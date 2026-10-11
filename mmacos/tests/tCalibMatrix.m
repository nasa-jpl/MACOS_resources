classdef tCalibMatrix < matlab.unittest.TestCase
%TCALIBMATRIX  CALIB's target x DOF-type matrix (PLAN_CONSOLIDATION 3.4, 2026-10-11).
%   Four targets (WFE, SPOT, WFE_ZMODE on Noll [4 7 8 11], BEAM = the chief ray driven to the
%   focal-plane vertex) x four DOF types, one SPOILED fixture each, all the f = 1 m paraboloid of
%   Rx_AsphCalib (model 128):
%     rigid    Rx_CalibRigid.in   tilted 1 mrad about x, TIP free
%     zernike  Rx_CalibZern.in    ANSI Z12 spoiled 2e-7 m, that mode free
%     aspheric Rx_AsphCalib.in    h^4 spoiled 5e-3, that term free
%     mixed    Rx_CalibMixed.in   tilt + h^4, TIP and the term free
%   WELL-POSED (13 cells): the solve must end converged / floor / cap with the spoil removed.
%   ILL-POSED, must fail BY NAME with the optics restored (3 cells): BEAM x {zernike, aspheric,
%   mixed} -- a beam-position row cannot see a shape term (a zero Jacobian column).
%   HOST SURVIVAL: SPOT with TIP AND TILT free on the rigid deck -- the unspoiled TILT is a
%   first-order null DOF of a centroid-referenced spot size, LM's first trial rotated the mirror
%   ~76 deg, the beam-frame chief ray missed the focal plane and MACOS_OPS STOPPED: the mex took
%   MATLAB down (rc=139, measured on the first matrix pass).  Now a named failure: run in a
%   matlab -batch subprocess so the pre-fix engine fails the TEST, not the suite.
    properties (Constant)
        Model = 128
        Fix = struct('rigid','Rx_CalibRigid.in', 'zernike','Rx_CalibZern.in', ...
                     'aspheric','Rx_AsphCalib.in', 'mixed','Rx_CalibMixed.in')
    end
    properties (TestParameter)
        well = {'rigid_WFE','rigid_SPOT','rigid_WFE_ZMODE','rigid_BEAM', ...
                'zernike_WFE','zernike_SPOT','zernike_WFE_ZMODE', ...
                'aspheric_WFE','aspheric_SPOT','aspheric_WFE_ZMODE', ...
                'mixed_WFE','mixed_SPOT','mixed_WFE_ZMODE'}
        ill  = {'zernike_BEAM','aspheric_BEAM','mixed_BEAM'}
    end
    methods (Static, Access = private)
        function [dof, tgt] = split_(c)
            k = find(c == '_', 1);  dof = c(1:k-1);  tgt = c(k+1:end);
        end
        function set_target_(tgt)
            switch tgt
                case 'WFE_ZMODE', macos.calib_set_target('WFE_ZMODE', [4 7 8 11]);
                case 'BEAM',      macos.calib_set_target('BEAM');
                                  macos.calib_set_beam('pos', 2, [0 0 -1]);
                otherwise,        macos.calib_set_target(tgt);
            end
        end
        function s = spoil_(dof)
            % the spoiled quantities: tilt (rad), h^4 coefficient, Z12 (m)
            psi = macos.get_elt_psi(1);
            s.tilt = acos(min(1, -psi(3) / norm(psi)));
            s.h4 = NaN;  s.z12 = NaN;
            if any(strcmp(dof, {'aspheric','mixed'})), a = macos.get_elt_asph(1, 1); s.h4 = a(1); end
            if strcmp(dof, 'zernike'), s.z12 = macos.get_elt_zrn_coef(1, 12); end
        end
        function load_(dof)
            macos.init(tCalibMatrix.Model);
            macos.load_rx(rx_fixture_path(tCalibMatrix.Fix.(dof)));
        end
    end
    methods (Test)
        function test_a_well_posed_cell_removes_its_spoil(tc, well)
            [dof, tgt] = tCalibMatrix.split_(well);
            tCalibMatrix.load_(dof);  tCalibMatrix.set_target_(tgt);
            r = macos.calib();
            tc.verifyTrue(ismember(r.stop_reason, {'converged','floor','cap'}), ...
                sprintf('%s: stop reason %s', well, r.stop_reason));
            s = tCalibMatrix.spoil_(dof);
            tc.verifyLessThan(s.tilt, 1e-8, [well ': the 1 mrad tilt is removed']);
            if ~isnan(s.h4),  tc.verifyLessThan(abs(s.h4), 1e-6, [well ': h^4 (seed 5e-3) is removed']); end
            if ~isnan(s.z12), tc.verifyLessThan(abs(s.z12), 1e-12, [well ': Z12 (seed 2e-7 m) is removed']); end
            tr = macos.trace(2);
            tc.verifyLessThan(tr.rmsWFE, 1e-10, [well ': the image is clean']);
        end
        function test_an_ill_posed_cell_fails_by_name_with_the_optics_restored(tc, ill)
            [dof, tgt] = tCalibMatrix.split_(ill);
            tCalibMatrix.load_(dof);  s0 = tCalibMatrix.spoil_(dof);
            tCalibMatrix.set_target_(tgt);
            tc.verifyError(@() macos.calib(), ?MException, [ill ': the solve must FAIL (a null column)']);
            st = macos.calib_stats();
            tc.verifyEqual(st.stop_reason, 'failed');
            s1 = tCalibMatrix.spoil_(dof);
            tc.verifyEqual([s1.tilt s1.h4 s1.z12], [s0.tilt s0.h4 s0.z12], 'AbsTol', 1e-15, ...
                'the optics come back exactly as they went in');
        end
        function test_a_trial_that_loses_the_chief_ray_does_not_kill_the_host(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's')); %#ok<NASGU>
            mm  = fileparts(fileparts(mfilename('fullpath')));
            scr = fullfile(wd, 'nulldof_leg.m');  fid = fopen(scr, 'w');
            fprintf(fid, ['run(''%s'');\nmacos.init(%d); macos.load_rx(''%s'');\n' ...
                'macos.calib_set_var_elt(1, ''TIP'', ''TILT''); macos.calib_set_target(''SPOT'');\n' ...
                'try, macos.calib(); fprintf(''CALIB_RETURNED\\n''); catch e, fprintf(''CALIB_FAILED %%s\\n'', e.message); end\n' ...
                'psi = macos.get_elt_psi(1); fprintf(''TILT %%.6e\\n'', acos(-psi(3)/norm(psi)));\n'], ...
                fullfile(mm, 'mmacos_setup.m'), tc.Model, rx_fixture_path(tc.Fix.rigid));
            fclose(fid);
            [st, out] = system(sprintf('cd %s && "%s" -batch "run(''%s'')" 2>&1', ...
                wd, fullfile(matlabroot, 'bin', 'matlab'), scr));
            tc.verifyEqual(st, 0, sprintf(['MATLAB must survive (pre-fix: MACOS_OPS STOPPED, ' ...
                'rc=139)\n%s'], out));
            tc.verifyNotEmpty(regexp(out, 'CALIB_FAILED', 'once'), 'the failure comes back as an error');
            tc.verifyNotEmpty(regexp(out, 'a trial step lost the chief ray', 'once'), 'and names its cause');
            t = str2double(regexp(out, 'TILT (\S+)', 'tokens', 'once'));
            tc.verifyEqual(t, 1e-3, 'RelTol', 1e-6, 'the optics are restored to the last accepted state');
        end
    end
end
