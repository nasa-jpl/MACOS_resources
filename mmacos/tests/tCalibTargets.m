classdef tCalibTargets < matlab.unittest.TestCase
%TCALIBTARGETS  The CALIB targets WFE_ZMODE (works) and OPL (removed) -- PLAN 3.1, 2026-10-10.
%   MEASURED on Rx_AsphCalib.in (a metre paraboloid whose h^4 term is spoiled to 5e-3):
%     WFE       h^4 5e-3 -> -1e-12, converged (the baseline)
%     WFE_ZMODE STOPPED THE HOST before the fix ('STOP program terminated by nerror'):
%               macos_mod_init_vars reset the Zernike-fit basis zcoType to 0 on every load
%               (after the LOAD handler had set ZernCircular), so zern_fit took the
%               non-circular branch with type 0, read ZernAnnuRatio(0) out of bounds, the
%               fit's SVD failed and nerror STOPped.  Now zcoType resets TO ZernCircular:
%               [4 11 12] -> h^4 -3.5e-13; [11] -> 1.4e-12 (mode 11 is primary spherical,
%               Noll 1-based); [12] leaves it (4.4e-3) -- the mode choice is honoured.
%     OPL       accepted by the parser and the api with NO objective behind it; CALIB
%               failed.  Removed: the api refuses target 5 by name, the veneer has no
%               'OPL' name, and an Rx 'OptTarget= OPL' is a named load refusal.
%   The WFE_ZMODE leg runs in a matlab -batch SUBPROCESS: the pre-fix mex kills MATLAB.
    properties (Constant)
        RxName = 'Rx_AsphCalib.in'
        Seed   = 5e-3
    end
    methods (Test)
        function test_wfe_zmode_converges_on_the_named_mode_in_a_subprocess(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's')); %#ok<NASGU>
            mm  = fileparts(fileparts(mfilename('fullpath')));
            scr = fullfile(wd, 'zmode_leg.m');  fid = fopen(scr, 'w');
            fprintf(fid, 'run(''%s'');\n', fullfile(mm, 'mmacos_setup.m'));
            for md = {'11', '12'}
                fprintf(fid, ['m = macos.Session(128); m.load_rx(''%s'');\n' ...
                    'm.calib_set_target(''WFE_ZMODE'', %s); r = m.calib();\n' ...
                    'a = macos.get_elt_asph(1, 4);\n' ...
                    'fprintf(''MODE%s %%d %%.6e\\n'', r.converged, a(1));\n'], ...
                    rx_fixture_path(tc.RxName), md{1}, md{1});
            end
            fclose(fid);
            [st, out] = system(sprintf('cd %s && "%s" -batch "run(''%s'')" 2>&1', ...
                wd, fullfile(matlabroot, 'bin', 'matlab'), scr));
            tc.verifyEqual(st, 0, sprintf(['WFE_ZMODE must not kill MATLAB (pre-fix: ' ...
                '''STOP program terminated by nerror'')\n%s'], out));
            t11 = regexp(out, 'MODE11 (\d) (\S+)', 'tokens', 'once');
            t12 = regexp(out, 'MODE12 (\d) (\S+)', 'tokens', 'once');
            tc.assertNotEmpty(t11, 'the mode-11 solve reported');
            tc.assertNotEmpty(t12, 'the mode-12 solve reported');
            tc.verifyEqual(str2double(t11{1}), 1, 'WFE_ZMODE [11] converges');
            tc.verifyLessThan(abs(str2double(t11{2})), 0.1 * tc.Seed, ...
                'mode 11 (primary spherical, Noll) drives the spoiled h^4 term out');
            tc.verifyGreaterThan(abs(str2double(t12{2})), 0.5 * tc.Seed, ...
                'non-vacuity: mode 12 does not see that term, so the mode choice is honoured');
        end
        function test_opl_is_refused_by_name(tc)
            tc.verifyError(@() macos.calib_set_target('OPL'), 'macos:calib_set_target:badName');
            m = macos.Session(128);  m.load_rx(rx_fixture_path(tc.RxName));
            tc.verifyError(@() m.calib_set_target(5), ?MException, ...
                'the engine refuses target 5 (OPL) -- the api answers FAIL');
        end
        function test_an_rx_opl_target_is_a_load_refusal(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's')); %#ok<NASGU>
            L = splitlines(string(fileread(rx_fixture_path(tc.RxName))));
            k = find(startsWith(strtrim(L), "OptTarget="), 1);
            tc.assertNotEmpty(k, 'the fixture declares an OptTarget=');
            L(k) = "        OptTarget=  OPL";
            p = fullfile(wd, 'opl.in');  fid = fopen(p, 'w');  fprintf(fid, '%s\n', L);  fclose(fid);
            macos.init(128);
            tc.verifyError(@() macos.load_rx(p), ?MException, 'OptTarget= OPL refuses the load');
        end
    end
end
