classdef tPupilFindMethod < matlab.unittest.TestCase
%TPUPILFINDMETHOD  reset_xp_method='pupil_find' in the dwd* supervisors.
%
%   Gates the pupil_find exit-pupil method (Luis's request, shipped
%   e3d08ea): the cone-convergence best-fit sphere placed ONCE per
%   configuration via reset_xp_guard's 'pupil_find' action, with the
%   per-field FEX reset off.  Runs at MODEL 512 (the zoom fixture) --
%   registered in its own 512 suite batch, never in SUITE_FAST.
%
%   The load-bearing assertion is the CONFIG-CORRECTNESS one: on a
%   two-configuration run the two placed spheres must DIFFER when the
%   configuration tilts the pupil mirror -- a shared sphere would prove
%   the finder fit the UNCONFIGURED deck (the save_rx round-trip in the
%   guard is exactly what prevents that).

    properties (Constant)
        FOV  = 2.90888e-4          % 1 arcmin half-field (5-field set)
        TILT = 1.45444e-4          % config tilt: 0.5 arcmin on the FSM
    end

    properties
        rx
        od
    end

    methods (TestClassSetup)
        function setup(tc)
            here = fileparts(mfilename('fullpath'));
            root = fileparts(here);
            run(fullfile(root, 'mmacos_setup.m'));
            addpath(fullfile(root, 'design', 'runners'));
            addpath(fullfile(root, 'design', 'src'));
            tc.rx = fullfile(root, 'templates', '50_sensitivities', ...
                             'zoom_5x5', 'jwst_ote_designc.in');
            tc.assumeTrue(exist(tc.rx, 'file') == 2, 'zoom deck not present');
            tc.od = fullfile(tempdir, 'tPupilFindMethod_out');
            if ~exist(tc.od, 'dir'), mkdir(tc.od); end
        end
    end

    methods (Test)
        function test_missing_stop_elt_errors_actionably(tc)
        % The method requires an ELEMENT stop (pupil_find sets the engine
        % stop at that element); without one the runner must refuse UP
        % FRONT with a message that names the remedy.  Two guards can
        % legitimately fire first depending on the deck: the runner's
        % pre-existing no-ApStop check (this zoom deck -- no id, message
        % names 'stop_elt') or the method's own pfNeedsStopElt (decks
        % WITH an obj-space ApStop, unsupported for pupil_find).  The
        % contract asserted is ACTIONABILITY, not one specific id.
            raised = false;
            try
                run_sensitivities(tc.rx, 'fov_rad', tc.FOV, ...
                    'channels', "dwdx", 'reset_xp_method', 'pupil_find', ...
                    'model_size', 512, 'out_dir', string(tc.od), ...
                    'name', "neg");
            catch e
                raised = true;
                tc.verifySubstring(e.message, 'stop_elt', ...
                    'the refusal must name the remedy');
            end
            tc.verifyTrue(raised, 'expected an up-front refusal');
        end

        function test_two_configs_place_two_different_spheres(tc)
        % One harvest, two configurations (FSM +-0.5 arcmin), method
        % pupil_find: metrics per config, vertices DIFFER, method
        % recorded, report carries the fit lines.
            T = table(["zoomA"; "zoomB"], [tc.TILT; -tc.TILT], ...
                'VariableNames', {'name', '25.Ry'});
            cfgs = macos.design.configs_from_table(T);
            a = run_sensitivities(tc.rx, 'fov_rad', tc.FOV, ...
                'channels', "dwdx", 'elts', 24, 'dofs', (3:5).', ...
                'stop_elt', 25, 'reset_xp_method', 'pupil_find', ...
                'configs', cfgs, 'ngridpts', 41, 'model_size', 512, ...
                'out_dir', string(tc.od), 'name', "pfm", ...
                'per_element', [], 'verbose', false);
            tc.verifyEqual(a.ox.reset_xp_method, 'pupil_find');
            tc.assertTrue(isfield(a.ox, 'pupil_find'), 'metrics missing');
            tc.assertEqual(numel(a.ox.pupil_find), 2, ...
                'one placement per configuration expected');
            dv = norm(a.ox.pupil_find(1).vtx - a.ox.pupil_find(2).vtx);
            tc.verifyGreaterThan(dv, 1e-9, ...
                ['placed spheres identical across configs -- the ' ...
                 'save_rx round-trip did not carry the configuration']);
            rep = fileread(fullfile(tc.od, 'pfm_sens_report.txt'));
            tc.verifySubstring(rep, 'pupil_find cfg');
        end

        function test_placed_sphere_keeps_the_reference_tilt_sensitive(tc)
        % THE psi-sign gate.  pupil_find used to force psi(3)<0 on the
        % written sphere normal ("toward the image"); on this deck the
        % Return stores psi(3)>0, so the flipped normal reflected the
        % sphere CENTER to the pupil side.  A reference sphere centered
        % at the pupil is rotation-invariant in path length, so the OPD
        % reference went TILT-BLIND (a 0.5' FSM tilt moved the map by
        % 2.4e-7 mm instead of ~3e-2) and every field carried the full
        % sag as a ~0.45 mm RMS bias.  Both symptoms are asserted.
            m = macos.Session(512);                          %#ok<NASGU>
            nE0 = macos.load_rx(tc.rx);
            % pupil_find on a save_rx product at light sampling, with the
            % stop set -- the same flow the supervisor guard uses (the raw
            % deck declares nGridpts=1024, which would make every probe
            % trace huge, and carries no ApStop=)
            macos.set_src_sampling(41);
            macos.stop(25);
            macos.modify();
            tmp = fullfile(tc.od, 'pf_tilt_gate.in');
            macos.save_rx(tmp);
            pf = pupil_find(tmp, tc.fieldset(), 'ep_elt', 25, ...
                'stop_elt', 25, 'xp_elt', nE0 - 1, ...
                'place', true, 'init', false);
            tc.assertTrue(pf.placed);
            macos.set_src_sampling(41);
            macos.modify();
            macos.trace(nE0 - 1);
            W0 = macos.opd();
            rms0 = sqrt(mean(W0(W0 ~= 0).^2));
            % Healthy placements measure 4.4e-3 (supervisor flow, ng 63)
            % to ~3e-2 mm (this flow, ng 41); the flipped-psi defect
            % measured 0.45 and the wrong-hemisphere variant 3.6 -- the
            % bound sits an order above healthy scatter, an order below
            % the defect.
            tc.verifyLessThan(rms0, 1e-1, sprintf( ...
                ['center-field nominal against the placed sphere is ' ...
                 '%.3g mm RMS -- the pupil-side (flipped-psi) sphere ' ...
                 'bias'], rms0));
            macos.perturb(25, 'rotation', [tc.TILT; tc.TILT; 0], ...
                'frame', 'local');
            macos.modify();
            macos.trace(nE0 - 1);
            W1 = macos.opd();
            v = (W0 ~= 0) & (W1 ~= 0);
            dmax = max(abs(W1(v) - W0(v)));
            tc.verifyGreaterThan(dmax, 1e-3, sprintf( ...
                ['a 0.5'' FSM tilt moved the OPD by only %.3g mm -- ' ...
                 'the placed reference sphere is tilt-blind (its ' ...
                 'center is on the pupil side: psi sign)'], dmax));
        end

        function test_config_sphere_is_independent_of_predecessors(tc)
        % THE leakage gate.  The per-config save_rx used to capture the
        % PREVIOUS configuration's pf-written sphere at nElt-1 (the
        % config snapshot/restore covers only the configuration's own
        % elements), so every configuration after the first was fit on a
        % compounded EP state.  The same configuration's sphere must not
        % depend on what ran before it: zoomB harvested SECOND (after
        % zoomA) must equal zoomB harvested ALONE.
            T2 = table(["zoomA"; "zoomB"], [tc.TILT; -tc.TILT], ...
                'VariableNames', {'name', '25.Ry'});
            a2 = run_sensitivities(tc.rx, 'fov_rad', tc.FOV, ...
                'channels', "dwdx", 'elts', 24, 'dofs', (3:5).', ...
                'stop_elt', 25, 'reset_xp_method', 'pupil_find', ...
                'configs', macos.design.configs_from_table(T2), ...
                'ngridpts', 41, 'model_size', 512, ...
                'out_dir', string(tc.od), 'name', "pfleakA", ...
                'per_element', [], 'verbose', false);
            T1 = table("zoomB", -tc.TILT, ...
                'VariableNames', {'name', '25.Ry'});
            a1 = run_sensitivities(tc.rx, 'fov_rad', tc.FOV, ...
                'channels', "dwdx", 'elts', 24, 'dofs', (3:5).', ...
                'stop_elt', 25, 'reset_xp_method', 'pupil_find', ...
                'configs', macos.design.configs_from_table(T1), ...
                'ngridpts', 41, 'model_size', 512, ...
                'out_dir', string(tc.od), 'name', "pfleakB", ...
                'per_element', [], 'verbose', false);
            vB_after_A = a2.ox.pupil_find(2).vtx;
            vB_alone   = a1.ox.pupil_find(1).vtx;
            tc.verifyLessThan(norm(vB_after_A - vB_alone), 1e-6, ...
                ['zoomB''s placed sphere depends on zoomA having run ' ...
                 'first -- the previous configuration''s pf write ' ...
                 'leaked into the save_rx round-trip']);
        end

        function test_resume_checkpoints_are_method_aware(tc)
        % THE resume gate.  A checkpoint written under fex and resumed
        % under pupil_find used to be served VERBATIM (the key was
        % channel+config only), silently making the two methods' outputs
        % identical.  Plant a poisoned bare-name checkpoint; the
        % pupil_find run must ignore it (its key carries '_pf') and
        % complete with real metrics.
            rd = fullfile(tc.od, 'pf_resume');
            if exist(rd, 'dir'), rmdir(rd, 's'); end
            mkdir(rd);
            o = struct('poison', 1);
            save(fullfile(rd, 'dwdx_zoomA.mat'), 'o');
            save(fullfile(rd, 'dwdx_zoomB.mat'), 'o');
            T = table(["zoomA"; "zoomB"], [tc.TILT; -tc.TILT], ...
                'VariableNames', {'name', '25.Ry'});
            a = run_sensitivities(tc.rx, 'fov_rad', tc.FOV, ...
                'channels', "dwdx", 'elts', 24, 'dofs', (3:5).', ...
                'stop_elt', 25, 'reset_xp_method', 'pupil_find', ...
                'configs', macos.design.configs_from_table(T), ...
                'resume_dir', string(rd), 'ngridpts', 41, ...
                'model_size', 512, 'out_dir', string(tc.od), ...
                'name', "pfres", 'per_element', [], 'verbose', false);
            tc.assertTrue(isfield(a.ox, 'pupil_find'), ...
                'run served the poisoned fex-keyed checkpoints');
            tc.verifyEqual(numel(a.ox.pupil_find), 2, ...
                'both configurations must be REcomputed, not resumed');
            rep = fileread(fullfile(tc.od, 'pfres_sens_report.txt'));
            tc.verifySubstring(rep, 'dwdx_pf_zoomA.mat', ...
                'pupil_find checkpoints must carry the method key');
        end
    end

    methods
        function F = fieldset(tc)
        % The stock 5-field set (center + 4 corners), as (K x 2) rad.
            F = [0 0; -tc.FOV tc.FOV; tc.FOV tc.FOV; ...
                 -tc.FOV -tc.FOV; tc.FOV -tc.FOV];
        end
    end
end
