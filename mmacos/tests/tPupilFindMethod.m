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
    end
end
