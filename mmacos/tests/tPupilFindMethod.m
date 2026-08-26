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

    methods (Test)
        function test_field_scope_places_per_combo_tilt_absorbing_spheres(tc)
        % pf_scope='field' (Dave, 2026-08-25): a 3x3 mini-cone fit per
        % (config, field) block, FEX baseline run at the cone CENTER, so
        % each combo's sphere axis and radius follow its OWN chief and
        % the field tilt is absorbed per block.  Measured healthy values
        % (this grid, ng 41): outer-field nominal 3.8e-3 mm (vs 0.46 mm
        % when the baseline ran at the deck's nominal chief, and ~0.64 mm
        % under config scope); across-field vtx spacing 4.0e-4 mm;
        % across-config spacing 1.1e-5 mm.
            T = table(["zoomA"; "zoomB"], [tc.TILT; -tc.TILT], ...
                'VariableNames', {'name', '25.Ry'});
            cfgs = macos.design.configs_from_table(T);
            m = macos.Session(512);
            out = macos.dw_dx_multi(m, tc.rx, 'field_x_rad', tc.FOV, ...
                'field_y_rad', tc.FOV, 'grid', '3x1', 'elts', 24, ...
                'dofs', (3:5).', 'configs', cfgs, 'stop_elt', 25, ...
                'ngridpts', 41, 'reset_xp_method', 'pupil_find', ...
                'pf_scope', 'field');
            tc.verifyEqual(out.pf_scope, 'field');
            tc.assertEqual(numel(out.pupil_find), 6, ...
                'one placement per (config, field) block expected');
            P = out.pupil_find;
            tc.verifyEqual([P.config], [1 1 1 2 2 2]);
            tc.verifyEqual([P.field],  [1 2 3 1 2 3]);
            vtx = reshape([P.vtx], 3, []).';
            tc.verifyGreaterThan(norm(vtx(1,:) - vtx(2,:)), 1e-5, ...
                'spheres must be DISTINCT across fields');
            tc.verifyGreaterThan(norm(vtx(1,:) - vtx(4,:)), 1e-6, ...
                'spheres must be DISTINCT across configurations');
            % the WRITTEN vertex is the combo's chief crossing (Dave
            % 2026-08-25): the bundle vertex stays a diagnostic, because
            % writing it injects its lateral offset as a pure-tilt frame
            % term (0.38 mm -> 4.4e-3 mm RMS of tilt, zero aberration).
            tc.verifyLessThan(max(abs(P(2).vtx - P(2).fex_vpt)), 1e-9, ...
                'field scope must write the chief-crossing vertex');
            tc.verifyGreaterThan(P(2).vtx_minus_fex, 1e-4, ...
                'the bundle diagnostic must be preserved (nonzero offset)');
            r = cellfun(@(W) sqrt(mean(W(W ~= 0).^2)), ...
                        out.per_field_w_nom_2d);
            tc.verifyLessThan(max(r(:)), 1e-3, sprintf( ...
                ['worst per-combo nominal is %.3g mm RMS -- fex scale ' ...
                 'expected (chief-vertex placement).  Config scope ' ...
                 'leaves ~0.64 mm at these fields; the bundle vertex ' ...
                 'leaves 4-8e-3 of pure tilt'], max(r(:))));
        end

        function test_object_space_apstop_deck_needs_no_stop_elt(tc)
        % Luis's case (2026-08-26): the stop declared OBJECT-SPACE in the
        % deck header (ApStop= 3-vector) -- the segmented-primary idiom,
        % where no single stop ELEMENT exists (e5hex1: 7 hex segments
        % share the primary).  Two gates, both non-vacuous against the
        % pre-fix tree:
        %   A. pupil_find without 'stop_elt' must leave the deck's stop
        %      in force -- it used to run macos.stop(ep_elt=1), i.e.
        %      override the pupil with ONE segment's aperture.  The
        %      discriminator: get_stop_info reports only ELEMENT stops
        %      and raises on an object-space stop, so post-fix it must
        %      raise, pre-fix it returned elt 1.
        %   B. the supervisor flow (Luis's run_sensitivities path) must
        %      accept reset_xp_method='pupil_find' with NO 'stop_elt' --
        %      it used to refuse up front (pfNeedsStopElt).
            here = fileparts(mfilename('fullpath'));
            root = fileparts(here);
            rxh = fullfile(root, 'templates', '50_sensitivities', ...
                           'run_dwdz_multi', 'e5hex1.in');
            tc.assumeTrue(exist(rxh, 'file') == 2, 'e5hex1 deck not present');
            fov = 1e-4;
            m = macos.Session(256);
            % -- A: direct pupil_find, deck stop preserved
            nE = macos.load_rx(rxh);
            macos.set_src_sampling(33);
            macos.modify();
            tmp = fullfile(tc.od, 'pf_objstop_gate.in');
            macos.save_rx(tmp);
            F = [0 0; -fov fov; fov fov; -fov -fov; fov -fov];
            pf = pupil_find(tmp, F, 'xp_elt', nE - 1, ...
                            'place', true, 'init', false);
            tc.assertTrue(pf.placed);
            stop_is_elt = true;
            try, macos.get_stop_info(); catch, stop_is_elt = false; end
            tc.verifyFalse(stop_is_elt, ...
                ['pupil_find overrode the deck''s object-space ApStop ' ...
                 'with an element stop (the segmented-primary pupil ' ...
                 'collapsed to one segment''s aperture)']);
            % The WRITTEN vertex on a segmented deck must be the FEX
            % chief crossing -- the paraxial anchor (Dave 2026-08-26).
            % The cone fit (stop-plane anchor from the deck ApStop,
            % entrance positions from the ray history -- Dave's
            % construction) is the pupil-structure DIAGNOSTIC: on this
            % deck it measures a real ~23 mm pupil smear (two-singlet
            % relay; differential chief 1133.3 / finite chief-pair
            % 1142.3 / annular cone zones 1156.2), which must NOT be
            % folded into the written reference.
            tc.verifyEqual(pf.vertex, 'chief', ...
                'segmented decks must force the chief-crossing vertex');
            tc.verifyLessThan(max(abs(pf.vtx_written(:) - pf.fex.vpt(:))), ...
                1e-9, ['the written vertex is not the FEX chief ' ...
                 'crossing -- a cone-fit vertex reached the Rx on a ' ...
                 'segmented deck']);
            % the stop-plane/history binning must produce a CLEAN
            % convergence surface: dep_rms 0.9 um measured; the earlier
            % index-grouped binning left 4.5 um, the M2-anchored one a
            % biased cloud -- the bound separates the constructions
            tc.verifyLessThan(pf.dep_rms, 2e-3, sprintf( ...
                ['cone-convergence departure %.3g mm RMS -- the ' ...
                 'stop-plane binning has degraded'], pf.dep_rms));
            % -- B: supervisor flow, no stop_elt
            out = macos.dw_dx_multi(m, rxh, 'field_x_rad', fov, ...
                'field_y_rad', fov, 'grid', '3x1', 'elts', 8, ...
                'dofs', 3, 'ngridpts', 33, ...
                'reset_xp_method', 'pupil_find');
            tc.assertTrue(isfield(out, 'pupil_find') && ...
                          ~isempty(out.pupil_find), ...
                ['supervisor pupil_find metrics missing -- the ' ...
                 'no-stop_elt path did not run the finder']);
        end

        function test_field_scope_probe_delta_is_not_load_bearing(tc)
        % The mini-cone half-width is a conditioning knob, not a result
        % knob: the fitted vertex must be stable under a 2x change.
        % Measured: 5.9e-5 mm between delta and 2*delta at the healthy
        % default (0.15x the field half-width).
            m = macos.Session(512);                          %#ok<NASGU>
            nE0 = macos.load_rx(tc.rx);
            macos.set_src_sampling(41);
            macos.stop(25);
            macos.modify();
            tmp = fullfile(tc.od, 'pf_delta_gate.in');
            macos.save_rx(tmp);
            d0 = 0.15 * tc.FOV;
            V = zeros(2, 3);
            for j = 1:2
                [gx, gy] = ndgrid([-1 0 1] * d0 * j);
                pf = pupil_find(tmp, [gx(:), gy(:)], 'ep_elt', 25, ...
                    'stop_elt', 25, 'xp_elt', nE0 - 1, ...
                    'place', false, 'init', false);
                V(j, :) = pf.vtx;
            end
            tc.verifyLessThan(norm(V(1,:) - V(2,:)), 1e-2, sprintf( ...
                ['fitted vertex moved %.3g mm under a 2x probe ' ...
                 'half-width change -- the cone fit is ' ...
                 'ill-conditioned at this delta'], norm(V(1,:) - V(2,:))));
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
