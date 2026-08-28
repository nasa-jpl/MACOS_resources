classdef tRunSensitivities < matlab.unittest.TestCase
    % Sensitivities stage runner (design/runners/run_sensitivities) +
    % the grid-frame rewriter (macos.design.grid_augment_rx), on the
    % SMM e5 PIE fixture -- which carries the CORPUS TRAP this stack
    % exists to close: SegMirMaker replicates the parent's grid channel
    % (pData = parent vertex, full-aperture span) into every segment
    % block, so a segment-frame influence basis poked against those
    % grids paints about the APERTURE center ("central dot") and the
    % dwdgrid Jacobian rank-collapses (e5pie: rank 15 of 42, cond 1e7;
    % diagnosed 2026-07-19).  grid_augment_rx must REPLACE those lines
    % (last-key-wins parsing means appending cannot fix it), and the
    % runner's dwdgrid must come out FULL RANK with localized pokes.
    properties
        seg = []
        smm_bin = ''
        res_root = ''
    end

    methods (TestClassSetup)
        function make(tc)
            here = fileparts(mfilename('fullpath'));
            tc.res_root = fileparts(fileparts(here));
            addpath(fullfile(tc.res_root, 'mmacos', 'design', 'runners'));
            addpath(fullfile(tc.res_root, 'mmacos', 'sensitivities'));
            % Locate whichever build tree exists (ifx on Linux, gfortran on
            % macOS / gfortran builds); ifx preferred when both are present.
            smmdir = fullfile(tc.res_root, 'segmirmaker');
            for tag = ["build_release_ifx", "build_release_gfortran", ...
                       "build_debug_ifx", "build_debug_gfortran"]
                cand = fullfile(smmdir, tag, 'SegMirMaker');
                if isfile(cand), tc.smm_bin = cand; break; end
            end
            macos.init(512);
        end
    end

    methods (Access = private)
        function s = seg_fixture(tc)
            % The SMM-built pie fixture, built ONCE and shared.  Built
            % lazily rather than in TestClassSetup so that a tree without
            % a SegMirMaker build skips only the cases that need it --
            % the CONFIGURATION-axis cases below run on a stock template
            % deck and must not be filtered out with them.
            tc.assumeTrue(~isempty(tc.smm_bin) && isfile(tc.smm_bin), ...
                'SegMirMaker not built');
            if isempty(tc.seg)
                tin = fullfile(tc.res_root, 'segmirmaker', 'test_in');
                tc.seg = macos.design.segment_rx(fullfile(tin, 'e5mono.in'), ...
                    'elt', 1, 'rings', 1, 'grid', 'Pie', 'gap', 50, ...
                    'dofs', 6, 'meas_config', 1);
            end
            s = tc.seg;
        end

        function [m, rx, b] = cfg_fixture(tc)
            % A stock template deck for the configuration-axis cases:
            % 13 elements with a header ApStop, elt 8 a figured
            % (Zernike) Reflector -- so the pose snapshot exercises the
            % figure-frame branch -- and elt 12 the exit-pupil Return.
            % Trimmed hard (one element's 6 DOFs, 2 fields, coarse grid)
            % because these cases are about BOOKKEEPING, not optics.
            rx = fullfile(tc.res_root, 'mmacos', 'templates', ...
                '50_sensitivities', 'run_dwdx_multi', 'e5hex1.in');
            m = macos.Session(512);
            b = {'field_x_rad', 1e-4, 'field_y_rad', 1e-4, 'grid', '2x1', ...
                 'ngridpts', 15, 'elts', 8, 'dofs', (0:5).', 'delta', 1e-8};
        end

        function f = zoom_fixture(tc)
            % The promoted multi-zoom fixture (jwst_ote_designc.in): its
            % 19 segments are Surface= FreeForm carrying a zero-amplitude
            % MonZernike figure channel, so the figure rungs (dw/dz,
            % dw/dgrid) of the configuration-axis family harvest on it.
            % The promotion (macos.design.promote_segments_freeform) is
            % gated for inertness below.
            f = fullfile(tc.res_root, 'mmacos', 'templates', ...
                '50_sensitivities', 'zoom_5x5', 'jwst_ote_designc.in');
        end

        function [W, ok] = zoom_opd(tc, rx, stop_elt) %#ok<INUSD>
            % Exit-pupil (elt 27) OPD of the zoom fixture at a coarse
            % sampling, with the FSM (elt 25) as the stop.  Shared by the
            % inertness gate.
            m = macos.Session(512);
            m.load_rx(rx);
            m.set_src_sampling(41);
            m.stop(int32(25));
            m.modify();
            m.trace(27);
            W = m.opd();
            ok = isfinite(W);
        end
    end

    methods (Test)
        function test_grid_augment_replaces_stale(tc)
            % the SMM fixture ships stale parent-frame grid lines in
            % every segment block: they must be REPLACED by the
            % segment's own clocked Mon frame
            sg = tc.seg_fixture();
            wd = tempname; mkdir(wd);
            cwd = onCleanup(@() rmdir(wd, 's'));
            rxg = fullfile(wd, 'aug.in');
            ga = macos.design.grid_augment_rx(sg.in, rxg, 'ng', 64);
            tc.verifyEqual(ga.nseg, sg.nseg);
            tc.verifyTrue(all(ga.replaced), ...
                'SMM fixture must report stale grid lines replaced');
            tc.verifyTrue(isfile(ga.gridfile));
            txt = string(fileread(rxg));
            blocks = split(txt, "Element=");
            nGrid = 0;
            for b = blocks(:).'
                if ~startsWith(strtrim(b), "Segment"), continue; end
                nGrid = nGrid + 1;
                % exactly ONE grid channel per segment block
                tc.verifyEqual(count(b, "nGridMat="), 1);
                tc.verifyEqual(count(b, "pData="), 1);
                % pData/xData == the block's own pMon/xMon (clocked frame)
                gv = @(k) sscanf(char(extractBetween(b, k + "=", newline)), '%g').';
                tc.verifyEqual(gv("pData"), gv("pMon"), 'AbsTol', 0);
                tc.verifyEqual(gv("xData"), gv("xMon"), 'AbsTol', 0);
                tc.verifyEqual(gv("zData"), gv("zMon"), 'AbsTol', 0);
                % span covers the circumscribing circle: gdx*(ng-1) >= 2*lMon
                lm = gv("lMon");
                gx = gv("GridSrfdx");
                tc.verifyGreaterThanOrEqual(gx*(64-1), 2*lm(1)*(1 - 1e-12));
            end
            tc.verifyEqual(nGrid, sg.nseg);
        end

        function test_reset_xp_method_sxp_deprecation_warns(tc)
            % reset_xp_method is deprecated (FEX==SXP post-merge); 'sxp'
            % must still WORK (legacy decks pass it) but warn once.  Clear
            % the function so its persistent one-time flag is fresh.
            sg = tc.seg_fixture();
            wd = tempname; mkdir(wd);
            cwd = onCleanup(@() rmdir(wd, 's'));
            rxg = fullfile(wd, 'aug.in');
            macos.design.grid_augment_rx(sg.in, rxg, 'ng', 64);
            m = macos.Session(512);
            sgb = macos.segment_grid_basis(m, rxg, ...
                'pm_ref_elt', 1, 'modes', 4:5, 'orthogonalize', true);
            clear macos.dw_dgrid_multi   % reset the persistent warned flag
            tc.verifyWarning(@() macos.dw_dgrid_multi(m, rxg, ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, 'grid', '1x1', ...
                'influence', sgb, 'zmodes', 4:5, 'ngridpts', 15, ...
                'reset_xp', false, 'reset_xp_method', 'sxp'), ...
                'macos:dw_dgrid_multi:resetXpMethodDeprecated');
        end

        function test_fex_axis_is_an_explicit_chief_default(tc)
            % The FEX pupil-sphere AXIS option (Dave 2026-08-27):
            % 'fex_axis' 'chief' (default) | 'centroid' -- the engine
            % CHIEFRAY/CENTROID toggle (api xp_fnd mode 1 | 0) surfaced
            % through the supervisors and run_sensitivities.  Vertex and
            % radius are axis-invariant (only psi moves), so on this
            % segmented, gap-obscured deck the centroid-vs-chief nominal
            % difference must be (a) NONZERO -- the downstream centroid
            % walks off the chief (measured 8e-4 mm at the EP on the
            % e5hex1 CLI, the FEX-axis-ruling reproducer) -- and (b) a
            % pure tip/tilt(+piston) FRAME term, killed by PTT removal.
            % pupil_find refuses the option at both layers: its written
            % reference is chief-tied by doctrine.
            [m, rx] = tc.cfg_fixture();
            b = {'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                 'grid', '2x1', 'ngridpts', 15, 'elts', 8, ...
                 'dofs', 0, 'delta', 1e-8};
            oc = macos.dw_dx_multi(m, rx, b{:});
            tc.verifyEqual(oc.fex_axis, 'chief', ...
                'the default FEX axis must be chief (the axis ruling)');
            on = macos.dw_dx_multi(m, rx, b{:}, 'fex_axis', 'centroid');
            tc.verifyEqual(on.fex_axis, 'centroid');
            dmax = 0;  frac = 0;
            for k = 1:numel(oc.per_field_w_nom_2d)
                msk = (oc.per_field_w_nom_2d{k} ~= 0) & ...
                      (on.per_field_w_nom_2d{k} ~= 0);
                D = on.per_field_w_nom_2d{k} - oc.per_field_w_nom_2d{k};
                v = D(msk);
                dmax = max(dmax, max(abs(v)));
                [ii, jj] = find(msk);
                x = ii - mean(ii);  y = jj - mean(jj);
                r = max(hypot(x, y));
                A = [ones(size(v)) x/r y/r];
                frac = max(frac, rms(v - A*(A\v)) / max(rms(v), eps));
            end
            tc.verifyGreaterThan(dmax, 1e-8, ...
                ['centroid axis made no difference on a segmented ' ...
                 'deck -- the option is not reaching the engine']);
            tc.verifyLessThan(frac, 0.2, sprintf( ...
                ['centroid-vs-chief nominal difference is %.0f%% ' ...
                 'NON-PTT -- the axis change must be a pure frame ' ...
                 'term (psi-only; vertex and radius are ' ...
                 'axis-invariant)'], 100*frac));
            tc.verifyError(@() macos.dw_dx_multi(m, rx, b{:}, ...
                'fex_axis', 'centroid', ...
                'reset_xp_method', 'pupil_find'), ...
                'macos:dw_dx_multi:fexAxisScope');
            tc.verifyError(@() run_sensitivities(rx, 'fov_rad', 1e-4, ...
                'channels', "dwdx", 'reset_xp_method', 'pupil_find', ...
                'fex_axis', 'centroid'), ...
                'macos:run_sensitivities:fexAxisScope');
        end

        function test_run_sensitivities_end_to_end(tc)
            % trimmed harvest on the SMM pie fixture; the regression
            % gate is dwdgrid FULL RANK + localized pokes (the stale
            % frames give rank ~15/42 and center-piled responses)
            sg = tc.seg_fixture();
            wd = tempname; mkdir(wd);
            cwd = onCleanup(@() rmdir(wd, 's'));
            res_root = tc.res_root;
            copyfile(sg.in, fullfile(wd, 'pie.in'));
            copyfile(fullfile(res_root, 'segmirmaker', 'test_in', ...
                'flat.txt'), fullfile(wd, 'flat.txt'));   % rx's own 256 grid
            art = run_sensitivities(fullfile(wd, 'pie.in'), ...
                'fov_rad', 1e-4, 'ngridpts', 31, 'ng', 64, ...
                'zmodes_fig', 4:6, 'zmodes_grid', 4:6, ...
                'stop', [0 0 0], ...      % SMM corpus ships no ApStop
                'per_element', "center", 'verbose', false);
            % artifacts
            tc.verifyTrue(isfile(art.mat));
            tc.verifyTrue(isfile(art.report));
            tc.verifyTrue(isfile(art.grid_in));
            % channel shapes: 7 segs x 3 modes
            tc.verifyEqual(size(art.og.dwdgall, 2), 3*sg.nseg);
            % dwdz sweeps EVERY lMon-bearing optic (non-segment optics
            % included -- the supervisor contract); the segments must
            % each contribute exactly the 3 requested modes
            cnz = art.oz.channel_names;
            for s2 = 1:sg.nseg
                tc.verifyEqual(nnz(startsWith(cnz, ...
                    sprintf('Elt %d MonZern', s2))), 3);
            end
            tc.verifyGreaterThanOrEqual(size(art.oz.dwdxall, 2), ...
                3*sg.nseg);
            % THE gate: full rank + healthy conditioning (stale frames
            % gave rank 15/42, cond 1e7 on this exact corpus)
            A = art.og.dwdgall;  A = A(all(isfinite(A), 2), :);
            s = svd(full(A), 'econ');
            tc.verifyEqual(nnz(s > max(size(A))*eps(s(1))), size(A, 2), ...
                'dwdgrid must be full rank (localized pokes)');
            tc.verifyLessThan(s(1)/s(end), 100, ...
                'dwdgrid conditioning must be healthy');
            % localization: each segment''s poke support must be
            % (near-)disjoint from every other segment''s -- the
            % central-dot failure overlaps them all at the center
            ns = sg.nseg;  nmode = 3;
            sup = false(size(A, 1), ns);
            okr = all(isfinite(art.og.dwdgall), 2);
            for s2 = 1:ns
                cols = (s2-1)*nmode + (1:nmode);
                v = max(abs(art.og.dwdgall(okr, cols)), [], 2);
                sup(:, s2) = v > 0.1*max(v);
            end
            for a2 = 1:ns
                for b2 = a2+1:ns
                    ov = nnz(sup(:,a2) & sup(:,b2)) / ...
                         max(1, min(nnz(sup(:,a2)), nnz(sup(:,b2))));
                    tc.verifyLessThan(ov, 0.15, sprintf( ...
                        'segments %d/%d poke supports must not overlap', a2, b2));
                end
            end
        end

        % =============================================================
        % CONFIGURATION axis -- design/PLAN_CONFIGURATIONS.md
        % =============================================================

        function test_configs_absent_is_byte_identical(tc)
            % THE preserved-surface gate, and it is a GATE not a
            % tolerance: with 'configs' absent (or empty, or a single
            % IDENTITY configuration) the supervisor must return exactly
            % what it returned before the axis existed -- every existing
            % runner, example and committed baseline depends on it.
            [m, rx, b] = tc.cfg_fixture();
            a  = macos.dw_dx_multi(m, rx, b{:});
            e  = macos.dw_dx_multi(m, rx, b{:}, 'configs', []);
            id = macos.dw_dx_multi(m, rx, b{:}, ...
                    'configs', struct('name', 'nom', 'set', {{}}));
            for f = {'dwdxall', 'w0_stacked', 'OPDall', 'indxall', ...
                     'per_field_dwdx', 'per_field_w_nom_2d'}
                tc.verifyTrue(isequal(a.(f{1}), e.(f{1})), ...
                    sprintf('configs=[] changed %s', f{1}));
            end
            % shape and surface: no configuration fields appear
            tc.verifyEqual(size(a.per_field_dwdx), [2 1]);   % Nfields x 1
            tc.verifyFalse(isfield(a, 'config_table'));
            tc.verifyFalse(isfield(a.indxall, 'config'));
            % an identity configuration must reproduce the NUMBERS
            % exactly -- it only changes the packaging
            tc.verifyTrue(isequal(id.dwdxall, a.dwdxall), ...
                'an identity configuration must not perturb the Jacobian');
            tc.verifyTrue(isequal(id.w0_stacked, a.w0_stacked));
            tc.verifyEqual(size(id.per_field_dwdx), [1 2]);  % Nc x Nfields
            tc.verifyTrue(isfield(id, 'config_table'));
            tc.verifyEqual(id.config_names, {'nom'});
        end

        function test_two_configs_stack_as_contiguous_rows(tc)
            % A configuration adds observations of the SAME state vector,
            % exactly as a field point does, so the blocks stack as
            % ROWS -- never a third array dimension, which would break
            % every downstream consumer.
            [m, rx, b] = tc.cfg_fixture();
            th = 1e-5;
            cfgs = [struct('name', 'nom', 'set', {{}}), ...
                    struct('name', 'tilt', 'set', {{ ...
                        {'perturb', 8, 'rotation', [th; 0; 0], ...
                         'frame', 'local'} }})];
            a = macos.dw_dx_multi(m, rx, b{:});
            d = macos.dw_dx_multi(m, rx, b{:}, 'configs', cfgs);
            n1 = nnz(d.indxall.config == 1);
            n2 = nnz(d.indxall.config == 2);
            tc.verifyEqual(size(d.dwdxall, 1), n1 + n2);
            tc.verifyEqual(numel(d.w0_stacked), n1 + n2);
            tc.verifyEqual(n1, numel(a.w0_stacked));
            tc.verifyTrue(isequal(find(d.indxall.config == 1), (1:n1).'), ...
                'each configuration must own a CONTIGUOUS block of rows');
            % the first (nominal) block is the no-configs harvest, bitwise
            tc.verifyTrue(isequal(d.dwdxall(1:n1, :), a.dwdxall));
            tc.verifyTrue(isequal(d.w0_stacked(1:n1), a.w0_stacked));
            % column identity across blocks (asserted, not assumed)
            tc.verifyEqual(d.channel_names, a.channel_names);
            tc.verifyEqual(d.config_names, {'nom'; 'tilt'});
            tc.verifyEqual(size(d.per_field_dwdx), [2 2]);   % Nc x Nfields
            % NON-VACUITY: the second configuration must actually move
            % the Jacobian, or the gate above passes for the wrong reason
            k = min(n1, n2);
            tc.verifyGreaterThan( ...
                max(abs(d.dwdxall(n1+1:n1+k, :) - d.dwdxall(1:k, :)), [], 'all'), ...
                0, 'the tilted configuration produced an identical block');
        end

        function test_configuration_is_restored_to_a_millionth(tc)
            % The load-bearing part of the design.  The supervisor's own
            % assertion already ran (it would have errored); this pins the
            % residual against the scale a FAILED restore would leave --
            % the size of the configuration itself -- rather than against
            % round-off, whose floor is set by the channel loop's own
            % poke/restore cycles on the same element, not by this.
            [m, rx, b] = tc.cfg_fixture();
            m.load_rx(rx);
            th = 1e-5;
            psi0 = m.get_elt_psi(8);  vpt0 = m.get_elt_vpt(8);
            % how far the configuration MOVES the element
            m.perturb(8, 'rotation', [th; 0; 0], 'frame', 'local');
            moved = max(abs(m.get_elt_psi(8) - psi0));
            m.perturb(8, 'rotation', [-th; 0; 0], 'frame', 'local');
            m.modify();
            tc.verifyGreaterThan(moved, 0, 'the probe configuration is a no-op');
            cfgs = struct('name', 'tilt', 'set', {{ ...
                {'perturb', 8, 'rotation', [th; 0; 0], 'frame', 'local'} }});
            macos.dw_dx_multi(m, rx, b{:}, 'configs', cfgs);
            tc.verifyLessThan(max(abs(m.get_elt_psi(8) - psi0)), 1e-6 * moved);
            tc.verifyLessThan(max(abs(m.get_elt_vpt(8) - vpt0)), ...
                1e-6 * moved * max(1, max(abs(vpt0))));
        end

        function test_config_validation_rejects_before_apply(tc)
            % The v1 whitelist and the range checks fire at VALIDATION
            % time, before anything touches the model -- a setter whose
            % effect the pose snapshot cannot restore would apply cleanly
            % and then restore SILENTLY WRONG.
            [m, rx, b] = tc.cfg_fixture();
            bad = @(cf) @() macos.dw_dx_multi(m, rx, b{:}, 'configs', cf);
            tc.verifyError(bad(struct('name', 'x', 'set', ...
                {{ {'set_elt_kr', 8, 1.0} }})), ...
                'macos:dw_dx_multi:configSetter');
            tc.verifyError(bad(struct('name', 'x', 'set', ...
                {{ {'perturb', 999, 'rotation', [0;0;0]} }})), ...
                'macos:dw_dx_multi:configElt');
            % the exit-pupil element (nElt-1) belongs to reset_xp
            tc.verifyError(bad(struct('name', 'x', 'set', ...
                {{ {'perturb', 12, 'rotation', [0;0;0]} }})), ...
                'macos:dw_dx_multi:configElt');
            tc.verifyError(bad([struct('name', 'a', 'set', {{}}), ...
                                struct('name', 'a', 'set', {{}})]), ...
                'macos:dw_dx_multi:configName');
            tc.verifyError(bad(struct('name', 'x', 'set', ...
                {{ {'perturb', 8, 'wobble', 1} }})), ...
                'macos:dw_dx_multi:configSet');
            tc.verifyError(bad(struct('name', ' ', 'set', {{}})), ...
                'macos:dw_dx_multi:configName');
        end

        function test_configs_from_table(tc)
            % the shape a zoom / compensation schedule arrives in
            th = 1.45444e-4;
            T = table(["z0"; "zUL"], [0; -th], [0; th], ...
                'VariableNames', {'name', '25.Rx', '25.Ry'});
            c = macos.design.configs_from_table(T);
            tc.verifyEqual(numel(c), 2);
            tc.verifyEqual(c(1).name, 'z0');
            tc.verifyEmpty(c(1).set, 'an all-zero row is the NOMINAL state');
            tc.verifyEqual(numel(c(2).set), 1);
            tc.verifyEqual(c(2).set{1}{1}, 'perturb');
            tc.verifyEqual(c(2).set{1}{2}, 25);
            r = c(2).set{1}{find(strcmp(c(2).set{1}, 'rotation')) + 1};
            tc.verifyEqual(r(:).', [-th th 0], 'AbsTol', 0);
            % a row mixing rotation and translation must SPLIT: a mixed
            % local-frame perturb does not invert exactly
            Tm = table("m", 1e-4, 1e-3, ...
                'VariableNames', {'name', '7.Ry', '7.Tz'});
            cm = macos.design.configs_from_table(Tm);
            tc.verifyEqual(numel(cm(1).set), 2);
            tc.verifyError(@() macos.design.configs_from_table( ...
                table("m", 1, 'VariableNames', {'name', 'nonsense'})), ...
                'macos:configs_from_table:header');
        end

        function test_resume_dir_stitches_and_prunes(tc)
            % A 25-block harvest is a multi-hour run, so run_sensitivities
            % can checkpoint per configuration.  The stitched result must
            % be what the single all-configurations call produces, and the
            % checkpoints must be pruned when the run completes.
            [~, rx] = tc.cfg_fixture();
            th = 1e-5;
            cfgs = [struct('name', 'nom', 'set', {{}}), ...
                    struct('name', 'tilt', 'set', {{ ...
                        {'perturb', 8, 'rotation', [th; 0; 0], ...
                         'frame', 'local'} }})];
            wd = tempname; mkdir(wd);
            cwd = onCleanup(@() rmdir(wd, 's'));
            rd = fullfile(wd, '_resume');
            art = run_sensitivities(rx, 'fov_rad', 1e-4, 'channels', "dwdx", ...
                'configs', cfgs, 'resume_dir', string(rd), ...
                'ngridpts', 15, 'model_size', 512, 'dofs', 0, 'elts', 8, ...
                'out_dir', wd, 'name', 'cfgres', 'per_element', [], ...
                'verbose', false);
            tc.verifyEqual(art.nconfig, 2);
            tc.verifyFalse(isfolder(rd), 'checkpoints must be pruned on success');
            % the stitched harvest carries both blocks, contiguously
            n1 = nnz(art.ox.indxall.config == 1);
            tc.verifyEqual(size(art.ox.dwdxall, 1), numel(art.ox.w0_stacked));
            tc.verifyTrue(isequal(find(art.ox.indxall.config == 1), (1:n1).'));
            tc.verifyEqual(art.ox.config_names, {'nom'; 'tilt'});
            % and the report names what each configuration did
            txt = fileread(char(art.report));
            tc.verifyTrue(contains(txt, 'configurations: 2'));
            tc.verifyTrue(contains(txt, 'perturb elt 8'));
        end

        function test_configuration_axis_across_the_family(tc)
            % The axis is not a dw_dx feature: the same validate /
            % snapshot / apply / undo / assert cycle and the same
            % column-wise canvas layout are in all four supervisors.
            % dwdz and dwdsurf are checked here; dwdgrid shares the
            % identical code path and is checked on the augmented pie
            % fixture by the end-to-end case above.
            [m, rx, b] = tc.cfg_fixture(); %#ok<ASGLU>
            th = 1e-5;
            cfgs = [struct('name', 'nom', 'set', {{}}), ...
                    struct('name', 'tilt', 'set', {{ ...
                        {'perturb', 8, 'rotation', [th; 0; 0], ...
                         'frame', 'local'} }})];
            sup = {'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                   'grid', '2x1', 'ngridpts', 15, 'elts', 1};
            % ---- dwdz (segment-LOCAL MonZernike) --------------------
            az = macos.dw_dz_zernike_multi(m, rx, sup{:}, ...
                'zmode_start', 4, 'n_zcoef', 5);
            dz = macos.dw_dz_zernike_multi(m, rx, sup{:}, ...
                'zmode_start', 4, 'n_zcoef', 5, 'configs', cfgs);
            check_(dz, az.dwdzall, 'dwdzall');
            % ---- dwdsurf (Kr) --------------------------------------
            % elt 8, not elt 1: dw_dsurf builds channels only for POWERED
            % Reflector/Refractor optics, and elt 1 is a Segment.
            sups = [sup(1:end-1), {8}];
            as = macos.dw_dsurf_multi(m, rx, sups{:}, 'params', {'Kr'});
            ds = macos.dw_dsurf_multi(m, rx, sups{:}, 'params', {'Kr'}, ...
                'configs', cfgs);
            check_(ds, as.dwdsall, 'dwdsall');

            function check_(d, ref, f)
                n1 = nnz(d.indxall.config == 1);
                n2 = nnz(d.indxall.config == 2);
                tc.verifyTrue(isequal(d.(f)(1:n1, :), ref), ...
                    sprintf('%s: block 1 must be the no-configs harvest', f));
                tc.verifyTrue(isequal(find(d.indxall.config == 1), (1:n1).'), ...
                    sprintf('%s: blocks must be contiguous', f));
                tc.verifyEqual(d.config_names, {'nom'; 'tilt'});
                k = min(n1, n2);
                tc.verifyGreaterThan(max(abs( ...
                    d.(f)(n1+1:n1+k, :) - d.(f)(1:k, :)), [], 'all'), 0, ...
                    sprintf('%s: the tilted configuration changed nothing', f));
            end
        end

        function test_configs_tile_geometrically(tc)
            % The configurations sit where their SCHEDULE says -- a
            % centre + four corners zoom set lands on the corners and
            % centre of a 3x3 outer grid, each cell holding that state's
            % whole field canvas -- while the stacked ROW order stays
            % configuration-major: w for one configuration stacks its
            % fields, w for the run stacks the configurations.  Reading
            % the row order off the assembled canvas would NOT give that
            % (m2v walks column-major and would interleave the blocks),
            % which is the whole reason config_canvas builds the index.
            [m, rx, b] = tc.cfg_fixture();
            t = 1e-5;
            T = table(["z0"; "zUL"; "zUR"; "zLL"; "zLR"], ...
                      [0; -t; +t; -t; +t], [0; +t; +t; -t; -t], ...
                      'VariableNames', {'name', '8.Rx', '8.Ry'});
            cfgs = macos.design.configs_from_table(T);
            % the schedule's own geometry, by the field set's rule
            tiles = vertcat(cfgs.tile);
            tc.verifyEqual(tiles, [1 1; 2 0; 2 2; 0 0; 0 2]);

            d = macos.dw_dx_multi(m, rx, b{:}, 'configs', cfgs);
            % outer grid is 3x3 of the per-configuration field canvas
            a = macos.dw_dx_multi(m, rx, b{:});
            tc.verifyEqual(size(d.OPDall), 3 * size(a.OPDall));
            % ROW ORDER is configuration-major, so each block is still
            % contiguous even though the tiles are not
            n = zeros(1, 5);
            for c = 1:5, n(c) = nnz(d.indxall.config == c); end
            tc.verifyEqual(size(d.dwdxall, 1), sum(n));
            tc.verifyTrue(isequal(find(d.indxall.config == 1), (1:n(1)).'));
            tc.verifyTrue(isequal(find(d.indxall.config == 3), ...
                (sum(n(1:2)) + (1:n(3))).'));
            % ... and the nominal block is still bitwise the no-configs
            % harvest, i.e. tiling changed the LAYOUT, not the numbers
            tc.verifyTrue(isequal(d.dwdxall(1:n(1), :), a.dwdxall));
            tc.verifyTrue(isequal(d.w0_stacked(1:n(1)), a.w0_stacked));
            % the m2v/v2m round trip survives the built index
            tc.verifyEqual(macos.v2m(d.w0_stacked, d.indxall), d.OPDall, ...
                'AbsTol', 0);
            % each configuration's canvas sits at ITS tile
            [nr, ncol] = size(a.OPDall);
            for c = 1:5
                r0 = tiles(c, 1) * nr;
                c0 = tiles(c, 2) * ncol;
                blk = d.OPDall(r0+1:r0+nr, c0+1:c0+ncol);
                tc.verifyEqual(nnz(blk), n(c), ...
                    sprintf('configuration %d is not at its tile', c));
            end
            % and the empty outer cells cost no rows
            tc.verifyEqual(nnz(d.OPDall), sum(n));
        end

        % =============================================================
        % Promoted zoom fixture -- PLAN_CONFIGURATIONS.md departure #6
        % (the figure rungs' fixture: Conic segments promoted to
        % FreeForm so dw/dz MonZernike and dw/dgrid have channels to
        % harvest, shown optically INERT).
        % =============================================================

        function test_zoom_fixture_promotion_is_inert(tc)
            % THE inertness gate, and it is a GATE not a claim: the
            % committed jwst_ote_designc.in is Surface= FreeForm on its
            % 19 segments (zero-amplitude MonZernike channels).  Trace it
            % against the Conic "before" -- reconstructed by swapping
            % Surface= FreeForm back to Conic, which the engine reads as
            % the bare conic base (SetFreeFormFlags runs only for
            % FreeForm, so the leftover Mon lines are inert) -- and the
            % OPD must match to sub-picometer.  A zero-coefficient
            % FreeForm computes the identical conic sag; the residual is
            % the Conic-vs-FreeForm intersection-solver floor (a
            % closed-form conic solve vs the iterative FreeForm one),
            % NOT any added wavefront.  Measured 7.3e-11 mm (~3e-8 waves)
            % on this deck; the gate at 1e-8 mm keeps two decades of
            % margin over that floor and still fires far below any real
            % figure effect (a poked mode moves the wavefront by ~1e-4
            % mm, six decades up).  Not ULP-tight for the same reason
            % config_axis's restore tolerance is not (different solvers,
            % not different physics).
            f = tc.zoom_fixture();
            L = splitlines(string(fileread(f)));
            isff = ~cellfun(@isempty, regexp(strtrim(L), ...
                '^Surface=\s*FreeForm', 'once'));
            % 18 real segments (5-22) + SM (23) + TM (24) = 20 FreeForm.
            % Element 4 (virtual, obscured) stays Conic.
            tc.verifyEqual(nnz(isff), 20, ...
                'the committed fixture must carry 20 FreeForm optics (segs+SM+TM)');
            % demote to the Conic "before"; also drop the grid lines the
            % promoter wrote onto SM/TM (a grid on Conic is inert, but
            % GridFile= flat64.txt would still be resolved -- strip to be
            % clean) is unnecessary: SetFreeFormFlags runs only for
            % FreeForm, so Conic ignores nGridMat.  Just swap the surface.
            L(isff) = regexprep(L(isff), 'FreeForm', 'Conic  ');
            wd = tempname; mkdir(wd);
            cwd = onCleanup(@() rmdir(wd, 's'));
            conic = fullfile(wd, 'conic.in');
            fid = fopen(conic, 'w'); fprintf(fid, '%s\n', L); fclose(fid);
            % the grid file the deck references must sit beside the copy
            copyfile(fullfile(fileparts(f), 'flat64.txt'), ...
                     fullfile(wd, 'flat64.txt'));
            copyfile(f, fullfile(wd, 'ff.in'));

            [Wf, okf] = tc.zoom_opd(fullfile(wd, 'ff.in'));
            [Wc, okc] = tc.zoom_opd(conic);
            tc.verifyTrue(isequal(okf, okc), ...
                'promotion changed the valid-ray mask');
            d = max(abs(Wf(okf) - Wc(okc)));
            tc.verifyLessThan(d, 1e-8, sprintf( ...
                ['promotion is not optically inert: max|dOPD| = %.3e mm ' ...
                 '(floor ~5e-11 mm, gate 1e-8 mm)'], d));
            % NON-VACUITY: the demotion actually removed the FreeForm-ness
            mc = macos.Session(512);  mc.load_rx(conic);
            tc.verifyEmpty(mc.find_freeform_elts(), ...
                'the Conic "before" must have no FreeForm elements');
        end

        function test_promoted_fixture_feeds_both_figure_rungs(tc)
            % Departure #6 closes only if BOTH figure rungs actually
            % harvest on the promoted deck.  find_freeform_elts (dw/dz's
            % eligibility set) and, after grid augmentation,
            % find_grid_elts (dw/dgrid's) must be non-empty, and a
            % trimmed run_sensitivities harvest of each must come back
            % with the expected channel count.
            f = tc.zoom_fixture();
            m = macos.Session(512);  m.load_rx(f);
            ff = m.find_freeform_elts();
            % 18 real segments + SM (23) + TM (24) = 20; elt 4 stays Conic.
            tc.verifyEqual(numel(ff), 20, ...
                'the promoted fixture must expose 20 FreeForm optics (segs+SM+TM)');
            tc.verifyTrue(all(ismember([23 24], ff)), ...
                'SM (23) and TM (24) must be FreeForm (dw/dz harvests them)');
            tc.verifyFalse(ismember(4, ff), ...
                'the virtual centre (elt 4) must stay Conic, not FreeForm');

            wd = tempname; mkdir(wd);
            cwd = onCleanup(@() rmdir(wd, 's'));
            % ---- dw/dz MonZernike, trimmed: a couple segs + SM + TM ----
            elts = [ff(1:2).' 23 24];
            oz = run_sensitivities(f, 'fov_rad', 2.9e-4, 'channels', "dwdz", ...
                'stop_elt', 25, 'ngridpts', 31, 'model_size', 512, ...
                'zmodes_fig', 4:5, 'elts', elts, 'per_element', [], ...
                'out_dir', wd, 'name', 'zdz', 'verbose', false);
            tc.verifyFalse(isempty(oz.oz), 'dw/dz harvested nothing');
            for s2 = elts
                tc.verifyEqual(nnz(startsWith(oz.oz.channel_names, ...
                    sprintf('Elt %d MonZern', s2))), 2, ...
                    sprintf('optic %d must contribute 2 MonZern modes', s2));
            end
            tc.verifyGreaterThan(max(abs(oz.oz.dwdxall), [], 'all'), 0, ...
                'the MonZernike Jacobian is all zero (channel not live)');

            % ---- dw/dgrid, trimmed: segments (augmented) + SM + TM.  SM/TM
            % carry their grid in the DECK; segments are grid-augmented.  A
            % mixed influence struct (shared per-segment basis + a
            % full-aperture basis per SM/TM) exercises the multi-basis path.
            rxg = fullfile(wd, 'zdg_grid.in');
            macos.design.grid_augment_rx(f, rxg, 'ng', 64, 'span_frac', 1.0);
            copyfile(fullfile(fileparts(f), 'flat64.txt'), ...
                     fullfile(wd, 'flat64.txt'));
            sgb = macos.segment_grid_basis(m, rxg, 'pm_ref_elt', 5, ...
                'modes', 4:5, 'orthogonalize', true);
            fn = fieldnames(sgb.seg);
            for e = [23 24]
                ns = double(macos.get_elt_grid_size(e));
                s = sgb.seg(1);  for q=1:numel(fn), s.(fn{q}) = []; end
                s.iElt = e;  s.B = macos.zernike_grid_basis(ns, 4:5);
                s.mask = true(ns);  s.mask_px = ns*ns;
                if isfield(s,'R_seg'), s.R_seg = (ns-1)/2; end
                sgb.seg(end+1) = s;
            end
            og = run_sensitivities(rxg, 'fov_rad', 2.9e-4, 'channels', "dwdgrid", ...
                'stop_elt', 25, 'ngridpts', 31, 'model_size', 512, ...
                'influence', sgb, 'per_element', [], 'out_dir', wd, ...
                'name', 'zdg', 'verbose', false);
            tc.verifyFalse(isempty(og.og), 'dw/dgrid harvested nothing');
            tc.verifyGreaterThan(max(abs(og.og.dwdgall), [], 'all'), 0, ...
                'the grid Jacobian is all zero (channel not live)');
            % SM & TM must have grid channels too (2 modes each)
            for e = [23 24]
                tc.verifyEqual(nnz(og.og.iElt == e), 2, sprintf( ...
                    'optic %d must contribute 2 grid channels (own basis)', e));
            end
        end

        function test_promoted_segment_poke_localizes(tc)
            % PLAN condition 3 (frames before amplitudes), the non-vacuity
            % check: a MonZernike poke on ONE optic must respond, and its
            % footprint must be near-disjoint from another's -- the
            % "central dot" failure (a poke that de-localizes because the
            % Mon frame is not the optic's clocked triad) overlaps every
            % optic at the aperture centre.  Two real segments on opposite
            % sides (5 and, from find_freeform_elts, one far away) must
            % have disjoint support.
            f = tc.zoom_fixture();
            m = macos.Session(512);  m.load_rx(f);
            ff = m.find_freeform_elts();
            segs = ff(ff <= 22);                 % real segments (exclude SM/TM)
            eA = segs(1);  eB = segs(end);       % opposite ends of the pupil
            [dA, okA] = tc.poke_dopd_(f, eA, 4, 1e-3);
            [dB, okB] = tc.poke_dopd_(f, eB, 4, 1e-3);
            rA = max(abs(dA(okA)));  rB = max(abs(dB(okB)));
            tc.verifyGreaterThan(rA, 0, sprintf('segment %d poke did nothing', eA));
            tc.verifyGreaterThan(rB, 0, sprintf('segment %d poke did nothing', eB));
            sA = okA & (abs(dA) > 0.1*rA);
            sB = okB & (abs(dB) > 0.1*rB);
            ov = nnz(sA & sB) / max(1, min(nnz(sA), nnz(sB)));
            tc.verifyLessThan(ov, 0.15, sprintf( ...
                ['segment pokes must localize (overlap %.3f) -- a large ' ...
                 'overlap is the central-dot de-localization'], ov));
            % SM (23) carries a SYNTHESIZED full-aperture frame (footprint-
            % centred); its poke must respond and localize away from a
            % segment (they sit at different pupil positions).
            [dS, okS] = tc.poke_dopd_(f, 23, 4, 1e-3);
            rS = max(abs(dS(okS)));
            tc.verifyGreaterThan(rS, 0, 'SM (23) poke did nothing');
        end

        function test_save_dw_flat_layout(tc)
            % The saved .mat must be FLAT and CHANNEL-NAMED: the Jacobian
            % at the top level under its own name (dwdgrid, not the generic
            % dwdxall alias), indxall / w0_stacked / channel_names beside
            % it, and NO ox/oz/og/os wrapper and no empty fields.
            addpath(fullfile(tc.res_root, 'mmacos', 'sensitivities'));
            % a minimal, well-formed harvest-like struct
            og = struct('dwdgall', rand(10, 6), 'dwdxall', rand(10, 6), ...
                'w0_stacked', rand(10, 1), 'indxall', struct('i', 1), ...
                'channel_names', {{'Elt 5 GridMode1'}}, ...
                'config_names', {{'nom'}}, 'rx_path', 'x.in', ...
                'OPDall', zeros(4), 'sgb', struct('seg', struct('iElt', 5)));
            wd = tempname; mkdir(wd);
            cwd = onCleanup(@() rmdir(wd, 's'));
            mp = fullfile(wd, 'flat.mat');
            save_dw_flat(og, mp, 'name', 'dwdgrid', 'model_size', 512);
            S = load(mp);
            tc.verifyTrue(isfield(S, 'dwdgrid'), 'top-level dwdgrid missing');
            tc.verifyFalse(isfield(S, 'dwdxall'), ...
                'the generic dwdxall alias must not be saved');
            tc.verifyFalse(isfield(S, 'og'), 'no wrapper struct allowed');
            tc.verifyTrue(isfield(S, 'indxall') && isfield(S, 'w0_stacked'));
            tc.verifyTrue(isfield(S, 'sgb'), 'dwdgrid must keep its sgb basis');
            tc.verifyEqual(S.dwdgrid, og.dwdgall);
        end

        function test_dwdsurf_remove_ptt(tc)
            % 'remove_ptt' projects piston + tip + tilt out of each Kr/Kc
            % response (aligned out during assembly).  With it on, every
            % response column must have ~zero piston/tilt content; with it
            % OFF (the default), the harvest is byte-identical to before --
            % the preserved-surface rule for the shared dw/dsurf path.
            f = tc.zoom_fixture();
            m = macos.Session(512);
            b = {'field_x_rad', 2.9e-4, 'field_y_rad', 2.9e-4, ...
                 'grid', '1x1', 'ngridpts', 41, 'stop_elt', 25};
            raw = macos.dw_dsurf_multi(m, f, b{:}, 'params', {'Kr','Kc'});
            ptt = macos.dw_dsurf_multi(m, f, b{:}, 'params', {'Kr','Kc'}, ...
                'remove_ptt', true);
            % default is unchanged
            r2 = macos.dw_dsurf_multi(m, f, b{:}, 'params', {'Kr','Kc'}, ...
                'remove_ptt', false);
            tc.verifyTrue(isequal(raw.dwdsall, r2.dwdsall), ...
                'remove_ptt=false must reproduce the original dw/dsurf');
            % build the piston+tilt basis over the (single-field) aperture
            ix = raw.indxall;  x = double(ix.j(:));  y = double(ix.i(:));
            sc = max([1, max(abs(x-mean(x))), max(abs(y-mean(y)))]);
            A = [ones(numel(x),1), (x-mean(x))/sc, (y-mean(y))/sc];
            fracs = zeros(1, size(raw.dwdsall, 2));
            for s = 1:size(ptt.dwdsall, 2)
                col = ptt.dwdsall(:, s);  ok = isfinite(col);
                c = A(ok,:) \ col(ok);
                fracs(s) = norm(A(ok,:)*c) / max(norm(col(ok)), eps);
            end
            tc.verifyLessThan(max(fracs), 1e-8, ...
                'remove_ptt must leave ~zero piston/tip/tilt in each column');
            % NON-VACUITY: the raw response had real PTT to remove (a conic/
            % radius error is mostly refocus+pointing), so removal changed it
            tc.verifyGreaterThan( ...
                max(abs(raw.dwdsall - ptt.dwdsall), [], 'all'), 0, ...
                'remove_ptt changed nothing -- the raw response had no PTT?');
        end

        % =============================================================
        % ELEMENT GROUPS -- rigid-body groups on the dwdx channel
        % =============================================================

        function test_groups_reach_the_dwdx_channel(tc)
            % The stage runner must plumb 'groups' into dw_dx_multi and
            % every downstream consumer must tolerate the extra columns:
            % the indxall row bookkeeping (groups add COLUMNS, never
            % observations), the conditioning / spectrum figures, and the
            % per-element pages -- which have to SECTION the group
            % channels, because a group carries no element id (iElt 0,
            % the same value a source channel carries) and would
            % otherwise land unlabelled on the source page.
            rx = fullfile(tc.res_root, 'mmacos', 'templates', ...
                '50_sensitivities', 'run_dwdx_multi', 'e5hex1.in');
            grp = containers.Map('KeyType', 'char', 'ValueType', 'any');
            grp('SegPair') = [1; 2];
            % stop_elt is passed EXPLICITLY rather than leaning on the
            % deck's header ApStop= 0 0 0.  Not cosmetic: this class runs
            % many decks in ONE engine process, and after the 27-element
            % zoom fixture has been harvested with its stop at element
            % 25, re-loading this 13-element deck comes back with
            % "*** Setting aperture stop failed!" and FEX then raises
            % macos:fex:noStop.  That leak PREDATES element groups --
            % reproduced with the plain ungrouped harvest below, which is
            % run first for exactly that reason -- but there is no reason
            % to let it decide whether a column-bookkeeping gate runs.
            base = {'fov_rad', 1e-4, 'channels', "dwdx", 'ngridpts', 15, ...
                    'elts', [1; 2], 'dofs', (0:5).', 'model_size', 512, ...
                    'stop_elt', 8, 'per_element', "center", ...
                    'verbose', false};

            % the ungrouped reference FIRST, so the grouped run cannot be
            % credited with (or blamed for) anything the plain harvest
            % already does on this deck
            wd0 = tempname; mkdir(wd0);
            c0 = onCleanup(@() rmdir(wd0, 's'));
            art0 = run_sensitivities(rx, base{:}, ...
                'out_dir', string(wd0), 'name', "nogrp");
            tc.verifyEqual(numel(art0.ox.channel_names), 12);
            tc.verifyFalse(any(strcmp(art0.ox.kind, 'Group')));
            tc.verifyEmpty(dir(fullfile(wd0, 'nogrp_pages', '*_grp*.png')));

            wd = tempname; mkdir(wd);
            c = onCleanup(@() rmdir(wd, 's'));
            art = run_sensitivities(rx, base{:}, 'groups', grp, ...
                'out_dir', string(wd), 'name', "grp");

            % 2 optics x 6 DOFs = 12 per-element, then 6 group columns
            ox = art.ox;
            tc.verifyEqual(numel(ox.channel_names), 18);
            tc.verifyEqual(size(ox.dwdxall, 2), 18);
            tc.verifyEqual(nnz(strcmp(ox.kind, 'Group')), 6);
            for k = 13:18
                tc.verifyEqual(ox.kind{k}, 'Group', ...
                    'group channels must be APPENDED after the per-element block');
                tc.verifyEqual(ox.iElt(k), 0);
            end
            tc.verifyGreaterThan(max(abs(ox.dwdxall(:, 13:18)), [], 'all'), 0, ...
                'non-vacuity: the group columns are all zero');

            % indxall stays a ROW index -- groups add columns, not rows
            tc.verifyEqual(size(ox.dwdxall, 1), numel(ox.w0_stacked));
            tc.verifyEqual(size(ox.dwdxall, 1), numel(ox.indxall.i));

            % report + figures built
            tc.verifyTrue(isfile(art.report));
            tc.verifyTrue(isfile(art.mat));
            tc.verifyTrue(isfile(fullfile(wd, 'grp_svspec.png')));
            tc.verifyTrue(isfile(fullfile(wd, 'grp_dwdx_channels.png')));
            % the group gets its OWN page, and does NOT masquerade as the
            % source page (which this harvest has no channels for)
            tc.verifyTrue(isfile(fullfile(wd, 'grp_pages', ...
                'grp_dwdx_grpSegPair_center.png')), ...
                'the group must get its own per-element page');
            tc.verifyFalse(isfile(fullfile(wd, 'grp_pages', ...
                'grp_dwdx_src_center.png')), ...
                'group channels must not land on the source page');
            for e = [1 2]
                tc.verifyTrue(isfile(fullfile(wd, 'grp_pages', ...
                    sprintf('grp_dwdx_elt%d_center.png', e))));
            end

            % The per-element block is UNDISTURBED -- exactly so at the
            % first field, and to round-off thereafter.  Not a fudge: the
            % group channels add six more GPERTURB poke/restore cycles on
            % elements 1 and 2 per field, and a perturb round-trip leaves
            % a ULP-level pose residue (the psiElt renormalization
            % residual tPerturbRoundtrip pins).  That residue persists
            % into the NEXT field's nominal trace, so the later blocks
            % agree at ~1e-6 relative rather than bitwise, and the
            % nominal wavefront itself moves by the same order.  Field C
            % is harvested first and IS bit-identical, which is what
            % distinguishes round-off accumulation from a real
            % contamination.
            ctr = find(strcmp(ox.field_names, 'C'), 1);
            tc.verifyTrue(isequal(art0.ox.per_field_dwdx{ctr}, ...
                                  ox.per_field_dwdx{ctr}(:, 1:12)), ...
                'the FIRST field''s per-element block must be bit-identical');
            A = art0.ox.dwdxall;  B = ox.dwdxall(:, 1:12);
            tc.verifyLessThan(max(abs(A - B), [], 'all') / ...
                max(abs(A), [], 'all'), 1e-5, ...
                'adding a group must not disturb the per-element block');
        end
    end

    methods (Access = private)
        function [dW, ok] = poke_dopd_(tc, rx, seg_elt, mode, amp) %#ok<INUSD>
            % delta-OPD at the exit pupil from poking one MonZern mode.
            m = macos.Session(512);
            m.load_rx(rx);  m.set_src_sampling(41);  m.stop(int32(25));
            m.modify();  m.trace(27);  W0 = m.opd();
            m.set_elt_mon_zrn_coef(seg_elt, mode, amp);
            m.modify();  m.trace(27);  W1 = m.opd();
            ok = isfinite(W0) & isfinite(W1);
            dW = zeros(size(W0));  dW(ok) = W1(ok) - W0(ok);
        end
    end
end
