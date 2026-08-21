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
    end
end
