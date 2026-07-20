classdef tRunCompare < matlab.unittest.TestCase
    % Compare stage runner (design/runners/run_compare): engine-vs-
    % linear poke comparison on the e5 PIE fixture with a REAL
    % (single-field, coarse-grid) dw_dx_multi harvest -- the agreement
    % gates ARE the physics check (truth minus Jacobian*poke = the
    % linearization error, which must be small at 100 nrad / 100 nm).
    %
    % Checks:
    %   (1) artifacts on disk (mat / report / gif / one frame per poke)
    %   (2) w agreement: engine OPD delta == dwdx column * poke to <5%
    %   (3) l agreement: engine METcalc delta == dldx column * poke
    %   (4) e: segment pokes move the edge sensors and the
    %       finite-rotation truth matches the dedx rows; hub pokes
    %       read exactly zero (sensors live on segments)
    %   (5) dwdu export dimensioned for the simulator stage
    properties
        seg
        wd
        met5      % run_met artifact struct
        ox        % real single-field dw_dx_multi harvest
        oz        % real single-field dw_dz (monzern) harvest
        og        % real single-field dw_dgrid harvest (augmented Rx)
    end

    methods (TestClassSetup)
        function make(tc)
            here = fileparts(mfilename('fullpath'));
            res_root = fileparts(fileparts(here));
            addpath(fullfile(res_root, 'mmacos', 'design', 'runners'));
            tin = fullfile(res_root, 'segmirmaker', 'test_in');
            bin = fullfile(res_root, 'segmirmaker', ...
                           'build_release_ifx', 'SegMirMaker');
            tc.assumeTrue(isfile(bin), 'SegMirMaker not built');
            macos.init(512);
            tc.seg = macos.design.segment_rx(fullfile(tin, 'e5mono.in'), ...
                'elt', 1, 'rings', 1, 'grid', 'Pie', 'gap', 50, ...
                'dofs', 6, 'meas_config', 1);
            tc.wd = tempname; mkdir(tc.wd);
            copyfile(tc.seg.in, fullfile(tc.wd, 'pie.in'));
            copyfile(tc.seg.hx, fullfile(tc.wd, 'pieHx.m'));
            copyfile(fullfile(tin, 'flat.txt'), fullfile(tc.wd, 'flat.txt'));
            % MET stage without merit (no jac): emits pie_met.in +
            % dedx/dldx/bodies/seg -- the met struct run_compare consumes
            tc.met5 = run_met(fullfile(tc.wd, 'pie.in'), ...
                'hx', fullfile(tc.wd, 'pieHx.m'), 'jac', '', ...
                'hub', 8, 'aft', 11, 'r_extra', 100, 'min_sep', 30, ...
                'ngridpts', 15, 'mc', 0, 'verbose', false);
            % REAL center-field rigid-body harvest on the met Rx, same
            % coarse grid run_compare will trace ('grid','1x1' = the
            % center field only; fp_mode none -- no ApStop in the SMM
            % corpus, and no sensed body is the FP)
            m = macos.Session(512);
            tc.ox = macos.dw_dx_multi(m, char(tc.met5.met_in), ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                'grid', '1x1', 'fp_mode', 'none', 'ngridpts', 15);
            tc.oz = macos.dw_dz_zernike_multi(m, char(tc.met5.met_in), ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                'grid', '1x1', 'kinds', {'monzern'}, ...
                'zmode_start', 4, 'n_zcoef', 5, ...
                'reset_xp', false, 'ngridpts', 15);
            % grid leg: grid-augmented Rx + a 2-mode per-segment basis.
            % 'zmodes' MUST match the sgb build -- run_compare rebuilds
            % the basis from og.zmodes (the echo, not the sgb itself)
            old = cd(tc.wd); restore = onCleanup(@() cd(old));
            grx = fullfile(tc.wd, 'pie_grid.in');
            macos.design.grid_augment_rx(fullfile(tc.wd, 'pie.in'), ...
                grx, 'ng', 128);
            sgb = macos.segment_grid_basis(m, char(grx), ...
                'pm_ref_elt', 1, 'modes', 4:5, 'orthogonalize', true);
            tc.og = macos.dw_dgrid_multi(m, char(grx), ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                'grid', '1x1', 'influence', sgb, 'zmodes', 4:5, ...
                'reset_xp', false, 'ngridpts', 15);
            % persist the harvest basis exactly as run_sensitivities
            % does -- run_compare must consume og.sgb verbatim (a
            % rebuild can rotate the last G-S mode across sessions)
            tc.og.sgb = sgb;
        end
        function cleanupdir(tc)
            tc.addTeardown(@() rmdir(tc.wd, 's'));
        end
    end

    methods (Test)
        function test_poke_agreement_end_to_end(tc)
            M = load(tc.met5.mat);
            art = run_compare(fullfile(tc.wd, 'pie.in'), ...
                'hx', fullfile(tc.wd, 'pieHx.m'), ...
                'jac', struct('ox', tc.ox, 'oz', tc.oz, 'og', tc.og), ...
                'met', M, ...
                'bodies', [1 2 8 9], 'dofs', [2 6], ... % Ry + Tz of two
                'dwell', 0, 'visible', false, ...       % segments + hub
                'ngridpts', 15, 'verbose', false, ...   % + the FPA-riding
                'out_dir', tc.wd, 'name', 'cmp');       %   aft ring
            % (1) artifacts: 8 x pokes + the z pokes (7 segs x modes
            % 4:5 + any non-segment monzern optic the harvest sweeps
            % in -- the e5 fixture has a FreeForm refractor)
            nz = numel(tc.oz.channel_names);
            ng = numel(tc.og.channel_names);
            tc.verifyTrue(isfile(art.mat));
            tc.verifyTrue(isfile(art.report));
            tc.verifyTrue(isfile(art.gif));
            fr = dir(fullfile(art.frames_dir, 'p*.png'));
            tc.verifySize(fr, [8 + nz + ng, 1]);
            T = art.table(strcmp({art.table.chan}, 'x'));
            tc.verifyNumElements(T, 8);
            % (2) OPD: every real poke's engine map matches dwdx*poke;
            % the aft body rides the FPA (no dwdx channels, no plain-
            % trace wavefront response) -> null-floor rows, w_rel NaN
            is_aft = [T.body] == 11;
            tc.verifyGreaterThan(min([T(~is_aft).w_rms_t]), 0, ...
                'every optic poke must move the wavefront');
            tc.verifyLessThan(max([T(~is_aft).w_rel]), 0.05);
            tc.verifyTrue(all(isnan([T(is_aft).w_rel])), ...
                'FPA pokes must hit the null-response floor');
            % (3) METcalc: beams respond and match dldx*poke
            tc.verifyGreaterThan(min([T.l_max_t]), 0, ...
                'every sensed body carries met hardware');
            tc.verifyLessThan(max([T.l_rel]), 0.01);
            % (4) edge sensors: segment pokes read (finite-rotation
            % truth vs the dedx rows), hub/aft pokes are zero
            is_seg = [T.body] <= 7;
            tc.verifyGreaterThan(min([T(is_seg).e_max_t]), 0);
            tc.verifyLessThan(max([T(is_seg).e_rel]), 1e-3);
            tc.verifyEqual(max([T(~is_seg).e_max_t]), 0);
            % (5) dwdu: control defaults to segments + hub = 8 bodies
            tc.verifyEqual(art.u_bodies, M.bodies(1:8));
            tc.verifySize(art.dwdu, [size(tc.ox.dwdxall, 1), 48]);
            % (6) z pokes: real OPD response matching dwdz*poke; the
            % engine METcalc/Hx are RIGID -> engine l/e read zero
            % (null), while dmdz carries the figure-sensing model
            Tz = art.table(strcmp({art.table.chan}, 'z'));
            tc.verifyNumElements(Tz, nz);
            tc.verifyGreaterThan(min([Tz.w_rms_t]), 0);
            tc.verifyLessThan(max([Tz.w_rel]), 0.05);
            tc.verifyTrue(all(isnan([Tz.l_rel])), ...
                'engine met must not respond to figure (rigid points)');
            tc.verifyGreaterThan(max([Tz.e_max_m]), 0, ...
                'dmdz must predict edge-piston response');
            % (7) dmdz structure: piston rows respond for the poked
            % segment; gap/shear rows are slope-order small (the
            % sensor's in-plane axes are perp to the LOCAL surface
            % normal while the engine Mon sag displaces along the
            % segment FACE normal zMon -- a curved segment projects a
            % few-percent component, real model response, not error)
            es = macos.design.edge_sensors(fullfile(tc.wd, 'pieHx.m'));
            tc.verifySize(art.dmdz, [size(M.dldx, 1) + es.nmeas, nz]);
            ep = art.dedz(es.axis == 1, :);
            eio = art.dedz(es.axis > 1, :);
            tc.verifyGreaterThan(max(abs(ep(:))), 0);
            tc.verifyLessThan(max(abs(eio(:))), 0.05 * max(abs(ep(:))));
            % (8) grid pokes: the harvest's engine response reproduces
            % from the REBUILT basis on a clean reload (the sgb
            % state-pollution + basis-consistency regression gate)
            Tg = art.table(strcmp({art.table.chan}, 'grid'));
            tc.verifyNumElements(Tg, ng);
            tc.verifyGreaterThan(min([Tg.w_rms_t]), 0);
            tc.verifyLessThan(max([Tg.w_rel]), 0.05);
            tc.verifyGreaterThan(max([Tg.e_max_m]), 0, ...
                'dmdgrid must predict edge-piston response');
            tc.verifySize(art.dmdgrid, [size(M.dldx, 1) + es.nmeas, ng]);
        end

        function test_zern_grid_engine_equivalence(tc)
            % THE CONVENTION GATE for zern_seg_eval (dmdz's mode
            % shapes): the same MonZernike mode sampled onto a grid
            % channel and poked via elt_grid_add must reproduce the
            % MonZernCoef poke's OPD -- this pins lMon normalization,
            % the Mon frame, AND the un-normalized-ANSI convention
            % against the engine itself (a NORM_RMS or lMon/Rseg error
            % shows as a large scale mismatch).
            old = cd(tc.wd); restore = onCleanup(@() cd(old));
            ga = macos.design.grid_augment_rx(fullfile(tc.wd, 'pie.in'), ...
                fullfile(tc.wd, 'pie_grid.in'), 'ng', 128);
            m = macos.Session(512);
            m.load_rx(fullfile(tc.wd, 'pie_grid.in'));
            m.set_src_sampling(31);
            s = 2;  elt = tc.seg.seg_elts(s);  mode = 5;
            f = tc.seg.frames(s);
            N = ga.ng;  gdx = ga.gdx(min(s, numel(ga.gdx)));
            c0 = (N + 1) / 2;
            [I, Jj] = ndgrid(1:N, 1:N);
            pts = f.rpt + f.xhat*((I(:).' - c0)*gdx) ...
                        + f.yhat*((Jj(:).' - c0)*gdx);
            map = reshape(macos.design.zern_seg_eval(f, mode, pts), N, N);
            wf = m.num_elt() - 1;
            m.trace(wf);  W0 = m.opd();
            c = 1e-4;                          % 100 nm in mm BaseUnits
            gc = macos.channels.GridChannel(m, elt, map);
            gc.apply(c);  m.trace(wf);  Wg = m.opd();  gc.restore();
            zc = macos.channels.MonZernChannel(m, elt, mode);
            zc.apply(c);  m.trace(wf);  Wz = m.opd();  zc.restore();
            mk = W0 ~= 0 & Wg ~= 0 & Wz ~= 0;
            dg = Wg(mk) - W0(mk);  dz = Wz(mk) - W0(mk);
            sel = abs(dz) > 0.1 * max(abs(dz));   % the segment's support
            tc.verifyGreaterThan(nnz(sel), 10);
            scale = dg(sel) \ dz(sel);
            tc.verifyEqual(scale, 1, 'AbsTol', 2e-2, ...
                'grid-sampled mode must reproduce the MonZern poke');
            % corr floor 0.995: the residual is 128-pt grid bilinear
            % discretization at coarse ray sampling; a frame or
            % convention error collapses it outright
            cc = corrcoef(dg(sel), dz(sel));
            tc.verifyGreaterThan(cc(1, 2), 0.995);
        end
    end
end
