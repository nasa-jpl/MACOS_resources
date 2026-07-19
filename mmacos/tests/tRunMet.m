classdef tRunMet < matlab.unittest.TestCase
    % MET stage runner (design/runners/run_met) + the shape-class
    % layout optimizer (macos.design.met_layout_opt) + the .in-file
    % rehydrator (macos.design.seg_from_rx), on the e5 PIE fixture
    % (center hexagon + 6 wedges = TWO shape classes).
    %
    % Checks:
    %   (1) seg_from_rx rebuilds frames/tiling from the .in alone
    %       (== the segment_rx in-memory struct)
    %   (2) met_layout_opt discovers the two boundary-congruence
    %       classes and REPLICATES one pattern per class: each wedge's
    %       launchers are IDENTICAL in its own face frame (Dave's
    %       shape-class rule -- congruent hardware)
    %   (3) run_met end-to-end (as-built + optimize with a trimmed
    %       grid): artifacts on disk, engine-FD == analytic dldx gate,
    %       Monte-Carlo == analytic merit
    properties
        seg
        jac
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
            % synthetic sensitivities .mat: a dw_dx_multi-shaped struct
            % (random but fixed) -- the runner's merit/products path is
            % pure linear algebra over it, so this exercises the code
            % without a multi-field harvest
            rng(11);
            elts = [tc.seg.seg_elts, 8, 11];      % m2 hub, fpa aft
            dofs = {'Rx','Ry','Rz','Tx','Ty','Tz'};
            cn = {};
            for k = [elts, 9]                      % +1 unsensed element
                for d = 1:6, cn{end+1} = sprintf('Elt %d %s', k, dofs{d}); end
            end
            ox = struct('dwdxall', randn(400, numel(cn)), ...
                        'channel_names', {cn});
            ox.dwdxall(3, :) = NaN;                % a masked field row
            % per-field blocks so run_met's tilt/non-tilt split runs:
            % one 20x20 field, column-major pixel order == row order
            ox.per_field_dwdx = {ox.dwdxall};
            ox.per_field_w_nom_2d = {zeros(20, 20)};
            tc.jac = ox;
        end
    end

    methods (Test)
        function test_seg_from_rx_rehydrates(tc)
            old = cd(tc.seg.run.workdir); restore = onCleanup(@() cd(old));
            rh = macos.design.seg_from_rx(tc.seg.in, 'hx', tc.seg.hx);
            tc.verifyEqual(rh.nseg, tc.seg.nseg);
            tc.verifyEqual(rh.seg_elts, tc.seg.seg_elts);
            tc.verifyEqual(rh.grid, tc.seg.grid);
            tc.verifyEqual(rh.width, tc.seg.width, 'RelTol', 1e-12);
            tc.verifyEqual(rh.gap, tc.seg.gap, 'RelTol', 1e-12);
            for s = 1:rh.nseg
                tc.verifyEqual(rh.frames(s).rpt, tc.seg.frames(s).rpt, ...
                    'AbsTol', 1e-9);
                tc.verifyEqual(rh.frames(s).xhat, tc.seg.frames(s).xhat, ...
                    'AbsTol', 1e-12);
                tc.verifyEqual(rh.frames(s).zhat, tc.seg.frames(s).zhat, ...
                    'AbsTol', 1e-12);
                tc.verifyEqual(rh.frames(s).lmon, tc.seg.frames(s).lmon, ...
                    'RelTol', 1e-9);
            end
        end

        function test_layout_opt_shape_classes(tc)
            old = cd(tc.seg.run.workdir); restore = onCleanup(@() cd(old));
            macos.load_rx(tc.seg.in);
            nb = tc.seg.nseg + 2;
            rng(3);
            D = randn(200, 6*nb);
            E = zeros(1, 6*nb);          % gauge-only sensing suffices here
            X = 1e-12*eye(6*nb);
            % min_sep 30: this fixture's corner launchers sit ~40 mm
            % apart under the trimmed [30 90 150] grid -- 50 mm has NO
            % feasible layout here (that infeasible path is exercised
            % by the optimizer's warning; run_met keeps as-built)
            out = macos.design.met_layout_opt(tc.seg, D, E, X, ...
                'hub', 8, 'aft', 11, 'r_extra', 100, ...
                'sig_edge', 1e-9, 'sig_met', 1e-9, ...
                'edge_off', 5, 'min_sep', 30, ...
                'families', "spread", ...
                'phi_grid', deg2rad([30 90 150]), 'nf_grid', 3, ...
                'fclk_coarse', 0, 'fclk_fine', deg2rad([0 30]), ...
                'nrefine', 2, 'topk', 2, 'verbose', false);
            % (2a) two classes: the center hexagon + the 6 wedges
            tc.verifyEqual(numel(out.classes), 2);
            nm = sort(cellfun(@numel, out.classes));
            tc.verifyEqual(nm, [1 6]);
            % (2b) shape-class replication: every wedge's launchers are
            % the SAME points in its own face frame (congruent parts).
            % Tolerance = the classifier's own congruence scale
            % (1e-3 of the boundary radius): tilted member frames
            % projected onto the flat tiling plane pick up physically
            % real cos(tilt) terms at the ~0.1 mm level on this
            % fixture -- congruent hardware, not an error.
            wc = out.classes{cellfun(@numel, out.classes) == 6};
            q0 = [];
            for s = wc(:).'
                f = tc.seg.frames(s);
                q = [f.xhat, f.yhat].' * (out.launch_pts{s} - f.rpt);
                if isempty(q0)
                    q0 = q;
                    tol = 1e-3 * max(vecnorm(q0));
                else
                    tc.verifyEqual(q, q0, 'AbsTol', tol, ...
                        'wedge patterns must be congruent in the segment frames');
                end
            end
            % (2c) the optimizer never loses to its own baseline
            tc.verifyLessThanOrEqual(out.rb, out.r0*(1 + 1e-12));
            % (2d) launcher separation gate holds on the winner
            tc.verifyTrue(isfinite(out.rb), 'a feasible layout must exist');
            LP = [out.launch_pts{:}];
            n = size(LP, 2);
            Dm = squeeze(vecnorm(reshape(LP,3,1,n) - reshape(LP,3,n,1)));
            Dm(1:n+1:end) = inf;
            tc.verifyGreaterThanOrEqual(min(Dm(:)), 30);
            % (2e) preset round-trip: the scale-free export re-applied
            % to the SAME build must reproduce the winner exactly
            ap = macos.design.met_layout_opt(tc.seg, 'apply', out.preset, ...
                'hub', 8, 'aft', 11, 'r_extra', 100, 'edge_off', 5, ...
                'min_sep', 30, 'verbose', false);
            tc.verifyEqual([ap.launch_pts{:}], LP, 'AbsTol', 1e-9, ...
                'preset apply must reproduce the winner launchers');
            tc.verifyEqual(ap.best.rfid, out.best.rfid, 'RelTol', 1e-12);
            tc.verifyEqual(ap.src_aft, out.src_aft, 'AbsTol', 1e-9);
        end

        function test_run_met_end_to_end(tc)
            wd = tempname; mkdir(wd);
            cwd = onCleanup(@() rmdir(wd, 's'));
            here = fileparts(mfilename('fullpath'));
            res_root = fileparts(fileparts(here));
            copyfile(tc.seg.in, fullfile(wd, 'pie.in'));
            copyfile(tc.seg.hx, fullfile(wd, 'pieHx.m'));
            copyfile(fullfile(res_root, 'segmirmaker', 'test_in', ...
                'flat.txt'), fullfile(wd, 'flat.txt'));
            art = run_met(fullfile(wd, 'pie.in'), ...
                'hx', fullfile(wd, 'pieHx.m'), 'jac', tc.jac, ...
                'hub', 8, 'aft', 11, 'r_extra', 100, ...
                'min_sep', 30, ...
                'sig_rot', 1e-6, 'sig_trans', 1e-6, ...
                'sig_edge', 1e-9, 'sig_met', 1e-9, ...
                'mc', 50, 'verbose', false, ...
                'opt', struct('families', "spread", ...
                    'phi_grid', deg2rad([30 90 150]), 'nf_grid', 3, ...
                    'fclk_coarse', 0, 'fclk_fine', deg2rad([0 30]), ...
                    'nrefine', 2, 'topk', 2));
            % artifacts on disk
            tc.verifyTrue(isfile(art.met_in));
            tc.verifyTrue(isfile(art.metopt_in));
            tc.verifyTrue(isfile(art.mat));
            tc.verifyTrue(isfile(art.report));
            for k = 1:numel(art.figs)
                tc.verifyTrue(isfile(art.figs{k}), art.figs{k});
            end
            % engine-FD == analytic dldx gate
            tc.verifyLessThan(art.gate, 5e-3);
            % merit sanity: sensing helps, optimizer never hurts
            tc.verifyLessThan(art.merits.edge_met, art.merits.prior);
            tc.verifyLessThanOrEqual(art.opt.rb, art.opt.r0*(1 + 1e-12));
            % engine-FD merit validation of the winner is tight
            tc.verifyLessThan(abs(art.rfd - art.opt.rb)/art.opt.rb, 0.02);
            % Monte-Carlo acceptance vs the analytic trace
            Hn = [art.dedx; art.dldx_opt];
            Rn = blkdiag(art.Re, 1e-18*eye(size(art.dldx_opt, 1)));
            Dk = art.dwdx(art.keep, :);
            Kn = (art.X*Hn') / (Hn*art.X*Hn' + Rn);
            ana = sqrt(trace((art.X - Kn*Hn*art.X)*(Dk'*Dk))/nnz(art.keep));
            tc.verifyLessThan(abs(art.mc - ana)/ana, 0.15);
            % named products dimensioned for the compare/simulate stages
            tc.verifySize(art.dxde, [6*9, size(art.dedx, 1)]);
            tc.verifySize(art.dwdl, [nnz(art.keep), size(art.dldx, 1)]);
        end
    end
end
