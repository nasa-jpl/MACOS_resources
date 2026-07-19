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
        seg
    end

    methods (TestClassSetup)
        function make(tc)
            here = fileparts(mfilename('fullpath'));
            res_root = fileparts(fileparts(here));
            addpath(fullfile(res_root, 'mmacos', 'design', 'runners'));
            addpath(fullfile(res_root, 'mmacos', 'sensitivities'));
            bin = fullfile(res_root, 'segmirmaker', ...
                           'build_release_ifx', 'SegMirMaker');
            tc.assumeTrue(isfile(bin), 'SegMirMaker not built');
            macos.init(512);
            tin = fullfile(res_root, 'segmirmaker', 'test_in');
            tc.seg = macos.design.segment_rx(fullfile(tin, 'e5mono.in'), ...
                'elt', 1, 'rings', 1, 'grid', 'Pie', 'gap', 50, ...
                'dofs', 6, 'meas_config', 1);
        end
    end

    methods (Test)
        function test_grid_augment_replaces_stale(tc)
            % the SMM fixture ships stale parent-frame grid lines in
            % every segment block: they must be REPLACED by the
            % segment's own clocked Mon frame
            wd = tempname; mkdir(wd);
            cwd = onCleanup(@() rmdir(wd, 's'));
            rxg = fullfile(wd, 'aug.in');
            ga = macos.design.grid_augment_rx(tc.seg.in, rxg, 'ng', 64);
            tc.verifyEqual(ga.nseg, tc.seg.nseg);
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
            tc.verifyEqual(nGrid, tc.seg.nseg);
        end

        function test_run_sensitivities_end_to_end(tc)
            % trimmed harvest on the SMM pie fixture; the regression
            % gate is dwdgrid FULL RANK + localized pokes (the stale
            % frames give rank ~15/42 and center-piled responses)
            wd = tempname; mkdir(wd);
            cwd = onCleanup(@() rmdir(wd, 's'));
            here = fileparts(mfilename('fullpath'));
            res_root = fileparts(fileparts(here));
            copyfile(tc.seg.in, fullfile(wd, 'pie.in'));
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
            tc.verifyEqual(size(art.og.dwdgall, 2), 3*tc.seg.nseg);
            % dwdz sweeps EVERY lMon-bearing optic (non-segment optics
            % included -- the supervisor contract); the segments must
            % each contribute exactly the 3 requested modes
            cnz = art.oz.channel_names;
            for s2 = 1:tc.seg.nseg
                tc.verifyEqual(nnz(startsWith(cnz, ...
                    sprintf('Elt %d MonZern', s2))), 3);
            end
            tc.verifyGreaterThanOrEqual(size(art.oz.dwdxall, 2), ...
                3*tc.seg.nseg);
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
            ns = tc.seg.nseg;  nmode = 3;
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
    end
end
