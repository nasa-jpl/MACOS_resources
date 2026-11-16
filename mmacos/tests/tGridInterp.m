classdef tGridInterp < matlab.unittest.TestCase
%TGRIDINTERP  GridSrfOrder= 3, the C1 bicubic grid interpolation, against the DM model's own influence function.
%   On the e2e coupled deck (metres; 256-node DM grids, 3.8 cells per
%   0.66-mm actuator pitch) the engine's per-ray path change for a DM
%   command is compared with the textbook prediction at the ray's footprint:
%       dOPL = 2 cos(theta) * sum_k a_k exp(ln C (r_k / pitch)^2)
%   (ctb_dm's Gaussian influence function, continuous; the engine sees only
%   its samples on the grid).  Bilinear interpolation (GridSrfOrder= 1, the
%   default) carries the sampled grid's interpolation error; the C1 bicubic
%   (GridSrfOrder= 3, 2026-10-31) a far smaller one.
%
%   Checks (a single actuator at 100 nm, then a rough 10-nm-rms state with
%   four 90-nm actuators):
%     (a) order 3's rms error against the prediction is below a pinned
%         fraction of order 1's, and below a pinned absolute fraction;
%     (b) order 1's error is where the record says (the pin is not vacuous:
%         the same prediction with the pitch planted 10 % wrong misses both);
%     (c) the two orders see the same sign convention.
%   Model 512 (the e2e deck's), SUITE_CTB_512.  The record: macos
%   notes/surfaces.md "Grid surfaces are C0", PLAN_CONSOLIDATION 3a.

    properties (Constant)
        ModelSize = 512
        % Measured 2026-10-31 on the private C1 build (the log line prints them): order 1 vs 3
        % rms error / rms prediction = [3.75e-2 6.04e-3] for one actuator at 100 nm and
        % [3.58e-2 5.69e-3] for the rough state; the pitch planted 10 % wrong reads 1.27e-1 at order 3.
        Ratio13Max  = 0.25    % (a) order 3 error / order 1 error, both cases (measured 0.16)
        Err3Max     = 1e-2    % (a) order 3 rms error / rms prediction (measured 6.0e-3; the
                              %     bicubic's own error at 3.8 cells per pitch plus second-order path terms)
        Err1Min     = 2e-2    % (b) order 1 is at least this far from the continuous surface (measured 3.6e-2)
    end

    properties
        work, rx, e2e
    end

    methods (TestClassSetup)
        function setup(tc)
            root = fullfile(fileparts(mfilename('fullpath')), '..');
            run(fullfile(root, 'mmacos_setup.m'));
            tc.e2e = fullfile(root, 'templates', '80_end_to_end', 'e2e');
            addpath(fullfile(root, 'templates', '30_instruments', 'bench_ctb'));
            tc.rx = fullfile(tc.e2e, 's9_coupled.in');
            tc.assumeTrue(isfile(tc.rx), 'the e2e stage-9 deck is not committed here');
            macos.init(tc.ModelSize);
            tc.work = tempname;  mkdir(tc.work);
            tc.addTeardown(@() rmdir(tc.work, 's'));
            prev = cd(tc.work);  tc.addTeardown(@() cd(prev));   % GridFile= resolves against the cwd
        end
    end

    methods (Static)
        function v = deckvec_(blk, k)
            tk = regexp(blk, [k '=\s*([^\n]*)'], 'tokens', 'once');  v = sscanf(tk{1}, '%f');  v = v(:);
        end
    end

    methods (Test)
        function test_c1_bicubic_tracks_the_influence_function(tc)
            cases = struct('name', {'one actuator 100 nm', 'rough state'}, 'a', {[], []});
            err = zeros(2, 2);  sgn = zeros(2, 2);  errbad = zeros(1, 2);
            for o = [1 3]
                r = ctb_dm_rx('rx_in', tc.rx, 'rx_out', fullfile(tc.work, sprintf('dm_o%d.in', o)), 'order', o);
                macos.load_rx(r.rx_out);
                u_mm = 1e-3 / r.cbm;                                   % mm -> the deck's unit (metres here), as s9 does
                dm = ctb_dm('ielt', r.ielt(1), 'ng', r.ng, 'gdx_mm', r.gdx_mm(1), 'nact', 32, 'beam_d_mm', 21.3 * u_mm);
                dm2 = ctb_dm('ielt', r.ielt(2), 'ng', r.ng, 'gdx_mm', r.gdx_mm(2), 'nact', 32, 'beam_d_mm', 21.3 * u_mm);
                dm2.apply(zeros(32^2, 1));
                blk = regexp(fileread(r.rx_out), 'EltName=\s*DM1.*?nECoord', 'match', 'once');
                gv = @(k) tGridInterp.deckvec_(blk, k);
                psi = gv('psiElt');  pD = gv('pData');  xD = gv('xData');  yD = gv('yData');
                % the path change is read at the first flat exit-pupil Return after the DMs: there the
                % per-ray path is the wavefront error (2 cos(theta) x sag); at the FPA the kinked
                % rays' geometric walk-off dominates it (the stage-11 record's 30 nm -> 250 nm)
                names = regexp(fileread(r.rx_out), '(?m)^\s*EltName=\s*(\S+)', 'tokens');  names = [names{:}];
                iEP = find(strcmp(names, 'Focus23_EPreturn'), 1);  tc.assertNotEmpty(iEP, 'the deck''s first EP Return after the DMs');
                % the commands
                rng(5);  act = find(dm.active(:));
                a1 = zeros(32^2, 1);  [~, c] = min(hypot(dm.acx(:), dm.acy(:)));  a1(c) = 100e-9;
                a2 = 10e-9 * randn(32^2, 1) .* dm.active(:);  big = act(randperm(numel(act), 4));  a2(big) = 90e-9 * sign(randn(4, 1));
                cases(1).a = a1;  cases(2).a = a2;
                % the flat reference and the footprints
                dm.apply(zeros(32^2, 1));  t = macos.trace(iEP);  i0 = macos.get_ray_info(t.nRays);
                macos.trace(r.ielt(1));  rf = macos.get_ray_info(t.nRays);
                ok = i0.ok_trace & rf.ok_trace;
                cth = abs(sum(rf.dir .* repmat(psi(:), 1, t.nRays), 1))';            % cos(incidence) per ray
                p = rf.pos - repmat(pD(:), 1, t.nRays);
                xg = (xD(:)' * p)';  yg = (yD(:)' * p)';                              % footprint in the grid frame
                lnC = log(dm.coupling);  pitch = dm.pitch_mm;
                for cs = 1:2
                    a = cases(cs).a;
                    dm.apply(a);  macos.trace(iEP);  i1 = macos.get_ray_info(t.nRays);
                    d = (i1.opl - i0.opl);  d = d(ok);  d = d - mean(d);
                    for planted = [0 1]
                        pp = pitch * (1 + 0.1 * planted);
                        sag = zeros(t.nRays, 1);
                        for k = find(a(:) ~= 0)'
                            rr2 = (xg - dm.acx(k)).^2 + (yg - dm.acy(k)).^2;
                            in = rr2 <= (3 * pitch)^2;                 % ctb_dm's stamp window, 3 pitches
                            sag(in) = sag(in) + a(k) * exp(lnC * rr2(in) / pp^2);
                        end
                        pred = 2 * cth .* sag;  pred = pred(ok);  pred = pred - mean(pred);
                        ep = norm(d - pred) / norm(pred);  em = norm(d + pred) / norm(pred);
                        if ~planted
                            [err(cs, 1 + (o == 3)), s] = min([ep em]);  sgn(cs, 1 + (o == 3)) = 3 - 2*s;
                        elseif o == 3
                            errbad(cs) = min(ep, em);
                        end
                    end
                end
            end
            fprintf('[grid interp] rms error / rms prediction, order 1 vs 3: %s = [%.3e %.3e], %s = [%.3e %.3e]; signs %s; order 3 with the pitch planted 10 %% wrong [%.3e %.3e]\n', ...
                cases(1).name, err(1, :), cases(2).name, err(2, :), mat2str(sgn), errbad);
            for cs = 1:2
                tc.verifyLessThan(err(cs, 2) / err(cs, 1), tc.Ratio13Max, sprintf('%s: order 3 below order 1 by the pinned ratio', cases(cs).name));
                tc.verifyLessThan(err(cs, 2), tc.Err3Max, sprintf('%s: order 3 tracks the influence function', cases(cs).name));
                tc.verifyGreaterThan(err(cs, 1), tc.Err1Min, sprintf('%s: order 1 carries the sampled grid''s error', cases(cs).name));
                tc.verifyGreaterThan(errbad(cs), 5 * err(cs, 2), sprintf('%s: a planted wrong pitch misses (the check is not vacuous)', cases(cs).name));
            end
            tc.verifyEqual(sgn(:, 1), sgn(:, 2), 'both orders see the same sign convention');
        end
    end
end
