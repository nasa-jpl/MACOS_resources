classdef tResolutionSweep < matlab.unittest.TestCase
%TRESOLUTIONSWEEP  The per-gauge resolution sweep's mechanism (cophi_dm96
%   ressweep_run / ressweep_hook; dm_gauge_lib/dmg_camres).  BRIEF_to_resolution_
%   sweep (2026-10-13).  SUITE_FAST-sized: pure MATLAB, plus reads of the sweep's
%   own records (no engine).
%
%   The operator (dmg_camres: the camera's pixels on the model's detector grid)
%     R1  the IDENTITY: cleared, suspended, or c = 1 with no offset returns the
%         frame itself (isequal) -- the anchor IS the record's code path
%     R2  area integration: the camera cells conserve flux (sum x c^2) for
%         integer AND non-integer c; replication carries one value per cell
%     R3  an offset of a whole camera pixel at c = 2 is the image shifted by
%         two model pixels (the registration row's operator moves the camera,
%         not the image's values)
%   The records (runs/ressweep_<gauge>/ressweep_<gauge>.mat), when present
%     R4  the anchor sampling (3.277, c = 1) reproduces the record's battery
%         rows TO THE DIGIT (the deck's printed precision): IFO = tg_psi_dm96_oap
%         redo96_oapph / camI_oap, HET = cophi_dm96 noise_oap_plain
%     R5  NON-VACUITY: the collapse APPEARS -- at 1 px per actuator the single-
%         actuator row is broken by name (see the test for the criterion), so
%         the binning reached the reading
    properties (Constant)
        LIB = fullfile(fileparts(mfilename('fullpath')), '..', 'templates', '40_benches', 'dm_gauge_lib');
        RUNS = fullfile(fileparts(mfilename('fullpath')), '..', 'templates', '40_benches', 'cophi_dm96', 'runs');
    end
    methods (TestClassSetup)
        function addlib(tc)
            addpath(tResolutionSweep.LIB);
            tc.addTeardown(@() rmpath(tResolutionSweep.LIB));
            tc.addTeardown(@() dmg_camres('clear'));
        end
    end
    methods (Test)
        function R1_identity(tc)
            rng(3);  I = rand(48, 48, 3);
            dmg_camres('clear');                                tc.verifyTrue(isequal(dmg_camres(I), I));
            dmg_camres('set', struct('c', 1));                  tc.verifyTrue(isequal(dmg_camres(I), I));
            dmg_camres('set', struct('c', 2));  dmg_camres('suspend', true);
            tc.verifyTrue(isequal(dmg_camres(I), I));
            dmg_camres('suspend', false);       tc.verifyFalse(isequal(dmg_camres(I), I));
            dmg_camres('clear');
        end
        function R2_flux_and_replication(tc)
            I = zeros(64);  I(17:48, 17:48) = rand(32);         % dark border: no flux leaves the grid
            for c = [1.5 2 3.277]
                [B, T] = dmg_camres('bin', I, struct('c', c, 'offset', [0 0], 'center', []));
                tc.verifyEqual(sum(B(:))*c^2, sum(I(:)), 'RelTol', 1e-12, sprintf('flux at c %.3f', c));
                Ir = dmg_camres('rep', B, T);
                tc.verifyLessThanOrEqual(numel(unique(Ir(Ir ~= 0))), nnz(B), sprintf('one value per cell at c %.3f', c));
            end
        end
        function R3_offset_moves_the_camera(tc)
            I = zeros(64);  I(30:35, 28:33) = 1;
            cf = struct('c', 2, 'offset', [0 0], 'center', [33 33]);
            [B0, T0] = dmg_camres('bin', I, cf);  I0 = dmg_camres('rep', B0, T0);
            cf.offset = [2 0];                                  % one camera pixel in x
            [B1, T1] = dmg_camres('bin', I, cf);  I1 = dmg_camres('rep', B1, T1);
            % replication stays on the NOMINAL cells, so the camera moved +1 cell
            % in x reads, at model column n, what the unshifted camera read at n + 2
            tc.verifyEqual(I1(:, 1:end-2), I0(:, 3:end), 'AbsTol', 1e-14);
            tc.verifyGreaterThan(sum(I0(:)), 0);
        end
        function R4_anchor_reproduces_the_record(tc)
            % IFO: redo96_oapph (the deck's same-surface record) and camI_oap
            % (the polarization-camera row's run) print the same three rows.
            % noise_oap_plain prints 0.9949 on the grid row: it ran on the mis-fed
            % plain mirror rig (SRC_TRIM, README 2026-10-12), not the record feed
            rec = struct('ifo', [0.9832 2.7 2; 0.9838 2.9 2; 0.9844 222.9 NaN], ...
                         'het', [0.9941 1.4 1; 0.9950 1.6 1; 0.9951 117.6 NaN]);   % HET: lockin_battery_oap (the plain mirror rig on the FIXED feed; noise-free, = the bucket reading)
            for g = {'ifo', 'het'}
                R = tResolutionSweep.anchor_(g{1});
                if isempty(R), continue; end
                for q = 1:3
                    r = R.rows(q, :);
                    tc.verifyEqual(round(r{2}*1e4)/1e4, rec.(g{1})(q, 1), sprintf('%s row %d gain', g{1}, q));
                    tc.verifyEqual(round(r{3}*10)/10, rec.(g{1})(q, 2), sprintf('%s row %d err', g{1}, q));
                    if ~isnan(rec.(g{1})(q, 3))
                        tc.verifyEqual(round(r{4}), rec.(g{1})(q, 3), sprintf('%s row %d floor', g{1}, q));
                    end
                end
            end
            tc.assumeTrue(~isempty(tResolutionSweep.anchor_('ifo')) || ~isempty(tResolutionSweep.anchor_('het')), ...
                'no ressweep records yet (run ressweep_run first)');
        end
        function R5_the_collapse_appears(tc)
            % at 1 camera px per actuator the single-actuator row FAILS: its
            % noise-free floor (one-step) leaves the record's ~2 pm for >= 10 pm,
            % and the dense random row's error at least triples -- measured: IFO
            % 13 pm / x6.6, HET 13 pm / x12 (runs/ressweep_*); every row at 1.5 and
            % above stays within 1.5x of the anchor's floor + 1 pm.  A binning that
            % never reached the reading would leave all rows equal.
            nfound = 0;
            for g = {'ifo', 'het'}
                f = fullfile(tResolutionSweep.RUNS, ['ressweep_' g{1}], ['ressweep_' g{1} '_results.mat']);
                if ~isfile(f), continue; end
                nfound = nfound + 1;
                L = load(f);  L.res = L.resp;  s = [L.res.s_target];
                ja = find(abs(s - 3.277) < 1e-6, 1);  j1 = find(abs(s - 1) < 1e-6, 1);
                if isempty(ja) || isempty(j1), continue; end
                ra = L.res(ja).R.rows;  r1 = L.res(j1).R.rows;
                tc.verifyGreaterThanOrEqual(r1{1, 4}, 10, sprintf('%s: the single row at 1 px/act must fail (floor)', g{1}));
                tc.verifyGreaterThanOrEqual(r1{3, 3}, 3*ra{3, 3}, sprintf('%s: dense random error at 1 px/act', g{1}));
                for j = find(s >= 1.5 - 1e-6)
                    rj = L.res(j).R.rows;
                    tc.verifyLessThanOrEqual(rj{1, 4}, 1.5*ra{1, 4} + 1, sprintf('%s: single floor at %.2f px/act', g{1}, s(j)));
                end
            end
            tc.assumeGreaterThan(nfound, 0, 'no ressweep records (run ressweep_run first)');
        end
    end
    methods (Static)
        function R = anchor_(g)
            R = [];  f = fullfile(tResolutionSweep.RUNS, ['ressweep_' g], ['ressweep_' g '_results.mat']);
            if ~isfile(f), return; end
            L = load(f);  L.res = L.resp;  j = find(abs([L.res.s_target] - 3.277) < 1e-6, 1);
            if ~isempty(j), R = L.res(j).R; end
        end
    end
end
