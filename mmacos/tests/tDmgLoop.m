classdef tDmgLoop < matlab.unittest.TestCase
%TDMGLOOP  The shared closed-loop hold metric (dm_gauge_lib/dmg_loop) on a
%   synthetic LINEAR instrument, pinned to the loop's own theory.
%
%   The instrument: a 24x24 DM whose lit disc images onto a 40x40 "detector"
%   as Gaussian blobs (a known response matrix J, one column per lit
%   actuator); frames are the map J*a; photon noise is white Gaussian at
%   sigma0/sqrt(nph/1e12) per pixel; the estimator is the exact pseudo-
%   inverse, optionally scaled by a gain G (a mis-calibrated reading).
%   Pure MATLAB, no engine: this class gates the LOOP CODE, which both
%   gauges (ZWFS `zwfs_run` stage 'loop', T-G IFO `tg96_run`) then run
%   unchanged (BRIEF_loop_metric.md, Dave 2026-09-11).
%
%   Gates:
%     G1  noiseless step: the residual contracts by exactly (1 - g*G) per
%         cycle (G = 1 and G = 0.8), to < 1e-9 of the step within K/2 cycles
%     G2  noise only: the steady-state rms equals sigma_n*sqrt(g/(2-g)), and
%         the in-run sigma_n equals an independent single-shot Monte Carlo
%     G2b random walk: rms_ss^2 = (sigma_d^2 + g^2 sigma_n^2)/(g(2-g))
%     G2c thermal ramp (noiseless): the residual settles at rate/g
%     G3  non-vacuity: a reading with a fixed differential bias converges
%         to a NON-zero surface equal to that bias through the estimator;
%         the unbiased reading converges to zero on the same seeds
%     G4  the drift realization is the same at every photon level and under
%         a different noise seed offset (seeded drift stream)
%     G5  a single-shot (noisy) reference is a fixed bias of size sigma_n;
%         a noiseless one leaves the bias at the 1/sqrt(nss) level
%     G6  Parseval: the spectrum's bands sum to the steady-state mean square
%     G7  a reading with NEGATIVE gain diverges at (1 + g|G|) per cycle and
%         the guard (opt.rmax) stops the run and flags it
%     G8  CAMERA drift (opt.cam, 2026-09-12): a per-pixel offset random-
%         walking between cycles.  A zero-sum two-frame reading (frames
%         +F + B and -F + B, read as their half difference) is EXACTLY
%         immune (its run with the drift equals its run without, same
%         seeds); a single-frame reading imprints o_k - o_0 on the DM (its
%         steady-state rms grows above its noise floor and its bias map
%         tracks minus the camera walk through the estimator)
%     G9  DESCENT (opt.start_rms, 2026-09-13): the loop opened at a surface
%         of the requested rms -- the set point's own field, rescaled, so
%         r(1) = |start_rms - rms(A0)| -- contracts at (1 - gG) from there,
%         .surf_rms reports the starting SURFACE, and .k_reach dates the
%         first cycle at or below each opt.reach level
%     G10 RE-CALIBRATION (opt.recal_every): ins.recal is called at exactly
%         those cycles, its estimator takes effect from the next cycle (a
%         mis-scaled reading whose recal returns the right scale changes its
%         contraction there), and its cost is added to .nstates
%     G11 WITHIN-MEASUREMENT DRIFT (opt.intra): intra 0 reproduces the run
%         without the knob to the last bit; with intra 1 a reading whose
%         frames are taken in sequence reads the MIDDLE of its own scan and
%         holds worse, while a simultaneous reading (which ignores
%         aux.dstep) is unchanged
%     G12 REFERENCE-ARM WALK (opt.ref_walk): a non-common-path phase walk
%         reaching the differential as a piston sets a hold floor that
%         scales with the walk, and .ref_phase is that walk
%     G13 THE UNWRAPPER (dmg_unwrap, 2026-09-13).  Capture is a WRAP
%         problem -- every phase reading returns a wrapped differential --
%         so the least-squares unwrapper is gated here, beside the loop it
%         serves: a wrapped ramp and a band-limited random surface of 1.5
%         and 3 waves peak-to-valley unwrap to the truth to 1e-12 with
%         zero residues; a map that was never wrapped passes through
%         unchanged; and a gradient beyond pi per pixel is REPORTED as
%         residues rather than silently unwrapped wrong

    properties (Constant)
        NACT = 24
        NPIX = 40
        SIG0 = 2e-6      % per-pixel noise at 1e12 photons (map units)
    end
    properties
        ins       % the synthetic instrument (G = 1)
        J
        lit
        Jp        % pinv(J)
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            ldir = fullfile(fileparts(mfilename('fullpath')), '..', 'templates', '40_benches', 'dm_gauge_lib');
            testCase.assertTrue(isfolder(ldir), sprintf('lib dir missing: %s', ldir));
            addpath(ldir);
            testCase.addTeardown(@() rmpath(ldir));
            n = testCase.NACT;  np = testCase.NPIX;
            [c, r] = meshgrid(1:n, 1:n);
            lit_ = hypot(c-(n+1)/2, r-(n+1)/2) <= 10.5;
            il = find(lit_);  nl = numel(il);
            % blobs: actuator (r,c) -> pixel (1.5*(c-1)+3, 1.5*(r-1)+3), sigma 1 px
            [pc, pr] = meshgrid(1:np, 1:np);
            J_ = zeros(np*np, nl);
            for q = 1:nl
                [rr, cc] = ind2sub([n n], il(q));
                u = 1.5*(cc-1) + 3;  v = 1.5*(rr-1) + 3;
                b = exp(-((pc-u).^2 + (pr-v).^2)/2);
                J_(:, q) = b(:);
            end
            testCase.J = J_;  testCase.lit = lit_;  testCase.Jp = pinv(J_);
            testCase.ins = testCase.mkins(1, zeros(np*np, 1));
        end
    end

    methods
        function ins = mkins(testCase, G, bias)
            % the synthetic instrument: gain G on the estimate, a fixed
            % differential bias (a map added to every differential)
            J_ = testCase.J;  Jp_ = testCase.Jp;  lit_ = testCase.lit;  n = testCase.NACT;
            il = find(lit_);  s0 = testCase.SIG0;
            ins.lit = lit_;
            ins.measure = @(cmd) J_ * cmd(il);
            ins.noisy = @(F, nph, seed) F + (isfinite(nph)) * s0/sqrt(nph/1e12) * ...
                randn(RandStream('mt19937ar', 'Seed', seed), size(F));
            ins.diff = @(F1, F0) F1 - F0 + bias;
            ins.est = @(m) est_local(m, Jp_, G, il, n);
        end

        function ins = mkins_cam(testCase, immune)
            % instruments for the camera-drift gate (G8): frames in PHOTON
            % units (1 electron = 1 frame unit), noisy adds shot noise and the
            % camera offset.  immune = false: one frame, diff = F1 - F0.
            % immune = true: two frames [+F + B, -F + B] with a bias B (the
            % phase-step analog), diff = half their difference, the offset
            % common to the scan (intra 0) cancels exactly.
            J_ = testCase.J;  Jp_ = testCase.Jp;  lit_ = testCase.lit;  n = testCase.NACT;  np = testCase.NPIX;
            il = find(lit_);  s0 = testCase.SIG0;  B = 5e-5;
            ins.lit = lit_;  ins.npix = np;
            sh = @(F, nph, rs) F + (isfinite(nph)) * s0/sqrt(nph/1e12) * randn(rs, size(F));
            if immune
                ins.measure = @(cmd) [J_*cmd(il) + B, -J_*cmd(il) + B];
                ins.noisy = @(F, nph, seed, varargin) noisy2(F, nph, seed, sh, np, varargin{:});
                ins.diff = @(F1, F0) (F1(:,1) - F1(:,2))/2 - (F0(:,1) - F0(:,2))/2;
            else
                ins.measure = @(cmd) J_*cmd(il) + B;
                ins.noisy = @(F, nph, seed, varargin) noisy1(F, nph, seed, sh, np, varargin{:});
                ins.diff = @(F1, F0) F1 - F0;
            end
            ins.est = @(m) est_local(m, Jp_, 1, il, n);
        end
        function ins = mkins_recal(testCase, G0, G1)
            % a MIS-SCALED reading (estimator gain G0) whose on-surface
            % re-calibration returns the right one (G1): the contraction
            % changes at the recal cycle and nowhere else
            J_ = testCase.J;  Jp_ = testCase.Jp;  lit_ = testCase.lit;  n = testCase.NACT;
            il = find(lit_);
            ins.lit = lit_;
            ins.measure = @(cmd) J_ * cmd(il);
            ins.noisy = @(F, nph, seed, varargin) F;
            ins.diff = @(F1, F0) F1 - F0;
            ins.est = @(m) est_local(m, Jp_, G0, il, n);
            ins.recal = @(cmd) struct('est', @(m) est_local(m, Jp_, G1, il, n), 'nstates', 7);
        end

        function ins = mkins_intra(testCase, sequential)
            % sequential = true: a two-frame reading whose frames are taken
            % at the two ENDS of its scan, read as their mean -- so a DM
            % advancing across the scan is read at its middle.
            % sequential = false: the same two frames captured at ONE
            % instant (a simultaneous pair), which ignores aux.dstep.
            J_ = testCase.J;  Jp_ = testCase.Jp;  lit_ = testCase.lit;  n = testCase.NACT;
            il = find(lit_);  s0 = testCase.SIG0;
            ins.lit = lit_;  ins.npix = testCase.NPIX;
            ins.measure = @(cmd, varargin) meas_intra(cmd, J_, il, sequential, varargin{:});
            ins.noisy = @(F, nph, seed, varargin) F + (isfinite(nph)) * s0/sqrt(nph/1e12) * ...
                randn(RandStream('mt19937ar', 'Seed', seed), size(F));
            ins.diff = @(F1, F0) mean(F1, 2) - mean(F0, 2);
            ins.est = @(m) est_local(m, Jp_, 1, il, n);
        end

        function ins = mkins_refwalk(testCase, sees)
            % sees = true: a NON-COMMON-PATH reading whose reference phase
            % lands on its map as a piston; false: a common-path one, which
            % has no such arm and ignores aux.ref_phase
            J_ = testCase.J;  Jp_ = testCase.Jp;  lit_ = testCase.lit;  n = testCase.NACT;
            il = find(lit_);  s0 = testCase.SIG0;
            ins.lit = lit_;
            ins.measure = @(cmd, varargin) meas_refwalk(cmd, J_, il, sees, varargin{:});
            ins.noisy = @(F, nph, seed, varargin) F + (isfinite(nph)) * s0/sqrt(nph/1e12) * ...
                randn(RandStream('mt19937ar', 'Seed', seed), size(F));
            ins.diff = @(F1, F0) F1 - F0;
            ins.est = @(m) est_local(m, Jp_, 1, il, n);
        end

        function s = sig_single(testCase, nph, nshot)
            % independent single-shot noise: rms over lit of est(noise) over nshot shots
            ins_ = testCase.ins;  F = ins_.measure(zeros(testCase.NACT));
            e2 = 0;
            for i = 1:nshot
                a = ins_.est(ins_.diff(ins_.noisy(F, nph, 50000 + i), F));
                e2 = e2 + mean(a(testCase.lit).^2);
            end
            s = sqrt(e2/nshot);
        end
    end

    methods (Test)
        function test_G1_noiseless_step_contracts_at_one_minus_gG(testCase)
            for G = [1 0.8]
                ins_ = testCase.mkins(G, zeros(testCase.NPIX^2, 1));
                o = struct('g', 0.5, 'K', 40, 'nph', Inf, 'seed', 3, ...
                           'drift', struct('kind', 'step', 'amp', 1e-6));
                L = dmg_loop(ins_, o);
                rho = 1 - o.g*G;
                testCase.verifyEqual(L.rms(1), 1e-6, 'RelTol', 1e-12, 'the step lands at cycle 1 with the requested rms');
                ratio = L.rms(2:20) ./ L.rms(1:19);
                testCase.verifyEqual(ratio, rho*ones(1, 19), 'AbsTol', 1e-9, sprintf('per-cycle contraction at G = %g', G));
                testCase.verifyEqual(L.rho, rho, 'AbsTol', 1e-6, 'fitted contraction');
                testCase.verifyEqual(L.tau, -1/log(rho), 'RelTol', 1e-5, 'time constant');
                testCase.verifyLessThan(L.rms(20), 1e-3 * L.rms(1), 'converged to < 1e-3 of the step within K/2 (the brief''s 1 nm -> 1 pm)');
                testCase.verifyLessThan(L.rms(end), 1e-8 * L.rms(1), 'and keeps contracting: < 1e-8 at K');
                testCase.verifyEqual(L.nstates, o.K + 1, 'one state per cycle plus the reference');
            end
        end

        function test_G2_noise_only_steady_state_matches_theory(testCase)
            g = 0.5;  nph = 1e12;
            o = struct('g', g, 'K', 800, 'nph', nph, 'seed', 11, 'drift', struct('kind', 'none'));
            L = dmg_loop(testCase.ins, o);
            s_mc = testCase.sig_single(nph, 200);
            testCase.verifyEqual(L.sig_n, s_mc, 'RelTol', 0.05, 'in-run single-shot noise vs an independent Monte Carlo');
            testCase.verifyEqual(L.ss, s_mc*sqrt(g/(2-g)), 'RelTol', 0.06, 'steady-state rms = sigma_n sqrt(g/(2-g))');
            testCase.verifyEqual(L.theory.ss_noise, L.sig_n*sqrt(g/(2-g)), 'RelTol', 1e-12, 'theory line uses the in-run sigma_n (rho not fitted on a flat history)');
            testCase.verifyLessThan(L.bias, 0.2*L.ss, 'no bias from noise alone (averages down over nss cycles)');
            % the noise-only floor scales as 1/sqrt(nph)
            o.nph = 1e14;  L2 = dmg_loop(testCase.ins, o);
            testCase.verifyEqual(L2.ss / L.ss, 0.1, 'RelTol', 0.1, 'floor scales as 1/sqrt(photons)');
        end

        function test_G2b_random_walk_steady_state_matches_theory(testCase)
            g = 0.5;  nph = 1e12;  sig_d = 3e-6;
            o = struct('g', g, 'K', 800, 'nph', nph, 'seed', 5, 'drift', struct('kind', 'walk', 'sigma', sig_d));
            L = dmg_loop(testCase.ins, o);
            th = sqrt((sig_d^2 + g^2*L.sig_n^2) / (g*(2-g)));
            testCase.verifyEqual(L.ss, th, 'RelTol', 0.08, 'walk: rms_ss^2 = (sigma_d^2 + g^2 sigma_n^2)/(g(2-g))');
            testCase.verifyEqual(L.theory.ss_walk, th, 'RelTol', 1e-12);
            testCase.verifyEqual(sqrt(mean(L.drift_rms.^2)), sig_d, 'RelTol', 0.05, 'the walk increments have the requested rms');
        end

        function test_G2c_thermal_ramp_settles_at_rate_over_g(testCase)
            g = 0.4;  rate = 5e-9;
            o = struct('g', g, 'K', 80, 'nph', Inf, 'seed', 2, 'drift', struct('kind', 'thermal', 'rate', rate));
            L = dmg_loop(testCase.ins, o);
            testCase.verifyEqual(L.rms(end), rate/g, 'RelTol', 1e-6, 'a ramp of r per cycle is held at a lag of r/g');
            testCase.verifyEqual(L.theory.lag_ramp, rate/g, 'RelTol', 1e-6);
            % the lag has the ramp's shape: low order only
            testCase.verifyGreaterThan(L.spec.band(1), 0.8*L.ss, 'the residual is the low-order thermal shape');
            testCase.verifyTrue(isnan(L.rho), 'no transient fit on a ramp run');
        end

        function test_G3_biased_reading_converges_to_a_nonzero_surface(testCase)
            np = testCase.NPIX;  n = testCase.NACT;
            % a fixed differential bias: 20 nm on one blob's worth of pixels
            ab = zeros(n);  ab(12, 9) = 2e-5;  bias = testCase.J * ab(testCase.lit);
            o = struct('g', 0.5, 'K', 60, 'nph', Inf, 'seed', 4, 'drift', struct('kind', 'none'));
            Lb = dmg_loop(testCase.mkins(1, bias), o);
            L0 = dmg_loop(testCase.mkins(1, zeros(np*np, 1)), o);
            expect = sqrt(mean(ab(testCase.lit).^2));      % the loop converges to -bias through the estimator
            testCase.verifyEqual(Lb.ss, expect, 'RelTol', 1e-6, 'biased reading: held surface = -(estimated bias)');
            testCase.verifyEqual(Lb.bias, expect, 'RelTol', 1e-6);
            testCase.verifyEqual(Lb.bias_map(12, 9), -2e-5, 'RelTol', 1e-6, 'and it sits where the bias is');
            testCase.verifyLessThan(L0.ss, 1e-15, 'unbiased reading on the same seeds: zero');
        end

        function test_G4_drift_realization_is_seeded_independently_of_noise(testCase)
            d = struct('kind', 'walk', 'sigma', 2e-6);
            o1 = struct('g', 0.5, 'K', 30, 'nph', Inf,  'seed', 9, 'drift', d);
            o2 = struct('g', 0.5, 'K', 30, 'nph', 1e13, 'seed', 9, 'drift', d);
            L1 = dmg_loop(testCase.ins, o1);  L2 = dmg_loop(testCase.ins, o2);
            testCase.verifyEqual(L2.drift_rms, L1.drift_rms, 'AbsTol', 0, 'identical drift increments at Inf and 1e13 photons');
            o3 = o1;  o3.seed = 10;  L3 = dmg_loop(testCase.ins, o3);
            testCase.verifyNotEqual(L3.drift_rms, L1.drift_rms, 'a different seed gives a different realization');
        end

        function test_G5_single_shot_reference_is_a_fixed_bias(testCase)
            nph = 1e12;
            o = struct('g', 0.5, 'K', 800, 'nph', nph, 'seed', 21, 'drift', struct('kind', 'none'), 'ref', 'noisy');
            Ln = dmg_loop(testCase.ins, o);
            o.ref = 'noiseless';  L0 = dmg_loop(testCase.ins, o);
            testCase.verifyEqual(Ln.bias, Ln.sig_n, 'RelTol', 0.15, 'noisy reference: the held surface carries its single-shot noise as a bias');
            testCase.verifyLessThan(L0.bias, 0.25*L0.sig_n, 'noiseless reference: no such bias');
            testCase.verifyEqual(Ln.ss^2 - Ln.bias^2, L0.ss^2, 'RelTol', 0.3, 'the fluctuation about the bias is the same loop noise');
        end

        function test_G7_negative_gain_diverges_and_is_flagged(testCase)
            ins_ = testCase.mkins(-0.5, zeros(testCase.NPIX^2, 1));
            o = struct('g', 0.5, 'K', 60, 'nph', Inf, 'seed', 1, 'drift', struct('kind', 'step', 'amp', 1e-6), 'rmax', 1e-3);
            L = dmg_loop(ins_, o);
            testCase.verifyTrue(L.diverged, 'flagged');
            testCase.verifyEqual(L.k_end, 32, 'stopped at the first cycle above rmax: 1.25^31 x 1 nm > 1 um');
            testCase.verifyEqual(L.rms(2:10) ./ L.rms(1:9), 1.25*ones(1, 9), 'AbsTol', 1e-9, 'grows at 1 + g|G|');
            testCase.verifyTrue(all(isnan(L.rms(33:end))), 'the cycles not run are NaN');
            testCase.verifyTrue(isnan(L.rho), 'no transient fit on a diverged run');
            L0 = dmg_loop(testCase.ins, o);
            testCase.verifyFalse(L0.diverged, 'the unit-gain reading on the same options does not trip the guard');
        end

        function test_G8_camera_drift_zero_sum_reading_immune_single_frame_not(testCase)
            o = struct('g', 0.5, 'K', 40, 'nph', 1e13, 'seed', 5, 'drift', struct('kind', 'none'));
            oc = o;  oc.cam = struct('walk', 2e-6, 'intra', 0);          % electrons = map units here
            im0 = dmg_loop(testCase.mkins_cam(true), o);
            im1 = dmg_loop(testCase.mkins_cam(true), oc);
            testCase.verifyEqual(im1.rms, im0.rms, 'RelTol', 1e-10, ...
                'a zero-sum reading must not see a within-scan-constant camera offset');
            sf0 = dmg_loop(testCase.mkins_cam(false), o);
            sf1 = dmg_loop(testCase.mkins_cam(false), oc);
            testCase.verifyGreaterThan(sf1.ss, 3*sf0.ss, 'the single-frame reading must imprint the camera walk');
            % the imprint IS the camera walk through the estimator: the bias
            % map over the tail tracks minus est(o_k - o_0)
            testCase.verifyTrue(isfield(sf1, 'cam_rms') && sf1.cam_rms(end) > 0, 'the record carries the camera walk');
            testCase.verifyGreaterThan(sf1.bias, 3*sf0.bias);
        end

        function test_G9_descent_opens_at_the_requested_surface(testCase)
            n = testCase.NACT;  lit_ = testCase.lit;
            rng(3);  A0 = zeros(n);  A0(lit_) = 30e-6*randn(nnz(lit_), 1);
            A0(lit_) = A0(lit_) / sqrt(mean(A0(lit_).^2)) * 30e-6;      % exactly 30 nm rms
            o = struct('A0', A0, 'g', 0.5, 'K', 60, 'nph', Inf, 'seed', 3, ...
                       'drift', struct('kind', 'none'), 'start_rms', 100e-6, ...
                       'reach', [10e-6 3e-9]);
            L = dmg_loop(testCase.ins, o);
            testCase.verifyEqual(L.surf_rms, 100e-6, 'RelTol', 1e-12, 'the STARTING surface has the requested rms');
            testCase.verifyEqual(L.rms(1), 70e-6, 'RelTol', 1e-12, 'so the residual to a 30 nm set point opens at 70 nm');
            testCase.verifyEqual(L.rms(2)/L.rms(1), 0.5, 'AbsTol', 1e-9, 'and contracts at 1 - gG from there');
            % k_reach: 70 nm x 0.5^k <= 10 nm at k = 4, so cycle 5; 3 pm at cycle 15
            k10 = find(L.rms <= 10e-6, 1);  k3 = find(L.rms <= 3e-9, 1);
            testCase.verifyEqual(L.k_reach, [k10 k3], 'the reach levels are dated from the residual history');
            testCase.verifyEqual(L.k_reach(1), 4, 'cycle 4 for 10 nm (70 x 0.5^3 = 8.75 nm)');
            testCase.verifyEqual(L.rho, 0.5, 'AbsTol', 1e-6, 'a descent gets a transient fit, as a step does');
            % with no start_rms the same options start AT the set point
            o2 = rmfield(o, 'start_rms');  L2 = dmg_loop(testCase.ins, o2);
            testCase.verifyLessThan(L2.rms(1), 1e-18, 'without the knob the loop opens at the set point');
            testCase.verifyEqual(L2.surf_rms, 30e-6, 'RelTol', 1e-12);
        end

        function test_G10_recalibration_runs_on_schedule_and_takes_effect(testCase)
            ins_ = testCase.mkins_recal(0.5, 1.0);
            o = struct('g', 0.5, 'K', 40, 'nph', Inf, 'seed', 3, ...
                       'drift', struct('kind', 'step', 'amp', 1e-6), 'recal_every', 10);
            L = dmg_loop(ins_, o);
            testCase.verifyEqual(L.k_recal, [10 20 30], 'recal at every tenth cycle, never at the last');
            testCase.verifyEqual(L.n_recal, 3);
            testCase.verifyEqual(L.nstates, o.K + 1 + 3*7, 'each re-calibration''s states are counted');
            rat = L.rms(2:end) ./ L.rms(1:end-1);
            testCase.verifyEqual(rat(2:9), 0.75*ones(1, 8), 'AbsTol', 1e-9, 'before the first recal: 1 - g*0.5');
            testCase.verifyEqual(rat(12:19), 0.50*ones(1, 8), 'AbsTol', 1e-9, 'after it: 1 - g*1.0');
            % non-vacuity: the same reading without the knob never improves
            o0 = o;  o0.recal_every = 0;  L0 = dmg_loop(ins_, o0);
            testCase.verifyEqual(L0.n_recal, 0);
            testCase.verifyEqual(L0.rms(2:end) ./ L0.rms(1:end-1), 0.75*ones(1, o.K-1), 'AbsTol', 1e-9);
            testCase.verifyLessThan(L.rms(end), 1e-3*L0.rms(end), 'and it is what makes the difference');
        end

        function test_G11_within_measurement_drift_reaches_a_sequential_reading(testCase)
            % WHAT IS GATED IS THE CONTRACT, not a sign.  A reading whose
            % frames straddle the scan reads the MIDDLE of its own scan to
            % first order, and under a pure random walk that half-step of
            % prediction can HELP as easily as hurt -- which way it goes is
            % the instrument's business and is measured on the engine, not
            % asserted here.  What the loop code must guarantee: the map
            % handed to the instrument is opt.intra times the NEXT cycle's
            % drift increment, intra 0 changes nothing, and a reading that
            % ignores aux.dstep is untouched.
            sq = testCase.mkins_intra(true);  sim = testCase.mkins_intra(false);
            d = struct('kind', 'walk', 'sigma', 2e-6);
            o = struct('g', 0.5, 'K', 60, 'nph', 1e13, 'seed', 7, 'drift', d);
            a0 = dmg_loop(sq, o);
            o0 = o;  o0.intra = 0;  b0 = dmg_loop(sq, o0);
            testCase.verifyEqual(b0.rms, a0.rms, 'AbsTol', 0, 'intra 0 is the run without the knob');
            % capture what the instrument is handed
            seen = {};
            sq2 = sq;  m0f = sq.measure;
            sq2.measure = @(cmd, varargin) grab(m0f, cmd, varargin{:});
            o1 = o;  o1.intra = 0.5;
            s1 = dmg_loop(sq2, o1);
            seen = grab();                                  % the aux of every cycle, in order
            testCase.verifyEqual(numel(seen), o.K, 'one aux per cycle');
            lit_ = testCase.lit;  rl = @(m) sqrt(mean(m(lit_).^2));
            got = cellfun(@(a) rl(a.dstep), seen);
            testCase.verifyEqual(got(1:o.K-1), 0.5*a0.drift_rms(2:o.K), 'RelTol', 1e-12, ...
                'cycle k is handed intra x the increment of cycle k+1');
            testCase.verifyNotEqual(s1.ss, a0.ss, 'and it reaches a sequential reading');
            m1 = dmg_loop(sim, o1);  m0 = dmg_loop(sim, o);
            testCase.verifyEqual(m1.rms, m0.rms, 'AbsTol', 0, 'a simultaneous reading ignores aux.dstep entirely');
        end

        function test_G12_reference_arm_walk_sets_a_floor_that_scales(testCase)
            % noiseless, no DM drift: whatever residual is left IS the
            % reference arm's, so the floor and its scaling are unambiguous
            o = struct('g', 0.5, 'K', 400, 'nph', Inf, 'seed', 5, 'drift', struct('kind', 'none'));
            sees = testCase.mkins_refwalk(true);  blind = testCase.mkins_refwalk(false);
            L0 = dmg_loop(sees, o);
            testCase.verifyLessThan(L0.ss, 1e-18, 'with no walk the noiseless loop holds exactly');
            w = [1e-8 1e-7];                              % map units per cycle (the synthetic's "rad")
            ss = zeros(size(w));
            for i = 1:numel(w)
                oi = o;  oi.ref_walk = w(i);  Li = dmg_loop(sees, oi);  ss(i) = Li.ss;
                testCase.verifyEqual(std(diff([0 Li.ref_phase])), w(i), 'RelTol', 0.15, 'the walk has the requested rms increment');
            end
            testCase.verifyGreaterThan(ss(1), 1e-12, 'the reference arm''s walk sets a floor of its own');
            testCase.verifyEqual(ss(2)/ss(1), 10, 'RelTol', 1e-9, 'and the floor is linear in the walk');
            % non-vacuity: a common-path reading has no such arm
            ob = o;  ob.ref_walk = w(2);
            Lb = dmg_loop(blind, ob);  Lb0 = dmg_loop(blind, o);
            testCase.verifyEqual(Lb.rms, Lb0.rms, 'AbsTol', 0, 'a common-path reading is untouched by it');
        end

        function test_G13_unwrapper_is_exact_below_the_pixel_gradient_limit(testCase)
            W = @(x) atan2(sin(x), cos(x));
            N = 64;  [x, y] = meshgrid(linspace(-1, 1, N));
            % (a) a wrapped ramp on a full box: the unweighted solve is exact
            t = 3*pi*(0.6*x + 0.4*y);                        % 1.5 waves across
            [u, i1] = dmg_unwrap(W(t), true(N));
            testCase.verifyEqual(i1.nres, 0, 'a ramp has no residues');
            testCase.verifyEqual(u - mean(u(:)), t - mean(t(:)), 'AbsTol', 1e-12, ...
                'a wrapped ramp unwraps to the truth');
            % (b) a band-limited random surface on a DISC mask, 1.5 and 3
            % waves peak to valley -- the masked (PCG) solve
            msk = hypot(x, y) <= 0.9;
            s = bandlimited(N, 4, 7);  s = s - mean(s(msk));
            for amp = [1.5 3.0]
                t2 = 2*pi*amp * s / (max(s(msk)) - min(s(msk)));
                [u2, i2] = dmg_unwrap(W(t2), msk);
                d = (u2(msk) - mean(u2(msk))) - (t2(msk) - mean(t2(msk)));
                testCase.verifyEqual(i2.nres, 0, sprintf('%g waves PV: no residues', amp));
                testCase.verifyLessThan(i2.maxgrad, pi, 'and it is inside the pixel-gradient limit');
                testCase.verifyEqual(max(abs(d)), 0, 'AbsTol', 1e-12, ...
                    sprintf('%g waves PV on a disc unwraps to the truth', amp));
                testCase.verifyTrue(i2.wrapped, 'and it reports that it did work');
            end
            % (c) a map that was never wrapped comes back unchanged
            s4 = 0.4*sin(2*pi*x).*cos(2*pi*y);
            [u4, i4] = dmg_unwrap(s4, msk);
            testCase.verifyEqual(max(abs(u4(msk) - s4(msk))), 0, 'AbsTol', 1e-12, ...
                'an unwrapped map passes through');
            testCase.verifyFalse(i4.wrapped, 'and it says so');
            testCase.verifyEqual(nnz(u4(~msk)), 0, 'off the mask it returns zero');
            % (d) beyond the pixel-gradient limit: REPORTED, not silently wrong
            t5 = 2*pi*40 * s / (max(s(msk)) - min(s(msk)));
            [~, i5] = dmg_unwrap(W(t5), msk);
            testCase.verifyGreaterThan(i5.nres, 0, ...
                'a gradient beyond pi per pixel must be reported as residues');
            testCase.verifyGreaterThan(i5.maxgrad, 3.0, 'the gradient itself is at the wrap');
        end

        function test_G6_spectrum_bands_sum_to_the_steady_state(testCase)
            o = struct('g', 0.5, 'K', 200, 'nph', 1e12, 'seed', 8, 'drift', struct('kind', 'walk', 'sigma', 2e-6));
            L = dmg_loop(testCase.ins, o);
            testCase.verifyEqual(sqrt(sum(L.spec.band.^2)), L.ss, 'RelTol', 1e-10, 'Parseval over the three bands');
            rb = L.spec.rms(~isnan(L.spec.rms));
            testCase.verifyEqual(sqrt(sum(rb.^2)), L.ss, 'RelTol', 1e-10, 'Parseval over the radial bins');
        end
    end
end

function Fn = noisy1(F, nph, seed, sh, np, cam)
rs = RandStream('mt19937ar', 'Seed', seed);
Fn = sh(F, nph, rs);
if nargin >= 6 && ~isempty(cam), Fn = Fn + reshape(cam.o, np*np, 1); end
end

function Fn = noisy2(F, nph, seed, sh, np, cam)
rs = RandStream('mt19937ar', 'Seed', seed);
Fn = [sh(F(:,1), nph/2, rs), sh(F(:,2), nph/2, rs)];
if nargin >= 6 && ~isempty(cam)
    o = reshape(cam.o, np*np, 1);  d = reshape(cam.d, np*np, 1);
    Fn = Fn + [o, o + d];                                    % frame 2 gets the within-scan increment
end
end

function s = bandlimited(N, nc, seed)
% a random surface with nc cycles across the grid -- smooth at the pixel
% scale, as a DM's surface is (4 detector px per actuator at 385 rays)
rng(seed);
F = zeros(N);
for p = -nc:nc
    for q = -nc:nc
        F(mod(p, N)+1, mod(q, N)+1) = (randn + 1i*randn) * exp(-(p^2 + q^2)/(2*(nc/2)^2));
    end
end
s = real(ifft2(F));
end

function a = est_local(m, Jp, G, il, n)
a = zeros(n);  a(il) = G * (Jp * m);
end

function out = grab(f, cmd, varargin)
% record the aux each cycle hands the instrument, then measure as usual
persistent seen
if nargin == 0, out = seen;  seen = {};  return; end
if ~isempty(varargin), seen{end+1} = varargin{1}; end %#ok<AGROW>
out = f(cmd, varargin{:});
end

function F = meas_intra(cmd, J_, il, sequential, varargin)
% two frames: at the start of the scan and, when the reading is sequential
% and the DM is advancing (aux.dstep), at its end
d = zeros(size(cmd));
if sequential && ~isempty(varargin) && ~isempty(varargin{1}) && isfield(varargin{1}, 'dstep') ...
        && ~isempty(varargin{1}.dstep), d = varargin{1}.dstep; end
F = [J_*cmd(il), J_*(cmd(il) + d(il))];
end

function F = meas_refwalk(cmd, J_, il, sees, varargin)
% the reference arm's phase reaches a non-common-path reading's map as a
% piston; a common-path reading has no such arm
p = 0;
if sees && ~isempty(varargin) && ~isempty(varargin{1}) && isfield(varargin{1}, 'ref_phase') ...
        && ~isempty(varargin{1}.ref_phase), p = varargin{1}.ref_phase; end
F = J_*cmd(il) + p;
end
