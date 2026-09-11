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

        function test_G6_spectrum_bands_sum_to_the_steady_state(testCase)
            o = struct('g', 0.5, 'K', 200, 'nph', 1e12, 'seed', 8, 'drift', struct('kind', 'walk', 'sigma', 2e-6));
            L = dmg_loop(testCase.ins, o);
            testCase.verifyEqual(sqrt(sum(L.spec.band.^2)), L.ss, 'RelTol', 1e-10, 'Parseval over the three bands');
            rb = L.spec.rms(~isnan(L.spec.rms));
            testCase.verifyEqual(sqrt(sum(rb.^2)), L.ss, 'RelTol', 1e-10, 'Parseval over the radial bins');
        end
    end
end

function a = est_local(m, Jp, G, il, n)
a = zeros(n);  a(il) = G * (Jp * m);
end
