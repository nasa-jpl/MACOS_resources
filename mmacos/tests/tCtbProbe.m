classdef tCtbProbe < matlab.unittest.TestCase
%TCTBPROBE  Pairwise-probe field estimation on the CTB (ctb_probe_estimate).
%   Give'on pairwise probes through the masked diffraction chain: the
%   dark-zone field is estimated from intensity images only and scored
%   against the engine's true field.  A small Jacobian (12 actuators per
%   DM, dark zone 3-5 lambda/D; 12 actuators reach 6 lambda/D) is measured
%   once in setup.  Nothing is written into the library tree.
%
%   Checks:
%     (a) noiseless default probes: the estimate matches the true field to
%         a pinned relative error; a planted wrong-sign difference image
%         makes it miss (the check is not vacuous)
%     (b) the probe-intensity diagnostic ratios are within a pinned band
%         of 1 at the default amplitude
%     (c) a probe 10x the default pushes the ratios off 1 (the meter works)
%     (d) ctb_efc closed on the estimate digs the TRUE contrast and
%         reports both curves; the default sensing path is tCtbDm's
%     (e) a Jacobian given as a path is checked as ctb_efc checks its cache
%     (f) 'probe_rel' scales the default amplitude (e2e stage 10's knob)
%
%   See also: ctb_probe_estimate, ctb_probe_solve, ctb_efc, tCtbDm.

    properties (Constant)
        ModelSize = 512
        % Measured 2026-10-28 on the re-pinned decks (resources 1b45af8),
        % the setup's Jacobian (nact 12, 3-5 lambda/D, central differences
        % at 2 nm since PLAN 1.2 (eeee))
        RelErrMax   = 1e-2     % (a) measured 6.865e-3 (one-sided pokes gave 1.724e-2,
                               %     pinned 2.5e-2; this pin catches a return to them)
        RatioBand   = 0.02     % (b) measured max |ratio - 1| = 0.0088 (one-sided 0.0078)
        RatioOff10x = 0.05     % (c) measured max |ratio - 1| = 0.119 at 10x (one-sided 0.120)
    end

    properties
        bench, J, ch
    end

    methods (TestClassSetup)
        function setup(tc)
            here     = fileparts(mfilename('fullpath'));
            tc.bench = fullfile(fileparts(here), 'templates', ...
                                '30_instruments', 'bench_ctb');
            tc.assumeTrue(isfile(fullfile(tc.bench, 'ctb_dm.in')), ...
                'bench_ctb decks not present');
            addpath(fullfile(fileparts(here), 'src'));
            addpath(tc.bench);
            prev = pwd;
            tc.addTeardown(@() cd(prev));
            cd(tc.bench);                  % GridFile= resolves against cwd
            tc.J = ctb_dm_jacobian('model_size', tc.ModelSize, 'nact', 12, ...
                'inner_lamD', 3, 'outer_lamD', 5, 'save', false, 'verbose', false);
            tc.ch = ctb_chain('rx', fullfile(tc.bench, 'ctb_dm.in'), ...
                'model_size', tc.ModelSize);
        end
    end

    methods (Test)
        function test_noiseless_estimate_matches_truth(tc)
            est = ctb_probe_estimate(tc.ch, tc.J, {});
            tc.verifyLessThan(est.rel_err, tc.RelErrMax, sprintf( ...
                'estimate vs truth: relative error %.3e', est.rel_err));
            % non-vacuity: one probe's difference image with the wrong sign
            dI = est.Ip - est.Im;  dI(:, 2) = -dI(:, 2);
            bad = ctb_probe_solve(dI, est.dE);
            rb = norm(bad - est.E_true) / norm(est.E_true);
            tc.verifyGreaterThan(rb, 10 * tc.RelErrMax, sprintf( ...
                'a wrong-sign difference image must make the estimate miss (%.3e)', rb));
        end

        function test_diagnostic_ratios_near_one(tc)
            est = ctb_probe_estimate(tc.ch, tc.J, {});
            tc.verifyLessThan(max(abs(est.ratio - 1)), tc.RatioBand, sprintf( ...
                'diagnostic ratios %s', mat2str(est.ratio, 4)));
        end

        function test_large_probe_moves_the_ratios(tc)
            e1 = ctb_probe_estimate(tc.ch, tc.J, {});
            e10 = ctb_probe_estimate(tc.ch, tc.J, {}, 'probe_nm', 10 * e1.probe_nm);
            tc.verifyGreaterThan(max(abs(e10.ratio - 1)), tc.RatioOff10x, sprintf( ...
                'a 10x probe must read off 1: ratios %s', mat2str(e10.ratio, 4)));
        end

        function test_efc_closes_on_the_estimate(tc)
            out = ctb_efc('jac', tc.J, 'niter', 3, 'save', false, ...
                'sensing', struct());
            % measured 1.877e-6 -> 9.02e-7 (perfect sensing: -> 8.98e-7);
            % one-sided Jacobian 8.33e-7 / 8.30e-7
            tc.verifyLessThan(out.c_after, 0.6 * out.c_before, ...
                'EFC on the probe estimate must dig the TRUE contrast in 3 iterations');
            n = numel(out.contrast);
            tc.verifyEqual(numel(out.contrast_est), n);
            tc.verifyTrue(all(isfinite(out.contrast_est(1:n-1))), ...
                'the estimated contrast must be reported for every iteration');
            tc.verifyTrue(all(isfinite(out.est_floor(1:n-1))));
        end

        function test_probe_rel_scales_the_default_amplitude(tc)
            % 'probe_rel' is a multiple of the default (speckle-level) amplitude;
            % a smaller probe is closer to the linear regime (ch08d's sweep)
            e1 = ctb_probe_estimate(tc.ch, tc.J, {});
            e3 = ctb_probe_estimate(tc.ch, tc.J, {}, 'probe_rel', 0.3);
            tc.verifyEqual(e1.probe_rel, 1);
            tc.verifyEqual(e3.probe_nm, 0.3 * e1.probe_nm, 'RelTol', 1e-12, ...
                'probe_rel 0.3 is three tenths of the default amplitude');
            tc.verifyLessThan(e3.rel_err, e1.rel_err, sprintf( ...
                'the smaller probe estimates better: %.3e vs %.3e', e3.rel_err, e1.rel_err));
            % an explicit amplitude wins over probe_rel
            e4 = ctb_probe_estimate(tc.ch, tc.J, {}, 'probe_rel', 0.3, 'probe_nm', e1.probe_nm);
            tc.verifyEqual(e4.probe_nm, e1.probe_nm, 'RelTol', 1e-12);
        end

        function test_jacobian_path_is_checked(tc)
            % (1) a path whose stored chain differs from the plant's is refused
            w = tempname;  mkdir(w);  tc.addTeardown(@() rmdir(w, 's'));
            J2 = tc.J;
            c = J2.chain_opts;
            k = find(strcmp(c(1:2:end), 'r_lyot_frac')) * 2;
            c{k} = c{k} + 0.1;
            J2.chain_opts = c;
            p1 = fullfile(w, 'jac_other_chain.mat');
            save(p1, '-struct', 'J2');
            tc.verifyError(@() ctb_probe_estimate(tc.ch, p1, {}), ...
                'ctb_jac_check:mismatch');
            % (2) a path that disagrees with the fingerprint beside it is refused
            J3 = tc.J;
            p2 = fullfile(w, 'jac_fp.mat');
            save(p2, '-struct', 'J3');
            J3.G = J3.G * 1.01;
            jac_fingerprint('write', fullfile(w, 'jac_fp.fp.json'), J3, struct());
            tc.verifyError(@() ctb_probe_estimate(tc.ch, p2, {}), ...
                'ctb_efc:fingerprint');
        end
    end
end
