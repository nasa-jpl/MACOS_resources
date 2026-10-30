classdef tE2eSensing < matlab.unittest.TestCase
%TE2ESENSING  e2e stage 10: pairwise-probe sensing on the coupled telescope + CTB model.
%   ctb_probe_estimate (Give'on pairwise probes through the masked chain)
%   on the committed coupled deck (s9_coupled.in, its DM deck written to a
%   scratch directory) behind the annular Lyot stop stage 9 picked, with a
%   small Jacobian (12 actuators per DM, dark zone 3-5 lambda/D) measured
%   once in setup -- tCtbProbe's configuration, on the telescope instead
%   of the bench.  Nothing is written into the library tree.
%
%   Checks:
%     (a) the stage's finding: at the estimator's default amplitude (the
%         probe as bright as the speckle, a 0.9 rad probe on the
%         telescope's halo) the estimate misses the true field by a pinned
%         margin, and at the stage's pick (0.3x) it matches to a pinned
%         relative error; the error ratio between the two is pinned (the
%         second-order scaling); a planted wrong-sign difference image
%         makes the good estimate miss (the check is not vacuous)
%     (b) the smallest probe estimates the field to a pinned error (the
%         Jacobian predicts the probes), while the probe-intensity
%         diagnostic reads the same at 0.1x and at the default, off 1 by a
%         pinned amount: on the telescope it is an amplitude-independent
%         second-order offset, not a Jacobian meter (the stage's finding)
%     (c) photon noise raises the floor in order: 1e6 > 1e8 > noiseless
%     (d) ctb_efc closed on the pick's estimate digs the TRUE contrast in
%         three iterations, to what perfect sensing digs, and reports both
%         curves
%   Model 512 (the CTB's), the SUITE_CTB_512 group of run_mmacos_tests.sh.
%
%   PLAN_CONSOLIDATION 3a, pairwise probing on the coupled system (CCL, 2026-10-30).

    properties (Constant)
        ModelSize = 512
        LyotInner = 0.08           % stage 9's pick (tE2eCoupled.LyotInner)
        PickRel   = 0.3            % stage 10's pick of the default amplitude
        % Measured 2026-10-30 on the committed coupled deck with the setup's Jacobian
        % (nact 12, 3-5 lambda/D, central differences at 2 nm, 264 columns, 600 s);
        % the pins are one-sided bounds on the measured values, named beside each
        RelErrMaxPick    = 3e-3     % (a) measured 1.220e-3 at 0.3x (1.77 nm)
        RelErrMinDefault = 1e-2     % (a) measured 1.368e-2 at the default (5.90 nm; the small Jacobian's
                                    %     probes are fainter than the stage's 37 nm, the dark zone 3-5 lambda/D brighter)
        RelErrMaxSmall   = 1e-3     % (b) at 0.1x (0.59 nm); the squared scaling from (a) puts it near 1.5e-4
        RatioDriftMax    = 0.02     % (b) the ratios move 0.003 between 0.3x and the default ([1.046 1.021 1.024] vs [1.043 1.019 1.022])
        RatioOffsetMin   = 0.02     % (b) measured 0.046 at 0.3x and 0.043 at the default (the stage's full Jacobian: 0.22)
        EfcGainMin       = 0.9      % (d) measured c_after/c_before = 0.825 (perfect sensing 0.824 in this
                                    %     12-actuator, 3-5 lambda/D configuration: the hole is the Jacobian's, not the sensing's)
        EfcVsPerfectMax  = 1.05     % (d) measured 1.0012: the estimate digs what perfect sensing digs
    end

    properties
        work, J, ch, e
    end

    methods (TestClassSetup)
        function setup(tc)
            root = fullfile(fileparts(mfilename('fullpath')), '..');
            run(fullfile(root, 'mmacos_setup.m'));
            e2e   = fullfile(root, 'templates', '80_end_to_end', 'e2e');
            bench = fullfile(root, 'templates', '30_instruments', 'bench_ctb');
            addpath(bench);
            compact = fullfile(e2e, 's9_coupled.in');
            tc.assumeTrue(isfile(compact), 'the e2e stage-9 deck is not committed here');
            tc.work = tempname;  mkdir(tc.work);
            tc.addTeardown(@() rmdir(tc.work, 's'));
            prev = cd(tc.work);  tc.addTeardown(@() cd(prev));   % GridFile= resolves against the cwd
            rr = ctb_dm_rx('rx_in', compact, 'rx_out', fullfile(tc.work, 's9_coupled_dm.in'));
            names = regexp(fileread(compact), '(?m)^\s*EltName=\s*(\S+)', 'tokens');  names = [names{:}];
            ix = @(nm) find(strcmp(names, nm), 1);
            tc.e = struct('DM1', ix('DM1'), 'DM2', ix('DM2'), 'Apodizer', ix('Apodizer'), 'FPM', ix('FPM'), ...
                          'Lyot', ix('Lyot'), 'ExitPupil', ix('ExitPupil'), 'FPA', ix('FPA'));
            chain = {'elt', tc.e, 'r_lyot_inner_frac', tc.LyotInner};
            tc.J = ctb_dm_jacobian('dmrx', rr, 'chain', chain, 'model_size', tc.ModelSize, 'nact', 12, ...
                                   'inner_lamD', 3, 'outer_lamD', 5, 'save', false, 'verbose', false);
            tc.ch = ctb_chain('rx', rr.rx_out, 'model_size', tc.ModelSize, chain{:});
        end
    end

    methods (Test)
        function test_default_rule_misses_and_the_pick_matches(tc)
            e0 = ctb_probe_estimate(tc.ch, tc.J, {});
            e1 = ctb_probe_estimate(tc.ch, tc.J, {}, 'probe_rel', tc.PickRel);
            tc.verifyEqual(e1.probe_nm, tc.PickRel * e0.probe_nm, 'RelTol', 1e-12, 'the pick is that fraction of the default');
            tc.verifyGreaterThan(e0.rel_err, tc.RelErrMinDefault, sprintf( ...
                'the default rule (%.1f nm on this halo) must miss the field: relative error %.3e', e0.probe_nm, e0.rel_err));
            tc.verifyLessThan(e1.rel_err, tc.RelErrMaxPick, sprintf( ...
                'the pick (%.1f nm) matches the field: relative error %.3e', e1.probe_nm, e1.rel_err));
            % the second-order scaling: the error ratio is about the amplitude ratio squared
            tc.verifyGreaterThan(e0.rel_err / e1.rel_err, 0.5 / tc.PickRel^2, sprintf( ...
                'error ratio %.2f against (1/%.1f)^2 = %.1f', e0.rel_err / e1.rel_err, tc.PickRel, 1/tc.PickRel^2));
            % non-vacuity: one probe's difference image with the wrong sign
            dI = e1.Ip - e1.Im;  dI(:, 2) = -dI(:, 2);
            bad = ctb_probe_solve(dI, e1.dE);
            rb = norm(bad - e1.E_true) / norm(e1.E_true);
            tc.verifyGreaterThan(rb, 10 * tc.RelErrMaxPick, sprintf( ...
                'a wrong-sign difference image must make the estimate miss (%.3e)', rb));
        end

        function test_jacobian_predicts_the_probes_and_the_ratio_is_not_the_meter(tc)
            % the smallest probe estimates the field to a pinned error (the Jacobian is right);
            % the probe-intensity diagnostic carries an amplitude-INDEPENDENT offset on the
            % telescope (the symmetric probe's second-order field projected on the shadow's
            % halo), so it reads the same at 0.1x and at the default while the estimate's
            % error moves a hundredfold -- the stage's finding
            es = ctb_probe_estimate(tc.ch, tc.J, {}, 'probe_rel', 0.1);
            e0 = ctb_probe_estimate(tc.ch, tc.J, {});
            tc.verifyLessThan(es.rel_err, tc.RelErrMaxSmall, sprintf( ...
                'the smallest probe (%.1f nm): relative error %.3e', es.probe_nm, es.rel_err));
            tc.verifyLessThan(max(abs(es.ratio - e0.ratio)), tc.RatioDriftMax, sprintf( ...
                'the diagnostic moves %s between 0.1x and the default: it does not see the nonlinearity the estimate does', ...
                mat2str(es.ratio - e0.ratio, 3)));
            tc.verifyGreaterThan(max(abs(es.ratio - 1)), tc.RatioOffsetMin, sprintf( ...
                'the diagnostic''s offset on the telescope is pinned: ratios %s', mat2str(es.ratio, 4)));
        end

        function test_photon_floors_are_ordered(tc)
            e1 = ctb_probe_estimate(tc.ch, tc.J, {}, 'probe_rel', tc.PickRel);
            e8 = ctb_probe_estimate(tc.ch, tc.J, {}, 'probe_rel', tc.PickRel, 'photons', 1e8);
            e6 = ctb_probe_estimate(tc.ch, tc.J, {}, 'probe_rel', tc.PickRel, 'photons', 1e6);
            tc.verifyGreaterThan(e8.floor, e1.floor, 'photon noise at 1e8 raises the floor');
            tc.verifyGreaterThan(e6.floor, 10 * e8.floor, sprintf( ...
                '1e6 photons: %.2e against 1e8: %.2e (a hundredfold fewer photons, a tenfold floor at least)', e6.floor, e8.floor));
        end

        function test_efc_closes_on_the_estimate(tc)
            out = ctb_efc('jac', tc.J, 'niter', 3, 'save', false, ...
                'sensing', struct('probe_rel', tc.PickRel));
            tc.verifyLessThan(out.c_after, tc.EfcGainMin * out.c_before, sprintf( ...
                'EFC on the probe estimate must dig the TRUE contrast: %.3e -> %.3e', out.c_before, out.c_after));
            outT = ctb_efc('jac', tc.J, 'niter', 3, 'save', false);
            tc.verifyLessThan(out.c_after, tc.EfcVsPerfectMax * outT.c_after, sprintf( ...
                'the estimate digs what perfect sensing digs: %.3e against %.3e', out.c_after, outT.c_after));
            n = numel(out.contrast);
            tc.verifyEqual(numel(out.contrast_est), n);
            tc.verifyTrue(all(isfinite(out.contrast_est(1:n-1))), 'the estimated contrast is reported for every iteration');
            tc.verifyTrue(all(isfinite(out.est_floor(1:n-1))));
        end
    end
end
