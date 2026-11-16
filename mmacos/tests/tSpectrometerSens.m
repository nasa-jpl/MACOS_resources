classdef tSpectrometerSens < matlab.unittest.TestCase
%TSPECTROMETERSENS  design/src/spectrometer_sens: the tolerance ladder of a spectrometer deck (addendum 49 step 1).
%
%   On the 1.5k Dyson of record (silica 130, dyson5 size:D:130), emitted by spectrometer_rx exactly as the record
%   emits it, scored by spectrometer_score on a 3 x 3 grid (cheap; the record's 7 x 7 is the runner's):
%     1  the MUST-FAIL leg: a zero-amount perturbation gives exactly zero rows (the text path itself moves nothing);
%     2  the compensator: a pure DETECTOR DEFOCUS is recovered by the focus compensator -- its residual vs the
%        compensated nominal is ~0 while the uncompensated row is not;
%     3  focus + x/y == focus: the scorer's metrics are relative, so a detector translation cannot move them;
%     4  non-vacuity: a real lens decenter moves at least one metric.
%   Model 128 (the dyson5 model; the brief's "model 256" would force a size transition in the fast batch).
    properties
        G
        M
        P
        pp
    end
    methods (TestClassSetup)
        function setup(tc)
            h = fileparts(mfilename('fullpath'));  root = fileparts(h);
            run(fullfile(root, 'mmacos_setup.m'));
            addpath(fullfile(root, 'design', 'src'));  addpath(fullfile(root, 'challenges', 'dyson5'));
            Z = load(fullfile(root, 'challenges', 'dyson5', 'dyson5_size.mat'));  rr = Z.OUT.rows;
            k = find(strcmp(string({rr.family}), 'D') & abs([rr.r_mm] - 130) < 1e-9 & strcmp(string({rr.variant}), 'solve'), 1);
            tc.assertNotEmpty(k, 'the 1.5k Dyson of record (size:D:130) is in dyson5_size.mat');
            tc.G = spectrometer_geom('dyson', rr(k).P);  tc.P = rr(k).P;
            macos.init(128);
            tc.M = spectrometer_rx(tc.G, [tempname '_tsens.in'], 'ngridpts', 21, 'name', 'tsens', 'apertures', true, 'margin', 5e-3);
            tc.pp = spectrometer_sens_defaults(tc.G, 'sellmeier', [0.6961663 0.4079426 0.8974794 0.004679148 0.01351206 97.934]);
        end
    end
    methods (TestClassTeardown)
        function teardown(tc), if exist(tc.M.file, 'file'), delete(tc.M.file); end, end
    end
    methods (Test)
        function test_a_zero_perturbation_gives_zero_rows(tc)
            q = tc.pp(strcmp({tc.pp.name}, 'grating_decenter_y'));  q.amount = 0;
            S = spectrometer_sens(tc.G, tc.M, tc.P, 'perts', q, 'comp', {}, 'lin_rows', {}, 'nx', 3, 'nlam', 3, 'quiet', true, 'xy_check', false);
            d = S.rows(1).d;
            tc.verifyEqual([d.smile d.keystone d.CRF d.SRF d.EE], zeros(1, 5), 'the text path moves nothing at zero amount');
        end
        function test_the_focus_compensator_recovers_a_detector_defocus(tc)
            q = tc.pp(strcmp({tc.pp.name}, 'detector_defocus'));  q.amount = 30e-6;
            S = spectrometer_sens(tc.G, tc.M, tc.P, 'perts', q, 'comp', {'focus'}, 'lin_rows', {}, 'nx', 3, 'nlam', 3, 'quiet', true, 'xy_check', false);
            r = S.rows(1);
            tc.verifyGreaterThan(abs(r.d.CRF) + abs(r.d.EE), 0.02, 'a 30 um detector defocus costs (non-vacuous)');
            tc.verifyLessThan(abs(r.comp.focus.resid.CRF), 0.01, 'the focus compensator recovers it (CRF)');
            tc.verifyLessThan(abs(r.comp.focus.resid.EE), 0.01, 'the focus compensator recovers it (EE)');
            tc.verifyEqual(r.comp.focus.dz_m - S.base_comp.focus.dz_m, -30e-6, 'AbsTol', 2e-6, 'and finds the defocus');
        end
        function test_a_detector_translation_cannot_move_the_metrics(tc)
            q = tc.pp(strcmp({tc.pp.name}, 'lens_decenter_y'));
            S = spectrometer_sens(tc.G, tc.M, tc.P, 'perts', q, 'comp', {}, 'lin_rows', {}, 'nx', 3, 'nlam', 3, 'quiet', true);
            a = S.xy_check.focus.abs;  b = S.xy_check.focus_xy.abs;
            tc.verifyEqual([b.smile b.keystone b.CRF b.SRF b.EE], [a.smile a.keystone a.CRF a.SRF a.EE], 'AbsTol', 1e-3, ...
                'focus + x/y == focus: the metrics are relative');
        end
        function test_a_real_perturbation_moves_a_metric(tc)
            q = tc.pp(strcmp({tc.pp.name}, 'lens_decenter_y'));  q.amount = 50e-6;
            S = spectrometer_sens(tc.G, tc.M, tc.P, 'perts', q, 'comp', {}, 'lin_rows', {}, 'nx', 3, 'nlam', 3, 'quiet', true, 'xy_check', false);
            d = S.rows(1).d;
            tc.verifyGreaterThan(max(abs([d.smile d.keystone d.CRF d.SRF d.EE])), 1e-4, 'a 50 um lens decenter is seen');
        end
    end
end
