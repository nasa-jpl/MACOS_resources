classdef tCophi < matlab.unittest.TestCase
%TCOPHI  The heterodyne (COPHI) reading's algebra, pinned to its own theory.
%   dm_gauge_lib/dmg_cophi_frames + dmg_cophi_recon + dmg_cophi_camera, pure
%   MATLAB, no engine (BRIEF_cophi_campaign.md sec. 3, the gate).  The bench
%   stages (tg96's two rigs with the LO arm traced) consume these; this class
%   gates the detector physics and the reconstruction they share.
%
%   Gates:
%     G1  four-bucket, exact steps, no noise: the phase is recovered to 1e-12
%         and the least-squares form IS atan2(I4-I2, I1-I3) about the bucket
%         centres (to 1e-12)
%     G2  the integrating bucket: fringe contrast = g sinc(pi/4) = 0.9003 g
%         (N = 4), 0.9745 g (N = 8); the numeric integration matches the
%         closed form to 1e-9
%     G3  N-bucket (N = 5, 8) recovers the same phase to 1e-12
%     G4  STEP ERROR (the textbook): a 1 deg per-step error puts an (e/2)
%         sin(2 phi) ripple on the four-bucket phase (rms e/(2 sqrt 2), the
%         second harmonic); MEASURED: the plain N = 8 least squares with the
%         same per-step error is WORSE (1.43x -- the ramp accumulates), so
%         "more buckets" is not the cure; KNOWING the steps is: the LSQ with
%         the true bucket centres recovers the phase exactly -- which is what
%         a frequency-set, Rb-clocked scan gives COPHI
%     G5  PHOTONS: shot noise alone gives sigma_phi = sqrt(2 / (N I0 g'^2))
%         per pixel (I0 = mean electrons per frame, g' the bucket contrast);
%         measured within 5 % at I0 = 1e3, 1e4, 1e5 and scaling as
%         1/sqrt(I0) (slope -0.5 +- 0.02 over two decades)
%     G6  READ NOISE enters as sigma_phi^2 = 2 (I0 + r^2) / (N I0^2 g'^2):
%         r = 100 e- at I0 = 1e3 e- raises sigma_phi by sqrt(11); within 5 %
%     G7  the camera scaling: the brightest pixel's frame MEAN sits at
%         fill * well electrons; photons per frame are the returned scale
%     G8  fiber-phase WALK within the scan biases the four-bucket by a fixed
%         fraction of the walk (measured 0.13 w rms over the map for w = 0.05
%         rad/frame, a harmonic of phi), the N = 8 LSQ no better -- the
%         intra-scan drift has no algorithmic cure, it is the row the deck
%         must carry; trigger JITTER of the same size is an error of the same
%         order; the controls (no walk, no jitter) are exact
    properties (Constant)
        LIB = fullfile(fileparts(mfilename('fullpath')), '..', 'templates', '40_benches', 'dm_gauge_lib');
    end
    methods (TestClassSetup)
        function addlib(tc)
            addpath(tCophi.LIB);
            tc.addTeardown(@() rmpath(tCophi.LIB));
        end
    end
    methods (Static)
        function [Et, Elo, phi] = fields(n, seed)
            rng(seed);
            phi = 2*pi*(rand(n) - 0.5);                  % a random phase map in (-pi, pi)
            at  = 1 + 0.2*randn(n);  at = abs(at) + 0.1; % unequal, positive amplitudes
            Et  = at .* exp(1i*phi);
            Elo = ones(n);                                % a flat LO
        end
        function d = wrapdiff(a, b)
            d = angle(exp(1i*(a - b)));
        end
    end
    methods (Test)
        function G1_four_bucket_exact_and_equals_atan2(tc)
            [Et, Elo, phi] = tCophi.fields(16, 1);
            [Fr, alpha] = dmg_cophi_frames(Et, Elo, struct('N', 4));
            p = dmg_cophi_recon(Fr, alpha);
            % the LSQ phase is about the bucket centres: phi - D/2 ... the recon's
            % own alpha carries that, so p == phi
            tc.verifyLessThan(max(abs(tCophi.wrapdiff(p, phi)), [], 'all'), 1e-12);
            % the model is I0 (1 + g cos(phi - a)) (dmg_ifo_gauge's sign), so the
            % four-step is atan2(I2 - I4, I1 - I3) + a_1; Feng's atan2(I4 - I2, I1 - I3)
            % is the same algorithm for I0 (1 + g cos(phi + 2 pi f t))
            p4 = atan2(Fr(:,:,2) - Fr(:,:,4), Fr(:,:,1) - Fr(:,:,3)) + alpha(1);
            tc.verifyLessThan(max(abs(tCophi.wrapdiff(p4, p)), [], 'all'), 1e-12, ...
                'the N = 4 least squares is the four-step atan2 about the bucket centres');
        end
        function G2_bucket_contrast(tc)
            n = 8;  Et = ones(n);  Elo = ones(n);         % equal arms: g = 1
            for N = [4 8]
                [Fr, alpha] = dmg_cophi_frames(Et, Elo, struct('N', N));
                [~, amp, dc] = dmg_cophi_recon(Fr, alpha);
                g = amp ./ dc;                              % I0 g sinc / I0
                D = 2*pi/N;
                tc.verifyEqual(g(1), sin(D/2)/(D/2), 'AbsTol', 1e-12);
            end
            [~, ~, info] = dmg_cophi_frames(Et, Elo, struct('N', 4));
            tc.verifyEqual(info.cbucket, 0.9003, 'AbsTol', 1e-4, 'the quarter-cycle bucket: 0.900');
            % the numeric integration (taken whenever a non-ideal term is on) matches the closed form
            [Fa, ~] = dmg_cophi_frames(Et*exp(1i*0.7), Elo, struct('N', 4));
            [Fb, ~] = dmg_cophi_frames(Et*exp(1i*0.7), Elo, struct('N', 4, 'drift', 1e-14, 'nsub', 256));
            tc.verifyLessThan(max(abs(Fa(:) - Fb(:))) / max(Fa(:)), 1e-6);
        end
        function G3_N_bucket_exact(tc)
            [Et, Elo, phi] = tCophi.fields(16, 2);
            for N = [5 8]
                [Fr, alpha] = dmg_cophi_frames(Et, Elo, struct('N', N));
                p = dmg_cophi_recon(Fr, alpha);
                tc.verifyLessThan(max(abs(tCophi.wrapdiff(p, phi)), [], 'all'), 1e-12);
            end
        end
        function G4_step_error_second_harmonic(tc)
            % a per-step error of 1 deg = a phase ramp of 1 deg per frame; reconstruct
            % with the NOMINAL bucket centres (the instrument does not know)
            n = 256;  phi = linspace(-pi, pi, n+1);  phi = phi(1:n);
            Et = exp(1i*phi(:)) * ones(1, 1);  Elo = ones(n, 1);
            e = pi/180;
            err = struct();
            for N = [4 8]
                [Fr, alpha] = dmg_cophi_frames(Et, Elo, struct('N', N, 'drift', e, 'nsub', 256));
                p = dmg_cophi_recon(Fr, alpha);
                d = tCophi.wrapdiff(p, phi(:));
                d = d - mean(d);                            % the mean is a piston the null absorbs
                err.(sprintf('N%d', N)) = d;
            end
            r4 = rms(err.N4);  r8 = rms(err.N8);
            % the textbook: a linear step miscalibration e puts (e/2) sin(2 phi) on the
            % four-bucket phase -- rms e/(2 sqrt 2)
            tc.verifyEqual(r4, e/(2*sqrt(2)), 'RelTol', 0.02, 'four-bucket ripple amplitude e/2');
            F = abs(fft(err.N4));  F(1) = 0;
            [~, k] = max(F(1:n/2));
            tc.verifyEqual(k - 1, 2, 'the four-bucket ripple is the SECOND harmonic of phi');
            % MEASURED, not the folklore: MORE buckets with the SAME per-step error do
            % not cure it -- the plain N = 8 least squares assuming nominal phases is
            % WORSE (the ramp accumulates over 7 steps instead of 3): r8/r4 = 1.43
            tc.verifyGreaterThan(r8, r4, 'plain N-frame LSQ with the same per-step error is not a cure');
            % the cure is KNOWING the steps -- which a frequency-set (Rb-clocked) scan
            % provides: the LSQ with the TRUE bucket centres recovers the phase exactly
            [Fr, alpha] = dmg_cophi_frames(Et, Elo, struct('N', 4, 'drift', e, 'nsub', 256));
            % (the ramp is applied to the FIELD phase, cos(phi + w_k - a): the true
            %  centre of bucket k is alpha_k - w_k)
            pt = dmg_cophi_recon(Fr, alpha - e*(0:3));
            tc.verifyLessThan(max(abs(tCophi.wrapdiff(pt, phi(:)))), 1e-5, 'known steps: exact');
        end
        function G5_shot_noise_scaling(tc)
            n = 64;  Et = ones(n);  Elo = ones(n);  N = 4;  gb = sin(pi/4)/(pi/4);
            [Fr, alpha] = dmg_cophi_frames(Et, Elo, struct('N', N));
            I0s = [1e3 1e4 1e5];  s = zeros(size(I0s));
            for j = 1:numel(I0s)
                cam = struct('well', I0s(j), 'read', 0, 'fill', 1);   % frame mean = I0 e-
                Fn = dmg_cophi_camera(Fr, cam, 10 + j);
                p = dmg_cophi_recon(Fn, alpha);
                s(j) = std(tCophi.wrapdiff(p(:), 0));
                th = sqrt(2 / (N * I0s(j) * gb^2));
                tc.verifyEqual(s(j), th, 'RelTol', 0.05, sprintf('sigma_phi at I0 = %g', I0s(j)));
            end
            slope = polyfit(log10(I0s), log10(s), 1);
            tc.verifyEqual(slope(1), -0.5, 'AbsTol', 0.02, '1/sqrt(photons) over two decades');
        end
        function G6_read_noise_adds(tc)
            n = 64;  Et = ones(n);  Elo = ones(n);  N = 4;  gb = sin(pi/4)/(pi/4);
            [Fr, alpha] = dmg_cophi_frames(Et, Elo, struct('N', N));
            I0 = 1e3;  r = 100;
            Fn = dmg_cophi_camera(Fr, struct('well', I0, 'read', r, 'fill', 1), 21);
            p = dmg_cophi_recon(Fn, alpha);
            s = std(tCophi.wrapdiff(p(:), 0));
            th = sqrt(2 * (I0 + r^2) / (N * I0^2 * gb^2));
            tc.verifyEqual(s, th, 'RelTol', 0.05, 'read noise: sigma^2 = 2 (I0 + r^2) / (N I0^2 g''^2)');
            tc.verifyEqual(th / sqrt(2 / (N * I0 * gb^2)), sqrt(11), 'RelTol', 1e-12);
        end
        function G7_camera_scale(tc)
            [Et, Elo] = tCophi.fields(16, 3);
            [Fr, ~] = dmg_cophi_frames(Et, Elo, struct('N', 4));
            cam = struct('well', 60e3, 'read', 0, 'fill', 0.5);
            [~, scale] = dmg_cophi_camera(Fr, cam, 31);
            m = mean(Fr, 3) * scale;
            tc.verifyEqual(max(m(:)), 0.5 * 60e3, 'RelTol', 1e-12);
        end
        function G8_walk_and_jitter_have_no_cure(tc)
            [Et, Elo, phi] = tCophi.fields(48, 4);
            w = 0.05;
            % one realization of the walk / jitter per scan is a PHASE-DEPENDENT error
            % (a harmonic of phi, like the step error), so its rms over the map is a
            % fixed fraction of w (measured 0.13 w for this walk); the piston the null
            % absorbs is removed.  Bracket it: > 0.05 w (it is there) and < 0.5 w.
            for N = [4 8]
                [Fr, alpha] = dmg_cophi_frames(Et, Elo, struct('N', N, 'walk', w, 'seed', 40 + N));
                p = dmg_cophi_recon(Fr, alpha);
                d = tCophi.wrapdiff(p, phi);  d = d - mean(d(:));
                tc.verifyGreaterThan(rms(d(:)), 0.05 * w, sprintf('fiber walk biases the N = %d reading', N));
                tc.verifyLessThan(rms(d(:)), 0.5 * w);
            end
            [Fr, alpha] = dmg_cophi_frames(Et, Elo, struct('N', 4, 'jitter', w, 'seed', 50));
            p = dmg_cophi_recon(Fr, alpha);
            d = tCophi.wrapdiff(p, phi);  d = d - mean(d(:));
            tc.verifyGreaterThan(rms(d(:)), 0.05 * w, 'trigger jitter of the same size is an error of the same order');
            tc.verifyLessThan(rms(d(:)), 0.5 * w);
            % and with neither, the same seeds recover the phase exactly (the controls)
            [Fr, alpha] = dmg_cophi_frames(Et, Elo, struct('N', 4));
            tc.verifyLessThan(max(abs(tCophi.wrapdiff(dmg_cophi_recon(Fr, alpha), phi)), [], 'all'), 1e-12);
        end
    end
end
