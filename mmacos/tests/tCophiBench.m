classdef tCophiBench < matlab.unittest.TestCase
%TCOPHIBENCH  COPHI's bench: the traced LO arm at the reference flat's seat
%   registers with the test arm at the camera (BRIEF_to_cophi.md sec. 1 + 3).
%   Its own class, NOT SUITE_FAST: it builds both TG96 rigs at model 1024
%   (~1 min each) through cophi_run's 'bench' stage, into a temp dir.
%
%   Gates (both rigs: the lens rig and the mirror rig of record):
%     B1  the chief rays of the two decks land on the same camera pixel (< 1 px)
%     B2  the LO covers the test arm's pupil image (> 0.99 of it above half
%         the LO's peak) and holds it inside its overfill
%     B3  the null on the flat DM -- the interference phase of the two traced
%         fields, LO placed on the test arm's pixels by camera position -- is
%         under 1 nm rms of surface (lens rig: under 0.01 nm, the two decks'
%         maps are the same there)
%     B4  fringe contrast over the pupil > 0.95 with the solved launches
%   Must-FAIL leg:
%     R1  the LO placed BY RAY INDEX (the engine's per-index detector field, as
%         the record's IFO can use it because its reference arm shares the test
%         arm's collimator) gives a null of > 50 nm on the mirror rig, where the
%         off-axis parabola and the LO's lens map the source grid differently:
%         the position placement is load-bearing, not cosmetic.
%   The battery legs (2026-10-12; one deck pass on the lens rig, model 1024):
%     B5  a 1 nm single-actuator poke at the hold-out reads 1.000 +- 0.01 through
%         the matrix measured on the 30 nm surface (the differential protocol),
%         one-step AND converged
%     B6  a 1 nrad LO tilt read through the matrix appears as a DM tilt plane:
%         the absorption (plane slope / (theta/2)) is MEASURED and stated, and
%         lies in [0.9, 1.1] -- the loop cannot tell LO tilt from DM tilt
%     R2  NO INTERFERENCE (an incoherent LO: the frames carry no fringe) is
%         refused BY NAME, cophi:noFringe (amp/dc under the contrast floor), not
%         read as a plausible zero phase; the same frames WITH the fringe read
%     R3  the camera at well = 0 (no light) is refused by the same name
    properties (Constant)
        DIR = fullfile(fileparts(mfilename('fullpath')), '..', 'templates', '40_benches', 'cophi_dm96');
    end
    properties
        R    % struct: .lens / .oap registration records
        NZ   % the gate deck pass: .rows .rows_conv .noise (lens)
        C    % the lens rig's reading context (the red legs)
        tmp  % the temp run root
    end
    methods (TestClassSetup)
        function build(tc)
            addpath(tCophiBench.DIR);
            tmp = tempname;  mkdir(tmp);
            tc.addTeardown(@() rmdir(tmp, 's'));
            o1 = cophi_run('stages', {'bench'}, 'tag', 'tb_lens', 'outdir', fullfile(tmp, 'lens'));
            o2 = cophi_run('stages', {'bench'}, 'tag', 'tb_oap', 'outdir', fullfile(tmp, 'oap'), 'bench.optics', 'oap');
            tc.R = struct('lens', o1.bench.reg, 'oap', o2.bench.reg);
            % the battery legs: one deck pass (lens), the poke at 1 nm, both estimators,
            % the LO-tilt probe; one camera, two draws (the camera rows are not gated here)
            cam = struct('baseline', cophi_params().cophi.cameras.baseline);
            o3 = cophi_run('stages', {'noise'}, 'tag', 'tb_gate', 'outdir', fullfile(tmp, 'gate'), ...
                'battery.dev_single', 1e-6, 'battery.est_mode', 'both', 'cophi.gate_probes', true, ...
                'cophi.cameras', cam, 'cophi.noise_nreal', 2, 'cophi.elec', struct('walk', 1e-3));
            tc.NZ = o3.noise;
            % the lens rig's reading context, for the red legs (decks in its run dir)
            old = cd(fullfile(tmp, 'lens'));  tc.addTeardown(@() cd(old));
            tc.C = cophi_ctx(o1.P, o1.bench.G, @(varargin) []);
        end
    end
    methods (Static)
        function nm = null_nm(Et, Elo, msk, lam_mm)
            X = sum(double(Et) .* conj(double(Elo)), 3);
            p = angle(X(msk) .* conj(mean(X(msk))));
            nm = std(p, 1) * lam_mm*1e6/(4*pi);
        end
    end
    methods (Test)
        function B1_chief_same_pixel(tc)
            for r = {'lens', 'oap'}
                R = tc.R.(r{1});
                chL = (R.frL.chief - R.frL.cen)*R.dxL/R.dxT + R.frT.cen;
                tc.verifyLessThan(max(abs(chL - R.frT.chief)), 1, r{1});
            end
        end
        function B2_lo_covers_the_pupil(tc)
            for r = {'lens', 'oap'}
                R = tc.R.(r{1});
                tc.verifyGreaterThan(R.cover, 0.99, r{1});
                tc.verifyLessThan(norm(R.cen_L - R.cen_T), R.rad_L - R.rad_T, r{1});
            end
        end
        function B3_null_on_the_flat_dm(tc)
            lam = 6.328e-4;
            nl = tCophiBench.null_nm(tc.R.lens.Et, tc.R.lens.Elo, tc.R.lens.mskT, lam);
            no = tCophiBench.null_nm(tc.R.oap.Et,  tc.R.oap.Elo,  tc.R.oap.mskT,  lam);
            tc.verifyLessThan(nl, 0.01, 'lens');
            tc.verifyLessThan(no, 1, 'oap');
        end
        function B4_contrast(tc)
            tc.verifyGreaterThan(tc.R.lens.g_med, 0.95);
            tc.verifyGreaterThan(tc.R.oap.g_med, 0.95);
        end
        function B5_one_nm_poke_through_the_matrix(tc)
            r1 = tc.NZ.rows(1, :);  rc = tc.NZ.rows_conv(1, :);
            fprintf('B5: %s one-step gain %.4f, converged %.4f\n', r1{1}, r1{2}, rc{2});
            tc.verifyEqual(r1{2}, 1, 'AbsTol', 0.01, 'one-step');
            tc.verifyEqual(rc{2}, 1, 'AbsTol', 0.01, 'converged');
        end
        function B6_lo_tilt_absorption_is_measured(tc)
            L = tc.NZ.noise.lo_tilt;
            fprintf('B6: LO tilt %.1g rad -> DM tilt plane absorption %.4f, plane %.3g pm, residual %.3g pm\n', ...
                L.theta, L.absorption, L.plane_pm, L.resid_pm);
            tc.verifyGreaterThan(L.absorption, 0.9);  tc.verifyLessThan(L.absorption, 1.1);
        end
        function R2_no_interference_is_refused_by_name(tc)
            E = tc.C.fieldf(zeros(tc.C.N_G));
            [Fr, al] = dmg_cophi_frames(E, tc.C.Elo, struct('N', 4));
            tc.verifyWarningFree(@() tc.C.read_frames(Fr, al), 'the fringe frames read');      % must-pass control
            I0 = sum(abs(E).^2 + abs(tc.C.Elo).^2, 3);                                         % incoherent: no fringe
            tc.verifyError(@() tc.C.read_frames(repmat(I0, 1, 1, 4), al), 'cophi:noFringe');
        end
        function R3_dark_camera_is_refused_by_name(tc)
            E = tc.C.fieldf(zeros(tc.C.N_G));
            [Fr, al] = dmg_cophi_frames(E, tc.C.Elo, struct('N', 4));
            Fd = dmg_cophi_camera(Fr, struct('well', 0, 'read', 0, 'fill', 0.5), 1);
            tc.verifyError(@() tc.C.read_frames(Fd, al), 'cophi:noFringe');
        end
        function R1_index_placement_must_fail_on_the_mirror_rig(tc)
            R = tc.R.oap;
            nidx = tCophiBench.null_nm(R.Et, R.Elo_native, R.mskT, 6.328e-4);
            tc.verifyGreaterThan(nidx, 50);
        end
    end
end
