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
%   To come with the battery stage (brief sec. 3): a 1 nm single-actuator poke
%   through the matrix, the LO-tilt row, the no-interference and well = 0 legs.
    properties (Constant)
        DIR = fullfile(fileparts(mfilename('fullpath')), '..', 'templates', '40_benches', 'cophi_dm96');
    end
    properties
        R    % struct: .lens / .oap registration records
    end
    methods (TestClassSetup)
        function build(tc)
            addpath(tCophiBench.DIR);
            tmp = tempname;  mkdir(tmp);
            tc.addTeardown(@() rmdir(tmp, 's'));
            o1 = cophi_run('stages', {'bench'}, 'tag', 'tb_lens', 'outdir', fullfile(tmp, 'lens'));
            o2 = cophi_run('stages', {'bench'}, 'tag', 'tb_oap', 'outdir', fullfile(tmp, 'oap'), 'bench.optics', 'oap');
            tc.R = struct('lens', o1.bench.reg, 'oap', o2.bench.reg);
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
        function R1_index_placement_must_fail_on_the_mirror_rig(tc)
            R = tc.R.oap;
            nidx = tCophiBench.null_nm(R.Et, R.Elo_native, R.mskT, 6.328e-4);
            tc.verifyGreaterThan(nidx, 50);
        end
    end
end
