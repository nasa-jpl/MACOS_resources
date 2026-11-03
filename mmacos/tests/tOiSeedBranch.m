classdef tOiSeedBranch < matlab.unittest.TestCase
%TOISEEDBRANCH  offset_imager's R2/R3 re-solve holds the seeded root branch (seed_R_m).
%
%   With X.eliminate = 'R2R3' the template re-solves R2 and R3 from
%   EFL + Petzval = 0 at EVERY S1/S3 iterate (OI_CLOSE -> OI_PARAXIAL).
%   That two-condition system has TWO roots (a convex and a concave M2).
%   The record path starts Newton at c2 = c3 = -1 /m every time; on the
%   dyson5 telescope's screened row (t1 140 mm, y2 0.3 -- BRIEF_dyson5_
%   beat5b section 4) that start lands on the CONCAVE root, so the
%   ladder silently solved a different telescope from the one screened.
%   The opt-in fix, offset_imager_params.seed_R_m, seeds R1 from it and
%   restarts every re-solve from the CURRENT radii (OI_PARAXIAL REQ.c0).
%
%   Legs:
%     1  MUST-FAIL (documents the defect): the record path (seed_R_m
%        empty) seeds this geometry with a CONCAVE M2.
%     2  the fix seeds it CONVEX, on the screened radii;
%     3  the fix HOLDS the branch through re-solves as R1 moves +-10 %
%        (what S1/S3 iterates do), where the record path's re-solve,
%        handed the same convex design, jumps to the concave root.
%   Engine-free except OI_CLOSE's on-axis FP pose (model 256).
%   Model size 256 group: ./run_mmacos_tests.sh freeform.  ~10 s.
%
%   See also OI_PARAXIAL, OI_SEED, OI_CLOSE, OFFSET_IMAGER_PARAMS.

    properties
        over
    end

    methods (TestClassSetup)
        function setup(tc)
            h = fileparts(mfilename('fullpath'));
            run(fullfile(fileparts(h), 'mmacos_setup.m'));
            addpath(fullfile(fileparts(h), 'templates', '10_telescopes', 'offset_imager'));
            macos.init(256);
            % the dyson5 screened row: f 126 mm, D 70 mm, t1 140, y2 0.3
            % (telescope_seed: R [400 63.65 75.69] mm, t2 37.93 mm)
            tc.over = struct('EPD_m', 0.070, 'Fno', 1.8, 'box_deg', [24.56 0.3], 'offset_deg', 0, ...
                             'z_m1_m', 0.2, 'spacings_m', [-0.140 0 0.0379331], 'seed_R1_m', -0.400, ...
                             'model', 256, 'sampling', 21);
        end
    end

    methods (Test)
        function test_record_path_lands_on_the_concave_root(tc)
            P = offset_imager_params(tc.over);           % seed_R_m empty: the record path
            X = oi_seed(P);
            tc.verifyGreaterThan(X.R(2), 0, sprintf(['record path seeded R2 = %.4f m (convex): ' ...
                'the defect this test documents is gone -- re-derive the fixture'], X.R(2)));
        end

        function test_seed_R_m_seeds_the_convex_branch(tc)
            o = tc.over;  o.seed_R_m = [-0.400 -0.0636535 -0.0756905];
            P = offset_imager_params(o);
            X = oi_seed(P);
            tc.verifyLessThan(X.R(2), 0, 'seed_R_m did not seed a convex M2');
            tc.verifyEqual(X.R, o.seed_R_m, 'AbsTol', 2e-4, 'seeded radii left the screened design');
            fo = oi_paraxial(X.R, [P.spacings_m(1) P.spacings_m(3)]);
            tc.verifyEqual(fo.EFL_m, 0.126, 'AbsTol', 1e-9);
            tc.verifyEqual(fo.petzval, 0, 'AbsTol', 1e-9);
        end

        function test_branch_held_through_resolves(tc)
            o = tc.over;  o.seed_R_m = [-0.400 -0.0636535 -0.0756905];
            Pf = offset_imager_params(o);                % the fix
            Pr = offset_imager_params(tc.over);          % the record path
            X0 = oi_seed(Pf);  X0.eliminate = 'R2R3';
            nconcave_rec = 0;
            for s = [0.90 0.95 1.00 1.05 1.10]
                X = X0;  X.R(1) = s*X0.R(1);             % an iterate moves R1
                Xf = oi_close(X, Pf, 'offset_deg', 0);
                tc.verifyLessThan(Xf.R(2), 0, sprintf('fix: R1 x %.2f re-solved to a concave M2 (%.4f m)', s, Xf.R(2)));
                Xr = oi_close(X, Pr, 'offset_deg', 0);
                nconcave_rec = nconcave_rec + (Xr.R(2) > 0);
            end
            % must-fail leg: the record path's re-solve does NOT hold it
            tc.verifyGreaterThan(nconcave_rec, 0, ['record path held the convex branch at every R1 -- ' ...
                'the leg that shows the fix is needed has no teeth']);
        end
    end
end
