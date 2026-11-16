classdef tBeamDirHook < matlab.unittest.TestCase
%TBEAMDIRHOOK  Telescope.optimize's chief-DIRECTION rows ('beam_dir').
%   The design-layer side of the engine's OptBeamDir= rows on a WFE target
%   (macos 64c0a90; api calib_set_beam('dir', ...)), added 2026-10-04 for
%   dyson5 TMA step 3: a Dyson behind the slit wants a TELECENTRIC image,
%   and the d205 section's exit pupil sat 22 mm before its image (42% of
%   the beam admitted at +-1.17 deg, BRIEF_dyson5_tma.md step 3).
%
%   What is pinned, on tAsphHook's TMA parent (f 330 mm, F/1.8), M3's
%   TIP/TILT the only DOFs, a chief-direction target 2 mrad off the nominal
%   at the detector:
%   (1) on a 1-urad field pair (the target reachable at both CALIB fields)
%       the solve WITH 'beam_dir' brings the chief within 2e-5 of the
%       target; the same solve WITHOUT it stays >1e-3 off (non-vacuity);
%       and the rows are OFF afterwards (a following plain solve does not
%       chase the old target -- session state must not leak);
%   (2) ONE target is scored at EVERY field, so on a non-telecentric design
%       the per-field chiefs STRADDLE it: at 1 arcmin this parent's two
%       chiefs differ by ~6e-3 at the image (its exit pupil is close to the
%       image), the solve puts them 3.2e-3 / 3.0e-3 either side and their
%       mean within 8.5e-5 of the target (measured).  That spread is the
%       pupil position the rows measure; tilting M3 cannot remove it -- the
%       layout DOFs must be free to make the image telecentric.
%   Bad input is refused.
    properties (Constant)
        ModelSize = 256
        Tilt      = 2e-3      % the target's rotation from the nominal chief, rad
    end
    methods (Access = private)
        function tel = parent_(~)
            D = 0.330/1.8;
            [R, t] = macos.design.tma_layout(D, 1.0, 1.8, 'secondary_mag', 3.5, ...
                                              'int_focus_m', -0.125*D, 'm3_behind_m', 0.6*D);
            tel = macos.design.Telescope('family','TMA','aperture_diameter_m',D,'wavelength_m',633e-9,'model_size',256);
            tel.add_mirror('M1','radius_m',R(1),'spacing_after_m',t(1));
            tel.add_mirror('M2','radius_m',R(2),'spacing_after_m',t(2),'convex',true);
            tel.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
            tel.add_focal_plane('FP');
            tel.build();
        end
        function d = chief_dir_(~, tel, ay)
            % the chief's direction AT the detector at field ay (rad, +y; 0 = nominal), on the design as built
            if nargin < 3, ay = 0; end
            tel.build();
            if ay ~= 0, macos.set_src_fov('src_dir', [0; sin(ay); cos(ay)]);  macos.modify(); end
            nE = macos.num_elt();  tr = macos.trace(nE);
            ri = macos.get_ray_info(tr.nRays);  d = ri.dir(:, 1);  d = d/norm(d);
        end
        function k = m3_(~, tel)
            k = find(strcmp({tel.spec.elt.name}, 'M3'), 1);
        end
    end
    methods (TestClassSetup)
        function setupClass(tc)
            macos.init(tc.ModelSize);
        end
    end
    methods (Test)
        function test_beam_dir_drives_the_chief_direction(tc)
            tel = tc.parent_();  d0 = tc.chief_dir_(tel);  k3 = tc.m3_(tel);
            tc.assertNotEmpty(k3, 'the parent has an M3');
            c = cos(tc.Tilt);  s = sin(tc.Tilt);
            tgt = [1 0 0; 0 c -s; 0 s c]*d0;           % 2 mrad about x from the nominal chief
            tc.assertGreaterThan(norm(d0 - tgt), 1e-3, 'the seed must start off the target');
            % control: the same DOFs and fields WITHOUT the direction rows
            ctl = tc.parent_();
            F = [0 1e-6];                                % a 1-urad field pair: the target is reachable at both fields
            ctl.optimize('fields', F, 'elts', k3, 'dofs', [1 1 0 0 0 0 0 0], 'max_iters', 40);
            dc = tc.chief_dir_(ctl);
            tc.verifyGreaterThan(norm(dc - tgt), 1e-3, ...
                sprintf('control: without beam_dir the chief must stay off the target (%.3e)', norm(dc - tgt)));
            % the hook
            tel.optimize('fields', F, 'elts', k3, 'dofs', [1 1 0 0 0 0 0 0], 'max_iters', 40, ...
                         'beam_dir', tgt, 'beam_wt', 1e6);
            d1 = tc.chief_dir_(tel);
            tc.verifyLessThan(norm(d1 - tgt), 2e-5, ...
                sprintf('with beam_dir the chief must meet the target: |d - t| = %.3e (seed %.3e)', norm(d1 - tgt), norm(d0 - tgt)));
            % the rows must be OFF afterwards: a plain solve from here must not keep chasing tgt
            tel.optimize('fields', F, 'elts', k3, 'dofs', [1 1 0 0 0 0 0 0], 'max_iters', 40);
            d2 = tc.chief_dir_(tel);
            tc.verifyGreaterThan(norm(d2 - tgt), 1e-3, ...
                sprintf('the direction rows leaked into the next solve: |d - t| = %.3e', norm(d2 - tgt)));
        end
        function test_one_target_is_scored_at_every_field(tc)
            tel = tc.parent_();  d0 = tc.chief_dir_(tel);  k3 = tc.m3_(tel);
            c = cos(tc.Tilt);  s = sin(tc.Tilt);  tgt = [1 0 0; 0 c -s; 0 s c]*d0;
            a = deg2rad(1/60);
            sp0 = norm(tc.chief_dir_(tel, a) - d0);
            tc.assertGreaterThan(sp0, 1e-3, sprintf('the parent must be non-telecentric at 1 arcmin (spread %.3e)', sp0));
            tel.optimize('fields_arcmin', 1, 'elts', k3, 'dofs', [1 1 0 0 0 0 0 0], 'max_iters', 40, ...
                         'beam_dir', tgt, 'beam_wt', 1e6);
            d1 = tc.chief_dir_(tel);  d2 = tc.chief_dir_(tel, a);
            tc.verifyGreaterThan(norm(d1 - tgt), 1e-3, 'a non-telecentric pair cannot both meet one target');
            tc.verifyGreaterThan(norm(d2 - tgt), 1e-3, 'a non-telecentric pair cannot both meet one target');
            tc.verifyLessThan(norm((d1 + d2)/2 - tgt), 2e-4, ...
                sprintf('the two fields must STRADDLE the target: |mean - t| = %.3e', norm((d1 + d2)/2 - tgt)));
        end
        function test_beam_dir_input_is_checked(tc)
            tel = tc.parent_();
            tc.verifyError(@() tel.optimize('fields_arcmin', 1, 'beam_dir', [0; 0; 0], 'max_iters', 2), ...
                'macos:design:Telescope:optimize:beamDir');
            tc.verifyError(@() tel.optimize('fields_arcmin', 1, 'beam_dir', [0 0; 0 0; 1 1], 'max_iters', 2), ...
                'macos:design:Telescope:optimize:beamDir');
        end
    end
end
