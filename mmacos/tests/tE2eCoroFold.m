classdef tE2eCoroFold < matlab.unittest.TestCase
%TE2ECOROFOLD  e2e stage 8: the CTB coronagraph folded into the telescope's shroud.
%   Replays the stage's recorded pick (e2e_params P.coro.pick) on the
%   committed stage-2 deck and gates it: the fold is an isometry (rays,
%   wavefront and DM walk equal to the unfolded case C), every surface
%   under the coronagraph AOI rule, the CTB inside the telescope's own
%   envelope with the recorded margin, clear of every telescope body and
%   beam and of itself, no lower than the bench.  Two non-vacuity checks:
%   the clearance measure reports a planted body in the beam as a negative
%   number and the primary's hole as the design; the relay scale moves the
%   physical lambda/D at the masks.  Model 256 (the stage's), the size=256
%   group of run_mmacos_tests.sh.
%
%   PLAN_CONSOLIDATION 3a, the fold (CCL, 2026-10-29).

    properties (Constant)
        ModelSize = 256
    end
    properties
        root; tel; Pc; Q; work
    end

    methods (TestClassSetup)
        function setupClass(tc)
            tc.root = fullfile(fileparts(mfilename('fullpath')), '..');
            addpath(fullfile(tc.root, 'design', 'src'));
            e2e = fullfile(tc.root, 'templates', '80_end_to_end', 'e2e');
            addpath(e2e);
            tc.tel = fullfile(e2e, 's2_instrument.in');
            tc.assumeTrue(isfile(tc.tel), 'the e2e stage-2 deck is not committed here');
            tc.Pc = ctb_template_params();
            P = e2e_params();  tc.Q = P.coro;
            tc.work = tempname;  mkdir(tc.work);
            macos.init(tc.ModelSize);
        end
    end
    methods (TestClassTeardown)
        function teardownClass(tc)
            if isfolder(tc.work), rmdir(tc.work, 's'); end
        end
    end

    methods (Test)
        function test_stop_image_and_unfolded_overflow(tc)
            % case C: DM1 at the stop's image; unfolded, the CTB leaves the envelope
            Rm = ctb_fold_build(tc.tel, fullfile(tc.work, 'm'), tc.Pc, struct('model', tc.ModelSize));
            s_img = ctb_stop_image(Rm, 3);
            tc.verifyEqual(s_img, 0.465, 'AbsTol', 0.004, 'the stop image after OAP1 (chapter 2: 465 mm)');
            sh = shroud_deck(tc.tel, struct('shroud_D_m', 0, 'model', tc.ModelSize), 'model', tc.ModelSize);
            tc.verifyEqual(sh.D, 5.0, 'AbsTol', 0.1, 'the telescope''s own hardware union, 1.25 x D');
            Rc = ctb_fold_build(tc.tel, fullfile(tc.work, 'C'), tc.Pc, struct('dm1_m', round(s_img, 4), 'model', tc.ModelSize));
            Mc = ctb_fold_measure(Rc, 'holes', struct('M1', 0.232), 'leg_from', 3, 'clear_as', tc.Q.clear_as, 'quiet', true);
            tc.verifyGreaterThan(Mc.shroud.ctb_max_r, sh.D/2 + 3, 'unfolded, the CTB reaches metres outside the envelope');
            tc.verifyEqual(Mc.rays(end), Mc.rays(1), 'every ray that reaches OAP1 reaches the FPA');
            tc.verifyEqual(Mc.walk.dm2_rpup, 0.44, 'AbsTol', 0.03, 'the chapter''s DM2 walk at 10" (0.44 of the pupil radius)');
        end

        function test_folded_pick_fits_and_is_an_isometry(tc)
            Q = tc.Q;  rp = Q.pick;
            Rm = ctb_fold_build(tc.tel, fullfile(tc.work, 'm2'), tc.Pc, struct('model', tc.ModelSize));
            dm1 = round(ctb_stop_image(Rm, 3), 4);
            holes = struct('M1', 0.232);
            Rc = ctb_fold_build(tc.tel, fullfile(tc.work, 'C2'), tc.Pc, struct('dm1_m', dm1, 'model', tc.ModelSize));
            Mc = ctb_fold_measure(Rc, 'holes', holes, 'leg_from', 3, 'clear_as', tc.Q.clear_as, 'quiet', true);
            Kp = struct('fold0_m', rp.fold_m, 'fold0_tilt_deg', rp.tilt_deg, 'plane_deg', Q.plane_deg, 'sign0', rp.sign, ...
                        'flip_at', rp.flip_at, 'relay_scale', rp.scale, 'dm1_m', dm1, 'model', tc.ModelSize);
            Rf = ctb_fold_build(tc.tel, fullfile(tc.work, 'F'), tc.Pc, Kp);
            Mf = ctb_fold_measure(Rf, 'holes', holes, 'leg_from', 3, 'clear_as', tc.Q.clear_as, 'quiet', true);
            sh = shroud_deck(tc.tel, struct('shroud_D_m', 0, 'model', tc.ModelSize), 'model', tc.ModelSize);
            Rgate = sh.D/2 + Q.shroud_grow_m;
            % the isometry
            tc.verifyEqual(Mf.rays, Mc.rays, 'rays at every station equal to the unfolded train');
            tc.verifyEqual(Mf.wfe_waves, Mc.wfe_waves, 'AbsTol', Q.wfe_tol_waves, 'the FPA wavefront is unchanged by flat folds');
            tc.verifyEqual(Mf.walk.dm2_m, Mc.walk.dm2_m, 'AbsTol', Q.walk_tol_m, 'the DM2 walk at 10" is unchanged');
            tc.verifyEqual(Mf.beam_r, Mc.beam_r, 'AbsTol', 1e-5, 'the beam radii are unchanged');
            % the rules, each a number
            aoi = [Mf.aoi.aoi_deg];  names = {Mf.aoi.name};
            tc.verifyLessThan(max(aoi), Q.aoi_max_deg, 'every surface under the coronagraph AOI rule');
            tc.verifyEqual(aoi(~strcmp(names, 'F0')), tc.Pc.AOI*ones(1, nnz(~strcmp(names, 'F0'))), 'AbsTol', 0.05, 'the OAPs and DMs at the template''s AOI');
            tc.verifyEqual(aoi(strcmp(names, 'F0')), rp.tilt_deg/2, 'AbsTol', 0.05, 'the added fold''s AOI is half its tilt');
            margin = Rgate - Mf.shroud.ctb_max_r;
            tc.verifyGreaterThanOrEqual(margin, Q.shroud_min_m, 'the CTB stays inside the envelope by the floor');
            tc.verifyEqual(margin, 0.235, 'AbsTol', 0.02, 'the recorded shroud margin of the pick (s8_report 2026-10-29)');
            tc.verifyGreaterThanOrEqual(Mf.clear.min, Q.clear_min_m, 'CTB vs telescope clearance');
            tc.verifyGreaterThan(Mf.clear.min, 0.12, 'the recorded CTB-vs-telescope clearance (~160 mm)');
            tc.verifyGreaterThanOrEqual(Mf.clear.min_addon, Q.clear_min_m, 'the CTB''s own clearance');
            tc.verifyGreaterThanOrEqual(Mf.zmin_ctb, Q.z_floor_m - 1e-3, 'no CTB body below the bench plane (a millimetre: the footprint''s mean sits microns under it)');
            zc = [Mf.clear.bodies([Mf.clear.bodies.elt] > Rf.n_base).z];
            tc.verifyLessThan(max(zc), 1.0, 'the folded CTB stays within a metre of height behind the bench');
            tc.verifyEqual(Rf.lamD_fpm_um, Rc.lamD_fpm_um, 'AbsTol', 1e-9, 'relay scale 1: the mask scale is the template''s');
        end

        function test_clearance_measure_is_not_vacuous(tc)
            % a planted body in a beam reads negative; the primary's hole is the design
            Rf = ctb_fold_build(tc.tel, fullfile(tc.work, 'F2'), tc.Pc, struct('dm1_m', 0.465, 'fold0_m', 0.3, 'fold0_tilt_deg', 8, ...
                                'plane_deg', 90, 'sign0', 1, 'flip_at', 4, 'model', tc.ModelSize));
            nF = macos.load_rx(Rf.full_in);
            C0 = deck_clearance(Rf.full_in, 'model', tc.ModelSize, 'holes', struct('M1', 0.232), 'elts', 3:nF, 'addon_from', Rf.n_base, 'quiet', true);
            tc.verifyGreaterThan(C0.min, 0, 'the pick clears');
            % plant: DM2's body grown to half a metre must stand in its neighbours' legs
            C1 = deck_clearance(Rf.full_in, 'model', tc.ModelSize, 'holes', struct('M1', 0.232), 'elts', 3:nF, 'addon_from', Rf.n_base, ...
                                'bodies', struct('DM2', 0.5), 'quiet', true);
            tc.verifyLessThan(C1.min_addon, 0, 'a planted half-metre DM2 is pierced by the CTB''s own legs');
            % the hole: without it the feed through M1 is a piercing of metres
            C2 = deck_clearance(Rf.full_in, 'model', tc.ModelSize, 'elts', 1:nF, 'addon_from', Rf.n_base, 'quiet', true);
            tc.verifyLessThan(C2.min_base, -1, 'without the hole, the M2 -> FM feed pierces the primary');
            tc.verifyGreaterThan(C0.min_base, -0.05, 'with the hole, the telescope''s own worst pair is its fold flat (mm-class)');
        end

        function test_relay_scale_moves_the_mask_scale(tc)
            R1 = ctb_fold_build(tc.tel, fullfile(tc.work, 's1'), tc.Pc, struct('dm1_m', 0.465, 'model', tc.ModelSize));
            R2 = ctb_fold_build(tc.tel, fullfile(tc.work, 's2'), tc.Pc, struct('dm1_m', 0.465, 'relay_scale', 0.5, 'model', tc.ModelSize));
            tc.verifyEqual(R2.lamD_fpm_um, 0.5*R1.lamD_fpm_um, 'RelTol', 1e-9, 'lambda/D at the FPM scales with the relay');
            tc.verifyEqual(R1.lamD_fpm_um, 1.35*0.5e-6/0.021375*1e6, 'RelTol', 1e-6, 'f4 * lambda / D_pup at the template''s numbers (31.6 um)');
        end
    end
end
