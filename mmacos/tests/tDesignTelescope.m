classdef tDesignTelescope < matlab.unittest.TestCase
%TDESIGNTELESCOPE  macos.design.Telescope 2-mirror builder (Sprint 2A-ii).
%   Asserts the closed-form layout + conics against the shared
%   optical-design fixtures (R/K to ~1e-5), and that the emitted
%   prescription LOADS via SMACOS and traces SPHERICAL-FREE on-axis for
%   the conic-secondary families (classical Cassegrain and Gregorian —
%   the latter exercises the concave-secondary psi=+z flip).
%
%   The full §5.3 coma signature (RC aplanat: field-WFE ∝ field²;
%   classical Cass ∝ field; DK largest) is validated in the runnable
%   reference harness (optical_design + the de-risk prototype) and is
%   documented in the reference memory; it folds into the matlab suite
%   once field-point evaluation lands.
%
%   Model size 128 (co-runs with the other design tests in the fast
%   suite); light circular grid.

    properties (Constant)
        ModelSize = 128
        GridNpts  = 21
        AbsTolRK  = 1e-5     % fixtures rounded to ~1e-6
        WfeTol    = 1e-11    % spherical-free margin (machine-zero is ~1e-15 m)
    end

    properties
        fixtures            % struct array of fixture designs
    end

    methods (TestClassSetup)
        function load_fixtures(tc)
            j = jsondecode(fileread( ...
                design_fixture_path('telescope_design_fixtures.json')));
            tc.fixtures = j.designs;
        end
    end

    methods
        function t = make_(tc, family, D, f, mg, bt)
        %MAKE_  Build a Telescope from fixture-style (D,f,m,β) inputs.
            t = macos.design.Telescope('family', family, ...
                'aperture_diameter_m', D, 'system_fnum', f/D, ...
                'primary_fnum', f/(mg*D), 'BFD_m', bt*f/mg, ...
                'model_size', tc.ModelSize, 'grid_npts', tc.GridNpts);
        end

        function t = make_tma_(tc)
        %MAKE_TMA_  The standard 3-mirror Seidel-seeded TMA fixture.
            t = macos.design.Telescope('family','TMA', ...
                'aperture_diameter_m',1.0, 'model_size',tc.ModelSize, ...
                'grid_npts',tc.GridNpts);
            t.add_mirror('M1','radius_m',8.0,'spacing_after_m',3.0);
            t.add_mirror('M2','radius_m',2.0,'spacing_after_m',4.5);
            t.add_mirror('M3','radius_m',4.0,'spacing_after','derive');
        end

        function t = make_sz_tma_(tc)
        %MAKE_SZ_TMA_  e5mono-derived sphere+Zernike unobscured TMA: M1 concave +
        %   M2 CONVEX (real intermediate image) + M3 concave, all base spheres
        %   (Kc=0), folded to clear the beam.  D=8 m, f/21.
            t = macos.design.Telescope('family','TMA', ...
                'aperture_diameter_m',8.0, 'model_size',tc.ModelSize, ...
                'grid_npts',tc.GridNpts, 'wavelength_m',1e-6);
            t.set_base_sphere(true);
            t.add_mirror('M1','radius_m',51.534,'spacing_after_m',22.0,'tilt_deg',-7.2);
            t.add_mirror('M2','radius_m', 8.871,'spacing_after_m',28.0,'tilt_deg', 8.46,'convex',true);
            t.add_mirror('M3','radius_m', 3.0,  'spacing_after','derive','tilt_deg',12.0);
            t.add_focal_plane('FP');
        end

        function v = parse_vec3_(~, txt, key)
        %PARSE_VEC3_  Extract the 3-vector after 'KEY=' in emitted Rx text.
        %   Lookbehind excludes OptChfRayDir/Pos (KEY is a suffix of those).
            pat = ['(?<![A-Za-z])' key '=\s*(\S+)\s+(\S+)\s+(\S+)'];
            tok = regexp(txt, pat, 'tokens', 'once');
            assert(~isempty(tok), 'key %s not found in emitted Rx', key);
            v = [str2double(tok{1}), str2double(tok{2}), str2double(tok{3})];
        end
    end

    methods (Test)
        function test_layout_conics_vs_fixtures(tc)
            % The hard regression gate (guide §4): every fixture's
            % R1/R2/sep/BFD/K1/K2 reproduced from design intent.
            for i = 1:numel(tc.fixtures)
                fx = tc.fixtures(i);
                in = fx.inputs;
                t  = tc.make_(fx.family, in.D_m, in.f_m, in.m, in.beta);
                d  = t.spec.derived; fo = fx.first_order; fc = fx.conics;
                nm = @(q) sprintf('%s: %s', fx.name, q);
                tc.verifyEqual(d.R1,  fo.R1_m,             'AbsTol', tc.AbsTolRK, nm('R1'));
                tc.verifyEqual(d.R2,  fo.R2_m,             'AbsTol', tc.AbsTolRK, nm('R2'));
                tc.verifyEqual(d.sep, fo.M1_M2_sep_m,      'AbsTol', tc.AbsTolRK, nm('sep'));
                tc.verifyEqual(d.bfd, fo.back_focal_dist_m,'AbsTol', tc.AbsTolRK, nm('BFD'));
                tc.verifyEqual(d.K1,  fc.K1,               'AbsTol', tc.AbsTolRK, nm('K1'));
                tc.verifyEqual(d.K2,  fc.K2,               'AbsTol', tc.AbsTolRK, nm('K2'));
            end
        end

        function test_cassegrain_loads_spherical_free(tc)
            t = tc.make_('Cassegrain', 1.0, 8.0, 4.0, 0.125);
            t.build();                                  % emit + validate-by-load
            tc.verifyTrue(macos.has_rx());
            tc.verifyEqual(macos.num_elt(), 3);
            s = macos.trace();
            tc.verifyLessThan(s.rmsWFE, tc.WfeTol, ...
                sprintf('classical Cassegrain not spherical-free: %.3e', s.rmsWFE));
        end

        function test_set_freeform_zernike_bites(tc)
        %TEST_SET_FREEFORM_ZERNIKE_BITES  A Zernike departure layered on a
        %   spherical-free Cassegrain mirror must (a) emit Surface=Zernike,
        %   (b) measurably deform the traced wavefront -- the engine actually
        %   applies it -- and (c) round-trip through save/load.  Guards the
        %   freeform DOF against a silent no-op (the ZernTypeL=0 trace-dispatch
        %   trap documented in macos/CLAUDE.md).
            t  = tc.make_('Cassegrain', 1.0, 8.0, 4.0, 0.125);
            t.build();
            s0 = macos.trace();                          % spherical-free baseline
            tc.verifyLessThan(s0.rmsWFE, tc.WfeTol, ...
                sprintf('baseline not spherical-free: %.3e', s0.rmsWFE));

            % 2 um of surface astigmatism (ANSI mode 6) on M1 -- a pure non-
            % rotationally-symmetric term the conic baseline has zero of, so it
            % shows up directly in the wavefront (not absorbed by refocus).
            t.set_freeform(1, 6, 2e-6);
            rx = t.build();                              % re-emit + reload
            s1 = macos.trace();
            tc.verifyGreaterThan(s1.rmsWFE, 1e-7, ...
                sprintf('Zernike departure did not bite (no-op?): %.3e', s1.rmsWFE));

            % (a) the element emitted a Zernike surface
            txt = fileread(rx);
            tc.verifyNotEmpty(regexp(txt, 'Surface=\s*Zernike', 'once'), ...
                'M1 did not emit Surface=Zernike');

            % (c) round-trip: a fresh load of the saved Rx reproduces the WFE
            macos.load_rx(rx);
            s2 = macos.trace();
            tc.verifyEqual(s2.rmsWFE, s1.rmsWFE, 'RelTol', 1e-9, ...
                'freeform Rx did not round-trip through save/load');

            % zeroing the departure returns to the spherical-free conic baseline
            t.set_freeform(1, 6, 0);
            t.build();
            s3 = macos.trace();
            tc.verifyEqual(s3.rmsWFE, s0.rmsWFE, 'AbsTol', 10*tc.WfeTol, ...
                'zeroing the Zernike did not restore the conic baseline');
        end

        function test_optimize_freeform_corrects_injected(tc)
        %TEST_OPTIMIZE_FREEFORM_CORRECTS_INJECTED  optimize_freeform must drive
        %   a known injected Zernike departure back out -- proving the optimize
        %   loop (re-emit + trace + minimise, radii/conics held) actually
        %   closes.  On-axis, single mode: well-posed and fast.
            t = tc.make_('Cassegrain', 1.0, 8.0, 4.0, 0.125);
            t.set_freeform(1, 6, 2e-6);                  % inject 2 um astig on M1
            t.build();  s_err = macos.trace();
            tc.verifyGreaterThan(s_err.rmsWFE, 1e-7, 'injected error not present');

            % on-axis only (fields_arcmin=[]): well-posed 1-DOF removal via the
            % native CALIB OptZern channel (radii/conics held).
            res   = t.optimize_freeform(1, 'modes',6, 'fields_arcmin',[], 'max_iters',60);
            s_fix = macos.trace();
            tc.verifyLessThan(s_fix.rmsWFE, s_err.rmsWFE/10, ...
                sprintf('did not correct injected astig: %.3e -> %.3e', ...
                        s_err.rmsWFE, s_fix.rmsWFE));
            tc.verifyLessThan(res.wfe_after(1), res.wfe_before(1)/10, ...
                'reported merit did not improve');
        end

        function test_base_sphere_zeros_conics(tc)
        %TEST_BASE_SPHERE_ZEROS_CONICS  set_base_sphere -> every mirror Kc=0
        %   (the Seidel conic seed is skipped; correction is all-Zernike).
            t = tc.make_sz_tma_();  t.build();
            for k = 1:3
                tc.verifyEqual(t.spec.elt(k).Kc, 0.0, ...
                    sprintf('mirror %d conic not zero under base_sphere', k));
            end
        end

        function test_convex_secondary_reproduces_e5mono(tc)
        %TEST_CONVEX_SECONDARY_REPRODUCES_E5MONO  the convex psi-flip emits M1/M2
        %   surface normals matching the e5mono reference (downstream centre of
        %   curvature for the convex secondary), to 1e-3.
            t = tc.make_sz_tma_();  t.build();
            tc.verifyEqual(t.spec.elt(1).psi, [0 -0.125333 -0.992115], ...
                'AbsTol',1e-3, 'M1 psi vs e5mono');
            tc.verifyEqual(t.spec.elt(2).psi, [0 -0.103482 -0.994631], ...
                'AbsTol',1e-3, 'M2 (convex) psi vs e5mono');
        end

        function test_convex_aware_focus(tc)
        %TEST_CONVEX_AWARE_FOCUS  paraxial_focus_ recovers the true f/# of a
        %   convex-secondary reimager (~f/21); the Seidel |radii| n-flip model
        %   mis-derives it (f/0.6).  Guards the convex-focus regression.
            t = tc.make_sz_tma_();  t.build();
            tc.verifyGreaterThan(t.spec.derived.fnum, 18, ...
                'convex-aware f/# too small -- seidel focus leaked through?');
            tc.verifyLessThan(t.spec.derived.fnum, 25, 'convex-aware f/# too large');
        end

        function test_convex_conic_seed_zero_and_optimizes(tc)
        %TEST_CONVEX_CONIC_SEED_ZERO_AND_OPTIMIZES  For a convex-secondary TMA
        %   the n-flip Seidel conic seed is unreliable, so seidel_seed returns a
        %   SAFE K=0 sphere seed (NOT garbage conics) + the correct unfolded
        %   focus; the conic optimize() then nulls the multi-field WFE to
        %   diffraction-limited (the j18mono f/20 geometry).
            t = macos.design.Telescope('family','TMA', ...
                'aperture_diameter_m',6.605, 'model_size',tc.ModelSize, ...
                'grid_npts',tc.GridNpts, 'wavelength_m',2.3e-6);
            t.add_mirror('M1','radius_m',15.879722,'spacing_after_m',7.169041556);
            t.add_mirror('M2','radius_m', 1.778913,'spacing_after_m',7.965313479,'convex',true);
            t.add_mirror('M3','radius_m', 3.016227,'spacing_after','derive');
            t.add_focal_plane('FP');
            t.build();
            % safe K=0 seed (NOT the broken n-flip conics) + correct f/20 focus
            tc.verifyEqual([t.spec.elt(1).Kc t.spec.elt(2).Kc t.spec.elt(3).Kc], ...
                [0 0 0], 'AbsTol',0, 'convex conic seed must be K=0');
            tc.verifyGreaterThan(t.spec.derived.fnum, 18, 'convex f/# wrong');
            tc.verifyLessThan(t.spec.derived.fnum, 25, 'convex f/# wrong');
            % the conic optimize crushes the field WFE to diffraction-limited
            res = t.optimize('fields_arcmin',[0.5 1.0], 'max_iters',150);
            tc.verifyLessThan(max(res.wfe_after)/2.3e-6, 0.07, ...
                'convex conic optimize did not reach the diffraction limit');
            tc.verifyLessThan(max(res.wfe_after), max(res.wfe_before)/1000, ...
                'convex conic optimize did not crush the WFE');
        end

        function test_zernmodes_single_line_emit_loads(tc)
        %TEST_ZERNMODES_SINGLE_LINE_EMIT_LOADS  >6 Zernike modes on a Surface=
        %   Zernike element must emit ZernModes on ONE line; wrapping at 6/row
        %   crashed the parser with a list-directed EOF (msmacosio.inc:1733).
            t = tc.make_sz_tma_();
            t.set_freeform(1, [3 4 5 9 11 12 13 19], zeros(1,8), 'type','BornWolf');
            rx = t.build();                          % re-emit + load; must not crash
            tc.verifyTrue(macos.has_rx(), 'Rx with 8 Zernike modes failed to load');
            line = regexp(fileread(rx), 'ZernModes=[^\n]*', 'match', 'once');
            tc.verifyEqual(numel(regexp(line, '\d+', 'match')), 8, ...
                'all 8 modes must be on the single ZernModes line');
        end

        function test_base_sphere_zernike_correction(tc)
        %TEST_BASE_SPHERE_ZERNIKE_CORRECTION  end-to-end recipe: uncorrected base
        %   spheres are aberration-dominated; optimizing the Zernike departures
        %   (CALIB OptZern) on-axis must crush the WFE by >100x.
            t = tc.make_sz_tma_();  t.build();
            macos.trace(numel(t.spec.elt));  W0 = macos.opd();
            wfe0 = std(W0(isfinite(W0) & W0~=0));     % m
            tc.verifyGreaterThan(wfe0/1e-6, 1000, 'base spheres should start badly aberrated');
            res = t.optimize_freeform([1 2 3], 'modes',[3 4 5 9 11], 'type','BornWolf', ...
                                      'fields_arcmin',[], 'max_iters',120);
            tc.verifyLessThan(res.wfe_after, res.wfe_before/100, ...
                sprintf('on-axis Zernike correction insufficient: %.3e -> %.3e m', ...
                        res.wfe_before, res.wfe_after));
        end

        function test_gregorian_loads_spherical_free(tc)
            % concave secondary => psiElt flips to +z; still spherical-free
            t = tc.make_('Gregorian', 1.0, 12.0, 4.0, 0.15);
            t.build();
            tc.verifyEqual(macos.num_elt(), 3);
            s = macos.trace();
            tc.verifyLessThan(s.rmsWFE, tc.WfeTol, ...
                sprintf('Gregorian not spherical-free: %.3e', s.rmsWFE));
        end

        function test_build_returns_loadable_rx_path(tc)
            t  = tc.make_('Cassegrain', 0.5, 7.5, 5.0, 0.20);
            rx = t.build();
            tc.verifyTrue(isfile(rx));
            tc.verifyEqual(t.spec.rx_path, rx);
        end

        function test_save_emits_nOutCord_block(tc)
            % The SMACOS-load-required trailing block must always be written.
            t = tc.make_('Cassegrain', 1.0, 8.0, 4.0, 0.125);
            f = [tempname '.in']; c = onCleanup(@() delete(f));
            t.save(f);                                  % emit only, no load
            txt = fileread(f);
            tc.verifyTrue(contains(txt, 'nOutCord'), 'emitted Rx missing nOutCord block');
            tc.verifyTrue(contains(txt, 'Element=  Reflector'));
            tc.verifyTrue(contains(txt, 'Surface=  Conic'));
        end

        function test_describe_runs_clean(tc)
            t = tc.make_('RC', 2.4, 57.6, 10.4, 0.271);
            evalc('t.describe()');                       % prints; must not error
        end

        function test_save_spec_roundtrip(tc)
            t = tc.make_('Dall-Kirkham', 1.0, 8.0, 4.0, 0.125);
            f = [tempname '.mat']; c = onCleanup(@() delete(f));
            t.save_spec(f);
            t2 = macos.design.Telescope.load_spec(f);
            tc.verifyEqual(t2.spec.family, t.spec.family);
            tc.verifyEqual(t2.spec.derived.K1, t.spec.derived.K1, 'AbsTol', 1e-12);
            tc.verifyEqual(t2.spec.derived.K2, t.spec.derived.K2, 'AbsTol', 1e-12);
        end

        function test_unit_sugar_mm_equals_m(tc)
            tm  = tc.make_('Cassegrain', 1.0, 8.0, 4.0, 0.125);
            tmm = macos.design.Telescope('family','Cassegrain', ...
                'aperture_diameter_mm',1000, 'system_fnum',8, ...
                'primary_fnum',2, 'BFD_mm',250, ...
                'model_size',tc.ModelSize, 'grid_npts',tc.GridNpts);
            tc.verifyEqual(tmm.spec.derived.R1, tm.spec.derived.R1, 'AbsTol',1e-12);
            tc.verifyEqual(tmm.spec.derived.bfd, tm.spec.derived.bfd, 'AbsTol',1e-12);
        end

        function test_emit_is_deterministic(tc)
            % Parity property (§3): same spec -> byte-identical .in.  (The
            % committed-golden cross-language anchor waits for the emitter
            % to stabilise in 2B/2C; determinism is the stable part now.)
            t  = tc.make_('Cassegrain', 1.0, 8.0, 4.0, 0.125);
            f1 = [tempname '.in']; f2 = [tempname '.in'];
            c  = onCleanup(@() delete(f1, f2));
            t.save(f1); t.save(f2);
            tc.verifyEqual(fileread(f2), fileread(f1));
        end

        function test_built_telescope_feeds_alignment(tc)
            % The §2 Stage 4 first result: a BUILT Cassegrain imported via
            % System recovers an M2 despace error -- closing the
            % builder -> analysis loop on the emitted prescription.
            t  = tc.make_('Cassegrain', 1.0, 8.0, 4.0, 0.125);
            rx = t.build();
            s  = macos.design.System.from_rx(rx, 'model_size', tc.ModelSize);
            s.vary(2, 'despace', 'bounds', [-2 2], 'unit', 'mm');   % M2 Tz
            base = s.evaluate(0);                  % aligned baseline (spherical-free)
            tc.verifyLessThan(base.merit, 1e-9, ...
                sprintf('built Cass baseline WFE not small: %.3e', base.merit));
            bad = s.evaluate(0.5);                 % +0.5 mm M2 despace
            tc.verifyGreaterThan(bad.merit, 1e3*base.merit + 1e-9, ...
                'M2 despace did not raise WFE');
            res = s.optimize('x0', 0.5, 'MaxIter', 40);
            tc.verifyLessThan(res.merit_opt, bad.merit/10, ...
                sprintf('optimize did not recover alignment: %.3e', res.merit_opt));
            tc.verifyLessThan(abs(res.x_opt), 0.05, ...     % recovered despace ~0 (mm)
                sprintf('recovered despace not ~0: %.4g mm', res.x_opt));
        end

        function test_add_pupil_inserts_exit_pupil(tc)
            % add_pupil inserts a FLAT image-Return + a SPHERICAL exit-pupil
            % Return BEFORE the focal plane, PRESERVING the FP (nElt += 2).
            % EP located by FEX; radius = chief-ray FP->EP, psi toward CoC.
            t  = tc.make_('Cassegrain', 1.0, 8.0, 4.0, 0.125);
            n0 = numel(t.spec.elt);                       % 3 (M1,M2,FP)
            t.add_pupil();
            p  = t.spec.pupil;
            tc.verifyEqual(numel(t.spec.elt), n0+2, '"lost the FP" -- nElt not +2');
            tc.verifyEqual(p.img_elt, n0);                % flat image return
            tc.verifyEqual(p.ep_elt,  n0+1);              % spherical EP return
            tc.verifyEqual(p.fp_elt,  n0+2);              % preserved FocalPlane
            tc.verifyEqual(t.spec.elt(p.img_elt).kind, 'Return');
            tc.verifyEqual(t.spec.elt(p.ep_elt).kind,  'Return');
            tc.verifyEqual(t.spec.elt(p.fp_elt).kind,  'FocalPlane');
            % real exit pupil behind the secondary (z<0), positive radius
            tc.verifyGreaterThan(p.ep_radius, 0);
            tc.verifyLessThan(p.ep_vpt(3), 0);
            % radius == chief-ray distance FP->EP; sphere Kr = -radius;
            % psi = -unit(FP->EP) points back toward the image (+z here)
            fpz = t.spec.elt(p.fp_elt).Vpt(3);
            tc.verifyEqual(p.ep_radius, abs(fpz - p.ep_vpt(3)), 'RelTol', 1e-9);
            tc.verifyEqual(t.spec.elt(p.ep_elt).Kr, -p.ep_radius, 'RelTol', 1e-9);
            tc.verifyGreaterThan(t.spec.elt(p.ep_elt).psi(3), 0.99);
            % augmented Rx loads + traces; wavefront still spherical-free
            tc.verifyTrue(macos.has_rx());
            tc.verifyEqual(macos.num_elt(), n0+2);
            s = macos.trace(numel(t.spec.elt));
            tc.verifyLessThan(s.rmsWFE, tc.WfeTol, ...
                sprintf('pupil refs perturbed the wavefront: %.3e', s.rmsWFE));
        end

        function test_tma_seidel_seed_matches_proof(tc)
            % Pure-math: the ported Seidel-seed solver reproduces the locked
            % proof_korsch f/8 conics (R=[8 2 4], t=[3 4.5,derive]).
            [K, tf, EFL] = macos.design.seidel_seed([8 2 4], [3 4.5], 1.0);
            tc.verifyEqual(K,   [-0.622 0.148 -3.904], 'AbsTol', 2e-3, 'seidel conics');
            tc.verifyEqual(EFL, 8.0, 'AbsTol', 1e-3, 'EFL (f/8)');
            tc.verifyEqual(tf,  2.0, 'AbsTol', 1e-3, 't_focus');
        end

        function test_tma_builder_emits_coaxial_korsch(tc)
            % N-mirror builder: add_mirror -> Seidel-seeded coaxial Korsch
            % that loads + traces at the seed residual (~0.15 lambda on-axis;
            % 3rd-order nulled, higher-order left for multi-field optimize).
            t = macos.design.Telescope('family','TMA', ...
                'aperture_diameter_m',1.0, 'model_size',tc.ModelSize, ...
                'grid_npts',tc.GridNpts);
            t.add_mirror('M1','radius_m',8.0,'spacing_after_m',3.0);
            t.add_mirror('M2','radius_m',2.0,'spacing_after_m',4.5);
            t.add_mirror('M3','radius_m',4.0,'spacing_after','derive');
            t.build();
            tc.verifyTrue(macos.has_rx());
            tc.verifyEqual(macos.num_elt(), 4);
            % builder conics == standalone solver (exact)
            Kref = macos.design.seidel_seed([8 2 4], [3 4.5], 1.0);
            tc.verifyEqual(t.spec.derived.K, Kref, 'AbsTol', 1e-12);
            % coaxial: all mirrors psi=(0,0,-1); fold vertices 0,-3,+1.5
            for k = 1:3
                tc.verifyEqual(t.spec.elt(k).psi, [0 0 -1], 'mirror psi not -z');
            end
            tc.verifyEqual(t.spec.elt(1).Vpt(3),  0.0, 'AbsTol', 1e-12);
            tc.verifyEqual(t.spec.elt(2).Vpt(3), -3.0, 'AbsTol', 1e-12);
            tc.verifyEqual(t.spec.elt(3).Vpt(3),  1.5, 'AbsTol', 1e-12);
            % on-axis: seidel-seed residual band (not yet diffraction-limited)
            s = macos.trace(4);
            tc.verifyLessThan(s.rmsWFE, 5e-7, ...
                sprintf('TMA on-axis WFE too large -- emission suspect: %.3e', s.rmsWFE));
            tc.verifyGreaterThan(s.rmsWFE, 1e-8, ...
                'TMA on-axis WFE implausibly small for an un-optimized seed');
        end

        function test_tma_multifield_optimize_native(tc)
            % Native multi-field CALIB drives the Seidel-seeded Korsch from
            % the seed residual (on-axis ~0.15 lambda, off-axis few-lambda)
            % to diffraction-limited across the field, varying ONLY conics
            % (radii/spacings fixed -- one shared physical system).
            t = macos.design.Telescope('family','TMA', ...
                'aperture_diameter_m',1.0, 'model_size',tc.ModelSize, ...
                'grid_npts',tc.GridNpts);
            t.add_mirror('M1','radius_m',8.0,'spacing_after_m',3.0);
            t.add_mirror('M2','radius_m',2.0,'spacing_after_m',4.5);
            t.add_mirror('M3','radius_m',4.0,'spacing_after','derive');
            t.build();
            Kseed = t.spec.derived.K;
            res = t.optimize('fields_arcmin',[1.2 2.4], 'max_iters',60);
            tc.verifyTrue(res.converged, 'CALIB did not converge');
            % off-axis improved by >100x -- the multi-field payoff
            tc.verifyGreaterThan(res.wfe_before(end)/res.wfe_after(end), 100, ...
                'off-axis WFE not strongly improved');
            tc.verifyLessThan(max(res.wfe_after), 5e-8, ...
                sprintf('not ~diffraction-limited: max %.3e m', max(res.wfe_after)));
            % conics actually moved and were written back to the spec
            tc.verifyGreaterThan(max(abs(res.conics - Kseed)), 0.1, 'conics did not move');
            tc.verifyEqual(t.spec.derived.K, res.conics, 'AbsTol', 1e-12);
            % the clean re-emitted design traces optimized (conic readback)
            s = macos.trace(4);
            tc.verifyLessThan(s.rmsWFE, 5e-8, ...
                sprintf('clean re-emit not optimized: %.3e m', s.rmsWFE));
        end

        function test_check_clipping_flags_coaxial_tma(tc)
            % The coaxial Korsch TMA SELF-OBSCURES: M1 (z=0) and the FP
            % (z=-0.5) sit inside the converging M2->M3 beam.  check_clipping
            % reconstructs the real ray bundle in 3-D (two orthogonal DRAW
            % projections) and must flag it -- this is the diagnosis that
            % motivates the off-axis fold builder.
            t = macos.design.Telescope('family','TMA', ...
                'aperture_diameter_m',1.0, 'model_size',tc.ModelSize, ...
                'grid_npts',tc.GridNpts);
            t.add_mirror('M1','radius_m',8.0,'spacing_after_m',3.0);
            t.add_mirror('M2','radius_m',2.0,'spacing_after_m',4.5);
            t.add_mirror('M3','radius_m',4.0,'spacing_after','derive');
            t.build();
            rep = t.check_clipping('quiet', true);     % standalone (loads Rx)
            tc.verifyEqual(numel(rep), 4, 'report should cover all 4 elements');
            tc.verifyFalse(all([rep.ok]), ...
                'coaxial TMA must register a body-in-beam conflict');
            % M1 is certainly pierced by the M2->M3 segment (crosses z=0 on-axis)
            tc.verifyGreaterThan(rep(1).obstructs, 0, ...
                'M1 body not detected in the converging M2->M3 beam');
            % and the signed clearance quantifies it: the foreign beam cuts
            % INSIDE M1's body edge -> negative clearance (vignette depth)
            tc.verifyLessThan(rep(1).clearance, 0, ...
                'M1 clearance should be negative (beam cuts the body)');
            % every footprint is a finite, non-negative radius
            tc.verifyTrue(all(isfinite([rep.foot_r]) & [rep.foot_r] >= 0));
        end

        function test_check_clipping_cassegrain_central_obs(tc)
            % An on-axis 2-mirror Cassegrain has the EXPECTED central
            % obscuration: the secondary sits in the incoming beam to the
            % primary.  check_clipping reports it as a real finding (the
            % check is honest about coaxial obscuration -- the off-axis
            % builder is what removes it).
            t = tc.make_('Cassegrain', 1.0, 8.0, 4.0, 0.125);
            t.build();
            rep = t.check_clipping('quiet', true);
            tc.verifyEqual(numel(rep), 3, 'M1 + secondary + FP');
            tc.verifyGreaterThan(rep(2).obstructs, 0, ...
                'secondary central obscuration not detected');
            tc.verifyTrue(all(isfinite([rep.margin])), 'margins must be finite');
        end

        function test_field_bias_zero_is_on_axis(tc)
            % set_field_bias(0) must emit the exact on-axis chief ray.
            t = tc.make_tma_();
            t.set_field_bias(0);
            f = [tempname '.in']; c = onCleanup(@() delete(f)); %#ok<NASGU>
            t.save(f);
            dir = tc.parse_vec3_(fileread(f), 'ChfRayDir');
            tc.verifyEqual(dir, [0 0 1], 'AbsTol', 1e-15, ...
                'zero field bias must stay on-axis');
        end

        function test_field_bias_emits_offaxis_chief_ray_pinned(tc)
            % A +y field bias tilts ChfRayDir to (0,sin a,cos a) and points
            % ChfRayPos anti-parallel (through the on-axis stop); the element
            % vertices stay PINNED on-axis (the off-axis-section invariant).
            a_arcmin = 3.0;  a = deg2rad(a_arcmin/60);
            t = tc.make_tma_();
            t.set_field_bias(a_arcmin);
            f = [tempname '.in']; c = onCleanup(@() delete(f)); %#ok<NASGU>
            t.save(f);
            txt = fileread(f);
            dir = tc.parse_vec3_(txt, 'ChfRayDir');
            tc.verifyEqual(dir, [0 sin(a) cos(a)], 'AbsTol', 1e-12, ...
                'biased chief-ray direction wrong');
            pos = tc.parse_vec3_(txt, 'ChfRayPos');
            tc.verifyEqual(pos./norm(pos), -[0 sin(a) cos(a)], 'AbsTol', 1e-9, ...
                'ChfRayPos must be anti-parallel to ChfRayDir (through the stop)');
            % vertices PINNED on-axis, psi axis-aligned -- no decenter/tilt
            for k = 1:3
                tc.verifyEqual(t.spec.elt(k).Vpt(1:2), [0 0], 'AbsTol', 1e-12, ...
                    'mirror decentered -- vertices must stay pinned');
                tc.verifyEqual(t.spec.elt(k).psi, [0 0 -1], ...
                    'mirror psi not axis-aligned');
            end
        end

        function test_field_bias_loads_traces_and_aberrates(tc)
            % The biased design loads + traces (rays not lost), and the
            % off-axis field is MORE aberrated than the on-axis seed -- the
            % bias bites, which is what optimize() will then correct.
            t0 = tc.make_tma_();  t0.build();
            s0 = macos.trace(4);

            t = tc.make_tma_();  t.set_field_bias(2.0);  t.build();
            tc.verifyTrue(macos.has_rx());
            tc.verifyEqual(macos.num_elt(), 4);
            s = macos.trace(4);
            tc.verifyTrue(isfinite(s.rmsWFE), 'biased trace WFE not finite');
            tc.verifyGreaterThan(s.rmsWFE, s0.rmsWFE, ...
                'field bias should add aberration vs the on-axis seed');
        end

        function test_field_bias_optimize_corrects_offaxis(tc)
            % optimize() centers its eval fields on the bias and re-derives
            % the conics, so the biased (off-axis) field is corrected.
            bias = 2.0;                       % arcmin
            t = tc.make_tma_();
            t.set_field_bias(bias);
            t.build();
            s_before = macos.trace(4);        % biased seed: aberrated
            res = t.optimize('fields_arcmin',[1.0 2.0], 'max_iters',60);
            tc.verifyTrue(res.converged, 'CALIB did not converge (biased)');
            % eval fields are absolute, centered on the bias (field 1 = bias)
            tc.verifyEqual(res.fields_arcmin(1), bias, 'AbsTol', 1e-9);
            tc.verifyEqual(res.fields_arcmin, bias + [0 1 2], 'AbsTol', 1e-9);
            s_after = macos.trace(4);         % re-emitted optimized @ bias
            tc.verifyLessThan(s_after.rmsWFE, s_before.rmsWFE, ...
                'optimize did not improve the biased-field WFE');
        end

        function test_aperture_decenter_zero_is_centered(tc)
            % dy=0 must emit the on-axis stop exactly.
            t = tc.make_tma_();  t.set_aperture_decenter(0);
            f = [tempname '.in']; c = onCleanup(@() delete(f)); %#ok<NASGU>
            t.save(f);
            tc.verifyEqual(tc.parse_vec3_(fileread(f), 'ApStop'), [0 0 0], ...
                'AbsTol', 1e-15, 'zero decenter must stay centered');
        end

        function test_aperture_decenter_emits_and_shifts_footprint(tc)
            % Decentering the aperture offsets the stop + chief ray and moves
            % the beam onto an OFF-AXIS patch of the pinned parent: the M1
            % footprint centroid shifts to +dy.  (Small dy so the beam still
            % fits inside the full-aperture parent -- larger decenters need
            % oversized parents, a follow-on.)
            dy = 0.03;
            t = tc.make_tma_();  t.set_aperture_decenter(dy);
            f = [tempname '.in']; c = onCleanup(@() delete(f)); %#ok<NASGU>
            t.save(f);
            txt = fileread(f);
            tc.verifyEqual(tc.parse_vec3_(txt, 'ApStop'), [0 dy 0], 'AbsTol', 1e-12, ...
                'stop not decentered');
            % vertices stay pinned on-axis
            for k = 1:3
                tc.verifyEqual(t.spec.elt(k).Vpt(1:2), [0 0], 'AbsTol', 1e-12);
            end
            t.build();
            b  = macos.draw_rays('YZ', 0, 4);    % YZ: V = Y
            m1 = (b.elt == 1);
            tc.assertTrue(any(m1(:)), 'no M1 crossings in the bundle');
            tc.verifyEqual(mean(b.V(m1)), dy, 'AbsTol', 0.01, ...
                'M1 footprint not centered on the aperture decenter');
        end

        function test_offaxis_tools_compose(tc)
            % field bias + aperture decenter compose in the emit: the chief
            % ray is tilted AND passes through the decentered stop.
            a = deg2rad(2/60);  dy = 0.02;
            t = tc.make_tma_();
            t.set_field_bias(2.0);  t.set_aperture_decenter(dy);
            f = [tempname '.in']; c = onCleanup(@() delete(f)); %#ok<NASGU>
            t.save(f);
            txt = fileread(f);
            tc.verifyEqual(tc.parse_vec3_(txt, 'ChfRayDir'), [0 sin(a) cos(a)], ...
                'AbsTol', 1e-12);
            tc.verifyEqual(tc.parse_vec3_(txt, 'ApStop'), [0 dy 0], 'AbsTol', 1e-12);
            % ChfRayPos = ApStop - stand*ChfRayDir : its y = dy - stand*sin(a),
            % so (ChfRayPos - ApStop) is anti-parallel to ChfRayDir.
            pos = tc.parse_vec3_(txt, 'ChfRayPos');
            v = pos - [0 dy 0];
            tc.verifyEqual(v./norm(v), -[0 sin(a) cos(a)], 'AbsTol', 1e-9, ...
                'chief ray does not pass through the decentered stop');
        end

        function test_offaxis_mirrors_emit_apnone(tc)
            % During the off-axis design phase the mirrors must not clip the
            % decentered/biased beam: Reflectors emit ApType=None.
            t = tc.make_tma_();  t.set_aperture_decenter(0.02);
            f = [tempname '.in']; c = onCleanup(@() delete(f)); %#ok<NASGU>
            t.save(f);
            txt = fileread(f);
            tc.verifyTrue(contains(txt, 'ApType=  None'), ...
                'off-axis mirrors should emit ApType=None');
            % on-axis design keeps Circular apertures
            t2 = tc.make_tma_();
            f2 = [tempname '.in']; c2 = onCleanup(@() delete(f2)); %#ok<NASGU>
            t2.save(f2);
            tc.verifyFalse(contains(fileread(f2), 'ApType=  None'), ...
                'on-axis design must keep Circular apertures');
        end

        function test_aperture_full_field_sizes_and_centers(tc)
            % aperture_full_field returns a per-element centered circle that
            % covers the traced footprints; with an aperture decenter the M1
            % aperture center tracks the decenter (XY plane: center(2)=Y).
            dy = 0.03;
            t = tc.make_tma_();  t.set_aperture_decenter(dy);  t.build();
            rep = t.aperture_full_field('quiet', true);
            tc.verifyEqual(numel(rep), 4, 'one entry per element');
            tc.verifyTrue(all([rep.radius] > 0), 'radii must be positive');
            tc.verifyTrue(all(isfinite([rep.radius])), 'radii must be finite');
            tc.verifyEqual(rep(1).center(2), dy, 'AbsTol', 0.02, ...
                'M1 aperture center not tracking the decenter');
        end

        function test_aperture_full_field_grows_with_fov(tc)
            % A wider field set covers more, so element apertures grow.
            t = tc.make_tma_();  t.set_field_bias(2.0);  t.build();
            r1 = t.aperture_full_field('fields', [0 deg2rad(2/60)], 'quiet', true);
            wide = [0            deg2rad(0/60);
                    0            deg2rad(4/60);
                    deg2rad( 2/60) deg2rad(2/60);
                    deg2rad(-2/60) deg2rad(2/60)];
            r2 = t.aperture_full_field('fields', wide, 'quiet', true);
            tc.verifyGreaterThanOrEqual(r2(end).radius, r1(end).radius, ...
                'FP aperture should not shrink with a wider field set');
        end

        function test_optimize_rigid_body_reemits_moved_design(tc)
            % Rigid-body DOFs (tilt + decenter + conic).  CALIB bakes the move
            % into psi/Vpt; the clean re-emit (moved psi/Vpt + conic surfaces)
            % reproduces the optimized WFE -- for rotationally-symmetric conics
            % the moved psi/Vpt fully define the surface (no TElt roll needed).
            t = tc.make_tma_();  t.set_field_bias(2.0);  t.build();
            seed = macos.trace(4).rmsWFE;                 % biased, un-optimized
            res = t.optimize('fields_arcmin',[1 2], 'max_iters',30, ...
                             'dofs',[1 1 0 1 1 0 0 1]);   % TIP TILT DX DY CONIC
            tc.verifyTrue(res.converged, 'rigid-body CALIB did not converge');
            % the re-emitted deliverable (now loaded) reproduces the optimized
            % WFE -> MACOS honored the moved psi/Vpt on re-load.  Compare with
            % an ABSOLUTE tol: both sit at the ~3e-9 diffraction-limited noise
            % floor (a lost tilt would leave WFE ~1e-6, caught by AbsTol).
            s = macos.trace(4);
            tc.verifyEqual(s.rmsWFE, res.wfe_after(1), 'AbsTol', 1e-8, ...
                're-emitted moved design does not reproduce the optimized WFE');
            tc.verifyLessThan(s.rmsWFE, 1e-8, ...
                'rigid+conic optimize not diffraction-limited');
            tc.verifyLessThan(s.rmsWFE, seed/10, ...
                'rigid+conic optimize did not substantially improve the seed');
            % the optimizer moved M2 off the pinned axis (recorded in spec)
            tc.verifyGreaterThan(norm(t.spec.elt(2).psi(:) - [0;0;-1]), 1e-7, ...
                'rigid-body DOFs did not move M2 (tilt expected)');
        end

        function test_emit_sensible_telt(tc)
            % The builder emits nECoord=6 + a sensible TElt: Z along the
            % outward surface normal (psi), X/Y tangent.  Trace-neutral, but
            % the interface frame for PERTURB/sensitivities.  On-axis M1
            % (psi=(0,0,-1)) -> the dmt6mono frame x=(-1,0,0) y=(0,1,0)
            % z=(0,0,-1); columns are emitted one per TElt line.
            t = tc.make_tma_();
            f = [tempname '.in']; c = onCleanup(@() delete(f)); %#ok<NASGU>
            t.save(f);
            txt = fileread(f);
            tc.verifyTrue(contains(txt, 'nECoord=  6'), 'expected nECoord=6');
            % first TElt line (column 1 = local x) for the first element is
            % (-1,0,0,0,0,0) for an on-axis mirror
            tok = regexp(txt, 'TElt=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens', 'once');
            col1 = [str2double(tok{1}) str2double(tok{2}) str2double(tok{3})];
            tc.verifyEqual(col1, [-1 0 0], 'AbsTol', 1e-12, ...
                'on-axis TElt column-1 (local x) not (-1,0,0)');
            % and the design still traces spherical-free (TElt is trace-neutral)
            t.build();
            tc.verifyLessThan(macos.trace(4).rmsWFE, 5e-7, ...
                'TElt emission must not change the ray trace');
        end

        function test_optimize_accepts_2d_field_cross(tc)
            % optimize() accepts an explicit 2-D (thx,thy) field set -- a CROSS
            % through the origin -- not just +y half-angles.  The seed Korsch is
            % rotationally symmetric, so the x-arm and y-arm at equal half-angle
            % aberrate equally; that equality proves the x field genuinely
            % reaches the trace (a collapse-to-+y bug would leave the x-arm at
            % the tiny on-axis WFE), and the optimizer balances the whole cross.
            a     = deg2rad(2.0/60);                    % 2 arcmin half-angle
            cross = [a 0; -a 0; 0 a; 0 -a];             % 4 off-axis arms (rad)
            t = tc.make_tma_();  t.build();
            res = t.optimize('fields', cross, 'max_iters',60);
            tc.verifyTrue(res.converged, 'CALIB did not converge on a 2-D field set');
            % result table is 2-D: (1+4) x 2, on-axis row 1 = [0 0]
            tc.verifySize(res.fields_xy_arcmin, [5 2]);
            tc.verifyEqual(res.fields_xy_arcmin(1,:), [0 0], 'AbsTol', 1e-9);
            tc.verifyEqual(max(abs(res.fields_xy_arcmin(:,1))), 2.0, 'AbsTol', 1e-6, ...
                'x-field component lost (collapsed to +y)');
            % x-arm (field 2) and y-arm (field 4) at the same half-angle must
            % aberrate equally -> the x field reached the trace, symmetrically.
            tc.verifyEqual(res.wfe_before(2), res.wfe_before(4), 'RelTol', 0.25, ...
                'x-arm WFE ~= y-arm WFE -> x field component did not reach the trace');
            tc.verifyGreaterThan(res.wfe_before(2), 5*res.wfe_before(1), ...
                'off-axis arm not aberrated vs on-axis -> field set inert');
            % and the conics balance the full 2-D field to diffraction-limited
            tc.verifyLessThan(max(res.wfe_after), 5e-8, ...
                sprintf('2-D field optimize not diffraction-limited: %.3e m', ...
                        max(res.wfe_after)));
        end

        function test_view_orthoviews_renders(tc)
            % view_layout's core is refactored into a shared draw-into-axes
            % helper (draw_plane_); view_orthoviews tiles it across planes for
            % a design-report figure.  Smoke: view_layout + both planes forms
            % (cellstr and token list) render to PNG (visible off) without
            % error -- guards the refactor end-to-end.
            t = tc.make_tma_();  t.build();
            cl = onCleanup(@() close('all','force'));  %#ok<NASGU>
            p1 = [tempname '.png'];  c1 = onCleanup(@() delete(p1)); %#ok<NASGU>
            t.view_layout('YZ', 'nrays',9, 'visible',false, 'save',p1);
            tc.verifyTrue(isfile(p1) && dir(p1).bytes > 0, ...
                'view_layout did not render a PNG');
            p2 = [tempname '.png'];  c2 = onCleanup(@() delete(p2)); %#ok<NASGU>
            t.view_orthoviews({'YZ','XZ'}, 'nrays',9, 'visible',false, 'save',p2);
            tc.verifyTrue(isfile(p2) && dir(p2).bytes > 0, ...
                'view_orthoviews (cellstr) did not render a PNG');
            p3 = [tempname '.png'];  c3 = onCleanup(@() delete(p3)); %#ok<NASGU>
            t.view_orthoviews('YZ XZ XY', 'nrays',9, 'visible',false, 'save',p3);
            tc.verifyTrue(isfile(p3) && dir(p3).bytes > 0, ...
                'view_orthoviews (token list) did not render a PNG');
            % view_field_map on a synthetic 3x3 grid scan (no real trace needed)
            [gx, gy] = meshgrid([-1 0 1], [-1 0 1]);
            sc = struct('fields',[gx(:) gy(:)], 'wfe', 0.01 + 0.02*(gx(:).^2+gy(:).^2));
            p4 = [tempname '.png'];  c4 = onCleanup(@() delete(p4)); %#ok<NASGU>
            t.view_field_map(sc, 'kind','contour', 'visible',false, 'save',p4);
            tc.verifyTrue(isfile(p4) && dir(p4).bytes > 0, ...
                'view_field_map did not render a PNG');
        end

        function test_field_set_builders(tc)
            % field_grid / field_cross produce the documented 2-D field sets
            % (the area + cross "modes"): right shape, span, origin handling,
            % and arcmin->rad units.
            fov = deg2rad(1.0/60);
            G = macos.design.field_grid(fov, 3);            % 3x3 incl center
            tc.verifySize(G, [9 2]);
            tc.verifyEqual(max(abs(G(:))), fov, 'AbsTol', 1e-15, 'grid must span +-fov');
            tc.verifyEqual(nnz(all(abs(G) < 1e-15, 2)), 1, 'grid must include (0,0)');
            G0 = macos.design.field_grid(fov, 3, 'origin', false);
            tc.verifySize(G0, [8 2]);                       % center dropped
            tc.verifyEqual(nnz(all(abs(G0) < 1e-15, 2)), 0, 'origin=false must drop center');
            Ga = macos.design.field_grid(1.0, 3, 'units','arcmin');   % arcmin -> rad
            tc.verifyEqual(max(abs(Ga(:))), fov, 'RelTol', 1e-12, 'arcmin units mismatch');
            C = macos.design.field_cross(fov, 3);           % 4 tips + shared center
            tc.verifySize(C, [5 2]);
            C0 = macos.design.field_cross(fov, 3, 'origin', false);
            tc.verifySize(C0, [4 2]);                       % the 4 arm tips
            tc.verifyEqual(sort(C0(:,1)).', [-fov 0 0 fov], 'AbsTol', 1e-15);
        end

        function test_field_ring_builder(tc)
            % field_ring produces the CIRCULAR-field set: n azimuths at the
            % given radius + the inner samples, no origin, arcmin->rad units.
            r = deg2rad(2.5/60);
            F = macos.design.field_ring(r);                 % 8 + 2 inner
            tc.verifySize(F, [10 2]);
            rad = hypot(F(:,1), F(:,2));
            tc.verifyEqual(max(rad), r, 'RelTol', 1e-12, 'ring must sit at radius');
            tc.verifyEqual(min(rad), 0.5*r, 'RelTol', 1e-12, 'inner ring at inner*radius');
            tc.verifyEqual(nnz(rad > 0.9*r), 8, 'default 8 outer azimuths');
            tc.verifyEqual(nnz(all(abs(F) < 1e-15, 2)), 0, 'ring must not include (0,0)');
            Fa = macos.design.field_ring(2.5, 'units','arcmin', 'inner',0);
            tc.verifySize(Fa, [8 2]);                       % inner=0 skips
            tc.verifyEqual(max(hypot(Fa(:,1),Fa(:,2))), r, 'RelTol', 1e-12, ...
                'arcmin units mismatch');
        end

        function test_trace_at_field_moves_field(tc)
            % trace_at_field re-emits + traces at a field OFFSET (the public
            % form of the realize_apertures per-field mechanism); [] restores
            % the nominal field.  An uncorrected off-axis field of a conic
            % telescope must show MORE aberration than on-axis, and the
            % restore must reproduce the nominal wavefront.
            t = tc.make_tma_();  t.build();
            nE = numel(t.spec.elt);
            macos.trace(nE);
            W0 = macos.opd();  v0 = W0(isfinite(W0) & W0 ~= 0);
            t.trace_at_field(deg2rad([0, 4.0/60]));         % 4' off-axis
            W1 = macos.opd();  v1 = W1(isfinite(W1) & W1 ~= 0);
            tc.verifyGreaterThan(std(v1), 2*std(v0), ...
                'off-axis field did not move (WFE unchanged)');
            t.trace_at_field([]);                           % restore nominal
            W2 = macos.opd();  v2 = W2(isfinite(W2) & W2 ~= 0);
            tc.verifyEqual(std(v2), std(v0), 'RelTol', 1e-9, ...
                'restore did not reproduce the nominal field');
        end

        function test_optimize_area_grid_dedup(tc)
            % optimize() accepts a full 2-D AREA grid INCLUDING the (0,0)
            % center; the on-axis row is dropped (it is the implicit field 1)
            % so n_fov is NOT inflated, and the area is balanced.
            fov = deg2rad(1.0/60);
            G = macos.design.field_grid(fov, 3);            % 9 pts incl center
            t = tc.make_tma_();  t.build();
            res = t.optimize('fields', G, 'max_iters', 60);
            tc.verifyTrue(res.converged, 'area-grid CALIB did not converge');
            % 9 grid points, center de-duped vs the implicit on-axis field
            % -> 1 + 8 = 9 FoV (NOT 1 + 9 = 10)
            tc.verifyEqual(res.n_fov, 9, 'on-axis grid point not de-duped');
            tc.verifySize(res.fields_xy_arcmin, [9 2]);
            tc.verifyLessThan(max(res.wfe_after), 5e-8, ...
                sprintf('area optimize not diffraction-limited: %.3e m', max(res.wfe_after)));
        end

        function test_add_mirror_convex_secondary(tc)
            % A convex secondary uses the SAME MACOS convention as any mirror:
            % KrElt = -|R| (negative).  It is convex by GEOMETRY -- it sits
            % before the M1 focus (Cassegrain spacing, t1 < f1), so the beam
            % reflects away from its centre of curvature (j18mono's convex SM).
            % A POSITIVE radius is the normal case (convex by geometry); a
            % NEGATIVE n-flip radius is also accepted and still emits KrElt<0
            % (see the signed-radius assertion below).  This is the validated f/8
            % Korsch [8 2 4] / [3 4.5]: M2 (R=2) sits before the M1 focus (f1=4),
            % so M2 is the convex secondary.
            t = macos.design.Telescope('family','TMA', 'aperture_diameter_m',1.0, ...
                'model_size',tc.ModelSize, 'grid_npts',tc.GridNpts);
            t.add_mirror('M1','radius_m',8.0, 'spacing_after_m',3.0);
            t.add_mirror('M2','radius_m',2.0, 'spacing_after_m',4.5);   % convex (t1<f1)
            t.add_mirror('M3','radius_m',4.0, 'spacing_after','derive');
            t.build();
            % KrElt = -|R| for EVERY mirror, convex secondary included.
            tc.verifyEqual(t.spec.elt(1).Kr, -8.0, 'AbsTol',1e-9, ...
                'concave primary must be KrElt = -|R|');
            tc.verifyEqual(t.spec.elt(2).Kr, -2.0, 'AbsTol',1e-9, ...
                'CONVEX secondary must STILL be KrElt = -|R| (convention, not a sign flip)');
            % convexity is the geometry: M2 sits before the M1 focus (t1 < f1).
            f1 = abs(t.spec.elt(1).Kr)/2;
            tc.verifyLessThan(t.spec.derived.t(1), f1, ...
                'convex secondary must sit before the M1 focus (Cassegrain spacing)');
            % and it actually TRACES (the old sign-only test never did).
            s = macos.trace(numel(t.spec.elt));
            tc.verifyGreaterThan(s.nRays, 0, 'convex-secondary TMA failed to trace');
            tc.verifyTrue(isfinite(s.rmsWFE), 'convex-secondary TMA WFE not finite');
            % a NEGATIVE n-flip radius is ACCEPTED (a slowing relay tertiary past
            % an intermediate focus carries one) and still emits KrElt=-|R| < 0 --
            % the sign drives only the Seidel conic/focus math, never KrElt.
            t2 = macos.design.Telescope('family','TMA', 'aperture_diameter_m',1.0, ...
                'model_size',tc.ModelSize, 'grid_npts',tc.GridNpts);
            t2.add_mirror('M1','radius_m', 8.0, 'spacing_after_m',3.0);
            t2.add_mirror('M2','radius_m',-2.0, 'spacing_after_m',4.5);   % signed (n-flip)
            t2.add_mirror('M3','radius_m', 4.0, 'spacing_after','derive');
            t2.build('', 'validate', false);                             % geometry/sign only
            tc.verifyLessThan(t2.spec.elt(2).Kr, 0, ...
                'signed n-flip radius must still emit KrElt = -|R| < 0');
            % a ZERO radius is still rejected.
            tc.verifyError(@() t2.add_mirror('Mx','radius_m',0.0, 'spacing_after','derive'), ...
                'macos:design:Telescope:sign');
        end

        function test_rc_unobscured_decenter_shrinks_with_mag(tc)
            % The unobscured off-axis RC (design/rc_unobscured) decenter is set
            % by the secondary's SHADOW, so a smaller secondary -- higher
            % magnification, i.e. a faster primary + slower system f/# -- needs
            % LESS off-axis decenter (geometric floor ~ D/2).
            t1 = macos.design.Telescope('family','RC', 'aperture_diameter_m',1.0, ...
                'primary_fnum',2.0, 'system_fnum',10.0, 'BFD_m',0.30, ...
                'wavelength_m',633e-9, 'model_size',tc.ModelSize);
            t1.build();  d_lo = t1.set_offaxis('all');     % m = 5
            t2 = macos.design.Telescope('family','RC', 'aperture_diameter_m',1.0, ...
                'primary_fnum',1.5, 'system_fnum',20.0, 'BFD_m',0.30, ...
                'wavelength_m',633e-9, 'model_size',tc.ModelSize);
            t2.build();  d_hi = t2.set_offaxis('all');     % m = 13.3 (smaller secondary)
            tc.verifyLessThan(d_hi, d_lo, ...
                'higher-mag RC should need less off-axis decenter');
            tc.verifyGreaterThan(d_hi, 0.5, 'decenter cannot beat the D/2 floor');
        end

        function test_rc_onaxis_obscured_aplanat(tc)
            % The on-axis RC (design/rc_onaxis) is an obscured aplanat:
            % diffraction-limited on-axis, with a real central obscuration the
            % secondary casts (~0.19 for f/2 + f/10).
            t = macos.design.Telescope('family','RC', 'aperture_diameter_m',1.0, ...
                'primary_fnum',2.0, 'system_fnum',10.0, 'BFD_m',0.30, ...
                'wavelength_m',633e-9, 'model_size',tc.ModelSize);
            t.build();
            s = macos.trace(numel(t.spec.elt));
            tc.verifyLessThan(s.rmsWFE, 5e-8, ...
                'on-axis RC not diffraction-limited (should be an aplanat)');
            rep = t.check_clipping('noload', true, 'quiet', true);
            iM2 = find(strcmp({rep.name}, 'M2'), 1);
            eps_o = 2 * rep(iM2).foot_r;          % obscuration ratio (D = 1)
            tc.verifyGreaterThan(eps_o, 0.05, 'expected a real central obscuration');
            tc.verifyLessThan(eps_o, 0.5, 'central obscuration unreasonably large');
        end
    end
end
