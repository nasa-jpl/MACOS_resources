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
    end
end
