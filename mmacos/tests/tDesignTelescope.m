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
    end
end
