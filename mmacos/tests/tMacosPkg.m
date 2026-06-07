classdef tMacosPkg < matlab.unittest.TestCase
%TMACOSPKG  Regression tests for the +macos/ function package.

    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_Cass_FarField.in'
        ExpectedNelt = 6
    end

    properties
        rx_path
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            testCase.rx_path = rx_fixture_path(testCase.RxName);
            macos.init(testCase.ModelSize);
        end
    end

    methods (TestMethodSetup)
        function setupMethod(testCase)
            macos.load_rx(testCase.rx_path);
        end
    end

    methods (Test)
        function test_has_rx_after_load(testCase)
            testCase.verifyTrue(macos.has_rx());
        end

        function test_num_elt_matches(testCase)
            testCase.verifyEqual(macos.num_elt(), testCase.ExpectedNelt);
        end

        function test_cbm_positive(testCase)
            testCase.verifyGreaterThan(macos.cbm(), 0);
        end

        function test_sys_units_is_struct(testCase)
            s = macos.sys_units();
            testCase.verifyTrue(isstruct(s));
            testCase.verifyTrue(isfield(s, 'base_unit_id'));
            testCase.verifyTrue(isfield(s, 'wave_unit_id'));
        end

        % --- Source --------------------------------------------------
        function test_src_sampling_roundtrip(testCase)
            macos.set_src_sampling(64);
            testCase.verifyEqual(macos.get_src_sampling(), 64);
            macos.set_src_sampling(96);
            testCase.verifyEqual(macos.get_src_sampling(), 96);
        end

        function test_src_wvl_roundtrip(testCase)
            wvl0 = macos.get_src_wvl();
            macos.set_src_wvl(wvl0 * 2);
            testCase.verifyEqual(macos.get_src_wvl(), wvl0 * 2, 'AbsTol', eps(wvl0*2));
            macos.set_src_wvl(wvl0);
        end

        % --- Element geometry ---------------------------------------
        function test_elt_vpt_returns_column_3vec(testCase)
            v = macos.get_elt_vpt(2);
            testCase.verifyEqual(size(v), [3 1]);
        end

        function test_elt_vpt_setter_roundtrip(testCase)
            v0 = macos.get_elt_vpt(2);
            macos.set_elt_vpt(2, v0 + [1e-9; 0; 0]);
            v1 = macos.get_elt_vpt(2);
            testCase.verifyEqual(v1(1) - v0(1), 1e-9, 'AbsTol', 1e-18);
            macos.set_elt_vpt(2, v0);
        end

        function test_elt_psi_returns_column_3vec(testCase)
            p = macos.get_elt_psi(2);
            testCase.verifyEqual(size(p), [3 1]);
        end

        function test_elt_rpt_returns_column_3vec(testCase)
            r = macos.get_elt_rpt(2);
            testCase.verifyEqual(size(r), [3 1]);
        end

        % --- Perturbations ------------------------------------------
        function test_perturb_single_returns_clean(testCase)
            macos.perturb(2, 'rotation', [1e-6; 0; 0], 'frame', 'global');
            macos.modify();
            s = macos.trace();
            testCase.verifyGreaterThan(s.nRays, 0);
        end

        function test_perturb_many_array_form(testCase)
            % Apply identical perturbations to elements 2 and 3.
            srfs = [2; 3];
            prb  = repmat([1e-6;0;0;0;0;0], 1, 2);
            macos.perturb_many(srfs, prb, [1; 1]);
            macos.modify();
            s = macos.trace();
            testCase.verifyGreaterThan(s.nRays, 0);
        end

        function test_perturb_src_runs(testCase)
            macos.perturb_src('rotation', [1e-6; 0; 0]);
            macos.modify();
            s = macos.trace();
            testCase.verifyGreaterThan(s.nRays, 0);
        end

        % --- Trace + diffraction buffers ----------------------------
        function test_trace_returns_struct(testCase)
            s = macos.trace();
            testCase.verifyTrue(isstruct(s));
            testCase.verifyTrue(isfield(s, 'nRays'));
            testCase.verifyTrue(isfield(s, 'rmsWFE'));
            testCase.verifyGreaterThan(s.nRays, 0);
        end

        function test_opd_is_square(testCase)
            macos.trace();
            W = macos.opd();
            testCase.verifyEqual(size(W,1), size(W,2));
        end

        function test_intensity_peak_positive(testCase)
            I = macos.intensity(testCase.ExpectedNelt);
            testCase.verifyGreaterThan(max(I(:)), 0);
        end

        % --- Ray status ---------------------------------------------
        function test_get_ray_status_returns_struct(testCase)
            s = macos.trace();
            r = macos.get_ray_status(s.nRays);
            testCase.verifyTrue(isstruct(r));
            for fn = {'status','fail_elt','n_ok','n_obscured','n_miss', ...
                      'n_bracket','n_max_iter','n_undef','n_lost'}
                testCase.verifyTrue(isfield(r, fn{1}), ...
                    sprintf('missing field %s', fn{1}));
            end
            testCase.verifyEqual(numel(r.status),  s.nRays);
            testCase.verifyEqual(numel(r.fail_elt), s.nRays);
        end

        function test_get_ray_status_cass_obscuration(testCase)
            % Rx_Cass_FarField has an M2 shadow that blocks the central
            % rays.  Expect Obscured > 0, but every ray must trace
            % cleanly (no Miss/Bracket/MaxIter/Undef) and the per-
            % category counts must add up to nRays.
            s = macos.trace();
            r = macos.get_ray_status(s.nRays);
            testCase.verifyGreaterThan(r.n_ok,       0);
            testCase.verifyGreaterThan(r.n_obscured, 0);
            testCase.verifyEqual(r.n_miss,     0);
            testCase.verifyEqual(r.n_bracket,  0);
            testCase.verifyEqual(r.n_max_iter, 0);
            testCase.verifyEqual(r.n_undef,    0);
            n_categorised = r.n_ok + r.n_obscured + r.n_miss + ...
                            r.n_bracket + r.n_max_iter + r.n_undef;
            testCase.verifyEqual(n_categorised, s.nRays);
            testCase.verifyEqual(r.n_lost, s.nRays - r.n_ok);
        end

        function test_get_ray_status_matches_ray_info(testCase)
            % get_ray_status (per-category) and get_ray_info (binary
            % ok_trace / ok_pass) must agree: status == 0 iff ok_pass,
            % and ok_trace must be true everywhere (no intersection
            % failures on Rx_Cass_FarField).
            s  = macos.trace();
            rs = macos.get_ray_status(s.nRays);
            ri = macos.get_ray_info(s.nRays);
            testCase.verifyTrue(all(ri.ok_trace));
            testCase.verifyEqual(rs.n_ok, nnz(ri.ok_pass));
            testCase.verifyTrue(all((rs.status == 0) == ri.ok_pass));
        end

        function test_complex_field_is_complex(testCase)
            cf = macos.complex_field(testCase.ExpectedNelt);
            testCase.verifyFalse(isreal(cf));
        end

        function test_intensity_matches_complex_mod_squared(testCase)
            % |cfield|^2 should equal intensity to numerical precision.
            cf = macos.complex_field(testCase.ExpectedNelt, ...
                'reset_trace', true);
            I  = macos.intensity(testCase.ExpectedNelt, ...
                'reset_trace', false);
            % Compare normalized to peak to avoid 1e+6 absolute scale.
            cf_int = abs(cf).^2;
            rel = max(abs(I(:) - cf_int(:))) / max(I(:));
            testCase.verifyLessThan(rel, 1e-10);
        end

        function test_dx_at_unit_conversion(testCase)
            dx_m  = macos.dx_at(testCase.ExpectedNelt);
            dx_mm = macos.dx_at(testCase.ExpectedNelt, 'mm');
            dx_um = macos.dx_at(testCase.ExpectedNelt, 'um');
            testCase.verifyEqual(dx_mm, dx_m * 1e3, 'AbsTol', 1e-18);
            testCase.verifyEqual(dx_um, dx_m * 1e6, 'AbsTol', 1e-12);
        end
    end
end
