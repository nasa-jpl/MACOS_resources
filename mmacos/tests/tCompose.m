classdef tCompose < matlab.unittest.TestCase
%TCOMPOSE  macos.compose() -- multi-wavelength PSF on a fixed pixel grid.
%   Mirrors pymacos tests/test_compose.py.  COMPOSE/ADD resample each
%   wavelength's PSF onto the SAME fixed grid and sum incoherently, so
%   the composite is exactly linear in the wavelength list -- the strong
%   regression invariant pinned here:
%       compose([a b]) == compose([a]) + compose([b])     (machine eps)
%       compose([a a]) == 2 * compose([a])

    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_Cass_FarField.in'
        Det       = 6        % FocalPlane
        Npix      = 64
    end

    properties
        w0
        dx
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            macos.init(testCase.ModelSize);
        end
    end

    methods (TestMethodSetup)
        function loadAndProbe(testCase)
            macos.load_rx(rx_fixture_path(testCase.RxName));
            macos.intensity(testCase.Det);            % establish dx at Det
            testCase.w0 = macos.get_src_wvl();
            testCase.dx = macos.dx_at(testCase.Det);  % SI metres
        end
    end

    methods (Test)
        function test_shape_and_positive(testCase)
            I = macos.compose(testCase.Det, testCase.w0, testCase.Npix, testCase.dx);
            testCase.verifyEqual(size(I), [testCase.Npix testCase.Npix]);
            testCase.verifyTrue(all(isfinite(I(:))));
            testCase.verifyGreaterThanOrEqual(min(I(:)), 0);
            testCase.verifyGreaterThan(max(I(:)), 0);
        end

        function test_linear_in_wavelength_list(testCase)
            w0 = testCase.w0; dx = testCase.dx; n = testCase.Npix; d = testCase.Det;
            I1  = macos.compose(d, w0,            n, dx);
            I2  = macos.compose(d, w0*1.02,       n, dx);
            I12 = macos.compose(d, [w0 w0*1.02],  n, dx);
            rel = max(abs(I12(:) - (I1(:)+I2(:)))) / max(max(I12(:)), 1e-300);
            testCase.verifyLessThan(rel, 1e-12);
        end

        function test_repeated_wavelength_scales(testCase)
            w0 = testCase.w0; dx = testCase.dx; n = testCase.Npix; d = testCase.Det;
            I1  = macos.compose(d, w0,       n, dx);
            Idd = macos.compose(d, [w0 w0],  n, dx);
            rel = max(abs(Idd(:) - 2*I1(:))) / max(max(Idd(:)), 1e-300);
            testCase.verifyLessThan(rel, 1e-12);
        end

        function test_dx_units_consistent(testCase)
            w0 = testCase.w0; dx = testCase.dx; n = testCase.Npix; d = testCase.Det;
            Im  = macos.compose(d, w0, n, dx,     'dx_unit', 'm');
            Imm = macos.compose(d, w0, n, dx*1e3, 'dx_unit', 'mm');
            testCase.verifyEqual(Imm, Im, 'AbsTol', 1e-9*max(max(Im(:)),1.0));
        end
    end
end
