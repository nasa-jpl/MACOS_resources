classdef tSysProp < matlab.unittest.TestCase
%TSYSPROP  macos.first_order_properties() (the SYSPROP command).
%   Validates the first-order struct + the EFL/lambda-D consistency
%   relations, and that the pixel-based quantities gate on a prior INT.

    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_Cass_FarField.in'
        Det       = 6        % FocalPlane
        ARCSEC_PER_RAD = 206264.806247096
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
        function loadRx(testCase)
            macos.load_rx(testCase.rx_path);
        end
    end

    methods (Test)
        function test_struct_has_fields(testCase)
            p = macos.first_order_properties();
            for fn = {'efl_baseunits','fno','dpup_m','obscuration', ...
                      'lambda_m','lamD_rad','lamD_arcsec','lamD_px', ...
                      'plate_arcsec_px','plate_px_rad', ...
                      'nyquist_baseunits','dx_focal_baseunits'}
                testCase.verifyTrue(isfield(p, fn{1}), ...
                    sprintf('missing field %s', fn{1}));
            end
        end

        function test_angular_relations(testCase)
            p = macos.first_order_properties(testCase.Det);
            testCase.verifyGreaterThan(p.fno, 0);
            testCase.verifyGreaterThan(p.dpup_m, 0);
            % lambda/D (rad) = lambda / D_EP
            testCase.verifyEqual(p.lamD_rad, p.lambda_m / p.dpup_m, ...
                'RelTol', 1e-9);
            % arcsec = rad * 206265
            testCase.verifyEqual(p.lamD_arcsec, ...
                p.lamD_rad * testCase.ARCSEC_PER_RAD, 'RelTol', 1e-6);
        end

        function test_px_gated_on_INT(testCase)
            % Pre-INT: pixel-based quantities are 0 (no propagation).
            p0 = macos.first_order_properties(testCase.Det);
            testCase.verifyEqual(p0.lamD_px, 0);
            testCase.verifyEqual(p0.plate_px_rad, 0);
            testCase.verifyEqual(p0.dx_focal_baseunits, 0);
            % Post-INT: they are set and self-consistent.
            macos.intensity(testCase.Det);
            p1 = macos.first_order_properties(testCase.Det);
            testCase.verifyGreaterThan(p1.lamD_px, 0);
            testCase.verifyGreaterThan(p1.plate_px_rad, 0);
            % lamD_px = lamD_rad * plate_px_rad
            testCase.verifyEqual(p1.lamD_px, ...
                p1.lamD_rad * p1.plate_px_rad, 'RelTol', 1e-6);
        end
    end
end
