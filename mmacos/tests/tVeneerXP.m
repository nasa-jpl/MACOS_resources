classdef tVeneerXP < matlab.unittest.TestCase
%TVENEERXP  spot / fex / get_xp / set_xp veneers on a STOP-bearing Rx.
%   These commands need a chief ray / exit pupil, which requires a STOP.
%   e5hex1 declares ApStop, so they compute; Rx_Cass_FarField is stopless
%   (there they fail gracefully — the engine GO-TO-reprompt infinite loop
%   was fixed in macos commit "SPOT: graceful abort on local-coord
%   failure").

    properties (Constant)
        ModelSize = 128
        RxName    = 'e5hex1.in'
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
        function loadAndTrace(testCase)
            macos.load_rx(testCase.rx_path);
            macos.trace();
        end
    end

    methods (Test)
        function test_spot_returns_struct(testCase)
            s = macos.spot(macos.num_elt());
            testCase.verifyTrue(isstruct(s));
            for fn = {'pts','centroid','shift','csys','nSpot'}
                testCase.verifyTrue(isfield(s, fn{1}), ...
                    sprintf('spot missing field %s', fn{1}));
            end
            testCase.verifyGreaterThan(s.nSpot, 0);
            testCase.verifyEqual(size(s.pts), [s.nSpot 2]);
            testCase.verifyEqual(numel(s.centroid), 2);
            testCase.verifyEqual(size(s.csys), [3 3]);
        end

        function test_spot_ref_options(testCase)
            % 'ref' and 'at' name-value options accepted + run clean.
            s = macos.spot(macos.num_elt(), 'ref', 'telt', 'at', 'elt');
            testCase.verifyGreaterThan(s.nSpot, 0);
        end

        function test_fex_returns_geometry(testCase)
            s = macos.fex(1);
            for fn = {'vpt','psi','rad','xp'}
                testCase.verifyTrue(isfield(s, fn{1}), ...
                    sprintf('fex missing field %s', fn{1}));
            end
            testCase.verifyEqual(size(s.vpt), [3 1]);
            testCase.verifyEqual(size(s.psi), [3 1]);
            testCase.verifyEqual(numel(s.xp), 7);
        end

        function test_xp_get_set_roundtrip(testCase)
            macos.fex(1);
            x0 = macos.get_xp();
            testCase.verifyEqual(size(x0.vpt), [3 1]);
            macos.set_xp(x0.vpt + [1e-6; 0; 0], x0.psi, x0.rad);
            x1 = macos.get_xp();
            testCase.verifyEqual(x1.vpt(1) - x0.vpt(1), 1e-6, 'AbsTol', 1e-12);
            macos.set_xp(x0.vpt, x0.psi, x0.rad);   % restore
        end

        function test_spot_stopless_fails_gracefully(testCase)
            % Regression for the infinite-loop fix: spot at a FocalPlane
            % on a stopless Rx must FAIL FAST (not hang).
            macos.load_rx(rx_fixture_path('Rx_Cass_FarField.in'));
            macos.trace();
            % spot_cmd fails -> mmacos raises (any MException); the point
            % is it RETURNS rather than hanging.
            testCase.verifyError(@() macos.spot(macos.num_elt()), ?MException);
        end
    end
end
