classdef tDwDsurf < matlab.unittest.TestCase
%TDWDSURF  Regression tests for macos.dw_dsurf (dw/dKr + dw/dKc).
%   Powered optics = Reflector/Refractor with |Kr| << 1e22.  Rx_Cass_FarField
%   has exactly two (M1=Elt 2, M2=Elt 3); the Obscuring stop (Elt 1), the
%   flats (Elt 4/6), and the POWERED Return / exit-pupil sphere (Elt 5) are
%   all correctly excluded.

    properties (Constant)
        ModelSize = 256
        RxName    = 'Rx_Cass_FarField.in'
        ExpectedPowered = 2
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

    methods (Test)
        function test_find_powered_only_reflectors(testCase)
            m = macos.Session(testCase.ModelSize);
            m.load_rx(testCase.rx_path);
            pe = macos.find_powered_elts(m, testCase.rx_path);
            testCase.verifyEqual(numel(pe), testCase.ExpectedPowered, ...
                'powered set should be the 2 conic Reflectors (M1, M2)');
            % powered RETURN (Elt 5) + Obscuring (Elt 1) + flats are excluded
            testCase.verifyEqual(pe(:).', [2 3], 'expected powered elts [2 3]');
        end

        function test_dwds_shape_and_nonzero(testCase)
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dsurf(m, testCase.rx_path);
            n_chan = 2 * testCase.ExpectedPowered;     % Kr + Kc per optic
            testCase.verifyEqual(numel(out.channel_names), n_chan, ...
                'channel count should be 2 (Kr,Kc) x n_powered');
            testCase.verifyEqual(size(out.dwds, 2), n_chan);
            testCase.verifyEqual(size(out.dwds, 1), numel(out.w_nom_vec));
            testCase.verifyGreaterThan(max(abs(out.dwds(:))), 0, ...
                '|dwds| max is zero -- the FD sweep did not perturb Kr/Kc');
            % element-major, param-minor order: Kr then Kc per optic
            testCase.verifyEqual(out.param(:).', {'Kr','Kc','Kr','Kc'});
            testCase.verifyEqual(out.iElt(:).', [2 2 3 3]);
        end

        function test_each_column_nonzero(testCase)
            % Every powered Kr AND Kc DOF moves the exit-pupil wavefront.
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dsurf(m, testCase.rx_path);
            for k = 1:size(out.dwds, 2)
                testCase.verifyGreaterThan(sqrt(mean(out.dwds(:,k).^2)), 0, ...
                    sprintf('%s column is identically zero', out.channel_names{k}));
            end
        end

        function test_params_subset(testCase)
            % params={'Kr'} yields one column per powered optic.
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dsurf(m, testCase.rx_path, 'params', {'Kr'});
            testCase.verifyEqual(numel(out.channel_names), testCase.ExpectedPowered);
            testCase.verifyEqual(out.param(:).', {'Kr','Kr'});
        end
    end
end
