classdef tDwDsurf < matlab.unittest.TestCase
%TDWDSURF  Regression tests for macos.dw_dsurf (dw/dKr + dw/dKc).
%   Powered optics = Reflector/Refractor/NSReflector/NSRefractor/Segment
%   with |Kr| << 1e22, ENGINE-queried (BRIEF_luis_round3: the old Rx-text
%   whitelist silently dropped NSReflector).  Rx_Cass_FarField has exactly
%   two (M1=Elt 2, M2=Elt 3); the Obscuring stop (Elt 1), the flats
%   (Elt 4/6), and the POWERED Return / exit-pupil sphere (Elt 5) are all
%   correctly excluded -- by TYPE, not by Kr.  The NS twin fixtures
%   (Rx_Cass_NS.in / Rx_Cass_NSseq.in, one Element= token apart) gate the
%   NSReflector path and the loud explicit-'elts' contract.

    properties (Constant)
        ModelSize = 256
        RxName    = 'Rx_Cass_FarField.in'
        RxNS      = 'Rx_Cass_NS.in'
        RxNSseq   = 'Rx_Cass_NSseq.in'
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

        function test_ns_reflector_discovered(testCase)
            % THE Luis 2026-09-05 bug: an NSReflector primary must be in the
            % powered set.  Pre-fix (Rx-text whitelist) this returns [3].
            m = macos.Session(testCase.ModelSize);
            rx = rx_fixture_path(testCase.RxNS);
            m.load_rx(rx);
            pe = macos.find_powered_elts(m, rx);
            testCase.verifyEqual(pe(:).', [2 3], ...
                'NSReflector M1 + Reflector M2 must both be powered');
        end

        function test_ns_explicit_elts_served(testCase)
            % 'elts' naming the NSReflector yields its Kr/Kc channels with
            % non-vacuous columns.  Pre-fix this errored 'nochan' after the
            % silent intersect emptied the set.
            m = macos.Session(testCase.ModelSize);
            rx = rx_fixture_path(testCase.RxNS);
            out = macos.dw_dsurf(m, rx, 'elts', 2);
            testCase.verifyEqual(out.iElt(:).', [2 2]);
            testCase.verifyEqual(out.param(:).', {'Kr','Kc'});
            for k = 1:2
                testCase.verifyGreaterThan(sqrt(mean(out.dwds(:,k).^2)), 0, ...
                    sprintf('%s column identically zero', out.channel_names{k}));
            end
        end

        function test_ns_matches_sequential_twin(testCase)
            % The twins differ by ONE Element= token; the NS dispatch traces
            % the same geometry, so the Jacobians must agree to round-off --
            % up to a PER-COLUMN PISTON: the NS deck's chief ray is
            % geometrically dead (fixture header), so each trace's DAvgl
            % piston reference averages one fewer ray (~1e-12 m/trace),
            % which the FD divides by delta into ~3e-6-relative uniform
            % column offsets.  Piston-removed, measured agreement is
            % 6e-17 relative (2026-09-05).
            m = macos.Session(testCase.ModelSize);
            o_ns  = macos.dw_dsurf(m, rx_fixture_path(testCase.RxNS));
            o_seq = macos.dw_dsurf(m, rx_fixture_path(testCase.RxNSseq));
            testCase.verifyEqual(size(o_ns.dwds), size(o_seq.dwds), ...
                'twin Jacobians must share support (annular launch)');
            scale = max(abs(o_seq.dwds(:)));
            for k = 1:size(o_ns.dwds, 2)
                d = o_ns.dwds(:,k) - o_seq.dwds(:,k);
                d = d - mean(d);
                testCase.verifyLessThan(max(abs(d)) / scale, 1e-12, ...
                    sprintf('%s: NS vs sequential twin mismatch beyond piston', ...
                            o_ns.channel_names{k}));
            end
        end

        function test_explicit_elts_error_named(testCase)
            % An explicitly requested, unserveable elt ERRORS with a named
            % reason -- never a silent drop (Dave's ruling, BRIEF_luis_round3).
            % Elt 5 is the POWERED exit-pupil Return: type-excluded.
            m = macos.Session(testCase.ModelSize);
            testCase.verifyError( ...
                @() macos.dw_dsurf(m, testCase.rx_path, 'elts', 5), ...
                'macos:channels:eltNotEligible');
            % out-of-range id: same contract
            testCase.verifyError( ...
                @() macos.dw_dsurf(m, testCase.rx_path, 'elts', 99), ...
                'macos:channels:eltNotEligible');
        end
    end
end
