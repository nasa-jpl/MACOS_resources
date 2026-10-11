classdef tSessionModes < matlab.unittest.TestCase
%TSESSIONMODES  Session modes survive load_rx, and load_rx says so (Dave 2026-10-11,
%   PLAN_CONSOLIDATION 4 item 2; CC's extension to the bindings on the one-semantics rule).
%   POLARIZATION / VECTOR / CENTROID / LNEG / CIR are SESSION modes set at the engine's
%   allocation, no longer reset by a load (before 2026-10-11 every load reset them --
%   the CLI's CENTROID before an `old` was lost silently).  The CLI's LOAD prints the
%   non-default ones; the bindings' load_rx raises warning 'mmacos:modesInForce' with
%   the same list (macos.modes, from the engine's ModesGet -- one source).
%   Pre-change mex: no modes_get, and a load reset polarization -- red.
    properties (Constant)
        Rx = 'Rx_AsphCalib.in'
    end
    methods (Test)
        function test_a_default_load_is_silent(tc)
            macos.init(128);
            tc.verifyWarningFree(@() macos.load_rx(rx_fixture_path(tc.Rx)));
            md = macos.modes();
            tc.verifyEmpty(md.in_force);
            tc.verifyFalse(md.pol);  tc.verifyTrue(md.lneg);
        end
        function test_polarization_survives_load_rx_and_is_named(tc)
            macos.init(128);  macos.load_rx(rx_fixture_path(tc.Rx));
            macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
            c = onCleanup(@() macos.polarization('off')); %#ok<NASGU>
            tc.verifyWarning(@() macos.load_rx(rx_fixture_path(tc.Rx)), 'mmacos:modesInForce');
            md = macos.modes();
            tc.verifyTrue(md.pol, 'polarization survives the load');
            tc.verifyEqual(md.in_force{1}, 'POLARIZATION');
            macos.polarization('off');
            tc.verifyWarningFree(@() macos.load_rx(rx_fixture_path(tc.Rx)), ...
                'back at the defaults, the load is silent again');
        end
    end
end
