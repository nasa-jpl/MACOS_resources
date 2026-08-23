classdef tStopReload < matlab.unittest.TestCase
%TSTOPRELOAD  An api element stop must not outlive load_rx.
%
%   Regression for the stop-state leak (found by TO 2026-08-22 in the
%   dw/dx-groups arc, fixed in the engine the same day):
%   stop_info_set (Session.stop) sets EltStopSet/StopElt in SAVEd
%   engine module state, and the LOAD reset in macos_cmd_loop.inc
%   cleared only RxStopSet/ifStopSet.  On the NEXT load_rx the stale
%   EltStopSet tripped ChkDf2's NS/Segment stop guard (iosub.inc)
%   mid-parse -- the guard clears BOTH ifStopSet and EltStopSet -- so
%   the new deck's own header ApStop= never took: the load printed
%   "*** Setting aperture stop failed!" and the first pupil query
%   died.  Pre-fix signature, measured: macos.fex(1) errors
%   macos:fex:noStop.  The guard only fires on a Segment/NS/typeless
%   element, so the second deck must carry Segments -- e5hex1 (7 hex
%   Segments + header ApStop= 0 0 0) is exactly the trap; an
%   object-space stop (Session.stop_obj) never leaked.  The fix also
%   scoped the ChkDf2 guard to iElt==StopElt (unscoped, ANY Segment
%   anywhere cancelled a valid element stop declared in a deck).

    properties (Constant)
        ModelSize = 128
        RxName    = 'e5hex1.in'
        StopEltId = 8   % the Reflector -- a legal element-stop site
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
        function test_element_stop_does_not_outlive_load(testCase)
            m = macos.Session(testCase.ModelSize);
            m.load_rx(testCase.rx_path);

            % Cold sanity: the deck's own object-space ApStop serves fex.
            s0 = macos.fex(1);
            testCase.verifyTrue(isfinite(s0.rad) && s0.rad ~= 0, ...
                'cold load: fex should work off the deck''s own ApStop');

            % Arm the leak: an api ELEMENT stop, then a fresh load.
            m.stop(int32(testCase.StopEltId));
            m.modify();

            m.load_rx(testCase.rx_path);
            % Pre-fix this errored macos:fex:noStop -- the leaked
            % EltStopSet made ChkDf2 cancel the reloaded deck's ApStop.
            s1 = macos.fex(1);
            testCase.verifyTrue(isfinite(s1.rad) && s1.rad ~= 0, ...
                'reload after element stop: the deck ApStop must survive');
            testCase.verifyEqual(s1.rad, s0.rad, 'AbsTol', 1e-9, ...
                'reloaded deck must reproduce the cold-load pupil');
        end
    end
end
