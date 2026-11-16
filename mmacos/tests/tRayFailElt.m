classdef tRayFailElt < matlab.unittest.TestCase
%TRAYFAILELT  An obscured ray's RayFailElt is the element that CLIPPED it.
%
%   PLAN_CONSOLIDATION 1.2 (macos, 2026-10-12).  CTRACE stamped obscuration
%   only after its element loop, with the loop's exit value iEndElt+1: on
%   Rx_Cass_FarField (the secondary's central obscuration is element 1) the
%   obscured rays read fail_elt 3 / 5 / 7 after trace(2) / trace(4) /
%   trace(6) -- the element AFTER the end of the trace -- while CPROPAGATE,
%   which walks the train element by element, already recorded 1.  CTRACE
%   now stamps at the clip, so the two paths agree.  The pre-fix engine
%   fails every leg below that reads fail_elt (3/5/7 vs 1; the grating's
%   end-of-trace stamp vs 2).
    properties (Constant)
        Model  = 128
        RxName = 'Rx_Cass_FarField.in'
    end
    methods (TestClassSetup)
        function setup(tc)
            macos.init(tc.Model);
        end
    end
    methods (Test)
        function test_trace_stamps_the_obscuring_element(tc)
            macos.load_rx(rx_fixture_path(tc.RxName));
            for e = [2 4 6]
                s = macos.trace(e);  st = macos.get_ray_status(s.nRays);
                o = st.fail_elt(st.status == 1);
                tc.verifyGreaterThan(numel(o), 0, 'the central obscuration must clip rays (else the gate is vacuous)');
                tc.verifyEqual(unique(o(:))', int32(1), ...
                    sprintf('trace(%d): obscured rays name element 1 (pre-fix: %d, the end of the trace + 1)', e, e + 1));
            end
        end
        function test_trace_and_propagation_agree(tc)
            macos.load_rx(rx_fixture_path(tc.RxName));
            s = macos.trace(6);  a = macos.get_ray_status(s.nRays);
            macos.load_rx(rx_fixture_path(tc.RxName));
            macos.intensity(6);  b = macos.get_ray_status(s.nRays);
            tc.verifyEqual(a.status, b.status, 'the same status per ray through CTRACE and CPROPAGATE');
            tc.verifyEqual(a.fail_elt, b.fail_elt, 'the same failing element per ray through CTRACE and CPROPAGATE');
        end
    end
end
