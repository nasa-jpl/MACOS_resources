classdef tSaveFixedPoint < matlab.unittest.TestCase
%TSAVEFIXEDPOINT  SAVE -> load -> SAVE reaches a fixed point; the source frame is not rebuilt on every load.
%   PLAN_CONSOLIDATION 6a (2026-10-09; found as CC's 5d, the dyson5 roll-180 joins).  OrthoSrcFrame
%   (mathsub.F) keeps the incoming source frame when the re-orthogonalised candidate differs by round-off
%   only -- but it compared x, y AND z, and z (zGrid) is never deck state: right after a load it was 0 or
%   the previous deck's, the test failed at order 1, and the frame was rebuilt from cross products on
%   every load.  A frame with round-off-sized components (a 180-deg roll leaves sin(pi) = 1.22e-16 in
%   xGrid) then never reached a fixed point: SAVE -> load -> SAVE walked those components an ulp per load
%   (33 corpus decks; measured: no two-cycle anywhere, 102 decks settle once -- the documented one-time
%   unitise of a hand-written vector -- and 34 drift).  Now the band compares x and y only and zGrid is
%   always written from the candidate.
%     1 three SAVE generations of Rx_Roll180Frame.in (= mmacos challenges/dyson5/
%       dyson5_t5f_cprime_centroid_roll180_e2e.in) are byte-identical.  Red pre-fix: xGrid / yGrid differ.
%     2 the cross-deck guard (CC): load and trace e5hex1, then load and trace the roll-180 deck in the SAME
%       session -- its source frame is bit-identical to the same deck's in a fresh session (the path that
%       would inherit the previous deck's zGrid).
    properties (Constant)
        ModelSize = 256
    end
    methods (Test)
        function test_three_save_generations_are_identical(tc)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            m = macos.Session(tc.ModelSize);
            src = rx_fixture_path('Rx_Roll180Frame.in');  g = cell(1, 3);
            for k = 1:3
                m.load_rx(src);  g{k} = fullfile(wd, sprintf('g%d.in', k));  m.save_rx(g{k});  src = g{k};
            end
            L = cellfun(@(f) splitlines(string(fileread(f))), g, 'uni', 0);
            for k = 2:3
                tc.assertEqual(numel(L{k}), numel(L{1}));
                d = find(L{k} ~= L{1});
                tc.verifyEmpty(d, sprintf('SAVE generation %d differs from generation 1: %s', k, strjoin(L{k}(d), ' | ')));
            end
        end
        function test_the_previous_decks_frame_is_not_inherited(tc)
            m = macos.Session(tc.ModelSize);
            m.load_rx(rx_fixture_path('Rx_Roll180Frame.in'));  m.trace(m.num_elt());  s0 = macos.get_src_csys();
            m.load_rx(rx_fixture_path('e5hex1.in'));  m.trace(m.num_elt());  sA = macos.get_src_csys();
            m.load_rx(rx_fixture_path('Rx_Roll180Frame.in'));  m.trace(m.num_elt());  s1 = macos.get_src_csys();
            tc.assertGreaterThan(norm(sA.zDir - s0.zDir), 0.1, 'the two decks have different chief rays (non-vacuous)');
            tc.verifyEqual([s1.xDir s1.yDir s1.zDir], [s0.xDir s0.yDir s0.zDir], 'the frame after another deck is bit-identical');
        end
    end
end
