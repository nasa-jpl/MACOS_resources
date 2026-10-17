classdef tSegRay < matlab.unittest.TestCase
%TSEGRAY  macos.segray -- the ray through a Segment element's centre (the CLI's SEGRAYTRACE
%   through the api's segray_trace; BRIEF_to_bindings B.2, 2026-10-17); twin of pymacos
%   test_segray.py.  Deck: e5hex1 (Segment elements 1-7, m2 = element 8, collimated source).
%   The CLI numbers are what `segraytrace` printed for segments 3 and 7 (SEGRAYTRACE.jou, both
%   compilers).  The analytic leg is the command's definition, from the deck's own ChfRayDir= /
%   RptElt= lines: the ray is parallel to the chief ray through the segment's reference point,
%   so its crossing at the segment lies on that line.
    properties (Constant)
        Model = 128
        % CLI segraytrace: crossing at the segment, at m2 (element 8); direction leaving the segment
        Seg3Pos = [-1.33333333333333e+03; -2.29984379552443e+03; 2.20877392583239e+02]
        Seg3M2  = [-1.94630316570224e+02; -5.81602731534171e+03; -2.12818888039439e+04]
        Seg3Dir = [5.21907694203786e-02; -1.61159073631132e-01; -9.85547500922033e-01]
        Seg7Pos = [2.66666666666667e+03; -8.65286482234703e+00; -6.90022891415174e+01]
        Seg7M2  = [3.90032347074229e+02; -5.47980841189830e+03; -2.13166480632806e+04]
        Seg7Dir = [-1.03208762760953e-01; -2.48028939046969e-01; -9.63239117085985e-01]
    end
    methods (Static, Access = private)
        function [ne, deck] = e5()
            deck = rx_fixture_path('e5hex1.in');
            macos.init(tSegRay.Model);
            macos.load_rx(deck);
            ne = macos.num_elt();
        end
        function [d, rpt] = deck_vectors(deck)
            txt = fileread(deck);
            t = regexp(txt, 'ChfRayDir=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens', 'once');
            d = str2double(t(:));  d = d / norm(d);
            t = regexp(txt, 'RptElt=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens');
            rpt = cell2mat(cellfun(@(c) str2double(c(:)), t, 'UniformOutput', false));
        end
    end
    methods (Test)
        function test_segray_reproduces_the_cli(tc)
            tSegRay.e5();
            cases = {3, tc.Seg3Pos, tc.Seg3M2, tc.Seg3Dir; 7, tc.Seg7Pos, tc.Seg7M2, tc.Seg7Dir};
            for k = 1:2
                [s, p, q, d] = cases{k, :};
                r = macos.segray(s);
                tc.verifyTrue(r.ok);  tc.verifyEqual([r.status r.fail_elt], [0 0]);
                tc.verifyEqual(r.pos(:,s), p, 'RelTol', 1e-13, 'AbsTol', 1e-12);
                tc.verifyEqual(r.pos(:,8), q, 'RelTol', 1e-13, 'AbsTol', 1e-12);
                tc.verifyEqual(r.dir(:,s), d, 'RelTol', 1e-12, 'AbsTol', 1e-15);
            end
        end
        function test_segray_crosses_on_the_chief_parallel_through_the_centre(tc)
            [ne, deck] = tSegRay.e5();
            [d, rpt] = tSegRay.deck_vectors(deck);
            for s = 1:7
                r = macos.segray(s);
                tc.verifyTrue(r.ok);  tc.verifyEqual(r.status, 0);
                tc.verifyLessThan(norm(cross(r.pos(:,s) - rpt(:,s), d)), 1e-8, sprintf('seg %d', s));
                tc.verifyEqual(vecnorm(r.dir), ones(1, ne), 'AbsTol', 1e-12);
            end
        end
        function test_segray_refuses_a_non_segment(tc)
            ne = tSegRay.e5();
            tc.verifyTrue(macos.segray(3).ok);
            for bad = [8 ne+1]                                  % m2 is a Reflector
                tc.verifyError(@() macos.segray(bad), ?MException);
            end
        end
        function test_segray_leaves_the_full_trace_unchanged(tc)
            ne = tSegRay.e5();
            macos.trace(ne);  W0 = macos.opd();  K0 = macos.opd_mask();
            macos.segray(3);
            macos.trace(ne);
            tc.verifyEqual(macos.opd_mask(), K0);
            tc.verifyEqual(macos.opd() .* K0, W0 .* K0, 'AbsTol', 0);
        end
    end
end
