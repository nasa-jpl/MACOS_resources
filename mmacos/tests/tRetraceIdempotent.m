classdef tRetraceIdempotent < matlab.unittest.TestCase
%TRETRACEIDEMPOTENT  Re-traces are bit-identical, and the round-off dead
%   bands that make them so neither swallow a real change nor stay silent.
%
%   Engine commit 81d3308 (2026-09-08) made identical traces identical: the
%   source-frame re-orthogonalisation at every grid setup and the object-
%   space STOP aim had no floating-point fixed point and alternated by a
%   few ulp, which every finite-difference column divided by its step
%   (Luis's "centre-channel speckle").  The cure is a DEAD BAND: the fresh
%   target is computed and the incoming state kept when they differ by
%   round-off only.  On 2026-10-02 the bands were tightened from 1e-14 /
%   1e-13 (45 / 450 ulp) to 16 ulp of the quantity's scale, MEASURED
%   against residuals <= 1 ulp across the corpus and a smallest genuine
%   update of 7e5 ulp, and a residual between 2 and 16 ulp -- larger than
%   the measured round-off, yet suppressed -- is now noted once per run and
%   counted (macos_api_mod deadband_notes_get).  Dave's question that
%   drove it: "how do we work around the 'didn't change' consequence of the
%   LSB dead band?" -- answer: keep it far below any real step, and make a
%   suppressed real step visible.
%
%   These tests pin all three properties:
%     * ten traces of the same deck give bit-identical OPD, with and without
%       an element stop, on a deck with an ApStop= header and on one without;
%     * SAVE -> load -> SAVE is byte-identical after the first round trip;
%     * a source-direction change ABOVE the band is honoured in the frame,
%       one INSIDE the band is kept out of the frame and counted, and one
%       at the round-off floor is silent.
    properties (Constant)
        ModelSize = 128
        Jwst  = 'templates/50_sensitivities/zoom_5x5/jwst_ote_designc.in'   % ApStop= header
        Hex   = 'templates/50_sensitivities/e5hex1/e5hex1.in'               % no header stop
        Ulp   = eps(1)
    end
    methods (Access = private)
        function p = deck(tc, rel)
            here = fileparts(mfilename('fullpath'));
            p = fullfile(fileparts(here), rel);
            tc.assumeTrue(isfile(p), ['fixture not found: ' p]);
        end
        function W = ten_traces_(~, n)
            nE = macos.num_elt();
            W = cell(n, 1);
            for k = 1:n, macos.trace(nE);  W{k} = macos.opd(); end
        end
        function assert_identical_(tc, W, what)
            for k = 2:numel(W)
                tc.verifyTrue(isequal(W{k}, W{1}), sprintf('%s: trace %d differs from trace 1 (max |dW| %.3e)', ...
                    what, k, max(abs(W{k}(:) - W{1}(:)))));
            end
        end
    end
    methods (TestClassSetup)
        function setupClass(tc)
            macos.init(tc.ModelSize);
        end
    end
    methods (Test)
        function test_ten_traces_bit_identical_header_stop(tc)
            macos.load_rx(tc.deck(tc.Jwst));
            tc.assert_identical_(tc.ten_traces_(10), 'jwst, ApStop header');
        end
        function test_ten_traces_bit_identical_element_stop(tc)
            macos.load_rx(tc.deck(tc.Jwst));
            macos.stop(25, [0 0]);
            tc.assert_identical_(tc.ten_traces_(10), 'jwst, stop 25');
        end
        function test_ten_traces_bit_identical_no_stop_and_segment_stop(tc)
            macos.load_rx(tc.deck(tc.Hex));
            tc.assert_identical_(tc.ten_traces_(10), 'e5hex1, no stop');
            macos.stop(1, [0 0]);
            tc.assert_identical_(tc.ten_traces_(10), 'e5hex1, stop 1');
        end
        function test_save_load_save_reaches_a_fixed_point_in_one_round_trip(tc)
            % The deck's ChfRayDir is unitised at load (msmacosio), which
            % moves it by 1 ulp on the FIRST load of a hand-written deck and
            % is then a fixed point -- so the first SAVE differs from the
            % deck, and every later round trip is byte-identical.  That
            % second property is the one the dead bands guarantee (before
            % them ChfRayPos alternated by 2 ulp on every load with an
            % ApStop= header).
            macos.load_rx(tc.deck(tc.Jwst));
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's')); %#ok<NASGU>
            A = fullfile(wd, 'A.in');  macos.save_rx(A);  macos.load_rx(A);
            B = fullfile(wd, 'B.in');  macos.save_rx(B);  macos.load_rx(B);
            C = fullfile(wd, 'C.in');  macos.save_rx(C);
            tc.verifyEqual(fileread(B), fileread(C), 'the second SAVE -> load -> SAVE must be byte-identical');
        end
        function test_band_honours_a_real_change_keeps_a_sub_band_one_and_counts_it(tc)
            macos.load_rx(tc.deck(tc.Hex));
            macos.trace(macos.num_elt());
            s0 = macos.get_src_csys();
            z0 = s0.zDir;  x0 = s0.xDir;
            n0 = mmacos('deadband_notes_get');
            % 1. ABOVE the band (4500 ulp): the frame must follow the new direction
            d1 = z0 + 1e-12 * x0;
            macos.set_src_fov('src_dir', d1);
            macos.trace(macos.num_elt());
            s1 = macos.get_src_csys();
            tc.verifyLessThan(norm(s1.zDir - d1/norm(d1)), 1e-15, 'a 4500-ulp direction change must be applied to the frame');
            tc.verifyEqual(mmacos('deadband_notes_get'), n0, 'a change above the band is not a note');
            % 2. INSIDE the band (4 ulp, above the 2-ulp quiet floor): kept out of the frame, counted
            d2 = s1.zDir + 4 * tc.Ulp * s1.xDir;
            macos.set_src_fov('src_dir', d2);
            macos.trace(macos.num_elt());
            s2 = macos.get_src_csys();
            tc.verifyTrue(isequal(s2.zDir, s1.zDir), 'a 4-ulp direction change must be kept out of the frame');
            tc.verifyGreaterThan(mmacos('deadband_notes_get'), n0, 'a suppressed 4-ulp change must be counted');
            n2 = mmacos('deadband_notes_get');
            % 3. At the round-off floor (1 ulp): kept, and silent
            d3 = s2.zDir + 1 * tc.Ulp * s2.xDir;
            macos.set_src_fov('src_dir', d3);
            macos.trace(macos.num_elt());
            s3 = macos.get_src_csys();
            tc.verifyTrue(isequal(s3.zDir, s2.zDir), 'a 1-ulp change is round-off and must be kept out of the frame');
            tc.verifyEqual(mmacos('deadband_notes_get'), n2, 'round-off must not be counted');
        end
    end
end
