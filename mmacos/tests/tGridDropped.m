classdef tGridDropped < matlab.unittest.TestCase
%TGRIDDROPPED  A grid FILE larger than the model's mGridMat is declined by name and COUNTED (engine 2026-10-10).
%   GridInit used to read such a file into GridMat(mGridMat,mGridMat,slot) past its slot, into
%   the neighbouring segments' slots, and the trace read it back out of bounds (found by the
%   gfortran debug build).  Every e5hex1 load at model 128 did it: its eight segment grids are a
%   256-point flat.txt.  Now (CC's ruling W) the grid is NOT loaded, the load proceeds, the surface
%   traces without it, and macos.grid_dropped counts the declines so a figure dropped at too
%   small a model is never silent in a harvest.
%     1 e5hex1 at model 128: the load succeeds, 8 grids declined, the trace runs.
%     2 the same deck at model 256 (mGridMat 256): nothing declined.
    methods (Test)
        function test_too_small_a_model_declines_and_counts(tc)
            macos.init(128);
            macos.load_rx(rx_fixture_path('e5hex1.in'));
            tc.verifyEqual(macos.grid_dropped(), 8, 'the eight 256-point segment grids, declined');
            tr = macos.trace(macos.num_elt() - 1);
            tc.verifyGreaterThan(tr.nRays, 0, 'the load proceeds and the deck traces');
        end
        function test_a_model_that_fits_declines_nothing(tc)
            macos.init(256);
            macos.load_rx(rx_fixture_path('e5hex1.in'));
            tc.verifyEqual(macos.grid_dropped(), 0);
        end
    end
end
