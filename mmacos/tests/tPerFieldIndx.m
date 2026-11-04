classdef tPerFieldIndx < matlab.unittest.TestCase
%TPERFIELDINDX  per_field_indx builds its pixel index from the VALID-RAY MASK.
%
%   A perfect on-axis telescope reads exactly 0 at every valid pupil pixel
%   under the chief-ray reference, and the Zemax Cassegrain template
%   (templates/20_segmentation/zemax_cassegrain) stopped in the per-element
%   page producer with "Vectors must be the same length": the index helper
%   rebuilt the index from the nominal map's NONZERO pixels, the core had
%   scattered the Jacobian rows with the opd_mask.  The core now exports
%   that mask (out.per_field_mask_2d) and the helper uses it.  No engine.

    methods (TestClassSetup)
        function paths(~)
            here = fileparts(mfilename('fullpath'));
            addpath(fullfile(here, '..', 'sensitivities'));
        end
    end

    methods (Static)
        function out = harvest(orient)
            % 5x5 pupil: a disc of 13 valid pixels, the centre one reads 0
            mask = false(5); W = zeros(5);
            for i = 1:5, for j = 1:5
                if (i-3)^2 + (j-3)^2 <= 4, mask(i,j) = true; W(i,j) = 1e-3*(i + 3*j); end
            end, end
            W(3,3) = 0;                       % the chief's pixel: valid, reads exactly 0
            W(2,3) = 0;                       % and one more, off the diagonal
            assert(nnz(mask) == 13 && nnz(W) == 11);
            out = struct('per_field_w_nom_2d', {{W}}, 'per_field_mask_2d', {{mask}}, ...
                         'field_names', {{'C'}}, 'opd_orient', orient);
            if strcmp(orient, 'xy')
                out.per_field_w_nom_2d = {W.'};  out.per_field_mask_2d = {mask.'};
            end
        end
    end

    methods (Test)
        function test_index_counts_every_valid_pixel(tc)
            out = tc.harvest('raw');
            idx = per_field_indx(out, 1);
            tc.verifyEqual(numel(idx.i), 13);                 % the Jacobian's row count
            % the rows round-trip onto the mask, zeros included
            rows = (1:13).';
            M = macos.v2m(rows, idx);
            tc.verifyEqual(M ~= 0, out.per_field_mask_2d{1});
            % must-fail: the old recipe (the map's nonzeros) loses the two zero pixels
            [~, bad] = macos.m2v(out.per_field_w_nom_2d{1});
            tc.verifyEqual(numel(bad.i), 11);
            tc.verifyError(@() macos.v2m(rows, bad), 'MATLAB:samelen');
        end

        function test_xy_orientation_remaps_the_mask(tc)
            raw = tc.harvest('raw');  xy = tc.harvest('xy');
            ir = per_field_indx(raw, 1);  ix = per_field_indx(xy, 1);
            tc.verifyEqual(numel(ix.i), 13);
            rows = (1:13).';
            tc.verifyEqual(macos.v2m(rows, ix), macos.v2m(rows, ir).');   % the xy page = raw transposed
        end

        function test_legacy_harvest_falls_back_to_the_map(tc)
            out = rmfield(tc.harvest('raw'), 'per_field_mask_2d');
            idx = per_field_indx(out, 1);
            tc.verifyEqual(numel(idx.i), 11);                 % as before the mask field existed
        end
    end
end
