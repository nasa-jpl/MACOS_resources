function out = embed_macos_array_in_proper_grid(arr, geom, allow_resample, fill_value)
%EMBED_MACOS_ARRAY_IN_PROPER_GRID  Centre-pad a macos array into PROPER's grid.
%   Mirror of pymacos:
%   tests/proper_compare/geometries/cass_farfield.py:_embed_macos_array_in_proper_grid
%
%   By construction macos's source-grid pixel pitch equals PROPER's
%   entrance-pupil pitch (proper_grid_n * beam_ratio = macos source
%   grid size).  No resampling needed; centre-pad only.
arguments
    arr            (:,:) double
    geom           (1,1) struct
    allow_resample (1,1) logical = false
    fill_value     (1,1) double = 0.0
end
macos_n = size(arr, 1);
inner_n = round(geom.proper_grid_n * geom.proper_beam_ratio);

if macos_n == inner_n
    n = geom.proper_grid_n;
    out = fill_value * ones(n, n);
    off = (n - macos_n) / 2;
    out(off+1:off+macos_n, off+1:off+macos_n) = arr;
    return
end

if ~allow_resample
    error('embed_macos_array_in_proper_grid:shapeMismatch', ...
        'macos array size %d does not match PROPER inner-pupil size %d', ...
        macos_n, inner_n);
end

% Bilinear resample to the full PROPER grid.
factor = geom.proper_grid_n / macos_n;
out = imresize(arr, factor, 'bilinear');
end
