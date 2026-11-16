function [yc, xc] = centroid_loc(arr)
%CENTROID_LOC  Intensity-weighted center of mass in fractional pixel coords.
%   Returns 0-based MATLAB-row / column fractional indices (mirrors
%   pymacos's _centroid_loc which uses NumPy convention).  Robust for
%   both Airy-like sharp PSFs and flat-top NF PSFs where peak position
%   inside a uniform region is noise-dominated.
arr = double(arr);
total = sum(arr(:));
if total <= 0
    n = size(arr, 1);
    yc = n/2 - 0.5;
    xc = n/2 - 0.5;
    return
end
[ny, nx] = size(arr);
% Use 0-based indices to match pymacos's NumPy convention.
[y, x] = ndgrid(0:ny-1, 0:nx-1);
yc = sum(arr .* y, 'all') / total;
xc = sum(arr .* x, 'all') / total;
end
