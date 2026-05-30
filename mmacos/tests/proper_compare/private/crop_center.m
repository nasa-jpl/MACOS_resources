function out = crop_center(arr, n)
%CROP_CENTER  Centre crop a 2D array to (n × n).
arguments
    arr (:,:) double
    n   (1,1) double {mustBeInteger, mustBePositive}
end
[sy, sx] = size(arr);
oy = floor((sy - n) / 2);
ox = floor((sx - n) / 2);
out = arr(oy+1:oy+n, ox+1:ox+n);
end
