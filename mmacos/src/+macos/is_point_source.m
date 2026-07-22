function tf = is_point_source()
%MACOS.IS_POINT_SOURCE  True iff the loaded Rx has a finite (point) source.
%   tf = macos.is_point_source() returns true for a point (finite-distance)
%   source, false for a collimated source.  See also: macos.get_src_size.
tf = logical(mmacos('src_finite'));
end
