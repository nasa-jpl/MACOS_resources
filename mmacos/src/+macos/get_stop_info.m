function s = get_stop_info()
%MACOS.GET_STOP_INFO  Element and vertex offset of the system stop.
%   s = macos.get_stop_info() returns a struct:
%       .elt     scalar   element id at which the aperture stop is defined
%       .offset  1×2      [dx,dy] offset from that surface's vertex
%
%   Errors (mmacos raises) if no stop is currently set.  See also:
%   macos.stop, macos.Session.stop.
[elt, offset] = mmacos('stop_info_get');
s.elt    = double(elt);
s.offset = offset(:)';
end
