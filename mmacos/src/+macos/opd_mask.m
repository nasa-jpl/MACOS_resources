function M = opd_mask(opts)
%MACOS.OPD_MASK  Which pixels of the last OPD map hold a ray (N x N logical).
%   M = macos.opd_mask() is TRUE exactly where the engine's last OPD wrote
%   the map (engine 2026-10-10, api opd_mask_get), in the same (i,j) =
%   (X,Y) layout as macos.opd().  Use it -- NOT W ~= 0 -- to find the
%   pupil: under the chief-ray reference (the default) a valid ray whose
%   path equals the chief's reads EXACTLY 0, the value of an empty pixel.
%   It follows the obscuration option (obs_set) exactly as the map does.
%
%   'orient'  'raw' (default) | 'xy' -- as macos.opd.
%
%   See also: macos.opd, macos.m2v, macos.opd_ref.
arguments
    opts.orient (1,:) char {mustBeMember(opts.orient, {'raw','xy'})} = 'raw'
end
N = size(mmacos('opd'), 1);
M = logical(mmacos('opd_mask_get', double(N)));
if strcmp(opts.orient, 'xy')
    M = M.';
end
end
