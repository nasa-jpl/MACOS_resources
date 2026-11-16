function calib_set_beam(kind, srf, target, varargin)
%MACOS.CALIB_SET_BEAM  Beam rows for the next CALIB, on ANY target.
%   macos.calib_set_beam('dir',  srf, n)   -- the chief ray's direction at
%       element srf is driven to the unit vector n (a telecentric image:
%       n = the detector normal).
%   macos.calib_set_beam('pos',  srf, p)   -- the beam position at srf
%       (chief ray, or the ray centroid with 'centroid' in
%       calib_set_beam_wt) is driven to the 3-vector p, base units.  Per-
%       field targets: macos.calib_set_beam_pos_fov.
%   macos.calib_set_beam('size', srf, r)   -- the beam radius at srf.
%   macos.calib_set_beam(kind, srf, [], 'off') switches that group off.
%
%   The rows ride on the WFE / SPOT / WFE_ZMODE target (since 2026-10-03;
%   before that only on OptTarget= BEAM) and are weighted against them by
%   calib_set_beam_wt.  Mirrors the Rx keywords OptBeamDir= / OptBeamPos= /
%   OptBeamSize= in element srf's block.
%
%   Example (telecentric at the focal plane 5, spot target):
%     m.calib_set_target('SPOT');
%     m.calib_set_beam('dir', 5, [0 0 1]);
%     m.calib_set_beam_wt(1e6);
KINDS = struct('dir', 1, 'pos', 2, 'size', 3);
key = lower(strtrim(char(kind)));
if ~isfield(KINDS, key)
    error('macos:calib_set_beam:badKind', 'kind must be ''dir'', ''pos'' or ''size'' (got %s)', key);
end
on = true;
if ~isempty(varargin) && any(strcmpi(varargin{1}, {'off', 'false'})), on = false; end
t = zeros(3, 1);
if on
    t(1:numel(target)) = double(target(:));
end
mmacos('calib_set_beam', double(KINDS.(key)), double(srf), t, double(on));
end
