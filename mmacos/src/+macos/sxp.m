function xp = sxp(mode)
%MACOS.SXP  Set / find the system exit pupil from the current chief ray.
%   xp = macos.sxp() re-derives the exit-pupil pose and radius from
%   the current chief-ray geometry and returns a 7-element row:
%
%       [xp_pos_x, xp_pos_y, xp_pos_z,           % EP centre
%        xp_dir_x, xp_dir_y, xp_dir_z,           % EP axis direction
%        xp_radius]                              % EP sphere radius
%
%   macos.sxp(MODE) picks the axis as macos.fex does: 1 = chief ray,
%   0 = beam centroid; omitted = the SESSION's axis (macos.modes().centroid;
%   was a fixed 1 that also refused 0 -- A.6, 2026-10-16).  A stop MUST be
%   set first via macos.stop or macos.stop_obj, otherwise sxp errors.
%
%   See also: macos.stop, macos.stop_obj.
arguments
    mode double {mustBeScalarOrEmpty, mustBeMember(mode, [0 1])} = []
end
if isempty(mode)
    md = macos.modes();  mode = double(~md.centroid);
end
xp = mmacos('sxp_fnd', mode);
xp = double(xp(:)).';
end
