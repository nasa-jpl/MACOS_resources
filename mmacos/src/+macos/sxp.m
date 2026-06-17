function xp = sxp(mode)
%MACOS.SXP  Set / find the system exit pupil from the current chief ray.
%   xp = macos.sxp() re-derives the exit-pupil pose and radius from
%   the current chief-ray geometry and returns a 7-element row:
%
%       [xp_pos_x, xp_pos_y, xp_pos_z,           % EP centre
%        xp_dir_x, xp_dir_y, xp_dir_z,           % EP axis direction
%        xp_radius]                              % EP sphere radius
%
%   macos.sxp(MODE) selects a specific SXP backend variant (see
%   macos_api_mod sxp_fnd); default MODE=1 runs the standard
%   chief-ray-distance-to-FP path.  A stop MUST be set first via
%   macos.stop or macos.stop_obj, otherwise sxp errors.
%
%   See also: macos.stop, macos.stop_obj.
arguments
    mode (1,1) double {mustBeInteger, mustBePositive} = 1
end
xp = mmacos('sxp_fnd', mode);
xp = double(xp(:)).';
end
