function set_src_fov(opts)
%MACOS.SET_SRC_FOV  Set the source field-of-view state ABSOLUTELY.
%   macos.set_src_fov('src_pos', POS, 'src_dir', DIR, 'zSrc', Z) writes
%   ChfRayPos, ChfRayDir, and the source distance directly.  Unlike
%   perturb_src (which applies an increment to the current state), this
%   sets each component to the supplied value.
%
%   Name-value pairs:
%     'src_pos'  [3×1] ChfRayPos in BaseUnits.  Default: current value.
%     'src_dir'  [3×1] ChfRayDir (will be renormalised by macos).
%                Default: current value.
%     'zSrc'     scalar source distance in BaseUnits.
%                Default: current value.
%
%   Any field not supplied is read from the current state via
%   get_src_fov so callers can change just one component.
%
%   Used by multi-field sensitivity loops to tip the source pointing
%   between field points without round-tripping through perturb_src.
%
%   See also: macos.get_src_fov, macos.perturb_src.
arguments
    opts.src_pos (3,1) double = NaN(3,1)
    opts.src_dir (3,1) double = NaN(3,1)
    opts.zSrc    (1,1) double = NaN
end
% Pull current state for any field the caller didn't override.
need_current = any(isnan(opts.src_pos)) || any(isnan(opts.src_dir)) ...
            || isnan(opts.zSrc);
if need_current
    cur = macos.get_src_fov();
    if any(isnan(opts.src_pos)), opts.src_pos = cur.src_pos; end
    if any(isnan(opts.src_dir)), opts.src_dir = cur.src_dir; end
    if isnan(opts.zSrc),         opts.zSrc    = cur.zSrc;    end
end
mmacos('set_src_fov', opts.zSrc, opts.src_pos, opts.src_dir);
end
