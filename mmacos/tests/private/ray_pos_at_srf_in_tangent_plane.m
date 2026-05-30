function pts = ray_pos_at_srf_in_tangent_plane(tmp_rx, lines, srf)
%RAY_POS_AT_SRF_IN_TANGENT_PLANE  Trace + project rays into srf's local 2D.
%   tmp_rx : char    — path to a temp .in file to write
%   lines  : string  — string array of the modified Rx file contents
%                      (one entry per line, no trailing newline)
%   srf    : int     — surface id where mask is applied
%
%   Returns P×2 [x y] ray positions in srf's local tangent plane,
%   restricted to rays that passed both the geometric trace and any
%   masks.  Mirror of pymacos
%   tests/test_masks.py:ray_pos_at_srf_in_tangent_plane.
arguments
    tmp_rx (1,:) char
    lines  (:,1) string
    srf    (1,1) double {mustBeInteger, mustBePositive}
end

% Write modified Rx.  writelines appends '\n' per entry.
writelines(lines, tmp_rx);

% Load the modified Rx and run two traces:
%   1) trace through to image plane (srf=-3 in pymacos => "image - 3 from end"
%      maps to the focal-plane sentinel — we use num_elt() since the test
%      prescriptions terminate at a FocalPlane element).
%   2) trace to the masked surface itself.
macos.load_rx(tmp_rx);
n_total = macos.num_elt();
macos.trace(n_total);

s = macos.trace(srf);
n_rays = s.nRays;
r = macos.get_ray_info(n_rays);
ray_pass = r.ok_trace & r.ok_pass;

% Map global positions into the surface's local 2D plane.
vpt  = macos.get_elt_vpt(srf);            % 3×1, BaseUnits, global frame
c    = macos.get_elt_csys(srf);
csys = c.csys(1:3, 1:3, 1);               % 3×3 rotation (global → local cols)
dp_glb2loc = csys.' * vpt;                % offset of vpt expressed in local

pts3 = r.pos(:, ray_pass) - vpt;          % 3×P, centered on vpt
pts3_loc = csys.' * pts3 + dp_glb2loc;    % 3×P in local coords
pts = pts3_loc(1:2, :).';                 % P×2 (x, y)
end
