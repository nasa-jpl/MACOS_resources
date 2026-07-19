function R = pie_rings(C2, w)
%PIE_RINGS  Robust ring classification for the pie tiling.
%   R = macos.design.pie_rings(C2, W) clusters segment centers (2 x n,
%   tiling-plane coordinates about the tiling center) into the center
%   cell + concentric wedge rings.  Single source of truth shared by
%   macos.design.seg_apertures, macos.design.seg_boundary, and
%   macos.view_rx.
%
%   The tolerance scales with the segment WIDTH, not machine epsilon:
%   real parents (figured / tilted) leave micron-level radius scatter
%   within a ring (the e2e Zernike-figured primary: ~5 um across the
%   ring-1 wedges, from the surface-normal tilt of the frame the
%   tiling plane is projected in), while distinct rings are separated
%   by ~W.  The old 1e-6*max(rc) tolerance split those wedges into
%   degenerate 1-2 member "rings", giving 2*pi/nnz sector spans and
%   go/sin(pi) apex blowups in the emitted apertures (2026-07-18).
%
%   R fields:
%     rc     1 x n segment-center radii
%     isctr  1 x n logical, center cell
%     rings  sorted ring radii (cluster representatives)
%     iring  1 x n ring index (0 for the center cell)
%     nmem   member count per ring
%
%   See also: macos.design.pie_wedge_geom.
arguments
    C2 (2,:) double
    w  (1,1) double {mustBePositive}
end
rc = vecnorm(C2);
tol = max(0.05*w, 1e-6*max(rc));
isctr = rc < tol;
rings = uniquetol(rc(~isctr), tol, 'DataScale', 1);
n = numel(rc);
iring = zeros(1, n);
nmem  = zeros(1, numel(rings));
for s = 1:n
    if isctr(s), continue; end
    [~, j] = min(abs(rings - rc(s)));
    iring(s) = j;
    nmem(j) = nmem(j) + 1;
end
R = struct('rc', rc, 'isctr', isctr, 'rings', rings, ...
           'iring', iring, 'nmem', nmem);
end
