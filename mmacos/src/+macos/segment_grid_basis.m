function out = segment_grid_basis(session, rx_path, opts)
%MACOS.SEGMENT_GRID_BASIS  Per-segment Zernike grid (GridMat) generator.
%   OUT = macos.segment_grid_basis(SESSION, RX_PATH, ...) steps through every
%   grid-bearing segment of a SEGMENTED prescription and builds, IN THAT
%   SEGMENT'S OWN clocked (xData,yData) frame, a bespoke aperture mask plus a
%   stack of Zernike figure modes sampled on the segment's N x N grid -- i.e.
%   the per-segment GridMat content.
%
%   Unlike macos.gs_zernike_segment_basis (one reference-segment basis reused
%   for every segment) this fits EACH segment's true footprint, so edge /
%   partial segments get their own mask and their modes are oriented by that
%   segment's grid axes.  For a regular flower (all full congruent hexes) the
%   per-segment masks come out congruent; the difference shows on clipped edge
%   segments of a real aperture.
%
%   Segment footprints come from tracing to a near-pupil REFERENCE surface (a
%   valid trace target carrying the segmented footprint -- you cannot trace to
%   a Segment element).  For source-defined segmentation (SegDemo*, e5*) put
%   the Reference just BEFORE the PM; for non-sequential segmentation, just
%   after.  This routine only POINTS at that Rx; it does not edit prescriptions.
%
%   Name-value:
%     'pm_ref_elt'  (req) element to trace to (the near-pupil Reference).
%     'seg_elts'    grid segments to build for.  Default macos.find_grid_elts().
%     'center_elts' Voronoi centres used to partition the traced rays.  Default:
%                   every Element=Segment in the Rx (includes a centre segment
%                   that may carry no grid).
%     'modes'       Zernike indices.  Default 4:15.
%     'orthogonalize' true  = Gram-Schmidt orthonormalize each segment's modes
%                            over its OWN aperture (manufacturer figure
%                            convention; circular modes cross-talk on a hex);
%                     false = plain circular Zernikes confined to the segment.
%                     Default true.
%     'zern_type'   Zernike convention of the raw modes:
%                     'ansi' (engine ZerntoMon1/NormANSI) | 'noll' (zernike_mode).
%                     Default 'ansi' (engine-exact).  Other macos conventions
%                     (Fringe, BornWolf, ExtFringe, Norm* variants -- see
%                     surfsub.F generators) are a TODO: add a branch in zern_local_.
%     'remove_rigid_body' project out piston+tip+tilt over each segment so the
%                   kept modes are pure figure (GS path only).  Default true.
%     'matlab_dir'  folder holding zernike_mode (Noll).  Default ~/matlab.
%
%   Returns OUT with fields N, gdx, modes, zern_type, orthogonalize,
%   remove_rigid_body, seg_elts, center_elts, and a struct array OUT.seg with
%   one entry per seg_elt:
%       .iElt .name .mask [NxN logical] .B [NxN x numel(modes)] (unit-RMS over
%       .mask) .pMon .xData .yData .mask_px .n_rays .R_seg
%   Each OUT.seg(s).B(:,:,k) is the GridMat for mode MODES(k) on that segment
%   (load it via macos.elt_grid_add, or sum_k coef_k*B(:,:,k) for a figure).
%   Leaves RX_PATH loaded in SESSION.
%
%   See also: macos.gs_zernike_segment_basis, macos.zernike_grid_basis,
%             macos.dw_dgrid_multi, macos.find_grid_elts, macos.elt_grid_add.
arguments
    session
    rx_path          (1,:) char
    opts.pm_ref_elt  (1,1) double {mustBeInteger, mustBePositive}
    opts.seg_elts    (1,:) double {mustBeInteger} = []
    opts.center_elts (1,:) double {mustBeInteger} = []
    opts.modes       (1,:) double = 4:15
    opts.orthogonalize     (1,1) logical = true
    opts.zern_type   (1,:) char {mustBeMember(opts.zern_type,{'noll','ansi'})} = 'ansi'
    opts.remove_rigid_body (1,1) logical = true
    opts.matlab_dir  (1,:) char = fullfile(getenv('HOME'), 'matlab')
end
if strcmpi(opts.zern_type, 'noll'), addpath(opts.matlab_dir); end   % zernike_mode
session.load_rx(rx_path);
txt = fileread(rx_path);

% ---- segment sets ----------------------------------------------------
center_elts = opts.center_elts;
if isempty(center_elts), center_elts = find_segment_elts_(txt); end
seg_elts = opts.seg_elts;
if isempty(seg_elts)
    % Grid-bearing SEGMENTS only: intersect the grid-bearing set (nGridMat>0)
    % with the Voronoi centres (Element=Segment).  find_grid_elts() keys on
    % nGridMat alone, so it also picks up grid-bearing elements that are NOT
    % part of the segmentation and cannot be Voronoi-partitioned -- a
    % conforming Reference (a passive trace target holding a Zernike basis
    % definition) or a downstream full-aperture refractor.  Those are dropped
    % here (a Reference is never a basis candidate; a full-aperture optic uses
    % the whole-grid basis path, not this per-segment one). -CC
    seg_elts = intersect(macos.find_grid_elts().', center_elts, 'stable');
end

% ---- Voronoi centres (segment origins), from the Rx ------------------
nc = numel(center_elts);
C  = zeros(3, nc);
for k = 1:nc, C(:,k) = local_vec_(txt, center_elts(k), 'RptElt'); end

% ---- trace to the near-pupil reference; pull the ray footprint -------
s  = session.trace(opts.pm_ref_elt);
ri = macos.get_ray_info(s.nRays);
P  = ri.pos;  ok = ri.ok_trace(:).';

% ---- per-segment bespoke mask + Zernike modes ------------------------
out = struct('N',[], 'gdx',[], 'modes',opts.modes, 'zern_type',opts.zern_type, ...
    'orthogonalize',opts.orthogonalize, 'remove_rigid_body',opts.remove_rigid_body, ...
    'seg_elts',seg_elts, 'center_elts',center_elts, 'seg',struct([]));
for ii = 1:numel(seg_elts)
    e     = seg_elts(ii);
    pMon  = local_vec_(txt, e, 'pMon');
    xData = local_vec_or_(txt, e, 'xData', 'xMon');   % grid axes (== mon frame)
    yData = local_vec_or_(txt, e, 'yData', 'yMon');
    N     = double(mmacos('elt_srf_grid_size', e, 1));
    gdx   = grid_dx_(txt, e);

    % project rays + centres into THIS segment's clocked (xData,yData) frame
    u  = xData.' * (P - pMon);   v  = yData.' * (P - pMon);
    cu = xData.' * (C - pMon);   cv = yData.' * (C - pMon);
    ic = find(center_elts == e, 1);
    if isempty(ic)
        error('macos:segment_grid_basis:center', ...
              'seg elt %d is not among center_elts', e);
    end
    d = (u - cu(:)).^2 + (v - cv(:)).^2;             % nc x nRays (Voronoi distance)
    [~, near] = min(d, [], 1);
    sel = ok & (near == ic);   us = double(u(sel));   vs = double(v(sel));

    % bespoke aperture mask on the segment grid (segment-local frame)
    c  = (N+1)/2;  [I, J] = ndgrid(1:N, 1:N);  GU = (I-c)*gdx;  GV = (J-c)*gdx;
    Kh = convhull(us, vs);   mask = inpolygon(GU, GV, us(Kh), vs(Kh));

    % Zernike modes over the mask (circular or Gram-Schmidt)
    [B, Rseg] = seg_modes_(mask, GU, GV, opts.modes, opts.zern_type, ...
                           opts.orthogonalize, opts.remove_rigid_body);

    seg = struct('iElt',e, 'name',elt_name_(txt,e), 'mask',mask, 'B',B, ...
        'pMon',pMon, 'xData',xData, 'yData',yData, 'mask_px',nnz(mask), ...
        'n_rays',numel(us), 'R_seg',Rseg);
    if isempty(out.seg), out.seg = seg; else, out.seg(ii) = seg; end
    out.N = N;  out.gdx = gdx;
end
end

% ======================================================================
%  Mode construction
% ======================================================================
function [B, Rseg] = seg_modes_(mask, GU, GV, modes, zern_type, ortho, rrb)
% Build the K-mode basis over MASK.  Centroid-centred, max-radius-normalised
% (matching zernike_mode), so 'ansi' and 'noll' share the same support disk.
N = size(mask, 1);  mv = mask(:);
gu = GU(mask);  gv = GV(mask);
cu = mean(gu);  cv = mean(gv);                       % segment centroid
Rseg = max(sqrt((gu - cu).^2 + (gv - cv).^2));       % circumscribing radius
rho  = sqrt((GU - cu).^2 + (GV - cv).^2) / Rseg;
th   = atan2(GV - cv, GU - cu);

% raw modes (prepend piston/tip/tilt only for GS rigid-body removal)
if ortho && rrb, rb = setdiff([1 2 3], modes); else, rb = []; end
allmodes = [rb, modes];   nz = numel(allmodes);
raw = zeros(N, N, nz);
for k = 1:nz
    raw(:,:,k) = zern_local_(allmodes(k), rho, th, zern_type, mask) .* mask;
end

if ortho
    % modified Gram-Schmidt over the segment aperture; drop the rigid-body lead
    Ball = zeros(N, N, nz);
    for k = 1:nz
        zk = raw(:,:,k);
        for j = 1:k-1
            bj = Ball(:,:,j);
            zk = zk - (sum(zk(mv).*bj(mv)) / sum(bj(mv).^2)) * bj;
        end
        zk = zk .* mask;
        Ball(:,:,k) = zk / sqrt(mean(zk(mv).^2));    % unit RMS over mask
    end
    B = Ball(:, :, numel(rb)+1:end);
else
    % plain circular Zernikes confined to the segment, unit RMS over mask
    B = zeros(N, N, numel(modes));
    for k = 1:numel(modes)
        zk = raw(:, :, k);
        B(:, :, k) = zk / sqrt(mean(zk(mv).^2));
    end
end
end

function Z = zern_local_(j, rho, th, zern_type, mask)
% Raw analytic Zernike mode j over (rho,th); type selects the convention.
switch lower(zern_type)
    case 'ansi'
        Z = ansi_zernike_(j, rho, th);               % engine ZerntoMon1 / NormANSI
    case 'noll'
        Z = zernike_mode(double(mask), j);           % ~/matlab Noll, mask support
    otherwise
        error('macos:segment_grid_basis:ztype', 'unsupported zern_type ''%s''', zern_type);
end
end

% ---- engine-exact ANSI Zernike (ported from macos.zernike_grid_basis) ----
function Z = ansi_zernike_(j, rho, th)
jj = j - 1;
n  = ceil((-3 + sqrt(9 + 8*jj)) / 2);
m  = 2*jj - n*(n + 2);
am = abs(m);
R  = zeros(size(rho));
for s = 0:((n - am)/2)
    c = (-1)^s * factorial(n - s) / ...
        (factorial(s) * factorial((n + am)/2 - s) * factorial((n - am)/2 - s));
    R = R + c * rho.^(n - 2*s);
end
if m >= 0, ang = cos(m*th); else, ang = sin(am*th); end
Z = norm_rms_ansi_(j) .* R .* ang;
end
function v = norm_rms_ansi_(j)
P = [1, 2, 2, sqrt(6), sqrt(3), sqrt(6), sqrt(8), sqrt(8), sqrt(8), sqrt(8), ...
     sqrt(10), sqrt(10), sqrt(5), sqrt(10), sqrt(10)];
if j > numel(P)
    error('macos:segment_grid_basis:mode', ...
        'NORM_RMS_PARAM_ANSI tabulated to mode 15 (got %d); extend P for ANSI.', j);
end
v = P(j);
end

% ======================================================================
%  Prescription parsing (D-exponent safe; EltName-block addressed)
% ======================================================================
function g = find_segment_elts_(txt)
% 1-based indices of every Element=Segment block, in file order.
ep = regexp(txt, '\n\s*EltName=', 'start');
g  = [];
for ie = 1:numel(ep)
    blk = block_(txt, ep, ie);
    if ~isempty(regexp(blk, 'Element=\s*Segment', 'once')), g(end+1) = ie; end %#ok<AGROW>
end
end
function blk = block_(txt, ep, ie)
if ie < numel(ep), blk = txt(ep(ie):ep(ie+1)-1); else, blk = txt(ep(ie):end); end
end
function blk = local_block_(txt, ie)
ep = regexp(txt, '\n\s*EltName=', 'start');
blk = block_(txt, ep, ie);
end
function v = local_vec_(txt, ie, key)
blk = local_block_(txt, ie);
t = regexp(blk, [key '=\s*([-\d.eEdD+]+)\s+([-\d.eEdD+]+)\s+([-\d.eEdD+]+)'], ...
           'tokens', 'once');
if isempty(t)
    error('macos:segment_grid_basis:key', 'key %s not found in elt %d', key, ie);
end
v = [dnum_(t{1}); dnum_(t{2}); dnum_(t{3})];
end
function v = local_vec_or_(txt, ie, key, alt)
blk = local_block_(txt, ie);
if ~isempty(regexp(blk, [key '='], 'once')), v = local_vec_(txt, ie, key);
else,                                        v = local_vec_(txt, ie, alt);  end
end
function x = grid_dx_(txt, ie)
blk = local_block_(txt, ie);
x = dnum_(regexp(blk, '(?<=GridSrfdx=)\s*[\d.eEdD+-]+', 'match', 'once'));
end
function nm = elt_name_(txt, ie)
blk = local_block_(txt, ie);
nm = strtrim(regexp(blk, '(?<=EltName=).*', 'match', 'once', 'lineanchors'));
end
function x = dnum_(s)                                 % Fortran D-exponent -> double
x = str2double(regexprep(strtrim(s), '[dD]', 'e'));
end
