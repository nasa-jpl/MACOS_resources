function [B, mask, info] = gs_zernike_segment_basis(session, rx_path, opts)
%MACOS.GS_ZERNIKE_SEGMENT_BASIS  Gram-Schmidt-orthonormalized Zernike basis
%   over a segment's TRUE (irregular) aperture -- a grid-poke influence basis
%   that respects the segment shape.  Circular Zernikes cross-talk on a hex /
%   edge segment; manufacturers spec figure over the segment aperture, so the
%   modes are orthonormalized over that aperture (MGS).
%
%   [B, mask, info] = macos.gs_zernike_segment_basis(SESSION, RX_PATH, ...)
%
%   You cannot trace to a Segment element (Invalid element type), so RX_PATH
%   must already contain a Reference surface with the PM conic placed just
%   ahead of the segments -- a valid trace target that still carries the
%   segmented footprint.  Prepare it once (copy a segment's KrElt/KcElt into a
%   Element=Reference / Surface=Conic block before the first segment); see
%   templates/50_sensitivities/run_dwdgrid_multi_singlesegbasis/
%   SegDemo3conic.in (elt 1 = the PM-conic Reference).  This routine
%   just points at that file -- it does NOT edit prescriptions.
%
%   The reference segment's rays are isolated by nearest-centre (Voronoi) over
%   the segment centres, hull-masked to its aperture, and the Noll Zernikes
%   (noll_mode, self-contained) are modified-Gram-Schmidt orthonormalized over that mask.
%   ONE basis covers all like-shaped segments -- each segment's local
%   (xMon,yMon) frame is already clocked to its orientation, so the basis is
%   applied in the local frame everywhere.
%
%   Name-value:
%     'pm_ref_elt' (req) element to trace to (the PM-conic Reference in RX_PATH).
%     'ref_seg'    (req) reference segment element; its local (xMon,yMon) frame
%                  defines the basis (pick one symmetric about an axis).
%     'seg_elts'   (req) all segment elements (used as the Voronoi centres).
%     'modes'      Noll Zernike indices.  Default 4:15.
%     'remove_rigid_body' project each mode against piston+tip+tilt over the
%                  segment, so the pokes are zero-mean / zero-tilt figure modes
%                  (no per-segment rigid-body offset in dW).  Default true.
%     'matlab_dir' DEPRECATED, ignored (Noll is now self-contained).
%
%   Returns B [N x N x numel(modes)] (orthonormal over MASK, unit RMS), the
%   logical aperture MASK, and an INFO struct.  Leaves RX_PATH loaded in
%   SESSION (the caller loads the working prescription afterwards).  Pass B as
%   the 'influence' basis to macos.dw_dgrid[_multi].
%
%   See also: macos.dw_dgrid_multi, macos.find_grid_elts.
arguments
    session
    rx_path        (1,:) char
    opts.pm_ref_elt(1,1) double {mustBeInteger, mustBePositive}
    opts.ref_seg   (1,1) double {mustBeInteger, mustBePositive}
    opts.seg_elts  (1,:) double {mustBeInteger, mustBePositive}
    opts.modes     (1,:) double = 4:15
    opts.remove_rigid_body (1,1) logical = true   % project out piston+tip+tilt
    opts.matlab_dir(1,:) char = fullfile(getenv('HOME'), 'matlab')
end
session.load_rx(rx_path);
txt = fileread(rx_path);

% ---- per-segment frames (origin + local axes), from the Rx ------------
ns = numel(opts.seg_elts);
C  = zeros(3, ns);   Xs = zeros(3, ns);   Ys = zeros(3, ns);
for k = 1:ns
    C(:,k)  = local_vec(txt, opts.seg_elts(k), 'RptElt');   % segment centre
    Xs(:,k) = local_vec(txt, opts.seg_elts(k), 'xMon');     % local x axis
    Ys(:,k) = local_vec(txt, opts.seg_elts(k), 'yMon');     % local y axis
end
iref = find(opts.seg_elts == opts.ref_seg, 1);
pMon = C(:, iref);   xMon = Xs(:, iref);   yMon = Ys(:, iref);  % == monomial frame
N    = double(mmacos('elt_srf_grid_size', opts.ref_seg, 1));
gdx  = str2num(regexp(txt, '(?<=GridSrfdx=)\s*[\d.eEdD+-]+', 'match', 'once'));  %#ok<ST2NM>

% ---- trace to the PM reference; pull the ray footprint ----------------
s  = session.trace(opts.pm_ref_elt);
ri = macos.get_ray_info(s.nRays);
P  = ri.pos;  ok = ri.ok_trace(:).';

% ---- Voronoi-isolate the reference segment, hull-mask -----------------
u = xMon.' * (P - pMon);   v = yMon.' * (P - pMon);
cu = xMon.' * (C - pMon);  cv = yMon.' * (C - pMon);
d = zeros(numel(opts.seg_elts), numel(u));
for k = 1:numel(opts.seg_elts), d(k,:) = (u-cu(k)).^2 + (v-cv(k)).^2; end
[~, near] = min(d, [], 1);
sel = ok & (near == iref);   us = double(u(sel));   vs = double(v(sel));
c = (N+1)/2;  [I, J] = ndgrid(1:N, 1:N);  GU = (I-c)*gdx;  GV = (J-c)*gdx;
K = convhull(us, vs);   mask = inpolygon(GU, GV, us(K), vs(K));

% ---- raw Noll Zernikes over the mask, MGS-orthonormalize --------------
% Over an IRREGULAR aperture the Noll modes lose their zero-mean/orthogonality
% (that holds only on a full disk), so each carries a piston + tilt.  Prepend
% piston/tip/tilt (Noll 1,2,3), orthonormalize the whole set, then DISCARD
% them -- the kept modes are then rigid-body-free (zero mean, zero tilt over
% the segment), so a poke is pure figure with no per-segment piston offset.
if opts.remove_rigid_body, rb = setdiff([1 2 3], opts.modes); else, rb = []; end
allmodes = [rb, opts.modes];
nz = numel(allmodes);  mv = mask(:);  Ball = zeros(N, N, nz);
for k = 1:nz
    zk = macos.noll_mode(double(mask), allmodes(k)) .* mask;
    for j = 1:k-1
        bj = Ball(:,:,j);
        zk = zk - (sum(zk(mv).*bj(mv)) / sum(bj(mv).^2)) * bj;
    end
    zk = zk .* mask;
    Ball(:,:,k) = zk / sqrt(mean(zk(mv).^2));
end
B = Ball(:, :, numel(rb)+1:end);       % keep the requested modes only
info = struct('N', N, 'gdx', gdx, 'mask_px', nnz(mask), 'n_seg_rays', numel(us), ...
              'modes', opts.modes, 'ref_seg', opts.ref_seg, 'seg_elts', opts.seg_elts, ...
              'seg_pMon', C, 'seg_xMon', Xs, 'seg_yMon', Ys);
end

% ======================================================================
function blk = local_block(txt, ie)             % element ie's block (EltName order)
% Split on EltName= (exactly one per element) rather than iElt= -- the
% per-element output-coordinate blocks repeat iElt= at line start, which makes
% "find iElt=N" ambiguous.  Assumes elements are in file order (ie-th EltName
% == element ie), which holds for any normally-written prescription.
ep = regexp(txt, '\n\s*EltName=', 'start');
if ie < numel(ep), blk = txt(ep(ie):ep(ie+1)-1); else, blk = txt(ep(ie):end); end
end
function v = local_vec(txt, ie, key)            % D-exponent-safe 3-vector
blk = local_block(txt, ie);
t = regexp(blk, [key '=\s*([-\d.eEdD+]+)\s+([-\d.eEdD+]+)\s+([-\d.eEdD+]+)'], 'tokens', 'once');
v = [str2num(t{1}); str2num(t{2}); str2num(t{3})];  %#ok<ST2NM>
end
