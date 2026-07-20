function out = dmet_dfig(seg, es, gm, opts)
%DMET_DFIG  Figure-state measurement Jacobians dmdz / dmdgrid.
%
%   out = macos.design.dmet_dfig(SEG, ES, GM, 'z_names', CN, ...) builds
%   the measurement-model sensitivity of the laser gauges l and the
%   edge sensors e to the FIGURE states of the segmented forward model
%   (Dave 2026-07-19: "evaluate the effect of grid and z DOFs at each
%   sensor location and use that to generate dmdz and dmdgrid"):
%
%       m = [l; e] = dmdx*x + dmdz*z + dmdgrid*g + m0
%
%   Physics: a figure DOF deforms the segment surface; a point MOUNTED
%   on that surface at world position p moves by  n_hat * f(p)  (the
%   mode shape value along the segment face normal).  Then
%     * an edge-sensor row reads  (a . n_hat) * f(p_q)  at its Hx
%       SensorPos -- the PISTON axis picks it up in full (a ~ n_hat)
%       while the in-plane gap/shear axes vanish through the same
%       projection (a perp n_hat), no special-casing;
%     * a gauge beam whose LAUNCHER rides segment s reads
%       -u_hat . n_hat * f(p_src)  (u_hat = beam direction); fiducials
%       ride the hub, which carries no figure DOFs here.
%   The engine's METcalc and the SegMirMaker Hx keep met/sensor points
%   RIGID (they do not ride figure), so these blocks are model-side --
%   they extend the estimator's H for the simulator stage.
%
%   Mode shapes:
%     z     MonZernike modes, engine-exact via macos.design.
%           zern_seg_eval (lMon-normalized, un-normalized ANSI --
%           gated by tRunCompare's grid-vs-MonZern equivalence test).
%     grid  influence maps from a macos.segment_grid_basis struct,
%           sampled at the point in the segment grid frame: bilinear
%           where the 2x2 pixel neighborhood is inside the basis mask,
%           else the NEAREST in-mask pixel value (edge sensors sit in
%           the inter-segment gap, just outside the ray-footprint mask
%           -- the physical surface extends there; nearest-pixel is
%           the modeling choice, noted per Dave's model review).
%
%   Inputs:
%     SEG   segment_rx / seg_from_rx struct (frames with rpt/xhat/
%           yhat/zhat/lmon, seg_elts, nseg)
%     ES    macos.design.edge_sensors output (new-format Hx: axis/
%           SensorPos; the per-row signed axis is recovered from the
%           row's own translation block, as in run_compare)
%     GM    macos.met_geom() output on the LOADED met Rx (src_pts/
%           tgt_pts/src_elt in met_get beam order) -- pass gm.n == 0
%           (or []) to skip the l blocks (no met Rx)
%   Options:
%     'z_names'    cellstr 'Elt <e> MonZern<j>' (oz.channel_names) --
%                  column order of dldz/dedz.  Default {} (skip z).
%     'g_names'    cellstr 'Elt <e> Grid<k>' (og.channel_names) --
%                  column order of dldg/dedg.  Default {} (skip grid).
%     'sgb'        segment_grid_basis struct (required with g_names)
%     'unit_to_m'  BaseUnits -> metres (CBM); readings are returned in
%                  SI per unit coefficient in BaseUnits, matching the
%                  dwdz/dwdgrid column convention.  Default 1.
%
%   out fields: .dldz (nl x nz), .dedz (ne x nz), .dldg (nl x ng),
%   .dedg (ne x ng), .dmdz = [dldz; dedz], .dmdgrid = [dldg; dedg]
%   (m = [l; e] order), .z_names, .g_names, .src_seg (1 x nl segment
%   index per beam, 0 = launcher not on a segment).
%
%   Columns for non-segment elements (a non-segment lMon optic swept
%   into dwdz) are ZERO -- their figure moves no segment-mounted
%   hardware in this model.
%
%   See also: macos.design.zern_seg_eval, macos.design.edge_sensors,
%             macos.met_geom, macos.segment_grid_basis, run_compare.

arguments
    seg (1,1) struct
    es (1,1) struct
    gm
    opts.z_names cell = {}
    opts.g_names cell = {}
    opts.sgb = []
    opts.unit_to_m (1,1) double {mustBePositive} = 1
end
nseg = seg.nseg;
cbm = opts.unit_to_m;
if isempty(gm) || ~isstruct(gm) || gm.n == 0
    nl = 0;
    src_seg = zeros(1, 0);
else
    nl = gm.n;
    % engine-truth launcher->segment map: src_elt is the element each
    % source point rides (hub/aft launchers map to 0)
    src_seg = zeros(1, nl);
    for b = 1:nl
        s = find(seg.seg_elts == gm.src_elt(b), 1);
        if ~isempty(s), src_seg(b) = s; end
    end
end
assert(isempty(opts.g_names) || ~isempty(opts.sgb), ...
    'dmet_dfig: g_names requires the segment_grid_basis struct (sgb)');
assert(any(es.axis > 0), ...
    'dmet_dfig: legacy Hx (no MeasAxis/SensorPos) -- regenerate');
assert(es.dof == 6, 'dmet_dfig: 6-DOF Hx required (got %d)', es.dof);

% per-segment precompute: sensor rows + signed axes + normal projections,
% launcher rows + beam-direction projections
pre = repmat(struct('rows', [], 'an', [], 'spos', [], ...
                    'beams', [], 'un', [], 'bpos', []), nseg, 1);
for s = 1:nseg
    f = seg.frames(s);
    Ts = [f.xhat, f.yhat, f.zhat];
    rows = find(any(es.meas_to_seg == s, 1));
    an = zeros(1, numel(rows));
    for k = 1:numel(rows)
        blk = es.dedx(rows(k), (s-1)*6 + (4:6)).';  % signed axis, triad
        an(k) = (Ts * blk).' * f.zhat;              % a . n_hat
    end
    beams = find(src_seg == s);
    un = zeros(1, numel(beams));
    for k = 1:numel(beams)
        d = gm.tgt_pts(:, beams(k)) - gm.src_pts(:, beams(k));
        un(k) = -(d.' / norm(d)) * f.zhat;          % -u_hat . n_hat
    end
    pre(s) = struct('rows', rows, 'an', an, ...
        'spos', es.sensor_pos(:, rows), 'beams', beams, 'un', un, ...
        'bpos', gm.src_pts(:, beams));
end

[dldz, dedz] = cols_(opts.z_names, 'MonZern', @(s, j, P) ...
    macos.design.zern_seg_eval(seg.frames(s), j, P));
if ~isempty(opts.g_names)
    G = grid_eval_(opts.sgb);
    [dldg, dedg] = cols_(opts.g_names, 'Grid', ...
        @(s, k, P) G(seg.seg_elts(s), k, P));
else
    dldg = zeros(nl, 0);  dedg = zeros(es.nmeas, 0);
end

out = struct('dldz', dldz, 'dedz', dedz, 'dldg', dldg, 'dedg', dedg, ...
    'dmdz', [dldz; dedz], 'dmdgrid', [dldg; dedg], ...
    'z_names', {opts.z_names}, 'g_names', {opts.g_names}, ...
    'src_seg', src_seg);

    function [dl, de] = cols_(names, kind, feval_)
        nzc = numel(names);
        dl = zeros(nl, nzc);
        de = zeros(es.nmeas, nzc);
        pat = ['^Elt (\d+) ' kind '(\d+)$'];
        for c = 1:nzc
            tok = regexp(names{c}, pat, 'tokens', 'once');
            assert(~isempty(tok), ...
                'dmet_dfig: unexpected %s channel name ''%s''', kind, names{c});
            elt = str2double(tok{1});
            j = str2double(tok{2});
            s = find(seg.seg_elts == elt, 1);
            if isempty(s), continue; end            % non-segment optic: zero
            p = pre(s);
            if ~isempty(p.rows)
                de(p.rows, c) = (p.an .* feval_(s, j, p.spos)).' * cbm;
            end
            if ~isempty(p.beams)
                dl(p.beams, c) = (p.un .* feval_(s, j, p.bpos)).' * cbm;
            end
        end
    end
end

% =========================================================================
function G = grid_eval_(sgb)
%GRID_EVAL_  Influence-map point evaluator over a segment_grid_basis.
%   G(elt, k, P, frame) samples mode k of segment element elt at the
%   3 x n world points P: bilinear inside the mask, nearest in-mask
%   pixel otherwise.  Pixel grid: GU/GV = (index - (N+1)/2) * gdx in
%   the clocked (xData, yData) frame about pMon (segment_grid_basis
%   construction; == the engine grid frame after grid_augment_rx).
N = sgb.N;  gdx = sgb.gdx;  c0 = (N + 1) / 2;
segs = sgb.seg;
ielt = [segs.iElt];
G = @sample_;
    function z = sample_(elt, k, P)
        e = find(ielt == elt, 1);
        assert(~isempty(e), 'dmet_dfig: no grid basis for elt %d', elt);
        B = segs(e).B(:, :, k);
        msk = segs(e).mask;
        d = P - segs(e).pMon;
        u = segs(e).xData.' * d;  v = segs(e).yData.' * d;
        i = u / gdx + c0;         j = v / gdx + c0;
        z = zeros(1, size(P, 2));
        [mi, mj] = find(msk);
        for q = 1:numel(z)
            i0 = floor(i(q));  j0 = floor(j(q));
            if i0 >= 1 && j0 >= 1 && i0 < N && j0 < N ...
                    && all(all(msk(i0:i0+1, j0:j0+1)))
                fi = i(q) - i0;  fj = j(q) - j0;
                z(q) = B(i0, j0)*(1-fi)*(1-fj) + B(i0+1, j0)*fi*(1-fj) ...
                     + B(i0, j0+1)*(1-fi)*fj  + B(i0+1, j0+1)*fi*fj;
            else
                [~, q0] = min((mi - i(q)).^2 + (mj - j(q)).^2);
                z(q) = B(mi(q0), mj(q0));
            end
        end
    end
end
