function out = met_layout_opt(seg, D, E, X, opts)
%MET_LAYOUT_OPT  Tier-3 MET-layout optimization with SHAPE-CLASS patterns.
%
%   out = macos.design.met_layout_opt(SEG, D, E, X, 'hub', h, ...)
%   minimizes the post-control wavefront residual
%       trace(dwdx * P_dx * dwdx'),
%       P_dx = X - X*H'*(H*X*H' + R)^-1*H*X,   H = [dedx; dldx]
%   over the MET launcher/fiducial layout, using the ANALYTIC gauge
%   Jacobian (macos.design.dldx_analytic == engine FD, tMet) -- the
%   product hoist of the e5_seg_metopt v3 search (2026-07-19).
%
%   SHAPE CLASSES (Dave 2026-07-18): the optimizer solves ONE launcher
%   pattern per segment SHAPE CLASS, expressed in the segment frame,
%   replicated to every same-shape segment.  Classes are discovered by
%   BOUNDARY CONGRUENCE (same vertex count + edge lengths of the
%   macos.design.seg_boundary polygon), so a pie tiling yields a
%   center-hexagon class + a wedge class, a hex tiling one class --
%   and imported rxpoly prescriptions classify with no special cases.
%   Patterns are 3 mirror pairs about each member's own symmetry axis
%   (its face-frame local x -- for SegMirMaker pie frames that is the
%   wedge bisector / center flat-normal, so same-shape members get
%   CONGRUENT hardware).  Multi-class layouts are searched by
%   coordinate descent (2 sweeps) + a joint top-K cross refinement.
%
%   INPUTS (body/column order everywhere: [segments..., hub(, aft)]):
%     SEG   segment_rx / seg_from_rx struct (frames, seg_elts, in).
%           The Rx must be LOADED in the current session (met_bodies /
%           base-unit queries).
%     D     dwdx  (nw x 6*nb, SI, local triads; NaN rows tolerated)
%     E     dedx  (ne x 6*nb, SI; hub/aft columns zero)
%     X     prior covariance (6*nb x 6*nb, SI)
%
%   OPTIONS (geometry in the Rx's BaseUnits; noise in SI metres):
%     'hub'        fiducial-carrier element index (required)
%     'aft'        aft launcher-ring element ([] = no aft leg)
%     'r_extra'    aft ring radius ([] = element radius + edge_off)
%     'sig_edge','sig_met'      sensor noise sigmas (required)
%     'edge_off','min_sep'      launcher clearance / separation gate
%     'fid_inset'  fiducial rim zone depth (default 25 mm equivalent)
%     'rfid_grid'  explicit fiducial radii ([] = rim zone from hub)
%     'nf_grid','families','phi_grid','psi_grid','c0_grid',
%     'delta_grid','fclk_coarse','fclk_fine','nrefine','topk'
%                  search grids (e5_seg_metopt v3 defaults)
%     'pattern_frame'  'segment' (default; Dave's shape-class rule) |
%                  'radial' (pattern about the array radial centerline)
%     'unit_to_m'  BaseUnits->m ([] = query the loaded session)
%     'verbose'    progress prints (default true)
%
%   OUT: .best (family/angs per class, pmap, nf, rfid, fclock),
%        .launch_pts {1 x nseg} 3x6 GLOBAL points (add_met override),
%        .classes {1 x nc} member index lists, .class_of (1 x nseg),
%        .src_aft, .r0/.w0m baseline, .rb/.wb best, .bodies,
%        .unit_to_m, .n_layouts, .feasible_frac
%
%   The engine-FD validation of the winner (dmet_dx on the realized
%   .in) belongs to the CALLER (see run_met / e5_seg_metopt).
%
%   See also: macos.design.add_met, macos.design.dldx_analytic,
%             macos.design.met_bodies, macos.design.seg_boundary,
%             run_met.

arguments
    seg (1,1) struct
    D double = []
    E double = []
    X double = []
    opts.apply = []             % PRESET APPLY MODE (no search): a
                                % preset struct (out.preset of an
                                % earlier run -- scale-free class
                                % patterns + fiducial inset + aft
                                % block).  Classes are matched by
                                % member count and the preset is
                                % REALIZED on this build's boundaries;
                                % returns launch_pts/best/preset with
                                % no merit evaluation (D/E/X unused).
                                % "Save the optimized configuration as
                                % the new as-built for future PIE
                                % builds" -- Dave 2026-07-19.
    opts.hub (1,1) double {mustBeInteger, mustBePositive}
    opts.aft double = []
    opts.r_extra double = []
    opts.sig_edge (1,1) double {mustBePositive} = 1e-9
    opts.sig_met (1,1) double {mustBePositive} = 1e-9
    opts.edge_off (1,1) double {mustBeNonnegative} = 0
    opts.min_sep (1,1) double {mustBeNonnegative} = 0
    opts.fid_inset double = []
    opts.rfid_grid double = []
    opts.nf_grid double = [3 6]
    opts.families string = ["spread" "cluster"]
    opts.phi_grid double = deg2rad(10:10:170)
    opts.psi_grid double = deg2rad(20:20:160)
    opts.c0_grid double = [0 pi]
    opts.delta_grid double = deg2rad([4 10 20])
    opts.fclk_coarse double = deg2rad(0:30:90)
    opts.fclk_fine double = deg2rad(0:15:105)
    opts.nrefine (1,1) double = 12
    opts.topk (1,1) double = 6
    opts.pattern_frame (1,1) string ...
        {mustBeMember(opts.pattern_frame, ["segment" "radial"])} = "segment"
    opts.unit_to_m double = []
    opts.gram double = []       % override for D'*D (e.g. tilt-removed
                                % wavefront Gram; Dave 2026-07-19 --
                                % optimize the NON-TILT WFE)
    opts.nw double = []         % row count matching 'gram' ([] = finite
                                % rows of D)
    opts.extra_layouts struct = struct([])  % named benchmark layouts:
                                % .name + .angs {1 x nc}; optional
                                % .pmap/.nf/.rfid/.fclock.  Evaluated
                                % over the fiducial grids with the
                                % min-sep gate BYPASSED (their actual
                                % separation is reported instead) and
                                % allowed to win when feasible.
    opts.corner_pairs (1,1) logical = true  % auto-benchmark (Dave
                                % 2026-07-19): launcher pairs AT each
                                % class's two max-radius boundary
                                % corners + a pair on the inside edge
                                % (the boundary point opposite)
    opts.sym_assign (1,1) logical = true    % ROTATIONAL fiducial
                                % assignment (Dave 2026-07-19: with 6
                                % fiducials the segments become
                                % ~interchangeable): each class member's
                                % map = the base pmap shifted by its
                                % clocking within the class, so every
                                % member's truss is congruent hardware
                                % AND congruent beam geometry.  Exact
                                % when the member clocking step is a
                                % multiple of the fiducial pitch
                                % (2*pi/nf); shifts are rounded.
    opts.verbose (1,1) logical = true
end
say = @(varargin) opts.verbose && fprintf(varargin{:});
nseg = seg.nseg;
has_aft = ~isempty(opts.aft);
nb = nseg + 1 + has_aft;
apply_mode = ~isempty(opts.apply);
tag = '';  if has_aft, tag = ', aft'; end
G = [];  nw = [];  bodies = [];
if ~apply_mode
    assert(size(D,2) == 6*nb && size(E,2) == 6*nb && isequal(size(X), [6*nb 6*nb]), ...
        'met_layout_opt: D/E/X must have %d columns (bodies = [segments, hub%s])', ...
        6*nb, tag);
    if ~isempty(opts.gram)
        G = opts.gram;  nw = opts.nw;
        assert(isequal(size(G), [6*nb 6*nb]) && ~isempty(nw), ...
            'met_layout_opt: gram must be %dx%d with nw given', 6*nb, 6*nb);
    else
        keep = all(isfinite(D), 2);
        Dk = D(keep, :);  G = Dk'*Dk;  nw = nnz(keep);
    end
    % engine-truth body frames ([segments, hub(, aft)])
    bodies = macos.design.met_bodies([seg.seg_elts, opts.hub, opts.aft]);
end

u2m = opts.unit_to_m;
if isempty(u2m) && ~apply_mode, u2m = mmacos('base_unit_to_metres'); end
if isempty(u2m), u2m = 1; end               % apply mode: geometry only

% hub fiducial plane + rim zone (text truth, by element INDEX)
[pv, ps, r_ap] = elt_geom_(seg.in, opts.hub);
assert(isfinite(r_ap) || ~isempty(opts.rfid_grid), ...
    'met_layout_opt: hub element %d has no ApVec/lMon -- pass rfid_grid', ...
    opts.hub);
[~, imin] = min(abs(ps));  e0 = zeros(3,1);  e0(imin) = 1;
xh = cross(ps, e0);  xh = xh/norm(xh);  yh = cross(ps, xh);
rfid_grid = opts.rfid_grid;
if isempty(rfid_grid)
    inset = opts.fid_inset;
    if isempty(inset), inset = 25e-3/u2m; end     % 25 mm equivalent
    rfid_grid = [r_ap - inset, r_ap - inset/2, r_ap];
end
say('hub elt %d: aperture radius %.4g -> fiducial radii [%s]\n', ...
    opts.hub, r_ap, join(string(rfid_grid), ' '));

% aft launcher ring frame: ring position/radius are STRUCTURE givens,
% but its CLOCKING and its own fiducial-assignment map are free -- the
% aft<->hub truss is solved FIRST as its own coordinate block, frozen
% for the segment sweeps, then revisited once (Dave 2026-07-19:
% "solve the simpler M3-SM truss first").  Frame construction mirrors
% add_met's extra_sources emission exactly (aft_clock = pi/6 == the
% legacy launch_clock default reproduces the as-built ring).
aftf = [];
if has_aft
    [pva, psa, ra] = elt_geom_(seg.in, opts.aft);
    r_a = opts.r_extra;
    if isempty(r_a)
        r_a = ra + opts.edge_off;
        assert(isfinite(r_a), ...
            'met_layout_opt: aft element %d needs r_extra', opts.aft);
    end
    [~, imina] = min(abs(psa));  ea = zeros(3, 1);  ea(imina) = 1;
    xa = cross(psa, ea);  xa = xa/norm(xa);  ya = cross(psa, xa);
    aftf = struct('pv', pva, 'xa', xa, 'ya', ya, 'r', r_a);
end

% true offset boundaries -> tiling-plane polygons about each center
B = macos.design.seg_boundary(seg, opts.edge_off);
c2 = [B.u, B.v].' * ([seg.frames.rpt] - B.c0);
P2 = cell(1, nseg);
for s = 1:nseg
    V = B.poly{s};  V = V(:, 1:end-1);                 % drop closure
    P2{s} = [B.u, B.v].' * (V - B.c0) - c2(:, s);      % center-relative
end

% per-member pattern reference angle in the tiling plane
ref_ang = zeros(1, nseg);
for s = 1:nseg
    if opts.pattern_frame == "segment"
        xs = seg.frames(s).xhat;
        ref_ang(s) = atan2(dot(xs, B.v), dot(xs, B.u));
    else
        ref_ang(s) = atan2(c2(2,s), c2(1,s));
        if norm(c2(:,s)) < 1e-9*max(1, r_ap), ref_ang(s) = 0; end
    end
end

% shape classes by boundary congruence IN THE PATTERN FRAME: members
% whose polar boundary profile r(ref_ang + phi) agrees are one class
% (robust to polygon tessellation -- polybuffer arcs / sliver vertices
% make edge-length signatures split congruent rxpoly wedges)
[classes, class_of] = classify_(P2, ref_ang);
nc = numel(classes);
say('%d shape class(es): %s\n', nc, strjoin(cellfun(@(m) ...
    sprintf('[%s]', num2str(m)), classes, 'uni', 0), ' '));

% per-member clocking offset within its class (for the rotational
% fiducial assignment): dang(s) = member angle - its class's first
% member's angle
dang = zeros(1, nseg);
for s = 1:nseg
    m1 = classes{class_of(s)}(1);
    dang(s) = angdiff_(ref_ang(s), ref_ang(m1));
end
ctx = struct('nseg', nseg, 'nb', nb, 'has_aft', has_aft, 'pv', pv, ...
    'xh', xh, 'yh', yh, 'E', E, 'X', X, 'G', G, 'nw', nw, ...
    'sige', opts.sig_edge, 'sigl', opts.sig_met, 'bodies', bodies, ...
    'ref_ang', ref_ang, 'class_of', class_of, 'min_sep', opts.min_sep, ...
    'aftf', aftf, 'u2m', u2m, 'u', B.u, 'v', B.v, 'c0', B.c0, ...
    'c2', c2, 'sym', opts.sym_assign, 'dang', dang);
ctx.P2 = P2;

% ---- PRESET APPLY MODE: realize a saved layout on THIS build ----------
if apply_mode
    pre = opts.apply;
    cur_nm = cellfun(@numel, classes);
    lay = struct('angs', {cell(1, nc)}, 'family', repmat("preset", 1, nc), ...
        'pmap', pre.pmap, 'nf', pre.nf, 'fclock', pre.fclock, ...
        'rfid', NaN, 'aft_clock', pi/6, 'aft_pmap', pre.pmap);
    if isfield(pre, 'aft_clock'), lay.aft_clock = pre.aft_clock; end
    if isfield(pre, 'aft_pmap'), lay.aft_pmap = pre.aft_pmap; end
    used = false(1, numel(pre.class_nmember));
    for c = 1:nc
        m = find(~used & pre.class_nmember == cur_nm(c), 1);
        assert(~isempty(m), ['met_layout_opt: preset has no class with ' ...
            '%d members to match this build''s class %d'], cur_nm(c), c);
        used(m) = true;
        lay.angs{c} = pre.angs{m};
    end
    assert(isfinite(r_ap), ...
        'met_layout_opt: hub element %d needs ApVec/lMon for rfid_inset', ...
        opts.hub);
    lay.rfid = r_ap - pre.rfid_inset;
    [okp, LPp, srcp] = place_(lay, ctx);
    msp = minsep_(srcp);
    say('preset applied: %d classes matched, rfid %.4g, min-sep %.4g (gate %g, %s)\n', ...
        nc, lay.rfid, msp, opts.min_sep, string(okp));
    pmm = zeros(nseg, 6);
    for s = 1:nseg, pmm(s, :) = seg_pmap_(lay, ctx, s); end
    out = struct('best', lay, 'launch_pts', {LPp}, 'classes', {classes}, ...
        'pmap_per_seg', pmm, 'sym_assign', opts.sym_assign, 'dang', dang, ...
        'class_of', class_of, 'src_aft', place_aft_(lay, ctx), ...
        'minsep', msp, 'feasible', okp, 'rfid_grid', lay.rfid, ...
        'hub_frame', struct('pv', pv, 'xh', xh, 'yh', yh), ...
        'preset', preset_of_(lay, classes, r_ap, seg, opts.pattern_frame));
    return
end

% baseline: spread [30 90 150] every class, Stewart struts, inner
% radius; aft ring at the legacy add_met clocking + the shared map
nf0 = 3;  if ~any(opts.nf_grid == 3), nf0 = opts.nf_grid(1); end
pm0 = mod((0:5), nf0) + 1;  if nf0 == 3, pm0 = [1 2 2 3 3 1]; end
base = struct('angs', {repmat({deg2rad([30 90 150 -30 -90 -150])}, 1, nc)}, ...
    'family', repmat("spread", 1, nc), 'pmap', pm0, 'nf', nf0, ...
    'rfid', rfid_grid(1), 'fclock', 0, ...
    'aft_clock', pi/6, 'aft_pmap', pm0);
[okb, ~, srcb] = place_(base, ctx);
say('baseline launcher min separation %.4g (gate %.4g, %s)\n', ...
    minsep_(srcb), opts.min_sep, string(okb));
[r0, w0m] = metric_(base, ctx);
say('baseline: rms %.3f nm, worst-mode %.3f nm\n', r0*1e9, w0m*1e9);

% ---- per-class candidate list (identical enumeration per class) --------
cands = enumerate_(opts);
say('%d candidate patterns per class (%d classes)\n', numel(cands), nc);

% ---- coordinate-descent sweeps: aft block first, then classes ----------
[~, corder] = sort(cellfun(@numel, classes), 'descend');
best = base;  rb = r0;  wb = w0m;
rank_r = cell(1, nc);  n_eval = 0;  n_feas = 0;
pm6 = [1 2 3 4 5 6; 1 4 2 5 3 6; 2 1 4 3 6 5];
    function aft_block_(label)
        % The aft<->hub leg is the SIMPLE truss: solve its clocking +
        % its own fiducial-assignment map first with the segments held,
        % freeze it for the class sweeps, revisit once after (Dave
        % 2026-07-19).  Evaluated NOGATE (aft params cannot change the
        % segment-launcher separations).
        if ~has_aft, return; end
        tic;
        bra = inf;  acb = best.aft_clock;  apb = best.aft_pmap;
        for ac = deg2rad(0:5:55)                 % 6-fold ring symmetry
            for ap = 1:size(pm6, 1)
                lay = best;
                lay.aft_clock = ac;  lay.aft_pmap = pm6(ap, :);
                for fc = opts.fclk_coarse
                    lay.fclock = fc;
                    r1 = metric_(lay, ctx, true);
                    n_eval = n_eval + 1;
                    if r1 < bra
                        bra = r1;  acb = ac;  apb = pm6(ap, :);
                    end
                end
            end
        end
        best.aft_clock = acb;  best.aft_pmap = apb;
        [rb, wb] = metric_(best, ctx);           % gated refresh
        say('aft block (%s): clock %.0f deg, pmap [%s] (%.1f s; rms %.3f nm nogate)\n', ...
            label, rad2deg(acb), join(string(apb), ' '), toc, bra*1e9);
    end
aft_block_('pre');
nsweep = 1 + (nc > 1);
for sweep = 1:nsweep
    for c = corder(:).'
        tic;
        rc = inf(numel(cands), 1);
        for q = 1:numel(cands)
            lay = best;
            lay.angs{c} = cands{q}.angs;
            lay.family(c) = cands{q}.family;
            lay.pmap = cands{q}.pmap;  lay.nf = cands{q}.nf;
            lay.rfid = rfid_grid(1);
            bf = inf;
            for fc = opts.fclk_coarse
                lay.fclock = fc;
                [r1, w1] = metric_(lay, ctx);  n_eval = n_eval + 1;
                if isfinite(r1), n_feas = n_feas + 1; end
                if r1 < bf, bf = r1; end
                if r1 < rb, best = lay; rb = r1; wb = w1; end
            end
            rc(q) = bf;
        end
        rank_r{c} = rc;
        say('sweep %d class %d: best rms %.3f nm (%.1f s, %d/%d feasible)\n', ...
            sweep, c, rb*1e9, toc, nnz(isfinite(rc)), numel(cands));
    end
end
aft_block_('revisit');

% ---- refinement: shortlist x fiducial grids ----------------------------
tic;
    function consider_(lay)
        for rf = rfid_grid
            for fc = opts.fclk_fine
                lay.rfid = rf;  lay.fclock = fc;
                [r1, w1] = metric_(lay, ctx);  n_eval = n_eval + 1;
                if r1 < rb, best = lay; rb = r1; wb = w1; end
            end
        end
    end
% (a) top-NREFINE single-class variations off the incumbent
for c = corder(:).'
    [~, order] = sort(rank_r{c});
    for q = order(1:min(opts.nrefine, numel(order))).'
        lay = best;
        lay.angs{c} = cands{q}.angs;  lay.family(c) = cands{q}.family;
        lay.pmap = cands{q}.pmap;  lay.nf = cands{q}.nf;
        consider_(lay);
    end
end
% (b) joint top-K cross between the two leading classes (nc > 1)
if nc > 1
    c1 = corder(1);  c2_ = corder(2);
    [~, o1] = sort(rank_r{c1});  [~, o2] = sort(rank_r{c2_});
    for q1 = o1(1:min(opts.topk, numel(o1))).'
        for q2 = o2(1:min(opts.topk, numel(o2))).'
            if cands{q1}.nf ~= cands{q2}.nf, continue; end
            lay = best;
            lay.angs{c1} = cands{q1}.angs;  lay.family(c1) = cands{q1}.family;
            lay.angs{c2_} = cands{q2}.angs;  lay.family(c2_) = cands{q2}.family;
            lay.pmap = cands{q1}.pmap;  lay.nf = cands{q1}.nf;
            consider_(lay);
        end
    end
end
say('refinement done (%.1f s); best rms %.3f nm, worst-mode %.3f nm\n', ...
    toc, rb*1e9, wb*1e9);
if ~isfinite(rb)
    warning('macos:design:met_layout_opt:infeasible', ...
        ['no candidate layout satisfies min_sep=%g -- the returned ' ...
         '"best" is the (infeasible) baseline; widen the pattern grids ' ...
         'or relax min_sep'], opts.min_sep);
end

% ---- named benchmark layouts (manual designs + auto corner-pairs) ------
% Evaluated over the fiducial grids and the canonical assignment maps
% with the min-sep gate BYPASSED: their true minimum separation is
% REPORTED (Dave 2026-07-19 -- corner pairs of adjacent segments sit a
% gap apart, and that hardware-envelope tension is a design datum, not
% a reason to hide the score).  A feasible extra may win outright.
exlist = num2cell(opts.extra_layouts);
if opts.corner_pairs
    exlist{end+1} = corner_layout_(P2, ref_ang, classes, opts.min_sep);
end
xres = struct('name', {}, 'rms', {}, 'worst', {}, 'minsep', {}, ...
              'feasible', {}, 'lay', {});
for q = 1:numel(exlist)
    eq = exlist{q};
    lay = best;                    % inherit the solved aft block etc.
    lay.angs = eq.angs;
    lay.family = repmat("manual", 1, nc);
    lay.nf = max(opts.nf_grid);
    if isfield(eq, 'nf') && ~isempty(eq.nf), lay.nf = eq.nf; end
    pms = pm6;
    if lay.nf ~= 6, pms = mod(pm6 - 1, lay.nf) + 1; end
    if isfield(eq, 'pmap') && ~isempty(eq.pmap), pms = eq.pmap; end
    [~, ~, srcq] = place_(lay, ctx);
    msq = minsep_(srcq);
    bq = inf;  wq = inf;  layq = lay;
    for pi_ = 1:size(pms, 1)
        lay.pmap = pms(pi_, :);
        for rf = rfid_grid
            for fc = opts.fclk_fine
                lay.rfid = rf;  lay.fclock = fc;
                [r1, w1] = metric_(lay, ctx, true);   % gate bypassed
                if r1 < bq, bq = r1; wq = w1; layq = lay; end
            end
        end
    end
    feas = msq >= opts.min_sep;
    ftxt = 'INFEASIBLE';  if feas, ftxt = 'feasible'; end
    xres(end+1) = struct('name', string(eq.name), 'rms', bq, ...
        'worst', wq, 'minsep', msq, 'feasible', feas, 'lay', layq); %#ok<AGROW>
    say('extra "%s": rms %.3f nm, worst %.3f nm, min-sep %.4g (%s vs gate %g)\n', ...
        eq.name, bq*1e9, wq*1e9, msq, ftxt, opts.min_sep);
    if feas && bq < rb, best = layq; rb = bq; wb = wq; end
end
for c = 1:nc
    say('  class %d (%s, %d members): angs [%s] deg\n', c, best.family(c), ...
        numel(classes{c}), join(string(round(rad2deg(best.angs{c}))), ' '));
end
say('  pmap [%s], nf=%d, rfid=%.4g, fclock=%.0f deg\n', ...
    join(string(best.pmap), ' '), best.nf, best.rfid, rad2deg(best.fclock));
if has_aft
    say('  aft ring: clock %.0f deg, pmap [%s]\n', ...
        rad2deg(best.aft_clock), join(string(best.aft_pmap), ' '));
end

[~, LP] = place_(best, ctx);
[~, LPb] = place_(base, ctx);
src_aft = place_aft_(best, ctx);
pmm = zeros(nseg, 6);
for s = 1:nseg, pmm(s, :) = seg_pmap_(best, ctx, s); end
out = struct('best', best, 'launch_pts', {LP}, 'classes', {classes}, ...
    'pmap_per_seg', pmm, 'sym_assign', opts.sym_assign, 'dang', dang, ...
    'class_of', class_of, 'src_aft', src_aft, 'r0', r0, 'w0m', w0m, ...
    'rb', rb, 'wb', wb, 'bodies', bodies, 'unit_to_m', u2m, ...
    'n_layouts', numel(cands), 'n_eval', n_eval, ...
    'feasible_frac', n_feas/max(1, n_eval), 'rfid_grid', rfid_grid, ...
    'base', base, 'base_launch_pts', {LPb}, 'extras', xres, ...
    'hub_frame', struct('pv', pv, 'xh', xh, 'yh', yh), ...
    'preset', preset_of_(best, classes, r_ap, seg, opts.pattern_frame));
% realized launcher points for each extra (reporting/figures)
for q = 1:numel(xres)
    [~, LPx] = place_(xres(q).lay, ctx);
    out.extras(q).launch_pts = LPx;
end
end

% ==========================================================================
function cands = enumerate_(opts)
%ENUMERATE_  Candidate patterns (angs + pmap + nf), e5_seg_metopt v3 space.
cands = {};
spread_pmaps = { ...
    3, [1 2 2 3 3 1; 1 1 2 2 3 3; 1 2 3 1 2 3]; ...
    6, [1 2 3 4 5 6; 1 4 2 5 3 6; 2 1 4 3 6 5]};
if any(opts.families == "spread")
    combos = nchoosek(1:numel(opts.phi_grid), 3);
    for ci = 1:size(combos, 1)
        phis = opts.phi_grid(combos(ci, :));
        for nfi = 1:size(spread_pmaps, 1)
            nf = spread_pmaps{nfi, 1};
            if ~any(opts.nf_grid == nf), continue; end
            pm = spread_pmaps{nfi, 2};
            for pi_ = 1:size(pm, 1)
                cands{end+1} = struct('family', "spread", ...
                    'angs', [phis, -phis], 'pmap', pm(pi_, :), ...
                    'nf', nf); %#ok<AGROW>
            end
        end
    end
end
if any(opts.families == "cluster")
    for c0v = opts.c0_grid
      for psi = opts.psi_grid
        for dl = opts.delta_grid
          h = dl/2;
          angs = [c0v-h, c0v+h, psi-h, psi+h, -psi-h, -psi+h];
          for nf = opts.nf_grid(:).'
            for d = 1:floor(nf/2)
              for o1 = 0:nf-1, for o2 = 0:nf-1, for o3 = 0:nf-1 %#ok<ALIGN>
                pa = [1+o1, 1+mod(o1+d,nf), 1+o2, 1+mod(o2+d,nf), ...
                      1+o3, 1+mod(o3+d,nf)];
                for ord = 0:1
                  if ord, pm = pa([2 1 4 3 6 5]); else, pm = pa; end
                  cands{end+1} = struct('family', "cluster", ...
                      'angs', angs, 'pmap', pm, 'nf', nf); %#ok<AGROW>
                end
              end, end, end
            end
          end
        end
      end
    end
end
end

function [classes, class_of] = classify_(P2, ref_ang)
%CLASSIFY_  Group segments by boundary congruence in the pattern frame:
%   the polar profile r(ref_ang(s) + phi) of member s must match a
%   class reference within 0.1% of its median radius.  This is exactly
%   the invariant pattern replication needs (a shared pattern lands on
%   identical boundary points in every member's frame), and it is
%   robust to tessellation differences between congruent polygons.
n = numel(P2);
phi = deg2rad(0:5:355);
class_of = zeros(1, n);
classes = {};
sigs = {};
for s = 1:n
    r = polar_r_(P2{s}, ref_ang(s) + phi);
    hit = 0;
    for c = 1:numel(sigs)
        if numel(sigs{c}) == numel(r) && ...
                max(abs(r - sigs{c})) < 1e-3*median(sigs{c})
            hit = c;  break;
        end
    end
    if hit == 0
        classes{end+1} = s;  sigs{end+1} = r;  hit = numel(classes); %#ok<AGROW>
    else
        classes{hit}(end+1) = s;
    end
    class_of(s) = hit;
end
end

function pre = preset_of_(lay, classes, r_ap, seg, pattern_frame)
%PRESET_OF_  Scale-free export of a layout: class pattern angles (in
%   the pattern frame), fiducial RIM INSET (not radius), aft block --
%   everything a future same-tiling build needs to realize this truss
%   on its own boundaries via the 'apply' mode.
grid = '';  if isfield(seg, 'grid'), grid = char(seg.grid); end
ri = NaN;  if isfinite(r_ap), ri = r_ap - lay.rfid; end
pre = struct('grid', grid, 'pattern_frame', char(pattern_frame), ...
    'class_nmember', cellfun(@numel, classes), 'angs', {lay.angs}, ...
    'family', lay.family, 'pmap', lay.pmap, 'nf', lay.nf, ...
    'fclock', lay.fclock, 'rfid_inset', ri, ...
    'aft_clock', lay.aft_clock, 'aft_pmap', lay.aft_pmap);
end

function ex = corner_layout_(P2, ref_ang, classes, min_sep)
%CORNER_LAYOUT_  Dave's manual benchmark (2026-07-19): launcher PAIRS
%   at each class's two outermost boundary corners + a third pair on
%   the inside edge (the boundary opposite, 180 deg in the pattern
%   frame).  For a regular class (all corners equal radius, e.g. the
%   center hexagon) the three pairs sit on alternating corners.
%   Within-pair arc sized so the pair itself clears min_sep (x1.1).
phi = deg2rad(-180:1:179);
angs = cell(1, numel(classes));
for c = 1:numel(classes)
    s = classes{c}(1);
    r = polar_r_(P2{s}, ref_ang(s) + phi);
    isMax = r > circshift(r, 1) & r >= circshift(r, -1);
    pk = find(isMax);
    [~, ord] = sort(r(pk), 'descend');
    % greedy angular declustering (arc plateaus produce peak runs)
    sel = [];
    for k = pk(ord)
        if isempty(sel) || all(abs(angdiff_(phi(k), phi(sel))) > deg2rad(15))
            sel(end+1) = k; %#ok<AGROW>
        end
        if numel(sel) >= 6, break; end
    end
    if numel(sel) >= 3 && (max(r(sel)) - min(r(sel(1:min(6, end))))) ...
            < 1e-2*max(r)
        pp = sort(phi(sel));                 % regular: alternate corners
        ctr = pp(1:2:end);  ctr = ctr(1:3);
    else
        ctr = [phi(sel(1)), phi(sel(2)), pi];  % 2 outer corners + inside
    end
    a = zeros(1, 6);
    for k = 1:3
        rc = polar_r_(P2{s}, ref_ang(s) + ctr(k));
        h = min(0.5, 0.55*min_sep/max(rc, eps));
        a((k-1)*2 + (1:2)) = [ctr(k) - h, ctr(k) + h];
    end
    angs{c} = a;
end
ex = struct('name', "corner_pairs", 'angs', {angs});
end

function d = angdiff_(a, b)
d = mod(a - b + pi, 2*pi) - pi;
end

function r = polar_r_(V, phi)
%POLAR_R_  Distance to a convex polygon boundary along angles PHI
%   (V: 2xM open polygon, CENTERED coordinates; origin must be inside).
M = size(V, 2);
d = [cos(phi); sin(phi)];
r = inf(size(phi));
for e = 1:M
    a = V(:, e);  ab = V(:, mod(e, M) + 1) - a;
    den = ab(1)*d(2, :) - ab(2)*d(1, :);
    ok = abs(den) > 1e-14;
    t = (ab(1)*a(2) - ab(2)*a(1)) ./ den;             % along the ray
    s = (d(1, :)*a(2) - d(2, :)*a(1)) ./ den;         % along the edge
    hit = ok & t > 0 & s >= -1e-9 & s <= 1 + 1e-9;
    r(hit) = min(r(hit), t(hit));
end
end

function [ok, LP, src] = place_(lay, c)
%PLACE_  Launcher positions: class pattern about each member's symmetry
%   axis, ON its true offset boundary.
src = zeros(3, 6*c.nseg);
LP = cell(1, c.nseg);
for s = 1:c.nseg
    phi = c.ref_ang(s) + lay.angs{c.class_of(s)};
    r = polar_r_(c.P2{s}, phi);
    p2 = c.c2(:, s) + r.*[cos(phi); sin(phi)];
    P6 = c.c0 + c.u*p2(1, :) + c.v*p2(2, :);
    LP{s} = P6;
    src(:, (s-1)*6 + (1:6)) = P6;
end
ok = minsep_(src) >= c.min_sep;
end

function d = minsep_(src)
n = size(src, 2);
D = squeeze(vecnorm(reshape(src, 3, 1, n) - reshape(src, 3, n, 1)));
D(1:n+1:end) = inf;
d = min(D(:));
end

function pm = seg_pmap_(lay, c, s)
%SEG_PMAP_  Segment s's fiducial indices: the base pmap, shifted by the
%   member's clocking within its class when the rotational (symmetric)
%   assignment is on -- congruent beam geometry per member.
fs = 0;
if c.sym, fs = round(c.dang(s) * lay.nf / (2*pi)); end
pm = mod(lay.pmap - 1 + fs, lay.nf) + 1;
end

function pts = place_aft_(lay, c)
%PLACE_AFT_  Aft launcher ring at the layout's clocking (== add_met's
%   extra_sources emission with extra_clock = lay.aft_clock).
pts = zeros(3, 0);
if ~c.has_aft, return; end
tl6 = lay.aft_clock + 2*pi*(0:5)/6;
pts = c.aftf.pv + c.aftf.r*(c.aftf.xa*cos(tl6) + c.aftf.ya*sin(tl6));
end

function [rms_w, worst] = metric_(lay, c, nogate)
%METRIC_  Post-control wavefront residual (Inf when infeasible, unless
%   NOGATE -- benchmark evaluation reports separation separately).
if nargin < 3, nogate = false; end
[ok, ~, src] = place_(lay, c);
if ~ok && ~nogate, rms_w = inf; worst = inf; return; end
thf = lay.fclock + 2*pi*(0:lay.nf-1)/lay.nf;
fid = c.pv + lay.rfid*(c.xh*cos(thf) + c.yh*sin(thf));
src = [src, place_aft_(lay, c)];
tgt = zeros(3, 6*c.nseg);
for s2 = 1:c.nseg
    tgt(:, (s2-1)*6+(1:6)) = fid(:, seg_pmap_(lay, c, s2));
end
sb = repelem(1:c.nseg, 6);
if c.has_aft
    apm = mod(lay.aft_pmap - 1, lay.nf) + 1;   % stay within 1..nf
    tgt = [tgt, fid(:, apm)];
    sb = [sb, repelem(c.nb, 6)];
end
Hl = macos.design.dldx_analytic(c.bodies, src, tgt, sb, ...
        (c.nseg+1)*ones(1, size(src, 2)), c.u2m);
H = [c.E; Hl];
R = blkdiag(c.sige^2*eye(size(c.E, 1)), c.sigl^2*eye(size(Hl, 1)));
P = c.X - c.X*H'*((H*c.X*H' + R) \ (H*c.X));
rms_w = sqrt(trace(P*c.G)/c.nw);
worst = sqrt(max(real(eig(P*c.G))));
end

function [pv, ps, r_ap] = elt_geom_(in_path, k)
%ELT_GEOM_  VptElt / psiElt / radius of element K from the Rx text.
lines = readlines(in_path);
starts = find(startsWith(strtrim(lines), "iElt="));
b1 = numel(lines);  if k < numel(starts), b1 = starts(k+1) - 1; end
b = strtrim(lines(starts(k):b1));
    function v = g3_(key)
        t = regexp(b(find(startsWith(b, key + "="), 1)), ...
            key + '=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens', 'once');
        v = str2double(string(t))';
    end
    function v = g1_(key)
        i = find(startsWith(b, key + "="), 1);
        v = NaN;
        if ~isempty(i)
            t = regexp(b(i), key + '=\s*(\S+)', 'tokens', 'once');
            v = str2double(string(t));
        end
    end
pv = g3_("VptElt");
ps = g3_("psiElt");  ps = ps/norm(ps);
r_ap = g1_("ApVec");
if ~isfinite(r_ap), r_ap = g1_("lMon"); end
end
