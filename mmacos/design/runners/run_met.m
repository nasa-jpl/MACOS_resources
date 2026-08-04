function art = run_met(seg_in, opts)
%RUN_MET  MET stage runner: segmented .in -> met .in + dedx/dldx + merit.
%
%   art = run_met(SEG_IN, 'hx', HXFILE, 'hub', H, ...) is the
%   metrology stage of the design pipeline
%       design -> segmentation -> sensitivities -> MET -> compare -> simulate
%   (see design/runners/README.md).  It takes the segmentation stage's
%   .in file (+ its SegMirMaker Hx sidecar) and the sensitivities
%   stage's Jacobian .mat, and emits:
%
%     <name>_met.in       the Rx with the AS-BUILT Stewart MET truss
%                         (6 launchers per segment on its true boundary,
%                         nf fiducials on the hub rim, optional aft
%                         launcher ring) -- macos.design.add_met
%     <name>_metopt.in    the same with the OPTIMIZED layout
%                         (macos.design.met_layout_opt: one launcher
%                         pattern per segment SHAPE CLASS, replicated
%                         in the segment frames) when 'optimize' is on
%     <name>_met.mat      dedx, dldx (FD == analytic gated), the merit
%                         table, and the estimator/controller products
%                         dxde, dxdl, dwde, dwdl for the compare/
%                         simulate stages
%     <name>_met_report.txt + layout/metric/view figures
%
%   Forward model on one state vector x (bodies = [segments, hub(, aft)],
%   per body [rot_xyz|trans_xyz] in its LOCAL triad, SI):
%       w = dwdx*x + w0,  e = dedx*x + e0,  l = dldx*x + l0
%   Configuration merit = post-control wavefront residual
%       trace(dwdx*P_dx*dwdx'),  P_dx = X - X*H'*(H*X*H'+R)^-1*H*X,
%       H = [dedx; dldx]
%   (Dave 2026-07-12), cross-checked by a Monte-Carlo estimate/control
%   loop.  The MMSE gain K = X*H'*(H*X*H'+R)^-1 partitions into the
%   named products  dxde = K(:,edge rows),  dxdl = K(:,gauge rows),
%   and  dwde = dwdx*dxde,  dwdl = dwdx*dxdl.
%
%   REQUIRED OPTIONS:
%     'hx'      SegMirMaker Hx.m sidecar (edge-sensor model -> dedx)
%     'hub'     fiducial-carrier element index (e.g. M2)
%   GEOMETRY (in the Rx's BaseUnits):
%     'aft'       aft launcher-ring element ([] = no aft leg).  Pick a
%                 body with LINE OF SIGHT to the hub fiducials (for a
%                 perforated primary: an on-axis bench body behind the
%                 hole -- run_met reports the per-beam hole margin).
%     'r_extra'   aft ring radius ([] = element radius + edge_off)
%     'r_fid'     fiducial ring radius ([] = hub rim - fid_inset)
%     'fid_inset' rim inset (default 25 mm equivalent)
%     'nf'        fiducial count (default 6)
%     'edge_off'  launcher clearance off the segment edge (default
%                 5 mm equivalent)
%     'min_sep'   launcher separation gate (default 50 mm equivalent)
%   MODEL / NOISE (SI):
%     'jac'       sensitivities .mat (dw_dx_multi output 'ox') or the
%                 struct itself ('' = skip merit/optimize/products)
%     'sig_rot','sig_trans'   prior (deploy) sigmas  (1e-6 rad / m)
%     'sig_edge','sig_met'    sensor noise sigmas    (1e-9 m)
%   CONTROL:
%     'optimize'  run met_layout_opt (default true when jac given)
%     'opt'       struct of met_layout_opt overrides (families, grids)
%     'ngridpts'  ray-grid override for the parity traces/figures
%                 ([] = keep the .in value).  The truss Jacobians are
%                 geometric (met points + Hx), so a coarse grid is a
%                 legitimate memory/runtime lever (Dave 2026-07-19).
%     'model_size' engine model (default 512), 'mc' MC draws (200),
%     'out_dir'   artifact directory (default: beside SEG_IN),
%     'name'      artifact basename (default: SEG_IN's basename),
%     'visible'   show figures (false), 'verbose' (true)
%
%   art: artifact paths + all matrices (see the save list at the end).
%
%   See also: macos.design.add_met, macos.design.met_layout_opt,
%             macos.design.edge_sensors, macos.design.dmet_dx,
%             macos.design.seg_from_rx, macos.design.met_view.

arguments
    seg_in (1,1) string
    opts.hx (1,1) string
    opts.hub (1,1) double {mustBeInteger, mustBePositive}
    opts.aft double = []
    opts.r_extra double = []
    opts.r_fid double = []
    opts.fid_inset double = []
    opts.nf (1,1) double {mustBeMember(opts.nf, [3 4 5 6])} = 6
    opts.edge_off double = []
    opts.min_sep double = []
    opts.jac = ''
    opts.sig_rot (1,1) double {mustBePositive} = 1e-6
    opts.sig_trans (1,1) double {mustBePositive} = 1e-6
    opts.sig_edge (1,1) double {mustBePositive} = 1e-9
    opts.sig_met (1,1) double {mustBePositive} = 1e-9
    opts.optimize (1,1) logical = true
    opts.optimize_with_edge (1,1) logical = false  % Dave 2026-07-19:
                                % the laser truss must stand alone --
                                % edge sensors are NOT in the optimizer
                                % merit (they still appear in the
                                % reported sensing cases and products)
    opts.opt (1,1) struct = struct()
    opts.hole_r double = []     % effective central-hole radius for the
                                % aft-beam clearance check ([] = the
                                % declared center-segment obscuration).
                                % Use when the design intent sizes the
                                % hole differently (e.g. to the SM
                                % shadow radius -- Dave 2026-07-19).
    opts.preset = []            % saved layout preset (struct, or path
                                % to a <name>_met_preset.mat): the
                                % AS-BUILT truss is realized from the
                                % preset's class patterns instead of
                                % the default boundary ring ("save the
                                % optimized configuration as the new
                                % as-built for future PIE builds" --
                                % Dave 2026-07-19).  Every optimized
                                % run writes <name>_met_preset.mat.
    opts.opt_preset = []        % LAYOUT-HELD REGEN (no search): apply a
                                % saved preset AS the optimizer winner --
                                % emit <name>_metopt.in, the optimized
                                % products (dldx_opt/dxde/dwde), and the
                                % [4b] "optimized" row from the applied
                                % layout, WITHOUT running met_layout_opt.
                                % Use to refresh the committed products
                                % against a new Jacobian generation while
                                % holding the reviewed layout exactly
                                % (mutually exclusive with 'optimize').
                                % Distinct from 'preset' (which makes the
                                % AS-BUILT be the preset); here the
                                % as-built stays the default boundary ring.
    opts.ngridpts double = []
    opts.model_size (1,1) double = 512
    opts.mc (1,1) double {mustBeInteger, mustBeNonnegative} = 200
    opts.out_dir (1,1) string = ""
    opts.name (1,1) string = ""
    opts.visible (1,1) logical = false
    opts.verbose (1,1) logical = true
end
assert(isfile(seg_in), 'run_met: %s not found', seg_in);
assert(isfile(opts.hx), 'run_met: Hx sidecar %s not found', opts.hx);
% opt_preset (layout-held regen) and optimize (fresh search) are mutually
% exclusive: opt_preset realizes a fixed winner, optimize searches for one.
if ~isempty(opts.opt_preset) && opts.optimize
    error('run_met:optPreset', ...
        ['opt_preset (hold a saved layout) and optimize (search for a ' ...
         'new one) are mutually exclusive -- set optimize=false to hold ' ...
         'the preset layout.']);
end
[ind, base] = fileparts(seg_in);
out_dir = opts.out_dir;  if strlength(out_dir) == 0, out_dir = ind; end
name = opts.name;        if strlength(name) == 0,   name = base; end
pth = @(suffix) char(fullfile(out_dir, name + suffix));
log_ = fopen(pth("_met_report.txt"), 'w');
closer = onCleanup(@() fclose(log_));
if opts.verbose
    say = @(varargin) fprintf(1, varargin{:}) + fprintf(log_, varargin{:});
else
    say = @(varargin) fprintf(log_, varargin{:});
end
say('==== run_met: %s ====\n', seg_in);

% engine up; artifact dir is the working dir (GridFile= resolves there)
old = cd(out_dir);  restore = onCleanup(@() cd(old));
macos.init(opts.model_size);
seg = macos.design.seg_from_rx(seg_in, 'hx', opts.hx);
if ~isempty(opts.ngridpts), macos.set_src_sampling(opts.ngridpts); end
nseg = seg.nseg;
cbm = mmacos('base_unit_to_metres');
mm = 1e-3/cbm;                             % "one millimetre" in BaseUnits
edge_off = opts.edge_off;  if isempty(edge_off), edge_off = 5*mm; end
min_sep = opts.min_sep;    if isempty(min_sep),  min_sep = 50*mm; end
fid_inset = opts.fid_inset;
if isempty(fid_inset), fid_inset = 25*mm; end
r_fid = opts.r_fid;
if isempty(r_fid)
    hi = macos.get_elt_info(opts.hub);
    r_ap = hi.lmon;
    if hi.ap_type == 1 && hi.ap_vec(1) > 0, r_ap = hi.ap_vec(1); end
    r_fid = r_ap - fid_inset;
end
t0 = macos.trace();
r0i = macos.get_ray_info(t0.nRays);
p0 = nnz(logical(r0i.ok_pass) & logical(r0i.ok_trace));
say('[0] %s: %d segments (%s tiling), %d elements; %d/%d rays pass\n', ...
    name, nseg, seg.grid, seg.n_elt, p0, t0.nRays);

%% -- [1] as-built truss ----------------------------------------------
belts = [seg.seg_elts, opts.hub, opts.aft];
nb = numel(belts);
if ~isempty(opts.preset)
    pre = opts.preset;
    if ischar(pre) || isstring(pre)
        Sp = load(pre);  pre = Sp.preset;
    end
    ap = macos.design.met_layout_opt(seg, 'apply', pre, ...
        'hub', opts.hub, 'aft', opts.aft, 'r_extra', opts.r_extra, ...
        'edge_off', edge_off, 'min_sep', min_sep, ...
        'verbose', opts.verbose);
    am = macos.design.add_met(char(seg_in), seg, 'hub', opts.hub, ...
        'r_fid', ap.best.rfid, 'nf', ap.best.nf, ...
        'fid_clock', ap.best.fclock, ...
        'launch_pts', ap.launch_pts, 'pair_map', ap.pmap_per_seg, ...
        'extra_sources', opts.aft, 'r_extra', opts.r_extra, ...
        'extra_clock', ap.best.aft_clock, ...
        'extra_pair_map', mod(ap.best.aft_pmap - 1, ap.best.nf) + 1, ...
        'edge_off', edge_off, 'out_in', pth("_met.in"));
    say('    (as-built realized from a saved %s-tiling preset)\n', pre.grid);
else
    am = macos.design.add_met(char(seg_in), seg, 'hub', opts.hub, ...
        'r_fid', r_fid, 'nf', opts.nf, 'extra_sources', opts.aft, ...
        'r_extra', opts.r_extra, 'edge_off', edge_off, ...
        'out_in', pth("_met.in"));
end
afttxt = '';
if ~isempty(opts.aft), afttxt = sprintf(' + aft ring on elt %d', opts.aft); end
say('\n[1] as-built truss: %d beams (6 launchers/segment%s), %d fiducials at r=%.4g on elt %d\n', ...
    am.n_beams, afttxt, opts.nf, r_fid, opts.hub);
macos.load_rx(am.in);
if ~isempty(opts.ngridpts), macos.set_src_sampling(opts.ngridpts); end
t1 = macos.trace();
r1i = macos.get_ray_info(t1.nRays);
p1 = nnz(logical(r1i.ok_pass) & logical(r1i.ok_trace));
say('    trace parity: %d/%d rays pass (bare %d/%d), rmsWFE %.4g (bare %.4g)\n', ...
    p1, t1.nRays, p0, t0.nRays, t1.rmsWFE, t0.rmsWFE);
% aft beams through a perforated primary: per-beam hole margin
hole = hole_margins_(seg, am, opts.aft, opts.hole_r);
if ~isempty(hole)
    say('    aft-beam hole clearance (effective hole r=%.4g%s):\n', ...
        hole.r_hole, hole.src);
    say('      crossing radii [%s] -> min margin %.4g (%s)\n', ...
        join(string(round(hole.r_cross, 4)), ' '), hole.margin, ...
        iif_(hole.margin > 0, 'CLEAR', 'BLOCKED'));
    if hole.margin <= 0
        warning('run_met:hole', ...
            'aft beams cross the primary OUTSIDE the hole -- no line of sight');
    end
end

%% -- [2] sensing Jacobians -------------------------------------------
% dedx: SegMirMaker Hx, readings in BaseUnits -> SI (rot cols x cbm;
% translation cols are BaseUnits/BaseUnits = dimensionless)
es = macos.design.edge_sensors(opts.hx);
dedx = zeros(es.nmeas, 6*nb);
dedx(:, 1:6*nseg) = es.dedx;
for s = 1:nseg
    dedx(:, (s-1)*6+(1:3)) = dedx(:, (s-1)*6+(1:3)) * cbm;
end
% dldx: engine FD vs the analytic rows -- the correctness gate
dm = macos.design.dmet_dx(belts);
dldx = dm.dldx;
bodies = macos.design.met_bodies(belts);
sb = repelem(1:nseg, 6);
if ~isempty(opts.aft), sb = [sb, repelem(nb, 6)]; end
dla = macos.design.dldx_analytic(bodies, am.src_pts, am.tgt_pts, ...
    sb, (nseg+1)*ones(1, am.n_beams), cbm);
scale = max(abs(dldx(:)));
gate = max(abs(dldx - dla), [], 'all') / scale;
say('\n[2] dedx %dx%d (SI);  dldx %dx%d engine-FD vs analytic: %.2e max rel  %s\n', ...
    es.nmeas, 6*nb, dm.nmeas, 6*nb, gate, iif_(gate < 5e-3, 'PASS', 'FAIL'));
if gate >= 5e-3
    warning('run_met:gate', ...
        'dldx FD/analytic disagree (%.2e) -- frames or units are off', gate);
end

%% -- [3] merit + estimator products ----------------------------------
X  = diag(repmat([opts.sig_rot^2*ones(1,3), opts.sig_trans^2*ones(1,3)], 1, nb));
Re = opts.sig_edge^2 * eye(es.nmeas);
Rl = opts.sig_met^2  * eye(dm.nmeas);
art = struct('met_in', am.in, 'metopt_in', '', 'mat', pth("_met.mat"), ...
    'report', pth("_met_report.txt"), 'figs', {{}});
D = [];  keep = [];  optout = [];  merits = [];  mcres = [];
K = [];  dxde = [];  dxdl = [];  dwde = [];  dwdl = [];  rfd = NaN;
ox = jac_struct_(opts.jac);
if ~isempty(ox)
    [D, missing, colidx] = dwdx_cols_(ox, belts);
    say('\n[3] dwdx: %d columns for bodies [%s] from the sensitivities .mat\n', ...
        size(D, 2), num2str(belts));
    if ~isempty(missing)
        error('run_met:jac', 'missing dwdx channels for elements: %s', ...
            num2str(missing));
    end
    keep = all(isfinite(D), 2);
    Dk = D(keep, :);  nw = nnz(keep);
    G = Dk'*Dk;                        % 54x54 -- NEVER form Dk*P*Dk'
    rmsw = @(P) sqrt(trace(P*G) / nw); % (Nw x Nw dense = tens of GB)
    post = @(H, R) X - X*H'*((H*X*H' + R) \ (H*X));
    merits = struct();
    merits.prior = rmsw(X);
    merits.edge  = rmsw(post(dedx, Re));
    merits.met   = rmsw(post(dldx, Rl));
    H = [dedx; dldx];  R = blkdiag(Re, Rl);
    Pab = post(H, R);
    merits.edge_met = rmsw(Pab);
    % HEADLINE NUMBERS ARE DIMENSIONLESS (Dave 2026-07-19: absolute nm
    % at arbitrary priors invites panic; MET tracks the post-WFSC drift
    % state, and its accuracy roadmap runs to picometres).  Two
    % configuration properties, both invariant to the scenario scales:
    %   WEM   = noise-scaled residual per unit sig_met, suite noise
    %           RATIOS held (Wavefront Error Multiplier).  Multiply by
    %           your gauge noise to forecast the sensing-limited
    %           wavefront.
    %   floor = prior-saturated residual as a FRACTION of the prior
    %           (directions H cannot see; better sensors do not help).
    % Each is evaluated on the FULL wavefront AND on the NON-TILT
    % wavefront (per-field piston+tip/tilt projected out -- tilt
    % control absorbs tilt, so non-tilt is the design driver).
    if isfield(ox, 'per_field_dwdx') && isfield(ox, 'per_field_w_nom_2d')
        [Gdt, Gt, nwdt] = detilt_gram_(ox, colidx);
    else
        warning('run_met:detilt', ...
            ['jac lacks per-field blocks -- tilt/non-tilt split ' ...
             'unavailable, using the full Gram for both']);
        Gdt = G;  Gt = zeros(size(G));  nwdt = nw;
    end
    prior_dt = sqrt(trace(X*Gdt)/nwdt);
    prior_t  = sqrt(trace(X*Gt)/nwdt);
    % per-config metric rows: WEM full / tilt / non-tilt + non-tilt
    % floor fraction (each metric from the two-point decomposition on
    % its own Gram; full^2 = tilt^2 + non-tilt^2 by orthogonality)
    wemrow_ = @(Hx, Rx) [ ...
        wem_pair_(X, Hx, Rx, G,   nw,   opts.sig_met), ...
        wem_pair_(X, Hx, Rx, Gt,  nwdt, opts.sig_met), ...
        wem_pair_(X, Hx, Rx, Gdt, nwdt, opts.sig_met)];
    flr_ = @(Hx, Rx) deal_flr_(X, Hx, Rx, Gdt, nwdt, opts.sig_met, prior_dt);
    say('    post-control wavefront residual (fraction of prior; nm at the\n');
    say('    configured scenario: prior %.2g rad / %.2g m, noise %.2g / %.2g m):\n', ...
        opts.sig_rot, opts.sig_trans, opts.sig_edge, opts.sig_met);
    say('      %-20s %10.4f   (%.4g nm)\n', 'prior (no sensing)', 1, merits.prior*1e9);
    say('      %-20s %10.4f   (%.4g nm)\n', 'edge sensors only', ...
        merits.edge/merits.prior, merits.edge*1e9);
    say('      %-20s %10.4f   (%.4g nm)\n', 'MET only', ...
        merits.met/merits.prior, merits.met*1e9);
    say('      %-20s %10.4f   (%.4g nm)\n', 'edge + MET', ...
        merits.edge_met/merits.prior, merits.edge_met*1e9);
    say('    as-built WEM (per unit gauge noise, suite ratios held;\n');
    say('    prior split: %.4g nm tilt / %.4g nm non-tilt):\n', ...
        prior_t*1e9, prior_dt*1e9);
    say('      %-12s %9s %9s %9s %12s\n', 'config', 'WEM_full', ...
        'WEM_tilt', 'WEM_ntilt', 'floor_ntilt');
    wm = wemrow_(dldx, Rl);
    say('      %-12s %9.3f %9.3f %9.3f %12.4f\n', 'MET only', ...
        wm(1), wm(2), wm(3), flr_(dldx, Rl));
    wa = wemrow_(H, R);
    say('      %-12s %9.3f %9.3f %9.3f %12.4f\n', 'edge+MET', ...
        wa(1), wa(2), wa(3), flr_(H, R));
    merits.wem = wa(1);  merits.wem_tilt = wa(2);  merits.wem_dt = wa(3);
    merits.wem_met = wm;
    merits.floor_frac_dt = flr_(H, R);
    sv = svd(H*sqrt(X));
    say('    observability: H %dx%d, sv(H*sqrt(X)) %.3e .. %.3e (cond %.2e)\n', ...
        size(H, 1), size(H, 2), sv(1), sv(end), sv(1)/max(sv(end), eps));
    % estimator gains + the named products (Dave's stage-5 list)
    K = (X*H') / (H*X*H' + R);
    n_e = size(dedx, 1);
    dxde = K(:, 1:n_e);  dxdl = K(:, n_e+1:end);
    dwde = Dk*dxde;     dwdl = Dk*dxdl;
    say('    products: dxde %dx%d, dxdl %dx%d, dwde %dx%d, dwdl %dx%d\n', ...
        size(dxde), size(dxdl), size(dwde), size(dwdl));
end

%% -- [4] layout optimization (shape-class patterns) ------------------
% Merit = MET-ONLY (the truss must stand alone -- edge sensors excluded
% unless 'optimize_with_edge'), on the NON-TILT wavefront Gram (tilt
% control absorbs tilt).  Dave 2026-07-19.
if opts.optimize && ~isempty(D)
    etxt = 'MET-only';  E_opt = zeros(0, size(dedx, 2));
    if opts.optimize_with_edge, etxt = 'edge+MET';  E_opt = dedx; end
    say('\n[4] met_layout_opt (one pattern per shape class; %s, NON-TILT merit):\n', etxt);
    nv = [fieldnames(opts.opt), struct2cell(opts.opt)].';
    optout = macos.design.met_layout_opt(seg, D, E_opt, X, ...
        'hub', opts.hub, 'aft', opts.aft, 'r_extra', opts.r_extra, ...
        'sig_edge', opts.sig_edge, 'sig_met', opts.sig_met, ...
        'edge_off', edge_off, 'min_sep', min_sep, ...
        'fid_inset', fid_inset, 'unit_to_m', cbm, ...
        'gram', Gdt, 'nw', nwdt, ...
        'verbose', opts.verbose, nv{:});
    if ~isfinite(optout.rb)
        say(['    NO feasible layout under min_sep=%g -- keeping the ' ...
             'as-built truss (widen grids or relax min_sep)\n'], min_sep);
    end
elseif ~isempty(opts.opt_preset) && ~isempty(D)
    % LAYOUT-HELD REGEN: realize the saved preset AS the optimizer winner
    % (no search).  Apply-mode returns the same optout shape the search
    % does (best/launch_pts/classes/pmap_per_seg) minus the merit rb; the
    % engine-FD validation block below computes rb honestly on the realized
    % layout, so we seed a finite placeholder purely to pass the gate.
    etxt = 'MET-only';  E_opt = zeros(0, size(dedx, 2));
    if opts.optimize_with_edge, E_opt = dedx; end
    pre = opts.opt_preset;
    if ischar(pre) || isstring(pre), Sp = load(pre); pre = Sp.preset; end
    say('\n[4] layout held from preset (no search; %s merit for the report):\n', etxt);
    optout = macos.design.met_layout_opt(seg, D, E_opt, X, ...
        'apply', pre, 'hub', opts.hub, 'aft', opts.aft, ...
        'r_extra', opts.r_extra, 'sig_edge', opts.sig_edge, ...
        'sig_met', opts.sig_met, 'edge_off', edge_off, ...
        'min_sep', min_sep, 'fid_inset', fid_inset, 'unit_to_m', cbm, ...
        'gram', Gdt, 'nw', nwdt, 'verbose', opts.verbose);
    if ~isfield(optout, 'rb') || isempty(optout.rb) || ~isfinite(optout.rb)
        optout.rb = 0;   % placeholder; the real merit is rfd, computed below
    end
end
if ~isempty(optout) && isfinite(optout.rb)
    for c = 1:numel(optout.classes)
        say('    class %d (%d members, %s): angs [%s] deg\n', c, ...
            numel(optout.classes{c}), optout.best.family(c), ...
            join(string(round(rad2deg(optout.best.angs{c}))), ' '));
    end
    am2 = macos.design.add_met(char(seg_in), seg, 'hub', opts.hub, ...
        'r_fid', optout.best.rfid, 'nf', optout.best.nf, ...
        'fid_clock', optout.best.fclock, ...
        'launch_pts', optout.launch_pts, 'pair_map', optout.pmap_per_seg, ...
        'extra_sources', opts.aft, 'r_extra', opts.r_extra, ...
        'extra_clock', optout.best.aft_clock, ...
        'extra_pair_map', mod(optout.best.aft_pmap - 1, optout.best.nf) + 1, ...
        'edge_off', edge_off, 'out_in', pth("_metopt.in"));
    art.metopt_in = am2.in;
    % engine-FD validation of the winner ON ITS OWN MERIT (MET-only or
    % edge+MET per the optimizer setting, non-tilt Gram)
    macos.load_rx(am2.in);
    if ~isempty(opts.ngridpts), macos.set_src_sampling(opts.ngridpts); end
    macos.trace();
    dm2 = macos.design.dmet_dx(belts);
    Hv = [E_opt; dm2.dldx];
    Rv = blkdiag(opts.sig_edge^2*eye(size(E_opt, 1)), ...
                 opts.sig_met^2*eye(size(dm2.dldx, 1)));
    Pv = X - X*Hv'*((Hv*X*Hv' + Rv) \ (Hv*X));
    rfd = sqrt(trace(Pv*Gdt)/nwdt);
    if ~isempty(opts.opt_preset)
        % held-from-preset: no search merit to compare against; rfd IS the
        % realized layout's merit.  Adopt it as optout.rb so downstream
        % labels/figures read a real number, not the placeholder.
        optout.rb = rfd;
        say('    engine-FD merit (non-tilt, layout held from preset): %.4g nm\n', ...
            rfd*1e9);
    else
        say('    engine-FD validation (non-tilt merit): %.4g nm (analytic %.4g, %.2f%% off)\n', ...
            rfd*1e9, optout.rb*1e9, 100*abs(rfd - optout.rb)/max(optout.rb, eps));
    end
    % winner products (edge+MET estimator -- exclusion was only for the
    % LAYOUT merit) supersede the as-built ones for downstream stages
    H2 = [dedx; dm2.dldx];
    R2 = blkdiag(Re, opts.sig_met^2*eye(size(dm2.dldx, 1)));
    K = (X*H2') / (H2*X*H2' + R2);
    n_e = size(dedx, 1);
    dxde = K(:, 1:n_e);  dxdl = K(:, n_e+1:end);
    dwde = Dk*dxde;     dwdl = Dk*dxdl;
    dldx_opt = dm2.dldx;
else
    dldx_opt = dldx;
    if opts.optimize && isempty(optout)
        say('\n[4] optimization SKIPPED (no sensitivities .mat given)\n');
    end
end

%% -- [4b] configuration comparison table ------------------------------
% Every realized/benchmark layout, each scored on the full / tilt /
% non-tilt wavefront, MET-only and edge+MET (Dave 2026-07-19).
if ~isempty(D)
    rows = {'as-built', dldx, NaN};
    if ~isempty(optout) && isfinite(optout.rb)
        rows(end+1, :) = {'optimized', dm2.dldx, NaN};
    end
    if ~isempty(optout) && isfield(optout, 'extras')
        % benchmark layouts exist only from a search; a held/applied
        % preset (opt_preset) returns no 'extras'.
        for q = 1:numel(optout.extras)
            xq = optout.extras(q);
            rows(end+1, :) = {char(xq.name), ...
                dldx_layout_(xq.lay, xq.launch_pts, optout, belts, ...
                             opts.aft, nseg, cbm), xq.minsep}; %#ok<AGROW>
        end
    end
    say('\n[4b] configuration comparison (WEM per unit gauge noise):\n');
    say('      %-14s | %8s %8s %8s | %8s %8s %8s | %9s %8s\n', ...
        'config', 'MET_full', 'MET_tilt', 'MET_nt', 'e+M_full', ...
        'e+M_tilt', 'e+M_nt', 'flr_nt', 'min_sep');
    cmp = struct('name', {}, 'wem_met', {}, 'wem_all', {}, ...
                 'floor_frac_nt', {}, 'minsep', {});
    for q = 1:size(rows, 1)
        dl = rows{q, 2};
        Rm = opts.sig_met^2*eye(size(dl, 1));
        wm = wemrow_(dl, Rm);
        wa = wemrow_([dedx; dl], blkdiag(Re, Rm));
        ff = flr_([dedx; dl], blkdiag(Re, Rm));
        mstxt = '';
        if isfinite(rows{q, 3}), mstxt = sprintf('%8.4g', rows{q, 3}); end
        say('      %-14s | %8.3f %8.3f %8.3f | %8.3f %8.3f %8.3f | %9.4f %s\n', ...
            rows{q, 1}, wm(1), wm(2), wm(3), wa(1), wa(2), wa(3), ff, mstxt);
        cmp(end+1) = struct('name', rows{q, 1}, 'wem_met', wm, ...
            'wem_all', wa, 'floor_frac_nt', ff, 'minsep', rows{q, 3}); %#ok<AGROW>
    end
    merits.cmp = cmp;
end

%% -- [5] Monte-Carlo acceptance --------------------------------------
if ~isempty(D) && opts.mc > 0
    Huse = [dedx; dldx_opt];
    Ruse = blkdiag(Re, opts.sig_met^2*eye(size(dldx_opt, 1)));
    Kuse = (X*Huse') / (Huse*X*Huse' + Ruse);
    Dk = D(keep, :);
    Gm = Dk'*Dk;
    ana = sqrt(trace((X - Kuse*Huse*X)*Gm)/nnz(keep));
    rng(1);
    acc = 0;
    for q = 1:opts.mc
        x = sqrt(diag(X)) .* randn(6*nb, 1);
        y = Huse*x + sqrt(diag(Ruse)) .* randn(size(Huse, 1), 1);
        wpost = Dk*(x - Kuse*y);
        acc = acc + mean(wpost.^2);
    end
    mcres = sqrt(acc/opts.mc);
    say('\n[5] Monte-Carlo (%d draws): %.3f nm rms  (analytic %.3f) [%.1f%%]\n', ...
        opts.mc, mcres*1e9, ana*1e9, 100*abs(mcres - ana)/ana);
end

%% -- [6] figures ------------------------------------------------------
% edge-sensor points (gray dots): shared-edge midpoints of the SMM Hx
% measurement pairs -- placement is SegMirMaker's, never optimized here
sen = zeros(3, 0);
for q = 2:size(es.meas_to_seg, 2)          % row 1 = master piston
    ij = es.meas_to_seg(:, q);
    if ij(1) ~= ij(2)
        sen(:, end+1) = (seg.frames(ij(1)).rpt + seg.frames(ij(2)).rpt)/2; %#ok<AGROW>
    end
end
figs = {};
fv = macos.design.met_view(seg, am, 'visible', opts.visible, ...
    'edge_off', edge_off, 'sensor_pts', sen, ...
    'title', sprintf('%s as-built MET (%d beams; gray dots = edge sensors)', ...
    name, am.n_beams), 'save', pth("_met_layout.png"));
close(fv);  figs{end+1} = pth("_met_layout.png");
if ~isempty(optout)
    ttl = sprintf('%s optimized MET (open = as-built)', name);
    if isfield(merits, 'cmp') && numel(merits.cmp) >= 2
        ttl = sprintf(['%s optimized MET: non-tilt WEM %.2f -> %.2f ' ...
            '(MET-only; open = as-built)'], name, ...
            merits.cmp(1).wem_met(3), merits.cmp(2).wem_met(3));
    end
    fv = macos.design.met_view(seg, am2, 'visible', opts.visible, ...
        'edge_off', edge_off, 'overlay_pts', [am.src_pts(:, 1:6*nseg)], ...
        'sensor_pts', sen, 'title', ttl, 'save', pth("_metopt_layout.png"));
    close(fv);  figs{end+1} = pth("_metopt_layout.png");
end
macos.load_rx(am.in);
if ~isempty(opts.ngridpts), macos.set_src_sampling(opts.ngridpts); end
macos.trace();
fg = macos.view_rx('visible', opts.visible, 'show', 'beam+met', ...
    'save', pth("_met_view_rx.png"));
close(fg);  figs{end+1} = pth("_met_view_rx.png");
if ~isempty(merits)
    f = figure('Visible', tf_(opts.visible));
    vals = [1, merits.edge, merits.met, merits.edge_met]/1;
    vals(2:4) = vals(2:4)/merits.prior;
    labs = {'prior', 'edge only', 'MET only', 'edge+MET'};
    if ~isempty(optout) && isfinite(optout.rb)
        Rm2 = opts.sig_met^2*eye(size(dldx_opt, 1));
        vals(end+1) = rmsw(post([dedx; dldx_opt], blkdiag(Re, Rm2))) ...
            / merits.prior;
        labs{end+1} = 'edge+MET opt';
    end
    bar(vals);  set(gca, 'XTickLabel', labs, 'YScale', 'log');
    ylabel('w_{post} / w_{prior}  (dimensionless)');  grid on;
    title(sprintf('%s MET metric: WEM %.2f (%d segs, %d edge + %d MET meas)', ...
        name, merits.wem, nseg, es.nmeas, dm.nmeas), 'Interpreter', 'none');
    print(f, pth("_met_metric.png"), '-dpng', '-r120');  close(f);
    figs{end+1} = pth("_met_metric.png");
end
art.figs = figs;
say('\n[6] figures: %s\n', strjoin(string(figs), ', '));

%% -- [7] save ---------------------------------------------------------
art.dedx = dedx;  art.dldx = dldx;  art.dldx_opt = dldx_opt;
art.dwdx = D;  art.keep = keep;  art.X = X;  art.Re = Re;  art.Rl = Rl;
art.K = K;  art.dxde = dxde;  art.dxdl = dxdl;
art.dwde = dwde;  art.dwdl = dwdl;
art.merits = merits;  art.opt = optout;  art.mc = mcres;  art.rfd = rfd;
art.bodies = belts;  art.gate = gate;  art.hole = hole;
art.seg = seg;  art.am = am;
if ~isempty(optout) && isfinite(optout.rb)
    preset = optout.preset;                 %#ok<NASGU> % future as-built
    art.preset = optout.preset;
    art.preset_file = pth("_met_preset.mat");
    save(art.preset_file, 'preset');
    say('preset saved: %s (reusable as-built for future %s builds)\n', ...
        art.preset_file, optout.preset.grid);
end
save(art.mat, '-struct', 'art', '-v7.3');
opttxt = '';
if ~isempty(optout), opttxt = sprintf(', %s_metopt.in', name); end
say('\nartifacts: %s_met.in%s, %s_met.mat, this report + figures\n', ...
    name, opttxt, name);
say('next stage: run_compare (linear model vs mmacos on x/z/grid states).\n');
end

% =========================================================================
function s = tf_(v)
s = 'off';  if v, s = 'on'; end
end

function s = iif_(cond, a, b)
s = b;  if cond, s = a; end
end

function ff = deal_flr_(X, H, R, G, nw, sig_ref, prior)
[~, flr] = wem_pair_(X, H, R, G, nw, sig_ref);
ff = flr / prior;
end

function dl = dldx_layout_(lay, LP, optout, belts, aft, nseg, cbm)
%DLDX_LAYOUT_  Analytic gauge Jacobian of a benchmark layout (validated
%   == engine FD by the [2] gate; benchmark extras are scored without
%   being realized as .in files).
thf = lay.fclock + 2*pi*(0:lay.nf-1)/lay.nf;
hf = optout.hub_frame;
fid = hf.pv + lay.rfid*(hf.xh*cos(thf) + hf.yh*sin(thf));
src = [LP{:}];
sb = repelem(1:nseg, 6);
tgt = zeros(3, 6*nseg);
for s = 1:nseg
    fs = 0;
    if optout.sym_assign
        fs = round(optout.dang(s) * lay.nf / (2*pi));
    end
    tgt(:, (s-1)*6+(1:6)) = fid(:, mod(lay.pmap - 1 + fs, lay.nf) + 1);
end
if ~isempty(aft)
    src = [src, optout.src_aft];
    sb = [sb, repelem(numel(belts), 6)];
    apm = mod(lay.aft_pmap - 1, lay.nf) + 1;
    tgt = [tgt, fid(:, apm)];
end
dl = macos.design.dldx_analytic(optout.bodies, src, tgt, sb, ...
        (nseg+1)*ones(1, size(src, 2)), cbm);
end

function [wem, flr, r] = wem_pair_(X, H, R, G, nw, sig_ref)
%WEM_PAIR_  Two-point WEM/floor decomposition on a given wavefront Gram:
%   r^2 = flr^2 + (wem*sig_ref)^2, from evaluating the posterior at R
%   and R/100 (all sensor noises scaled together, ratios held).
post = @(Rq) X - X*H'*((H*X*H' + Rq) \ (H*X));
r   = sqrt(trace(post(R)*G)/nw);
r10 = sqrt(trace(post(R/100)*G)/nw);
wem = sqrt(max(0, (r^2 - r10^2)*100/99)) / sig_ref;
flr = sqrt(max(0, (100*r10^2 - r^2)/99));
end

function ox = jac_struct_(jac)
%JAC_STRUCT_  Accept a dw_dx_multi output struct or a .mat holding one.
ox = [];
if isstruct(jac), ox = jac; return; end
if (ischar(jac) || isstring(jac)) && strlength(string(jac)) > 0
    assert(isfile(jac), 'run_met: sensitivities .mat %s not found', jac);
    S = load(jac);
    fn = fieldnames(S);
    for k = 1:numel(fn)
        v = S.(fn{k});
        if isstruct(v) && isfield(v, 'dwdxall') && isfield(v, 'channel_names')
            % rigid-body channel set = names like 'Elt 3 Ry'
            if any(~cellfun('isempty', regexp(v.channel_names, ...
                    '^Elt \d+ [RT][xyz]$', 'once')))
                ox = v;  return;
            end
        end
    end
    error('run_met:jac', ...
        'no dw_dx_multi rigid-body output found in %s', jac);
end
end

function [D, missing, idx] = dwdx_cols_(ox, belts)
%DWDX_COLS_  Extract the sensed bodies' 6-DOF columns in [rot|trans] order.
cn = ox.channel_names;
dofs = {'Rx', 'Ry', 'Rz', 'Tx', 'Ty', 'Tz'};
D = zeros(size(ox.dwdxall, 1), 6*numel(belts));
idx = zeros(1, 6*numel(belts));
missing = [];
for b = 1:numel(belts)
    for d = 1:6
        i = find(strcmp(cn, sprintf('Elt %d %s', belts(b), dofs{d})), 1);
        if isempty(i)
            missing(end+1) = belts(b); %#ok<AGROW>
            break
        end
        D(:, (b-1)*6 + d) = ox.dwdxall(:, i);
        idx((b-1)*6 + d) = i;
    end
end
missing = unique(missing);
end

function [Gdt, Gt, nwdt] = detilt_gram_(ox, idx)
%DETILT_GRAM_  Split the sensed dwdx columns' Gram into the NON-TILT
%   part (per-field piston+tip/tilt projected OUT -- tilt control
%   absorbs tilt, so this is the design driver; Dave 2026-07-19) and
%   the TILT part (projection ONTO piston+tilt -- what the tilt loop
%   must absorb).  The split is orthogonal: G_full = Gdt + Gt.  Works
%   from the multi-field supervisor's per-field blocks; only Grams are
%   needed downstream (row order is irrelevant to a trace).
nz = numel(idx);
Gdt = zeros(nz);  Gt = zeros(nz);
nwdt = 0;
for f = 1:numel(ox.per_field_dwdx)
    Bf = ox.per_field_dwdx{f}(:, idx);
    W2 = ox.per_field_w_nom_2d{f};
    [ri, ci] = find(isfinite(W2));              % column-major == m2v order
    ok = all(isfinite(Bf), 2);
    if numel(ri) ~= size(Bf, 1)
        % fall back: pupil coords unavailable -> piston-only removal
        T = ones(nnz(ok), 1);
    else
        ri = ri(ok);  ci = ci(ok);
        T = [ones(numel(ri), 1), ...
             ri - mean(ri), ci - mean(ci)];
    end
    Bf = Bf(ok, :);
    Bp = T*(T \ Bf);                            % piston+tilt component
    Bt = Bf - Bp;
    Gdt = Gdt + Bt'*Bt;
    Gt  = Gt + Bp'*Bp;
    nwdt = nwdt + size(Bt, 1);
end
end

function hole = hole_margins_(seg, am, aft, hole_r)
%HOLE_MARGINS_  Aft-beam crossing radii at the primary plane vs the
%   central hole (declared center-segment Circle obscuration, or the
%   HOLE_R design override, e.g. the SM shadow radius).
hole = [];
if isempty(aft), return; end
src = '';
if ~isempty(hole_r)
    r_hole = hole_r;
    src = ' (design override)';
else
    obs = macos.get_elt_obs(seg.seg_elts(1));
    ic = find(obs.type == 1, 1);
    if isempty(ic), return; end
    r_hole = obs.vec(1, ic);
    src = ' (declared obscuration)';
end
p0 = seg.frames(1).rpt;  nrm = seg.frames(1).zhat;
q = am.n_beams - 5 : am.n_beams;              % aft beams are appended last
s = am.src_pts(:, q);  t = am.tgt_pts(:, q);
tt = (nrm.'*(p0 - s)) ./ (nrm.'*(t - s));
xp = s + (t - s).*tt;
rc = vecnorm(xp - p0 - nrm*(nrm.'*(xp - p0)));
hole = struct('r_hole', r_hole, 'r_cross', rc, ...
              'margin', r_hole - max(rc), 'src', src);
end
