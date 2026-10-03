function art = run_simulator(seg_in, opts)
%RUN_SIMULATOR  Simulate stage runner: play a state time history through mmacos.
%
%   art = run_simulator(SEG_IN, 'hx', HX, 'jac', JAC, 'met', METMAT,
%   'ts', TS) is the simulate stage of the design pipeline
%       design -> segmentation -> sensitivities -> MET -> compare -> SIMULATE
%   (see design/runners/README.md).  It ingests a TIME SERIES in the
%   rigid-body states x, the segment-figure MonZernike coefficients z,
%   and the grid influence coefficients g (Dave 2026-07-20), plays the
%   history through the mmacos ENGINE, and produces a movie showing
%   BOTH the UNCORRECTED and the CORRECTED performance (Dave
%   2026-07-20: the history opens with um-to-mm misalignments; an
%   initial image-based wavefront control
%       u = -pinv(dwdu) * w(first frame, uncorrected)
%   is solved from the engine wavefront and HELD while the history
%   drifts on with nm-to-um random-walk steps -- the corrected leg
%   starts near-nominal and degrades, the uncorrected leg shows what
%   the control bought).  Each frame shows
%       - the center-field OPD change, uncorrected and corrected,
%       - the PIX psf and the COMPOSE broadband psf (corrected leg,
%         log10 nominal-peak norm),
%       - bar charts of the measurement m = [l; e] on the CORRECTED
%         (i.e. actual, control-applied) system,
%       - the ACCUMULATING rms-WFE (uncorrected vs corrected, log
%         scale) and Strehl-vs-time curves (uncorrected lambda0,
%         corrected lambda0, corrected broadband; peak ratio to the
%         nominal-state psf).
%   Frames are saved and assembled into a GIF, alongside a per-frame
%   report that cross-checks the corrected-leg engine DRIFT INCREMENT
%   (frame t minus frame 1) against the linear model
%   w = dwdx*x + dwdz*z + dwdgrid*g -- the regime the estimator
%   operates in (the absolute frame-1 state is deliberately nonlinear
%   at um-to-mm amplitudes and is handled by the iterated control).
%
%   TWO-PASS ENGINE SCHEDULE (correctness, not convenience): the
%   engine perturb path is INCREMENTAL, and single-axis rotation
%   increments applied in a fixed loop order do not commute --
%   toggling +-u every frame would leave a SYSTEMATIC ~|u_rot|^2
%   non-closure per cycle that accumulates linearly into a phantom
%   rotation of the control bodies (~urad over 100 frames at 50 urad
%   control values).  So the runner plays the WHOLE uncorrected
%   history first (storing the per-frame OPD + psf peak), reloads the
%   Rx for a clean state, and plays the corrected history in a second
%   pass -- within each pass the large state is applied once and the
%   per-frame increments are nm-scale, where the commutator residual
%   is negligible.
%
%   MEASUREMENT MODEL: the m bars are the validated linear measurement
%   model  m = [dldx; dedx]*(x+u) + dmdz*z + dmdgrid*g  (s6: engine-
%   vs-model to 1.2e-6 on l, 6e-8 on e for x; the engine METcalc/Hx
%   hold met and sensor points RIGID, so the model is the ONLY source
%   of the figure terms).  When the simulation runs on the met Rx (no
%   grid states in the history) the engine l is also computed on the
%   corrected leg and cross-checked in the report.
%
%   TIME SERIES (the 'ts' struct):
%     .dt   frame period, seconds (default 1)
%     .x    6*nb x T   rigid states, met-body order, [Rx Ry Rz Tx Ty
%                      Tz] per body; rotations rad, translations SI m
%                      (the RigidBodyChannel contract).  Column 1
%                      carries the initial misalignment the wavefront
%                      control corrects.
%     .z    nz x T     MonZernike coefficients, jac oz.channel_names
%                      row order, prescription BaseUnits of surface
%     .g    ng x T     grid influence coefficients, og.channel_names
%                      row order, BaseUnits of surface
%   Any of x/z/g may be absent or empty (zeros); at least one must be
%   present.  A t=0 NOMINAL frame is prepended to the movie.
%
%   REQUIRED OPTIONS:
%     'hx'    SegMirMaker Hx.m sidecar (axis/loc/SensorPos form)
%     'jac'   sensitivities .mat (ox + optional oz/og) or ox struct
%     'met'   MET-stage .mat (run_met output) or equivalent struct
%     'ts'    the time-series struct above
%   OPTIONS:
%     'wfc_on_frame'  history frame at which the WFC initialises
%                   (default 1).  Frames before it run UNINITIALISED
%                   (Uh = 0, corrected == uncorrected) -- "no system
%                   starts perfect" (Dave 2026-07-20); the movie opens
%                   at the as-deployed WFE for a couple frames, then
%                   the control turns on.
%     'loop_senses_figure'  whether the MET loop's measurement includes
%                   the figure (dmz/dmg) contribution (default true).
%                   Set FALSE to model a truss that reads RIGID POSE
%                   only -- figure is a separate WFS domain, so it
%                   accumulates unseen by the loop and only the
%                   periodic image-based WFC (which sees the wavefront)
%                   removes its rigid-correctable part.
%     'meas_bias'   nmet x T additive measurement-bias trajectory
%                   (default []): a slow metrology calibration drift.
%                   The loop faithfully holds the BIASED reading, so
%                   the true pose walks off unseen; the image-based
%                   reset (unbiased) re-references past it.  Use to
%                   demonstrate the WF Maintenance Activity's core job.
%     'wfc_reset_tol'  ridge tol for the RESET solve (default []=wfc_tol).
%                   A tighter value engages the weakly-observed lateral
%                   DOFs that counter segment focus/astig on a parabolic
%                   parent (Dave 2026-07-21).
%     'wfc'         solve the initial wavefront control (default
%                   true; false = open-loop only, single pass,
%                   identical legs)
%     'met_loop'    close the metrology loop after the initial WFC
%                   (default true; Dave 2026-07-20: the post-WFC state
%                   is the control TARGET -- estimate the pose and
%                   correct any deviation, "without weighting the WF
%                   impact").  This is the RBCS estimator/controller
%                   of Tesch, "RBCS Algorithms" ch 2.3 + 3.3:
%                     estimator (weighted LS / BLUE, eq 11):
%                       dx_est = R_meas * (m - m_ref),
%                       R_meas = (H'N^{-1}H + Rx^{-1})^{-1} H'N^{-1}
%                     controller (min pose error, eq 16-17):
%                       u_t = u_{t-1} - kp * dx_est(control DOFs)
%                   where H = dmdx = d[l;e]/dx.  A RAW pseudo-inverse
%                   (the basic-LS estimator, eq 10) DIVERGES: it
%                   amplifies un-modelled measurement content (figure
%                   drift, linearisation residual) by 1/sigma_min in
%                   the weakly-observed rigid directions and the
%                   integrating loop runs away (the s7 blow-up,
%                   2026-07-20).  The state prior Rx regularises it:
%                   DOFs that drift (PTT) are trusted from metrology,
%                   DOFs that barely move (lateral) and the pinned aft
%                   are pulled to the prior.  This is STATE weighting
%                   (noise + disturbance statistics), NOT wavefront-
%                   impact weighting (Tesch eq 19, deliberately not
%                   used per Dave).  met_loop=false HOLDS the initial
%                   u and the drift accumulates.  Figure drift aliases
%                   into dx_est through the l / e_piston rows (the
%                   simple estimator has no figure states) --
%                   negligible at realistic few-nm figure drift;
%                   figure states via dmdz/dmdgrid are s7b's job.
%     'meas_noise'  metrology measurement noise sigma (N = diag(sigma^2)),
%                   SI m.  Scalar, or [sigma_l sigma_e] for the beam-
%                   length and edge-sensor blocks (default [1e-12 1e-9]
%                   = pm laser truss, nm edge sensors).  Sets the
%                   estimator regularisation SCALE against state_prior.
%     'state_prior' pose-state prior sigma (Rx = diag(sigma^2)), SI,
%                   per DOF -- the disturbance covariance.  "auto"
%                   (default) reads it from the ts drift statistics
%                   (per-DOF std of the deviation from frame 1, with a
%                   floor); or pass an ndof vector.  Encodes the DOF-
%                   class structure (PTT >> lateral >> pinned aft).
%     'ctrl_gain'   proportional control gain kp (default 0.5; <1 for
%                   loop margin against sensitivity-matrix error --
%                   Tesch sec 3.3, the TCE integrator makes the loop
%                   robust to steady-state gain error)
%     'ctrl_reg'    controller command-size penalty rho (eq 17-18,
%                   default 0; the estimator prior does the
%                   conditioning, this only caps command magnitude)
%     'wfc_reset_times'  times (s) at which to RE-RUN the image-based
%                   wavefront control mid-history, like the initial
%                   one -- Tesch's periodic WF Maintenance Activity
%                   (updates the calibration pose x_cal).  Each reset
%                   re-nulls the accumulated wavefront with a fresh
%                   ridge solve on the control DOFs and re-references
%                   the MET loop target to the new pose, so the drift
%                   is knocked back to the (uncontrollable figure)
%                   floor and the corrected leg's slow rise restarts
%                   from there.  Default [] = none (Dave 2026-07-20).
%     'wfc_iters'   Gauss-Newton refinements of the initial control
%                   (default 3): image-based WFC in practice ITERATES
%                   -- at um-to-mm misalignments the ~0.5% engine
%                   nonlinearity of each column leaves a um-scale
%                   residual after ONE linear solve (e2e: 202 um ->
%                   1.3 um one-shot vs 40 nm predicted, 2026-07-20);
%                   each refinement re-measures the engine wavefront
%                   at the corrected state and solves the ridge for a
%                   correction update.  The state path is MONOTONE
%                   (never toggles back), respecting the two-pass
%                   non-closure rule.  1 = the literal one-shot.
%     'control'     control-body indices into the met bodies list
%                   (default segments + hub -- "segment and SM x")
%     'wfc_tol'     Tikhonov ridge weight, relative to the largest
%                   singular value of dwdu (default 1e-3): the solve
%                   is  u = -V*diag(sv/(sv^2+lam^2))*U'*w  with
%                   lam = wfc_tol*max(sv) -- the OSC controller form.
%                   This is CONTROL regularization, not numerical
%                   hygiene.  A plain pinv (or a too-small ridge)
%                   inverts near-degenerate dwdu combinations
%                   (several piston-like Tz directions, weak in-plane
%                   DOFs), filling u with huge canceling commands
%                   that the engine honors only to first order -- a
%                   WORSE psf despite a lower fitted rms (caught by
%                   the fixture test 2026-07-20).  A hard SVD CUTOFF
%                   at a safe level over-truncates instead: on the
%                   e2e history a 1e-2 cutoff left 624 nm of the
%                   202 um initial error uncorrected (residual ~
%                   cutoff x initial).  The ridge corrects every
%                   direction with sv >> lam while bounding the
%                   command in each direction by |w|/(2*lam) --
%                   noise cannot blow up.
%     'met_rx'      Rx for the met-geometry ingest and (when the
%                   history has no grid states) the simulation itself
%                   (default: metopt_in if present, else met_in)
%     'grid_rx'     Rx for histories WITH grid states (default:
%                   og.rx_path -- the grid-augmented Rx of the
%                   harvest; it declares no metrology, so engine l is
%                   unavailable there)
%     'wavelengths' COMPOSE band, engine WaveUnits (default: the Rx
%                   source wavelength * [0.97 1 1.03])
%     'npix'        COMPOSE detector pixels per side (default 128)
%     'pix_dx'      COMPOSE pixel pitch, SI m (default: the native
%                   diffraction pitch dx_at(psf_elt) at the center
%                   wavelength)
%     'psf_elt'     psf element (default: num_elt -- the focal plane)
%     'psf_crop'    PIX display crop, pixels, centered on the nominal
%                   psf peak (default 128)
%     'psf_floor'   log10 display floor (default 1e-8)
%     'dwell'       seconds per GIF frame (default 0.8; settable --
%                   Dave 2026-07-19 on pacing)
%     'gif'         assemble <name>_sim.gif (default true)
%     'ngridpts'    ray-grid override -- MUST match the jac harvest
%                   (default [] = keep the .in values)
%     'model_size'  engine model (default 512)
%     'out_dir'     artifact directory (default: beside SEG_IN)
%     'name'        artifact basename (default: SEG_IN's basename)
%     'visible'     show the figure live (default true)
%     'verbose'     (default true)
%
%   art: .table (per-frame metrics), .t, .rms_wfe_unc, .rms_wfe_corr,
%        .strehl_unc, .strehl_corr, .strehl_bb, .u, .u_bodies,
%        .m_hist, .frames_dir, .gif, .report, .mat.
%
%   See also: run_compare, run_met, run_sensitivities,
%             macos.compose, macos.intensity, macos.design.dmet_dfig.

arguments
    seg_in (1,1) string
    opts.hx (1,1) string
    opts.jac
    opts.met
    opts.ts (1,1) struct
    opts.wfc (1,1) logical = true
    opts.wfc_on_frame (1,1) double {mustBeInteger, mustBePositive} = 1
    opts.met_loop (1,1) logical = true
    opts.loop_senses_figure (1,1) logical = true
    opts.meas_bias double = []
    opts.meas_noise (1,:) double {mustBePositive} = [1e-12 1e-9]
    opts.state_prior = "auto"
    opts.ctrl_gain (1,1) double {mustBePositive} = 0.5
    opts.ctrl_reg (1,1) double {mustBeNonnegative} = 0
    opts.wfc_reset_times (1,:) double = []
    opts.wfc_reset_tol double = []
    opts.wfc_iters (1,1) double {mustBeInteger, mustBePositive} = 3
    opts.control double = []
    opts.wfc_tol (1,1) double {mustBePositive} = 1e-3
    opts.met_rx (1,1) string = ""
    opts.grid_rx (1,1) string = ""
    opts.wavelengths (1,:) double = []
    opts.npix (1,1) double {mustBeInteger, mustBePositive} = 128
    opts.pix_dx (1,1) double {mustBeNonnegative} = 0
    opts.psf_elt (1,1) double {mustBeInteger} = -1
    opts.psf_crop (1,1) double {mustBeInteger, mustBePositive} = 128
    opts.psf_floor (1,1) double {mustBePositive} = 1e-8
    opts.dwell (1,1) double {mustBeNonnegative} = 0.8
    opts.gif (1,1) logical = true
    opts.ngridpts double = []
    opts.model_size (1,1) double = 512
    opts.out_dir (1,1) string = ""
    opts.name (1,1) string = ""
    opts.visible (1,1) logical = true
    opts.verbose (1,1) logical = true
end
assert(isfile(seg_in), 'run_simulator: %s not found', seg_in);
assert(isfile(opts.hx), 'run_simulator: Hx sidecar %s not found', opts.hx);
[ind, base] = fileparts(seg_in);
out_dir = opts.out_dir;  if strlength(out_dir) == 0, out_dir = ind; end
name = opts.name;        if strlength(name) == 0,   name = base; end
pth = @(suffix) char(fullfile(out_dir, name + suffix));
log_ = fopen(pth("_sim_report.txt"), 'w');
closer = onCleanup(@() fclose(log_));
if opts.verbose
    say = @(varargin) fprintf(1, varargin{:}) + fprintf(log_, varargin{:});
else
    say = @(varargin) fprintf(log_, varargin{:});
end
say('==== run_simulator: %s ====\n', seg_in);

%% -- [0] ingest the upstream artifacts + the time series ---------------
M = met_struct_(opts.met);
seg = M.seg;  nseg = seg.nseg;  bodies = M.bodies;  nb = numel(bodies);
met_rx = opts.met_rx;
dldx = M.dldx;
if strlength(met_rx) == 0
    if isfield(M, 'metopt_in') && strlength(string(M.metopt_in)) > 0 ...
            && isfile(M.metopt_in)
        met_rx = string(M.metopt_in);
        if isfield(M, 'dldx_opt') && ~isempty(M.dldx_opt)
            dldx = M.dldx_opt;
        end
    else
        met_rx = string(M.met_in);
    end
end
assert(isfile(met_rx), 'run_simulator: met Rx %s not found', met_rx);
nl = size(dldx, 1);
J = jac_all_(opts.jac);
ox = J.ox;
icf = find(strcmp(ox.field_names, 'C'), 1);
assert(~isempty(icf), 'run_simulator: jac has no center field ''C''');
[Bc, ~, nochan] = dwdx_cols_(ox.per_field_dwdx{icf}, ox.channel_names, bodies);
[~, jnx] = macos.m2v(ox.per_field_w_nom_2d{icf});
es = macos.design.edge_sensors(opts.hx);
assert(any(es.axis > 0), ['run_simulator: legacy Hx (no MeasAxis/' ...
    'SensorPos) -- regenerate with the 2026-07-19 SegMirMaker']);
ie_p = find(es.axis == 1);  ie_g = find(es.axis == 2);
ie_s = find(es.axis == 3);
% dedx from the CURRENT Hx sidecar (rot cols x cbm once cbm known) --
% never the met .mat copy (stale-generation trap, 2026-07-19)
dedx = zeros(es.nmeas, 6*nb);
dedx(:, 1:6*nseg) = es.dedx;
control = opts.control;
if isempty(control), control = 1:min(nseg+1, nb); end   % segments + SM/hub
assert(~opts.wfc || ~any(ismember(bodies(control), nochan)), ...
    'run_simulator: a control body has no dwdx channels -- not actuatable');
ucols = reshape(((control(:)-1)*6 + (1:6)).', 1, []);

% the time series
ts = opts.ts;
dt = 1;  if isfield(ts, 'dt'), dt = ts.dt; end
X = [];  Z = [];  G = [];
if isfield(ts, 'x'), X = ts.x; end
if isfield(ts, 'z'), Z = ts.z; end
if isfield(ts, 'g'), G = ts.g; end
T = max([size(X, 2), size(Z, 2), size(G, 2)]);
assert(T > 0, 'run_simulator: ts has no frames (provide ts.x / ts.z / ts.g)');
if isempty(X), X = zeros(6*nb, T); end
if isempty(Z) && ~isempty(J.oz), Z = zeros(numel(J.oz.channel_names), T); end
if isempty(G) && ~isempty(J.og), G = zeros(numel(J.og.channel_names), T); end
assert(size(X, 1) == 6*nb && size(X, 2) == T, ...
    'run_simulator: ts.x must be %d x %d (6*nb x T, met-body order)', 6*nb, T);
use_z = ~isempty(Z) && any(Z(:) ~= 0);
use_g = ~isempty(G) && any(G(:) ~= 0);
assert(~use_z || ~isempty(J.oz), 'run_simulator: ts.z given but jac has no oz');
assert(~use_g || ~isempty(J.og), 'run_simulator: ts.g given but jac has no og');
if use_z
    assert(size(Z, 1) == numel(J.oz.channel_names) && size(Z, 2) == T, ...
        'run_simulator: ts.z must be %d x %d (oz.channel_names order)', ...
        numel(J.oz.channel_names), T);
end
if use_g
    assert(size(G, 1) == numel(J.og.channel_names) && size(G, 2) == T, ...
        'run_simulator: ts.g must be %d x %d (og.channel_names order)', ...
        numel(J.og.channel_names), T);
end

% simulation Rx: the grid-augmented Rx when the history moves grid
% states (the only Rx whose grid channels match the harvest basis),
% else the met Rx (engine l then available for cross-check)
if use_g
    sim_rx = opts.grid_rx;
    if strlength(sim_rx) == 0, sim_rx = string(J.og.rx_path); end
    assert(isfile(sim_rx), ...
        'run_simulator: grid-augmented Rx %s not found', sim_rx);
else
    sim_rx = met_rx;
end
say(['[0] %d bodies (%d segments), T = %d frames, dt = %g s\n' ...
     '    states: x %dx%d%s%s;  sim Rx %s\n'], nb, nseg, T, dt, ...
    size(X), iif_(use_z, sprintf(', z %dx%d', size(Z)), ''), ...
    iif_(use_g, sprintf(', g %dx%d', size(G)), ''), sim_rx);
if ~isempty(nochan)
    say('    NOTE: no dwdx channels for element(s) %s -- zero linear w columns\n', ...
        num2str(nochan));
end

%% -- [1] met-Rx ingest: cbm, met geometry, figure-sensing blocks -------
old = cd(out_dir);  restore = onCleanup(@() cd(old));
m = macos.Session(opts.model_size);
m.load_rx(char(met_rx));
if ~isempty(opts.ngridpts), m.set_src_sampling(opts.ngridpts); end
cbm = m.cbm();
for s = 1:nseg
    dedx(:, (s-1)*6+(1:3)) = dedx(:, (s-1)*6+(1:3)) * cbm;
end
% dwdx units adapter (f5d648f): supervisors emit BaseUnits-OPD; this
% simulator works in SI metres (dW maps *cbm), so scale the rigid-body
% columns once -- matches the dwdz/dwdg *cbm at their use sites and
% run_compare's identical adapter.  Metre decks: no-op.  Absence on a
% mm deck under-commanded u by 1/cbm (tRunCompare time-history fail,
% triaged 2026-08-28).
Bc = Bc * cbm;
gmet = macos.met_geom();
dmz = [];  dmg = [];
if use_z || use_g
    dfa = {};
    if use_z, dfa = [dfa, {'z_names', J.oz.channel_names}]; end
    if use_g
        assert(isfield(J.og, 'sgb') && ~isempty(J.og.sgb), ...
            ['run_simulator: og carries no sgb influence basis -- ' ...
             're-harvest with the current run_sensitivities (the basis ' ...
             'is part of the Jacobian; a rebuild is not bit-stable)']);
        dfa = [dfa, {'g_names', J.og.channel_names, 'sgb', J.og.sgb}];
    end
    df = macos.design.dmet_dfig(seg, es, gmet, dfa{:}, 'unit_to_m', cbm);
    if use_z, dmz = [df.dldz; df.dedz]; end
    if use_g, dmg = [df.dldg; df.dedg]; end
end
dmdx = [dldx; dedx];
if ~isempty(opts.meas_bias)
    assert(isequal(size(opts.meas_bias), [size(dmdx, 1), T]), ...
        'run_simulator: meas_bias must be %d x %d (nmet x T)', ...
        size(dmdx, 1), T);
end

%% -- [2] simulation baseline ------------------------------------------
chx = {};  chz = {};  chg = {};              % filled by load_sim_rx_
Wb = [];  maskb = [];
load_sim_rx_();
n_elt = m.num_elt();
wf_elt = n_elt - 1;
psf_elt = opts.psf_elt;  if psf_elt < 1, psf_elt = n_elt; end
m.trace(wf_elt);
Wb = m.opd();  maskb = Wb ~= 0;
engine_l = false;
l0b = [];
if ~use_g
    l0 = macos.met();
    if l0.n == nl, engine_l = true;  l0b = l0.l; end
end
wvl0 = macos.get_src_wvl();
wband = opts.wavelengths;
if isempty(wband), wband = wvl0 * [0.97 1 1.03]; end
P0 = macos.intensity(psf_elt);
pix_dx = opts.pix_dx;
if pix_dx <= 0, pix_dx = macos.dx_at(psf_elt, 'm'); end
C0 = macos.compose(psf_elt, wband, opts.npix, pix_dx);
macos.set_src_wvl(wvl0);
p0max = max(P0(:));  c0max = max(C0(:));
assert(p0max > 0 && c0max > 0, ...
    'run_simulator: nominal psf is empty at elt %d', psf_elt);
[pi0, pj0] = find(P0 == p0max, 1);
crop = min(opts.psf_crop, min(size(P0)));
ci = crop_(pi0, crop, size(P0, 1));  cj = crop_(pj0, crop, size(P0, 2));
say(['[1] baseline on %s: %d rays in pupil; psf @ elt %d, peak %.3g;\n' ...
     '    COMPOSE band [%s] WaveUnits, %d px @ %.3g um;%s engine l %s\n'], ...
    sim_rx, nnz(maskb), psf_elt, p0max, ...
    strtrim(sprintf('%g ', wband)), opts.npix, pix_dx*1e6, ...
    iif_(engine_l, '', ' (grid Rx: no metrology)'), ...
    iif_(engine_l, 'cross-checked (corrected leg)', 'model-only'));
say('[2] channels in motion: %d rigid, %d zernike, %d grid\n', ...
    numel(chx), numel(chz), numel(chg));

%% -- [3] pass 1, UNCORRECTED: play the raw history ---------------------
% Stores the per-frame OPD map + psf peak; the as-deployed frame (wof)
% feeds the wavefront-control solve.  Skipped when wfc is off.  The WFC
% is DELAYED to history frame wof (Dave 2026-07-20: "no system starts
% perfect" -- the first frames stand uninitialized before control turns
% on), so Uh = 0 for t < wof and the corrected leg == uncorrected there.
tv = (0:T) * dt;
rms_unc = zeros(1, T+1);
st_unc = ones(1, T+1);
dWu_hist = cell(1, T);
U = zeros(6*nb, 1);
wof = min(max(opts.wfc_on_frame, 1), T);       % WFC-on history frame
Pdep = P0;  Cdep = C0;                          % as-deployed psf/compose
if opts.wfc
    say('[3] pass 1 (uncorrected):');
    for t = 1:T
        dWu_hist{t} = engine_delta_(t, zeros(6*nb, 1));
        rms_unc(t+1) = rms_(dWu_hist{t}(isfinite(dWu_hist{t}))) * 1e9;
        st_unc(t+1) = strehl_opd_(dWu_hist{t}, cbm, wvl0);
        if t == 1                              % as-deployed display frame
            Pdep = macos.intensity(psf_elt);
            Cdep = macos.compose(psf_elt, wband, opts.npix, pix_dx);
            macos.set_src_wvl(wvl0);
        end
        if mod(t, 10) == 0, say(' %d', t); end
    end
    say(' done\n');
    % u = -ridge(dwdu) * w(frame wof): image-based sensing, center-field
    % dwdx columns of the control bodies, Tikhonov-ridge inverse (the
    % OSC controller form) so weak directions are corrected without
    % noise blowing up into huge canceling commands -- see 'wfc_tol'.
    % The reset (WF-maintenance) uses a TIGHTER ridge so it engages the
    % weakly-observed LATERAL DOFs that counter segment astigmatism on a
    % parabolic parent (Dave 2026-07-21) -- 'wfc_reset_tol'.
    Bu = Bc(:, ucols);
    [Us, S, Vs] = svd(Bu, 'econ');
    sv = diag(S);
    lam = opts.wfc_tol * max(sv);
    ridge_ = @(w) -Vs * ((sv ./ (sv.^2 + lam^2)) .* (Us' * w));
    rtol = opts.wfc_tol;  if ~isempty(opts.wfc_reset_tol), rtol = opts.wfc_reset_tol; end
    lamr = rtol * max(sv);
    ridge_reset_ = @(w) -Vs * ((sv ./ (sv.^2 + lamr^2)) .* (Us' * w));
    w1 = wvec_(dWu_hist{wof});
    u = ridge_(w1);
    U(ucols) = u;
    say(['    initial wavefront control @ frame %d (t=%g s): uncorrected w ' ...
         'rms %.4g um -> predicted residual %.4g nm\n    (u on %d bodies, ' ...
         'ridge %.1g, %d/%d directions with sv > lam, |u| max %.4g)\n'], ...
        wof, wof*dt, rms_(w1)*1e6, rms_(w1 + Bu*u)*1e9, numel(control), ...
        opts.wfc_tol, nnz(sv > lam), numel(sv), max(abs(u)));
    % clean engine state for pass 2: the +-u toggle must never happen
    % (see the two-pass note in the header)
    for c = [chx, chz, chg], c{1}.ch.restore(); end
    load_sim_rx_();
    m.trace(wf_elt);
    Wb2 = m.opd();
    assert(isequal(Wb2 ~= 0, maskb), ...
        'run_simulator: baseline mask changed across the pass-2 reload');
    Wb = Wb2;
    % Gauss-Newton refinements: at um-to-mm misalignments the engine's
    % ~0.5% per-column nonlinearity leaves a um-scale residual after
    % one linear solve -- re-measure and re-solve (monotone state
    % path: the correction only ever refines, never toggles back)
    for it = 2:opts.wfc_iters
        wk = wvec_(engine_delta_(wof, U));
        du = ridge_(wk);
        U(ucols) = U(ucols) + du;
        say('    wfc iteration %d: engine residual %.4g nm -> update |du| max %.4g\n', ...
            it, rms_(wk)*1e9, max(abs(du)));
    end
else
    say('[3] wfc off: open-loop only (single pass, legs identical)\n');
end

% -- [3b] the metrology loop: RBCS estimator + controller --------------
% The RBCS pose estimator/controller (Tesch, "RBCS Algorithms" ch 2.3
% + 3.3).  The post-WFC state (X(:,1)+U) is the control TARGET; each
% frame the estimator infers the pose deviation from the sensed drift
% dm = m - m_ref and the controller nulls it on the control DOFs.  The
% whole history is pure algebra on the LINEAR measurement model -- what
% the sensors deliver (s6-validated) -- computed before the engine pass.
Uh = repmat(U, 1, T);                   % column t = control at frame t
Ms = zeros(size(dmdx, 1), T);           % sensed drift dm (the bars)
jnz = [];  jng = [];                    % figure canvas pixel maps (wlin_)
if use_z, [~, jnz] = macos.m2v(J.oz.per_field_w_nom_2d{icf_of_(J.oz)}); end
if use_g, [~, jng] = macos.m2v(J.og.per_field_w_nom_2d{icf_of_(J.og)}); end
if opts.wfc && opts.met_loop
    % --- estimator gain R_meas: weighted LS / BLUE (Tesch eq 11) ------
    % R_meas = (H' N^{-1} H + Rx^{-1})^{-1} H' N^{-1}, H = dmdx.  N is
    % the measurement-noise cov, Rx the state (disturbance) prior.  A
    % raw pinv (basic LS, eq 10) amplifies un-modelled dm content by
    % 1/sigma_min in the weakly-observed rigid directions -> the
    % integrating loop diverges (the s7 blow-up, 2026-07-20); the Rx
    % prior regularises so weak/pinned DOFs fall back to prior 0.
    mn = opts.meas_noise;
    if isscalar(mn), sig_m = mn * ones(size(dmdx, 1), 1);
    else
        assert(numel(mn) == 2, ['run_simulator: meas_noise is scalar ' ...
            'or [sigma_l sigma_e]']);
        sig_m = [mn(1)*ones(nl, 1); mn(2)*ones(size(dmdx,1)-nl, 1)];
    end
    Ni = 1 ./ (sig_m.^2);                          % N^{-1} diagonal
    sp = opts.state_prior;
    if (isstring(sp) || ischar(sp)) && strcmpi(string(sp), "auto")
        % disturbance prior from the drift statistics: per-DOF std of
        % the deviation from the target frame (PTT >> lateral), floored
        sig_x = std(X - X(:, 1), 0, 2);
        fl = max(sig_x) * 1e-3 + 1e-15;            % floor for pinned DOFs
        sig_x = max(sig_x, fl);
    else
        sig_x = sp(:);
        assert(numel(sig_x) == 6*nb, ['run_simulator: state_prior must ' ...
            'be "auto" or a %d-vector'], 6*nb);
    end
    Rxi = 1 ./ (sig_x.^2);                         % Rx^{-1} diagonal
    HtNi = dmdx' .* Ni.';                          % H' N^{-1}  (ndof x nmet)
    Rmeas = (HtNi * dmdx + diag(Rxi)) \ HtNi;      % BLUE gain (ndof x nmet)
    kp = opts.ctrl_gain;  rho = opts.ctrl_reg;
    % --- controller (min pose error, Tesch eq 16-17) -----------------
    % actuators ARE the control rigid DOFs, so dx/du is a selection
    % matrix and G = (rho I + (dx/du)'(dx/du))^{-1}(dx/du)' just picks
    % the control-DOF components of the pose error, scaled by 1/(1+rho)
    reset_fr = unique(round(opts.wfc_reset_times / dt));
    reset_fr = reset_fr(reset_fr > wof & reset_fr <= T);
    Uh(:, 1:wof-1) = 0;                             % uninitialised (as deployed)
    m_ref = msense_(wof, U);                        % loop target at WFC-on frame
    for t = wof+1:T
        Ms(:, t) = msense_(t, Uh(:, t-1)) - m_ref;
        dx_est = Rmeas * Ms(:, t);                 % estimated deviation
        Uh(:, t) = Uh(:, t-1);
        Uh(ucols, t) = Uh(ucols, t-1) ...
            - (kp / (1 + rho)) * dx_est(ucols);    % null the pose error
        if any(t == reset_fr)
            % WF Maintenance: re-run the image-based control on the
            % current TRUE wavefront (rigid + figure; unbiased -- image
            % sensing sees no metrology bias) and re-reference the loop
            % target to the re-nulled pose.  The tighter reset ridge
            % engages the lateral DOFs that counter segment focus/astig
            % on a parabolic parent (Dave 2026-07-21).
            Uh(ucols, t) = Uh(ucols, t) + ridge_reset_(wlin_(t));
            m_ref = msense_(t, Uh(:, t));
        end
    end
    say(['[3b] metrology loop ON (RBCS weighted-LS estimator + ' ...
         'pose-error controller):\n' ...
         '     R_meas = (H''N^-1 H + Rx^-1)^-1 H''N^-1, kp %.2g, ' ...
         'rho %.2g; sensor sigma [%.2g %.2g] m;\n' ...
         '     state prior sigma %.2g..%.2g m (%s); loop senses figure: %d; ' ...
         'meas bias: %d\n'], kp, rho, ...
        sig_m(1), sig_m(end), min(sig_x), max(sig_x), ...
        iif_((isstring(sp)||ischar(sp)) && strcmpi(string(sp),"auto"), ...
             'auto from drift stats', 'user'), opts.loop_senses_figure, ...
        ~isempty(opts.meas_bias));
    if ~isempty(reset_fr)
        say('     WF-maintenance recontrol at t = %s s (reset ridge %.1g)\n', ...
            num2str(reset_fr * dt), rtol);
    end
elseif opts.wfc
    Uh(:, 1:wof-1) = 0;                             % uninitialised (as deployed)
    say('[3b] metrology loop OFF: initial u HELD from frame %d for the rest\n', wof);
end

% per-frame linear predictions (both legs) for the cross-check + the
% fixed display limits of the whole movie
dWl_cor = cell(1, T);
cmax_u = eps;  cmax_c = eps;
for t = 1:T
    D = zmap_(Bc * X(:, t), jnx, size(Wb));
    if use_z, D = D + zmap_(J.oz.per_field_dwdz{icf_of_(J.oz)} * Z(:, t) * cbm, ...
            jnz, size(Wb)); end
    if use_g, D = D + zmap_(J.og.per_field_dwdg{icf_of_(J.og)} * G(:, t) * cbm, ...
            jng, size(Wb)); end
    cmax_u = max(cmax_u, max(abs(D(:))));
    dWl_cor{t} = D + zmap_(Bc * Uh(:, t), jnx, size(Wb));
    cmax_c = max(cmax_c, max(abs(dWl_cor{t}(:))));
end
Mh = dmdx * (X + Uh);                   % measurements on the ACTUAL system
for t = 1:T, Mh(:, t) = Mh(:, t) + mfig_(t); end
if opts.wfc && opts.met_loop
    Mbar = Ms;                          % bars show the SENSED DRIFT
    mlbl = '\deltam sensed drift (model)';
else
    Mbar = Mh;                          % bars show the absolute m
    mlbl = 'm (model, actual system)';
end
yl_l = sym_lim_(Mbar(1:nl, :));  yl_e = sym_lim_(Mbar(nl+1:end, :));

%% -- [4] pass 2, CORRECTED: the movie ----------------------------------
fdir = pth("_sim_frames");
if ~exist(fdir, 'dir'), mkdir(fdir); end
stale = dir(fullfile(fdir, 't*.png'));       % stale-frame trap (s6 lesson)
for q = 1:numel(stale), delete(fullfile(fdir, stale(q).name)); end
gif_file = '';
if opts.gif, gif_file = pth("_sim.gif"); end
vis = 'off';  if opts.visible, vis = 'on'; end
fg = figure('Visible', vis, 'Position', [40 60 1500 880], 'Color', 'w');
fig_closer = onCleanup(@() close(fg));

rms_cor = zeros(1, T+1);
st_cor = ones(1, T+1);  st_bb = ones(1, T+1);
dWc1 = [];                              % corrected frame-1 increment ref
Tab = struct('t', {}, 'rms_unc_nm', {}, 'rms_corr_nm', {}, 'w_rel', {}, ...
    'strehl_unc', {}, 'strehl_corr', {}, 'strehl_bb', {}, ...
    'l_max_nm', {}, 'e_max_nm', {}, 'l_rel', {});
% the t=0 (as-deployed) point on both curves is the frame-1 uncorrected
% state -- no system starts perfect (Dave 2026-07-20)
if opts.wfc, rms_unc(1) = rms_unc(2);  st_unc(1) = st_unc(2); end
say(['\n[4]   t(s)  rms_unc(nm) rms_corr(nm)   w_rel   S_unc  S_corr  ' ...
     'S_bb   |l|max(nm) |e|max(nm)  l_rel\n']);
for t = 0:T
    if t > 0
        dWc = engine_delta_(t, Uh(:, t));
        mkc = isfinite(dWc);
        l_rel = NaN;
        if engine_l
            dl_t = macos.met().l - l0b;
            l_rel = max(abs(dl_t - Mh(1:nl, t))) / max(max(abs(dl_t)), 1e-12);
        end
        P = macos.intensity(psf_elt);
        C = macos.compose(psf_elt, wband, opts.npix, pix_dx);
        macos.set_src_wvl(wvl0);
        % engine-vs-linear on the DRIFT INCREMENT (frame t minus frame
        % 1) -- the regime the estimator operates in.  The absolute
        % frame-1 state is deliberately nonlinear at um-to-mm
        % amplitudes; the iterated control compensates it, so an
        % absolute comparison would only re-measure that compensation.
        if t == 1
            dWc1 = dWc;
            w_rel = NaN;                          % increment reference
        else
            dI = dWc - dWc1;
            DLi = dWl_cor{t} - dWl_cor{1};
            mki = mkc & isfinite(dWc1);
            dI(mki) = dI(mki) - mean(dI(mki));
            DLi(mki) = DLi(mki) - mean(DLi(mki));
            w_rel = rms_(dI(mki) - DLi(mki)) / max(rms_(dI(mki)), eps);
        end
        lm = Mbar(1:nl, t);  em = Mbar(nl+1:end, t);
        if opts.wfc, dWu = dWu_hist{t}; else, dWu = dWc; end
    else
        % t=0: the AS-DEPLOYED, uninitialised system (Dave: no system
        % starts perfect) -- show the uncorrected frame-1 wavefront/psf,
        % not a perfect nominal.  (Strehl is still normalised to the
        % diffraction-limited P0.)
        if opts.wfc
            dWu = dWu_hist{1};  dWc = dWu;  P = Pdep;  C = Cdep;
        else
            dWu = zeros(size(Wb));  dWu(~maskb) = NaN;  dWc = dWu;
            P = P0;  C = C0;
        end
        w_rel = NaN;  l_rel = NaN;
        lm = zeros(nl, 1);  em = zeros(es.nmeas, 1);
    end
    k = t + 1;
    rms_cor(k) = rms_(dWc(isfinite(dWc))) * 1e9;
    st_cor(k) = strehl_opd_(dWc, cbm, wvl0);        % exact aperture Strehl
    st_bb(k) = max(C(:)) / c0max;                   % psf-peak (COMPOSE panel only)
    if ~opts.wfc, rms_unc(k) = rms_cor(k);  st_unc(k) = st_cor(k); end
    Tab(k) = struct('t', tv(k), 'rms_unc_nm', rms_unc(k), ...
        'rms_corr_nm', rms_cor(k), 'w_rel', w_rel, ...
        'strehl_unc', st_unc(k), 'strehl_corr', st_cor(k), ...
        'strehl_bb', st_bb(k), 'l_max_nm', max([abs(lm); 0])*1e9, ...
        'e_max_nm', max([abs(em); 0])*1e9, 'l_rel', l_rel);
    say('    %6.0f  %11.4g  %11.4g  %7s  %6.3f  %6.3f  %6.3f  %10.3g %10.3g  %s\n', ...
        tv(k), rms_unc(k), rms_cor(k), rel_txt_(w_rel), st_unc(k), ...
        st_cor(k), st_bb(k), Tab(k).l_max_nm, Tab(k).e_max_nm, rel_txt_(l_rel));

    ttl_tag = '';
    if opts.wfc && t < wof, ttl_tag = ' (as deployed, uninitialised)';
    elseif opts.wfc && t == wof, ttl_tag = ' (wavefront control applied)';
    elseif opts.wfc && any(round(opts.wfc_reset_times/dt) == t)
        ttl_tag = ' (WF maintenance recontrol)';
    elseif t == 0, ttl_tag = ' (nominal)';
    end
    draw_frame_(fg, sprintf('%s | t = %.0f s%s', name, tv(k), ttl_tag), ...
        dWu, dWc, P(ci, cj)/p0max, C/c0max, lm, em, ie_p, ie_g, ie_s, ...
        tv(1:k), rms_unc(1:k), rms_cor(1:k), st_unc(1:k), st_cor(1:k), ...
        st_bb(1:k), cmax_u*1e9, cmax_c*1e9, tv(end), opts.psf_floor, ...
        opts.wfc, yl_l, yl_e, mlbl, opts.wfc_reset_times);
    drawnow;
    if opts.visible && opts.dwell > 0, pause(opts.dwell); end
    fr = fullfile(fdir, sprintf('t%03d.png', t));
    print(fg, fr, '-dpng', '-r100');
    if opts.gif
        [q, cmp] = rgb2ind(imread(fr), 256);
        if t == 0
            imwrite(q, cmp, gif_file, 'gif', 'LoopCount', Inf, ...
                'DelayTime', max(opts.dwell, 0.05));
        else
            imwrite(q, cmp, gif_file, 'gif', 'WriteMode', 'append', ...
                'DelayTime', max(opts.dwell, 0.05));
        end
    end
end
for c = [chx, chz, chg], c{1}.ch.restore(); end

%% -- [5] summary + artifacts ------------------------------------------
say('\n[5] final: rms WFE %.4g nm uncorrected vs %.4g nm corrected;\n', ...
    rms_unc(end), rms_cor(end));
say('    Strehl %.4f uncorrected vs %.4f corrected (%.4f broadband); worst w_rel %s\n', ...
    st_unc(end), st_cor(end), st_bb(end), rel_txt_(max([Tab.w_rel])));
if engine_l
    say('    engine-l cross-check (corrected leg): worst l_rel %s\n', ...
        rel_txt_(max([Tab.l_rel])));
else
    say('    m bars are the LINEAR measurement model (grid Rx declares no metrology;\n');
    say('    the engine METcalc/Hx hold met/sensor points rigid for figure states)\n');
end
art = struct('table', Tab, 't', tv, 'rms_wfe_unc', rms_unc, ...
    'rms_wfe_corr', rms_cor, 'strehl_unc', st_unc, 'strehl_corr', st_cor, ...
    'strehl_bb', st_bb, 'u', U(ucols), 'u_bodies', bodies(control), ...
    'u_full', U, 'u_hist', Uh(ucols, :), 'met_loop', ...
    (opts.wfc && opts.met_loop), 'frames_dir', fdir, 'gif', gif_file, ...
    'report', pth("_sim_report.txt"), 'mat', pth("_sim.mat"), ...
    'sim_rx', char(sim_rx), 'met_rx', char(met_rx), 'bodies', bodies, ...
    'wavelengths', wband, 'npix', opts.npix, 'pix_dx', pix_dx, ...
    'psf_elt', psf_elt, 'dt', dt, 'engine_l', engine_l);
art.m_hist = Mh;  art.ts = ts;
if use_z, art.dmdz = dmz; end
if use_g, art.dmdgrid = dmg; end
save(art.mat, '-struct', 'art', '-v7.3');
say('\nartifacts: %s_sim.mat, frames in %s_sim_frames/%s, this report\n', ...
    name, name, iif_(opts.gif, sprintf(', %s_sim.gif', name), ''));

% ---- nested: (re)load the sim Rx and rebuild the state channels -------
    function load_sim_rx_()
        m.load_rx(char(sim_rx));
        if ~isempty(opts.ngridpts), m.set_src_sampling(opts.ngridpts); end
        chx = {};  chz = {};  chg = {};
        xrows = find(any(X ~= 0, 2)).';
        if opts.wfc, xrows = unique([xrows, ucols]); end
        for r = xrows
            kb = ceil(r / 6);  d = r - (kb-1)*6;
            chx{end+1} = struct('row', r, 'ch', ...
                macos.channels.RigidBodyChannel(m, bodies(kb), d - 1)); %#ok<AGROW>
        end
        if use_z
            for r = find(any(Z ~= 0, 2)).'
                tok = regexp(J.oz.channel_names{r}, ...
                    '^Elt (\d+) MonZern(\d+)$', 'tokens', 'once');
                assert(~isempty(tok), ...
                    'run_simulator: unexpected z channel %s', ...
                    J.oz.channel_names{r});
                chz{end+1} = struct('row', r, 'ch', ...
                    macos.channels.MonZernChannel(m, ...
                    str2double(tok{1}), str2double(tok{2}))); %#ok<AGROW>
            end
        end
        if use_g
            gch = macos.channels.grid_channels(m, J.og.sgb);
            names_g = cellfun(@(c) c.name(), gch, 'UniformOutput', false);
            for r = find(any(G ~= 0, 2)).'
                i = find(strcmp(names_g, J.og.channel_names{r}), 1);
                assert(~isempty(i), 'run_simulator: sgb lacks channel %s', ...
                    J.og.channel_names{r});
                chg{end+1} = struct('row', r, 'ch', gch{i}); %#ok<AGROW>
            end
        end
    end

% ---- nested: apply a full state (x-vector offset UADD) + trace --------
    function dW = engine_delta_(t_, Uadd)
        for c_ = chx, c_{1}.ch.apply(X(c_{1}.row, t_) + Uadd(c_{1}.row)); end
        for c_ = chz, c_{1}.ch.apply(Z(c_{1}.row, t_)); end
        for c_ = chg, c_{1}.ch.apply(G(c_{1}.row, t_)); end
        m.trace(wf_elt);
        W_ = m.opd();
        dW = nan(size(Wb));
        mk_ = maskb & (W_ ~= 0);
        dW(mk_) = (W_(mk_) - Wb(mk_)) * cbm;
        dW(mk_) = dW(mk_) - mean(dW(mk_));        % piston-removed WFE
    end

% ---- nested: a dW map as an m2v-ordered vector on the jac pixels ------
    function w = wvec_(dW)
        w = zeros(numel(jnx.i), 1);
        v_ = dW(sub2ind(size(dW), jnx.i, jnx.j));
        w(isfinite(v_)) = v_(isfinite(v_));
    end

% ---- nested: the figure-state contribution to the measurement ---------
    function mf = mfig_(t_)
        mf = zeros(size(dmdx, 1), 1);
        if use_z, mf = mf + dmz * Z(:, t_); end
        if use_g, mf = mf + dmg * G(:, t_); end
    end

% ---- nested: the metrology's SENSED measurement at frame t_ ------------
% Rigid pose (dmdx*(X+Ucol)); + figure only if the loop senses it
% (loop_senses_figure -- the truss reads rigid pose, figure is a
% separate WFS domain); + a slow measurement BIAS if supplied (Run A:
% metrology calibration drift the image-based reset re-references past).
    function ms = msense_(t_, Ucol)
        ms = dmdx * (X(:, t_) + Ucol);
        if opts.loop_senses_figure, ms = ms + mfig_(t_); end
        if ~isempty(opts.meas_bias), ms = ms + opts.meas_bias(:, t_); end
    end

% ---- nested: the linear wavefront (rigid + figure) at jnx pixels -------
    function w = wlin_(t_)
        W = zmap_(Bc * (X(:, t_) + Uh(:, t_)), jnx, size(Wb));
        if use_z, W = W + zmap_(J.oz.per_field_dwdz{icf_of_(J.oz)} ...
                * Z(:, t_) * cbm, jnz, size(Wb)); end
        if use_g, W = W + zmap_(J.og.per_field_dwdg{icf_of_(J.og)} ...
                * G(:, t_) * cbm, jng, size(Wb)); end
        w = W(sub2ind(size(W), jnx.i, jnx.j));
        w(~isfinite(w)) = 0;
    end
end

% =========================================================================
function s = iif_(cond, a, b)
s = b;  if cond, s = a; end
end

function v = rms_(x)
if isempty(x), v = 0; else, v = sqrt(mean(x(:).^2)); end
end

function s = strehl_opd_(dW, cbm, wvl0)
%STREHL_OPD_  Exact aperture Strehl from the OPD deviation (<= 1 always).
%   S = |<exp(i 2pi W/lambda)>|^2 over the valid pupil pixels, W the
%   piston-removed wavefront deviation from nominal.  dW is in SI metres
%   (opd*cbm); the raw WaveUnits OPD is dW/cbm and lambda (wvl0) is in
%   WaveUnits, so W/lambda = (dW/cbm)/wvl0 is dimensionless (waves).
%   Robust to the sub-pixel psf-peak sampling that let max(P)/max(P0)
%   read marginally > 1 (Dave 2026-07-21).
w = dW(isfinite(dW)) / (cbm * wvl0);
if isempty(w), s = 1; else, s = abs(mean(exp(1i * 2 * pi * w)))^2; end
end

function s = rel_txt_(v)
if isnan(v), s = 'null';  else, s = sprintf('%.2e', v); end
end

function i = icf_of_(o)
i = find(strcmp(o.field_names, 'C'), 1);
end

function D = zmap_(col, jn, sz)
%ZMAP_  Scatter an m2v-ordered column onto a zeros canvas of size SZ.
assert(isequal(jn.size, sz), ...
    'run_simulator: jac canvas %dx%d vs engine %dx%d -- ngridpts must match', ...
    jn.size(1), jn.size(2), sz(1), sz(2));
D = zeros(sz);
D(sub2ind(jn.size, jn.i, jn.j)) = col;
end

function idx = crop_(c, w, n)
a = max(1, min(c - floor(w/2), n - w + 1));
idx = a:(a + w - 1);
end

function yl = sym_lim_(A)
v = max([abs(A(:)); eps]) * 1e9 * 1.15;
yl = [-v, v];
end

function draw_frame_(fg, ttl, dWu, dWc, Pn, Cn, l, e, ie_p, ie_g, ie_s, ...
    tv, ru, rc, su, sc, sb, cmax_u, cmax_c, t_end, floorv, wfc, ...
    yl_l, yl_e, mlbl, reset_times)
%DRAW_FRAME_  One movie frame: OPD unc | OPD corr | PIX | COMPOSE over
%   m bars over the accumulating rms-WFE / Strehl curves.
clf(fg);
tl = tiledlayout(fg, 3, 8, 'Padding', 'compact', 'TileSpacing', 'compact');
title(tl, ttl, 'Interpreter', 'none', 'FontWeight', 'bold');

opd_panel_(nexttile(tl, 1, [1 2]), dWu, cmax_u, ...
    iif_(wfc, 'OPD uncorrected', 'OPD (open loop)'), ...
    sprintf('rms WFE %.4g nm', ru(end)));
opd_panel_(nexttile(tl, 3, [1 2]), dWc, cmax_c, ...
    iif_(wfc, 'OPD corrected', 'OPD (open loop)'), ...
    sprintf('rms WFE %.4g nm', rc(end)));

ax = nexttile(tl, 5, [1 2]);
imagesc(ax, log10(max(Pn, floorv)));
axis(ax, 'image');  set(ax, 'XTick', [], 'YTick', []);
clim(ax, [log10(floorv), 0]);  colormap(ax, hot);  colorbar(ax);
title(ax, iif_(wfc, 'PIX psf, corrected (log10)', 'PIX psf (log10)'), ...
    'FontWeight', 'normal');

ax = nexttile(tl, 7, [1 2]);
imagesc(ax, log10(max(Cn, floorv)));
axis(ax, 'image');  set(ax, 'XTick', [], 'YTick', []);
clim(ax, [log10(floorv), 0]);  colormap(ax, hot);  colorbar(ax);
title(ax, 'COMPOSE broadband psf (log10)', 'FontWeight', 'normal');

ax = nexttile(tl, 9, [1 3]);
bar(ax, l * 1e9, 'FaceColor', [0.2 0.45 0.75]);
ylim(ax, yl_l);  grid(ax, 'on');
ylabel(ax, 'l (nm)');  xlabel(ax, 'beam');
title(ax, ['metrology ' mlbl], 'FontWeight', 'normal');

ax = nexttile(tl, 12, [1 5]);
hold(ax, 'on');
bar(ax, ie_p, e(ie_p) * 1e9, 0.9, 'FaceColor', [0.2 0.45 0.75]);
bar(ax, ie_g, e(ie_g) * 1e9, 0.9, 'FaceColor', [0.85 0.55 0.15]);
bar(ax, ie_s, e(ie_s) * 1e9, 0.9, 'FaceColor', [0.5 0.7 0.3]);
ylim(ax, yl_e);  grid(ax, 'on');
ylabel(ax, 'e (nm)');  xlabel(ax, 'sensor row');
legend(ax, {'piston', 'gap', 'shear'}, 'Location', 'northeast', ...
    'Orientation', 'horizontal');
title(ax, ['edge sensors ' mlbl], 'FontWeight', 'normal');

ax = nexttile(tl, 17, [1 4]);
if wfc
    semilogy(ax, tv, max(ru, 1e-3), '-o', 'LineWidth', 1.4, ...
        'MarkerSize', 4, 'Color', [0.75 0.25 0.2]);
    hold(ax, 'on');
    semilogy(ax, tv, max(rc, 1e-3), '-o', 'LineWidth', 1.4, ...
        'MarkerSize', 4, 'Color', [0.2 0.45 0.75], ...
        'MarkerFaceColor', [0.2 0.45 0.75]);
    legend(ax, {'uncorrected', 'corrected'}, 'Location', 'best');
else
    plot(ax, tv, rc, '-o', 'LineWidth', 1.4, 'MarkerSize', 4, ...
        'Color', [0.2 0.45 0.75], 'MarkerFaceColor', [0.2 0.45 0.75]);
end
grid(ax, 'on');  xlim(ax, [0, max(t_end, eps)]);
reset_mark_(ax, reset_times, tv(end));
xlabel(ax, 't (s)');  ylabel(ax, 'rms WFE (nm)');
title(ax, 'accumulating rms WFE', 'FontWeight', 'normal');

% Strehl at the center wavelength only (the broadband trace is dropped
% -- peak-ratio Strehl can read marginally >1 from sub-pixel psf-peak
% sampling, which is confusing; Dave 2026-07-21).  The COMPOSE
% broadband psf itself is still shown in the top-right panel.
ax = nexttile(tl, 21, [1 4]);
hold(ax, 'on');
if wfc
    plot(ax, tv, su, '-o', 'LineWidth', 1.4, 'MarkerSize', 4, ...
        'Color', [0.75 0.25 0.2]);
end
plot(ax, tv, sc, '-o', 'LineWidth', 1.4, 'MarkerSize', 4, ...
    'Color', [0.2 0.45 0.75], 'MarkerFaceColor', [0.2 0.45 0.75]);
grid(ax, 'on');  xlim(ax, [0, max(t_end, eps)]);  ylim(ax, [0, 1.05]);
reset_mark_(ax, reset_times, tv(end));
xlabel(ax, 't (s)');  ylabel(ax, 'Strehl (\lambda_0)');
if wfc, legend(ax, {'uncorrected', 'corrected'}, 'Location', 'best'); end
title(ax, 'Strehl vs time (peak ratio to nominal)', 'FontWeight', 'normal');
end

function reset_mark_(ax, reset_times, t_now)
%RESET_MARK_  Dashed vertical markers at the WF-maintenance recontrol times.
for rt = reset_times(:).'
    if rt <= t_now + eps
        xl = xline(ax, rt, '--', 'Color', [0.5 0.2 0.6], 'LineWidth', 1.1);
        xl.Annotation.LegendInformation.IconDisplayStyle = 'off';  % no 'data1'
    end
end
end

function opd_panel_(ax, D, ~, ttl, xl)
% Per-frame AUTOSCALE (Dave 2026-07-21): a fixed scale shared with the
% uncorrected (100 um) panel washes out the corrected map's nm-scale
% structure -- each panel scales to its own frame's range instead (the
% colorbar carries the value).
im = imagesc(ax, D * 1e9);  set(im, 'AlphaData', isfinite(D));
axis(ax, 'image');  set(ax, 'XTick', [], 'YTick', []);
cm = max(abs(D(isfinite(D))) * 1e9);  if isempty(cm) || cm == 0, cm = 1; end
clim(ax, [-cm, cm]);  colormap(ax, parula);
cb = colorbar(ax);  cb.Label.String = '\DeltaOPD (nm)';
title(ax, ttl, 'FontWeight', 'normal');
xlabel(ax, xl);                               % running WFE (Dave)
end

function [B, colmap, nochan] = dwdx_cols_(Bf, cn, bodies)
%DWDX_COLS_  Sensed bodies' 6-DOF columns in [rot|trans] order (zero
%   columns + NOCHAN id for bodies the harvest has no channels for).
DOFN = {'Rx', 'Ry', 'Rz', 'Tx', 'Ty', 'Tz'};
nb = numel(bodies);
B = zeros(size(Bf, 1), 6*nb);
colmap = zeros(1, 6*nb);
nochan = [];
for b = 1:nb
    for d = 1:6
        i = find(strcmp(cn, sprintf('Elt %d %s', bodies(b), DOFN{d})), 1);
        if isempty(i)
            nochan(end+1) = bodies(b); %#ok<AGROW>
            break
        end
        B(:, (b-1)*6 + d) = Bf(:, i);
        colmap((b-1)*6 + d) = i;
    end
end
nochan = unique(nochan);
end

function M = met_struct_(met)
%MET_STRUCT_  Accept a run_met .mat path or an equivalent struct.
if isstruct(met), M = met;
else
    assert(isfile(met), 'run_simulator: met .mat %s not found', met);
    M = load(met);
end
for f = {'dedx', 'dldx', 'bodies', 'seg', 'met_in'}
    assert(isfield(M, f{1}), 'run_simulator: met struct lacks %s', f{1});
end
end

function J = jac_all_(jac)
%JAC_ALL_  Accept the s4 .mat (ox + optional oz/og), or the ox struct.
J = struct('ox', [], 'oz', [], 'og', []);
if isstruct(jac) && isfield(jac, 'ox')
    J.ox = jac.ox;
    if isfield(jac, 'oz'), J.oz = jac.oz; end
    if isfield(jac, 'og'), J.og = jac.og; end
elseif isstruct(jac) && isfield(jac, 'dwdxall')
    J.ox = jac;
else
    assert(isfile(jac), 'run_simulator: jac .mat %s not found', jac);
    S = load(jac);
    if isfield(S, 'ox'), J.ox = S.ox; end
    if isfield(S, 'oz'), J.oz = S.oz; end
    if isfield(S, 'og'), J.og = S.og; end
    assert(~isempty(J.ox), ...
        'run_simulator: no dw_dx_multi rigid-body output in %s', jac);
end
for f = {'per_field_dwdx', 'per_field_w_nom_2d', 'field_names'}
    assert(isfield(J.ox, f{1}), ...
        'run_simulator: jac lacks %s (re-harvest with the multi supervisor)', f{1});
end
if ~isempty(J.oz)
    assert(isfield(J.oz, 'per_field_dwdz'), 'run_simulator: oz lacks per_field_dwdz');
end
if ~isempty(J.og)
    assert(isfield(J.og, 'per_field_dwdg') && isfield(J.og, 'rx_path'), ...
        'run_simulator: og lacks per_field_dwdg/rx_path');
end
end
