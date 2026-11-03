% example_bench_ifo_pol_slice2.m
% ========================================================================
%  BENCH IFO POLARIZATION: BS-AOI vs MECHANICAL-CLEARANCE TRADE (SLICE 2/3)
% ========================================================================
%  Slice 1 established, at the canonical 45-deg fold, a polarization-honest
%  Twyman-Green whose coated beam-splitter produces an arm-differential
%  diattenuation D=0.0721 / retardance=0.0835 rad -- a real common-mode
%  asymmetry (external air->Al reflect in the test arm vs internal glass->Al
%  in the reference arm), verified against a full-train textbook Fresnel
%  closed form to all digits.
%
%  Slice 2 sweeps the BS angle DOWN from 45 deg and asks what smaller AOI
%  BUYS (less polarization diattenuation/retardance -> higher fringe
%  visibility) against what it COSTS (a larger fold turn crowds the beams --
%  a mechanical-clearance penalty the ray trace itself cannot see).  The
%  deliverable is the trade CURVE and its knee, not a prior.
%
%  THREE scores per AOI:
%   (1) fringe VISIBILITY from the arm-differential means
%       V = sqrt((1 + sqrt(1-D^2) cos(ret))/2), the worst-case (45-deg-to-
%       axes) polarization fringe contrast set by the differential D/ret.
%       Reported as visibility, not raw D/ret.
%   (2) the pupil-variation PSI phase error (co-pol fringe, piston removed).
%       Expected to stay near round-off while the recombining beams are
%       collimated and common-path -- CONFIRMING that is a result.
%   (3) mechanical clearance: the minimum beam-ENVELOPE separation between
%       the incoming front-end bundle and the folded test-arm bundle
%       (MIN_SEP style, from macos.ray_hist -- same physicality standard as
%       the MET launcher work).  Falls as the fold tightens.
%
%  GATE (curve gate, extends slice-1 Gate 1 to the whole sweep): at every
%  AOI the engine's mean D and retardance must match the full-train closed
%  form -- external-reflect x net-transit-pair (test) vs transit-pair x
%  internal-reflect (ref).  The single NET glass transit pair is load-
%  bearing: dropping it shifts D by ~78% (the transit pair is itself a
%  strong s/p diattenuator; a one-reflection model is wrong).  NON-VACUITY:
%  one deliberately mis-modeled point (drop the transit pair) is asserted to
%  FAIL the same gate.
%
%  Run: cd ~/dev/MACOS_resources/mmacos/templates/90_polarization/bench_ifo_pol
%       matlab -batch "run('example_bench_ifo_pol_slice2.m')"
% ========================================================================

addpath('~/dev/MACOS_resources/mmacos/src');
exdir = fileparts(mfilename('fullpath'));
if isempty(exdir), exdir = pwd; end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);

MODEL = 256;
macos.init(MODEL);

% ---- parameters ---------------------------------------------------------
% Al coating at 632.8 nm (HeNe); Bench BaseUnits = mm, so an optically-
% thick physical layer thickness is given in mm (matches tJonesPupil).
nAl  = 1.45;  kAl = 7.54;  thkAl = 2.0e-4;   % mm
nG   = 1.5;                                   % BS/compensator glass index
Nal  = complex(nAl, -kAl);
lam_nm = 632.8;                               % HeNe
NGRID  = 63;

% AOI sweep: 45 deg down.  Below ~15 deg the fold turn exceeds 150 deg and
% the returning beam runs back over the front end; the clearance score is
% designed to capture exactly that collapse.
aoi_list = [45 42.5 40 37.5 35 32.5 30 27.5 25 22.5 20 17.5 15];
naoi = numel(aoi_list);

coat = @(i) macos.coating(i, 'index', nAl, 'extinc', kAl, 'thickness', thkAl);

% MIN_SEP physicality reference (NOT a prior on the answer -- a stated
% datum, the same role the 50 mm rule plays in the MET work).  The
% collimated beam radius here is ~12 mm; two bundles are "clear" when their
% envelopes leave a comparable gap.
MIN_SEP_REF = 10.0;   % mm

fprintf('=== IFO POLARIZATION SLICE 2: BS-AOI vs CLEARANCE TRADE ===\n');
fprintf('Model %d, ngrid %d, Al n=%.2f k=%.2f thk=%.1f nm, %d AOI points\n\n', ...
    MODEL, NGRID, nAl, kAl, thkAl*1e6, naoi);

% ---- result accumulators ------------------------------------------------
R.aoi        = aoi_list(:);
R.aoi_meas   = nan(naoi,1);   % engine-measured mean AOI at the BS
R.traced_ok  = false(naoi,1);
R.nref_test  = NaN;  R.nref_ref = NaN;   % 45-deg successful-ray reference
R.D_eng      = nan(naoi,1);   R.ret_eng   = nan(naoi,1);
R.D_var      = nan(naoi,1);   R.ret_var   = nan(naoi,1);
R.D_ana      = nan(naoi,1);   R.ret_ana   = nan(naoi,1);
R.gate_curve = false(naoi,1);
R.g1_mag     = nan(naoi,1);   R.g1_ph     = nan(naoi,1);
R.vis        = nan(naoi,1);   R.vis_cost  = nan(naoi,1);
R.vis_ana    = nan(naoi,1);
R.psi_nm     = nan(naoi,1);
R.clear_mm   = nan(naoi,1);

% =========================================================================
%  SWEEP
% =========================================================================
for ia = 1:naoi
    aoi = aoi_list(ia);
    fprintf('--- AOI %.1f deg (fold turn %.1f deg) ---\n', aoi, 180-2*aoi);

    % ---- build + emit both arms at this BS angle ------------------------
    % The compensator normal, BS-substrate transit faces, and the internal-
    % reflect return all key off the shared bs token / recomputed chief, so
    % they track the BS angle automatically; only the reflect turn is set by
    % BS_AOI.  A grazing-internal geometry can throw in add_bs_transmit
    % (refract TIR / degenerate crossing) -- caught here as the sweep floor.
    try
        G = macos.design.twyman_green('BS_AOI', aoi, 'ngridpts', NGRID);
        G.bt.emit(sprintf('s2_test_%02d.in', ia));
        G.br.emit(sprintf('s2_ref_%02d.in',  ia));
    catch ME
        fprintf('  BUILD FAILED (%s) -- geometry infeasible below this AOI\n', ME.identifier);
        break;
    end
    ftest = sprintf('s2_test_%02d.in', ia);  fref = sprintf('s2_ref_%02d.in', ia);
    iBS_t = find(strcmp({G.bt.E.name}, 'BSrefl'),  1);
    iBS_r = find(strcmp({G.br.E.name}, 'BScrefr'), 1);
    iRC_t = G.T.iRC;  iRC_r = G.R.iRC;  iDET_t = G.T.iDET;  iDET_r = G.R.iDET;

    % ---- trace-clean gate: emitted geometry must trace before scoring ---
    % nRays from macos.trace is the SUCCESSFUL ray count.  A re-emitted rig
    % whose folded beam clips or walks off drops it; require both arms keep
    % >=90% of the 45-deg reference (captured on the first scored AOI).
    macos.load_rx(ftest);  macos.polarization('off');
    tt = macos.trace(iDET_t);  ngood_t = tt.nRays;
    macos.load_rx(fref);   macos.polarization('off');
    tr = macos.trace(iDET_r);  ngood_r = tr.nRays;
    if isnan(R.nref_test), R.nref_test = ngood_t;  R.nref_ref = ngood_r; end
    frac = min(ngood_t/R.nref_test, ngood_r/R.nref_ref);
    fprintf('  trace-clean: test %d / ref %d good rays (%.1f%% of 45-deg ref)\n', ...
        ngood_t, ngood_r, 100*frac);
    if ngood_t < 200 || ngood_r < 200 || frac < 0.90
        fprintf('  TRACE NOT CLEAN -- skipping this AOI\n');
        continue;
    end
    R.traced_ok(ia) = true;

    % ---- coated single-surface Fresnel + measured AOI (per-angle Gate 1)-
    % Confirms the coating machinery tracks the BS angle: the engine s/p
    % amplitudes at the coated reflection match the textbook bare-interface
    % Fresnel coefficients at the ACTUAL AOI.  (Reference incident field
    % taken just before the BS, per slice-1 finding 2.)
    macos.load_rx(ftest);  coat(iBS_t);
    macos.polarization('on', 'Ex', [1/sqrt(2) 0], 'Ey', [1/sqrt(2) 0]);
    macos.trace(iBS_t-1);  ri = macos.ray_field(iBS_t-1);
    macos.trace(iBS_t);    rf = macos.ray_field(iBS_t);
    m = (rf.status == 0) & (ri.status == 0);
    [g1m, g1p, aoi_meas] = fresnel_gate(rf, ri, m, N1_(), Nal);
    R.g1_mag(ia) = g1m;  R.g1_ph(ia) = g1p;  R.aoi_meas(ia) = aoi_meas;
    fprintf('  Gate1 (coating vs Fresnel): AOI_meas %.3f, |RS/RP| resid %.2e, phase %.2e\n', ...
        aoi_meas, g1m, g1p);
    assert(g1m < 1e-11 && g1p < 1e-11, ...
        'AOI %.1f: coating s/p disagree with textbook Fresnel', aoi);

    % ---- arm Jones pupils at recombination, differential M --------------
    macos.load_rx(ftest);  coat(iBS_t);
    jt = macos.jones_pupil(iRC_t);
    macos.load_rx(fref);   coat(iBS_r);
    jr = macos.jones_pupil(iRC_r, 'axis', jt.axis, 'xref', jt.xref);
    mask = jt.mask & jr.mask;
    assert(nnz(mask) > 100, 'AOI %.1f: too few common rays at recomb', aoi);

    M = arm_differential(jt, jr, mask);       % M = J_test * inv(J_ref)
    pm = macos.pol_maps(struct('J', M, 'mask', mask));
    R.D_eng(ia)   = pm.mean.D;    R.D_var(ia)   = pm.var_rms.D;
    R.ret_eng(ia) = pm.mean.ret;  R.ret_var(ia) = pm.var_rms.ret;
    fprintf('  differential: D=%.5f (var %.1e), ret=%.5f rad (var %.1e)\n', ...
        pm.mean.D, pm.var_rms.D, pm.mean.ret, pm.var_rms.ret);

    % ---- CURVE GATE: engine mean D/ret vs full-train closed form --------
    [Da, reta, ~, ~, Va] = analytic_diff(aoi_meas, 1, nG, Nal);   % dtp = 1
    R.D_ana(ia) = Da;  R.ret_ana(ia) = reta;  R.vis_ana(ia) = Va;
    dD  = abs(pm.mean.D   - Da)/max(Da,eps);
    dR  = abs(pm.mean.ret - reta);
    R.gate_curve(ia) = (dD < 1e-2) && (dR < 1e-3);
    fprintf('  curve gate: closed-form D=%.5f ret=%.5f -> relD %.2e, absRet %.2e  [%s]\n', ...
        Da, reta, dD, dR, tern(R.gate_curve(ia), 'PASS', 'FAIL'));
    assert(R.gate_curve(ia), ...
        'CURVE GATE FAILED at AOI %.1f: engine D/ret disagree with closed form', aoi);

    % ---- (1) fringe visibility from the arm-differential means ----------
    D = pm.mean.D;  ret = pm.mean.ret;
    V = sqrt(max(0, (1 + sqrt(max(0,1-D^2))*cos(ret))/2));
    R.vis(ia) = V;  R.vis_cost(ia) = 1 - V;
    fprintf('  (1) fringe visibility V=%.6f  cost 1-V=%.2e  (closed-form V=%.6f)\n', ...
        V, 1-V, Va);

    % ---- (2) PSI pupil-variation phase error (co-pol fringe) ------------
    psi_nm = psi_pupil_variation(jt, jr, mask, lam_nm);
    R.psi_nm(ia) = psi_nm;
    fprintf('  (2) PSI pupil-variation phase error: %.3e nm @ %.1f nm\n', psi_nm, lam_nm);

    % ---- (3) mechanical clearance: min beam-envelope separation ---------
    % The test-arm EXCURSION (compensator -> test optic -> return) swings
    % back toward the incoming front-end beam as the fold turn grows.  The
    % output port (recomb/L2/detector) always crosses near the BS by design
    % and is NOT a crowding concern, so it is excluded.  Clearance = min
    % over excursion nodes of [perpendicular distance to the incoming
    % source->BS beam segment] minus the two beam-envelope radii (MIN_SEP).
    iTO  = find(strcmp({G.bt.E.name}, 'TestOptic'), 1);
    iOut = find(strcmp({G.bt.E.name}, 'BStxfo'),    1);   % start of output port
    macos.load_rx(ftest);  macos.polarization('off');
    macos.ray_hist('on');  tt = macos.trace(iDET_t);
    h = macos.ray_hist(tt.nRays);
    macos.ray_hist('off');
    [clr, cnode] = beam_clearance(h, iBS_t, iOut);
    R.clear_mm(ia) = clr;
    cname = '';
    if cnode >= 1 && cnode <= numel(G.bt.E), cname = G.bt.E(cnode).name; end
    fprintf('  (3) beam-envelope clearance to incoming beam: %.2f mm (closest excursion elt %d:%s)\n\n', ...
        clr, cnode, cname);
end

% =========================================================================
%  NON-VACUITY: the curve gate must REJECT a mis-modeled (drop-pair) point
% =========================================================================
% Pick the anchored 45-deg point; recompute the closed form with the net
% transit pair DROPPED (dtp=0, the wrong one-reflection model) and assert
% the engine's measured D does NOT match it.
fprintf('=== NON-VACUITY CHECK (drop the transit pair) ===\n');
ia0 = find(R.traced_ok & isfinite(R.D_eng), 1);
assert(~isempty(ia0), 'non-vacuity: no scored AOI available');
[Dbad, retbad] = analytic_diff(R.aoi_meas(ia0), 0, nG, Nal);   % dtp = 0
dD_bad = abs(R.D_eng(ia0) - Dbad)/max(Dbad,eps);
fprintf('  AOI %.1f: engine D=%.5f vs drop-pair closed form D0=%.5f -> relD %.2f\n', ...
    R.aoi_meas(ia0), R.D_eng(ia0), Dbad, dD_bad);
assert(dD_bad > 0.5, ...
    'NON-VACUITY FAILED: engine matches the drop-pair model -> gate has no teeth');
fprintf('  drop-pair model MISSES by %.0f%% and would FAIL the curve gate -> gate is non-vacuous\n\n', ...
    100*dD_bad);

% =========================================================================
%  TRADE CURVE + KNEE
% =========================================================================
ok = R.traced_ok & isfinite(R.clear_mm) & isfinite(R.vis_cost);
aoi_ok = R.aoi(ok);  cost_ok = R.vis_cost(ok);  clr_ok = R.clear_mm(ok);
% knee = smallest AOI whose clearance still meets the physicality floor;
% the polarization payoff is the visibility cost achieved there.
feasible = clr_ok >= MIN_SEP_REF;
if any(feasible)
    [~, kk] = min(aoi_ok(feasible));
    fa = aoi_ok(feasible);  fc = cost_ok(feasible);
    knee_aoi  = fa(kk);  knee_cost = fc(kk);
else
    knee_aoi = NaN;  knee_cost = NaN;
end
fprintf('=== TRADE SUMMARY ===\n');
fprintf(' AOI    D_eng    ret_eng   viscost    clear(mm)  gate\n');
for ia = 1:naoi
    if ~R.traced_ok(ia), continue; end
    fprintf(' %4.1f  %.5f  %.5f  %.3e  %8.2f   %s\n', R.aoi(ia), R.D_eng(ia), ...
        R.ret_eng(ia), R.vis_cost(ia), R.clear_mm(ia), tern(R.gate_curve(ia),'ok','--'));
end
fprintf('\nKNEE (clearance floor %.0f mm): AOI %.4g deg, visibility cost %.3e\n', ...
    MIN_SEP_REF, knee_aoi, knee_cost);
fprintf('Reading: visibility cost falls monotonically as AOI drops (the\n');
fprintf('polarization payoff of smaller AOI); clearance falls with it (the\n');
fprintf('mechanical price of the larger fold).  The knee is the smallest AOI\n');
fprintf('that still clears -- below it the folded arm crowds the front end.\n\n');

% ---- trade-curve figure -------------------------------------------------
try
    f = figure('Color','w','Position',[100 100 760 560]);
    tiledlayout(f,2,1,'TileSpacing','compact','Padding','compact');

    nexttile; hold on; grid on;
    yyaxis left
    plot(aoi_ok, cost_ok, '-o','LineWidth',1.4,'MarkerFaceColor','auto');
    ylabel('fringe-visibility cost  1-V');  set(gca,'YScale','log');
    yyaxis right
    plot(aoi_ok, R.ret_eng(ok), '-s','LineWidth',1.0);
    ylabel('differential retardance (rad)');
    xlabel('BS angle of incidence (deg)');
    title('Polarization payoff of smaller AOI (curve-gated to the closed form)');

    nexttile; hold on; grid on;
    plot(aoi_ok, clr_ok, '-o','LineWidth',1.4,'MarkerFaceColor','auto');
    yline(MIN_SEP_REF, '--', sprintf('MIN\\_SEP ref = %.0f mm', MIN_SEP_REF), ...
        'LabelHorizontalAlignment','left');
    if isfinite(knee_aoi)
        xline(knee_aoi, ':', sprintf('knee %.3g deg', knee_aoi), ...
            'Color',[0.85 0.33 0.1], 'LabelVerticalAlignment','bottom');
    end
    xlabel('BS angle of incidence (deg)');  ylabel('min beam-envelope clearance (mm)');
    title('Mechanical clearance price of the larger fold');
    set(gca,'XDir','reverse');   % sweep direction: 45 -> smaller, left to right
    ax = flipud(findall(f,'Type','Axes'));
    for a = ax(:)', set(a,'XDir','reverse'); end

    saveas(f, 'bench_ifo_pol_slice2_trade.png');
    fprintf('Saved bench_ifo_pol_slice2_trade.png\n');
catch ME
    fprintf('(figure skipped: %s)\n', ME.message);
end

% ---- persist ------------------------------------------------------------
results = R;
results.MODEL = MODEL;  results.NGRID = NGRID;
results.nAl = nAl;  results.kAl = kAl;  results.thkAl = thkAl;  results.nG = nG;
results.MIN_SEP_REF = MIN_SEP_REF;
results.knee_aoi = knee_aoi;  results.knee_cost = knee_cost;
results.nonvacuity_relD = dD_bad;
save('bench_ifo_pol_slice2_results.mat', 'results');
fprintf('Saved bench_ifo_pol_slice2_results.mat\n');

fprintf('\n=== SLICE 2 COMPLETE: all curve gates pass, non-vacuity confirmed ===\n');

% =========================================================================
%  local functions
% =========================================================================
function v = N1_(), v = 1.0; end

function s = tern(c, a, b), if c, s = a; else, s = b; end, end

function [D, ret, Ms, Mp, V] = analytic_diff(aoi_deg, dtp, nG, Nal)
%ANALYTIC_DIFF  Full-train closed-form arm-differential Jones.
%   Planar fold: every glass face is parallel to the BS, so the whole train
%   is DIAGONAL in a common s/p basis and the differential reduces to two
%   complex ratios.  Test-arm coated path = external air->Al reflection;
%   ref-arm = internal glass->Al reflection.  Their glass OPL balances
%   (compensator), but the transmissive plates cross two air/glass
%   boundaries per thickness while the internal-reflect return crosses only
%   one per thickness -- leaving DTP net glass transit pairs in the
%   differential (DTP=1 physical).  Each transit pair is a real (zero-
%   retardance) s/p diattenuator and is LOAD-BEARING in D.  Written from
%   Born & Wolf, not the engine, so the r_p sign is pinned non-circularly.
    N1  = 1.0;
    th  = deg2rad(aoi_deg);
    thg = asin(sin(th)/nG);
    [rse, rpe] = fresnel_rt(N1, Nal, th);      % external air->Al reflect (test)
    [rsi, rpi] = fresnel_rt(nG,  Nal, thg);    % internal glass->Al reflect (ref)
    [~,~, tsag, tpag] = fresnel_rt(N1, nG, th);   % air -> glass  (into plate)
    [~,~, tsga, tpga] = fresnel_rt(nG, N1, thg);  % glass -> air  (out of plate)
    ts_pair = (tsag*tsga)^dtp;
    tp_pair = (tpag*tpga)^dtp;
    Ms = ts_pair * (rse/rsi);
    Mp = tp_pair * (rpe/rpi);
    D  = abs(abs(Ms)^2 - abs(Mp)^2)/(abs(Ms)^2 + abs(Mp)^2);
    ret = abs(angle(Ms) - angle(Mp));  ret = min(ret, 2*pi - ret);
    V  = sqrt(max(0,(1 + sqrt(max(0,1-D^2))*cos(ret))/2));
end

function [r_s, r_p, t_s, t_p] = fresnel_rt(N1, N2, th1)
%FRESNEL_RT  Textbook Born&Wolf amplitude coefficients (ray-following p).
    c1 = cos(th1);
    s2 = (N1/N2)*sin(th1);
    c2 = sqrt(1 - s2.^2);
    r_s = (N1*c1 - N2*c2)./(N1*c1 + N2*c2);
    r_p = (N2*c1 - N1*c2)./(N2*c1 + N1*c2);
    t_s = 2*N1*c1./(N1*c1 + N2*c2);
    t_p = 2*N1*c1./(N2*c1 + N1*c2);
end

function [g1m, g1p, aoi_deg] = fresnel_gate(rf, ri, m, N1, Nal)
%FRESNEL_GATE  Slice-1 single-surface 45-deg gate, generalized to any AOI.
%   Convention-independent RS/RP = (Es/Ep)(qp/qs) from the measured reflected
%   (rf) and incident (ri) fields, vs textbook bare-interface Fresnel.
    kox=rf.kx(m); koy=rf.ky(m); koz=rf.kz(m);
    nx=rf.nx(m);  ny=rf.ny(m);  nz=rf.nz(m);
    kd  = kox.*nx + koy.*ny + koz.*nz;
    kix = kox-2*kd.*nx;  kiy = koy-2*kd.*ny;  kiz = koz-2*kd.*nz;
    sx=kiy.*nz-kiz.*ny; sy=kiz.*nx-kix.*nz; sz=kix.*ny-kiy.*nx;
    sm=sqrt(sx.^2+sy.^2+sz.^2); sx=sx./sm; sy=sy./sm; sz=sz./sm;
    pix=sy.*kiz-sz.*kiy; piy=sz.*kix-sx.*kiz; piz=sx.*kiy-sy.*kix;
    prx=sy.*koz-sz.*koy; pry=sz.*kox-sx.*koz; prz=sx.*koy-sy.*kox;
    Es = rf.Ex(m).*sx  + rf.Ey(m).*sy  + rf.Ez(m).*sz;
    Ep = rf.Ex(m).*prx + rf.Ey(m).*pry + rf.Ez(m).*prz;
    qs = ri.Ex(m).*sx  + ri.Ey(m).*sy  + ri.Ez(m).*sz;
    qp = ri.Ex(m).*pix + ri.Ey(m).*piy + ri.Ez(m).*piz;
    ratio_meas = (Es./Ep).*(qp./qs);
    cthi = abs(kix.*nx + kiy.*ny + kiz.*nz);
    ctht = sqrt(1 - (N1/Nal)^2*(1 - cthi.^2));
    RPa = (Nal*cthi - N1*ctht)./(Nal*cthi + N1*ctht);
    RSa = (N1*cthi - Nal*ctht)./(N1*cthi + Nal*ctht);
    g1m = max(abs(abs(ratio_meas) - abs(RSa./RPa)));
    g1p = max(abs(angle(ratio_meas./(RSa./RPa))));
    aoi_deg = mean(acosd(cthi));
end

function M = arm_differential(jt, jr, mask)
%ARM_DIFFERENTIAL  M = J_test * inv(J_ref) per ray (2x2 vectorized).
    a = jr.J(:,:,1,1); b = jr.J(:,:,1,2); c = jr.J(:,:,2,1); d = jr.J(:,:,2,2);
    det = a.*d - b.*c;
    inv11 =  d./det; inv12 = -b./det; inv21 = -c./det; inv22 = a./det;
    t11 = jt.J(:,:,1,1); t12 = jt.J(:,:,1,2); t21 = jt.J(:,:,2,1); t22 = jt.J(:,:,2,2);
    M = complex(nan(size(jt.J)), nan(size(jt.J)));
    M(:,:,1,1) = t11.*inv11 + t12.*inv21;
    M(:,:,1,2) = t11.*inv12 + t12.*inv22;
    M(:,:,2,1) = t21.*inv11 + t22.*inv21;
    M(:,:,2,2) = t21.*inv12 + t22.*inv22;
    for aa=1:2, for bb=1:2
        Mab = M(:,:,aa,bb); Mab(~mask) = NaN+1i*NaN; M(:,:,aa,bb) = Mab;
    end, end
end

function psi_nm = psi_pupil_variation(jt, jr, mask, lam_nm)
%PSI_PUPIL_VARIATION  RMS pupil variation of the co-pol fringe phase.
%   x-pol input (a pure p-eigenstate in this planar fold), piston removed.
    t11=jt.J(:,:,1,1); t21=jt.J(:,:,2,1);
    r11=jr.J(:,:,1,1); r21=jr.J(:,:,2,1);
    ft1=t11; ft2=t21;                    % test field for e_in = (1,0)
    fr1=r11; fr2=r21;                    % ref  field for e_in = (1,0)
    fringe = conj(ft1).*fr1 + conj(ft2).*fr2;
    phi = angle(fringe(mask));
    phi = phi - angle(mean(exp(1i*phi)));
    phi = mod(phi + pi, 2*pi) - pi;
    psi_nm = std(phi)/(2*pi)*lam_nm;
end

function [clr, cnode] = beam_clearance(h, iBS, iOut)
%BEAM_CLEARANCE  Min beam-ENVELOPE separation between the test-arm EXCURSION
%   and the incoming source->BS beam (MIN_SEP style, from ray_hist).
%   Node 0 = source (slot 1), element k -> slot k+1.  The incoming beam is
%   the source->BS capsule (envelope radius = max ray offset along it).  The
%   excursion is every element STRICTLY between the BS and the output port
%   (iBS < k < iOut): compensator, test optic, return passes -- the nodes
%   that swing back toward the front end as the fold turn grows.  Reported =
%   min over excursion nodes of [dist(node, incoming segment) - rho_node -
%   rho_incoming].  Positive = clear; falls monotonically as AOI drops.
    P = h.P;  ok = h.ok;  nNode = size(P,3);
    c = nan(3, nNode);  rho = zeros(1, nNode);
    for k = 1:nNode
        sel = ok(:,k);
        if nnz(sel) < 10, continue; end
        Pk = squeeze(P(:, sel, k));
        ck = mean(Pk, 2);  c(:,k) = ck;
        rho(k) = max(sqrt(sum((Pk - ck).^2, 1)));
    end
    src = c(:,1);  bs = c(:, iBS+1);          % incoming beam: source -> BS
    % incoming beam radius = the COLLIMATED radius at the last pre-BS node
    % (L1pow), NOT the tilted BS footprint (which elongates with AOI and
    % would inflate the envelope); rho(iBS) is the L1pow node (elt iBS-1).
    rho_in = rho(iBS);
    clr = inf;  cnode = 0;
    for k = (iBS+1):(iOut-1)                   % element indices in the excursion
        node = c(:, k+1);
        if any(isnan(node)), continue; end
        gap = point_seg_dist(node, src, bs) - rho(k+1) - rho_in;
        if gap < clr, clr = gap;  cnode = k; end
    end
    if ~isfinite(clr), clr = NaN; end
end

function d = point_seg_dist(p, a, b)
%POINT_SEG_DIST  Distance from point P to segment [A B].
    ab = b - a;  t = dot(p - a, ab)/max(dot(ab,ab), 1e-12);
    t = min(max(t,0),1);
    d = norm(p - (a + t*ab));
end
