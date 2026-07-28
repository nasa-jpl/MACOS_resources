% example_bs_angle_trade.m
% ========================================================================
%  USER-FACING: WHAT BEAM-SPLITTER ANGLE SHOULD *MY* BENCH USE?
% ========================================================================
%  Table constraints usually force a Twyman-Green's fold angle: a classic
%  layout turns the beam by 90 deg (BS AOI = 45 deg), a compressed layout
%  by ~45 deg (BS AOI = 22.5 deg).  Smaller BS angle BUYS polarization
%  performance -- the coated splitter's arm-differential diattenuation and
%  retardance fall roughly as AOI^2, raising fringe visibility -- and
%  COSTS mechanical clearance, because the test-arm excursion swings back
%  toward the incoming beam as the fold tightens.
%
%  This example sweeps the BS angle over a user-set range on the stock
%  Bench Twyman-Green, scores each angle three ways, and reports:
%    * the trade table (visibility cost, PSI pupil error, clearance);
%    * the KNEE -- the smallest AOI whose clearance still meets your
%      floor (the knee scales with YOUR beam radius: re-run, don't quote);
%    * your TARGET angle's row, with its polarization payoff vs 45 deg.
%
%  Every swept point is verified against a full-train textbook Fresnel
%  closed form (the "curve gate") -- if engine and textbook disagree the
%  example STOPS rather than print a wrong trade.  The development-grade
%  version of this study, with its non-vacuity checks and the fuller gate
%  set, is example_bench_ifo_pol_slice2.m; results + review trail in
%  macos/REVIEW_POL_IFO_SLICE2_2026-07-27.md.
%
%  EDIT THE "USER PARAMETERS" BLOCK, then:
%      cd ~/dev/MACOS_resources/mmacos/examples/design/bench_ifo_pol
%      matlab -batch "run('example_bs_angle_trade.m')"
% ========================================================================

addpath('~/dev/MACOS_resources/mmacos/src');
exdir = fileparts(mfilename('fullpath'));
if isempty(exdir), exdir = pwd; end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);

% ========================================================================
%  USER PARAMETERS -- edit these for your bench
% ========================================================================
FOLD_TARGET_DEG = 45;    % your table's beam fold (source-to-reflected angle);
                         % BS AOI = FOLD_TARGET_DEG/2
MIN_SEP_MM      = 10;    % clearance floor between beam envelopes (mm);
                         % same physicality role as the MET work's 50 mm rule
AOI_SWEEP       = [45 40 35 30 25 20 17.5 15];   % deg; target AOI is added
lam_nm          = 632.8; % HeNe
nAl = 1.45;  kAl = 7.54; thkAl = 2.0e-4;   % Al coating; thickness mm (opaque)
nG  = 1.5;               % BS / compensator glass index
MODEL = 256;  NGRID = 63;

% ------------------------------------------------------------------------
aoi_target = FOLD_TARGET_DEG/2;
aoi_list = sort(unique([AOI_SWEEP aoi_target]), 'descend');
naoi = numel(aoi_list);
Nal  = complex(nAl, -kAl);
coat = @(i) macos.coating(i, 'index', nAl, 'extinc', kAl, 'thickness', thkAl);

macos.init(MODEL);
fprintf('=== BS-ANGLE TRADE for your bench ===\n');
fprintf('target fold %.1f deg (BS AOI %.2f), clearance floor %.1f mm, %d angles\n\n', ...
    FOLD_TARGET_DEG, aoi_target, MIN_SEP_MM, naoi);

R.aoi = aoi_list(:);
[R.D, R.ret, R.vis_cost, R.psi_nm, R.clear_mm] = deal(nan(naoi,1));
R.traced_ok = false(naoi,1);
nref_t = NaN;  nref_r = NaN;

for ia = 1:naoi
    aoi = aoi_list(ia);
    fprintf('--- AOI %.2f deg (fold %.1f deg) ---\n', aoi, 2*aoi);

    % build + emit both arms at this BS angle
    try
        G = macos.design.twyman_green('BS_AOI', aoi, 'ngridpts', NGRID);
        ftest = sprintf('ex_trade_test_%02d.in', ia);
        fref  = sprintf('ex_trade_ref_%02d.in',  ia);
        G.bt.emit(ftest);  G.br.emit(fref);
    catch ME
        fprintf('  build infeasible below this AOI (%s) -- sweep floor\n', ME.identifier);
        break;
    end
    iBS_t = find(strcmp({G.bt.E.name}, 'BSrefl'),  1);
    iBS_r = find(strcmp({G.br.E.name}, 'BScrefr'), 1);
    iRC_t = G.T.iRC;  iRC_r = G.R.iRC;  iDET_t = G.T.iDET;

    % trace-clean: the emitted geometry must keep its rays before scoring
    macos.load_rx(ftest);  macos.polarization('off');
    tt = macos.trace(iDET_t);  ngood_t = tt.nRays;
    macos.load_rx(fref);   macos.polarization('off');
    tr = macos.trace(G.R.iDET);  ngood_r = tr.nRays;
    if isnan(nref_t), nref_t = ngood_t;  nref_r = ngood_r; end
    if ngood_t < 200 || ngood_r < 200 || ...
            min(ngood_t/nref_t, ngood_r/nref_r) < 0.90
        fprintf('  trace not clean (%d/%d good rays) -- skipped\n', ngood_t, ngood_r);
        continue;
    end
    R.traced_ok(ia) = true;

    % arm Jones pupils at recombination; differential M = J_test inv(J_ref)
    macos.load_rx(ftest);  coat(iBS_t);
    jt = macos.jones_pupil(iRC_t);
    macos.load_rx(fref);   coat(iBS_r);
    jr = macos.jones_pupil(iRC_r, 'axis', jt.axis, 'xref', jt.xref);
    mask = jt.mask & jr.mask;
    assert(nnz(mask) > 100, 'AOI %.1f: too few common rays', aoi);
    M  = arm_differential(jt, jr, mask);
    pm = macos.pol_maps(struct('J', M, 'mask', mask));
    R.D(ia) = pm.mean.D;  R.ret(ia) = pm.mean.ret;

    % CURVE GATE: engine vs full-train textbook closed form.  A failure
    % here means the install is broken -- do not trust the trade.
    [Da, reta] = analytic_diff(aoi, 1, nG, Nal);
    assert(abs(pm.mean.D - Da)/max(Da,eps) < 1e-2 && ...
           abs(pm.mean.ret - reta) < 1e-3, ...
        'CURVE GATE FAILED at AOI %.1f: engine disagrees with textbook', aoi);

    % (1) fringe visibility from the differential means
    V = sqrt(max(0, (1 + sqrt(max(0,1-pm.mean.D^2))*cos(pm.mean.ret))/2));
    R.vis_cost(ia) = 1 - V;
    % (2) PSI pupil-variation phase error (piston-removed co-pol fringe)
    R.psi_nm(ia) = psi_pupil_variation(jt, jr, mask, lam_nm);
    % (3) clearance: test-arm excursion vs the incoming beam envelope
    iOut = find(strcmp({G.bt.E.name}, 'BStxfo'), 1);
    macos.load_rx(ftest);  macos.polarization('off');
    macos.ray_hist('on');  tt = macos.trace(iDET_t);
    h = macos.ray_hist(tt.nRays);  macos.ray_hist('off');
    R.clear_mm(ia) = beam_clearance(h, iBS_t, iOut);

    fprintf('  D=%.4f ret=%.4f  1-V=%.2e  PSI %.1e nm  clearance %.1f mm\n', ...
        pm.mean.D, pm.mean.ret, R.vis_cost(ia), R.psi_nm(ia), R.clear_mm(ia));
end

% ========================================================================
%  THE TRADE, YOUR ANGLE, AND THE KNEE
% ========================================================================
ok = R.traced_ok & isfinite(R.clear_mm) & isfinite(R.vis_cost);
assert(nnz(ok) >= 3, 'too few scored angles for a trade');
i45 = find(ok & abs(R.aoi-45) < 1e-9, 1);
itg = find(ok & abs(R.aoi-aoi_target) < 1e-9, 1);
feasible = ok & (R.clear_mm >= MIN_SEP_MM);
iknee = find(feasible, 1, 'last');            % smallest feasible AOI (desc order)

fprintf('\n%6s %6s %8s %8s %10s %10s %10s  %s\n', 'AOI', 'fold', 'D', 'ret', ...
    '1-V', 'PSI(nm)', 'clear(mm)', '');
for ia = 1:naoi
    if ~ok(ia), continue; end
    tag = '';
    if ia == itg,   tag = ' <-- YOUR TARGET'; end
    if ia == iknee, tag = [tag ' <-- KNEE'];  end %#ok<AGROW>
    fprintf('%6.2f %6.1f %8.4f %8.4f %10.2e %10.2e %10.2f %s\n', ...
        R.aoi(ia), 2*R.aoi(ia), R.D(ia), R.ret(ia), R.vis_cost(ia), ...
        R.psi_nm(ia), R.clear_mm(ia), tag);
end

fprintf('\n');
if ~isempty(itg) && ~isempty(i45)
    fprintf('Your fold (%.1f deg): visibility cost %.2e (%.0fx better than 90-deg fold),\n', ...
        FOLD_TARGET_DEG, R.vis_cost(itg), R.vis_cost(i45)/R.vis_cost(itg));
    fprintf('  and pol-PSI coating-aliasing scales with ret: x%.2f of the 45-deg value.\n', ...
        R.ret(itg)/R.ret(i45));
    if R.clear_mm(itg) >= MIN_SEP_MM
        fprintf('  Clearance %.1f mm meets your %.1f mm floor: FEASIBLE.\n', ...
            R.clear_mm(itg), MIN_SEP_MM);
    else
        fprintf('  Clearance %.1f mm MISSES your %.1f mm floor: open the fold.\n', ...
            R.clear_mm(itg), MIN_SEP_MM);
    end
end
if ~isempty(iknee)
    fprintf('Knee for THIS bench: AOI %.1f deg (fold %.1f deg), cost %.2e at %.1f mm clearance.\n', ...
        R.aoi(iknee), 2*R.aoi(iknee), R.vis_cost(iknee), R.clear_mm(iknee));
    fprintf('The knee scales with beam radius -- re-run with your geometry, do not quote this one.\n');
end

% ---- figure + artifacts -------------------------------------------------
f = figure('Visible', 'off', 'Position', [100 100 760 640]);
subplot(2,1,1);
semilogy(R.aoi(ok), R.vis_cost(ok), 'o-', 'LineWidth', 1.2); hold on;
if ~isempty(itg), semilogy(R.aoi(itg), R.vis_cost(itg), 'rs', 'MarkerSize', 12, 'LineWidth', 1.5); end
if ~isempty(iknee), semilogy(R.aoi(iknee), R.vis_cost(iknee), 'kd', 'MarkerSize', 12, 'LineWidth', 1.5); end
grid on; xlabel('BS AOI (deg)'); ylabel('visibility cost 1-V');
legend({'swept', 'your target', 'knee'}, 'Location', 'northwest');
title('Polarization visibility cost vs BS angle (coated Al splitter)');
subplot(2,1,2);
plot(R.aoi(ok), R.clear_mm(ok), 'o-', 'LineWidth', 1.2); hold on;
yline(MIN_SEP_MM, '--', sprintf('floor %.0f mm', MIN_SEP_MM), ...
    'LabelHorizontalAlignment', 'left', 'HandleVisibility', 'off');
if ~isempty(itg), plot(R.aoi(itg), R.clear_mm(itg), 'rs', 'MarkerSize', 12, 'LineWidth', 1.5); end
grid on; xlabel('BS AOI (deg)'); ylabel('beam-envelope clearance (mm)');
title('Mechanical clearance vs BS angle');
print(f, '-dpng', '-r120', 'bs_angle_trade.png');
save('bs_angle_trade_results.mat', 'R', 'FOLD_TARGET_DEG', 'MIN_SEP_MM', ...
    'lam_nm', 'nAl', 'kAl', 'thkAl', 'nG');
fprintf('\nwrote bs_angle_trade.png + bs_angle_trade_results.mat\n');

% ========================================================================
%  local functions -- these MIRROR example_bench_ifo_pol_slice2.m (kept
%  self-contained per the example convention, cf. tJonesPupil's zern_;
%  if a third consumer appears, promote to private/).
% ========================================================================
function [D, ret] = analytic_diff(aoi_deg, dtp, nG, Nal)
%ANALYTIC_DIFF  Full-train closed-form arm-differential (Born & Wolf).
    N1  = 1.0;
    th  = deg2rad(aoi_deg);
    thg = asin(sin(th)/nG);
    [rse, rpe] = fresnel_rt(N1, Nal, th);
    [rsi, rpi] = fresnel_rt(nG,  Nal, thg);
    [~,~, tsag, tpag] = fresnel_rt(N1, nG, th);
    [~,~, tsga, tpga] = fresnel_rt(nG, N1, thg);
    Ms = (tsag*tsga)^dtp * (rse/rsi);
    Mp = (tpag*tpga)^dtp * (rpe/rpi);
    D  = abs(abs(Ms)^2 - abs(Mp)^2)/(abs(Ms)^2 + abs(Mp)^2);
    ret = abs(angle(Ms) - angle(Mp));  ret = min(ret, 2*pi - ret);
end

function [r_s, r_p, t_s, t_p] = fresnel_rt(N1, N2, th1)
%FRESNEL_RT  Textbook amplitude coefficients (ray-following p-hat).
    c1 = cos(th1);
    s2 = (N1/N2)*sin(th1);
    c2 = sqrt(1 - s2.^2);
    r_s = (N1*c1 - N2*c2)./(N1*c1 + N2*c2);
    r_p = (N2*c1 - N1*c2)./(N2*c1 + N1*c2);
    t_s = 2*N1*c1./(N1*c1 + N2*c2);
    t_p = 2*N1*c1./(N2*c1 + N1*c2);
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
    t11=jt.J(:,:,1,1); t21=jt.J(:,:,2,1);
    r11=jr.J(:,:,1,1); r21=jr.J(:,:,2,1);
    fringe = conj(t11).*r11 + conj(t21).*r21;
    phi = angle(fringe(mask));
    phi = phi - angle(mean(exp(1i*phi)));
    phi = mod(phi + pi, 2*pi) - pi;
    psi_nm = std(phi)/(2*pi)*lam_nm;
end

function [clr, cnode] = beam_clearance(h, iBS, iOut)
%BEAM_CLEARANCE  Min beam-envelope separation, excursion vs incoming beam.
    P = h.P;  ok = h.ok;  nNode = size(P,3);
    c = nan(3, nNode);  rho = zeros(1, nNode);
    for k = 1:nNode
        sel = ok(:,k);
        if nnz(sel) < 10, continue; end
        Pk = squeeze(P(:, sel, k));
        ck = mean(Pk, 2);  c(:,k) = ck;
        rho(k) = max(sqrt(sum((Pk - ck).^2, 1)));
    end
    src = c(:,1);  bs = c(:, iBS+1);
    rho_in = rho(iBS);
    clr = inf;  cnode = 0;
    for k = (iBS+1):(iOut-1)
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
