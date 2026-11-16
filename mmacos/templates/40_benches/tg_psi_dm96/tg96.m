function out = tg96()
%TG96  The optimized-plate gauge laid out for a Xinetics 96x96 DM.
%   Dave's tasking (2026-09-03): lay out the bench for the shallow-plate
%   option, ENFORCING CLEARANCES, for a 96x96-actuator Xinetics DM
%   (1.0 mm pitch -> 96 mm aperture, the aperture class where the cube
%   goes custom), then run the v2-equivalent analysis battery.
%
%   Stage A  clearance-driven layout solve: the smallest BS incidence
%            angle whose beam/body separations clear declared housing
%            half-widths with a 25 mm margin inside a 700 mm leg cap.
%            Margins print as NUMBERS per leg (clearance-gate rule).
%   Stage B  build the rig at the solved geometry.  The v1 rig scales
%            UNIFORMLY by s = 96/56 (every length x s, conics kept):
%            angles are preserved, so the tuned lens nulls carry over.
%   Stage C  battery: arm azimuths / departure, unaligned piston gain,
%            null, single-actuator poke, full 96x96 checkerboard closure.
%   Stage D  actuator-lattice calibration (the tg_widen protocol at
%            96x96): transfer curve to Nyquist = 48 cyc/pupil + held-out
%            random command.  The question this stage answers: can the
%            gauge resolve 1 mm actuators?
%
%   Writes tg96_report.txt, tg96_run.mat, tg96_layout.png,
%   tg96_transfer.png, tg96_closure.png.  Helpers copied verbatim from
%   tg_widen (the established self-contained pattern).
%   Run:  cd <this dir>;  matlab -batch "tg96"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
addpath(fullfile(exdir, '..', '..', '90_polarization', 'tg_psi_dm'));  % dm_influence_map (dir moved to 40_benches 2026-09-04)
cd(exdir);
rep = fopen('tg96_report.txt','w');
say = @(varargin) say_(rep, varargin{:});

% ---- the DM and the scale ----------------------------------------
NACT = 96;  PITCH = 1.0;  DM_MM = NACT*PITCH;      % 96 mm aperture
s    = DM_MM/56;                                   % uniform scale vs v1
LAM  = 6.328e-4;  QWP = 0.25;  THETAS = [0 45 90 135];
MODEL = 1024;  NGRID = 385;  N_G = 384;  DX_G = 0.28;  % 3.57 px/actuator
% NGRID 385 (Dave 2026-09-03, 'a detector capable of resolving the
% imagery, up to 1 Mpix square'): the wavefront seed inherits the ray
% grid's sampling, so the pupil image spans ~NGRID px at the detector.
% 63 px gave detector Nyquist 31.5 cyc/pupil -- BLIND at the 96x96 DM's
% 48 (the run of record for that finding is tg96_report_ng63.txt).
% 385 px -> ~4 px/actuator, Nyquist ~192 cyc/pupil, inside the 1024^2
% (1 Mpix) grid with FFT padding.
% MODEL 1024 is LOAD-BEARING: mGridMat is NOT the model size -- 512 caps
% grids at 256 (macos_param.txt) and the nGridMat= parse is unguarded
% (BRIEF_gridmat_guard.md), so a 384 grid on model 512 corrupts the heap.
POKE  = 50e-6;                                     % mm (50 nm commands)
say('=== TG96: optimized plate gauge for a Xinetics %dx%d (%.0f mm) ===\n', ...
    NACT, NACT, DM_MM);
say('uniform scale s = %.4f from the 56 mm v1 rig; model %d, grid %dx%.2f mm\n\n', ...
    s, MODEL, N_G, DX_G);

% ---- Stage A: clearance-driven layout solve ----------------------
%  At plate incidence AOI the two legs at the splitter separate at
%  2*AOI; each leg's body must clear the OTHER leg's beam:
%      L_leg * sind(2*AOI) >= beam_r + body_half + MARGIN
beam_r  = s*30;                                    % scaled R_TO_AP
HW_DM   = 90;    % Xinetics 96x96 housing half-width (declared estimate)
HW_REF  = 60;    % reference flat + PZT stage half-width
HW_CAM  = 50;    % detector assembly half-width
MARGIN  = 25;    % clearance spec, mm
LEG_CAP = 700;   % max leg length on the bench
legs = {'DM arm vs ref beam',       HW_DM;
        'ref arm vs DM beam',       HW_REF;
        'camera leg vs source beam',HW_CAM};
say('Stage A -- clearance solve (beam_r %.1f, margin %.0f, leg cap %.0f):\n', ...
    beam_r, MARGIN, LEG_CAP);
need = cellfun(@(h) beam_r + h + MARGIN, legs(:,2));
th_min = max(asind(need/LEG_CAP))/2;
AOI = ceil(th_min);
Lreq = need/sind(2*AOI);
say('  binding angle %.2f deg -> BS_AOI = %d deg\n', th_min, AOI);
for k = 1:size(legs,1)
    say('  %-27s need %6.1f mm sep -> leg >= %5.0f mm\n', legs{k,1}, need(k), Lreq(k));
end
D_BS_TO = ceil(max(Lreq(1), s*250)/50)*50;
say('  chosen DM leg %d mm; achieved separations and margins there:\n', D_BS_TO);
for k = 1:size(legs,1)
    m = D_BS_TO*sind(2*AOI) - (need(k) - MARGIN);
    say('  %-27s separation %6.1f mm, margin %+6.1f mm (spec >= %.0f)\n', ...
        legs{k,1}, D_BS_TO*sind(2*AOI), m, MARGIN);
end
say('\n');

% ---- Stage A2: sampling budget (Dave 2026-09-03: sampling is part of
%      the DESIGN -- size every interface in the signal chain against
%      the finest feature it must carry, and assert the margin) ------
act_nyq = NACT/2;                                  % cyc/pupil the DM commands
det_nyq = NGRID/2;                                 % pupil image spans ~NGRID px
grid_ppa = PITCH/DX_G;                             % surface-grid px per actuator
say('Stage A2 -- sampling budget (finest feature: actuator Nyquist %.0f cyc/pupil):\n', act_nyq);
say('  detector: ~%d px across pupil -> Nyquist %5.1f cyc/pup  margin %4.1fx (need >= 2)\n', ...
    NGRID, det_nyq, det_nyq/act_nyq);
say('  DM surface grid: %.2f px/actuator                        (need >= 3)\n', grid_ppa);
say('  diffraction grid: %d^2 for a %d-px image                (padding %4.1fx)\n\n', ...
    MODEL, NGRID, MODEL/NGRID);
assert(det_nyq >= 2*act_nyq, 'sampling budget: detector Nyquist %.1f < 2x actuator Nyquist %.1f', det_nyq, act_nyq);
assert(grid_ppa >= 3, 'sampling budget: %.2f grid px/actuator < 3', grid_ppa);
assert(MODEL >= 2*NGRID, 'sampling budget: grid %d gives < 2x padding on a %d-px image', MODEL, NGRID);

% ---- Stage B: build at the solved geometry -----------------------
macos.init(MODEL);
assert(N_G <= macos.grid_size_max(), 'N_G %d exceeds mGridMat %d at this model size', N_G, macos.grid_size_max());
% detector-tail parameters: the re-tuned set from tg96_tail.m when
% present (Dave 2026-09-04), else the geometrically scaled v1 winner.
T_FL_F = s*25.02100857;  T_FL_Kc = -2.11278288;
T_DMF  = s*6.277463741;  T_TRIM  = s*1.085330067;
if isfile('tg96_tail.mat')
    tl = load('tg96_tail.mat');
    T_FL_F = tl.out.FL_F;  T_FL_Kc = tl.out.FL_Kc;
    T_DMF  = tl.out.D_MASK_FL;  T_TRIM = tl.out.DET_TRIM;
    say('Tail: RE-TUNED set (null %.3f nm at opt res; seed was %.3f)\n', ...
        tl.out.null_nm, tl.out.seed_null_nm);
else
    say('Tail: geometrically scaled v1 set (tg96_tail.mat not found)\n');
end
macos.write_grid_file('tg96_flat.txt', zeros(N_G));
mk = @(gf) macos.design.twyman_green('polarizing',true, 'ngridpts',NGRID, ...
    'BS_AOI',AOI, ...
    'F1',s*500, 'F2',s*250, 'D_LENS',s*60, 'R_BAFFLE',s*12.5, 'D_SB',s*250, ...
    'BS_T',s*1.5, 'D_L1_BS',s*150, 'D_BS_TO',D_BS_TO, 'D_BS_CMP',s*100, ...
    'R_TO_AP',s*30, 'L1_Kr',s*236.866, 'L1_Kc',-0.5829, ...
    'L2_Kr',-s*124.076, 'L2_Kc',-0.5826, ...
    'to_grid_file',gf, 'to_grid_n',N_G, 'to_grid_dx',DX_G, ...
    'qwp_ret',QWP, 'pol_in_deg',45, 'qwp_test_deg',0, 'qwp_ref_deg',45, ...
    'out_qwp_deg',0, 'analyzer_deg',0, ...
    'tail_arch','fieldlens', 'FL_F',T_FL_F, 'FL_Kc',T_FL_Kc, ...
    'FL_D',s*12, 'D_MASK_FL',T_DMF, 'DET_TRIM',T_TRIM);
Gf = mk('tg96_flat.txt');
Gf.bt.emit('tg96_test.in');  Gf.br.emit('tg96_ref.in');
say('Stage B -- rig built at BS_AOI %d deg; emitted tg96_{test,ref}.in\n\n', AOI);

AT = arm_desc('tg96_test.in', Gf.bt, Gf.T, 0);
AR = arm_desc('tg96_ref.in',  Gf.br, Gf.R, 45);

% ---- Stage C: the v2-equivalent battery --------------------------
say('Stage C -- battery (design azimuths, unaligned):\n');
az_t = arm_azimuth(AT, QWP, 0);  az_r = arm_azimuth(AR, QWP, 45);
dep  = wrap180(az_t - az_r - 90);
say('  arm azimuths: test %+.4f, ref %+.4f -> departure %+.4f deg\n', az_t, az_r, dep);

Sr = analyzer_basis(AR, QWP, []);
S0 = analyzer_basis(AT, QWP, []);
I0 = frame(S0, Sr, 0);  msk = I0 > 0.1*max(I0(:));
p_null = fourstep(S0, Sr, THETAS);
h_null = (p_null - median(p_null(msk))) * LAM/(4*pi) * 1e6;
say('  null: %.4f nm rms surface (%.1f pm) with nothing aligned\n', ...
    std(h_null(msk)), 1e3*std(h_null(msk)));

dp = angle(exp(1i*(fourstep(analyzer_basis(AT,QWP,20e-6*ones(N_G)),Sr,THETAS) - p_null)));
gain = median(dp(msk))/(4*pi*20e-6/LAM);
say('  20 nm piston: |gain| %.5f (unaligned scale error %+.3f%%)\n', ...
    abs(gain), 100*(abs(gain)-1));

Mp = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, ...
                      'pattern','single', 'poke',150e-6);
hp = meas_surface(AT, QWP, Mp, Sr, p_null, THETAS, LAM);
say('  single actuator at 150 nm: recovered peak %.1f nm\n', 1e6*max(abs(hp(msk))));

% REGISTRATION BY TWO POKES (deterministic -- no correlation search).
% Runs 3-6 taught: the checkerboard's correlation landscape ripples at
% every integer actuator, so any refiner can park an arbitrary integer
% offset (run 6 measured 2.25 actuators); and a center poke cannot pick
% parity (it is parity-invariant).  So: ray affine -> scale/rotation;
% poke A (center, parity-invariant) -> translation EXACTLY by blob
% centroid match; poke B (off-center) -> parity, by direct overlap, and
% the selection metric doubles as the gate.
A2c = zeros(NACT);  A2c(30,64) = 1;
Mp2 = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', 150e-6*A2c);
hp2 = meas_surface(AT, QWP, Mp2, Sr, p_null, THETAS, LAM);
Mdm = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, ...
                       'pattern','checker', 'poke',POKE);
hm = meas_surface(AT, QWP, Mdm, Sr, p_null, THETAS, LAM);
axs = ((1:N_G)-(N_G+1)/2)*DX_G;
[map, reg] = register_two_pokes(AT, Gf.T, Mp, hp, Mp2, hp2, N_G, DX_G, msk);
say('  registration by two pokes: parity %d of 8, meas sign %+d; |pokeB corr| %.4f (runner-up %.4f)\n', ...
    reg.par, reg.sign, abs(reg.pokeB_corr), reg.runner_up);
say('  parity corr table: %s\n', sprintf('%.3f ', reg.table));
assert(abs(reg.pokeB_corr) >= 0.8, 'registration gate FAILED: |pokeB corr| %.3f', abs(reg.pokeB_corr));
assert(abs(reg.pokeB_corr) - reg.runner_up >= 0.3, 'registration gate FAILED: parity separation %.3f', ...
       abs(reg.pokeB_corr) - reg.runner_up);
hm = reg.sign * hm;                    % gauge sign calibration, applied
tt = interpn(axs, axs, Mdm, map.Xt, map.Yt, 'spline', 0);
hv = 1e6*(hm(msk) - mean(hm(msk)));  tv = 1e6*(tt(msk) - mean(tt(msk)));
cc = corrcoef(hv, tv);
say('  96x96 checkerboard closure: truth %.3f nm rms, measured %.3f, residual %.3f nm rms, corr %.6f\n', ...
    std(tv), std(hv), std(hv-tv), cc(1,2));
say('  registration: mag %.4f, anamorphism %.2f%%, nonlinearity %.4f mm\n\n', ...
    map.mag, map.anam_pct, map.nonlin_mm);
fig = figure('Visible','off','Position',[60 60 1100 420]);
subplot(1,2,1); imagesc(fullmap(hm,msk)); axis image off; colorbar;
title(sprintf('measured: %.2f nm rms', std(hv)));
subplot(1,2,2); imagesc(fullmap(tt,msk)); axis image off; colorbar;
title(sprintf('truth: %.2f nm rms  (residual %.3f nm)', std(tv), std(hv-tv)));
print(fig,'tg96_closure.png','-dpng','-r130');

% ---- Stage D: transfer curve to 48 cyc/pupil ---------------------
say('Stage D -- transfer curve on the 96x96 lattice (Nyquist 48 cyc/pupil):\n');
PQ = [1 1; 2 2; 4 4; 8 8; 16 16; 24 24; 32 32; 48 48; 64 64; 80 80; 96 96; 48 0];
nM = size(PQ,1);  ii = (1:NACT)';
Tt = zeros(nnz(msk), nM);  Mm = zeros(nnz(msk), nM);  frq = zeros(1,nM);
t0 = tic;
for k = 1:nM
    p = PQ(k,1); q = PQ(k,2);
    Ak = cos(pi*p*ii/NACT) * cos(pi*q*ii/NACT)';  Ak = Ak/max(abs(Ak(:)));
    Mk = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'act', POKE*Ak);
    hk = reg.sign * meas_surface(AT, QWP, Mk, Sr, p_null, THETAS, LAM);
    tk = interpn(axs, axs, Mk, map.Xt, map.Yt, 'spline', 0);
    Mm(:,k) = hk(msk) - mean(hk(msk));
    Tt(:,k) = tk(msk) - mean(tk(msk));
    frq(k)  = hypot(p,q)/2;
end
G = Tt \ Mm;
say('  %d modes x 3 traces in %.1f s\n', nM, toc(t0));
say('  %-9s %8s %8s %11s\n', 'mode(p,q)', 'cyc/pup', 'gain', 'cross-talk');
for k = 1:nM
    x = G(:,k); x(k) = 0;
    say('  %3d,%-5d %8.1f %8.4f %11.4f\n', PQ(k,1), PQ(k,2), frq(k), G(k,k), norm(x));
end
Mrnd = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, 'poke',POKE, ...
                        'pattern','random', 'seed',11);
hv2 = reg.sign * meas_surface(AT, QWP, Mrnd, Sr, p_null, THETAS, LAM);
tv2 = interpn(axs, axs, Mrnd, map.Xt, map.Yt, 'spline', 0);
hv2 = hv2(msk)-mean(hv2(msk));  tv2 = tv2(msk)-mean(tv2(msk));
c2  = G \ (Tt \ hv2);
hc2 = hv2 - Tt*(Tt\hv2) + Tt*c2;
r2  = corrcoef(hc2-tv2, tv2);
say('  held-out random: %.4f -> %.4f nm rms (input %.2f nm rms, resid/truth corr %+.2f)\n', ...
    1e6*std(hv2-tv2), 1e6*std(hc2-tv2), 1e6*std(tv2), r2(1,2));

% ---- Stage E: the DIFFERENTIAL metric (Dave 2026-09-04) -----------
%  The instrument's product is a DEVIATION: measure state B, measure
%  B+delta, difference the MEASUREMENTS, score against delta.  The
%  common systematic cancels to first order; what survives is its
%  dependence on the surface -- the number a ZWFS comparison needs.
say('Stage E -- differential metric (deviation measured about a working state):\n');
d_sng = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, ...
                         'pattern','single', 'poke',10e-6);
d_rnd = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, ...
                         'pattern','random', 'seed',23, 'poke',16e-6);
bases = {'flat', zeros(N_G), S0; 'random 30 nm', Mrnd, []};
devs  = {'single act 10 nm', d_sng; 'random 10 nm rms', d_rnd};
say('  %-14s %-18s %8s %10s %10s %8s\n', 'base', 'deviation', 'gain', 'resid nm', 'resid %', 'corr');
for bi = 1:2
    if isempty(bases{bi,3}), Sb = analyzer_basis(AT, QWP, bases{bi,2});
    else, Sb = bases{bi,3}; end
    p_B = fourstep(Sb, Sr, THETAS);
    for di = 1:2
        Sd = analyzer_basis(AT, QWP, bases{bi,2} + devs{di,2});
        dmeas = reg.sign * angle(exp(1i*(fourstep(Sd,Sr,THETAS) - p_B))) * LAM/(4*pi);
        dt = interpn(axs, axs, devs{di,2}, map.Xt, map.Yt, 'spline', 0);
        a = 1e6*(dmeas(msk)-mean(dmeas(msk)));  b = 1e6*(dt(msk)-mean(dt(msk)));
        g = b\a;  res = std(a - g*b);  cco = corrcoef(a, b);
        say('  %-14s %-18s %8.4f %10.4f %9.1f%% %8.4f\n', ...
            bases{bi,1}, devs{di,1}, g, res, 100*res/std(b), cco(1,2));
    end
end

fig = figure('Visible','off','Position',[60 60 640 430]);
dg = zeros(1,11);  for k=1:11, dg(k) = G(k,k); end
plot(frq(1:11), dg, 'o-', 'LineWidth',1.6); grid on;
xlabel('spatial frequency (cycles across the pupil)'); ylabel('measured gain');
title('TG96 transfer curve: can the gauge resolve 1 mm actuators?');
print(fig,'tg96_transfer.png','-dpng','-r140');

draw_layout(AOI, D_BS_TO, beam_r, HW_DM, HW_REF, HW_CAM, s);
out = struct('AOI',AOI, 'D_BS_TO',D_BS_TO, 's',s, 'dep',dep, 'gain',gain, ...
             'null_nm',std(h_null(msk)), 'frq',frq, 'PQ',PQ, 'G',G, ...
             'closure_resid_nm',std(hv-tv), 'closure_corr',cc(1,2));
save('tg96_run.mat','out');
say('\nwrote tg96_report.txt + tg96_run.mat + figures\n');
fclose(rep);
end

% ---- local: display + layout -------------------------------------
function say_(rep, varargin)
    fprintf(1, varargin{:});  fprintf(rep, varargin{:});
end

function Z = fullmap(h, msk)
    Z = nan(size(msk));  Z(msk) = h(msk) - mean(h(msk));  Z = 1e6*Z;
end

function draw_layout(AOI, L_dm, br, hwdm, hwref, hwcam, s)
    f = figure('Visible','off','Position',[40 40 940 640]); hold on; axis equal;
    a = 2*AOI;                     % angle between transmitted and reflected legs
    L_ref = max(300, s*100+150);  L_out = max(350, s*200+150);  L_in = 400;
    Pdm  = L_dm *[cosd(a) sind(a)];
    Pref = [L_ref 0];
    Pin  = [-L_in 0];
    Pout = -L_out*[cosd(a) sind(a)];
    drawbeam(Pin,  [0 0], br, [0.85 0.45 0.2]);
    drawbeam([0 0], Pref, br, [0.85 0.45 0.2]);
    drawbeam([0 0], Pdm,  br, [0.25 0.45 0.8]);
    drawbeam([0 0], Pout, br, [0.35 0.65 0.35]);
    drawbody(Pdm,  hwdm, 40, [0.75 0.82 1.0]);  text(Pdm(1)-40, Pdm(2)+55, 'DM 96\times96');
    drawbody(Pref, hwref, 30, [1.0 0.88 0.72]); text(Pref(1)-40, Pref(2)+45, 'ref + PZT');
    drawbody(Pout, hwcam, 40, [0.8 1.0 0.8]);   text(Pout(1)-60, Pout(2)-55, 'detector leg');
    plot(0,0,'ks','MarkerSize',12,'MarkerFaceColor','y');
    text(15,-35, sprintf('plate BS @ %d\\circ', AOI));
    title(sprintf('TG96 layout: BS %d\\circ, DM leg %d mm, beam %.0f mm', AOI, L_dm, 2*br));
    xlabel('mm'); ylabel('mm'); grid on;
    print(f,'tg96_layout.png','-dpng','-r130');
end

function drawbeam(p1, p2, r, c)
    d = p2 - p1;  n = [-d(2) d(1)]/norm(d)*r;
    patch('XData',[p1(1)+n(1) p2(1)+n(1) p2(1)-n(1) p1(1)-n(1)], ...
          'YData',[p1(2)+n(2) p2(2)+n(2) p2(2)-n(2) p1(2)-n(2)], ...
          'FaceColor',c, 'FaceAlpha',0.30, 'EdgeColor','none');
end

function drawbody(P, hw, dpth, c)
    rectangle('Position',[P(1)-hw P(2)-dpth/2 2*hw dpth], 'FaceColor',c, 'EdgeColor','k');
end

% ==== helpers, copied verbatim from tg_widen.m ====================
function A = arm_desc(rx, b, ix, base_deg)
    nm = {b.E.name};
    A = struct('rx', rx, 'b', b, 'iPol', find(strcmp(nm,'PolIn'),1), ...
        'iQ', find(contains(nm,'QWP') & ~strcmp(nm,'OutQWP')), ...
        'base', base_deg, 'qwp_deg', base_deg, 'oq_deg', 0, 'iTO', [], ...
        'iRC', ix.iRC, 'iOQ', ix.iOutQWP, 'iAn', ix.iAnalyzer, 'iDET', ix.iDET);
    if isfield(ix,'iTO'), A.iTO = ix.iTO; end
end

function a = lax(psi, deg)
    u1 = macos.design.Bench.perp(psi(:));  u2 = cross(psi(:), u1);
    a = cosd(deg)*u1 + sind(deg)*u2;  a = a(:).';
end

function x = wrap180(x)
    x = mod(x + 90, 180) - 90;
end

function load_arm(A, QWP, an_deg, grid)
    macos.load_rx(A.rx);  b = A.b;
    if nargin >= 4 && ~isempty(grid)
        macos.set_elt_grid(A.iTO, macos.get_elt_grid_spacing(A.iTO), grid);
    end
    macos.polarizer(A.iPol, 'axis', lax(b.E(A.iPol).psi, 45));
    qa = lax(b.E(A.iQ(1)).psi, A.qwp_deg);
    for j = 1:2, macos.waveplate(A.iQ(j), 'axis', qa, 'retardance', QWP); end
    macos.waveplate(A.iOQ, 'axis', lax(b.E(A.iOQ).psi, A.oq_deg), 'retardance', QWP);
    macos.polarizer(A.iAn, 'axis', lax(b.E(A.iAn).psi, an_deg));
    macos.polarization('on', 'Ex',[1/sqrt(2) 0], 'Ey',[1/sqrt(2) 0]);
    macos.vector_diffraction(true);
end

function E = arm_field(A, QWP, an_deg, grid)
    load_arm(A, QWP, an_deg, grid);
    E = cat(3, macos.complex_field(A.iDET,'plane',1), ...
               macos.complex_field(A.iDET,'plane',2), ...
               macos.complex_field(A.iDET,'plane',3));
end

function S = analyzer_basis(A, QWP, grid)
    E0  = arm_field(A, QWP,  0, grid);
    E45 = arm_field(A, QWP, 45, grid);
    E90 = arm_field(A, QWP, 90, grid);
    S = struct('A', E0, 'C', E90, 'B', 2*E45 - E0 - E90);
end

function E = synth(S, th)
    c = cosd(th);  s = sind(th);
    E = c^2*S.A + c*s*S.B + s^2*S.C;
end

function e = arm_state(A, QWP, iElt)
    load_arm(A, QWP, 0);
    macos.trace(iElt);  f = macos.ray_field(iElt);
    ok = f.status == 0;
    psi = A.b.E(iElt).psi(:);
    u1 = macos.design.Bench.perp(psi);  u2 = cross(psi, u1);
    e1 = f.Ex*u1(1) + f.Ey*u1(2) + f.Ez*u1(3);
    e2 = f.Ex*u2(1) + f.Ey*u2(2) + f.Ez*u2(3);
    r  = e2(ok)./e1(ok);  a = median(abs(e1(ok)));
    e  = [a; a*(median(real(r)) + 1i*median(imag(r)))];
end

function az = arm_azimuth(A, QWP, qwp_deg)
    A.qwp_deg = qwp_deg;
    e = arm_state(A, QWP, A.iRC);
    az = 0.5*atan2d(2*real(conj(e(1))*e(2)), abs(e(1))^2 - abs(e(2))^2);
end

function I = frame(Sx, Sr, th)
    I = sum(abs(synth(Sx,th) + synth(Sr,th)).^2, 3);
end

function p = fourstep(Sx, Sr, th)
    I1 = frame(Sx,Sr,th(1));  I2 = frame(Sx,Sr,th(2));
    I3 = frame(Sx,Sr,th(3));  I4 = frame(Sx,Sr,th(4));
    p  = atan2(I2-I4, I1-I3);
end

function h = meas_surface(A, QWP, M, Sr, p_null, THETAS, LAM)
    d = angle(exp(1i*(fourstep(analyzer_basis(A, QWP, M), Sr, THETAS) - p_null)));
    h = d * LAM/(4*pi);
end

function [map, reg] = register_two_pokes(A, ix, MpA, hpA, MpB, hpB, N_G, DX_G, msk)
%  Ray affine (scale/rotation) + poke-A blob centroid (translation, exact)
%  + poke-B overlap (parity selection among 8; doubles as the gate).
    macos.load_rx(A.rx);
    s1 = macos.trace(ix.iTO);   ito  = macos.get_ray_info(s1.nRays);
    s2 = macos.trace(ix.iDET);  idet = macos.get_ray_info(s2.nRays);
    okr = ito.ok_trace(:) & ito.ok_pass(:) & idet.ok_trace(:) & idet.ok_pass(:);
    psi1 = macos.get_elt_psi(ix.iTO);  vpt1 = macos.get_elt_vpt(ix.iTO);
    u1 = macos.design.Bench.perp(psi1);  v1 = cross(psi1, u1);
    xy_to = [u1.'; v1.'] * (ito.pos - vpt1);
    psi2 = macos.get_elt_psi(ix.iDET);
    u2 = macos.design.Bench.perp(psi2);  v2 = cross(psi2, u2);
    xy_d = [u2.'; v2.'] * (idet.pos - idet.pos(:,1));
    xy_to = xy_to(:,okr);  xy_d = xy_d(:,okr);
    Aaf = [xy_d.' ones(nnz(okr),1)] \ xy_to.';
    Lm  = Aaf(1:2,:).';
    [~,Ss,~] = svd(Lm);  sm = diag(Ss);
    nl  = xy_to - (Lm*xy_d + Aaf(3,:).');
    map = struct('mag', sqrt(abs(det(Lm))), 'anam_pct', 100*(sm(1)/sm(2)-1), ...
                 'nonlin_mm', sqrt(mean(sum(nl.^2,1))));
    N = size(hpA,1);  [cg, rg] = meshgrid(1:N, 1:N);
    cx = sum(cg(msk))/nnz(msk);  cy = sum(rg(msk))/nnz(msk);
    dxp = macos.dx_at(ix.iDET, 'mm');
    a1 = (cg-cx)*dxp;  a2 = (rg-cy)*dxp;
    axs = ((1:N_G)-(N_G+1)/2)*DX_G;
    sc = map.mag;                        % isotropic (anam ~0): DM mm per det mm
    % measured blob-A centroid in detector-plane coords (weighted, top 5%)
    w = abs(hpA - median(hpA(msk)));  w(~msk) = 0;
    w(w < 0.05*max(w(:))) = 0;
    bx = sum(a1(:).*w(:))/sum(w(:));  by = sum(a2(:).*w(:))/sum(w(:));
    % truth blob positions on the DM (mm)
    [~,iA] = max(abs(MpA(:)));  [rA, cA] = ind2sub(size(MpA), iA);
    pA = [axs(cA) axs(rA)];
    hpB0 = hpB(msk) - mean(hpB(msk));
    cands = {a1,a2; a1,-a2; -a1,a2; -a1,-a2; a2,a1; a2,-a1; -a2,a1; -a2,-a1};
    % blob-A centroid under each parity: parity acts on (a1,a2), and the
    % centroid transforms the same way -- evaluate directly.
    bA = {[bx by]; [bx -by]; [-bx by]; [-bx -by]; [by bx]; [by -bx]; [-by bx]; [-by -bx]};
    reg = struct('pokeB_corr',0,'par',0);  tab = zeros(1,8);  best_abs = -1;
    for c = 1:8
        Xc = sc*cands{c,1};  Yc = sc*cands{c,2};
        Xc = Xc - (sc*bA{c}(1) - pA(1));      % translation: blob A -> truth A
        Yc = Yc - (sc*bA{c}(2) - pA(2));
        tps = interpn(axs, axs, MpB, Xc, Yc, 'spline', 0);
        cc = corrcoef(hpB0, tps(msk)-mean(tps(msk)));
        tab(c) = cc(1,2);
        if abs(cc(1,2)) > best_abs
            best_abs = abs(cc(1,2));
            reg.pokeB_corr = cc(1,2);  reg.par = c;  reg.Xt = Xc;  reg.Yt = Yc;
        end
    end
    st = sort(abs(tab), 'descend');  reg.runner_up = st(2);  reg.table = tab;
    % The four-step sign is DECK-DEPENDENT (the v1 example's h = +-psi
    % lambda/4pi gate); the poke-B overlap sign IS that calibration.
    reg.sign = sign(reg.pokeB_corr);
    map.Xt = reg.Xt;  map.Yt = reg.Yt;
end

function [c, ht, Xt, Yt] = reg_corr(p, A1, A2, c_d, Fx, Fy, axs, Mdm, hm, msk)
    s = exp(p(4));  ct = cos(p(3));  st = sin(p(3));
    X = s*(ct*A1 - st*A2) + c_d(1) + p(1);
    Y = s*(st*A1 + ct*A2) + c_d(2) + p(2);
    Xt = Fx(X,Y);  Yt = Fy(X,Y);
    ht = interpn(axs, axs, Mdm, Xt, Yt, 'spline', 0);
    ht = ht - mean(ht(msk));
    cc = corrcoef(hm(msk), ht(msk));  c = cc(1,2);
end
