% example_bench_layout.m
% ===================================================================
%  MACOS DESIGN LAYER -- LAYING OUT AND OPTIMIZING AN OPTICAL BENCH
% ===================================================================
%  A generic folded-bench worked example: build a complete test-bench
%  train SEQUENTIALLY with the macos.design.Bench add-optic utilities,
%  then optimize it in three physically-staged steps.  The bench is the
%  classic surface-gauge / interferometer test-arm topology:
%
%    point source -> baffle (pupil stop) -> L1 (collimating singlet)
%      -> beam-splitter REFLECT -> DM (test mirror, retro)
%      -> beam-splitter TRANSMIT (through the tilted plate, real
%         walk-off) -> Fold1 -> Fold2 -> L2 (imaging singlet)
%      -> focal-mask Reference (internal focus) -> Detector at the
%         DM-PUPIL IMAGE.
%
%  Everything is laid out in the X-Y plane with REAL angles of
%  incidence -- a folded 3-D bench, not an unfolded paraxial stack.
%  The Bench builder tracks the chief ray analytically (mirror turns,
%  Snell refraction through the BS plate) so every element is emitted
%  already centered on the beam.
%
%  THE THREE STAGES (why staged: each conjugate condition lives on its
%  own plane, and a single wavefront solve at the detector could trade
%  them against each other -- e.g. mis-collimate at the DM and hide it
%  at focus, which is fatal for a surface gauge):
%
%    Stage A  COLLIMATE AT THE DM.  Solve L1 (Kr,Kc) to minimize the
%             ray-direction spread vs the chief at the DM.  A geometric
%             cost -- a WFE cost is gamed by the reference sphere
%             absorbing defocus (the optimizer runs to a flat lens).
%    Stage B  FOCUS AT THE MASK.  Freeze L1, solve L2 (Kr,Kc) to
%             minimize the TRANSVERSE ray spot on the flat mask plane.
%    Stage C  DETECTOR AT THE DM-PUPIL IMAGE.  The detector must be
%             CONJUGATE to the DM so DM surface detail maps sharply.
%             Seeded by the thin-lens conjugate, then trimmed with the
%             engine using the classic conjugate test: TILT the DM a
%             little -- at the DM's image plane the beam does NOT
%             translate (tilt at a pupil changes only angle).  Minimize
%             the tilt-induced chief displacement over detector
%             position.
%
%  Conventions baked in (engine-verified; see macos.design.Bench help):
%  collimator = flat front / powered back, Kr=+(n-1)f; imager = powered
%  front, Kr=-(n-1)f with psi=+chief; point-source Aperture = FULL cone
%  angle in RADIANS; every Rx ends with the nOutCord/Tout terminator.
%
%  Run interactively:  >> run('.../example_bench_layout.m')
%  (Batch check: matlab -batch "run('.../example_bench_layout.m')")
%  Requires MACOS_HOME (else the MEX aborts on init).
% ===================================================================

addpath('~/dev/MACOS_resources/mmacos/src');     % +macos on path
exdir = fileparts(mfilename('fullpath'));
if isempty(exdir), exdir = pwd; end
assert(~isempty(getenv('MACOS_HOME')), ...
    'MACOS_HOME must be set (engine needs macos_param.txt).');

MODEL = 256;
macos.init(MODEL);

% ---- bench parameters (all lengths mm; edit and re-run) ------------
F1        = 500;     % L1 collimator EFL
F2        = 250;     % L2 imager EFL
D_LENS    = 60;      % lens clear diameter
N_GLASS   = 1.5;
R_BAFFLE  = 12.5;    % pupil-stop radius -> collimated beam ~ 2*R*F1/D_SB
D_SB      = 250;     % source -> baffle
FILL      = 0.95;    % source cone fill of the baffle (<1: no vignetting)
BS_T      = 8;       % beam-splitter plate thickness
D_L1_BS   = 150;     % L1 powered surface -> BS
D_BS_DM   = 250;     % BS -> DM
D_BS_F1   = 300;     % BS back face -> Fold1
D_F1_F2   = 300;     % Fold1 -> Fold2
D_F2_L2   = 200;     % Fold2 -> L2 powered surface
DM_TILT   = 2e-4;    % Stage-C probe tilt (rad)

% ---- build the bench sequentially ----------------------------------
AP = 2*atan(R_BAFFLE/D_SB)*FILL;    % full cone angle (rad), fills FILL of stop
b = macos.design.Bench('bench_layout', 'aperture', AP, 'ngridpts', 63);

iBAF = b.add_baffle(D_SB, R_BAFFLE);
L1s  = b.add_lens(F1 - D_SB, F1, D_LENS, 'mode','collimate', 'n',N_GLASS, 'name','L1');
[~, bs] = b.add_bs_reflect(D_L1_BS, [0;-1;0], 'thickness',BS_T, 'n',N_GLASS);
iDM  = b.add_mirror(D_BS_DM, 'name','DM', 'aprad',30);   % retro test mirror
b.add_bs_transmit(bs);
b.add_fold(D_BS_F1, [1;0;0], 'name','Fold1');
b.add_fold(D_F1_F2, [0;-1;0], 'name','Fold2');
L2s  = b.add_lens(D_F2_L2, F2, D_LENS, 'mode','focus', 'n',N_GLASS, 'name','L2');
iMASK = b.add_reference(F2 - L2s.thickness, 'FocalMask');

% detector seed: thin-lens conjugate of the DM through L2
s_o  = b.E(L2s.i_pow).s - b.E(iDM).s;    % DM -> L2 chief path
s_i  = 1/(1/F2 - 1/s_o);                 % L2 -> pupil image
iDET = b.add_detector(s_i - (b.E(iMASK).s - b.E(L2s.i_pow).s), 'Detector');
mag  = s_i/s_o;                          % pupil-image magnification
b.print_chain();

rx_seed = fullfile(exdir, 'bench_layout_seed.in');
b.emit(rx_seed);
fprintf('seed prescription: %s\n', rx_seed);

% ---- input-parameter schematic (what a user must specify) ----------
%  Every leg is labeled with the parameter that set it; unlabeled legs
%  are derived (lens thicknesses, the plate crossing, conjugates).
leg_labels = { ...
    'D_SB',            ... 1  Baffle
    '',                ... 2  L1flat (F1 - D_SB - t, derived)
    '(L1 thickness)',  ... 3  L1pow
    'D_L1_BS',         ... 4  BSrefl
    'D_BS_DM',         ... 5  DM
    '(retro to BS)',   ... 6  BStxf
    '(thru BS_T)',     ... 7  BStxb
    'D_BS_F1',         ... 8  Fold1
    'D_F1_F2',         ... 9  Fold2
    'D_F2_L2',         ... 10 L2pow
    '(L2 thickness)',  ... 11 L2flat
    'F2 - t (focus)',  ... 12 FocalMask
    's_i (conjugate)'};  % 13 Detector
fsk = b.sketch('labels', leg_labels, 'title', ...
    'bench_layout inputs -- legs = add_* DIST args (mm); source cone + stop set the pupil');
set(fsk, 'Position', [100 100 1500 1000]);
png0 = fullfile(exdir, 'bench_layout_params.png');
print(fsk, png0, '-dpng', '-r150');
fprintf('parameter schematic: %s\n', png0);

% ---- verify: load, trace, builder-vs-engine chief agreement --------
macos.load_rx(rx_seed);
nE = macos.num_elt();
assert(nE == numel(b.E), 'engine read %d elements, builder has %d', nE, numel(b.E));
s1 = macos.trace(1);   sN = macos.trace(nE);
fprintf('rays: %d at baffle, %d at detector\n', s1.nRays, sN.nRays);
assert(sN.nRays == s1.nRays, 'vignetting: %d rays lost past the baffle', ...
    s1.nRays - sN.nRays);
dchief = zeros(1, nE);
for k = 1:nE
    sk = macos.trace(k);
    info = macos.get_ray_info(sk.nRays);
    dchief(k) = norm(info.pos(:,1) - b.E(k).vpt);
end
fprintf('builder-vs-engine chief agreement: max %.3g mm\n', max(dchief));
assert(max(dchief) < 1e-6, 'chief-ray model disagrees with the engine');

out = struct('params', struct('F1',F1,'F2',F2,'AP',AP,'mag_seed',mag), ...
             'rx_seed', rx_seed, 'nrays', sN.nRays, 'chief_err', max(dchief));

% ---- Stage A: collimate at the DM (solve L1) -----------------------
fprintf('\n=== Stage A: collimate at DM (elt %d), solve L1 ===\n', iDM);
c0 = eval_at(rx_seed, @() collimation_cost(iDM));
[out.L1_Kr, out.L1_Kc] = optimize_conic(rx_seed, L1s.i_pow, @() collimation_cost(iDM));
rxA = fullfile(exdir, 'bench_layout_stageA.in');
macos.save_rx(rxA);
cA = eval_at(rxA, @() collimation_cost(iDM));
fprintf('  L1: Kr %.6g -> %.6g, Kc %.3g -> %.4g;  ray spread %.3g -> %.3g mrad rms\n', ...
    L1s.Kr, out.L1_Kr, L1s.Kc, out.L1_Kc, 1e3*sqrt(c0), 1e3*sqrt(cA));

% ---- Stage B: focus at the mask (solve L2, L1 frozen) --------------
fprintf('\n=== Stage B: focus at mask (elt %d), solve L2 ===\n', iMASK);
f0 = eval_at(rxA, @() spot_cost(iMASK));
[out.L2_Kr, out.L2_Kc] = optimize_conic(rxA, L2s.i_pow, @() spot_cost(iMASK));
rxB = fullfile(exdir, 'bench_layout_stageB.in');
macos.save_rx(rxB);
fB = eval_at(rxB, @() spot_cost(iMASK));
fprintf('  L2: Kr %.6g -> %.6g, Kc %.3g -> %.4g;  focus spot %.3g -> %.3g um rms\n', ...
    L2s.Kr, out.L2_Kr, L2s.Kc, out.L2_Kc, 1e3*f0, 1e3*fB);

% ---- Stage C: detector at the DM-pupil image -----------------------
%  Tilt the DM; at the DM's conjugate the beam does not translate.
fprintf('\n=== Stage C: detector at the DM pupil image ===\n');
macos.load_rx(rxB);
psi0  = macos.get_elt_psi(iDM);
Rz    = [cos(DM_TILT) -sin(DM_TILT) 0; sin(DM_TILT) cos(DM_TILT) 0; 0 0 1];
psi_t = Rz*psi0;
vdet0 = b.E(iDET).vpt;   cdet = b.dir;    % chief direction at the detector
shift = @(dz) tilt_shift(rxB, iDM, psi_t, iDET, vdet0, cdet, dz);
sh0 = shift(0);
dz  = fminbnd(shift, -0.6*s_i, 0.6*s_i, optimset('TolX',1e-3,'Display','off'));
shN = shift(dz);
fprintf('  thin-lens seed off by %.3f mm; tilt-induced beam shift %.3g -> %.3g um\n', ...
    dz, 1e3*sh0, 1e3*shN);

% final prescription: stage-B conics + trimmed detector, DM untouched
macos.load_rx(rxB);
macos.set_elt_vpt(iDET, vdet0 + dz*cdet);
rx_opt = fullfile(exdir, 'bench_layout_opt.in');
macos.save_rx(rx_opt);
out.rx_opt = rx_opt;  out.det_trim_mm = dz;

% pupil-image size check vs the thin-lens magnification
macos.load_rx(rx_opt);
sD = macos.trace(iDET);  info = macos.get_ray_info(sD.nRays);
ok = info.ok_trace(:) & info.ok_pass(:);
d  = info.pos(:,ok) - info.pos(:,1);  d = d - cdet*(cdet.'*d);
r_img = max(sqrt(sum(d.^2,1)));
sB = macos.trace(iDM);  infoB = macos.get_ray_info(sB.nRays);
okB = infoB.ok_trace(:) & infoB.ok_pass(:);
dB  = infoB.pos(:,okB) - infoB.pos(:,1);
r_dm = max(sqrt(sum(dB.^2,1)));
out.r_pupil_dm = r_dm;  out.r_pupil_img = r_img;  out.mag_engine = r_img/r_dm;
fprintf('  pupil: %.2f mm radius at DM -> %.2f mm at detector (mag %.3f, thin-lens %.3f)\n', ...
    r_dm, r_img, r_img/r_dm, mag);

% ---- render + save -------------------------------------------------
macos.load_rx(rx_opt);
macos.modify();                       % dirty the trace so ray_hist populates
macos.trace(macos.num_elt());
f1 = macos.view_rx('show','beam','bundle','rings','nrings',3,'nspokes',12, ...
                   'bodies','solid');
set(f1, 'Color','w', 'Position',[100 100 1400 1000]);
view(2); axis equal; grid on;
title('bench\_layout -- beam through optics (fold plane)');
png1 = fullfile(exdir, 'bench_layout_view_rx.png');
print(f1, png1, '-dpng', '-r150');
f2 = macos.view_std();
set(f2, 'Color','w', 'Position',[100 100 1600 1100]);
png2 = fullfile(exdir, 'bench_layout_view_std.png');
print(f2, png2, '-dpng', '-r150');
fprintf('rendered: %s\n          %s\n', png1, png2);

save(fullfile(exdir, 'bench_layout.mat'), 'out');
fprintf('\nDONE.  Optimized prescription: %s\n', rx_opt);

% ===================================================================
%  LOCAL FUNCTIONS
% ===================================================================
function [Kr, Kc] = optimize_conic(rx, pow_elt, costfn)
%OPTIMIZE_CONIC  fminsearch over (Kr,Kc) of POW_ELT minimizing COSTFN;
%   RX is reloaded fresh each evaluation.  Leaves the engine loaded at
%   the optimum (caller saves with macos.save_rx).
    macos.load_rx(rx);
    x0 = [macos.get_elt_kr(pow_elt), macos.get_elt_kc(pow_elt)];
    f = @(x) eval_conic(rx, pow_elt, x, costfn);
    o = optimset('Display','off','TolX',1e-6,'TolFun',1e-16,'MaxFunEvals',500);
    x = fminsearch(f, x0, o);
    macos.load_rx(rx);
    macos.set_elt_kr(pow_elt, x(1));  macos.set_elt_kc(pow_elt, x(2));
    Kr = x(1);  Kc = x(2);
end

function c = eval_conic(rx, pow_elt, x, costfn)
    macos.load_rx(rx);
    macos.set_elt_kr(pow_elt, x(1));  macos.set_elt_kc(pow_elt, x(2));
    c = costfn();
end

function c = eval_at(rx, costfn)
    macos.load_rx(rx);
    c = costfn();
end

function c = collimation_cost(dm_elt)
%COLLIMATION_COST  Mean-squared ray angle vs the chief at DM_ELT (rad^2).
    s = macos.trace(dm_elt);
    if s.nRays < 10, c = 1e6; return; end
    info = macos.get_ray_info(s.nRays);
    ok = info.ok_trace(:) & info.ok_pass(:);
    D  = info.dir(:, ok);  D = D ./ vecnorm(D);
    dch = info.dir(:,1) / norm(info.dir(:,1));
    ct = max(min(dch.'*D, 1), -1);
    c  = mean(acos(ct).^2);
end

function r = spot_cost(foc_elt)
%SPOT_COST  RMS TRANSVERSE ray spread on the flat plane at FOC_ELT (mm).
%   Transverse (perp to the chief), so defocus cannot be hidden the way
%   a WFE-vs-reference-sphere metric hides it.
    s = macos.trace(foc_elt);
    if s.nRays < 10, r = 1e6; return; end
    info = macos.get_ray_info(s.nRays);
    ok = info.ok_trace(:) & info.ok_pass(:);
    P  = info.pos(:, ok);
    pch = info.pos(:,1);  dch = info.dir(:,1)/norm(info.dir(:,1));
    d  = P - pch;  dt = d - dch*(dch.'*d);
    r  = sqrt(mean(sum(dt.^2, 1)));
end

function sh = tilt_shift(rx, dm_elt, psi_t, det_elt, vdet0, cdet, dz)
%TILT_SHIFT  Transverse chief displacement at the (moved) detector when
%   the DM is tilted -- zero when the detector sits at the DM conjugate.
    macos.load_rx(rx);
    macos.set_elt_vpt(det_elt, vdet0 + dz*cdet);
    macos.set_elt_psi(dm_elt, psi_t);
    s = macos.trace(det_elt);
    if s.nRays < 10, sh = 1e6; return; end
    info = macos.get_ray_info(s.nRays);
    d = info.pos(:,1) - (vdet0 + dz*cdet);
    d = d - cdet*(cdet.'*d);
    sh = norm(d);
end
