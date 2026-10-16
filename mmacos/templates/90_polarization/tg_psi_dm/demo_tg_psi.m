% demo_tg_psi.m
% =========================================================================
%  LIVE DEMO -- polarization phase-shifting Twyman-Green gauging a DM
% =========================================================================
%  Seven beats, each a few seconds, each leaving a PNG behind:
%
%    1  BUILD      one call puts the whole bench on the chief ray
%    2  LAYOUT     look at it
%    3  ALIGN      the beamsplitter has rotated an arm -- find it, fix it
%    4  NULL       flat DM: the fringe field with nothing to measure
%    5  POKE       drive ONE actuator; the fringes bend over it
%    6  SWEEP      rotate the analyzer -- the fringes walk, with NO re-trace
%    7  RECOVER    four-step PSI: the DM surface beside the truth
%
%  Every beat prints its own numbers, and every beat writes
%  demo_beat<N>_*.png into this directory, so a live hang costs ten seconds
%  rather than the demo -- just show the PNG.
%
%  The full, gated version of this measurement (five gates, the beta^2 law,
%  the alignment counterfactual, the closure against the injected map) is
%  example_tg_psi_dm.m in this directory.  This script is the narrative cut.
%
%  Run:  cd <this dir>;  demo_tg_psi          % interactive, pauses per beat
%        matlab -batch "run('demo_tg_psi.m'); exit(0)"   % regenerate the PNGs
% =========================================================================

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);

LAM   = 6.328e-4;                  % mm (HeNe)
MODEL = 256;   NGRID = 63;
N_G   = 256;   DX_G  = 0.35;       % DM grid: 89 mm span
NACT  = 16;    PITCH = 3.5;        % 16x16 actuators at 3.5 mm
QWP   = 0.25;
THETAS = [0 45 90 135];
TAIL = {'tail_arch','fieldlens', 'FL_F',25.02100857, 'FL_Kc',-2.11278288, ...
        'D_MASK_FL',6.277463741, 'DET_TRIM',1.085330067};
INTERACTIVE = usejava('desktop');   % pause between beats only when a human is here
%  The live animation needs figures, not a human, so it is gated separately --
%  set TG_DEMO_ANIMATE=1 to exercise that path in batch (it is the one part of
%  the demo a headless run would otherwise never execute, and the demo is the
%  thing that has to work in front of a room).
ANIMATE = INTERACTIVE || ~isempty(getenv('TG_DEMO_ANIMATE'));
NTURNS  = 3;  if ~INTERACTIVE, NTURNS = 1; end
macos.init(MODEL);

% =========================================================================
beat(1, 'BUILD -- the whole bench in one call', INTERACTIVE);
% =========================================================================
%  Bench places every optic on the chief ray it tracks analytically, so the
%  only things named here are physical: focal lengths, plate thickness,
%  standoffs, and the polarization azimuths.  'polarizing' inserts the REAL
%  engine polarizing elements (TrPolarizer / WavePlate), not a Jones model
%  bolted on afterwards -- they see the rays the same way every other
%  surface does.
mkrig = @(gf) macos.design.twyman_green( ...
    'polarizing',   true,  ...   % insert the polarizing train
    'ngridpts',     NGRID, ...
    'pol_in_deg',   45,    ...   % input polarizer, both arms
    'qwp_test_deg', 0,     ...   % test-arm QWP  (double-passed = half-wave)
    'qwp_ref_deg',  45,    ...   % ref-arm  QWP
    'out_qwp_deg',  0,     ...   % output QWP, shared leg
    'qwp_ret',      QWP,   ...
    'to_grid_file', gf, 'to_grid_n', N_G, 'to_grid_dx', DX_G, TAIL{:});

macos.write_grid_file('demo_flat.txt', zeros(N_G));
G = mkrig('demo_flat.txt');
G.bt.emit('demo_test.in');  G.br.emit('demo_ref.in');
print_train('TEST ARM', G.bt);
print_train('REF  ARM', G.br);
fprintf('\n  -> demo_test.in (%d elements), demo_ref.in (%d elements)\n', ...
    numel(G.bt.E), numel(G.br.E));

AT = arm_desc('demo_test.in', G.bt, G.T, 0);      % test arm
AR = arm_desc('demo_ref.in',  G.br, G.R, 45);     % reference arm

% =========================================================================
beat(2, 'LAYOUT -- look at it', INTERACTIVE);
% =========================================================================
macos.load_rx('demo_test.in');
f = macos.view_std('title', 'Polarization-PSI Twyman-Green, test arm', ...
                   'save', 'demo_beat2_layout.png');
fprintf('  source -> baffle -> L1 -> polarizer -> BS -> compensator -> QWP ->\n');
fprintf('  DM (retro) -> QWP -> BS -> recomb -> output QWP -> analyzer ->\n');
fprintf('  L2 -> focal mask -> field lens -> detector at the DM pupil image\n');
fprintf('  saved demo_beat2_layout.png\n');

% =========================================================================
beat(3, 'ALIGN -- the beamsplitter has rotated an arm', INTERACTIVE);
% =========================================================================
%  The design intent: a double-passed QWP at azimuth alpha acts as a
%  half-wave plate, so the two arms should leave at -45 and +45 -- ORTHOGONAL.
%  Measure it.  Everything between the polarizer and the recombination plane
%  that is not at normal incidence is a diattenuator (t_s /= t_p), and a
%  diattenuator rotates a linear state.
az_r  = arm_azimuth(AR, QWP, AR.base);
az_t0 = arm_azimuth(AT, QWP, 0);
want  = az_r - 90;
fprintf('  reference arm leaves at %+8.4f deg   (design %+8.4f)\n', az_r, 45);
fprintf('  test      arm leaves at %+8.4f deg   (design %+8.4f)\n', az_t0, -45);
fprintf('  -> %.4f deg from orthogonal.  A waveplate is UNITARY, so nothing\n', ...
    abs(az_t0 - want));
fprintf('     downstream can repair this; the arm waveplate is the only knob.\n');
al = [0, -0.5*(az_t0-want)];  az = [az_t0, arm_azimuth(AT, QWP, al(2))];
for it = 1:3
    if abs(az(end)-want) < 1e-7, break; end
    sl = (az(end)-az(end-1))/(al(end)-al(end-1));
    al(end+1) = al(end) + (want-az(end))/sl;  az(end+1) = arm_azimuth(AT, QWP, al(end)); %#ok<SAGROW>
end
alpha_t = al(end);  oq = az_r + 45;
AT = set_pol_align(AT, alpha_t, oq);
AR = set_pol_align(AR, AR.base, oq);
fprintf('  turn the test-arm waveplate %+.4f deg -> arms leave at %+8.4f deg\n', ...
    alpha_t, az(end));
fprintf('  output QWP to %+.4f deg (45 deg off the pair) -> orthogonal CIRCULAR\n', oq);
st = arm_state(AT, QWP, AT.iOQ);  sr = arm_state(AR, QWP, AR.iOQ);
fprintf('  check: test |b/a| %.5f arg %+7.2f | ref |b/a| %.5f arg %+7.2f | <t|r> %.1e\n', ...
    abs(st(2)/st(1)), rad2deg(angle(st(2)/st(1))), ...
    abs(sr(2)/sr(1)), rad2deg(angle(sr(2)/sr(1))), ...
    abs(st'*sr)/(norm(st)*norm(sr)));
fprintf('  (skip this beat and the gauge reads 11.7%% high -- see gate 5 of\n');
fprintf('   example_tg_psi_dm.m, which runs both configurations)\n');

% =========================================================================
beat(4, 'NULL -- flat DM, nothing to measure', INTERACTIVE);
% =========================================================================
%  THREE traces per arm, and then every analyzer angle is free: the detector
%  field is bilinear in the analyzer axis, so E(t) = c^2 A + c s B + s^2 C.
t0 = tic;
Sr    = analyzer_basis(AR, QWP, []);
Sflat = analyzer_basis(AT, QWP, []);
fprintf('  6 traces in %.2f s -- from here the analyzer costs nothing\n', toc(t0));
I_null = frame(Sflat, Sr, 0);
msk = I_null > 0.1*max(I_null(:));
p_null = fourstep(Sflat, Sr, THETAS);
fprintf('  fringe field is FLAT: recovered phase %.3e rad rms in the pupil\n', ...
    std(p_null(msk) - median(p_null(msk))));
fprintf('  = %.4f nm of surface -- the null of a compensated Twyman-Green\n', ...
    1e6*std(p_null(msk)-median(p_null(msk)))*LAM/(4*pi));

% =========================================================================
beat(5, 'POKE -- drive one actuator', INTERACTIVE);
% =========================================================================
%  A LIVE poke: rewrite the DM grid in the loaded model (macos.set_elt_grid,
%  which invalidates the cached trace for you) rather than re-emitting a
%  deck.  This is the same call an actuator command loop would make.
M1 = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, ...
                      'poke',150e-6, 'pattern','single');
S1 = analyzer_basis(AT, QWP, M1);
I_poke = frame(S1, Sr, 0);
fprintf('  one actuator at 150 nm -> the fringes bend over it\n');
d1 = angle(exp(1i*(fourstep(S1,Sr,THETAS) - p_null)));
fprintf('  local surface change %.1f nm peak (injected %.1f nm)\n', ...
    1e6*max(abs(d1(msk)))*LAM/(4*pi), 150);
fprintf('  (a few percent low: the detector''s pupil image of the DM smooths a\n');
fprintf('   SINGLE actuator, which is the sharpest thing the DM can make)\n');
b5 = beam_box(msk, 6);  cl5 = [0 max([I_null(msk); I_poke(msk)])];
f5 = figure('Color','w','Position',[80 80 1200 460]);
subplot(1,2,1); imagesc(sub(I_null,b5)); axis image off; clim(cl5); colorbar;
title('flat DM: the null (a dark fringe)');
subplot(1,2,2); imagesc(sub(I_poke,b5)); axis image off; clim(cl5); colorbar;
title('one actuator at 150 nm');
print(f5, 'demo_beat5_poke.png', '-dpng', '-r150');
fprintf('  saved demo_beat5_poke.png\n');

% =========================================================================
beat(6, 'SWEEP -- rotate the analyzer, no re-trace', INTERACTIVE);
% =========================================================================
%  Nothing in the interferometer moves.  The phase steps come from rotating
%  a polarizer in the output leg -- and because the field is bilinear in the
%  analyzer axis, all of these frames come out of the six traces of beat 4.
nsw = 36;  ths = (0:nsw-1)/nsw*180;
t0 = tic;  Isw = zeros([size(I_null) nsw]);
for k = 1:nsw, Isw(:,:,k) = frame(S1, Sr, ths(k)); end
fprintf('  %d analyzer frames synthesized in %.3f s -- ZERO traces\n', nsw, toc(t0));
pk = find(msk);  [~,ip] = max(abs(d1(pk)));  [pr,pc] = ind2sub(size(I_null), pk(ip));
tr = squeeze(Isw(pr,pc,:));
b6 = beam_box(msk, 6);  cl6 = [0 max(Isw(:))];
f6 = figure('Color','w','Position',[80 80 1500 700]);
for k = 1:4
    subplot(2,4,k); imagesc(sub(frame(S1,Sr,THETAS(k)),b6)); axis image off;
    clim(cl6);          % ONE scale for all four: the whole field modulates
    title(sprintf('analyzer %d\\circ  (step %d/4)', THETAS(k), k));
end
subplot(2,4,[5 6]);
plot(ths, tr/max(tr), 'b-', 'LineWidth',1.5); hold on;
plot(THETAS, interp1([ths 180],[tr;tr(1)]/max(tr),THETAS), 'ro','MarkerFaceColor','r');
grid on; xlabel('analyzer angle \theta (deg)'); ylabel('normalized intensity');
title('one pixel: I(\theta) = A + B cos2\theta + C sin2\theta');
legend({'synthesized','the four steps'}, 'Location','south');
subplot(2,4,[7 8]);
%  the whole sweep as ONE static image: a cut across the beam through the
%  poked actuator, stacked against analyzer angle.  This is what the live
%  animation shows, in a form that survives being a PNG on a slide.
kymo = squeeze(Isw(pr, b6(3):b6(4), :)).';
imagesc(1:size(kymo,2), ths, kymo);  clim(cl6);
xlabel('across the beam (px)'); ylabel('analyzer angle \theta (deg)');
title(sprintf('the fringes walk: one cut through the actuator, all %d angles', nsw));
print(f6, 'demo_beat6_sweep.png', '-dpng', '-r150');
fprintf('  saved demo_beat6_sweep.png\n');
if ANIMATE                          % the live animation
    fprintf('  animating the sweep (close the window to stop)...\n');
    fa = figure('Color','w','Position',[200 200 620 620]);
    ax = axes(fa);  im = imagesc(ax, sub(Isw(:,:,1),b6));  axis(ax,'image','off');
    cl = [min(Isw(:)) max(Isw(:))];  clim(ax, cl);
    for turn = 1:NTURNS
        for k = 1:nsw
            if ~isvalid(fa), break; end
            set(im, 'CData', sub(Isw(:,:,k),b6));
            title(ax, sprintf('analyzer %5.1f\\circ   (fringe phase 2\\theta)', ths(k)));
            drawnow;  if INTERACTIVE, pause(0.04); end
        end
        if ~isvalid(fa), break; end
    end
    if isvalid(fa)
        print(fa, 'demo_beat6_animation_frame.png', '-dpng', '-r120');
        fprintf('  animation ran %d turns x %d frames\n', NTURNS, nsw);
    end
end

% =========================================================================
beat(7, 'RECOVER -- four-step PSI against the truth', INTERACTIVE);
% =========================================================================
%  The full DM command now, and the differential protocol a real gauge uses:
%  measure flat, measure poked, subtract in the complex domain.  Every static
%  term cancels and nothing needs unwrapping.
[Mdm, dminfo] = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, ...
                                 'poke',50e-6, 'pattern','checker');
Sdm  = analyzer_basis(AT, QWP, Mdm);
dphi = angle(exp(1i*(fourstep(Sdm,Sr,THETAS) - p_null)));
h    = dphi * LAM/(4*pi);           % surface height, mm (sign gated in the example)
fprintf('  %dx%d actuators, checkerboard at 50 nm: %.2f nm rms, %.1f nm PtV injected\n', ...
    NACT, NACT, 1e6*std(Mdm(:)), 1e6*(max(Mdm(:))-min(Mdm(:))));
[best, map] = register_to_dm(AT, G.T, Mdm, N_G, DX_G, h, msk);
%  Score the interior: the outermost ring resamples the truth with rays on one
%  side only, so its residual is an artefact of the comparison, not the gauge.
mskin = erode_disc(msk, 0.92);
res  = best.hm(mskin) - best.ht(mskin);
fprintf('  the detector images the DM at mag %.3f with %.3f%% anamorphic stretch\n', ...
    map.mag, map.anam_pct);
fprintf('  and %.4f mm of nonlinear distortion -- measured from the trace, not assumed\n', ...
    map.nonlin_mm);
fprintf('\n  RECOVERED %.2f nm rms   TRUTH %.2f nm rms   RESIDUAL %.3f nm rms (%.1f%%)\n\n', ...
    1e6*std(best.hm(msk)), 1e6*std(best.ht(msk)), 1e6*std(res), ...
    100*std(res)/std(best.ht(mskin)));
NN = size(h,1);  box = beam_box(msk, 6);
cl = [-1 1]*1e6*max(abs(best.ht(msk)));
f7 = figure('Color','w','Position',[80 80 1560 420]);
tl = tiledlayout(f7,1,4,'TileSpacing','compact','Padding','compact');
nexttile; imagesc(sub(frame(Sdm,Sr,0),box)); axis image off; colorbar; title('interferogram');
show(1e6*best.hm, msk, NN, box, cl, sprintf('MEASURED (%.2f nm rms)', 1e6*std(best.hm(msk))));
show(1e6*best.ht, msk, NN, box, cl, sprintf('TRUTH (%.2f nm rms)', 1e6*std(best.ht(msk))));
show(1e6*(best.hm-best.ht), msk, NN, box, [-1 1]*4e6*std(res), ...
     sprintf('residual %.3f nm rms interior', 1e6*std(res)));
title(tl, 'Polarization-PSI Twyman-Green: a DM measured, and the map it was given');
print(f7, 'demo_beat7_recovery.png', '-dpng', '-r150');
fprintf('  saved demo_beat7_recovery.png\n');
fprintf('\n=== demo complete.  Artefacts: demo_*.in, demo_beat*.png ===\n');

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================
function beat(n, ttl, interactive)
    fprintf('\n');
    fprintf('#########################################################\n');
    fprintf('#  BEAT %d -- %s\n', n, ttl);
    fprintf('#########################################################\n');
    if interactive && n > 1
        input('   [enter to run this beat] ', 's');
    end
end

function print_train(ttl, b)
    fprintf('\n  %s\n', ttl);
    for k = 1:numel(b.E)
        e = b.E(k);
        mark = '';
        if ismember(e.element, {'TrPolarizer','WavePlate'}), mark = '   <-- polarization'; end
        if ~isempty(e.gridfile), mark = '   <-- the DM'; end
        fprintf('   %2d  %-11s %-9s %-10s s=%8.3f mm%s\n', ...
            k, e.name, e.element, e.surface, e.s, mark);
    end
end

function A = arm_desc(rx, b, ix, base_deg)
    nm = {b.E.name};
    A = struct('rx', rx, 'b', b, 'iPol', find(strcmp(nm,'PolIn'),1), ...
        'iQ', find(contains(nm,'QWP') & ~strcmp(nm,'OutQWP')), ...
        'base', base_deg, 'qwp_deg', base_deg, 'oq_deg', 0, 'iTO', [], ...
        'iRC', ix.iRC, 'iOQ', ix.iOutQWP, 'iAn', ix.iAnalyzer, 'iDET', ix.iDET);
    if isfield(ix,'iTO'), A.iTO = ix.iTO; end
end

function A = set_pol_align(A, qwp_deg, oq_deg)
    A.qwp_deg = qwp_deg;  A.oq_deg = oq_deg;
end

function a = lax(psi, deg)
    u1 = macos.design.Bench.perp(psi(:));  u2 = cross(psi(:), u1);
    a = cosd(deg)*u1 + sind(deg)*u2;  a = a(:).';
end

function load_arm(A, QWP, an_deg, grid)
%LOAD_ARM  Load the deck, optionally rewrite the DM grid IN THE LOADED MODEL
%   (the live poke), then set every polarizing element.
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

function I = frame(Sx, Sr, th)
    I = sum(abs(synth(Sx,th) + synth(Sr,th)).^2, 3);
end

function p = fourstep(Sx, Sr, th)
    I1 = frame(Sx,Sr,th(1));  I2 = frame(Sx,Sr,th(2));
    I3 = frame(Sx,Sr,th(3));  I4 = frame(Sx,Sr,th(4));
    p  = atan2(I2-I4, I1-I3);
end

function e = arm_state(A, QWP, iElt)
    load_arm(A, QWP, 0, []);
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

function [best, map] = register_to_dm(A, ix, Mdm, N_G, DX_G, h, msk)
%REGISTER_TO_DM  The instrument's pupil mapping, measured from the trace: one
%   (DM position, detector position) pair per surviving ray.
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
    Fx = scatteredInterpolant(xy_d(1,:).', xy_d(2,:).', xy_to(1,:).', 'linear','linear');
    Fy = scatteredInterpolant(xy_d(1,:).', xy_d(2,:).', xy_to(2,:).', 'linear','linear');
    N = size(h,1);  [cg, rg] = meshgrid(1:N, 1:N);
    cx = sum(cg(msk))/nnz(msk);  cy = sum(rg(msk))/nnz(msk);
    dxp = macos.dx_at(ix.iDET, 'mm');
    a1 = (cg-cx)*dxp;  a2 = (rg-cy)*dxp;
    c_d = mean(xy_d, 2);
    axs = ((1:N_G)-(N_G+1)/2)*DX_G;
    hm = h - mean(h(msk));
    cands = {a1,a2; a1,-a2; -a1,a2; -a1,-a2; a2,a1; a2,-a1; -a2,a1; -a2,-a1};
    best = struct('c',-inf, 'i',1);
    for c = 1:size(cands,1)
        [cc, ht] = reg_corr([0 0 0 0], cands{c,1}, cands{c,2}, c_d, Fx, Fy, axs, Mdm, hm, msk);
        if cc > best.c, best = struct('c',cc, 'ht',ht, 'i',c); end
    end
    A1 = cands{best.i,1};  A2 = cands{best.i,2};
    p = fminsearch(@(q) -reg_corr(q,A1,A2,c_d,Fx,Fy,axs,Mdm,hm,msk), [0 0 0 0], ...
                   optimset('TolX',1e-7,'TolFun',1e-10,'Display','off'));
    [c2, ht2] = reg_corr(p, A1, A2, c_d, Fx, Fy, axs, Mdm, hm, msk);
    if c2 > best.c, best.c = c2;  best.ht = ht2; end
    best.hm = hm;
end

function [c, ht] = reg_corr(p, A1, A2, c_d, Fx, Fy, axs, Mdm, hm, msk)
    s = exp(p(4));  ct = cos(p(3));  st = sin(p(3));
    X = s*(ct*A1 - st*A2) + c_d(1) + p(1);
    Y = s*(st*A1 + ct*A2) + c_d(2) + p(2);
    ht = interpn(axs, axs, Mdm, Fx(X,Y), Fy(X,Y), 'spline', 0);
    ht = ht - mean(ht(msk));
    cc = corrcoef(hm(msk), ht(msk));  c = cc(1,2);
end

function show(Z, msk, N, box, cl, ttl)
    q = nan(N);  q(msk) = Z(msk);  q = sub(q, box);
    nexttile; imagesc(q, 'AlphaData', ~isnan(q)); axis image off;
    if ~isempty(cl), clim(cl); end
    colorbar; title(ttl);
end

function b = beam_box(msk, pad)
%BEAM_BOX  Padded bounding box of the illuminated pixels -- the beam is a
%   small disc on the padded diffraction array.
    [rr, cc] = find(msk);  N = size(msk,1);
    b = [max(1,min(rr)-pad) min(N,max(rr)+pad) max(1,min(cc)-pad) min(N,max(cc)+pad)];
end

function Z = sub(Z, b), Z = Z(b(1):b(2), b(3):b(4)); end

function m = erode_disc(msk, frac)
%ERODE_DISC  Keep the inner FRAC of the illuminated disc (no toolbox needed).
    N = size(msk,1);  [cg, rg] = meshgrid(1:N, 1:N);
    cx = mean(cg(msk));  cy = mean(rg(msk));
    r  = sqrt((cg-cx).^2 + (rg-cy).^2);
    m  = msk & (r <= frac*max(r(msk)));
end
