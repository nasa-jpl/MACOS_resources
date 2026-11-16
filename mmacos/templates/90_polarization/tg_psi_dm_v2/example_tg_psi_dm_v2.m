%EXAMPLE_TG_PSI_DM_V2  Polarization PSI Twyman-Green with a REAL polarizing
%   beamsplitter: a cemented MacNeille cube, modelled as coated engine
%   surfaces at 45 degrees inside the prism glass.
%
%   v1 (../tg_psi_dm) carries the polarization CONCEPTUALLY: the splitter is a
%   perfect-conductor plate plus a compensator, and each arm opens with an
%   IDEAL TrPolarizer at normal incidence.  That rig found a real defect -- the
%   beamsplitter rotates the test arm 7.479 deg and the gauge reads 11.7% high.
%   v2 replaces the concept with the COMPONENT and asks what is left.
%
%   Nothing here is new engine capability.  MACOS has carried trustworthy
%   coated s/p physics at arbitrary AOI since 2026-07-27/28 (the r_p sign fix,
%   the incident-medium fix, the transmission radiometric factor, and the
%   published-data Mueller anchor at ~1e-14), so a PBS is coating physics and
%   needs no ray splitting and no RfPolarizer element: the test arm's deck
%   carries the diagonal in TRANSMISSION going out and REFLECTION coming back,
%   the reference arm's the other way round, and each arm's double-passed QWP
%   swaps its state between the coating's own s and p eigenaxes.  That is the
%   physical "all light to the output port" routing, still two decks because
%   the engine does not split rays.
%
%   Run:  matlab -batch "run('example_tg_psi_dm_v2.m')"     (~4 min)
%   Gates: mmacos/tests/tTgPol2.m
%
%   v1 is NOT modified and NOT superseded; it is built here as the
%   counterexample every comparison is scored against.

clear;  close all;
exdir = fileparts(mfilename('fullpath'));
if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
addpath(fullfile(exdir, '..', 'tg_psi_dm'));   % dm_influence_map (v1, read-only)
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);                         % GridFile= resolves relative to the cwd

MODEL = 256;                       % >= N_G, so the GridData map is not resampled
macos.init(MODEL);

% ---- parameters ---------------------------------------------------------
LAM     = 6.328e-4;                % mm (HeNe)
NGRID   = 63;                      % source ray grid
N_G     = 256;   DX_G = 0.35;      % DM grid: 256 nodes at 0.35 mm = 89 mm span
NACT    = 16;    PITCH = 3.5;      % 16x16 actuators at 3.5 mm = 56 mm DM
POKE_NM = 50;                      % command amplitude, nm of surface
QWP     = 0.25;                    % quarter-wave retardance (waves)
THETAS  = [0 45 90 135];           % the four analyzer steps (deg)
%  Detector-leg architecture: the l2_trade winner, carried over verbatim.
%  The cube build subtracts its own half-side from D_RC_L2 so the
%  Recomb->L2->mask->detector conjugate is the SAME geometry as the plate
%  rig -- otherwise these trims would silently no longer apply.
TAIL = {'tail_arch','fieldlens', 'FL_F',25.02100857, 'FL_Kc',-2.11278288, ...
        'D_MASK_FL',6.277463741, 'DET_TRIM',1.085330067};

fprintf('\n=========================================================\n');
fprintf(' POL-PSI TWYMAN-GREEN v2 -- A REAL MacNeille PBS CUBE\n');
fprintf('=========================================================\n');
fprintf('model %d | ray grid %d | DM %dx%d act @ %.2f mm | pokes %g nm | %.1f nm\n\n', ...
    MODEL, NGRID, NACT, NACT, PITCH, POKE_NM, LAM*1e6);

% =========================================================================
%  1.  The coating: a MacNeille design, and why its TERMINATION matters
% =========================================================================
%  MacNeille (US 2,403,731, 1946), in Macleod's form: choose the prism index
%  so that the internal angle at every H/L interface is Brewster's.  With the
%  classic visible pair (ZnS 2.35 / cryolite 1.35) that fixes n_g = 1.6555 --
%  a dense flint, which is why real MacNeille cubes are not made of BK7.
PBS  = macos.design.pbs_macneille();                  % (1/2 H  L  1/2 H)^4
PBSq = macos.design.pbs_macneille('design','qw');     % H(LH)^4 -- the trap
fprintf('=== the coating ===\n');
fprintf('  prism index (MacNeille condition)  n_g = %.6f\n', PBS.n_glass);
fprintf('  internal angles: theta_H %.4f  theta_L %.4f  (sum %.6f, Brewster)\n', ...
    PBS.theta_H, PBS.theta_L, PBS.theta_H + PBS.theta_L);
fprintf('  %d layers, %s, %.0f quarter waves\n', ...
    size(PBS.layers,1), PBS.design, PBS.qw_total);
fprintf('  layer thickness (nm): '); fprintf('%.2f ', PBS.thk*1e6); fprintf('\n');
fprintf('  ANALYTIC (Macleod characteristic matrix, glass in / glass out):\n');
fprintf('    symmetric (1/2H L 1/2H)^4 : R_s %.8f  R_p %.3e  T_p %.8f\n', ...
    PBS.rt.Rs, PBS.rt.Rp, PBS.rt.Tp);
fprintf('    odd-QW    H(LH)^4         : R_s %.8f  R_p %.3e  T_p %.8f\n', ...
    PBSq.rt.Rs, PBSq.rt.Rp, PBSq.rt.Tp);
fprintf('  -> Brewster at the H/L interfaces equalizes the tilted p admittances\n');
fprintf('     (both 2.7101), so for p the whole stack is ONE HOMOGENEOUS SLAB.\n');
fprintf('     Its two boundaries with the prism are NOT Brewster, so what is\n');
fprintf('     left depends only on the slab''s total p phase: %.0f quarter waves\n', PBS.qw_total);
fprintf('     is a half-wave ABSENTEE (R_p = 0); %.0f is a quarter-wave layer\n', PBSq.qw_total);
fprintf('     (R_p = %.2f%%).  Both satisfy the textbook condition.\n\n', 100*PBSq.rt.Rp);

% =========================================================================
%  2.  The DM surface -> GridFile, and both rigs
% =========================================================================
[Mdm, dminfo] = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, ...
                                 'poke',POKE_NM*1e-6, 'pattern','checker');
macos.write_grid_file('dm2_poked.txt', Mdm);
macos.write_grid_file('dm2_flat.txt',  zeros(N_G));
fprintf('DM map: %.3f nm rms, %.3f nm PtV, %.1f mm span, %.2f mm influence radius\n\n', ...
    1e6*std(Mdm(:)), 1e6*(max(Mdm(:))-min(Mdm(:))), dminfo.span_mm, dminfo.width_mm);

mk2 = @(gf, varargin) macos.design.twyman_green('pbs','cube', 'polarizing',true, ...
    'ngridpts',NGRID, 'to_grid_file',gf, 'to_grid_n',N_G, 'to_grid_dx',DX_G, ...
    'qwp_ret',QWP, TAIL{:}, varargin{:});
mk1 = @(gf) macos.design.twyman_green('polarizing',true, 'ngridpts',NGRID, ...
    'to_grid_file',gf, 'to_grid_n',N_G, 'to_grid_dx',DX_G, 'qwp_ret',QWP, TAIL{:});

G  = mk2('dm2_poked.txt');   G0 = mk2('dm2_flat.txt');
G.bt.emit('v2_test.in');  G0.bt.emit('v2_base.in');  G.br.emit('v2_ref.in');
V1 = mk1('dm2_flat.txt');
V1.bt.emit('v1_test.in');  V1.br.emit('v1_ref.in');

AT = arm_desc('v2_test.in', G.bt,  G.T,  G.P.qwp_test_deg);   % DM poked
AB = arm_desc('v2_base.in', G0.bt, G0.T, G.P.qwp_test_deg);   % DM flat
AR = arm_desc('v2_ref.in',  G.br,  G.R,  G.P.qwp_ref_deg);
B1T = arm_desc('v1_test.in', V1.bt, V1.T, V1.P.qwp_test_deg); % the v1 counterexample
B1R = arm_desc('v1_ref.in',  V1.br, V1.R, V1.P.qwp_ref_deg);
OQ  = G.P.out_qwp_deg;

fprintf('v2 test arm %2d elements | cube faces+diagonal at %s (out) / %s (back)\n', ...
    numel(G.bt.E), mat2str(G.T.iPBSf), mat2str(G.T.iPBSr));
fprintf('v2 ref  arm %2d elements | %s (out) / %s (back)\n', ...
    numel(G.br.E), mat2str(G.R.iPBSf), mat2str(G.R.iPBSr));
fprintf('v1 test arm %2d | v1 ref arm %2d elements (the plate rig, for comparison)\n\n', ...
    numel(V1.bt.E), numel(V1.br.E));
%  The two arms must recombine into the SAME output leg, and -- the cube's
%  whole point -- their glass paths must balance with NO compensator.
assert(norm(G.bt.E(G.T.iRC).psi(:) - G.br.E(G.R.iRC).psi(:)) < 1e-12, ...
    'the two arms do not share the output-leg direction');
d_path = abs(G.bt.path_len - G.br.path_len);
fprintf('arm geometric path difference: %.3e mm (no compensator plate in the rig)\n\n', d_path);

% =========================================================================
%  GATE 1 -- the emitted deck really carries the stack, in the right place
% =========================================================================
fprintf('=== GATE 1: Coating= round-trips, and Tranche-1 order holds ===\n');
g1 = struct();
for f = {'v2_test.in','v2_ref.in'}
    D = read_deck(f{1});
    ipol = find(ismember(D.element, {'TrPolarizer','WavePlate'}));
    ipo  = find(~strcmp(D.proptype, 'Geometric'));
    assert(isempty(ipo) || max(ipol) < min(ipo), ...
        'GATE 1 FAILED: %s has a polarizing element after a physical-optics leg', f{1});
    fprintf('  %-12s pol elements at %s; %d physical-optics legs\n', ...
        f{1}, mat2str(ipol(:).'), numel(ipo));
end
%  Read the stack BACK OUT OF THE ENGINE.  The Rx keyword is in WAVES at the
%  deck's Wavelen and the parser scales it by Wavelen/IndRef; coat_get returns
%  PHYSICAL thickness.  So this round-trip is a real check of that conversion,
%  not a tautology -- it is the one place the two coating units meet.
macos.load_rx('v2_test.in');
iD = G.T.iPBSf(2);   iF = G.T.iPBSf(1);
cD = macos.coating(iD);   cF = macos.coating(iF);
g1.thk_rel = max(abs(cD.thickness(:) - PBS.thk(:))./PBS.thk(:));
g1.idx_err = max(abs(cD.index(:) - PBS.layers(:,1)));
fprintf('  diagonal (elt %d): %d layers | thickness round-trip %.2e | index %.2e\n', ...
    iD, cD.n_layer, g1.thk_rel, g1.idx_err);
fprintf('  face     (elt %d): %d layer  n %.3f, %.2f nm (MgF2 quarter wave = %.2f nm)\n', ...
    iF, cF.n_layer, cF.index(1), cF.thickness(1)*1e6, LAM*1e6/(4*1.38));
assert(cD.n_layer == size(PBS.layers,1), 'GATE 1 FAILED: layer count');
assert(g1.thk_rel < 1e-12 && g1.idx_err < 1e-12, ...
    'GATE 1 FAILED: Coating= does not round-trip (%.2e / %.2e)', g1.thk_rel, g1.idx_err);
%  The tripwire a source read cannot give: the grid must respond to the
%  analyzer.  Run it on a LINEAR arm state (oq_deg = 0, no output QWP action
%  yet) -- a circular state is analyzer-invariant in power and would pass
%  vacuously.
g1.an_ratio = sum(sum(grid_intensity(set_pol_align(AT,AT.base,0), QWP, 90))) / ...
              sum(sum(grid_intensity(set_pol_align(AT,AT.base,0), QWP,  0)));
fprintf('  grid power at analyzer 90 / at 0 = %.4e  (1.0 would mean the grid is blind)\n', ...
    g1.an_ratio);
assert(abs(g1.an_ratio - 1) > 0.1, 'GATE 1 FAILED: the grid does not see the analyzer');
%  A PSI that adds two arms pixel by pixel is only meaningful if the two arms
%  land on the SAME pixels.  In v1 the arms carry different glass-path
%  bookkeeping so an exact tie is not expected; the cube's ports are
%  symmetric, so here it should be much tighter -- check, do not assume.
[dxT, cT, rT] = det_geom(AT);   [dxR, cR, rR] = det_geom(AR);
g1.dpitch_rel = abs(dxT-dxR)/max(dxT,dxR);
g1.dcen = norm(cT-cR);   g1.drad = abs(rT-rR);
fprintf('  detector pitch test %.9f mm | ref %.9f mm | relative diff %.2e\n', ...
    dxT, dxR, g1.dpitch_rel);
fprintf('  centroid offset %.2e px | second-moment radius diff %.2e px\n', g1.dcen, g1.drad);
assert(g1.dpitch_rel < 1e-9, 'GATE 1 FAILED: detector sampling differs (%.2e)', g1.dpitch_rel);
assert(g1.dcen < 0.05 && g1.drad < 0.05, 'GATE 1 FAILED: the two beams are not registered');
fprintf('GATE 1 PASS\n\n');

% =========================================================================
%  GATE 2 -- the engine's coated 45-deg surface IS the textbook stack
% =========================================================================
%  The whole v2 claim rests on this.  Probe the diagonal with a PURE s and a
%  PURE p input and take the amplitude ratio across it; compare with
%  macos.design.thinfilm_rt, which is written from Macleod ch. 2 and NEVER
%  transcribed from elemsub.F (an "analytic" copied out of the engine is
%  circular in exactly the coefficient it should check -- the r_p-sign lesson).
%
%  The cemented cube collapses every normalization subtlety: n_inc == n_sub,
%  so the radiometric factor sqrt(n_sub cos_sub/(n_inc cos_inc)) and Macleod's
%  tangential-vs-Fresnel factor cos_sub/cos_inc are BOTH identically 1, and
%  the engine's power-amplitude TP/TS compare directly with sqrt(T).
fprintf('=== GATE 2: engine coated diagonal vs the Macleod analytic ===\n');
%  source frame for the +x chief: xGrid = yhat (= p at the diagonal),
%  yGrid = zhat (= s).  PolIn is set to the probe axis so it passes untouched.
g2 = struct();
g2.eng = probe_rt('v2_test.in', G.T.iPBSf, 'v2_ref.in', G.R.iPBSf);
A = PBS.rt;
fprintf('%8s %20s %20s %11s\n', '', 'engine', 'analytic', 'rel err');
nm = {'T_s','T_p','R_s','R_p'};  fl = {'Ts','Tp','Rs','Rp'};
for q = 1:4
    e = g2.eng.(fl{q});  a = A.(fl{q});
    if a > 1e-9                       % R_p is a NULL: a relative error on a
        r = abs(e-a)/a;  tag = '';    % quantity that is zero is meaningless,
    else                              % so report it absolutely and say so.
        r = abs(e-a);    tag = ' (abs)';
    end
    fprintf('%8s %20.12e %20.12e %11.2e%s\n', nm{q}, e, a, r, tag);
    g2.rel.(fl{q}) = r;
end
g2.en_s = g2.eng.Rs + g2.eng.Ts;   g2.en_p = g2.eng.Rp + g2.eng.Tp;
fprintf('  ENERGY (the two arms'' decks measure R and T of the SAME coating):\n');
fprintf('    R_s + T_s = %.12f      R_p + T_p = %.12f\n', g2.en_s, g2.en_p);
fprintf('  extinction  T_p/T_s = %.4g : 1   (a real MacNeille cube is ~1e3)\n', ...
    g2.eng.Tp/g2.eng.Ts);
assert(g2.rel.Rs < 1e-7 && g2.rel.Tp < 1e-7, ...
    'GATE 2 FAILED: engine R_s/T_p disagree with the analytic');
assert(g2.rel.Ts < 1e-4, 'GATE 2 FAILED: T_s off by %.2e', g2.rel.Ts);
assert(g2.eng.Rp < 1e-9, 'GATE 2 FAILED: the MacNeille p-null is not there (R_p=%.2e)', g2.eng.Rp);
assert(abs(g2.en_s-1) < 1e-9 && abs(g2.en_p-1) < 1e-9, 'GATE 2 FAILED: energy not conserved');
%  NON-VACUITY -- install the odd-QW stack, which satisfies the SAME textbook
%  Brewster condition, and the p-null assertion above must fail.  Without this
%  the gate would pass on any stack the analytic happens to agree with.
Gq = mk2('dm2_flat.txt', 'pbs_coat', PBSq.layers);
Gq.bt.emit('v2_qw_test.in');  Gq.br.emit('v2_qw_ref.in');
g2.qw = probe_rt('v2_qw_test.in', Gq.T.iPBSf, 'v2_qw_ref.in', Gq.R.iPBSf);
fprintf('  NON-VACUITY, the odd-QW stack: engine R_p = %.4e (analytic %.4e), T_p = %.6f\n', ...
    g2.qw.Rp, PBSq.rt.Rp, g2.qw.Tp);
%  1e-3, not 1e-9: R at a 45-deg interface varies quadratically with AOI and
%  the beam is collimated to a few tens of microradians, not perfectly, so the
%  pupil median carries a real (physical) spread.  Measured 5e-5.
assert(abs(g2.qw.Rp - PBSq.rt.Rp)/PBSq.rt.Rp < 1e-3, ...
    'GATE 2 FAILED: engine and analytic disagree on the odd-QW stack (%.2e)', ...
    abs(g2.qw.Rp - PBSq.rt.Rp)/PBSq.rt.Rp);
assert(~(g2.qw.Rp < 1e-9), 'GATE 2 non-vacuity FAILED: the p-null assertion cannot fail');
fprintf('GATE 2 PASS\n\n');

% =========================================================================
%  GATE 3 -- the arms leave ORTHOGONAL, with no alignment step at all
% =========================================================================
%  v1 needed a solved waveplate clock here (+3.768 deg) because its splitter
%  rotated the test arm.  The cube does not, and the reason is structural:
%  each arm's state sits ON a coating eigenaxis (the test arm on p, the
%  reference arm on s), where a diattenuator cannot rotate anything.  v1 put
%  the state at 45 deg to the splitter's axes, where the rotation is FIRST
%  order in the diattenuation.
fprintf('=== GATE 3: arm states at the recombination plane ===\n');
g3 = struct();
g3.az_t  = arm_azimuth(AB,  QWP, AB.base);
g3.az_r  = arm_azimuth(AR,  QWP, AR.base);
g3.sep   = abs(wrap180(g3.az_t - g3.az_r)) - 90;
g3.v1_t  = arm_azimuth(B1T, QWP, B1T.base);
g3.v1_r  = arm_azimuth(B1R, QWP, B1R.base);
g3.v1sep = abs(wrap180(g3.v1_t - g3.v1_r)) - 90;
fprintf('  v2 cube : test %+11.6f deg | ref %+11.6f deg | from orthogonal %+.3e deg\n', ...
    g3.az_t, g3.az_r, g3.sep);
fprintf('  v1 plate: test %+11.6f deg | ref %+11.6f deg | from orthogonal %+.4f deg\n', ...
    g3.v1_t, g3.v1_r, g3.v1sep);
fprintf('  -> the cube removes the beamsplitter rotation by a factor %.3g\n', ...
    abs(g3.v1sep)/max(abs(g3.sep), eps));
assert(abs(g3.sep) < 1e-3, 'GATE 3 FAILED: v2 arms %.3e deg from orthogonal', g3.sep);
assert(abs(g3.v1sep) > 1, 'GATE 3 non-vacuity FAILED: v1 was supposed to be misaligned');
%  With the pair already orthogonal LINEAR, a quarter wave at 45 deg to them
%  makes them orthogonal CIRCULAR -- what the rotating analyzer needs.  In v2
%  that azimuth is a DESIGN CONSTANT, not a solved quantity.
[AT, AB, AR] = deal(set_pol_align(AT, AT.base, OQ), ...
                    set_pol_align(AB, AB.base, OQ), ...
                    set_pol_align(AR, AR.base, OQ));
[st, sr] = deal(arm_state(AT, QWP, AT.iOQ), arm_state(AR, QWP, AR.iOQ));
g3.ortho = abs(st'*sr)/(norm(st)*norm(sr));
fprintf('  after the output QWP (fixed at %g deg): test |b/a| %.6f arg %+7.2f\n', ...
    OQ, abs(st(2)/st(1)), rad2deg(angle(st(2)/st(1))));
fprintf('                                          ref  |b/a| %.6f arg %+7.2f\n', ...
    abs(sr(2)/sr(1)), rad2deg(angle(sr(2)/sr(1))));
fprintf('  circular is |b/a| = 1, arg = -+90;  |<test|ref>| = %.2e\n', g3.ortho);
assert(g3.ortho < 1e-4, 'GATE 3 FAILED: arms %.2e from orthogonal after the QWP', g3.ortho);
fprintf('GATE 3 PASS\n\n');

% =========================================================================
%  GATE 4 -- three traces per arm still span every analyzer angle
% =========================================================================
fprintf('=== GATE 4: the analyzer basis is still exact on the cube rig ===\n');
tb = tic;
St = analyzer_basis(AT, QWP);      % DM poked
Sb = analyzer_basis(AB, QWP);      % DM flat
Sr = analyzer_basis(AR, QWP);      % reference arm
g4 = struct('t_basis', toc(tb), 'theta',[], 'rel',[]);
fprintf('  9 traces (3 per arm x 3 decks) in %.2f s\n', g4.t_basis);
for th = [23.7 71.3 137.2 -18.0]
    Edir = arm_field(AT, QWP, th);
    r = max(abs(reshape(synth(St,th) - Edir, [], 1)))/max(abs(Edir(:)));
    g4.theta(end+1) = th;  g4.rel(end+1) = r;
    fprintf('  theta %7.2f deg : synthesized vs direct trace, rel %.3e\n', th, r);
end
assert(max(g4.rel) < 1e-8, 'GATE 4 FAILED: the analyzer basis does not span the sweep');
%  and the sweep it generates must be a pure cos(2 theta).  The 6-theta bin is
%  a gate ONLY on directly traced frames: a frame synthesized from the
%  quadratic basis is degree 2 in 2 theta BY CONSTRUCTION and would pass
%  against any engine at all.
nsw = 64;  th_sw = (0:nsw-1)/nsw*180;
Isw = zeros(size(St.A,1), size(St.A,2), nsw);
for k = 1:nsw, Isw(:,:,k) = sum(abs(synth(St,th_sw(k)) + synth(Sr,th_sw(k))).^2, 3); end
Ibar = mean(Isw,3);  msk = Ibar > 0.1*max(Ibar(:));
[v_s, h4_s, h6_s] = harmonics(Isw, msk);
nd = 12;  th_d = (0:nd-1)/nd*180;
Id = zeros(size(St.A,1), size(St.A,2), nd);
td = tic;
for k = 1:nd
    Id(:,:,k) = sum(abs(arm_field(AT,QWP,th_d(k)) + arm_field(AR,QWP,th_d(k))).^2, 3);
end
[v_d, h4_d, h6_d] = harmonics(Id, msk);
g4.vis = v_d;  g4.h4 = h4_d;  g4.h6 = h6_d;  g4.h4_syn = h4_s;  g4.t_direct = toc(td);
fprintf('  %d TRACED angles (%.2f s): visibility %.6f | 4t/2t %.3e | 6t/2t %.3e\n', ...
    nd, g4.t_direct, v_d, h4_d, h6_d);
fprintf('  synthesized             : visibility %.6f | 4t/2t %.3e | 6t/2t %.3e (0 by construction)\n', ...
    v_s, h4_s, h6_s);
assert(g4.vis > 0.99, 'GATE 4 FAILED: fringe visibility %.4f', g4.vis);
assert(h6_d < 1e-8, 'GATE 4 FAILED: traced sweep has content above 4 theta (%.2e)', h6_d);
fprintf('GATE 4 PASS\n\n');

% =========================================================================
%  GATE 5 -- scale, and WHAT AN ALIGNMENT ERROR NOW COSTS
% =========================================================================
%  Pin the chain on a known input first: a uniform grid piston dz shortens the
%  double-passed path by 2 dz, so the recovered phase must move by 4 pi dz/lam.
fprintf('=== GATE 5: PSI scale on a known 20 nm surface piston ===\n');
dz = 20e-6;  expect = 4*pi*dz/LAM;
macos.write_grid_file('dm2_piston.txt', dz*ones(N_G));
Gp = mk2('dm2_piston.txt');  Gp.bt.emit('v2_piston.in');
Ap = set_pol_align(arm_desc('v2_piston.in', Gp.bt, Gp.T, G.P.qwp_test_deg), ...
                   G.P.qwp_test_deg, OQ);
Sp = analyzer_basis(Ap, QWP);
[dphi_grid, ~, vis_al] = psi_diff(Sp, Sb, Sr, THETAS, msk);
Ash = AB;  Ash.shift = struct('elt', G0.T.iTO, 'dz', dz);   % the same, rigidly
dphi_shift = psi_diff(analyzer_basis(Ash,QWP), Sb, Sr, THETAS, msk);
g5 = struct('grid_rad', median(dphi_grid(msk)), 'shift_rad', median(dphi_shift(msk)), ...
            'expect_rad', expect, 'vis', median(vis_al(msk)));
g5.sign = sign(g5.grid_rad);
g5.gain = abs(g5.grid_rad)/expect;
g5.err_nm = abs(abs(g5.grid_rad) - expect)/(4*pi)*LAM*1e6;
SGN = g5.sign;
fprintf('  grid piston %g nm -> %+.6f rad (expect %+.6f)  gain %.6f, err %.4f nm\n', ...
    dz*1e6, g5.grid_rad, g5.sign*expect, g5.gain, g5.err_nm);
fprintf('  rigid optic shift    -> %+.6f rad  (grid vs rigid %.2e rad)\n', ...
    g5.shift_rad, abs(g5.grid_rad - g5.shift_rad));
fprintf('  visibility %.6f\n', g5.vis);
assert(abs(g5.gain - 1) < 1e-3, 'GATE 5 FAILED: PSI gain %.6f', g5.gain);
assert(abs(g5.grid_rad - g5.shift_rad) < 1e-3*expect, ...
    'GATE 5 FAILED: a grid piston and a rigid shift disagree');
%  Now the v2 calibration story.  v1's headline was that a waveplate error
%  costs SCALE and hides from the fringe monitor.  Measure the same ladder on
%  both rigs.  The cube inverts it: on the REFLECTING return the coating
%  re-projects the state onto its own eigenaxis, so an azimuth error is
%  CLEANED (to r_p, which is zero) and costs only throughput; on the
%  TRANSMITTING return it is cleaned only to the extinction ratio t_s/t_p.
fprintf('\n  --- arm-waveplate azimuth error: what each rig does with it ---\n');
eps_l = [0 2 3.768 5 10];
g5.eps = eps_l;  g5.v2_sep_t = zeros(size(eps_l));  g5.v2_sep_r = g5.v2_sep_t;
g5.v1_sep = g5.v2_sep_t;  g5.v2_gain = g5.v2_sep_t;  g5.v2_vis = g5.v2_sep_t;
az_r2 = arm_azimuth(AR, QWP, AR.base);   az_t2 = arm_azimuth(AB, QWP, AB.base);
az_r1 = arm_azimuth(B1R, QWP, B1R.base);
fprintf('  %5s | %12s %12s | %12s | %10s %10s\n', 'eps', ...
    'v2 |sep|-90', '(ref-arm err)', 'v1 |sep|-90', 'v2 gain', 'v2 vis');
for q = 1:numel(eps_l)
    e = eps_l(q);
    g5.v2_sep_t(q) = abs(wrap180(arm_azimuth(AB,  QWP, AB.base+e)  - az_r2)) - 90;
    g5.v2_sep_r(q) = abs(wrap180(arm_azimuth(AR,  QWP, AR.base+e)  - az_t2)) - 90;
    g5.v1_sep(q)   = abs(wrap180(arm_azimuth(B1T, QWP, B1T.base+e) - az_r1)) - 90;
    Ae = set_pol_align(AB, AB.base + e, OQ);
    Ape = set_pol_align(Ap, Ap.base + e, OQ);
    Sbe = analyzer_basis(Ae, QWP);  Spe = analyzer_basis(Ape, QWP);
    [de, ~, ve] = psi_diff(Spe, Sbe, Sr, THETAS, msk);
    g5.v2_gain(q) = abs(median(de(msk)))/expect;
    g5.v2_vis(q)  = median(ve(msk));
    fprintf('  %5.1f | %12.3e %12.3e | %12.4f | %10.6f %10.6f\n', e, ...
        g5.v2_sep_t(q), g5.v2_sep_r(q), g5.v1_sep(q), g5.v2_gain(q), g5.v2_vis(q));
end
fprintf('  -> the cube''s REFLECTING arm is cleaned to r_p (= 0): %.1e deg at 10 deg of error.\n', ...
    abs(g5.v2_sep_t(end)));
fprintf('     Its TRANSMITTING arm is cleaned only to t_s/t_p = %.4f: %.3f deg.\n', ...
    sqrt(g2.eng.Ts/g2.eng.Tp), abs(g5.v2_sep_r(end)));
fprintf('     v1, with no cleanup at all: %.2f deg.\n', abs(g5.v1_sep(end)));
fprintf('     (the v1 column is NON-MONOTONIC on purpose: v1 STARTS %.3f deg off, so\n', ...
    abs(g5.v1_sep(1)));
fprintf('      its curve dips through a minimum near eps = 3.768 -- which IS the\n');
fprintf('      waveplate clock v1 has to solve for.  v2 has no such minimum to find.)\n');
fprintf('  -> and in v2 the error costs CONTRAST (%.6f -> %.6f), not SCALE\n', ...
    g5.v2_vis(1), g5.v2_vis(end));
fprintf('     (gain %.6f -> %.6f).  That is the INVERSE of the v1 finding:\n', ...
    g5.v2_gain(1), g5.v2_gain(end));
fprintf('     the PBS turns an invisible systematic into a visible one.\n');
assert(abs(g5.v2_gain(end) - 1) < 1e-3, ...
    'GATE 5 FAILED: v2 gain moved to %.6f under a 10 deg waveplate error', g5.v2_gain(end));
assert(abs(g5.v2_sep_t(end)) < 1e-3, 'GATE 5 FAILED: the reflecting arm was not cleaned');
assert(abs(g5.v1_sep(end)) > 1, 'GATE 5 non-vacuity FAILED: v1 must be sensitive here');
fprintf('GATE 5 PASS: h = %+d * psi * lambda/(4 pi), gain %.6f\n\n', SGN, g5.gain);

% =========================================================================
%  GATE 6 -- where the light goes
% =========================================================================
%  A PBS earns its place by putting BOTH returns in the output port; the plate
%  rig throws half of each arm back at the source.  Score it against the
%  declared stack: the ideal cube delivers |t_p r_s|^2 of the input through
%  four AR'd faces per arm.
fprintf('=== GATE 6: output-port efficiency ===\n');
g6e = struct();
g6e.P_v2 = sum(sum(sum(abs(Sb.A + Sr.A).^2)));
S1b = analyzer_basis(set_pol_align(B1T, B1T.base, 0), QWP);
S1r = analyzer_basis(set_pol_align(B1R, B1R.base, 0), QWP);
g6e.P_v1 = sum(sum(sum(abs(S1b.A + S1r.A).^2)));
g6e.ratio = g6e.P_v2/g6e.P_v1;
%  the coating+face budget, from the declared stack alone
ar = macos.design.thinfilm_rt([1.38, LAM/(4*1.38)], 1.0, PBS.n_glass, 0, LAM);
g6e.face_T = ar.Ts;
g6e.pred = g2.eng.Tp*g2.eng.Rs*g6e.face_T^4;
fprintf('  detected power  v2 cube %.6f | v1 plate %.6f | ratio %.4f\n', ...
    g6e.P_v2, g6e.P_v1, g6e.ratio);
fprintf('  per-arm coating budget T_p*R_s = %.6f, four AR faces at T = %.6f each\n', ...
    g2.eng.Tp*g2.eng.Rs, g6e.face_T);
fprintf('    -> predicted per-arm throughput %.6f (bare glass would be %.6f)\n', ...
    g6e.pred, g2.eng.Tp*g2.eng.Rs*(1-((PBS.n_glass-1)/(PBS.n_glass+1))^2)^4);
assert(g6e.ratio > 2, 'GATE 6 FAILED: the cube should roughly double the delivered power');
fprintf('GATE 6 PASS\n\n');

% =========================================================================
%  7.  The measurement and the closure
% =========================================================================
fprintf('=== MEASUREMENT: four-step polarization PSI (differential) ===\n');
[dphi, ~, vis] = psi_diff(St, Sb, Sr, THETAS, msk);
h = SGN * dphi * LAM/(4*pi);
fprintf('  fringe visibility %.6f | recovered surface %.3f nm rms in the pupil\n', ...
    median(vis(msk)), 1e6*std(h(msk)));
dphi_ls = psi_diff(St, Sb, Sr, th_sw, msk);
d_est = angle(exp(1i*(dphi - dphi_ls)));
est_nm = 1e6*std(d_est(msk)*LAM/(4*pi));
fprintf('  four-step vs %d-angle least squares: %.3e nm rms of surface\n', nsw, est_nm);

fprintf('\n=== CLOSURE: recovered surface vs the injected DM map ===\n');
[best, dmap] = register_to_dm(AT, G.T, Mdm, N_G, DX_G, h, msk);
res  = best.hm(msk) - best.ht(msk);
mskin = erode_disc(msk, 0.92);
resin = best.hm(mskin) - best.ht(mskin);
g7 = struct('corr',best.c, 'axes',best.name, 'resid_nm',1e6*std(res), ...
            'resid_in_nm',1e6*std(resin), 'truth_nm',1e6*std(best.ht(msk)), ...
            'recov_nm',1e6*std(best.hm(msk)), 'mag',dmap.mag, ...
            'anam_pct',dmap.anam_pct, 'rot_deg',dmap.rot_deg, 'nonlin_mm',dmap.nonlin_mm);
fprintf(['  detector->DM magnification %.4f, rotation %+.2f deg, anamorphic %.4f%%,\n' ...
         '  nonlinear distortion %.4f mm rms; axis convention %s\n'], ...
    g7.mag, g7.rot_deg, g7.anam_pct, g7.nonlin_mm, g7.axes);
fprintf('  truth %.2f nm rms | recovered %.2f nm rms | correlation %.6f\n', ...
    g7.truth_nm, g7.recov_nm, g7.corr);
fprintf('  residual %.3f nm rms interior (%.3f whole pupil)\n', g7.resid_in_nm, g7.resid_nm);
assert(g7.corr > 0.99, 'CLOSURE FAILED: correlation %.4f', g7.corr);

% =========================================================================
%  8.  Figures and artefacts
% =========================================================================
fig = figure('Position',[80 80 1500 460], 'Color','w', 'Visible','off');
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
bb = beam_box(msk,4);
cl = [-1 1]*1e6*max(abs(best.ht(msk)));
show(best.ht*1e6, msk, MODEL, bb, cl, sprintf('injected DM surface (%.2f nm rms)', g7.truth_nm));
show(best.hm*1e6, msk, MODEL, bb, cl, sprintf('v2 recovered (%.2f nm rms)', g7.recov_nm));
show((best.hm-best.ht)*1e6, msk, MODEL, bb, [], ...
    sprintf('residual (%.3f nm rms interior)', g7.resid_in_nm));
exportgraphics(fig, 'tg_psi_dm_v2_recovery.png', 'Resolution',140);

fig2 = figure('Position',[80 80 1150 430], 'Color','w', 'Visible','off');
subplot(1,2,1);
semilogy(eps_l, max(abs(g5.v1_sep),1e-12), 'o-', 'LineWidth',1.6); hold on;
semilogy(eps_l, max(abs(g5.v2_sep_r),1e-12), 's-', 'LineWidth',1.6);
semilogy(eps_l, max(abs(g5.v2_sep_t),1e-12), '^-', 'LineWidth',1.6); grid on;
xlabel('arm-waveplate azimuth error (deg)');
ylabel('|arm separation| - 90 (deg)');
title('who cleans an alignment error');
legend({'v1 plate (no cleanup)','v2 cube, transmitting arm (to t_s/t_p)', ...
        'v2 cube, reflecting arm (to r_p = 0)'}, 'Location','southeast');
subplot(1,2,2);
yyaxis left;  plot(eps_l, g5.v2_gain, 'o-', 'LineWidth',1.6); ylabel('PSI scale gain');
ylim([0.99 1.01]);
yyaxis right; plot(eps_l, g5.v2_vis, 's-', 'LineWidth',1.6); ylabel('fringe visibility');
grid on; xlabel('arm-waveplate azimuth error (deg)');
title('v2: the error costs contrast, not scale');
exportgraphics(fig2, 'tg_psi_dm_v2_sensitivity.png', 'Resolution',140);

save('tg_psi_dm_v2.mat', 'g1','g2','g3','g4','g5','g6e','g7','best','h','msk', ...
     'PBS','PBSq','eps_l','LAM','MODEL','NGRID','N_G','DX_G','d_path','-v7.3');

fprintf('\n=========================================================\n');
fprintf(' SUMMARY -- v1 plate (ideal polarizers) vs v2 MacNeille cube\n');
fprintf('=========================================================\n');
fprintf(' %-34s %14s %14s\n', '', 'v1 plate', 'v2 cube');
fprintf(' %-34s %14.4f %14.2e\n', 'arm rotation from orthogonal (deg)', g3.v1sep, g3.sep);
fprintf(' %-34s %14.5f %14.6f\n', 'PSI scale gain, as designed', 1.11661, g5.gain);
fprintf(' %-34s %14.6f %14.6f\n', 'fringe visibility', 0.996612, g5.vis);
fprintf(' %-34s %14.4f %14.4f\n', 'delivered power (arb, same source)', g6e.P_v1, g6e.P_v2);
fprintf(' %-34s %14s %14.4f\n', 'sep after 10 deg WP error (deg)', ...
    sprintf('%.4f', abs(g5.v1_sep(end))), abs(g5.v2_sep_r(end)));
fprintf(' %-34s %14s %14s\n', 'alignment step needed', 'yes (+3.768 deg)', 'none');
fprintf(' %-34s %14s %14.3e\n', 'compensator plate', 'required', d_path);
fprintf('\n coating: engine vs Macleod  R_s %.2e | T_p %.2e | R_p %.2e (null)\n', ...
    g2.rel.Rs, g2.rel.Tp, g2.eng.Rp);
fprintf(' DM closure: corr %.6f, residual %.3f nm rms interior on %.2f nm\n', ...
    g7.corr, g7.resid_in_nm, g7.truth_nm);
fprintf(' artefacts: v2_*.in, v1_*.in, tg_psi_dm_v2.mat, tg_psi_dm_v2_*.png\n');
fprintf('=========================================================\n');

% =========================================================================
%  LOCAL FUNCTIONS
%  Carried over from ../tg_psi_dm/example_tg_psi_dm.m rather than shared:
%  v1 is FROZEN as the demo default, and a shared helper file would make
%  every v2 edit a v1 risk.  Merging the two (and bench_ifo_pol's PSI/Jones
%  path) into one source is the deferred stretch item -- see the README.
% =========================================================================
function A = arm_desc(rx, b, ix, base_deg)
%ARM_DESC  Everything a trace of one arm needs: the deck, the pol element
%   indices, and that arm's waveplate azimuths.
    nm = {b.E.name};
    A = struct('rx', rx, 'b', b, ...
        'iPol', find(strcmp(nm,'PolIn'),1), ...
        'iQ',   find(contains(nm,'QWP') & ~strcmp(nm,'OutQWP')), ...
        'base', base_deg, 'qwp_deg', base_deg, 'oq_deg', 0, ...
        'iRC', ix.iRC, 'iOQ', ix.iOutQWP, 'iAn', ix.iAnalyzer, ...
        'iDET', ix.iDET, 'shift', []);
    assert(numel(A.iQ) == 2, 'arm_desc: expected a double-passed arm QWP');
end

function A = set_pol_align(A, qwp_deg, oq_deg)
    A.qwp_deg = qwp_deg;  A.oq_deg = oq_deg;
end

function a = lax(psi, deg)
%LAX  A polarization axis DEG degrees from local x in the transverse plane of
%   a beam along PSI -- the same right-handed frame the Bench emitter uses, so
%   "45 deg" is the same physical direction in every folded leg.  On the cube
%   rig local 0 is the diagonal's p axis and local 90 its s axis, in EVERY
%   leg, which is why both arm plates and the output plate sit at 45.
    u1 = macos.design.Bench.perp(psi(:));  u2 = cross(psi(:), u1);
    a = cosd(deg)*u1 + sind(deg)*u2;  a = a(:).';
end

function x = wrap180(x)
    x = mod(x + 90, 180) - 90;
end

function load_arm(A, QWP, an_deg)
%LOAD_ARM  Load the deck and set every polarizing element.  A double-passed
%   physical plate has ONE global fast axis: derive it from the FORWARD
%   element's frame and give the same vector to both passes.
    macos.load_rx(A.rx);  b = A.b;
    if ~isempty(A.shift)
        p = macos.get_elt_psi(A.shift.elt);  v = macos.get_elt_vpt(A.shift.elt);
        macos.set_elt_vpt(A.shift.elt, v + A.shift.dz*p);
    end
    macos.polarizer(A.iPol, 'axis', lax(b.E(A.iPol).psi, 45));
    qa = lax(b.E(A.iQ(1)).psi, A.qwp_deg);
    for j = 1:2, macos.waveplate(A.iQ(j), 'axis', qa, 'retardance', QWP); end
    macos.waveplate(A.iOQ, 'axis', lax(b.E(A.iOQ).psi, A.oq_deg), 'retardance', QWP);
    macos.polarizer(A.iAn, 'axis', lax(b.E(A.iAn).psi, an_deg));
    macos.polarization('on', 'Ex',[1/sqrt(2) 0], 'Ey',[1/sqrt(2) 0]);
    macos.vector_diffraction(true);
end

function E = arm_field(A, QWP, an_deg)
%ARM_FIELD  The complex VECTOR field at the detector, N x N x 3 (Ex,Ey,Ez).
    load_arm(A, QWP, an_deg);
    E = cat(3, macos.complex_field(A.iDET,'plane',1), ...
               macos.complex_field(A.iDET,'plane',2), ...
               macos.complex_field(A.iDET,'plane',3));
end

function I = grid_intensity(A, QWP, an_deg)
    load_arm(A, QWP, an_deg);
    I = macos.intensity(A.iDET);
end

function e = arm_state(A, QWP, iElt)
%ARM_STATE  The arm's Jones state (a,b) at element IELT, in that leg's
%   transverse frame, as a pupil median -- read where the physics puts it
%   (per ray) rather than after a grid assembly.
    load_arm(A, QWP, 0);
    macos.trace(iElt);  f = macos.ray_field(iElt);
    ok = f.status == 0;
    psi = A.b.E(iElt).psi(:);
    u1 = macos.design.Bench.perp(psi);  u2 = cross(psi, u1);
    e1 = f.Ex*u1(1) + f.Ey*u1(2) + f.Ez*u1(3);
    e2 = f.Ex*u2(1) + f.Ey*u2(2) + f.Ez*u2(3);
    r  = e2(ok)./e1(ok);
    a  = median(abs(e1(ok)));
    e  = [a; a*(median(real(r)) + 1i*median(imag(r)))];
end

function az = arm_azimuth(A, QWP, qwp_deg)
    A.qwp_deg = qwp_deg;
    e = arm_state(A, QWP, A.iRC);
    az = 0.5*atan2d(2*real(conj(e(1))*e(2)), abs(e(1))^2 - abs(e(2))^2);
end

function S = analyzer_basis(A, QWP)
%ANALYZER_BASIS  Three traces that span every analyzer angle.
    E0  = arm_field(A, QWP, 0);
    E45 = arm_field(A, QWP, 45);
    E90 = arm_field(A, QWP, 90);
    S = struct('A', E0, 'C', E90, 'B', 2*E45 - E0 - E90);
end

function E = synth(S, th)
    c = cosd(th);  s = sind(th);
    E = c^2*S.A + c*s*S.B + s^2*S.C;
end

function R = probe_rt(rx_t, iT, rx_r, iR)
%PROBE_RT  The engine's R and T of the cemented diagonal, per polarization.
%   Probe with a PURE s or PURE p input and take the amplitude ratio ACROSS
%   the diagonal -- the field just after the entrance face vs the field just
%   after the diagonal.  The glass between them is lossless, so the
%   propagation phase has unit modulus and drops out of |ratio|.
%
%   Source frame for the +x chief: xGrid = yhat, which is the diagonal's p
%   axis; yGrid = zhat, its s axis.  PolIn is set to the probe axis so it
%   passes the state untouched instead of projecting it.
    R = struct();
    P = {'v2_test', rx_t, iT, 'T'; 'v2_ref', rx_r, iR, 'R'};
    for k = 1:2
        for pol = {'s','p'}
            macos.load_rx(P{k,2});
            if strcmp(pol{1},'s'), ax = [0 0 1];  Ex = [0 0];  Ey = [1 0];
            else,                  ax = [0 1 0];  Ex = [1 0];  Ey = [0 0]; end
            ip = P{k,3};                       % [face diagonal face]
            macos.polarizer(ip(1)-1, 'axis', ax);   % PolIn precedes the cube
            macos.polarization('on','Ex',Ex,'Ey',Ey);
            macos.vector_diffraction(true);
            macos.trace(ip(1));  f1 = macos.ray_field(ip(1));
            macos.trace(ip(2));  f2 = macos.ray_field(ip(2));
            ok = (f1.status == 0) & (f2.status == 0);
            a1 = sqrt(abs(f1.Ex).^2 + abs(f1.Ey).^2 + abs(f1.Ez).^2);
            a2 = sqrt(abs(f2.Ex).^2 + abs(f2.Ey).^2 + abs(f2.Ez).^2);
            R.([P{k,4} '' pol{1}]) = median(a2(ok)./a1(ok))^2;
        end
    end
    R = struct('Ts',R.Ts, 'Tp',R.Tp, 'Rs',R.Rs, 'Rp',R.Rp);
end

function [dphi, frames, vis] = psi_diff(Sx, Sb, Sr, thetas, msk)
%PSI_DIFF  Differential polarization PSI: the analyzer sequence on the
%   measured state and on the baseline, each against the same reference arm,
%   subtracted in the complex domain so every static term cancels.
    [px, frames, vis] = psi_frames(Sx, Sr, thetas);
    pb = psi_frames(Sb, Sr, thetas);
    dphi = angle(exp(1i*(px - pb)));
    if nargin >= 5 && ~isempty(msk), dphi(~msk) = 0; end
end

function [psi, frames, vis] = psi_frames(Sx, Sr, thetas)
    nt = numel(thetas);
    frames = cell(1, nt);
    for q = 1:nt
        frames{q} = sum(abs(synth(Sx,thetas(q)) + synth(Sr,thetas(q))).^2, 3);
    end
    if nt == 4 && max(abs(thetas(:).' - [0 45 90 135])) < 1e-12
        psi = atan2(frames{2}-frames{4}, frames{1}-frames{3});
    else
        t2 = 2*deg2rad(thetas(:));
        M = [ones(nt,1) cos(t2) sin(t2)];
        Sm = zeros(nt, numel(frames{1}));
        for q = 1:nt, Sm(q,:) = frames{q}(:).'; end
        c = M \ Sm;
        psi = reshape(atan2(c(3,:), c(2,:)), size(frames{1}));
    end
    % Visibility from the FITTED harmonic, never from min/max over the
    % samples: four samples of a cos(2t) fringe only touch its extremes when
    % the fringe phase lines up with them.
    t2 = 2*deg2rad(thetas(:));
    M2 = [ones(nt,1) cos(t2) sin(t2)];
    Sm2 = zeros(nt, numel(frames{1}));
    for q = 1:nt, Sm2(q,:) = frames{q}(:).'; end
    cc = M2 \ Sm2;
    vis = reshape(sqrt(cc(2,:).^2 + cc(3,:).^2)./max(abs(cc(1,:)),eps), size(frames{1}));
end

function [vis, h4, h6] = harmonics(I, msk)
%HARMONICS  Visibility and 4-theta / 6-theta content of a stack of analyzer
%   frames spanning [0,180) uniformly; FFT bin k holds the 2*(k-1)-theta term.
    n = size(I,3);
    F = fft(I, [], 3)/n;
    h0 = abs(F(:,:,1));  h1 = 2*abs(F(:,:,2));
    h2 = 2*abs(F(:,:,3));  h3 = 2*abs(F(:,:,4));
    vis = median(h1(msk)./h0(msk));
    h4  = mean(h2(msk))/mean(h1(msk));
    h6  = mean(h3(msk))/mean(h1(msk));
end

function D = read_deck(fn)
%READ_DECK  Element / Surface / PropType per element, straight from the
%   emitted prescription -- assertions made against what the engine will read.
    L = regexp(fileread(fn), '\n', 'split');
    D = struct('element',{{}}, 'surface',{{}}, 'proptype',{{}});
    for k = 1:numel(L)
        t = strtrim(L{k});
        if startsWith(t,'Element='), D.element{end+1} = strtrim(extractAfter(t,'Element=')); end
        if startsWith(t,'Surface='), D.surface{end+1} = strtrim(extractAfter(t,'Surface=')); end
        if startsWith(t,'PropType='), D.proptype{end+1} = strtrim(extractAfter(t,'PropType=')); end
    end
end

function [dxp, cen, rad] = det_geom(A)
%DET_GEOM  Detector sampling and the beam's FLUX-WEIGHTED centroid and
%   second-moment radius, in PIXELS.  Weighted, not thresholded: a mask
%   centroid is quantized to whole pixels and would report a reassuring exact
%   zero whatever the real misregistration was.
    macos.load_rx(A.rx);
    I = macos.intensity(A.iDET);
    dxp = macos.dx_at(A.iDET, 'mm');
    N = size(I,1);  [cg, rg] = meshgrid(1:N, 1:N);
    w = I/sum(I(:));
    cen = [sum(cg(:).*w(:)); sum(rg(:).*w(:))];
    rad = sqrt(sum(((cg(:)-cen(1)).^2 + (rg(:)-cen(2)).^2).*w(:)));
end

function [best, map] = register_to_dm(A, ix, Mdm, N_G, DX_G, h, msk)
%REGISTER_TO_DM  Build the instrument's pupil mapping from the trace itself
%   and use it to bring the injected DM map onto detector pixels.
    macos.load_rx(A.rx);
    s1 = macos.trace(ix.iTO);   ito  = macos.get_ray_info(s1.nRays);
    s2 = macos.trace(ix.iDET);  idet = macos.get_ray_info(s2.nRays);
    okr = ito.ok_trace(:) & ito.ok_pass(:) & idet.ok_trace(:) & idet.ok_pass(:);
    % DM-plane ray coordinates in the GRID frame the deck declares
    psi1 = macos.get_elt_psi(ix.iTO);  vpt1 = macos.get_elt_vpt(ix.iTO);
    u1 = macos.design.Bench.perp(psi1);  v1 = cross(psi1, u1);
    xy_to = [u1.'; v1.'] * (ito.pos - vpt1);
    psi2 = macos.get_elt_psi(ix.iDET);
    u2 = macos.design.Bench.perp(psi2);  v2 = cross(psi2, u2);
    xy_d = [u2.'; v2.'] * (idet.pos - idet.pos(:,1));
    xy_to = xy_to(:,okr);  xy_d = xy_d(:,okr);
    d = ito.pos(:,okr) - ito.pos(:,1);
    dch = ito.dir(:,1)/norm(ito.dir(:,1));  d = d - dch*(dch.'*d);
    r_beam = max(sqrt(sum(d.^2,1)));
    % affine part = the classical distortion report
    Aaf = [xy_d.' ones(nnz(okr),1)] \ xy_to.';
    Lm  = Aaf(1:2,:).';
    [Us,Ss,Vs] = svd(Lm);  sm = diag(Ss);  Rr = Us*Vs.';
    nl = xy_to - (Lm*xy_d + Aaf(3,:).');
    map = struct('mag', sqrt(abs(det(Lm))), 'anam_pct', 100*(sm(1)/sm(2)-1), ...
        'rot_deg', atan2d(Rr(2,1),Rr(1,1)), ...
        'nonlin_mm', sqrt(mean(sum(nl.^2,1))), 'r_beam', r_beam, 'Lm', Lm, 'Aaf', Aaf);
    Fx = scatteredInterpolant(xy_d(1,:).', xy_d(2,:).', xy_to(1,:).', 'linear','linear');
    Fy = scatteredInterpolant(xy_d(1,:).', xy_d(2,:).', xy_to(2,:).', 'linear','linear');
    % WF pixel centres -> detector mm.  The discrete row/col axis convention
    % is the classic grid-orientation trap, so resolve it empirically over all
    % eight candidates and PRINT the winner.
    N = size(h,1);  [cg, rg] = meshgrid(1:N, 1:N);
    cx = sum(cg(msk))/nnz(msk);  cy = sum(rg(msk))/nnz(msk);
    dxp = macos.dx_at(ix.iDET, 'mm');
    a1 = (cg-cx)*dxp;  a2 = (rg-cy)*dxp;
    c_d = mean(xy_d, 2);
    axs = ((1:N_G)-(N_G+1)/2)*DX_G;
    hm = h - mean(h(msk));
    cands = {a1,a2,'x=+col,y=+row'; a1,-a2,'x=+col,y=-row'; ...
             -a1,a2,'x=-col,y=+row'; -a1,-a2,'x=-col,y=-row'; ...
             a2,a1,'x=+row,y=+col'; a2,-a1,'x=+row,y=-col'; ...
             -a2,a1,'x=-row,y=+col'; -a2,-a1,'x=-row,y=-col'};
    best = struct('c',-inf, 'i',1);
    for c = 1:size(cands,1)
        [cc, ht] = reg_corr([0 0 0 0], cands{c,1}, cands{c,2}, c_d, Fx, Fy, axs, Mdm, hm, msk);
        if cc > best.c, best = struct('c',cc, 'ht',ht, 'name',cands{c,3}, 'i',c); end
    end
    A1 = cands{best.i,1};  A2 = cands{best.i,2};
    obj = @(p) -reg_corr(p, A1, A2, c_d, Fx, Fy, axs, Mdm, hm, msk);
    p = fminsearch(obj, [0 0 0 0], optimset('TolX',1e-7,'TolFun',1e-10,'Display','off'));
    [c2, ht2, Xt, Yt] = reg_corr(p, A1, A2, c_d, Fx, Fy, axs, Mdm, hm, msk);
    if c2 > best.c, best.c = c2;  best.ht = ht2; end
    best.hm = hm;  best.p = p;
    map.Xt = Xt;  map.Yt = Yt;
end

function [c, ht, Xt, Yt] = reg_corr(p, A1, A2, c_d, Fx, Fy, axs, Mdm, hm, msk)
%REG_CORR  Truth-vs-recovery correlation under a similarity adjustment
%   P = [dx dy rot log_scale] of the pixel->detector coordinates.  Those four
%   numbers ARE the instrument calibration a bench gets from a fiducial.
%   Spline resampling: bilinear costs hundreds of pm at actuator-scale
%   structure.
    s = exp(p(4));  ct = cos(p(3));  st = sin(p(3));
    X = s*(ct*A1 - st*A2) + c_d(1) + p(1);
    Y = s*(st*A1 + ct*A2) + c_d(2) + p(2);
    Xt = Fx(X,Y);  Yt = Fy(X,Y);
    ht = interpn(axs, axs, Mdm, Xt, Yt, 'spline', 0);
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
%BEAM_BOX  Bounding box of the illuminated pixels, padded -- the beam is a
%   small disc on the padded diffraction array, and an uncropped panel spends
%   90% of its area on black.
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
