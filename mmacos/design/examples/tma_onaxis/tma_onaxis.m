% tma_onaxis.m  (mmacos/design/examples/ -- a parameterized design driver)
% =====================================================================
%  PARAMETERIZED THREE-MIRROR ANASTIGMAT DESIGNER (Korsch / j18mono form)
% =====================================================================
%  A j18mono-form TMA -- concave primary, CONVEX secondary (convex by
%  GEOMETRY: it sits before the M1 focus, KrElt=-|R|), concave tertiary
%  BEHIND the primary -- with the three j18 features:
%    1. the Cassegrain feed forms a REAL intermediate focus BETWEEN M1 and
%       M2 (the field-stop / metrology-injection plane);
%    2. a SLIGHT off-axis field bias tips the focal plane OUT of the M2->M3
%       beam (the j18 "slightly off-axis detector"); the central obscuration
%       (M2 in the incoming beam) remains -- this is the obscured baseline,
%       whose unobscured eccentric-pupil cousin is design/examples/tma_offaxis;
%    3. the exit pupil after M3 is ASSESSED (FEX), NOT constrained.
%
%  THE BIAS TRADEOFF (why this is a SWEEP).  The bias must be large enough
%  to clear the FP, but off-axis aberration grows ~QUADRATICALLY with it, so
%  the LEAST bias that clears the FP gives the best wavefront.  This script
%  SWEEPS the bias, prints the clearance-vs-WFE table, and RECOMMENDS (and
%  builds) the smallest bias that both clears the FP and is diffraction-
%  limited.
%
%  THE LAYOUT + CORRECTION.  macos.design.tma_layout gives the first-order
%  Korsch (closed-form Cassegrain feed + M3 relay; convex-secondary aware --
%  the unfolded paraxial, not the n-flip).  add_mirror(...,'convex',true)
%  makes seidel_seed return the correct unfolded focus + a K=0 sphere seed
%  (the n-flip |radii| conic seed is unreliable for a convex reimager), and
%  optimize('engine','native') over the 3 conics nulls 3rd-order spherical +
%  coma + astigmatism -> diffraction-limited.
%
%  Run:  >> run('.../design/examples/tma_onaxis/tma_onaxis.m')
% =====================================================================

addpath('~/dev/MACOS_resources/mmacos/src');
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

% ====================  USER DESIGN CHOICES  ==========================
APERTURE_M    = 1.0;    % aperture diameter D (m)
PRIMARY_FNUM  = 1.5;    % M1 f/#  (f1 = PRIMARY_FNUM*D, R1 = 2*f1)
SYSTEM_FNUM   = 8.0;    % system f/#  (EFL = SYSTEM_FNUM*D)
SECONDARY_MAG = 8.0;    % Cassegrain feed magnification (> 1)
INT_FOCUS_D   = -0.125; % intermediate-focus z, units of D (between M1 & M2)
M3_BEHIND_D   = 0.6;    % tertiary this far behind the primary, units of D
FOV_ARCMIN    = 2.0;    % half-field optimized/evaluated about the bias
BIAS_SWEEP_ARCMIN = [1 2 3 4 6];   % off-axis biases to explore (arcmin)
% ---------------------------------------------------------------------
LAM   = 633e-9;         % wavelength (m)
MODEL = 256;            % diffraction grid model size
DIFFRACTION_LIMIT = 0.07;   % ~lambda/14 RMS (waves)
% =====================================================================

D = APERTURE_M;
fprintf('====================================================================\n');
fprintf(' Korsch TMA designer (j18 form) | D=%.2f m | primary f/%.1f | system f/%.1f\n', ...
        D, PRIMARY_FNUM, SYSTEM_FNUM);
fprintf('====================================================================\n');

%% -- [1] first-order Korsch layout (bias-independent) ----------------
[R, t, lay] = macos.design.tma_layout(D, PRIMARY_FNUM, SYSTEM_FNUM, ...
                  'secondary_mag', SECONDARY_MAG, ...
                  'int_focus_m',  INT_FOCUS_D*D, ...
                  'm3_behind_m',  M3_BEHIND_D*D);
fprintf('\n[1] layout: R=[%.3f %.3f %.3f] m  t=[%.3f %.3f] m  EFL=%.3f m (f/%.2f)\n', ...
        R(1),R(2),R(3), t(1),t(2), lay.EFL, lay.fnum);
fprintf('    intermediate focus z=%.3f m (between M1 at 0 and M2 at %.3f); M3 z=%.3f m behind M1\n', ...
        lay.int_focus_z, -t(1), lay.m3_z);

%% -- [2] SWEEP the off-axis bias: clearance vs WFE -------------------
fprintf('\n[2] bias sweep (least bias that clears the FP wins -- WFE grows ~bias^2):\n');
fprintf('      bias(arcmin)   RMS WFE(waves)   FP out of beam?\n');
nb = numel(BIAS_SWEEP_ARCMIN);
wfe = nan(1,nb);  clr = false(1,nb);  pick = 0;
for i = 1:nb
    bias = BIAS_SWEEP_ARCMIN(i);
    ti = build_tma_(R, t, D, LAM, MODEL);                 % fresh K=0 convex TMA
    ti.set_field_bias(bias);
    ti.optimize('fields_arcmin', linspace(0,FOV_ARCMIN,3), ...
                'dofs', [0 0 0 0 0 0 0 1], 'max_iters', 120);
    macos.trace(numel(ti.spec.elt));  wfe(i) = rms_waves(macos.opd(), LAM);
    rep = ti.check_clipping('noload', true, 'quiet', true);
    clr(i) = rep(end).ok;                                  % FP is the last optic
    fprintf('      %8.1f       %10.4f       %s\n', bias, wfe(i), ...
            ternary(clr(i),'YES','no  (FP in beam)'));
    if pick == 0 && clr(i) && wfe(i) < DIFFRACTION_LIMIT   % first (= least) that qualifies
        pick = i;
    end
end

%% -- [3] recommend the least bias that clears AND is diffraction-limited
if pick == 0
    [~, pick] = min(wfe + 1e3*(~clr));     % fallback: best clearing WFE
    fprintf('\n[3] none reached the diffraction limit -- using the best clearing bias.\n');
end
bias = BIAS_SWEEP_ARCMIN(pick);
fprintf('\n[3] RECOMMENDED bias = %g'' (least bias that clears the FP and is %s)\n', ...
        bias, ternary(wfe(pick)<DIFFRACTION_LIMIT,'diffraction-limited','best-effort'));

%% -- [4] build the recommended design + assess the exit pupil --------
t3 = build_tma_(R, t, D, LAM, MODEL);
t3.set_field_bias(bias);
r4 = t3.optimize('fields_arcmin', linspace(0,FOV_ARCMIN,3), ...
                 'dofs', [0 0 0 0 0 0 0 1], 'max_iters', 150);
nE = numel(t3.spec.elt);  macos.trace(nE);  wfe_f = rms_waves(macos.opd(), LAM);
fprintf('[4] design @ bias %g'': %.0f -> %.4f waves -> %s  (K=[%.4f %.4f %.4f])\n', ...
        bias, max(r4.wfe_before)/LAM, wfe_f, ...
        ternary(wfe_f<DIFFRACTION_LIMIT,'DIFFRACTION-LIMITED','residual'), ...
        t3.spec.elt(1).Kc, t3.spec.elt(2).Kc, t3.spec.elt(3).Kc);

rep = t3.check_clipping('noload', true, 'quiet', true);
fprintf('    clearance (obscured baseline -- M2 central obscuration expected):\n');
for k = 1:numel(rep)
    fprintf('      %-10s : %s\n', rep(k).name, ternary(rep(k).ok,'clear','OBSTRUCTS beam'));
end

t3.add_pupil();  ep = t3.spec.pupil.ep_vpt(3);  epr = t3.spec.pupil.ep_radius;
fprintf('    exit pupil after M3 (FEX, assessed): z=%.3f m (%.2f*D from M1), radius=%.3f m%s\n', ...
        ep, ep/D, epr, ternary(abs(ep/D)>5,'  [near-telecentric -> freeform driver]',''));

%% -- [5] save deliverable + layout figure ---------------------------
rxfile = fullfile(exdir,'tma_onaxis.in');  matfile = fullfile(exdir,'tma_onaxis.mat');
t3.save(rxfile);  t3.save_spec(matfile);
fprintf('\n[5] saved: %s\n           + %s\n', rxfile, matfile);
try
    f1 = t3.view_orthoviews({'YZ','XZ'}, 'nrays', 11);
    saveas(f1, fullfile(exdir,'tma_onaxis_layout.png'));
    fprintf('    layout: tma_onaxis_layout.png\n');
catch ME, fprintf('    layout skipped (%s)\n', ME.message); end

%% -- Summary --------------------------------------------------------
fprintf('\n--------------------------------------------------------------------\n');
fprintf(' D=%.2f m | f/%.0f | EFL %.2f m | Cass focus between M1 & M2 | bias %g'' | WFE %.4f wv\n', ...
        D, SYSTEM_FNUM, lay.EFL, bias, wfe_f);
fprintf(' obscured baseline (M2 central obscuration); FP biased out of beam; EP after M3 assessed.\n');
fprintf(' Unobscured eccentric-pupil cousin: design/examples/tma_offaxis.\n');
fprintf('====================================================================\n');

% ---- helpers --------------------------------------------------------
function t = build_tma_(R, t_sp, D, LAM, MODEL)
%BUILD_TMA_  A fresh convex-secondary Korsch TMA (K=0 seed) from the layout.
    t = macos.design.Telescope('family','TMA', 'aperture_diameter_m',D, ...
            'wavelength_m',LAM, 'model_size',MODEL);
    t.add_mirror('M1','radius_m',R(1),'spacing_after_m',t_sp(1));
    t.add_mirror('M2','radius_m',R(2),'spacing_after_m',t_sp(2),'convex',true);
    t.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
    t.add_focal_plane('FP');
    t.build();
end
function w = rms_waves(W, lam)
    v = W(isfinite(W) & W ~= 0);
    if isempty(v), w = NaN; else, w = std(v) / lam; end
end
function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
