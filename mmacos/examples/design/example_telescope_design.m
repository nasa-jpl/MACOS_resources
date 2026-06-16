% example_telescope_design.m
% ===================================================================
%  MACOS DESIGN LAYER — TELESCOPE DESIGN WALKTHROUGH
% ===================================================================
%  A template for designing a telescope from scratch with the MACOS
%  design layer, and the worked example for the manual.
%
%  The idea: you state design *intent* (which family, a handful of
%  first-order numbers); the builder derives the full prescription in
%  closed form (first-order layout + conic constants, Schroeder (m,β)
%  convention) and hands MACOS a ready-to-trace `.in`.  Everything
%  downstream — evaluate, vary, optimize — is the same analysis surface
%  an imported CodeV/Zemax prescription uses.
%
%  Reference (the optics): MACOS_resources/optical_design/
%      TELESCOPE_DESIGN_REFERENCE.md   (closed forms, families)
%      fixtures/                       (regression ground truth)
%  Plan: macos/PLAN_DESIGN_LAYER.md  §2 (this flow), §5 (the math).
%
%  To adapt this template: change the family + the four first-order
%  numbers in Stage 1.  Everything else follows.
%
%  Run:  matlab -batch "run('.../example_telescope_design.m')"   (exit 0)
% ===================================================================

addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..'));   % +macos on path
MODEL = 256;

%% -- Stage 1 -------------------------------------------------------
%  DECLARE INTENT.  A 6 m f/20 Ritchey-Chretien on an f/2 primary,
%  1 m back focal distance.  (Families: Cassegrain | RC | Gregorian |
%  Dall-Kirkham.  Lengths take _m or _mm; f/#'s are dimensionless.)
fprintf('==================================================================\n');
fprintf(' Stage 1 — design intent: 6 m f/20 Ritchey-Chretien\n');
fprintf('==================================================================\n');
t = macos.design.Telescope('family','RC', ...
        'aperture_diameter_mm', 6000, 'primary_fnum', 2.0, ...
        'system_fnum', 20.0, 'BFD_mm', 1000, 'model_size', MODEL);

%% -- Stage 2 -------------------------------------------------------
%  BUILD + INSPECT.  build() derives the §5 table, emits the .in, and
%  validates it by loading through SMACOS.  describe() shows every
%  derived value with its provenance — audit it before you trust it.
fprintf('\n Stage 2 — build + describe\n');
fprintf('------------------------------------------------------------------\n');
rx = t.build();
t.describe();
fprintf('\n  emitted + SMACOS-validated prescription: %s\n', rx);

%% -- Stage 3 -------------------------------------------------------
%  TRUST BEFORE OPTIMIZING.  Hand the emitted Rx to the analysis core
%  (the same surface from_rx gives an imported design) and confirm the
%  nominal is well-corrected.  An RC nulls spherical AND coma, so the
%  on-axis wavefront error is at the trace floor.
fprintf('\n Stage 3 — single-point evaluation (on-axis)\n');
fprintf('------------------------------------------------------------------\n');
s = macos.design.System.from_rx(rx, 'model_size', MODEL, 'init', false);
s.vary(2, 'despace', 'bounds', [-5 5], 'unit', 'mm');   % M2 despace = the DOF
wfe_nom = s.evaluate(0).merit;
fprintf('  on-axis RMS WFE (nominal) = %.3e m   (RC: spherical + coma corrected)\n', wfe_nom);

%% -- Stage 4 -------------------------------------------------------
%  THE FAMILY IDEA.  The named families share the SAME first-order
%  layout for given (D, f/#, primary f/#, BFD) and differ only in the
%  conic constants.  Build a classical Cassegrain on the identical
%  layout and compare what the builder derives.
fprintf('\n Stage 4 — same layout, different conics (RC vs classical Cassegrain)\n');
fprintf('------------------------------------------------------------------\n');
tc = macos.design.Telescope('family','Cassegrain', ...
        'aperture_diameter_mm', 6000, 'primary_fnum', 2.0, ...
        'system_fnum', 20.0, 'BFD_mm', 1000, 'model_size', MODEL);
dR = t.spec.derived; dC = tc.spec.derived;
fprintf('  shared layout : R1 = %.4f m   R2 = %.4f m   M1-M2 sep = %.4f m\n', ...
        dR.R1, dR.R2, dR.sep);
fprintf('  RC          : K1 = % .6f   K2 = % .6f   (primary slightly hyperbolic)\n', dR.K1, dR.K2);
fprintf('  Cassegrain  : K1 = % .6f   K2 = % .6f   (parabolic primary)\n', dC.K1, dC.K2);
fprintf('  => RC trades a slightly-hyperbolic primary to also null coma (aplanat),\n');
fprintf('     widening the usable field; the classical Cass is simpler but coma-limited.\n');

%% -- Stage 5 -------------------------------------------------------
%  ALIGN.  Put a 2 mm despace on the secondary and let the optimizer
%  drive it back to minimum wavefront error — the inner loop a real
%  build/alignment campaign runs.
fprintf('\n Stage 5 — align a perturbed secondary (M2 despace)\n');
fprintf('------------------------------------------------------------------\n');
wfe_off = s.evaluate(2.0).merit;                 % +2 mm M2 despace
res     = s.optimize('x0', 2.0, 'MaxIter', 40);
fprintf('  perturbed (+2.000 mm) : RMS WFE = %.3e m\n', wfe_off);
fprintf('  recovered             : RMS WFE = %.3e m   at despace = % .4f mm\n', ...
        res.merit_opt, res.x_opt);
fprintf('  (optimizer returned the secondary to within %.1f um of nominal)\n', ...
        abs(res.x_opt)*1e3);

%% -- Stage 6 -------------------------------------------------------
%  EXPORT.  save() writes a stand-alone .in (loadable in interactive
%  macos, shareable, diffable); save_spec() persists the design struct
%  so the builder state can be reloaded and re-optimized later.
fprintf('\n Stage 6 — export\n');
fprintf('------------------------------------------------------------------\n');
out_rx   = fullfile(tempdir, 'rc_6m_f20.in');
out_spec = fullfile(tempdir, 'rc_6m_f20.mat');
t.save(out_rx);
t.save_spec(out_spec);
fprintf('  prescription : %s\n', out_rx);
fprintf('  design spec  : %s\n', out_spec);

fprintf('\n  done.\n');
exit(0)
