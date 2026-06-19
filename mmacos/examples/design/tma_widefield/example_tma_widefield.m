% example_tma_widefield.m
% ===================================================================
%  MACOS DESIGN LAYER -- "DOES EACH MIRROR EARN ITS KEEP?"
%  Wide-field 2-mirror (Ritchey-Chretien) vs 3-mirror (Korsch TMA)
% ===================================================================
%  Multi-mirror telescopes exist to widen the corrected field of view.
%  This example makes that quantitative: it builds a 2-mirror RC and a
%  3-mirror TMA at the same aperture and f/#, optimises BOTH over the
%  same wide field, and compares the residual wavefront error vs field.
%
%  The physics: an RC nulls 3rd-order spherical AND coma with its two
%  conics, so it is excellent on-axis -- but it has NO degree of freedom
%  left for field astigmatism, which grows ~field^2.  A TMA's third conic
%  supplies exactly that DOF (a Korsch anastigmat nulls spherical + coma
%  + astigmatism).  So as the field opens up the 2-mirror hits a "wall"
%  and the 3-mirror sails past it.
%
%  What the builder does for you:
%    * RC  : closed-form layout + conics from (D, f/#, primary f/#, BFD).
%    * TMA : add_mirror() the three radii + spacings; the builder solves
%            the paraxial focus and SEIDEL-SEEDS the three conics (null
%            S_I/II/III), then optimize() refines them multi-field.
%    * optimize('fields_arcmin', ...) varies ONLY the conics (radii and
%      spacings stay fixed -- one shared physical system) to minimise the
%      FoV-weighted RMS WFE via MACOS's native CALIB optimizer.
%
%  NOTE the design is at D = 1 m for clarity; aberrations in waves depend
%  on f/# and field angle, not absolute size, so the table below is
%  scale-invariant -- multiply every length by your real aperture.
%
%  Run interactively:  >> run('.../example_tma_widefield.m')
%  (Batch check: matlab -batch "run('.../example_tma_widefield.m'); exit(0)")
% ===================================================================

addpath('~/dev/MACOS_resources/mmacos/src');     % +macos on path
exdir = pwd;                                       % save artifacts here
MODEL = 256;
LAM   = 633e-9;                                    % HeNe, for waves
FIELDS_ARCMIN = [2 5 10 15 20];                    % off-axis half-angles
DIFFRACTION_LIMIT = 0.1;                           % "well-corrected" (waves)

%% -- Stage 1 -- build the 2-mirror Ritchey-Chretien (f/8) ----------
fprintf('==================================================================\n');
fprintf(' Stage 1 -- 2-mirror Ritchey-Chretien, D=1 m, f/8\n');
fprintf('==================================================================\n');
rc = macos.design.Telescope('family','RC', ...
        'aperture_diameter_m',1.0, 'primary_fnum',2.0, ...
        'system_fnum',8.0, 'BFD_m',0.25, 'model_size',MODEL);
rc.build();  rc.describe();

%% -- Stage 2 -- build the 3-mirror Korsch TMA (f/8) ---------------
fprintf('\n Stage 2 -- 3-mirror Korsch TMA, D=1 m, f/8 (Seidel-seeded)\n');
fprintf('------------------------------------------------------------------\n');
tma = macos.design.Telescope('family','TMA', ...
        'aperture_diameter_m',1.0, 'model_size',MODEL);
tma.add_mirror('M1','radius_m',8.0, 'spacing_after_m',3.0);
tma.add_mirror('M2','radius_m',2.0, 'spacing_after_m',4.5);
tma.add_mirror('M3','radius_m',4.0, 'spacing_after','derive');  % focus
tma.build();  tma.describe();

%% -- Stage 3 -- optimise BOTH over the same wide field ------------
fprintf('\n Stage 3 -- multi-field conic optimization (0 - %g arcmin)\n', ...
        max(FIELDS_ARCMIN));
fprintf('------------------------------------------------------------------\n');
res_rc  = rc.optimize('fields_arcmin', FIELDS_ARCMIN, 'max_iters', 80);
res_tma = tma.optimize('fields_arcmin', FIELDS_ARCMIN, 'max_iters', 80);

%% -- Stage 4 -- compare residual WFE vs field --------------------
fprintf('\n Stage 4 -- residual RMS wavefront error vs field\n');
fprintf('------------------------------------------------------------------\n');
fld   = res_tma.fields_arcmin;
wf_rc = res_rc.wfe_after  / LAM;       % waves
wf_tm = res_tma.wfe_after / LAM;
fprintf('  field(arcmin)   RC 2-mirror(lambda)   TMA 3-mirror(lambda)   gain\n');
for i = 1:numel(fld)
    fprintf('     %5.1f          %10.4f            %10.4f         %5.1fx\n', ...
        fld(i), wf_rc(i), wf_tm(i), wf_rc(i)/wf_tm(i));
end
rc_field  = max(fld(wf_rc <= DIFFRACTION_LIMIT));  if isempty(rc_field),  rc_field  = 0; end
tma_field = max(fld(wf_tm <= DIFFRACTION_LIMIT));   if isempty(tma_field), tma_field = 0; end
fprintf('\n  well-corrected field (< %.2f lambda RMS):  RC <= %g'' ,  TMA <= %g''\n', ...
        DIFFRACTION_LIMIT, rc_field, tma_field);
fprintf('  => the 3rd mirror''s conic nulls field astigmatism the RC cannot:\n');
fprintf('     the 2-mirror walls at ~0.5-1.6 lambda; the TMA holds the field.\n');

%% -- Stage 5 -- plot WFE vs field --------------------------------
fig = figure('Visible','off','Position',[100 100 720 480]);
semilogy(fld, wf_rc, 'o-', 'LineWidth',1.6, 'DisplayName','RC (2-mirror)'); hold on;
semilogy(fld, wf_tm, 's-', 'LineWidth',1.6, 'DisplayName','Korsch TMA (3-mirror)');
yline(DIFFRACTION_LIMIT, 'k--', '0.1\lambda diffraction limit', 'HandleVisibility','off');
grid on; xlabel('field half-angle (arcmin)'); ylabel('RMS WFE (waves @ 633 nm)');
title('Wide-field correction: each mirror earns its keep'); legend('Location','northwest');
png = fullfile(exdir, 'tma_widefield_wfe.png');
print(fig, png, '-dpng', '-r150');
fprintf('\n  WFE-vs-field plot saved: %s\n', png);

%% -- Stage 6 -- add the exit pupil + export the TMA design --------
fprintf('\n Stage 6 -- add exit pupil + export the optimised TMA\n');
fprintf('------------------------------------------------------------------\n');
tma.add_pupil();                                   % deliverable exit pupil
p = tma.spec.pupil;
fprintf('  TMA exit pupil @ z = %.4f m,  reference radius = %.4f m\n', ...
        p.ep_vpt(3), p.ep_radius);
tma.save(     fullfile(exdir, 'tma_widefield.in'));
tma.save_spec(fullfile(exdir, 'tma_widefield.mat'));
rc.save(      fullfile(exdir, 'rc_widefield.in'));     % the comparison RC
rc.save_spec( fullfile(exdir, 'rc_widefield.mat'));
fprintf('  saved: tma_widefield.{in,mat}, rc_widefield.{in,mat}\n');

fprintf('\n  done.  (designs + plot on disk; MATLAB stays open)\n');
