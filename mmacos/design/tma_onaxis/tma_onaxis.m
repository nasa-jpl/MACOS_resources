% tma_onaxis.m  (mmacos/design/ -- a parameterized design driver)
% =====================================================================
%  PARAMETERIZED ON-AXIS THREE-MIRROR ANASTIGMAT (Korsch / JWST form)
% =====================================================================
%  An obscured on-axis TMA: concave primary, convex secondary (convex by
%  GEOMETRY -- it sits before the M1 focus, KrElt=-|R|), and a tertiary
%  placed BEHIND the primary, with a real intermediate focus between M2
%  and M3 (the field-stop / metrology-injection plane).  Built from the
%  validated Korsch fixture (optical_design/fixtures/tma_fixture.json),
%  scaled to your aperture + f-numbers.
%
%  WHY A TMA: two mirrors (Cassegrain/RC) null spherical + coma but the
%  third mirror is what nulls ASTIGMATISM and relieves field curvature --
%  the reason wide-field space imagers (JWST, Roman) are TMAs.
%
%  THE CORRECTION RECIPE (this driver):
%    1. tma_layout  -> first-order layout (Seidel-seeded conics).
%    2. optimize over a small FIELD (radius+conic) -- a FIELD set is
%       essential: on-axis alone is degenerate (6 DOF, 1 aberration ->
%       the optimizer drives M3 to a junk figure).  The off-axis points
%       add coma/astigmatism constraints -> a well-posed, sane solution.
%    3. optimize_aspheres on M1 -- a few even-radial AsphCoef terms
%       (Surface=Aspheric) mop up the higher-order spherical the conic
%       seed leaves at a fast primary -> diffraction-limited.
%    4. add_pupil -- inserts the exit-pupil Return surface (j18-like
%       M1/M2/M3 + FP_return/ExitPupil/FP).
%
%  EXIT PUPIL (read this): the EP distance is set by the first-order RADII,
%  and the WFE optimization MOVES those radii -- so EP-distance and WFE are
%  coupled and conics cannot independently hold both.  This driver does NOT
%  constrain the EP (it reports where FEX finds it; often near-telecentric).
%  Controlling exit_pupil_dist -- and the FSM-at-pupil fold that needs it --
%  is a FREEFORM problem (fix the radii for the EP, correct the wavefront
%  with Zernike departures that don't move the radii); that is the next
%  driver.  See TELESCOPE_DESIGN_REFERENCE.md sections 6.4 + 7.
%
%  Run:  >> run('.../design/tma_onaxis/tma_onaxis.m')
% =====================================================================

addpath('~/dev/MACOS_resources/mmacos/src');
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

% ====================  USER DESIGN CHOICES  ==========================
APERTURE_M    = 1.0;    % aperture diameter D (m)
PRIMARY_FNUM  = 1.5;    % M1 f/#   (fast primary, JWST-like)
SYSTEM_FNUM   = 8.0;    % system f/#
M3_BEHIND_D   = 0.6;    % tertiary this far behind the primary, in units of D
FOV_ARCMIN    = 5.0;    % optimize/evaluate WFE over +-this field half-angle
ASPHERE_TERMS = 3;      % even-radial AsphCoef terms on M1 (h^4,h^6,h^8)
% ---------------------------------------------------------------------
LAM   = 633e-9;         % wavelength (m)
MODEL = 256;            % diffraction grid model size
DIFFRACTION_LIMIT = 0.07;   % ~lambda/14 RMS (waves)
% =====================================================================

D = APERTURE_M;
fprintf('====================================================================\n');
fprintf(' On-axis TMA | D=%.2f m | primary f/%.1f | system f/%.1f | M3 %.2f*D behind M1\n', ...
        D, PRIMARY_FNUM, SYSTEM_FNUM, M3_BEHIND_D);
fprintf('====================================================================\n');

%% -- Stage 1 -- first-order layout (Korsch fixture, scaled) -----------
[R, t, lay] = macos.design.tma_layout(D, PRIMARY_FNUM, SYSTEM_FNUM, ...
                  'm3_behind_m', M3_BEHIND_D*D);
t3 = macos.design.Telescope('family','TMA', 'aperture_diameter_m',D, ...
        'wavelength_m',LAM, 'model_size',MODEL);
t3.add_mirror('M1','radius_m',R(1),'spacing_after_m',t(1));
t3.add_mirror('M2','radius_m',R(2),'spacing_after_m',t(2));
t3.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
t3.build();
fprintf('\n[1] layout: R=[%.3f %.3f %.3f] m  spacings=[%.3f %.3f] m  EFL=%.3f m (f/%.2f)\n', ...
        R(1),R(2),R(3), t(1),t(2), lay.EFL, lay.fnum);
fprintf('    M3 at z=%.3f m (%.2f*D behind M1); intermediate focus between M2 and M3\n', ...
        t3.spec.elt(3).Vpt(3), M3_BEHIND_D);

%% -- Stage 2 -- correct: field optimize (radius+conic) + M1 aspheres --
fov  = linspace(0, FOV_ARCMIN, 3);
r_fo = t3.optimize('fields_arcmin', fov, 'dofs', [0 0 0 0 0 0 1 1], 'max_iters', 150);
wfe_fo = rms_waves(macos.opd(), LAM);
r_as = t3.optimize_aspheres([1], 'nterms', ASPHERE_TERMS);
fprintf('\n[2] correction: seed -> field-optimize(radius+conic) = %.4f wv -> +M1 aspheres = %.4f wv\n', ...
        wfe_fo, r_as.wfe_after);
fprintf('    mirrors (sane): M1 Kc=%.3f  M2 Kc=%.3f  M3 Kc=%.3f\n', ...
        t3.spec.elt(1).Kc, t3.spec.elt(2).Kc, t3.spec.elt(3).Kc);

%% -- Stage 3 -- WFE vs field -----------------------------------------
scanF = [zeros(numel(fov),1), deg2rad(fov(:)/60)];
scan  = t3.realize_apertures('fields', scanF, 'margin', 0.05, 'quiet', true);
w_fld = scan.wfe(:).';  f_fld = scan.fields(:,2).';
dl    = max([0, f_fld(w_fld <= DIFFRACTION_LIMIT)]);
fprintf('\n[3] WFE vs field:\n');
fprintf('      field(arcmin):'); fprintf(' %6.1f', f_fld); fprintf('\n');
fprintf('      WFE  (waves) :'); fprintf(' %6.4f', w_fld); fprintf('\n');
fprintf('      diffraction-limited (< %.2f lambda) out to %.1f arcmin\n', DIFFRACTION_LIMIT, dl);

%% -- Stage 4 -- save the OPTICS (M1/M2/M3/FP) + figures ---------------
%   Save + draw the OPTICAL CORE here, BEFORE add_pupil appends the (far,
%   near-telecentric) ExitPupil Return -- otherwise the layout view spans
%   tens of metres and the telescope collapses to a dot.
rxfile = fullfile(exdir,'tma_onaxis.in');  matfile = fullfile(exdir,'tma_onaxis.mat');
t3.save(rxfile);  t3.save_spec(matfile);
fprintf('\n[4] saved: %s\n           + %s\n', rxfile, matfile);
try
    f1 = figure('Position',[60 60 660 430]);
    ir = ~isnan(w_fld);  plot(f_fld(ir), w_fld(ir), '-o', 'LineWidth',1.5); grid on; hold on;
    yline(DIFFRACTION_LIMIT, '--', '\lambda/14', 'Color',[.6 0 0]);
    xlabel('field half-angle (arcmin)'); ylabel('RMS WFE (waves)');
    title(sprintf('On-axis TMA -- WFE vs field (D=%.1f m, f/%.0f)', D, SYSTEM_FNUM));
    saveas(f1, fullfile(exdir,'tma_onaxis_wfe.png'));
catch ME, fprintf('    WFE plot skipped (%s)\n', ME.message); end
try
    f2 = t3.view_layout('YZ', 'nrays', 13);   % the optical core only
    saveas(f2, fullfile(exdir,'tma_onaxis_layout.png'));
catch ME, fprintf('    layout skipped (%s)\n', ME.message); end

%% -- Stage 5 -- exit pupil (UNCONSTRAINED -- reported only) -----------
t3.add_pupil();
ep = t3.spec.pupil.ep_vpt(3);
fprintf('\n[5] exit pupil (FEX): z=%.3f m (%.2f*D from M1)  radius=%.3f m\n', ...
        ep, ep/D, t3.spec.pupil.ep_radius);
if abs(ep/D) > 5
    fprintf('    NOTE: EP near-telecentric -- no accessible pupil; a finite EP + FSM\n');
    fprintf('          fold (a REALIZABLE layout) is the freeform driver (B).\n');
end

%% -- Summary ---------------------------------------------------------
fprintf('\n--------------------------------------------------------------------\n');
fprintf(' D=%.2f m | f/%.0f | EFL %.2f m | M3 %.2f*D behind M1 | on-axis WFE %.4f wv\n', ...
        D, SYSTEM_FNUM, lay.EFL, M3_BEHIND_D, r_as.wfe_after);
fprintf(' diffraction-limited out to %.1f arcmin\n', dl);
fprintf(' EXIT PUPIL UNCONSTRAINED (z=%.2f m).  Pupil-controlled + FSM fold = freeform driver (B).\n', ep);
fprintf('====================================================================\n');

% ---- helpers --------------------------------------------------------
function w = rms_waves(W, lam)
    v = W(isfinite(W) & W ~= 0);
    if isempty(v), w = NaN; else, w = std(v) / lam; end
end
