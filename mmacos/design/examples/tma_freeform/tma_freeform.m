% tma_freeform.m  (mmacos/design/examples/ -- a parameterized design driver)
% =====================================================================
%  OFF-AXIS UNOBSCURED THREE-MIRROR FREEFORM DESIGNER
% =====================================================================
%  The companion to wf2_freeform, where freeform genuinely earns its
%  keep.  Pick the knobs below; the script seeds a coaxial Korsch TMA
%  (3 conics null 3rd-order S_I/II/III), takes an UNOBSCURED off-axis
%  section (eccentric pupil), refigures the 3 conics + rigid bodies to
%  the best a 3-mirror can do over the field, then layers freeform
%  Zernike departures on M2 + M3 -- holding every radius and conic --
%  and re-optimizes.  Verifies it is unobscured and saves the
%  deliverable (.in + .mat + figures).
%
%  WHY THE THIRD MIRROR CHANGES EVERYTHING (ref: optical_design/
%  TELESCOPE_DESIGN_REFERENCE.md §6.4, §7).  A 2-mirror has just ONE
%  surface away from the pupil, so a freeform there gives only field-
%  LINEAR aberration control (wf2_freeform: ~1x field-area gain over a
%  wide symmetric field).  An off-axis TMA has TWO surfaces away from
%  the pupil (M2, M3), and unobscuring breaks rotational symmetry, so
%  the residual is non-rotationally-symmetric -- exactly the regime
%  where freeform Zernike departures (placed per Nodal Aberration
%  Theory) cancel the field-dependent nodes the conics cannot reach.
%  Freeform breaks the 3-conic limit by ~8x here, to diffraction-
%  limited over a wide unobscured field.
%
%  Run:  >> run('.../design/examples/tma_freeform/tma_freeform.m')
% =====================================================================

addpath('~/dev/MACOS_resources/mmacos/src');
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

% ====================  USER DESIGN CHOICES  ==========================
APERTURE_M   = 1.0;        % aperture diameter D (m)
% Coaxial Korsch TMA seed (validated proof_korsch f/8 fixture, scales with D):
M1_RADIUS_M  = 8.0;  M1_SPACING_M = 3.0;   % M1 radius + M1->M2 spacing
M2_RADIUS_M  = 2.0;  M2_SPACING_M = 4.5;   % M2 radius + M2->M3 spacing
M3_RADIUS_M  = 4.0;                        % M3 radius (M3->focus derived)
FOV_ARCMIN   = 3.0;        % design HALF-field (arcmin): optimize + evaluate over +-this
FREEFORM_ELT = [2 3];      % mirrors to make freeform (M2, M3 -- both off the pupil)
FREEFORM_MODES = [5 6 7 8 11];   % ANSI Zernike DOF: defocus, astig, coma, spherical
LAM          = 633e-9;     % wavelength (m)
MODEL        = 256;        % diffraction grid model size
DIFFRACTION_LIMIT = 0.071; % ~lambda/14 (waves RMS) -- the Marechal criterion
% =====================================================================

D = APERTURE_M;
opt_fov  = [FOV_ARCMIN/2, FOV_ARCMIN];          % off-axis fields fed to the optimizer
fprintf('====================================================================\n');
fprintf(' Off-axis unobscured 3-mirror freeform | D=%.2f m | half-field +-%g''\n', D, FOV_ARCMIN);
fprintf('====================================================================\n');

%% -- Stage 1 -- coaxial Korsch TMA seed ------------------------------
t = macos.design.Telescope('family','TMA', 'aperture_diameter_m',D, ...
        'wavelength_m',LAM, 'model_size',MODEL);
t.add_mirror('M1','radius_m',M1_RADIUS_M,'spacing_after_m',M1_SPACING_M);
t.add_mirror('M2','radius_m',M2_RADIUS_M,'spacing_after_m',M2_SPACING_M);
t.add_mirror('M3','radius_m',M3_RADIUS_M,'spacing_after','derive');
t.build();  nE = numel(t.spec.elt);             % elts exist only after resolve
macos.trace(nE);
fprintf('\n[1] coaxial Korsch TMA seeded: on-axis %.4f waves (%d elts)\n', ...
        rms_waves(macos.opd(), LAM), nE);

%% -- Stage 2 -- unobscured off-axis section --------------------------
d = t.set_offaxis('all');
t.build('','init',false);  macos.trace(nE);
fprintf('[2] unobscured off-axis: decenter %.2f m (%.2f D); eccentric-pupil WFE %.1f waves\n', ...
        d, d/D, rms_waves(macos.opd(), LAM));

%% -- Stage 3 -- refigure the 3 conics to the 3-mirror LIMIT ----------
t.optimize('fields_arcmin', [], 'dofs', [0 0 0 0 0 0 1 1], 'max_iters', 80);   % axial refigure
t.set_offaxis('none');
r0 = t.optimize('fields_arcmin', opt_fov, 'dofs', [1 1 0 0 1 0 1 1], 'max_iters', 150);
t.set_offaxis('none');
w_conic = r0.wfe_after / LAM;                    % per-field (on-axis + opt_fov)
fprintf('[3] 3-conic + rigid optimize: worst-field %.4f waves (3-mirror limit)\n', max(w_conic));

%% -- Stage 4 -- freeform M2 + M3, re-optimize over the field ---------
rf = t.optimize_freeform(FREEFORM_ELT, 'modes',FREEFORM_MODES, ...
                         'fields_arcmin',opt_fov, 'max_iters',200);
w_free = rf.wfe_after / LAM;
fprintf(['[4] + freeform M%s (modes %s, CALIB OptZern, radii/conics held):\n' ...
         '    worst-field %.4f -> %.4f waves  (%.1fx) -> %s\n'], ...
        mat2str(FREEFORM_ELT), mat2str(FREEFORM_MODES), max(w_conic), max(w_free), ...
        max(w_conic)/max(w_free), ...
        ternary(max(w_free) < DIFFRACTION_LIMIT, 'DIFFRACTION-LIMITED', 'residual'));

%% -- Stage 5 -- verify UNOBSCURED ------------------------------------
rep = t.check_clipping('noload', true, 'quiet', true);
fprintf('[5] clearance: %d/%d optics clear -> %s\n', sum([rep.ok]), numel(rep), ...
        ternary(all([rep.ok]), 'UNOBSCURED', '** OBSCURED'));

%% -- Stage 6 -- WFE-across-field map (rc_unobscured model) -----------
try
    scan = t.realize_apertures('fields', ...
        macos.design.field_grid(FOV_ARCMIN, 7, 'units','arcmin'), ...
        'margin', 0.05, 'quiet', true);
    f1 = t.view_field_map(scan, 'kind','contour');
    p1 = fullfile(exdir, 'tma_freeform_wfe.png');  saveas(f1, p1);
    fprintf('[6] WFE-across-field map: %s\n', p1);
catch ME, fprintf('[6] WFE map skipped (%s)\n', ME.message); end

% before/after field bars (3-conic limit vs + freeform, per field)
try
    fov_lbl = [0, opt_fov];
    f2 = figure('Visible','off', 'Position',[80 80 720 500]);
    bar(categorical(string(fov_lbl)+''''), [w_conic(:), w_free(:)]);  grid on;
    yline(DIFFRACTION_LIMIT, '--k', 'diffraction limit (\lambda/14)');
    xlabel('field half-angle (arcmin)');  ylabel('RMS WFE (waves)');
    title(sprintf('Off-axis TMA: freeform M%s breaks the 3-conic limit', mat2str(FREEFORM_ELT)));
    legend({'3-conic limit','+ freeform'}, 'Location','northwest');
    p2 = fullfile(exdir, 'tma_freeform_fields.png');  saveas(f2, p2);
    fprintf('    per-field before/after: %s\n', p2);
catch ME, fprintf('    field bars skipped (%s)\n', ME.message); end

% orthographic layout
try
    f3 = t.view_orthoviews({'YZ','XZ'}, 'nrays', 9);
    p3 = fullfile(exdir, 'tma_freeform_layout.png');  saveas(f3, p3);
    fprintf('    orthographic layout: %s\n', p3);
catch ME, fprintf('    layout skipped (%s)\n', ME.message); end

%% -- Stage 7 -- save the deliverable ---------------------------------
t.add_pupil(numel(t.spec.elt));
rxfile  = fullfile(exdir, 'tma_freeform.in');
matfile = fullfile(exdir, 'tma_freeform.mat');
t.save(rxfile);  t.save_spec(matfile);
fprintf('[7] saved: %s\n           + %s\n', rxfile, matfile);

%% -- Summary ---------------------------------------------------------
fprintf('\n--------------------------------------------------------------------\n');
fprintf(' D=%.2f m | off-axis %.2f D (unobscured) | half-field +-%g''\n', D, d/D, FOV_ARCMIN);
fprintf(' worst-field RMS WFE: 3-conic limit %.4f -> + freeform %.4f waves  (%.1fx)\n', ...
        max(w_conic), max(w_free), max(w_conic)/max(w_free));
fprintf(' freeform on M%s holds all radii/conics; the 3rd mirror gives the field-\n', ...
        mat2str(FREEFORM_ELT));
fprintf(' dependent control a 2-mirror cannot (cf. wf2_freeform: ~1x over a wide field)\n');
fprintf('====================================================================\n');

% ---- helpers --------------------------------------------------------
function w = rms_waves(W, lam)
    v = W(isfinite(W) & W ~= 0);
    if isempty(v), w = NaN; else, w = std(v) / lam; end
end
function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
