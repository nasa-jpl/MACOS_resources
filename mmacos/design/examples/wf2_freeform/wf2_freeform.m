% wf2_freeform.m  (mmacos/design/examples/ -- a parameterized design driver)
% =====================================================================
%  WIDE-FIELD 2-MIRROR FREEFORM DESIGNER
% =====================================================================
%  Pick the knobs below; the script builds an aplanatic Ritchey-Chretien
%  and shows what a freeform Zernike departure on the secondary buys --
%  in two views, holding every radius and conic (the layout) FIXED:
%
%    A. PER-FIELD POWER.  At a single off-axis field, freeform NULLS the
%       astigmatism the 2-mirror conics leave behind -- ~25x, to
%       diffraction-limited.
%    B. WIDE-FIELD REALITY.  Over a symmetric +-FOV field, freeform
%       extends the diffraction-limited field only modestly: a 2-mirror
%       has just ONE surface away from the pupil, so its departure gives
%       field-LINEAR astigmatism control, which cannot cancel the field-
%       SQUARED astigmatism across a wide field.  That is exactly what a
%       THIRD mirror buys (-> the off-axis 3-mirror freeform driver).
%
%  Deliverable: the wide-field design (.in + .mat).  Headline figure:
%  the RMS-WFE-across-field map (modeled on rc_unobscured_wfe.png --
%  realize_apertures + view_field_map), plus a before/after field curve.
%  Method ref: optical_design/TELESCOPE_DESIGN_REFERENCE.md §7.2 (fix the
%  geometry, then correct the wavefront with departures that don't move
%  the first order) + §7.3 (NAT: a surface away from the pupil acts
%  field-dependently).
%
%  Run:  >> run('.../design/examples/wf2_freeform/wf2_freeform.m')
% =====================================================================

addpath('~/dev/MACOS_resources/mmacos/src');
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

% ====================  USER DESIGN CHOICES  ==========================
APERTURE_M   = 1.0;        % aperture diameter D (m)
PRIMARY_FNUM = 2.0;        % M1 f/#
SYSTEM_FNUM  = 8.0;        % system f/#  (mag = SYSTEM/PRIMARY)
BFD_M        = 0.25;       % back focal distance: focus this far behind M1 (m)
FOV_ARCMIN   = 7.0;        % design HALF-field (arcmin): optimize + evaluate over +-this
FREEFORM_ELT = 2;          % mirror(s) to make freeform (2 = secondary, away from pupil).
                           %   Try [1 2] for both; M1 is at the pupil (field-constant).
FREEFORM_MODES = [5 6 11]; % ANSI Zernike DOF: 5 defocus, 6 astig, 11 spherical
LAM          = 633e-9;     % wavelength (m)
MODEL        = 256;        % diffraction grid model size
DIFFRACTION_LIMIT = 0.071; % ~lambda/14 (waves RMS) -- the Marechal criterion
% =====================================================================

D = APERTURE_M;
fov_scan = linspace(0, FOV_ARCMIN, 8);          % field-scan sample half-angles
opt_fov  = [FOV_ARCMIN/2, FOV_ARCMIN];          % off-axis fields fed to the optimizer
mk = @() macos.design.Telescope('family','RC', 'aperture_diameter_m',D, ...
        'primary_fnum',PRIMARY_FNUM, 'system_fnum',SYSTEM_FNUM, 'BFD_m',BFD_M, ...
        'wavelength_m',LAM, 'model_size',MODEL);
fprintf('====================================================================\n');
fprintf(' Wide-field 2-mirror freeform | D=%.2f m | primary f/%.1f | system f/%.1f | +-%g''\n', ...
        D, PRIMARY_FNUM, SYSTEM_FNUM, FOV_ARCMIN);
fprintf('====================================================================\n');

%% -- Stage 1 -- aplanatic RC parent + nominal field WFE --------------
t = mk();  nE = numel(t.spec.elt);
t.build();  macos.trace(nE);
fprintf('\n[1] aplanatic RC: on-axis %.4f waves; nominal field scan (astig wall)\n', ...
        rms_waves(macos.opd(), LAM));
w_rc = field_scan(t, fov_scan, nE, LAM);

%% -- Stage 2 -- field-balance the conics (the 2-mirror LIMIT) --------
t.optimize('fields_arcmin', opt_fov, 'dofs', [0 0 0 0 0 0 1 1], 'max_iters', 100);
w_conic = field_scan(t, fov_scan, nE, LAM);
fprintf('[2] conic field-balance (radius+conic): worst-field %.4f waves (2-mirror astig limit)\n', ...
        max(w_conic));

%% -- Stage 3 -- freeform the secondary, re-optimize over the field ---
t.optimize_freeform(FREEFORM_ELT, 'modes',FREEFORM_MODES, ...
                    'fields_arcmin',opt_fov, 'max_iters',150);
w_free = field_scan(t, fov_scan, nE, LAM);
r_conic = dl_radius(w_conic, fov_scan, DIFFRACTION_LIMIT);
r_free  = dl_radius(w_free,  fov_scan, DIFFRACTION_LIMIT);
fprintf(['[3] + freeform on M%s (modes %s, CALIB OptZern, radii/conics held):\n' ...
         '    worst-field %.4f waves; diffraction-limited field %.1f'' -> %.1f''  (%.2fx area)\n'], ...
        mat2str(FREEFORM_ELT), mat2str(FREEFORM_MODES), max(w_free), ...
        r_conic, r_free, (r_free/max(r_conic,eps))^2);

%% -- Stage 4 -- PER-FIELD POWER: null one off-axis field (the 25x) ---
% A fresh copy, biased to the field EDGE: conic-optimize there (aplanatic,
% residual = astigmatism), then freeform that single field to the floor.
t2 = mk();
t2.set_field_bias(FOV_ARCMIN);
rc2 = t2.optimize('fields_arcmin',[], 'dofs',[0 0 0 0 0 0 1 1], 'max_iters',80);
rf2 = t2.optimize_freeform(FREEFORM_ELT, 'modes',[5 6 8 11], 'fields_arcmin',[], 'max_iters',150);
w_edge_conic = rc2.wfe_after(end)/LAM;  w_edge_free = rf2.wfe_after(end)/LAM;
fprintf(['[4] single off-axis field (%g''): conic-limit %.4f -> freeform %.4f waves' ...
         '  (%dx, %s)\n'], FOV_ARCMIN, w_edge_conic, w_edge_free, ...
        round(w_edge_conic/max(w_edge_free,eps)), ...
        ternary(w_edge_free < DIFFRACTION_LIMIT, 'DIFFRACTION-LIMITED', 'residual'));

%% -- Stage 5 -- WFE-across-field MAP (rc_unobscured model) -----------
% realize_apertures over a 2-D field grid, then view_field_map -- the same
% RMS-WFE-over-field report figure rc_unobscured produces.
try
    scan = t.realize_apertures('fields', ...
        macos.design.field_grid(FOV_ARCMIN, 7, 'units','arcmin'), ...
        'margin', 0.05, 'quiet', true);
    f1 = t.view_field_map(scan, 'kind','contour');
    p1 = fullfile(exdir, 'wf2_freeform_wfe.png');  saveas(f1, p1);
    fprintf('[5] WFE-across-field map: %s\n', p1);
catch ME, fprintf('[5] WFE map skipped (%s)\n', ME.message); end

% before/after field curve (the freeform-extends-the-field comparison)
try
    f2 = figure('Visible','off', 'Position',[80 80 780 540]);  hold on; grid on;
    plot(fov_scan, w_rc,    '-o', 'LineWidth',1.3, 'DisplayName','RC nominal');
    plot(fov_scan, w_conic, '-s', 'LineWidth',1.3, 'DisplayName','conic field-balance (2-mirror limit)');
    plot(fov_scan, w_free,  '-^', 'LineWidth',1.9, 'DisplayName',sprintf('+ freeform M%s', mat2str(FREEFORM_ELT)));
    yline(DIFFRACTION_LIMIT, '--k', 'diffraction limit (\lambda/14)', 'HandleVisibility','off');
    xlabel('field half-angle (arcmin)');  ylabel('RMS WFE (waves)');
    title(sprintf('WFE vs field: freeform on a 2-mirror (D=%g m, f/%g)', D, SYSTEM_FNUM));
    legend('Location','northwest');  set(gca,'YScale','log');  ylim([1e-3 max(w_rc)*1.5]);
    p2 = fullfile(exdir, 'wf2_freeform_fieldcurve.png');  saveas(f2, p2);
    fprintf('    before/after field curve: %s\n', p2);
catch ME, fprintf('    field curve skipped (%s)\n', ME.message); end

% orthographic layout
try
    f3 = t.view_orthoviews({'YZ','XZ'}, 'nrays', 9);
    p3 = fullfile(exdir, 'wf2_freeform_layout.png');  saveas(f3, p3);
    fprintf('    orthographic layout: %s\n', p3);
catch ME, fprintf('    layout skipped (%s)\n', ME.message); end

%% -- Stage 6 -- save the deliverable ---------------------------------
t.add_pupil(numel(t.spec.elt));
rxfile  = fullfile(exdir, 'wf2_freeform.in');
matfile = fullfile(exdir, 'wf2_freeform.mat');
t.save(rxfile);  t.save_spec(matfile);
fprintf('[6] saved: %s\n           + %s\n', rxfile, matfile);

%% -- Summary ---------------------------------------------------------
fprintf('\n--------------------------------------------------------------------\n');
fprintf(' D=%.2f m | primary f/%.1f | system f/%.1f | BFD=%.0f mm | half-field +-%g''\n', ...
        D, PRIMARY_FNUM, SYSTEM_FNUM, BFD_M*1000, FOV_ARCMIN);
fprintf(' PER-FIELD : a single off-axis field nulls %.4f -> %.4f waves (%dx, diffraction-limited)\n', ...
        w_edge_conic, w_edge_free, round(w_edge_conic/max(w_edge_free,eps)));
fprintf(' WIDE-FIELD: diffraction-limited half-field %.1f'' -> %.1f'' (%.2fx area), worst %.3f->%.3f waves\n', ...
        r_conic, r_free, (r_free/max(r_conic,eps))^2, max(w_conic), max(w_free));
fprintf(' freeform on M%s holds all radii/conics (layout unchanged); full wide-field needs a 3rd mirror\n', ...
        mat2str(FREEFORM_ELT));
fprintf('====================================================================\n');

% ---- helpers --------------------------------------------------------
function w = field_scan(t, fovs, nE, lam)
%FIELD_SCAN  RMS WFE (waves) at each +y field half-angle, via the EMITTED
%   biased chief ray (set_field_bias -> build) -- the working off-axis
%   mechanism (set_src_fov does NOT move the field for a design trace).
    w = zeros(size(fovs));
    for i = 1:numel(fovs)
        t.set_field_bias(fovs(i));  t.build('', 'init', false);
        macos.trace(nE);  w(i) = rms_waves(macos.opd(), lam);
    end
    t.set_field_bias(0);  t.build('', 'init', false);   % restore on-axis
end
function r = dl_radius(w, fov, dl)
%DL_RADIUS  Outer edge of the in-spec band, contiguous from the best
%   field (after field-balancing the curve dips mid-field, so measure
%   the width of the diffraction-limited band).
    in = w < dl;
    if ~any(in), r = 0; return; end
    [~, c] = min(w);
    hi = c;  while hi < numel(in) && in(hi+1), hi = hi+1; end
    r = fov(hi);
end
function w = rms_waves(W, lam)
    v = W(isfinite(W) & W ~= 0);
    if isempty(v), w = NaN; else, w = std(v) / lam; end
end
function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
