% rc_unobscured.m  (mmacos/design/ -- a parameterized design driver)
% =====================================================================
%  PARAMETERIZED UNOBSCURED OFF-AXIS RITCHEY-CHRETIEN DESIGNER
% =====================================================================
%  Pick the four design knobs below; the script builds the on-axis RC
%  parent, takes an UNOBSCURED off-axis section (eccentric pupil),
%  refigures + field-optimizes it to diffraction-limited, verifies it is
%  unobscured, and saves the deliverable (.in + .mat + figures).
%
%  THE OFF-AXIS TRADE (why these knobs).  An on-axis Cassegrain/RC has a
%  central obscuration: the secondary M2 shadows the incoming beam.  To
%  make it unobscured you use an eccentric (decentered) sub-aperture of
%  the same parent so the used cone misses M2's shadow.  The decenter
%  needed is set by the SECONDARY's size:
%        off-axis decenter ~ D/2 + (secondary radius)      [floor ~ D/2]
%  so you go LESS off-axis by SHRINKING the secondary, i.e. a FASTER
%  primary and/or SLOWER system f/# (higher magnification).  Measured
%  (D=1 m, BFD=0.3 m):
%        primary f/2  + system f/10  ->  0.89 D
%        primary f/2  + system f/20  ->  0.67 D
%        primary f/1.5 + system f/20 ->  0.64 D   (near the D/2 floor)
%        primary f/3  (slow)          ->  secondary too big, cannot clear
%  The focus sits BEHIND M1 by BFD (Cassegrain back focus), so the focal
%  plane / instrument is accessible behind the primary.
%
%  Because an RC is APLANATIC, the eccentric sub-aperture is spherical-
%  AND coma-free at the axial field by construction; the radius+conic
%  refigure cleans up the residual and the rigid-body DOFs then balance
%  the linear off-axis astigmatism over the design field.
%
%  Run:  >> run('.../design/rc_unobscured/rc_unobscured.m')
% =====================================================================

addpath('~/dev/MACOS_resources/mmacos/src');
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

% ====================  USER DESIGN CHOICES  ==========================
APERTURE_M   = 1.0;     % aperture diameter D (m)
PRIMARY_FNUM = 1.5;     % M1 f/#    (FASTER -> smaller secondary -> less off-axis)
SYSTEM_FNUM  = 20.0;    % system f/# (SLOWER -> smaller secondary -> less off-axis)
OFF_AXIS_M   = [];      % off-axis decenter (m).  [] = AUTO (minimum to clear);
                        % a number sets it explicitly (>= the minimum).
% ---- secondary choices ----------------------------------------------
BFD_M        = 0.30;    % back focal distance: focus this far BEHIND M1 (m)
FOV_ARCMIN   = 0.5;     % design HALF-field (arcmin): optimize + evaluate over +-this
LAM          = 633e-9;  % wavelength (m)
MODEL        = 256;     % diffraction grid model size
DIFFRACTION_LIMIT = 0.07;   % ~lambda/14 (waves)
% =====================================================================

D = APERTURE_M;
fprintf('====================================================================\n');
fprintf(' Unobscured off-axis RC | D=%.2f m | primary f/%.1f | system f/%.1f | BFD=%.0f mm\n', ...
        D, PRIMARY_FNUM, SYSTEM_FNUM, BFD_M*1000);
fprintf('====================================================================\n');

%% -- Stage 1 -- on-axis RC parent ------------------------------------
t = macos.design.Telescope('family','RC', 'aperture_diameter_m',D, ...
        'primary_fnum',PRIMARY_FNUM, 'system_fnum',SYSTEM_FNUM, 'BFD_m',BFD_M, ...
        'wavelength_m',LAM, 'model_size',MODEL);
t.set_field_points(macos.design.field_grid(FOV_ARCMIN, 3, 'units','arcmin'));
nE = numel(t.spec.elt);
t.build();
macos.trace(nE);  wfe0 = rms_waves(macos.opd(), LAM);
fprintf('\n[1] on-axis RC built: RMS WFE = %.4f waves (aplanatic)\n', wfe0);

%% -- Stage 2 -- unobscured off-axis section --------------------------
dmin = t.set_offaxis('all');                 % minimum decenter to clear (applies it)
if isempty(OFF_AXIS_M)
    d = dmin;  why = 'auto: minimum to clear the secondary shadow';
elseif OFF_AXIS_M + 1e-9 >= dmin
    d = t.set_offaxis('all', 'dist', OFF_AXIS_M);
    why = sprintf('your choice (minimum to clear = %.2f D)', dmin/D);
else
    d = dmin;  % keep the minimum (already applied)
    why = sprintf('** %.2f D requested < %.2f D minimum -> bumped to minimum', ...
                  OFF_AXIS_M/D, dmin/D);
end
macos.trace(nE);  wfe_oa = rms_waves(macos.opd(), LAM);
fprintf('[2] off-axis decenter = %.3f m (%.2f D)  [floor ~0.50 D]\n    %s\n', d, d/D, why);
fprintf('    eccentric-pupil WFE (un-refigured) = %.3f waves\n', wfe_oa);

%% -- Stage 3 -- axial refigure (radius + conic) ----------------------
t.optimize('fields_arcmin', [], 'dofs', [0 0 0 0 0 0 1 1], 'max_iters', 80);
t.set_offaxis('none');
macos.trace(nE);  wfe_ax = rms_waves(macos.opd(), LAM);
fprintf('[3] axial optimize (radius+conic): RMS WFE = %.4f waves -> %s\n', wfe_ax, ...
        ternary(wfe_ax < DIFFRACTION_LIMIT, 'DIFFRACTION-LIMITED', 'residual'));

%% -- Stage 4 -- balance over the design field ------------------------
optF = macos.design.field_grid(FOV_ARCMIN, 3, 'units','arcmin', 'origin',false);
resf = t.optimize('fields', optF, 'dofs', [1 1 0 0 1 0 1 1], 'max_iters', 150);
t.set_offaxis('none');
wfe_fb = max(resf.wfe_before)/LAM;  wfe_fa = max(resf.wfe_after)/LAM;
fprintf(['[4] field optimize (+-%g'' area, tip/tilt/dy+radius+conic):\n' ...
         '    worst-field WFE  %.3f -> %.4f waves -> %s\n'], FOV_ARCMIN, wfe_fb, wfe_fa, ...
        ternary(wfe_fa < DIFFRACTION_LIMIT, 'DIFFRACTION-LIMITED', 'residual'));

%% -- Stage 5 -- field scan: WFE map + clear apertures ----------------
fprintf('\n[5] field scan over the +-%g'' field -> WFE map + apertures\n', FOV_ARCMIN);
scan = t.realize_apertures('fields', macos.design.field_grid(FOV_ARCMIN, 7, 'units','arcmin'), ...
                           'margin', 0.05, 'quiet', true);

%% -- Stage 6 -- verify UNOBSCURED ------------------------------------
rep = t.check_clipping('noload', true, 'quiet', true);
fprintf('[6] clearance: %d/%d optics clear -> %s\n', sum([rep.ok]), numel(rep), ...
        ternary(all([rep.ok]), 'UNOBSCURED', '** OBSCURED'));
assert(all([rep.ok]), 'design is obscured -- increase OFF_AXIS_M');

%% -- Stage 7 -- exit pupil + save deliverable ------------------------
t.add_pupil(numel(t.spec.elt));
rxfile  = fullfile(exdir, 'rc_unobscured.in');
matfile = fullfile(exdir, 'rc_unobscured.mat');
t.save(rxfile);  t.save_spec(matfile);
fprintf('[7] saved: %s\n           + %s\n', rxfile, matfile);

%% -- Stage 8 -- design-report figures --------------------------------
try
    f1 = t.view_field_map(scan, 'kind','contour');
    p1 = fullfile(exdir, 'rc_unobscured_wfe.png');  saveas(f1, p1);
    fprintf('[8] WFE field map: %s\n', p1);
catch ME, fprintf('[8] WFE map skipped (%s)\n', ME.message); end
try
    f2 = t.view_orthoviews({'YZ','XZ'}, 'nrays', 9);
    p2 = fullfile(exdir, 'rc_unobscured_layout.png');  saveas(f2, p2);
    fprintf('    orthographic layout: %s\n', p2);
catch ME, fprintf('    layout skipped (%s)\n', ME.message); end

%% -- Summary ---------------------------------------------------------
fprintf('\n--------------------------------------------------------------------\n');
fprintf(' D=%.2f m | primary f/%.1f | system f/%.1f (mag %.1f) | BFD=%.0f mm\n', ...
        D, PRIMARY_FNUM, SYSTEM_FNUM, SYSTEM_FNUM/PRIMARY_FNUM, BFD_M*1000);
fprintf(' off-axis decenter = %.3f m (%.2f D)   [geometric floor ~0.50 D]\n', d, d/D);
fprintf(' RMS WFE: on-axis %.4f -> eccentric %.2f -> axial %.4f -> field %.4f waves\n', ...
        wfe0, wfe_oa, wfe_ax, wfe_fa);
fprintf(' UNOBSCURED, focus %.0f mm behind M1, diffraction-limited over +-%g arcmin\n', ...
        BFD_M*1000, FOV_ARCMIN);
fprintf('====================================================================\n');

% ---- helpers --------------------------------------------------------
function w = rms_waves(W, lam)
    v = W(isfinite(W) & W ~= 0);
    if isempty(v), w = NaN; else, w = std(v) / lam; end
end
function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
