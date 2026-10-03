% example_rc_offaxis.m
% ===================================================================
%  MACOS DESIGN LAYER -- UNOBSCURED OFF-AXIS RITCHEY-CHRETIEN
%  Build an on-axis 2-mirror RC, take it OFF-AXIS as an eccentric-pupil
%  section, clear the beam, and re-optimize to diffraction-limited.
% ===================================================================
%  An on-axis Cassegrain/RC has the secondary sitting in the middle of
%  the incoming beam (the central obscuration) and the focus behind a
%  hole in the primary.  To make it UNOBSCURED you use an OFF-AXIS
%  SECTION of the same parent: decenter the used sub-aperture far enough
%  that (a) the secondary no longer shadows the off-axis incoming beam
%  and (b) the M2->focus return beam clears the primary's edge -- no hole,
%  no obscuration anywhere.
%
%  THE OFF-AXIS-SECTION BUILDING BLOCK (engine-true).  MACOS's conic
%  surface measures its sag from VptElt (the parent VERTEX) along psiElt
%  (the parent AXIS); RptElt is the section POLE -- the point on the
%  parent surface at the center of the used sub-aperture, which carries
%  the local TElt frame and the perturbation center.  An off-axis section
%  is therefore just RptElt != VptElt on the SAME parent figure (exactly
%  how JWST's segmented primary is written -- j18sc.in).  set_offaxis()
%  emits that for every mirror automatically.
%
%  THE RECIPE (e5mono / dmt6mono "design on-axis, then move off-axis"):
%    1. build the on-axis RC                      -> diffraction-limited
%    2. set_offaxis('all')                         -> eccentric pupil, every
%       (decenter until every mirror body clears)     mirror clears the beam
%    3. optimize(radius+conic, on-axis field)      -> refigure the two
%                                                      mirrors for the off-axis
%                                                      zone -> diffraction-limited
%    4. optimize(tip/tilt/dy+radius+conic, AREA)   -> balance the linear
%                                                      off-axis astigmatism over
%                                                      the 2-D field
%  Because an RC is APLANATIC, two mirrors have just enough freedom
%  (each radius + conic) to recover the eccentric-pupil wavefront at the
%  axial field; the rigid-body DOFs then balance the FIELD-dependent
%  (linear off-axis) astigmatism the rotationally-symmetric conics cannot.
%
%  ONE FIELD INPUT, AREA SCORING.  The design field is given ONCE
%  (FOV_ARCMIN, the half-field).  From it the script builds 2-D field-angle
%  GRIDS (macos.design.field_grid): a coarse 3x3 AREA grid for the field
%  optimize (CALIB caps at 12 FoV, so 3x3 = 9 is the practical area), and a
%  fine 7x7 grid for the WFE field MAP + the clear-aperture scan.  Optimized
%  field == evaluated field by construction.  The off-axis pupil is
%  eccentric in y, so WFE(thx) != WFE(thy) -- the report shows the full 2-D
%  map and the x/y cross-sections.  (For a lighter run, swap field_grid for
%  macos.design.field_cross -- the cross mode.)
%
%  Run interactively:  >> run('.../example_rc_offaxis.m')
%  (Batch check: matlab -batch "run('.../example_rc_offaxis.m'); exit(0)")
% ===================================================================

addpath('~/dev/MACOS_resources/mmacos/src');     % +macos on path
exdir = fileparts(mfilename('fullpath'));
if isempty(exdir), exdir = pwd; end
% ---- user inputs (edit for your telescope) -----------------------
MODEL = 256;
LAM   = 633e-9;                                    % HeNe, for waves
D     = 1.0;                                       % aperture (m)
FOV_ARCMIN = 0.5;                                  % design HALF-field (arcmin):
                                                   % the SINGLE field input --
                                                   % optimize + evaluate over a
                                                   % +-FOV 2-D AREA.
NOPT  = 3;                                          % AREA-optimize grid (3x3 = 9 FoV)
NMAP  = 7;                                          % WFE field-map / scan grid
DIFFRACTION_LIMIT = 0.07;                           % ~lambda/14 (waves)

% Field sets derived from the SINGLE FoV input (2-D grids, arcmin -> rad):
optF = macos.design.field_grid(FOV_ARCMIN, NOPT, 'units','arcmin', 'origin',false);
mapF = macos.design.field_grid(FOV_ARCMIN, NMAP, 'units','arcmin');

fprintf('==================================================================\n');
fprintf(' Unobscured off-axis Ritchey-Chretien  (D=%.1f m, f/10)\n', D);
fprintf('==================================================================\n');

%% -- Stage 1 -- on-axis RC (the parent) ---------------------------
t = macos.design.Telescope('family','RC', 'aperture_diameter_mm', D*1000, ...
        'primary_fnum', 2.0, 'system_fnum', 10.0, 'BFD_mm', 300, ...
        'wavelength_m', LAM, 'model_size', MODEL);
t.set_field_points(macos.design.field_grid(FOV_ARCMIN, NOPT, 'units','arcmin'));
nE = numel(t.spec.elt);
t.build();
macos.trace(nE);  wfe_onaxis = rms_waves(macos.opd(), LAM);
fprintf('\n[1] on-axis RC built: RMS WFE = %.4f waves (aplanatic, on-axis)\n', ...
        wfe_onaxis);

%% -- Stage 2 -- take it off-axis as an eccentric-pupil section -----
% Decenter the beam until EVERY mirror clears (M2 out of the incoming
% beam AND M1 out of the return beam).  For a JWST-like TMA you would
% instead name only the optic to extract, e.g. set_offaxis('M3').
d = t.set_offaxis('all');
macos.trace(nE);  wfe_offaxis = rms_waves(macos.opd(), LAM);
fprintf(['[2] set_offaxis(''all''): beam decentered %.3f m (%.2f * D)\n' ...
         '    eccentric-pupil WFE (un-refigured) = %.3f waves\n'], ...
        d, d/D, wfe_offaxis);

%% -- Stage 3 -- refigure the two mirrors for the off-axis zone -----
t.optimize('fields_arcmin', [], 'dofs', [0 0 0 0 0 0 1 1], 'max_iters', 80);
t.set_offaxis('none');                  % refresh section poles at the new conics
macos.trace(numel(t.spec.elt));  wfe_opt = rms_waves(macos.opd(), LAM);
fprintf(['[3] optimize(radius+conic, axial field): RMS WFE = %.4f waves' ...
         '  -> %s\n'], wfe_opt, ...
         ternary(wfe_opt < DIFFRACTION_LIMIT, 'DIFFRACTION-LIMITED', 'residual'));

%% -- Stage 4 -- balance over the +-FOV AREA (3x3 grid) -------------
% The axial refigure is diffraction-limited ONLY on-axis -- an eccentric-
% pupil 2-mirror has strong LINEAR off-axis astigmatism, so the WFE grows
% fast with field.  Re-balance over the 2-D AREA (corners + edge midpoints)
% using the rigid-body tip/tilt/decenter DOFs (the field-dependent
% correction the rotationally-symmetric conics cannot supply) + radius+conic.
resf = t.optimize('fields', optF, 'dofs', [1 1 0 0 1 0 1 1], 'max_iters', 150);
t.set_offaxis('none');                  % refresh section poles after the moves
wfe_field_before = max(resf.wfe_before)/LAM;       % axial design, over the area
wfe_field_after  = max(resf.wfe_after)/LAM;        % field-balanced
fprintf(['[4] optimize over +-%g'' AREA (3x3, tip/tilt/dy+radius+conic):\n' ...
         '    worst-field WFE  %.3f -> %.3f waves  -> %s\n'], ...
        FOV_ARCMIN, wfe_field_before, wfe_field_after, ...
        ternary(wfe_field_after < DIFFRACTION_LIMIT, 'DIFFRACTION-LIMITED', 'residual'));

%% -- Stage 5 -- field scan: WFE field map + real clear apertures ---
% Scan the SAME design field on a fine 2-D grid, record WFE over the field,
% and size each optic's clear aperture to the full-field beam footprint
% (Circular on M1/M2, Square on the focal plane).  These ApVecs are emitted
% into the Rx and used to draw each optic at its real size.
fprintf('\n[5] field scan over the +-%g'' area (%dx%d) -> WFE map + apertures:\n', ...
        FOV_ARCMIN, NMAP, NMAP);
scan = t.realize_apertures('fields', mapF, 'margin', 0.05);

%% -- Stage 6 -- prove there is no clipping (with real apertures) ---
fprintf('\n[6] clearance check (no body sits in any beam):\n');
t.build('check', true);
rep = t.check_clipping('noload', true);
assert(all([rep.ok]), 'unexpected clipping in the final design');

%% -- Stage 7 -- exit pupil + freeze the deliverable ---------------
t.add_pupil(numel(t.spec.elt));         % accessible exit pupil before the FP
rxfile  = fullfile(exdir, 'rc_offaxis.in');
matfile = fullfile(exdir, 'rc_offaxis.mat');
t.save(rxfile);
t.save_spec(matfile);
fprintf('\n[7] saved deliverable: %s\n               + spec: %s\n', rxfile, matfile);

%% -- Stage 8 -- design-report figures -----------------------------
% (a) WFE field MAP over the 2-D field; (b) x/y cross-sections through the
% map; (c) orthographic layout (optics at their measured clear apertures).
try
    f1 = t.view_field_map(scan, 'kind','contour');     % the 2-D WFE map
    p1 = fullfile(exdir, 'rc_offaxis_wfe_map.png');  saveas(f1, p1);
    fprintf('[8] WFE field map: %s\n', p1);
catch ME
    fprintf('[8] WFE map skipped (%s)\n', ME.message);
end
try
    fa = scan.fields;  wv = scan.wfe(:);  tol = 1e-9;
    rx = abs(fa(:,2)) < tol;   [fxv, ix] = sort(fa(rx,1));  wx = wv(rx);  wx = wx(ix);
    ry = abs(fa(:,1)) < tol;   [fyv, iy] = sort(fa(ry,2));  wy = wv(ry);  wy = wy(iy);
    f2 = figure('Position',[60 60 660 430]);
    plot(fxv, wx, '-o', 'LineWidth',1.5);  hold on;
    plot(fyv, wy, '-s', 'LineWidth',1.5);  grid on;
    yline(DIFFRACTION_LIMIT, '--', '\lambda/14', 'Color',[.6 0 0]);
    xlabel('field angle (arcmin)');  ylabel('RMS WFE (waves)');
    legend('WFE vs \theta_x','WFE vs \theta_y', 'Location','north');
    title(sprintf('Off-axis RC -- field cross-sections (balanced over +-%g'' area)', ...
                  FOV_ARCMIN));
    p2 = fullfile(exdir, 'rc_offaxis_wfe_xsec.png');  saveas(f2, p2);
    fprintf('    WFE x/y cross-sections: %s\n', p2);
catch ME
    fprintf('    WFE cross-sections skipped (%s)\n', ME.message);
end
try
    f3 = t.view_orthoviews({'YZ','XZ'}, 'nrays', 9);   % design-report layout
    p3 = fullfile(exdir, 'rc_offaxis_layout.png');  saveas(f3, p3);
    fprintf('    orthographic layout (YZ+XZ): %s\n', p3);
catch ME
    fprintf('    view_orthoviews skipped (%s)\n', ME.message);
end

%% -- Summary ------------------------------------------------------
fprintf('\n------------------------------------------------------------------\n');
fprintf(' on-axis RC               : %.4f waves\n', wfe_onaxis);
fprintf(' off-axis (raw)           : %.4f waves   @ decenter %.2f*D\n', wfe_offaxis, d/D);
fprintf(' off-axis (axial refigure): %.4f waves\n', wfe_opt);
fprintf(' over +-%g'' area  : %.4f -> %.4f waves  (axial-opt -> area-opt)\n', ...
        FOV_ARCMIN, wfe_field_before, wfe_field_after);
fprintf('------------------------------------------------------------------\n');
fprintf(' Eccentric-pupil refigure (radius+conic) gives a diffraction-\n');
fprintf(' limited axial field; rigid tip/tilt/decenter then BALANCES the\n');
fprintf(' linear off-axis astigmatism over the 2-D field.  The off-axis\n');
fprintf(' pupil is eccentric in y, so WFE(thx) != WFE(thy) -- both shown.\n');
fprintf(' The layout stays unobscured; apertures sized to the full field.\n');
fprintf('==================================================================\n');

% ---- local helpers ------------------------------------------------
function w = rms_waves(W, lam)
    v = W(isfinite(W) & W ~= 0);
    if isempty(v), w = NaN; else, w = std(v) / lam; end
end
function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
