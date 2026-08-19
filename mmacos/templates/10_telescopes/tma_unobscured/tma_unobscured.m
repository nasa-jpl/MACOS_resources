% tma_unobscured.m  (mmacos/templates/10_telescopes/tma_unobscured/)
% =====================================================================
%  THE VISIBLE-BAND UNOBSCURED TMA -- coronagraph / imager /
%  spectrometer front end.
% =====================================================================
%  Companion to tma_unobscured_search.m (run that FIRST -- it walks the
%  slower-M1 ladder and saves the geometry that is all-clear with every
%  mirror's AOI spread under the coronagraph polarization preference).
%  This script rebuilds the chosen design, runs the visible-band
%  correction ladder with every step visible, aligns the true focal
%  plane from a grid of field foci, places the exit pupil (emitted
%  PropType=FarField -- the PSF/Strehl hook), and emits the design
%  report.
%
%  The design story (Dave, 2026-07-06):
%   - visible telescope: 500 nm center wavelength -- the same figure is
%     4.6x more waves than at j18's 2.3 um, so the correction ladder
%     has to work harder for the same DL verdict;
%   - SLOWER M1 than the centered j18: gentler section AOI (< 15 deg
%     spread per mirror, the coronagraph polarization rule) and a
%     smaller clearing decenter;
%   - M2 stays CLOSE to the source->M1 beam: the minimal-clearance
%     eccentric section IS that rule (plus the launch-shroud metric);
%   - deliverable: an unobscured front end with an accessible exit
%     pupil for the coronagraph, and a flat, correctly-TILTED focal
%     plane for the imager / spectrometer pickoffs.
%
%  Run:  >> run('.../tma_unobscured/tma_unobscured.m')
% =====================================================================
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/design/src'));
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

gfile = fullfile(exdir,'tma_unobscured_geometry.mat');
if ~exist(gfile,'file')
    error('run tma_unobscured_search.m first (no %s).', gfile);
end
g = load(gfile);  c = g.chosen;
DIFFRACTION_LIMIT = 0.071;                   % Marechal, waves RMS

fprintf('====================================================================\n');
fprintf(' Unobscured visible TMA | f/%.1f primary, %g nm, decenter %.2f x D\n', ...
        c.f1, c.lambda*1e9, c.decenter/c.D);
fprintf('====================================================================\n');

%% -- [1] rebuild the chosen geometry ----------------------------------
t = macos.design.Telescope('family','TMA','aperture_diameter_m',c.D, ...
        'model_size',256,'wavelength_m',c.lambda,'grid_npts',41);
t.add_mirror('M1','radius_m',c.R(1),'spacing_after_m',c.TBET(1),'conic',c.K(1));
t.add_mirror('M2','radius_m',c.R(2),'spacing_after_m',c.TBET(2),'convex',true, ...
             'conic',c.K(2));
t.add_mirror('M3','radius_m',c.R(3),'spacing_after','derive','conic',c.K(3));
t.add_focal_plane('FP');
t.build();
t.set_offaxis('none', 'dist', c.decenter);   % replay the found section
e = t.spec.elt;
fprintf('\n[1] chain (eccentric-pupil sections, decenter %.3f m):\n', c.decenter);
for k = 1:numel(e)
    fprintf('    %-4s %-10s Vpt=[%8.3f %8.3f %8.3f]\n', ...
            e(k).name, e(k).kind, e(k).Vpt);
end

%% -- [2] the visible-band correction ladder ----------------------------
% Every step visible (Dave 2026-07-05).  At 500 nm the DL bar is 35 nm
% RMS -- the same ladder that cruised at 2.3 um has to earn it here.
ladder = @(F) wfe_field_diag(t, F, 'quiet',true);
lprint = @(tag,d) fprintf( ...
    '    %-24s raw %7.3f | -tilt %7.3f | -focus %7.3f | -astig %7.3f\n', ...
    tag, max(d.rms_raw), max(d.rms_tilt), max(d.rms_focus), max(d.rms_astig));
fprintf('\n[2] correction ladder (waves RMS @ %g nm, field center):\n', ...
        c.lambda*1e9);
d0 = ladder([0 0]);
lprint('[2a] as found:', d0);

% [2b] refigure radius+conic for the off-axis zone (axial field): the
% finder's conics were solved for the CENTERED parent; the eccentric
% section sees a different zone of it.
t.optimize('fields_arcmin',[],'dofs',[0 0 0 0 0 0 1 1],'max_iters',120);
t.set_offaxis('none');                       % refresh section poles
d1 = ladder([0 0]);
lprint('[2b] off-axis refigure:', d1);

% [2c] balance over the science field (tip/tilt/dy + radius + conic --
% pointing DOFs are safe here: no field bias to eat, clearance is
% re-verified in [4]).
optF = macos.design.field_grid(c.field_rad, 3, 'units','arcmin', ...
                               'origin', false);
t.optimize('fields',optF,'dofs',[1 1 0 0 1 0 1 1],'max_iters',150);
t.set_offaxis('none');
d2 = ladder([0 0]);
lprint('[2c] field balance:', d2);

% [2d] M1 stop-surface freeform, balanced over the field: at 500 nm
% the conic residual is usually still above the DL bar; the stop
% surface takes the field-constant part with the least field damage.
iM1 = find(strcmp({t.spec.elt.name},'M1'), 1);
rf = t.optimize_freeform(iM1, 'modes',5:15, ...
        'fields_arcmin',[c.field_rad/2 c.field_rad], 'max_iters',100); %#ok<NASGU>
d3 = ladder([0 0]);
lprint('[2d] + freeform M1:', d3);
w0 = max(d3.rms_raw);
fprintf('    field center: %.3f waves @ %g nm -> %s\n', w0, c.lambda*1e9, ...
        ternary(w0 < DIFFRACTION_LIMIT,'DIFFRACTION-LIMITED','residual'));

% [2e] the TRUE focal plane from a grid of field foci (2x2 prelim,
% 5x5 = final design): the off-axis section's FP is tilted wrt the
% chief just like the biased centered design's.
NGRID = 5;  SPAN = 0.25;
fa = t.align_focal_plane('grid',NGRID, 'span_arcmin',SPAN);
fprintf(['[2e] true FP from %dx%d field foci (+center): tilt %.3f deg, ' ...
         'defocus removed %+.3f mm,\n' ...
         '     field-curvature sag %+.1f to %+.1f um\n'], ...
        NGRID, NGRID, fa.tilt_deg, fa.defocus_m*1e3, ...
        min(fa.sag_m)*1e6, max(fa.sag_m)*1e6);

%% -- [3] what the field costs ------------------------------------------
fprintf(['\n[3] FOV ladder (worst on ring, waves RMS @ %g nm; blur on ' ...
         '-tilt):\n    %9s %9s %9s\n'], c.lambda*1e9, 'ring','raw','-tilt');
for rr = [0.1 0.25 0.5 1.0]
    dr = ladder(macos.design.field_ring(rr,'units','arcmin'));
    wt = max(dr.rms_tilt);
    fprintf('    %8.2f'' %9.3f %9.3f%s\n', rr, max(dr.rms_raw), wt, ...
            ternary(wt < DIFFRACTION_LIMIT,'  <- DL',''));
end

%% -- [4] clearance + packaging (re-checked AFTER the correction) -------
scan = t.realize_apertures('fields', ...
        macos.design.field_grid(c.field_rad,5,'units','arcmin'), ...
        'margin',0.05,'quiet',true);                              %#ok<NASGU>
rep = t.check_clipping('noload',true);
fprintf('\n[4] clearance: %s (minimal decenter = M2 as close to the\n', ...
        ternary(all([rep.obstructs] == 0), ...
                'UNOBSCURED -- every body clear','** OBSCURED **'));
fprintf('    source->M1 beam as the %.0f%%-of-D margin allows)\n', ...
        c.margin*100);

%% -- [5] deliverables ---------------------------------------------------
t.add_pupil(numel(t.spec.elt));              % EP emits PropType=FarField
rxfile  = fullfile(exdir,'tma_unobscured.in');
matfile = fullfile(exdir,'tma_unobscured.mat');
t.save(rxfile);  t.save_spec(matfile);
fprintf('\n[5] saved: %s\n           + %s\n', rxfile, matfile);
try
    f2 = t.view_orthoviews({'YZ','XZ'},'nrays',9);
    saveas(f2, fullfile(exdir,'tma_unobscured_layout.png'));
    fg = figure('Visible','off');
    contourf(fa.map.thx_arcmin, fa.map.thy_arcmin, fa.map.sag_m*1e6, ...
             15, 'LineColor','none');
    axis equal tight; colormap(parula); cb = colorbar;
    cb.Label.String = 'focus sag from fitted FP  [\mum]';
    xlabel('\theta_x  [arcmin]'); ylabel('\theta_y  [arcmin]');
    title(sprintf('field curvature (FP tilt %.3f\\circ)', fa.tilt_deg));
    saveas(fg, fullfile(exdir,'tma_unobscured_fpmap.png')); close(fg);
    fprintf(['    figures: tma_unobscured_layout.png + ' ...
             'tma_unobscured_fpmap.png\n']);
catch ME, fprintf('    figures skipped (%s)\n', ME.message); end

%% -- [6] the design report ----------------------------------------------
fprintf('\n[6] design report:\n');
% rings stay INSIDE the designed field: realize_apertures ([4]) sized
% the clear apertures to the +-FIELD_RAD envelope, so a wider ring
% loses all rays at the realized stops.
rpt = design_report(t, 'rings_arcmin',[0.1 0.25 c.field_rad], 'align',fa, ...
        'file',fullfile(exdir,'tma_unobscured_report.txt'));      %#ok<NASGU>
fprintf('    report: tma_unobscured_report.txt\n');

function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
