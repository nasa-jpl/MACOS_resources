function OUT = s1_telescope(over)
%S1_TELESCOPE  e2e6m stage 1: the 6 m unobscured visible telescope.
%
%   A thin instance runner over the `freeform_unobscured` template's
%   design flow (sphere+Zernike tilted folds) at the e2e6m parameter set:
%   D = 6 m, 500 nm, an arcminute-class field, and the campaign's three
%   hard gates -- OTA f/# in [12, 20], an 8 m DEPLOYED shroud diameter,
%   and an UNOBSCURED train.
%
%     [1] all-sphere 0th-order layout (picked by s1_layout_search)
%     [2] staged Zernike (freeform) correction, centre -> inner -> full
%     [3] the true focal plane from a grid of field foci
%     [4] dense WFE-vs-field map + clearance + AOI + packaging
%     [5] exit pupil, save, and the RELOAD RAY-COUNT GATE
%     [6] figures: layout, field map, focal-plane curvature, shroud fit
%     [7] the design report
%
%   METRIC TAG, stated once and quoted with every number in this stage:
%   RMS wavefront error at 500 nm, referenced to the exit pupil that
%   `add_pupil` emits (PropType=FarField), measured on the stage's own
%   fitted focal plane; "raw" keeps tilt, "-tilt" removes piston and
%   tip/tilt.  This is the TELESCOPE-ONLY configuration, anchored at the
%   TELESCOPE's best-focus exit pupil.  Once the S3 back end is attached,
%   ultimate performance moves to the CORONAGRAPH's exit pupil and these
%   numbers stay as this configuration's record.
%
%   APERTURES ARE OFF (rule 1 of the four aperture rules; see
%   e2e6m_params).  The clearance verdict comes from `check_clipping`,
%   which measures beam footprints directly and needs no emitted stops.
%
%   OUT = S1_TELESCOPE()      run at the default parameter set
%   OUT = S1_TELESCOPE(OVER)  ... with e2e6m_params overrides
%
%   See also E2E6M_PARAMS, S1_LAYOUT_SEARCH, FREEFORM_UNOBSCURED,
%   E2E6M_SHROUD_FIG.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    setup_(here);
    P = e2e6m_params(over);
    if isempty(P.outdir), P.outdir = here; end
    tag = fullfile(P.outdir, 's1');

    L = {};
    t0 = tic;

    L = say_(L, '==================== e2e6m S1 -- the 6 m unobscured telescope');
    L = say_(L, 'D = %.2f m | lambda = %g nm | model %d | grid %d', ...
             P.D_m, P.lambda_m*1e9, P.model, P.gridn);
    L = say_(L, 'metric: RMS WFE @ %g nm, exit-pupil referenced (add_pupil FarField),', ...
             P.lambda_m*1e9);
    L = say_(L, '        on the stage''s fitted focal plane; raw keeps tilt, -tilt removes');
    L = say_(L, '        piston+tip/tilt.  TELESCOPE-ONLY config, anchored at the');
    L = say_(L, '        TELESCOPE best-focus exit pupil.');
    L = say_(L, 'gates : f/# in [%g %g] | shroud <= %.1f m (deployed, diameter-only)', ...
             P.fno_band, P.shroud_D_m);
    L = say_(L, '        | UNOBSCURED | DL <= %.3f waves | AOI spread < %g deg (preference)', ...
             P.dl_waves, P.aoi_limit);
    L = say_(L, 'apertures OFF (rule 1); clearance from check_clipping footprints');

    % ================= [1] the all-sphere 0th-order layout ================
    t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_m',P.D_m, 'model_size',P.model, ...
            'wavelength_m',P.lambda_m, 'grid_npts',P.gridn);
    t.set_base_sphere(true);
    nm = {'M1','M2','M3'};
    for k = 1:3
        args = {'radius_m',P.tel.R_m(k), 'tilt_deg',P.tel.tilt_deg(k)};
        if k < 3, args = [args {'spacing_after_m',P.tel.T_m(k)}]; %#ok<AGROW>
        else,     args = [args {'spacing_after','derive'}];       %#ok<AGROW>
        end
        if P.tel.convex(k), args = [args {'convex',true}]; end %#ok<AGROW>
        t.add_mirror(nm{k}, args{:});
    end
    t.add_focal_plane('FP');
    t.build();
    nE = numel(t.spec.elt);
    macos.trace(nE);
    wfe0 = rms_waves_(macos.opd(), P.lambda_m);
    L = say_(L, '\n[1] all-sphere layout  R = [%.3f %.3f %.3f] m, T = [%.3f %.3f] m,', ...
             P.tel.R_m, P.tel.T_m);
    L = say_(L, '    tilt = [%.2f %.2f %.2f] deg -> uncorrected %.0f waves', ...
             P.tel.tilt_deg, wfe0);

    % ================= [2] staged freeform correction =====================
    % Bauer-style intermediate objectives: diffraction-limit the centre,
    % hold the inner field, then the full field.  Fields are 2-D
    % area-weighted about the y-z fold plane (sample thx >= 0, weight
    % thx > 0 twice) -- the design is mirror-symmetric in x.
    amin = @(A) deg2rad(A/60);
    h  = P.tel.fov_arcmin;
    F1 = (h/2)*[1 0; 0 1; 0 -1; 1 1; 1 -1];
    F2 =  h   *[1 0; 0 1; 0 -1; 1 1; 1 -1];
    wt = @(Fam) [1, 1 + (Fam(:,1).' > 0)];
    Fall = [0 0; amin([F1; F2])];

    % [2a] CONICS FIRST, when enabled.  A conic is first-order NEUTRAL --
    % it changes the shape, not the paraxial power -- so it is the free
    % way to take out the bulk of a tilted fast primary's aberration.
    % Without it the Zernike stage has to supply conic-level sag: the
    % first run's M1 departure came out at 2.4 MILLIMETRES, which is not
    % the "micron-scale departure that does not move the chief ray" the
    % sphere+Zernike doctrine assumes, and the solve moved the EFL by a
    % factor of two.  Conics restore the doctrine's premise.
    if P.tel.conic_stage
        t.set_base_sphere(false);
        rc = t.optimize('fields', amin(F2), 'dofs',[0 0 0 0 0 0 0 1], ...
                        'max_iters', P.tel.conic_iters);
        L = say_(L, '\n[2a] conic solve (first-order neutral): worst %.3f -> %.4f waves', ...
                 max(rc.wfe_before)/P.lambda_m, max(rc.wfe_after)/P.lambda_m);
        L = say_(L, '     K = [%s]', sprintf('%+.5f ', rc.conics));
        efl_c = efl_(t, nE);
        L = say_(L, '     EFL after conics %.3f m -> f/%.2f (must be unchanged)', ...
                 efl_c, efl_c/P.D_m);
    end

    % NORMALIZATION RADIUS.  set_freeform defaults lmon to the element
    % BODY ap_r; on this train the bodies are the design-phase 3 m stubs
    % while the beam is 0.3 m at M2 and 0.08 m at M3, so every mode would
    % be normalized to ~10-40x the lit patch -- over which they are nearly
    % degenerate (they all look like tilt/astig there), the solve goes
    % ill-conditioned, and CALIB nulls one field with huge canceling
    % coefficients.  Measure the FULL-FIELD footprint instead and hand it
    % over per mirror.  This uses the frame-correct aperture_full_field
    % (element-local ApVec frame, fixed 2026-08-24) purely as a RULER --
    % no apertures are applied here (rule 1).
    apr = t.aperture_full_field('fields', Fall, 'margin', 0.0, 'quiet', true);
    % lMon is measured from the element's Mon ORIGIN (the vertex), not from
    % the footprint's own centre, so the enclosing radius is
    % |centre| + radius.  Using the footprint radius alone leaves the far
    % rim of an off-centre patch OUTSIDE the normalization circle, where
    % the Zernikes extrapolate and the solve fights its own basis.
    lfoot = arrayfun(@(a) norm(a.center) + a.radius, apr(1:3));
    switch lower(P.tel.lmon_mode)
        case 'body'
            lmon = NaN(1,3);        % set_freeform's default: the body ap_r
        case 'auto'
            lmon = P.tel.lmon_margin * lfoot;
        otherwise
            error('s1_telescope:lmon','lmon_mode must be body|auto');
    end
    L = say_(L, '\n[2] staged Zernike (freeform) correction, waves RMS @ %g nm:', ...
             P.lambda_m*1e9);
    L = say_(L, '    full-field footprint radius (m): %.4f %.4f %.4f', lfoot);
    L = say_(L, '    lmon mode "%s" -> %s', P.tel.lmon_mode, ...
             tern_(strcmpi(P.tel.lmon_mode,'body'), 'element body ap_r', ...
                   sprintf('%.4f %.4f %.4f', lmon)));
    L = say_(L, '    modes (%s): %s', P.tel.ztype, mat2str(P.tel.modes));
    zarg = {'modes',P.tel.modes, 'type',P.tel.ztype, 'max_iters',P.tel.iters, ...
            'lmon',lmon};
    r0 = t.optimize_freeform([1 2 3], zarg{:}, 'fields_arcmin',[]);
    L = say_(L, '    S0 centre        : %.0f -> %.4f', ...
             r0.wfe_before/P.lambda_m, r0.wfe_after/P.lambda_m);

    r1 = t.optimize_freeform([1 2 3], zarg{:}, ...
             'fields',amin(F1), 'weights',wt(F1));
    L = say_(L, '    S1 inner 2-D     : worst %.3f -> %.4f', ...
             max(r1.wfe_before)/P.lambda_m, max(r1.wfe_after)/P.lambda_m);
    F = [F1; F2];  w = wt(F);
    r2 = t.optimize_freeform([1 2 3], zarg{:}, 'fields',amin(F), 'weights',w);
    wv = r2.wfe_after(:).'/P.lambda_m;
    wfe_worst = max(wv);
    wfe_aw = sqrt(sum(w.*wv.^2)/sum(w));
    L = say_(L, '    S2 full 2-D +-%g'' : worst %.4f, area-weighted %.4f -> %s', ...
             h, wfe_worst, wfe_aw, ...
             tern_(wfe_worst < P.dl_waves, 'DIFFRACTION-LIMITED', 'residual'));

    efl_corr = efl_(t, nE);
    L = say_(L, '    EFL after correction %.3f m -> f/%.2f (spheres alone: see [4])', ...
             efl_corr, efl_corr/P.D_m);

    % ================= [3] the true focal plane ===========================
    span = min(0.25, P.tel.fov_arcmin/2);
    fa = t.align_focal_plane('grid',P.tel.fp_grid, 'span_arcmin',span);
    L = say_(L, '\n[3] fitted FP from a %dx%d field-foci grid: tilt %.3f deg,', ...
             P.tel.fp_grid, P.tel.fp_grid, fa.tilt_deg);
    L = say_(L, '    defocus removed %+.3f mm, field-curvature sag %+.1f to %+.1f um', ...
             fa.defocus_m*1e3, min(fa.sag_m)*1e6, max(fa.sag_m)*1e6);

    % ================= [4] field map + the packaging gates =================
    Fmap = macos.design.field_grid(P.tel.fov_arcmin, P.tel.map_n, 'units','arcmin');
    dmap = wfe_field_diag(t, Fmap, 'quiet',true);
    scan = struct('fields', Fmap*180*60/pi, 'wfe', dmap.rms_raw(:));
    rep  = t.check_clipping('noload',true,'quiet',true);
    clear_ok = all([rep.ok]);
    pk  = packaging_report(t,'quiet',true);
    ao  = aoi_report(t,'quiet',true);
    efl = efl_(t, nE);  fno = efl/P.D_m;
    L = say_(L, '\n[4] first order + packaging');
    L = say_(L, '    EFL %.3f m -> f/%.2f  [%s]', efl, fno, ...
             gate_(fno >= P.fno_band(1) && fno <= P.fno_band(2)));
    L = say_(L, '    shroud %.3f m diameter (%.3f x D), train %.2f m  [%s]', ...
             2*pk.shroud_radius_m, pk.shroud_over_D, pk.length_m, ...
             gate_(2*pk.shroud_radius_m <= P.shroud_D_m));
    L = say_(L, '    clearance %d/%d bodies clear  [%s]', ...
             sum([rep.ok]), numel(rep), gate_(clear_ok));
    L = say_(L, '    dense %dx%d map over +-%g'': raw max %.4f, -tilt max %.4f waves  [%s]', ...
             P.tel.map_n, P.tel.map_n, P.tel.fov_arcmin, ...
             max(dmap.rms_raw), max(dmap.rms_tilt), ...
             gate_(max(dmap.rms_tilt) < P.dl_waves));
    L = say_(L, '    AOI spread per mirror: %s deg (preference < %g)', ...
             sprintf('%.1f ', [ao.aoi_spread_deg]), P.aoi_limit);
    rr = pairs_({t.spec.elt.name}, max([pk.elts.r_body],[pk.elts.r_beam]));
    L = say_(L, '    per-element radial extent (m): %s', ...
             sprintf('%s %.3f   ', rr{:}));

    % ================= [5] exit pupil, save, RELOAD GATE ===================
    t.add_pupil(numel(t.spec.elt));           % EP emits PropType=FarField
    rxfile  = [tag '_telescope.in'];
    matfile = [tag '_telescope.mat'];
    t.save(rxfile);  t.save_spec(matfile);
    gate = reload_gate_(rxfile, P.model);
    L = say_(L, '\n[5] saved %s (+ .mat)', rxfile);
    L = say_(L, '    RELOAD GATE (rule 3): %d elements, %d/%d rays pass  [%s]', ...
             gate.nelt, gate.npass, gate.nray, gate_(gate.ok));
    if ~gate.ok
        L = say_(L, '    ** the saved deck does not reproduce the in-session model.');
    end
    t.build('', 'init', false);                % back to the session model

    % ================= [6] figures =========================================
    figs = struct();
    try
        f1 = t.view_field_map(scan,'kind','contour');
        figs.map = [tag '_wfe_field.png'];  saveas(f1, figs.map);  close(f1);
        f2 = t.view_orthoviews({'YZ','XZ'},'nrays',9);
        figs.layout = [tag '_layout.png'];  saveas(f2, figs.layout);  close(f2);
        fg = figure('Visible','off');
        contourf(fa.map.thx_arcmin, fa.map.thy_arcmin, fa.map.sag_m*1e6, ...
                 15, 'LineColor','none');
        axis equal tight; colormap(parula); cb = colorbar;
        cb.Label.String = 'focus sag from the fitted FP  [\mum]';
        xlabel('\theta_x  [arcmin]'); ylabel('\theta_y  [arcmin]');
        title(sprintf('field curvature (FP tilt %.3f\\circ)', fa.tilt_deg));
        figs.fpmap = [tag '_fpmap.png'];  saveas(fg, figs.fpmap);  close(fg);
        sh = e2e6m_shroud_fig(t, [tag '_shroud.png'], ...
                'shroud_D_m',P.shroud_D_m, ...
                'title', sprintf('%s S1 -- launch-shroud fit (deployed, diameter-only)', P.name));
        figs.shroud = sh.png;
        L = say_(L, '\n[6] figures: layout, WFE field map, FP curvature, shroud fit');
        L = say_(L, '    shroud figure measures %.3f m against the %.1f m gate  [%s]', ...
                 sh.shroud_D_m, sh.gate_D_m, gate_(sh.pass));
    catch ME
        L = say_(L, '\n[6] figures FAILED: %s', ME.message);
    end

    % ================= [7] the design report ================================
    rings = unique([0.25 0.5 P.tel.fov_arcmin]);
    rpt = design_report(t, 'rings_arcmin',rings, 'align',fa, ...
            'dl_waves',P.dl_waves, 'quiet',true, ...
            'file',[tag '_design_report.txt']);
    L = say_(L, '\n[7] design report -> s1_design_report.txt');

    L = say_(L, '\nS1 DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen([tag '_report.txt'],'w');  fprintf(fid,'%s\n',txt);  fclose(fid);
    fprintf('%s\n', txt);

    OUT = struct('P',P, 'wfe0_waves',wfe0, 'stages',{{r0,r1,r2}}, ...
                 'wfe_worst_waves',wfe_worst, 'wfe_aw_waves',wfe_aw, ...
                 'map',dmap, 'scan',scan, 'align',fa, 'clip',rep, ...
                 'pack',pk, 'aoi',ao, 'EFL_m',efl, 'fno',fno, ...
                 'reload',gate, 'figs',figs, 'report',rpt, 'text',txt, ...
                 'rx',rxfile, 'mat',matfile, 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save([tag '_run.mat'],'OUT');
end

% =========================================================================
function setup_(here)
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
end

function L = say_(L, varargin)
%SAY_  Append one report line and echo it.  A plain local function (not
%   nested): this file calls run() on mmacos_setup, which a static
%   workspace would forbid.
    s = sprintf(varargin{:});
    L{end+1} = s;
    fprintf('%s\n', s);
end

function g = reload_gate_(rxfile, model)
%RELOAD_GATE_  Rule 3: a save is only good if the SAVED deck reproduces
%   the in-session model.  Load it standalone and count rays -- this is
%   what catches the realize_apertures frame defect instantly (a
%   tilted-fold deck with global-XY aperture centres loses every ray).
    macos.init(model);
    n = macos.load_rx(rxfile);
    s = macos.trace(n);
    r = macos.get_ray_info(s.nRays);
    np = nnz(logical(r.ok_pass) & logical(r.ok_trace));
    g = struct('nelt',n, 'npass',np, 'nray',s.nRays, 'ok', np > 0.9*s.nRays);
end

function E = efl_(t, fk)
    del = 0.1*pi/180/60;
    t.trace_at_field([0  del]);  s = macos.trace(fk);
    b1 = macos.get_ray_info(s.nRays);
    t.trace_at_field([0 -del]);        macos.trace(fk);
    b2 = macos.get_ray_info(s.nRays);
    t.trace_at_field([]);
    E = norm(b1.pos(:,1) - b2.pos(:,1))/(2*del);
end

function w = rms_waves_(W, lam)
    v = W(isfinite(W) & W ~= 0 & abs(W) < 1e30);
    if isempty(v), w = NaN; else, w = std(v)/lam; end
end

function c = pairs_(names, vals)
    c = cell(1, 2*numel(names));
    c(1:2:end) = names;
    c(2:2:end) = num2cell(vals);
end

function s = gate_(ok), if ok, s = 'PASS'; else, s = 'FAIL'; end, end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
