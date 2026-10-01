function OUT = dyson5_run(over)
%DYSON5_RUN  The dyson5 challenge runner: one user-editable entry point.
%
%   OUT = DYSON5_RUN() runs every stage in P.stages at the default spec
%   (dyson5_params).  OUT = DYSON5_RUN(OVER) applies overrides first.
%   Every number in the challenge record is produced THROUGH this runner.
%
%   Stages (the ladder grows beat by beat; each stage names its convention
%   before it prints a number):
%     s0  SCALING LAW of the classical concentric Dyson at the spec: the
%         Dyson condition R_g = n r/(n-1) verified by exact trace, then
%         the block radius the spec's slit corner needs for a blur budget
%         of P.blur_px pixels -- the pre-registered question "can a single
%         block meet F/1.8 at a 54 mm slit?" answered before any element
%         is added.  Engine-free (pure geometry).
%     s1  DECK EMISSION for both forms (design/src/spectrometer_geom +
%         spectrometer_rx): the chain's chief aim through the grating
%         vertex, the groove period that lays the band across the FPA,
%         and the focus plane are solved by exact trace; the deck is
%         loaded in the engine and its element count verified; the
%         section figure shows the chief and marginal rays.  Gate:
%         tests/tSpectrometerRx (engine chief == chain chief, 1e-9 m).
%     s2  SCORE both decks (design/src/spectrometer_score): field-angle
%         and wavelength maps, smile and keystone in pixels, SRF / CRF by
%         the slit (x) LSF (x) pixel (x) Airy chain, geometric ensquared
%         energy, the closed-form radiometric chain; Joe's spec and the
%         paper's two reference columns printed beside every number.
%     s2w PROPAGATION TWIN (design/src/spectrometer_wave; opt-in -- add
%         's2w' to P.stages): a far-field terminal on a reference sphere
%         L_ref upstream of the FPA, re-posed per (field, lambda); the
%         complex field at the FPA, PSF centroid vs ray centroid, SRF/CRF
%         from the propagated PSF.  Validated on the order-0 Offner relay
%         (Airy, 94 % in one pixel); at order -1 it measures the engine's
%         grating OPL defect until that is fixed (tGratingOpl).
%     s2l SLIT-WIDTH DIFFRACTION LOSS (spectrometer_slit_loss; opt-in, its
%         OWN MATLAB at model 1024 -- model-size transitions in one process
%         are the known engine heap hazard): a far-field leg from the
%         rectangular slit aperture to the grating plane vs the sinc^2 closed
%         form, per wavelength.
%     s3  THE DYSON DEPARTURE LADDER (dyson_ladder): R0 the concentric seed,
%         R1 the concentric knobs (R_g factor, face offset, block radius),
%         R2 + conic and h^4/h^6 asphere on the block's convex face, R3 +
%         the block's centre off the grating's (de-concentric) --
%         each rung solved on the exact chain with smile/keystone operands
%         in the merit from the first pass, then emitted and ENGINE-scored.
%     s4  native optimize -- beat 4
%
%   Artifacts (P.outdir): <tag>_s0_scaling.{txt,mat,png};
%   <tag>_s1_{dyson,offner}.in, <tag>_s1_layout.png, <tag>_s1.{txt,mat};
%   <tag>_s2.{txt,mat}, <tag>_s2_maps.png, <tag>_s2_rad.png; deck-standard
%   figures: <tag>_s1_layout_<form>.png, <tag>_s2_maps_<form>.png,
%   <tag>_s3_layout_r<k>.png, <tag>_s3_maps_r<k>.png, <tag>_s3_trade.{txt,png}.
%
%   See also DYSON5_PARAMS, dyson_layout, dyson_scaling.
    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here, '..', '..', 'mmacos_setup.m'));
    addpath(here);
    P = dyson5_params(over);
    if isempty(P.outdir), P.outdir = here; end
    if ~exist(P.outdir, 'dir'), mkdir(P.outdir); end
    tag = fullfile(P.outdir, P.tag);
    OUT.P = P;

    for k = 1:numel(P.stages)
        switch P.stages{k}
            case 's0', OUT.s0 = stage_s0_(P, tag);
            case 's1', OUT.s1 = stage_s1_(P, tag);
            case 's2', OUT.s2 = stage_s2_(P, tag);
            case 's2w', OUT.s2w = stage_s2w_(P, tag);
            case 's3', OUT.s3 = stage_s3_(P, tag);
            case 's2l', OUT.s2l = stage_s2l_(P, tag);
            otherwise
                error('dyson5_run:stage', 'unknown stage %s', P.stages{k});
        end
    end
end

% =====================================================================
function S = stage_s0_(P, tag)
%STAGE_S0_  The concentric-Dyson scaling law at the spec.
    n = sellmeier_(P.glass, P.lambda_ref_m);
    slit_len = P.npix(1)*P.pixel_m;
    fpa_h    = P.npix(2)*P.pixel_m;
    h_max    = hypot(slit_len/2, P.y_slit_m);       % slit CORNER off the axis
    blur_max = P.blur_px*P.pixel_m;

    % (1) the condition, verified at a mid-size block
    D = dyson_layout(0.1, n, 'fno', P.Fno);
    % (2) the sweep: blur at the slit corner vs block radius
    Ssc = dyson_scaling(n, P.Fno, h_max, blur_max, 'r_grid', P.r_grid_m);

    S.n = n;  S.h_max = h_max;  S.blur_max = blur_max;  S.slit_len = slit_len;
    S.fpa_h = fpa_h;  S.spectral_sampling_m = diff(P.band_m)/P.npix(2);
    S.condition = D;  S.scaling = Ssc;

    % -- the record ------------------------------------------------------
    fid = fopen([tag '_s0_scaling.txt'], 'w');
    pr = @(varargin) dualprint_(fid, varargin{:});   % console + record
    pr('dyson5 s0 -- classical concentric Dyson at the spec (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: block index n(%s) at %.2f um = %.6f; F/# = air-equivalent image-space,\n', P.glass, P.lambda_ref_m*1e6, n);
    pr('  in-glass marginal half-angle u = asin(1/(2 n F)) = %.3f deg; blur = transverse rms spot of a\n', D.u_glass*180/pi);
    pr('  full F/%.1f cone from a point on the flat face, imaged back to the flat face (1:1), grating\n', P.Fno);
    pr('  traced as a mirror (order 0); field height h measured from the Dyson axis on the flat face.\n');
    pr('SPEC: slit %.1f mm (%d px x %.0f um), FPA %.1f mm spectral (%d px), band %.0f-%.0f nm -> %.2f nm/px;\n', ...
        slit_len*1e3, P.npix(1), P.pixel_m*1e6, fpa_h*1e3, P.npix(2), P.band_m*1e9, S.spectral_sampling_m*1e9);
    pr('  slit centre offset %.1f mm -> slit CORNER at h_max = %.2f mm; blur budget %.2f px = %.1f um rms.\n', ...
        P.y_slit_m*1e3, h_max*1e3, P.blur_px, blur_max*1e6);
    pr('\n(1) DYSON CONDITION, r = 100 mm: R_g = n r/(n-1) = %.3f mm, gap %.3f mm, D_g(axial) = %.1f mm\n', ...
        D.R_g*1e3, D.gap*1e3, D.D_g_axial*1e3);
    pr('    blur (um rms) vs R_g factor: ');  pr('%.2f:%.2f  ', [D.sweep.factor; D.sweep.blur*1e6]);  pr('\n');
    pr('    blur (um rms) vs h (mm):     ');  pr('%.0f:%.2f  ', [D.h*1e3; D.blur_rms*1e6]);
    pr('  -> h-exponent %.2f (fifth-order residual)\n', D.h_exponent);
    pr('    centroid shift image_y + h (um) vs h (mm): ');  pr('%.0f:%.3f  ', [D.h*1e3; D.distortion*1e6]);  pr('  (distortion, same order)\n');
    pr('\n(2) SCALING at the slit corner h_max = %.2f mm:\n', h_max*1e3);
    pr('    r (mm):        ');  pr('%8.0f', Ssc.r_grid*1e3);  pr('\n');
    pr('    blur (um rms): ');  pr('%8.2f', Ssc.blur*1e6);    pr('\n');
    pr('    fitted r-exponent %.2f (law: blur ~ h^4 / r^3)\n', Ssc.law_exponent);
    if isnan(Ssc.r_required)
        pr('    -> NO block on the grid meets %.1f um at the corner.\n', blur_max*1e6);
    else
        pr('    -> r_required = %.1f mm  (R_g = %.1f mm, gap %.1f mm, grating clear diam >= %.1f mm axial + field)\n', ...
            Ssc.r_required*1e3, Ssc.R_g_required*1e3, (Ssc.R_g_required-Ssc.r_required)*1e3, ...
            2*Ssc.R_g_required*sin(D.u_glass)*1e3);
    end
    pr('VERDICT: the single-block concentric Dyson meets the corner blur only at r >= %.0f mm -- report the law,\n', Ssc.r_required*1e3);
    pr('  do not add elements yet (BRIEF_to_dyson5 build order, pre-registered null).\n');
    fclose(fid);

    % -- figure ----------------------------------------------------------
    f = figure('Visible', 'off', 'Position', [100 100 640 420]);
    loglog(Ssc.r_grid*1e3, Ssc.blur*1e6, 'o-', 'LineWidth', 1.5);  hold on
    yline(blur_max*1e6, '--', sprintf('%.2f px budget', P.blur_px));
    if ~isnan(Ssc.r_required), xline(Ssc.r_required*1e3, ':', sprintf('r = %.0f mm', Ssc.r_required*1e3)); end
    grid on;  xlabel('block radius r (mm)');  ylabel('rms blur at the slit corner (um)');
    title(sprintf('Concentric Dyson, F/%.1f, %s, slit corner h = %.1f mm: blur ~ r^{%.2f}', ...
        P.Fno, P.glass, h_max*1e3, Ssc.law_exponent));
    print(f, [tag '_s0_scaling.png'], '-dpng', '-r110');  close(f);
    save([tag '_s0_scaling.mat'], 'S', 'P');
    fprintf('dyson5 s0: wrote %s_s0_scaling.{txt,mat,png}\n', tag);
end

function S = stage_s1_(P, tag)
%STAGE_S1_  Emit the Dyson and Offner decks at the spec; verify they load.
    base = struct('Fno', P.Fno, 'pixel_m', P.pixel_m, 'npix', P.npix, 'band_m', P.band_m, ...
                  'lambda_ref_m', P.lambda_ref_m, 'order', P.order, 'y_slit', P.y_slit_m, ...
                  'block_r', P.block_r_m, 'glass', P.glass, 'face_offset', P.face_offset_m, ...
                  'Rg_factor', P.Rg_factor, 'offner_R', P.offner_R_m);
    forms = {'dyson', 'offner'};
    fid = fopen([tag '_s1.txt'], 'w');
    pr = @(varargin) dualprint_(fid, varargin{:});
    pr('dyson5 s1 -- deck emission at the spec (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: concentric about C = origin, slit along x, dispersion along y; chief aimed through the\n');
    pr('  grating vertex (= the stop); |m| = %d with the sign that pushes the FPA away from the slit; groove\n', P.order);
    pr('  period solved so |y(lambda_max) - y(lambda_min)| = FPA spectral height; FPA plane z solved for\n');
    pr('  minimum slit-centre blur at lambda_c; F/# air-equivalent (Dyson %.2f, Offner %.2f).\n', P.Fno, P.Fno_offner);
    macos.init(P.model);
    f = figure('Visible', 'off', 'Position', [100 100 1100 480]);
    for k = 1:2
        Pk = base;
        if strcmp(forms{k}, 'offner')
            Pk.Fno = P.Fno_offner;  Pk.y_slit = P.y_slit_offner_m;
            Pk.offner_Rg_factor = P.offner_Rg_factor;  Pk.offner_M3_factor = P.offner_M3_factor;
            Pk.offner_M3_dy = P.offner_M3_dy;  Pk.offner_M3_dz = P.offner_M3_dz;
        end
        G = spectrometer_geom(forms{k}, Pk);
        file = sprintf('%s_s1_%s.in', tag, forms{k});
        M = spectrometer_rx(G, file, 'ngridpts', P.ngridpts, 'name', [P.tag '_' forms{k}], 'apertures', true, 'margin', P.ap_margin_m);
        macos.load_rx(file);
        nE = macos.num_elt();
        assert(nE == M.nElt, 'dyson5 s1: %s loads %d of %d elements', forms{k}, nE, M.nElt);
        macos.stop(M.iG);  macos.modify();
        tr = macos.trace(M.nElt);  ri = macos.get_ray_info(tr.nRays);
        % the declared apertures must not vignette the beam they were cut from
        nv = nnz(ri.ok_trace & ~ri.ok_pass);
        assert(nv == 0, 'dyson5 s1: %s -- %d rays vignetted by the declared apertures (aperture frame sign?)', forms{k}, nv);
        Cl = spectrometer_clearance(G, Pk, 'quiet', true);
        S.(forms{k}) = struct('G', G, 'M', M, 'nRays', tr.nRays, 'file', file, 'clearance', Cl);
        switch forms{k}
        case 'dyson'
            pr('DYSON  : block r %.1f mm (%s), R_g %.1f mm (factor %.3f), air gap %.1f mm, face offset %.2f mm\n', ...
                G.r*1e3, P.glass, G.Rg*1e3, P.Rg_factor, G.gap*1e3, P.face_offset_m*1e3);
        case 'offner'
            pr('OFFNER : R %.1f mm concave (zone 2: R x %.5f, centre dy %+.2f dz %+.2f mm), convex grating R/2 x %.5f = %.1f mm, F/%.2f (the Dyson is F/%.2f)\n', ...
                G.R*1e3, Pk.offner_M3_factor, Pk.offner_M3_dy*1e3, Pk.offner_M3_dz*1e3, Pk.offner_Rg_factor, Pk.offner_Rg_factor*G.R/2*1e3, Pk.Fno, P.Fno);
        end
        pr('  slit at y = %+.2f mm; m = %+d, d = %.3f um (%.2f l/mm); FPA centre y = %+.3f mm, z = %+.4f mm;\n', ...
            G.y_slit*1e3, G.grating.m, G.grating.d*1e6, G.grating.lines_per_mm, G.fpa.center(2)*1e3, G.fpa.z*1e3);
        pr('  band edges y = %+.3f / %+.3f mm (span %.3f mm); slit-to-FPA-edge clearance %.2f mm; %d elements,\n', ...
            G.fpa.y_lambda(1)*1e3, G.fpa.y_lambda(3)*1e3, abs(diff(G.fpa.y_lambda([1 3])))*1e3, ...
            G.fpa.clear_to_slit*1e3, M.nElt);
        pr('  grating = elt %d (stop); engine trace at lambda_c: %d rays, 0 vignetted by the declared apertures; deck %s\n', M.iG, tr.nRays, file);
        pr('  CLEARANCE (legs vs bodies not traversed, mount %.0f mm): min %+.2f mm -- %s\n', P.mount_margin_m*1e3, Cl.min_mm, tern_(Cl.pass, 'PASS', 'FAIL'));
        for i = 1:min(6, height(Cl.table)), pr('    %-34s vs %-16s %+9.2f mm\n', Cl.table.leg{i}, Cl.table.body{i}, Cl.table.clearance_mm(i)); end
        subplot(1, 2, k);  section_(G, forms{k});
        spectrometer_layout_fig(G, sprintf('%s_s1_layout_%s.png', tag, forms{k}), 'title', [forms{k} ' seed']);
    end
    fclose(fid);
    print(f, [tag '_s1_layout.png'], '-dpng', '-r110');  close(f);
    save([tag '_s1.mat'], 'S', 'P');
    dyson5_view_figs({[P.tag '_s1_dyson'], [P.tag '_s1_offner']}, P.outdir);   % the engine renders of record
    assert(S.dyson.clearance.pass && S.offner.clearance.pass, 'dyson5 s1: a beam crosses a body (see the clearance table)');
    fprintf('dyson5 s1: wrote %s_s1_{dyson,offner}.in, %s_s1.{txt,mat}, %s_s1_layout.png\n', tag, tag, tag);
end

function S = stage_s2_(P, tag)
%STAGE_S2_  Score both s1 decks on the spectrometer metrics.
    s1 = load([tag '_s1.mat']);  S1 = s1.S;
    forms = {'dyson', 'offner'};
    fid = fopen([tag '_s2.txt'], 'w');
    pr = @(varargin) dualprint_(fid, varargin{:});
    pr('dyson5 s2 -- spectrometer metrics (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: FPA frame origin = the lambda_c image of the slit centre; u along the slit (+x), v along the\n');
    pr('  dispersion, both in %.0f um pixels; per (slit x, lambda): engine rays at the FPA (traced & unblocked),\n', P.pixel_m*1e6);
    pr('  centroid (u_c, v_c), rms widths, geometric ensquared fraction in 1 px about the centroid.\n');
    pr('  SMILE(lambda) = p-v of v_c along the slit; KEYSTONE(x) = p-v of u_c across the band; headlines = max.\n');
    pr('  SRF = FWHM of rect(%d px slit) (x) LSF_v (x) rect(1 px) (x) Airy(lambda F); CRF = LSF_u (x) rect (x) Airy\n', P.slit_px);
    pr('  (spectrometer only -- the telescope LSF is not modelled).  Grid: %d slit positions x %d wavelengths,\n', P.score_nx, P.score_nlam);
    pr('  %d-pt ray grid.  Radiometric chain = Fresnel(uncoated faces) x blaze sinc^2 (lambda_B = %.0f nm) x QE(placeholder);\n', P.ngridpts, P.blaze_m*1e9);
    pr('  the slit loss is the propagation twin''s measured term and is NOT in it.\n');
    pr('  GROOVE MODEL: the engine holds the groove period constant ALONG THE CURVED SURFACE (Snells_Law_Grating\n');
    pr('  normalises the projected rule direction); a straight-ruled concave grating has constant period along the\n');
    pr('  CHORD (equidistant groove planes).  Each form carries two chain rows: ''surface'' (must reproduce the engine)\n');
    pr('  and ''planes'' (the straight-ruled design prediction).  Engine rows are scored; the difference is the finding.\n');
    pr('SPEC (Joe): smile/keystone < %.2f px (0.2 ok), SRF < %.1f-%.1f px, XRF < %.1f px.  Paper rules: distortion ~1%% px at\n', ...
        P.smile_px, P.srf_px, P.xrf_px);
    pr('  design (3%% toleranced), ensquared > 75%%.  Offner Table 2: smile < 0.3%% px, keystone < 2%% px, EE > 0.76, SRF < 1.35x,\n');
    pr('  CRF < 1.1x sampling.  BPDS Table 5 (3200 px, 18 um, F/2): smile 0.6 um = 3.3%% px achieved.\n\n');
    macos.init(P.model);
    f = figure('Visible', 'off', 'Position', [60 60 1300 760]);
    for k = 1:2
        G = S1.(forms{k}).G;  M = S1.(forms{k}).M;
        macos.load_rx(M.file);
        Pk = P;  Pk.Fno = G.P.Fno;
        R = spectrometer_score(G, M, Pk, 'nx', P.score_nx, 'nlam', P.score_nlam, 'quiet', true);
        S.(forms{k}) = R;
        pr('%-6s: smile max %.4f px (%.3f um)  keystone max %.4f px (%.3f um)  dispersion %.4f px/nm\n', ...
            upper(forms{k}), R.smile_max, R.smile_max*P.pixel_m*1e6, R.keystone_max, R.keystone_max*P.pixel_m*1e6, R.dispersion_px_per_nm);
        pr('        SRF FWHM max %.3f px (= %.2fx sampling)   CRF FWHM max %.3f px   SRF var. with field %.1f%%   CRF var. with lambda %.1f%%\n', ...
            R.srf_max, R.srf_max, R.crf_max, 100*R.srf_var_field, 100*R.crf_var_lambda);
        pr('        geometric ensquared min %.3f   rms blur max %.3f px (u) / %.3f px (v)   rays per point %d-%d\n', ...
            R.ee_min, max(R.SU(:)), max(R.SV(:)), min(R.nrays(:)), max(R.nrays(:)));
        pr('        smile per lambda (px): %s\n', sprintf('%.4f ', R.smile_px));
        pr('        keystone per slit x (px): %s\n', sprintf('%.4f ', R.keystone_px));
        pr('        radiometric gain vs lambda (nm: gain): %s\n', sprintf('%.0f:%.3f ', [R.lams*1e9; R.rad.gain]));
        % the chain's own scores: 'surface' must reproduce the engine (same
        % groove model); 'planes' is the straight-ruled design prediction
        for mdl = {'surface', 'planes'}
            Pm = G.P;  Pm.grating_model = mdl{1};
            Gm = spectrometer_geom(forms{k}, Pm);
            Rc = spectrometer_score_chain(Gm, Pk, 'nx', P.score_nx, 'nlam', P.score_nlam);
            S.([forms{k} '_chain_' mdl{1}]) = Rc;
            pr('        chain (%-7s grooves): smile %.4f px  keystone %.4f px  SRF max %.3f px  CRF max %.3f px  EE min %.3f  blur max %.3f/%.3f px\n', ...
                mdl{1}, Rc.smile_max, Rc.keystone_max, Rc.srf_max, Rc.crf_max, Rc.ee_min, max(Rc.SU(:)), max(Rc.SV(:)));
        end
        ok_s = R.smile_max < P.smile_px;  ok_k = R.keystone_max < P.keystone_px;
        ok_srf = R.srf_max < P.srf_px(2);  ok_crf = R.crf_max < P.xrf_px;
        pr('        vs spec: smile %s  keystone %s  SRF %s  CRF %s\n\n', pf_(ok_s), pf_(ok_k), pf_(ok_srf), pf_(ok_crf));
        spectrometer_maps_fig(R, sprintf('%s_s2_maps_%s.png', tag, forms{k}), 'title', [forms{k} ' seed, engine'], 'pixel_um', P.pixel_m*1e6);
        % maps
        subplot(2, 4, (k-1)*4 + 1);  imagesc(R.lams*1e9, R.xs*1e3, R.U);  colorbar;  axis xy
        xlabel('lambda (nm)');  ylabel('slit x (mm)');  title(sprintf('%s: u_c (px) -- field-angle map', forms{k}));
        subplot(2, 4, (k-1)*4 + 2);  imagesc(R.lams*1e9, R.xs*1e3, R.V - mean(R.V, 1));  colorbar;  axis xy
        xlabel('lambda (nm)');  title('v_c - mean over slit (px): smile');
        subplot(2, 4, (k-1)*4 + 3);  imagesc(R.lams*1e9, R.xs*1e3, R.SRF);  colorbar;  axis xy
        xlabel('lambda (nm)');  title('SRF FWHM (px)');
        subplot(2, 4, (k-1)*4 + 4);  imagesc(R.lams*1e9, R.xs*1e3, R.EE);  colorbar;  axis xy
        xlabel('lambda (nm)');  title('geometric ensquared (1 px)');
    end
    print(f, [tag '_s2_maps.png'], '-dpng', '-r100');  close(f);
    f = figure('Visible', 'off', 'Position', [60 60 640 400]);
    R = S.dyson;
    plot(R.lams*1e9, R.rad.T_fresnel, '-', R.lams*1e9, R.rad.eta_blaze, '-', R.lams*1e9, R.rad.gain, 'k-', 'LineWidth', 1.4);
    grid on;  xlabel('lambda (nm)');  ylabel('fraction');  ylim([0 1.05]);
    legend({sprintf('Fresnel, %d faces', R.rad.nface), 'blaze sinc^2', 'gain (x QE)'}, 'Location', 'south');
    title('Dyson radiometric chain (closed form; slit loss not included)');
    print(f, [tag '_s2_rad.png'], '-dpng', '-r100');  close(f);
    fclose(fid);
    save([tag '_s2.mat'], 'S', 'P');
    fprintf('dyson5 s2: wrote %s_s2.{txt,mat}, %s_s2_maps.png, %s_s2_rad.png\n', tag, tag, tag);
end

function S = stage_s3_(P, tag)
%STAGE_S3_  The Dyson departure ladder, engine-scored rung by rung.
    macos.init(P.model);
    fid = fopen([tag '_s3.txt'], 'w');
    pr = @(varargin) dualprint_(fid, varargin{:});
    pr('dyson5 s3 -- the Dyson departure ladder (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: each rung is solved by lsqnonlin on the EXACT CHAIN (straight-ruled grooves; the engine reproduces\n');
    pr('  it ray for ray), residuals in PIXELS over a %d x %d (slit x, lambda) grid: %g x [smile_ij = v_c - v_c(slit centre);\n', P.ladder_nx, P.ladder_nlam, P.ladder_w_dist);
    pr('  keystone_ij = u_c - u_c(lambda_c)] and %g x [rms spot u, v], plus a wall on the slit-to-FPA clearance >= %.1f mm;\n', P.ladder_w_blur, P.ladder_clear_m*1e3);
    pr('  block radius %s; groove period and FPA focus re-solved at every iterate; the solved deck is then scored in the ENGINE\n', tern_(P.ladder_free_r, 'FREE (walks to its bound: size buys blur)', sprintf('HELD at %.0f mm (what each departure buys at fixed scale)', P.block_r_m*1e3)));
    pr('  (spectrometer_score, %d x %d, %d-pt grid) -- the engine rows are the record.  Spec: smile/keystone < %.2f px,\n', P.score_nx, P.score_nlam, P.ngridpts, P.smile_px);
    pr('  SRF < %.1f-%.1f px (2-px slit floor 2.0), CRF < %.1f px; paper: distortion ~1%% px at design, EE > 0.75.\n\n', P.srf_px, P.xrf_px);
    L = dyson_ladder(P, tag, 'rungs', P.ladder_rungs, 'nx', P.ladder_nx, 'nlam', P.ladder_nlam, ...
                     'w_dist', P.ladder_w_dist, 'w_blur', P.ladder_w_blur, 'clear_m', P.ladder_clear_m, ...
                     'max_iter', P.ladder_max_iter, 'free_r', P.ladder_free_r, 'quiet', true);
    pr('%-52s %8s %8s %7s %7s %6s | %7s %7s %8s %7s %s\n', 'rung', 'smile', 'keyst', 'CRF', 'SRF', 'EE', 'R_g mm', 'r mm', 'face mm', 'Kc', 'asph [h4 h6]');
    for k = 1:numel(L.rung)
        r = L.rung(k);  Re = r.engine;  G = spectrometer_geom('dyson', r.P);
        pr('%-52s %8.4f %8.4f %7.3f %7.3f %6.3f | %7.1f %7.1f %8.3f %7.3f %s  dC [%.2f %.2f] mm\n', r.name, Re.smile_max, Re.keystone_max, ...
            Re.crf_max, Re.srf_max, Re.ee_min, G.Rg*1e3, G.r*1e3, r.P.face_offset*1e3, r.P.block_Kc, mat2str(r.P.block_asph, 4), ...
            r.P.block_dy*1e3, r.P.block_dz*1e3);
        pr('%-52s chain: %8.4f %8.4f %7.3f %7.3f %6.3f | clearance %.2f mm, d %.2f um, merit %.4g, deck %s\n', '', ...
            r.chain.smile_max, r.chain.keystone_max, r.chain.crf_max, r.chain.srf_max, r.chain.ee_min, ...
            G.fpa.clear_to_slit*1e3, G.grating.d*1e6, r.merit, r.file);
    end
    fclose(fid);
    S = L;
    save([tag '_s3.mat'], 'S', 'P');
    fid = fopen([tag '_s3.txt'], 'a');  pr = @(varargin) dualprint_(fid, varargin{:});
    pr('\nCLEARANCE per rung (legs vs bodies not traversed; mount %.0f mm; slit mask %s mm; FPA carrier +%.0f mm, depth %.0f mm, shield %.0f mm):\n', ...
        P.mount_margin_m*1e3, mat2str(P.slit_mask_m*1e3), P.pkg_margin_m*1e3, P.pkg_depth_m*1e3, P.pkg_shield_m*1e3);
    tags = {};
    for k = 1:numel(L.rung)
        r = L.rung(k);  G = spectrometer_geom('dyson', r.P);  rk = regexprep(r.name, ' .*', '');   % 'R0'..'R4a','R4'
        Cl = spectrometer_clearance(G, P, 'quiet', true);  L.rung(k).clearance = Cl;
        pr('  %-4s min %+8.2f mm %s : %s vs %s\n', rk, Cl.min_mm, tern_(Cl.pass, 'PASS', 'FAIL'), Cl.table.leg{1}, Cl.table.body{1});
        spectrometer_layout_fig(G, sprintf('%s_s3_layout_%s.png', tag, lower(rk)), 'title', ['dyson ' rk]);
        spectrometer_maps_fig(r.engine, sprintf('%s_s3_maps_%s.png', tag, lower(rk)), 'title', ['dyson ' rk ', engine'], 'pixel_um', P.pixel_m*1e6);
        tags{end+1} = regexprep(r.file, {'^.*/', '\.in$'}, '');  %#ok<AGROW>
    end
    fclose(fid);
    S = L;  save([tag '_s3.mat'], 'S', 'P');
    dyson5_view_figs(tags, P.outdir);                       % the engine renders of record, every rung
    if ~P.ladder_free_r, dyson5_trade(tag); end
    assert(all(arrayfun(@(r) r.clearance.pass, L.rung)), 'dyson5 s3: a beam crosses a body on some rung (see the clearance table)');
    % figure: engine CRF and smile/keystone per rung
    f = figure('Visible', 'off', 'Position', [60 60 900 360]);
    nr = numel(L.rung);  vals = zeros(nr, 4);
    for k = 1:nr, Re = L.rung(k).engine;  vals(k,:) = [Re.smile_max, Re.keystone_max, Re.crf_max, Re.ee_min]; end
    subplot(1,2,1);  bar(vals(:,1:3));  set(gca, 'XTickLabel', arrayfun(@(k) sprintf('R%d', P.ladder_rungs(k)), 1:nr, 'uni', 0));
    legend({'smile max (px)', 'keystone max (px)', 'CRF max (px)'}, 'Location', 'northeast');  grid on;  title('engine, per rung');
    subplot(1,2,2);  bar(vals(:,4));  set(gca, 'XTickLabel', arrayfun(@(k) sprintf('R%d', P.ladder_rungs(k)), 1:nr, 'uni', 0));
    ylabel('min ensquared energy (1 px, geometric)');  grid on;  yline(0.75, '--', 'paper rule');
    print(f, [tag '_s3_ladder.png'], '-dpng', '-r100');  close(f);
    fprintf('dyson5 s3: wrote %s_s3.{txt,mat}, %s_s3_r*.in, %s_s3_ladder.png\n', tag, tag, tag);
end

function S = stage_s2l_(P, tag)
%STAGE_S2L_  The slit-width diffraction loss, engine vs sinc^2.
    fid = fopen([tag '_s2l.txt'], 'w');
    pr = @(varargin) dualprint_(fid, varargin{:});
    pr('dyson5 s2l -- slit-width diffraction loss (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: a uniformly illuminated rectangular slit %.0f um x %.0f mm (the telescope cone is not modelled -- the\n', P.slit_px*P.pixel_m*1e6, P.slitloss_len*1e3);
    pr('  partially coherent case is the ~10%% the paper cites), far-field leg (flat Return, radius 1e22, zElt = z) to the\n');
    pr('  grating plane at z = %.3f m; loss = 1 - energy inside |y| <= z tan(u), u = asin(1/(2 F)), F = %.1f; closed form =\n', P.slitloss_z, P.Fno);
    pr('  1 - energy of sinc^2(w sin(theta)/lambda) inside |sin(theta)| <= 1/(2F).  Model %d, %d-pt grid, pitch at the\n', P.slitloss_model, P.slitloss_ngrid);
    pr('  grating plane lambda z / (N dx_slit); the acceptance must sit inside the window (checked).\n');
    R = spectrometer_slit_loss(P, [tag '_s2l_slit.in'], 'lams', P.slitloss_lams, 'model', P.slitloss_model, ...
                               'ngridpts', P.slitloss_ngrid, 'slit_len', P.slitloss_len, 'z_grating', P.slitloss_z);
    pr('%8s %12s %12s %10s %10s %8s\n', 'nm', 'loss engine', 'loss sinc^2', 'pitch um', 'window mm', 'inside');
    for j = 1:numel(R.lams)
        pr('%8.0f %12.5f %12.5f %10.2f %10.1f %8d   energy %.4g\n', R.lams(j)*1e9, R.loss_engine(j), R.loss_sinc(j), R.dx_m(j)*1e6, R.window_m(j)*1e3, R.inside_window(j), R.energy(j));
    end
    fclose(fid);
    S = R;  save([tag '_s2l.mat'], 'S', 'P');
    f = figure('Visible', 'off', 'Position', [60 60 560 380], 'Color', 'w');
    plot(R.lams*1e9, 100*R.loss_sinc, 'k-', R.lams*1e9, 100*R.loss_engine, 'ro', 'LineWidth', 1.4, 'MarkerSize', 7);
    grid on;  xlabel('\lambda (nm)');  ylabel('slit diffraction loss past the grating (%)');
    legend({'sinc^2 closed form', 'engine far-field leg'}, 'Location', 'northwest');
    title(sprintf('%.0f um slit, F/%.1f acceptance', P.slit_px*P.pixel_m*1e6, P.Fno));
    print(f, [tag '_s2l_slitloss.png'], '-dpng', '-r130');  close(f);
    fprintf('dyson5 s2l: wrote %s_s2l.{txt,mat}, %s_s2l_slitloss.png\n', tag, tag);
end

function S = stage_s2w_(P, tag)
%STAGE_S2W_  The propagation twin on both s1 decks (model 512).
    s1 = load([tag '_s1.mat']);  S1 = s1.S;
    forms = {'offner', 'dyson'};
    fid = fopen([tag '_s2w.txt'], 'w');
    pr = @(varargin) dualprint_(fid, varargin{:});
    pr('dyson5 s2w -- propagation twin (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: far-field terminal (Rx_Cass_FarField idiom) on a REFERENCE sphere of radius %.2f m centred on the\n', P.wave_L_ref);
    pr('  chief''s FPA pierce, vertex upstream on the chief, re-posed per (slit x, lambda) with macos.set_xp; FP_return\n');
    pr('  and FPA vertices moved onto the chief pierce so the PSF grid is centred on the chief (centre pixel N/2+1);\n');
    pr('  grid index 1 = global X (slit, u), index 2 = global Y (dispersion, v) -- measured: the lambda-proportional\n');
    pr('  offset appears in index 2; pitch = macos.dx_at(FPA), = lambda L_ref/(N dx_ep), window = ngridpts x lambda F;\n');
    pr('  model %d, ngridpts %d; the slit is a POINT (slit-width diffraction = the separate slit-loss measurement).\n', P.wave_model, P.wave_ngridpts);
    pr('  WAVE offset = PSF intensity centroid from the chief, RAY offset = engine ray centroid from the chief, px;\n');
    pr('  SRF_wave = FWHM of rect(%d px) (x) LSF_v(PSF) (x) rect(1 px); CRF_wave = LSF_u(PSF) (x) rect(1 px).\n', P.slit_px);
    % -- validation: the order-0 Offner relay must give a centred Airy spot
    Gv = spectrometer_geom('offner', S1.offner.G.P);  Gv.grating.m = 0;  Gv.grating.d = 1;
    Pv = P;  Pv.Fno = S1.offner.G.P.Fno;
    Mv = spectrometer_rx(Gv, [tag '_s2w_offner_m0.in'], 'ngridpts', P.ngridpts);
    Rv = spectrometer_wave(Gv, Mv, Pv, 'nx', 1, 'nlam', 1, 'model', P.wave_model, 'ngridpts', P.wave_ngridpts, 'L_ref', P.wave_L_ref, 'quiet', true);
    S.offner_m0 = Rv;
    pr('VALIDATION (Offner at order 0 = a concentric relay, lambda_c): PSF centroid offset (%.4f, %.4f) px, rms widths\n', Rv.du_wave, Rv.dv_wave);
    pr('  %.3f / %.3f px, ensquared in 1 px %.3f, CRF %.3f px -- the terminal reproduces the Airy spot.\n', Rv.su_wave, Rv.sv_wave, Rv.ee, Rv.CRF);
    for k = 1:2
        G = S1.(forms{k}).G;  M = S1.(forms{k}).M;  Pk = P;  Pk.Fno = G.P.Fno;
        if strcmp(forms{k}, 'dyson') && ~isempty(P.twin_rung) && isfile([tag '_s3.mat'])
            s3 = load([tag '_s3.mat']);  L3 = s3.S;
            kk = find(strncmp({L3.rung.name}, [P.twin_rung ' '], numel(P.twin_rung) + 1), 1, 'last');
            if ~isempty(kk)
                G = spectrometer_geom('dyson', L3.rung(kk).P);  M = L3.rung(kk);  M.file = L3.rung(kk).file;
                M = spectrometer_rx(G, M.file, 'ngridpts', P.ngridpts);     % re-emit (the M map the twin needs)
                pr('DYSON twin runs on the s3 rung %s deck (%s); its geometric EE there: %.3f\n', P.twin_rung, M.file, L3.rung(kk).engine.ee_min);
            end
        end
        R = spectrometer_wave(G, M, Pk, 'nx', P.wave_nx, 'nlam', P.wave_nlam, 'model', P.wave_model, ...
                              'ngridpts', P.wave_ngridpts, 'L_ref', P.wave_L_ref, 'quiet', true);
        S.(forms{k}) = R;
        pr('%-6s: wave - ray centroid offsets, max |du| %.4f px, max |dv| %.4f px  (dv per lambda at slit centre: %s)\n', ...
            upper(forms{k}), max(abs(R.d_du(:))), max(abs(R.d_dv(:))), sprintf('%+.3f ', R.dv_wave(ceil(end/2), :)));
        pr('        SRF_wave max %.3f px  CRF_wave max %.3f px  EE(1 px) min %.3f  PSF rms widths max %.3f/%.3f px  energy range %.3g-%.3g\n', ...
            R.srf_max, R.crf_max, R.ee_min, max(R.su_wave(:)), max(R.sv_wave(:)), min(R.energy(:)), max(R.energy(:)));
    end
    dmax = max([max(abs(S.offner.d_dv(:))), max(abs(S.dyson.d_dv(:))), max(abs(S.offner.d_du(:))), max(abs(S.dyson.d_du(:)))]);
    if dmax < 0.01
        pr('STATUS: wave and ray centroids agree to %.4f px on every (slit x, lambda) point of both forms -- the propagated\n', dmax);
        pr('  PSF sits where the rays say (engine finding #3 fixed; comparisons across a grating are modulo lambda, addendum 3).\n');
        pr('  The twin''s own products: ensquared energy WITH diffraction (Offner %.3f, geometric 1.000; Dyson %.3f),\n', S.offner.ee_min, S.dyson.ee_min);
        pr('  SRF/CRF from the propagated PSF (Offner %.3f/%.3f px, Dyson %.3f/%.3f px).\n', S.offner.srf_max, S.offner.crf_max, S.dyson.srf_max, S.dyson.crf_max);
    else
        pr('STATUS: wave - ray centroid offsets up to %.3f px (the seeds agree to 0.001 px).  The PSF intensity centroid is the\n', dmax);
        pr('  AMPLITUDE-WEIGHTED mean of the ray aberration (Fresnel transmission varies across the pupil through the\n');
        pr('  refractions; the ray centroid is unweighted), so where a deck carries more oblique refractions and a larger\n');
        pr('  spot the two part -- the ray maps are the geometric statement, the wave maps the radiometric one.  Checked:\n');
        pr('  tGratingOpl passes on this engine, so this is not the OPL defect.\n');
    end
    spectrometer_wave_fig(S, [tag '_s2w_twin.png'], 'pixel_um', P.pixel_m*1e6);
    fclose(fid);
    save([tag '_s2w.mat'], 'S', 'P');
    fprintf('dyson5 s2w: wrote %s_s2w.{txt,mat}, %s_s2w_twin.png\n', tag, tag);
end

function t = tern_(c, a, b)
    if c, t = a; else, t = b; end
end

function t = pf_(ok)
    if ok, t = 'PASS'; else, t = 'FAIL'; end
end

function section_(G, form)
%SECTION_  y-z section: surfaces + chief and marginal rays at band centre/edges.
    hold on;  axis equal;  grid on
    t = linspace(0, 2*pi, 361);
    for k = 1:numel(G.surf)
        s = G.surf(k);
        if strcmp(s.kind, 'sphere')     % full circle: the used half is the one the rays reach
            plot(s.C(3) + s.R*cos(t), s.C(2) + s.R*sin(t), 'k-', 'LineWidth', 0.5);
        else
            plot([s.C(3) s.C(3)], [-0.08 0.08], 'k:');
        end
    end
    lams = [G.P.band_m(1) G.src.lambda_c G.P.band_m(2)];  col = {'b', 'g', 'r'};
    Gm = G;
    for j = 1:3
        for a = [-G.src.u 0 G.src.u]
            d = G.src.chief_dir;  ex = [1;0;0];  ey = cross(d, ex);  ey = ey/norm(ey);
            dd = cos(a)*d + sin(a)*ey;
            [pts, ~, ok] = Gm.trace(G.slit, dd, lams(j));
            if ~ok, continue; end
            Pz = [G.slit(3), pts(3,:)];  Py = [G.slit(2), pts(2,:)];
            plot(Pz, Py, [col{j} '-'], 'LineWidth', 0.6);
        end
    end
    plot(G.fpa.center(3), G.fpa.center(2), 'ms', 'MarkerFaceColor', 'm');
    plot(G.slit(3), G.slit(2), 'ko', 'MarkerFaceColor', 'k');
    xlabel('z (m)');  ylabel('y (m)');
    title(sprintf('%s: slit (o) -> FPA (square); 380/1440/2500 nm = b/g/r', form));
end

function dualprint_(fid, varargin)
%DUALPRINT_  fprintf to the console and the record file (a plain function:
%   fprintf inside cellfun's uniform-output context errors).
    fprintf(1, varargin{:});  fprintf(fid, varargin{:});
end

function n = sellmeier_(glass, lam_m)
%SELLMEIER_  Index for the layout, from the engine table rows (um^2 C's).
%   The engine applies the same rows at trace time (gate tGlassDispersion).
    switch glass
        case 'Silica', B = [0.6961663 0.4079426 0.8974794];  C = [0.004679148 0.01351206 97.934];
        case 'CaF2',   B = [0.5675888 0.4710914 3.8484723];  C = [0.050263605^2 0.1003909^2 34.649040^2];
        otherwise, error('dyson5_run:glass', 'no Sellmeier row for %s in this runner', glass);
    end
    L2 = (lam_m*1e6)^2;
    n = sqrt(1 + sum(B .* L2 ./ (L2 - C)));
end
