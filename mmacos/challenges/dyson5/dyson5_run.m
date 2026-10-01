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
%     s3  native optimize -- beat 3
%
%   Artifacts (P.outdir): <tag>_s0_scaling.{txt,mat,png};
%   <tag>_s1_{dyson,offner}.in, <tag>_s1_layout.png, <tag>_s1.{txt,mat};
%   <tag>_s2.{txt,mat}, <tag>_s2_maps.png, <tag>_s2_rad.png.
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
            case 's3'
                fprintf('dyson5_run: stage %s is queued for the next beat (BRIEF_to_dyson5 build order).\n', P.stages{k});
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
        Pk = base;  if strcmp(forms{k}, 'offner'), Pk.Fno = P.Fno_offner; end
        G = spectrometer_geom(forms{k}, Pk);
        file = sprintf('%s_s1_%s.in', tag, forms{k});
        M = spectrometer_rx(G, file, 'ngridpts', P.ngridpts, 'name', [P.tag '_' forms{k}]);
        macos.load_rx(file);
        nE = macos.num_elt();
        assert(nE == M.nElt, 'dyson5 s1: %s loads %d of %d elements', forms{k}, nE, M.nElt);
        macos.stop(M.iG);  macos.modify();
        tr = macos.trace(M.nElt);
        S.(forms{k}) = struct('G', G, 'M', M, 'nRays', tr.nRays, 'file', file);
        switch forms{k}
        case 'dyson'
            pr('DYSON  : block r %.1f mm (%s), R_g %.1f mm (factor %.3f), air gap %.1f mm, face offset %.2f mm\n', ...
                G.r*1e3, P.glass, G.Rg*1e3, P.Rg_factor, G.gap*1e3, P.face_offset_m*1e3);
        case 'offner'
            pr('OFFNER : R %.1f mm concave, convex grating R/2 = %.1f mm, F/%.2f\n', G.R*1e3, G.R/2*1e3, Pk.Fno);
        end
        pr('  slit at y = %+.2f mm; m = %+d, d = %.3f um (%.2f l/mm); FPA centre y = %+.3f mm, z = %+.4f mm;\n', ...
            G.y_slit*1e3, G.grating.m, G.grating.d*1e6, G.grating.lines_per_mm, G.fpa.center(2)*1e3, G.fpa.z*1e3);
        pr('  band edges y = %+.3f / %+.3f mm (span %.3f mm); slit-to-FPA-edge clearance %.2f mm; %d elements,\n', ...
            G.fpa.y_lambda(1)*1e3, G.fpa.y_lambda(3)*1e3, abs(diff(G.fpa.y_lambda([1 3])))*1e3, ...
            G.fpa.clear_to_slit*1e3, M.nElt);
        pr('  grating = elt %d (stop); engine trace at lambda_c: %d rays; deck %s\n', M.iG, tr.nRays, file);
        subplot(1, 2, k);  section_(G, forms{k});
    end
    fclose(fid);
    print(f, [tag '_s1_layout.png'], '-dpng', '-r110');  close(f);
    save([tag '_s1.mat'], 'S', 'P');
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
