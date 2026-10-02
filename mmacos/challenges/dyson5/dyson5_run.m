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
%     s4  NATIVE OPTIMIZE (dyson_native; opt-in): CALIB, the engine's own
%         multi-field x multi-wavelength least squares, on the R4 deck --
%         SPOT target (max ray distance to the chief per field, lambda) over
%         grating position, block face radius + conic, meniscus faces, focus,
%         with the double-pass copies linked -- in chunks of iterations with
%         the smile/keystone WALLS held on the chain between chunks (CALIB
%         has no distortion operand; the ask is on CC's list).  Every chunk
%         is read back from the engine, mapped into the chain and proven by
%         an identity check before it is scored and gated (walls, clearance).
%     s5  R5, THE FOLD PRISM (opt-in): an entrance plate on the slit side and
%         a mirror-coated fold prism on the image side, both cemented to the
%         block; the FPA folds away from the slit's plane so a detector
%         package WITH a cold shield fits.  The shield height is the
%         parameter: each height in P.fold_shield_sweep_m sets the air gap
%         beyond the prism, the design is re-solved on the chain (ladder
%         rung R5, warm-started from R4 then along the sweep), engine-scored
%         and put through the clearance gate; the record is the height
%         P.fold_shield_m.  Table + deck-standard figures + trade row.
%     s4env THE CLOSURE ENVELOPE (dyson5_envelope; opt-in; addendum 11): for
%         which parameters does the R4 design close?  R4 re-solved from the
%         record one axis at a time (F-number, block radius, slit length,
%         pixel, glass), engine-scored, judged against the spec with the
%         failing metric named; solves on their bounds are not called closed;
%         then the two-axis corner of the first failures.  Table + figure +
%         the sentence the run-it-yourself slide needs.
%
%     t1  THE TELESCOPE (beat 5, addendum 7; opt-in): the fore-optics that
%         feed the slit at EMIT's parameters (420 km, 60 m ground sample ->
%         0.143 mrad per pixel, f = 126 mm, 70 mm at F/1.8, 24.6 deg across
%         track onto the 54 mm slit).  A coaxial three-mirror anastigmat's
%         off-axis section, first-order seed (telescope_seed: a flat field
%         and the exit pupil at the spectrometer's -- the Dyson is telecentric
%         at the slit to 0.09 deg), a flat fold so the 0.7 m spectrometer
%         lies outside the telescope, rungs T1-T3 solved on the exact chain
%         (telescope_ladder: conics + radii + spacings + bias; + h^4/h^6
%         aspheres; + M2/M3 decentre and tilt) with the pupil match and the
%         clearance wall in the merit, each rung emitted with apertures and
%         ENGINE-scored at the slit (telescope_score: spot, slit admittance,
%         telecentricity, pupil match as the chief's miss of the grating
%         vertex, field flatness, mapping), the clearance gate on the
%         combined chain with the R4 spectrometer's bodies.
%     t2  END TO END (opt-in, after t1): the telescope of record prepended to
%         the R4 and the R5 spectrometers as ONE prescription each (e2e_geom),
%         a collimated field source with the GRATING as the stop, scored by
%         the spectrometer's scorer over fields x wavelengths (smile,
%         keystone, SRF, CRF, ensquared energy, and the fraction of the
%         launched bundle the grating admits per field), the clearance gate
%         across both instruments, the engine renders.
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
            case 's4',  OUT.s4  = stage_s4_(P, tag);
            case 's5',  OUT.s5  = stage_s5_(P, tag);
            case 's4env', OUT.s4env = stage_s4env_(P, tag);
            case 't1',  OUT.t1  = stage_t1_(P, tag);
            case 't2',  OUT.t2  = stage_t2_(P, tag);
            case 't3',  OUT.t3  = stage_t3_(P, tag);
            case 't3s', OUT.t3s = stage_t3s_(P, tag);
            case 't3w', OUT.t3w = stage_t3w_(P, tag);
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
                               'ngridpts', P.slitloss_ngrid, 'slit_len', P.slitloss_len, 'z_grating', P.slitloss_z, ...
                               'propagating', P.slitloss_propagating);
    pr('  RESOLVED (dyson5_s2l_tests.txt, addendum 9): a window of 1.11 x the acceptance aliased the tail back inside (0.30 x\n');
    pr('  at 380 nm); the planar FFT far field carries energy at |sin theta| > 1 that no physical far field does (1.36 x at\n');
    pr('  2500 nm) -- the record now uses a window >= 2 x the acceptance and normalises to the propagating region; the\n');
    pr('  residual few %% is the pitch across a 4-sidelobe acceptance at 2500 nm.\n');
    pr('%8s %12s %12s %7s %10s %10s %8s %8s\n', 'nm', 'loss engine', 'loss sinc^2', 'ratio', 'pitch um', 'win/acc', 'evanesc.', 'energy');
    for j = 1:numel(R.lams)
        pr('%8.0f %12.5f %12.5f %7.3f %10.2f %10.2f %8.4f %8.3g\n', R.lams(j)*1e9, R.loss_engine(j), R.loss_sinc(j), R.loss_engine(j)/R.loss_sinc(j), R.dx_m(j)*1e6, R.window_ratio(j), R.evanescent_frac(j), R.energy(j));
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
                G = spectrometer_geom('dyson', L3.rung(kk).P);
                % the twin's own copy of the rung deck (never overwrite the
                % ladder's: it carries the declared apertures the tables and
                % renders are read from)
                M = spectrometer_rx(G, sprintf('%s_s2w_%s_src.in', tag, lower(P.twin_rung)), 'ngridpts', P.ngridpts, ...
                                    'apertures', true, 'margin', P.ap_margin_m);
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

function N = stage_s4_(P, tag)
%STAGE_S4_  The native optimize on the rung of record (P.native_rung).
    fn = [tag '_s3.mat'];
    assert(isfile(fn), 'dyson5 s4 needs the ladder record %s (run s3 first)', fn);
    S3 = load(fn);  L = S3.S;
    k = find(strncmp({L.rung.name}, [P.native_rung ' '], numel(P.native_rung) + 1), 1);
    assert(~isempty(k), 'dyson5 s4: rung %s is not in %s', P.native_rung, fn);
    r4 = L.rung(k);
    % BLOCKED on the engine (2026-10-01, BRIEF_dyson5_beat4c.md section 3.4):
    % CALIB's derivative loop steps the SPOT objective at the wavefront-map
    % stride (design_optim.F ~:792 `off=off+opd_size`, 16384 for a 30-long
    % objective) -- a heap stomp on the second (field, wavelength) that kills
    % the MATLAB process.  The stage refuses to run until the fix is on the
    % engine of record; set over.native_enabled = true to run it then.
    assert(P.native_enabled, ['dyson5 s4: the native optimize is blocked on the engine (CALIB SPOT-target derivative ' ...
           'stride, design_optim.F ~:792; CC''s lane -- BRIEF_dyson5_beat4c.md 3.4).  Pass native_enabled = true once fixed.']);
    macos.init(P.model);
    fid = fopen([tag '_s4.txt'], 'w');  pr = @(varargin) dualprint_(fid, varargin{:});
    pr('dyson5 s4 -- the native optimize on %s (%s)\n', P.native_rung, datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: CALIB (design_optim.F, LM) on the rung''s deck with the double-pass copies LINKED (Link=; PERTURB/ROC/\n');
    pr('  CONIC/ASPH apply to both passes); target SPOT = the max ray distance to the chief at the FPA, one number per (slit\n');
    pr('  position, wavelength), %d x %d fields, equal weights, driven to 0; variables: block convex face\n', P.native_nx, P.native_nlam);
    pr('  ROC+CONIC (asphere %s), meniscus faces PIST+ROC, FPA PIST; variable set ''%s'' (''all'' adds the grating''s DY+PIST, which\n', tern_(P.native_asph, 'FREE', 'HELD'), P.native_varset);
    pr('  the blur merit cannot police: measured keystone 0.003 -> 15 px in five iterations, wall-rejected); groove period HELD.  Chunks of %d\n', P.native_chunk);
    pr('  iterations; after each the engine state is read back, mapped into the chain (identity to 1e-9 m), ENGINE-scored\n');
    pr('  on the %d x %d grid and gated: smile and keystone <= %.3f px (half the spec, Dave''s wall on iterates), clearance\n', P.score_nx, P.score_nlam, P.native_wall_px);
    pr('  PASS; a breach restores the last accepted state.  The rung of record is re-emitted CLEAN from the mapped chain.\n\n');
    N = dyson_native(P, tag, r4.P, 'nx', P.native_nx, 'nlam', P.native_nlam, 'chunk', P.native_chunk, ...
                     'max_chunks', P.native_max_chunks, 'wall_px', P.native_wall_px, 'tol_px', P.native_tol_px, ...
                     'asph', P.native_asph, 'varset', P.native_varset, 'quiet', true);
    H = N.history;
    pr('%-5s %5s %8s %8s %7s %7s %6s %8s %9s  %s\n', 'chunk', 'iters', 'smile', 'keyst', 'CRF', 'SRF', 'EE', 'clear', 'identity', 'status');
    for i = 1:height(H)
        pr('%-5d %5d %8.4f %8.4f %7.3f %7.3f %6.3f %8.2f %9.1e  %s %s\n', H.chunk(i), H.iters(i), H.smile(i), H.keystone(i), H.crf(i), H.srf(i), H.ee(i), ...
           H.clear_mm(i), H.identity_m(i), tern_(H.accepted(i), 'accepted', 'REJECTED'), H.note{i});
    end
    Re = N.rung.engine;  Re0 = r4.engine;  Pn = N.rung.P;  Gn = spectrometer_geom('dyson', Pn);
    pr('\n%-34s %8s %8s %7s %7s %6s\n', 'engine score', 'smile', 'keyst', 'CRF', 'SRF', 'EE');
    pr('%-34s %8.4f %8.4f %7.3f %7.3f %6.3f\n', [P.native_rung ' of record'], Re0.smile_max, Re0.keystone_max, Re0.crf_max, Re0.srf_max, Re0.ee_min);
    pr('%-34s %8.4f %8.4f %7.3f %7.3f %6.3f\n', 'R4n native', Re.smile_max, Re.keystone_max, Re.crf_max, Re.srf_max, Re.ee_min);
    pr('R4n parameters (chain frame, grating centre at the origin): block r %.3f mm Kc %.5f; block centre dy %.3f dz %.3f mm;\n', ...
       Pn.block_r*1e3, Pn.block_Kc, Pn.block_dy*1e3, Pn.block_dz*1e3);
    pr('  slit plane z %.3f mm, y_slit %.3f mm, face offset %.3f mm; meniscus vertex %.2f mm, t %.3f mm, c %.4f / %.4f /m;\n', ...
       Pn.slit_dz*1e3, Pn.y_slit*1e3, Pn.face_offset*1e3, Pn.men_z*1e3, Pn.men_t*1e3, Pn.men_ca, Pn.men_cb);
    pr('  FPA z %.4f mm (R4: %.4f); grating shift in the engine frame [%.3f %.3f %.3f] mm; band span %.1f px of %d.\n', ...
       Pn.fpa_z*1e3, N.seed.P.fpa_z*1e3, N.shift_m*1e3, N.span_px, N.span_spec_px);
    Cl = spectrometer_clearance(Gn, P, 'quiet', true);  N.rung.clearance = Cl;
    pr('CLEARANCE R4n: min %+.2f mm %s : %s vs %s\n', Cl.min_mm, tern_(Cl.pass, 'PASS', 'FAIL'), Cl.table.leg{1}, Cl.table.body{1});
    fclose(fid);
    spectrometer_layout_fig(Gn, sprintf('%s_s4_layout_r4n.png', tag), 'title', 'dyson R4n (native)');
    spectrometer_maps_fig(Re, sprintf('%s_s4_maps_r4n.png', tag), 'title', 'dyson R4n (native), engine', 'pixel_um', P.pixel_m*1e6);
    S = N;  save([tag '_s4.mat'], 'S', 'P');
    dyson5_view_figs({regexprep(N.rung.file, {'^.*/', '\.in$'}, '')}, P.outdir);
    if ~P.ladder_free_r, dyson5_trade(tag); end
    assert(Cl.pass, 'dyson5 s4: the native design crosses a body (see the clearance table)');
end

function N = stage_s5_(P, tag)
%STAGE_S5_  R5, the fold prism, under the clearance gate: cold-shield height sweep.
    fn = [tag '_s3.mat'];
    assert(isfile(fn), 'dyson5 s5 needs the ladder record %s (run s3 first)', fn);
    S3 = load(fn);  L3 = S3.S;
    k = find(strncmp({L3.rung.name}, 'R4 ', 3), 1);
    assert(~isempty(k), 'dyson5 s5: rung R4 is not in %s', fn);
    r4 = L3.rung(k);
    macos.init(P.model);
    fid = fopen([tag '_s5.txt'], 'w');  pr = @(varargin) dualprint_(fid, varargin{:});
    pr('dyson5 s5 -- R5, the fold prism, under the clearance gate (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: R4 of record + an entrance PLATE (slit %.1f mm in air before it, cemented to the block face) and a\n', P.fold_slit_gap_m*1e3);
    pr('  mirror-coated FOLD PRISM cemented under the image (fold plane %.0f mm below the face at 45 deg, folding toward -y,\n', P.fold_h_m*1e3);
    pr('  away from the slit; TIR fails at F/1.8 in silica), exit face placed for first-order conjugate symmetry, the FPA\n');
    pr('  (normal +y, dispersion along +z) an AIR GAP beyond it.  COLD-SHIELD HEIGHT h is the parameter: air gap = h + %.1f mm,\n', P.fold_shield_clear_m*1e3);
    pr('  the SAME on the slit side (slit to plate), so the concentric form keeps equal object and image media;\n');
    pr('  at each h the R5 rung (R4''s variables + the face offset at the fold''s scale, exit distance >= 7.5 mm) is re-solved\n');
    pr('  on the exact chain (lsqnonlin, %d iterations, warm-started from R4 then along the sweep), the deck emitted with\n', P.fold_max_iter);
    pr('  apertures and ENGINE-scored (%d x %d), the clearance gate run with the FPA package (54 x 9 mm + %.0f mm, %.0f mm deep,\n', P.score_nx, P.score_nlam, P.pkg_margin_m*1e3, P.pkg_depth_m*1e3);
    pr('  shield h toward the prism) in the FOLDED frame; plate and prism are one cemented part with the block.  Record: %s.\n\n', ...
       tern_(isempty(P.fold_shield_m), 'the tallest shield that closes', sprintf('h = %.0f mm', P.fold_shield_m*1e3)));
    P5 = P;  P5.fold_h = P.fold_h_m;  P5.slit_gap = P.fold_slit_gap_m;  P5.plate = true;  P5.face_offset_m = P.fold_face_offset_m;
    seed = r4.P;  seed.face_offset = P.fold_face_offset_m;
    hs = P.fold_shield_sweep_m;
    if ~isempty(P.fold_shield_m) && ~any(abs(hs - P.fold_shield_m) < 1e-9), hs = sort([hs, P.fold_shield_m]); end
    closes_ = @(r, Cl) Cl.pass && r.engine.smile_max < P.smile_px && r.engine.keystone_max < P.smile_px && r.engine.crf_max < P.xrf_px ...
                       && r.engine.srf_max < max(P.srf_px(end), P.slit_px) + 0.1;
    rows = {};  rungs = [];                       % the ladder's rung struct + clearance + shield_m (built from the first)
    pr('%-6s %8s %8s %8s %8s %7s %7s %6s %8s %7s %8s  %s\n', 'h mm', 'gap mm', 'face mm', 'smile', 'keyst', 'CRF', 'SRF', 'EE', 'clear', 'PASS', 'fold_e', 'worst pair');
    for i = 1:numel(hs)
        h = hs(i);
        % the SAME air gap on both sides (slit to plate = FPA to prism): the
        % concentric form wants equal object and image media, and air on the
        % image side alone cost CRF 1.19 -> 1.54 px and EE 0.79 -> 0.41 at a
        % 3 mm gap (measured 2026-10-01); the slit mask then sits h + 1 mm off
        % the plate, which it can
        P5.fpa_gap = max(P.fold_slit_gap_m, h + P.fold_shield_clear_m);  P5.slit_gap = P5.fpa_gap;  P5.pkg_shield_m = h;
        deck = sprintf('%s_s5_r5_h%02.0f.in', tag, h*1e3);
        L = dyson_ladder(P5, tag, 'rungs', 6, 'seed', seed, 'deck', deck, 'nx', P.ladder_nx, 'nlam', P.ladder_nlam, ...
                         'w_dist', P.ladder_w_dist, 'w_blur', P.ladder_w_blur, 'clear_m', 0, 'max_iter', P.fold_max_iter, 'quiet', true);
        r = L.rung(1);  G = spectrometer_geom('dyson', r.P);  Re = r.engine;
        Cl = spectrometer_clearance(G, P5, 'quiet', true);
        r.clearance = Cl;  r.shield_m = h;
        if isempty(rungs), rungs = r; else, rungs(end+1) = r; end   %#ok<AGROW>
        pr('%-6.1f %8.2f %8.2f %8.4f %8.4f %7.3f %7.3f %6.3f %8.2f %7s %8.2f  %s vs %s\n', h*1e3, P5.fpa_gap*1e3, r.P.face_offset*1e3, ...
           Re.smile_max, Re.keystone_max, Re.crf_max, Re.srf_max, Re.ee_min, Cl.min_mm, tern_(Cl.pass, 'PASS', 'FAIL'), G.fold.e*1e3, ...
           Cl.table.leg{1}, Cl.table.body{1});
        rows(end+1, :) = {h*1e3, P5.fpa_gap*1e3, r.P.face_offset*1e3, Re.smile_max, Re.keystone_max, Re.crf_max, Re.srf_max, Re.ee_min, ...
                          Cl.min_mm, Cl.pass, G.fold.e*1e3, r.file};   %#ok<AGROW>
        seed = r.P;                                   % warm start along the sweep
    end
    % REFINE the record height: the sweep's solves are short (P.fold_max_iter)
    % and warm-started along the sweep, and the landscape is multimodal (the
    % 2026-10-01 sweep landed h = 2 and 5 mm in a 29 mm-face basin at CRF 1.38
    % while h = 0 and 10 mm found 17 / 23.5 mm faces at CRF 1.30); so the
    % record height is re-solved from the BEST sweep point's design (lowest
    % engine CRF) with twice the iterations, and the better of the two stands
    ok_ = arrayfun(@(r) closes_(r, r.clearance), rungs);
    if isempty(P.fold_shield_m)
        % the record height: the TALLEST shield whose point closes (the spec and
        % the clearance gate); none closing -> the lowest height, reported as such
        if any(ok_), ir = find(ok_, 1, 'last'); else, ir = 1; end
        h = rungs(ir).shield_m;
        pr('record height: %s\n', tern_(any(ok_), sprintf('%.0f mm, the tallest shield that closes', h*1e3), 'NO height closes -- the lowest is the record, and does not close'));
    else
        ir = find(abs([rungs.shield_m] - P.fold_shield_m) < 1e-9, 1);  h = P.fold_shield_m;
    end
    [~, ib] = min(arrayfun(@(r) r.engine.crf_max, rungs));
    P5.fpa_gap = max(P.fold_slit_gap_m, h + P.fold_shield_clear_m);  P5.slit_gap = P5.fpa_gap;  P5.pkg_shield_m = h;
    deck = sprintf('%s_s5_r5_h%02.0f_refined.in', tag, h*1e3);
    Lr = dyson_ladder(P5, tag, 'rungs', 6, 'seed', rungs(ib).P, 'deck', deck, 'nx', P.ladder_nx, 'nlam', P.ladder_nlam, ...
                      'w_dist', P.ladder_w_dist, 'w_blur', P.ladder_w_blur, 'clear_m', 0, 'max_iter', 2*P.fold_max_iter, 'quiet', true);
    rr = Lr.rung(1);  Gr = spectrometer_geom('dyson', rr.P);  rr.clearance = spectrometer_clearance(Gr, P5, 'quiet', true);  rr.shield_m = h;
    pr('refine h = %.0f mm from the best sweep basin (h = %.0f mm, face %.2f mm), %d iterations: face %.2f mm, CRF %.3f EE %.3f smile %.4f keystone %.4f, clearance %+.2f mm %s\n', ...
       h*1e3, rungs(ib).shield_m*1e3, rungs(ib).P.face_offset*1e3, 2*P.fold_max_iter, rr.P.face_offset*1e3, rr.engine.crf_max, rr.engine.ee_min, ...
       rr.engine.smile_max, rr.engine.keystone_max, rr.clearance.min_mm, tern_(rr.clearance.pass, 'PASS', 'FAIL'));
    rows(end+1, :) = {h*1e3, P5.fpa_gap*1e3, rr.P.face_offset*1e3, rr.engine.smile_max, rr.engine.keystone_max, rr.engine.crf_max, rr.engine.srf_max, ...
                      rr.engine.ee_min, rr.clearance.min_mm, rr.clearance.pass, Gr.fold.e*1e3, rr.file};
    if rr.clearance.pass && rr.engine.crf_max < rungs(ir).engine.crf_max
        rungs(end+1) = rr;  ir = numel(rungs);  pr('  -> the refined solve is the record\n');
    else
        pr('  -> the sweep solve stands as the record\n');
    end
    T = cell2table(rows, 'VariableNames', {'shield_mm', 'air_gap_mm', 'face_offset_mm', 'smile_px', 'keystone_px', 'CRF_px', 'SRF_px', ...
                                           'EE_1px', 'clearance_mm', 'pass', 'fold_exit_mm', 'deck'});
    T.Properties.RowNames = [arrayfun(@(k) sprintf('sweep%d', k), 1:numel(hs), 'uni', 0), {'refined'}];
    rec = rungs(ir);  rec.name = sprintf('R5 fold prism, shield %.0f mm (record)', h*1e3);
    Re = rec.engine;  Re4 = r4.engine;  G5 = spectrometer_geom('dyson', rec.P);
    pr('\n%-40s %8s %8s %7s %7s %6s %9s\n', 'engine score', 'smile', 'keyst', 'CRF', 'SRF', 'EE', 'clear mm');
    pr('%-40s %8.4f %8.4f %7.3f %7.3f %6.3f %9.2f (no shield; slit mask vs package)\n', 'R4 of record', Re4.smile_max, Re4.keystone_max, Re4.crf_max, Re4.srf_max, Re4.ee_min, r4.clearance.min_mm);
    pr('%-40s %8.4f %8.4f %7.3f %7.3f %6.3f %9.2f (%s vs %s)\n', rec.name, Re.smile_max, Re.keystone_max, Re.crf_max, Re.srf_max, Re.ee_min, ...
       rec.clearance.min_mm, rec.clearance.table.leg{1}, rec.clearance.table.body{1});
    pr('R5 of record: face offset %.2f mm (plate %.2f mm thick), fold plane %.0f mm below the face, exit face %.2f mm beyond the fold,\n', ...
       rec.P.face_offset*1e3, G5.fold.plate_t*1e3, G5.fold.h*1e3, G5.fold.e*1e3);
    pr('  air gap %.2f mm, FPA centre [%.1f %.1f %.1f] mm (slit at [%.1f %.1f %.1f]); R_g factor %.5f, block Kc %.5f, asph %s, dC [%.2f %.2f] mm,\n', ...
       G5.fold.fpa_gap*1e3, G5.fpa.center*1e3, G5.slit*1e3, rec.P.Rg_factor, rec.P.block_Kc, mat2str(rec.P.block_asph, 4), rec.P.block_dy*1e3, rec.P.block_dz*1e3);
    pr('  meniscus vertex %.2f mm, t %.2f mm, c %.4f / %.4f /m; groove period %.3f um; deck %s\n', rec.P.men_z*1e3, rec.P.men_t*1e3, rec.P.men_ca, rec.P.men_cb, ...
       G5.grating.d*1e6, rec.file);
    fclose(fid);
    spectrometer_layout_fig(G5, sprintf('%s_s5_layout_r5.png', tag), 'title', 'dyson R5 (fold prism)');
    spectrometer_maps_fig(Re, sprintf('%s_s5_maps_r5.png', tag), 'title', 'dyson R5 (fold prism), engine', 'pixel_um', P.pixel_m*1e6);
    % sweep figure: image quality and clearance vs shield height
    f = figure('Visible', 'off', 'Position', [40 40 1000 360], 'Color', 'w');
    Ts = T(1:numel(hs), :);  Tr = T(end, :);
    subplot(1, 3, 1);  plot(Ts.shield_mm, Ts.CRF_px, 'o-', Ts.shield_mm, Ts.SRF_px, 's-', Tr.shield_mm, Tr.CRF_px, 'kp', 'MarkerSize', 10);  yline(1.5, '--', 'spec CRF');  grid on;
    xlabel('cold-shield height (mm)');  ylabel('FWHM (px)');  legend({'CRF max', 'SRF max', 'refined record'}, 'Location', 'best');  title('response functions (engine)');
    subplot(1, 3, 2);  plot(Ts.shield_mm, Ts.EE_1px, 'o-', Tr.shield_mm, Tr.EE_1px, 'kp', 'MarkerSize', 10);  yline(0.75, '--', 'paper > 0.75');  grid on;
    xlabel('cold-shield height (mm)');  ylabel('min ensquared (1 px)');  title('ensquared energy (engine)');
    subplot(1, 3, 3);  plot(Ts.shield_mm, Ts.clearance_mm, 'o-', Ts.shield_mm, Ts.face_offset_mm, 's-', Tr.shield_mm, Tr.face_offset_mm, 'kp', 'MarkerSize', 10);  yline(0, 'k--');  grid on;
    xlabel('cold-shield height (mm)');  ylabel('mm');  legend({'min clearance', 'face offset (plate)'}, 'Location', 'best');  title('clearance gate, plate thickness');
    print(f, sprintf('%s_s5_sweep.png', tag), '-dpng', '-r130');  close(f);
    N = struct('rung', rec, 'sweep', T, 'rungs', rungs, 'R4', r4);
    S = N;  save([tag '_s5.mat'], 'S', 'P');
    dyson5_view_figs({regexprep(rec.file, {'^.*/', '\.in$'}, '')}, P.outdir);
    if ~P.ladder_free_r, dyson5_trade(tag); end
    if ~rec.clearance.pass, warning('dyson5 s5: the R5 design of record crosses a body (see the clearance table)'); end
end

function E = stage_s4env_(P, tag)
%STAGE_S4ENV_  The closure envelope (addendum 11) around the R4 of record.
    fn = [tag '_s3.mat'];
    assert(isfile(fn), 'dyson5 s4env needs the ladder record %s (run s3 first)', fn);
    S3 = load(fn);  L3 = S3.S;
    k = find(strncmp({L3.rung.name}, 'R4 ', 3), 1);  r4 = L3.rung(k);
    macos.init(P.model);
    fid = fopen([tag '_s4env.txt'], 'w');  pr = @(varargin) dualprint_(fid, varargin{:});
    pr('dyson5 s4env -- the closure envelope around R4 (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: one axis at a time from the design of record, every point a full R4 solve (all eleven variables,\n');
    pr('  lsqnonlin %d iterations on the exact chain, warm-started from R4), its deck emitted with apertures and ENGINE-\n', P.env_max_iter);
    pr('  scored (%d x %d); CLOSES = smile and keystone < %.2f px, CRF < %.1f px, SRF < %.1f px (the %d-px slit is the floor,\n', P.score_nx, P.score_nlam, P.smile_px, P.xrf_px, max(P.srf_px(end), P.slit_px) + 0.1, P.slit_px);
    pr('  the optics may add 0.1 px) AND no variable on a bound (a solve on its bounds is not a closed design; the meniscus\n');
    pr('  curvature bounds are widened to [0.05, 8] /m here, since R4 of record sits on 0.5).  The FPA stays 54 x 9 mm: the pixel count follows the slit length\n');
    pr('  and the pixel pitch.  Then the two-axis corner: the first failing value of the first two failing axes, together.\n\n');
    E = dyson5_envelope(P, tag, r4.P, 'max_iter', P.env_max_iter, 'quiet', true);
    T = E.table;
    pr('%-16s %-10s %8s %8s %7s %7s %6s  %-5s %-14s %s\n', 'axis', 'value', 'smile', 'keyst', 'CRF', 'SRF', 'EE', 'close', 'fails', 'on bounds');
    for i = 1:height(T)
        pr('%-16s %-10s %8.4f %8.4f %7.3f %7.3f %6.3f  %-5s %-14s %s\n', T.axis{i}, T.value{i}, T.smile_px(i), T.keystone_px(i), T.CRF_px(i), T.SRF_px(i), T.EE_1px(i), ...
           tern_(T.closes(i), 'yes', 'NO'), T.fails{i}, T.on_bounds{i});
    end
    if ~isempty(E.corner)
        c = E.corner(1);  row = c.row;
        pr('CORNER %s = %s with %s = %s: %s (fails: %s; on bounds: %s)\n', c.axes{1}, dyson5_vstr_(c.values{1}), c.axes{2}, dyson5_vstr_(c.values{2}), ...
           tern_(row{8}, 'closes', 'does NOT close'), row{9}, row{10});
    else
        pr('CORNER: fewer than two axes fail -- no two-axis corner to run\n');
    end
    pr('\n%s\n', E.sentence);
    fclose(fid);
    S = E;  save([tag '_s4env.mat'], 'S', 'P');
end

function S = stage_t1_(P, tag)
%STAGE_T1_  The telescope (beat 5): seed, ladder, engine score at the slit, clearance with R4.
    GD = dyson_of_record_(P, tag, 'R4');
    macos.init(P.model);
    fid = fopen([tag '_t1.txt'], 'w');  pr = @(varargin) dualprint_(fid, varargin{:});
    ifov = P.tel_gsd_m/P.tel_alt_m;  f = P.pixel_m/ifov;  D = f/P.Fno;  fov = P.npix(1)*ifov;
    pr('dyson5 t1 -- the telescope that feeds the slit (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: EMIT parameters -- altitude %.0f km, ground sample %.0f m -> IFOV %.4f mrad per %.0f um pixel -> f = %.1f mm,\n', P.tel_alt_m*1e-3, P.tel_gsd_m, ifov*1e3, P.pixel_m*1e6, f*1e3);
    pr('  D = f/%.1f = %.1f mm, field = %d px x IFOV = %.2f deg across track (along the slit, +x).  Form: a coaxial positive-\n', P.Fno, D*1e3, P.npix(1), fov*180/pi);
    pr('  negative-positive three-mirror anastigmat (concave M1, convex M2, concave M3; KrElt = -|R|, psi to the centre of\n');
    pr('  curvature), its off-axis section selected by a field BIAS across the slit, a FLAT fold after M3 turning the beam into\n');
    pr('  the spectrometer so the 0.7 m block lies outside the telescope; the field is the sky line that images ONTO the\n');
    pr('  straight slit (the across-slit angle solved per field), so the image of a straight sky line is curved -- the push-\n');
    pr('  broom''s orthorectified truth.  First-order seed (telescope_seed): EFL, a FLAT field (Petzval sum 0) and the EXIT\n');
    pr('  PUPIL at the spectrometer''s apparent entrance pupil, which the R4 chain puts %.1f m behind the slit (the Dyson is\n', 16.84);
    pr('  telecentric at the slit to 0.09 deg); in that limit t2 = f y2 and phi3 = 1/t2 -- the family is one-dimensional\n');
    pr('  in the beam compression y2 at a given t1.  Seed t1 = %.0f mm, y2 = %.2f, bias %.0f deg, chief FOLD angles at M1/M2/M3\n', P.tel_t1_m*1e3, P.tel_y2, P.tel_bias_deg);
    pr('  %s deg (Bauer: the coaxial section of this family cannot be unobscured at F/1.8 -- its spacings are the beam''s\n', mat2str(P.tel_tilt_deg));
    pr('  size -- so the chief is folded at each mirror; a scan over fold angles found this open layout); spheres (the Seidel\n');
    pr('  n-flip seed does not describe this geometry).  Rungs on the EXACT CHAIN (lsqnonlin, residuals in px over %d fields):\n', P.tel_nfield);
    pr('  T0 the LAYOUT (bias, spacings, M2/M3 decentre + tilt, the fold''s distance; the wall dominant, the image terms\n');
    pr('  weak); T1 conics + radii + spacings + bias; T2 + h^4, h^6 aspheres on all three; T3 + everything.  Merit:\n');
    pr('  %g x rms spot (u, v), %g x image line off the slit, %g x end fields vs the slit ends, %g x departure from a linear\n', P.tel_w.blur, P.tel_w.v, P.tel_w.map, P.tel_w.ftheta);
    pr('  map, %g x the chief''s miss of the grating vertex when sent on through the Dyson (mm), %g x the best-focus offset\n', P.tel_w.pupil, P.tel_w.flat);
    pr('  along the slit normal per field (100 um; a FLAT field), %g x the clearance wall\n', P.tel_w.clear);
    pr('  (legs %.0f mm beyond the mount from every mirror, the fold, and the spectrometer''s bodies).  Each rung is then\n', P.tel_clear_m*1e3);
    pr('  emitted (collimated source, apertures from footprints + %.0f mm) and scored in the ENGINE at the slit over %d fields\n', P.ap_margin_m*1e3, P.tel_score_nfield);
    pr('  (telescope_score; the engine == chain identity is the gate tTelescopeRx), and the clearance gate is run on the\n');
    pr('  COMBINED chain with the R4 spectrometer''s bodies (spectrometer_clearance).  Units: px of %.0f um; walk = the chief''s\n', P.pixel_m*1e6);
    pr('  landing on the grating from its vertex, mm; flatness = p-v of the best-focus offset along the slit normal, um.\n\n');
    % the apparent pupil + the seed
    G0 = telescope_geom(struct('f', f, 'D', D, 'fov', fov, 'R', [0.3 0.1 0.3], 't', [0.1 0.1 0.1]), GD);
    Lapp = G0.pupil.L_app;
    Sd = telescope_seed(f, D, Lapp, P.tel_t1_m, P.tel_y2);
    assert(Sd.ok, 'dyson5 t1: the first-order seed does not close (t1 %.3f, y2 %.2f)', P.tel_t1_m, P.tel_y2);
    pr('SPECTROMETER PUPIL: the Dyson''s aim lines from the slit centre and ends cross %.2f m behind the slit (edge chief %.3f deg).\n', Lapp, G0.pupil.edge_angle_deg);
    pr('SEED (first order): R [%.1f %.1f %.1f] mm, t [%.1f %.1f %.1f] mm, powers [%.4f %.4f %.4f] /m (sum %.1e), f %.2f mm, y3 %.3f\n\n', ...
       Sd.R*1e3, Sd.t*1e3, Sd.phi, sum(Sd.phi), Sd.f*1e3, Sd.y(3));
    Pt0 = struct('f', f, 'D', D, 'fov', fov, 'bias', P.tel_bias_deg*pi/180, 'R', Sd.R, 't', Sd.t, 'Kc', [0 0 0], 'A', zeros(3, 2), ...
                 'dec', [0 0 0], 'tilt', P.tel_tilt_deg(:)'*pi/180, 'fold_dir', P.tel_fold_dir(:), 'lambda_c', GD.src.lambda_c, ...
                 'D_src', D*P.tel_oversize, 'name', [P.tag '_tel'], 'fold_gap', P.tel_fold_gap_m);
    L = telescope_ladder(Pt0, GD, P, 'rungs', P.tel_rungs, 'nfield', P.tel_nfield, 'nring', P.tel_nring, ...
                         'w_blur', P.tel_w.blur, 'w_v', P.tel_w.v, 'w_map', P.tel_w.map, 'w_ftheta', P.tel_w.ftheta, ...
                         'w_pupil', P.tel_w.pupil, 'w_flat', P.tel_w.flat, 'w_clear', P.tel_w.clear, 'clear_m', P.tel_clear_m, ...
                         'max_iter', P.tel_max_iter, 'fold_gap', P.tel_fold_gap_m, 'quiet', false);
    pr('%-4s %8s %7s %6s %6s %7s %7s %7s %8s %8s %8s %6s  %s\n', 'rung', 'merit', 'spot', 'ee1', 'slit', 'telec', 'pupil', 'walk', 'flat', 'ends', 'IFOV', 'clear', 'on bounds');
    pr('%-4s %8s %7s %6s %6s %7s %7s %7s %8s %8s %8s %6s\n', '', '', 'px', 'min', 'min', 'deg', 'deg', 'mm', 'um p-v', 'px', 'ratio', 'mm');
    tags = {};
    for k = 1:numel(L.rung)
        r = L.rung(k);  h = r.chain.headline;
        pr('%-4s %8.3g %7.3f %6.3f %6.3f %7.3f %7.3f %7.2f %8.1f %8.2f %8.3f %6.2f  %s\n', [r.name ' chain'], r.merit, h.s_max_px, h.ee1_min, h.slit_min, h.tel_max_deg, ...
           h.err_max_deg, h.walk_max_mm, h.flat_pv_um, max(abs(h.end_err_px)), max(abs(h.ifov_ratio_range - 1)) + 1, r.cmin_mm, strjoin(r.on_bounds, ' '));
        % emit + engine score + full clearance (with the R4 spectrometer)
        GT = telescope_geom(r.P, GD);
        F = GT.footprints('nx', 5, 'nlam', 1, 'nring', P.tel_nring);
        d0 = GT.field_dir(0);  [p0, ok] = GT.aim_pt(d0, GT.src.lambda_c);  assert(ok);
        file = sprintf('%s_t1_%s.in', tag, lower(r.name));
        M = spectrometer_rx(GT, file, 'ngridpts', P.ngridpts, 'name', sprintf('%s_tel_%s', P.tag, r.name), 'apertures', true, 'margin', P.ap_margin_m, ...
                            'footprints', F, 'source', struct('dir', d0, 'pos', p0, 'aperture', GT.src.D_src), 'wavelen', GT.src.lambda_c);
        M.iStop = GT.iStop;
        macos.load_rx(file);
        assert(macos.num_elt() == M.nElt, 'dyson5 t1: %s loads %d of %d elements', file, macos.num_elt(), M.nElt);
        Re = telescope_score(GT, M, P, 'nfield', P.tel_score_nfield, 'quiet', true);
        he = Re.headline;
        % identity: the engine's CHIEF at the slit vs the chain's, centre field
        % (the gate tTelescopeRx checks every ray; this is the record's one line)
        d0c = GT.field_dir(0);  [p0c, ~] = GT.aim_pt(d0c, GT.src.lambda_c);
        [pcc, ~, ~] = GT.trace(p0c, d0c, GT.src.lambda_c);
        macos.stop(M.iStop);  macos.set_src_fov('src_pos', p0c, 'src_dir', d0c, 'zSrc', 1e22);  macos.modify();
        sc = macos.trace(M.nElt);  ric = macos.get_ray_info(sc.nRays);
        ident = norm(ric.pos(:, 1) - pcc(:, end));
        GE = e2e_geom(GT, GD);
        Cl = spectrometer_clearance(GE, P, 'quiet', true);
        pr('%-4s %8s %7.3f %6.3f %6.3f %7.3f %7.3f %7.2f %8.1f %8.2f %8.3f %6.2f  %s (%s vs %s); engine chief vs chain %.1e m; %d rays/field; deck %s\n', ...
           [r.name ' engine'], '', he.s_max_px, he.ee1_min, he.slit_min, he.tel_max_deg, he.err_max_deg, he.walk_max_mm, he.flat_pv_um, max(abs(he.end_err_px)), ...
           max(abs(he.ifov_ratio_range - 1)) + 1, Cl.min_mm, tern_(Cl.pass, 'PASS', 'FAIL'), Cl.table.leg{1}, Cl.table.body{1}, ident, he.nrays_min, file);
        L.rung(k).engine = Re;  L.rung(k).clearance = Cl;  L.rung(k).file = file;  L.rung(k).M = M;
        telescope_maps_fig(Re, sprintf('%s_t1_maps_%s.png', tag, lower(r.name)), 'title', sprintf('telescope %s, engine, at the slit', r.name), 'pixel_um', P.pixel_m*1e6);
        tags{end+1} = regexprep(file, {'^.*/', '\.in$'}, '');   %#ok<AGROW>
    end
    r = L.rung(end);  Re = r.engine;
    pr('\nTELESCOPE OF RECORD (%s): R [%.2f %.2f %.2f] mm, spacings [%.2f %.2f %.2f] mm, bias %.3f deg, conics %s,\n', r.name, r.P.R*1e3, r.P.t*1e3, r.P.bias*180/pi, mat2str(r.P.Kc, 5));
    pr('  aspheres h^4 %s /m^3, h^6 %s /m^5; fold angles %s deg, M2/M3 decentre %s mm; the flat fold %.1f mm before the slit; EFL by the map %.2f mm\n', ...
       mat2str(r.P.A(:,1)', 4), mat2str(r.P.A(:,2)', 4), mat2str(r.P.tilt*180/pi, 4), mat2str(r.P.dec(2:3)*1e3, 3), r.P.fold_gap*1e3, Re.efl_fit_m*1e3);
    pr('  per field (deg): %s\n  spot px:  %s\n  slit:     %s\n  walk mm:  %s\n  focus um: %s\n', sprintf('%+.1f ', Re.fields*180/pi), sprintf('%.3f ', Re.S), sprintf('%.3f ', Re.SLIT), sprintf('%.2f ', Re.walk_m*1e3), sprintf('%.0f ', Re.zbf_m*1e6));
    pr('  CLEARANCE with R4 (legs vs bodies not traversed, mount %.0f mm): min %+.2f mm %s\n', P.mount_margin_m*1e3, r.clearance.min_mm, tern_(r.clearance.pass, 'PASS', 'FAIL'));
    for i = 1:min(6, height(r.clearance.table)), pr('    %-34s vs %-16s %+9.2f mm\n', r.clearance.table.leg{i}, r.clearance.table.body{i}, r.clearance.table.clearance_mm(i)); end
    fclose(fid);
    S = L;  S.seed = Sd;  S.Lapp = Lapp;  S.record = r;
    save([tag '_t1.mat'], 'S', 'P');
    dyson5_view_figs(tags(end), P.outdir);
    fprintf('dyson5 t1: wrote %s_t1.{txt,mat}, %s_t1_t*.in, %s_t1_maps_*.png\n', tag, tag, tag);
end

function S = stage_t3_(P, tag)
%STAGE_T3_  Beat 5b: the telescope through the offset_imager ladder (oi_story), one run per (envelope t1, offset) case.
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', 'templates', '10_telescopes', 'offset_imager'));
    GD = dyson_of_record_(P, tag, 'R4');
    ifov = P.tel_gsd_m/P.tel_alt_m;  f = P.pixel_m/ifov;  D = f/P.Fno;  fov = P.npix(1)*ifov;
    G0 = telescope_geom(struct('f', f, 'D', D, 'fov', fov, 'R', [0.3 0.1 0.3], 't', [0.1 0.1 0.1]), GD);
    Lapp = G0.pupil.L_app;
    odir = fullfile(P.outdir, 't3');  if ~exist(odir, 'dir'), mkdir(odir); end
    [~, tb] = fileparts(tag);
    box = [fov*180/pi, P.tel3_box_al_deg];
    sn = {'s1', 's2', 's3', 's4', 's5'};
    rows = struct('t1', {}, 'off', {}, 'y2', {}, 'seed', {}, 'ok', {}, 'msg', {}, 'map', {}, 'clear_mm', {}, 'worst', {}, ...
                  'exit_err', {}, 'diam_mm', {}, 'bbox_mm', {}, 'sep_mm', {}, 'X3', {});
    for c = 1:size(P.tel3_cases, 1)
        t1 = P.tel3_cases(c, 1);  off = P.tel3_cases(c, 2);  y2 = P.tel_y2;
        otag = sprintf('%s_t3_t%03d_off%02d', tb, round(t1*1e3), round(off));
        if size(P.tel3_cases, 2) >= 3                    % [t1 off y2]: a screened row (addendum 21)
            y2 = P.tel3_cases(c, 3);
            if abs(y2 - P.tel_y2) > 1e-12, otag = sprintf('%s_y%02d', otag, round(100*y2)); end
        end
        fsum = fullfile(odir, [otag '_sum.mat']);
        if P.tel3_reuse && isfile(fsum)
            R = load(fsum);
            if ~isfield(R.row, 'y2'), R.row.y2 = P.tel_y2; end   % rows recorded before the y2 column
            rows(end+1) = orderfields(R.row, rows);  %#ok<AGROW>
            fprintf('dyson5 t3: reused %s\n', fsum);  continue
        end
        Sd = telescope_seed(f, D, Lapp, t1, y2);
        row = struct('t1', t1, 'off', off, 'y2', y2, 'seed', Sd, 'ok', false, 'msg', '', 'map', nan(1, 5), 'clear_mm', nan(1, 5), ...
                     'worst', {repmat({''}, 1, 5)}, 'exit_err', nan(1, 5), 'diam_mm', nan(1, 3), 'bbox_mm', nan(1, 3), ...
                     'sep_mm', tand(off)*t1*1e3, 'X3', []);
        if ~Sd.ok
            row.msg = sprintf('first-order seed does not close at t1 %.3f m, y2 %.2f', t1, y2);
            save(fsum, 'row');  rows(end+1) = orderfields(row, rows);  continue  %#ok<AGROW>
        end
        over = struct('name', otag, 'tag', otag, 'outdir', odir, 'EPD_m', D, 'Fno', P.Fno, 'lambda_m', P.tel3_lambda_m, ...
                      'box_deg', box, 'offset_deg', off, 'nsolve', P.tel3_nsolve, 'nsolve_s5', P.tel3_nsolve_s5, ...
                      'z_m1_m', P.tel3_z_m1_m, 'spacings_m', [-Sd.t(1) 0 Sd.t(2)], 'seed_R1_m', -Sd.R(1), 'seed_R_m', -Sd.R, ...
                      'clear_m', P.tel3_clear_m, 'exit_dir', P.tel3_exit_dir, 'model', P.tel3_model, ...
                      'sampling', P.tel3_sampling, 'gn_iters', P.tel3_gn_iters, 'stages', P.tel3_stages);
        try
            O = offset_imager(over);       % the ladder only: oi_story's counter (a) is an S5-class solve, out of addendum 20's S1-S3 scope
            row.ok = true;
            for k = 1:numel(O.ladder)
                j = find(strcmp(sn, O.ladder(k).stage));  st = O.(O.ladder(k).stage);  g = st.gates;
                row.map(j) = O.ladder(k).map_max_nm;
                if isfield(st.map, 'valid') && ~st.map.valid, row.map(j) = Inf; end   % every field lost: INVALID, not finite
                row.clear_mm(j) = g.clear_min_m*1e3;  row.exit_err(j) = g.exit_err_deg;
                [~, iw] = min([g.clear_table.min_m]);  row.worst{j} = g.clear_table(iw).leg;
            end
            last = O.(O.ladder(end).stage);  row.X3 = last.X;
            [row.diam_mm, row.bbox_mm] = t3_size_(last.X, last.G, O.P, off);
        catch err
            row.msg = err.message;
            fprintf(2, 'dyson5 t3: case t1 %.0f mm, offset %g deg FAILED: %s\n', t1*1e3, off, err.message);
        end
        save(fsum, 'row');  rows(end+1) = orderfields(row, rows);  %#ok<AGROW>
    end
    fid = fopen([tag '_t3.txt'], 'w');  pr = @(varargin) dualprint_(fid, varargin{:});
    pr('dyson5 t3 -- the telescope through the offset_imager ladder: ENVELOPE x OFFSET scan (addendum 20) (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: templates/10_telescopes/offset_imager (offset_imager ladder, stages %s) called with EPD %.1f mm, F/%.1f (EFL %.1f mm an\n', mat2str(P.tel3_stages), D*1e3, P.Fno, f*1e3);
    pr('  identity), box %.2f deg cross-track (the slit) x %.2f deg along-track centred OFF deg along-track.  Envelope: t1 (M1 -> M2\n', box);
    pr('  = M1 -> stop) through telescope_seed (EFL %.0f mm, Petzval 0, exit pupil %.2f m behind the slit, y2 = %.2f), mapped to the\n', f*1e3, Lapp, P.tel_y2);
    pr('  template as R1 = -|R1|, spacings [-t1 0 +t2], stop at M2; R2/R3 re-solved by the template from EFL + Petzval 0.\n');
    pr('  Metric = the template''s: strict RMS WFE at %.2f um, centroid reference, exit-pupil anchor, piston only; headline =\n', P.tel3_lambda_m*1e6);
    pr('  dense 11x11 map MAXIMUM (nm).  Clearance = oi_clear min over the nine beam-leg x mirror pairs (disk model: footprint\n');
    pr('  centre + 1.15x radius per field), WORST PAIR NAMED as "leg x obstacle"; packages = floor >= %.0f mm (addendum 20).\n', P.tel3_pack_m*1e3);
    pr('  walk = t1 tan(OFF) (addendum 20 asks >= 80 mm).  Size from the last stage''s rays at the box centre + corners: mirror\n');
    pr('  diameter = max pairwise distance of the footprint points (no mount margin); envelope = bounding box of the M1, M2, M3\n');
    pr('  footprints and the focal-plane spot, mm (template frame: z = the entering beam).\n\n');
    for r = rows
        pr('t1 %3.0f mm  y2 %.2f  OFF %4.1f deg  walk %5.1f mm  seed R [%.0f %.1f %.1f] t [%.0f %.1f %.1f] mm\n', r.t1*1e3, r.y2, r.off, r.sep_mm, r.seed.R*1e3, r.seed.t*1e3);
        if ~r.ok, pr('   FAILED: %s\n\n', r.msg);  continue, end
        for j = find(~isnan(r.map))
            if isinf(r.map(j)), ms = '  INVALID'; else, ms = sprintf('%9.1f', r.map(j)); end
            pr('   %s  map max %s nm   clearance %+7.1f mm  worst %-14s  exit err %.3f deg\n', upper(sn{j}), ms, r.clear_mm(j), r.worst{j}, r.exit_err(j));
        end
        bb = nan(1, 3);  bb(1:numel(r.bbox_mm)) = r.bbox_mm;
        pr('   mirror diameters M1/M2/M3 %.0f / %.0f / %.0f mm (largest %.0f);  envelope %.0f x %.0f x %.0f mm (x y z)\n', r.diam_mm, max(r.diam_mm), bb);
        if r.map(1) > P.tel3_s1_conv_nm, pr('   S1 NOT CONVERGED (%.0f nm > %.0f): no verdict on image OR clearance (addendum 22)\n', r.map(1), P.tel3_s1_conv_nm); end
        pr('\n');
    end
    pr('%7s %4s %5s %6s | %9s %9s %-14s | %7s %9s | %s\n', 't1 mm', 'y2', 'OFF', 'walk', 'last nm', 'clr mm', 'worst pair', 'max D', 'max dim', 'verdict (last stage reached)');
    for r = rows
        if ~r.ok, pr('%7.0f %4.2f %5.1f %6.1f | FAILED -- NO VERDICT (the run ended before S3 scored)\n', r.t1*1e3, r.y2, r.off, r.sep_mm);  continue, end
        j = find(~isnan(r.map), 1, 'last');
        if j < 3 || isinf(r.map(j)), v = 'S3 not reached / INVALID';
        else, v = tern_(r.clear_mm(j) >= P.tel3_pack_m*1e3, tern_(r.map(j) < 250, 'PACKAGES, images', 'packages, image > 250 nm'), 'does not package');
        end
        if r.map(1) > P.tel3_s1_conv_nm, v = [v ' -- NO VERDICT (S1 not converged)']; end
        bb = nan(1, 3);  bb(1:numel(r.bbox_mm)) = r.bbox_mm;
        pr('%7.0f %4.2f %5.1f %6.1f | %9.4g %+9.1f %-14s | %7.0f %9.0f | %s\n', r.t1*1e3, r.y2, r.off, r.sep_mm, r.map(j), r.clear_mm(j), r.worst{j}, max(r.diam_mm), max(bb), v);
    end
    pr('\nPer-case runs: %s/%s_t3_t<mm>_off<deg>_{REPORT,STORY}.md, decks, figures.\n', odir, tb);
    fclose(fid);
    S = struct('rows', rows, 'box', box);
    save([tag '_t3.mat'], 'S');
end

function S = stage_t3s_(P, tag)
%STAGE_T3S_  Addendum 21: the first-order nine-pair clearance screen over telescope_seed's family (t1 x y2 x offset).
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', 'templates', '10_telescopes', 'offset_imager'));
    GD = dyson_of_record_(P, tag, 'R4');
    ifov = P.tel_gsd_m/P.tel_alt_m;  f = P.pixel_m/ifov;  D = f/P.Fno;  fov = P.npix(1)*ifov;
    G0 = telescope_geom(struct('f', f, 'D', D, 'fov', fov, 'R', [0.3 0.1 0.3], 't', [0.1 0.1 0.1]), GD);
    Lapp = G0.pupil.L_app;  by = P.tel3_box_al_deg/2;  xh = fov*90/pi;
    fid = fopen([tag '_t3s.txt'], 'w');  pr = @(varargin) dualprint_(fid, varargin{:});
    pr('dyson5 t3s -- the FIRST-ORDER clearance screen of the telecentric three-mirror family (addendum 21) (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: tma_screen (design/src): OI_CLEAR''s nine leg x obstacle pairs (the offset_imager gate''s), evaluated\n');
    pr('  PARAXIALLY -- chief through M2''s vertex (the stop) + the axial marginal, meridional rays, the box centre and the\n');
    pr('  two along-track extremes (+-%.2f deg) at zero cross-track angle; glass = one disk per field, footprint centre +\n', by);
    pr('  1.15 x the marginal height, in the element''s plane (FP normal to the axis); a leg crossing a disk is negative by\n');
    pr('  its in-plane depth.  Layout = telescope_seed''s family at f %.0f mm, D %.0f mm, Petzval 0, exit pupil %.2f m behind\n', f*1e3, D*1e3, Lapp);
    pr('  the slit: knobs t1 (M1 -> stop), y2 (compression at M2; t2 = f y2 and the back focus follow), offset.  Mirror\n');
    pr('  diameters = footprint extent incl. the cross-track +-%.2f deg (max of x, y); length = z extent of M1..FP; height =\n', xh);
    pr('  y extent.  PASS = all nine >= +5 mm.  First order only: sag, aberrated footprints, conics ignored.\n\n');
    % 1. validation against the engine gate on the template's own seeds
    pr('VALIDATION -- screen vs the engine oi_clear on the template''s seed (spheres, R2/R3 from EFL + Petzval 0, its BFD), mm:\n');
    macos.init(P.tel3_model);
    val = struct('t1', {}, 'off', {}, 'eng', {}, 'scr', {}, 'stop_y', {});
    for c = 1:size(P.tel3s_validate, 1)
        t1 = P.tel3s_validate(c, 1);  off = P.tel3s_validate(c, 2);
        Sd = telescope_seed(f, D, Lapp, t1, P.tel_y2);
        Pt = offset_imager_params(struct('EPD_m', D, 'Fno', P.Fno, 'box_deg', [fov*180/pi, P.tel3_box_al_deg], 'offset_deg', off, ...
             'z_m1_m', P.tel3_z_m1_m, 'spacings_m', [-Sd.t(1) 0 Sd.t(2)], 'seed_R1_m', -Sd.R(1), 'clear_m', P.tel3_clear_m, 'model', P.tel3_model));
        X = oi_seed(Pt);  [X, G, fo] = oi_close(X, Pt, 'offset_deg', off);  X.fpa = oi_apply_fpa(X);  G.fpa = X.fpa;
        [~, dv] = oi_clear(X, G, Pt, off);
        Sc = tma_screen([2/abs(X.R(1)) -2/abs(X.R(2)) 2/abs(X.R(3))], [abs(X.spacings(1)) X.spacings(3)], -fo.BFD_m, D, off, 'by_deg', by);
        val(end+1) = struct('t1', t1, 'off', off, 'eng', dv(:)', 'scr', Sc.d(:)', 'stop_y', X.stopC(2));  %#ok<AGROW>
        [~, ie] = min(dv);
        pr('  t1 %3.0f mm off %2.0f deg: floor engine %+6.1f (%s)  screen %+6.1f (%s);  max |diff| %.1f mm; stop y %.1f mm\n', ...
           t1*1e3, off, min(dv)*1e3, Sc.pairs{ie}, Sc.dmin*1e3, Sc.worst, max(abs(dv(:) - Sc.d(:)))*1e3, X.stopC(2)*1e3);
    end
    % 2. the scan
    T1 = P.tel3s_t1_m;  Y2 = P.tel3s_y2;  OF = P.tel3s_off_deg;
    rows = struct('t1', {}, 'y2', {}, 'off', {}, 'd', {}, 'dmin', {}, 'worst', {}, 't2', {}, 'bfd', {}, 'diam', {}, 'len', {}, 'hgt', {}, 'R', {});
    nbad = 0;
    for t1 = T1
        for y2 = Y2
            Sd = telescope_seed(f, D, Lapp, t1, y2);
            if ~Sd.ok, nbad = nbad + 1;  continue, end
            for off = OF
                Sc = tma_screen(Sd.phi, Sd.t(1:2), Sd.t3, D, off, 'by_deg', by, 'xhalf_deg', xh);
                rows(end+1) = struct('t1', t1, 'y2', y2, 'off', off, 'd', Sc.d', 'dmin', Sc.dmin, 'worst', Sc.worst, 't2', Sd.t(2), ...
                                     'bfd', Sd.t3, 'diam', Sc.diam, 'len', Sc.len, 'hgt', Sc.hgt, 'R', Sd.R);  %#ok<AGROW>
            end
        end
    end
    pr('\nSCAN: t1 %s mm x y2 %s x offset %s deg -> %d rows (%d (t1, y2) seeds do not close)\n', mat2str(T1*1e3), mat2str(Y2), mat2str(OF), numel(rows), nbad);
    dmin = [rows.dmin];  pass = dmin >= P.tel3_pack_m;  offs = [rows.off];
    pr('rows passing all nine >= %.0f mm: %d of %d; at offset <= %.0f deg: %d\n', P.tel3_pack_m*1e3, nnz(pass), numel(rows), P.tel3s_off_max, nnz(pass & offs <= P.tel3s_off_max));
    hdr = sprintf('%6s %4s %4s | %s | %7s %-12s | %5s %5s | %5s %5s %5s | %5s %5s', 't1', 'y2', 'off', ...
          strjoin(cellfun(@(s) sprintf('%6s', s), {'iM1xM2','iM1xM3','iM1xFP','12xM3','12xFP','23xM1','23xFP','3FxM1','3FxM2'}, 'UniformOutput', false), ' '), ...
          'floor', 'worst pair', 't2', 'BFD', 'D1', 'D2', 'D3', 'len', 'hgt');
    prow = @(r) pr('%6.0f %4.1f %4.0f | %s | %+7.1f %-12s | %5.0f %5.0f | %5.0f %5.0f %5.0f | %5.0f %5.0f\n', r.t1*1e3, r.y2, r.off, ...
          sprintf('%+6.1f ', r.d*1e3), r.dmin*1e3, r.worst, r.t2*1e3, r.bfd*1e3, r.diam*1e3, r.len*1e3, r.hgt*1e3);
    pr('\nBEST ROW PER OFFSET (max floor over t1 x y2), mm:\n%s\n', hdr);
    for off = OF
        k = find(offs == off);  [~, b] = max(dmin(k));  prow(rows(k(b)));
    end
    pr('\nBEST ROW PER t1 AT OFFSET <= %.0f deg:\n%s\n', P.tel3s_off_max, hdr);
    for t1 = T1
        k = find([rows.t1] == t1 & offs <= P.tel3s_off_max);  if isempty(k), continue, end
        [~, b] = max(dmin(k));  prow(rows(k(b)));
    end
    % the binding pair at the best row of each offset, and which pair binds most often
    w = {rows.worst};  [u, ~, iu] = unique(w);  cnt = accumarray(iu(:), 1);
    pr('\nbinding (worst) pair over all rows: %s\n', strjoin(arrayfun(@(i) sprintf('%s %d', u{i}, cnt(i)), 1:numel(u), 'UniformOutput', false), ', '));
    if any(pass)
        pr('\nPASSING ROWS (all nine >= %.0f mm):\n%s\n', P.tel3_pack_m*1e3, hdr);
        for r = rows(pass), prow(r); end
    end
    fclose(fid);
    S = struct('rows', rows, 'val', val, 'f', f, 'D', D, 'Lapp', Lapp);
    save([tag '_t3s.mat'], 'S');
    % full table as CSV beside it
    fc = fopen([tag '_t3s.csv'], 'w');
    fprintf(fc, 't1_mm,y2,off_deg,%s,floor_mm,worst,t2_mm,bfd_mm,D1_mm,D2_mm,D3_mm,len_mm,hgt_mm\n', ...
            strjoin(strrep(strrep(tma_screen_pairs_(), ' x ', '_x_'), '->', '_'), ','));
    for r = rows
        fprintf(fc, '%.0f,%.2f,%.1f,%s,%.2f,%s,%.2f,%.2f,%.1f,%.1f,%.1f,%.1f,%.1f\n', r.t1*1e3, r.y2, r.off, ...
                strjoin(arrayfun(@(v) sprintf('%.2f', v), r.d*1e3, 'UniformOutput', false), ','), r.dmin*1e3, r.worst, ...
                r.t2*1e3, r.bfd*1e3, r.diam*1e3, r.len*1e3, r.hgt*1e3);
    end
    fclose(fc);
end

function S = stage_t3w_(P, tag)
%STAGE_T3W_  Addendum 23: the y2 continuation of the S1 parent at t1 fixed, then S3 (/ S4) at the offset from it, with the hard stop.
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', 'templates', '10_telescopes', 'offset_imager'));
    GD = dyson_of_record_(P, tag, 'R4');
    ifov = P.tel_gsd_m/P.tel_alt_m;  f = P.pixel_m/ifov;  D = f/P.Fno;  fov = P.npix(1)*ifov;
    G0 = telescope_geom(struct('f', f, 'D', D, 'fov', fov, 'R', [0.3 0.1 0.3], 't', [0.1 0.1 0.1]), GD);
    Lapp = G0.pupil.L_app;  t1 = P.tel3w_t1_m;  off = P.tel3w_off_deg;  it = P.tel3w_iters;
    odir = fullfile(P.outdir, 't3');  if ~exist(odir, 'dir'), mkdir(odir); end
    [~, tb] = fileparts(tag);  wtag = fullfile(odir, sprintf('%s_t3w', tb));
    box = [fov*180/pi, P.tel3_box_al_deg];
    mkP = @(Sd, name) offset_imager_params(struct('name', name, 'tag', name, 'outdir', odir, 'EPD_m', D, 'Fno', P.Fno, ...
              'lambda_m', P.tel3_lambda_m, 'box_deg', box, 'offset_deg', off, 'nsolve', P.tel3w_nsolve, 'z_m1_m', P.tel3_z_m1_m, ...
              'spacings_m', [-Sd.t(1) 0 Sd.t(2)], 'seed_R1_m', -Sd.R(1), 'seed_R_m', -Sd.R, 'clear_m', P.tel3_clear_m, ...
              'exit_dir', P.tel3_exit_dir, 'model', P.tel3_model, 'sampling', P.tel3_sampling, 'gn_iters', it, 'hold_R1', P.tel3w_hold_R1));
    macos.init(P.tel3_model);
    fid = fopen([tag '_t3w.txt'], 'w');  pr = @(varargin) dualprint_(fid, varargin{:});
    pr('dyson5 t3w -- the y2 continuation of the three-mirror parent, then the offset solve (addendum 23) (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('NOTE: a first run with R1 FREE was stopped after step y2 0.600: S1 reached 79.0 nm only by walking R1 0.700 -> 2.615 m\n');
    pr('  with K1 -141 (40 iterations, capped) -- an effective y2 of ~0.89, not the screened family; carried onto the next step\n');
    pr('  it started at 8.7 mm.  This run holds R1.\n');
    pr('CONVENTIONS: the offset_imager template''s S1 (on-axis box, symmetric conics + h^4/h^6/h^8 aspheres, R2/R3 eliminated by\n');
    pr('  EFL %.0f mm + Petzval 0 on the branch held by seed_R_m) walked in y2 at t1 = %.0f mm: each step''s spacings and R1 are\n', f*1e3, t1*1e3);
    pr('  telescope_seed''s family point (exit pupil %.2f m behind the slit), its conics / aspheres / FPA refit CARRIED from the\n', Lapp);
    pr('  previous solved step; R1 %s in S1 / S3 (y2 sets M1''s power: phi1 = (1 - y2)/t1), so every step IS its family point;\n', tern_(P.tel3w_hold_R1, 'HELD', 'free'));
    pr('  S4 frees the radii (the template''s S4).  Solve set %s (across the slit x along the %.1f deg strip); every solve runs to oi_solve''s own\n', mat2str(P.tel3w_nsolve), P.tel3_box_al_deg);
    pr('  stop (a rejected step, or < 0.1 %% gain) with a cap of %d -- "CAPPED" if it ends at the cap.  A step COUNTS at S1\n', it);
    pr('  dense-map max <= %.0f nm; a failed step is halved once, then the walk ends.  Metric = the template''s (strict RMS WFE at\n', P.tel3_s1_conv_nm);
    pr('  %.2f um, 11 x 11 dense-map max over the 24.6 x 0.3 deg box).  Per step, at the %g deg offset (stop posed there): M3''s\n', P.tel3_lambda_m*1e6, off);
    pr('  footprint reach rho/|R3| at the box corners, the fraction of launched rays reaching the FP at the +-12.3 deg cross-track\n');
    pr('  edges, and the exit-chief error vs %s.  Then S3 at %g deg seeded from the last counted step, accepted when the gate\n', mat2str(P.tel3_exit_dir), off);
    pr('  (oi_clear, disk model) reads >= %.0f mm afterwards, else S4 (tilts/decenters + the clearance hinge) from it.\n\n', P.tel3_pack_m*1e3);
    % ---- the walk ------------------------------------------------------------
    y2s = P.tel3w_y2;  steps = struct('y2', {}, 'ok', {}, 'map', {}, 'avg', {}, 'iters', {}, 'capped', {}, 'R', {}, 'K', {}, ...
                                      'asph', {}, 'diag', {}, 'X', {}, 'deck', {}, 'screen', {});
    Xprev = [];  k = 1;  halved = false;  y2prev = NaN;
    pr('%5s %10s %10s %6s %-24s %-24s | %7s %9s %8s %7s\n', 'y2', 'S1 max nm', 'avg nm', 'iters', 'R1 R2 R3 (mm)', 'K1 K2 K3', 'rho/R3', 'edge kept', 'exit err', 'scr flr');
    while k <= numel(y2s)
        y2 = y2s(k);
        Sd = telescope_seed(f, D, Lapp, t1, y2);
        name = sprintf('%s_t3w_y%03d', tb, round(1000*y2));
        Pw = mkP(Sd, name);
        X = oi_seed(Pw);
        if ~isempty(Xprev)                          % warm start: the solved shape carried onto the new family point
            X.K = Xprev.K;  X.asph = Xprev.asph;  X.fpa_refit = Xprev.fpa_refit;
        end
        X.eliminate = 'R2R3';
        [X, h] = oi_solve(X, Pw, 'S1', 'offset', 0, 'iters', it);
        [X, G] = oi_close(X, Pw, 'offset_deg', 0);  X.fpa = oi_apply_fpa(X);  G.fpa = X.fpa;
        [~, mp] = oi_map_fig(X, G, Pw, 0, sprintf('t3w S1 y2 %.3f (on axis)', y2), fullfile(odir, [name '_s1_map.png']));
        dg = t3w_diag_(X, Pw, off);
        sc = tma_screen([2/abs(X.R(1)) -2/abs(X.R(2)) 2/abs(X.R(3))], [Sd.t(1) Sd.t(2)], Sd.t3, D, off, 'by_deg', P.tel3_box_al_deg/2);
        deck = fullfile(odir, [name '_s1.in']);  t3w_write_deck_(X, Pw, deck);
        okk = mp.max_nm <= P.tel3_s1_conv_nm && (~isfield(mp, 'valid') || mp.valid);
        st = struct('y2', y2, 'ok', okk, 'map', mp.max_nm, 'avg', mp.avg_nm, 'iters', h.iters, 'capped', h.iters >= it, ...
                    'R', X.R, 'K', X.K, 'asph', X.asph, 'diag', dg, 'X', X, 'deck', deck, 'screen', sc.dmin);
        steps(end+1) = st;  %#ok<AGROW>
        pr('%5.3f %10.1f %10.1f %4d%s %-24s %-24s | %7.3f %9.3f %8.3f %+7.1f  %s\n', y2, mp.max_nm, mp.avg_nm, h.iters, tern_(st.capped, 'C', ' '), ...
           sprintf('%.1f %.2f %.2f', abs(X.R)*1e3), sprintf('%.3g %.3g %.3g', X.K), dg.rho_R3, dg.edge_kept, dg.exit_err, sc.dmin*1e3, tern_(okk, 'counts', 'DOES NOT COUNT'));
        if okk
            Xprev = X;  y2prev = y2;  halved = false;  k = k + 1;
        elseif ~halved && ~isnan(y2prev)
            mid = (y2prev + y2)/2;  y2s = [y2s(1:k-1), mid, y2s(k:end)];  halved = true;
            pr('      -> halving the step: y2 %.3f inserted\n', mid);
        else
            pr('      -> the walk ends (a halved step failed, or the first step)\n');  break
        end
    end
    % ---- the offset solve from the last counted step the screen passes ---------
    cnt = steps([steps.ok]);
    scr = P.tel3w_screen_pass_m;  if isnan(scr), scr = P.tel3_pack_m; end
    pass = cnt([cnt.screen] >= scr);
    S = struct('steps', steps, 'off', off);
    if isempty(pass)
        if isempty(cnt), lasty = NaN; else, lasty = cnt(end).y2; end
        pr('\nHARD STOP: no counted step passes the screen (last counted y2 %.3f) -- the walk did not reach the packaging corner.\n', lasty);
        S.stop = 'walk';  t3w_close_(fid, S, tag);  return
    end
    base = pass(end);  Sd = telescope_seed(f, D, Lapp, t1, base.y2);
    name = sprintf('%s_t3w_y%03d', tb, round(1000*base.y2));  Pw = mkP(Sd, name);
    X = base.X;  X.fpa_refit = [0 0];  X.eliminate = 'R2R3';
    X.stop_fixed = false;  [X, ~] = oi_close(X, Pw, 'offset_deg', off);  X.stop_fixed = true;     % pose the stop at the offset once, then free (the template's S3 entry)
    pr('\nS3 at %g deg from the counted step y2 %.3f (stop posed at y %.2f mm):\n', off, base.y2, X.stopC(2)*1e3);
    [X3, h3] = oi_solve(X, Pw, 'S3', 'iters', it);
    R3s = t3w_score_(X3, Pw, off, fullfile(odir, [name '_s3']), 'S3');
    pr('  S3: %d iters%s, map max %.1f nm avg %.1f, clearance %+.1f mm (%s), exit err %.3f deg, rho/R3 %.3f, edge kept %.3f\n', ...
       h3.iters, tern_(h3.iters >= it, ' CAPPED', ''), R3s.map, R3s.avg, R3s.clear_mm, R3s.worst, R3s.exit_err, R3s.diag.rho_R3, R3s.diag.edge_kept);
    S.s3 = R3s;  fin = R3s;
    if R3s.clear_mm < P.tel3_pack_m*1e3
        pr('  S3 does not hold the gate -> S4 (tilts/decenters + the clearance hinge) from it:\n');
        X4 = X3;  X4.eliminate = 'R3';
        [X4, h4] = oi_solve(X4, Pw, 'S4', 'iters', it, 'walls', @(a, b) false, 'clear', true);
        R4s = t3w_score_(X4, Pw, off, fullfile(odir, [name '_s4']), 'S4');
        pr('  S4: %d iters%s, map max %.1f nm avg %.1f, clearance %+.1f mm (%s), exit err %.3f deg, rho/R3 %.3f, edge kept %.3f\n', ...
           h4.iters, tern_(h4.iters >= it, ' CAPPED', ''), R4s.map, R4s.avg, R4s.clear_mm, R4s.worst, R4s.exit_err, R4s.diag.rho_R3, R4s.diag.edge_kept);
        S.s4 = R4s;  fin = R4s;
    end
    % ---- the hard stop -------------------------------------------------------------
    why = {};
    if fin.clear_mm >= P.tel3_pack_m*1e3 && fin.map > P.tel3w_img_max_nm
        why{end+1} = sprintf('the offset solve ends at %.1f nm > %.0f nm with the gate satisfied (%+.1f mm)', fin.map, P.tel3w_img_max_nm, fin.clear_mm); end
    if fin.clear_mm < P.tel3_pack_m*1e3
        why{end+1} = sprintf('the offset solve does not hold the gate (%+.1f mm, %s)', fin.clear_mm, fin.worst); end
    if 1 - fin.diag.edge_kept > P.tel3w_vig_max
        why{end+1} = sprintf('%.1f %% of rays lost at the cross-track edge (> %.0f %%)', 100*(1 - fin.diag.edge_kept), 100*P.tel3w_vig_max); end
    if base.y2 > 0.4 + 1e-9 && isempty(why)
        pr('  note: the offset solve ran from y2 %.3f, the last counted step the screen passes (the walk did not count y2 0.4)\n', base.y2); end
    if isempty(why)
        pr('\nNO HARD STOP FIRES: the three-mirror lives at y2 %.3f, %g deg -- next: the three residual rows, then S4 / S5.\n', base.y2, off);  S.stop = '';
    else
        pr('\nHARD STOP (beat 5c earned): %s.\n', strjoin(why, '; '));  S.stop = strjoin(why, '; ');
    end
    t3w_close_(fid, S, tag);
end

function t3w_close_(fid, S, tag)
    fclose(fid);
    for k = 1:numel(S.steps), S.steps(k).X = rmfield(S.steps(k).X, intersect(fieldnames(S.steps(k).X), {'cache'})); end
    save([tag '_t3w.mat'], 'S');
end

function dg = t3w_diag_(X, Pw, off)
%T3W_DIAG_  At the offset (stop posed there, as the template's S3 entry does): M3's footprint reach rho/|R3| at the box
%   corners, the fraction of launched rays reaching the FP at the +-12.3 deg cross-track edges, the exit-chief error.
    Xd = X;  Xd.fpa_refit = [0 0];  Xd.stop_fixed = false;
    dg = struct('rho_R3', NaN, 'edge_kept', 0, 'exit_err', NaN, 'stop_y_mm', NaN);
    try
        [Xd, G] = oi_close(Xd, Pw, 'offset_deg', off);  Xd.fpa = oi_apply_fpa(Xd);  G.fpa = Xd.fpa;
    catch
        return
    end
    dg.stop_y_mm = Xd.stopC(2)*1e3;
    [dg.rho_R3, dg.edge_kept] = t3w_edge_(Xd, G, Pw, off);
    try, g = oi_gates(Xd, G, Pw, off);  dg.exit_err = g.exit_err_deg; catch, end
end

function [rr, kept] = t3w_edge_(X, G, Pw, off)
    bx = Pw.box_deg(1)/2;  by = Pw.box_deg(2)/2;
    F = [-bx off-by; -bx off+by; bx off-by; bx off+by];
    Dk = X;  Dk.EPD_m = Pw.EPD_m;  Dk.WL_m = Pw.lambda_m;  Dk.sampling = Pw.sampling;  Dk.name = Pw.name;
    rr = NaN;  kept = 0;
    try, sc = oi_score(oi_deck(Dk), G, F, 'rays', true); catch, return, end
    nl = 0;  nk = 0;  r = 0;
    for q = 1:size(F, 1)
        E = sc.rays{q};
        if ~iscell(E) || numel(E) < 5, nl = nl + 1;  continue, end   % no state: count the field as lost
        n0 = numel(E{1}.ok);  nl = nl + n0;  nk = nk + nnz(E{5}.ok);
        p3 = E{4}.pos(:, E{4}.ok);
        if ~isempty(p3), r = max(r, max(hypot(p3(1, :), p3(2, :) - X.yde(3)))); end
    end
    kept = nk/max(nl, 1);  rr = r/abs(X.R(3));
end

function R = t3w_score_(X, Pw, off, stem, lbl)
    [X, G] = oi_close(X, Pw, 'offset_deg', off);  X.fpa = oi_apply_fpa(X);  G.fpa = X.fpa;
    [~, mp] = oi_map_fig(X, G, Pw, off, sprintf('t3w %s at %g deg', lbl, off), [stem '_map.png']);
    oi_layout_fig(X, G, Pw, off, sprintf('t3w %s at %g deg', lbl, off), [stem '_layout.png']);
    g = oi_gates(X, G, Pw, off);  [~, iw] = min([g.clear_table.min_m]);
    t3w_write_deck_(X, Pw, [stem '.in']);
    [rr, kept] = t3w_edge_(X, G, Pw, off);
    R = struct('X', X, 'map', mp.max_nm, 'avg', mp.avg_nm, 'valid', ~isfield(mp, 'valid') || mp.valid, 'clear_mm', g.clear_min_m*1e3, ...
               'worst', g.clear_table(iw).leg, 'exit_err', g.exit_err_deg, 'diag', struct('rho_R3', rr, 'edge_kept', kept), 'deck', [stem '.in']);
    if ~R.valid, R.map = Inf; end
end

function t3w_write_deck_(X, Pw, file)
    Dk = X;  Dk.EPD_m = Pw.EPD_m;  Dk.WL_m = Pw.lambda_m;  Dk.sampling = Pw.sampling;  Dk.name = Pw.name;
    fh = fopen(file, 'w');  fprintf(fh, '%s', oi_deck(Dk));  fclose(fh);
end

function p = tma_screen_pairs_()
    p = {'in->M1 x M2','in->M1 x M3','in->M1 x FP', 'M1->M2 x M3','M1->M2 x FP', 'M2->M3 x M1','M2->M3 x FP', 'M3->FP x M1','M3->FP x M2'};
end

function [diam, bbox] = t3_size_(X, G, Pt, off)
%T3_SIZE_  Mirror footprint diameters + the optics' bounding box from the template's own rays (box centre + corners).
    bx = Pt.box_deg(1)/2;  by = Pt.box_deg(2)/2;
    F = [0 off; -bx off-by; -bx off+by; bx off-by; bx off+by];
    D = X;  D.EPD_m = Pt.EPD_m;  D.WL_m = Pt.lambda_m;  D.sampling = Pt.sampling;  D.name = Pt.name;
    sc = oi_score(oi_deck(D), G, F, 'rays', true);
    iel = [1 3 4 5];  pts = cell(1, 4);           % M1, M2, M3, FP (the stop Reference is element 2)
    for q = 1:numel(sc.rays)
        E = sc.rays{q};  if ~iscell(E), continue, end
        for k = 1:4
            e = E{iel(k)};  pts{k} = [pts{k}, e.pos(:, e.ok)];
        end
    end
    diam = nan(1, 3);
    for k = 1:3
        p = pts{k};  if size(p, 2) < 2, continue, end
        p = p(:, 1:max(1, floor(size(p, 2)/2000)):end);
        dm = 0;
        for i = 1:size(p, 2), dm = max(dm, max(vecnorm(p - p(:, i)))); end
        diam(k) = dm*1e3;
    end
    A = [pts{:}];
    bbox = (max(A, [], 2) - min(A, [], 2))'*1e3;
end

function S = stage_t2_(P, tag)
%STAGE_T2_  End to end: the telescope of record into R4 and R5, one deck each, the spectrometer's scorer.
    fn = [tag '_t1.mat'];
    assert(isfile(fn), 'dyson5 t2 needs the telescope record %s (run t1 first)', fn);
    T1 = load(fn);  rt = T1.S.record;
    macos.init(P.model);
    fid = fopen([tag '_t2.txt'], 'w');  pr = @(varargin) dualprint_(fid, varargin{:});
    pr('dyson5 t2 -- telescope + spectrometer, end to end (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: the telescope of record (t1, rung %s) prepended to each spectrometer of record as ONE prescription\n', rt.name);
    pr('  (e2e_geom): sky -> M1 -> M2 -> M3 -> fold -> slit (a pass-through Reference) -> the spectrometer''s surfaces -> FPA.\n');
    pr('  COLLIMATED source of %.1f mm (%.2f x the 70 mm), the GRATING declared the STOP (macos.stop): per (field, lambda) the\n', rt.P.D_src*1e3, P.tel_oversize);
    pr('  chain''s own launch whose chief passes the grating vertex is written as the source; the telescope''s apertures are its\n');
    pr('  footprints + %.0f mm, the spectrometer''s its own record apertures, the grating''s its F/%.1f footprint + 0.2 mm (the stop).\n', P.ap_margin_m*1e3, P.Fno);
    pr('  Scored by spectrometer_score in the FPA frame (u along the slit, v along the dispersion, px): smile, keystone, SRF,\n');
    pr('  CRF, ensquared energy, and PASS = the fraction of the launched bundle the grating admits (the pupil match in energy)\n');
    pr('  over %d fields x %d wavelengths; the clearance gate on the combined chain; the engine renders.\n\n', P.e2e_nfield, P.e2e_nlam);
    pr('%-6s %8s %8s %7s %7s %6s %6s %8s  %s\n', 'deck', 'smile', 'keyst', 'CRF', 'SRF', 'EE', 'pass', 'clear', 'worst pair');
    tags = {};
    for q = 1:numel(P.e2e_rungs)
        rg = P.e2e_rungs{q};
        GD = dyson_of_record_(P, tag, rg);
        GT = telescope_geom(rt.P, GD);  GE = e2e_geom(GT, GD);
        nT = numel(GT.surf);
        Ft = GE.footprints('nx', 5, 'nlam', 3, 'nring', P.tel_nring);
        Fd = GD.footprints('nx', 3, 'nlam', 3, 'nring', 2);
        F = [Ft(1:nT), Fd];
        marg = [P.ap_margin_m*ones(1, nT), P.ap_margin_m*ones(1, numel(GD.surf))];  marg(GE.iG) = 0.2e-3;
        d0 = GE.field_dir(0);  [p0, ~, ok] = GE.launch_field(0, GE.src.lambda_c);  assert(ok);
        file = sprintf('%s_t2_%s.in', tag, lower(rg));
        M = spectrometer_rx(GE, file, 'ngridpts', P.ngridpts, 'name', sprintf('%s_e2e_%s', P.tag, rg), 'apertures', true, 'margin', marg, ...
                            'footprints', F, 'source', struct('dir', d0, 'pos', p0, 'aperture', GE.src.D_src), 'wavelen', GE.src.lambda_c);
        macos.load_rx(file);
        assert(macos.num_elt() == M.nElt, 'dyson5 t2: %s loads %d of %d elements', file, macos.num_elt(), M.nElt);
        fields = linspace(-GE.src.fov/2, GE.src.fov/2, P.e2e_nfield);
        Pk = P;  Pk.Fno = GD.P.Fno;
        R = spectrometer_score(GE, M, Pk, 'fields', fields, 'nlam', P.e2e_nlam, 'quiet', true);
        Cl = spectrometer_clearance(GE, P, 'quiet', true);
        pr('%-6s %8.4f %8.4f %7.3f %7.3f %6.3f %6.3f %+8.2f  %s vs %s (%s); %d elements; deck %s\n', rg, R.smile_max, R.keystone_max, R.crf_max, R.srf_max, R.ee_min, ...
           min(R.pass_frac(:)), Cl.min_mm, Cl.table.leg{1}, Cl.table.body{1}, tern_(Cl.pass, 'PASS', 'FAIL'), M.nElt, file);
        pr('        pass fraction per field: %s\n', sprintf('%.3f ', min(R.pass_frac, [], 2)));
        pr('        smile per lambda (px): %s\n        keystone per field (px): %s\n', sprintf('%.4f ', R.smile_px), sprintf('%.4f ', R.keystone_px));
        S.(rg) = struct('G', GE, 'M', M, 'score', R, 'clearance', Cl, 'file', file);
        spectrometer_maps_fig(R, sprintf('%s_t2_maps_%s.png', tag, lower(rg)), 'title', sprintf('telescope + %s, end to end, engine', rg), 'pixel_um', P.pixel_m*1e6);
        tags{end+1} = regexprep(file, {'^.*/', '\.in$'}, '');   %#ok<AGROW>
    end
    fclose(fid);
    save([tag '_t2.mat'], 'S', 'P');
    dyson5_view_figs(tags, P.outdir);
    fprintf('dyson5 t2: wrote %s_t2.{txt,mat}, %s_t2_*.in, %s_t2_maps_*.png\n', tag, tag, tag);
end

function GD = dyson_of_record_(P, tag, rg)
%DYSON_OF_RECORD_  The R4 (s3 record) or R5 (s5 record) chain, rebuilt from its parameter set.
    switch rg
        case 'R4'
            fn = [tag '_s3.mat'];  assert(isfile(fn), 'dyson5: the ladder record %s is needed (run s3)', fn);
            S3 = load(fn);  k = find(strncmp({S3.S.rung.name}, 'R4 ', 3), 1);  assert(~isempty(k));
            GD = spectrometer_geom('dyson', S3.S.rung(k).P);
        case 'R5'
            fn = [tag '_s5.mat'];  assert(isfile(fn), 'dyson5: the fold-prism record %s is needed (run s5)', fn);
            S5 = load(fn);  GD = spectrometer_geom('dyson', S5.S.rung.P);
        otherwise
            error('dyson5: unknown spectrometer of record %s', rg);
    end
end

function s = dyson5_vstr_(v)
    if isempty(v), s = 'corner'; elseif ischar(v) || isstring(v), s = char(v); elseif numel(v) > 1, s = mat2str(v, 4); else, s = sprintf('%.4g', v); end
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
