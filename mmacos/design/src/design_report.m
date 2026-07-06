function rep = design_report(t, opts)
%DESIGN_REPORT  One-page design report -- identity, first-order optical
%   properties, field performance, focal-plane geometry, exit pupil, and
%   packaging (Dave 2026-07-06: "emit a design report -- WFE, FOV, FP
%   tilt, fno at M1 and at FP, etc.").
%
%   rep = design_report(t) measures everything live on the built design
%   and prints the report; rep.text carries the formatted block for
%   saving alongside the example artifacts.
%
%   Name-value:
%     'rings_arcmin' WFE-ladder ring radii about the (biased) field
%                    center (default [0.25 0.5 1.0]); the center point
%                    is always the first row.
%     'align'        an align_focal_plane result struct: adds the
%                    defocus-removed / field-curvature-sag / best-focus
%                    blur lines (the FP tilt itself is measured live
%                    from the trace either way).
%     'yardstick_m'  extra wavelength for a second waves column
%                    (default [] = design wavelength only).
%     'dl_waves'     diffraction-limit threshold (default 0.071 waves
%                    RMS, Marechal).
%     'strehl'       add a Strehl column (default true; needs add_pupil
%                    so the OPD is exit-pupil-referenced).  Strehl =
%                    far-field intensity peak / unaberrated-aperture
%                    peak, computed exactly as the coherent sum over
%                    the de-tilted OPD; evaluated at the worst-WFE
%                    point of each ring.  (The EP's PropType=FarField
%                    additionally makes INT at the FP the PSF -- the
%                    hook for full PSF products.)
%     'file'         also write rep.text to this path.
%     'quiet'        suppress printing (default false).
%
%   Returns rep with: .D_m .lambda_m .EFL_m .fno_m1 .fno_fp
%   .plate_um_per_arcsec .lamD_mas .bias_arcmin .rings_arcmin
%   .wfe_raw/.wfe_tilt (per row, waves @ design lambda) .strehl (per
%   row; NaN rows when disabled/no pupil) .fp_tilt_deg .pupil (struct
%   or []) .shroud_over_D .train_length_m .clear_ok .obstructors
%   (names) .aoi (per powered mirror) .text
%
%   See also wfe_field_diag, packaging_report, aoi_report,
%   macos.design.Telescope/align_focal_plane.
    arguments
        t
        opts.rings_arcmin (1,:) double = [0.25 0.5 1.0]
        opts.align struct = struct([])
        opts.yardstick_m (1,:) double = []
        opts.dl_waves (1,1) double = 0.071
        opts.strehl (1,1) logical = true
        opts.file (1,:) char = ''
        opts.quiet (1,1) logical = false
    end
    e   = t.spec.elt;
    lam = t.spec.wavelength;
    D   = t.spec.in.D;
    L   = {};   % report lines

    % ---- identity + mirror list --------------------------------------
    L{end+1} = sprintf('==================== DESIGN REPORT ====================');
    L{end+1} = sprintf(' family %s | D = %.3f m | lambda = %g um | %d elements', ...
                       upper(t.spec.family), D, lam*1e6, numel(e));
    for k = 1:numel(e)
        if abs(e(k).Kr) < 1e21
            L{end+1} = sprintf('   %-10s R = %9.4f m  K = %+9.5f  Vpt=[%7.3f %7.3f %7.3f]', ...
                e(k).name, abs(e(k).Kr), e(k).Kc, e(k).Vpt);
        else
            L{end+1} = sprintf('   %-10s %-10s (flat)        Vpt=[%7.3f %7.3f %7.3f]', ...
                e(k).name, e(k).kind, e(k).Vpt);
        end
    end

    % ---- first-order properties --------------------------------------
    % EFL measured LIVE: chief-ray image displacement per field angle
    % (spec.derived.EFL is the Seidel-seed 3rd-order estimate -- known
    % unreliable for convex-secondary reimagers and folded chains).
    fk  = find(strcmp({e.kind},'FocalPlane'), 1, 'last');
    EFL = NaN;
    try
        del = 0.1 * pi/180/60;                     % 0.1' probe
        t.trace_at_field([0  del]);
        s = macos.trace(fk);  b1 = macos.get_ray_info(s.nRays);
        t.trace_at_field([0 -del]);
        macos.trace(fk);      b2 = macos.get_ray_info(s.nRays);
        t.trace_at_field([]);
        EFL = norm(b1.pos(:,1) - b2.pos(:,1)) / (2*del);
    catch
        if isfield(t.spec,'derived') && isfield(t.spec.derived,'EFL')
            EFL = t.spec.derived.EFL;
        end
    end
    ip     = find(abs([e.Kr]) < 1e21, 1);          % first powered mirror
    fno_m1 = abs(e(ip).Kr) / 2 / D;
    fno_fp = EFL / D;                              % working f/# at the FP
    plate  = EFL * 4.8481368e-6 * 1e6;             % um per arcsec
    lamD   = lam / D * 206264.806 * 1e3;           % mas
    L{end+1} = ' -- first order --';
    L{end+1} = sprintf('   EFL %.2f m | f/# at M1 %.2f | f/# at FP %.2f', ...
                       EFL, fno_m1, fno_fp);
    L{end+1} = sprintf('   plate scale %.1f um/arcsec | lambda/D %.2f mas', ...
                       plate, lamD);

    % ---- field performance (WFE ladder + Strehl) -----------------------
    bias = 0;                                       % stored in radians
    if isfield(t.spec,'field_bias'), bias = t.spec.field_bias*180*60/pi; end
    % Strehl needs the add_pupil EP (PropType=FarField on the EP->FP hop):
    % the PSF peak from INT at the detector, ratioed to the peak of the
    % UNABERRATED aperture -- same amplitudes, phase flattened at the EP
    % via a conjugate-phase mask, same INT sampling (Dave 2026-07-06).
    fk    = find(strcmp({e.kind},'FocalPlane'), 1, 'last');
    doS   = opts.strehl && isfield(t.spec,'pupil') && ~isempty(t.spec.pupil);
    L{end+1} = ' -- field performance (RMS WFE about the field center) --';
    ycol = '';
    if ~isempty(opts.yardstick_m)
        ycol = sprintf('  -tilt @%.3gum', opts.yardstick_m(1)*1e6);
    end
    scol = '';  if doS, scol = '    Strehl'; end
    L{end+1} = sprintf('   bias %+.1f'' | %8s %9s %9s%s%s', bias, ...
                       'field', 'raw', '-tilt', ycol, scol);
    rows = [0, opts.rings_arcmin];
    wr = zeros(size(rows));  wt = zeros(size(rows));
    sr = nan(size(rows));
    for i = 1:numel(rows)
        if rows(i) == 0
            F1 = [0 0];  tag = 'center';
        else
            F1 = macos.design.field_ring(rows(i),'units','arcmin');
            tag = sprintf('%.2f'' ring', rows(i));
        end
        d = wfe_field_diag(t, F1, 'quiet', true);
        [wr(i), iw] = max(d.rms_raw);  wt(i) = max(d.rms_tilt);
        yv = '';
        if ~isempty(opts.yardstick_m)
            w2 = wt(i)*lam/opts.yardstick_m(1);
            dl = '';  if w2 < opts.dl_waves, dl = '  <- DL'; end
            yv = sprintf('  %9.3f%s', w2, dl);
        end
        sv = '';
        if doS
            sr(i) = strehl_(t, F1(iw,:), lam);
            sv = sprintf('  %8.3f', sr(i));
        end
        L{end+1} = sprintf('   %13s %9.3f %9.3f%s%s', tag, wr(i), wt(i), yv, sv);
    end
    if doS
        L{end+1} = ['   (Strehl = far-field peak / unaberrated-aperture ' ...
                    'peak, exact coherent sum over the'];
        L{end+1} = ['    EP-referenced OPD, tilt removed; worst ring ' ...
                    'point.  PSF products: INT at the FP.)'];
    end

    % ---- focal-plane geometry -----------------------------------------
    tilt = NaN;
    if ~isempty(fk) && fk > 1
        s = macos.trace(fk-1);  a = macos.get_ray_info(s.nRays);
        macos.trace(fk);        b = macos.get_ray_info(s.nRays);
        dch = b.pos(:,1) - a.pos(:,1);  dch = dch / norm(dch);
        tilt = acosd(min(1, abs(dot(e(fk).psi(:), dch))));
    end
    L{end+1} = ' -- focal plane --';
    L{end+1} = sprintf('   FP tilt %.3f deg wrt the arriving chief ray', tilt);
    if ~isempty(opts.align)
        fa = opts.align;
        L{end+1} = sprintf(['   aligned from %d field foci: defocus ' ...
            'removed %+.3f mm; field-curvature'], size(fa.foci,2), ...
            fa.defocus_m*1e3);
        L{end+1} = sprintf(['   sag %+.1f to %+.1f um (rms %.1f um); ' ...
            'best-focus blur %.2e m (center)'], min(fa.sag_m)*1e6, ...
            max(fa.sag_m)*1e6, fa.fit_rms_m*1e6, fa.spot_rms_m(1));
    end

    % ---- exit pupil -----------------------------------------------------
    pup = [];
    if isfield(t.spec,'pupil') && ~isempty(t.spec.pupil)
        pup = t.spec.pupil;
        L{end+1} = ' -- exit pupil (add_pupil / FEX) --';
        L{end+1} = sprintf(['   %.3f m from the image, at ' ...
            '[%7.3f %7.3f %7.3f] (elt %d)'], pup.ep_radius, ...
            pup.ep_vpt, pup.ep_elt);
    else
        L{end+1} = ' -- exit pupil: not placed (run add_pupil) --';
    end

    % ---- packaging + clearance -----------------------------------------
    pk  = packaging_report(t, 'quiet', true);
    cc  = t.check_clipping('noload', true);
    obs = {cc([cc.obstructs] > 0).name};
    note = '';
    if ~isempty(pup)
        % post-add_pupil the FP body ALWAYS intersects its own
        % FP_return<->EP reference legs (they emanate from the image on
        % the FP) -- self-conflict by construction, not an obstruction
        obs = obs(~strcmp(obs, e(fk).name));
        note = sprintf(' (%s vs its own pupil-retrace legs excluded)', ...
                       e(fk).name);
    end
    aoi = aoi_report(t);
    L{end+1} = ' -- packaging / clearance --';
    L{end+1} = sprintf('   shroud %.2f x D | train length %.2f m', ...
                       pk.shroud_over_D, pk.length_m);
    if isempty(obs)
        L{end+1} = ['   clearance: every body CLEAR of every beam' note];
    else
        L{end+1} = sprintf('   clearance: body-in-beam at %s%s', ...
                           strjoin(obs, ', '), note);
    end
    for k = 1:numel(aoi)
        L{end+1} = sprintf(['   %-10s AOI %5.1f deg ' ...
            '(spread %.1f deg across the beam)'], aoi(k).name, ...
            aoi(k).aoi_chief_deg, aoi(k).aoi_spread_deg);
    end
    L{end+1} = '=======================================================';

    rep = struct('D_m',D, 'lambda_m',lam, 'EFL_m',EFL, ...
        'fno_m1',fno_m1, 'fno_fp',fno_fp, ...
        'plate_um_per_arcsec',plate, 'lamD_mas',lamD, ...
        'bias_arcmin',bias, 'rings_arcmin',rows, ...
        'wfe_raw',wr, 'wfe_tilt',wt, 'strehl',sr, ...
        'fp_tilt_deg',tilt, ...
        'pupil',pup, 'shroud_over_D',pk.shroud_over_D, ...
        'train_length_m',pk.length_m, 'clear_ok',isempty(obs), ...
        'obstructors',{obs}, 'aoi',aoi, ...
        'text',sprintf('%s\n', L{:}));
    if ~opts.quiet, fprintf('%s', rep.text); end
    if ~isempty(opts.file)
        fid = fopen(opts.file,'w');  fprintf(fid,'%s',rep.text);  fclose(fid);
    end
end

function S = strehl_(t, fxy, lam)
%STREHL_  Strehl at one field point: the far-field intensity peak
%   ratio, PSF peak over the unaberrated-aperture peak, computed
%   EXACTLY as the coherent sum over the exit-pupil-referenced OPD
%   (piston+tip/tilt removed -- field tilt is distortion, not blur;
%   same judgment as the -tilt WFE column):
%       S = |mean(exp(i*2*pi/lambda * W_detilted))|^2
%   This is the peak of the far-field PSF at its own center, ratioed
%   to the flat-wavefront peak of the same aperture.  It is NOT taken
%   from the pixelated INT map: an off-axis field's PSF walks across
%   the FarField INT window and the pixel-peak ratio then measures
%   window sampling, not image quality (verified: 1' ring 0.98 exact
%   vs 0.28 from INT pixels).  The FarField EP emitted by add_pupil is
%   still the enabler for full PSF products (INT at the detector).
    if any(abs(fxy) > 1e-15), t.trace_at_field(fxy);
    else,                     t.trace_at_field([]);   end
    W = macos.opd();
    m = isfinite(W) & (W ~= 0);
    [ry, rx] = find(m);
    ux = rx - mean(rx);  uy = ry - mean(ry);
    r  = max(hypot(ux, uy));
    A  = [ones(nnz(m),1), ux/r, uy/r];
    Wd = W(m) - A*(A \ W(m));                % de-tilted OPD (m)
    S  = abs(mean(exp(1i*2*pi/lam*Wd)))^2;
    if any(abs(fxy) > 1e-15), t.trace_at_field([]); end
end
