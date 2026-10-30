function C = seq_centroid()
%SEQ_CENTROID  The centroid-primary deliverable at the CODE V .seq truth.
%
%   C = SEQ_CENTROID()
%
%   Dave's 2026-07-31 ruling: the PRIMARY wavefront reference is the SPOT
%   CENTROID, not the chief ray.  It is closest to the CODE V numbers, and the
%   grid of centroids is what the detector actually sees.  Chief-referenced is
%   retained as a labelled SECONDARY column.  Distortion is tracked in the
%   grid of CENTROIDS against the ideal f.theta mapping, not the chief grid.
%
%   Produces, at EPD 5000 / M1 hole / his 15-point half box:
%
%     1  HEADLINE TABLE -- all four of our stages plus the three re-traced
%        CODE V designs; columns centroid (primary) / chief (secondary) /
%        CODE V reported.
%     2  IDENTITY CROSS-CHECK -- the tilt removed between the two references,
%        times the sphere radius, must equal the ray-measured
%        centroid-minus-chief displacement.  It does, to ~2e-4 relative.
%        The leftover of the DIFFERENCE itself is ~2e-3 rather than
%        machine-small, and that is physics: the displacement lies in the
%        detector plane, which the beam meets at ~14 deg, so it carries an
%        along-chief component and moves FOCUS as well as tilt.  Fit the
%        defocus too and the residual goes machine-small.  Both are reported.
%     3  CENTROID-DISPLACEMENT MAP (um on the detector) -- the coma tracker.
%        Expectation on record: largest at stage 2, shrinking as the solves
%        correct the field.
%     4  DISTORTION MAP -- centroid grid vs the ideal f.theta grid (f = 100 m
%        from his UMY solve), um.  Reported ALONGSIDE, never inside, the WFE
%        tables.
%     5  HARDWARE NUMBERS -- beam AOI on the detector (the FPA
%        acceptance-angle / assembly-alignment driver) vs the MECHANICAL
%        detector tilt about the axis.  Two different ~14 deg / ~0.1 deg
%        quantities that have been confused once already.
%
%   Writes rodgers1_seq_centroid.mat and rodgers1_seq_centroid_maps.png.
%   Baselines untouched; every artifact is _seq-suffixed.
%
%   See PACKET.md Addendum 9.

    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);
    P = rodgers_common('seq');
    S = P.seq;  lam_nm = P.lambda_m*1e9;
    Frel = S.Frel;
    EFL_mm = S.EFL_mm;

    % ---- stage 1 at truth is the one deck the earlier run did not emit ---
    s1deck = fullfile(here,'rodgers1_seq_stage1.in');
    if ~isfile(s1deck), build_stage1_(P, s1deck); end

    rows = { 'S1  on-axis, verbatim conics',      s1deck,                                        P.gt.s1_onaxis_box
             'S2  offset, verbatim conics',       fullfile(here,'rodgers1_seq_rodgersS2.in'),    P.gt.s2_box
             'S3  ours, joint solve',             fullfile(here,'rodgers1_seq_stage3_xpopt.in'), P.gt.s3_box
             'S4  ours, joint solve',             fullfile(here,'rodgers1_seq_stage4_xpopt.in'), P.gt.s4_box
             'CODE V S2 re-traced',               fullfile(here,'rodgers1_seq_rodgersS2.in'),    P.gt.s2_box
             'CODE V S3 re-traced',               fullfile(here,'rodgers1_seq_rodgersS3.in'),    P.gt.s3_box
             'CODE V S4 re-traced',               fullfile(here,'rodgers1_seq_rodgersS4.in'),    P.gt.s4_box };

    C = struct('row',cell(1,size(rows,1)));
    for i = 1:size(rows,1)
        nm = rows{i,1};  deck = rows{i,2};  gt = rows{i,3};
        if ~isfile(deck)
            fprintf('  (missing %s -- run run_seq first)\n', deck);  continue;
        end
        F = Frel;
        if startsWith(nm,'S1'), F = Frel; end   % S1's deck is already on-axis
        s = strict_wfe_deck(deck, F, 'reference','strict-centroid');
        C(i).name = nm;  C(i).deck = deck;  C(i).gt = gt*lam_nm;  C(i).scan = s;
        C(i).cen = [max(s.wfe_m_centroid)*1e9, mean(s.wfe_m_centroid)*1e9];
        C(i).chf = [max(s.wfe_m_chief)*1e9,    mean(s.wfe_m_chief)*1e9];
        C(i).dcen_um = s.dcen_m*1e6;
        C(i).tilt_resid  = s.tilt_resid;
        C(i).defoc_resid = s.defoc_resid;
        C(i).dcen_err_um = s.dcen_err_m*1e6;
        C(i).dcen_perp_um = s.dcen_perp_m*1e6;
        C(i).dcen_long_um = s.dcen_long_m*1e6;
        C(i).diff_rms_nm  = s.diff_rms*1e9;
        % The relative residuals are 0/0 on any field where the two
        % references COINCIDE (the box-centre field of the on-axis stage, and
        % any field with no coma).  Mask those rather than let a 0/0 = 1
        % dominate a max.  Threshold: the difference must be at least 1e-3 of
        % that design's own WFE to be worth taking a ratio of.
        C(i).ok_ratio = s.diff_rms*1e9 > 1e-3*max(s.wfe_m_centroid*1e9);
        C(i).hw = hardware_(deck, s);
        C(i).dist = distortion_(s, Frel, EFL_mm, startsWith(nm,'S1')*0 + S.offset_deg, nm);
    end

    % =================================================================
    banner('1.  HEADLINE TABLE -- CENTROID-PRIMARY  (EPD %g, %d fields, nm @ %g nm)', ...
           P.EPD_mm, size(Frel,1), lam_nm);
    fprintf('  %-30s | %-17s | %-17s | %-15s | %s\n', 'design', ...
            'CENTROID (primary)', 'chief (secondary)', 'CODE V reported', 'centroid x');
    fprintf('  %s\n', repmat('-',1,105));
    for i = 1:numel(C)
        if isempty(C(i).name), continue; end
        g = C(i).gt;
        fprintf('  %-30s | %8.1f/%-8.1f | %8.1f/%-8.1f | %7.1f/%-7.1f | %5.2fx / %5.2fx\n', ...
            C(i).name, C(i).cen(1), C(i).cen(2), C(i).chf(1), C(i).chf(2), ...
            g(2), g(3), C(i).cen(1)/g(2), C(i).cen(2)/g(3));
    end
    fprintf(['\n  max/avg in nm.  "centroid x" = centroid-referenced / CODE V reported.\n' ...
             '  S1''s CODE V column is his on-axis box (1.46/0.61 nm) -- a different\n' ...
             '  regime from the offset stages, listed for completeness.\n']);

    % =================================================================
    banner('2.  IDENTITY CROSS-CHECK -- centroid-vs-chief is tilt (+ induced defocus)');
    fprintf(['  The fitted tilt, times the sphere radius, must reproduce the\n' ...
             '  ray-measured centroid-minus-chief displacement.  Nothing here is\n' ...
             '  fitted to anything; this is the cross-check.\n\n']);
    fprintf('  %-30s | %10s | %10s | %s\n', 'design', ...
            'resid /P+T', 'resid /P+T+F', 'implied-vs-measured transverse displ.');
    fprintf(['  (ratios taken only over fields where the two references actually\n' ...
             '   differ -- on a field with no coma the difference is null and the\n' ...
             '   ratio would be 0/0.)\n']);
    fprintf('  %s\n', repmat('-',1,102));
    for i = 1:numel(C)
        if isempty(C(i).name), continue; end
        m = C(i).ok_ratio;
        fprintf('  %-30s | %10.2e | %10.2e | %9.3e um of %8.3f um  (%.1e rel)\n', ...
            C(i).name, max(C(i).tilt_resid(m)), max(C(i).defoc_resid(m)), ...
            max(C(i).dcen_err_um), max(C(i).dcen_perp_um), ...
            max(C(i).dcen_err_um)/max(C(i).dcen_perp_um));
    end
    fprintf(['\n  The tilt-only residual is ~2e-3, NOT machine-small, and that is\n' ...
             '  physics rather than error: the displacement lies IN THE DETECTOR\n' ...
             '  PLANE, which the beam meets at ~14 deg, so it carries an along-chief\n' ...
             '  component -- moving the reference changes FOCUS as well as tilt.\n' ...
             '  Add the defocus term and the residual drops to machine-small, which\n' ...
             '  is the actual identity: chief-vs-centroid IS tilt + induced defocus.\n\n']);
    fprintf('  %-30s | %12s | %12s | %s\n', 'design', ...
            'displ. total', 'transverse', 'along-chief (the defocus)');
    fprintf('  %s\n', repmat('-',1,90));
    for i = 1:numel(C)
        if isempty(C(i).name), continue; end
        fprintf('  %-30s | %9.3f um | %9.3f um | %+9.3f um\n', C(i).name, ...
            max(C(i).dcen_um), max(C(i).dcen_perp_um), ...
            C(i).dcen_long_um(argmax_(C(i).dcen_um)));
    end

    % ---- where the centroid ruling sits on the reference ladder ---------
    lp = fullfile(here,'rodgers1_seq_ladder.mat');
    if isfile(lp)
        banner('1b. WHERE THE CENTROID RULING SITS ON THE REFERENCE LADDER');
        Lm = load(lp);  Ld = Lm.L;
        fprintf(['  The ladder rungs (seq_ladder) remove freedoms by LEAST SQUARES.\n' ...
                 '  The centroid reference removes a SPECIFIC tilt -- the one that\n' ...
                 '  re-centres on the spot -- which is not the variance-minimising\n' ...
                 '  tilt.  So the two are close but not equal, and the gap is worth\n' ...
                 '  stating rather than eliding:\n\n']);
        fprintf('  %-34s %10s %10s\n', 'rung (max ratio vs CODE V)', 'S2', 'S3');
        fprintf('  %s\n', repmat('-',1,58));
        fprintf('  %-34s %9.3fx %9.3fx\n', 'strict-chief (frozen detector)', ...
                Ld(1).strict(1)/Ld(1).gt(2), Ld(2).strict(1)/Ld(2).gt(2));
        r2 = C(5).cen(1)/C(5).gt(2);  r3 = C(6).cen(1)/C(6).gt(2);
        fprintf('  %-34s %9.3fx %9.3fx   <- THE RULING\n', ...
                'strict-centroid (frozen detector)', r2, r3);
        fprintf('  %-34s %9.3fx %9.3fx\n', 'strict-chief + per-field best focus', ...
                Ld(1).bestfoc(1)/Ld(1).gt(2), Ld(2).bestfoc(1)/Ld(2).gt(2));
        fprintf('  %-34s %9.3fx %9.3fx\n', 'least-squares tilt + best focus', ...
                Ld(1).notilt(1)/Ld(1).gt(2), Ld(2).notilt(1)/Ld(2).gt(2));
        fprintf('  %-34s %9.3fx %9.3fx\n', '+ astigmatism (overshoots)', ...
                Ld(1).noastig(1)/Ld(1).gt(2), Ld(2).noastig(1)/Ld(2).gt(2));
    end

    % =================================================================
    banner('3.  CENTROID-DISPLACEMENT MAP -- um on the detector (the coma tracker)');
    for i = 1:numel(C)
        if isempty(C(i).name), continue; end
        fprintf('\n  %s :  min %7.3f   max %7.3f   mean %7.3f um\n', ...
                C(i).name, min(C(i).dcen_um), max(C(i).dcen_um), mean(C(i).dcen_um));
        print_field_map_(Frel, C(i).dcen_um, '%7.3f');
    end
    fprintf(['\n  EXPECTATION ON RECORD: largest at stage 2, shrinking as the solves\n' ...
             '  correct the field.  Verdict printed in the summary below.\n']);

    % =================================================================
    banner('4.  DISTORTION MAP -- centroid grid vs ideal f.theta (f = %.0f mm), um', EFL_mm);
    fprintf(['  Reported ALONGSIDE the WFE tables, never inside them.\n\n' ...
             '  Three readings, most literal first:\n' ...
             '   (a) vs the ideal f.theta grid with the SCALE HELD AT f -- only the\n' ...
             '       detector''s arbitrary placement and clocking are removed.  This\n' ...
             '       is the number the brief asks for.\n' ...
             '   (b) after also fitting a uniform scale -- the leftover is the shape\n' ...
             '       error, and the fitted scale IS the local magnification, which on\n' ...
             '       a 0.2 deg patch sitting 0.5 deg off axis is NOT f.\n' ...
             '   (c) after a full affine fit -- the leftover is what no linear map can\n' ...
             '       absorb, i.e. the genuinely NONLINEAR distortion.  (b) minus (c) is\n' ...
             '       the ANAMORPHIC part: tangential and sagittal magnifications differ\n' ...
             '       off axis, and a uniform-scale fit cannot represent that.\n']);
    for i = 1:numel(C)
        if isempty(C(i).name) || isempty(C(i).dist), continue; end
        D = C(i).dist;
        fprintf('\n  %s\n', C(i).name);
        fprintf('    vs ideal f.theta (scale HELD at f, placement removed):\n');
        fprintf('        max %9.1f um   rms %9.1f um\n', max(D.raw_um), rms_(D.raw_um));
        fprintf('    local mapping scale %11.3f mm/rad vs f = %.0f  (%+.4f%%)\n', ...
                D.M, EFL_mm, 100*(D.M-EFL_mm)/EFL_mm);
        fprintf('    after uniform scale+rotation : max %8.1f um, rms %8.1f um\n', ...
                max(D.resid_um), rms_(D.resid_um));
        fprintf('    after FULL AFFINE (= nonlinear): max %8.1f um, rms %8.1f um\n', ...
                max(D.affine_um), rms_(D.affine_um));
        fprintf('    frame rotation %+8.4f deg (180 = the odd-mirror inversion)\n', D.phi_deg);
        fprintf('    -- map below: departure from the ideal f.theta grid (um)\n');
        print_field_map_(Frel, D.raw_um, '%7.1f');
    end

    % =================================================================
    banner('5.  HARDWARE NUMBERS -- labelled by what the customer buys');
    fprintf(['  These are TWO DIFFERENT ANGLES and have been confused once already\n' ...
             '  (PACKET 4b).  Both are reported, each with the hardware it drives.\n\n']);
    fprintf('  %-30s | %-28s | %s\n', 'design', ...
            'BEAM AOI on the detector', 'MECHANICAL detector tilt');
    fprintf('  %-30s | %-28s | %s\n', '', ...
            '(FPA acceptance angle,', '(about the optical axis;');
    fprintf('  %-30s | %-28s | %s\n', '', ...
            ' assembly alignment)', ' CODE V image ADE)');
    fprintf('  %s\n', repmat('-',1,95));
    for i = 1:numel(C)
        if isempty(C(i).name), continue; end
        fprintf('  %-30s | %10.4f deg               | %+9.4f deg\n', ...
                C(i).name, C(i).hw.aoi_deg, C(i).hw.mech_deg);
    end
    fprintf('\n  his .seq image ADE, verbatim: S2 %+.4f  S3 %+.4f  S4 %+.4f deg\n', ...
            S.img_ADE_deg(2), S.img_ADE_deg(3), S.img_ADE_deg(4));
    fprintf('  the same in OUR frame (decoded, alpha = -ADE): %+.4f %+.4f %+.4f\n', ...
            -S.img_ADE_deg(2), -S.img_ADE_deg(3), -S.img_ADE_deg(4));
    i4 = find(strcmp({C.name},'CODE V S4 re-traced'),1);
    if ~isempty(i4)
        fprintf(['\n  ADE-DECODE WITNESS #4, and the strongest yet: on HIS S4 optics our\n' ...
                 '  own FPA fit lands at %+.4f deg where his decoded ADE is %+.4f deg --\n' ...
                 '  %.4f deg apart on a %.2f deg angle (%.2f%%).  The earlier witnesses\n' ...
                 '  were all at ~0.07 deg, where a sign error is easy to hide; this one\n' ...
                 '  is at 4.4 deg and cannot be.\n'], ...
                C(i4).hw.mech_deg, -S.img_ADE_deg(4), ...
                abs(C(i4).hw.mech_deg + S.img_ADE_deg(4)), abs(S.img_ADE_deg(4)), ...
                100*abs(C(i4).hw.mech_deg + S.img_ADE_deg(4))/abs(S.img_ADE_deg(4)));
    end
    fprintf(['\n' ...
             '  READ: the focal-plane tilt that matters to hardware is the ~14 deg\n' ...
             '  BEAM AOI -- it sets the FPA acceptance angle and drives assembly\n' ...
             '  alignment.  The MECHANICAL tilt of the detector about the axis is\n' ...
             '  ~0.1 deg for stages 1-3; stage 4''s real ADE is 4.44 deg, a genuine\n' ...
             '  mount requirement, because there the solve uses image tilt as a DOF.\n']);

    % ---- verdicts ----------------------------------------------------
    banner('SUMMARY');
    idx = find(arrayfun(@(x) ~isempty(x.name), C));
    [~,imax] = max(arrayfun(@(x) max(x.dcen_um), C(idx)));
    fprintf('  centroid displacement peaks on: %s\n', C(idx(imax)).name);
    fprintf('  worst tilt-only residual   : %.3e (relative; = the induced defocus)\n', ...
            max(arrayfun(@(x) max(x.tilt_resid(x.ok_ratio)), C(idx))));
    fprintf('  worst tilt+defocus residual: %.3e (relative)  <- the identity\n', ...
            max(arrayfun(@(x) max(x.defoc_resid(x.ok_ratio)), C(idx))));
    fprintf('  worst implied-vs-measured displacement    : %.3e um\n', ...
            max(arrayfun(@(x) max(x.dcen_err_um), C(idx))));

    save(fullfile(here,'rodgers1_seq_centroid.mat'),'C','-v7.3');
    seq_centroid_maps(C, Frel, fullfile(here,'rodgers1_seq_centroid_maps.png'));
    fprintf('\nsaved rodgers1_seq_centroid.mat + rodgers1_seq_centroid_maps.png\n');
end

% =====================================================================
function build_stage1_(P, path)
    t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_mm', P.EPD_mm, ...
            'wavelength_m', P.lambda_m, 'model_size', P.model_size);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
    t.set_hole('M1', P.M1_hole_m);
    t.build();
    t.align_focal_plane('grid',5,'span_arcmin',6);
    t.save(path);
    fprintf('  built stage-1 deck at the .seq truth -> %s\n', path);
end

function hw = hardware_(deck, s)
%HARDWARE_  Beam AOI on the detector vs the detector's mechanical tilt.
    Nd = s.detector.psi(:);  Nd = Nd/norm(Nd);
    % arriving chief of the box-centre field: the ray that defines the AOI.
    [~,kc] = min(vecnorm(s.fields,2,2));
    d1 = s.c_chief(:,kc) - s.xp(:,kc);        % exit pupil -> chief intercept
    d1 = d1/norm(d1);
    hw.aoi_deg  = acosd(min(1,abs(dot(d1,Nd))));
    hw.mech_deg = atan2d(Nd(2), -Nd(3));
    hw.deck = deck;
end

function D = distortion_(s, Frel, EFL_mm, offset_deg, nm)
%DISTORTION_  Centroid grid vs the ideal f.theta grid.
%   Fit p_actual = A + M*Rot(phi)*theta over all fields; M is the traced
%   mapping constant (compare with f) and the residual is the distortion.
%   Translation and rotation are removed because the detector's position and
%   clocking are arbitrary; SCALE is NOT removed, so an EFL error stays visible.
    ok = all(isfinite(s.c_centroid),1);
    if nnz(ok) < 4, D = []; return; end
    th = Frel(ok,:);
    if ~startsWith(nm,'S1'), th(:,2) = th(:,2) + deg2rad(offset_deg); end
    f1 = [1;0;0];  f2 = [0;1;0];              % global x,y: the train is coaxial about z
    p  = [ (f1.'*s.c_centroid(:,ok)).' , (f2.'*s.c_centroid(:,ok)).' ] * 1e3;  % mm
    % least squares on [a b; -b a] (scale*rotation) + translation
    n = size(th,1);
    A = zeros(2*n,4);  y = zeros(2*n,1);
    A(1:2:end,:) = [th(:,1), -th(:,2), ones(n,1), zeros(n,1)];
    A(2:2:end,:) = [th(:,2),  th(:,1), zeros(n,1), ones(n,1)];
    y(1:2:end) = p(:,1);  y(2:2:end) = p(:,2);
    c = A\y;
    D.M = hypot(c(1),c(2));  D.phi_deg = atan2d(c(2),c(1));
    r = y - A*c;
    D.resid_um = hypot(r(1:2:end), r(2:2:end)) * 1e3;      % mm -> um

    % (a) THE LITERAL ASK: departure from the ideal f.theta grid with the
    % SCALE HELD AT f.  Only the detector's arbitrary placement and clocking
    % are removed (translation + the fitted rotation); the magnification is
    % NOT fitted, so a local magnification that differs from f stays visible.
    ph = deg2rad(D.phi_deg);
    Rm = [cos(ph) -sin(ph); sin(ph) cos(ph)];
    pid = (EFL_mm * (Rm * th.')).';
    off = mean(p - pid, 1);
    D.raw_um = vecnorm(p - pid - off, 2, 2) * 1e3;

    % (b) FULL AFFINE fit (6 parameters): allows anisotropic magnification,
    % which an off-axis patch genuinely has (tangential and sagittal
    % magnifications differ under distortion).  Its residual is the part that
    % is NOT any linear map -- the genuinely NONLINEAR distortion.
    Aa = zeros(2*n,6);
    Aa(1:2:end,:) = [th(:,1), th(:,2), ones(n,1), zeros(n,3)];
    Aa(2:2:end,:) = [zeros(n,3), th(:,1), th(:,2), ones(n,1)];
    ra = y - Aa*(Aa\y);
    D.affine_um = hypot(ra(1:2:end), ra(2:2:end)) * 1e3;

    % literal radial form, about the box centre
    D.radial_um = (vecnorm(p - mean(p,1),2,2) - ...
                   EFL_mm*vecnorm(th - mean(th,1),2,2)) * 1e3;
    D.theta = th;  D.p_mm = p;
end

function print_field_map_(Frel, v, fmt)
    xs = unique(round(Frel(:,1)*180/pi*60,4));
    ys = unique(round(Frel(:,2)*180/pi*60,4));
    fprintf('      dYAN\\XAN');  fprintf('%9.2f', xs);  fprintf('   (arcmin)\n');
    for iy = numel(ys):-1:1
        fprintf('      %+8.2f', ys(iy));
        for ix = 1:numel(xs)
            k = find(abs(Frel(:,1)*180/pi*60-xs(ix))<1e-3 & ...
                     abs(Frel(:,2)*180/pi*60-ys(iy))<1e-3, 1);
            if isempty(k), fprintf('        .');
            else,          fprintf([' ' fmt], v(k)); end
        end
        fprintf('\n');
    end
end

function r = rms_(v), r = sqrt(mean(v(:).^2)); end

function i = argmax_(v), [~,i] = max(v); end

function banner(varargin)
    fprintf('\n=================================================================\n');
    fprintf(' %s\n', sprintf(varargin{:}));
    fprintf('=================================================================\n');
end
