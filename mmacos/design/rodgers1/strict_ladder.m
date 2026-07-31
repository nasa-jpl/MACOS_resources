function L = strict_ladder(map_n, variant, Kuse)
%STRICT_LADDER  Where does the stage-2 strict-metric magnitude live?
%
%   Diagnostic for the gate-3 miss.  For each field of the stage-2
%   bias-relative box it reports a ladder of reference-surface freedoms,
%   each MORE permissive than Dave's strict metric, so we can see which
%   freedom (if any) would have to be granted to reach Rodgers' numbers:
%
%     strict     sphere centred on the chief-ray intercept on the FROZEN
%                detector, piston-only removal            <- Dave's ruling
%     bestfoc    same, but the sphere centre slid ALONG THE CHIEF to the
%                per-field best focus (1-D minimisation).  The residual is
%                the floor ANY detector surface -- plane or curved -- can
%                reach, because a surface can only choose where along each
%                chief ray the image point sits.
%     -tilt      bestfoc, additionally least-squares tip/tilt removed
%     -astig     bestfoc, additionally astigmatism (rho^2 cos2t, sin2t)
%                removed
%
%   Pupil coordinates for the fits are the ray intersections on M1
%   (the entrance pupil = the stop), projected transverse to the chief and
%   normalised to the marginal ray.
%
%   Also reports the per-field longitudinal focus shift required, in mm,
%   and the sag of the align_focal_plane surface, so a "his FPA is curved"
%   hypothesis can be sized.

    if nargin < 1 || isempty(map_n), map_n = 5; end
    if nargin < 2 || isempty(variant), variant = 'epd4060'; end
    isseq = strcmpi(variant,'seq');
    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);
    if isseq, P = rodgers_common('seq'); else, P = rodgers_common();  P.EPD_mm = 4060; end
    if nargin < 3 || isempty(Kuse), Kuse = P.K_nom; end

    t = build_tma(P, Kuse, P.offset_deg);
    fprintf('  variant %s: EPD %g mm, conics %.12f %.12f %.12f\n', ...
            variant, P.EPD_mm, Kuse);
    fp = t.align_focal_plane('grid',5,'span_arcmin',6);
    nE = numel(t.spec.elt);
    V0 = t.spec.elt(nE).Vpt(:);  N0 = t.spec.elt(nE).psi(:); N0 = N0/norm(N0);
    fprintf('  FPA fit: tilt %.4f deg, plane-fit rms %.4g mm, sag range %.4g mm\n', ...
            fp.tilt_deg, fp.fit_rms_m*1e3, (max(fp.sag_m)-min(fp.sag_m))*1e3);

    % box-RELATIVE: trace_at_field adds spec.field_bias itself.
    if isseq
        Frel = P.seq.Frel;            % his 15-point half box
    else
        Frel = macos.design.field_grid(P.fov_half_deg*60, map_n, 'units','arcmin');
    end
    Fbias = Frel;
    K = size(Fbias,1);
    L = struct('fields',Fbias,'bias_deg',t.spec.field_bias*180/pi,'strict',nan(K,1),'bestfoc',nan(K,1), ...
               'notilt',nan(K,1),'noastig',nan(K,1),'dfoc_mm',nan(K,1));

    % exit pupil, once (it moves by << its distance across the box)
    t.trace_at_field(Fbias(1,:));  s = macos.trace(nE);
    r0 = macos.get_ray_info(s.nRays);
    t.trace_at_field(Fbias(1,:) + [1e-5 0]);  macos.trace(nE);
    rp = macos.get_ray_info(s.nRays);
    XP = fex_cross(r0.pos(:,1), r0.dir(:,1), rp.pos(:,1), rp.dir(:,1));

    fprintf('\n  %-16s %10s %10s %10s %10s %10s\n', 'field (arcmin)', ...
            'strict', 'bestfoc', '-tilt', '-astig', 'dfoc/mm');
    for k = 1:K
        t.trace_at_field(Fbias(k,:));
        s  = macos.trace(nE);   rb = macos.get_ray_info(s.nRays);
        macos.trace(1);         ra = macos.get_ray_info(s.nRays);
        macos.trace(nE);
        ok = rb.ok_trace(:) & rb.ok_pass(:) & ra.ok_trace(:) & ra.ok_pass(:);
        ok(1) = false;
        Pb = rb.pos(:,ok);  D = rb.dir(:,ok);  Lo = rb.opl(ok);
        p1 = rb.pos(:,1);   d1 = rb.dir(:,1);

        c0 = p1 + d1*(dot(N0, V0 - p1)/dot(N0, d1));
        Rf = @(c) norm(XP - c);
        W  = @(c) strict_sphere_opl(Pb, D, Lo, c, Rf(c));
        L.strict(k) = std(W(c0));

        % --- best focus: slide the centre along the chief ---------------
        f  = @(u) std(W(c0 + d1*u));
        u  = fminbnd(f, -0.5, 0.5, optimset('TolX',1e-9));
        L.bestfoc(k) = f(u);   L.dfoc_mm(k) = u*1e3;

        % --- pupil coords at M1, transverse to the chief ---------------
        Pa = ra.pos(:,ok);
        e3 = d1/norm(d1);
        e1 = [1;0;0] - e3*dot([1;0;0],e3);  e1 = e1/norm(e1);
        e2 = cross(e3,e1);
        xq = (e1.'*(Pa - mean(Pa,2))).';  yq = (e2.'*(Pa - mean(Pa,2))).';
        rn = max(hypot(xq,yq));  xq = xq/rn;  yq = yq/rn;

        Wb = W(c0 + d1*u);
        A1 = [ones(numel(xq),1), xq, yq];
        L.notilt(k)  = std(Wb - A1*(A1\Wb));
        A2 = [A1, xq.^2 - yq.^2, 2*xq.*yq];
        L.noastig(k) = std(Wb - A2*(A2\Wb));

        fprintf('  [%+6.2f %+6.2f] %10.4g %10.4g %10.4g %10.4g %10.4g\n', ...
            Frel(k,1)*180/pi*60, Frel(k,2)*180/pi*60, ...
            L.strict(k)*1e9, L.bestfoc(k)*1e9, L.notilt(k)*1e9, ...
            L.noastig(k)*1e9, L.dfoc_mm(k));
    end
    t.trace_at_field([]);
    fprintf('\n  SUMMARY (nm)        min        max        avg\n');
    for f = {'strict','bestfoc','notilt','noastig'}
        v = L.(f{1})*1e9;
        fprintf('   %-10s %10.4g %10.4g %10.4g\n', f{1}, min(v), max(v), mean(v));
    end
    fprintf('   focus shift needed: %.4g .. %.4g mm  (range %.4g mm)\n', ...
            min(L.dfoc_mm), max(L.dfoc_mm), max(L.dfoc_mm)-min(L.dfoc_mm));
    fprintf('   Rodgers S2 box:     %10.4g %10.4g %10.4g\n', ...
            P.gt.s2_box(1)*1e3, P.gt.s2_box(2)*1e3, P.gt.s2_box(3)*1e3);
    save(fullfile(here,sprintf('rodgers1_%s_strict_ladder.mat',variant)),'L','fp','XP');
end

function X = fex_cross(p1,d1,p2,d2)
    d1 = d1/norm(d1);  d2 = d2/norm(d2);
    w0 = p1 - p2;  b = dot(d1,d2);  den = 1 - b^2;
    if abs(den) < 1e-14, X = p1; return; end
    s1 = ( b*dot(d2,w0) - dot(d1,w0)) / den;
    s2 = ( dot(d2,w0) - b*dot(d1,w0)) / den;
    X  = 0.5*((p1 + d1*s1) + (p2 + d2*s2));
end

function t = build_tma(P, K, bias_deg)
    t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_mm', P.EPD_mm, ...
            'wavelength_m', P.lambda_m, 'model_size', P.model_size);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',K(1),'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',K(2),'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',K(3),'spacing_after','derive');
    if isfield(P,'M1_hole_m') && P.M1_hole_m > 0
        t.set_hole('M1', P.M1_hole_m);      % CODE V "CIR HOL" on M1
    end
    if bias_deg ~= 0, t.set_field_bias(bias_deg*60); end
    t.build();
end
