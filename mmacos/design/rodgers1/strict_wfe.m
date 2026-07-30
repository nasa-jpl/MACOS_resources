function out = strict_wfe(t, F, opts)
%STRICT_WFE  Dave's strict per-field WFE metric, computed constructively.
%
%   out = STRICT_WFE(T, F) evaluates, for every field offset in F (K x 2,
%   radians, box-relative -- trace_at_field adds the design's field bias),
%   the RMS wavefront error of telescope T referenced to
%
%       a sphere ANCHORED AT THE EXIT PUPIL, CENTRED on that field's
%       CHIEF-RAY INCIDENCE POINT on the FROZEN DETECTOR,
%       with PISTON-ONLY removal.
%
%   Nothing here relies on an engine reference-surface default.  Per field
%   we harvest the raw ray data at the terminal FocalPlane -- position,
%   direction and cumulative OPL (macos.get_ray_info) -- and reduce it in
%   MATLAB.  Rays are straight after the last surface (index 1), so the
%   OPL to ANY surface intersecting that final segment is algebraic and
%   EXACT; no paraxial / 2*rho^2 expansion is used anywhere.
%
%   Construction, per field f:
%     c_f : the chief ray's intersection with the frozen detector plane
%           (Vd, Nd).  This is the sphere CENTRE.
%     X_f : the exit pupil, found the way FEX finds it -- the crossing of
%           two probe chief rays at neighbouring fields (tracesub.F:312-368).
%           This ANCHORS the sphere.
%     R_f : |X_f - c_f|, the sphere radius.  Note R only enters the metric
%           through the O(eps^2/R) term below (eps = transverse ray
%           aberration), so the answer is insensitive to it near focus and
%           needs it only when the detector is far off focus.
%     W   : for each ray with terminal point P, direction d, OPL L,
%              v  = P - c_f ,  a = v.d ,  e = |v|^2
%              s  = -a - sqrt(R^2 + a^2 - e)          (upstream crossing)
%              W  = L + s
%           i.e. the OPL from the source to the reference sphere, exactly.
%     wfe = std(W) over the rays, piston-only (std removes the mean; no
%           tip/tilt, no defocus, no astigmatism is fitted out).
%
%   WHY THE SPHERE AND NOT THE DETECTOR PLANE.  The engine's own
%   macos.trace(FP).rmsWFE is std(CumRayL) at each ray's OWN intercept on
%   the detector (tracesub.F SUBROUTINE OPD, :163-239).  On a TILTED
%   detector -- and a fast anastigmat's off-axis focal surface is tilted by
%   ~14 deg here -- that carries a spurious term (transverse ray
%   aberration) x tan(tilt), which on this design is ~20x the wavefront
%   error itself.  Referencing to a sphere centred on the chief intercept
%   removes it and leaves the field-to-field focus/astigmatism walk that
%   the fixed detector genuinely imposes.
%
%   Name-value:
%     'detector'  struct('Vpt',1x3,'psi',1x3) overriding the emitted FP
%                 plane (metres / unit vector).  DEFAULT: the emitted
%                 FocalPlane -- i.e. whatever the caller froze.
%     'dz'        scalar (m): additionally displace the detector plane
%                 along its own normal.  The displaced-detector
%                 discriminator (gate 1) uses this.
%     'probe'     probe field offset for the exit-pupil finder, rad
%                 (default 1e-5; FEX uses 5e-6).
%     'xp'        1x3 fixed exit pupil (metres) -- skips the probe trace
%                 (2x faster).  Default [] = per-field FEX-style probe.
%     'quiet'     suppress the per-field line (default true).
%
%   Returns OUT with fields .fields (K x 2 rad), .wfe (K, waves),
%   .wfe_m (K, metres), .nrays (K), .c (3 x K sphere centres),
%   .xp (3 x K exit pupils), .R (K radii), .detector (the plane used),
%   .eng_rms_m (K, the engine's own FP rmsWFE in METRES, for contrast).
%
%   NOTE ON ENGINE UNITS.  macos.trace(k).rmsWFE is returned in BaseUnits
%   (metres for these decks), NOT in waves -- unlike
%   realize_apertures(...).wfe, which divides by lambda.  Multiplying
%   rmsWFE by lambda_nm understates it by 1e6.

    arguments
        t
        F (:,2) double
        opts.detector struct = struct([])
        opts.dz (1,1) double = 0
        opts.probe (1,1) double = 1e-5
        opts.xp (:,:) double = []
        opts.quiet (1,1) logical = true
    end

    nE = numel(t.spec.elt);
    if isempty(fieldnames(opts.detector))
        Vd = t.spec.elt(nE).Vpt(:);   Nd = t.spec.elt(nE).psi(:);
    else
        Vd = opts.detector.Vpt(:);    Nd = opts.detector.psi(:);
    end
    Nd = Nd / norm(Nd);
    Vd = Vd + Nd*opts.dz;

    K = size(F,1);
    out = struct('fields',F, 'wfe',nan(K,1), 'wfe_m',nan(K,1), ...
                 'nrays',zeros(K,1), 'c',nan(3,K), 'xp',nan(3,K), ...
                 'R',nan(K,1), 'eng_rms_m',nan(K,1), ...
                 'detector',struct('Vpt',Vd.','psi',Nd.'), 'dz',opts.dz, ...
                 'lambda',t.spec.wavelength);

    for k = 1:K
        t.trace_at_field(F(k,:));
        s  = macos.trace(nE);
        ri = macos.get_ray_info(s.nRays);
        out.eng_rms_m(k) = s.rmsWFE;                 % BaseUnits (m)
        ok = ri.ok_trace(:) & ri.ok_pass(:);
        ok(1) = false;                               % engine OPD skips the chief
        if nnz(ok) < 10, continue; end

        p1 = ri.pos(:,1);  d1 = ri.dir(:,1);         % the chief ray
        % ---- sphere centre: chief incidence on the FROZEN detector -----
        den = dot(Nd, d1);
        if abs(den) < 1e-12, continue; end
        c = p1 + d1 * (dot(Nd, Vd - p1) / den);
        out.c(:,k) = c;

        % ---- exit pupil: FEX's two-probe chief-ray crossing ------------
        if ~isempty(opts.xp)
            X = opts.xp(:);
        else
            t.trace_at_field(F(k,:) + [opts.probe 0]);
            macos.trace(nE);
            rp = macos.get_ray_info(s.nRays);
            X  = line_cross(p1, d1, rp.pos(:,1), rp.dir(:,1));
        end
        out.xp(:,k) = X;
        R = norm(X - c);
        out.R(k) = R;

        % ---- exact OPL to the reference sphere -------------------------
        W = strict_sphere_opl(ri.pos(:,ok), ri.dir(:,ok), ri.opl(ok), c, R);
        out.wfe_m(k) = std(W);
        out.wfe(k)   = out.wfe_m(k) / t.spec.wavelength;
        out.nrays(k) = nnz(ok);
        if ~opts.quiet
            fprintf('   field %2d [%+8.4f %+8.4f]'' : %10.4g nm  (%d rays)\n', ...
                    k, F(k,1)*180/pi*60, F(k,2)*180/pi*60, out.wfe_m(k)*1e9, nnz(ok));
        end
    end
    t.trace_at_field([]);
end

% ---------------------------------------------------------------------
function X = line_cross(p1, d1, p2, d2)
%LINE_CROSS  Mid-point of the mutual perpendicular of two skew lines --
%   the exit pupil as FEX defines it (FindCrossPt, tracesub.F:4844).
    d1 = d1/norm(d1);  d2 = d2/norm(d2);
    w0 = p1 - p2;  b = dot(d1,d2);
    den = 1 - b^2;
    if abs(den) < 1e-14, X = p1; return; end       % telecentric
    s1 = ( b*dot(d2,w0) - dot(d1,w0)) / den;
    s2 = ( dot(d2,w0) - b*dot(d1,w0)) / den;
    X  = 0.5*((p1 + d1*s1) + (p2 + d2*s2));
end
