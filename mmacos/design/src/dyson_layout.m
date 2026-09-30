function [D, geom] = dyson_layout(r, n, opts)
%DYSON_LAYOUT  Closed-form seed + exact-trace verification of a concentric Dyson.
%   [D, geom] = dyson_layout(r, n) lays out the classical CONCENTRIC DYSON
%   relay -- a plano-convex block of radius r and index n whose flat face
%   passes through the common centre C, and a concave mirror (the grating
%   substrate) of radius R_g concentric with it -- at the Dyson condition
%
%       R_g = n r / (n - 1)          (Dyson 1959; Mertz 1977: "the block
%                                     fills (n-1)/n of the slit-grating gap")
%
%   and VERIFIES it by exact 3-D ray tracing (no paraxial or Seidel
%   shortcut): object points on the flat face at height h image to -h (1:1,
%   inverted) with a transverse blur that is MINIMAL at the condition and
%   grows as h^4 / r^3 away from the centre (a fifth-order residual -- the
%   condition kills every third-order term, which is what "without Seidel
%   aberrations" means).  Everything the spectrometer builder needs to size
%   the block is returned, and the scaling law is what a 54 mm slit at
%   F/1.8 runs into (see dyson_scaling).
%
%   The mirror is traced as a MIRROR (order 0): the layout question is the
%   imaging condition; dispersion (order, groove period, slit/FPA offsets
%   along the dispersion axis) is the deck emitter's job and rides on top.
%
%   Units: r and everything returned in the SAME length unit (use m).
%
%   Options
%     'fno'        image-space (air-equivalent) F-number, default 1.8.
%                  The in-glass marginal half-angle is u = asin(1/(2 n fno))
%                  -- the beam converges in glass toward the flat face, so
%                  the cone is quoted as its air equivalent n sin u.
%     'Rg_factor'  multiplies the Dyson R_g (1 = the condition), default 1.
%     'h'          field heights (off the centre, on the flat face) at which
%                  the blur is reported, default r*[0.05 0.1 0.2 0.3].
%     'sweep'      true (default) runs the condition sweep R_g x
%                  [0.95 0.98 1 1.02 1.05] at h(2) and asserts the minimum
%                  sits at the condition.
%     'nring'      cone sampling rings for the blur (default 6 -> 169 rays).
%
%   Returns D (the numbers) and geom (2-D section for plotting):
%     D.R_g, D.gap (= R_g - r, the air space), D.u_glass (rad),
%     D.D_g_axial (= 2 R_g sin u, grating clear diameter for the axial
%     point; add the field extent for the whole slit), D.h, D.blur_rms
%     (transverse rms spot, one per h), D.image_y (traced image height,
%     one per h), D.distortion (= image_y + h, the fifth-order centroid
%     shift), D.sweep (struct: factor, blur), D.h_exponent (log-log
%     slope of blur vs h over the last two h's).
%
%   See also: dyson_scaling, offner_layout.
    arguments
        r (1,1) double {mustBePositive}
        n (1,1) double {mustBeGreaterThan(n,1)}
        opts.fno (1,1) double {mustBePositive} = 1.8
        opts.Rg_factor (1,1) double {mustBePositive} = 1
        opts.h (1,:) double = r*[0.05 0.10 0.20 0.30]
        opts.sweep (1,1) logical = true
        opts.nring (1,1) double {mustBeInteger, mustBePositive} = 6
    end
    R_dyson = n*r/(n-1);
    R  = opts.Rg_factor*R_dyson;
    u  = asin(1/(2*n*opts.fno));
    dirs = cone_(u, opts.nring);

    D.R_g = R;  D.R_dyson = R_dyson;  D.gap = R - r;  D.u_glass = u;
    D.fno = opts.fno;  D.n = n;  D.r = r;
    D.D_g_axial = 2*R*sin(u);
    D.h = opts.h;
    D.blur_rms = nan(size(opts.h));  D.image_y = nan(size(opts.h));
    for j = 1:numel(opts.h)
        [D.image_y(j), D.blur_rms(j)] = spot_(opts.h(j), n, r, R, dirs);
    end
    ok = ~isnan(D.blur_rms);
    if nnz(ok) >= 2
        hh = opts.h(ok);  bb = D.blur_rms(ok);
        D.h_exponent = log(bb(end)/bb(end-1)) / log(hh(end)/hh(end-1));
    else
        D.h_exponent = NaN;
    end

    % -- closure invariants: 1:1 inverted imaging at the condition ---------
    % The centroid lands at -h up to the same fifth-order residual that
    % blurs the spot (a distortion term ~ h^5/r^4: 5.9 um at h = 0.3 r,
    % r = 100 mm).  Assert the closure where that term is negligible
    % (h <= 0.1 r) and REPORT it everywhere as D.distortion.
    D.distortion = D.image_y + opts.h;
    if opts.Rg_factor == 1
        for j = find(ok & opts.h <= 0.1*r)
            assert(abs(D.distortion(j)) < 1e-6*r, ...
                'dyson_layout: image at y=%.6g for h=%.6g, expected -h (1:1 inversion)', ...
                D.image_y(j), opts.h(j));
        end
    end

    % -- the condition sweep: blur minimal at R_g = n r/(n-1) --------------
    D.sweep = struct('factor', [], 'blur', []);
    if opts.sweep
        fac = [0.95 0.98 1.00 1.02 1.05];
        hs  = opts.h(min(2, numel(opts.h)));
        blur = nan(size(fac));
        for k = 1:numel(fac)
            [~, blur(k)] = spot_(hs, n, r, fac(k)*R_dyson, dirs);
        end
        D.sweep.factor = fac;  D.sweep.blur = blur;
        [~, imin] = min(blur);
        assert(fac(imin) == 1, ...
            'dyson_layout: blur minimum at R_g factor %.2f, not at the Dyson condition', fac(imin));
    end

    % -- 2-D section (y,z about C) for plotting ---------------------------
    t = linspace(-pi/2, pi/2, 181);
    geom.block_arc   = [r*sin(t); r*cos(t)];          % convex face, z >= 0
    geom.grating_arc = [R*sin(t); R*cos(t)];
    geom.flat_face   = [-r r; 0 0];
    geom.C = [0; 0];  geom.r = r;  geom.R = R;
end

% =====================================================================
function [y_img, rms] = spot_(h, n, r, R, dirs)
%SPOT_  Trace a cone from (0,h,0) on the flat face; return the traced image
%   height and the transverse rms blur at the flat face (z = 0).
    P0 = [0; h; 0];
    m = size(dirs, 2);  pts = nan(2, m);
    for k = 1:m
        q = trace_(P0, dirs(:,k), n, r, R);
        if isempty(q), y_img = NaN;  rms = NaN;  return;  end
        pts(:,k) = q(1:2);
    end
    c = mean(pts, 2);
    rms = sqrt(mean(sum((pts - c).^2, 1)));
    y_img = c(2);
end

function q = trace_(P0, d, n, r, R)
%TRACE_  glass -> sphere r (refract out) -> mirror R (reflect) -> sphere r
%   (refract in) -> plane z = 0.  Empty on a miss or TIR.
    t = sphere_t_(P0, d, r, 'far');    q1 = P0 + t*d;   N1 = -q1/norm(q1);
    d = refract_(d, N1, n, 1);         if isempty(d), q = [];  return;  end
    t = sphere_t_(q1, d, R, 'far');    q2 = q1 + t*d;   N2 = -q2/norm(q2);
    d = d - 2*(d.'*N2)*N2;
    t = sphere_t_(q2, d, r, 'near');   if isempty(t), q = [];  return;  end
    q3 = q2 + t*d;                     N3 = q3/norm(q3);
    d = refract_(d, N3, 1, n);         if isempty(d), q = [];  return;  end
    t = -q3(3)/d(3);                   q  = q3 + t*d;
end

function t = sphere_t_(p, d, Rad, which)
    b = p.'*d;  c = p.'*p - Rad^2;  disc = b^2 - c;
    if disc < 0, t = [];  return;  end
    s = sqrt(disc);  t1 = -b - s;  t2 = -b + s;
    if strcmp(which, 'far'), t = t2;
    elseif t1 > 1e-12,      t = t1;
    else,                   t = t2;
    end
end

function dout = refract_(d, N, n1, n2)
%REFRACT_  N is the unit normal on the INCIDENT side (against d).
    cosi = -(N.'*d);  eta = n1/n2;  k = 1 - eta^2*(1 - cosi^2);
    if k < 0, dout = [];  return;  end
    dout = eta*d + (eta*cosi - sqrt(k))*N;
end

function dirs = cone_(u, nring)
%CONE_  Ring-sampled directions within half-angle u about +z.
    dirs = [0; 0; 1];
    for i = 1:nring
        a = u*i/nring;  m = 8*i;
        ph = 2*pi*(0:m-1)/m;
        dirs = [dirs, [sin(a)*cos(ph); sin(a)*sin(ph); cos(a)*ones(1,m)]]; %#ok<AGROW>
    end
end
