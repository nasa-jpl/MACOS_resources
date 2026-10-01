function G = spectrometer_geom(form, P)
%SPECTROMETER_GEOM  Concentric imaging-spectrometer geometry + exact tracer.
%   G = spectrometer_geom('dyson', P) / spectrometer_geom('offner', P)
%   builds the surface chain of a concentric spectrometer about the common
%   centre C = origin (axis z, slit along x, dispersion along y), SOLVES the
%   three things a seed leaves open -- the chief aim through the grating
%   vertex (the stop), the groove period d that lays the band across the
%   FPA's spectral height, and the FPA focus plane -- by EXACT 3-D ray
%   tracing of that chain (no paraxial shortcut), and returns everything
%   spectrometer_rx needs to emit the MACOS deck plus a tracer handle the
%   gates use to predict what the engine must do with the same deck.
%
%   Forms (all lengths m, angles rad; see SPECTROMETER_DESIGN_REFERENCE.md)
%   'dyson'  JPL form: plano-convex block (radius P.block_r, glass P.glass
%            with Sellmeier index n(lambda)), flat face at z = P.face_offset
%            (the slit and FPA sit in AIR at z = 0, a hair in front of the
%            face; the classical seed has the face THROUGH C), air gap,
%            concave grating of radius P.Rg_factor * n(lambda_ref) r/(n-1)
%            (Dyson condition at factor 1), concentric; grooves along x.
%            Chain: flat(1->n) | sphere r (n->1) | grating (reflect,
%            order m) | sphere r (1->n) | flat (n->1) | FPA plane.
%   'offner' all-reflective: concave sphere P.offner_R used twice, convex
%            grating of radius R/2 at the stop (vertex on the -z axis),
%            slit at ring radius P.y_slit, image at -y_slit (m = 0).
%            Chain: concave | grating | concave | FPA plane (z = 0).
%
%   Required P fields: Fno (air-equivalent image-space), pixel_m, npix
%   [spatial spectral], band_m [min max], lambda_ref_m (index + layout
%   wavelength), order (|m|, sign solved so dispersion pushes the FPA AWAY
%   from the slit), y_slit (slit-centre offset from the axis), and per form
%   block_r / glass / face_offset / Rg_factor  or  offner_R.  Optional Dyson
%   departures: block_Kc / block_asph (face figure), block_dz / block_dy
%   (centre off the grating's), men_z / men_t / men_ca / men_cb (a meniscus
%   corrector in the air gap: vertex z, thickness, face curvatures).  Optional
%   P.grating_model = 'planes' (DEFAULT since the engine fix of 2026-09-30:
%            straight-ruled, equidistant groove PLANES, period constant along
%            the chord -- what elemsub.F Snells_Law_Grating now traces) |
%            'surface' (period constant ALONG the surface: the PRE-FIX engine,
%            kept for the beat-2 record's two-column comparison).
%
%   Returns G with .surf (the chain: struct array with .kind 'plane'|
%   'sphere', .C centre, .R radius, .n_out, .act 'refract'|'reflect'|
%   'grating'|'stop', .root 'far'|'near', .vpt vertex point, .psi (toward
%   the CoC for spheres; against the incoming beam for planes), .name,
%   .glass), .src (slit point, chief dir, cone half-angle u), .grating
%   (index into surf, m, d, groove dir), .fpa (centre, z plane, axes),
%   .n (function handle n(lambda)), and .trace = @(p0, d, lambda) ->
%   [hit points per surface, final point on the FPA plane].
    arguments
        form (1,:) char {mustBeMember(form, {'dyson','offner'})}
        P struct
    end
    G.form = form;
    G.n = @(lam) sellmeier_(P.glass, lam);
    lam_c = mean(P.band_m);
    u_air = asin(1/(2*P.Fno));                 % cone half-angle at the slit, in air
    G.src.u = u_air;  G.src.lambda_c = lam_c;
    H_fpa = P.npix(2)*P.pixel_m;

    switch form
    case 'dyson'
        n0 = G.n(P.lambda_ref_m);
        r  = P.block_r;  Rg = P.Rg_factor*n0*r/(n0-1);  dz = P.face_offset;
        ys = P.y_slit;
        S = struct('kind',{},'C',{},'R',{},'n_out',{},'act',{},'root',{}, ...
                   'vpt',{},'psi',{},'name',{},'glass',{},'Kc',{},'A',{});
        S(1) = plane_([0;0;dz], [0;0;-1], 'glass', 'refract', 'BlockFaceIn', P.glass);
        % the block's centre may leave the grating's (de-concentric departure):
        % P.block_dz along the axis, P.block_dy along the dispersion direction
        Cb = [0; 0; 0];
        if isfield(P, 'block_dz'), Cb(3) = P.block_dz; end
        if isfield(P, 'block_dy'), Cb(2) = P.block_dy; end
        S(2) = sphere_(Cb, r, 1, 'refract', 'far', 'BlockSphereOut', '', Cb + [0;0;r]);
        S(3) = sphere_([0;0;0], Rg, 1, 'grating', 'far', 'Grating', '', [0;0;Rg]);
        S(4) = sphere_(Cb, r, 'glass', 'refract', 'near', 'BlockSphereIn', P.glass, Cb + [0;0;r]);
        % the block's convex face may depart from the sphere: conic constant
        % P.block_Kc and even-asphere coefficients P.block_asph (engine
        % AsphCoef convention: coef(i) multiplies h^(2i+2) of the sag along
        % +psi, h the height off the vertex axis) -- Carbon-I's departure
        Kc_b = 0;  A_b = [];
        if isfield(P, 'block_Kc'), Kc_b = P.block_Kc; end
        if isfield(P, 'block_asph'), A_b = P.block_asph(:)'; end
        if Kc_b ~= 0 || any(A_b ~= 0)
            S(2).kind = 'asph';  S(2).Kc = Kc_b;  S(2).A = A_b;
            S(4).kind = 'asph';  S(4).Kc = Kc_b;  S(4).A = A_b;
        end
        S(5) = plane_([0;0;dz], [0;0;1], 1, 'refract', 'BlockFaceOut', '');
        S(6) = plane_([0;0;0], [0;0;1], 1, 'stop', 'FPA', '');
        % R4, the compact variant's MENISCUS corrector in the air gap: two
        % spherical faces A (vertex z_a) and B (vertex z_a + t_m) of the block
        % glass, traversed outward (air->glass->air) and back.  Surface
        % curvatures c_a, c_b (1/m, centre at vertex + 1/c along +z; a
        % concentric shell, c = 1/z_vertex, is a null and the natural seed).
        if isfield(P, 'men_z') && ~isempty(P.men_z) && P.men_z > 0
            za = P.men_z;  tm = P.men_t;  ca = P.men_ca;  cb = P.men_cb;
            Ma_out = msph_(za, ca, 'glass', P.glass, 'MenA_out');
            Mb_out = msph_(za + tm, cb, 1, '', 'MenB_out');
            Mb_in  = msph_(za + tm, cb, 'glass', P.glass, 'MenB_in');
            Ma_in  = msph_(za, ca, 1, '', 'MenA_in');
            S = [S(1:2), Ma_out, Mb_out, S(3), Mb_in, Ma_in, S(4:6)];
            iG = 5;
        else
            iG = 3;
        end
        G.Rg = Rg;  G.r = r;  G.gap = Rg - r;
        slit = [0; ys; 0];
    case 'offner'
        R = P.offner_R;  ys = P.y_slit;
        % the convex grating radius: R/2 exactly is the concentric seed, which
        % carries astigmatism growing with the ring radius; Offner's correction
        % makes it slightly different (P.offner_Rg_factor x R/2)
        fg = 1;  if isfield(P, 'offner_Rg_factor'), fg = P.offner_Rg_factor; end
        S = struct('kind',{},'C',{},'R',{},'n_out',{},'act',{},'root',{}, ...
                   'vpt',{},'psi',{},'name',{},'glass',{},'Kc',{},'A',{});
        % the second concave zone (M3) may differ from the first: its own
        % radius factor and a centre offset (dy along the dispersion, dz) --
        % the classical Offner spectrometer's corrections
        fm3 = 1;  C3 = [0;0;0];
        if isfield(P, 'offner_M3_factor'), fm3 = P.offner_M3_factor; end
        if isfield(P, 'offner_M3_dy'), C3(2) = P.offner_M3_dy; end
        if isfield(P, 'offner_M3_dz'), C3(3) = P.offner_M3_dz; end
        S(1) = sphere_([0;0;0], R,   1, 'reflect', 'far',  'M1', '', [0;0;0]);
        S(2) = sphere_([0;0;0], fg*R/2, 1, 'grating', 'near', 'Grating', '', [0;0;-fg*R/2]);
        S(3) = sphere_(C3, fm3*R, 1, 'reflect', 'far',  'M3', '', C3);
        S(4) = plane_([0;0;0], [0;0;-1], 1, 'stop', 'FPA', '');
        % M1/M3 vertices: the chief hit points (set after the aim solve)
        G.R = R;
        iG = 2;  slit = [0; ys; 0];
    end
    G.iG = iG;  G.slit = slit;  G.y_slit = ys;
    G.grating.groove = [1;0;0];                 % grooves along the slit
    G.grating.sdir   = [0;1;0];                 % dispersion direction (projected per hit)
    G.grating.m = 0;  G.grating.d = Inf;        % order 0 while aiming / focusing
    if isfield(P, 'grating_model'), G.grating.model = P.grating_model; else, G.grating.model = 'planes'; end   % default 'planes' since the engine fix of 2026-09-30 (chord-ruled grooves); 'surface' = the pre-fix engine

    % -- chief aim: the ray from the slit centre through the grating vertex
    d0 = aim_(S, slit, S(iG).vpt, G, lam_c, iG);
    G.src.chief_dir = d0;
    if strcmp(form, 'offner')                   % vertices at the chief hits
        [pts] = trace_chain_(S, slit, d0, lam_c, G);
        S(1).vpt = pts(:,1);  S(1).psi = -pts(:,1)/norm(pts(:,1));
        S(3).vpt = pts(:,3);  S(3).psi = (S(3).C - pts(:,3))/norm(S(3).C - pts(:,3));
    end
    G.surf = S;

    % -- order sign + groove period: band across the FPA, away from the slit
    y_of = @(lam, m, d) img_y_(S, slit, d0, lam, G, m, d);
    y0 = y_of(lam_c, 0, Inf);                   % m = 0 image of the slit centre
    % trial: m = +|m| with a coarse d; sign chosen so the lambda_c image is
    % farther from the slit than the m = 0 image
    d_try = 50e-6;
    yp = y_of(lam_c, +P.order, d_try);  ym = y_of(lam_c, -P.order, d_try);
    if abs(yp - ys) > abs(ym - ys), m = +P.order; else, m = -P.order; end
    % secant on d: |y(lambda_max) - y(lambda_min)| = H_fpa
    f = @(d) abs(y_of(P.band_m(2), m, d) - y_of(P.band_m(1), m, d)) - H_fpa;
    d1 = d_try;  d2 = d_try*2;  f1 = f(d1);  f2 = f(d2);
    for it = 1:60
        d3 = d2 - f2*(d2 - d1)/(f2 - f1);
        if d3 <= 0, d3 = 0.5*d2; end
        d1 = d2;  f1 = f2;  d2 = d3;  f2 = f(d2);
        if abs(f2) < 1e-12, break; end
    end
    G.grating.m = m;  G.grating.d = d2;
    G.grating.lines_per_mm = 1e-3/d2;
    G.fpa.y_lambda = [y_of(P.band_m(1), m, d2), y_of(lam_c, m, d2), y_of(P.band_m(2), m, d2)];
    G.fpa.y0_order0 = y0;

    % -- focus: FPA plane z that minimises the slit-centre blur at lambda_c
    zf = focus_(S, slit, d0, lam_c, G, u_air);
    G.fpa.z = zf;
    S(end).C = [0; 0; zf];
    G.surf = S;
    G.fpa.center = [0; G.fpa.y_lambda(2); zf];
    G.fpa.xhat = [1;0;0];  G.fpa.yhat = [0;1;0];  G.fpa.normal = S(end).psi;
    G.fpa.H = H_fpa;  G.fpa.W = P.npix(1)*P.pixel_m;
    G.fpa.clear_to_slit = abs(G.fpa.center(2) - ys) - H_fpa/2;
    % ChfRayPos is where the engine STARTS its rays (PtSource builds the grid
    % there and propagates forward), so it must lie between the slit and the
    % first surface; the source point is ChfRayPos + zSource*ChfRayDir with
    % zSource = -gap.  NB the engine's stop re-aim changes ChfRayDir and so
    % MOVES the source point by gap*(d_aimed - d_written): hand the engine
    % an already-aimed chief (G.aim) when the slit point must be exact.
    G.src.zsrc_gap = 0.2e-3;
    if strcmp(form, 'dyson')
        assert(G.src.zsrc_gap < P.face_offset, 'spectrometer_geom: face_offset must exceed the %.1e m ChfRayPos gap', G.src.zsrc_gap);
    end
    G.P = P;

    % -- tracer handle for the gates (chief or any ray) --------------------
    G.trace = @(p0, d, lam) trace_chain_(S, p0, d, lam, G);
    G.cone  = @(u, nring) cone_(u, nring);
    G.aim   = @(p0, lam) aim_(S, p0, S(iG).vpt, G, lam, iG);
    G.bundle = @(varargin) bundle_(G, varargin{:});
    G.footprints = @(varargin) footprints_(G, varargin{:});
end

function B = bundle_(G, opts)
%BUNDLE_  The multi-field, multi-lambda ray bundle through the chain: slit
%   centre + both ends (+ opts.nx extra points), band edges + centre, chief +
%   nring rings of marginals.  B.P (3, nRay, nSurf+1): slit point then every
%   surface hit; B.D the directions after each surface; B.ok.
    arguments
        G struct
        opts.nx (1,1) double = 3
        opts.nlam (1,1) double = 3
        opts.nring (1,1) double = 2
    end
    W = G.P.npix(1)*G.P.pixel_m;  xs = linspace(-W/2, W/2, opts.nx);
    lams = linspace(G.P.band_m(1), G.P.band_m(2), opts.nlam);
    dirs0 = cone_(G.src.u, opts.nring);
    nS = numel(G.surf);  P = [];  D = [];  ok = logical([]);  meta = [];
    for xs_ = xs
        slit = G.slit + [xs_; 0; 0];
        for lam = lams
            d0 = aim_(G.surf, slit, G.surf(G.iG).vpt, G, lam, G.iG);
            ez = d0;  ex = cross([0;1;0], ez);  ex = ex/norm(ex);  ey = cross(ez, ex);
            dd = ex*dirs0(1,:) + ey*dirs0(2,:) + ez*dirs0(3,:);
            for k = 1:size(dd, 2)
                [pts, dr, okk] = trace_chain_(G.surf, slit, dd(:,k), lam, G);
                if ~okk, continue; end
                P(:, end+1, :) = reshape([slit, pts], 3, 1, nS+1);           %#ok<AGROW>
                D(:, end+1, :) = reshape([dd(:,k), dr], 3, 1, nS+1);        %#ok<AGROW>
                ok(end+1, 1) = true;  meta(end+1, :) = [xs_, lam, k];     %#ok<AGROW>
            end
        end
    end
    B.P = P;  B.D = D;  B.ok = ok;  B.meta = meta;  B.nx = opts.nx;  B.nlam = opts.nlam;  B.nring = opts.nring;
end

function F = footprints_(G, opts)
%FOOTPRINTS_  Per-surface beam footprint in the surface's APERTURE frame
%   (x_ap = global x projected into the vertex tangent plane, y_ap =
%   psi x x_ap -- the engine's xObs/yObs with xObs = x written), about the
%   VERTEX: centre (xc, yc), enclosing radius about that centre, extents.
%   The emitter declares ApType Circular with ApVec = (radius + margin, xc,
%   yc); the clearance tool and the trade table size the bodies from it.
    arguments
        G struct
        opts.nx (1,1) double = 3
        opts.nlam (1,1) double = 3
        opts.nring (1,1) double = 2
    end
    B = bundle_(G, 'nx', opts.nx, 'nlam', opts.nlam, 'nring', opts.nring);
    nS = numel(G.surf);
    F = struct('xc', {}, 'yc', {}, 'radius', {}, 'xlim', {}, 'ylim', {}, 'xap', {}, 'yap', {}, 'n', {});
    for k = 1:nS
        S = G.surf(k);  psi = S.psi(:)/norm(S.psi);
        xap = [1;0;0] - ([1;0;0]'*psi)*psi;  xap = xap/norm(xap);  yap = cross(psi, xap);
        H = squeeze(B.P(:, :, k+1));  rho = H - S.vpt(:);
        px = (rho'*xap)';  py = (rho'*yap)';
        xc = 0.5*(min(px) + max(px));  yc = 0.5*(min(py) + max(py));
        F(k) = struct('xc', xc, 'yc', yc, 'radius', max(hypot(px - xc, py - yc)), ...
                      'xlim', [min(px) max(px)], 'ylim', [min(py) max(py)], 'xap', xap, 'yap', yap, 'n', numel(px));
    end
end

% =====================================================================
function s = msph_(zv, c, n_out, glass, name)
%MSPH_  Meniscus face: vertex on the axis at zv, curvature c (1/m, centre at
%   zv + 1/c along +z; |c| < 1e-9 -> a plane), root 'auto' (nearest positive).
    if abs(c) < 1e-9
        s = plane_([0;0;zv], [0;0;-1], n_out, 'refract', name, glass);
    else
        R = 1/abs(c);  C = [0; 0; zv + 1/c];
        s = sphere_(C, R, n_out, 'refract', 'auto', name, glass, [0;0;zv]);
    end
end

function s = plane_(C, normal, n_out, act, name, glass)
%PLANE_  C = a point on the plane; psi = the normal (against the incoming beam).
    s = struct('kind', 'plane', 'C', C(:), 'R', NaN, 'n_out', n_out, 'act', act, ...
               'root', '', 'vpt', C(:), 'psi', normal(:)/norm(normal), 'name', name, 'glass', glass, ...
               'Kc', 0, 'A', []);
end

function s = sphere_(C, R, n_out, act, root, name, glass, vpt)
%SPHERE_  Sphere of radius R about C; vpt = the vertex point used by the
%   emitter (on the sphere), psi = toward the centre of curvature.
    psi = C(:) - vpt(:);
    if norm(psi) < 1e-15, psi = [0;0;-1]; else, psi = psi/norm(psi); end
    s = struct('kind', 'sphere', 'C', C(:), 'R', R, 'n_out', n_out, 'act', act, ...
               'root', root, 'vpt', vpt(:), 'psi', psi, 'name', name, 'glass', glass, ...
               'Kc', 0, 'A', []);
end

function [pts, dirs, ok] = trace_chain_(S, p0, d, lam, G)
%TRACE_CHAIN_  Sequential exact trace through the chain; pts(:,k) = hit on
%   surface k, dirs(:,k) = direction AFTER surface k.  ok = false on a miss.
    nS = numel(S);  pts = nan(3, nS);  dirs = nan(3, nS);  ok = true;
    p = p0(:);  d = d(:)/norm(d);  n_cur = 1;
    for k = 1:nS
        s = S(k);
        switch s.kind
        case 'plane'
            N = s.psi(:)/norm(s.psi);
            t = -((p - s.C(:))'*N)/(d'*N);
            if ~(t > 0), ok = false; return; end
            q = p + t*d;
        case 'sphere'
            t = sphere_t_(p - s.C(:), d, s.R, s.root);
            if isempty(t), ok = false; return; end
            q = p + t*d;  N = (q - s.C(:))/norm(q - s.C(:));
        case 'asph'
            % axisymmetric conic + even asphere about the axis psi through the
            % vertex: F(q) = (q - vpt).psi - sag(h) = 0, sag along +psi (toward
            % the CoC), h = |(q - vpt) - ((q - vpt).psi) psi|.  Newton from the
            % base-sphere root.
            t = sphere_t_(p - s.C(:), d, s.R, s.root);
            if isempty(t), ok = false; return; end
            a = s.psi(:);
            for it = 1:30
                q = p + t*d;  v = q - s.vpt(:);  z = v'*a;  hv = v - z*a;  h = norm(hv);
                [sg, dsg] = sag_(h, s.R, s.Kc, s.A);
                F = z - sg;
                if h > 0, hh = hv/h; else, hh = zeros(3,1); end
                gradF = a - dsg*hh;
                dF = gradF'*d;
                t = t - F/dF;
                if abs(F) < 1e-14, break; end
            end
            q = p + t*d;  v = q - s.vpt(:);  z = v'*a;  hv = v - z*a;  h = norm(hv);
            [~, dsg] = sag_(h, s.R, s.Kc, s.A);
            if h > 0, hh = hv/h; else, hh = zeros(3,1); end
            N = a - dsg*hh;  N = N/norm(N);
        end
        switch s.act
        case 'refract'
            n2 = idx_(s.n_out, G, lam);
            Ni = N;  if Ni'*d > 0, Ni = -Ni; end       % normal against the incoming ray
            d = refract_(d, Ni, n_cur, n2);
            if isempty(d), ok = false; return; end
            n_cur = n2;
        case 'reflect'
            d = d - 2*(d'*N)*N;
        case 'grating'
            % Two groove models.  'surface': the period d is constant ALONG
            % THE CURVED SURFACE -- the grating vector is m lambda/d times the
            % UNIT projection of sdir into the local tangent plane (this is
            % what the engine's Snells_Law_Grating does: it normalises shat).
            % 'planes': straight-ruled -- equidistant parallel groove PLANES
            % with spacing d along sdir, so the tangential kick is m lambda/d
            % times the UN-normalised projection (magnitude cos of the local
            % tilt), the classical Rowland concave grating.
            sdir = G.grating.sdir(:) - (G.grating.sdir(:)'*N)*N;
            if ~isfield(G.grating, 'model') || strcmp(G.grating.model, 'surface'), sdir = sdir/norm(sdir); end
            d_t = d - (d'*N)*N + (G.grating.m*lam/G.grating.d/n_cur)*sdir;
            val = 1 - norm(d_t)^2;
            if val < 0, ok = false; return; end
            d = d_t - sign(d'*N)*sqrt(val)*N;         % reflected: normal part flips
        case 'stop'
            % image plane: nothing
        end
        pts(:,k) = q;  dirs(:,k) = d;  p = q;
    end
end

function [sg, dsg] = sag_(h, R, Kc, A)
%SAG_  Conic (radius R > 0, constant Kc) + even asphere sag along +psi and
%   its h-derivative; A(i) multiplies h^(2i+2) (engine AsphCoef convention).
    c = 1/R;  h2 = h*h;  rt = sqrt(1 - (1+Kc)*c*c*h2);
    sg = c*h2/(1 + rt);
    dsg = c*h/rt;
    for i = 1:numel(A)
        sg  = sg + A(i)*h^(2*i+2);
        dsg = dsg + (2*i+2)*A(i)*h^(2*i+1);
    end
end

function n = idx_(spec, G, lam)
    if ischar(spec), n = G.n(lam); else, n = spec; end
end

function t = sphere_t_(p, d, Rad, which)
    b = p'*d;  c = p'*p - Rad^2;  disc = b^2 - c;
    if disc < 0, t = [];  return;  end
    s = sqrt(disc);  t1 = -b - s;  t2 = -b + s;
    if strcmp(which, 'far'), t = t2;
    elseif t1 > 1e-12,       t = t1;      % 'near' and 'auto': the nearest positive root
    else,                    t = t2;
    end
    if ~(t > 1e-12), t = []; end
end

function dout = refract_(d, N, n1, n2)
    cosi = -(N'*d);  eta = n1/n2;  k = 1 - eta^2*(1 - cosi^2);
    if k < 0, dout = [];  return;  end
    dout = eta*d + (eta*cosi - sqrt(k))*N;
end

function d0 = aim_(S, p0, target, G, lam, iG)
%AIM_  Launch direction from p0 whose ray hits surface iG at TARGET (Newton
%   on two angles about the straight-line direction).
    v = target(:) - p0(:);  d0 = v/norm(v);
    ex = [1;0;0];  ey = cross(d0, ex);  ey = ey/norm(ey);  ex = cross(ey, d0);
    th = [0;0];
    dir_of = @(th) rot_(d0, ex, ey, th);
    miss = @(th) miss_(S, p0, dir_of(th), lam, G, iG, target, ex, ey);
    for it = 1:40
        m0 = miss(th);
        if norm(m0) < 1e-13, break; end
        h = 1e-7;  J = zeros(2);
        J(:,1) = (miss(th + [h;0]) - m0)/h;  J(:,2) = (miss(th + [0;h]) - m0)/h;
        th = th - J\m0;
    end
    d0 = dir_of(th);
end

function d = rot_(d0, ex, ey, th)
    d = d0 + th(1)*ex + th(2)*ey;  d = d/norm(d);
end

function m = miss_(S, p0, d, lam, G, iG, target, ex, ey)
    Gm = G;  Gm.grating.m = 0;
    [pts, ~, ok] = trace_chain_(S(1:iG), p0, d, lam, Gm);
    if ~ok, m = [1;1]; return; end
    v = pts(:,iG) - target(:);
    m = [v'*ex; v'*ey];
end

function y = img_y_(S, p0, d0, lam, G, m, d)
    Gm = G;  Gm.grating.m = m;  Gm.grating.d = d;
    [pts, ~, ok] = trace_chain_(S, p0, d0, lam, Gm);
    assert(ok, 'spectrometer_geom: chief lost at lambda = %.4g (m=%d, d=%.3g)', lam, m, d);
    y = pts(2, end);
end

function zf = focus_(S, p0, d0, lam, G, u)
%FOCUS_  z of the FPA plane minimising the rms blur of the slit-centre cone.
    dirs = cone_(u, 4);  dirs = aimcone_(d0, dirs);
    blur = @(z) blur_(S, p0, dirs, lam, G, z);
    % golden-section on [-5, +5] mm about z = 0
    a = -5e-3;  b = 5e-3;  gr = (sqrt(5)-1)/2;
    c = b - gr*(b-a);  dd = a + gr*(b-a);  fc = blur(c);  fd = blur(dd);
    for it = 1:60
        if fc < fd, b = dd;  dd = c;  fd = fc;  c = b - gr*(b-a);  fc = blur(c);
        else,       a = c;   c = dd;  fc = fd;  dd = a + gr*(b-a); fd = blur(dd);
        end
        if abs(b-a) < 1e-9, break; end
    end
    zf = 0.5*(a+b);
end

function v = blur_(S, p0, dirs, lam, G, z)
    S(end).C = [0;0;z];
    q = nan(2, size(dirs,2));
    for k = 1:size(dirs,2)
        [pts, ~, ok] = trace_chain_(S, p0, dirs(:,k), lam, G);
        if ~ok, v = 1; return; end
        q(:,k) = pts(1:2, end);
    end
    c = mean(q, 2);  v = sqrt(mean(sum((q - c).^2, 1)));
end

function dirs = aimcone_(d0, dirs)
%AIMCONE_  Rotate +z-centred cone directions onto the chief d0.
    ez = d0(:)/norm(d0);  ex = cross([0;1;0], ez);
    if norm(ex) < 1e-12, ex = [1;0;0]; end
    ex = ex/norm(ex);  ey = cross(ez, ex);
    dirs = ex*dirs(1,:) + ey*dirs(2,:) + ez*dirs(3,:);
end

function dirs = cone_(u, nring)
    dirs = [0; 0; 1];
    for i = 1:nring
        a = u*i/nring;  m = 8*i;  ph = 2*pi*(0:m-1)/m;
        dirs = [dirs, [sin(a)*cos(ph); sin(a)*sin(ph); cos(a)*ones(1,m)]]; %#ok<AGROW>
    end
end

function n = sellmeier_(glass, lam_m)
%SELLMEIER_  The engine table's rows (macos_glass_list.txt), C in um^2.
    switch glass
        case 'Silica', B = [0.6961663 0.4079426 0.8974794];  C = [0.004679148 0.01351206 97.934];
        case 'CaF2',   B = [0.5675888 0.4710914 3.8484723];  C = [0.050263605^2 0.1003909^2 34.649040^2];
        otherwise, error('spectrometer_geom:glass', 'no Sellmeier row for %s', glass);
    end
    L2 = (lam_m*1e6)^2;
    n = sqrt(1 + sum(B .* L2 ./ (L2 - C)));
end
