function G = tms_geom(P, GD)
%TMS_GEOM  The two-mirror modified Schwarzschild (TMS) that feeds the slit, as an exact chain.
%   G = tms_geom(P, GD) builds, in telescope_geom's schema (so chain_trace,
%   chain_bundle, chain_footprints, telescope_score_chain, spectrometer_rx,
%   telescope_score, e2e_geom and spectrometer_clearance read it unchanged),
%   the review's TMS (Mouroulis & Green 2018 sec. 5.1): a CONVEX primary and a
%   CONCAVE secondary on one parent axis, the stop VIRTUAL behind the primary
%   (at the secondary's front focal point, so the output is telecentric), an
%   inverted telephoto.  The stop is represented by its image in OBJECT space
%   -- the entrance pupil, a 'pass' plane ahead of M1 through whose centre
%   every chief is aimed (G.iStop = 1; the emitted deck's Reference stop).
%   Cross-track field along x (the slit), the used section selected by a field
%   BIAS about x and/or an entrance-pupil DECENTRE in y (the TMS's own
%   unobscuring freedoms), then PLACED in the spectrometer's frame exactly as
%   telescope_geom does (GD = [] builds the telescope alone: image at the
%   origin, exit chief along +z).
%
%   P (metres, radians): f, D (entrance-pupil diameter), fov, bias, R [R1 R2]
%   (|radii|), d (M1 -> M2), t2 (M2 -> image along the axis), z_ep (the
%   entrance pupil's axial position, signed, < 0 = ahead of M1), dec_ep (its
%   y decentre), Kc [2], A (2 x n even aspheres, engine AsphCoef), dec [2]
%   (mirror y decentres), D_src, standoff, lambda_c, name; fold_d / fold_dir
%   as telescope_geom (a flat fold in the M2 -> image leg).
%   TMS_PARAXIAL gives the closed-form first order (R, d, t2, z_ep).
%
%   See also TMS_PARAXIAL, TELESCOPE_GEOM, E2E_GEOM, TELESCOPE_SCORE_CHAIN.
    arguments
        P struct
        GD = []
    end
    f = P.f;  D = P.D;
    R = P.R(:)';  t = [P.d, P.t2];
    Kc = fld_(P, 'Kc', [0 0]);  A = fld_(P, 'A', zeros(2, 2));
    dec = fld_(P, 'dec', [0 0]);  z_ep = P.z_ep;  dec_ep = fld_(P, 'dec_ep', 0);
    bias = fld_(P, 'bias', 0);  D_src = fld_(P, 'D_src', D);
    standoff = fld_(P, 'standoff', 0.5*D);  lam_c = fld_(P, 'lambda_c', 1.0e-6);
    name = fld_(P, 'name', 'tms');
    if size(A, 1) ~= 2, A = reshape(A, 2, []); end
    assert(all(isfinite([R t Kc A(:)' dec bias z_ep dec_ep])), 'tms_geom: a non-finite parameter (R %s, d/t2 %s, Kc %s)', mat2str(R, 4), mat2str(t, 4), mat2str(Kc, 4));

    % ---- the local chain: the entrance pupil ('pass', the virtual stop's
    % object-space image), the convex M1 at the origin (psi along the beam,
    % toward its centre of curvature; the NEAR wall), the concave M2 t(1)
    % back along the reflected beam (psi against the beam; the FAR wall),
    % the image t(2) beyond M2.  Light arrives along +z.
    S = struct('kind',{},'C',{},'R',{},'n_out',{},'act',{},'root',{}, ...
               'vpt',{},'psi',{},'name',{},'glass',{},'Kc',{},'A',{});
    pep = [0; dec_ep; z_ep];
    S(1) = struct('kind', 'plane', 'C', pep, 'R', NaN, 'n_out', 1, 'act', 'pass', 'root', '', ...
                  'vpt', pep, 'psi', [0; 0; -1], 'name', 'TelEP', 'glass', '', 'Kc', 0, 'A', []);
    rotx = @(a) [1 0 0; 0 cos(a) -sin(a); 0 sin(a) cos(a)];
    din = [0; 0; 1];  v = [0; 0; 0];  sgn = [+1, -1];  roots = {'near', 'far'};
    for k = 1:2
        a = sgn(k)*din;                               % psi toward the centre of curvature
        vpt = v + [0; dec(k); 0];
        C = vpt + R(k)*a;
        kind = 'sphere';  if Kc(k) ~= 0 || any(A(k, :) ~= 0), kind = 'asph'; end
        S(k+1) = struct('kind', kind, 'C', C, 'R', R(k), 'n_out', 1, 'act', 'reflect', 'root', roots{k}, ...
                        'vpt', vpt, 'psi', a, 'name', sprintf('TelM%d', k), 'glass', '', 'Kc', Kc(k), 'A', A(k, :));
        dout = din - 2*(din'*a)*a;  dout = dout/norm(dout);
        v = v + t(k)*dout;  din = dout;
    end
    z = [S.vpt];  z = z(3, :);
    launch = struct('C', [0; 0; min([z, 0]) - standoff], 'N', [0; 0; 1]);
    dloc = @(thx, db) [sin(thx); -cos(thx)*sin(bias + db); cos(thx)*cos(bias + db)];   % field thx along the slit, at the bias (+ db)
    Gl.n = @(lam) 1;  Gl.grating = struct('m', 0, 'd', Inf, 'lines_per_mm', 0, 'sdir', [0;1;0], 'model', 'planes');
    fold_d = fld_(P, 'fold_d', 0);  fold_dir = fld_(P, 'fold_dir', [0; 1; 0]);  fold_dir = fold_dir(:)/norm(fold_dir);

    % ---- the centre-field chief in the local frame: through the entrance
    % pupil's centre (the virtual stop), M1, M2 and on toward the image; the
    % fold (if any) sits fold_d down that leg; the image plane is t2 from M2
    % by path, normal against the arriving chief
    [p0c, okc] = chain_aim(S, launch, dloc(0, 0), 1, lam_c, Gl);
    assert(okc, 'tms_geom: the centre-field chief does not reach the entrance pupil''s centre');
    [pts, dirs, okt] = chain_trace(S(1:3), p0c, dloc(0, 0), lam_c, Gl);
    assert(okt, 'tms_geom: the centre-field chief is lost before M2');
    q3 = pts(:, 3);  d3 = dirs(:, 3);                % the chief leaving M2
    if fold_d > 0
        assert(fold_d < t(2), 'tms_geom: the fold (%.1f mm after M2) must sit before the image (t2 = %.1f mm)', fold_d*1e3, t(2)*1e3);
        qf = q3 + fold_d*d3;  nf = fold_dir - d3;  nf = nf/norm(nf);    % psi = unit(d_out - d_in): faces the beam
        S(4) = struct('kind', 'plane', 'C', qf, 'R', NaN, 'n_out', 1, 'act', 'reflect', 'root', '', ...
                      'vpt', qf, 'psi', nf, 'name', 'TelFold', 'glass', '', 'Kc', 0, 'A', []);
        q_c = qf + (t(2) - fold_d)*fold_dir;  e_c = fold_dir;
    else
        q_c = q3 + t(2)*d3;  e_c = d3;
    end
    S(end+1) = struct('kind', 'plane', 'C', q_c, 'R', NaN, 'n_out', 1, 'act', 'stop', 'root', '', ...
                      'vpt', q_c, 'psi', -e_c, 'name', 'Slit', 'glass', '', 'Kc', 0, 'A', []);
    nS = numel(S);

    % ---- placement: rotate about x so the exit chief is the Dyson's aim, and
    % translate the image point onto the slit
    if ~isempty(GD)
        cd = GD.src.chief_dir(:);  slit = GD.slit(:);
    else
        cd = [0; 0; 1];  slit = [0; 0; 0];
    end
    ang = atan2(e_c(2)*cd(3) - e_c(3)*cd(2), e_c(2)*cd(2) + e_c(3)*cd(3));   % from e_c to cd, about x
    Rm = rotx(ang);  tr = slit - Rm*q_c;
    for k = 1:nS
        S(k).C = Rm*S(k).C + tr;  S(k).vpt = Rm*S(k).vpt + tr;  S(k).psi = Rm*S(k).psi;
    end
    launch.C = Rm*launch.C + tr;  launch.N = Rm*launch.N;
    % the terminal plane is the SPECTROMETER's slit plane (normal against the
    % beam), through the slit point
    if ~isempty(GD)
        S(nS).C = slit;  S(nS).vpt = slit;  S(nS).psi = [0; 0; -1];
    end
    G.form = 'tms';  G.name = name;
    G.surf = S;  G.iStop = 1;  G.launch = launch;  G.slit = slit;  G.iSlit = nS;
    G.place = struct('ang', ang, 'R', Rm, 'tr', tr, 'exit_dir_local', e_c, 'image_local', q_c);
    G.src = struct('fov', P.fov, 'D', D, 'D_src', D_src, 'bias', bias, 'dec_ep', dec_ep, 'lambda_c', lam_c, 'chief_dir', cd, 'zsrc', 1e22);
    G.P = P;  G.n = Gl.n;  G.grating = Gl.grating;
    G.field_dir_raw = @(thx, db) Rm*dloc(thx, db);   % the sky direction (thx along the slit, the bias + db across it)
    G.axis = Rm*[0; 0; 1];                          % the parent axis, placed
    G.station0 = 'Sky';
    % THE FIELD LINE: a push-broom field is whatever curve on the sky images
    % onto the STRAIGHT slit, so each field angle thx along the slit is paired
    % with the across-slit angle (bias + db) whose chief lands ON the slit
    % line (v = 0).  field_dir(thx) solves db by secant on the chief (two or
    % three traces) and caches it; the raw direction at db = 0 is
    % field_dir_raw.  The image of a straight line on the sky is then
    % curved -- the pushbroom's orthorectified truth, not an aberration
    % (Mouroulis & Green: "the image of the slit on the ground is curved").
    G.field_dir = @(thx) field_on_slit_(S, launch, Rm, dloc, slit, 1, lam_c, Gl, thx);

    % ---- the spectrometer's apparent entrance pupil, seen from the slit: the
    % Dyson's aim lines from the slit centre and ends cross (least squares)
    % L_app beyond the slit along the chief -- the pupil-match target
    if ~isempty(GD)
        W = GD.P.npix(1)*GD.P.pixel_m;  xs = [-W/2 0 W/2];
        Am = zeros(3);  bb = zeros(3, 1);  dirs3 = zeros(3, 3);
        for i = 1:3
            pi_ = slit + [xs(i); 0; 0];  di = GD.aim(pi_, GD.src.lambda_c);  dirs3(:, i) = di;
            Mk = eye(3) - di*di';  Am = Am + Mk;  bb = bb + Mk*pi_;
        end
        xp = Am\bb;
        G.pupil = struct('point', xp, 'L_app', (xp - slit)'*cd, 'aims', dirs3, 'xs', xs, ...
                         'edge_angle_deg', acosd(dirs3(:, 1)'*dirs3(:, 2)));
        G.req_dir = @(x) GD.aim(slit + [x; 0; 0], GD.src.lambda_c);
        G.dyson = struct('P', GD.P, 'slit', GD.slit, 'chief_dir', GD.src.chief_dir, 'lambda_c', GD.src.lambda_c, 'u', GD.src.u, ...
                         'trace', GD.trace, 'iG', GD.iG, 'vptG', GD.surf(GD.iG).vpt, 'surf', GD.surf);
    else
        G.pupil = struct('point', [NaN;NaN;NaN], 'L_app', Inf, 'aims', [], 'xs', [], 'edge_angle_deg', 0);
        G.req_dir = @(x) cd;
    end

    % ---- handles (G is complete; the handles capture this copy)
    G.trace = @(p0, d, lam) chain_trace(S, p0, d, lam, G);
    G.aim_pt = @(d, lam) chain_aim(S, launch, d, 1, lam, G);
    G.bundle = @(varargin) chain_bundle(G, varargin{:});
    G.footprints = @(varargin) chain_footprints(G, chain_bundle(G, varargin{:}));
end

function d = field_on_slit_(S, launch, Rm, dloc, slit, iStop, lam, Gl, thx)
%FIELD_ON_SLIT_  The sky direction at field thx whose chief lands on the slit line.
    % the chain here is the PLACED one (S, launch), so trace in global
    % coordinates; v = the chief's y at the slit plane minus the slit's y
    db = 0;  v_of = @(db) vhit_(S, launch, Rm*dloc(thx, db), slit, iStop, lam, Gl);
    v0 = v_of(0);
    if isnan(v0), d = Rm*dloc(thx, 0); return; end
    db1 = 1e-3;  v1 = v_of(db1);
    for it = 1:12
        if isnan(v1) || v1 == v0, break; end
        db2 = db1 - v1*(db1 - db)/(v1 - v0);
        db = db1;  v0 = v1;  db1 = db2;  v1 = v_of(db1);
        if abs(v1) < 1e-12, break; end
    end
    d = Rm*dloc(thx, db1);
end

function v = vhit_(S, launch, d, slit, iStop, lam, Gl)
    [p0, ok] = chain_aim(S, launch, d, iStop, lam, Gl);
    if ~ok, v = NaN; return; end
    [pts, ~, okt] = chain_trace(S, p0, d, lam, Gl);
    if ~okt, v = NaN; return; end
    v = pts(2, end) - slit(2);
end

function v = fld_(P, f, d)
    if isfield(P, f) && ~isempty(P.(f)), v = P.(f); else, v = d; end
end
