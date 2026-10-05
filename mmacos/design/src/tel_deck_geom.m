function G = tel_deck_geom(deck, P, GD)
%TEL_DECK_GEOM  An exact chain from a COAXIAL telescope deck (macos.design.Telescope emission) with an object-space stop.
%   G = tel_deck_geom(DECK, P, GD) reads a Telescope-emitted deck whose
%   mirrors are coaxial conics (psiElt toward the centre of curvature,
%   KrElt = -|R|, VptElt on the parent axis) with a field BIAS (ChfRayDir)
%   and an object-space STOP point (ApStop -- e.g. the TMA step-2b eccentric
%   section's 205 mm pupil decentre), and builds the chain in
%   telescope_geom's schema so e2e_geom / spectrometer_rx /
%   spectrometer_score / spectrometer_clearance read it unchanged:
%     surfaces  the mirrors (root by geometry: psi against the arriving
%               beam = concave = the far wall), then the image plane where
%               the exit chief crosses the deck's FocalPlane;
%     the stop  NOT a surface (at an off-axis section the stop point can sit
%               behind M1's sag): every chief is the straight object-space
%               line through ApStop -- G.aim_pt and the slit-line field
%               solve aim that way;
%     placement as telescope_geom (GD = [] keeps the telescope alone with the
%               image at the origin and the exit chief along +z).
%   P: fov (full cross-track field, rad), name, lambda_c (default the deck's
%   Wavelen), standoff (launch plane ahead of the frontmost vertex, default
%   0.5 D), D_src (the launched bundle, default the deck's Aperture).
%   GD = []: the terminal plane is the DECK's FocalPlane (engine-faithful --
%   it may be tilted to the exit chief); with GD it is the Dyson's slit.  The cross-track field runs along x; the bias is the deck's
%   ChfRayDir angle in the y-z plane.
%   NOTE: chain_bundle aims through G.surf(G.iStop)'s vertex, which is not
%   this telescope's stop -- use it only inside a joined chain whose stop is
%   a surface (e2e_geom: the grating).
%
%   See also TELESCOPE_GEOM, TMS_GEOM, E2E_GEOM.
    arguments
        deck (1,:) char
        P struct
        GD = []
    end
    txt = fileread(deck);
    hdr = txt(1:strfind(txt, 'nElt=') - 1);
    cdir = vec_(hdr, 'ChfRayDir');  ep = vec_(hdr, 'ApStop');  D = num_(hdr, 'Aperture');
    lam_c = num_(hdr, 'Wavelen');  if isfield(P, 'lambda_c'), lam_c = P.lambda_c; end
    blocks = regexp(txt, '\n\s*iElt=', 'split');  blocks = blocks(2:end);   % line-anchored: psiElt= also contains 'iElt='
    S = struct('kind',{},'C',{},'R',{},'n_out',{},'act',{},'root',{}, 'vpt',{},'psi',{},'name',{},'glass',{},'Kc',{},'A',{});
    fp = [];
    for b = blocks
        bt = b{1};
        el = strtrim(regexp(bt, 'Element=\s*(\S+)', 'tokens', 'once'));  el = el{1};
        nm = regexp(bt, 'EltName=\s*(\S+)', 'tokens', 'once');  nm = nm{1};
        vpt = vec_(bt, 'VptElt');  psi = vec_(bt, 'psiElt');  psi = psi/norm(psi);
        if strcmp(el, 'Reflector')
            R = abs(num_(bt, 'KrElt'));  Kc = num_(bt, 'KcElt');  A = [];
            if ~isempty(regexp(bt, '(?m)^\s*AsphCoef=', 'once'))
                A = vec_(bt, 'AsphCoef')';
                if ~isempty(regexp(bt, '(?m)^\s*nAsphCoef=', 'once'))
                    nA = num_(bt, 'nAsphCoef');
                    assert(numel(A) == nA, 'tel_deck_geom: %s carries nAsphCoef=%d but its AsphCoef= line holds %d values (a wrapped line is not read)', nm, nA, numel(A));
                end
            end
            kind = 'sphere';  if Kc ~= 0 || any(A ~= 0), kind = 'asph'; end
            S(end+1) = struct('kind', kind, 'C', vpt + R*psi, 'R', R, 'n_out', 1, 'act', 'reflect', 'root', '', ...
                              'vpt', vpt, 'psi', psi, 'name', ['Tel' nm], 'glass', '', 'Kc', Kc, 'A', A); %#ok<AGROW>
        elseif strcmp(el, 'FocalPlane')
            fp = struct('vpt', vpt, 'psi', psi);
        end
    end
    assert(~isempty(fp), 'tel_deck_geom: no FocalPlane in %s', deck);
    % roots by geometry: trace the nominal (zero-field) chief through the vertices' axis
    standoff = D/2;  if isfield(P, 'standoff'), standoff = P.standoff; end
    zz = [S.vpt];  launch = struct('C', [0; 0; min([zz(3, :), ep(3)]) - standoff], 'N', [0; 0; 1]);
    Gl.n = @(lam) 1;  Gl.grating = struct('m', 0, 'd', Inf, 'lines_per_mm', 0, 'sdir', [0;1;0], 'model', 'planes');
    din = cdir/norm(cdir);
    for k = 1:numel(S)                                % psi against the arriving beam = concave = the far wall
        if S(k).psi'*din < 0, S(k).root = 'far'; else, S(k).root = 'near'; end
        din = din - 2*(din'*S(k).psi)*S(k).psi;  din = din/norm(din);
    end
    bias = atan2(cdir(2), cdir(3));                   % the deck's field bias in y-z
    dloc = @(thx, db) [sin(thx); cos(thx)*sin(bias + db); cos(thx)*cos(bias + db)];
    aimL = @(Sx, launchx, epx, d) epx - d*(((epx - launchx.C)'*launchx.N)/(d'*launchx.N));   % on the launch plane, along the line through the stop
    % the exit chief (zero field) and the image point on the deck's FocalPlane
    d0 = dloc(0, 0);  p0 = aimL(S, launch, ep, d0);
    [pts, dirs, ok] = chain_trace(S, p0, d0, lam_c, Gl);
    assert(ok, 'tel_deck_geom: the centre-field chief is lost');
    q3 = pts(:, end);  e_c = dirs(:, end);
    q_c = q3 + e_c*(((fp.vpt - q3)'*fp.psi)/(e_c'*fp.psi));
    S(end+1) = struct('kind', 'plane', 'C', q_c, 'R', NaN, 'n_out', 1, 'act', 'stop', 'root', '', ...
                      'vpt', q_c, 'psi', fp.psi, 'name', 'Slit', 'glass', '', 'Kc', 0, 'A', []);   % the DECK's focal plane (tilted to the chief); a join replaces it with the slit
    nS = numel(S);
    % ---- placement (telescope_geom's): exit chief -> the Dyson's aim, image -> the slit
    if ~isempty(GD), cd = GD.src.chief_dir(:);  slit = GD.slit(:); else, cd = [0; 0; 1];  slit = [0; 0; 0]; end
    Rm = rot_to_(e_c, cd);  tr = slit - Rm*q_c;
    for k = 1:nS, S(k).C = Rm*S(k).C + tr;  S(k).vpt = Rm*S(k).vpt + tr;  S(k).psi = Rm*S(k).psi; end
    launch.C = Rm*launch.C + tr;  launch.N = Rm*launch.N;  epP = Rm*ep + tr;
    if ~isempty(GD), S(nS).C = slit;  S(nS).vpt = slit;  S(nS).psi = [0; 0; -1]; end
    name = 'teldeck';  if isfield(P, 'name'), name = P.name; end
    G.form = 'teldeck';  G.name = name;  G.deck = deck;
    G.surf = S;  G.iStop = 1;  G.launch = launch;  G.slit = slit;  G.iSlit = nS;  G.ep = epP;
    G.place = struct('R', Rm, 'tr', tr, 'exit_dir_local', e_c, 'image_local', q_c);
    Dsrc = D;  if isfield(P, 'D_src'), Dsrc = P.D_src; end   % the launched (overfilling) bundle when the grating is the stop
    G.src = struct('fov', P.fov, 'D', D, 'D_src', Dsrc, 'bias', bias, 'lambda_c', lam_c, 'chief_dir', cd, 'zsrc', 1e22);
    G.P = P;  G.n = Gl.n;  G.grating = Gl.grating;
    G.field_dir_raw = @(thx, db) Rm*dloc(thx, db);
    G.axis = Rm*[0; 0; 1];  G.station0 = 'Sky';
    G.field_dir = @(thx) field_on_slit_(S, launch, epP, Rm, dloc, slit, lam_c, Gl, thx, aimL);
    if ~isempty(GD)
        W = GD.P.npix(1)*GD.P.pixel_m;  xs = [-W/2 0 W/2];
        Am = zeros(3);  bb = zeros(3, 1);  dirs3 = zeros(3, 3);
        for i = 1:3
            pi_ = slit + [xs(i); 0; 0];  di = GD.aim(pi_, GD.src.lambda_c);  dirs3(:, i) = di;
            Mk = eye(3) - di*di';  Am = Am + Mk;  bb = bb + Mk*pi_;
        end
        xp = Am\bb;
        G.pupil = struct('point', xp, 'L_app', (xp - slit)'*cd, 'aims', dirs3, 'xs', xs, 'edge_angle_deg', acosd(dirs3(:, 1)'*dirs3(:, 2)));
        G.req_dir = @(x) GD.aim(slit + [x; 0; 0], GD.src.lambda_c);
        G.dyson = struct('P', GD.P, 'slit', GD.slit, 'chief_dir', GD.src.chief_dir, 'lambda_c', GD.src.lambda_c, 'u', GD.src.u, ...
                         'trace', GD.trace, 'iG', GD.iG, 'vptG', GD.surf(GD.iG).vpt, 'surf', GD.surf);
    else
        G.pupil = struct('point', [NaN;NaN;NaN], 'L_app', Inf, 'aims', [], 'xs', [], 'edge_angle_deg', 0);
        G.req_dir = @(x) cd;
    end
    G.trace = @(p0, d, lam) chain_trace(S, p0, d, lam, G);
    G.aim_pt = @(d, lam) deal_ok_(aimL(S, launch, epP, d/norm(d)));
    G.bundle = @(varargin) chain_bundle(G, varargin{:});
    G.footprints = @(varargin) chain_footprints(G, chain_bundle(G, varargin{:}));
end

function [p, ok] = deal_ok_(p), ok = all(isfinite(p)); end

function [d, db1] = field_on_slit_(S, launch, ep, Rm, dloc, slit, lam, Gl, thx, aimL)
%FIELD_ON_SLIT_  The sky direction at field thx whose chief (through the stop point) lands on the slit line.
    v_of = @(db) vhit_(S, launch, ep, Rm*dloc(thx, db), slit, lam, Gl, aimL);
    db = 0;  v0 = v_of(0);
    if isnan(v0), d = Rm*dloc(thx, 0);  db1 = NaN; return; end
    db1 = 1e-3;  v1 = v_of(db1);
    for it = 1:12
        if isnan(v1) || v1 == v0, break; end
        db2 = db1 - v1*(db1 - db)/(v1 - v0);
        db = db1;  v0 = v1;  db1 = db2;  v1 = v_of(db1);
        if abs(v1) < 1e-12, break; end
    end
    d = Rm*dloc(thx, db1);
end

function v = vhit_(S, launch, ep, d, slit, lam, Gl, aimL)
    d = d/norm(d);  p0 = aimL(S, launch, ep, d);
    [pts, ~, okt] = chain_trace(S, p0, d, lam, Gl);
    if ~okt, v = NaN; return; end
    v = pts(2, end) - slit(2);
end

function R = rot_to_(a, b)
%ROT_TO_  The rotation about x taking direction a to b (both in the y-z plane, as telescope_geom's placement).
    ang = atan2(a(2)*b(3) - a(3)*b(2), a(2)*b(2) + a(3)*b(3));
    R = [1 0 0; 0 cos(ang) -sin(ang); 0 sin(ang) cos(ang)];
end

function v = vec_(t, key)
    % LINE-ANCHORED: 'AsphCoef=' must not match inside 'nAsphCoef=' (nor 'iElt=' inside 'psiElt=') -- the parse read
    % nAsphCoef's '2' as the h^4 coefficient and the bridge gate caught it (2026-10-04)
    m = regexp(t, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens', 'once');  v = sscanf(strrep(m{1}, 'D', 'E'), '%f');
end

function x = num_(t, key)
    v = vec_(t, key);  x = v(1);
end
