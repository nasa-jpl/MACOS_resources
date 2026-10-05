function S = dyson5_t5f(P, tag, q)
%DYSON5_T5F  End to end for a telescope deck the chain cannot model (Surface=FreeForm): the join placed from ENGINE traces.
%   Addendum 44 (CC's call, 2026-10-05).  t5e's join is built from tel_deck_geom's exact chain and gated chain == engine;
%   a FreeForm telescope (two Zernike channels) has no chain model yet, so here the ENGINE is the only model:
%     (1) the centre chief through the deck's ApStop is traced to the deck's FocalPlane: image point q_c, exit chief e_c;
%     (2) the placement is t5e's -- rotation about x taking e_c onto the Dyson's chief (+ P.tel5e_roll_deg about it),
%         the image onto the slit -- and is applied to the TELESCOPE deck's text (every point / direction keyword, the
%         TElt / Tout frames, the Mon / FF frames); the Slit + Dyson blocks are taken VERBATIM from a t5e end-to-end deck
%         of the same Dyson (P.tel5f_e2e_template), which already sits in the Dyson's frame;
%     (3) spectrometer_score runs on the joined deck with ENGINE launches: per field the sky direction whose chief lands on
%         the slit line (secant on the telescope deck), then the chief aimed through the grating by the engine's own
%         macos.stop(iG) (the instrument's stop) -- no chain anywhere.
%   GATE (replaces the bridge gate): on a CONIC/aspheric telescope deck the same join must reproduce t5e's row.
%   q (optional, addendum 45): q.quiet = true -> no record files, no prints (the e2e residual of tGM (c));
%   q.GD = the Dyson geometry (cached); q.GE0 = an e2e chain (tel_deck_geom + e2e_geom of an ASPHERIC deck of the same
%   Dyson) -> the LIVE CLEARANCE: spectrometer_clearance VERBATIM on the joined deck with its bundle and footprints from
%   ENGINE rays (tEC_ below) and the telescope bodies at the ENGINE's moved vertices/axes (S.clearance).
    here = fileparts(mfilename('fullpath'));
    deck = P.tel5f_deck;  if ~isfile(deck), deck = fullfile(here, deck); end
    tmpl = fullfile(here, P.tel5f_e2e_template);  sfx = P.tel5f_suffix;
    lam = 633e-9;  npx = P.tel_npix_xt;  if isnan(npx), npx = P.npix(1); end
    ifov = P.tel_gsd_m/P.tel_alt_m;  fov = npx*ifov;  f = P.pixel_m/ifov;
    if nargin < 3, q = struct(); end
    quiet = isfield(q, 'quiet') && q.quiet;
    if isfield(q, 'GD'), GD = q.GD; else, GD = dyson_(P, tag); end
    macos.init(P.tel3_model);
    if quiet, fid = -1;  pr = @(varargin) []; else, fid = fopen([tag '_t5f' sfx '.txt'], 'w');  pr = @(varargin) dp_(fid, varargin{:}); end
    pr('dyson5 t5f -- end to end, the join placed from ENGINE traces (FreeForm decks) (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: telescope %s, Dyson %s (Slit + Dyson blocks from %s); roll %g deg; no chain: the bridge gate is replaced\n', ...
       P.tel5f_deck, P.tel_dyson, P.tel5f_e2e_template, P.tel5e_roll_deg);
    pr('  by the identity gate (this join on a conic deck == t5e''s row).\n\n');
    % ---- (1) the telescope in its own frame
    txt = fileread(deck);  hdr = txt(1:strfind(txt, 'nElt=') - 1);
    cdir = vec_(hdr, 'ChfRayDir');  cdir = cdir/norm(cdir);  apst = vec_(hdr, 'ApStop');  bias = atan2(cdir(2), cdir(3));
    macos.load_rx(deck);  nT = macos.num_elt();
    dloc = @(thx, db) [sin(thx); cos(thx)*sin(bias + db); cos(thx)*cos(bias + db)];
    [qc, ec] = chief_(dloc(0, 0), apst, nT);
    cd = GD.src.chief_dir(:);  slit = GD.slit(:);
    Rm = rot_to_(ec, cd);
    if P.tel5e_roll_deg ~= 0
        a = P.tel5e_roll_deg*pi/180;  k = cd/norm(cd);  K = [0 -k(3) k(2); k(3) 0 -k(1); -k(2) k(1) 0];  Rm = (eye(3) + sin(a)*K + (1 - cos(a))*K*K)*Rm;
    end
    tr = slit - Rm*qc;
    % the slit plane in the TELESCOPE frame: through qc, normal Rm'*[0 0 1], "y" = Rm'*[0 1 0]
    ns = Rm'*[0; 0; 1];  ys = Rm'*[0; 1; 0];  xs = Rm'*[1; 0; 0];
    onslit = @(p, d) p + d*((qc - p)'*ns)/(d'*ns);
    vfun = @(thx, db) ys'*(onslit_chief_(dloc(thx, db), apst, nT, onslit) - qc);
    xfun = @(thx, db) xs'*(onslit_chief_(dloc(thx, db), apst, nT, onslit) - qc);
    % the field on the slit line (secant in db), and the strip the slit admits
    W = GD.P.npix(1)*GD.P.pixel_m;
    dbof = @(thx) secant_(@(db) vfun(thx, db), 0, 1e-3);
    xe = abs(xfun(fov/2, dbof(fov/2)));
    if xe > W/2, ta = fzero(@(t) abs(xfun(t, dbof(t))) - W/2, [0 fov/2]); else, ta = fov/2; end
    f_loc = abs(xfun(1e-4, dbof(1e-4)) - xfun(-1e-4, dbof(-1e-4)))/(2*tan(1e-4));  f_edge = xe/tan(fov/2);
    pr('TELESCOPE (engine): plate local %.1f mm, edge %.1f mm (spec %.1f); strip +-%.2f mm at the image vs slit %.1f mm; admitted +-%.3f of +-%.3f deg\n', ...
       f_loc*1e3, f_edge*1e3, f*1e3, xe*1e3, W*1e3, ta*180/pi, fov/2*180/pi);
    % the scored fields' slit-line directions, solved on the TELESCOPE deck before the joined deck is loaded
    fe = linspace(-ta, ta, P.e2e_nfield);  dbe = arrayfun(dbof, fe);
    % ---- (2) the joined deck: the telescope text transformed + the template's Slit/Dyson blocks
    efile = sprintf('%s_t5f%s_e2e.in', tag, sfx);  if quiet, efile = [tempname '_t5f_e2e.in']; end
    [nTel, ttxt] = transform_deck_(txt, Rm, tr);              % header + M1..M3 (the FocalPlane dropped)
    tt = fileread(tmpl);  tb = regexp(tt, '(?m)^\s*iElt=\s*\d+', 'start');
    ntm = numel(tb);  iSlitT = find(cellfun(@(c) contains(c, 'EltName=  Slit'), arrayfun(@(i) tt(tb(i):min(tb(i)+200, end)), 1:ntm, 'uni', 0)), 1);
    dys = tt(tb(iSlitT):end);  nDys = ntm - iSlitT + 1;
    dys = renumber_(dys, nTel);
    nE = nTel + nDys;
    ttxt = regexprep(ttxt, '(?m)^(\s*nElt=\s*)\d+', ['$1' num2str(nE)], 'once');
    fidd = fopen(efile, 'w');  fprintf(fidd, '%s\n%s', ttxt, dys);  fclose(fidd);
    macos.load_rx(efile);  assert(macos.num_elt() == nE, 'dyson5 t5f: %s loads %d of %d elements', efile, macos.num_elt(), nE);
    iG = nTel + (find(cellfun(@(c) contains(c, 'Element=  Grating'), regexp(dys, '(?m)^\s*iElt=', 'split')), 1) - 1);
    iSlit = nTel + 1;
    pr('JOINED DECK %s: %d telescope + %d Slit/Dyson elements; grating = element %d, slit = element %d\n', efile, nTel, nDys, iG, iSlit);
    % ---- (3) engine launches + spectrometer_score
    Rp = Rm;  tp = tr;  dsky = @(thx) Rp*dloc(thx, interp1(fe, dbe, thx, 'linear', 'extrap'));
    G = GD;  G.e2e = true;  G.iG = iG;  G.iSlit = iSlit;  G.slit = slit;
    G.launch_field = @(thx, lamq) launch_(dsky(thx), Rp*apst + tp, iG, lamq);
    G.trace = @(p0, d0, lamq) trace_slit_(p0, d0, iSlit, nE);
    M = struct('iG', iG, 'nElt', nE, 'file', efile, 'iSlit', iSlit);
    Pk = P;  Pk.Fno = GD.P.Fno;
    RE = spectrometer_score(G, M, Pk, 'fields', fe, 'nlam', P.e2e_nlam, 'quiet', true);
    pr('END TO END (engine join, %d fields x %d wavelengths over +-%.3f deg, the grating the stop):\n', P.e2e_nfield, P.e2e_nlam, ta*180/pi);
    pr('  smile %.4f px  keystone %.4f px  CRF %.3f px  SRF %.3f px  EE %.3f  grating admits %.3f\n', RE.smile_max, RE.keystone_max, RE.crf_max, RE.srf_max, RE.ee_min, min(RE.pass_frac(:)));
    pr('  smile per lambda (px): %s\n  keystone per field (px): %s\n  pass fraction per field: %s\n', sprintf('%.4f ', RE.smile_px), sprintf('%.4f ', RE.keystone_px), sprintf('%.3f ', min(RE.pass_frac, [], 2)));
    pr('  SRF per field (max over lambda): %s\n  CRF per field: %s\n', sprintf('%.2f ', max(RE.SRF, [], 2)), sprintf('%.2f ', max(RE.CRF, [], 2)));
    S = struct('e2e', RE, 'file', efile, 'admit_half_rad', ta, 'Rm', Rm, 'tr', tr, 'f_loc', f_loc, 'f_edge', f_edge, 'clearance', []);
    if isfield(q, 'GE0') && ~isempty(q.GE0)
        G.efile = efile;
        [Cl, Cl0] = tEC_(q.GE0, G, nE, iSlit, fe, P, @(thx, lamq) launch_(dsky(thx), Rp*apst + tp, iG, lamq));
        S.clearance = Cl;  S.clearance0 = Cl0;
        isTel = @(T) startsWith(string(T.leg), "Tel") & startsWith(string(T.body), "Tel");
        T1 = Cl.table(isTel(Cl.table), :);  T0 = Cl0.table(isTel(Cl0.table), :);
        pr('LIVE CLEARANCE (spectrometer_clearance verbatim; bundle + footprints from ENGINE rays on the joined deck, telescope bodies at\n');
        pr('  the engine''s vertices/axes, lifted onto the base sphere -- the FreeForm sag is not in the body): min %+.2f mm (%s vs %s) %s;\n', ...
           Cl.min_mm, Cl.table.leg{1}, Cl.table.body{1}, tern5f_(Cl.pass, 'PASS', 'FAIL'));
        if ~isempty(T1), pr('  telescope-internal worst %+.2f mm with the mount, %+.2f mm without (%s vs %s)\n', T1.clearance_mm(1), T0.clearance_mm(1), T1.leg{1}, T1.body{1}); end
        it = find(strncmp(Cl.dep_names, 'Tel', 3));
        pr('  the mirrors'' lit surface vs their BEST-FIT body spheres (residual NOT in the bodies; read against the margins): %s mm\n', ...
           strjoin(arrayfun(@(k) sprintf('%s %.2f', Cl.dep_names{k}, Cl.dep_mm(k)), it, 'uni', 0), ', '));
        for i = 1:min(8, height(Cl.table)), pr('    %-34s vs %-14s %+8.2f mm\n', Cl.table.leg{i}, Cl.table.body{i}, Cl.table.clearance_mm(i)); end
    end
    if ~quiet, fclose(fid);  save([tag '_t5f' sfx '.mat'], 'S'); end
end

function [Cl, Cl0] = tEC_(GE0, G, nE, iSlit, fe, P, launch)
%TEC_  spectrometer_clearance on the JOINED ENGINE DECK (addendum 45 (b): the geometry moves, so no carry-over).
%   GE0 is an e2e chain of the same Dyson (its Dyson surfaces, slit, FPA, mechanical boxes are the joined deck's: the template
%   blocks sit in the Dyson's frame).  Replaced from the engine: the TELESCOPE surfaces' vertex and axis (base-sphere centre
%   kept at the chain's signed offset along the axis), the bundle (3 fields x 3 wavelengths, the chief + the pupil's outer
%   rays, every station from the engine's per-element hits; the Sky station = each ray carried back to the launch plane),
%   and the footprints (every passing ray of those 9 launches, in each surface's aperture frame).
    S = GE0.surf;  nS = numel(S);
    nmE = regexp(fileread(G.efile), '(?m)^\s*EltName=\s*(\S+)', 'tokens');  nmE = cellfun(@(c) c{1}, nmE, 'uni', 0);   % our own emitted deck
    assert(numel(nmE) == nE, 'dyson5 t5f clearance: %d EltName lines for %d elements', numel(nmE), nE);
    map = zeros(1, nS);
    for k = 1:nS
        i = find(strcmp(nmE, S(k).name) | strcmp(strcat('Tel', nmE), S(k).name), 1);
        assert(~isempty(i), 'dyson5 t5f clearance: chain surface %s has no engine element', S(k).name);  map(k) = i;
    end
    F0 = GE0.footprints('nx', 3, 'nlam', 3, 'nring', 2);
    for k = 1:nS
        if strncmp(S(k).name, 'Tel', 3)
            i = map(k);  v = macos.get_elt_vpt(i);  ps = macos.get_elt_psi(i);  v = v(:);  ps = ps(:)/norm(ps);
            s0 = (S(k).C(:) - S(k).vpt(:))'*S(k).psi(:)/norm(S(k).psi);
            S(k).vpt = v;  S(k).psi = ps;  S(k).C = v + s0*ps;
            xa = F0(k).xap(:);  xa = xa - (xa'*ps)*ps;  xa = xa/norm(xa);  F0(k).xap = xa;  F0(k).yap = cross(ps, xa);
        end
    end
    lams = linspace(P.band_m(1), P.band_m(2), 3);  fq = fe([1 ceil(end/2) end]);
    PP = [];  Hall = cell(1, nS);
    % the engine's grid samples CELLS: its outermost rays sit about one spacing inside the aperture edge (gate 3, 2026-10-05:
    % M1 footprint +-96.7 mm vs the chain's true edge +-100.8 mm, clearance 3.6 mm optimistic).  So the source aperture is
    % scaled for these traces until the outermost ray sits ON the edge (Ap/2) -- the chain's disc -- and restored after.
    % ALSO the circular grid is a LATTICE clipped by the circle, so its edge is ragged (the axis rays sit ~1 spacing inside
    % the edge, the diagonals on it): these traces run on a 201-point grid (edge within ~0.5 % of D), restored after.
    sz = macos.get_src_size();  Ap = sz.aperture;  ng0 = macos.get_src_sampling();  macos.set_src_sampling(201);
    [p0, d0, ok] = launch(fq(2), lams(2));  assert(ok);  s = macos.trace(1);  ri = macos.get_ray_info(s.nRays);
    w = ri.pos(:, ri.ok_trace(:)) - ri.pos(:, 1);  w = w - d0*(d0'*w);  rm0 = max(vecnorm(w));
    macos.set_src_size(Ap*(Ap/2)/rm0, sz.obscuration);  macos.modify();
    for i = 1:3
        for j = 1:3
            [p0, d0, ok] = launch(fq(i), lams(j));  if ~ok, continue, end
            pos = nan(3, 0, nE);
            for e = 1:nE
                s = macos.trace(e);  ri = macos.get_ray_info(s.nRays);
                if e == 1, pos = nan(3, s.nRays, nE); end
                pos(:, :, e) = ri.pos;
            end
            okr = ri.ok_trace(:) & ri.ok_pass(:);  okr(1) = true;
            for k = 1:nS, Hall{k} = [Hall{k}, pos(:, okr, map(k))]; end
            w = squeeze(pos(:, :, 1)) - pos(:, 1, 1);  w = w - d0*(d0'*w);  rr = vecnorm(w);  rm = max(rr(okr));
            e1 = w(:, 2) - d0*(d0'*w(:, 2));  e1 = e1/norm(e1);  e2 = cross(d0, e1);  az = atan2(e2'*w, e1'*w);
            sel = 1;                                          % the chief + 48 edge rays and 16 half-radius rays, even in azimuth
            for ring = [1 0.5]
                cand = find(okr(:) & abs(rr(:) - ring*rm) <= 0.02*rm);  na = 48*(ring == 1) + 16*(ring < 1);
                for a = 2*pi*(0:na-1)/na - pi
                    [~, m] = min(abs(angle(exp(1i*(az(cand) - a)))));  sel(end+1) = cand(m);
                end
            end
            sel = unique(sel(:), 'stable');
            Q = pos(:, sel, map);  sky = Q(:, :, 1) - d0*(d0'*(Q(:, :, 1) - p0));
            PP = cat(2, PP, cat(3, sky, Q));
        end
    end
    macos.set_src_size(Ap, sz.obscuration);  macos.set_src_sampling(ng0);  macos.modify();
    B = struct('P', PP);
    % the TELESCOPE bodies sit on the BEST-FIT SPHERE of each mirror's own lit hits.  The chain lifts a conic/asphere body onto
    % its BASE sphere (spectrometer_clearance body_pts_), which on these eccentric sections drops the conic: M1's lit patch
    % departs from its base sphere by ~20 mm (h^4/8R^3 at h = 0.29 m, R 0.37 m) -- every telescope body in the chain record
    % was misplaced by its patch's conic sag (found 2026-10-05, addendum 45).  The fit leaves the higher orders (S.fit_mm).
    fitres = nan(1, nS);
    for k = 1:nS
        if ~strncmp(S(k).name, 'Tel', 3), continue, end
        H = Hall{k};  M = [2*H', ones(size(H, 2), 1)];  u = M\sum(H.^2, 1)';  c0 = u(1:3);  R0 = sqrt(u(4) + c0'*c0);
        ps = mean(H, 2) - c0;  ps = ps/norm(ps);
        S(k).C = c0;  S(k).R = R0;  S(k).vpt = c0 + R0*ps;  S(k).psi = ps;
        xa = F0(k).xap(:);  xa = xa - (xa'*ps)*ps;  xa = xa/norm(xa);  F0(k).xap = xa;  F0(k).yap = cross(ps, xa);
        fitres(k) = max(abs(vecnorm(H - c0) - R0))*1e3;
    end
    F = F0;
    for k = 1:nS
        H = Hall{k} - S(k).vpt(:);  u = F0(k).xap(:)'*H;  v = F0(k).yap(:)'*H;
        F(k).xlim = [min(u) max(u)];  F(k).ylim = [min(v) max(v)];  F(k).xc = mean(F(k).xlim);  F(k).yc = mean(F(k).ylim);
        F(k).radius = max(hypot(u - F(k).xc, v - F(k).yc));  F(k).n = numel(u);
    end
    Gx = GE0;  Gx.surf = S;  Gx.bundle = @(varargin) B;  Gx.footprints = @(varargin) F;  Gx.iSlit = iSlit;
    if isfield(GE0, 'iSlit'), Gx.iSlit = GE0.iSlit; end
    Cl = spectrometer_clearance(Gx, P, 'quiet', true);
    Cl.dep_mm = fitres;              % the lit surface's residual from its body's (best-fit) sphere: the margin the body model eats
    Cl.dep_names = {S.name};
    Pm = P;  Pm.mount_margin_m = 0;  Cl0 = spectrometer_clearance(Gx, Pm, 'quiet', true);
end

function t = tern5f_(c, a, b), if c, t = a; else, t = b; end, end

function [q, e] = chief_(d, apst, nE)
    macos.set_src_fov('src_pos', apst - d, 'src_dir', d, 'zSrc', 1e22);  macos.modify();
    s = macos.trace(nE);  ri = macos.get_ray_info(s.nRays);  q = ri.pos(:, 1);  e = ri.dir(:, 1)/norm(ri.dir(:, 1));
end

function p = onslit_chief_(d, apst, nE, onslit)
    [q, e] = chief_(d, apst, nE);  p = onslit(q, e);
end

function db1 = secant_(fn, db0, db1)
    v0 = fn(db0);  v1 = fn(db1);
    for it = 1:15
        if v1 == v0, break, end
        db2 = db1 - v1*(db1 - db0)/(v1 - v0);  db0 = db1;  v0 = v1;  db1 = db2;  v1 = fn(db1);
        if abs(v1) < 1e-12, break, end
    end
end

function [p0, d0, ok] = launch_(d, apst, iG, lam)
%LAUNCH_  The engine aims the chief through the grating (the instrument stop) for sky direction d AT WAVELENGTH lam: the
%   slit-to-grating path refracts through the block, so the aim is chromatic (the gate found it: aimed at the previous
%   trace's wavelength, the first lambda's smile read 2.63 px where t5e's chain gave 2.46).
    d0 = d/norm(d);  macos.set_src_wvl(lam);
    macos.set_src_fov('src_pos', apst - d0, 'src_dir', d0, 'zSrc', 1e22);  macos.modify();
    macos.stop(iG);  macos.stop(iG);                       % twice: the engine's stop aim is one pass short (spectrometer_score)
    s = macos.get_src_fov();  p0 = s.src_pos(:);  d0 = s.src_dir(:)/norm(s.src_dir);  ok = all(isfinite(p0));
end

function [pc, dc, ok] = trace_slit_(p0, d0, iSlit, nE) %#ok<INUSD>
%TRACE_SLIT_  spectrometer_score's chain-shaped trace, from the ENGINE: only the slit column is filled (all it reads).
    macos.set_src_fov('src_pos', p0, 'src_dir', d0, 'zSrc', 1e22);  macos.modify();
    s = macos.trace(iSlit);  ri = macos.get_ray_info(s.nRays);  pc = nan(3, iSlit);  pc(:, iSlit) = ri.pos(:, 1);
    dc = nan(3, iSlit);  dc(:, iSlit) = ri.dir(:, 1);  ok = logical(ri.ok_trace(1));
end

function [nTel, out] = transform_deck_(txt, Rm, tr)
%TRANSFORM_DECK_  Apply p -> Rm p + tr / d -> Rm d to a Telescope deck's header and element blocks (TElt / Tout rows, the
%   Mon / FF frames); drop the FocalPlane BLOCK (split on the iElt= lines, so its leading lines go with it).
    pts = {'VptElt', 'RptElt', 'pMon', 'pFF', 'ChfRayPos', 'ApStop'};
    dirs = {'psiElt', 'xMon', 'yMon', 'zMon', 'xFF', 'yFF', 'zFF', 'ChfRayDir', 'xGrid', 'yGrid'};
    st = regexp(txt, '(?m)^\s*iElt=', 'start');  parts = [{txt(1:st(1)-1)}, arrayfun(@(k) txt(st(k):ifelse_(k < numel(st), st(min(k+1, end))-1, numel(txt))), 1:numel(st), 'uni', 0)];
    keepb = [true, ~cellfun(@(b) ~isempty(regexp(b, '(?m)^\s*Element=\s*FocalPlane', 'once')), parts(2:end))];
    parts = parts(keepb);  o = strings(0, 1);
    for pp = 1:numel(parts)
        L = splitlines(string(parts{pp}));  inFrame = 0;
        if strlength(L(end)) == 0, L(end) = []; end
        for i = 1:numel(L)
            s = L(i);  key = regexp(char(s), '^\s*(\w+)=', 'tokens', 'once');
            if ~isempty(key), key = key{1}; else, key = ''; end
            if any(strcmp(key, {'TElt', 'Tout'})), inFrame = 6; end
            if inFrame > 0
                v = sscanf(strrep(char(regexprep(s, '^\s*\w+=', '')), 'D', 'E'), '%f')';
                r = 7 - inFrame;  b = 1 + 3*(r > 3);  v(b:b+2) = v(b:b+2)*Rm';
                lead = regexp(char(s), '^\s*\w+=', 'match', 'once');  if isempty(lead), lead = '                   '; end
                s = string(lead) + sprintf('  %.16E', v);  inFrame = inFrame - 1;
            elseif any(strcmp(key, pts))
                v = vec_(char(s), key);  s = sprintf('%18s=  %.16E  %.16E  %.16E', key, Rm*v + tr);
            elseif any(strcmp(key, dirs))
                v = vec_(char(s), key);  s = sprintf('%18s=  %.16E  %.16E  %.16E', key, Rm*v);
            end
            o(end+1, 1) = s; %#ok<AGROW>
        end
    end
    out = strjoin(o, newline);
    nTel = numel(regexp(char(out), '(?m)^\s*iElt=', 'start'));
end

function v = ifelse_(c, a, b), if c, v = a; else, v = b; end, end

function s = renumber_(s, n0)
    b = regexp(s, '(?m)^(\s*iElt=\s*)(\d+)', 'tokens');  idx = regexp(s, '(?m)^\s*iElt=\s*\d+', 'start');
    for k = numel(idx):-1:1
        m = regexp(s(idx(k):end), '^\s*iElt=\s*\d+', 'match', 'once');
        s = [s(1:idx(k)-1) sprintf('%17s=  %d', 'iElt', n0 + k) s(idx(k)+numel(m):end)];
    end
    if isempty(b), return, end
end

function R = rot_to_(a, b)
    ang = atan2(a(2)*b(3) - a(3)*b(2), a(2)*b(2) + a(3)*b(3));  R = [1 0 0; 0 cos(ang) -sin(ang); 0 sin(ang) cos(ang)];
end

function v = vec_(t, key), m = regexp(t, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens', 'once');  v = sscanf(strrep(m{1}, 'D', 'E'), '%f'); end

function GD = dyson_(P, tag)
    tk = strsplit(P.tel_dyson, ':');  Z = load(fullfile(fileparts(tag), 'dyson5_size.mat'));  r = Z.OUT.rows;
    k = find(strcmp(string({r.family}), tk{2}) & abs([r.r_mm] - str2double(tk{3})) < 1e-9 & strcmp(string({r.variant}), 'solve'), 1);
    GD = spectrometer_geom('dyson', r(k).P);
end

function dp_(fid, varargin), fprintf(fid, varargin{:});  fprintf(varargin{:}); end
