function fs = focal_surface(rx, opts)
%FOCAL_SURFACE  Measure a system's best-focus SURFACE and emit the deck
%   geometry for it -- the companion to the FEX curved-iElt+1 radius fix.
%
%   fs = FOCAL_SURFACE(RX) scans the operating envelope (fields, and
%   optionally a configuration schedule), records the best-focus image
%   point of each (config,field) as a cloud of 3-D points, fits a PLANE or
%   a SPHERE to that cloud BY CHOICE, and optionally writes a revised .in
%   with the deck's focal-surface elements replaced by the fit.
%
%   WHY THIS EXISTS.  Since 2026-08-28 FEX/SXP take the exit-pupil radius
%   to element iElt+1's actual SURFACE rather than its tangent plane, so a
%   curved focal surface is finally honoured.  That makes the DECLARED
%   surface load-bearing: if the deck's focal sphere is not where the
%   system actually focuses, the OPD reference is wrong off axis by the
%   difference.  This measures where it actually focuses.
%
%   THE CLOUD POINT IS THE Z4-NULL SPHERE CENTRE.  Per (config,field):
%   run macos.fex, then -- holding the written vertex and axis FIXED --
%   find the reference radius that NULLS the defocus term of the OPD.  The
%   centre of that sphere, vpt - R_null*psi, is the field's best-focus
%   image point.  It is the MEDIAL focus: astigmatism splits the
%   tangential and sagittal foci around it, and that split is signal the
%   fit residuals must REPORT, not error to be squeezed out.
%
%   By construction the cloud point does not depend on the FEX radius FEX
%   started from -- it is where a4 = 0, wherever the search began -- so it
%   is the same on the pre-fix and post-fix engines.  'verify' checks the
%   linearity the solve assumes rather than trusting it.
%
%   Name-value:
%     'fov_rad'    half-field for the built-in field sets (rad).  Required
%                  unless 'fields' is given.
%     'fields'     explicit K x 2 [dthx dthy] field offsets (rad).  Beats
%                  'fov_rad'/'grid'.
%     'grid'       'NxM' -- an N x M field grid over +-fov_rad.  Default
%                  '3x3'.  A 5-point centre+corners set is 'cross5'.
%     'configs'    configuration schedule (config_axis form), or [].  The
%                  cloud is measured at every (config,field) pair; how far
%                  the surface MOVES between configurations is part of the
%                  answer, so do not average it away.
%     'fit'        'sphere' (default) | 'plane'.  BY CHOICE -- there is no
%                  auto-selection.  The other model's residual is PRINTED
%                  for comparison but never substituted.
%     'stop_elt'   stop element (0 = leave the deck's own ApStop in force).
%     'xp_elt'     exit-pupil element FEX writes (default nElt-1).
%     'elts'       elements that receive the fitted surface.  Default []
%                  = auto-detect every element sharing the CURRENT declared
%                  focal surface (same Kr, psi and Vpt as element nElt).
%                  Ambiguity is refused loudly, never guessed.
%     'ngridpts'   ray-grid sampling (default 63).
%     'model_size' engine grid (default 256).
%     'init'       call macos.init first (default false; the caller owns init).
%     'write'      path for the emitted .in ([] = measure only).  Never
%                  overwrites the input.
%     'dR'         radius offsets used to calibrate d(a4)/d(rad), in
%                  BaseUnits (default [-2 -1 1 2]).
%     'verify'     re-measure a4 at the solved radius and report the
%                  residual (default true).
%     'verbose'    print the per-point table (default true).
%
%   Returns fs with fields:
%     .pts      K x 3 cloud (best-focus image points, deck BaseUnits)
%     .pt       K x 1 struct array: .name .cfg .dir .vpt .psi .rad
%                 .a4 .slope .R_null .a4_resid .ok
%     .fit      .kind 'sphere'|'plane', .centre/.radius or .point/.normal,
%                 .resid (K x 1 signed), .rms, .max, .sigma (parameter
%                 uncertainties), .cond
%     .other    the NOT-chosen model's .rms/.max, for comparison only
%     .deck     .elts .kr .psi .vpt (what was found), .emit (what would be
%                 written), .written (path or '')
%     .split    astigmatic/field spread diagnostics
%
%   See also: PUPIL_FIND, MACOS.FEX, MACOS.GET_XP, MACOS.SET_XP.
    arguments
        rx (1,:) char
        opts.fov_rad    (1,1) double = NaN
        opts.fields     (:,2) double = zeros(0,2)
        opts.grid       (1,:) char   = '3x3'
        opts.configs                 = []
        opts.fit        (1,:) char {mustBeMember(opts.fit,{'sphere','plane'})} = 'sphere'
        opts.stop_elt   (1,1) double {mustBeInteger, mustBeNonnegative} = 0
        opts.xp_elt     (1,1) double {mustBeInteger, mustBeNonnegative} = 0
        opts.elts       (1,:) double = []
        opts.ngridpts   (1,1) double {mustBeInteger, mustBePositive} = 63
        opts.model_size (1,1) double {mustBeInteger, mustBePositive} = 256
        opts.init       (1,1) logical = false
        opts.write      (1,:) char   = ''
        opts.dR         (1,:) double = [-2 -1 1 2]
        opts.verify     (1,1) logical = true
        opts.verbose    (1,1) logical = true
    end

    if opts.init, macos.init(opts.model_size); end

    % ---- field set -----------------------------------------------------
    F = resolve_fields_(opts);
    nF = size(F, 1);

    % ---- load, stop, sampling -----------------------------------------
    macos.load_rx(rx);
    macos.set_src_sampling(opts.ngridpts);
    if opts.stop_elt > 0, macos.stop(int32(opts.stop_elt)); end
    % Deck-declared object-space ApStop (stop_elt == 0): parse the header
    % ONCE so each point can re-issue it after setting its field -- the
    % stop-enforced chief ruling (Dave 2026-08-28) applies to this form
    % too; a fresh load re-aims only at the NOMINAL field.
    opts.ap_stop_pos = [];
    if opts.stop_elt == 0
        tok = regexp(fileread(rx), '^\s*ApStop=\s*([^\n%]*)', ...
                     'tokens', 'once', 'lineanchors');
        if ~isempty(tok)
            v = sscanf(tok{1}, '%f');
            if numel(v) >= 3, opts.ap_stop_pos = double(v(1:3)); end
        end
    end
    nE  = macos.num_elt();
    xpE = opts.xp_elt;  if xpE == 0, xpE = nE - 1; end
    nom = macos.get_src_fov();

    % ---- which elements carry the declared focal surface ---------------
    deck = find_focal_elts_(nE, opts.elts);

    % ---- configuration schedule ---------------------------------------
    % macos.design.configs_from_table emits these; the supervisors run them
    % through the PRIVATE config_axis validator, which is unreachable from
    % design/src.  Here every measurement RELOADS the Rx and re-applies the
    % configuration from scratch, so there is no snapshot/undo to get wrong
    % -- the reload IS the restore.
    cfgs = opts.configs;
    if ~isempty(cfgs)
        if ~isstruct(cfgs) || ~all(isfield(cfgs, {'name','set'}))
            error('macos:focal_surface:configs', ...
                  '''configs'' must be the macos.design.configs_from_table struct array');
        end
    end
    nC = max(1, numel(cfgs));

    if opts.verbose
        fprintf(['[focal_surface] %s\n  nElt=%d  xp_elt=%d  fields=%d  ' ...
                 'configs=%d  ng=%d  model=%d\n'], rx, nE, xpE, nF, nC, ...
                opts.ngridpts, opts.model_size);
        fprintf(['  declared focal surface: elts %s  Kr=%.6f  ' ...
                 'psi=[%.6f %.6f %.6f]\n'], mat2str(deck.elts), deck.kr, deck.psi);
    end

    % ---- the cloud ------------------------------------------------------
    pt = struct('name', {}, 'cfg', {}, 'dir', {}, 'vpt', {}, 'psi', {}, ...
                'rad', {}, 'a4', {}, 'slope', {}, 'R_null', {}, ...
                'a4_resid', {}, 'centre', {}, 'ok', {});
    for ic = 1:nC
        if isempty(cfgs), cfg = []; else, cfg = cfgs(ic); end
        for k = 1:nF
            p = measure_point_(rx, opts, nom, F(k,:), xpE, ...
                               sprintf('f%02d', k), cfg);
            pt(end+1) = p; %#ok<AGROW>
        end
    end

    good = [pt.ok];
    P = vertcat(pt(good).centre);
    if size(P,1) < 4
        error('macos:focal_surface:cloud', ...
              ['only %d usable cloud points -- a fit needs at least 4 ' ...
               '(and 4 is a minimum, not a recommendation)'], size(P,1));
    end

    % ---- fit ------------------------------------------------------------
    [fitS, fitP] = deal(fit_sphere_(P), fit_plane_(P));
    if strcmp(opts.fit, 'sphere')
        fit = fitS;  other = struct('kind','plane','rms',fitP.rms,'max',fitP.max);
    else
        fit = fitP;  other = struct('kind','sphere','rms',fitS.rms,'max',fitS.max);
    end

    % ---- report ---------------------------------------------------------
    fs = struct();
    fs.rx    = rx;
    fs.pts   = P;
    fs.pt    = pt;
    fs.fit   = fit;
    fs.other = other;
    fs.deck  = deck;
    fs.split = split_diag_(pt(good), P, fit);
    fs.fields = F;

    if opts.verbose, print_report_(fs, opts); end

    % ---- emission -------------------------------------------------------
    fs.deck.emit    = emit_geometry_(fs, nom, deck);
    fs.deck.written = '';
    if ~isempty(opts.write)
        fs.deck.written = write_deck_(rx, opts.write, fs);
        if opts.verbose
            fprintf('[focal_surface] wrote %s\n', fs.deck.written);
        end
    end
end


% ======================================================================
function F = resolve_fields_(opts)
    if ~isempty(opts.fields)
        F = opts.fields;  return
    end
    if ~isfinite(opts.fov_rad)
        error('macos:focal_surface:fields', ...
              'give ''fields'' explicitly or ''fov_rad'' for a built-in set');
    end
    f = opts.fov_rad;
    if strcmpi(opts.grid, 'cross5')
        F = [0 0; -f +f; +f +f; -f -f; +f -f];
        return
    end
    t = regexp(lower(opts.grid), '^(\d+)x(\d+)$', 'tokens', 'once');
    if isempty(t)
        error('macos:focal_surface:grid', ...
              '''grid'' must be ''NxM'' or ''cross5''; got %s', opts.grid);
    end
    n = str2double(t{1});  m = str2double(t{2});
    ax = @(q) ternary_(q > 1, linspace(-f, f, max(q,2)), 0);
    [X, Y] = meshgrid(ax(n), ax(m));
    F = [X(:), Y(:)];
end

function v = ternary_(c, a, b)
    if c, v = a; else, v = b; end
end

function nm = cfg_name_(cfg)
    if isempty(cfg), nm = '-'; else, nm = char(cfg.name); end
end


function apply_cfg_(cfg)
%APPLY_CFG_  Run one configuration's setters on the freshly loaded Rx.
%   Accepts the RAW {fname, elt, args...} cell entries configs_from_table
%   emits.  Deliberately a SUBSET of the supervisors' whitelist: the pose
%   setters plus perturb.  Anything else errors rather than applying and
%   then silently not being restored -- the same reasoning config_axis
%   gives, reached by reloading instead of snapshotting.
    if isempty(cfg), return; end
    for k = 1:numel(cfg.set)
        e = cfg.set{k};
        if iscell(e)
            fn = char(e{1});  elt = double(e{2});  args = e(3:end);
        else                                   % already-validated struct
            fn = e.fn;  elt = e.elt;  args = {};
        end
        switch fn
        case 'perturb'
            if iscell(e)
                macos.perturb(elt, args{:});
            else
                macos.perturb(elt, 'rotation', e.rotation, ...
                    'translation', e.translation, 'frame', e.frame);
            end
        case 'set_elt_vpt'
            macos.set_elt_vpt(elt, pick_(iscell(e), args, e, 'value'));
        case 'set_elt_psi'
            macos.set_elt_psi(elt, pick_(iscell(e), args, e, 'value'));
        case 'set_elt_rpt'
            macos.set_elt_rpt(elt, pick_(iscell(e), args, e, 'value'));
        otherwise
            error('macos:focal_surface:configSetter', ...
                  ['''%s'' is not an accepted focal_surface configuration ' ...
                   'setter (perturb, set_elt_vpt/psi/rpt)'], fn);
        end
    end
    macos.modify();
end


function v = pick_(israw, args, e, fld)
    if israw, v = args{1}; else, v = e.(fld); end
end


% ======================================================================
function deck = find_focal_elts_(nE, forced)
%FIND_FOCAL_ELTS_  Every element sharing element nElt's declared surface.
%   The jwst OTE deck carries the SAME sphere at elts 26 and 28 (the focal
%   Return and the detector Reference); both must move together or the
%   deck becomes inconsistent.  Matching is on Kr AND psi AND Vpt -- an
%   element that merely shares the radius is a different surface.
    kr  = macos.get_elt_kr(nE);
    psi = macos.get_elt_psi(nE);
    vpt = macos.get_elt_vpt(nE);
    if ~isempty(forced)
        elts = sort(forced(:).');
    else
        elts = [];
        for i = 1:nE
            if abs(macos.get_elt_kr(i) - kr) <= 1e-9 * max(1, abs(kr)) && ...
               norm(macos.get_elt_psi(i) - psi) <= 1e-12 && ...
               norm(macos.get_elt_vpt(i) - vpt) <= 1e-9 * max(1, norm(vpt))
                elts(end+1) = i; %#ok<AGROW>
            end
        end
        if isempty(elts)
            error('macos:focal_surface:noelts', ...
                  'element %d''s surface matched nothing -- pass ''elts''', nE);
        end
    end
    deck = struct('elts', elts, 'kr', kr, 'psi', psi(:), 'vpt', vpt(:), ...
                  'kc', macos.get_elt_kc(nE));
end


% ======================================================================
function p = measure_point_(rx, opts, nom, dxy, xpE, nm, cfg)
%MEASURE_POINT_  One cloud point: the Z4-nulling sphere centre.
%
%   The Rx is RELOADED for every point, so a configuration never has to be
%   undone and no field inherits the previous field's written exit pupil.
%
%   Stop order (Dave's ruling 2026-08-28 -- the stop-enforced chief IS
%   the field's chief ray): the stop is RE-ISSUED after the field is set,
%   so the chief re-aims through the stop at that field -- the CLI
%   STOP/PERTURB convention, now also the dw_d* supervisors'.  The A/B
%   report measured the two orders differ by up to 0.76 mm in the
%   written radius but hold the sphere CENTRE to 8e-5 mm -- and the
%   centre is what this routine returns, so the cloud barely moves;
%   the convention is enforced anyway.  Do the right thing always.
    p = struct('name', nm, 'cfg', cfg_name_(cfg), 'dir', nan(3,1), ...
               'vpt', nan(3,1), 'psi', nan(3,1), 'rad', NaN, 'a4', NaN, ...
               'slope', NaN, 'R_null', NaN, 'a4_resid', NaN, ...
               'centre', nan(1,3), 'ok', false);
    try
        macos.load_rx(rx);
        macos.set_src_sampling(opts.ngridpts);
        if opts.stop_elt > 0, macos.stop(int32(opts.stop_elt)); end
        apply_cfg_(cfg);

        % Field.  Same convention as the dw_d* supervisors -- add the
        % offset to the deck's own ChfRayDir and renormalise; the frame is
        % the deck's, never re-derived here.
        v = nom.src_dir(:) + [dxy(1); dxy(2); 0];
        d = v / norm(v);
        p.dir = d;
        macos.set_src_fov('src_pos', nom.src_pos, 'src_dir', d, 'zSrc', nom.zSrc);
        % stop-enforced chief: re-issue the stop at THIS field (either
        % form -- the fresh load above aimed it at the NOMINAL field)
        if opts.stop_elt > 0
            macos.stop(int32(opts.stop_elt));
        elseif ~isempty(opts.ap_stop_pos)
            macos.stop_obj(opts.ap_stop_pos(1), opts.ap_stop_pos(2), ...
                           opts.ap_stop_pos(3));
        end
        macos.modify();

        macos.fex(xpE);
        xp0 = macos.get_xp();
        p.vpt = xp0.vpt(:);  p.psi = xp0.psi(:);  p.rad = xp0.rad;

        wfe = macos.num_elt() - 1;
        s0  = opd_stats_(wfe);
        p.a4 = s0.a4;

        % Calibrate d(a4)/d(rad) with the vertex and axis HELD.  Sliding
        % the vertex along psi with the radius compensated leaves the
        % sphere centre -- and hence the OPD -- unchanged, so the radius
        % is the only free parameter here.
        dR = opts.dR(:).';
        a4 = zeros(size(dR));
        for q = 1:numel(dR)
            macos.set_xp(xp0.vpt, xp0.psi, xp0.rad + dR(q));
            macos.modify();
            a4(q) = getfield(opd_stats_(wfe), 'a4'); %#ok<GFLD>
        end
        cf = polyfit([0 dR], [p.a4 a4], 1);
        p.slope  = cf(1);
        p.R_null = xp0.rad - cf(2) / cf(1);

        if opts.verify
            macos.set_xp(xp0.vpt, xp0.psi, p.R_null);
            macos.modify();
            p.a4_resid = getfield(opd_stats_(wfe), 'a4'); %#ok<GFLD>
        end

        % Best-focus image point = centre of the Z4-null sphere.
        p.centre = (xp0.vpt(:) - p.R_null * xp0.psi(:)).';
        p.ok = true;
    catch ME
        warning('macos:focal_surface:point', '%s/%s failed: %s', ...
                p.cfg, nm, ME.message);
    end
end


function s = opd_stats_(wfe)
%OPD_STATS_  Piston/tip/tilt/focus fit of the current OPD map.
%   Coordinates are ray-grid INDEX pixels centred on the footprint
%   CENTROID (the canvas centre is not the beam centre on an obscured
%   pupil) and normalised to the footprint's own max radius, so the pixel
%   pitch cancels -- macos.dx_at at an exit pupil returns 0 because no
%   diffraction grid has been propagated there.  a4 is the coefficient of
%   (2*rho^2 - 1); the ANSI Z4 coefficient is a4/sqrt(3).
    macos.trace(wfe);
    W = macos.opd();
    mask = (W ~= 0);
    [I, J] = ndgrid(1:size(W,1), 1:size(W,2));
    ci = mean(I(mask));  cj = mean(J(mask));
    xv = I(mask) - ci;   yv = J(mask) - cj;
    w  = W(mask);
    r0 = max(hypot(xv, yv));
    rx_ = xv / r0;  ry_ = yv / r0;
    rho2 = rx_.^2 + ry_.^2;
    A = [ones(numel(w),1), rx_, ry_, (2*rho2 - 1)];
    c = A \ w;
    s.a4  = c(4);
    s.rms = sqrt(mean(w.^2));
    s.rms_pttf = sqrt(mean((w - A*c).^2));
    s.r0  = r0;
    s.n   = numel(w);
end


% ======================================================================
function f = fit_sphere_(P)
%FIT_SPHERE_  Algebraic (Kasa) seed + Gauss-Newton on the true residual.
    n = size(P,1);
    A = [2*P, ones(n,1)];
    b = sum(P.^2, 2);
    s = A \ b;
    c = s(1:3).';
    R = sqrt(max(s(4) + c*c.', 0));
    for it = 1:50                       %#ok<NASGU>  Gauss-Newton refine
        d  = P - c;
        rn = sqrt(sum(d.^2, 2));
        res = rn - R;
        J  = [-d ./ rn, -ones(n,1)];    % d(res)/d[cx cy cz R]
        step = J \ (-res);
        c = c + step(1:3).';
        R = R + step(4);
        if norm(step) <= 1e-12 * max(1, R), break; end
    end
    d  = P - c;
    rn = sqrt(sum(d.^2, 2));
    res = rn - R;
    J  = [-d ./ rn, -ones(n,1)];
    f = struct('kind','sphere','centre',c(:),'radius',R, ...
               'resid',res,'rms',sqrt(mean(res.^2)),'max',max(abs(res)));
    [f.sigma, f.cond] = param_sigma_(J, res);
end


function f = fit_plane_(P)
%FIT_PLANE_  LS plane: centroid + smallest right singular vector.
    c = mean(P, 1);
    [~, S, V] = svd(P - c, 0);
    nrm = V(:,3);
    res = (P - c) * nrm;
    f = struct('kind','plane','point',c(:),'normal',nrm(:), ...
               'resid',res,'rms',sqrt(mean(res.^2)),'max',max(abs(res)));
    sv = diag(S);
    f.cond = sv(1) / max(sv(2), eps);   % in-plane spread anisotropy
    f.sigma = f.rms / sqrt(max(size(P,1) - 3, 1)) * ones(3,1);
end


function [sig, kappa] = param_sigma_(J, res)
%PARAM_SIGMA_  Jacobian-based 1-sigma parameter uncertainties.
    n = size(J,1);  p = size(J,2);
    dof = max(n - p, 1);
    s2 = sum(res.^2) / dof;
    C = J.' * J;
    kappa = cond(C);
    if rcond(C) < eps
        sig = inf(p,1);
    else
        sig = sqrt(s2 * diag(inv(C)));
    end
end


% ======================================================================
function sp = split_diag_(pt, P, fit)
%SPLIT_DIAG_  What the residuals are made of.
%   The astigmatic tangential/sagittal split puts real structure in the
%   cloud that NO plane or sphere can absorb.  Report it beside the fit
%   residual so a nonzero residual is not read as a failed fit.
    sp = struct();
    sp.cloud_extent = max(P) - min(P);
    sp.resid_rms = fit.rms;
    sp.resid_max = fit.max;
    if isfield(pt, 'cfg') && ~isempty(pt)
        cn = unique({pt.cfg});
        sp.n_config = numel(cn);
        if numel(cn) > 1
            % how far the surface moves between configurations
            mu = zeros(numel(cn), 3);
            for i = 1:numel(cn)
                sel = strcmp({pt.cfg}, cn{i});
                mu(i,:) = mean(vertcat(pt(sel).centre), 1);
            end
            sp.config_motion = max(mu) - min(mu);
            sp.config_motion_norm = norm(sp.config_motion);
        else
            sp.config_motion = zeros(1,3);
            sp.config_motion_norm = 0;
        end
    end
    a4r = [pt.a4_resid];
    sp.a4_resid_max = max(abs(a4r(isfinite(a4r))));
    sp.slope = [pt.slope];
end


% ======================================================================
function e = emit_geometry_(fs, nom, deck)
%EMIT_GEOMETRY_  The replacement Kr/psi/Vpt for the focal-surface elements.
%   Vertex   = the NOMINAL chief's intersection with the fitted surface.
%   Normal   = the surface normal there, with the HEMISPHERE COPIED from
%              the element's existing psi.  Do NOT invent a sign rule --
%              the pupil_find psi-hemisphere defect is the cautionary tale.
%   Kr sign  = copied from the deck's existing convention.
%   Kc       = 0 (a fitted sphere/plane has no conic term).
    e = struct();
    o = deck.vpt(:);                     % ray along the deck's own axis
    d = -deck.psi(:);                    % ...toward the surface
    switch fs.fit.kind
    case 'sphere'
        c = fs.fit.centre(:);  R = fs.fit.radius;
        f = o - c;
        bq = 2*dot(f,d);  cq = dot(f,f) - R^2;
        disc = bq^2 - 4*cq;
        if disc < 0
            error('macos:focal_surface:emit', ...
                  'deck axis misses the fitted sphere -- cannot place a vertex');
        end
        t = [(-bq - sqrt(disc))/2, (-bq + sqrt(disc))/2];
        [~, ix] = min(abs(t));           % the crossing nearest the old vertex
        vtx = o + t(ix)*d;
        nrm = (vtx - c) / R;             % outward surface normal
        if dot(nrm, deck.psi(:)) < 0, nrm = -nrm; end   % hemisphere COPIED
        e.kr = sign(deck.kr) * abs(R);   % sign convention COPIED
        e.kc = 0;
        e.psi = nrm(:);
        e.vpt = vtx(:);
        e.kind = 'sphere';
        e.radius = R;
        e.centre = c;
    case 'plane'
        q = fs.fit.point(:);  nrm = fs.fit.normal(:);
        den = dot(d, nrm);
        if abs(den) < 1e-12
            error('macos:focal_surface:emit', ...
                  'deck axis is parallel to the fitted plane');
        end
        vtx = o + (dot(q - o, nrm)/den) * d;
        if dot(nrm, deck.psi(:)) < 0, nrm = -nrm; end   % hemisphere COPIED
        e.kr = -1e22;                    % the corpus flat sentinel
        e.kc = 0;
        e.psi = nrm(:);
        e.vpt = vtx(:);
        e.kind = 'plane';
    end
    e.elts = deck.elts;
    e.dvpt = norm(e.vpt - deck.vpt(:));
    e.dpsi = rad2deg(real(acos(min(1, max(-1, dot(e.psi, deck.psi(:)))))));
    e.dkr  = e.kr - deck.kr;
end


% ======================================================================
function outp = write_deck_(rx, outp, fs)
%WRITE_DECK_  Emit a NEW .in with the focal-surface elements replaced.
%   Never overwrites the input.  Edits are per-element textual
%   replacements of KrElt/KcElt/psiElt/VptElt/RptElt inside the blocks
%   named by fs.deck.emit.elts, which keeps every other keyword -- and the
%   file's own layout -- byte-identical.
    if strcmp(fullfile(outp), fullfile(rx))
        error('macos:focal_surface:write', ...
              'refusing to overwrite the input prescription');
    end
    txt = fileread(rx);
    nl  = sprintf('\n');
    lines = strsplit(txt, nl, 'CollapseDelimiters', false);

    % element block boundaries: an 'Element=' line starts a block
    isElt = ~cellfun(@isempty, regexp(lines, '^\s*Element\s*=', 'once'));
    starts = find(isElt);
    e = fs.deck.emit;
    for i = 1:numel(e.elts)
        k = e.elts(i);
        if k > numel(starts)
            error('macos:focal_surface:write', ...
                  'element %d has no Element= block in %s', k, rx);
        end
        lo = starts(k);
        if k < numel(starts), hi = starts(k+1) - 1; else, hi = numel(lines); end
        lines(lo:hi) = subst_block_(lines(lo:hi), e);
    end
    fid = fopen(outp, 'w');
    if fid < 0, error('macos:focal_surface:write', 'cannot write %s', outp); end
    fprintf(fid, '%s', strjoin(lines, nl));
    fclose(fid);
end


function L = subst_block_(L, e)
    L = set_key_(L, 'KrElt',  sprintf('%.10E', e.kr));
    L = set_key_(L, 'KcElt',  sprintf('%.10E', e.kc));
    L = set_key_(L, 'psiElt', sprintf('%.10E %.10E %.10E', e.psi));
    L = set_key_(L, 'VptElt', sprintf('%.10E %.10E %.10E', e.vpt));
    L = set_key_(L, 'RptElt', sprintf('%.10E %.10E %.10E', e.vpt));
end


function L = set_key_(L, key, val)
    pat = ['^(\s*' key '\s*=\s*).*$'];
    hit = find(~cellfun(@isempty, regexp(L, pat, 'once')), 1);
    if isempty(hit), return; end          % key absent -> leave the deck alone
    pre = regexp(L{hit}, ['^\s*' key '\s*=\s*'], 'match', 'once');
    L{hit} = [pre val];
end


% ======================================================================
function print_report_(fs, opts)
    fprintf('\n%-6s %-8s %13s %13s %12s %12s\n', 'cfg', 'field', ...
            'FEX rad', 'R_null', 'a4(FEX)', 'a4 resid');
    for i = 1:numel(fs.pt)
        p = fs.pt(i);
        if ~p.ok, fprintf('%-6s %-8s   (failed)\n', p.cfg, p.name); continue; end
        fprintf('%-6s %-8s %13.6f %13.6f %12.4e %12.4e\n', ...
                p.cfg, p.name, p.rad, p.R_null, p.a4, p.a4_resid);
    end
    f = fs.fit;
    fprintf('\n[fit = %s, BY CHOICE -- no auto-selection]\n', f.kind);
    if strcmp(f.kind, 'sphere')
        fprintf('  centre = [%.6f %.6f %.6f]   radius = %.6f\n', f.centre, f.radius);
        fprintf('  1-sigma: centre [%.3e %.3e %.3e]  radius %.3e   cond %.3e\n', ...
                f.sigma(1:3), f.sigma(4), f.cond);
        fprintf('  deck declared Kr = %.6f  ->  fitted |R| = %.6f  (delta %.6f)\n', ...
                fs.deck.kr, f.radius, abs(fs.deck.kr) - f.radius);
    else
        fprintf('  point = [%.6f %.6f %.6f]   normal = [%.6f %.6f %.6f]\n', ...
                f.point, f.normal);
        fprintf('  in-plane anisotropy (cond) %.3e\n', f.cond);
    end
    fprintf('  residual rms %.6e   max %.6e   (%d points)\n', ...
            f.rms, f.max, size(fs.pts,1));
    fprintf('  the OTHER model (%s), for comparison only: rms %.6e  max %.6e\n', ...
            fs.other.kind, fs.other.rms, fs.other.max);
    s = fs.split;
    fprintf(['  cloud extent [%.4f %.4f %.4f]   a4 verify residual max %.3e\n'], ...
            s.cloud_extent, s.a4_resid_max);
    if isfield(s,'config_motion_norm')
        fprintf('  configuration motion of the cloud mean: %.6f\n', ...
                s.config_motion_norm);
    end
    if size(fs.pts,1) < 6
        fprintf(['  NOTE: %d points is thin for a %s fit -- the parameter ' ...
                 'sigmas above are the honest read.\n'], size(fs.pts,1), f.kind);
    end
end
