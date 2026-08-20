function [X, G, fo] = oi_close(X, P, opts)
%OI_CLOSE  First-order closure of an offset_imager candidate design.
%
%   [X, G, FO] = OI_CLOSE(X, P) enforces the template's first-order
%   IDENTITIES on the design struct X (afocal4 doctrine: identities are
%   re-derived at every iterate, never penalized in a merit):
%
%     1  EFL = P.EFL_m EXACTLY -- R3 is eliminated: solved by secant on
%        the paraxial chain (OI_PARAXIAL) given R1, R2 and the spacings.
%     2  STOP POSE -- S1 (on-axis box): the stop centre is the axial
%        point [0 0 z_stop].  Offset box: the ENTRANCE-PUPIL
%        construction: the EP centre is measured from the engine (the
%        crossing, in object space, of two aimed near-axis chiefs), the
%        box-centre chief is launched THROUGH the EP centre, and the
%        stop centre is where that traced chief crosses the stop plane.
%        The physical aperture follows the used field the way a real
%        aperture would -- and on the rodgers3 instance this
%        construction is what CODE V's stop YDE encodes.
%     3  FP POSE -- the FP is posed ON the traced box-centre exit chief
%        (recenter by construction), at the paraxial back-focus distance
%        along it, normal to it.  Stage refits then open [dz, tilt]
%        about this pose (OI_SOLVE 'fpa').
%
%   The closure is ENGINE-TRUTH where it matters: EP and chief poses
%   come from traced rays of the candidate itself, not from the paraxial
%   model (which only pins EFL/BFD scalars).
%
%   Returns the closed X (R3, stopC, fpa updated), the scorer geometry
%   G (stopC / z_m1 / fpa), and FO with the first-order record:
%   .EFL_m .BFD_m .petzval .EP (3x1) .stop_semi_m .plate_um_per_amin.
%
%   Name-value:
%     'offset_deg'  box-centre YAN, deg (default P.offset_deg)
%     'repose_fpa'  true (default) = re-pose the FP; false keeps X.fpa
%                   (stage refits own it once a stage has solved it)
%     'repose_stop' true (default) = re-pose the stop; false keeps X.stopC
%
%   See also OI_PARAXIAL, OI_DECK, OI_SCORE, OFFSET_IMAGER.

    arguments
        X struct
        P struct
        opts.offset_deg (1,1) double = NaN
        opts.repose_fpa (1,1) logical = true
        opts.repose_stop (1,1) logical = true
    end
    if isnan(opts.offset_deg), opts.offset_deg = P.offset_deg; end

    % ---- 1. EFL identity: eliminate R3 ------------------------------------
    tnet = [X.spacings(1) + X.spacings(2), X.spacings(3)];
    c3 = secant_(@(c3) efl_err_(X.R(1), X.R(2), 1/c3, tnet, P.EFL_m), ...
                 1/X.R(3));
    X.R(3) = 1/c3;
    fo = oi_paraxial(X.R, tnet);

    % ---- stations ----------------------------------------------------------
    z_stop = X.z_m1 + X.spacings(1);

    % ---- 2. stop pose -------------------------------------------------------
    if opts.repose_stop
        if abs(opts.offset_deg) < 1e-12
            X.stopC = [0; 0; z_stop];
        else
            % 2a. axial stop first, measure the EP from two aimed chiefs
            X0 = X;  X0.stopC = [0; 0; z_stop];
            X0.fpa = far_fpa_(X0, fo);
            EP = ep_measure_(X0, P);
            % 2b. box-centre chief through the EP centre; its crossing at
            %     the stop plane is the stop centre
            cdir = tancomp_(0, opts.offset_deg);
            X.stopC = stop_from_ep_(X0, P, EP, cdir, z_stop);
            fo.EP = EP;
        end
    end

    % ---- 3. FP pose ----------------------------------------------------------
    if opts.repose_fpa
        cdir = tancomp_(0, opts.offset_deg);
        [pc, dc] = exit_chief_(X, P, cdir, fo);
        X.fpa = struct('Vpt', pc + abs(fo.BFD_m)*dc, 'psi', -dc);
    end

    % ---- scorer geometry + first-order extras --------------------------------
    G = struct('stopC',X.stopC, 'z_m1',X.z_m1, 'fpa',X.fpa);
    fo.stop_semi_m = stop_semi_(X, P);
    fo.plate_um_per_amin = abs(fo.EFL_m)*tand(1/60)*1e6;
end

% =========================================================================
function e = efl_err_(R1, R2, R3, tnet, EFLt)
    o = oi_paraxial([R1 R2 R3], tnet);
    e = o.EFL_m - EFLt;
end

function c = secant_(f, c0)
    c1 = c0*(1+1e-6) + 1e-9;
    f0 = f(1/(1/c0));  f1 = f(c1);   %#ok<NASGU>
    f0 = f(c0);  f1 = f(c1);
    for it = 1:60
        if abs(f1 - f0) < eps, break; end
        c2 = c1 - f1*(c1 - c0)/(f1 - f0);
        c0 = c1;  f0 = f1;  c1 = c2;  f1 = f(c1);
        if abs(f1) < 1e-15, break; end
    end
    if abs(f1) > 1e-9
        error('oi_close:efl','EFL closure did not converge (err %g m)', f1);
    end
    c = c1;
end

function fpa = far_fpa_(X, fo)
%FAR_FPA_  Crude axial FP (only used while measuring the EP on axis).
    z_m3 = X.z_m1 + sum(X.spacings);
    fpa = struct('Vpt',[0;0;z_m3 - abs(fo.BFD_m)], 'psi',[0;0;1]);
end

function EP = ep_measure_(X0, P, dth)
%EP_MEASURE_  Entrance-pupil centre: crossing in OBJECT space of two
%   near-axis chiefs aimed through the axial stop (native machinery).
    if nargin < 3, dth = 0.5; end     % deg
    [p1, d1] = obj_chief_(X0, P, tancomp_(0, +dth));
    [p2, d2] = obj_chief_(X0, P, tancomp_(0, -dth));
    EP = fex_cross_(p1, d1, p2, d2);
end

function [pm1, cdir] = obj_chief_(X, P, cdir)
%OBJ_CHIEF_  Aim the chief through the stop (native), return its M1
%   intersection; the object-space line is (pm1, cdir).
    tmp = [tempname '.in'];
    cu  = onCleanup(@() delete_(tmp));
    txt = oi_deck(deck_fill_(X, P));
    emit_src_(txt, tmp, seed_pos_(X, cdir), cdir);
    macos.load_rx(tmp);
    if ~macos.has_rx(), error('oi_close:load','closure deck failed to load'); end
    macos.stop(2, [0 0]);
    tr = macos.trace(1);
    ri = macos.get_ray_info(tr.nRays);
    pm1 = ri.pos(:,1);
end

function stopC = stop_from_ep_(X0, P, EP, cdir, z_stop)
%STOP_FROM_EP_  Launch the box-centre chief through the EP centre
%   (UNAIMED -- the EP defines it), trace to M1, extrapolate the post-M1
%   ray to the stop plane.
    tmp = [tempname '.in'];
    cu  = onCleanup(@() delete_(tmp));
    txt = oi_deck(deck_fill_(X0, P));
    p0  = EP - (0.75/cdir(3))*cdir;
    emit_src_(txt, tmp, p0, cdir);
    macos.load_rx(tmp);
    if ~macos.has_rx(), error('oi_close:load','closure deck failed to load'); end
    tr = macos.trace(1);
    ri = macos.get_ray_info(tr.nRays);
    p = ri.pos(:,1);  d = ri.dir(:,1);
    t = (z_stop - p(3))/d(3);
    q = p + d*t;
    stopC = [q(1); q(2); z_stop];
end

function [pc, dc] = exit_chief_(X, P, cdir, fo)
%EXIT_CHIEF_  Aimed box-centre chief state at M3 (position + direction).
    tmp = [tempname '.in'];
    cu  = onCleanup(@() delete_(tmp));
    Xf = X;  Xf.fpa = far_fpa_(X, fo);
    % keep the FP off the beam: pose it crudely along -psi of m3 exit; a
    % far flat plane is enough to trace THROUGH m3
    txt = oi_deck(deck_fill_(Xf, P));
    emit_src_(txt, tmp, seed_pos_(X, cdir), cdir);
    macos.load_rx(tmp);
    if ~macos.has_rx(), error('oi_close:load','closure deck failed to load'); end
    macos.stop(2, [0 0]);
    tr = macos.trace(4);              % M3 is element 4 in the template train
    ri = macos.get_ray_info(tr.nRays);
    pc = ri.pos(:,1);  dc = ri.dir(:,1);
end

function r = stop_semi_(X, P)
%STOP_SEMI_  Traced beam semi-diameter at the stop plane (box centre).
    tmp = [tempname '.in'];
    cu  = onCleanup(@() delete_(tmp));
    txt = oi_deck(deck_fill_(X, P));
    cdir = tancomp_(0, P.offset_deg);
    emit_src_(txt, tmp, seed_pos_(X, cdir), cdir);
    macos.load_rx(tmp);
    if ~macos.has_rx(), r = nan; return; end
    macos.stop(2, [0 0]);
    tr = macos.trace(2);
    ri = macos.get_ray_info(tr.nRays);
    ok = ri.ok_trace(:) & ri.ok_pass(:);  ok(1) = false;
    Q = ri.pos(1:2,ok);
    r = max(vecnorm(Q - mean(Q,2), 2, 1));
end

function D = deck_fill_(X, P)
    D = X;
    D.EPD_m = P.EPD_m;  D.WL_m = P.lambda_m;
    D.sampling = P.sampling;  D.name = P.name;
end

function p = seed_pos_(X, cdir)
    z_stop = X.z_m1 + X.spacings(1);
    sC = X.stopC;  if isempty(sC), sC = [0;0;z_stop]; end
    cdR = [cdir(1); cdir(2); -cdir(3)];
    tq  = (X.z_m1 - sC(3))/cdir(3);
    q   = sC - tq*cdR;
    p   = q - (0.75/cdir(3))*cdir;
end

function emit_src_(txt0, tmp, p0, cdir)
    v3 = @(v) sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));
    s = regexprep(txt0, '(ChfRayDir=\s*)[^\n]*', ['$1' v3(cdir)]);
    s = regexprep(s,    '(ChfRayPos=\s*)[^\n]*', ['$1' v3(p0)]);
    fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
end

function d = tancomp_(xan_deg, yan_deg)
    d = [tand(xan_deg); tand(yan_deg); 1];
    d = d/norm(d);
end

function X = fex_cross_(p1,d1,p2,d2)
    d1 = d1/norm(d1);  d2 = d2/norm(d2);
    w0 = p1 - p2;  b = dot(d1,d2);  den = 1 - b^2;
    if abs(den) < 1e-14, X = p1; return; end
    s1 = ( b*dot(d2,w0) - dot(d1,w0)) / den;
    s2 = ( dot(d2,w0) - b*dot(d1,w0)) / den;
    X  = 0.5*((p1 + d1*s1) + (p2 + d2*s2));
end

function delete_(p), if exist(p,'file'), delete(p); end, end
