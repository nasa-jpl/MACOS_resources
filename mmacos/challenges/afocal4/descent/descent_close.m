function C = descent_close(P, S, opts)
%DESCENT_CLOSE  First-order closure of an N-mirror coaxial afocal train.
%
%   C = DESCENT_CLOSE(P, S) closes the three first-order conditions an
%   afocal telescope with an interface pupil is SPECIFIED by, for any number
%   of powered mirrors N >= 3:
%
%     recollimate            u_out = 0            (the afocal condition)
%     magnify by P.M         |y_out| = (D/2)/M    (the Lagrange statement)
%     land the exit pupil    pupil_dist = S.iface (the interface condition)
%
%   They stay EXACT CLOSURES and never become merit terms -- the S4 ruling,
%   carried up to N mirrors.  AFOCAL4_BUILD holds them at 1e-9 on the
%   4-mirror family; this holds them the same way, and by the same algebra.
%
%   THE GENERALIZATION IS THREE LINES, AND THEY ARE AFOCAL4_CLOSE'S OWN.
%   In the 4-mirror 'field' form (AFOCAL4_CLOSE's FIELD_D_) the marginal ray
%   fixes everything but the pupil:
%
%       phi_N   = u/y_out                 the LAST mirror recollimates
%       t_{N-1} = (y_out - y)/u           the LAST spacing sets M
%       phi_{N-1}                         the only real unknown: the chief
%                                         ray is the residue, and this is
%                                         the one power the pupil condition
%                                         actually buys
%
%   with (y, u) the marginal state leaving mirror N-2.  Nothing in that is
%   about four mirrors: propagate the paraxial marginal and chief through
%   mirrors 1..N-2 with their FREE radii and spacings, and the last three
%   quantities close in exactly the same substitution.  So the descent's
%   closure is the 4-mirror closure with a longer front end -- which is why
%   N = 4 reproduces AFOCAL4_CLOSE's 'field' branch to machine precision
%   (asserted, not claimed: TAFOCAL4DESCENT).
%
%   TWO THINGS THAT ARE *NOT* INHERITED AND HAD TO BE DERIVED:
%
%   1  THE SIGN OF THE EXIT MARGINAL HEIGHT.  AFOCAL4_CLOSE hard-codes
%      y_out = -(D/2)/M because a 4-mirror 'field' train forms exactly ONE
%      intermediate image, hence one axis crossing, hence a negative exit
%      height.  At other N the number of crossings is not fixed and the sign
%      is a PROPERTY OF THE LAYOUT, not a constant.  Both signs are closed
%      here and the one that puts the last mirror at a POSITIVE spacing
%      wins -- the same rule AFOCAL4_CLOSE's 'relay' branch already states
%      for its own second image ("that sign is taken from the requirement
%      that M4 sit at a positive distance, not assumed").  If both signs
%      close a positive spacing the layout is ambiguous and it says so
%      rather than picking one.
%
%   2  A SIGN CHANGE IS NOT A ROOT (RESULTS rule 11).  d(phi_{N-1}) is a
%      RATIONAL function of the power, so it changes sign across its POLES
%      as well as its zeros and FZERO converges onto a pole quite happily --
%      returning a "layout" with a collimator 1e14 m away.  Every candidate
%      is therefore CLOSED and CHECKED (finite positive spacings, the pupil
%      where it was asked for, both first-order identities intact) and the
%      first one that is actually a telescope wins.  First = LOWEST |power|,
%      because the weakest penultimate mirror is the one closest to the
%      train it was grown from.  This is AFOCAL4_PHI4's discipline, kept.
%
%   S (the free parameters -- everything the closure does NOT consume):
%     .N        number of ELEMENTS the closure indexes (>= 3).  A free
%               element given a near-infinite radius is a FLAT: it still
%               reflects (so it still sets the packaging parity) but it is
%               not a powered mirror, and C.n_powered / C.n_flat report the
%               two counts apart.  Conflating them is how "N = 6 cannot be
%               built" gets said about a train that can -- see
%               DESCENT_REMOVE.
%     .R        1 x (N-2) radius MAGNITUDES, mirrors 1..N-2
%     .convex   1 x (N-2) logical
%     .t        1 x (N-2) spacings, mirror k -> k+1
%     .iface    interface standoff past the last mirror, m
%     .K        1 x N conics (carried through; the closure is paraxial and
%               does not read them)
%
%   Name-value:
%     'phi'     impose phi_{N-1} instead of closing it, in which case the
%               interface distance FLOATS to wherever the pupil goes.  The
%               diagnostic mode AFOCAL4_CLOSE offers for the same reason.
%     'window'  scan window for phi_{N-1}, /m ([-1.5 9])
%     'npts'    scan resolution (241)
%
%   Returns C with .N .names .R .t .convex .K .phi (all N) .fo (the paraxial
%   trace) .z .behind_m1 .y_out_sign .pupil_dist .found, and .residual (the
%   three closure conditions, which must all be at 1e-12 or the closure is
%   not a closure).
%
%   See also AFOCAL4_CLOSE, AFOCAL4_PHI4, AFOCAL_FIRST_ORDER, DESCENT_BUILD.

    arguments
        P (1,1) struct
        S (1,1) struct
        opts.phi    (1,1) double = NaN
        opts.window (1,2) double = [-1.5 9]
        opts.npts   (1,1) double = 241
    end

    N = S.N;
    if N < 3
        error('macos:design:descent_close:N', ...
              ['an afocal train with an interface pupil needs at least 3 ' ...
               'powered mirrors (asked for %d): two close the afocal and ' ...
               'magnification conditions, and the pupil needs a third.'], N);
    end
    need = N - 2;
    chk_(S, 'R', need);   chk_(S, 'convex', need);   chk_(S, 't', need);
    if ~isfield(S,'K') || numel(S.K) ~= N
        error('macos:design:descent_close:K', 'S.K must be 1x%d.', N);
    end

    % ---- the front end: mirrors 1..N-2, free ----------------------------
    % t has N-2 entries: t(N-2) is mirror N-2 -> mirror N-1, which is the
    % spacing FIELD_D_ calls `a`.  So the paraxial trace of the front end
    % runs over the first N-3 spacings and the state is read leaving the
    % last of its mirrors.
    Rf = S.R(:).';   cf = logical(S.convex(:)).';   tf = S.t(:).';
    if need == 1
        % N = 3: the "front end" is M1 alone; nothing to trace through.
        f1  = tern_(cf(1), -Rf(1)/2, Rf(1)/2);
        ymk = P.D/2;             umk = -ymk/f1;
        yck = P.stop_ahead;      uck = 1 - yck/f1;
    else
        p   = afocal_first_order(Rf, tf(1:need-1), cf, 'D',P.D, ...
                                 'stop_ahead',P.stop_ahead);
        ymk = p.y_marginal(need);                % height ON mirror N-2 ...
        umk = p.u_marginal(need);                % ... and the angle AFTER it
        yck = p.y_chief(need);
        uck = p.u_chief(need);
    end
    % propagate the free spacing t(N-2) onto mirror N-1
    a  = tf(need);
    ym = ymk + a*umk;      um = umk;
    yc = yck + a*uck;      uc = uck;

    % ---- the closure ----------------------------------------------------
    % Both signs of the exit marginal height are closed; the one with a
    % positive last spacing is the layout.
    ytry = [-(P.D/2)/P.M, +(P.D/2)/P.M];
    best = [];   amb = false;
    for yout = ytry
        if isnan(opts.phi)
            [phi, ok] = root_(ym, um, yc, uc, yout, S.iface, opts);
            if ~ok, continue; end
        else
            phi = opts.phi;
        end
        [b, phiN] = close_(phi, ym, um, yout);
        if ~isfinite(b) || b <= 0.02, continue; end
        d = chief_(phi, ym, um, yc, uc, yout);
        cand = struct('phi',phi, 'b',b, 'phiN',phiN, 'd',d, 'yout',yout);
        if isempty(best), best = cand; else, amb = true; end
    end
    if isempty(best)
        C = struct('found',false, 'N',N);
        return;
    end
    if amb
        % Not fatal, but it must never be silent: two layouts satisfy the
        % same specification and the caller is entitled to know which it got.
        warning('macos:design:descent_close:ambiguous', ...
                ['both exit-marginal signs close a positive layout at N = ' ...
                 '%d; taking the first (y_out < 0).'], N);
    end

    phi   = best.phi;   b = best.b;   phiN = best.phiN;
    phiAll = [pow_(Rf, cf), phi, phiN];
    R  = [Rf, abs(2/phi), abs(2/phiN)];
    cv = [cf,  phi  < 0,   phiN < 0];
    t  = [tf, b];

    % A near-zero power is a FLAT, not a mirror (the descent's ruling 4), and
    % the count that matters for the requirement table is POWERED mirrors --
    % while the count that matters for the packaging PARITY is elements, i.e.
    % reflections.  Both are reported, because conflating them is exactly how
    % "N = 6 cannot be built" gets said about a train that can.
    isflat = abs(phiAll) < 1e-6;
    C = struct('found',true, 'N',N, 'n_elements',N, ...
               'n_powered',nnz(~isflat), 'n_flat',nnz(isflat), ...
               'flat_at',find(isflat), 'names',{names_(N)}, 'R',R, 't',t, ...
               'convex',cv, 'K',S.K(:).', 'phi',phiAll, 'iface',S.iface, ...
               'y_out_sign',sign(best.yout), 'pupil_dist',best.d);
    C.fo = afocal_first_order(R, t, cv, 'D',P.D, 'stop_ahead',P.stop_ahead);
    % z stations: the beam folds along z, one flip per reflection, M1 at 0.
    C.z = stations_(t);
    C.behind_m1 = C.z(end);
    % ---- and the closure has to BE one ----------------------------------
    C.residual = [C.fo.u_out, C.fo.mag/P.M - 1, C.fo.pupil_dist - S.iface];
    C.closed = max(abs(C.residual)) < 1e-9;
end

% =====================================================================
function [b, phiN] = close_(phi, ym, um, yout)
%CLOSE_  The MARGINAL half of the closure, verbatim from AFOCAL4_CLOSE's
%   FIELD_D_: the marginal ray fixes everything but the pupil, so imposing
%   the two first-order conditions is not a solve at all -- it is
%   substitution.  B sets the magnification, PHIN recollimates.
    u2   = um - ym*phi;
    b    = (yout - ym)/u2;              % mirror N-1 -> N: this sets M
    phiN = u2/yout;                     % the last mirror: this recollimates
end

function d = chief_(phi, ym, um, yc, uc, yout)
%CHIEF_  The residue: where the exit pupil lands for a penultimate power
%   PHI.  This is the one number PHI actually buys.
    [b, phiN] = close_(phi, ym, um, yout);
    uc2 = uc - yc*phi;
    yc3 = yc + b*uc2;
    uc3 = uc2 - yc3*phiN;
    d   = -yc3/uc3;
end

function [phi, ok] = root_(ym, um, yc, uc, yout, iface, opts)
%ROOT_  The pupil condition's root in the penultimate power -- scanned and
%   CHECKED, never bracketed and trusted (RESULTS rule 11).
    xs = linspace(opts.window(1), opts.window(2), opts.npts);
    g  = arrayfun(@(x) safe_(@() chief_(x, ym, um, yc, uc, yout) - iface), xs);
    k  = find(isfinite(g(1:end-1)) & isfinite(g(2:end)) & ...
              sign(g(1:end-1)) ~= sign(g(2:end)));
    % lowest |power| first: the weakest penultimate mirror is the one
    % closest to the train this rung was grown from.
    [~, ord] = sort(min(abs(xs(k)), abs(xs(k+1))));
    for j = k(ord(:).')
        try
            x = fzero(@(y) chief_(y, ym, um, yc, uc, yout) - iface, ...
                      [xs(j) xs(j+1)]);
        catch
            continue;
        end
        b = close_(x, ym, um, yout);
        if ~isfinite(b) || b <= 0.02 || b > 12, continue; end
        if abs(chief_(x, ym, um, yc, uc, yout) - iface) > 1e-9, continue; end
        phi = x;   ok = true;   return;
    end
    phi = NaN;   ok = false;
end

function y = safe_(f)
    try, y = f(); catch, y = NaN; end
end

function p = pow_(R, cvx)
    f = R/2;   f(cvx) = -f(cvx);   p = 1./f;
end

function z = stations_(t)
%STATIONS_  Vertex z with M1 at 0, the beam folding along z one flip per
%   reflection.  Sky is at -z, so BEHIND the primary is +z (the S4b sign
%   convention, and the whole packaging wall depends on it).
    z = zeros(1, numel(t)+1);
    s = -1;
    for k = 1:numel(t)
        z(k+1) = z(k) + s*t(k);   s = -s;
    end
end

function n = names_(N)
    n = arrayfun(@(k) sprintf('M%d',k), 1:N, 'UniformOutput',false);
end

function chk_(S, f, n)
    if ~isfield(S,f) || numel(S.(f)) ~= n
        error('macos:design:descent_close:shape', ...
              'S.%s must be 1x%d for N = %d.', f, n, S.N);
    end
end

function v = tern_(c,a,b), if c, v = a; else, v = b; end, end
