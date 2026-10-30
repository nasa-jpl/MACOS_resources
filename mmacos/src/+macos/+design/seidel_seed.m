function [K, t_focus, EFL, S] = seidel_seed(R, t_between, D, convex, opts)
%SEIDEL_SEED  Third-order anastigmat conic seed for a coaxial mirror train.
%   [K, t_focus, EFL] = macos.design.seidel_seed(R, t_between, D) solves the
%   conic constants K (1xN) that null the Seidel spherical, coma and
%   astigmatism sums (S_I, S_II, S_III) of an N-mirror coaxial reflective
%   train, and returns the paraxial focus distance t_focus (last mirror ->
%   image, positive downstream) and the |EFL|.
%
%   [...] = seidel_seed(R, t_between, D, CONVEX) takes a 1xN logical giving
%   each mirror's ACTUAL curvature sense (true = convex to the light that
%   reaches it) -- when ANY entry is true.  An all-false CONVEX, like the
%   3-argument call, means the classical alternation concave, convex,
%   concave, ... (Cassegrain front, Korsch TMA): the Telescope builder's
%   default, which infers a Korsch secondary's convexity from geometry and
%   never flags it.  An all-concave train therefore cannot be seeded by
%   flags alone (none exists in the corpus; add an option if one appears).
%
%   [K, t_focus, EFL, S] = seidel_seed(..., 'stop', s) places the aperture
%   stop on mirror s (default 1); S returns the Seidel sums of the seeded
%   train (S.SI, S.SII, S.SIII, S.SIV), each ~0 for the solved ones.
%
%   THE MODEL (rewritten 2026-10-03).  Physical paraxial trace in a FIXED
%   frame: light enters along +z; every mirror flips the travel direction
%   d; the index before a mirror is n = d and after it n' = -d (the "n-flip"
%   of Welford / Smith); vertex curvatures carry their fixed-frame sign
%   (centre of curvature at +z of the vertex -> c > 0: a concave mirror has
%   c = -d/R, a convex one c = +d/R); the slope u is dy/dz; the transfer uses
%   the SIGNED vertex displacement z(k+1) - z(k) = -d(k) t(k); refraction is
%   n' u' = n u - y c (n' - n).  The Seidel sums are Welford's, with the
%   refraction invariants A = n (y c + u), Abar = n (ybar c + ubar) and the
%   conic term K c^3 (n' - n) y^4 scaled by (ybar/y)^k for S_II, S_III.
%   This nulls a single concave mirror's spherical aberration at K = -1
%   (the paraboloid) and reproduces the classical-Cassegrain and
%   Ritchey-Chretien closed forms (Schroeder) to round-off -- the gates in
%   tDesignTelescope.
%
%   WHY IT WAS REWRITTEN.  The 2026-06 port (`seidel_seed_nflip`, kept
%   verbatim) used |R| with a positive thickness after every reflection, so
%   the ray height at the THIRD mirror was wrong unless the M2->M3 space was
%   afocal (the proof_korsch and tma_fixture cases, which is why they
%   passed): on a Petzval-flat telecentric three-mirror (dyson5 beat 5,
%   R = [700 125 152] mm, t = [140 76] mm) it placed the focus 16 mm BEHIND
%   M3 and gave K3 = +265, where the exact first order has 141 mm in front
%   and the EFL 126.7 mm.  Its "convex" path got the focus right but handed
%   back K = 0.  The two-mirror cases never exposed it because their gates
%   check the Seidel residuals, not the image position.
%
%   Inputs (consistent units; metres in the design layer):
%     R         1xN vertex radii as POSITIVE MAGNITUDES (|KrElt|).
%     t_between 1x(N-1) vertex spacings along the light, positive.
%     D         aperture diameter (the marginal ray starts at y = D/2, u = 0).
%     convex    1xN logical, see above.
%   Outputs:
%     K         1xN conic constants (Schroeder / MACOS KcElt convention).
%     t_focus   mirror N -> image along the light (positive = downstream).
%     EFL       |effective focal length|.
%
%   N = 3 nulls S_I/II/III exactly; N = 2 nulls S_I and S_II (the RC); N > 3
%   returns the minimum-norm seed for optimize() to refine; N = 1 nulls S_I
%   (the paraboloid).
%
%   See also: macos.design.Telescope, macos.design.tma_layout,
%   macos.design.seidel_seed_nflip (legacy, for the record only).
    arguments
        R         (1,:) double
        t_between (1,:) double
        D         (1,1) double {mustBePositive}
        convex    (1,:) logical = false(1, numel(R))
        opts.stop (1,1) double {mustBeInteger, mustBePositive} = 1
    end
    N = numel(R);
    % An all-false CONVEX (the 3-argument call, and the Telescope builder's
    % default, which infers a Korsch secondary's convexity from geometry and
    % never flags it) means the classical alternation; any true flag makes
    % the vector the ACTUAL sense of every mirror.  This is exactly what the
    % 2026-06 code did on every existing call, so no caller moves.
    neg = R < 0;  R = abs(R);                  % a signed radius marks a convex mirror
    if any(neg), convex = convex | neg; end    % (the builder accepts 'radius_m', -|R|)
    if any(R == 0)
        error('macos:design:seidel_seed:zeroR', 'a mirror radius of 0 has no meaning here.');
    end
    if ~any(convex), convex = logical(mod(0:N-1, 2)); end
    if numel(t_between) ~= N-1
        error('macos:design:seidel_seed:dims', ...
            't_between must have N-1 = %d entries (got %d).', N-1, numel(t_between));
    end
    if numel(convex) ~= N
        error('macos:design:seidel_seed:convexdims', ...
            'convex must have N = %d entries (got %d).', N, numel(convex));
    end
    if opts.stop > N
        error('macos:design:seidel_seed:stop', 'stop mirror %d > N = %d.', opts.stop, N);
    end
    th = deg2rad(0.05);                        % small field for the chief ray

    % --- geometry in the fixed frame: travel direction, signed curvature, vertex z
    d  = (-1).^(0:N-1);                        % direction of the light REACHING mirror k
    c  = -d ./ R;  c(convex) = -c(convex);     % concave: centre on the incoming side
    dz = [-d(1:N-1) .* t_between, 0];          % z(k+1) - z(k) along the light after mirror k
    n  = d;  np = -d;

    % --- chief ray through the stop mirror's vertex: linear in its height at M1
    yb1 = chief_height_(c, dz, n, np, th, opts.stop);

    % --- base sums (spheres) and the conic sensitivities
    [base, g, img] = sums_(c, dz, n, np, D/2, 0, yb1, th, zeros(1, N));
    b = -[base.SI; base.SII; base.SIII];
    switch N
        case 1,     K = -base.SI / g(1, 1);
        case 2,     K = (g(1:2, :) \ b(1:2)).';
        case 3,     K = (g \ b).';
        otherwise,  K = (pinv(g) * b).';
    end
    [S, ~, img] = sums_(c, dz, n, np, D/2, 0, yb1, th, K);
    t_focus = img.t_focus;
    EFL     = img.EFL;
end

% =====================================================================
function yb1 = chief_height_(c, dz, n, np, th, s)
%CHIEF_HEIGHT_  Height at M1 of the chief ray (slope th) that crosses mirror s on axis.
    if s == 1, yb1 = 0; return; end
    y0 = height_at_(c, dz, n, np, 0, th, s);
    y1 = height_at_(c, dz, n, np, 1, th, s);
    yb1 = -y0 / (y1 - y0);                     % y(s) is affine in y(1)
end

function ys = height_at_(c, dz, n, np, y, u, s)
    for k = 1:s-1
        up = (n(k)*u - y*c(k)*(np(k)-n(k))) / np(k);
        y  = y + dz(k)*up;  u = up;
    end
    ys = y;
end

function [S, g, img] = sums_(c, dz, n, np, y, u, yb, ub, K)
%SUMS_  Welford's Seidel sums over the train, the conic sensitivities, the image.
    N = numel(c);  y1 = y;
    SI = 0; SII = 0; SIII = 0; SIV = 0;
    H  = n(1)*(ub*y - u*yb);                   % Lagrange invariant
    g  = zeros(3, N);
    for k = 1:N
        A   = n(k)*(y*c(k) + u);
        Ab  = n(k)*(yb*c(k) + ub);
        up  = (n(k)*u  - y *c(k)*(np(k)-n(k))) / np(k);
        ubp = (n(k)*ub - yb*c(k)*(np(k)-n(k))) / np(k);
        dun = up/np(k) - u/n(k);
        gI  = c(k)^3*(np(k)-n(k))*y^4;         % dS_I/dK of this mirror
        rho = 0;  if y ~= 0, rho = yb/y; end
        g(:, k) = [gI; gI*rho; gI*rho^2];
        SI   = SI   - A*A  *y*dun + K(k)*gI;
        SII  = SII  - A*Ab *y*dun + K(k)*gI*rho;
        SIII = SIII - Ab*Ab*y*dun + K(k)*gI*rho^2;
        SIV  = SIV  - H*H*c(k)*(1/np(k) - 1/n(k));
        if k < N
            y = y + dz(k)*up;  yb = yb + dz(k)*ubp;
        end
        u = up;  ub = ubp;
    end
    S = struct('SI', SI, 'SII', SII, 'SIII', SIII, 'SIV', SIV);
    % image: y + t u = 0 along the light leaving mirror N (direction -n(N) = np(N))
    img.t_focus = (-y/u) * np(N);
    img.EFL     = abs(y1/u);
end
