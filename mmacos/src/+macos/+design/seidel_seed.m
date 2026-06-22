function [K, t_focus, EFL] = seidel_seed(R, t_between, D)
%SEIDEL_SEED  Korsch-anastigmat conic seed for a coaxial mirror train.
%   [K, t_focus, EFL] = macos.design.seidel_seed(R, t_between, D) solves
%   the conic constants K (1xN) that null the 3rd-order Seidel spherical,
%   coma and astigmatism sums (S_I, S_II, S_III) for an N-mirror coaxial
%   reflective layout, using the n-flip unfolded paraxial model (ported
%   from optical_design/seidel.py, which is validated against the trusted
%   2-mirror RC and classical-Cassegrain fixtures).  It also returns the
%   paraxial focus distance t_focus (last mirror -> image) and |EFL|.
%
%   Inputs (consistent units; metres in the design layer):
%     R         1xN mirror vertex radii as POSITIVE MAGNITUDES in the n-flip
%               unfolded model (the validated reference is all-positive:
%               seidel_seed([8 2 4],[3 4.5],1) -> f/8 with a convex secondary).
%               Convexity is the GEOMETRY (a secondary before the M1 focus,
%               t1 < f1), not the radius sign -- a negative R here would
%               corrupt the paraxial trace.  The emitter stores KrElt = -|R|
%               for every mirror (MACOS convention; see j18mono's convex SM).
%     t_between 1x(N-1) vertex spacings M1->M2, ..., M(N-1)->MN.
%     D         aperture diameter.
%
%   Outputs:
%     K         1xN conic constants (Schroeder/MACOS KcElt convention).
%     t_focus   MN->image vertex distance (paraxial marginal focus).
%     EFL       |effective focal length|.
%
%   N=3 nulls S_I/II/III exactly (Korsch TMA).  N>3 returns the minimum-
%   norm seed (extra conics) for multi-field optimisation to refine.
%   N<3 errors (the 2-mirror families use the closed forms in resolve_).
%
%   Reference target (proof_korsch): seidel_seed([8 2 4],[3 4.5],1) ->
%   K ~ [-0.622 0.148 -3.904], EFL = 8 (f/8).
%
%   See also: macos.design.Telescope, optical_design/seidel.py.
    arguments
        R         (1,:) double
        t_between (1,:) double
        D         (1,1) double {mustBePositive}
    end
    N = numel(R);
    if N < 3
        error('macos:design:seidel_seed:tooFew', ...
            'need >= 3 mirrors for the Seidel anastigmat seed (got %d).', N);
    end
    if numel(t_between) ~= N-1
        error('macos:design:seidel_seed:dims', ...
            't_between must have N-1 = %d entries (got %d).', N-1, numel(t_between));
    end
    th = deg2rad(0.05);                  % small field for coma/astig scaling

    % --- paraxial marginal focus after the last mirror -> t_focus ---
    n = 1.0; y = D/2; u = 0.0;
    for k = 1:N-1
        c = 1/R(k); np_ = -n;
        u = (n*u - y*((np_-n)*c))/np_;  y = y + t_between(k)*u;  n = np_;
    end
    c = 1/R(N); np_ = -n;
    u = (n*u - y*((np_-n)*c))/np_;
    t_focus = -y/u;
    t = [t_between, t_focus];             % full spacing list (last = focus)

    % --- base trace (all spheres) for S_I/II/III ---
    base = seidel_trace_(R, zeros(1,N), t, D/2, 0, 0, th);

    % --- aspheric sensitivities g_j = [dS_I; dS_II; dS_III]/dK_j ---
    n = 1.0; y = D/2; u = 0.0; yb = 0.0; ub = th;
    g = zeros(3, N);
    for k = 1:N
        c = 1/R(k); np_ = -n;
        gI = (np_-n)*c^3*y^4;
        rho = 0; if y ~= 0, rho = yb/y; end
        g(:,k) = [gI; gI*rho; gI*rho*rho];
        phi = (np_-n)*c;
        up  = (n*u  - y*phi)/np_;
        ubp = (n*ub - yb*phi)/np_;
        y = y + t(k)*up;  yb = yb + t(k)*ubp;  u = up; ub = ubp;  n = np_;
    end

    b = -[base.SI; base.SII; base.SIII];
    if N == 3
        K = (g \ b).';                   % exact 3x3
    else
        K = (pinv(g) * b).';             % min-norm seed for N>3
    end

    r   = seidel_trace_(R, K, t, D/2, 0, 0, th);
    EFL = abs(r.EFL);
end

% =====================================================================
function out = seidel_trace_(R, K, t, y1, u1, yb1, ub1)
%SEIDEL_TRACE_  n-flip unfolded paraxial + Seidel sums for coaxial mirrors.
%   Marginal ray (y1,u1), chief ray (yb1,ub1); each mirror flips n -> -n.
%   Returns S_I..S_IV and EFL = -y1/u_final.
    n = 1.0; y = y1; u = u1; yb = yb1; ub = ub1;
    SI = 0; SII = 0; SIII = 0; SIV = 0;
    H = n*(ub*y - u*yb);                  % Lagrange invariant (constant)
    for k = 1:numel(R)
        c = 1/R(k); np_ = -n;
        phi = (np_-n)*c;
        A  = n*(y*c + u);                 % marginal refraction invariant
        Ab = n*(yb*c + ub);              % chief refraction invariant
        up  = (n*u  - y*phi)/np_;
        ubp = (n*ub - yb*phi)/np_;
        dun = up/np_ - u/n;
        sI   = -A*A   * y * dun;
        sII  = -A*Ab  * y * dun;
        sIII = -Ab*Ab * y * dun;
        sIV  = -H*H * c * (1.0/np_ - 1.0/n);
        A4  = K(k)*c^3/8.0;               % 4th-order conic departure
        sIa = 8.0*(np_-n)*A4*y^4;
        rho = 0; if y ~= 0, rho = yb/y; end
        SI   = SI   + sI   + sIa;
        SII  = SII  + sII  + sIa*rho;
        SIII = SIII + sIII + sIa*rho*rho;
        SIV  = SIV  + sIV;
        y  = y  + t(k)*up;
        yb = yb + t(k)*ubp;
        u = up; ub = ubp; n = np_;
    end
    EFL = Inf; if u ~= 0, EFL = -y1/u; end
    out = struct('SI',SI,'SII',SII,'SIII',SIII,'SIV',SIV,'EFL',EFL);
end
