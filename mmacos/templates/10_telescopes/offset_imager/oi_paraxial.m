function out = oi_paraxial(R, t, req)
%OI_PARAXIAL  First-order properties / seed solve of the 3-mirror chain.
%
%   out = OI_PARAXIAL(R, T) computes first-order properties of a coaxial
%   three-mirror imager given SIGNED CODE V radii R = [R1 R2 R3] (m,
%   positive = centre of curvature at +z of the vertex) and SIGNED
%   thicknesses T = [t_m1_m2, t_m2_m3] (m, negative = beam travels -z;
%   the stop plane between m1 and m2 carries no power so only the NET
%   m1->m2 spacing enters).  Returns:
%     .EFL_m     effective focal length (h = EFL*tan(theta))
%     .BFD_m     SIGNED thickness m3 -> paraxial image (CODE V sign)
%     .petzval   c1 - c2 + c3 (1/m); 0 = flat field
%
%   out = OI_PARAXIAL(R1_seed, T, REQ) SEED SOLVE: given the M1 radius
%   (scalar first argument), the spacings, and requirements
%   REQ.EFL_m (+ optional REQ.BFD_m), solve the remaining radii:
%     no BFD requirement:  c2, c3 from { EFL = req, petzval = 0 }
%     with REQ.BFD_m:      c1, c2, c3 from { EFL, petzval = 0, BFD }
%   Returns the same struct plus .R = [R1 R2 R3] of the solved seed.
%
%   CONVENTIONS (verified against the rodgers3 r1 deck by real rays --
%   see the template README): paraxial trace in reduced angles w = n*u
%   with n = +1 flipping sign at each mirror; refraction
%   w' = w - y*phi, phi_i = c_i*(n' - n); transfer y' = y + (t/n')*w.
%   EFL = -y_in/w_out for an infinite-conjugate input (y_in, w_in = 0);
%   BFD = -y_last*n_out/w_out (signed CODE V thickness).
%
%   See also OFFSET_IMAGER_PARAMS, OI_DECK.

    if nargin < 3
        out = props_(R, t);
        return
    end

    % ---- seed solve ------------------------------------------------------
    R1 = R(1);
    if isfield(req,'BFD_m') && ~isempty(req.BFD_m)
        % three unknowns c = [c1 c2 c3]
        x0 = [1/R1; -1; -1];
        f  = @(c) [efl_([c(1) c(2) c(3)], t) - req.EFL_m;
                   c(1) - c(2) + c(3);
                   bfd_([c(1) c(2) c(3)], t) - req.BFD_m];
    else
        % two unknowns c = [c2 c3], c1 fixed by the R1 seed
        c1 = 1/R1;
        x0 = [-1; -1];
        f  = @(c) [efl_([c1 c(1) c(2)], t) - req.EFL_m;
                   c1 - c(1) + c(2)];
    end
    x = newton_(f, x0);
    if numel(x) == 3, c = x(:)'; else, c = [1/R1, x(1), x(2)]; end
    out = props_(1./c, t);
    out.R = 1./c;
end

% =========================================================================
function out = props_(R, t)
    c = 1./R(:)';
    [EFL, BFD] = trace_(c, t);
    out = struct('EFL_m',EFL, 'BFD_m',BFD, ...
                 'petzval', c(1) - c(2) + c(3), 'R', R(:)');
end

function e = efl_(c, t),  [e, ~] = trace_(c, t);  end
function b = bfd_(c, t),  [~, b] = trace_(c, t);  end

function [EFL, BFD] = trace_(c, t)
%TRACE_  y-w paraxial trace of the 3-mirror chain, infinite conjugate.
    y = 1;  w = 0;  n = 1;
    tt = [t(1) t(2)];
    for i = 1:3
        np  = -n;                       % mirror flips the index sign
        phi = c(i)*(np - n);
        w   = w - y*phi;
        if i < 3
            y = y + (tt(i)/np)*w;
        end
        n = np;
    end
    EFL = -1/w;                         % y_in = 1
    BFD = -y*n/w;
end

function x = newton_(f, x0)
%NEWTON_  Damped Newton with FD Jacobian (tiny well-conditioned systems).
    x = x0;  h = 1e-9;
    for it = 1:60
        r = f(x);
        if norm(r) < 1e-14, return; end
        J = zeros(numel(r), numel(x));
        for j = 1:numel(x)
            xp = x; xp(j) = xp(j) + h;
            J(:,j) = (f(xp) - r)/h;
        end
        dx = -J\r;
        % damp: halve until the residual shrinks (or give up the damping)
        s = 1;
        for k = 1:20
            if norm(f(x + s*dx)) < norm(r), break; end
            s = s/2;
        end
        x = x + s*dx;
    end
    if norm(f(x)) > 1e-9
        error('oi_paraxial:seed', ...
              'first-order seed did not converge (residual %g)', norm(f(x)));
    end
end
