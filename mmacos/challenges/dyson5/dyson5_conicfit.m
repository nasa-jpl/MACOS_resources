function T = dyson5_conicfit(deck, opts)
%DYSON5_CONICFIT  Is a solved FreeForm mirror an off-axis ASPHERE?  Fit a conic of free vertex, axis, R and K (and, second
%   form, + h^4 / h^6 about that vertex) to the mirror's REAL surface over its LIT patch -- the engine's own ray hits, every
%   field -- and report the residual departure (addendum 45, CC's ask after (b): M2 carries 1.3-2.1 mm of pole-centred
%   Zernike freeform at 56-185 mrad on every solve; if a re-fitted base takes it to tens of um the mirror is an off-axis
%   asphere, not a freeform, and the manufacturing story changes).
%   Surface model: with q = p - V, z = a.q, rho^2 = |q|^2 - z^2,  F = c (rho^2 + (1+K) z^2) - 2 z - 2 (A4 rho^4 + A6 rho^6) = 0
%   (A = 0 for the pure conic); residual = F / |grad F| (the normal distance to first order), m.  Start: the deck's own
%   vertex, axis (psiElt), KrElt, KcElt -- for the FreeForm decks of tFF/tGM that is the PARENT conic + the moves.
%   opts: .elts (1:3), .dirs (nf x 3 source directions), .apst, .stand (the tEP aim), .model (256).
    arguments
        deck (1,:) char
        opts.elts (1,:) double = 1:3
        opts.dirs (:,3) double = []
        opts.apst (3,1) double = [0; 0.19; 0]
        opts.stand (1,1) double = 1.0
        opts.model (1,1) double = 256
        opts.quiet (1,1) logical = false
    end
    macos.init(opts.model);  macos.load_rx(deck);
    txt = fileread(deck);  bl = regexp(txt, '\n\s*iElt=', 'split');
    T = struct('elt', {}, 'n', {}, 'conic', {}, 'asph', {}, 'start_res_um', {});
    for k = opts.elts
        H = [];
        for q = 1:size(opts.dirs, 1)
            d = opts.dirs(q, :)';  macos.set_src_fov('src_pos', opts.apst - opts.stand*d, 'src_dir', d, 'zSrc', 1e22);  macos.modify();
            s = macos.trace(k);  ri = macos.get_ray_info(s.nRays);  H = [H, ri.pos(:, ri.ok_trace(:))]; %#ok<AGROW>
        end
        b = bl{k+1};  V0 = vec_(b, 'VptElt');  a0 = vec_(b, 'psiElt');  a0 = a0/norm(a0);  Kr = vec_(b, 'KrElt');  Kc = vec_(b, 'KcElt');
        e1 = null(a0');  e2 = e1(:, 2);  e1 = e1(:, 1);
        c0 = 1/Kr(1);  r1 = res_([V0; 0; 0; c0; Kc(1); 0; 0], H, a0, e1, e2);  r2 = res_([V0; 0; 0; -c0; Kc(1); 0; 0], H, a0, e1, e2);
        if rms(r2) < rms(r1), c0 = -c0; end                     % the sign pairing of axis and curvature the deck implies
        x0 = [V0; 0; 0; c0; Kc(1); 0; 0];  rs = res_(x0, H, a0, e1, e2);
        o = optimoptions('lsqnonlin', 'Display', 'off', 'Algorithm', 'levenberg-marquardt', 'ScaleProblem', 'jacobian', ...
                         'MaxFunctionEvaluations', 20000, 'MaxIterations', 2000, 'FunctionTolerance', 1e-20, 'StepTolerance', 1e-14);
        m0 = logical([1 1 1 1 1 1 0 0 0]);  m1 = logical([1 1 1 1 1 1 1 0 0]);   % STAGED: K held first (a free K from a far start
        xa = x0;  xa(m0) = lsqnonlin(@(y) res_(put_(x0, m0, y), H, a0, e1, e2), x0(m0), [], [], o);   % wandered on the 3k (b) M2/M3: axis ~1.5 rad)
        x1 = xa;  x1(m1) = lsqnonlin(@(y) res_(put_(xa, m1, y), H, a0, e1, e2), xa(m1), [], [], o);  r1 = res_(x1, H, a0, e1, e2);
        x2 = lsqnonlin(@(y) res_(y, H, a0, e1, e2), x1, [], [], o);  r2 = res_(x2, H, a0, e1, e2);
        T(end+1) = struct('elt', k, 'n', size(H, 2), 'conic', sum_(x1, r1, V0, a0, e1, e2, H), 'asph', sum_(x2, r2, V0, a0, e1, e2, H), ...
                          'start_res_um', [rms(rs) max(rs) - min(rs)]*1e6); %#ok<AGROW>
        if ~opts.quiet
            c = T(end).conic;  h = T(end).asph;
            fprintf(['M%d (%d lit hits): the deck''s own parent conic + moves leaves rms %.1f / p-v %.1f um;\n' ...
                     '   best conic (vertex, axis, R, K free): rms %.2f / p-v %.2f um, max slope %.2f mrad; R %.5f m, K %.4f, vertex moved %.2f mm, axis tilted %.2f mrad\n' ...
                     '   + h4/h6 about that vertex:          rms %.2f / p-v %.2f um, max slope %.2f mrad; R %.5f m, K %.4f, A4 %.4g, A6 %.4g\n'], ...
                    k, T(end).n, T(end).start_res_um, c.rms_um, c.pv_um, c.slope_mrad, c.R, c.K, c.dV_mm, c.tilt_mrad, ...
                    h.rms_um, h.pv_um, h.slope_mrad, h.R, h.K, h.A4, h.A6);
        end
    end
end

function x = put_(x, m, y), x(m) = y; end

function r = res_(x, H, a0, e1, e2)
    V = x(1:3);  a = a0 + x(4)*e1 + x(5)*e2;  a = a/norm(a);  c = x(6);  K = x(7);  A4 = x(8);  A6 = x(9);
    Q = H - V;  z = a'*Q;  rho2 = sum(Q.^2, 1) - z.^2;
    F = c*(rho2 + (1 + K)*z.^2) - 2*z - 2*(A4*rho2.^2 + A6*rho2.^3);
    dF_drho2 = c - 2*(2*A4*rho2 + 3*A6*rho2.^2);  dF_dz = c*2*(1 + K)*z - 2;
    G = dF_drho2.*2.*(Q - a*z) + a*dF_dz;                   % grad of rho2 = 2 (q - a z), grad of z = a
    r = (F./vecnorm(G))';
end

function s = sum_(x, r, V0, a0, e1, e2, H)
    a = a0 + x(4)*e1 + x(5)*e2;  a = a/norm(a);
    % the residual's SLOPE: a total-degree-8 polynomial in the patch's own (u, v) (scaled to +-1) fitted to r, max |grad| at the hits
    f1 = e1 - a*(a'*e1);  f1 = f1/norm(f1);  f2 = cross(a, f1);  Q = H - mean(H, 2);  u = f1'*Q;  v = f2'*Q;  L = max(hypot(u, v));
    u = u/L;  v = v/L;  [I, J] = meshgrid(0:8);  m = (I + J) <= 8;  I = I(m);  J = J(m);
    Ir = I';  Jr = J';                               % (u(:).^I' parses as (u(:).^I)')
    M = (u(:).^Ir).*(v(:).^Jr);  cf = M\r(:);
    Mu = (Ir.*u(:).^max(Ir - 1, 0)).*(v(:).^Jr);  Mv = (u(:).^Ir).*(Jr.*v(:).^max(Jr - 1, 0));
    sl = hypot(Mu*cf, Mv*cf)/L;
    s = struct('rms_um', rms(r)*1e6, 'pv_um', (max(r) - min(r))*1e6, 'R', 1/x(6), 'K', x(7), 'A4', x(8), 'A6', x(9), ...
               'dV_mm', norm(x(1:3) - V0)*1e3, 'tilt_mrad', acos(min(1, abs(a'*a0)))*1e3, 'slope_mrad', max(sl)*1e3, 'x', x);
end

function v = vec_(t, key), m = regexp(t, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens', 'once');  v = sscanf(strrep(m{1}, 'D', 'E'), '%f'); end
