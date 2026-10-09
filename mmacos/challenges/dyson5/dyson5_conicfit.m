function T = dyson5_conicfit(deck, opts)
%DYSON5_CONICFIT  Is a solved FreeForm mirror an off-axis ASPHERE?  Fit a conic of free vertex, axis, R and K (and, second
%   form, + h^4 / h^6 about that vertex) to the mirror's REAL surface over its LIT patch -- the engine's own ray hits, every
%   field -- and report the residual departure (addendum 45, CC's ask after (b): M2 carries 1.3-2.1 mm of pole-centred
%   Zernike freeform at 56-185 mrad on every solve; if a re-fitted base takes it to tens of um the mirror is an off-axis
%   asphere, not a freeform, and the manufacturing story changes).
%   Surface model: MACOS's explicit Aspheric sag along the axis from the vertex (res_); residual = normal distance, m.
%   FORM TRAP: fit in the ENGINE's form.  MACOS Aspheric ADDS h4/h6 as explicit sag on top of the conic; a fit that puts them
%   INSIDE the implicit conic equation agrees only near K = -1 and missed B1's M3 (K -42) by 7 mm on emission.  And a small
%   SAG residual is not a small IMAGE residual: score the emitted deck (1.5k (b): 17 um p-v at 1.7 mrad slope -> edge 13 -> 119 um).  Start: the deck's own
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
        opts.emit (1,:) char = ''          % write the deck with each fitted mirror as Surface= Aspheric (vertex, axis, R, K, h4/h6)
    end
    macos.init(opts.model);  macos.load_rx(deck);
    txt = fileread(deck);  bl = regexp(txt, '\n\s*iElt=', 'split');
    T = struct('elt', {}, 'n', {}, 'conic', {}, 'asph', {}, 'start_res_um', {});  X = {};  HH = {};  SG = [];  AX = {};
    for k = opts.elts
        H = [];
        for q = 1:size(opts.dirs, 1)
            d = opts.dirs(q, :)';  macos.set_src_fov('src_pos', opts.apst - opts.stand*d, 'src_dir', d, 'zSrc', 1e22);  macos.modify();
            s = macos.trace(k);  ri = macos.get_ray_info(s.nRays);  H = [H, ri.pos(:, ri.ok_trace(:))]; %#ok<AGROW>
        end
        b = bl{k+1};  V0 = vec_(b, 'VptElt');  a0 = vec_(b, 'psiElt');  a0 = a0/norm(a0);  Kr = vec_(b, 'KrElt');  Kc = vec_(b, 'KcElt');
        e1 = null(a0');  e2 = e1(:, 2);  e1 = e1(:, 1);
        c0 = 1/Kr(1);  r1 = res_([V0; 0; 0; c0; Kc(1); 0; 0], H, a0, e1, e2);  r2 = res_([V0; 0; 0; -c0; Kc(1); 0; 0], H, a0, e1, e2);
        sgn = 1;  if rms(r2) < rms(r1), c0 = -c0;  sgn = -1; end   % the sign pairing of axis and curvature the deck implies
        x0 = [V0; 0; 0; c0; Kc(1); 0; 0];  rs = res_(x0, H, a0, e1, e2);
        o = optimoptions('lsqnonlin', 'Display', 'off', 'Algorithm', 'levenberg-marquardt', 'ScaleProblem', 'jacobian', ...
                         'MaxFunctionEvaluations', 20000, 'MaxIterations', 2000, 'FunctionTolerance', 1e-20, 'StepTolerance', 1e-14);
        m0 = logical([1 1 1 1 1 1 0 0 0]);  m1 = logical([1 1 1 1 1 1 1 0 0]);   % STAGED: K held first (a free K from a far start
        xa = x0;  xa(m0) = lsqnonlin(@(y) res_(put_(x0, m0, y), H, a0, e1, e2), x0(m0), [], [], o);   % wandered on the 3k (b) M2/M3: axis ~1.5 rad)
        x1 = xa;  x1(m1) = lsqnonlin(@(y) res_(put_(xa, m1, y), H, a0, e1, e2), xa(m1), [], [], o);  r1 = res_(x1, H, a0, e1, e2);
        x2 = lsqnonlin(@(y) res_(y, H, a0, e1, e2), x1, [], [], o);  r2 = res_(x2, H, a0, e1, e2);
        X{k} = x2;  HH{k} = H;  SG(k) = sgn;  AX{k} = {a0, e1, e2}; %#ok<AGROW>
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
    if ~isempty(opts.emit)
        best = Inf;
        for sk = [1 -1]
            for sa = [1 -1]
                t = emit_txt_(txt, X, SG, AX, opts.elts, sk, sa);  fid = fopen(opts.emit, 'w');  fprintf(fid, '%s', t);  fclose(fid);
                try, macos.load_rx(opts.emit); catch, continue, end %#ok<CTCH>
                e = 0;
                for k = opts.elts
                    Hk = [];
                    for q = 1:size(opts.dirs, 1)
                        d = opts.dirs(q, :)';  macos.set_src_fov('src_pos', opts.apst - opts.stand*d, 'src_dir', d, 'zSrc', 1e22);  macos.modify();
                        s = macos.trace(k);  ri = macos.get_ray_info(s.nRays);  Hk = [Hk, ri.pos(:, ri.ok_trace(:))]; %#ok<AGROW>
                    end
                    ax = AX{k};  if isempty(Hk), e = Inf; break, end
                    rk = res_(X{k}, Hk, ax{1}, ax{2}, ax{3});  e = max(e, max(abs(rk)));
                    fprintf('   emit check sk %+d sa %+d M%d: %d hits, residual to the fit rms %.3g / max %.3g um\n', sk, sa, k, size(Hk, 2), rms(rk)*1e6, max(abs(rk))*1e6);
                end
                if e < best, best = e;  bs = [sk sa]; end
            end
        end
        t = emit_txt_(txt, X, SG, AX, opts.elts, bs(1), bs(2));  fid = fopen(opts.emit, 'w');  fprintf(fid, '%s', t);  fclose(fid);
        fprintf('EMITTED %s: every mirror Surface= Aspheric from the fit (conventions sk %+d, sa %+d); its traced hits lie on the fitted surfaces to %.3g um (max)\n', ...
                opts.emit, bs(1), bs(2), best*1e6);
        T(1).emit = struct('file', opts.emit, 'sk', bs(1), 'sa', bs(2), 'hit_on_fit_um', best*1e6);
    end
end

function x = put_(x, m, y), x(m) = y; end

function t = emit_txt_(txt, X, SG, AX, elts, sk, sa)
%EMIT_TXT_  Each fitted mirror as Surface= Aspheric: VptElt = the fitted vertex, psiElt = the fitted axis, KrElt / KcElt /
%   AsphCoef from the fit (sk, sa = the deck's sign conventions for R and the h4/h6, found empirically by the caller); the Mon
%   and FF channels dropped; RptElt and TElt (the pole frame) kept.
    t = char(txt);  st = [regexp(t, '(?m)^\s*iElt=', 'start'), numel(t) + 1];
    parts = [{t(1:st(1)-1)}, arrayfun(@(k) t(st(k):st(k+1)-1), 1:numel(st)-1, 'uni', 0)];
    drop = {'MonZernType', 'nMonZernCoef', 'MonZernModes', 'MonZernCoef', 'lMon', 'FFZernType', 'nFFZernCoef', 'FFZernModes', ...
            'FFZernCoef', 'lFF', 'pFF', 'xFF', 'yFF', 'zFF', 'pMon', 'xMon', 'yMon', 'zMon', 'nAsphCoef', 'AsphCoef'};
    for k = elts
        x = X{k};  ax = AX{k};  a = ax{1} + x(4)*ax{2} + x(5)*ax{3};  a = a/norm(a);
        L = splitlines(string(parts{k+1}));  o = strings(0, 1);  inCoef = false;
        for i = 1:numel(L)
            sl = L(i);  key = regexp(char(sl), '^\s*(\w+)=', 'tokens', 'once');
            if isempty(key)
                if inCoef, continue, end                       % continuation rows of a dropped multi-row key
                o(end+1, 1) = sl; continue %#ok<AGROW>
            end
            key = key{1};  inCoef = any(strcmp(key, drop));
            if inCoef, continue, end
            switch key
                case 'Surface', sl = "          Surface=  Aspheric";
                case 'KrElt', sl = sprintf("            KrElt=%.16E", sk*SG(k)/x(6));
                case 'KcElt', sl = sprintf("            KcElt=%.16E\n        nAsphCoef=  2\n         AsphCoef=  %.16E %.16E", x(7), sa*x(8), sa*x(9));
                case 'VptElt', sl = sprintf("%17s=  %.16E  %.16E  %.16E", 'VptElt', x(1:3));
                case 'psiElt', sl = sprintf("%17s=  %.16E  %.16E  %.16E", 'psiElt', a);
            end
            o(end+1, 1) = sl; %#ok<AGROW>
        end
        parts{k+1} = char(strjoin(o, newline) + newline);
    end
    t = [parts{:}];
end

function r = res_(x, H, a0, e1, e2)
%RES_  Normal distance (first order) of the points H from the surface in MACOS's EXPLICIT Aspheric form: along the axis a from
%   the vertex V, sag(rho) = c rho^2 / (1 + sqrt(1 - (1+K) c^2 rho^2)) + A4 rho^4 + A6 rho^6 (the h4/h6 ADD to the conic sag;
%   an implicit form that puts them inside the conic equation agrees only near K = -1 -- it missed the B1 M3 (K -350) by 7 mm).
    V = x(1:3);  a = a0 + x(4)*e1 + x(5)*e2;  a = a/norm(a);  c = x(6);  K = x(7);  A4 = x(8);  A6 = x(9);
    Q = H - V;  z = a'*Q;  rho2 = max(sum(Q.^2, 1) - z.^2, 0);
    w = sqrt(max(1 - (1 + K)*c^2*rho2, 1e-12));  zs = c*rho2./(1 + w) + A4*rho2.^2 + A6*rho2.^3;
    dz = c./w + 4*A4*rho2 + 6*A6*rho2.^2;               % d sag / d rho, divided by rho
    r = ((z - zs)./sqrt(1 + rho2.*dz.^2))';
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
