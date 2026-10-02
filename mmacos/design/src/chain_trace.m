function [pts, dirs, ok] = chain_trace(S, p0, d, lam, G)
%CHAIN_TRACE  Sequential exact 3-D trace through a surface chain.
%   [pts, dirs, ok] = chain_trace(S, p0, d, lam, G) traces the ray from p0
%   along d (any length) through the chain S in order; pts(:,k) = the hit on
%   surface k, dirs(:,k) = the direction AFTER surface k, ok = false on a
%   miss (a surface not reached along +t, a refraction past the critical
%   angle, an evanescent grating order).  This is spectrometer_geom's own
%   tracer, lifted verbatim (2026-10-01, dyson5 beat 5) so the telescope
%   chain and the end-to-end chain trace with the SAME code the Dyson gates
%   pin against the engine at 1e-9 m (tSpectrometerRx); spectrometer_geom's
%   trace_chain_ now calls this.
%
%   Surface records (the spectrometer_geom schema): .kind 'plane' | 'sphere'
%   | 'asph' (conic + even asphere about the axis psi through vpt, sag along
%   +psi, A(i) on h^(2i+2) -- the engine's AsphCoef convention); .C the
%   plane point / sphere centre; .R; .Kc; .A; .vpt; .psi; .root 'far' |
%   'near' | 'auto' (which sphere intersection is the surface); .act
%   'refract' (n_out = a number, or a glass name resolved through G.n(lam))
%   | 'reflect' | 'grating' (G.grating.m, .d, .sdir, .model) | 'pass' (a
%   station that records the hit and changes nothing: the slit plane inside
%   an end-to-end chain) | 'stop' (the terminal image plane).
    nS = numel(S);  pts = nan(3, nS);  dirs = nan(3, nS);  ok = true;
    p = p0(:);  d = d(:)/norm(d);  n_cur = 1;
    for k = 1:nS
        s = S(k);
        switch s.kind
        case 'plane'
            N = s.psi(:)/norm(s.psi);
            t = -((p - s.C(:))'*N)/(d'*N);
            if ~(t > 0), ok = false; return; end
            q = p + t*d;
        case 'sphere'
            t = sphere_t_(p - s.C(:), d, s.R, s.root);
            if isempty(t), ok = false; return; end
            q = p + t*d;  N = (q - s.C(:))/norm(q - s.C(:));
        case 'asph'
            % axisymmetric conic + even asphere about the axis psi through the
            % vertex: F(q) = (q - vpt).psi - sag(h) = 0, sag along +psi (toward
            % the CoC), h = |(q - vpt) - ((q - vpt).psi) psi|.  Newton from the
            % base-sphere root.
            t = sphere_t_(p - s.C(:), d, s.R, s.root);
            if isempty(t), ok = false; return; end
            a = s.psi(:);
            for it = 1:30
                q = p + t*d;  v = q - s.vpt(:);  z = v'*a;  hv = v - z*a;  h = norm(hv);
                [sg, dsg] = sag_(h, s.R, s.Kc, s.A);
                F = z - sg;
                if h > 0, hh = hv/h; else, hh = zeros(3,1); end
                gradF = a - dsg*hh;
                dF = gradF'*d;
                t = t - F/dF;
                if abs(F) < 1e-14, break; end
            end
            q = p + t*d;  v = q - s.vpt(:);  z = v'*a;  hv = v - z*a;  h = norm(hv);
            [~, dsg] = sag_(h, s.R, s.Kc, s.A);
            if h > 0, hh = hv/h; else, hh = zeros(3,1); end
            N = a - dsg*hh;  N = N/norm(N);
        end
        switch s.act
        case 'refract'
            n2 = idx_(s.n_out, G, lam);
            Ni = N;  if Ni'*d > 0, Ni = -Ni; end       % normal against the incoming ray
            d = refract_(d, Ni, n_cur, n2);
            if isempty(d), ok = false; return; end
            n_cur = n2;
        case 'reflect'
            d = d - 2*(d'*N)*N;
        case 'grating'
            % Two groove models.  'surface': the period d is constant ALONG
            % THE CURVED SURFACE -- the grating vector is m lambda/d times the
            % UNIT projection of sdir into the local tangent plane (this is
            % what the engine's Snells_Law_Grating does: it normalises shat).
            % 'planes': straight-ruled -- equidistant parallel groove PLANES
            % with spacing d along sdir, so the tangential kick is m lambda/d
            % times the UN-normalised projection (magnitude cos of the local
            % tilt), the classical Rowland concave grating.
            sdir = G.grating.sdir(:) - (G.grating.sdir(:)'*N)*N;
            if ~isfield(G.grating, 'model') || strcmp(G.grating.model, 'surface'), sdir = sdir/norm(sdir); end
            d_t = d - (d'*N)*N + (G.grating.m*lam/G.grating.d/n_cur)*sdir;
            val = 1 - norm(d_t)^2;
            if val < 0, ok = false; return; end
            d = d_t - sign(d'*N)*sqrt(val)*N;         % reflected: normal part flips
        case 'pass'
            % a recorded station (the slit plane in an end-to-end chain): nothing
        case 'stop'
            % image plane: nothing
        end
        pts(:,k) = q;  dirs(:,k) = d;  p = q;
    end
end

function [sg, dsg] = sag_(h, R, Kc, A)
%SAG_  Conic (radius R > 0, constant Kc) + even asphere sag along +psi and
%   its h-derivative; A(i) multiplies h^(2i+2) (engine AsphCoef convention).
    c = 1/R;  h2 = h*h;  rt = sqrt(1 - (1+Kc)*c*c*h2);
    sg = c*h2/(1 + rt);
    dsg = c*h/rt;
    for i = 1:numel(A)
        sg  = sg + A(i)*h^(2*i+2);
        dsg = dsg + (2*i+2)*A(i)*h^(2*i+1);
    end
end

function n = idx_(spec, G, lam)
    if ischar(spec), n = G.n(lam); else, n = spec; end
end

function t = sphere_t_(p, d, Rad, which)
    b = p'*d;  c = p'*p - Rad^2;  disc = b^2 - c;
    if disc < 0, t = [];  return;  end
    s = sqrt(disc);  t1 = -b - s;  t2 = -b + s;
    if strcmp(which, 'far'), t = t2;
    elseif t1 > 1e-12,       t = t1;      % 'near' and 'auto': the nearest positive root
    else,                    t = t2;
    end
    if ~(t > 1e-12), t = []; end
end

function dout = refract_(d, N, n1, n2)
    cosi = -(N'*d);  eta = n1/n2;  k = 1 - eta^2*(1 - cosi^2);
    if k < 0, dout = [];  return;  end
    dout = eta*d + (eta*cosi - sqrt(k))*N;
end
