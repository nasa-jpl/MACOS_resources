function G = tls_section(P, X, file)
%TLS_SECTION  The zig-zag section in 3-D: three off-axis conic sections on the chief.
%
%   G = TLS_SECTION(P, X) builds the section of the design vector X
%   (TLS_DESIGN; a first-order struct FO is accepted and seeded through
%   TLS_DESIGN).  It lays the chief ray out in the global y-z plane
%   (light enters along +z; the strip is along global x, OUT of the fold
%   plane) and puts at each chief hit an OFF-AXIS CONIC SECTION whose local
%   radii there are FO's first-order radii: R_t (in the fold plane) and R_s
%   (across it), R_t/R_s = 1/cos^2(AOI).  G = TLS_SECTION(P, FO, FILE) also
%   writes the MACOS deck FILE.
%
%   Geometry.  Mirror k turns the chief by 180 - 2*AOI(k) about +x with sense
%   P.turn(k); the section's local normal (toward its centre of curvature)
%   bisects the incoming and outgoing chiefs (concave) or its reverse
%   (convex, phi < 0).  A conic of revolution with parent radius R and conic
%   K, cut at height h from its axis, has there
%       R_s = h/sin(theta)          (theta = angle normal <-> axis)
%       R_t = R_s^3 / R^2,          R_s^2 = R^2 - K h^2
%   so given (R_t, R_s) and theta:  R = sqrt(R_s^3/R_t), h = R_s sin(theta),
%   K = (R^2 - R_s^2)/h^2.  theta = AOI (the default) gives K = -1 for every
%   mirror -- three off-axis PARABOLOIDS, the exact first-order section; any
%   other theta (P.axis_theta_deg) is a third-order choice made in the
%   figure stage.  The parent axis leans from the normal toward the incoming
%   chief (P.axis_side +1) or the outgoing one (-1).
%
%   MACOS conventions (macos/CLAUDE.md, the add_oap pattern): KrElt = -|R|,
%   psiElt = the parent axis toward the centre of curvature, VptElt = the
%   PARENT vertex, RptElt = the section pole (the chief hit), TElt = the
%   section frame (z = local normal, x = global x, the strip axis).  The stop
%   is an ELEMENT stop on M2 (P.stop) written as the element's `ApStop= dx dy`:
%   the offset of the pole from the parent vertex in M2's TElt x/y, which is
%   what the engine's STOP matches (tracesub STOP -> ChiefRayAiming).
%
%   G fields: d (3x4 chief directions: input, after M1..M3), H (3x4 chief
%   hits M1..M3 and the slit), n (3x3 local normals), m (struct per mirror:
%   name, concave, Rt, Rs, R, K, theta, h, axis, vertex, pole, sag, frame,
%   offset), slit (point, normal), src (pos, dir), stop_elt, stop_offset.
%
%   See also TLS_FIRST_ORDER, TMA_LONGSLIT_RUN.
if isfield(X, 'phi'), X = tls_design(P, X); end
L = X.legs;  aoi = X.aoi;  th = X.theta;
rotx = @(a) [1 0 0; 0 cos(a) -sin(a); 0 sin(a) cos(a)];
d = zeros(3, 4);  H = zeros(3, 4);  n = zeros(3, 3);
d(:, 1) = [0; 0; 1];  H(:, 1) = [0; 0; 0];
names = {'M1', 'M2', 'M3'};
m = struct('name', {}, 'concave', {}, 'Rt', {}, 'Rs', {}, 'R', {}, 'K', {}, 'theta', {}, 'h', {}, ...
           'axis', {}, 'vertex', {}, 'pole', {}, 'sag', {}, 'frame', {}, 'offset', {}, 'asph', {});
for k = 1:3
    din = d(:, k);
    dout = rotx(P.turn(k)*deg2rad(180 - 2*aoi(k)))*din;
    d(:, k+1) = dout;
    b = (dout - din)/norm(dout - din);               % faces the incoming beam
    cc = X.Rt(k) > 0;                                % concave
    nh = b;  if ~cc, nh = -b; end                    % toward the centre of curvature
    n(:, k) = nh;
    % the parent axis: theta from the normal, leaning toward the incoming chief
    % (side +1: -din on a concave mirror, +din on a convex one) or the outgoing
    sg = 1;  if ~cc, sg = -1; end
    if P.axis_side(k) > 0, r1 = -sg*din; else, r1 = sg*dout; end
    e = r1 - (r1.'*nh)*nh;  e = e/norm(e);
    t = deg2rad(th(k));
    ax = cos(t)*nh + sin(t)*e;
    Rt = abs(X.Rt(k));  Rs = abs(X.Rs(k));
    % The parent conic (R, K) and the pole height h such that conic + asphere
    % together give, at the pole, the designed normal angle theta to the axis
    % and the designed local radii.  For any surface of revolution z(h):
    %   tan(theta) = z'(h),  R_s = h/sin(theta),  R_t = (1 + z'^2)^1.5/z''.
    % With the asphere a(h) = sum A_j h^(2j+2) given, the conic must supply
    %   zc' = tan(theta) - a',  zc'' = (1 + tan^2)^1.5/R_t - a''.
    % Conic: zc' = h/sqrt(Q), zc'' = R^2/Q^1.5, Q = R^2 - (1+K) h^2.
    % Closed form -- so no DOF can bend the chief off the layout.
    A = X.asph(k, :).*1e-6./X.hlit(k).^(2*(1:size(X.asph, 2)) + 2);
    jj = 1:numel(A);
    h = Rs*sin(t);
    if abs(t) < 1e-12
        assert(abs(Rt - Rs) < 1e-12*Rs && ~any(A), 'tls_section: theta = 0 needs R_t = R_s and no asphere (mirror %d)', k);
        R = Rs;  K = 0;  zc = 0;  a = 0;
    else
        a   = sum(A.*h.^(2*jj + 2));
        a1  = sum((2*jj + 2).*A.*h.^(2*jj + 1));
        a2  = sum((2*jj + 2).*(2*jj + 1).*A.*h.^(2*jj));
        s1  = tan(t) - a1;  s2 = (1 + tan(t)^2)^1.5/Rt - a2;
        assert(s1 > 0 && s2 > 0, 'tls_section: the asphere on %s leaves no conic through the pole (slope %.3g, curvature %.3g)', names{k}, s1, s2);
        sq  = h/s1;  Q = sq^2;
        R   = sqrt(s2*Q^1.5);
        K   = (R^2 - Q)/h^2 - 1;
        zc  = h^2/(R + sq);                          % conic sag at h
    end
    tt = (cos(t)*ax - nh);  if norm(tt) > 0, tt = tt/norm(tt); end   % from the axis toward the pole
    V = H(:, k) - h*tt - (zc + a)*ax;                % parent vertex
    % self-check: the pole is ON the aspheric surface: transverse h, the conic satisfied at z - a
    pv = H(:, k) - V;  zz = pv.'*ax - a;  hh = norm(pv - (pv.'*ax)*ax);
    assert(abs(hh - h) < 1e-9 && abs(hh^2 - 2*R*zz + (1 + K)*zz^2) < 1e-9*R^2, 'tls_section: section construction failed at %s', names{k});
    % section frame: z = the local normal, x = global x (the strip axis)
    fx = [1; 0; 0] - nh(1)*nh;  fx = fx/norm(fx);  fy = cross(nh, fx);
    Fr = [fx fy nh];
    m(k) = struct('name', names{k}, 'concave', cc, 'Rt', X.Rt(k), 'Rs', X.Rs(k), 'R', R, 'K', K, 'theta', th(k), 'h', h, ...
                  'axis', ax, 'vertex', V, 'pole', H(:, k), 'sag', zc + a, 'frame', Fr, 'offset', [pv.'*fx, pv.'*fy], 'asph', A);
    H(:, k+1) = H(:, k) + L(k)*dout;
end
H(:, 4) = H(:, 4) + X.slit_dz*d(:, 4);              % focus
G = struct('d', d, 'H', H, 'n', n, 'm', m, 'X', X);
G.slit = struct('point', H(:, 4), 'normal', -d(:, 4));
G.src = struct('dir', d(:, 1), 'pos', H(:, 1) - 0.6*d(:, 1));
G.stop_elt = find(strcmpi(names, char(P.stop)));
if isempty(G.stop_elt), G.stop_elt = 2; end
G.stop_offset = m(G.stop_elt).offset;
if nargin > 2 && ~isempty(file), write_deck_(P, G, file); end
end

% ---------------------------------------------------------------------------
function write_deck_(P, G, file)
v3 = @(a) sprintf('%.16E  %.16E  %.16E', a(1), a(2), a(3));
L = {};
L{end+1} = '% MACOS prescription emitted by tma_longslit (templates/10_telescopes/tma_longslit)';
L{end+1} = '% Zig-zag long-slit TMA: three off-axis conic sections, element stop on M2, telecentric slit';
L{end+1} = '% Source Definition';
L{end+1} = ['        ChfRayDir=  ' v3(G.src.dir)];
L{end+1} = ['        ChfRayPos=  ' v3(G.src.pos)];
L{end+1} = '          zSource=1.0E+22';
L{end+1} = '        BaseUnits=  m';
L{end+1} = '        WaveUnits=  m';
L{end+1} = '           IndRef=1.0E+00';
L{end+1} = '           Extinc=0.0E+00';
L{end+1} = sprintf('          Wavelen=%.16E', P.lambda_m);
L{end+1} = '             Flux=1.0E+00';
L{end+1} = sprintf('         Aperture=%.16E', P.D_m);
L{end+1} = '         Obscratn=0.0E+00';
L{end+1} = '         GridType=  Circular';
L{end+1} = sprintf('         nGridpts=  %d', P.ngridpts);
L{end+1} = ['            xGrid=  ' v3([1 0 0])];
L{end+1} = ['            yGrid=  ' v3([0 1 0])];
L{end+1} = '% Element Definitions';
L{end+1} = '             nElt=  4';
for k = 1:3
    e = G.m(k);
    L{end+1} = sprintf('             iElt=  %d', k);                       %#ok<AGROW>
    L{end+1} = ['          EltName=  ' e.name];                            %#ok<AGROW>
    L{end+1} = '          Element=  Reflector';                           %#ok<AGROW>
    asp = any(e.asph ~= 0) || G.X.declare_asph;
    mon = isfield(G.X, 'mon') && (any(G.X.mon(k, :) ~= 0) || (isfield(G.X, 'declare_mon') && G.X.declare_mon));
    assert(~(asp && mon), 'tls_section: %s carries both an asphere and a pole-frame freeform (one surface type per element)', e.name);
    if asp, L{end+1} = '          Surface=  Aspheric';                   %#ok<AGROW>
    elseif mon, L{end+1} = '          Surface=  Monomial';               %#ok<AGROW>
    else, L{end+1} = '          Surface=  Conic'; end                     %#ok<AGROW>
    L{end+1} = sprintf('            KrElt=%.16E', -abs(e.R));             %#ok<AGROW>
    L{end+1} = sprintf('            KcElt=%.16E', e.K);                   %#ok<AGROW>
    if asp
        L{end+1} = sprintf('        nAsphCoef=  %d', numel(e.asph));                     %#ok<AGROW>
        L{end+1} = ['         AsphCoef=  ' strtrim(sprintf('%.16E ', e.asph))];         %#ok<AGROW>
    end
    if mon
        mc = zeros(1, 120);
        for t = 1:size(G.X.mon_terms, 1)
            i = G.X.mon_terms(t, 1);  j = G.X.mon_terms(t, 2);
            mc(1 + (i - 1)*(i + 2)/2 + j + 1) = G.X.mon(k, t)*1e-6;     % MonomialEval: k = 1 + sum_{n<i}(n+1) + j + 1
        end
        for r = 1:20
            s6 = strtrim(sprintf('%.16E ', mc(6*r - 5:6*r)));
            if r == 1, L{end+1} = ['          MonCoef=  ' s6]; else, L{end+1} = ['                    ' s6]; end   %#ok<AGROW>
        end
        L{end+1} = sprintf('             lMon=%.16E', G.X.lmon(k));    %#ok<AGROW>
        L{end+1} = ['             pMon=  ' v3(e.pole)];                %#ok<AGROW>
        L{end+1} = ['             xMon=  ' v3(e.frame(:, 1))];         %#ok<AGROW>
        L{end+1} = ['             yMon=  ' v3(e.frame(:, 2))];         %#ok<AGROW>
        L{end+1} = ['             zMon=  ' v3(e.frame(:, 3))];         %#ok<AGROW>
    end
    L{end+1} = ['           psiElt=  ' v3(e.axis)];                       %#ok<AGROW>
    L{end+1} = ['           VptElt=  ' v3(e.vertex)];                     %#ok<AGROW>
    L{end+1} = ['           RptElt=  ' v3(e.pole)];                       %#ok<AGROW>
    L{end+1} = '           IndRef=1.0E+00';                               %#ok<AGROW>
    L{end+1} = '           Extinc=0.0E+00';                               %#ok<AGROW>
    L{end+1} = '             nObs=  0';                                   %#ok<AGROW>
    L{end+1} = '           ApType=  None';                                %#ok<AGROW>
    if k == G.stop_elt
        L{end+1} = sprintf('           ApStop=  %.16E  %.16E', e.offset(1), e.offset(2));   %#ok<AGROW>
    end
    L{end+1} = '         PropType=  Geometric';                           %#ok<AGROW>
    L{end+1} = sprintf('             zElt=%.16E', G.X.legs(k));           %#ok<AGROW>
    L = [L, telt_(e.frame)];                                              %#ok<AGROW>
end
L{end+1} = '             iElt=  4';
L{end+1} = '          EltName=  Slit';
L{end+1} = '          Element=  FocalPlane';
L{end+1} = '          Surface=  Flat';
L{end+1} = '            KrElt=-1.0000000000000000E+22';
L{end+1} = '            KcElt=0.0000000000000000E+00';
L{end+1} = ['           psiElt=  ' v3(G.slit.normal)];
L{end+1} = ['           VptElt=  ' v3(G.slit.point)];
L{end+1} = ['           RptElt=  ' v3(G.slit.point)];
L{end+1} = '           IndRef=1.0E+00';
L{end+1} = '           Extinc=0.0E+00';
L{end+1} = '             nObs=  0';
L{end+1} = '           ApType=  None';
L{end+1} = '         PropType=  Geometric';
L{end+1} = '             zElt=1.0000000000000000E+20';
fz = G.slit.normal;  fx = [1; 0; 0];  fy = cross(fz, fx);
L = [L, telt_([fx fy fz])];
L{end+1} = '% Output Coordinate System Definition';
L{end+1} = '         nOutCord=  5';
T = [1 0 0 0 0 0 0; 0 1 0 0 0 0 0; 0 0 0 1 0 0 0; 0 0 0 0 1 0 0; 0 0 0 0 0 0 1];
for i = 1:5
    s = sprintf('%.16E  ', T(i, 1:6));  s = [s sprintf('%.1E', T(i, 7))];   %#ok<AGROW>
    if i == 1, L{end+1} = ['             Tout=  ' s]; else, L{end+1} = ['                    ' s]; end %#ok<AGROW>
end
fid = fopen(file, 'w');  assert(fid > 0, 'tls_section: cannot write %s', file);
fprintf(fid, '%s\n', L{:});  fclose(fid);
end

function L = telt_(F)
T = blkdiag(F, F);
L = {'          nECoord=  6'};
for i = 1:6
    s = strtrim(sprintf('%.16E  ', T(i, :)));
    if i == 1, L{end+1} = ['             TElt=  ' s]; else, L{end+1} = ['                    ' s]; end %#ok<AGROW>
end
end
