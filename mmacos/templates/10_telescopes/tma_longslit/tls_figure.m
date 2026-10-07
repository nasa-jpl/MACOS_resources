function [X, R] = tls_figure(P, X0, dofs, opts)
%TLS_FIGURE  One rung of the figure ladder: an engine-traced least-squares solve of the section.
%
%   [X, R] = TLS_FIGURE(P, X0, DOFS) varies the design-vector fields named in
%   DOFS (TLS_DESIGN) from X0 and returns the solved X and a record R.  DOFS
%   is a cell of group names:
%     'Rt' 'Rs' 'theta'   the three local radii / off-axis angles (3 each)
%     'slit_dz'           focus (1)
%     'asph'              h^4, h^6 sag per mirror at its lit radius (6)
%     'aoi'               the three chief AOIs (3)  -- the layout
%     'legs'              the three chief path lengths (3) -- the layout
%   With X.closure (TLS_DESIGN's default) R_t and R_s are NOT DOFs: they are
%   re-derived from the first order of the current legs and AOIs at every
%   evaluation (TLS_FIRST_ORDER), so EFL, the back focus and telecentricity
%   hold exactly (paraxially) and the solve moves only the figure (theta ->
%   the conic, the aspheres), the focus and the layout.
%   Every evaluation regenerates the deck from X (TLS_SECTION: the chief stays
%   on the three poles, the stop stays exactly at the M2 pole) and traces each
%   solve field in the ENGINE.  Rows per field, all in um:
%     SPOT   every ray of the field's seed set: its in-plane offset on the slit
%            from the field's centroid, AS PLACED (no refocus), x sqrt(254/N)
%            (the tGM / CALIB SPOT balance) x the field's P.solve_field_wt;
%            the ALONG-slit component x P.w_spot_x, the across-slit x
%            P.w_spot_y (the along-slit width is the CRF the Dyson cannot fix;
%            the across-slit width is truncated by the slit itself); a lost
%            ray is a 1 mm row
%     PLATE  the chief's slit-axis position minus f tan(theta)   (x P.w_plate)
%     BOW    the chief's across-slit position minus the centre field's: the
%            strict chief intercept, a straight slit needs a straight image
%            (x P.w_bow)
%     CBOW   the same for the field's spot CENTROID (x P.w_bow): the
%            spectrometer's smile reads centroids, and coma moves the centroid
%            off the chief -- R4 of 2026-10-07 held the chief bow to 2.7 um
%            while the centroid bowed 9.1 um (0.5 px) and the e2e smile read
%            0.73 px; R5's 0.6 um centroid bow gave 0.09 px
%     TELE   the chief's angle to the slit normal, both components, rad x
%            P.w_tel (um per rad)
%     CONE   the working F/# 1/(2 sin u) of the passing rays, slit-axis and
%            across-slit, outside P.cone_fnum: a hinge, x P.w_cone (um per
%            unit F/#)
%     WD     the working distance below P.work_dist_m (layout rungs): hinge,
%            x P.w_wd (um per m)
%     PLATE_Y the along-track (fold-plane) focal length at the centre and
%            edge fields, from a 0.05 deg along-track probe, vs f (x P.w_plate)
%            -- the first-order half of the paper's F/# anamorphicity
%            constraint (the CONE rows are the aberrated half)
%   Solve fields: P.solve_fields_deg (default 0..strip half-angle in 5: the
%   section is mirror-symmetric about the fold plane, so +theta and -theta
%   are the same field).  Solver: lsqnonlin Levenberg-Marquardt with Jacobian
%   scaling (the dyson5 tGM settings; residuals in um, not metres).
%
%   R fields: dofs, x0, x, names, cost0, cost, exitflag, iterations,
%   evaluations, seconds, firstorderopt.
%
%   See also TLS_DESIGN, TLS_SECTION, TLS_MEASURE, TMA_LONGSLIT_RUN.
arguments
    P struct
    X0 struct
    dofs cell
    opts.maxfev (1,1) double = P.maxfev
    opts.quiet (1,1) logical = false
end
[x0, names, put, lb, ub] = pack_(X0, dofs, P);
fd = P.solve_fields_deg;  if isempty(fd), fd = linspace(0, P.strip_half_deg, 5); end
if ~isfield(P, 'solve_field_wt') || isempty(P.solve_field_wt), P.solve_field_wt = ones(1, numel(fd)); end
assert(numel(P.solve_field_wt) == numel(fd), 'tls_figure: solve_field_wt has %d weights for %d solve fields', numel(P.solve_field_wt), numel(fd));
if ~isfield(P, 'w_spot_x'), P.w_spot_x = 1; end
if ~isfield(P, 'w_spot_y'), P.w_spot_y = 1; end
deck = [tempname '_tlsfig.in'];  ftmp = [tempname '_tlsfld.in'];
cln = onCleanup(@() cellfun(@(f) delete_if_(f), {deck, ftmp}));
% the fixed ray set: what passes at the seed
c = struct('P', P, 'X0', X0, 'put', put, 'fd', fd, 'deck', deck, 'ftmp', ftmp, 'f', P.f_m);
c.sel = cell(1, numel(fd));
X = put(X0, x0);  G = tls_section(P, X, deck);  txt = fileread(deck);
for q = 1:numel(fd)
    ri = field_trace_(txt, fd(q), ftmp);  ok = ri.ok_trace & ri.ok_pass;  ok(1) = false;  c.sel{q} = find(ok);
end
c.G0 = G;
c.nrows = sum(2*cellfun(@numel, c.sel) + 7) + 1 + 2;
c.dy_deg = 0.05;                                  % the along-track plate-scale probe
fun = @(x) resid_(x, c);
tic;  r0 = fun(x0);  t1 = toc;
if ~opts.quiet
    fprintf('tls_figure: %d DOF (%s), %d rows, %d fields, seed cost %.4e, %.2f s per evaluation\n', numel(x0), strjoin(dofs, '+'), ...
            numel(r0), numel(fd), sum(r0.^2), t1);
end
o = optimoptions('lsqnonlin', 'Display', 'off', 'MaxFunctionEvaluations', opts.maxfev, 'MaxIterations', 2000, ...
                 'FunctionTolerance', 1e-12, 'StepTolerance', 1e-10, 'OptimalityTolerance', 1e-6, ...
                 'Algorithm', 'levenberg-marquardt', 'ScaleProblem', 'jacobian');
tic;
if opts.maxfev > 0
    [x, rn, ~, ef, out] = lsqnonlin(fun, x0, lb, ub, o);
else
    x = x0;  rn = sum(r0.^2);  ef = NaN;  out = struct('iterations', 0, 'funcCount', 1, 'firstorderopt', NaN);
end
sec = toc;
X = put(X0, x);
R = struct('dofs', {dofs}, 'names', {names}, 'x0', x0, 'x', x, 'cost0', sum(r0.^2), 'cost', rn, 'exitflag', ef, ...
           'iterations', out.iterations, 'evaluations', out.funcCount, 'seconds', sec, 'firstorderopt', out.firstorderopt, ...
           'fields_deg', fd, 'rows', rowsplit_(fun(x), c));
if ~opts.quiet
    fprintf('tls_figure: exitflag %d, %d iterations, %d evaluations, %.0f s, cost %.4e -> %.4e\n', ef, out.iterations, ...
            out.funcCount, sec, R.cost0, R.cost);
end
end

% ---------------------------------------------------------------------------
function [x0, names, put, lb, ub] = pack_(X, dofs, P)
% DOF vector in solve units: radii / legs / focus in mm, angles in deg, asphere in um of sag
x0 = [];  names = {};  map = {};  lb = [];  ub = [];
for g = dofs(:)'
    switch g{1}
        case {'Rt', 'Rs'}, v = X.(g{1})*1e3;  nm = arrayfun(@(k) sprintf('%s%d mm', g{1}, k), 1:3, 'uni', 0);
        case 'theta',      v = X.theta;       nm = arrayfun(@(k) sprintf('theta%d deg', k), 1:3, 'uni', 0);
        case 'aoi',        v = X.aoi;         nm = arrayfun(@(k) sprintf('aoi%d deg', k), 1:3, 'uni', 0);
        case 'legs',       v = X.legs*1e3;    nm = arrayfun(@(k) sprintf('leg%d mm', k), 1:3, 'uni', 0);
        case 'slit_dz',    v = X.slit_dz*1e3; nm = {'slit dz mm'};
        case 'asph',       v = reshape(X.asph', 1, []);  nm = {'M1 a4 um', 'M1 a6 um', 'M2 a4 um', 'M2 a6 um', 'M3 a4 um', 'M3 a6 um'};
        case {'mon', 'mon34'}                                       % pole-frame freeform: all terms, or degree 3-4 only
            sel = true(1, size(X.mon_terms, 1));  if strcmp(g{1}, 'mon34'), sel = X.mon_terms(:, 1)' <= 4; end
            [kk, tt] = ndgrid(1:3, find(sel));  v = X.mon(sub2ind(size(X.mon), kk(:), tt(:)))';
            nm = arrayfun(@(a, b) sprintf('M%d x%dy%d um', a, X.mon_terms(b, 2), X.mon_terms(b, 1) - X.mon_terms(b, 2)), kk(:)', tt(:)', 'uni', 0);
            map(end+1, :) = {'mon', numel(x0) + (1:numel(v)), sub2ind(size(X.mon), kk(:), tt(:))};   %#ok<AGROW>
            x0 = [x0, v(:)'];  names = [names, nm];  lb = [lb, -inf(1, numel(v))];  ub = [ub, inf(1, numel(v))];   %#ok<AGROW>
            continue
        otherwise, error('tls_figure: unknown DOF group ''%s''', g{1});
    end
    map(end+1, :) = {g{1}, numel(x0) + (1:numel(v)), []};   %#ok<AGROW>
    x0 = [x0, v(:)'];  names = [names, nm];             %#ok<AGROW>
    lo = -inf(1, numel(v));  hi = inf(1, numel(v));
    if strcmp(g{1}, 'theta'), lo(:) = 1;  hi(:) = 80; end   % the off-axis angle: 90 deg is the degenerate equator section
    % the layout stays the FORM it was seeded as: unbounded, the AOIs collapse to ~2 deg (a near-coaxial train that
    % self-obscures by 70 mm -- try 4's R3, 2026-10-07); the bounds are the template's P.aoi_span_deg / P.leg_span
    if strcmp(g{1}, 'aoi'),  lo = P.aoi_deg - P.aoi_span_deg;  hi = P.aoi_deg + P.aoi_span_deg; end
    if strcmp(g{1}, 'legs'), lo = P.legs_m*1e3*(1 - P.leg_span);  hi = P.legs_m*1e3*(1 + P.leg_span); end
    lb = [lb, lo];  ub = [ub, hi];                      %#ok<AGROW>
end
put = @(X, x) put_(X, x, map, P);
end

function X = put_(X, x, map, P)
for i = 1:size(map, 1)
    v = x(map{i, 2});
    switch map{i, 1}
        case {'Rt', 'Rs'}, X.(map{i, 1}) = v*1e-3;
        case {'theta', 'aoi'}, X.(map{i, 1}) = v;
        case 'legs', X.legs = v*1e-3;
        case 'slit_dz', X.slit_dz = v*1e-3;
        case 'asph', X.asph = reshape(v, 2, 3)';  X.declare_asph = true;
        case 'mon', X.mon(map{i, 3}) = v;  X.declare_mon = true;
    end
end
% FIRST-ORDER CLOSURE (afocal4 doctrine: identities are re-derived, never penalized): the local radii at the poles are
% re-solved from the first order of the CURRENT legs and AOIs, so EFL, the back focus and telecentricity hold exactly
% at every iterate.  (Two solves with R_t/R_s as free DOFs traded the first order for spots -- F/2.37 across the slit,
% then the slit 117 mm in and the chiefs 2.2 deg off: soft rows lose to ~12k spot rows.)
if isfield(X, 'closure') && X.closure
    Pc = P;  Pc.legs_m = X.legs;  Pc.aoi_deg = X.aoi;
    FO = tls_first_order(Pc);  X.Rt = FO.Rt;  X.Rs = FO.Rs;
end
end

function ri = field_trace_(txt, fdeg, ftmp, adeg)
if nargin < 4, adeg = 0; end                      % adeg: along-track (fold-plane) field angle, deg
sx = sind(fdeg);  sy = sind(adeg);  d = [sx; sy; sqrt(1 - sx^2 - sy^2)];
t2 = regexprep(txt, '(?m)^(\s*ChfRayDir=).*$', sprintf('$1  %.16E  %.16E  %.16E', d), 'dotexceptnewline');
fid = fopen(ftmp, 'w');  fwrite(fid, t2);  fclose(fid);
macos.load_rx(ftmp);  nE = macos.num_elt();  s = macos.trace(nE);  ri = macos.get_ray_info(s.nRays);
end

function r = resid_(x, c)
P = c.P;  X = c.put(c.X0, x);
try
    G = tls_section(P, X, c.deck);
catch
    r = 1e3*ones(c.nrows, 1);  return            % no conic meets the pole: a wall, not an error
end
txt = fileread(c.deck);
ez = -G.slit.normal(:);  ex = [1; 0; 0];  ey = cross(ez, ex);
r = [];  yc = NaN;
for q = 1:numel(c.fd)
    sl = c.sel{q};  n = numel(sl);
    ri = field_trace_(txt, c.fd(q), c.ftmp);
    ok = ri.ok_trace & ri.ok_pass;  okq = ok(sl);
    if nnz(okq) < 0.9*n, r = [r; 1e3*ones(2*n + 7, 1)]; continue, end   %#ok<AGROW> lost rays: a wall
    Pq = ri.pos(:, sl);  d = Pq - mean(Pq(:, okq), 2);  du = ex'*d;  dv = ey'*d;  du(~okq) = 1e-3;  dv(~okq) = 1e-3;
    w = sqrt(254/n)*P.solve_field_wt(q);                               % per-field weight (outer fields up)
    pc = ri.pos(:, 1) - G.slit.point(:);  cx = ex'*pc;  cy = ey'*pc;
    cen = mean(Pq(:, okq), 2) - G.slit.point(:);  ccy = ey'*cen;          % the field's spot CENTROID across the slit
    if q == 1, yc = cy;  ycc = ccy; end                                 % field 1 = the centre (fd(1) = 0)
    cd = ri.dir(:, 1)/norm(ri.dir(:, 1));
    Dd = ri.dir(:, ok)./vecnorm(ri.dir(:, ok));
    ax = atan2(Dd.'*ex, Dd.'*ez);  ay = atan2(Dd.'*ey, Dd.'*ez);
    Fx = 1/(2*sin((max(ax) - min(ax))/2));  Fy = 1/(2*sin((max(ay) - min(ay))/2));
    hinge = @(F) max(0, F - P.cone_fnum(2) - P.cone_tol) + max(0, P.cone_fnum(1) - P.cone_tol - F);
    r = [r; P.w_spot_x*w*du(:)*1e6; P.w_spot_y*w*dv(:)*1e6; ...
         P.w_plate*(cx - c.f*tand(c.fd(q)))*1e6; P.w_bow*(cy - yc)*1e6; ...
         P.w_tel*atan2(cd'*ex, cd'*ez); P.w_tel*atan2(cd'*ey, cd'*ez); ...
         P.w_cone*hinge(Fx); P.w_cone*hinge(Fy); ...
         P.w_bow*(ccy - ycc)*1e6];                                       %#ok<AGROW> CBOW: the centroid bow (what smile reads)
end
r = [r; P.w_wd*max(0, P.work_dist_m - (X.legs(3) + X.slit_dz))];   % the slit moves with the focus
% PLATE_Y: the along-track (fold-plane) focal length, at the centre and the edge field -- without it the solve trades
% the fold-plane first order for spots (R1 of 2026-10-07: F/2.37 across the slit, the slit 20 mm off)
for q = [1 numel(c.fd)]
    ri0 = field_trace_(txt, c.fd(q), c.ftmp);  ri1 = field_trace_(txt, c.fd(q), c.ftmp, c.dy_deg);
    dy = ey'*(ri1.pos(:, 1) - ri0.pos(:, 1));
    r = [r; P.w_plate*(abs(dy) - c.f*tand(c.dy_deg))*1e6];   %#ok<AGROW>
end
end

function S = rowsplit_(r, c)
% the cost by row family at the solution (for the rung table)
S = struct('spot', 0, 'plate', 0, 'bow', 0, 'tele', 0, 'cone', 0, 'wd', 0, 'plate_y', 0, 'cbow', 0);
i = 0;
for q = 1:numel(c.fd)
    n = numel(c.sel{q});
    S.spot = S.spot + sum(r(i + (1:2*n)).^2);  i = i + 2*n;
    S.plate = S.plate + r(i + 1)^2;  S.bow = S.bow + r(i + 2)^2;  S.tele = S.tele + sum(r(i + (3:4)).^2);
    S.cone = S.cone + sum(r(i + (5:6)).^2);  S.cbow = S.cbow + r(i + 7)^2;  i = i + 7;
end
S.wd = r(end - 2)^2;  S.plate_y = sum(r(end - 1:end).^2);
end

function delete_if_(f), if exist(f, 'file'), delete(f); end, end
