function C = tls_clearance_joined(P, efile, opts)
%TLS_CLEARANCE_JOINED  Clearance of a telescope + spectrometer JOINED deck, every body from ENGINE rays.
%
%   C = TLS_CLEARANCE_JOINED(P, EFILE) loads the joined deck EFILE (a dyson5_t5f
%   end-to-end emission: the telescope's mirrors, then the Slit and the Dyson
%   blocks), launches the strip's centre and edge fields through the deck's
%   object-space ApStop and aims each through the grating (the instrument's
%   stop, macos.stop twice as t5f does), traces every element and scores the
%   clearance with the dyson5 record's rule (TLS_CLEARANCE): every body is its
%   lit footprint over the launched fields grown by the mount (P.mount_m),
%   lifted onto a quadric fitted to its own ray hits; the slit mask is a plate
%   (slit length + 10 mm) x 4 mm.  Surfaces that are one physical part (the
%   block's two face passes, its two sphere passes) are one body.
%
%   WHY NOT t5f's live clearance: it lifts each mirror onto its PARENT's base
%   sphere about the parent vertex; these sections' poles sit up to ~1.7 m off
%   their parent axes, beyond that sphere.  The footprint-quadric body is the
%   lit surface itself.
%
%   Scored: every TELESCOPE leg (input -> M1 -> M2 -> M3 -> Slit) against
%   every body it does not start or end on, and every SPECTROMETER leg against
%   the telescope bodies.  The spectrometer against itself is the record's
%   (that geometry is the Dyson of record, unchanged).
%   C.table rows {leg, body, clear_mm}, worst first; C.min_mm; C.pass.
%   Options: 'fields_deg' (default [-1 0 1]*P.strip_half_deg), 'lambda_m' (the Dyson's centre).
arguments
    P struct
    efile (1,:) char
    opts.fields_deg (1,:) double = [-1 0 1]*P.strip_half_deg
    opts.lambda_m (1,1) double = NaN
    opts.ntel (1,1) double = 3
    opts.sample_m (1,1) double = 4e-3
    opts.nrim (1,1) double = 120
end
mount = 5e-3;  if isfield(P, 'mount_m'), mount = P.mount_m; end
txt = fileread(efile);  hdr = txt(1:strfind(txt, 'nElt=') - 1);
v3 = @(t, k) vec_(t, k);
apst = v3(hdr, 'ApStop');  dc = v3(hdr, 'ChfRayDir');  dc = dc/norm(dc);  xg = v3(hdr, 'xGrid');  xg = xg - (xg'*dc)*dc;  xg = xg/norm(xg);
names = regexp(txt, '(?m)^\s*EltName=\s*(\S+)', 'tokens');  names = cellfun(@(c) c{1}, names, 'uni', 0);
macos.load_rx(efile);  nE = macos.num_elt();
iG = find(strcmp(names, 'Grating'), 1);  iS = find(strcmp(names, 'Slit'), 1);
lam = opts.lambda_m;  if isnan(lam), lam = macos.get_src_wvl(); end
% ---- the paths: every element's hits, per field (passing rays only, the chief kept)
nF = numel(opts.fields_deg);  hits = cell(nE, 1);  paths = cell(1, nF);
for q = 1:nF
    th = deg2rad(opts.fields_deg(q));  d = cos(th)*dc + sin(th)*xg;
    macos.set_src_wvl(lam);  macos.set_src_fov('src_pos', apst - d, 'src_dir', d, 'zSrc', 1e22);  macos.modify();
    macos.stop(iG);  macos.stop(iG);
    Pk = cell(1, nE);  okall = [];
    for k = 1:nE
        s = macos.trace(k);  ri = macos.get_ray_info(s.nRays);  Pk{k} = ri.pos;
        okk = ri.ok_trace & ri.ok_pass;  if isempty(okall), okall = okk; else, okall = okall & okk; end
    end
    sf = macos.get_src_fov();  p0 = Pk{1}(:, okall) - 0.5*sf.src_dir(:)/norm(sf.src_dir);
    pts = zeros(3, nnz(okall), nE + 1);  pts(:, :, 1) = p0;
    for k = 1:nE, pts(:, :, k + 1) = Pk{k}(:, okall);  hits{k} = [hits{k}, Pk{k}(:, okall)]; end
    paths{q} = pts;
end
% ---- bodies: physical parts (stems), footprint + mount on a quadric fitted to the part's hits
stem = regexprep(names, '(In|Out)$', '');
parts = unique(stem, 'stable');
B = struct('name', {}, 'elts', {}, 'pts', {}, 'c', {}, 'n', {}, 'ex', {}, 'ey', {}, 'a', {}, 'b', {});
for p = 1:numel(parts)
    el = find(strcmp(stem, parts{p}));
    if any(el == iS), continue, end                                   % the slit: the mask plate below
    H = [hits{el}];  if size(H, 2) < 10, continue, end
    c = mean(H, 2);  [U, ~, ~] = svd(H - c, 'econ');  ex = U(:, 1);  ey = U(:, 2);  nz = cross(ex, ey);
    u = ex'*(H - c);  v = ey'*(H - c);  w = nz'*(H - c);
    Q = [u(:).^2, v(:).^2, u(:).*v(:), u(:), v(:), ones(numel(u), 1)];  qc = Q\w(:);
    a = (max(u) - min(u))/2 + mount;  b = (max(v) - min(v))/2 + mount;  cu = (max(u) + min(u))/2;  cv = (max(v) + min(v))/2;
    [uu, vv] = meshgrid(-a:opts.sample_m:a, -b:opts.sample_m:b);  in = (uu/a).^2 + (vv/b).^2 <= 1;
    uu = uu(in) + cu;  vv = vv(in) + cv;
    ww = [uu.^2, vv.^2, uu.*vv, uu, vv, ones(numel(uu), 1)]*qc;
    pts = c + ex*uu' + ey*vv' + nz*ww';
    B(end+1) = struct('name', parts{p}, 'elts', el, 'pts', pts, 'c', c + ex*cu + ey*cv, 'n', nz, 'ex', ex, 'ey', ey, 'a', a, 'b', b);   %#ok<AGROW>
end
% the slit mask plate, in the plane of the slit hits
H = hits{iS};  c = mean(H, 2);  [U, ~, ~] = svd(H - c, 'econ');  ex = U(:, 1);  ey = cross(cross(ex, U(:, 2)), ex);  ey = ey/norm(ey);
a = P.slit_m/2 + 5e-3;  b = 2e-3;  [uu, vv] = meshgrid(-a:opts.sample_m/2:a, -b:opts.sample_m/4:b);
B(end+1) = struct('name', 'SlitMask', 'elts', iS, 'pts', c + ex*uu(:)' + ey*vv(:)', 'c', c, 'n', cross(ex, ey), 'ex', ex, 'ey', ey, 'a', a, 'b', b);
isTelBody = arrayfun(@(x) all(x.elts <= opts.ntel), B);
% ---- legs: station j -> j+1 (station 1 = the input plane, station k+1 = element k)
stn = [{'input'}, names];
rows = {};
for j = 1:nE
    telLeg = j <= opts.ntel + 1;                                      % input->M1 ... M3->Slit
    A = [];  Bq = [];
    for q = 1:nF
        pq = paths{q};  nr = size(pq, 2);  r = vecnorm(pq(:, :, 2) - mean(pq(:, :, 2), 2));
        [~, o] = sort(r, 'descend');  sel = unique([1, o(1:min(opts.nrim, nr))]);
        A = [A, pq(:, sel, j)];  Bq = [Bq, pq(:, sel, j + 1)];        %#ok<AGROW>
    end
    for k = 1:numel(B)
        if any(B(k).elts == j - 1) || any(B(k).elts == j), continue, end      % the leg's own end bodies (element j-1 -> j)
        if ~telLeg && ~isTelBody(k), continue, end                           % spectrometer vs itself: the record's
        cl = seg_body_(A, Bq, B(k));
        rows(end+1, :) = {sprintf('%s -> %s', stn{j}, stn{j+1}), B(k).name, cl*1e3};   %#ok<AGROW>
    end
end
[~, o] = sort(cell2mat(rows(:, 3)));  rows = rows(o, :);
C = struct('table', {rows}, 'min_mm', rows{1, 3}, 'pass', rows{1, 3} > 0, 'mount_m', mount, 'fields_deg', opts.fields_deg);
end

function cl = seg_body_(A, Bq, bd)
D = Bq - A;  L2 = sum(D.^2, 1);
den = bd.n.'*D;  t = (bd.n.'*(bd.c - A))./den;  hit = abs(den) > 1e-15 & t > 0 & t < 1;
if any(hit)
    Q = A(:, hit) + D(:, hit).*t(hit);
    e = sqrt(((bd.ex.'*(Q - bd.c))/bd.a).^2 + ((bd.ey.'*(Q - bd.c))/bd.b).^2);
    if any(e < 1), cl = -(1 - min(e))*min(bd.a, bd.b); return, end
end
cl = inf;  S = bd.pts;  ns = size(S, 2);  chunk = max(1, floor(4e6/ns));
for i0 = 1:chunk:size(A, 2)
    ii = i0:min(size(A, 2), i0 + chunk - 1);  a = A(:, ii);  d = D(:, ii);  l2 = L2(ii);
    tt = ((S.'*d) - sum(a.*d, 1))./max(l2, 1e-30);  tt = min(max(tt, 0), 1);
    px = a(1, :) + d(1, :).*tt;  py = a(2, :) + d(2, :).*tt;  pz = a(3, :) + d(3, :).*tt;
    dd = (px - S(1, :).').^2 + (py - S(2, :).').^2 + (pz - S(3, :).').^2;
    cl = min(cl, sqrt(min(dd(:))));
end
end

function v = vec_(t, k)
m = regexp(t, ['(?m)^\s*' k '=\s*([^\n]*)'], 'tokens', 'once');
assert(~isempty(m), 'tls_clearance_joined: no %s= in the header', k);
v = sscanf(strrep(m{1}, 'D', 'E'), '%f');
end
