function S = spectrometer_sens(G, M, P, opts)
%SPECTROMETER_SENS  The tolerance ladder of a spectrometer deck: one perturbation at a time, scored by the engine.
%
%   S = spectrometer_sens(G, M, P) perturbs the emitted deck M.file of the
%   spectrometer_geom chain G ONE PERTURBATION AT A TIME, on the deck TEXT
%   (rigid moves of element groups about a pivot, radius, conic, period, index,
%   the slit launch, the detector), re-loads, and scores with spectrometer_score
%   -- the record's own scorer, so every number is the engine's.  Per row:
%     d.smile d.keystone d.CRF d.SRF d.EE   (px; and per unit perturbation)
%   the 2x linearity ratio on the rows named in opts.lin_rows, and the
%   COMPENSATOR residuals:
%     'focus'     the detector moved along its normal (one-axis stage)
%     'focus+xy'  + the detector translated in its plane
%     'clock'     the detector rotated about its normal (smile / keystone tilt)
%   each the metric set at the compensator's optimum (focus: the mean square
%   ray width; clock: smile + keystone).  NOTE the scorer's smile / keystone /
%   FWHM are RELATIVE (max - min over the slit or the band; widths about the
%   centroid), so a pure detector or slit TRANSLATION leaves them unchanged:
%   'focus+xy' == 'focus' by construction -- measured, not assumed (S.xy_check).
%
%   The deck geometry convention is spectrometer_rx's (the slit along x, the
%   dispersion along y, the optical axis z): decenter x = along the slit, y =
%   across it (the dispersion direction); tilt x/y = about those axes.
%
%   opts:
%     'perts'     struct array of perturbations (default: SPECTROMETER_SENS_DEFAULTS
%                 for the Dyson form -- the addendum 49 list); fields name, kind,
%                 elts (EltNames), amount, unit, pivot (EltName whose vertex is the
%                 pivot; '' = the group's first element)
%     'lin_rows'  names of the rows checked at 2x (default {'lens_tilt_x', 'grating_decenter_y'})
%     'comp'      compensators to run (default {'focus', 'clock'}; 'focus+xy' once,
%                 on the first row, as S.xy_check)
%     'e2e'       [] or a struct {template, tel_deck, dyson5 (a dyson5_params
%                 override struct)}: each row ALSO scored end to end through
%                 dyson5_t5f with the template's Dyson blocks perturbed the same
%                 way (centroid launch, roll 0 unless e2e.dyson5 says otherwise)
%     'nx','nlam' the scorer's grid (default 7 x 7)
%     'quiet'
%   S.base (the nominal score), S.rows (one per perturbation), S.units (a line).
%
%   See also SPECTROMETER_SCORE, SPECTROMETER_RX, DYSON5_T5F.
arguments
    G struct
    M struct
    P struct
    opts.perts = []
    opts.lin_rows (1,:) cell = {'lens_tilt_x', 'grating_decenter_y'}
    opts.comp (1,:) cell = {'focus', 'clock'}
    opts.e2e = []
    opts.nx (1,1) double = 7
    opts.nlam (1,1) double = 7
    opts.quiet (1,1) logical = false
    opts.xy_check (1,1) logical = true
end
perts = opts.perts;  if isempty(perts), perts = spectrometer_sens_defaults(G); end
txt0 = fileread(M.file);
tmp = [tempname '_sens.in'];  cln = onCleanup(@() delete_if_(tmp));
px = P.pixel_m;
score = @(txt, Gx) score_(txt, Gx, M, P, tmp, opts.nx, opts.nlam);
base = score(txt0, G);
S = struct('base', base, 'rows', [], 'units', sprintf(['metrics in px of %.0f um (and um); per unit = per the row''s unit; ' ...
           'linearity = d(2x)/(2 d(1x)) (1 = linear)'], px*1e6));
pr = @(varargin) [];  if ~opts.quiet, pr = @(varargin) fprintf(varargin{:}); end
pr('spectrometer_sens: nominal smile %.4f keystone %.4f CRF %.3f SRF %.4f EE %.3f px\n', base.smile, base.keystone, base.CRF, base.SRF, base.EE);
% the COMPENSATED nominal: a compensator's residual is judged against the nominal at ITS compensator optimum (the record's
% deck is not at the focus that minimises the mean square width -- refocusing it alone moves CRF -- so a residual against
% the raw nominal would credit the compensator with the nominal's own defocus)
S.base_comp = struct();
for c = opts.comp, S.base_comp.(strrep(c{1}, '+', '_')) = compensate_(txt0, G, c{1}, score); end
if ~isempty(opts.e2e)
    q0 = struct('name', 'nominal', 'kind', 'decenter_x', 'elts', {{'Grating'}}, 'amount', 0, 'unit', 'm', 'pivot', 'Grating');
    S.base_e2e = e2e_(opts.e2e, q0, G);
    pr('  e2e nominal: smile %.4f keystone %.4f CRF %.3f SRF %.4f EE %.3f px\n', S.base_e2e.smile, S.base_e2e.keystone, S.base_e2e.CRF, S.base_e2e.SRF, S.base_e2e.EE);
end
for k = 1:numel(perts)
    q = perts(k);
    [t1, G1] = apply_(txt0, G, q, 1);  r1 = score(t1, G1);
    row = struct('name', q.name, 'kind', q.kind, 'amount', q.amount, 'unit', q.unit, ...
                 'd', dif_(r1, base), 'abs', r1, 'lin', NaN, 'comp', struct(), 'e2e', [], 'd_e2e', []);
    if any(strcmp(q.name, opts.lin_rows))
        [t2, G2] = apply_(txt0, G, q, 2);  r2 = score(t2, G2);  d2 = dif_(r2, base);  d1 = row.d;
        f = fieldnames(d1);  lr = struct();
        for i = 1:numel(f), if abs(d1.(f{i})) > 1e-9, lr.(f{i}) = d2.(f{i})/(2*d1.(f{i})); else, lr.(f{i}) = NaN; end, end
        row.lin = lr;
    end
    for c = opts.comp
        row.comp.(strrep(c{1}, '+', '_')) = compensate_(t1, G1, c{1}, score);
    end
    if k == 1 && opts.xy_check
        S.xy_check = struct('focus', compensate_(t1, G1, 'focus', score), 'focus_xy', compensate_(t1, G1, 'focus+xy', score));
    end
    if ~isempty(opts.e2e)
        row.e2e = e2e_(opts.e2e, q, G);
        row.d_e2e = dif_(row.e2e, S.base_e2e);
        pr('  %-24s e2e: dsmile %+8.4f dkey %+8.4f dCRF %+7.4f dSRF %+7.4f dEE %+7.4f px\n', q.name, row.d_e2e.smile, row.d_e2e.keystone, ...
           row.d_e2e.CRF, row.d_e2e.SRF, row.d_e2e.EE);
    end
    for c = fieldnames(row.comp)'
        row.comp.(c{1}).resid = dif_(row.comp.(c{1}).abs, S.base_comp.(c{1}).abs);   % vs the compensated nominal
    end
    S.rows = [S.rows, row];
    pr('  %-24s %8.3g %-6s | dsmile %+8.4f dkey %+8.4f dCRF %+7.4f dSRF %+7.4f dEE %+7.4f px\n', q.name, q.amount, q.unit, ...
       row.d.smile, row.d.keystone, row.d.CRF, row.d.SRF, row.d.EE);
end
end

% ===========================================================================
function r = score_(txt, Gx, M, P, tmp, nx, nlam)
fid = fopen(tmp, 'w');  fwrite(fid, txt);  fclose(fid);
macos.load_rx(tmp);
Mx = M;  Mx.file = tmp;
R = spectrometer_score(Gx, Mx, P, 'nx', nx, 'nlam', nlam, 'quiet', true);
r = struct('smile', R.smile_max, 'keystone', R.keystone_max, 'CRF', R.crf_max, 'SRF', R.srf_max, 'EE', R.ee_min, ...
           'w2', mean(R.SU(:).^2 + R.SV(:).^2, 'omitnan'));
end

function d = dif_(r, b)
d = struct('smile', r.smile - b.smile, 'keystone', r.keystone - b.keystone, 'CRF', r.CRF - b.CRF, 'SRF', r.SRF - b.SRF, 'EE', r.EE - b.EE);
end

function c = compensate_(txt, Gx, kind, score)
% the compensator's optimum and the metrics there
det = {'PreFPA', 'FPA'};
switch kind
    case 'focus'
        f = @(dz) score(move_(txt, det, eye(3), [0; 0; dz], det{2}), Gx).w2;
        dz = fminbnd(f, -5e-4, 5e-4, optimset('TolX', 1e-7));
        r = score(move_(txt, det, eye(3), [0; 0; dz], det{2}), Gx);  c = struct('dz_m', dz, 'abs', r);
    case 'focus+xy'
        f = @(v) score(move_(txt, det, eye(3), v(:), det{2}), Gx).w2;
        v = fminsearch(f, [0 0 0], optimset('TolX', 1e-8, 'MaxFunEvals', 60));
        r = score(move_(txt, det, eye(3), v(:), det{2}), Gx);  c = struct('dxyz_m', v, 'abs', r);
    case 'clock'
        f = @(a) metr_(score(move_(txt, det, rot_([0; 0; 1], a), [0; 0; 0], det{2}), Gx));
        a = fminbnd(f, -2e-3, 2e-3, optimset('TolX', 1e-8));
        r = score(move_(txt, det, rot_([0; 0; 1], a), [0; 0; 0], det{2}), Gx);  c = struct('clock_rad', a, 'abs', r);
    otherwise, error('spectrometer_sens: unknown compensator %s', kind);
end
end
function m = metr_(r), m = r.smile + r.keystone; end

% ===========================================================================
function [txt, Gx] = apply_(txt, G, q, s)
% apply perturbation q at s x its amount; returns the deck text and the (launch) chain
a = s*q.amount;  Gx = G;
switch q.kind
    case 'decenter_x', txt = move_(txt, q.elts, eye(3), [a; 0; 0], q.pivot);
    case 'decenter_y', txt = move_(txt, q.elts, eye(3), [0; a; 0], q.pivot);
    case 'despace',    txt = move_(txt, q.elts, eye(3), [0; 0; a], q.pivot);
    case 'tilt_x',     txt = move_(txt, q.elts, rot_([1; 0; 0], a), [0; 0; 0], q.pivot);
    case 'tilt_y',     txt = move_(txt, q.elts, rot_([0; 1; 0], a), [0; 0; 0], q.pivot);
    case 'clock',      txt = move_(txt, q.elts, rot_([0; 0; 1], a), [0; 0; 0], q.pivot);
    case 'radius',     txt = scale_key_(txt, q.elts, 'KrElt', 1 + a);
    case 'period'      % line density +a: the period d -> d/(1 + a)
        txt = scale_key_(txt, q.elts, 'RuleWidth', 1/(1 + a));
    case 'index',      txt = index_(txt, q.elts, a, q);
    case 'thickness'   % the sphere vertex moves away from the face by a (both passes)
        txt = move_(txt, q.elts, eye(3), [0; 0; a], q.pivot);
    case 'slit_x', Gx.slit = G.slit + [a; 0; 0];
    case 'slit_y', Gx.slit = G.slit + [0; a; 0];
    case 'slit_z', Gx.slit = G.slit + a*G.src.chief_dir(:)/norm(G.src.chief_dir);
    otherwise, error('spectrometer_sens: unknown perturbation kind %s', q.kind);
end
end

function txt = move_(txt, elts, R, t, pivot)
% rigid move of the named elements' blocks: points p -> c + R (p - c) + t, directions d -> R d
[pre, blk, nm] = split_(txt);
if isempty(pivot), pivot = elts{1}; end
c = vec_(blk{find(strcmp(nm, pivot), 1)}, 'VptElt');
pts = {'VptElt', 'RptElt'};  dirs = {'psiElt', 'xObs', 'h1HOE'};
for k = find(ismember(nm, elts))
    L = splitlines(string(blk{k}));
    for i = 1:numel(L)
        key = regexp(char(L(i)), '^\s*(\w+)=', 'tokens', 'once');  if isempty(key), continue, end
        key = key{1};
        if any(strcmp(key, pts)), v = vec_(char(L(i)), key);  L(i) = sprintf('%17s=  %.16E  %.16E  %.16E', key, c + R*(v - c) + t);
        elseif any(strcmp(key, dirs)), v = vec_(char(L(i)), key);  L(i) = sprintf('%17s=  %.16E  %.16E  %.16E', key, R*v);
        end
    end
    blk{k} = char(strjoin(L, newline));
end
txt = [pre, blk{:}];
end

function txt = scale_key_(txt, elts, key, f)
[pre, blk, nm] = split_(txt);
for k = find(ismember(nm, elts))
    v = num_(blk{k}, key);
    blk{k} = regexprep(blk{k}, ['(?m)^\s*' key '=.*$'], sprintf('%17s=  %.16E', key, v*f), 'once', 'dotexceptnewline');
end
txt = [pre, blk{:}];
end

function txt = index_(txt, elts, dn, q)
%   (the scaling is exact at q.lambda0_m; spectrometer_sens_dn(q) gives dn over the band -- the residual dispersion)
% the glass index raised by dn at q.lambda0_m: the Sellmeier B coefficients scaled by (1 + e), e = 2 n dn / (n^2 - 1),
% written as a per-element GlassCoef= after GlassElt= (the engine applies it at every wavelength)
B = q.sellmeier(1:3);  C = q.sellmeier(4:6);  L2 = (q.lambda0_m*1e6)^2;
n2 = 1 + sum(B*L2./(L2 - C));  e = 2*sqrt(n2)*dn/(n2 - 1);
coef = [B*(1 + e), C];
[pre, blk, nm] = split_(txt);
for k = find(ismember(nm, elts))
    if isempty(regexp(blk{k}, '(?m)^\s*GlassElt=', 'once')), continue, end
    blk{k} = regexprep(blk{k}, '(?m)^(\s*GlassElt=[^\n]*)$', sprintf('$1\n        GlassCoef=  %s', strtrim(sprintf('%.16E ', coef))), 'once', 'dotexceptnewline');
end
txt = [pre, blk{:}];
end

function [pre, blk, nm] = split_(txt)
st = regexp(txt, '(?m)^\s*iElt=', 'start');
pre = txt(1:st(1)-1);  st(end+1) = numel(txt) + 1;
blk = arrayfun(@(k) txt(st(k):st(k+1)-1), 1:numel(st)-1, 'uni', 0);
nm = cellfun(@(b) tok_(b, 'EltName'), blk, 'uni', 0);
end

function s = tok_(b, key)
m = regexp(b, ['(?m)^\s*' key '=\s*(\S+)'], 'tokens', 'once');  if isempty(m), s = ''; else, s = m{1}; end
end
function v = vec_(t, key)
m = regexp(t, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens', 'once');  v = sscanf(strrep(m{1}, 'D', 'E'), '%f');  v = v(1:3);
end
function x = num_(t, key)
m = regexp(t, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens', 'once');  x = sscanf(strrep(m{1}, 'D', 'E'), '%f', 1);
end
function R = rot_(ax, a)
ax = ax/norm(ax);  K = [0 -ax(3) ax(2); ax(3) 0 -ax(1); -ax(2) ax(1) 0];  R = eye(3) + sin(a)*K + (1 - cos(a))*K*K;
end

% ===========================================================================
function E = e2e_(e, q, G)
% the row end to end: the template's Dyson blocks perturbed the same way, joined by dyson5_t5f (centroid launch, quiet).
% A SLIT row is the Dyson moved by -v relative to the telescope (the join's Slit element is a pass-through, so moving it
% alone changes nothing physical).
tt = fileread(e.template);
st = regexp(tt, '(?m)^\s*iElt=', 'start');
nmT = cellfun(@(i) tok_(tt(i:min(i + 400, end)), 'EltName'), num2cell(st), 'uni', 0);
iS = find(strcmp(nmT, 'Slit'), 1);
head = tt(1:st(iS) - 1);  dys = tt(st(iS):end);
[~, ~, nmD] = split_(dys);
kinds = {'slit_x', 'slit_y', 'slit_z'};
if any(strcmp(q.kind, kinds))
    v = [0; 0; 0];  v(strcmp(q.kind, kinds)) = q.amount;
    if strcmp(q.kind, 'slit_z'), v = q.amount*G.src.chief_dir(:)/norm(G.src.chief_dir); end
    dys = move_(dys, nmD, eye(3), -v, 'Slit');
else
    dys = apply_(dys, G, q, 1);
end
ddir = fileparts(e.template);
pname = 'tls_sens_e2e_tmpl.in';  fid = fopen(fullfile(ddir, pname), 'w');  fwrite(fid, [head dys]);  fclose(fid);
cln = onCleanup(@() delete_if_(fullfile(ddir, pname)));
ov = struct('tel_dyson', 'size:F:240', 'tel5f_launch', 'centroid', 'tel5e_roll_deg', 0);
if isfield(e, 'dyson5'), f = fieldnames(e.dyson5); for i = 1:numel(f), ov.(f{i}) = e.dyson5.(f{i}); end, end
ov.tel5f_deck = e.tel_deck;  ov.tel5f_e2e_template = pname;  ov.tel5f_suffix = '_sens';
Pd = dyson5_params(ov);
old = cd(ddir);  back = onCleanup(@() cd(old));
S5 = dyson5_t5f(Pd, 'dyson5', struct('quiet', true));
R = S5.e2e;
E = struct('smile', R.smile_max, 'keystone', R.keystone_max, 'CRF', R.crf_max, 'SRF', R.srf_max, 'EE', R.ee_min);
end

function delete_if_(f), if exist(f, 'file'), delete(f); end, end
