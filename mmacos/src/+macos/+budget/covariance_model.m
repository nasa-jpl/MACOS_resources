function cm = covariance_model(g, M, cols)
%COVARIANCE_MODEL  Per-group covariance descriptor in the Jacobian's units.
%   CM = macos.budget.covariance_model(G, M, COLS) builds the statistics a
%   budget group contributes, converted into the SAME native units as the
%   assembled Jacobian columns M.J (so the kernel macos.budget.cov_budget /
%   mc_budget is pure linear algebra).  G is one group record (fields
%   .sigma .units .dist .range .cov .cov_data), M is the assembled model
%   (macos.budget.assemble), COLS the group's logical column mask.
%
%   CM fields:
%     .type         'iid' | 'full'
%     .var          iid per-DOF variance in native-unit^2 (the covariance-
%                   analysis second moment of the error distribution)
%     .sigma_native iid representative 1-sigma, native units
%     .Sigma        full covariance (nc x nc), native units (type 'full')
%     .sampler      @() -> nc x 1 random draw, native units, honouring .dist
%     .class        native-unit class of the columns ('angle'|'length_si'|
%                   'length_base'|'none')
%
%   Distribution second moments (so covariance == Monte-Carlo by
%   construction): normal sigma s -> var s^2; uniform half-width h ->
%   var h^2/3; worstcase bound h -> var h^2 (the error sits at +/-h).
%
%   See also macos.budget.cov_budget, macos.budget.mc_budget,
%   macos.budget.assemble.

nc = nnz(cols);
cls = native_class_(M, cols);
cm.class = cls;

dist = lower(strtrim(char(g.dist)));
if isempty(dist), dist = 'normal'; end

% ---- full / correlated covariance supplied explicitly -------------------
if isfield(g, 'cov') && any(strcmpi(g.cov, {'full', 'psd'})) ...
        && isfield(g, 'cov_data') && ~isempty(g.cov_data)
    Sig = g.cov_data;
    if isscalar(Sig), Sig = Sig * eye(nc); end
    assert(isequal(size(Sig), [nc nc]), ...
        'budget:cov', 'group "%s": cov_data must be %dx%d', g.name, nc, nc);
    % cov_data is taken already in native units (document in read_spec)
    cm.type = 'full';
    cm.Sigma = Sig;
    cm.var = NaN; cm.sigma_native = sqrt(mean(diag(Sig)));
    L = chol(Sig + 0*eye(nc), 'lower');
    cm.sampler = @() L * randn(nc, 1);
    return;
end

% ---- iid group: a single engineering sigma / range in CM.class units ----
switch dist
    case 'normal'
        s = to_native_(g.sigma, g.units, cls, M.cbm);
        cm.var = s^2;  cm.sigma_native = s;
        cm.sampler = @() s .* randn(nc, 1);
    case 'uniform'
        h = half_range_(g, cls, M.cbm);
        cm.var = h^2 / 3;  cm.sigma_native = sqrt(cm.var);
        cm.sampler = @() (2*rand(nc, 1) - 1) .* h;
    case 'worstcase'
        h = half_range_(g, cls, M.cbm);
        cm.var = h^2;  cm.sigma_native = h;
        cm.sampler = @() sign(randn(nc, 1)) .* h;
    otherwise
        error('budget:dist', 'group "%s": unknown Dist "%s"', g.name, dist);
end
cm.type = 'iid';
cm.Sigma = [];
end

% -------------------------------------------------------------------------
function h = half_range_(g, cls, cbm)
% Half-width of a uniform/worstcase range [Min,Max], native units.
assert(isfield(g, 'range') && numel(g.range) == 2 && all(isfinite(g.range)), ...
    'budget:range', 'group "%s": uniform/worstcase needs Min and Max', g.name);
h = 0.5 * (g.range(2) - g.range(1));
h = to_native_(h, g.units, cls, cbm);
end

% -------------------------------------------------------------------------
function cls = native_class_(M, cols)
% Native-unit class shared by a group's columns (require homogeneity).
ch  = M.meta.channel(cols);
dof = M.meta.dof(cols);
par = M.meta.param(cols);
c = cell(nnz(cols), 1);
for j = 1:numel(c)
    switch ch{j}
        case 'dwdx'
            if dof(j) <= 2, c{j} = 'angle'; else, c{j} = 'length_si'; end
        case {'dwdz', 'dwdgrid'}
            c{j} = 'length_base';
        case 'dwdsurf'
            if strcmpi(par{j}, 'Kc'), c{j} = 'none'; else, c{j} = 'length_base'; end
        otherwise
            c{j} = 'length_base';
    end
end
u = unique(c);
assert(isscalar(u), 'budget:units', ...
    'a group mixes unit classes (%s) -- split it', strjoin(u, ', '));
cls = u{1};
end

% -------------------------------------------------------------------------
function x = to_native_(x, units, cls, cbm)
% Scale an engineering value to the column's native unit.
u = lower(strtrim(char(units)));
u = regexprep(u, '\s*/\s*zi$', '');     % strip "/Zi" per-mode annotation
u = regexprep(u, '\s*/\s*mode$', '');
switch cls
    case 'angle'
        x = x * angle_to_rad_(u);
    case 'length_si'
        x = x * length_to_m_(u);
    case 'length_base'
        x = x * length_to_m_(u) / cbm;   % metre -> BaseUnit
    case 'none'
        % dimensionless (Kc) or unmodelled unit -- pass through
end
end

function f = angle_to_rad_(u)
switch u
    case {'rad', 'radian', 'radians'}, f = 1;
    case {'mrad'},                     f = 1e-3;
    case {'urad', 'microrad'},         f = 1e-6;
    case {'nrad'},                     f = 1e-9;
    case {'arcsec'},                   f = pi/180/3600;
    otherwise, error('budget:unit', 'unknown angle unit "%s"', u);
end
end

function f = length_to_m_(u)
switch u
    case {'m', 'metre', 'meter'}, f = 1;
    case {'cm'},                  f = 1e-2;
    case {'mm'},                  f = 1e-3;
    case {'um', 'micron'},        f = 1e-6;
    case {'nm'},                  f = 1e-9;
    case {'pm'},                  f = 1e-12;
    otherwise, error('budget:unit', 'unknown length unit "%s"', u);
end
end
