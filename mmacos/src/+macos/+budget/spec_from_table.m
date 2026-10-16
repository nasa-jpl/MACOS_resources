function spec = spec_from_table(T, M, varargin)
%SPEC_FROM_TABLE  Resolve a user group table into a validated budget spec.
%   SPEC = macos.budget.spec_from_table(T, M) turns the canonical group
%   table T (one row per budget line; see macos.budget.read_spec /
%   suggest_groups) into a spec whose every group carries a resolved
%   logical column mask (.cols) over the assembled model M and a
%   covariance descriptor (.cm, from macos.budget.covariance_model).  It
%   warns on columns claimed by two groups and reports columns no group
%   claims.
%
%   Table columns (case-insensitive; missing optional ones default):
%     Name              required
%     Elements          role name (via opts.roles) | 'all' | numeric list
%     DOFs              tilt|clock|decenter|despace|Rx..Tz | list
%     Modes             Noll modes '4-10' | '4,5,6' | numeric
%     Param             Kr | Kc
%     Channel           dwdx|dwdz|dwdsurf|dwdgrid (else inferred)
%     Dist              normal | uniform | worstcase   (default normal)
%     Sigma             1-sigma (normal), engineering units
%     Units             nrad|urad|nm|um|mm|m|...        (default by class)
%     Min, Max          range for uniform/worstcase
%     Cov               iid | full | psd               (default iid)
%     Control           ''|WFC|MET  (tags the actuator DOFs: any non-empty
%                       value makes the group a control group)
%     Section           optional: the budget section the row sits in (the
%                       workbook's column A, e.g. 'Primary Mirror'); default
%                       'Optics'
%
%   Options:
%     'roles'   containers.Map name -> iElt vector (e.g. 'pm_segs' -> [3 4..])
%     'wfs_sigma'  {sigma, units} (or a scalar in nm): the WFS (sensing)
%               noise, 1-sigma PER WAVEFRONT SAMPLE (iid over the rays the
%               Jacobian rows sample) -- Redding 2014 Eq. 32's West = s^2 I.
%               Feeds the post-WFC sensing-error row.  Default: none.
%     'act_sigma'  actuator noise per CONTROL group (Eq. 31): a table with
%               Name / Sigma / Units, or a cell {name, sigma, units; ...},
%               NAME = a group whose Control column is set.  Its sigma is
%               per actuated DOF, in that group's own unit class (a
%               rotation group takes an angle).  Default: none.
%
%   SPEC fields: .groups (struct array: name selector cols cm is_control
%   stage), .assigned (logical Nz), .orphans (idx of unclaimed columns),
%   .noverlap (count of doubly-claimed columns), .wfs (struct: sigma
%   [native BaseUnit], sigma_eng, units; [] when not given), .act (struct
%   array: name group cols cm, one per control group given an act_sigma).
%
%   See also macos.budget.select_columns, macos.budget.covariance_model.

p = inputParser;
p.addParameter('roles', containers.Map('KeyType', 'char', 'ValueType', 'any'));
p.addParameter('wfs_sigma', [], @(x) isempty(x) || isnumeric(x) || iscell(x));
p.addParameter('act_sigma', [], @(x) isempty(x) || istable(x) || iscell(x));
p.parse(varargin{:});
roles = p.Results.roles;

T = canon_table_(T);
nrow = height(T);
nz = numel(M.meta.channel);
assigned = false(nz, 1);
noverlap = 0;

groups = struct('name', {}, 'selector', {}, 'cols', {}, 'cm', {}, ...
    'is_control', {}, 'stage', {}, 'section', {});
hasSection = ismember('Section', T.Properties.VariableNames);
names = {};
for r = 1:nrow
    g = struct();
    g.name = char(T.Name(r));
    assert(~isempty(g.name), 'budget:spec', 'row %d has empty Name', r);
    assert(~ismember(g.name, names), 'budget:spec', ...
        'duplicate group name "%s"', g.name);
    names{end+1} = g.name; %#ok<AGROW>

    sel = selector_(T(r, :), roles);
    cols = macos.budget.select_columns(M.meta, sel);
    if ~any(cols)
        warning('budget:spec', 'group "%s" selects no columns', g.name);
    end
    noverlap = noverlap + nnz(cols & assigned);
    assigned = assigned | cols;

    g.selector = sel;
    g.cols = cols;
    g.is_control = ~isempty(char(T.Control(r)));
    g.stage = lower(char(T.Control(r)));
    g.section = 'Optics';
    if hasSection
        sv = char_(T.Section(r));
        if ~isempty(sv), g.section = sv; end
    end
    % statistics -> covariance descriptor in native units
    gg = stat_(T(r, :));
    gg.name = g.name;
    g.cm = macos.budget.covariance_model(gg, M, cols);
    groups(end+1) = g; %#ok<AGROW>
end

if noverlap > 0
    warning('budget:spec', ...
        '%d column(s) claimed by more than one group (double-counted)', noverlap);
end
orphans = find(~assigned);
if ~isempty(orphans)
    warning('budget:spec', '%d Jacobian column(s) claimed by no group', ...
        numel(orphans));
end

spec = struct('groups', groups, 'assigned', assigned, ...
    'orphans', orphans, 'noverlap', noverlap);
spec.wfs = wfs_(p.Results.wfs_sigma, M);
spec.act = act_(p.Results.act_sigma, groups, M);
end

% =========================================================================
function w = wfs_(x, M)
% WFS noise, per wavefront sample, -> native BaseUnit (the metric's unit).
w = [];
if isempty(x), return; end
if isnumeric(x), x = {x, 'nm'}; end
s = double(x{1});  u = char(x{2});
g = struct('name', 'WFS', 'sigma', s, 'units', u, 'dist', 'normal', 'cov', 'iid');
cm = macos.budget.covariance_model(g, struct('cbm', M.cbm, 'meta', ...
    struct('channel', {{'dwdz'}}, 'dof', NaN, 'param', {{''}})), true);
w = struct('sigma', cm.sigma_native, 'sigma_eng', s, 'units', u, ...
    'per_eng', cm.per_eng);
end

% -------------------------------------------------------------------------
function a = act_(x, groups, M)
% One actuator-noise record per control group named in X.
a = struct('name', {}, 'group', {}, 'cols', {}, 'cm', {});
if isempty(x), return; end
if istable(x), x = [cellstr(string(x.Name)), num2cell(double(x.Sigma)), ...
        cellstr(string(x.Units))]; end
for r = 1:size(x, 1)
    gname = char(x{r, 1});
    k = find(strcmp({groups.name}, gname), 1);
    assert(~isempty(k), 'budget:act', 'act_sigma names no group "%s"', gname);
    assert(groups(k).is_control, 'budget:act', ...
        'act_sigma group "%s" is not a control group (Control column empty)', gname);
    g = struct('name', [gname ' Actuator'], 'sigma', double(x{r, 2}), ...
        'units', char(x{r, 3}), 'dist', 'normal', 'cov', 'iid');
    cm = macos.budget.covariance_model(g, M, groups(k).cols);
    a(end+1) = struct('name', g.name, 'group', gname, ...
        'cols', groups(k).cols, 'cm', cm); %#ok<AGROW>
end
end

% =========================================================================
function sel = selector_(row, roles)
sel = struct();
% elements
el = row.Elements;
elv = resolve_elements_(el, roles);
if ~isempty(elv), sel.iElt = elv; end
% dofs
dv = resolve_dofs_(row.DOFs);
if ~isempty(dv), sel.dof = dv; end
% modes
mv = resolve_list_(row.Modes);
if ~isempty(mv), sel.mode = mv; end
% param
pv = char_(row.Param);
if ~isempty(pv), sel.param = {pv}; end
% channel: explicit, else inferred from which fields are set
 chv = char_(row.Channel);
if ~isempty(chv)
    sel.channel = {chv};
else
    if ~isempty(mv),        sel.channel = {'dwdz'};
    elseif ~isempty(pv),    sel.channel = {'dwdsurf'};
    elseif ~isempty(dv),    sel.channel = {'dwdx'};
    end
end
end

% -------------------------------------------------------------------------
function v = resolve_elements_(el, roles)
if isnumeric(el), v = el(:).'; v = v(~isnan(v)); return; end
s = strtrim(char(string(el)));
if isempty(s) || strcmpi(s, 'all'), v = []; return; end
if isKey(roles, s), v = roles(s); v = v(:).'; return; end
% numeric list "3 4 5" or "3,4,5"
num = str2double(regexp(s, '[-\d.]+', 'match'));
if all(~isnan(num)) && ~isempty(num), v = num; return; end
error('budget:elements', 'cannot resolve Elements "%s" (no such role)', s);
end

% -------------------------------------------------------------------------
function v = resolve_dofs_(d)
if isnumeric(d), v = d(:).'; v = v(~isnan(v)); return; end
s = lower(strtrim(char(string(d))));
if isempty(s), v = []; return; end
map = struct('tilt', [0 1], 'clock', 2, 'clocking', 2, ...
    'decenter', [3 4], 'despace', 5, 'piston', 5, ...
    'rx', 0, 'ry', 1, 'rz', 2, 'tx', 3, 'ty', 4, 'tz', 5);
v = [];
for tok = regexp(s, '[a-z]+|\d+', 'match')
    t = tok{1};
    if isfield(map, t), v = [v, map.(t)]; %#ok<AGROW>
    elseif ~isnan(str2double(t)), v = [v, str2double(t)]; %#ok<AGROW>
    end
end
v = unique(v);
end

% -------------------------------------------------------------------------
function v = resolve_list_(x)
% "4-10", "11,13,15", "4:2:10", or numeric -> vector
if isnumeric(x), v = x(:).'; v = v(~isnan(v)); return; end
s = strtrim(char(string(x)));
if isempty(s), v = []; return; end
rng = regexp(s, '^(\d+)\s*-\s*(\d+)$', 'tokens', 'once');
if ~isempty(rng), v = str2double(rng{1}):str2double(rng{2}); return; end
num = str2double(regexp(s, '\d+', 'match'));
v = num(~isnan(num));
end

% -------------------------------------------------------------------------
function g = stat_(row)
g.sigma = tonum_(row.Sigma);
g.units = char_(row.Units);
g.dist  = char_(row.Dist); if isempty(g.dist), g.dist = 'normal'; end
g.cov   = char_(row.Cov);  if isempty(g.cov),  g.cov  = 'iid'; end
g.cov_data = [];
mn = tonum_(row.Min); mx = tonum_(row.Max);
g.range = [mn, mx];
end

% -------------------------------------------------------------------------
function T = canon_table_(T)
if isstruct(T), T = struct2table(T, 'AsArray', true); end
need = {'Name','Elements','DOFs','Modes','Param','Channel', ...
    'Dist','Sigma','Units','Min','Max','Cov','Control'};
vn = T.Properties.VariableNames;
lut = containers.Map(lower(vn), vn);
for k = 1:numel(need)
    key = lower(need{k});
    if isKey(lut, key)
        if ~strcmp(lut(key), need{k})
            T.Properties.VariableNames{strcmp(vn, lut(key))} = need{k};
        end
    else
        % add a missing optional column with empty defaults
        if any(strcmp(need{k}, {'Sigma','Min','Max'}))
            T.(need{k}) = nan(height(T), 1);
        else
            T.(need{k}) = repmat({''}, height(T), 1);
        end
    end
    vn = T.Properties.VariableNames;
end
end

function s = char_(x)
if iscell(x), x = x{1}; end
if ismissing(x), s = ''; return; end
s = strtrim(char(string(x)));
if strcmpi(s, 'nan'), s = ''; end
end

function v = tonum_(x)
if iscell(x), x = x{1}; end
if isnumeric(x), v = double(x); return; end
v = str2double(string(x));
end
