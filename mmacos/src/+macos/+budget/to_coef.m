function coef = to_coef(rc, M, spec, varargin)
%TO_COEF  Assemble the coef.* coefficient struct for a budget spreadsheet.
%   COEF = macos.budget.to_coef(RC, M, SPEC) builds the coefficient struct
%   that a budget workbook's Coefficients sheet consumes, following the
%   HabEx/ATLAST layout: one entry per budget LINE with Pre/Post x iSeq
%   (1 = WFC, 2 = metrology) coefficients.  RC is macos.budget.cov_budget's
%   result (with its .met stage when one was run); M supplies cbm and the
%   metric name (may be [] with 'cbm' given); SPEC is accepted for the v1
%   signature and not needed.
%
%   COEF.<key> per line (key = sanitized line name), v1 fields first:
%       .name .class .units      line name, native-unit class, sigma units
%       .alpha                   pre-WFC coefficient (native, sigma-free)
%       .alpha_post              post-WFC coefficient (Eq. 30; = .alpha
%                                when nothing is controlled, Pu = I)
%       .alpha_met, .alpha_met_post   iSeq = 2 pre/post-metrology (native).
%                                With no metrology stage the drift is not
%                                corrected: alpha_met = alpha and
%                                alpha_met_post = alpha_met.
%       .sigma                   1-sigma, native units
%       .wfe                     post-WFC RMS WFE (nm) -- v1's .wfe
%   and the workbook view (2026-10-15):
%       .kind     'error' | 'wfs' | 'act' | 'met' (gauge noise) | 'drift'
%                 (a drift group with no initial-error line of its name)
%       .section  the budget section (group table 'Section'; noise rows
%                 'WFC', gauge / drift-only rows 'Metrology')
%       .sigma_eng, .coef_pre, .coef_post     the INITIAL-alignment input
%                 (engineering units) and nm per engineering unit, so the
%                 sheet's E = C*D and F = C*E reproduce .wfe_pre / .wfe
%       .sigma_drift, .units_drift, .coef_met, .coef_met_post   the same
%                 for the DRIFT input (0 when the line has no drift)
%       .wfe_pre, .wfe_met, .wfe_met_post     nm
%       .total    sqrt(wfe^2 + wfe_met_post^2) (the HabEx 'Total WFE')
%   A 'full'-covariance group has its covariance folded in: sigma_eng = 1,
%   units '(cov)', coef = the WFE itself.
%
%   COEF.total            sqrt(sum of every line's total^2) = the Total WFE
%   COEF.total_pre_wfc / .total_post_wfc / .total_pre_met / .total_post_met
%   COEF.meta             cbm, tonm, order (line keys), controlled, has_met,
%                         metric
%
%   See also macos.budget.rollup, macos.budget.write_xlsx,
%   macos.budget.write_workbook.

p = inputParser;
dcbm = 1;  metric = '';
if ~isempty(M)
    dcbm = M.cbm;
    if isfield(M, 'units') && isfield(M.units, 'metric'), metric = M.units.metric; end
end
p.addParameter('cbm', dcbm, @isscalar);
p.parse(varargin{:});
tonm = p.Results.cbm * 1e9;

ctl = isfield(rc, 'controlled') && rc.controlled;
met = [];
if isfield(rc, 'met') && ~isempty(rc.met), met = rc.met; end

L = {};                                    % line structs, in order
ng = numel(rc.names);
for k = 1:ng
    e = blank_();
    e.name = rc.names{k};  e.kind = 'error';
    e.class = rc.class{k};
    e.section = getc_(rc, 'section', k, 'Optics');
    e.units = getc_(rc, 'units', k, '');
    e.sigma = rc.sigma(k);
    a_pre = getv_(rc, 'alpha_pre', k, rc.alpha(k));
    e.alpha = a_pre;
    e.alpha_post = a_pre;
    if ctl, e.alpha_post = rc.alpha(k); end
    full = isfield(rc, 'type') && strcmp(rc.type{k}, 'full');
    pe = getv_(rc, 'per_eng', k, 1);
    if full
        e.sigma_eng = 1;  e.units = '(cov)';
        e.coef_pre = a_pre * tonm;  e.coef_post = e.alpha_post * tonm;
    else
        e.sigma_eng = getv_(rc, 'sigma_eng', k, rc.sigma(k));
        e.coef_pre = a_pre * tonm * pe;  e.coef_post = e.alpha_post * tonm * pe;
    end
    e.wfe_pre = getv_(rc, 'wfe_pre', k, rc.wfe(k)) * tonm;
    e.wfe = rc.wfe(k) * tonm;
    % drift (iSeq = 2) on the same line name
    e.alpha_met = e.alpha;  e.alpha_met_post = e.alpha;
    e.coef_met = e.coef_pre;  e.coef_met_post = e.coef_pre;
    if ~isempty(met)
        j = find(strcmp(met.names, e.name), 1);
        if ~isempty(j), e = drift_(e, met, j, tonm); end
    end
    L{end+1} = e; %#ok<AGROW>
end

% WFC noise rows (iSeq = 1) -- and their iSeq = 2 twins by name
ex = [];  if isfield(rc, 'extra'), ex = rc.extra; end
for q = 1:numel(ex)
    e = noise_(ex(q), tonm, 1);
    if ~isempty(met)
        j = find(strcmp({met.extra.name}, e.name), 1);
        if ~isempty(j), e = noise2_(e, met.extra(j), tonm); end
    end
    L{end+1} = e; %#ok<AGROW>
end

% metrology-stage lines with no iSeq = 1 line of their name
if ~isempty(met)
    have = cellfun(@(z) z.name, L, 'UniformOutput', false);
    for j = 1:numel(met.names)
        if ismember(met.names{j}, have), continue; end
        e = blank_();
        e.name = met.names{j};  e.kind = 'drift';  e.section = 'Metrology';
        if isfield(met, 'section'), e.section = met.section{j}; end
        e = drift_(e, met, j, tonm);
        L{end+1} = e; %#ok<AGROW>
    end
    for j = 1:numel(met.extra)
        if ismember(met.extra(j).name, have), continue; end
        e = noise2_(noise_(met.extra(j), tonm, 2), met.extra(j), tonm);
        L{end+1} = e; %#ok<AGROW>
    end
end

coef = struct();
keys = cell(numel(L), 1);
tot2 = 0;
for i = 1:numel(L)
    e = L{i};
    e.total = sqrt(e.wfe^2 + e.wfe_met_post^2);
    tot2 = tot2 + e.total^2;
    key = sanitize_(e.name);
    while isfield(coef, key) || any(strcmp(keys(1:i-1), key)), key = [key '_']; end %#ok<AGROW>
    keys{i} = key;
    coef.(key) = e;
end
coef.total = sqrt(tot2);
coef.total_pre_wfc = getv_(rc, 'total_pre', 1, rc.total) * tonm;
coef.total_post_wfc = rc.total * tonm;
coef.total_pre_met = 0;  coef.total_post_met = 0;
if ~isempty(met)
    coef.total_pre_met = met.total_pre * tonm;
    coef.total_post_met = met.total_post * tonm;
end
coef.meta = struct('cbm', p.Results.cbm, 'tonm', tonm, ...
    'order', {keys}, 'controlled', ctl, 'has_met', ~isempty(met), ...
    'metric', metric);
end

% =========================================================================
function e = blank_()
e = struct('name', '', 'class', '', 'units', '', 'alpha', 0, ...
    'alpha_post', 0, 'alpha_met', 0, 'alpha_met_post', 0, 'sigma', 0, ...
    'wfe', 0, 'kind', '', 'section', '', 'sigma_eng', 0, 'coef_pre', 0, ...
    'coef_post', 0, 'sigma_drift', 0, 'units_drift', '', 'coef_met', 0, ...
    'coef_met_post', 0, 'wfe_pre', 0, 'wfe_met', 0, 'wfe_met_post', 0, ...
    'total', 0);
end

function e = drift_(e, met, j, tonm)
% fill a line's iSeq = 2 slots from drift group J of the metrology stage
pe = met.per_eng(j);
e.alpha_met = met.alpha_met(j);
e.alpha_met_post = met.alpha_met_post(j);
e.sigma_drift = met.sigma_eng(j);
e.units_drift = met.units{j};
e.coef_met = e.alpha_met * tonm * pe;
e.coef_met_post = e.alpha_met_post * tonm * pe;
e.wfe_met = met.wfe_pre(j) * tonm;
e.wfe_met_post = met.wfe_post(j) * tonm;
end

function e = noise_(x, tonm, stage)
% a noise row (wfs / act / met); its coefficient is post-control only
e = blank_();
e.name = x.name;  e.kind = x.kind;  e.class = x.kind;
e.section = 'WFC';  if strcmp(x.kind, 'met'), e.section = 'Metrology'; end
e.units = x.units;  e.sigma = x.sigma;
if stage == 1
    e.sigma_eng = x.sigma_eng;
    e.alpha_post = x.alpha;
    e.coef_post = x.alpha * tonm * x.per_eng;
    e.wfe = x.wfe * tonm;
end
end

function e = noise2_(e, x, tonm)
% the iSeq = 2 (drift-stage) slots of a noise row
e.sigma_drift = x.sigma_eng;  e.units_drift = x.units;
e.alpha_met_post = x.alpha;
e.coef_met_post = x.alpha * tonm * x.per_eng;
e.wfe_met_post = x.wfe * tonm;
end

function v = getv_(s, f, k, dflt)
if isfield(s, f) && numel(s.(f)) >= k, v = s.(f)(k); else, v = dflt; end
end

function c = getc_(s, f, k, dflt)
if isfield(s, f) && numel(s.(f)) >= k && ~isempty(s.(f){k}), c = s.(f){k}; else, c = dflt; end
end

function s = sanitize_(nm)
s = regexprep(char(nm), '[^A-Za-z0-9]', '_');
if isempty(s) || ~isletter(s(1)), s = ['g_' s]; end
end
