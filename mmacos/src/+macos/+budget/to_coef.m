function coef = to_coef(rc, M, spec, varargin)
%TO_COEF  Assemble the coef.* coefficient struct for a budget spreadsheet.
%   COEF = macos.budget.to_coef(RC, M, SPEC) builds the coefficient struct
%   that a budget workbook's Coefficients sheet consumes, following the
%   HabEx/ATLAST layout: a sigma-free alpha per group, with Pre/Post x
%   iSeq (1=WFC, 2=metrology) slots.  v1 (uncontrolled) fills iSeq=1 Pre;
%   Post and iSeq=2 are declared NaN so the controls phase writes the same
%   shape without changing the schema.
%
%   COEF.<key> per group (key = sanitized group name):
%       .name      original group name
%       .class     native-unit class
%       .units     declared sigma units
%       .alpha     pre-WFC coefficient (iSeq=1)  [sigma-free for iid]
%       .alpha_post        NaN (controls phase)
%       .alpha_met, .alpha_met_post   NaN (iSeq=2, metrology phase)
%       .sigma     the group's 1-sigma (native units)
%       .wfe       sigma*alpha (nm, via COEF.meta.tonm)
%   COEF.total     total RSS WFE (nm)
%   COEF.meta      cbm, tonm (BaseUnit->nm), group order, controlled flag
%
%   See also macos.budget.rollup, macos.budget.write_xlsx.

p = inputParser;
p.addParameter('cbm', M.cbm, @isscalar);
p.parse(varargin{:});
tonm = p.Results.cbm * 1e9;

coef = struct();
ng = numel(rc.names);
keys = cell(ng, 1);
for k = 1:ng
    key = sanitize_(rc.names{k});
    keys{k} = key;
    coef.(key) = struct( ...
        'name', rc.names{k}, ...
        'class', rc.class{k}, ...
        'units', units_(spec, k), ...
        'alpha', rc.alpha(k), ...
        'alpha_post', NaN, ...
        'alpha_met', NaN, ...
        'alpha_met_post', NaN, ...
        'sigma', rc.sigma(k), ...
        'wfe', rc.wfe(k) * tonm);
end
coef.total = rc.total * tonm;
coef.meta = struct('cbm', p.Results.cbm, 'tonm', tonm, ...
    'order', {keys}, 'controlled', rc.controlled, ...
    'metric', M.units.metric);
end

function u = units_(spec, k)
u = '';
try
    g = spec.groups(k);
    if isfield(g, 'selector'), end
catch
end
end

function s = sanitize_(nm)
s = regexprep(char(nm), '[^A-Za-z0-9]', '_');
if isempty(s) || ~isletter(s(1)), s = ['g_' s]; end
end
