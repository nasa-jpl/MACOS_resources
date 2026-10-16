function T = read_spec(src, varargin)
%READ_SPEC  Load a budget group table from Excel, CSV, or a .m, uniformly.
%   T = macos.budget.read_spec(SRC) returns the canonical group table from
%   any of:
%     - a MATLAB table or struct array (returned/normalized as-is)
%     - a .csv path            (readtable)
%     - a .xlsx/.xls path      (readtable; sheet 'WFE' or 'Budget' if present,
%                               else the first)
%     - a .m path or function name that RETURNS the table (feval), or a
%       script that leaves a variable T / spec / groups in scope
%
%   One schema, three authoring surfaces (Dave 2026-10-14).  Column names
%   are matched case-insensitively by macos.budget.spec_from_table.
%
%   Options:
%     'sheet'  worksheet name/index for .xlsx (overrides the default pick)
%
%   See also macos.budget.spec_from_table, macos.budget.suggest_groups.

p = inputParser;
p.addParameter('sheet', '', @(x) ischar(x) || isstring(x) || isnumeric(x));
p.parse(varargin{:});

if istable(src), T = src; return; end
if isstruct(src), T = struct2table(src, 'AsArray', true); return; end

src = char(src);
[~, ~, ext] = fileparts(src);
switch lower(ext)
    case '.csv'
        T = readtable(src, 'TextType', 'string');
    case {'.xlsx', '.xls'}
        shts = sheetnames(src);
        sh = p.Results.sheet;
        if isempty(sh)
            pick = find(ismember(lower(shts), {'wfe','budget','groups'}), 1);
            if isempty(pick), sh = 1; else, sh = shts(pick); end
        end
        T = readtable(src, 'Sheet', sh, 'TextType', 'string');
    case '.m'
        [d, fn] = fileparts(src);
        addpath(d); restore = onCleanup(@() rmpath(d));
        T = feval(fn);
    otherwise
        error('budget:read', 'unsupported spec source "%s"', src);
end
assert(istable(T), 'budget:read', 'spec source did not yield a table');
end
