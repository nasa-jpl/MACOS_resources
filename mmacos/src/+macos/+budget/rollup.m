function T = rollup(rc, varargin)
%ROLLUP  Assemble the budget table (HabEx calculator columns) from the results.
%   T = macos.budget.rollup(RC) returns a MATLAB table, one row per budget
%   line (macos.budget.to_coef's lines: error groups, then the WFC noise
%   rows, then metrology-only lines), then the summary rows, in nm:
%
%     Section Name Kind | Sigma Units (initial-alignment input, engineering
%     units) | Pre_WFC_nm Post_WFC_nm | Drift_Sigma Drift_Units |
%     Pre_MET_nm Post_MET_nm | Total_nm = sqrt(Post_WFC^2 + Post_MET^2)
%
%   (the HabEx WFE sheet's C/D | E F | G/H | I J | K).  Summary rows
%   (Kind 'summary'): '<section> WFE' per section (RSS of its lines, every
%   column), 'Everything Else' (all lines outside the FOCUS section),
%   'TOTAL' (all lines), 'Design WFE' (input), 'Payload WFE' =
%   sqrt(TOTAL^2 + Design^2), 'WFE Requirement' (input), 'WFE Margin' =
%   1 - Payload/Requirement (a FRACTION, in Total_nm).
%
%   Options:
%     'mc'          mc_budget result: MC_Post_WFC_nm column (groups by
%                   order, noise rows by name)
%     'mc_met'      mc_met result: MC_Post_MET_nm column (by name)
%     'cbm'         BaseUnit -> metre (default 1 = metric already metres)
%     'design_wfe'  the design residual, nm (default 0)
%     'requirement' the WFE requirement, nm (default NaN)
%     'focus'       the section reported against 'Everything Else'
%                   (default: the first line's section)
%     'coef'        a to_coef result to use (default: built from RC)
%
%   See also macos.budget.to_coef, macos.budget.cov_budget,
%   macos.budget.write_csv, macos.budget.write_xlsx.

p = inputParser;
p.addParameter('mc', [], @(x) isempty(x) || isstruct(x));
p.addParameter('mc_met', [], @(x) isempty(x) || isstruct(x));
p.addParameter('cbm', 1, @isscalar);
p.addParameter('design_wfe', 0, @isscalar);
p.addParameter('requirement', NaN, @isscalar);
p.addParameter('focus', '', @(x) ischar(x) || isstring(x));
p.addParameter('coef', [], @(x) isempty(x) || isstruct(x));
p.parse(varargin{:});
o = p.Results;
tonm = o.cbm * 1e9;
coef = o.coef;
if isempty(coef), coef = macos.budget.to_coef(rc, [], [], 'cbm', o.cbm); end

keys = coef.meta.order;  n = numel(keys);
Section = cell(n, 1);  Name = cell(n, 1);  Kind = cell(n, 1);
Units = cell(n, 1);  Drift_Units = cell(n, 1);
Sigma = nan(n, 1);  Drift_Sigma = nan(n, 1);
Pre_WFC_nm = nan(n, 1);  Post_WFC_nm = nan(n, 1);
Pre_MET_nm = nan(n, 1);  Post_MET_nm = nan(n, 1);  Total_nm = nan(n, 1);
MC_Post_WFC_nm = nan(n, 1);  MC_Post_MET_nm = nan(n, 1);
for i = 1:n
    e = coef.(keys{i});
    Section{i} = e.section;  Name{i} = e.name;  Kind{i} = e.kind;
    Units{i} = e.units;  Drift_Units{i} = e.units_drift;
    Sigma(i) = e.sigma_eng;  Drift_Sigma(i) = e.sigma_drift;
    Pre_WFC_nm(i) = e.wfe_pre;  Post_WFC_nm(i) = e.wfe;
    Pre_MET_nm(i) = e.wfe_met;  Post_MET_nm(i) = e.wfe_met_post;
    Total_nm(i) = e.total;
end
% Monte-Carlo columns
ng = numel(rc.names);
if ~isempty(o.mc)
    for i = 1:n
        k = find(strcmp(rc.names, Name{i}), 1);
        if ~isempty(k) && k <= ng && strcmp(Kind{i}, 'error')
            MC_Post_WFC_nm(i) = o.mc.wfe(k) * tonm;
        elseif isfield(o.mc, 'extra') && ~isempty(o.mc.extra)
            j = find(strcmp({o.mc.extra.name}, Name{i}), 1);
            if ~isempty(j), MC_Post_WFC_nm(i) = o.mc.extra(j).wfe * tonm; end
        end
    end
end
if ~isempty(o.mc_met)
    for i = 1:n
        j = find(strcmp(o.mc_met.names, Name{i}), 1);
        if ~isempty(j)
            MC_Post_MET_nm(i) = o.mc_met.wfe_post(j) * tonm;
        else
            j = find(strcmp({o.mc_met.extra.name}, Name{i}), 1);
            if ~isempty(j), MC_Post_MET_nm(i) = o.mc_met.extra(j).wfe * tonm; end
        end
    end
end

% ---- summary rows -------------------------------------------------------
cols = [Pre_WFC_nm Post_WFC_nm Pre_MET_nm Post_MET_nm Total_nm];
mcc  = [MC_Post_WFC_nm MC_Post_MET_nm];
rss = @(m) sqrt(sum(m.^2, 1));
secs = unique(Section, 'stable');
focus = char(o.focus);  if isempty(focus) && n > 0, focus = Section{1}; end
S = {};  V = [];  VM = [];
for s = 1:numel(secs)
    in = strcmp(Section, secs{s});
    S{end+1} = sprintf('%s WFE', secs{s}); %#ok<AGROW>
    V(end+1, :) = rss(cols(in, :));  VM(end+1, :) = rssn_(mcc(in, :)); %#ok<AGROW>
end
ee = ~strcmp(Section, focus);
S{end+1} = 'Everything Else';  V(end+1, :) = rss(cols(ee, :));  VM(end+1, :) = rssn_(mcc(ee, :));
S{end+1} = 'TOTAL';            V(end+1, :) = rss(cols);          VM(end+1, :) = rssn_(mcc);
tot = V(end, 5);
pay = sqrt(tot^2 + o.design_wfe^2);
S{end+1} = 'Design WFE';       V(end+1, :) = [NaN NaN NaN NaN o.design_wfe];  VM(end+1, :) = NaN;
S{end+1} = 'Payload WFE';      V(end+1, :) = [NaN NaN NaN NaN pay];           VM(end+1, :) = NaN;
S{end+1} = 'WFE Requirement';  V(end+1, :) = [NaN NaN NaN NaN o.requirement]; VM(end+1, :) = NaN;
S{end+1} = 'WFE Margin';       V(end+1, :) = [NaN NaN NaN NaN 1 - pay/o.requirement]; VM(end+1, :) = NaN;
m = numel(S);
Section = [Section; repmat({'WFE Summary'}, m, 1)];
Name = [Name; S(:)];  Kind = [Kind; repmat({'summary'}, m, 1)];
Units = [Units; repmat({'nm'}, m, 1)];  Drift_Units = [Drift_Units; repmat({''}, m, 1)];
Sigma = [Sigma; nan(m, 1)];  Drift_Sigma = [Drift_Sigma; nan(m, 1)];
Pre_WFC_nm = [Pre_WFC_nm; V(:, 1)];  Post_WFC_nm = [Post_WFC_nm; V(:, 2)];
Pre_MET_nm = [Pre_MET_nm; V(:, 3)];  Post_MET_nm = [Post_MET_nm; V(:, 4)];
Total_nm = [Total_nm; V(:, 5)];
MC_Post_WFC_nm = [MC_Post_WFC_nm; VM(:, 1)];  MC_Post_MET_nm = [MC_Post_MET_nm; VM(:, 2)];
Units{end} = 'fraction';

T = table(Section, Name, Kind, Sigma, Units, Pre_WFC_nm, Post_WFC_nm, ...
    Drift_Sigma, Drift_Units, Pre_MET_nm, Post_MET_nm, Total_nm);
if ~isempty(o.mc), T.MC_Post_WFC_nm = MC_Post_WFC_nm; end
if ~isempty(o.mc_met), T.MC_Post_MET_nm = MC_Post_MET_nm; end
T.Properties.Description = sprintf( ...
    'error budget: %d lines; Total WFE %.4g nm, Payload %.4g nm', n, tot, pay);
end

function r = rssn_(m)
% RSS over rows, NaN where every row is NaN
r = sqrt(sum(m.^2, 1, 'omitnan'));
r(all(isnan(m), 1)) = NaN;
if isempty(m), r = nan(1, 2); end
end
