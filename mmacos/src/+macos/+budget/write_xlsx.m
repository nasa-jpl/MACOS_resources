function write_xlsx(T, coef, path, varargin)
%WRITE_XLSX  Write a fresh self-contained error-budget workbook.
%   macos.budget.write_xlsx(T, COEF, PATH) writes a two-sheet .xlsx the
%   tool owns: a user-facing "WFE" sheet (the roll-up table from
%   macos.budget.rollup) and a "Coefficients" sheet (the sigma-free alpha
%   per group, Pre/Post x iSeq, from macos.budget.to_coef).  Uses MATLAB's
%   built-in writetable (no Excel / openpyxl needed).
%
%   See also macos.budget.rollup, macos.budget.to_coef.

p = inputParser;
p.addParameter('title', 'MACOS error budget', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});

if exist(path, 'file'), delete(path); end     % fresh workbook

% ---- WFE sheet: the user-facing roll-up ---------------------------------
writetable(T, path, 'Sheet', 'WFE');

% ---- Coefficients sheet: alpha per group, Pre/Post x iSeq ---------------
keys = coef.meta.order;
ng = numel(keys);
Name = strings(ng, 1); Class = strings(ng, 1); Units = strings(ng, 1);
Sigma = nan(ng, 1);
Alpha_preWFC = nan(ng, 1); Alpha_postWFC = nan(ng, 1);
Alpha_preMET = nan(ng, 1); Alpha_postMET = nan(ng, 1);
WFE_nm = nan(ng, 1);
for k = 1:ng
    c = coef.(keys{k});
    Name(k) = string(c.name);  Class(k) = string(c.class);
    Units(k) = string(c.units);  Sigma(k) = c.sigma;
    Alpha_preWFC(k) = c.alpha;  Alpha_postWFC(k) = c.alpha_post;
    Alpha_preMET(k) = c.alpha_met;  Alpha_postMET(k) = c.alpha_met_post;
    WFE_nm(k) = c.wfe;
end
Name(end+1) = "TOTAL"; Class(end+1) = ""; Units(end+1) = "";
Sigma(end+1) = NaN; Alpha_preWFC(end+1) = NaN; Alpha_postWFC(end+1) = NaN;
Alpha_preMET(end+1) = NaN; Alpha_postMET(end+1) = NaN; WFE_nm(end+1) = coef.total;
C = table(Name, Class, Units, Sigma, Alpha_preWFC, Alpha_postWFC, ...
    Alpha_preMET, Alpha_postMET, WFE_nm);
writetable(C, path, 'Sheet', 'Coefficients');

fprintf('wrote %s (WFE + Coefficients; total %.4g nm)\n', path, coef.total);
end
