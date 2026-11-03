function write_xlsx(T, coef, path, varargin)
%WRITE_XLSX  Write a fresh self-contained error-budget workbook.
%   macos.budget.write_xlsx(T, COEF, PATH) writes a two-sheet .xlsx the
%   tool owns: a user-facing "WFE" sheet (the roll-up table from
%   macos.budget.rollup) and a "Coefficients" sheet (one row per budget
%   line: the coefficients in nm per engineering unit, Pre/Post x iSeq, and
%   the native sigma-free alphas, from macos.budget.to_coef).  Plain values,
%   no formulas -- the live-formula calculator is macos.budget.
%   write_workbook.  Uses MATLAB's
%   built-in writetable (no Excel / openpyxl needed).
%
%   See also macos.budget.rollup, macos.budget.to_coef.

p = inputParser;
p.addParameter('title', 'MACOS error budget', @(x) ischar(x) || isstring(x));
p.parse(varargin{:});

if exist(path, 'file'), delete(path); end     % fresh workbook

% ---- WFE sheet: the user-facing roll-up ---------------------------------
writetable(T, path, 'Sheet', 'WFE');

% ---- Coefficients sheet: one row per budget line, Pre/Post x iSeq -------
% nm per engineering unit (E = C*D on the WFE side), plus the native alphas
keys = coef.meta.order;
n = numel(keys);
Name = strings(n, 1); Kind = strings(n, 1); Section = strings(n, 1);
Units = strings(n, 1); Units_drift = strings(n, 1);
Sigma = nan(n, 1); Sigma_drift = nan(n, 1);
Coef_preWFC = nan(n, 1); Coef_postWFC = nan(n, 1);
Coef_preMET = nan(n, 1); Coef_postMET = nan(n, 1);
Alpha_preWFC = nan(n, 1); Alpha_postWFC = nan(n, 1);
Alpha_preMET = nan(n, 1); Alpha_postMET = nan(n, 1);
Total_nm = nan(n, 1);
for k = 1:n
    c = coef.(keys{k});
    Name(k) = string(c.name);  Kind(k) = string(c.kind);
    Section(k) = string(c.section);  Units(k) = string(c.units);
    Units_drift(k) = string(c.units_drift);
    Sigma(k) = c.sigma_eng;  Sigma_drift(k) = c.sigma_drift;
    Coef_preWFC(k) = c.coef_pre;  Coef_postWFC(k) = c.coef_post;
    Coef_preMET(k) = c.coef_met;  Coef_postMET(k) = c.coef_met_post;
    Alpha_preWFC(k) = c.alpha;  Alpha_postWFC(k) = c.alpha_post;
    Alpha_preMET(k) = c.alpha_met;  Alpha_postMET(k) = c.alpha_met_post;
    Total_nm(k) = c.total;
end
C = table(Section, Name, Kind, Units, Sigma, Coef_preWFC, Coef_postWFC, ...
    Units_drift, Sigma_drift, Coef_preMET, Coef_postMET, Total_nm, ...
    Alpha_preWFC, Alpha_postWFC, Alpha_preMET, Alpha_postMET);
writetable(C, path, 'Sheet', 'Coefficients');

fprintf('wrote %s (WFE + Coefficients; total %.4g nm)\n', path, coef.total);
end
