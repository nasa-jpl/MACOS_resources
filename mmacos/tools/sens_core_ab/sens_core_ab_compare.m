function nbad = sens_core_ab_compare(dir_pre, dir_post)
%SENS_CORE_AB_COMPARE  Byte-compare two sens_core_ab output directories.
%   NBAD = sens_core_ab_compare(DIR_PRE, DIR_POST) compares every
%   run-set .mat present in both directories field-by-field (isequaln).
%   Prints each differing field with max|diff| where numeric; error-
%   parity sets (same AB_ERROR string both sides) count as a pass.
%   NBAD = number of mismatching fields; 0 = byte-identical.
d = dir(fullfile(dir_pre, '*.mat'));
nbad = 0;  nset = 0;
for s = 1:numel(d)
    nm = erase(d(s).name, '.mat');
    fpost = fullfile(dir_post, d(s).name);
    if ~exist(fpost, 'file')
        fprintf('%s: MISSING on the post side\n', nm);
        nbad = nbad + 1;  continue;
    end
    nset = nset + 1;
    a = load(fullfile(dir_pre, d(s).name));
    b = load(fpost);
    fa = sort(fieldnames(a.S));  fb = sort(fieldnames(b.S));
    if ~isequal(fa, fb)
        fprintf('%s: FIELD SETS differ: pre-only {%s} post-only {%s}\n', nm, ...
            strjoin(setdiff(fa, fb), ','), strjoin(setdiff(fb, fa), ','));
        nbad = nbad + 1;
    end
    for k = 1:numel(fa)
        if ~ismember(fa{k}, fb), continue; end
        va = a.S.(fa{k});  vb = b.S.(fa{k});
        if ~isequaln(va, vb)
            fprintf('%s.%s: differs', nm, fa{k});
            if isnumeric(va) && isnumeric(vb) && isequal(size(va), size(vb))
                fprintf(' (max|d| %.3e)', max(abs(double(va(:)) - double(vb(:)))));
            end
            fprintf('\n');
            nbad = nbad + 1;
        end
    end
    if ismember('AB_ERROR', fa) && ismember('AB_ERROR', fb) ...
            && isequal(a.S.AB_ERROR, b.S.AB_ERROR)
        fprintf('%s: error parity (both sides): %s\n', nm, a.S.AB_ERROR);
    end
end
if nbad == 0
    fprintf('A/B: ALL %d RUN SETS BYTE-IDENTICAL (or error-parity)\n', nset);
else
    fprintf('A/B: %d mismatching fields -- NOT equivalent\n', nbad);
end
end
