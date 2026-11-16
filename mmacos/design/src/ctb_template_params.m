function P = ctb_template_params(ex)
%CTB_TEMPLATE_PARAMS  The CTB template's PARAMETER BLOCK, read from its file.
%   P = CTB_TEMPLATE_PARAMS() evaluates the "PARAMETER BLOCK" of
%   templates/30_instruments/bench_ctb/example_ctb.m (every length in mm:
%   F_OAP, AOI, R_OAP, R_DM, FILL, WAVLEN, the free legs L_*) and returns
%   it as the struct P that the template's own build uses -- so a chain
%   built elsewhere (the mmacos manual's chapter 2, the e2e stage 8 fold)
%   runs on the template's numbers without a copy that could drift.  The
%   template file is read, never written.  P = CTB_TEMPLATE_PARAMS(EX)
%   reads the block from the file EX instead.
%
%   See also CTB_FOLD_BUILD.
    if nargin < 1 || isempty(ex)
        here = fileparts(mfilename('fullpath'));
        ex = fullfile(here, '..', '..', 'templates', '30_instruments', 'bench_ctb', 'example_ctb.m');
    end
    assert(isfile(ex), 'ctb_template_params: %s not found', ex);
    el = splitlines(string(fileread(ex)));
    k0 = find(contains(el, "PARAMETER BLOCK"), 1) + 2;            % past the '% ====' rule
    k1 = k0 - 1 + find(strtrim(el(k0:end)) == "", 1) - 1;          % through the line before the blank
    assert(~isempty(k0) && k1 >= k0, 'ctb_template_params: no PARAMETER BLOCK in %s', ex);
    eval(char(strjoin(el(k0:k1), newline)));                       % defines P
    assert(exist('P', 'var') == 1 && isfield(P, 'F_OAP'), 'ctb_template_params: the block did not define P.F_OAP');
end
