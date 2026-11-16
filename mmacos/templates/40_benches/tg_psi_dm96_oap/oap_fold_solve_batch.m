function oap_fold_solve_batch(varargin)
%OAP_FOLD_SOLVE_BATCH  matlab -batch wrapper for oap_fold_solve.  The explicit
%   exit is REQUIRED under -batch (a loaded mex hangs the implicit exit).
try
    oap_fold_solve(varargin{:});
catch e
    fprintf(2, '%s\n', getReport(e, 'extended', 'hyperlinks', 'off'));
    exit(1);
end
exit(0);
end
