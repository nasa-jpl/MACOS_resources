function ressweep_run_batch(varargin)
%RESSWEEP_RUN_BATCH  matlab -batch wrapper for ressweep_run (the explicit exit is
%   REQUIRED under -batch; exit(1) on failure with the report printed).
try
    ressweep_run(varargin{:});
catch e
    fprintf(2, '%s\n', getReport(e, 'extended', 'hyperlinks', 'off'));
    exit(1);
end
exit(0);
end
