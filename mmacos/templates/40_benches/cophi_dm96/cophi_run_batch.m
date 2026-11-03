function cophi_run_batch(varargin)
%COPHI_RUN_BATCH  matlab -batch wrapper for cophi_run (the explicit exit is
%   REQUIRED under -batch; exit(1) on failure with the report printed).  Use
%   cophi_run itself interactively -- never put exit in the user-facing runner.
try
    cophi_run(varargin{:});
catch e
    fprintf(2, '%s\n', getReport(e, 'extended', 'hyperlinks', 'off'));
    exit(1);
end
exit(0);
end
