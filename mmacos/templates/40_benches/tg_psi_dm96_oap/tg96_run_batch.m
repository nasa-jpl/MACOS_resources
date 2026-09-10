function tg96_run_batch(varargin)
%TG96_RUN_BATCH  matlab -batch wrapper for tg96_run.  The explicit exit is
%   REQUIRED under -batch (a loaded mex hangs the implicit exit); exit(1) on
%   failure with the report printed so a launcher can tell them apart.  Use
%   tg96_run itself interactively -- NEVER put exit in the user-facing runner.
try
    tg96_run(varargin{:});
catch e
    fprintf(2, '%s\n', getReport(e, 'extended', 'hyperlinks', 'off'));
    exit(1);
end
exit(0);
end
