function zwfs_run_batch(varargin)
%ZWFS_RUN_BATCH  zwfs_run for `matlab -batch`: same arguments, then exit.
%   matlab -batch "zwfs_run_batch('tag','ng385', 'NGRID',385)"
%   The explicit exit is REQUIRED in batch mode (a loaded mex hangs
%   MATLAB's implicit exit -- mmacos/CLAUDE.md); a failure exits 1 with
%   the report printed, so a launcher can tell the two apart.  Use
%   zwfs_run itself from an interactive session.
try
    zwfs_run(varargin{:});
catch e
    fprintf(2, '%s\n', getReport(e, 'extended', 'hyperlinks', 'off'));
    exit(1);
end
exit(0);
end
