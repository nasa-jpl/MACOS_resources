function pdi_run_batch(varargin)
%PDI_RUN_BATCH  pdi_run for `matlab -batch`: same arguments, then exit.
%   matlab -batch "pdi_run_batch('tag','pfdeck', pdi_params, 'stages',{'bench','battery'})"
%   The explicit exit is REQUIRED in batch mode (a loaded mex hangs MATLAB's
%   implicit exit -- mmacos/CLAUDE.md); a failure exits 1 with the report
%   printed, so a launcher can tell the two apart.
try
    pdi_run(varargin{:});
catch e
    fprintf(2, '%s\n', getReport(e, 'extended', 'hyperlinks', 'off'));
    exit(1);
end
exit(0);
end
