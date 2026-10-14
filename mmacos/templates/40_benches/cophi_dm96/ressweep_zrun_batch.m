function ressweep_zrun_batch(varargin)
%RESSWEEP_ZRUN_BATCH  matlab -batch wrapper for ressweep_zrun (the explicit exit is
%   REQUIRED under -batch; exit(1) on failure with the report printed).
try
    ressweep_zrun(varargin{:});
catch e
    fprintf(2, '%s\n', getReport(e, 'extended', 'hyperlinks', 'off'));
    exit(1);
end
exit(0);
end
