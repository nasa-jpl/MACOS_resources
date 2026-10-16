function psri_clearance_batch(varargin)
%PSRI_CLEARANCE_BATCH  matlab -batch wrapper (the explicit exit is REQUIRED).
try
    psri_clearance(varargin{:});
catch e
    fprintf(2, '%s\n', getReport(e, 'extended', 'hyperlinks', 'off'));
    exit(1);
end
exit(0);
end
