function oap_loss_probe_batch(varargin)
%OAP_LOSS_PROBE_BATCH  matlab -batch wrapper (the explicit exit is REQUIRED).
try
    oap_loss_probe(varargin{:});
catch e
    fprintf(2, '%s\n', getReport(e, 'extended', 'hyperlinks', 'off'));
    exit(1);
end
exit(0);
end
