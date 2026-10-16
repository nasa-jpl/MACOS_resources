function oap_conj_probe_batch(varargin)
%OAP_CONJ_PROBE_BATCH  matlab -batch wrapper (the explicit exit is REQUIRED).
try
    oap_conj_probe(varargin{:});
catch e
    fprintf(2, '%s\n', getReport(e, 'extended', 'hyperlinks', 'off'));
    exit(1);
end
exit(0);
end
