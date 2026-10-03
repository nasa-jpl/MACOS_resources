function oap_focus_probe_batch(varargin)
%OAP_FOCUS_PROBE_BATCH  matlab -batch wrapper (the explicit exit is REQUIRED).
%   The 'tag' the launcher passes is not an option of the probe; drop it.
k = find(strcmp(varargin,'tag'), 1);
if ~isempty(k), varargin(k:k+1) = []; end
try
    oap_focus_probe(varargin{:});
catch e
    fprintf(2, '%s\n', getReport(e, 'extended', 'hyperlinks', 'off'));
    exit(1);
end
exit(0);
end
