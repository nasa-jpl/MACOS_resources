function tg96_tail_batch(varargin)
%TG96_TAIL_BATCH  matlab -batch wrapper for tg96_tail (the explicit exit is
%   REQUIRED under -batch: a loaded mex hangs the implicit one).
try
    tg96_tail(varargin{:});
catch e
    fprintf(2, '%s\n', getReport(e, 'extended', 'hyperlinks', 'off'));
    exit(1);
end
exit(0);
end
