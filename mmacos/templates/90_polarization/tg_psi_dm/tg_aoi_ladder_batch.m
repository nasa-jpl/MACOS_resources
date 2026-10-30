function tg_aoi_ladder_batch(varargin)
%TG_AOI_LADDER_BATCH  matlab -batch wrapper for tg_aoi_ladder (the explicit
%   exit is REQUIRED under -batch: a loaded mex hangs the implicit one).
%   First argument is the AOI list, the rest are tg_aoi_ladder options.
try
    tg_aoi_ladder(varargin{:});
catch e
    fprintf(2, '%s\n', getReport(e, 'extended', 'hyperlinks', 'off'));
    exit(1);
end
exit(0);
end
