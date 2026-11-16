function tg96_pupil_batch(varargin)
%TG96_PUPIL_BATCH  matlab -batch entry for the pupil-image tools (run it yourself).
%   tg96_pupil_batch('tool','sim'|'q'|'both', 'rig','lens'|'oap'|'both', name/value ...)
%   runs tg96_pupilsim (the detailed simulation) and/or tg96_pupilq (the crossing-cloud
%   quality assessment) on one or both rigs, reading the knobs of record from
%   tg96_params.m (the P.pupil block) with any name/value here overriding them.
%   The explicit exit is REQUIRED under -batch (a loaded mex hangs the implicit exit);
%   exit(1) on failure with the report printed.  Use the tools themselves interactively.
o = struct('tool','both','rig','both');
rest = {};
for k = 1:2:numel(varargin)
    if any(strcmp(varargin{k}, {'tool','rig'})), o.(varargin{k}) = varargin{k+1}; else, rest = [rest varargin(k:k+1)]; end %#ok<AGROW>
end
try
    P = tg96_params();
    knobs = {};
    if isfield(P, 'pupil'), f = fieldnames(P.pupil); for k = 1:numel(f), knobs = [knobs {f{k}, P.pupil.(f{k})}]; end, end %#ok<AGROW>
    rigs = {o.rig};  if strcmp(o.rig, 'both'), rigs = {'lens', 'oap'}; end
    for r = rigs
        if any(strcmp(o.tool, {'q', 'both'})),   tg96_pupilq('rig', r{1}, rest{:}); end
        if any(strcmp(o.tool, {'sim', 'both'})), tg96_pupilsim('rig', r{1}, knobs{:}, rest{:}); end
    end
catch e
    fprintf(2, '%s\n', getReport(e, 'extended', 'hyperlinks', 'off'));
    exit(1);
end
exit(0);
end
