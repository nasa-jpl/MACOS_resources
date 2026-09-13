function out = pdi_run(varargin)
%PDI_RUN  The point-diffraction readings' runner: zwfs_run on pdi_params.
%   out = pdi_run()                       the PDI record (pdi_params)
%   out = pdi_run(P)                      an edited pdi_params sheet
%   out = pdi_run('name', value, ...)     call-line overrides (dotted names OK)
%   out = pdi_run(P, 'tag', 'x', ...)     both
%   Stages, outputs and rules are zwfs_run's (README "Run it yourself");
%   headless: ./pdi_batch.sh TAG "pdi_params, <the same name/value args>".
%   Outputs land in <this dir>/runs/<tag>/ unless 'outdir' is given.
%   The code is SHARED: zwfs_run + ../dm_gauge_lib, nothing is copied.
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
addpath(fullfile(exdir, '..', 'zwfs_dm96'));                % zwfs_run, zwfs_params, zwfs_mask
is = find(cellfun(@isstruct, varargin), 1);     % the sheet may sit anywhere (the batch
if ~isempty(is)                                 % wrapper puts 'tag', TAG first)
    P = varargin{is};  varargin(is) = [];
else
    P = pdi_params();
end
P = applyargs_(P, varargin);
if isempty(P.outdir), P.outdir = fullfile(exdir, 'runs', P.tag); end
out = zwfs_run(P);
end

function P = applyargs_(P, args)
% the tag has to be resolved BEFORE the output directory is, so the
% name/value pairs are applied here as well as (idempotently) in zwfs_run
assert(mod(numel(args), 2) == 0, 'pdi_run: name/value pairs expected');
for i = 1:2:numel(args)
    parts = strsplit(args{i}, '.');
    P = setfield(P, parts{:}, args{i+1}); %#ok<SFLD>
end
end
