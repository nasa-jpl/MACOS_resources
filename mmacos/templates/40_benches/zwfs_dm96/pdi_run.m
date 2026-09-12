function out = pdi_run(varargin)
%PDI_RUN  The point-diffraction readings' runner: zwfs_run on pdi_params.
%   out = pdi_run()                       the PDI record (pdi_params)
%   out = pdi_run(P)                      an edited pdi_params sheet
%   out = pdi_run('name', value, ...)     call-line overrides (dotted names OK)
%   out = pdi_run(P, 'tag', 'x', ...)     both
%   Stages, outputs and rules are zwfs_run's (README "Run it yourself");
%   headless: ./zwfs_batch.sh TAG "pdi_params, <the same name/value args>".
if ~isempty(varargin) && isstruct(varargin{1})
    P = varargin{1};  varargin(1) = [];
else
    P = pdi_params();
end
out = zwfs_run(P, varargin{:});
end
