% grid_bug_demo.m  (mmacos/examples/grid_surface/)
% =====================================================================
%  GRAPHICAL demo: a FreeForm-grid figure error in the wavefront
% =====================================================================
%  Loads a grid-bearing FreeForm Rx, plots the OPD, ADDS a known figure-
%  error map to a segment's grid surface, re-plots, and shows the OPD
%  difference -- which is the applied figure error (the grid bites).
%
%  HISTORY: this once looked like a no-op -- the grid setters updated the
%  data but did NOT mark the trace dirty, so trace() returned the CACHED
%  OPD.  FIXED in macos_api_mod: the elt_srf_grid_data* setters now call
%  modified_rx, so a grid change dirties the trace and a plain trace()
%  picks it up (the engine math was always fine -- the grid sag enters L
%  via SFFSrf, surfsub.F:2437).  Drive via the knobs below.
%
%  Run (interactive, to SEE it):  >> run('.../grid_bug_demo.m')
% =====================================================================

addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/src'));

% ====================  KNOBS (drive these)  ==========================
ELT  = [];           % which FreeForm elt to perturb ([] = first grid elt)
AMP  = 5e-6;         % figure-error amplitude (m), peak
PAT  = 'astig';      % 'astig' | 'focus' | 'spot' (the shape added to the grid)
MODEL = 256;
% =====================================================================

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
old = cd(exdir);  cleanup = onCleanup(@() cd(old));   % GridFile resolves from cwd

m = macos.Session(MODEL);
m.load_rx(fullfile(exdir,'FFSegDemoAll.in'));         % Rx + GridFile are local
ff = m.find_freeform_elts();
srf = ELT;  if isempty(srf), srf = ff(1); end
fprintf('FreeForm grid elements: %s; perturbing elt %d\n', mat2str(ff(:).'), srf);

% ---- OPD BEFORE -----------------------------------------------------
m.trace();   W0 = m.opd();

% ---- build a known figure-error map + ADD it to the grid -----------
N = size(m.zrn_freeform(srf).grid.mat, 1);
[X,Y] = meshgrid(linspace(-1,1,N));
switch PAT
    case 'focus', dz = AMP*(X.^2 + Y.^2);
    case 'spot',  dz = AMP*exp(-((X.^2+Y.^2)/0.1));
    otherwise,    dz = AMP*(X.^2 - Y.^2);          % astigmatism
end
m.elt_grid_add(srf, dz);
gnow = m.zrn_freeform(srf).grid.mat;
fprintf('grid read-back: applied RMS = %.3e m (data layer %s)\n', ...
        std(dz(:)), tern(std(gnow(:))>0,'updated OK','NOT updated'));

% ---- OPD AFTER -- the grid setter now invalidates the trace --------
% macos_api_mod's grid setters call modified_rx, so a grid change dirties
% the trace and a plain trace() picks it up -- no explicit modify() needed.
% (Before the fix this was a silent no-op: trace() returned the cached OPD.)
m.trace();   W1 = m.opd();
dW = W1 - W0;
fprintf('grid_add -> trace -> opd:  OPD max|diff| = %.3e  <- the grid bites (setter dirties the trace)\n', ...
        max(abs(dW(:))));

% ---- GRAPHICS: applied grid | OPD before | OPD after | OPD diff -----
f = figure('Name','mmacos FreeForm-grid no-op','Position',[50 50 1150 850]);
subplot(2,2,1); imagesc(dz);   axis image off; colorbar;
    title(sprintf('applied grid figure error (elt %d, %s, %.0f um)', srf, PAT, AMP*1e6));
subplot(2,2,2); imagesc(W0);   axis image off; colorbar; title('OPD before');
subplot(2,2,3); imagesc(W1);   axis image off; colorbar; title('OPD after grid change');
subplot(2,2,4); imagesc(dW);   axis image off; colorbar;
    title(sprintf('OPD difference (max %.2e m) -- the applied grid figure error', max(abs(dW(:)))));
colormap(parula);
try
    saveas(f, fullfile(exdir, 'grid_bug_demo.png'));
catch, end

% ---- helpers --------------------------------------------------------
function r = rms_(W), v = W(isfinite(W) & W~=0); r = std(v); end
function s = tern(c,a,b), if c, s=a; else, s=b; end, end
