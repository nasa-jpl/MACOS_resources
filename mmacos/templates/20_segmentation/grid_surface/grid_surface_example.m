% grid_surface_example.m  (mmacos/templates/20_segmentation/)
% =====================================================================
%  GRID-DATA SURFACE API walkthrough on FFSegDemoAll.in
% =====================================================================
%  Exercises the elt_srf_grid_* family on a 7-segment FreeForm primary
%  (segments S2..S7 carry a grid-data component) -- read the live grid,
%  apply a measured-figure-error / DM displacement map, scale it, add to
%  it, round-trip it, and trace to see the wavefront change.
%
%  Rx: macos/ZGD_example/FFSegDemoAll.in (+ its GridFile
%  zern41em5z155em3.txt alongside).  We cd there so MACOS resolves the
%  GridFile, which is read by name at load.
%
%  Run:  >> run('.../templates/20_segmentation/grid_surface/grid_surface_example.m')
% =====================================================================

addpath('~/dev/MACOS_resources/mmacos/src');
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
rx    = fullfile(exdir, 'FFSegDemoAll.in');             % Rx + GridFile are local
old   = cd(exdir);  cleanup = onCleanup(@() cd(old));   % GridFile resolves from cwd

m = macos.Session(256);
m.load_rx(rx);
fprintf('loaded FFSegDemoAll: %d elements\n', m.num_elt());

% ---- which elements carry a grid surface? --------------------------
ff = m.find_freeform_elts();
fprintf('FreeForm elements: %s\n', mat2str(ff));
srf = ff(1);

% ---- query the live grid (raw family) ------------------------------
N  = mmacos('elt_srf_grid_size', double(srf), 1);          % live sampling
[dx, G0] = mmacos('elt_srf_grid_data', double(srf), 0.0, ...
                  zeros(N), 0.0, double(N), double(N));      % getter (setter flag 0)
fprintf('elt %d grid: %dx%d, dx=%.4g, sag RMS=%.4e\n', srf, N, N, dx, std(G0(:)));

% baseline trace
s0 = m.trace();
fprintf('\nbaseline RMS WFE = %.4e\n', s0.rmsWFE);

% ---- apply a known figure-error map via elt_grid_add ---------------
[X, Y] = meshgrid(linspace(-1, 1, N));
amp = 1e-6;                                  % 1 um peak sag
dz  = amp * (X.^2 - Y.^2);                   % a known astigmatism-like map (Rx units)
m.elt_grid_add(srf, dz);
s1 = m.trace();
fprintf('+ figure-error (1 um astig): RMS WFE = %.4e  (delta %.4e)\n', ...
        s1.rmsWFE, s1.rmsWFE - s0.rmsWFE);

% ---- round-trip: read the grid back, confirm the add ----------------
ffa = m.zrn_freeform(srf);
applied = ffa.grid.mat - G0;
fprintf('round-trip max|read-back - applied| = %.3e (should be ~0)\n', ...
        max(abs(applied(:) - dz(:))));

% ---- scale the grid in place (ARRAY form: iElt(N), scalar(N), N=#elts) --
mmacos('elt_srf_grid_data_scale', double(srf), 0.5, 1.0);   % N=1 (one element)
s2 = m.trace();
fprintf('after scale x0.5: RMS WFE = %.4e  (delta %.4e)\n', s2.rmsWFE, s2.rmsWFE-s0.rmsWFE);

% ---- zero the ENTIRE grid and trace --------------------------------
% sag RMS ~1.8e-4 m; zeroing it removes that figure error so the WFE moves
% (the grid setters call modified_rx -> the next trace re-runs; before that
% fix this was a silent no-op).
mmacos('elt_srf_grid_data', double(srf), dx, zeros(N), 1.0, double(N), double(N));
gz = m.zrn_freeform(srf);
fprintf('\nzeroed grid: read-back sag RMS = %.3e (should be 0)\n', std(gz.grid.mat(:)));
s4 = m.trace();
fprintf('after ZEROING the grid: RMS WFE = %.4e  (delta from baseline %.4e -- grid was applied)\n', ...
        s4.rmsWFE, s4.rmsWFE - s0.rmsWFE);

fprintf('\n--- grid-surface API exercised: size/data(get,set)/grid_add/scale/zrn_freeform ---\n');
