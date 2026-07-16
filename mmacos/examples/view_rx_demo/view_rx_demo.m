%VIEW_RX_DEMO  The general prescription visualizer on stock .in files.
%
% macos.view_rx works on ANY loaded prescription -- no design-layer
% structs: the beam is the engine DRAW command's real traced fans in
% true 3-D (macos.draw_rays3d, both meridians), each optic is drawn as
% surface cross-section curves through its actual beam footprint (any
% surface type, Segment/NS included), and laser-MET gauge paths render
% whenever the Rx declares nMetPos/tMetElt/metBeamFlg (macos.met_geom).
%
% Three cases, PNGs land beside this script:
%   1. CassWithExitPupil.in  -- classic double-pass Cassegrain (manual)
%   2. CoroExample.in        -- coronagraph train (manual)
%   3. e5mono + hand-added met blocks -- MET paths on a PLAIN Rx that
%      never went near the design layer (the .in is saved beside the
%      script so you can inspect the met keywords)
%
% Modify the knobs and re-run.  Requires a built mmacos mex.

MODEL  = 512;          % one model size for all cases (avoid transitions)
NRAYS  = 15;           % rays drawn per fan

here = fileparts(mfilename('fullpath'));
res_root = fileparts(fileparts(fileparts(here)));      % MACOS_resources
man = fullfile(fileparts(res_root), 'macos', 'docs', 'macos-manual', 'examples');
tin = fullfile(res_root, 'segmirmaker', 'test_in');

macos.init(MODEL);

%% ---- 1. classic Cassegrain with exit-pupil return ---------------------
fprintf('[1] CassWithExitPupil.in\n');
old = cd(man); c1 = onCleanup(@() cd(old));
macos.load_rx(fullfile(man, 'CassWithExitPupil.in'));
macos.trace();
f = macos.view_rx('nrays', NRAYS, 'visible', false, ...
    'title', 'CassWithExitPupil.in -- macos.view_rx', ...
    'save', fullfile(here, 'view_rx_cass.png'));
close(f);

%% ---- 2. coronagraph train ---------------------------------------------
fprintf('[2] CoroExample.in\n');
macos.load_rx(fullfile(man, 'CoroExample.in'));
macos.trace();
f = macos.view_rx('nrays', NRAYS, 'visible', false, ...
    'title', 'CoroExample.in -- macos.view_rx', ...
    'save', fullfile(here, 'view_rx_coro.png'));
close(f);

%% ---- 3. MET paths on a plain prescription ------------------------------
% Hand-splice engine met keywords into e5mono (two launchers on m2
% beamed to two fiducials on the focal-plane bench) -- the same element
% syntax any user Rx can carry; view_rx picks the gauges up through
% macos.met_geom with no design-layer involvement.
fprintf('[3] e5mono + met blocks\n');
lines = readlines(fullfile(tin, 'e5mono.in'));
v3 = @(p) sprintf('  %.15E  %.15E  %.15E', p);
a = [0; -5471.177517626807; -21308.82954482988];       % m2 vertex
b = a + [400; 0; 0];
c = [0; -6571.126153057798; 3678.032705099662];        % fpa vertex
d = c + [0; 400; 0];
im2 = find(strtrim(lines) == "EltName=  m2", 1);
lines = [lines(1:im2); ...
    "          nMetPos=  2"; string(v3(a)); string(v3(b)); ...
    "          tMetElt=  5  2"; "  1  0"; "  0  1"; lines(im2+1:end)];
ifpa = find(strtrim(lines) == "EltName=  fpa", 1);
lines = [lines(1:ifpa); ...
    "          nMetPos=  2"; string(v3(c)); string(v3(d)); lines(ifpa+1:end)];
met_in = fullfile(here, 'e5mono_met.in');
writelines(lines, met_in);
copyfile(fullfile(tin, 'flat.txt'), fullfile(here, 'flat.txt'));
cd(here);                                  % GridFile= resolves from cwd
macos.load_rx(met_in);
macos.trace();
f = macos.view_rx('nrays', NRAYS, 'visible', false, ...
    'title', 'e5mono + met keywords -- gauges via macos.met_geom', ...
    'save', fullfile(here, 'view_rx_met.png'));
close(f);

fprintf('done: view_rx_cass/coro/met.png + e5mono_met.in beside the script\n');
