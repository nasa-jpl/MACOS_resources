%VIEW_RX_DEMO  The general prescription visualizer on stock .in files.
%
% macos.view_rx works on ANY loaded prescription -- no design-layer
% structs: the beam is a sparse-but-FILLED rings-and-spokes bundle cut
% from the engine's per-trace ray-position history (macos.ray_hist --
% the full traced grid, true 3-D), each optic renders as a THIN flat-
% toned sag-following shell (true aperture boundary on the real conic
% sag; exact hex tiles for Segment elements on hex-segmented sources;
% consecutive refractors JOIN into one glass solid;
% Reference/Return/FocalPlane/Obscuring draw as outline frames), and
% laser-MET gauge paths render whenever the Rx declares
% nMetPos/tMetElt/metBeamFlg (macos.met_geom).
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
NRAYS  = 25;           % bundle ray budget ('rim'/'fans' modes)

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
    'ray_color', [0.72 0.0 0.72], ...   % channel color (cf. LightTools decks)
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

%% ---- 4. segmented hex primary: e5hex1, standard 3-view -----------------
% 7 EXACT hex Segment tiles (engine tiling truth via src_seg_get:
% width/gap + one global clocking -- tiles don't overlap and the gaps
% read), the m2 hub, and the JOINED lens_s1/lens_s2 glass solid.
% macos.view_std draws the three standard beam-aligned panels -- front
% (looking back up the beam at M1's face), iso, side -- with the layout
% convention: SOURCE AT LEFT, light travels right.  Fine-tune any panel
% with its [az el] option.
fprintf('[4] e5hex1.in (segmented hex primary), standard views\n');
macos.load_rx(fullfile(here, 'e5hex1.in'));    % copy committed beside script
macos.trace();
f = macos.view_std('visible', false, ...
    'args', {'ray_color', [0.9 0.45 0.0]}, ...
    'title', 'e5hex1.in -- macos.view_std (front / back / iso / side)', ...
    'save', fullfile(here, 'view_rx_e5hex1.png'));
close(f);

fprintf('done: view_rx_cass/coro/met/e5hex1.png + e5mono_met.in beside the script\n');
