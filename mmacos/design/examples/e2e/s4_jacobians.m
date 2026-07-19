%S4_JACOBIANS  Stage 4: sensitivity Jacobians on the segmented system.
%
% Stage 4 of the end-to-end worked example: the LINEAR MODEL substrate
% for stages 5 (MET configuration) and 6 (simulator).  Harvests the
% three wavefront-sensitivity channels on the stage-3 artifact
% (e2e_<P.seg.variant>.in, default pie) over the +-2' science field
% (center + 4 corners), each in the canonical state-vector form
%
%     wall = dwdxall * x + w0_stacked        (macos.dw_d*_multi)
%
%   [1] dwdx    rigid-body 6-DOF per element (per body
%               [rot_xyz|trans_xyz] in its LOCAL/TElt triad -- the
%               Sprint-2D ordering); segments are the controlled
%               bodies, the relay mirrors ride along for disturbance
%               studies.
%   [2] dwdz    segment FIGURE channel: FF-Zernike coefficient
%               derivatives (kinds={'ffzern'}) modes 4..11 on every
%               segment.
%   [3] dwdgrid segment grid-poke channel on a GRID-AUGMENTED variant
%               (e2e_<v>_grid.in): each segment gets a flat 256 grid
%               in its CLOCKED Mon frame (pData..zData = pMon..zMon --
%               the frames must match or pokes don't localize), and
%               the pokes are aperture-confined Zernike influence maps
%               (macos.zernike_grid_basis).
%   [4] conditioning report: size / rank / singular-value spectrum per
%               channel + per-segment column norms (who moves the
%               wavefront how hard).
%   [5] figures: tiled nominal-OPD canvas + the three SV spectra.
%
% Artifacts (beside this script): s4_report.txt, s4_jacobians.mat
% (dwdx/dwdz/dwdg outputs), e2e_<v>_grid.in + flat.txt, s4_svspec.png,
% s4_opdall.png.
%
% Run AFTER s3_segmentation.m.

addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/design/src'));
P = e2e_params();
here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
v  = char(P.seg.variant);
rx = fullfile(here, sprintf('e2e_%s.in', v));
assert(isfile(rx), 's4 needs e2e_%s.in -- run s3_segmentation.m first', v);

FOV   = P.inst.fov_arcmin * pi/180/60;    % +-2' corners (science patch)
ZMODES_FIG  = 4:11;                        % dwdz: figure modes/segment
ZMODES_GRID = [4 5 6 7 8 9];               % dwdgrid pokes/segment
NG    = 256;                               % grid size (model 512)

log_ = fopen(fullfile(here, 's4_report.txt'), 'w');
say = @(varargin) fprintf(1, varargin{:}) + fprintf(log_, varargin{:});
say('==== e2e stage 4: sensitivity Jacobians on e2e_%s.in ====\n', v);
say('field set: center + 4 corners at +-%g'' (the science patch)\n\n', ...
    P.inst.fov_arcmin);

% segment bookkeeping from the artifact text (Seg blocks lead the file)
txt = fileread(rx);
nseg = numel(regexp(txt, 'Element=\s*Segment', 'match'));
segs = 1:nseg;

%% -- [1] dwdx: rigid-body 6-DOF per element ---------------------------
m = macos.Session(P.seg.model_size);
say('[1] dwdx (rigid-body, 6 DOF/element, LOCAL triads)...\n');
ox = macos.dw_dx_multi(m, rx, 'field_x_rad', FOV, 'field_y_rad', FOV);
say('    dwdxall %d x %d over %d fields; channels: %s ...\n', ...
    size(ox.dwdxall, 1), size(ox.dwdxall, 2), size(ox.field_table, 1), ...
    strjoin(ox.channel_names(1:min(3, end)), ', '));

%% -- [2] dwdz: segment FF-Zernike figure channel ----------------------
say('\n[2] dwdz (segment figure, FF-Zernike modes %s)...\n', mat2str(ZMODES_FIG));
oz = macos.dw_dz_zernike_multi(m, rx, 'field_x_rad', FOV, 'field_y_rad', FOV, ...
    'kinds', {'ffzern'}, 'zmode_start', ZMODES_FIG(1), 'n_zcoef', ZMODES_FIG(end));
say('    dwdzall %d x %d; channels: %s ...\n', size(oz.dwdxall, 1), ...
    size(oz.dwdxall, 2), strjoin(oz.channel_names(1:min(3, end)), ', '));

%% -- [3] dwdgrid: per-segment grid pokes on the grid variant ----------
% augment each Segment block with a flat 256 grid in its clocked Mon
% frame; span sized to cover the wedge/hexagon footprint
say('\n[3] dwdgrid: grid-augmenting the artifact...\n');
gdx = 2*0.7*P.D_m/2 / (NG - 1);           % half-extent 0.7*(D/2) per segment
rxg = fullfile(here, sprintf('e2e_%s_grid.in', v));
L = splitlines(string(fileread(rx)));
outL = strings(0, 1);
inseg = false;  monf = containers.Map;
for i = 1:numel(L)
    ln = L(i);
    outL(end+1) = ln; %#ok<AGROW>
    tl = strtrim(ln);
    if startsWith(tl, 'Element=')
        inseg = contains(tl, 'Segment');
        monf = containers.Map;
    end
    if inseg
        for key = ["pMon" "xMon" "yMon" "zMon"]
            if startsWith(tl, key + "=")
                monf(char(key)) = regexprep(char(tl), '^\w+=', '');
            end
        end
        if startsWith(tl, "zMon=") && monf.Count == 4
            outL(end+1) = sprintf('         nGridMat=  %d', NG); %#ok<AGROW>
            outL(end+1) = "         GridFile=  flat.txt"; %#ok<AGROW>
            outL(end+1) = sprintf('        GridSrfdx=%.6E', gdx); %#ok<AGROW>
            outL(end+1) = "            pData=" + monf('pMon'); %#ok<AGROW>
            outL(end+1) = "            xData=" + monf('xMon'); %#ok<AGROW>
            outL(end+1) = "            yData=" + monf('yMon'); %#ok<AGROW>
            outL(end+1) = "            zData=" + monf('zMon'); %#ok<AGROW>
        end
    end
end
fid = fopen(rxg, 'w'); fprintf(fid, '%s\n', outL); fclose(fid);
if ~isfile(fullfile(here, 'flat.txt'))
    macos.write_grid_file(fullfile(here, 'flat.txt'), zeros(NG));
end
oldd = cd(here);  restore = onCleanup(@() cd(oldd));   % GridFile= from cwd
nge = macos.load_rx(rxg);  tg = macos.trace(nge);      % sanity: loads+traces
say('    e2e_%s_grid.in: %d elts, %d/%d rays (grid frames = clocked Mon)\n', ...
    v, nge, nnz(logical(macos.get_ray_info(tg.nRays).ok_pass)), tg.nRays);

% aperture-confined Zernike pokes inside each segment footprint
lMon = str2num(regexp(txt, '(?<=lMon=)\s*[\d.eEdD+-]+', 'match', 'once')); %#ok<ST2NM>
ap_frac = min(lMon / (((NG-1)/2)*gdx), 1);
infl = macos.zernike_grid_basis(NG, ZMODES_GRID, ap_frac);
say('    pokes: modes %s, aperture %.0f%% of grid half-width\n', ...
    mat2str(ZMODES_GRID), 100*ap_frac);
og = macos.dw_dgrid_multi(m, rxg, 'field_x_rad', FOV, 'field_y_rad', FOV, ...
    'influence', infl);
say('    dwdgall %d x %d\n', size(og.dwdgall, 1), size(og.dwdgall, 2));

%% -- [4] conditioning report -----------------------------------------
say('\n[4] channel conditioning (finite rows only):\n');
say('    %-8s %12s %6s %10s %10s %10s\n', 'channel', 'size', 'rank', ...
    'sv_max', 'sv_min+', 'cond+');
J = {'dwdx', ox.dwdxall; 'dwdz', oz.dwdxall; 'dwdgrid', og.dwdgall};
SV = cell(3, 1);
for q = 1:3
    A = J{q,2};  A = A(all(isfinite(A), 2), :);
    s = svd(full(A), 'econ');  SV{q} = s;
    tol = max(size(A)) * eps(max(s));
    rk = nnz(s > tol);
    sp = s(s > tol);
    say('    %-8s %5dx%-6d %6d %10.3e %10.3e %10.3e\n', J{q,1}, ...
        size(A,1), size(A,2), rk, s(1), sp(end), s(1)/sp(end));
end
% per-segment rigid-body column norms: who moves the wavefront hardest
say('\n    dwdx per-segment column norms (rms waves per unit DOF):\n');
say('      seg     rx        ry        rz        tx        ty        tz\n');
cn = ox.channel_names;                     % 'Elt 1 Rx', 'Elt 1 Ry', ...
for s = segs
    ic = find(startsWith(cn, sprintf('Elt %d ', s)));
    if numel(ic) == 6
        nv = sqrt(mean(ox.dwdxall(:, ic).^2, 1, 'omitnan'));
        say('      %3d  %s\n', s, sprintf('%9.2e ', nv));
    else
        say('      %3d  (channels not found: %d)\n', s, numel(ic));
    end
end

%% -- [5] figures ------------------------------------------------------
f = figure('Visible', 'off', 'Position', [0 0 760 520]);
mk = {'-o', '-s', '-^'};
for q = 1:3
    semilogy(SV{q}/SV{q}(1), mk{q}, 'MarkerSize', 3); hold on
end
grid on; legend({'dwdx', 'dwdz', 'dwdgrid'}, 'Location', 'southwest');
xlabel('singular value index'); ylabel('\sigma_i / \sigma_1');
title(sprintf('e2e s4: Jacobian spectra (%s, %d segs, 5 fields, +-%g'')', ...
      v, nseg, P.inst.fov_arcmin));
print(f, fullfile(here, 's4_svspec.png'), '-dpng', '-r120'); close(f);
f = figure('Visible', 'off', 'Position', [0 0 900 700]);
imagesc(ox.OPDall); axis image; colorbar;
title('e2e s4: nominal OPD, field canvas (center + 4 corners)');
print(f, fullfile(here, 's4_opdall.png'), '-dpng', '-r120'); close(f);
say('\n[5] figures: s4_svspec.png + s4_opdall.png\n');

save(fullfile(here, 's4_jacobians.mat'), 'ox', 'oz', 'og', 'P', '-v7.3');
say('\nStage 4 complete: s4_jacobians.mat + s4_report.txt + e2e_%s_grid.in\n', v);
say('Next: s5_met.m (shape-class launcher patterns; dedx/dldx join dwdx here).\n');
fclose(log_);
