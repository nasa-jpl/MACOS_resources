%E5_PIE  Pie segmentation with physical polygonal apertures.
%
% Manual worked example (companion to ../e5_seg, the hex case).  Run
% after mmacos_setup.  Every step writes a figure beside this script;
% numeric findings collect in findings.txt.
%
%   Step 1  segment the e5 monolithic primary into a 1-ring PIE
%           (macos.design.segment_rx: 7 segments = center + 6 wedges).
%   Step 2  measure each segment's TRUE ray footprint from the trace
%           (poke one segment, diff the OPD) and overlay the tiling
%           boundary reconstruction (macos.design.seg_boundary).  The
%           central cell of the (X,L,R) hex-coordinate tiling is a
%           HEXAGON, not a disc -- the footprint proves it.
%   Step 3  re-emit the prescription with each segment's physical
%           boundary declared as a polygonal aperture
%           (segment_rx emit_apertures=true -> macos.design.
%           seg_apertures: center hexagon + convex chorded sectors
%           with inner-sector obscurations; engine convention for
%           non-convex shapes is convex aperture minus convex
%           obscurations).
%   Step 4  trace parity: rays the source tiling places in the
%           inter-segment GAPS are clipped by the apertures --
%           physically honest (they have no glass under them); the
%           wavefront is unchanged where rays survive.
%   Step 5  perturbation honesty: decenter one wedge 100 mm -- rays
%           walking off the physical segment clip only when the
%           aperture is declared.
%   Step 6  launcher placement on the Rx-DECLARED edges: with the
%           polygons in the prescription, macos.design.seg_boundary
%           auto-switches to its 'rxpoly' source, so add_met places
%           launchers on the aperture polygons themselves (the general
%           case, covering imported segmented prescriptions), and
%           met_view renders the MET scene.
%
% See also: assess history in README.md (this example grew out of the
% 2026-07-16 poly-aperture assessment).

PAD  = 0;       % aperture clearance beyond the physical edge, mm.
                % 0 = the physical segment (gap rays clip); gap/2 would
                % put the aperture at the tiling midline (trace-neutral).
POKE = 0.100;   % wedge decenter for step 5, metres
NF   = 6;       % fiducials on the M2 rim (step 6)
R_FID = 590;    % fiducial ring radius, mm (~25 inside M2's 615 rim)
R_EXTRA = 100;  % aft/M3 launcher ring radius, mm (that truss's size)

here = fileparts(mfilename('fullpath'));
res_root = fullfile(getenv('HOME'), 'dev', 'MACOS_resources');
tin = fullfile(res_root, 'segmirmaker', 'test_in');
log_ = fopen(fullfile(here, 'findings.txt'), 'w');
say = @(varargin) fprintf(1, varargin{:}) + fprintf(log_, varargin{:});

%% ---- Step 1: pie segmentation ------------------------------------------
seg = macos.design.segment_rx(fullfile(tin, 'e5mono.in'), 'elt', 1, ...
    'rings', 1, 'grid', 'Pie', 'gap', 50, 'dofs', 6, 'meas_config', 1);
say('step 1: pie system, %d segments, width %.1f mm, gap %g mm\n', ...
    seg.nseg, seg.width, seg.gap);

old = cd(seg.run.workdir); restore = onCleanup(@() cd(old));
macos.init(512);
macos.load_rx(seg.in);
t0 = macos.trace();
pass_ = @(t) nnz(subsref(macos.get_ray_info(t.nRays), ...
    substruct('.', 'ok_pass')));
p0 = pass_(t0);
w0 = macos.opd();
say('        baseline: %d src rays, %d pass, rmsWFE %.6g mm\n', ...
    t0.nRays, p0, t0.rmsWFE);

%% ---- Step 2: true footprints vs the tiling boundary --------------------
% Poke each segment in turn (local piston) and mark the OPD pixels that
% move.  OPD is piston-removed, so every pixel shifts by a constant --
% the segment's own pixels sit at a distinct level; split on deviation
% from the median.
N = size(w0, 1);
labels = zeros(N);                                % pixel -> segment id
for s = 1:seg.nseg
    macos.load_rx(seg.in);
    macos.perturb(seg.seg_elts(s), 'translation', [0;0;1e-6], 'frame','local');
    macos.modify(); macos.trace();
    d = macos.opd() - w0;
    ok = isfinite(d) & (w0 ~= 0);
    dev = abs(d - median(d(ok)));
    labels(ok & dev > 0.25*max(dev(ok))) = s;
end
macos.load_rx(seg.in);                            % restore nominal
B = macos.design.seg_boundary(seg);               % tiling reconstruction
% Calibrate the OPD pixel grid to global X/Y from the trace itself:
% the source grid axes (xGrid/yGrid in the Rx) need not point along
% +X/+Y -- e5's xGrid is (-1,0,0) -- so fit the pixel->mm map from the
% footprint centroids instead of assuming a convention.
[xmm, ymm, toimg] = pupil_axes_(labels, B, seg);
f = figure('Visible','off', 'Position', [0 0 720 660]);
imagesc(xmm, ymm, toimg(labels), 'AlphaData', toimg(labels) > 0);
axis xy image; hold on
colormap([0.94 0.94 0.94; lines(seg.nseg)]); clim([-0.5 seg.nseg+0.5]);
overlay_(B, seg.nseg, 'k-', 1.4);
title('Step 2 -- traced segment footprints + tiling boundary');
xlabel('pupil x, mm'); ylabel('pupil y, mm');
text(0, 0, 'hexagonal center cell', 'Horiz', 'center', 'FontWeight', 'bold');
print(f, fullfile(here, 'e5pie_step2_footprints.png'), '-dpng', '-r120');
close(f);
say('step 2: footprint map + boundary overlay -> e5pie_step2_footprints.png\n');

%% ---- Step 3: emit the physical apertures --------------------------------
segA = macos.design.segment_rx(fullfile(tin, 'e5mono.in'), 'elt', 1, ...
    'rings', 1, 'grid', 'Pie', 'gap', 50, 'dofs', 6, 'meas_config', 1, ...
    'emit_apertures', true, 'ap_pad', PAD);
copyfile(segA.in, fullfile(here, 'e5pie_polyap.in'));
ap = segA.apertures;
say('step 3: apertures emitted (pad %g mm): center %s with %d vertices, wedges %d each -> e5pie_polyap.in\n', ...
    PAD, 'hexagon', size(ap.poly{1}, 2), size(ap.poly{2}, 2));
f = figure('Visible','off', 'Position', [0 0 720 660]);
imagesc(xmm, ymm, toimg(labels), 'AlphaData', 0.35*(toimg(labels) > 0));
axis xy image; hold on
colormap([0.94 0.94 0.94; lines(seg.nseg)]); clim([-0.5 seg.nseg+0.5]);
for s = 1:segA.nseg
    P = ap.poly{s}(:, [1:end 1]);
    plot(P(1,:), P(2,:), 'k-', 'LineWidth', 1.6);
    if ~isempty(ap.obs{s})
        O = ap.obs{s}(:, [1:end 1]);
        plot(O(1,:), O(2,:), 'r--', 'LineWidth', 1.0);
    end
end
title('Step 3 -- emitted aperture polygons (black) + inner obscurations (red)');
xlabel('pupil x, mm'); ylabel('pupil y, mm');
print(f, fullfile(here, 'e5pie_step3_apertures.png'), '-dpng', '-r120');
close(f);

%% ---- Step 4: trace parity (gap rays clip) --------------------------------
cd(segA.run.workdir);                             % its own GridFile dir
macos.load_rx(segA.in);
t1 = macos.trace();
p1 = pass_(t1);
w1 = macos.opd();
rs = macos.get_ray_status(t1.nRays);
fe = rs.fail_elt(rs.status ~= 0);
say('step 4: with apertures: %d pass (baseline %d), rmsWFE %.6g mm (%+.2f%%)\n', ...
    p1, p0, t1.rmsWFE, 100*(t1.rmsWFE - t0.rmsWFE)/t0.rmsWFE);
say('        %d rays clipped; first-fail element(s): %s\n', ...
    p0 - p1, mat2str(unique(fe(fe > 0)).'));
dm = (w0 ~= 0 & isfinite(w0)) & ~(w1 ~= 0 & isfinite(w1));
f = figure('Visible','off', 'Position', [0 0 720 660]);
imagesc(xmm, ymm, toimg(double(dm)), 'AlphaData', 0.25 + 0.75*toimg(double(dm)));
axis xy image; hold on
colormap([0.94 0.94 0.94; 0.85 0.2 0.2]); clim([0 1]);
overlay_(B, seg.nseg, 'k-', 0.8);
title(sprintf('Step 4 -- %d clipped rays: the inter-segment gaps + rim margin', nnz(dm)));
xlabel('pupil x, mm'); ylabel('pupil y, mm');
print(f, fullfile(here, 'e5pie_step4_clipped.png'), '-dpng', '-r120');
close(f);

%% ---- Step 5: perturbation honesty ---------------------------------------
% decenter wedge 2 by POKE: rays walking off the physical segment must
% not survive.  Without the aperture some do (they re-intersect the
% mathematical surface where no glass exists); with it, the loss lands
% at the DECLARED edge.
pass5 = zeros(1, 2);
for variant = 1:2
    if variant == 1
        cd(seg.run.workdir);  macos.load_rx(seg.in);
    else
        cd(segA.run.workdir); macos.load_rx(segA.in);
    end
    macos.perturb(seg.seg_elts(2), 'translation', [POKE; 0; 0], 'frame','local');
    macos.modify();
    tp = macos.trace();
    pass5(variant) = pass_(tp);
end
say('step 5: %g mm wedge decenter: %d rays pass with apertures, %d without --\n', ...
    POKE*1e3, pass5(2), pass5(1));
say('        the aperture-less trace keeps %d rays that have no glass under them\n', ...
    pass5(1) - pass5(2));
wp = macos.opd();
f = figure('Visible','off', 'Position', [0 0 720 660]);
dmp = (w1 ~= 0 & isfinite(w1)) & ~(wp ~= 0 & isfinite(wp));
imagesc(xmm, ymm, toimg(double(dmp)), 'AlphaData', 0.25 + 0.75*toimg(double(dmp)));
axis xy image; hold on
colormap([0.94 0.94 0.94; 0.85 0.2 0.2]); clim([0 1]);
overlay_(B, seg.nseg, 'k-', 0.8);
title(sprintf('Step 5 -- %g mm decenter of wedge 2: %d rays clip at the aperture', ...
      POKE*1e3, p1 - pass5(2)));
xlabel('pupil x, mm'); ylabel('pupil y, mm');
print(f, fullfile(here, 'e5pie_step5_poke.png'), '-dpng', '-r120');
close(f);

%% ---- Step 6: launchers on the Rx-declared edges + MET scene -------------
Brx = macos.design.seg_boundary(segA, 5);
say('step 6: seg_boundary source = %s (auto-detected from the Rx)\n', Brx.kind);
am = macos.design.add_met(segA.in, segA, 'hub', segA.nseg+1, ...
    'r_fid', R_FID, 'nf', NF, 'extra_sources', segA.n_elt-2, ...
    'r_extra', R_EXTRA, 'edge_off', 5);
macos.load_rx(am.in); macos.trace();
fv = macos.design.met_view(segA, am, 'visible', false, 'edge_off', 5, ...
    'title', 'e5 PIE: launchers on the Rx-declared aperture polygons', ...
    'save', fullfile(here, 'e5pie_step6_met.png'));
close(fv);
say('        %d MET beams; launchers sampled from the aperture polygons -> e5pie_step6_met.png\n', ...
    am.n_beams);
fclose(log_);
fprintf('findings.txt + 5 figures written to %s\n', here);

% ---------------------------------------------------------------------------
function [xmm, ymm, toimg] = pupil_axes_(labels, B, seg)
%PUPIL_AXES_  Calibrate the OPD pixel grid to global X/Y (mm).
%   Fits the affine pixel->mm map from the per-segment footprint
%   centroids (pixel space) against the tiling-region centroids (global
%   X/Y from the boundary polygons), so the overlay is exact whatever
%   the source grid orientation (xGrid/yGrid may be mirrored or flipped
%   wrt the global axes).  Returns ascending mm axes plus TOIMG, the
%   flip that puts a pixel matrix into that axis convention.
N = size(labels, 1);
[JJ, II] = meshgrid(1:N, 1:N);
Ppix = zeros(0, 3);  Pmm = zeros(0, 2);
for s = 1:seg.nseg
    m = labels == s;
    if ~any(m, 'all'), continue; end
    shp = polyshape(B.poly{s}(1, 1:end-1), B.poly{s}(2, 1:end-1));
    [cx, cy] = centroid(shp);
    Ppix(end+1, :) = [mean(JJ(m)), mean(II(m)), 1]; %#ok<AGROW>
    Pmm(end+1, :)  = [cx, cy];                      %#ok<AGROW>
end
A = (Ppix \ Pmm).';                    % 2x3: [X;Y] = A*[col;row;1]
% the engine OPD grid stores x along the ROW index (see the GridMat /
% write_grid_file orientation convention) -- if the fit says the map is
% transposed, swap the pixel axes and refit
swp = abs(A(1,1)) + abs(A(2,2)) < abs(A(1,2)) + abs(A(2,1));
if swp
    Ppix(:, [1 2]) = Ppix(:, [2 1]);
    A = (Ppix \ Pmm).';
end
offd = max(abs([A(1,2) A(2,1)])) / max(abs([A(1,1) A(2,2)]));
if offd > 0.05
    warning('e5_pie:grid', ...
        'pupil grid rotated %.2f wrt global X/Y; overlay approximate', offd);
end
assert(all(isfinite([A(:); 0])), 'pupil grid calibration failed');
xmm = A(1,1)*(1:N) + A(1,3);
ymm = A(2,2)*(1:N) + A(2,3);
fx = xmm(1) > xmm(end);  fy = ymm(1) > ymm(end);
if fx, xmm = flip(xmm); end
if fy, ymm = flip(ymm); end
toimg = @(M) flip_(flip_(tr_(M, swp), fx, 2), fy, 1);
end

function M = flip_(M, tf, dim)
if tf, M = flip(M, dim); end
end

function M = tr_(M, tf)
if tf, M = M.'; end
end

function overlay_(B, nseg, style, lw)
for s = 1:nseg
    plot(B.poly{s}(1,:), B.poly{s}(2,:), style, 'LineWidth', lw);
end
end
