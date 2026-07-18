%S3_SEGMENTATION  Stage 3: segment the e2e primary -- PIE and 2-ring HEX.
%
% Stage 3 of the end-to-end worked example.  Segments M1 of the stage-2
% system (parent = s2_instrument.in, element 1) TWO ways, keeping both
% as artifacts (Dave 2026-07-18):
%
%   pie   1-ring PIE -> 7 segments: a center HEXAGON + 6 wedges whose
%         inner edges abut it along straight chords  ->  e2e_pie.in
%   hex2  2-ring HEX -> 19 hexagonal segments        ->  e2e_hex2.in
%
% Segment geometry SCALES WITH THE APERTURE: SegMirMaker's default
% segment size is Aperture/(2*rings+1), so the 4 m e2e primary (half
% the 8 m e5 fixtures) gets half-size segments for free; the explicit
% knob that scales with it is the gap (50 -> 25 mm, P.seg.gap_mm).
% NOTE the units: this prescription family is BaseUnits=m (the e5
% fixtures are mm), so every length handed to segment_rx is in metres.
%
% Per variant:
%   [1] bare segmentation (segment_rx splice; no declared apertures)
%       + load/trace parity vs the monolithic parent.  The parent's
%       M1 central-hole obscuration (set_hole emission) rides onto the
%       CENTER segment automatically (segment_rx carry_obs).
%   [2] traced segment footprints (poke one segment, diff the OPD)
%       overlaid with the tiling boundary + emitted aperture polygons
%   [3] re-emit with each segment's PHYSICAL polygonal aperture
%       declared (emit_apertures: pie = center hexagon + chorded
%       sectors, hex = exact corners); gap rays clip -- physically
%       honest -- and the artifact .in is THIS version
%   [4] views (view_std 4-panel) of the segmented system; the M1 hole
%       renders on the center segment (view_rx obscuration layer)
%   [5] standalone reload verification of the artifact
%
% P.seg.variant picks which artifact stages 4-6 consume (one knob;
% both .in files always exist).  Report -> s3_report.txt.
%
% The M1 freeform rides along: SegMirMaker replicates the full parent
% surface (conic + the stage-1/2 Zernike figure) on every segment, so
% the segmented system reproduces the stage-2 wavefront wherever rays
% survive.

addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/design/src'));
P = e2e_params();
here  = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
parent = fullfile(here, 's2_instrument.in');
assert(isfile(parent), 's3 needs s2_instrument.in -- run s2_instrument.m first');

gap_m = P.seg.gap_mm * 1e-3;             % BaseUnits=m in this family
vs = P.seg.variants;
log_ = fopen(fullfile(here, 's3_report.txt'), 'w');
say = @(varargin) fprintf(1, varargin{:}) + fprintf(log_, varargin{:});
say('==== e2e stage 3: segmentation of M1 (parent s2_instrument.in) ====\n');
say('parent: D = %g m, BaseUnits=m; gap %g mm; segment size defaults to\n', ...
    P.D_m, P.seg.gap_mm);
say('Aperture/(2*rings+1) -- the tiling scales with the aperture\n\n');

% ---- monolithic parent baseline ------------------------------------
macos.init(P.seg.model_size);
macos.load_rx(parent);
t0 = macos.trace();
r0 = macos.get_ray_info(t0.nRays);
p0 = nnz(logical(r0.ok_pass) & logical(r0.ok_trace));
say('[0] monolithic parent: %d src rays, %d pass, rmsWFE %.4g (WaveUnits)\n\n', ...
    t0.nRays, p0, t0.rmsWFE);

S = struct();                            % per-variant results for the .mat
for iv = 1:numel(vs)
    v = char(vs(iv));
    switch v
        case 'pie',  rings = 1; grid = 'Pie';
        case 'hex2', rings = 2; grid = 'Hex';
        otherwise, error('unknown segmentation variant "%s"', v);
    end
    say('---- variant "%s": rings=%d, grid=%s ----\n', v, rings, grid);

    %% [1] bare segmentation + trace parity ---------------------------
    seg = macos.design.segment_rx(parent, 'elt', 1, ...
        'rings', rings, 'grid', grid, 'gap', gap_m, ...
        'dofs', P.seg.dofs, 'meas_config', P.seg.meas_config, ...
        'grid_npts', P.seg.grid_npts);
    say('[1] %d segments, width %.4g m flat-to-flat, gap %g m, %d elements\n', ...
        seg.nseg, seg.width, seg.gap, seg.n_elt);
    say('    M1 hole carried onto the center segment: %d obscuration(s)\n', ...
        seg.carried_obs);
    for s = 1:seg.nseg
        fr = seg.frames(s);
        say('      %-8s center [%+8.4f %+8.4f %+8.4f] m  lMon %.4g\n', ...
            fr.name, fr.rpt(1), fr.rpt(2), fr.rpt(3), fr.lmon);
    end
    macos.load_rx(seg.in);
    t1 = macos.trace();
    r1 = macos.get_ray_info(t1.nRays);
    p1 = nnz(logical(r1.ok_pass) & logical(r1.ok_trace));
    w1 = macos.opd();
    say('    bare segmented: %d src rays, %d pass, rmsWFE %.4g (parent %.4g, %+.2f%%)\n', ...
        t1.nRays, p1, t1.rmsWFE, t0.rmsWFE, 100*(t1.rmsWFE - t0.rmsWFE)/t0.rmsWFE);

    %% [2] traced footprints vs the tiling boundary -------------------
    N = size(w1, 1);
    labels = zeros(N);
    for s = 1:seg.nseg
        macos.load_rx(seg.in);
        macos.perturb(seg.seg_elts(s), 'translation', [0;0;1e-7], 'frame','local');
        macos.modify(); macos.trace();
        d = macos.opd() - w1;
        ok = isfinite(d) & (w1 ~= 0);
        dev = abs(d - median(d(ok)));
        labels(ok & dev > 0.25*max(dev(ok))) = s;
    end
    macos.load_rx(seg.in);
    B = macos.design.seg_boundary(seg);

    %% [3] physical polygonal apertures -------------------------------
    segA = macos.design.segment_rx(parent, 'elt', 1, ...
        'rings', rings, 'grid', grid, 'gap', gap_m, ...
        'dofs', P.seg.dofs, 'meas_config', P.seg.meas_config, ...
        'grid_npts', P.seg.grid_npts, ...
        'emit_apertures', P.seg.emit_apertures, 'ap_pad', P.seg.ap_pad);
    macos.load_rx(segA.in);
    tA = macos.trace();
    rA = macos.get_ray_info(tA.nRays);
    pA = nnz(logical(rA.ok_pass) & logical(rA.ok_trace));
    rs = macos.get_ray_status(tA.nRays);
    fe = rs.fail_elt(rs.status ~= 0);
    say('[3] physical apertures (pad %g): %d pass (bare %d) -- %d gap/rim rays clip;\n', ...
        P.seg.ap_pad, pA, p1, p1 - pA);
    say('    first-fail elements %s; rmsWFE %.4g (%+.2f%% vs parent)\n', ...
        mat2str(unique(fe(fe > 0)).'), tA.rmsWFE, ...
        100*(tA.rmsWFE - t0.rmsWFE)/t0.rmsWFE);

    % footprint + boundary + aperture-polygon figure
    [xm, ym, toimg] = pupil_axes_(labels, B, seg);
    f = figure('Visible','off', 'Position', [0 0 720 660]);
    imagesc(xm, ym, toimg(labels), 'AlphaData', 0.55*(toimg(labels) > 0));
    axis xy image; hold on
    colormap([0.94 0.94 0.94; lines(seg.nseg)]); clim([-0.5 seg.nseg+0.5]);
    for s = 1:seg.nseg
        plot(B.poly{s}(1,:), B.poly{s}(2,:), 'k-', 'LineWidth', 0.8);
        Pp = segA.apertures.poly{s}(:, [1:end 1]);
        plot(Pp(1,:), Pp(2,:), 'k-', 'LineWidth', 1.6);
        if ~isempty(segA.apertures.obs{s})
            O = segA.apertures.obs{s}(:, [1:end 1]);
            plot(O(1,:), O(2,:), 'r--', 'LineWidth', 1.0);
        end
    end
    title(sprintf('e2e s3 (%s): traced footprints + emitted apertures (%d segs)', ...
          v, seg.nseg));
    xlabel('pupil x, m'); ylabel('pupil y, m');
    print(f, fullfile(here, sprintf('s3_footprints_%s.png', v)), '-dpng', '-r120');
    close(f);
    say('    footprint/aperture figure: s3_footprints_%s.png\n', v);

    %% [4] views ------------------------------------------------------
    try
        fv = macos.view_std('args', {'show','beam'}, 'visible', false, ...
            'title', sprintf('e2e s3: segmented primary (%s, %d segments)', ...
                             v, seg.nseg), ...
            'save', fullfile(here, sprintf('s3_views_%s.png', v)));
        close(fv);
        say('[4] standard views: s3_views_%s.png\n', v);
    catch ME, say('[4] view_std skipped (%s)\n', ME.message); end

    %% [5] artifact + standalone reload -------------------------------
    art = fullfile(here, sprintf('e2e_%s.in', v));
    copyfile(segA.in, art);
    copyfile(segA.hx, fullfile(here, sprintf('e2e_%sHx.m', v)));
    macos.init(P.seg.model_size);
    nv = macos.load_rx(art);
    sv = macos.trace(nv);
    rv = macos.get_ray_info(sv.nRays);
    np = nnz(logical(rv.ok_pass) & logical(rv.ok_trace));
    say('[5] artifact e2e_%s.in: standalone reload %d elts, %d/%d rays pass -> %s\n\n', ...
        v, nv, np, sv.nRays, tern_(np == pA, 'VERIFIED', '** MISMATCH **'));

    S.(v) = struct('seg', seg, 'segA', segA, 'labels', labels, ...
                   'src', t1.nRays, ...
                   'pass_bare', p1, 'pass_ap', pA, 'pass_parent', p0, ...
                   'rms_parent', t0.rmsWFE, 'rms_bare', t1.rmsWFE, ...
                   'rms_ap', tA.rmsWFE, 'artifact', art);
end

say('==== stage-3 summary ====\n');
say('%-6s %5s %9s %10s %9s %12s\n', 'name','nseg','width_m','pass/src','clipped','rmsWFE d%');
for iv = 1:numel(vs)
    v = char(vs(iv)); q = S.(v);
    say('%-6s %5d %9.4g %6d/%-5d %8d %+11.2f\n', v, q.seg.nseg, q.seg.width, ...
        q.pass_ap, q.src, q.pass_bare - q.pass_ap, ...
        100*(q.rms_ap - q.rms_parent)/q.rms_parent);
end
say('(rmsWFE deltas vs parent = pupil-sampling coverage: the segment tilings\n');
say(' sample the pupil differently from the parent Circular grid)\n');
say('downstream variant (P.seg.variant) = "%s" -> stages 4-6 consume e2e_%s.in\n', ...
    P.seg.variant, P.seg.variant);
fclose(log_);
save(fullfile(here, 's3_segmentation.mat'), 'P', 'S', 'parent');
fprintf('\nStage 3 complete: e2e_pie.in + e2e_hex2.in + s3_report.txt + figures.\n');
fprintf('Next: s4_jacobians.m (dwdx / dwdz / dwdgrid on e2e_%s.in).\n', P.seg.variant);

% ---------------------------------------------------------------------
function s = tern_(c, a, b), if c, s = a; else, s = b; end, end

function [xmm, ymm, toimg] = pupil_axes_(labels, B, seg)
%PUPIL_AXES_  Calibrate the OPD pixel grid to global X/Y (parent units).
%   Same construction as e5_pie: fit the affine pixel->pupil map from
%   per-segment footprint centroids vs tiling-region centroids, so the
%   overlay is exact whatever the source grid orientation.
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
A = (Ppix \ Pmm).';
swp = abs(A(1,1)) + abs(A(2,2)) < abs(A(1,2)) + abs(A(2,1));
if swp
    Ppix(:, [1 2]) = Ppix(:, [2 1]);
    A = (Ppix \ Pmm).';
end
assert(all(isfinite(A(:))), 'pupil grid calibration failed');
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
