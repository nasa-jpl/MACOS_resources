function [fig, xun, yun, toimg] = seg_footprint_view(labels, B, seg, opts)
%SEG_FOOTPRINT_VIEW  Footprint map + tiling/aperture overlay figure.
%   [FIG, XUN, YUN, TOIMG] = macos.design.seg_footprint_view(LABELS,
%   B, SEG) renders the per-segment footprint map (LABELS from
%   macos.design.seg_footprints) in PUPIL coordinates, with the tiling
%   boundary (B from macos.design.seg_boundary) and optionally the
%   emitted aperture polygons overlaid.  Shared by the e5_pie and
%   e2e-s3 runners -- the figure logic lives here, the runners stay
%   thin narratives (Dave: general-purpose runners).
%
%   Pixel-to-pupil calibration is fit from the footprints themselves:
%   an affine pixel->pupil map from per-segment footprint centroids
%   (pixel space) vs tiling-region centroids (boundary polygons), so
%   the overlay is exact whatever the source grid orientation
%   (xGrid/yGrid may be mirrored or flipped wrt the global axes).  The
%   engine OPD grid stores x along the ROW index; if the fit says the
%   map is transposed, the pixel axes are swapped and refit.  Warns if
%   the pupil grid is rotated >5% wrt global X/Y.
%
%   Returns the figure handle plus the calibrated ascending axes XUN /
%   YUN and TOIMG, the flip that puts any N x N pixel matrix into that
%   axis convention -- so callers can annotate (text, extra layers)
%   before printing.
%
%   Options:
%     'apertures'  seg_apertures output: overlays ap.poly (black,
%                  heavy) + ap.obs (red dashed) ([] = none)
%     'boundary'   overlay the B tiling polygons (default true)
%     'alpha'      footprint fill alpha (default 1)
%     'units'      axis-label unit string (default 'mm')
%     'title'      figure title ('' = none)
%     'save'       PNG path ('' = caller prints); figure stays open
%     'visible'    (default false)
%
%   See also: macos.design.seg_footprints, macos.design.seg_boundary,
%             macos.design.seg_apertures.
arguments
    labels (:,:) double
    B (1,1) struct
    seg (1,1) struct
    opts.apertures = []
    opts.boundary (1,1) logical = true
    opts.alpha (1,1) double = 1
    opts.units (1,:) char = 'mm'
    opts.title (1,:) char = ''
    opts.save  (1,:) char = ''
    opts.visible (1,1) logical = false
end

% ---- affine pixel->pupil calibration -----------------------------------
N = size(labels, 1);
[JJ, II] = meshgrid(1:N, 1:N);
Ppix = zeros(0, 3);  Pun = zeros(0, 2);
for s = 1:seg.nseg
    m = labels == s;
    if ~any(m, 'all'), continue; end
    shp = polyshape(B.poly{s}(1, 1:end-1), B.poly{s}(2, 1:end-1));
    [cx, cy] = centroid(shp);
    Ppix(end+1, :) = [mean(JJ(m)), mean(II(m)), 1]; %#ok<AGROW>
    Pun(end+1, :)  = [cx, cy];                      %#ok<AGROW>
end
A = (Ppix \ Pun).';                    % 2x3: [X;Y] = A*[col;row;1]
swp = abs(A(1,1)) + abs(A(2,2)) < abs(A(1,2)) + abs(A(2,1));
if swp
    Ppix(:, [1 2]) = Ppix(:, [2 1]);
    A = (Ppix \ Pun).';
end
offd = max(abs([A(1,2) A(2,1)])) / max(abs([A(1,1) A(2,2)]));
if offd > 0.05
    warning('macos:design:seg_footprint_view:rotated', ...
        'pupil grid rotated %.2f wrt global X/Y; overlay approximate', offd);
end
assert(all(isfinite(A(:))), 'pupil grid calibration failed');
xun = A(1,1)*(1:N) + A(1,3);
yun = A(2,2)*(1:N) + A(2,3);
fx = xun(1) > xun(end);  fy = yun(1) > yun(end);
if fx, xun = flip(xun); end
if fy, yun = flip(yun); end
toimg = @(M) flip_(flip_(tr_(M, swp), fx, 2), fy, 1);

% ---- figure ------------------------------------------------------------
vis = 'off';  if opts.visible, vis = 'on'; end
fig = figure('Visible', vis, 'Position', [0 0 720 660]);
imagesc(xun, yun, toimg(labels), 'AlphaData', opts.alpha*(toimg(labels) > 0));
axis xy image; hold on
colormap([0.94 0.94 0.94; lines(seg.nseg)]); clim([-0.5 seg.nseg+0.5]);
if opts.boundary
    for s = 1:seg.nseg
        plot(B.poly{s}(1,:), B.poly{s}(2,:), 'k-', 'LineWidth', ...
             tern_(isempty(opts.apertures), 1.4, 0.8));
    end
end
if ~isempty(opts.apertures)
    ap = opts.apertures;
    for s = 1:seg.nseg
        P = ap.poly{s}(:, [1:end 1]);
        plot(P(1,:), P(2,:), 'k-', 'LineWidth', 1.6);
        if ~isempty(ap.obs{s})
            O = ap.obs{s}(:, [1:end 1]);
            plot(O(1,:), O(2,:), 'r--', 'LineWidth', 1.0);
        end
    end
end
% 'Interpreter','none': these titles carry artifact stems, and TeX turns
% an underscore into a subscript -- "s2_segmented" renders as "s2" with a
% subscripted "s".  Committed figures end up on slides.
if ~isempty(opts.title), title(opts.title, 'Interpreter', 'none'); end
xlabel(sprintf('pupil x, %s', opts.units));
ylabel(sprintf('pupil y, %s', opts.units));
if ~isempty(opts.save), print(fig, opts.save, '-dpng', '-r120'); end
end

% ---------------------------------------------------------------------------
function M = flip_(M, tf, dim)
if tf, M = flip(M, dim); end
end

function M = tr_(M, tf)
if tf, M = M.'; end
end

function v = tern_(c, a, b)
if c, v = a; else, v = b; end
end
