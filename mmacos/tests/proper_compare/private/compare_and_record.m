function metrics = compare_and_record(name, macos_int, proper_int, dx_m, opts)
%COMPARE_AND_RECORD  macos vs PROPER intensity comparison metrics.
%
%   Minimal port of pymacos
%   tests/proper_compare/conftest.py:compare_and_record — returns the
%   metrics struct without the plotting / .mat-export / report.md side
%   effects.  Tests assert on metrics.max_abs_aligned (or .max_abs).
%
%   Both inputs are assumed to be on the same physical sampling.
%   Strehl-normalises each side independently before differencing.
%
%   Returned struct fields:
%       .peak_m  .peak_p             intensity peaks
%       .sum_m   .sum_p              intensity sums
%       .dx_pix  .dy_pix             centroid offset macos→PROPER (pix)
%       .norm_kind                   'peak' or 'sum'
%       .max_abs                     max|a-b| (position-aware)
%       .rms_abs                     RMS|a-b|
%       .max_abs_aligned             after centroid alignment (shape only)
arguments
    name              (1,:) char
    macos_int         (:,:) double
    proper_int        (:,:) double
    dx_m              (1,1) double
    opts.norm_kind    (1,:) char {mustBeMember(opts.norm_kind, {'peak','sum'})} = 'peak'
    opts.align_centroid (1,1) logical = true
    opts.results_dir  (1,:) char = ''   % if non-empty, write <name>.png here
    opts.crop_pixels  (1,1) double = 64
    opts.log_scale    (1,1) logical = true
end
if ~isequal(size(macos_int), size(proper_int))
    error('compare_and_record:shapeMismatch', ...
        'shape mismatch %s vs %s', mat2str(size(macos_int)), ...
        mat2str(size(proper_int)));
end

peak_m = max(macos_int(:));
peak_p = max(proper_int(:));
sum_m  = sum(macos_int(:));
sum_p  = sum(proper_int(:));

switch opts.norm_kind
    case 'peak'
        a = macos_int  / max(peak_m, eps);
        b = proper_int / max(peak_p, eps);
    case 'sum'
        a = macos_int  / max(sum_m,  eps);
        b = proper_int / max(sum_p,  eps);
end

diff_ab  = a - b;
max_abs  = max(abs(diff_ab(:)));
rms_abs  = sqrt(mean(diff_ab(:).^2));

[mcy, mcx] = centroid_loc(macos_int);
[pcy, pcx] = centroid_loc(proper_int);
dy = round(pcy - mcy);
dx = round(pcx - mcx);

if opts.align_centroid
    a_aligned = circshift(a, [dy, dx]);
else
    a_aligned = a;
end
max_abs_aligned = max(abs(a_aligned(:) - b(:)));

metrics.name            = name;
metrics.dx_m            = dx_m;
metrics.peak_m          = peak_m;
metrics.peak_p          = peak_p;
metrics.sum_m           = sum_m;
metrics.sum_p           = sum_p;
metrics.dx_pix          = dx;
metrics.dy_pix          = dy;
metrics.norm_kind       = opts.norm_kind;
metrics.max_abs         = max_abs;
metrics.rms_abs         = rms_abs;
metrics.max_abs_aligned = max_abs_aligned;

% --- Optional 3-panel PNG (macos | PROPER | difference) -----------
if ~isempty(opts.results_dir)
    if ~exist(opts.results_dir, 'dir')
        mkdir(opts.results_dir);
    end
    n_full = size(macos_int, 1);
    nc = min(opts.crop_pixels, n_full);
    if mod(nc, 2) == 1, nc = nc - 1; end
    am = crop_center(macos_int,  nc);
    ap = crop_center(proper_int, nc);
    ad = crop_center(diff_ab,    nc);
    extent_um = (nc / 2) * dx_m * 1e6;

    fig = figure('Visible', 'off', 'Position', [100, 100, 1300, 450], ...
                 'Color', 'w');

    % Panel 1: macos intensity
    subplot(1, 3, 1);
    plot_panel(am, [-extent_um, extent_um], ...
               sprintf('macos INT  peak=%.3e', peak_m), ...
               'viridis', opts.log_scale, peak_m);

    % Panel 2: PROPER intensity
    subplot(1, 3, 2);
    plot_panel(ap, [-extent_um, extent_um], ...
               sprintf('PROPER  peak=%.3e', peak_p), ...
               'viridis', opts.log_scale, peak_p);

    % Panel 3: difference (diverging)
    subplot(1, 3, 3);
    dlim = max(abs(ad(:)));
    if dlim == 0, dlim = eps; end
    plot_panel(ad, [-extent_um, extent_um], ...
               sprintf('(macos - PROPER) %s-norm  max|.|=%.2e', ...
                       opts.norm_kind, max_abs), ...
               'redblue', false, dlim);

    sgtitle(strrep(name, '_', '\_'));
    png_path = fullfile(opts.results_dir, [name, '.png']);
    exportgraphics(fig, png_path, 'Resolution', 110);
    close(fig);
end
end


function plot_panel(data, ext_um, ttl, cmap_name, log_scale, max_val)
%PLOT_PANEL  Single imagesc panel with log/diverging coloring + colorbar.
if log_scale && max_val > 0
    % Log scale via log10; floor to peak * 1e-6 to avoid -inf.
    floor_val = max(1e-6 * max_val, eps);
    data_show = log10(max(data, floor_val));
    cmin = log10(floor_val);
    cmax = log10(max_val);
    imagesc([ext_um(1), ext_um(2)], [ext_um(1), ext_um(2)], data_show);
    clim([cmin, cmax]);
    cb = colorbar;
    cb.Label.String = 'log_{10}(intensity)';
elseif strcmp(cmap_name, 'redblue')
    imagesc([ext_um(1), ext_um(2)], [ext_um(1), ext_um(2)], data);
    clim([-max_val, max_val]);
    colorbar;
else
    imagesc([ext_um(1), ext_um(2)], [ext_um(1), ext_um(2)], data);
    colorbar;
end
axis xy image;
xlabel('um');
ylabel('um');
title(ttl, 'FontSize', 9, 'Interpreter', 'none');

if strcmp(cmap_name, 'viridis')
    colormap(gca, viridis_safe());
elseif strcmp(cmap_name, 'redblue')
    colormap(gca, redblue_safe());
end
end


function cmap = viridis_safe()
%VIRIDIS_SAFE  MATLAB's built-in 'parula' is colormap-compatible with
%   matplotlib's viridis intent (perceptually uniform).  parula is
%   always available; viridis was added in later releases.
try
    cmap = viridis(256);
catch
    cmap = parula(256);
end
end


function cmap = redblue_safe()
%REDBLUE_SAFE  Build a diverging red-white-blue colormap.
n = 128;
r = [linspace(0, 1, n)'; ones(n, 1)];
g = [linspace(0, 1, n)'; linspace(1, 0, n)'];
b = [ones(n, 1); linspace(1, 0, n)'];
cmap = [r, g, b];
end

