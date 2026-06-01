% verify.m -- MATLAB sanity check for the multi-field .mat outputs
% in this directory.  Loads dwdzall_*.mat and dwdxall_*.mat (if
% present), reconstructs the tiled OPDall via v2m / indxall, and
% emits per-channel sensitivity PNGs (one figure per ~16 channels).
%
% Helpers (m2v, v2m, mimg, pad) live in the parent sensitivities/
% directory -- addpath'd at the top so this script can run standalone.
%
% Usage:
%   cd /path/to/this/example/dir
%   matlab -batch "verify"          % batch, prints PNGs and exits
%   matlab -r "verify; keyboard"    % interactive, figures stay open

format compact

here = fileparts(mfilename('fullpath'));
if isempty(here)
    here = pwd;
end
parent_sens = fileparts(here);
addpath(parent_sens);

mats = dir(fullfile(here, '*all_*.mat'));
if isempty(mats)
    fprintf('no *all_*.mat files in %s\n', here);
    return
end

for k = 1:length(mats)
    matpath = fullfile(mats(k).folder, mats(k).name);
    fprintf('\n=== %s ===\n', mats(k).name);
    S = load(matpath);

    if ~isfield(S, 'dwdxall')
        fprintf('  skipping (no dwdxall field)\n');
        continue;
    end

    [nray, nchan] = size(S.dwdxall);
    nfields = size(S.field_table, 1);
    fprintf('  dwdxall: %d rays x %d channels\n', nray, nchan);
    fprintf('  fields (%d): ', nfields);
    for fi = 1:nfields
        fprintf('%s ', char(S.field_names{fi}));
    end
    fprintf('\n');
    fprintf('  ChfRayDir_nom = [%g %g %g]\n', S.chfraydir_nom);
    fprintf('  rx = %s, model_size=%d, delta=%g\n', ...
        S.rx, S.model_size, S.delta);

    % --- Tile OPDall round-trip ---
    OPDall = v2m(S.w0_stacked, S.indxall);
    f1 = figure('Visible', 'off', 'Position', [100 100 800 800]);
    mimg(OPDall, -1);
    title(sprintf('OPDall (m2v round-trip) - %s', mats(k).name), ...
        'interpreter', 'none');
    out_png = fullfile(here, ...
        ['verify_OPDall_' replace(mats(k).name, '.mat', '.png')]);
    print(f1, '-dpng', out_png);
    fprintf('  wrote %s\n', out_png);
    close(f1);

    % --- dwdxall <-> dwdzall alias check (if present) ---
    if isfield(S, 'dwdzall')
        max_diff = max(abs(S.dwdxall(:) - S.dwdzall(:)));
        fprintf('  alias check: max|dwdxall - dwdzall| = %.3e\n', max_diff);
    end

    % --- Per-channel sensitivity tiles ---
    nrow = 4; ncol = 4; per_fig = nrow * ncol;
    n_figs = ceil(nchan / per_fig);
    for fi = 1:n_figs
        fig_start = (fi - 1) * per_fig + 1;
        fig_end = min(fig_start + per_fig - 1, nchan);
        fig = figure('Visible', 'off', 'Position', [100 100 1400 1100]);
        for sub_k = 1:(fig_end - fig_start + 1)
            ii = fig_start + sub_k - 1;
            subplot(nrow, ncol, sub_k);
            dwcol = v2m(S.dwdxall(:, ii), S.indxall);
            mimg(dwcol, -1);
            ttl = char(S.channel_names{ii});
            title(ttl, 'interpreter', 'none', 'fontsize', 8);
        end
        out_png = fullfile(here, sprintf( ...
            'verify_sens_%s_fig%02d.png', ...
            replace(mats(k).name, '.mat', ''), fi));
        print(fig, '-dpng', out_png);
        fprintf('  wrote %s (chans %d-%d)\n', out_png, fig_start, fig_end);
        close(fig);
    end
end

fprintf('\nverify done.\n');
