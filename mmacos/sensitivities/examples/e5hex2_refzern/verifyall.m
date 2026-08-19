% verifyall.m -- MATLAB sanity check + per-channel sensitivity
% display for the multi-field .mat outputs in this directory.
%
% Layout, per Dave's spec (2026-05-31):
%   dwdxall:  one row PER ELEMENT, 6 columns (Rx, Ry, Rz, Tx, Ty, Tz).
%             N_elt elements -> N_elt × 6 subplot grid in ONE figure.
%             Each subplot tiles the multi-field sensitivity via the
%             OPDall canvas (v2m round-trip on indxall).
%   dwdzall:  one figure PER ELEMENT, 6×7 grid of 42 Zernike modes.
%             Subplot title shows the mode index.
%
% Display helpers (macos.m2v/v2m/mimg/pad) are +macos package
% functions -- run mmacos_setup once per session.
%
% Figures stay open in interactive sessions.  In batch mode
% (`matlab -batch verifyall`) MATLAB is headless; PNGs are written
% either way.
%
% Usage:
%   cd /path/to/this/example/dir
%   matlab                            % interactive
%       >> verifyall
%   matlab -batch "verifyall"         % batch + PNGs

format compact

here = fileparts(mfilename('fullpath'));
if isempty(here)
    here = pwd;
end
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

    % --- Tile OPDall round-trip --------------------------------------
    OPDall = macos.v2m(S.w0_stacked, S.indxall);
    f1 = figure('Position', [100 100 800 800]);
    macos.mimg(OPDall, -1);
    title(sprintf('OPDall (m2v round-trip) - %s', mats(k).name), ...
        'interpreter', 'none');
    out_png = fullfile(here, ...
        ['verify_OPDall_' replace(mats(k).name, '.mat', '.png')]);
    print(f1, '-dpng', out_png);
    fprintf('  wrote %s\n', out_png);

    % --- dwdxall <-> dwdzall alias check (Zernike runs) --------------
    if isfield(S, 'dwdzall')
        max_diff = max(abs(S.dwdxall(:) - S.dwdzall(:)));
        fprintf('  alias check: max|dwdxall - dwdzall| = %.3e\n', max_diff);
    end

    % --- Per-channel sensitivity tiles -------------------------------
    %   Two distinct layouts based on what the channel-name structure
    %   reveals.  Parsing rule: every channel name starts with "Elt N "
    %   then a kind-specific suffix.
    %     dwdx:  "Elt N <Rx|Ry|Rz|Tx|Ty|Tz>"
    %     dwdz:  "Elt N <MonZern|Zern|FFZern> <mode_idx>"
    [elt_ids, kinds, mode_or_dof] = parse_channels(S.channel_names);
    is_dwdx = all(ismember(kinds, {'Rx','Ry','Rz','Tx','Ty','Tz'}));
    is_dwdz = all(ismember(kinds, {'MonZern','Zern','FFZern'}));

    tag_clean = replace(replace(mats(k).name, '.mat', ''), '/', '_');

    if is_dwdx
        plot_dwdxall(S, elt_ids, kinds, mode_or_dof, here, tag_clean);
    elseif is_dwdz
        plot_dwdzall(S, elt_ids, kinds, mode_or_dof, here, tag_clean);
    else
        fprintf('  channel-name structure not recognized; ');
        fprintf('falling back to flat grid layout\n');
        plot_flat(S, here, tag_clean);
    end
end

fprintf('\nverifyall done.\n');


% =====================================================================
function [elt_ids, kinds, mode_or_dof] = parse_channels(channel_names)
%   For each channel name, extract:
%     elt_ids(k)     integer element id (NaN if not parseable)
%     kinds{k}       kind token ('Rx' / 'MonZern' / 'Zern' / ...)
%     mode_or_dof(k) trailing integer mode index (NaN for dwdx)
%
%   Two name formats produced by channels.py:
%     dwdx:  "Elt N <Dof>"            e.g. "Elt 1 Rx"
%     dwdz:  "Elt N <Kind><mode_int>" e.g. "Elt 1 MonZern4"
%
%   The dwdz form has NO SPACE between kind and mode; we split it
%   with a regex on the trailing digits.
    nchan = length(channel_names);
    elt_ids = nan(nchan, 1);
    kinds = cell(nchan, 1);
    mode_or_dof = nan(nchan, 1);
    DWDX_DOFS = {'Rx','Ry','Rz','Tx','Ty','Tz'};
    for ii = 1:nchan
        nm = strtrim(char(channel_names{ii}));
        toks = regexp(nm, '\s+', 'split');
        if length(toks) < 3 || ~strcmp(toks{1}, 'Elt')
            kinds{ii} = nm;
            continue;
        end
        elt_ids(ii) = str2double(toks{2});
        suffix = toks{3};
        if ismember(suffix, DWDX_DOFS)
            kinds{ii} = suffix;             % dwdx: kind = DOF
        else
            % dwdz: split "MonZern4" -> kind='MonZern', mode=4
            tok = regexp(suffix, '^([A-Za-z]+)(\d+)$', 'tokens', 'once');
            if isempty(tok)
                kinds{ii} = suffix;          % unknown form
            else
                kinds{ii} = tok{1};
                mode_or_dof(ii) = str2double(tok{2});
            end
        end
    end
end


% =====================================================================
function plot_dwdxall(S, elt_ids, kinds, ~, here, tag_clean)
%   N_elt rows × 6 columns layout.  Row = element (in element-id
%   order), Col = DOF (Rx, Ry, Rz, Tx, Ty, Tz).
    DOF_ORDER = {'Rx', 'Ry', 'Rz', 'Tx', 'Ty', 'Tz'};
    n_dof = numel(DOF_ORDER);
    unique_elts = unique(elt_ids(~isnan(elt_ids)));
    n_elt = numel(unique_elts);
    fprintf('  dwdx layout: %d elements × %d DOFs = %d subplots\n', ...
        n_elt, n_dof, n_elt * n_dof);

    fig = figure('Position', [50 50 ...
        min(120 * n_dof + 200, 1700) ...
        min(120 * n_elt + 100, 1100)]);
    nchan = size(S.dwdxall, 2);
    for ie = 1:n_elt
        for jd = 1:n_dof
            sub_k = (ie - 1) * n_dof + jd;
            subplot(n_elt, n_dof, sub_k);
            idx = find(elt_ids == unique_elts(ie) & ...
                       strcmp(kinds, DOF_ORDER{jd}), 1);
            if isempty(idx) || idx > nchan
                axis off;
                continue;
            end
            dwcol = macos.v2m(S.dwdxall(:, idx), S.indxall);
            macos.mimg(dwcol, -1);
            title(sprintf('Elt %d %s', unique_elts(ie), ...
                DOF_ORDER{jd}), 'interpreter', 'none', 'fontsize', 7);
        end
    end
    drawnow
    out_png = fullfile(here, ['verify_grid_' tag_clean '.png']);
    print(fig, '-dpng', out_png);
    fprintf('  wrote %s (%d × %d grid)\n', out_png, n_elt, n_dof);
end


% =====================================================================
function plot_dwdzall(S, elt_ids, kinds, modes, here, tag_clean)
%   Layout is chosen adaptively from the per-pair mode count:
%
%   - If every (element, kind) pair has <= 12 modes (typical
%     Z4..Z15 = 12 modes-per-element runs), use a COMPACT layout:
%     one figure total, rows = (element, kind) pairs in element
%     order, cols = mode index.  Multiple elements per page.
%
%   - Otherwise (e.g. Z4..Z45 = 42 modes-per-element runs), fall
%     back to SEPARATE layout: one figure per (element, kind) pair,
%     7 rows × 6 cols = 42 subplots.
    [pair_elts, pair_kinds] = collect_pairs(elt_ids, kinds);
    n_pairs = numel(pair_elts);
    pair_n_modes = zeros(n_pairs, 1);
    for pp = 1:n_pairs
        mask = (elt_ids == pair_elts(pp)) & ...
               strcmp(kinds, pair_kinds{pp});
        pair_n_modes(pp) = sum(mask);
    end
    max_modes = max(pair_n_modes);

    if max_modes <= 12
        plot_dwdzall_compact(S, elt_ids, kinds, modes, here, ...
            tag_clean, pair_elts, pair_kinds, max_modes);
    else
        plot_dwdzall_separate(S, elt_ids, kinds, modes, here, ...
            tag_clean, pair_elts, pair_kinds);
    end
end


% =====================================================================
function plot_dwdzall_compact(S, elt_ids, kinds, modes, here, ...
    tag_clean, pair_elts, pair_kinds, max_modes)
%   ONE figure: rows = (element, kind) pairs, cols = mode index.
%   Good for Z4..Z15 (12 modes-per-element) regime where every
%   element fits on the same page.
    n_pairs = numel(pair_elts);
    fprintf('  dwdz layout (compact): %d pairs × %d modes = ', ...
        n_pairs, max_modes);
    fprintf('%d subplots in 1 figure\n', n_pairs * max_modes);

    fig = figure('Position', [50 50 ...
        min(110 * max_modes + 250, 1800) ...
        min(110 * n_pairs + 150, 1100)]);
    for pp = 1:n_pairs
        ie = pair_elts(pp);
        kk = pair_kinds{pp};
        mask = (elt_ids == ie) & strcmp(kinds, kk);
        idxs = find(mask);
        ms = modes(mask);
        [ms_sorted, order] = sort(ms);
        idxs_sorted = idxs(order);
        for jj = 1:numel(idxs_sorted)
            sub_k = (pp - 1) * max_modes + jj;
            subplot(n_pairs, max_modes, sub_k);
            ii = idxs_sorted(jj);
            dwcol = macos.v2m(S.dwdxall(:, ii), S.indxall);
            macos.mimg(dwcol, -1);
            if pp == 1
                title(sprintf('Z%d', ms_sorted(jj)), ...
                    'interpreter', 'none', 'fontsize', 8);
            end
            if jj == 1
                ylabel(sprintf('Elt %d %s', ie, kk), ...
                    'interpreter', 'none', 'fontsize', 8);
            end
        end
    end
    drawnow
    out_png = fullfile(here, ['verify_grid_' tag_clean '.png']);
    print(fig, '-dpng', out_png);
    fprintf('  wrote %s\n', out_png);
end


% =====================================================================
function plot_dwdzall_separate(S, elt_ids, kinds, modes, here, ...
    tag_clean, pair_elts, pair_kinds)
%   ONE figure per (element, kind) pair.  Layout: 7 rows × 6 cols
%   = 42 subplots (fits Z4..Z45).  Used when the per-element mode
%   count is too large to fit alongside other elements on one page.
    N_ROWS = 7;  N_COLS = 6;  PER_FIG = N_ROWS * N_COLS;  % = 42
    n_pairs = numel(pair_elts);
    fprintf('  dwdz layout (separate): %d pair(s), %d × %d = %d', ...
        n_pairs, N_ROWS, N_COLS, PER_FIG);
    fprintf(' subplots per figure\n');
    for pp = 1:n_pairs
        ie = pair_elts(pp);
        kk = pair_kinds{pp};
        mask = (elt_ids == ie) & strcmp(kinds, kk);
        idxs = find(mask);
        ms = modes(mask);
        [ms_sorted, order] = sort(ms);
        idxs_sorted = idxs(order);

        fig = figure('Position', [50 50 1500 1700]);
        for jj = 1:numel(idxs_sorted)
            if jj > PER_FIG
                break;
            end
            ii = idxs_sorted(jj);
            subplot(N_ROWS, N_COLS, jj);
            dwcol = macos.v2m(S.dwdxall(:, ii), S.indxall);
            macos.mimg(dwcol, -1);
            title(sprintf('Z%d', ms_sorted(jj)), ...
                'interpreter', 'none', 'fontsize', 8);
        end
        sgtitle(sprintf('dwdz - Elt %d, %s (%d modes)', ie, kk, ...
            numel(idxs_sorted)), 'interpreter', 'none');
        drawnow
        out_png = fullfile(here, sprintf( ...
            'verify_grid_%s_Elt%d_%s.png', tag_clean, ie, kk));
        print(fig, '-dpng', out_png);
        fprintf('  wrote %s (%d modes)\n', out_png, numel(idxs_sorted));
    end
end


% =====================================================================
function [pair_elts, pair_kinds] = collect_pairs(elt_ids, kinds)
%   Distinct (elt_id, kind) pairs in first-seen order.
    seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    pair_elts = [];
    pair_kinds = {};
    for ii = 1:numel(elt_ids)
        if isnan(elt_ids(ii))
            continue;
        end
        key = sprintf('%d|%s', elt_ids(ii), kinds{ii});
        if isKey(seen, key)
            continue;
        end
        seen(key) = true;
        pair_elts(end+1, 1) = elt_ids(ii);  %#ok<AGROW>
        pair_kinds{end+1, 1} = kinds{ii};   %#ok<AGROW>
    end
end


% =====================================================================
function plot_flat(S, here, tag_clean)
%   Fallback: flat 4×4 grid (the original layout) when channel names
%   don't fit either the dwdx or dwdz schema.
    nchan = size(S.dwdxall, 2);
    nrow = 4; ncol = 4; per_fig = nrow * ncol;
    n_figs = ceil(nchan / per_fig);
    for fi = 1:n_figs
        fig_start = (fi - 1) * per_fig + 1;
        fig_end = min(fig_start + per_fig - 1, nchan);
        fig = figure('Position', [100 100 1400 1100]);
        for sub_k = 1:(fig_end - fig_start + 1)
            ii = fig_start + sub_k - 1;
            subplot(nrow, ncol, sub_k);
            dwcol = macos.v2m(S.dwdxall(:, ii), S.indxall);
            macos.mimg(dwcol, -1);
            ttl = char(S.channel_names{ii});
            title(ttl, 'interpreter', 'none', 'fontsize', 8);
        end
        drawnow
        out_png = fullfile(here, sprintf( ...
            'verify_flat_%s_fig%02d.png', tag_clean, fi));
        print(fig, '-dpng', out_png);
        fprintf('  wrote %s (chans %d-%d)\n', out_png, fig_start, fig_end);
    end
end
