function out = ctb_e2e_figures(opts)
%CTB_E2E_FIGURES  Deck figures for the coronagraph on the telescope: views and a station walk.
%   out = CTB_E2E_FIGURES() reads the e2e thread's products (templates/
%   80_end_to_end/e2e: s8_fold_full.in, the telescope with the folded CTB
%   behind it; s8_fold_ctb.in, the folded bench alone; s9_coupled_dm.in,
%   the coupled diffraction deck with grid-data DMs; s9_run.mat, its
%   station map and Lyot pick) and writes two figures beside the deck
%   sources in this directory, each the tool's own output:
%
%   ctb_e2e_views.png   four views of the folded train, drawn by
%       macos.view_rx: the whole train from the side (elevation, z up:
%       light enters from below, the 4 m primary at z = 0, the secondary
%       6.3 m below it, the bench on the primary's back), the whole train
%       isometric, the same elevation zoomed on the bench behind the
%       primary, and the folded bench alone with its elements labelled.
%       The end-on shroud fit is stage 8's own s8_shroud.png (copied
%       beside this figure as ctb_e2e_shroud.png).
%
%   ctb_e2e_walk.png    the field at eight planes through the telescope
%       and the coronagraph, static (flat mirrors, the annular Lyot of
%       stage 9's pick, N=512, the deck's wavelength): the primary M1, DM1,
%       DM2, the apodizer, the focal-plane mask (after the occulter), the
%       Lyot stop (after the stop), the exit pupil and the camera.  Row 1
%       log10 amplitude; row 2 phase -- at the pupil planes from DM1 on the
%       ray-traced wavefront (macos.opd at that element, waves, the
%       chief-ray reference, piston and tilt removed: the beam's physical
%       phase, free of the tilt and curvature of the propagation frame that
%       the field's own phase carries on the grid), at the focal planes the
%       propagated field's phase.  The primary shows its traced pupil only
%       (at a curved mirror the surface path difference is the sag), and
%       the telescope's foci are ray-traced planes in this deck (the
%       diffraction chain starts at DM1) and are not shown.  The dark-zone contrast read from
%       the last plane is printed and returned so the walk can be checked
%       against s9_report.txt's static number.
%
%   Run:  >> out = ctb_e2e_figures;      (headless; figures invisible)
%   See also: ctb_chain, ctb_vortex_stations, macos.view_rx, s9_diffraction.
    arguments
        opts.outdir  (1,:) char = ''
        opts.visible (1,1) logical = false
        opts.model_size (1,1) double = 512
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', '..', 'src'));
    e2e = fullfile(here, '..', '..', '80_end_to_end', 'e2e');
    if isempty(opts.outdir), opts.outdir = here; end
    vis = 'off';  if opts.visible, vis = 'on'; end
    full_in  = fullfile(e2e, 's8_fold_full.in');
    bench_in = fullfile(e2e, 's8_fold_ctb.in');
    dm_in    = fullfile(e2e, 's9_coupled_dm.in');
    for f = {full_in, bench_in, dm_in, fullfile(e2e, 's9_run.mat')}
        assert(isfile(f{1}), 'ctb_e2e_figures: %s is missing -- run the e2e stages 8 and 9 first', f{1});
    end
    S9 = load(fullfile(e2e, 's9_run.mat'));
    e  = S9.e;                                % the compact deck's station map
    lyot_inner = S9.S.coupled.lyot_pick;      % the blocker, pupil radii (0.08)

    % ---- figure 1: the views -------------------------------------------------
    prev = cd(e2e);  c0 = onCleanup(@() cd(prev));      % GridFile= resolves against the cwd
    macos.init(opts.model_size);
    fig = figure('Visible', vis, 'Color', 'w', 'Position', [40 40 1900 1300]);
    tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, 'The coronagraph testbed folded behind the telescope (e2e stage 8): macos.view_rx', ...
        'FontWeight', 'bold', 'Interpreter', 'none');
    macos.load_rx(full_in);
    ax = nexttile(tl);
    macos.view_rx('ax', ax, 'view', [0 0], 'labels', false, 'show', 'beam', ...
        'title', 'the whole train from the side (z up): light from below, the 4 m primary at z = 0, the secondary 6.3 m below, the bench on its back');
    ax = nexttile(tl);
    macos.view_rx('ax', ax, 'view', [-35 18], 'labels', false, 'show', 'beam', 'title', 'the whole train, isometric');
    ax = nexttile(tl);
    macos.view_rx('ax', ax, 'view', [0 0], 'labels', false, 'show', 'beam', ...
        'title', 'the same elevation, zoomed on the bench behind the primary');   % labels would land outside the zoomed limits
    xlim(ax, [-2.6 2.6]);  zlim(ax, [-0.4 1.0]);  ylim(ax, [-2.6 2.6]);
    macos.load_rx(bench_in);
    ax = nexttile(tl);
    macos.view_rx('ax', ax, 'view', [-35 18], 'labels', true, 'show', 'beam', ...
        'title', 'the folded bench alone: F0 at the feed, the accordion in the vertical plane');
    fp1 = fullfile(opts.outdir, 'ctb_e2e_views.png');
    exportgraphics(fig, fp1, 'Resolution', 150);  close(fig);
    fprintf('[e2e figures] wrote %s\n', fp1);
    copyfile(fullfile(e2e, 's8_shroud.png'), fullfile(opts.outdir, 'ctb_e2e_shroud.png'));
    fprintf('[e2e figures] copied s8_shroud.png -> ctb_e2e_shroud.png\n');

    % ---- figure 2: the station walk ------------------------------------------
    ch = ctb_chain('rx', dm_in, 'model_size', opts.model_size, 'elt', e, 'r_lyot_inner_frac', lyot_inner);
    names = {'M1', 'DM1', 'DM2', 'Apodizer', 'FPM', 'Lyot', 'ExitPupil', 'FPA'};
    idx   = [1, e.DM1, e.DM2, e.Apodizer, e.FPM, e.Lyot, e.ExitPupil, e.FPA];
    kind  = {'pupil', 'pupil', 'pupil', 'pupil', 'focal', 'pupil', 'pupil', 'focal'};
    what  = {'primary mirror', 'DM1, the stop image', 'DM2', 'apodizer pupil', ...
             'after the occulter', 'after the annular stop', 'exit pupil', 'camera'};
    % the ray-traced wavefront at each pupil plane after the primary (OPD, waves, the
    % chief reference, piston and tilt removed: the engine measures the path at the
    % element's own tilted surface, so a 5-degree fold carries ~1e3 waves of tilt).
    % The primary is left out: at a curved mirror the path difference at the surface
    % is the mirror's own sag, not a wavefront.
    addpath(e2e);  P = e2e_params();  lam_m = P.lambda_m;
    Wr = cell(size(idx));
    for k = find(strcmp(kind, 'pupil') & idx > 1)
        macos.trace(idx(k));
        W = macos.opd() / lam_m;  m = logical(macos.opd_mask());
        Wr{k} = remove_pt_(W, m);
    end
    E = walk_(ch, e, idx);
    Efpa = E{end};
    dz = ch.dz_mask(3, 15);
    contrast = mean(abs(Efpa(dz)).^2) / ch.peak_bare;
    fprintf('[e2e figures] static dark zone 3-15 lambda/D from the walk: %.4e (s9_report.txt: %.4e)\n', contrast, S9.S.coupled.contrast_lyot);

    fig = figure('Visible', vis, 'Color', 'w', 'Position', [40 40 2200 620]);
    tl = tiledlayout(fig, 2, numel(names), 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, sprintf(['The field through the telescope and the coronagraph, static: flat mirrors, annular Lyot ' ...
        '(blocker %.2f), N=%d -- log amplitude (top) and phase (bottom) at each plane'], lyot_inner, ch.N), 'FontWeight', 'bold');
    for k = 1:numel(names)
        A = abs(E{k});  pk = max(A(:));
        w = crop_w_(A, kind{k}, ch);
        ax = nexttile(tl, k);
        show_(ax, log10(max(A / pk, 1e-6)), w, ch, [-6 0], parula);
        title(ax, sprintf('%s (E%d)', names{k}, idx(k)), 'Interpreter', 'none');
        xlabel(ax, what{k});
        if k == 1, ylabel(ax, 'log_{10} |E| / max'); end
        ax = nexttile(tl, numel(names) + k);
        if k == 1
            axis(ax, 'off');
            text(ax, 0.5, 0.5, {'ray trace only:', 'the wavefront is read', 'from DM1 on'}, ...
                'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontSize', 9);
            ylabel(ax, 'phase: wavefront (waves) / field phase (rad)', 'Visible', 'on');
        elseif strcmp(kind{k}, 'pupil')
            W = Wr{k};  wrms = std(W(:), 'omitnan');  wmax = max(abs(W(:)), [], 'omitnan');
            show_opd_(ax, W, [-wmax wmax]);
            xlabel(ax, sprintf('ray-traced wavefront, %.4f waves rms', wrms));
        else
            sup = A >= 1e-3 * pk;
            ph = angle(E{k});  ph(~sup) = NaN;
            show_(ax, ph, w, ch, [-pi pi], hsv);
            xlabel(ax, 'field phase as propagated');
        end
    end
    fp2 = fullfile(opts.outdir, 'ctb_e2e_walk.png');
    exportgraphics(fig, fp2, 'Resolution', 150);  close(fig);
    fprintf('[e2e figures] wrote %s\n', fp2);
    out = struct('fig_views', fp1, 'fig_walk', fp2, 'contrast', contrast, 'contrast_s9', S9.S.coupled.contrast_lyot);
end

% ======================================================================
function E = walk_(ch, e, idx)
%WALK_  One masked pass, the complex field captured at each plane of idx.
    E = cell(1, numel(idx));
    macos.intensity(e.Apodizer);                           % propagate to the apodizer
    for k = find(idx < e.Apodizer)
        E{k} = macos.complex_field(idx(k), 'reset_trace', false);
    end
    E{idx == e.Apodizer} = macos.complex_field(e.Apodizer, 'reset_trace', false);
    macos.intensity(e.FPM, 'reset_trace', false);
    macos.apodize_complex(e.FPM, ch.masks.F);
    E{idx == e.FPM} = macos.complex_field(e.FPM, 'reset_trace', false);
    macos.intensity(e.Lyot, 'reset_trace', false);
    macos.apodize(e.Lyot, ch.masks.L);
    E{idx == e.Lyot} = macos.complex_field(e.Lyot, 'reset_trace', false);
    E{idx == e.ExitPupil} = macos.complex_field(e.ExitPupil, 'reset_trace', false);
    E{idx == e.FPA} = macos.complex_field(e.FPA, 'reset_trace', false);
end

function W = remove_pt_(W, m)
%REMOVE_PT_  Least-squares piston and tilt off the OPD over its mask; NaN outside it.
    [I, J] = ndgrid(1:size(W, 1), 1:size(W, 2));
    A = [ones(nnz(m), 1), I(m), J(m)];
    c = A \ W(m);
    W = W - (c(1) + c(2) * I + c(3) * J);
    W(~m) = NaN;
end

function show_opd_(ax, W, cl)
%SHOW_OPD_  The OPD cropped to its own support (the ray grid's centre is not the field grid's).
    [i, j] = find(~isnan(W));
    ix = max(1, min(i) - 4) : min(size(W, 1), max(i) + 4);
    jx = max(1, min(j) - 4) : min(size(W, 2), max(j) + 4);
    M = W(ix, jx);
    imagesc(ax, M.', 'AlphaData', ~isnan(M.'));
    axis(ax, 'image', 'xy');
    colormap(ax, parula);  clim(ax, cl);
    set(ax, 'XTick', [], 'YTick', [], 'Color', [1 1 1]);
end

function w = crop_w_(A, kind, ch)
%CROP_W_  Half-width of the crop: the beam's own support on a pupil plane, +/-18 lambda/D on a focal one.
    if strcmp(kind, 'focal')
        w = ceil(18 * ch.lamD_px);
    else
        [i, j] = find(A > 1e-3 * max(A(:)));
        w = ceil(1.1 * max([abs(i - ch.center_px); abs(j - ch.center_px)]));
        w = max(w, 8);
    end
end

function show_(ax, M, w, ch, cl, cmap)
    c = ch.center_px;
    ix = max(1, c - w) : min(size(M, 1), c + w);
    imagesc(ax, M(ix, ix).', 'AlphaData', ~isnan(M(ix, ix).'));
    axis(ax, 'image', 'xy');
    colormap(ax, cmap);  clim(ax, cl);
    set(ax, 'XTick', [], 'YTick', [], 'Color', [1 1 1]);
end
