function out = ctb_vortex_stations(opts)
%CTB_VORTEX_STATIONS  Complex field at the 7 stations, with/without polarization.
%   out = CTB_VORTEX_STATIONS() walks the vortex chain (charge 4, Lyot
%   0.60, no apodizer) through the seven key planes of the slide-2
%   convention -- DM1, DM2, Apodizer, FPM, Lyot, ExitPupil, FPA -- and
%   records the COMPLEX field at each, twice: the scalar chain, and the
%   co-polarized (Jxx-screened) chain of the coated train.  Two figures:
%
%   ctb_vortex_stations.png -- rows x stations grid:
%     row 1  log10 amplitude, scalar chain
%     row 2  phase, scalar chain (the vortex spiral at the focus planes)
%     row 3  log10 |E_pol - E_scalar| / max|E_scalar| -- what the coated
%            mirrors DO to the propagating field (per station)
%     row 4  log10 cross-polarized amplitude |E_yx| / max|E_scalar| --
%            the light the coatings move into the orthogonal state
%
%   ctb_pol_maps.png -- what the mirrors do at the pupil (macos.pol_maps
%   on the coated-train Jones pupil): diattenuation map, retardance map,
%   cross-pol amplitude |Jyx|/|Jxx|, and the co-pol differential phase
%   arg(Jyy/Jxx).
%
%   Static chain (flat DMs), monochromatic, N=512 -- matching the
%   slide-2 static convention.  Requires the cached Jones screens
%   (ctb_pol_screens.mat; recomputed by ctb_efc_physics('pol',true) if
%   absent -- run that first).
%
%   Run:  >> out = ctb_vortex_stations;
%   See also: ctb_chain, ctb_efc_physics, macos.pol_maps, ctb_coro_compare.
    arguments
        opts.outdir  (1,:) char = ''
        opts.visible (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', '..', 'src'));
    if isempty(opts.outdir), opts.outdir = here; end

    r  = ctb_dm_rx();
    ch = ctb_chain('rx', r.rx_out, 'model_size', 512, ...
        'fpm_kind','vortex', 'charge',4, 'apodizer',false, 'r_lyot_frac',0.60);
    e = ch.elt;  N = ch.N;
    sp = fullfile(here, 'ctb_pol_screens.mat');
    assert(isfile(sp), ...
        'ctb_vortex_stations: run ctb_efc_physics(''pol'',true) once to cache the screens');
    SC = load(sp);
    Jxx = SC.J(:,:,1,1);  Jyx = SC.J(:,:,2,1);

    names = {'DM1','DM2','Apodizer','FPM','Lyot','ExitPupil','FPA'};
    kind  = {'pupil','pupil','pupil','focal','pupil','pupil','focal'};

    Es = walk_(ch, e, names, []);         % scalar
    Ep = walk_(ch, e, names, Jxx);        % co-polarized, coated train
    Ex = walk_(ch, e, names, Jyx);        % cross-polarized chain

    % ---- figure 1: the station grid ------------------------------------
    vis = 'off'; if opts.visible, vis = 'on'; end
    fig = figure('Visible',vis, 'Color','w', 'Position',[40 40 1960 1150]);
    tl = tiledlayout(fig, 4, 7, 'TileSpacing','compact', 'Padding','compact');
    title(tl, ['Vortex chain (charge 4, Lyot 0.60), station by station -- ' ...
        'scalar vs the coated train (static, monochromatic, N=512)'], ...
        'FontWeight','bold');
    for k = 1:7
        pk = max(abs(Es{k}(:)));
        w = crop_w_(kind{k}, ch);
        % row 1: scalar amplitude
        ax = nexttile(tl, k);
        show_(ax, log10(max(abs(Es{k})/pk, 1e-8)), w, ch, [-8 0], parula);
        title(ax, names{k}, 'Interpreter','none');
        if k == 1, ylabel(ax, 'log_{10}|E| scalar'); end
        % row 2: scalar phase
        ax = nexttile(tl, 7 + k);
        ph = angle(Es{k});  ph(abs(Es{k}) < 1e-6*pk) = NaN;
        show_(ax, ph, w, ch, [-pi pi], hsv);
        if k == 1, ylabel(ax, 'phase, scalar'); end
        % rows 3-4 exist only from the screen plane on: the Jones
        % screens apply at the Apodizer, so there is no polarized or
        % cross-polarized field upstream -- blank DM1/DM2 rather than
        % show the unscreened beam there
        blank = k <= 2;
        % row 3: the coating effect on the co-pol field
        ax = nexttile(tl, 14 + k);
        M3 = log10(max(abs(Ep{k} - Es{k})/pk, 1e-10));
        if blank, M3 = NaN(size(M3)); end
        show_(ax, M3, w, ch, [-10 -2], parula);
        if blank, text(ax, 0.5, 0.5, 'upstream of screens', 'Units','normalized', ...
                'HorizontalAlignment','center', 'FontSize', 8, 'Color',[.6 .6 .6]); end
        if k == 1, ylabel(ax, 'log_{10}|E_{pol}-E_{scal}|'); end
        % row 4: cross-polarized amplitude
        ax = nexttile(tl, 21 + k);
        M4 = log10(max(abs(Ex{k})/pk, 1e-10));
        if blank, M4 = NaN(size(M4)); end
        show_(ax, M4, w, ch, [-10 -2], parula);
        if blank, text(ax, 0.5, 0.5, 'upstream of screens', 'Units','normalized', ...
                'HorizontalAlignment','center', 'FontSize', 8, 'Color',[.6 .6 .6]); end
        if k == 1, ylabel(ax, 'log_{10}|E_{yx}|'); end
    end
    cb = colorbar; cb.Layout.Tile = 'east';
    fp1 = fullfile(opts.outdir, 'ctb_vortex_stations.png');
    exportgraphics(fig, fp1, 'Resolution', 130);
    close(fig);
    fprintf('[stations] wrote %s\n', fp1);

    % ---- figure 2: what the mirrors do at the pupil --------------------
    pm = macos.pol_maps(struct('J', SC.J, 'mask', abs(SC.J(:,:,1,1)) > 0));
    fig = figure('Visible',vis, 'Color','w', 'Position',[60 60 1500 400]);
    tl = tiledlayout(fig, 1, 4, 'TileSpacing','compact', 'Padding','compact');
    title(tl, ['What the coated mirrors do (MgF_2/Al, all ten): the ' ...
        'Jones pupil of the train, ray-traced to the exit pupil'], ...
        'FontWeight','bold');
    panels = { pm.D,                         'diattenuation D';
               pm.ret,                       'retardance (rad)';
               abs(Jyx) ./ max(abs(Jxx), eps), 'cross-pol |J_{yx}/J_{xx}|';
               angle(SC.J(:,:,2,2) ./ Jxx),  'co-pol \Delta phase (rad)' };
    for k = 1:4
        ax = nexttile(tl);
        M = panels{k,1};  M(~pm.mask) = NaN;
        imagesc(ax, M.', 'AlphaData', ~isnan(M.'));
        axis(ax, 'image', 'xy');  set(ax, 'Color', [1 1 1]);
        colormap(ax, parula);  colorbar(ax);
        title(ax, panels{k,2});
        set(ax, 'XTick', [], 'YTick', []);
    end
    fp2 = fullfile(opts.outdir, 'ctb_pol_maps.png');
    exportgraphics(fig, fp2, 'Resolution', 150);
    close(fig);
    fprintf('[stations] wrote %s\n', fp2);
    out = struct('fig_stations', fp1, 'fig_polmaps', fp2, ...
        'ret_mean', pm.mean.ret, 'ret_rms', pm.var_rms.ret, ...
        'D_mean', pm.mean.D, 'D_rms', pm.var_rms.D);
end

% ======================================================================
function E = walk_(ch, e, names, S)
%WALK_  One masked pass, capturing the complex field at each station.
    E = cell(1, numel(names));
    macos.intensity(e.Apodizer);
    E{1} = macos.complex_field(e.DM1, 'reset_trace', false);
    E{2} = macos.complex_field(e.DM2, 'reset_trace', false);
    if ~isempty(S)
        macos.apodize_complex(e.Apodizer, S);
    end
    E{3} = macos.complex_field(e.Apodizer, 'reset_trace', false);
    macos.intensity(e.FPM, 'reset_trace', false);
    macos.apodize_complex(e.FPM, ch.masks.F);
    E{4} = macos.complex_field(e.FPM, 'reset_trace', false);
    macos.intensity(e.Lyot, 'reset_trace', false);
    macos.apodize(e.Lyot, ch.masks.L);
    E{5} = macos.complex_field(e.Lyot, 'reset_trace', false);
    E{6} = macos.complex_field(e.ExitPupil, 'reset_trace', false);
    E{7} = macos.complex_field(e.FPA, 'reset_trace', false);
end

function w = crop_w_(kind, ch)
    if strcmp(kind, 'focal')
        w = ceil(18 * ch.lamD_px);          % +/-18 lam/D
    else
        w = 160;                            % pupil footprint (+margin)
    end
end

function show_(ax, M, w, ch, cl, cmap)
    c = ch.center_px;
    ix = max(1, c-w) : min(size(M,1), c+w);
    imagesc(ax, M(ix, ix).', 'AlphaData', ~isnan(M(ix, ix).'));
    axis(ax, 'image', 'xy');
    colormap(ax, cmap);  clim(ax, cl);
    set(ax, 'XTick', [], 'YTick', [], 'Color', [1 1 1]);
end
