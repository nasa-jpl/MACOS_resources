function out = ctb_vortex_bandwidth(opts)
%CTB_VORTEX_BANDWIDTH  EFC floor vs bandwidth, polarized vortex chain.
%   out = CTB_VORTEX_BANDWIDTH() maps the closed-loop dark-zone floor of
%   the polarized vortex chain (charge 4, Lyot 0.60, coated train,
%   unpolarized input) against bandwidth: monochromatic, then 5% with 3
%   colors, 10% with 5, 20% with 9 (Dave's rule: ~2.5% control-
%   wavelength spacing held constant, so wide bands are not
%   under-sampled between control wavelengths).
%
%   Cost control: ONE Jacobian is measured over the SUPERSET of control
%   wavelengths (the 9-color 20% grid covers every band), and each
%   bandwidth's EFC uses the subset of its per-lambda blocks -- the
%   sweep pays the poke bill once (~100 min) instead of per band.
%   Saved as ctb_dm_jacobian_N512_phys_bwsweep.mat (gitignored).
%
%   Figure: floor vs bandwidth (log-log-ish), with the static floors
%   and the polarization floor for reference.
%
%   Run:  >> out = ctb_vortex_bandwidth;      (~2 hr total, engine)
%   See also: ctb_efc_physics, ctb_phys_summary.
    arguments
        opts.bands   (1,:) double = [0 0.05 0.10 0.20]
        opts.colors  (1,:) double = [1 3 5 9]
        opts.niter   (1,1) double = 12
        opts.outdir  (1,:) char = ''
        opts.visible (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', '..', 'src'));
    if isempty(opts.outdir), opts.outdir = here; end
    chain = {'fpm_kind','vortex','charge',4,'apodizer',false,'r_lyot_frac',0.60};

    % control-wavelength sets per band (uniform grids), and their superset
    assert(numel(opts.colors) == numel(opts.bands));
    sets = cell(1, numel(opts.bands));
    for b = 1:numel(opts.bands)
        if opts.bands(b) == 0 || opts.colors(b) == 1
            sets{b} = 1.0;
        else
            sets{b} = 1 + opts.bands(b) * linspace(-0.5, 0.5, opts.colors(b));
        end
    end
    super = unique(round([sets{:}], 6));

    % ---- the one Jacobian, superset wavelengths ------------------------
    jp = fullfile(here, 'ctb_dm_jacobian_N512_phys_bwsweep.mat');
    if isfile(jp)
        JJ = load(jp);
    else
        JJ = ctb_efc_physics('band', true, 'lfracs', super, ...
            'chain', chain, 'tag', 'bwsweep', 'jac_only', true);
    end
    assert(isequal(JJ.lfracs(:).', super(:).'), ...
        'ctb_vortex_bandwidth: cached sweep Jacobian has different wavelengths');

    % ---- per-band EFC on the subset blocks -----------------------------
    floors = zeros(1, numel(opts.bands));
    statics = zeros(1, numel(opts.bands));
    polfl  = zeros(1, numel(opts.bands));
    for b = 1:numel(opts.bands)
        lf = sets{b};
        [tf, loc] = ismember(round(lf, 6), round(super, 6));
        assert(all(tf));
        Jb = JJ;
        keep = [];
        newoff = 0;
        for k = 1:numel(loc)
            l = loc(k);
            rows = JJ.rowoff(l)+1 : JJ.rowoff(l+1);
            keep = [keep rows];                                %#ok<AGROW>
            newoff(end+1) = newoff(end) + numel(rows);         %#ok<AGROW>
        end
        Jb.G = JJ.G(keep, :);
        Jb.rowoff = newoff;
        Jb.lfracs = lf;
        tag = sprintf('bw%02d', round(100*opts.bands(b)));
        o = ctb_efc_physics('band', numel(lf) > 1, 'lfracs', lf, ...
            'pol', true, 'chain', chain, 'jac', Jb, 'niter', opts.niter, ...
            'tag', tag);
        floors(b) = o.c_after;  statics(b) = o.c_before;  polfl(b) = o.pol_floor;
        fprintf('[bw] band %2.0f%% (%d colors): static %.3e -> floor %.3e (pol floor %.2e)\n', ...
            100*opts.bands(b), numel(lf), o.c_before, o.c_after, o.pol_floor);
    end

    % ---- figure --------------------------------------------------------
    vis = 'off'; if opts.visible, vis = 'on'; end
    fig = figure('Visible',vis, 'Color','w', 'Position',[80 80 760 520]);
    ax = axes(fig);  hold(ax,'on');  grid(ax,'on');
    bw = 100 * opts.bands;
    semilogy(ax, bw, statics, 's--', 'LineWidth', 1.2, ...
        'DisplayName', 'static (pre-control)');
    semilogy(ax, bw, floors, 'o-', 'LineWidth', 1.6, 'MarkerSize', 7, ...
        'DisplayName', 'closed-loop floor');
    semilogy(ax, bw, polfl, '^:', 'LineWidth', 1.0, ...
        'DisplayName', 'polarization floor (uncontrollable)');
    set(ax, 'YScale','log');
    xlabel(ax, 'bandwidth (%)');
    ylabel(ax, 'dark-zone mean contrast (3-15 \lambda_0/D)');
    for b = 1:numel(bw)
        text(ax, bw(b), floors(b)*0.55, sprintf('%d colors', opts.colors(b)), ...
            'FontSize', 8, 'HorizontalAlignment','center', 'Color',[.35 .35 .35]);
    end
    title(ax, ['Polarized vortex chain: performance vs bandwidth -- ' ...
        'charge 4, Lyot 0.60, 2.5% control-wavelength spacing'], 'FontWeight','bold');
    legend(ax, 'Location', 'east');
    fp = fullfile(opts.outdir, 'ctb_vortex_bandwidth.png');
    exportgraphics(fig, fp, 'Resolution', 150);
    close(fig);
    fprintf('[bw] wrote %s\n', fp);
    out = struct('bands', opts.bands, 'floors', floors, 'statics', statics, ...
        'pol_floors', polfl, 'figure', fp);
    save(fullfile(opts.outdir, 'ctb_vortex_bandwidth.mat'), '-struct', 'out');
end
