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
%   Name-value 'chain' (ctb_chain config cell) and 'tag' generalize the
%   sweep to other mask configs: the default tag 'bwsweep' reproduces
%   the shipped file names exactly; any other tag suffixes every cache
%   and output so configs never collide.  Cached Jacobians are verified
%   against their stored chain_opts stamp (ctb_jac_check).
%
%   Run:  >> out = ctb_vortex_bandwidth;      (~2 hr total, engine)
%   See also: ctb_efc_physics, ctb_phys_summary, ctb_study.
    arguments
        opts.bands   (1,:) double = [0 0.05 0.10 0.20]
        opts.colors  (1,:) double = [1 3 5 9]
        opts.niter   (1,1) double = 12
        opts.chain   (1,:) cell = {'fpm_kind','vortex','charge',4, ...
                                   'apodizer',false,'r_lyot_frac',0.60}
        opts.tag     (1,:) char = 'bwsweep'  % non-default keeps its own
                                             % caches/outputs (_<tag>)
        opts.outdir  (1,:) char = ''
        opts.visible (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', '..', 'src'));
    if isempty(opts.outdir), opts.outdir = here; end
    chain = opts.chain;
    % default tag reproduces the shipped file names exactly; a study tag
    % suffixes every cache/output so configs never collide
    dflt = strcmp(opts.tag, 'bwsweep');
    osfx = '';  if ~dflt, osfx = ['_' opts.tag]; end

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
    jp = fullfile(here, sprintf('ctb_dm_jacobian_N512_phys_%s.mat', opts.tag));
    if isfile(jp)
        JJ = load(jp);
        ctb_jac_check(JJ, chain, jp);
    else
        JJ = ctb_efc_physics('band', true, 'lfracs', super, ...
            'chain', chain, 'tag', opts.tag, 'jac_only', true);
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
        if ~dflt, tag = sprintf('%s_bw%02d', opts.tag, round(100*opts.bands(b))); end
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
        text(ax, bw(b), floors(b)*0.55, ...
            sprintf('%d color%s', opts.colors(b), repmat('s', 1, opts.colors(b)>1)), ...
            'FontSize', 8, 'HorizontalAlignment','center', 'Color',[.35 .35 .35]);
    end
    cv = @(k, d) chaincfg_(chain, k, d);
    title(ax, {'Polarized vortex chain: performance vs bandwidth', ...
        sprintf('charge %d, Lyot %.2f, 2.5%% control-wavelength spacing', ...
        cv('charge', 4), cv('r_lyot_frac', 0.50))}, 'FontWeight','bold');
    legend(ax, 'Location', 'east');
    fp = fullfile(opts.outdir, ['ctb_vortex_bandwidth' osfx '.png']);
    exportgraphics(fig, fp, 'Resolution', 150);
    close(fig);
    fprintf('[bw] wrote %s\n', fp);
    out = struct('bands', opts.bands, 'floors', floors, 'statics', statics, ...
        'pol_floors', polfl, 'figure', fp);
    save(fullfile(opts.outdir, ['ctb_vortex_bandwidth' osfx '.mat']), '-struct', 'out');
end

function v = chaincfg_(chain, key, dflt)
    v = dflt;
    for i = 1:2:numel(chain)-1
        if strcmp(chain{i}, key), v = chain{i+1}; end
    end
end
