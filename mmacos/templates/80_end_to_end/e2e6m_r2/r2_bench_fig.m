function r2_bench_fig()
%R2_BENCH_FIG  Layout renders at interpretable size (Dave item 3).
%
%   Two figures:
%     r2_train_iso.png    the full DM-bearing train, 3D render (context)
%     r2_back_plane.png   the back end as a 2D FOLD-PLANE ELEVATION --
%        a bench drawing.  A 6-deg near-normal-fold accordion is almost
%        collinear in 3D (measured: every 3D view renders a stick), so
%        the interpretable picture is the chief path and the named
%        optics projected into the plane the folds actually live in,
%        to scale, with the DM pocket readable.
%
%   The elevation is data-driven: element positions from the committed
%   bench deck, plane by SVD of the positions (no assumed frame).

    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(struct());

    macos.init(P.model);
    % ---- context render --------------------------------------------------
    rx = fullfile(P.outdir, 'r1_seg_full.in');
    macos.load_rx(rx);
    n = macos.num_elt();
    macos.view_std('front', [], 'back', [], ...
                   'iso', [-35 22], 'side', [], ...
                   'title', sprintf(['full train: segmented 6 m telescope + ' ...
                       'DM-bearing back end (%d elts)'], n), ...
                   'save', fullfile(P.outdir,'r2_train_iso.png'), ...
                   'visible', false);

    % ---- fold-plane elevation -------------------------------------------
    rxb = fullfile(P.outdir, 'r1_seg_back.in');
    macos.load_rx(rxb);
    nb = macos.num_elt();
    nm = regexp(fileread(rxb), '^\s*EltName=\s*(\S+)', 'tokens','lineanchors');
    nm = cellfun(@(c) c{1}, nm, 'UniformOutput', false);
    Pv = zeros(3, nb);
    for k = 1:nb, Pv(:,k) = macos.get_elt_vpt(k); end
    c0 = mean(Pv, 2);
    [U,~,~] = svd(Pv - c0, 'econ');
    e1 = U(:,1);  e2 = U(:,2);
    u = e1.' * (Pv - c0);  v = e2.' * (Pv - c0);
    if u(end) < u(1), u = -u; end          % light left-to-right

    f = figure('Visible','off','Color','w','Position',[40 40 1800 480]);
    ax = axes(f);  hold(ax,'on');
    plot(ax, u, v, '-', 'Color',[0.55 0.75 0.55], 'LineWidth', 1.2);
    for k = 1:nb
        cls = class_(nm{k});
        scatter(ax, u(k), v(k), 55, 'filled', ...
                'MarkerFaceColor', clr_(cls), 'MarkerEdgeColor',[0.25 0.25 0.25]);
        side = 1 - 2*(v(k) < median(v));
        text(ax, u(k), v(k) + side*0.075, nm{k}, ...
             'HorizontalAlignment','center', 'FontSize', 8.5, ...
             'FontWeight','bold', 'Interpreter','none');
    end
    axis(ax,'equal');  grid(ax,'on');  box(ax,'on');
    ylim(ax, [min(v)-0.28, max(v)+0.28]);
    xlim(ax, [min(u)-0.25, max(u)+0.25]);
    xlabel(ax,'m along the bench (fold plane)');
    ylabel(ax,'m across');
    title(ax, sprintf(['the back end in its own fold plane, to scale: OAP1 ' ...
        'collimator, DM1/DM2 pocket, 7-OAP mask relay (%d elts; 6\\circ ' ...
        'same-side folds)'], nb));
    png = fullfile(P.outdir,'r2_back_plane.png');
    exportgraphics(f, png, 'Resolution', 160);
    close(f);
    d = dir(png);
    fprintf('r2_bench_fig: elevation %d elts, span %.2f x %.2f m -> %s (%d bytes)\n', ...
            nb, max(u)-min(u), max(v)-min(v), png, d.bytes);
end

function c = class_(name)
    if any(strcmp(name, {'DM1','DM2'})),                     c = 'dm';
    elseif any(strcmp(name, {'Apodizer','Lyot','Backend'})), c = 'mask';
    elseif any(strcmp(name, {'FPM','FieldStop'})),           c = 'focus';
    elseif any(strcmp(name, {'Science'})),                   c = 'det';
    else,                                                    c = 'mirror';
    end
end

function c = clr_(kind)
    switch kind
        case 'mirror', c = [0.55 0.70 0.90];
        case 'dm',     c = [0.98 0.70 0.30];
        case 'mask',   c = [0.75 0.60 0.90];
        case 'focus',  c = [1.00 1.00 1.00];
        case 'det',    c = [0.35 0.35 0.40];
    end
end
