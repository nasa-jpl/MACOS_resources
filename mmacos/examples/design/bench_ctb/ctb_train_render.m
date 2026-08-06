function out = ctb_train_render(opts)
%CTB_TRAIN_RENDER  Deck-grade layout figure of the CTB, compact over full.
%   One figure, XZ projection (the fold plane of this planar bench -- the
%   beam runs along +X and folds in Z; the XZ view is the informative one,
%   YZ/XY are degenerate).  Compact model (ctb_dcr.in) stacked ABOVE the
%   full surface-to-surface model (ctb_s2s_dcr.in), SHARED axis scale + a
%   common scale bar, with the coronagraph mask planes and the exit-pupil
%   reference spheres marked.
%
%   Conventions (established figure rules, no external STYLE doc in tree):
%   white ground, real traced rays in muted green, element/station labels
%   placed OFF the ray lines, mask planes flagged with vertical ticks +
%   text, EP spheres marked at their vertices.  Headless-safe.
%
%   out = CTB_TRAIN_RENDER() writes ctb_train_render.png beside the decks.
%   Name-value: 'model_size','outdir','visible'.
    arguments
        opts.model_size (1,1) double = 512
        opts.outdir     (1,:) char   = ''
        opts.visible    (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    if isempty(opts.outdir), opts.outdir = here; end
    addpath(fullfile(here,'..','..','..','src'));
    assert(~isempty(getenv('MACOS_HOME')),'MACOS_HOME must be set.');

    M(1) = local_model_('compact', fullfile(here,'ctb_dcr.in'), ...
        struct('DM1',2,'DM2',5,'Apodizer',13,'FPM',17,'Lyot',20, ...
               'ExitPupil',30,'FPA',31), opts.model_size);
    M(2) = local_model_('full', fullfile(here,'ctb_s2s_dcr.in'), ...
        struct('DM1',2,'DM2',5,'Apodizer',16,'FPM',22,'Lyot',27, ...
               'ExitPupil',43,'FPA',44), opts.model_size);

    % shared limits = the PHYSICAL bench window (station vertices), NOT the
    % ray excursions: the diffraction EP-return spheres sit far off-bench
    % (x ~ -3700) and would otherwise compress the real train into a sliver.
    sx = []; sy = [];
    for k = 1:2
        fn = fieldnames(M(k).mark);
        for i = 1:numel(fn)
            sx(end+1) = M(k).mark.(fn{i})(1); %#ok<AGROW>
            sy(end+1) = M(k).mark.(fn{i})(2); %#ok<AGROW>
        end
        % include OAP1 + the source so the source->OAP1->DM1 leg is in-window
        sx(end+1) = M(k).mark_OAP1(1); sy(end+1) = M(k).mark_OAP1(2); %#ok<AGROW>
        sx(end+1) = M(k).src(1);       sy(end+1) = M(k).src(2);       %#ok<AGROW>
    end
    padx = 0.05*(max(sx)-min(sx));
    xl = [min(sx)-padx max(sx)+padx];
    % vertical: use the ray fold spread WITHIN the physical window
    vv = [];
    for k = 1:2
        inwin = M(k).U>=xl(1) & M(k).U<=xl(2) & M(k).U~=0;
        vv = [vv; M(k).V(inwin)]; %#ok<AGROW>
    end
    vc = (max(vv)+min(vv))/2; vh = 0.6*(max(vv)-min(vv)) + 0.05*(xl(2)-xl(1));
    yl = [vc-vh vc+vh];
    M(1).win = xl; M(2).win = xl;

    vis = 'off'; if opts.visible, vis='on'; end
    fig = figure('Visible',vis,'Color','w','Position',[60 60 1180 720]);
    tl  = tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');
    title(tl,'CTB coronagraph -- real-ray layout (XZ fold plane)', ...
        'FontWeight','bold','Interpreter','none');

    for k = 1:2
        ax = nexttile(tl); hold(ax,'on');
        draw_panel_(ax, M(k), xl, yl, k==2);   % scale bar only on lower panel
        % model name in the bottom-RIGHT corner (top edge holds the rotated
        % station labels; bottom-left holds the scale bar on the lower panel)
        text(ax, xl(2)-0.01*(xl(2)-xl(1)), yl(1)+0.10*(yl(2)-yl(1)), ...
            upper(M(k).name), 'FontWeight','bold','FontSize',11, ...
            'Interpreter','none','HorizontalAlignment','right', ...
            'BackgroundColor',[1 1 1 0.6]);
    end

    figpath = fullfile(opts.outdir,'ctb_train_render.png');
    exportgraphics(fig, figpath, 'Resolution',150);
    if ~opts.visible, close(fig); end
    fprintf('[train] wrote %s\n', figpath);
    out = struct('models',{{M.name}},'figure',figpath);
end

% ---------------------------------------------------------------------
function m = local_model_(name, rx, elt, N)
    macos.init(N);
    macos.load_rx(rx);
    cbm = macos.cbm();
    macos.trace(macos.num_elt());
    b = macos.draw_rays('XZ', 0, macos.num_elt());
    m.name = name; m.rx = rx; m.elt = elt; m.cbm = cbm;
    m.U = b.U; m.V = b.V; m.nper = b.nper; m.nray = b.nray;
    m.u = b.U(b.U~=0); m.v = b.V(b.V~=0);
    % station vertices (XZ = [x,z]); this bench is z=0 so V~=x-fold... use vpt
    fn = fieldnames(elt);
    for i = 1:numel(fn)
        vp = macos.get_elt_vpt(elt.(fn{i}));       % [x;y;z] source coords
        m.mark.(fn{i}) = [vp(1); vp(3)];           % XZ -> (x, z)
    end
    % OAP1 (element 1) + the source: the source->OAP1 leg is upstream of DM1
    % and would otherwise fall off the left edge of a DM1..FPA window.
    vp1 = macos.get_elt_vpt(1); m.mark_OAP1 = [vp1(1); vp1(3)];
    sf = macos.get_src_fov();                      % ChfRayPos of the source
    m.src = [sf.src_pos(1); sf.src_pos(3)];
    % EP reference sphere vertices (ExitPupil-1 is the NF1/FF sphere set)
    m.ep_sphere = [];
    for s = [elt.FPM-1, elt.Lyot, elt.ExitPupil]     % sphere / pupil marks
        try, vp = macos.get_elt_vpt(s); m.ep_sphere(:,end+1) = [vp(1);vp(3)]; catch, end
    end
end

function p = merge_mark_(m, nm)
% station mark by name; OAP1 lives in its own field (mark_OAP1).
    if strcmp(nm,'OAP1'), p = m.mark_OAP1; else, p = m.mark.(nm); end
end

function draw_panel_(ax, m, xl, yl, want_bar)
    % rays (muted green, thin).  Keep only the crossings INSIDE the physical
    % window, then draw ONE continuous polyline through them: this drops the
    % off-bench EP-return-sphere excursions (x ~ -3700) WITHOUT dropping the
    % physical beam segment that connects the two real optics bracketing the
    % detour (the earlier both-endpoints-in-window test deleted that segment,
    % so the beam appeared to vanish between elements).  The reference-sphere
    % legs are geometrically backtracks to the SAME chief pierce, so
    % connecting the surviving in-window points reproduces the real beam path.
    for r = 1:m.nray
        n = m.nper(r); if n < 2, continue; end
        u = m.U(1:n,r); v = m.V(1:n,r);
        in = u>=xl(1) & u<=xl(2) & v>=yl(1) & v<=yl(2);
        if nnz(in) < 2, continue; end
        plot(ax, u(in), v(in), '-', 'Color',[0 0.55 0.15 0.35], 'LineWidth',0.4);
    end
    % mask-plane station markers + labels off the ray lines (OAP1 first, then
    % the DM1..FPA stations); label struct fields are pulled by name.
    stn  = {'OAP1','DM1','DM2','Apodizer','FPM','Lyot','ExitPupil','FPA'};
    pmark = @(nm) merge_mark_(m, nm);
    % OAP1 sits at nearly the same X as DM2 on this folded bench, so drop its
    % label lower to avoid overprinting the DM2 label.
    ytop = yl(2)-0.08*(yl(2)-yl(1));
    ylow = yl(2)-0.24*(yl(2)-yl(1));
    for i = 1:numel(stn)
        p = pmark(stn{i});
        plot(ax, p(1), p(2), 'k.', 'MarkerSize',9);
        ytxt = ytop; if strcmp(stn{i},'OAP1'), ytxt = ylow; end
        text(ax, p(1), ytxt, stn{i}, ...
            'Rotation',90, 'FontSize',7, 'Interpreter','none', ...
            'HorizontalAlignment','left','VerticalAlignment','middle', ...
            'Color',[0.2 0.2 0.2]);
        xline(ax, p(1), ':', 'Color',[0.7 0.7 0.7], 'LineWidth',0.3, ...
            'HandleVisibility','off');
    end
    % the source (chief-ray origin) -- distinct marker
    plot(ax, m.src(1), m.src(2), 'p', 'MarkerEdgeColor',[0.85 0.4 0], ...
        'MarkerFaceColor',[1 0.7 0.2], 'MarkerSize',11);
    text(ax, m.src(1), yl(2)-0.08*(yl(2)-yl(1)), 'source', ...
        'Rotation',90,'FontSize',7,'Interpreter','none', ...
        'HorizontalAlignment','left','VerticalAlignment','middle', ...
        'Color',[0.85 0.4 0]);
    % EP reference spheres (magenta rings)
    if ~isempty(m.ep_sphere)
        plot(ax, m.ep_sphere(1,:), m.ep_sphere(2,:), 'o', ...
            'MarkerEdgeColor',[0.7 0 0.7], 'MarkerSize',7, 'LineWidth',1.0);
    end
    xlim(ax,xl); ylim(ax,yl); axis(ax,'equal'); box(ax,'on');
    set(ax,'XTick',[],'YTick',[]);
    xlabel(ax,sprintf('beam axis X  (BaseUnits, span %.0f)',xl(2)-xl(1)));
    if want_bar
        % scale bar: 500 mm (cbm-aware; deck BaseUnits = mm)
        L = 500;
        x0 = xl(1)+0.06*(xl(2)-xl(1)); y0 = yl(1)+0.10*(yl(2)-yl(1));
        plot(ax,[x0 x0+L],[y0 y0],'k-','LineWidth',2.5);
        text(ax,x0+L/2,y0+0.05*(yl(2)-yl(1)),sprintf('%d %s',L,'mm'), ...
            'HorizontalAlignment','center','FontSize',8);
    end
    hold(ax,'off');
end
