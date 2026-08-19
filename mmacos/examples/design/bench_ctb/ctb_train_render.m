function out = ctb_train_render(opts)
%CTB_TRAIN_RENDER  Deck-grade layout figure of the CTB (full s2s model).
%   One figure, XZ projection (the fold plane of this planar bench -- the
%   beam runs along +X and folds in Z; XZ is the informative view, YZ/XY
%   are degenerate).  The full surface-to-surface model (ctb_s2s_dcr.in) is
%   rendered as its real traced-ray fold path source->OAP1->...->FPA, with
%   every optic dotted+labelled AT ITS ACTUAL BEAM CROSSING.
%
%   Two things that trip up a naive layout render, handled here:
%     - The OAP vertices (VptElt) are the PARENT-axis vertices at z=0, far
%       from where the off-axis beam actually strikes (z=-113..-643).  So a
%       dot at the vertex sits on a flat z=0 line, NOT on the beam.  Dots
%       are placed at the mean RAY CROSSING at each element instead.
%     - The diffraction reference/return surfaces (exit-pupil spheres at
%       x~-3700, focus-return planes) are propagation bookkeeping, not the
%       physical beam.  The polyline is drawn through REAL optics only
%       (Reflector + FocalPlane), so it does not backtrack to a reference
%       plane sitting just left of a real fold (the "extra bit past DM1").
%
%   out = CTB_TRAIN_RENDER() writes ctb_train_render.png beside the deck.
%   Name-value: 'rx','model_size','outdir','visible'.
    arguments
        opts.rx         (1,:) char   = ''
        opts.model_size (1,1) double = 512
        opts.outdir     (1,:) char   = ''
        opts.visible    (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    if isempty(opts.rx),     opts.rx     = fullfile(here,'ctb_s2s_dcr.in'); end
    if isempty(opts.outdir), opts.outdir = here; end
    addpath(fullfile(here,'..','..','..','src'));
    assert(~isempty(getenv('MACOS_HOME')),'MACOS_HOME must be set.');

    % full-deck optical train, light order (elt -> label).  Reflectors +
    % focal plane + the three mask stations; these are the physical stops.
    stn = { 1,'OAP1'; 2,'DM1'; 5,'DM2'; 8,'OAP2'; 13,'OAP3'; 16,'Apodizer'; ...
           19,'OAP4'; 22,'FPM'; 24,'OAP5'; 27,'Lyot'; 30,'OAP6'; 35,'OAP7'; ...
           41,'OAP8'; 44,'FPA' };
    real_types = {'Reflector','FocalPlane'};              % the beam path

    macos.init(opts.model_size);
    macos.load_rx(opts.rx);
    nE  = macos.num_elt();
    cbm = macos.cbm();
    macos.trace(nE);
    b = macos.draw_rays('XZ', 0, nE);

    % which elements are real optics (beam-defining)?
    isreal = false(1,nE);
    for e = 1:nE
        info = macos.get_elt_info(e);
        isreal(e) = any(strcmp(info.type, real_types));
    end

    % mean beam crossing (U,V) per element, over rays that reach it
    Uc = nan(1,nE); Vc = nan(1,nE);
    for e = 1:nE
        u = []; v = [];
        for r = 1:b.nray
            idx = find(b.elt(1:b.nper(r),r) == e, 1);
            if ~isempty(idx) && b.U(idx,r) ~= 0
                u(end+1) = b.U(idx,r); v(end+1) = b.V(idx,r); %#ok<AGROW>
            end
        end
        if ~isempty(u), Uc(e) = mean(u); Vc(e) = mean(v); end
    end
    % source (chief-ray origin)
    sf = macos.get_src_fov(); src = [sf.src_pos(1); sf.src_pos(3)];

    % ---- window: source + all labelled-station beam crossings ----------
    sx = [src(1)]; sy = [src(2)];
    for i = 1:size(stn,1)
        e = stn{i,1};
        if isfinite(Uc(e)), sx(end+1)=Uc(e); sy(end+1)=Vc(e); end %#ok<AGROW>
    end
    padx = 0.05*(max(sx)-min(sx));
    xl = [min(sx)-padx max(sx)+padx];
    vc = (max(sy)+min(sy))/2; vh = 0.6*(max(sy)-min(sy)) + 0.06*(xl(2)-xl(1));
    yl = [vc-vh vc+vh];

    % ---- figure --------------------------------------------------------
    vis = 'off'; if opts.visible, vis='on'; end
    fig = figure('Visible',vis,'Color','w','Position',[60 60 1180 460]);
    ax  = axes(fig); hold(ax,'on');

    % beam: per ray, connect source -> real-optic crossings (in order),
    % keeping only in-window points (drops the off-bench reference spheres).
    for r = 1:b.nray
        u = src(1); v = src(2);
        for s = 1:b.nper(r)
            e = b.elt(s,r);
            if e>=1 && e<=nE && isreal(e) && b.U(s,r)~=0
                u(end+1)=b.U(s,r); v(end+1)=b.V(s,r); %#ok<AGROW>
            end
        end
        in = u>=xl(1) & u<=xl(2) & v>=yl(1) & v<=yl(2);
        if nnz(in) < 2, continue; end
        plot(ax, u(in), v(in), '-', 'Color',[0 0.55 0.15 0.30], 'LineWidth',0.4);
    end

    % source marker + label
    plot(ax, src(1), src(2), 'p', 'MarkerEdgeColor',[0.85 0.4 0], ...
        'MarkerFaceColor',[1 0.7 0.2], 'MarkerSize',12);
    label_(ax, src(1), 'source', yl, [0.85 0.4 0], 0);

    % optic dots ON THE BEAM at their actual crossings + labels.  Labels ride
    % the top edge at the optic's X; where two optics sit at nearly the same X
    % (OAP1~DM2, OAP7~FPA on this folded bench) the second label is dropped a
    % row so they don't overprint.
    valid = find(arrayfun(@(i) isfinite(Uc(stn{i,1})), 1:size(stn,1)));
    [~,ord] = sort(cellfun(@(e) Uc(e), stn(valid,1)));      % by X
    order = valid(ord);
    xspan = xl(2)-xl(1); lastx = -inf; row = 0;
    for j = 1:numel(order)
        i = order(j); e = stn{i,1}; nm = stn{i,2};
        ismask = any(strcmp(nm,{'Apodizer','FPM','Lyot'}));
        if ismask
            plot(ax, Uc(e), Vc(e), 'o', 'MarkerEdgeColor',[0.7 0 0.7], ...
                'MarkerSize',7, 'LineWidth',1.1);           % mask station
        else
            plot(ax, Uc(e), Vc(e), 'ko', 'MarkerFaceColor','k', 'MarkerSize',5);
        end
        if Uc(e)-lastx < 0.03*xspan, row = row+1; else, row = 0; end
        lastx = Uc(e);
        label_(ax, Uc(e), nm, yl, [0.2 0.2 0.2], row);
    end

    xlim(ax,xl); ylim(ax,yl); axis(ax,'equal'); box(ax,'on');
    set(ax,'XTick',[],'YTick',[]);
    title(ax,'CTB coronagraph -- real-ray layout (full model, XZ fold plane)', ...
        'FontWeight','bold','Interpreter','none');
    xlabel(ax,sprintf('beam axis X  (BaseUnits, span %.0f)', xl(2)-xl(1)));
    % scale bar (500 mm; deck BaseUnits = mm)
    L = 500; x0 = xl(1)+0.05*(xl(2)-xl(1)); y0 = yl(1)+0.08*(yl(2)-yl(1));
    plot(ax,[x0 x0+L],[y0 y0],'k-','LineWidth',2.5);
    text(ax,x0+L/2,y0+0.05*(yl(2)-yl(1)),sprintf('%d mm',L), ...
        'HorizontalAlignment','center','FontSize',8);

    figpath = fullfile(opts.outdir,'ctb_train_render.png');
    exportgraphics(fig, figpath, 'Resolution',150);
    if ~opts.visible, close(fig); end
    fprintf('[train] wrote %s\n', figpath);
    out = struct('rx',opts.rx,'figure',figpath, 'Uc',Uc,'Vc',Vc);
end

% ---------------------------------------------------------------------
function label_(ax, x, nm, yl, col, row)
% rotated label above the axis top so it never sits on a ray line; ROW>0
% drops it lower to clear a near-coincident neighbour.
    if nargin < 6, row = 0; end
    ytxt = yl(2) - (0.09 + 0.16*row)*(yl(2)-yl(1));
    text(ax, x, ytxt, nm, ...
        'Rotation',90, 'FontSize',7.5, 'Interpreter','none', ...
        'HorizontalAlignment','left', 'VerticalAlignment','middle', 'Color',col);
    xline(ax, x, ':', 'Color',[0.75 0.75 0.75], 'LineWidth',0.3, ...
        'HandleVisibility','off');
end
