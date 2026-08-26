function OUT = r2_sequence_fig()
%R2_SEQUENCE_FIG  The optical SEQUENCE sketch -- both legs (Dave item 2).
%
%   A light-order schematic, PM -> ... -> detector for the coronagraph
%   AND imager legs -- a diagram of WHAT the light visits in what
%   order, not a scaled render (the scaled renders are r2_bench_fig's
%   job).  Data-driven: the node lists are read from the committed
%   decks (r1_seg_full.in + round 1's s3_imager_leg.in), so the sketch
%   cannot drift from the trains it describes; only the CLASSIFICATION
%   (mirror / DM / mask site / focus / detector) is local.
%
%   Writes r2_sequence.png.

    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(struct());

    coro = names_(fullfile(P.outdir,'r1_seg_full.in'));
    imgr = names_(fullfile(P.r1dir, 's3_imager_leg.in'));

    % collapse the 19 segments into one PM node; the trunk both legs
    % share is PM -> M2 -> M3 -> OAP1
    nseg = nnz(startsWith(coro, 'Seg'));
    trunk = {sprintf('PM (%d seg)', nseg), 'M2', 'M3', 'OAP1'};
    k1 = find(strcmp(coro, 'OAP1'), 1);
    legA = coro(k1+1:end).';               % the coronagraph leg after OAP1
    legB = imgr(~strcmp(imgr,'OAP1')).';   % the imager leg after OAP1

    f = figure('Visible','off','Color','w','Position',[40 40 1500 560]);
    ax = axes(f,'Position',[0.02 0.03 0.96 0.88]);  hold(ax,'on');
    axis(ax,[0 15 0 6.2]);  axis(ax,'off');

    % ROW PLAN (no line ever crosses a box): the shared trunk and the
    % imager leg share the TOP row (the imager IS the trunk's
    % continuation when the pick-off deploys); the coronagraph leg
    % serpentines on the two rows below, entered by an elbow routed
    % through the clear band under the trunk.
    prev = [];
    x = 0.6;
    for k = 1:numel(trunk)
        prev = node_(ax, x, 5.4, trunk{k}, class_(trunk{k}), prev);
        x = x + 1.55;
    end
    branch = prev;                          % OAP1: both legs leave here

    prev = branch;
    for k = 1:numel(legB)
        xx = 7.8 + (k-1)*1.55;
        prev = node_(ax, xx, 5.4, legB{k}, class_(legB{k}), prev);
    end
    text(ax, 6.5, 5.62, 'pick-off deployed', 'FontSize', 7.5, ...
         'HorizontalAlignment','center', 'Color',[0.35 0.35 0.35]);

    xs = 0.6 + (0:7)*1.55;
    prev = [];                              % elbow drawn manually below
    for k = 1:numel(legA)
        row = 3.9 - 1.5*floor((k-1)/8);
        col = mod(k-1, 8) + 1;
        if mod(floor((k-1)/8),2) == 0, xx = xs(col); else, xx = xs(9-col); end
        prev = node_(ax, xx, row, legA{k}, class_(legA{k}), prev);
    end
    % the elbow: OAP1 bottom -> clear band -> left margin -> DM1 left
    gc = [0.45 0.45 0.45];
    plot(ax, [branch(1) branch(1)], [5.09 4.65], '-', 'Color',gc, 'LineWidth',1.1);
    plot(ax, [branch(1) 0.05], [4.65 4.65], '-', 'Color',gc, 'LineWidth',1.1);
    plot(ax, [0.05 0.05], [4.65 3.9], '-', 'Color',gc, 'LineWidth',1.1);
    plot(ax, [0.05 -0.08+xs(1)-0.55], [3.9 3.9], '-', 'Color',gc, 'LineWidth',1.1);
    patch(ax, xs(1)-0.55-[0.14 0 0.14], 3.9+[0.06 0 -0.06], gc, 'EdgeColor','none');

    text(ax, 0.1, 6.0, ['light order, both instruments -- shared trunk + ' ...
        'imager leg (top row), coronagraph leg (serpentine below)'], ...
        'FontSize', 11, 'FontWeight','bold');
    lg = {'powered mirror','DM','mask / pupil site','focus station','detector'};
    cl = {clr_('mirror'), clr_('dm'), clr_('mask'), clr_('focus'), clr_('det')};
    for k = 1:numel(lg)
        rectangle(ax,'Position',[8.9+1.22*(k-1) 0.55 0.26 0.26], ...
                  'Curvature',0.4,'FaceColor',cl{k},'EdgeColor',[0.3 0.3 0.3]);
        text(ax, 8.9+1.22*(k-1)+0.13, 0.40, lg{k}, 'FontSize',7, ...
             'HorizontalAlignment','center');
    end

    png = fullfile(P.outdir,'r2_sequence.png');
    exportgraphics(f, png, 'Resolution', 160);
    close(f);
    fprintf('r2_sequence_fig: %d trunk + %d coro + %d imager nodes -> %s\n', ...
            numel(trunk), numel(legA), numel(legB), png);
    OUT = struct('png',png, 'trunk',{trunk}, 'legA',{legA}, 'legB',{legB});
end

% =========================================================================
function nm = names_(rx)
    nm = regexp(fileread(rx), '^\s*EltName=\s*(\S+)', 'tokens','lineanchors');
    nm = cellfun(@(c) c{1}, nm, 'UniformOutput', false);
end

function c = class_(name)
    if any(strcmp(name, {'DM1','DM2'})),                     c = 'dm';
    elseif any(strcmp(name, {'Apodizer','Lyot','Backend','SharedPupil'})), c = 'mask';
    elseif any(strcmp(name, {'FPM','FieldStop'})),           c = 'focus';
    elseif any(strcmp(name, {'Science','Imager'})),          c = 'det';
    else,                                                    c = 'mirror';
    end
end

function c = clr_(kind)
    switch kind
        case 'mirror', c = [0.80 0.88 0.97];
        case 'dm',     c = [0.99 0.83 0.60];
        case 'mask',   c = [0.88 0.80 0.95];
        case 'focus',  c = [1.00 1.00 1.00];
        case 'det',    c = [0.45 0.45 0.50];
    end
end

function h = node_(ax, x, y, label, kind, prev)
    w = 1.25;  hh = 0.62;
    ec = [0.30 0.30 0.30];
    if strcmp(kind,'focus'), ls = '--'; else, ls = '-'; end
    rectangle(ax, 'Position',[x-w/2 y-hh/2 w hh], 'Curvature',0.35, ...
              'FaceColor', clr_(kind), 'EdgeColor', ec, 'LineStyle', ls);
    tc = [0 0 0];  if strcmp(kind,'det'), tc = [1 1 1]; end
    text(ax, x, y, label, 'HorizontalAlignment','center', ...
         'FontSize', 8.5, 'FontWeight','bold', 'Color', tc, 'Interpreter','none');
    h = [x y];
    if ~isempty(prev)
        d = h - prev;  L = norm(d);  u = d/L;
        a = prev + u*( (abs(u(1))>abs(u(2)))*w/2 + (abs(u(1))<=abs(u(2)))*hh/2 + 0.03 );
        b = h    - u*( (abs(u(1))>abs(u(2)))*w/2 + (abs(u(1))<=abs(u(2)))*hh/2 + 0.03 );
        plot(ax, [a(1) b(1)], [a(2) b(2)], '-', 'Color',[0.45 0.45 0.45], ...
             'LineWidth', 1.1);
        n = [-u(2) u(1)];
        tri = [b; b - 0.14*u + 0.06*n; b - 0.14*u - 0.06*n];
        patch(ax, tri(:,1), tri(:,2), [0.45 0.45 0.45], 'EdgeColor','none');
    end
end
