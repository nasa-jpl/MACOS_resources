function sh = shroud_deck(rx, P, opts)
%SHROUD_DECK  Launch-shroud radial extent of a SPLICED prescription.
%   SH = SHROUD_DECK(RX, P) loads RX, traces it, and measures every
%   hardware element's radial extent about the launch axis:
%
%       extent(k) = hypot(cx, cy) + r_footprint
%
%   i.e. how far off the incoming-beam axis the element's illuminated
%   patch reaches.  SH.D = 2*max(extent) over hardware is the diameter
%   the observatory has to fit through.
%
%   ELEMENT=RETURN IS EXCLUDED, everything else is kept.  Return
%   surfaces are the propagator's return planes and exit-pupil
%   reference SPHERES -- mathematics, not glass anyone builds -- and an
%   exit-pupil sphere sits at a radius with nothing to do with the
%   hardware envelope.  Counting them inflated this gate by 0.86 m the
%   first time it was written (e2e6m S1 LOG).  Element=Reference
%   markers ARE hardware: a mask or pupil site is a real mount.
%
%   Read from the deck TEXT, because a spliced deck has no design spec.
%
%   SH = SHROUD_DECK(RX, P, 'extra', RX2) additionally measures RX2 and
%   returns the UNION -- for an observatory whose instruments live on
%   separate configurations of one structure (e.g. a deployable
%   pick-off feeding a second camera): both are hardware whichever one
%   the light is currently using.
%
%   Name-value:
%     'extra'   further deck(s) to union in (char or cellstr).
%     'png'     write the end-on figure here.
%     'model'   engine model size (default P.model).
%     'labels'  legend names per deck (default derived from filenames).
%
%   SH fields: .D (union diameter, m), .len (train length along the
%     launch axis), .per_deck (struct array with .rx .D .len .n_hw),
%     .png.
%
%   See also PACKAGING_REPORT, S3_BACKEND, S3_IMAGER.

    arguments
        rx        (1,:) char
        P         struct
        opts.extra           = {}
        opts.png  (1,:) char = ''
        opts.model(1,1) double = 0
        opts.labels          = {}
    end
    if opts.model <= 0, opts.model = P.model; end
    decks = [{rx}, cellstr_(opts.extra)];
    labs  = cellstr_(opts.labels);
    if numel(labs) ~= numel(decks)
        labs = cell(1,numel(decks));
        for k = 1:numel(decks), [~,labs{k}] = fileparts(decks{k}); end
    end

    % Accumulate into a cell and concatenate: assigning struct-by-struct
    % into a pre-declared empty struct array is order- and field-set
    % sensitive, and silently works for one deck while failing on two.
    tmp = cell(1, numel(decks));
    for k = 1:numel(decks)
        tmp{k} = one_deck_(decks{k}, opts.model);
    end
    per = [tmp{:}];
    sh = struct('D', 2*max(arrayfun(@(p) max(p.extent_hw), per)), ...
                'len', max([per.len]), 'per_deck', per, 'png', opts.png);

    if ~isempty(opts.png)
        draw_(per, labs, sh, P, opts.png);
    end
end

% ---------------------------------------------------------------------
function q = one_deck_(rx, model)
    macos.init(model);
    nE = macos.load_rx(rx);
    kinds = regexp(fileread(rx), '(?m)^\s*Element=\s*(\S+)', 'tokens');
    enm   = regexp(fileread(rx), '(?m)^\s*EltName=\s*(\S+)', 'tokens');
    names = repmat({''},1,nE);
    for k = 1:min(nE, numel(enm)), names{k} = enm{k}{1}; end
    hw = true(1,nE);
    for k = 1:min(nE, numel(kinds))
        hw(k) = ~strcmpi(kinds{k}{1}, 'Return');
    end
    macos.ray_hist('on');  s = macos.trace(nE);
    h = macos.ray_hist(s.nRays);  macos.ray_hist('off');
    C = nan(3,nE);  R = nan(1,nE);
    for k = 1:nE
        m = h.ok(:,k+1);  m(1) = false;
        if nnz(m) < 3, continue; end
        Q = h.P(:, m, k+1);
        C(:,k) = mean(Q,2);
        R(k)   = max(vecnorm(Q - C(:,k), 2, 1));
    end
    ok = isfinite(R);
    ext = hypot(C(1,:), C(2,:)) + R;
    q = struct('rx',rx, 'D', 2*max(ext(ok & hw)), ...
               'len', max(C(3,ok)) - min(C(3,ok)), ...
               'n_hw', nnz(ok & hw), 'C', C, 'R', R, 'hw', hw, ...
               'names', {names}, 'extent_hw', ext(ok & hw));
end

function draw_(per, labs, sh, P, png)
%DRAW_  End-on view.  Decks after the first draw ONLY the elements the
%   earlier decks do not already have: two instrument legs on one
%   telescope share ~all their elements, so plotting both in full paints
%   the second leg exactly over the first and the figure silently shows
%   one leg while the legend claims two.
    f = figure('Visible','off','Position',[100 100 820 780]);
    ax = axes(f, 'Position',[0.11 0.30 0.60 0.62]);
    hold(ax,'on'); axis(ax,'equal');
    th = linspace(0,2*pi,361);  Rg = P.shroud_D_m/2;
    hg = plot(ax, Rg*cos(th), Rg*sin(th), 'k-', 'LineWidth', 2.0);
    cols = lines(max(numel(per),3));
    hleg = gobjects(1,numel(per));  seen = {};  uniq = cell(1,numel(per));
    for d = 1:numel(per)
        p = per(d);  idx = find(isfinite(p.R));
        mine = {};
        for j = 1:numel(idx)
            k = idx(j);
            nm = p.names{k};
            if d > 1 && ~isempty(nm) && any(strcmp(seen, nm)), continue; end
            mine{end+1} = nm;                                     %#ok<AGROW>
            st = '-';  w = 1.2;
            if ~p.hw(k), st = ':';  w = 0.7;  end   % not hardware, not gated
            hh = plot(ax, p.C(1,k)+p.R(k)*cos(th), p.C(2,k)+p.R(k)*sin(th), ...
                      st, 'Color', cols(d,:), 'LineWidth', w);
            if ~isgraphics(hleg(d)), hleg(d) = hh; end
        end
        uniq{d} = mine;
        seen = [seen, p.names(idx)];                              %#ok<AGROW>
    end
    hu = plot(ax, sh.D/2*cos(th), sh.D/2*sin(th), '--', ...
              'Color',[0.85 0.2 0.2], 'LineWidth',1.4);
    xlabel(ax,'x  [m]'); ylabel(ax,'y  [m]'); grid(ax,'on'); box(ax,'on');
    keep = isgraphics(hleg);
    lab2 = labs(:).';
    for d = 1:numel(per)
        if d > 1, lab2{d} = sprintf('%s (its own optics)', lab2{d}); end
    end
    legend(ax, [hg hleg(keep) hu], ...
           [{sprintf('%.1f m shroud', P.shroud_D_m)}, lab2(keep), ...
            {sprintf('hardware union %.3f m', sh.D)}], ...
           'Location','southoutside', 'Interpreter','none');
    title(ax, sprintf('end-on: hardware union %.3f m against the %.1f m gate (%s)', ...
          sh.D, P.shroud_D_m, tern_(sh.D <= P.shroud_D_m,'FITS','OVER')), ...
          'Interpreter','none');
    lim = 1.3*Rg;  xlim(ax,[-lim lim]);  ylim(ax,[-lim lim]);

    % inset: the instrument cluster, which is centimetre-class against a
    % 6 m primary and is otherwise a few pixels
    ins = axes(f, 'Position',[0.735 0.60 0.245 0.245]);
    hold(ins,'on'); axis(ins,'equal');
    xs = []; ys = [];
    for d = 1:numel(per)
        p = per(d);  idx = find(isfinite(p.R));
        for j = 1:numel(idx)
            k = idx(j);
            if p.R(k) > 0.5, continue; end          % skip the big mirrors
            plot(ins, p.C(1,k)+p.R(k)*cos(th), p.C(2,k)+p.R(k)*sin(th), ...
                 '-', 'Color', cols(d,:), 'LineWidth', 1.1);
            xs(end+1) = p.C(1,k); ys(end+1) = p.C(2,k);           %#ok<AGROW>
        end
    end
    if ~isempty(xs)
        pad = max(0.25, 0.6*max(range_(xs), range_(ys)));
        xlim(ins, [mean(xs)-pad mean(xs)+pad]);
        ylim(ins, [mean(ys)-pad mean(ys)+pad]);
    end
    grid(ins,'on'); box(ins,'on');
    title(ins,'instrument legs (zoom)','FontSize',9,'Interpreter','none');
    exportgraphics(f, png, 'Resolution', 150);  close(f);
end

function r = range_(v), r = max(v) - min(v); end

function c = cellstr_(v)
    if isempty(v), c = {}; elseif ischar(v), c = {v}; else, c = cellstr(v); end
end
function s = tern_(c,a,b), if c, s=a; else, s=b; end, end
