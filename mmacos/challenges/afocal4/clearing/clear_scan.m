function R = clear_scan(P, D0, opts)
%CLEAR_SCAN  Is there ANY station where the collimator fits its own feed cone?
%
%   R = CLEAR_SCAN(P, D0) is step 1 of BRIEF_afocal4_clear: sweep the
%   collimator's station and plot what the part DEMANDS against what the
%   feed beam leaves AVAILABLE, so that a fold or a tilt is spent only after
%   the cheap answer has been looked at.  It sweeps two axes, because the
%   design has exactly two that move the collimator without touching the
%   customer interface:
%
%     'standoff'  the FIELD MIRROR's standoff -- the only DOF the closure
%                 leaves free once the interface is fixed, and therefore the
%                 literal "collimator station" knob: it moves the collimator
%                 through a metre of z.
%     'iface'     the INTERFACE standoff -- the S4 ruling's operating point.
%                 Included because the law below says it is the dominant
%                 variable, which is not obvious and is worth measuring.
%
%   THE TWO CURVES, both engine truth over the whole field box:
%     r_demand  the collimator's UNION footprint radius -- how big the part
%               has to be to carry every field.
%     r_avail   r_demand plus the signed bare-glass clearance floor -- i.e.
%               the largest that part could be and still only TOUCH the feed
%               beam.  Below r_demand means the beam is inside the glass.
%   They never cross.  CLEAR_LAW says why in one number, and the reason is
%   structural rather than a property of this particular layout: both
%   footprints are scaled copies of the same off-axis field box, so they can
%   separate only if their scales differ by more than (bias+half)/(bias-half)
%   = 2.43, while the collimator's own scale is PINNED at M * iface by the
%   interface specification and the feed's can only reach the intermediate
%   image height.  Every field-proportional remedy is inside that bound.
%
%   THIS IS A GEOMETRY SCAN, NOT A RE-SOLVE, and that is deliberate.  Each
%   point re-closes the first order exactly (AFOCAL4_BUILD) and carries D0's
%   conics unchanged: the question "does a body stand in a beam" is answered
%   by the layout, and a conic moves a footprint by less than a millimetre
%   on these decks.  Re-solving every point would cost hours and change the
%   answer by nothing -- but it WOULD change the wavefront, so no merit
%   number from this scan is quotable and none is reported.
%
%   Name-value:
%     'axes'      which sweeps to run: {'standoff','iface'} (both)
%     'standoff'  the field-mirror standoffs to try, m
%     'iface'     the interface standoffs to try, m
%     'decks'     extra committed prescriptions to measure as they are
%                 (full paths).  These carry their OWN re-solved front ends,
%                 so they are reported as a separate block and never mixed
%                 into a sweep that holds the front end.
%     'fields'    the field set (default P.Fsolve, his 3x3 box)
%     'body_k'/'body_pad'   the declared body allowance (1.15, 15 mm); the
%                 scan ALSO reports bare lit glass (1.0, 0) every time
%     'fig'       draw the figure (true);  'save' write it (path or '')
%     'quiet'
%
%   Returns R with .standoff .iface .decks (struct arrays of the measured
%   points) and .law (CLEAR_LAW on D0's own deck).
%
%   See also CLEAR_LAW, AFOCAL4_UNION, CLEAR_TILT, AFOCAL4_CLEARING.

    arguments
        P (1,1) struct
        D0 (1,1) struct
        opts.axes     (1,:) cell = {'standoff','iface'}
        opts.standoff (1,:) double = [-0.60 -0.45 -0.30 -0.20 -0.10 -0.0386 ...
                                       0.05 0.15 0.25 0.40]
        opts.iface    (1,:) double = [0.05 0.09 0.14 0.20 0.26 0.30 0.343 0.40]
        opts.decks    (1,:) cell = {}
        opts.fields   (:,2) double = []
        opts.body_k   (1,1) double = 1.15
        opts.body_pad (1,1) double = 0.015
        opts.fig      (1,1) logical = true
        opts.save     (1,:) char = ''
        opts.quiet    (1,1) logical = false
    end
    F = opts.fields;   if isempty(F), F = P.Fsolve; end
    if ~isfield(D0,'tilt_deg'), D0.tilt_deg = 0; end

    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>
    R = struct('P',P, 'D0',D0, 'fields',F, 'standoff',[], 'iface',[], ...
               'decks',[], 'law',[]);

    % ---- the law, on the design the sweep starts from --------------------
    afocal4_build(P, D0, tmp, 'verify',false);
    nE = macos.num_elt();
    R.law = clear_law(tmp, 'fields',F, 'leg',2, 'elt',nE-1, 'M',P.M, ...
                      'quiet',opts.quiet);
    R.need = R.law.need;

    % ---- axis 1: the collimator station ----------------------------------
    if any(strcmp(opts.axes,'standoff'))
        if ~opts.quiet
            hdr_('the COLLIMATOR STATION (field-mirror standoff), interface held');
            head_('s_FM mm');
        end
        R.standoff = sweep_(P, D0, 'fm_standoff', opts.standoff, F, opts, tmp);
    end

    % ---- axis 2: the interface standoff ----------------------------------
    if any(strcmp(opts.axes,'iface'))
        if ~opts.quiet
            hdr_(['the INTERFACE STANDOFF (the S4 operating point), front ' ...
                  'end and conics held']);
            head_('iface mm');
        end
        R.iface = sweep_(P, D0, 'iface', opts.iface, F, opts, tmp);
    end

    % ---- and the committed decks, measured as they are -------------------
    if ~isempty(opts.decks)
        if ~opts.quiet
            hdr_(['the COMMITTED trade curve, each deck as it was solved ' ...
                  '(its own front end)']);
            head_('deck');
        end
        S = blank_();
        for i = 1:numel(opts.decks)
            [~,nm] = fileparts(opts.decks{i});
            S(end+1) = measure_(opts.decks{i}, nm, NaN, F, opts, opts.quiet); %#ok<AGROW>
        end
        R.decks = S;
    end

    if opts.fig, R.fig = figure_(R, opts); end
end

% =====================================================================
function S = blank_()
%BLANK_  The ONE place the scan's row shape is declared.  MATLAB's
%   `arr(k) = s` fails on dissimilar structs and only when REACHED -- i.e.
%   after the expensive part is already spent (RESULTS rule 7, which this
%   file has now paid for twice).  Every accumulator seeds from here, so a
%   field added to MEASURE_ cannot drift out of step with two separate
%   empty-struct literals.
    S = struct('tag',{},'val',{},'z_body',{},'r_dem',{},'r_avail',{}, ...
               'floor_bare',{},'floor_body',{},'floor_all_bare',{}, ...
               'floor_all_body',{},'worst_all',{},'gate_ok',{}, ...
               'ratio',{},'offset_m',{},'nLost',{},'worst',{});
end

function S = sweep_(P, D0, field, vals, F, opts, tmp)
    S = blank_();
    for i = 1:numel(vals)
        D = D0;   D.(field) = vals(i);
        try
            afocal4_build(P, D, tmp, 'verify',false);
        catch ME
            if ~opts.quiet
                fprintf('  %9.1f   WALL: %s\n', vals(i)*1e3, one_line_(ME.message));
            end
            continue;
        end
        S(end+1) = measure_(tmp, sprintf('%s %.4f',field,vals(i)), vals(i), ...
                            F, opts, opts.quiet); %#ok<AGROW>
    end
end

function s = measure_(deck, tag, val, F, opts, quiet)
%MEASURE_  One scan point: the gate at the declared allowance, the gate at
%   bare lit glass, and the law -- all on the same loaded deck.
    Kb = afocal4_union(deck, 'fields',F, 'body_k',1.0, 'body_pad',0.0, ...
                       'quiet',true);
    Km = afocal4_union(deck, 'fields',F, 'body_k',opts.body_k, ...
                       'body_pad',opts.body_pad, 'init',false, 'quiet',true);
    nE = numel(Kb.foot_r);   e = nE - 1;
    L  = clear_law(deck, 'fields',F, 'leg',2, 'elt',e, 'init',false, 'quiet',true);
    % the pair the brief is about, not merely the worst pair: on a cleared
    % design some OTHER pair becomes the floor, and reporting only the floor
    % would hide whether the collimator itself came free.
    ip = find([Kb.pair.leg] == 2 & [Kb.pair.obst] == e, 1);
    fb = Kb.pair(ip).d_m;
    im = find([Km.pair.leg] == 2 & [Km.pair.obst] == e, 1);
    fm = Km.pair(im).d_m;
    % BOTH floors, always.  The collimator pair is what this stage is
    % about, but the GATE's verdict is the minimum over every pair -- and
    % on the 50 mm trade point those two disagree in the way that matters:
    % the collimator pair clears there (+37 mm) while the deck still fails,
    % on its field-mirror -> collimator leg against the cold stop's body.
    % Reporting only the pair under study would read as "50 mm clears".
    s = struct('tag',tag, 'val',val, 'z_body',Kb.vpt(3,e), ...
               'r_dem',Kb.foot_r(e), 'r_avail',Kb.foot_r(e) + fb, ...
               'floor_bare',fb, 'floor_body',fm, ...
               'floor_all_bare',Kb.floor_m, 'floor_all_body',Km.floor_m, ...
               'worst_all',Km.worst_name, 'gate_ok',Km.ok, ...
               'ratio',L.ratio, 'offset_m',L.offset_m, 'nLost',Km.nLost, ...
               'worst',Km.worst_name);
    if ~quiet
        v = val*1e3;   if isnan(val), v = NaN; end
        fprintf(['  %9.1f %+9.4f %9.1f %9.1f %9.2f %9.2f %10.2f %8.4f %9.2f ' ...
                 '%5d  %s\n'], v, s.z_body, s.r_dem*1e3, s.r_avail*1e3, ...
                fb*1e3, fm*1e3, Km.floor_m*1e3, s.ratio, s.offset_m*1e3, ...
                s.nLost, short_(tag));
    end
end

function hdr_(t)
    fprintf('\n  --- %s ---\n', t);
end

function head_(c1)
    fprintf('  %9s %9s %9s %9s %9s %9s %10s %8s %9s %5s  %s\n', c1, 'z_col m', ...
            'demand', 'avail mm', 'bare mm', 'body mm', 'GATE mm', 'ratio', ...
            'offset', 'lost', 'point');
    fprintf(['  %9s %9s %9s %9s %9s %9s %10s\n'], '', '', '(collimator pair)', ...
            '', '', '', '(all pairs)');
end

function f = figure_(R, opts)
%FIGURE_  Two panels: what the part DEMANDS against what the feed beam
%   leaves AVAILABLE, on each of the design's two axes.  The titles carry
%   the signed gap and how many points of each sweep actually clear -- a
%   headline that overstates the result is worse than no headline, and the
%   interface-standoff axis DOES cross at its short end (against bare glass,
%   at a standoff whose delivered deck has a five-metre back end).
    f = figure('Position',[80 80 1180 560], 'Color','w');
    tl = tiledlayout(f, 1, 2, 'Padding','compact', 'TileSpacing','compact');
    pans = {R.standoff, R.iface};
    labs = {'field-mirror standoff s_{FM}  (mm)', 'interface standoff  (mm)'};
    ttl  = {'the collimator STATION', 'the interface STANDOFF'};
    ax1 = [];
    for i = 1:2
        S = pans{i};
        ax = nexttile(tl);
        if isempty(S), axis(ax,'off');  title(ax, [ttl{i} ' -- not run']);  continue; end
        if isempty(ax1), ax1 = ax; end
        x = [S.val]*1e3;
        plot(ax, x, [S.r_dem]*1e3, '-o', 'LineWidth',1.8, 'Color',[0.85 0.20 0.15], ...
             'MarkerFaceColor',[0.85 0.20 0.15], 'MarkerSize',4);
        hold(ax,'on');
        plot(ax, x, [S.r_avail]*1e3, '-s', 'LineWidth',1.8, 'Color',[0.10 0.40 0.75], ...
             'MarkerFaceColor',[0.10 0.40 0.75], 'MarkerSize',4);
        fill(ax, [x fliplr(x)], [[S.r_dem]*1e3 fliplr([S.r_avail]*1e3)], ...
             [0.85 0.20 0.15], 'FaceAlpha',0.10, 'EdgeColor','none');
        grid(ax,'on');   box(ax,'on');
        xlabel(ax, labs{i});
        ylabel(ax, 'radius about the collimator''s footprint centre  (mm)');
        nb = nnz([S.floor_bare] > 0);   nm = nnz([S.floor_body] > 0);
        title(ax, {sprintf('%s  --  ratio %.2f .. %.2f  (needs %.2f)', ttl{i}, ...
                   min([S.ratio]), max([S.ratio]), R.need), ...
                   sprintf(['bare-glass gap %+.0f .. %+.0f mm; clears at %d ' ...
                   'of %d points bare, %d with the allowance'], ...
                   min([S.floor_bare])*1e3, max([S.floor_bare])*1e3, nb, ...
                   numel(S), nm)});
    end
    if ~isempty(ax1)
        lg = legend(ax1, {'r_{demand}: the union footprint the part must carry', ...
              'r_{available}: the largest part that only TOUCHES the feed beam'}, ...
              'Orientation','horizontal', 'Box','off');
        lg.Layout.Tile = 'south';
    end
    ttlstr = sprintf(['the collimator against its own feed beam   --   the ' ...
        'field box demands a walk ratio of %.3f; the station reaches %.2f, ' ...
        'the standoff %.2f'], R.need, mx_(R.standoff), mx_(R.iface));
    annotation(f, 'textbox', [0.02 0.955 0.96 0.04], 'String', ttlstr, ...
        'HorizontalAlignment','center', 'EdgeColor','none', ...
        'FontWeight','bold', 'FontSize',11);
    tl.OuterPosition = [0 0 1 0.94];
    if ~isempty(opts.save)
        exportgraphics(f, opts.save, 'Resolution',150);
        fprintf('  wrote %s\n', opts.save);
    end
end

function v = mx_(S)
    v = NaN;   if ~isempty(S), v = max([S.ratio]); end
end
function s = short_(t)
    s = t;   if numel(s) > 28, s = s(1:28); end
end
function s = one_line_(m)
    s = regexprep(m, '\s+', ' ');   if numel(s) > 90, s = [s(1:90) '...']; end
end
function del_(p),  if exist(p,'file'), delete(p); end,  end
