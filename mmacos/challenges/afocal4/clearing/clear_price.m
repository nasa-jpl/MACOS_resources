function R = clear_price(P, D0, opts)
%CLEAR_PRICE  What an extraction tilt buys, and what it costs.
%
%   R = CLEAR_PRICE(P, D0) sweeps the field mirror's extraction tilt and
%   reports, at each angle, the clearance it wins and the design quality it
%   spends -- the exchange rate, which is the deliverable, not a chosen
%   angle.  Same shape of statement as the S4 ruling's interface-standoff
%   trade: a parameter carried and reported rather than optimised away.
%
%   WHY A TILT AND NOT SOMETHING CHEAPER.  CLEAR_LAW measures that the
%   collimator's footprint and the feed beam's are both scaled copies of
%   the same off-axis field box, with scales in a ratio of 1.30 where 2.43
%   is needed, and that the collimator's scale is PINNED at M * iface by the
%   interface specification.  CLEAR_SCAN then shows the ratio never gets
%   past 1.48 over a metre and a half of collimator travel.  A tilt is the
%   cheapest thing that is not bound by that law: it separates the two
%   bundles by a FIELD-INDEPENDENT 2*alpha*d, which CLEAR_LAW reports
%   directly as the fitted offset.
%
%   TWO PRICES, AND THE DIFFERENCE BETWEEN THEM IS THE POINT.
%     RAW      the tilt applied to the delivered design as it stands.  This
%              is what the mechanism costs before anything is done about it.
%     RESOLVED the same tilt with the conics, the field-mirror standoff and
%              the front end re-solved around it ('resolve', a list of
%              angles).  A raw price quoted as THE price would overstate it:
%              the parent's conics were solved for an untilted train.
%
%   THE WAVEFRONT IS NOT THE PRICE HERE, and that is worth saying because it
%   is the opposite of the usual expectation.  A mirror at the field
%   conjugate carries almost no beam per field -- 1.8 mm of footprint
%   against a 113 mm union -- so swinging it barely touches the wavefront;
%   what it moves is the PUPIL, which is the one thing the fourth mirror was
%   added to control.  The columns to read are blur, breathing and wander.
%
%   Name-value:
%     'tilt'      angles to sweep, deg
%     'resolve'   angles at which to also re-solve (default none)
%     'resolved_mat'  cell of .mat paths, each holding a finished
%                 CLEAR_SOLVE result in a variable R.  A re-solve is a
%                 45-minute artifact, so it is checkpointed and re-read
%                 rather than re-run every time the report is rebuilt (the
%                 save-resumable-workspaces rule).  Points loaded this way
%                 are re-GATED and re-scored here, not trusted from the
%                 file -- only the converged DESIGN is taken from it.
%     'dofs'      DOF set for those re-solves ({'conic','standoff','front'})
%     'max_fev'   evaluation budget per re-solve (0 = P.solve default)
%     'axis'      tilt axis (default [1 0 0]; CLEAR_SCAN's own measurement
%                 says the bias-plane axis buys more clearance per degree
%                 than the perpendicular one)
%     'fields'    field set (default P.Fsolve)
%     'body_k'/'body_pad'   declared allowance (1.15, 15 mm)
%     'deck_dir'  where to keep the re-solved decks ('' = temporary)
%     'fig'/'save'/'quiet'
%
%   Returns R with .raw and .resolved (struct arrays: .tilt_deg .floor_bare
%   .floor_body .offset_mm .wfe_nm .blur_um .breathe_pct .wander_um .mag
%   .nLost .worst .deck) and .parent (the untilted score).
%
%   See also CLEAR_TILT, CLEAR_SOLVE, CLEAR_SCAN, CLEAR_LAW, AFOCAL4_UNION.

    arguments
        P (1,1) struct
        D0 (1,1) struct
        opts.tilt     (1,:) double = [-14 -12 -10 -9 -8 -6 -4 -2 0 2 4 6 8]
        opts.resolve  (1,:) double = []
        opts.resolved_mat (1,:) cell = {}
        opts.dofs     (1,:) cell = {'conic','standoff','front'}
        opts.max_fev  (1,1) double = 0
        opts.axis     (1,3) double = [1 0 0]
        opts.fields   (:,2) double = []
        opts.body_k   (1,1) double = 1.15
        opts.body_pad (1,1) double = 0.015
        opts.deck_dir (1,:) char = ''
        opts.fig      (1,1) logical = true
        opts.save     (1,:) char = ''
        opts.quiet    (1,1) logical = false
    end
    F = opts.fields;   if isempty(F), F = P.Fsolve; end
    if ~isfield(D0,'tilt_deg'), D0.tilt_deg = 0; end

    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>

    % ---- the raw price ----------------------------------------------------
    if ~opts.quiet
        fprintf('\n  --- the RAW price of an extraction tilt (no re-solve) ---\n');
        head_();
    end
    R.raw = arrayfun(@(a) point_(P, setf_(D0,'tilt_deg',a), tmp, F, opts, ''), ...
                     opts.tilt);
    R.parent = R.raw([R.raw.tilt_deg] == 0);

    % ---- and the price after a re-solve -----------------------------------
    R.resolved = R.raw([]);
    if ~isempty(opts.resolve) || ~isempty(opts.resolved_mat)
        if ~opts.quiet
            fprintf(['\n  --- the RESOLVED price: conics, standoff and front ' ...
                     'end re-solved around the tilt ---\n']);
            head_();
        end
        Q = P;
        if opts.max_fev > 0, Q.solve.max_fev = opts.max_fev; end
        for a = opts.resolve
            dk = tmp;
            if ~isempty(opts.deck_dir)
                dk = fullfile(opts.deck_dir, sprintf('afocal4_clear_t%+03.0f.in', a*10));
            end
            S = clear_solve(Q, setf_(D0,'tilt_deg',a), 'dofs',opts.dofs, ...
                    'deck',dk, 'axis',opts.axis, 'max_iter',400, ...
                    'label',sprintf('tilt %+.1f deg', a), 'quiet',true);
            R.resolved(end+1) = point_(P, S.D, dk, F, opts, dk, S); %#ok<AGROW>
        end
        % checkpointed re-solves: take only the converged DESIGN from the
        % file and rebuild, re-gate and re-score it here, so a stale or
        % differently-sampled record cannot contribute a number to the
        % report.
        for i = 1:numel(opts.resolved_mat)
            Z = load(opts.resolved_mat{i}, 'R');
            dk = tmp;
            if ~isempty(opts.deck_dir)
                dk = fullfile(opts.deck_dir, ...
                     sprintf('afocal4_clear_t%+03.0f.in', Z.R.D.tilt_deg*10));
            end
            R.resolved(end+1) = point_(P, Z.R.D, dk, F, opts, dk); %#ok<AGROW>
        end
        [~, io] = sort([R.resolved.tilt_deg]);
        R.resolved = R.resolved(io);
    end

    if opts.fig, R.fig = figure_(R, opts); end
end

% =====================================================================
function s = point_(P, D, deck, F, opts, keep, S)
%POINT_  Build (or take) one deck, gate it, score it, and read the offset
%   the tilt actually put in.  S, when given, is a finished CLEAR_SOLVE and
%   its score is used verbatim rather than recomputed at different sampling.
    if nargin < 7 || isempty(S)
        clear_build(P, D, deck, 'axis',opts.axis, 'verify',false);
        S = struct('S', afocal4_score(P, deck, 'fields',F, ...
                                      'nodes',P.solve.nodes_score));
    end
    Kb = afocal4_union(deck, 'fields',F, 'body_k',1.0, 'body_pad',0.0, ...
                       'quiet',true);
    Km = afocal4_union(deck, 'fields',F, 'body_k',opts.body_k, ...
                       'body_pad',opts.body_pad, 'init',false, 'quiet',true);
    nE = numel(Kb.foot_r);
    L  = clear_law(deck, 'fields',F, 'leg',2, 'elt',nE-1, 'init',false, ...
                   'quiet',true);
    sc = S.S;
    s = struct('tilt_deg',D.tilt_deg, 'floor_bare',Kb.floor_m, ...
               'floor_body',Km.floor_m, 'offset_mm',L.offset_m*1e3, ...
               'ratio',L.ratio, 'wfe_nm',sc.wfe_max_nm, 'blur_um',sc.blur_um, ...
               'breathe_pct',sc.breathe_pct, 'wander_um',sc.wander_um, ...
               'mag',sc.mag_centre_chief, 'nLost',Km.nLost, ...
               'worst',Km.worst_name, 'deck',keep, 'score',sc, 'D',D);
    if ~opts.quiet
        fprintf(['  %+7.2f %9.2f %9.2f %9.1f %9.1f %9.1f %8.4f %9.1f %9.5f ' ...
                 '%5d  %s\n'], s.tilt_deg, s.floor_bare*1e3, s.floor_body*1e3, ...
                s.offset_mm, s.wfe_nm, s.blur_um, s.breathe_pct, s.wander_um, ...
                s.mag, s.nLost, s.worst);
    end
end

function head_()
    fprintf('  %7s %9s %9s %9s %9s %9s %8s %9s %9s %5s  %s\n', 'tilt deg', ...
            'bare mm','body mm','offset','WFE nm','blur um','breathe%', ...
            'wander um','M','lost','binding pair');
end

function f = figure_(R, opts)
%FIGURE_  The exchange rate on one page: clearance won (left axis) against
%   the pupil terms paid (right axis, normalised to the untilted design), so
%   "what it buys" and "what it costs" are read off the same abscissa.
    f = figure('Position',[80 80 1180 520], 'Color','w');
    tl = tiledlayout(f,1,2,'Padding','compact','TileSpacing','compact');
    p0 = R.parent;

    ax = nexttile(tl);
    x = [R.raw.tilt_deg];
    yyaxis(ax,'left');
    plot(ax, x, [R.raw.floor_body]*1e3, '-o','LineWidth',1.8,'MarkerSize',4, ...
         'MarkerFaceColor','auto');
    hold(ax,'on');
    plot(ax, x, [R.raw.floor_bare]*1e3, '--s','LineWidth',1.2,'MarkerSize',4);
    yline(ax, 0, 'k-', 'LineWidth',1.0);
    ylabel(ax, 'union clearance floor  (mm)');
    yyaxis(ax,'right');
    plot(ax, x, [R.raw.blur_um]/max(p0.blur_um,eps), '-^','LineWidth',1.6, ...
         'MarkerSize',4);
    ylabel(ax, 'pupil blur, relative to the untilted design');
    xlabel(ax, 'extraction tilt on the field mirror  (deg)');
    grid(ax,'on');   box(ax,'on');
    title(ax, 'the RAW exchange rate');
    legend(ax, {'floor, declared body','floor, bare lit glass','zero', ...
                'pupil blur / parent'}, 'Location','northeast', 'Box','off');

    ax2 = nexttile(tl);
    if isempty(R.resolved)
        axis(ax2,'off');   title(ax2,'re-solved points -- not run');
    else
        nm = {'WFE','blur','breathe','wander'};
        g  = @(s) [s.wfe_nm/p0.wfe_nm, s.blur_um/p0.blur_um, ...
                   s.breathe_pct/p0.breathe_pct, s.wander_um/p0.wander_um];
        Raw = cell2mat(arrayfun(@(s) g(s), ...
              R.raw(ismember([R.raw.tilt_deg],[R.resolved.tilt_deg])).', ...
              'UniformOutput',false));
        Res = cell2mat(arrayfun(@(s) g(s), R.resolved.', 'UniformOutput',false));
        b = bar(ax2, [mean(Raw,1); mean(Res,1)].');
        b(1).FaceColor = [0.85 0.33 0.10];   b(2).FaceColor = [0.10 0.45 0.75];
        set(ax2, 'XTickLabel', nm);
        ylabel(ax2, 'relative to the untilted delivered design');
        yline(ax2, 1, 'k--');
        grid(ax2,'on');   box(ax2,'on');
        title(ax2, sprintf('what a re-solve buys back (tilt %s deg)', ...
              strjoin(arrayfun(@(v) sprintf('%+.0f',v), ...
              [R.resolved.tilt_deg], 'UniformOutput',false), ', ')));
        legend(ax2, {'raw','re-solved'}, 'Location','northwest','Box','off');
    end
    annotation(f,'textbox',[0.02 0.955 0.96 0.04], 'String', ...
        ['an extraction tilt buys clearance the field-walk law forbids -- ' ...
         'and is paid for in the pupil, not the wavefront'], ...
        'HorizontalAlignment','center','EdgeColor','none', ...
        'FontWeight','bold','FontSize',11);
    tl.OuterPosition = [0 0 1 0.94];
    if ~isempty(opts.save)
        exportgraphics(f, opts.save, 'Resolution',150);
        fprintf('  wrote %s\n', opts.save);
    end
end

function D = setf_(D, f, v),  D.(f) = v;  end
function del_(p),  if exist(p,'file'), delete(p); end,  end
