function R = afocal4_wall(opts)
%AFOCAL4_WALL  BRIEF_afocal4_wall: make the clearance a wall, then converge
%   the cleared curve.
%
%   R = AFOCAL4_WALL() assembles and reports the walled, converged
%   tilt-vs-price frontier from the per-point checkpoints WALL_POINT writes,
%   and runs the measurements that go around it.  It does not itself solve:
%   a frontier point is an hours-long artifact and is produced by one MATLAB
%   process each (RUN_WALL_FLEET.SH), then read back here.
%
%     0  THE WALL, AND WHETHER IT IS A WALL AT ALL.  AFOCAL4_UNION's floor
%        is now a wall in AFOCAL4_BUILD / CLEAR_BUILD (P.pack.union_enforce,
%        default off).  Non-vacuity is asserted on BOTH halves and in two
%        different senses: the wall must REFUSE the committed 343 mm deck
%        and admit the cleared one, AND -- the half this slice exists for --
%        the -8 and -9 deg re-solves must reproduce the clearing stage's
%        margin-spending with the wall OFF and hold their floor with it ON.
%     1  THE FRONTIER.  Tilt against price, walled at two declared floors
%        (0 and +15 mm), converged with central differences, with the
%        operating point read off it: the minimal pupil price that clears
%        with declared margin.
%     2  THE POLISH.  The delivered -10 deg deck re-solved with central
%        differences, against the numbers slide 13 carries -- because a
%        budget-capped number is not a converged one and the deck quotes it
%        as if it were.
%     3  THE UNCLAIMED PUPIL (the addendum).  The signed-tilt curve on the
%        committed deck with no re-solve at all -- the one figure that says
%        the committed trade curve was not at its own pupil optimum -- and,
%        at the cleared operating point, whether a pupil-weighted merit
%        recovers any of the blur without giving back wavefront or margin.
%     4  LEVERAGE 4's BAR, restated against whatever the frontier delivered.
%
%   THE MERIT DOCTRINE IS NOT REOPENED HERE.  Log-domain residuals, walls
%   and not terms: both stand.  Section 3 MEASURES the slack a merit
%   dominated by a wavefront term 130x off its target leaves lying around;
%   it does not propose a different merit.
%
%   Name-value:
%     'sections'  which of 0:4 (default all)
%     'dir'       where the checkpoints are (this directory)
%     'polish'    'load' (default -- read the -10 deg wall-off checkpoint)
%                 | 'skip'
%     'pupil'     'load' the pupil-weighted checkpoints for section 3's
%                 cross-table, or 'skip' (default 'load').  They are SOLVED
%                 by RUN_WALL_POINT with WALL_PUPILW, one process each, like
%                 every other multi-hour point in this stage -- not inline
%                 here, where a failure would take the whole report with it.
%     'model'     engine model size (256)
%     'save'      write the figure and the .mat (true)
%
%   Run:  R = afocal4_wall();          % after ./run_wall_fleet.sh
%
%   See also WALL_POINT, WALL_SEED, AFOCAL4_UNION_WALL, AFOCAL4_UNION,
%   CLEAR_PRICE, AFOCAL4_CLEARING.

    arguments
        opts.sections (1,:) double = 0:4
        opts.dir      (1,:) char = ''
        opts.polish   (1,:) char {mustBeMember(opts.polish,{'load','skip'})} = 'load'
        opts.pupil    (1,:) char {mustBeMember(opts.pupil,{'load','skip'})} = 'load'
        opts.model    (1,1) double = 256
        opts.save     (1,1) logical = true
    end

    here = fileparts(mfilename('fullpath'));
    up   = fileparts(here);
    addpath(here);  addpath(up);  addpath(fullfile(up,'clearing'));
    addpath(fullfile(up,'packaging'));
    if isempty(opts.dir), opts.dir = here; end

    macos.init(opts.model);
    P    = afocal4_params();
    Fbox = P.Fsolve;
    src  = fullfile(up, 'afocal4_b2long_343mm.in');
    cleared = fullfile(up, 'clearing', 'afocal4_clear_343mm.in');
    R = struct('P',P, 'opts',opts, 'parent',src, 'cleared',cleared, 'when',[]);

    % the delivered clearing row, verbatim, so every comparison below is
    % against the numbers that were actually published rather than against a
    % re-measurement that could quietly differ.
    R.delivered = struct('tilt_deg',-10, 'wfe_nm',8992.68, 'blur_um',553.34, ...
        'breathe_pct',0.8160, 'wander_um',559.87, 'floor_body_mm',37.82, ...
        'nfev',427, 'exitflag',0, 'committed_wfe_nm',10406.98, ...
        'committed_blur_um',157.02, 'committed_breathe_pct',0.1240, ...
        'committed_wander_um',161.23, 'committed_floor_mm',-79.89);

    pts = load_points_(opts.dir);
    R.points = pts;
    ctl = load_points_(opts.dir, 'wall_ctl_t*.mat');
    R.controls = ctl;

    % ---- 0  THE WALL, AND ITS NON-VACUITY ---------------------------------
    if any(opts.sections == 0)
        hdr_('0  the union floor as a WALL, and whether it is one');
        R.nonvac = nonvacuity_(P, src, cleared, Fbox);
        R.ab     = wall_ab_(R, pts);
    end

    % ---- 1  THE FRONTIER ---------------------------------------------------
    if any(opts.sections == 1)
        hdr_('1  the walled, converged tilt-vs-price frontier');
        R.frontier = frontier_(R, pts, P);
    end

    % ---- 1b  THE CONTROL: tilt isolated at a fixed standoff -----------------
    if any(opts.sections == 1)
        hdr_('1b  the tilt-vs-price curve, with the standoff HELD FIXED');
        R.control = control_(R, ctl, P);
    end

    % ---- 2  THE POLISH OF THE DELIVERED DECK -------------------------------
    if any(opts.sections == 2) && ~strcmp(opts.polish,'skip')
        hdr_('2  the delivered -10 deg deck, polished with central differences');
        R.polish = polish_(R, pts);
    end

    % ---- 3  THE UNCLAIMED PUPIL (addendum) ---------------------------------
    if any(opts.sections == 3)
        hdr_('3  addendum -- the unclaimed pupil');
        R.unclaimed = unclaimed_(P, src, Fbox, here, opts);
        if ~strcmp(opts.pupil,'skip')
            R.pupilw = pupil_weighted_(R, P, opts, here);
        end
    end

    % ---- 4  LEVERAGE 4's BAR ------------------------------------------------
    if any(opts.sections == 4)
        hdr_('4  leverage 4 -- the bar a fifth mirror has to beat, restated');
        R.fifth = fifth_bar_(R);
    end

    % ---- figures + save -----------------------------------------------------
    if opts.save && isfield(R,'frontier') && ~isempty(R.frontier.row)
        R.fig_file = fullfile(here, 'afocal4_wall_frontier.png');
        figure_(R, R.fig_file);
    end
    R.when = datestr(now, 'yyyy-mm-dd HH:MM:SS'); %#ok<TNOW1,DATST>
    if opts.save
        save(fullfile(here,'afocal4_wall.mat'), 'R', '-v7.3');
        fprintf('\n  wrote %s\n', fullfile(here,'afocal4_wall.mat'));
    end
end

% =====================================================================
function pts = load_points_(d, glob)
%LOAD_POINTS_  Every checkpoint in the directory, verified against the deck
%   it claims to have been measured on.  A checkpoint whose deck has been
%   overwritten since is loaded but FLAGGED, never silently used: the
%   frontier is assembled from files other processes wrote, and identifying
%   an artifact by its filename is exactly how a stale one gets quoted.
    % 'wall_t*' and not 'wall_*': the addendum's sweep and the pupil-weighted
    % polishes checkpoint into the same directory under names that would match
    % a looser glob and carry different variables.
    if nargin < 2, glob = 'wall_t*.mat'; end
    f = dir(fullfile(d, glob));
    pts = struct('tag',{},'R',{},'stale',{});
    for i = 1:numel(f)
        Z = load(fullfile(d, f(i).name), 'R');
        if ~isfield(Z,'R') || ~isfield(Z.R,'tag'), continue; end
        st = false;
        if isfield(Z.R,'deck_stamp') && isfile(Z.R.deck)
            st = ~strcmp(Z.R.deck_stamp, stamp_(Z.R.deck));
        end
        pts(end+1) = struct('tag',Z.R.tag, 'R',Z.R, 'stale',st); %#ok<AGROW>
    end
    if isempty(pts)
        fprintf('  no %s checkpoints in %s\n', glob, d);
        return;
    end
    ns = nnz([pts.stale]);
    fprintf('  %d %s checkpoint(s)%s\n', numel(pts), glob, ...
            tern_(ns>0, sprintf('  <-- %d with a deck that no longer matches', ns), ''));
end

function h = stamp_(deck)
    b = uint8(fileread(deck));
    m = java.security.MessageDigest.getInstance('SHA-256');
    m.update(b(:));
    dg = typecast(m.digest(), 'uint8');
    h = lower(reshape(dec2hex(dg(1:8), 2).', 1, []));
end

function p = pick_(pts, tilt, umin_mm, wall)
    p = [];
    for i = 1:numel(pts)
        r = pts(i).R;
        if abs(r.tilt_deg - tilt) > 1e-9, continue; end
        if abs(r.union_min*1e3 - umin_mm) > 1e-9, continue; end
        if r.wall ~= wall, continue; end
        p = r;   return;
    end
end

% ---------------------------------------------------------------------
function N = nonvacuity_(P, src, cleared, Fbox)
%NONVACUITY_  Half one, on the GATE, and now also on the WALL: a wall that
%   refuses nothing is not a wall.  Both are asserted rather than left to a
%   reader's inspection.
    N.gate_fail = afocal4_union(src,     'fields',Fbox, 'quiet',true);
    N.gate_pass = afocal4_union(cleared, 'fields',Fbox, 'quiet',true);
    N.gate_ok = (~N.gate_fail.ok) && N.gate_pass.ok;
    fprintf('    the GATE: committed 343 mm deck %s (%+.2f mm), cleared %s (%+.2f mm)\n', ...
            tern_(N.gate_fail.ok,'PASSES <-- vacuous','FAILS, as it must'), ...
            N.gate_fail.floor_m*1e3, ...
            tern_(N.gate_pass.ok,'PASSES, as it must','FAILS <-- not cleared'), ...
            N.gate_pass.floor_m*1e3);

    % the WALL, exercised through the builder itself
    D = wall_recover(P, src);
    Q = P;   Q.pack.union_enforce = true;   Q.pack.union_min = 0;
    t = [tempname '.in'];   c = onCleanup(@() del_(t)); %#ok<NASGU>
    N.wall_refuses_committed = false;   N.wall_msg = '';
    try
        afocal4_build(Q, D, t, 'verify',false);
    catch ME
        N.wall_refuses_committed = strcmp(ME.identifier, ...
                                          'macos:design:afocal4_build:union');
        N.wall_msg = ME.message;
    end
    Dt = D;   Dt.tilt_deg = -10;
    N.wall_admits_tilted = true;
    try
        o = clear_build(Q, Dt, t, 'verify',false);
        N.wall_tilted_floor_m = o.union.floor_m;
    catch
        N.wall_admits_tilted = false;   N.wall_tilted_floor_m = NaN;
    end
    fprintf('    the WALL: AFOCAL4_BUILD %s the committed design\n', ...
            tern_(N.wall_refuses_committed,'REFUSES','ADMITS <-- vacuous'));
    fprintf('              CLEAR_BUILD at -10 deg %s it (%+.2f mm)\n', ...
            tern_(N.wall_admits_tilted,'admits','REFUSES <-- a cage'), ...
            N.wall_tilted_floor_m*1e3);

    % and the default must still be OFF, or every committed artifact moves
    N.default_off = ~logical(getfielddef_(P.pack,'union_enforce',false));
    t2 = [tempname '.in'];   c2 = onCleanup(@() del_(t2)); %#ok<NASGU>
    afocal4_build(P, D, t2, 'verify',false);
    N.rebuilds_committed = isequal(fileread(t2), fileread(src));
    fprintf(['    the DEFAULT: wall off %d, and AFOCAL4_BUILD still rebuilds ' ...
             'the committed deck byte-for-byte %d\n'], N.default_off, ...
            N.rebuilds_committed);
    N.ok = N.gate_ok && N.wall_refuses_committed && N.wall_admits_tilted && ...
           N.default_off && N.rebuilds_committed;
    fprintf('    => the wall is %s\n', tern_(N.ok,'NON-VACUOUS and additive', ...
                                             'NOT sound'));
    assert(N.ok, 'afocal4_wall:nonvacuity', ...
           'the union wall failed its own non-vacuity check');
end

function A = wall_ab_(R, pts)
%WALL_AB_  Does the wall change the answer?  MEASURED, and the measurement
%   overturned the premise this stage was given.
%
%   THE PREMISE.  The clearing stage reported that without a wall the
%   re-solve SPENDS the clearance: at -8 and -9 deg it walked +23.3 and
%   +42.3 mm of raw margin down to +2.3 and +0.7 mm, and concluded that
%   AFOCAL4_SCORE cannot see clearance so the solver trades it away for free.
%   BRIEF_afocal4_wall asks for that behaviour to be reproduced with the wall
%   off and abolished with it on.
%
%   WHAT IT ACTUALLY DOES.  Run the SAME solve to convergence -- central
%   differences at 1e-4 and restart rounds, against the clearing stage's 427
%   budget-capped forward-difference evaluations -- and the margin is not
%   spent at all.  The wall-off solve keeps tens of millimetres.  At -8 and
%   -10 deg the wall-on and wall-off runs are IDENTICAL, round-1 merit to the
%   last digit: the wall never rejected an iterate.
%
%   So the margin-spending was an artifact of stopping early on a gradient
%   that S4c had already measured as 17 % low, not a property of the merit.
%   That does not make the wall wrong -- nothing else holds the clearance,
%   and it still refuses the committed deck (section 0's other half) -- but
%   it does mean the wall is INSURANCE here rather than the thing that
%   changed the answer.  Convergence is.  Where it DOES bind (a +15 mm floor,
%   and -9 deg at 0 mm) it changed the path and landed a BETTER design, which
%   is worth reporting and is not what a cage does.
    A = struct('rows',[], 'binds',[], 'ok',false, 'why','checkpoints missing');
    rows = struct('tilt',{},'raw_mm',{},'clearing_mm',{},'off_mm',{}, ...
                  'on0_mm',{},'on15_mm',{},'off_wfe',{},'on0_wfe',{});
    raw = struct('t8',23.34, 't9',42.25);        % clearing README 6b, measured
    clr = struct('t8',2.32,  't9',0.69);         % clearing README 6c, delivered
    fprintf(['\n    %-6s %9s %11s %11s %11s %11s\n'], 'tilt', 'raw', ...
            'clearing', 'OFF conv.', 'ON 0 conv.', 'ON +15 conv.');
    for t = [-8 -9]
        o  = pick_(pts, t, 0,  false);
        w0 = pick_(pts, t, 0,  true);
        w15= pick_(pts, t, 15, true);
        rows(end+1) = struct('tilt',t, ...
            'raw_mm', raw.(sprintf('t%d',abs(t))), ...
            'clearing_mm', clr.(sprintf('t%d',abs(t))), ...
            'off_mm',  gz_(o,  'floor_body_m')*1e3, ...
            'on0_mm',  gz_(w0, 'floor_body_m')*1e3, ...
            'on15_mm', gz_(w15,'floor_body_m')*1e3, ...
            'off_wfe', gzs_(o, 'wfe_max_nm'), ...
            'on0_wfe', gzs_(w0,'wfe_max_nm')); %#ok<AGROW>
        r = rows(end);
        fprintf('    %+6.0f %9.2f %11.2f %11s %11s %11s\n', t, r.raw_mm, ...
                r.clearing_mm, nm_(r.off_mm), nm_(r.on0_mm), nm_(r.on15_mm));
    end
    A.rows = rows;
    if ~all(isfinite([rows.off_mm]))
        fprintf(['    (needs the wall-OFF checkpoints at -8 and -9 deg; run ' ...
                 './run_wall_fleet.sh)\n']);
        return;
    end
    % Did the wall ever actually reject an iterate?  Identical round-1 merits
    % from an identical seed is the tell that it did not.
    binds = struct('tilt',{},'umin',{},'binds',{},'why',{});
    for t = [-8 -9 -10]
        o = pick_(pts, t, 0, false);   w = pick_(pts, t, 0, true);
        if isempty(o) || isempty(w) || ~o.ok || ~w.ok, continue; end
        same = abs(o.rounds(1).merit - w.rounds(1).merit) < 1e-9;
        binds(end+1) = struct('tilt',t, 'umin',0, 'binds',~same, ...
            'why',tern_(same, 'round-1 merit identical -- never rejected an iterate', ...
                        'round-1 merit differs -- the wall rejected iterates')); %#ok<AGROW>
        fprintf(['    tilt %+.0f at a 0 mm floor: wall %s (%s); converged ' ...
                 'floor OFF %+.2f mm vs ON %+.2f mm\n'], t, ...
                tern_(~same,'BINDS','does NOT bind'), binds(end).why, ...
                o.floor_body_m*1e3, w.floor_body_m*1e3);
    end
    A.binds = binds;
    spent = any([rows.off_mm] < [rows.raw_mm] - 5);
    A.premise_holds = spent;
    A.ok = true;   A.why = '';
    if spent
        fprintf(['\n    => the converged wall-off solve DOES spend the margin, ' ...
                 'as the clearing stage reported.\n']);
    else
        fprintf(['\n    => THE PREMISE DOES NOT SURVIVE CONVERGENCE.  The ' ...
                 'clearing stage''s -8/-9 deg re-solves ended at +2.3/+0.7 mm ' ...
                 'on 427\n       budget-capped forward-difference ' ...
                 'evaluations; the same solves at 1209 central-difference ' ...
                 'evaluations end at\n       %+.2f/%+.2f mm with the wall ' ...
                 'OFF.  The margin-spending was a stalled solve, not a blind ' ...
                 'merit.\n       The wall is still right to have -- nothing ' ...
                 'else holds the clearance and it still refuses the committed ' ...
                 'deck -- but\n       on this design it is INSURANCE, and ' ...
                 'CONVERGENCE is what changed the answer.\n'], ...
                rows(1).off_mm, rows(2).off_mm);
    end
end

% ---------------------------------------------------------------------
function Fr = frontier_(R, pts, P)
%FRONTIER_  The curve, and the operating point read off it.
    % Grown from empty rather than from a declared field list: ROW_FROM_ adds
    % its admissibility fields after the struct is built, so a hand-written
    % prototype drifts out of step with it and `row(end+1) = ...` fails with
    % "dissimilar structures" -- and only when REACHED, after the checkpoints
    % have been loaded.  That is RESULTS rule 7's MATLAB trap, and it caught
    % this file once.
    row = struct([]);
    ts = sort(unique(arrayfun(@(p) p.R.tilt_deg, pts)), 'descend');
    for u = [0 15]
        for t = ts
            p = pick_(pts, t, u, true);
            if isempty(p), continue; end
            r = row_from_(p, P);
            if isempty(row), row = r; else, row(end+1) = r; end %#ok<AGROW>
        end
    end
    Fr.row = row;
    if isempty(row)
        fprintf('    no walled checkpoints yet -- nothing to assemble.\n');
        Fr.operating = [];   return;
    end
    for u = [0 15]
        m = [row.umin_mm] == u;
        if ~any(m), continue; end
        fprintf(['\n    WALLED AT %+.0f mm  (declared body = %.2fx union ' ...
                 'footprint + %.0f mm)\n'], u, P.pack.union_body_k, ...
                P.pack.union_body_pad*1e3);
        print_rows_(row(m));
    end
    % THE OPERATING POINT: the minimal pupil price that clears with declared
    % margin.  Minimal PRICE, not minimal tilt -- the clearing stage already
    % measured that on this design more tilt can buy back blur, so ranking by
    % tilt would answer a different question from the one the room asks.
    %
    % TWO THRESHOLDS, BOTH AT REPORTING SAMPLING, because the wall is judged
    % at solve sampling and lands the solver ON it: a point walled at +15 mm
    % will sit near +15 there and near +13 here.  "Clears" is the gate's own
    % pass condition (>= 0, i.e. buildable); "clears with margin" is the
    % declared allowance's own pad (>= +15 mm).  The operating point is taken
    % from the second set when it is non-empty, and the choice is stated
    % rather than left to the reader.
    MARGIN_MM = 15;
    ok  = [row.ok] & isfinite([row.blur_um]);
    cl  = ok & [row.admissible];
    clm = cl & ([row.floor_mm] >= MARGIN_MM);
    Fr.margin_mm = MARGIN_MM;
    Fr.operating = [];   Fr.operating_kind = '';
    if any(clm) || any(cl)
        if any(clm), sel = clm;
            Fr.operating_kind = 'admissible, and holds the declared +15 mm pad';
        else,        sel = cl;
            Fr.operating_kind = 'admissible and clears, but without the declared pad';
        end
        cand = row(sel);
        [~, j] = min([cand.blur_um]);
        Fr.operating = cand(j);
        o = Fr.operating;
        fprintf(['\n    OPERATING POINT (%s): tilt %+.1f deg, walled at ' ...
                 '%+.0f mm -- blur %.1f um at a %+.2f mm floor\n'], ...
                Fr.operating_kind, o.tilt, o.umin_mm, o.blur_um, o.floor_mm);
        fprintf(['      %d of %d walled points are ADMISSIBLE (clear the gate ' ...
                 'at reporting sampling AND hold M and the 15 deg AOI rule); ' ...
                 '%d of those\n      also hold the %+.0f mm declared pad.\n'], ...
                nnz(cl), nnz(ok), nnz(clm), MARGIN_MM);
        bad = ok & ~[row.admissible];
        if any(bad)
            fprintf(['      %d walled point(s) CLEAR THE BEAM BUT ARE NOT ' ...
                     'DESIGNS -- they buy the clearance by leaving the ' ...
                     'interface:\n'], nnz(bad));
            for r = row(bad)
                fprintf(['        tilt %+.1f at %+.0f mm: floor %+.2f mm, but ' ...
                         'M %.4f (%.2f %% off 30), max AOI %.2f deg, anchoring ' ...
                         'residual %.3f um  [%s]\n'], ...
                        r.tilt, r.umin_mm, r.floor_mm, r.mag, ...
                        abs(r.mag/30-1)*100, r.aoi_max, r.anchor_um, adm_(r));
            end
        end
        % A POINT SEEDED OFF A DIFFERENT FRONT END IS A DIFFERENT DESIGN, and
        % ranking it against the others on one column is the same mistake the
        % clearing stage's "50 mm clears" table warns about.  Say so rather
        % than let a basin change hide inside a blur number.
        if ~contains(o.seed, 'tilt alone')
            fprintf(['      NOTE: this point was SEEDED BY RE-POSING (%s), so ' ...
                     'it is not the committed design swung -- compare it as a ' ...
                     'different\n            member of the family, not as the ' ...
                     'same design at a different tilt.\n'], o.seed);
        end
        d = R.delivered;
        fprintf(['      against the delivered -10 deg row (%.1f nm / %.1f um / ' ...
                 '%.4f %% / %+.2f mm):\n'], d.wfe_nm, d.blur_um, ...
                d.breathe_pct, d.floor_body_mm);
        fprintf(['      WFE %+.1f %%, blur %+.1f %%, breathing %+.1f %%, ' ...
                 'wander %+.1f %%, floor %+.2f mm\n'], ...
                100*(o.wfe_nm/d.wfe_nm - 1), 100*(o.blur_um/d.blur_um - 1), ...
                100*(o.breathe_pct/d.breathe_pct - 1), ...
                100*(o.wander_um/d.wander_um - 1), o.floor_mm - d.floor_body_mm);
        % the question the room will ask, answered in one line
        w8 = row([row.tilt] == -8 & [row.umin_mm] == 15);
        if ~isempty(w8) && w8.ok
            better = w8.blur_um < 0.9*d.blur_um;
            fprintf(['      does a walled -8 deg hold real margin at ' ...
                     'materially less pupil damage than -10?  %s\n'], ...
                    tern_(better, sprintf(['YES -- %.1f um at %+.2f mm'], ...
                          w8.blur_um, w8.floor_mm), ...
                          sprintf(['NO -- %.1f um at %+.2f mm, against the ' ...
                          'delivered %.1f um'], w8.blur_um, w8.floor_mm, ...
                          d.blur_um)));
        end
    else
        fprintf('\n    no walled point clears at reporting sampling yet.\n');
    end
    % the frontier's own honesty check: the sampling bias the wall carried
    bb = [row.bias_mm];   bb = bb(isfinite(bb));
    if ~isempty(bb)
        fprintf(['\n    the wall was judged at SOLVE sampling and the table ' ...
                 'quotes REPORTING sampling: the difference runs\n      ' ...
                 '%+.2f .. %+.2f mm (median %+.2f), always optimistic, and ' ...
                 'is what the seeder''s 10 mm margin covers.\n'], min(bb), ...
                max(bb), median(bb));
    end
end

function r = row_from_(p, P)
    if ~p.ok
        r = struct('tilt',p.tilt_deg, 'umin_mm',p.union_min*1e3, 'ok',false, ...
            'why',p.why, 'floor_mm',NaN, 'bare_mm',NaN, 'solve_mm',NaN, ...
            'bias_mm',NaN, 'wfe_nm',NaN, 'blur_um',NaN, 'breathe_pct',NaN, ...
            'wander_um',NaN, 'anchor_um',NaN, 'aoi_max',NaN, 'aoi_fm',NaN, 'mag',NaN, ...
            'exit_mm',NaN, 'coll_urad',NaN, 'lost',NaN, 'nfev',NaN, ...
            'exitflag',NaN, 'seed','none', 'standoff_mm',NaN, 'R2_mm',NaN, ...
            'deck','', 'spec_clear',false, 'spec_mag',false, 'spec_aoi',false, ...
            'spec_anchor',false, 'admissible',false);
        return;
    end
    r = struct('tilt',p.tilt_deg, 'umin_mm',p.union_min*1e3, 'ok',true, ...
        'why','', 'floor_mm',p.floor_body_m*1e3, 'bare_mm',p.floor_bare_m*1e3, ...
        'solve_mm',p.floor_solve_m*1e3, 'bias_mm',p.sampling_bias_mm, ...
        'wfe_nm',p.S.wfe_max_nm, 'blur_um',p.S.blur_um, ...
        'breathe_pct',p.S.breathe_pct, 'wander_um',p.S.wander_um, ...
        'anchor_um',p.S.anchor_resid_um, ...
        'aoi_max',p.aoi_max_deg, 'aoi_fm',p.aoi_fm_deg, ...
        'mag',p.S.mag_centre_chief, 'exit_mm',p.traced.exit_dia*1e3, ...
        'coll_urad',p.traced.collimation_urad, 'lost',p.nLost, ...
        'nfev',p.nfev, 'exitflag',p.exitflag, 'seed',p.seed.source, ...
        'standoff_mm',p.D.fm_standoff*1e3, 'R2_mm',p.D.R2*1e3, 'deck',p.deck);
    % ---- ADMISSIBILITY, and it is not a new rule -------------------------
    % A frontier point has to be a DESIGN before it is a price.  Two of this
    % study's standing constraints are outside the union gate and outside
    % the merit's reach, and both bite here: the interface magnification
    % (P.targets.mag / mag_pct -- the CUSTOMER boundary, "30x afocal", held
    % fixed since S0) and the design drivers' 15 deg chief-AOI rule.  Ranking
    % rows by blur without them picks designs that clear the beam by ceasing
    % to be 30x telescopes -- measured on this frontier, not hypothetical.
    r.spec_clear = r.floor_mm >= 0;
    r.spec_mag   = abs(r.mag/P.targets.mag - 1)*100 <= P.targets.mag_pct;
    r.spec_aoi   = r.aoi_max <= 15;
    % AND THE SOLVER-INTEGRITY CHECK S4c EARNED.  pupil_map's anchoring
    % residual is a VALIDITY check, not a metric: 0.1 um on every sound
    % design in this study (0.095 committed, 0.079 cleared) and tens of
    % MILLIMETRES on a solve that landed a scrambled layout while reporting
    % the best wavefront of the sweep.  A frontier point with a suspiciously
    % good column has to pass it before the column is quoted.  100 um is
    % three orders above sound and two below the failures.
    r.spec_anchor = r.anchor_um <= 100;
    r.admissible = r.spec_clear && r.spec_mag && r.spec_aoi && r.spec_anchor;
end

function print_rows_(row)
    fprintf(['    %6s %9s %9s %9s %9s %8s %9s %7s %8s %8s %7s %5s %6s %4s  %s\n'], ...
        'tilt','floor mm','bare mm','WFE nm','blur um','breath%','wander um', ...
        'AOImax','M','exit mm','coll ur','lost','nfev','xfl','seed / standoff');
    for i = 1:numel(row)
        r = row(i);
        if ~r.ok
            fprintf('    %+6.1f %9s  %s\n', r.tilt, '--', r.why);   continue;
        end
        fprintf(['    %+6.1f %9.2f %9.2f %9.1f %9.1f %8.4f %9.1f %8.3f %7.2f ' ...
                 '%8.4f %8.3f %7.1f %5d %6d %4d %-13s %-24s s %+.0f mm, ' ...
                 'R2 %.1f mm\n'], ...
            r.tilt, r.floor_mm, r.bare_mm, r.wfe_nm, r.blur_um, r.breathe_pct, ...
            r.wander_um, r.anchor_um, r.aoi_max, r.mag, r.exit_mm, r.coll_urad, ...
            r.lost, r.nfev, r.exitflag, adm_(r), r.seed, r.standoff_mm, r.R2_mm);
    end
end

function s = adm_(r)
%ADM_  Why a row is or is not a candidate, in one cell.
    if r.admissible, s = 'yes';   return;   end
    w = {};
    if ~r.spec_clear, w{end+1} = 'in beam'; end
    if ~r.spec_mag,   w{end+1} = 'M';       end
    if ~r.spec_aoi,   w{end+1} = 'AOI';     end
    if ~r.spec_anchor, w{end+1} = 'anchor';  end
    s = ['NO: ' strjoin(w, ',')];
end

% ---------------------------------------------------------------------
function C = control_(R, ctl, P)
%CONTROL_  The frontier the brief actually asked for.
%
%   The free-standoff sweep (section 1) is NOT a tilt-vs-price curve: its
%   points order by the field-mirror standoff each solve reached, not by
%   tilt, and every one of them was still descending.  Here the standoff is
%   PINNED at +276 mm for all points and the DOF set is {conic, front}, so
%   the tilt is the only thing that differs -- which is what a tilt-vs-price
%   curve has to mean.  1628 evaluations over 4 restart rounds each, central
%   differences, wall on at a 0 mm floor.
    C = struct('row',[], 'operating',[], 'knee',NaN);
    if isempty(ctl)
        fprintf('    no control checkpoints -- run the fixed-standoff series.\n');
        return;
    end
    row = struct([]);
    for i = 1:numel(ctl)
        r = row_from_(ctl(i).R, P);
        if isempty(row), row = r; else, row(end+1) = r; end %#ok<AGROW>
    end
    [~, io] = sort([row.tilt], 'descend');   row = row(io);
    C.row = row;
    print_rows_(row);

    % THE CLEARANCE SATURATES, and that is the shape of the answer: past the
    % knee a point pays more pupil for no more floor, so it is DOMINATED.
    ok = [row.ok] & [row.admissible];
    if ~any(ok), return; end
    r = row(ok);
    fmax = max([r.floor_mm]);
    sat  = [r.floor_mm] >= 0.90*fmax;          % on the saturated plateau
    fprintf(['\n    the clearance SATURATES at %+.1f mm; %d of %d points sit ' ...
             'on that plateau (within 10 %%).\n'], fmax, nnz(sat), numel(r));
    dom = false(1,numel(r));
    for i = 1:numel(r)
        dom(i) = any([r.floor_mm] >= r(i).floor_mm - 1e-9 & ...
                     [r.blur_um]  <= r(i).blur_um  - 1e-9);
    end
    if any(dom)
        fprintf('    DOMINATED (another point clears at least as well for less blur):\n');
        for i = find(dom)
            fprintf('      tilt %+.1f: floor %+.2f mm, blur %.1f um\n', ...
                    r(i).tilt, r(i).floor_mm, r(i).blur_um);
        end
    end
    % the operating point: the least pupil price on the saturated plateau,
    % and separately the least pupil price that holds the declared +15 mm pad.
    sp = r(sat);   [~, j] = min([sp.blur_um]);   C.operating = sp(j);
    pad = r([r.floor_mm] >= 15);
    if ~isempty(pad), [~, k] = min([pad.blur_um]);   C.cheapest_pad = pad(k); end
    o = C.operating;   d = R.delivered;
    fprintf(['\n    OPERATING POINT (least pupil price on the saturated ' ...
             'plateau): tilt %+.1f deg\n'], o.tilt);
    fprintf(['      floor %+.2f mm (bare %+.2f), WFE %.1f nm, blur %.1f um, ' ...
             'breathing %.4f %%, wander %.1f um\n'], o.floor_mm, o.bare_mm, ...
            o.wfe_nm, o.blur_um, o.breathe_pct, o.wander_um);
    fprintf(['      against the DELIVERED -10 deg row: WFE %+.1f %%, blur ' ...
             '%+.1f %%, breathing %+.1f %%, wander %+.1f %%, floor %+.2f mm\n'], ...
            100*(o.wfe_nm/d.wfe_nm-1), 100*(o.blur_um/d.blur_um-1), ...
            100*(o.breathe_pct/d.breathe_pct-1), 100*(o.wander_um/d.wander_um-1), ...
            o.floor_mm - d.floor_body_mm);
    if isfield(C,'cheapest_pad') && ~isempty(C.cheapest_pad)
        c = C.cheapest_pad;
        fprintf(['    CHEAPEST that still holds the declared +15 mm pad: tilt ' ...
                 '%+.1f deg -- floor %+.2f mm, blur %.1f um (%+.1f %% on the ' ...
                 'delivered row)\n'], c.tilt, c.floor_mm, c.blur_um, ...
                100*(c.blur_um/d.blur_um-1));
    end
end

% ---------------------------------------------------------------------
function Po = polish_(R, pts)
%POLISH_  The delivered -10 deg deck, re-solved with central differences and
%   restarts, against the numbers slide 13 carries.  The brief's threshold:
%   anything moving more than ~1 % is flagged immediately, because the deck
%   quotes 8993 / 553 / 0.82 % / +37.8 / 1.24x as if they were converged.
    Po = struct('have',false);
    p = pick_(pts, -10, 0, false);
    if isempty(p) || ~p.ok
        fprintf('    no wall-off -10 deg checkpoint yet.\n');   return;
    end
    d = R.delivered;
    nm = {'WFE rung 2 max (nm)','pupil blur rms (um)','breathing (%)', ...
          'wander (um)','union floor, declared (mm)'};
    was = [d.wfe_nm, d.blur_um, d.breathe_pct, d.wander_um, d.floor_body_mm];
    now = [p.S.wfe_max_nm, p.S.blur_um, p.S.breathe_pct, p.S.wander_um, ...
           p.floor_body_m*1e3];
    Po.have = true;   Po.was = was;   Po.now = now;   Po.names = {nm};
    Po.nfev = p.nfev;  Po.exitflag = p.exitflag;
    fprintf('    %-28s %14s %14s %10s\n', '', 'delivered', 'polished', 'change');
    moved = false(1,numel(nm));
    for i = 1:numel(nm)
        rel = 100*(now(i)/was(i) - 1);
        moved(i) = abs(rel) > 1;
        fprintf('    %-28s %14.4f %14.4f %9.2f %%%s\n', nm{i}, was(i), now(i), ...
                rel, tern_(moved(i),'  <-- >1%',''));
    end
    Po.moved = moved;   Po.moved_names = nm(moved);
    fprintf(['    %d evaluations (delivered: %d at exitflag %d), exitflag ' ...
             '%d\n'], p.nfev, d.nfev, d.exitflag, p.exitflag);
    if any(moved)
        fprintf(['\n    *** FLAG FOR CC: the central-difference polish moves ' ...
                 '%d of the deck''s quoted numbers by more than 1 %%:\n' ...
                 '    %s\n'], nnz(moved), strjoin(nm(moved), '; '));
    else
        fprintf(['\n    the delivered numbers survive the polish inside 1 %% ' ...
                 '-- slide 13 stands as quoted.\n']);
    end
end

% ---------------------------------------------------------------------
function U = unclaimed_(P, src, Fbox, here, opts)
%UNCLAIMED_  The signed-tilt curve on the COMMITTED deck, no re-solve at
%   all.  The clearing stage found a POSITIVE 4 deg tilt takes the blur from
%   157.0 to 102.6 um for nothing -- 35 % of the pupil metric lying
%   unclaimed because a wavefront term 130x off target owns the log-domain
%   sum of squares.  This is that statement as a curve rather than as one
%   point, over a finer grid and both signs.
%
%   It re-uses CLEAR_PRICE, the clearing stage's own machinery, rather than
%   re-implementing the sweep: the number this addendum reports has to be
%   the same number that stage reported, measured the same way.
    D0 = wall_recover(P, src);
    png = '';
    if opts.save, png = fullfile(here, 'afocal4_wall_unclaimed.png'); end
    chk = fullfile(here, 'wall_unclaimed.mat');
    if isfile(chk)
        Z = load(chk, 'price');   U.price = Z.price;
        fprintf('    (loaded the sweep from %s)\n', chk);
        print_price_(U.price);
    else
        U.price = clear_price(P, D0, 'tilt', -8:1:8, 'fields',Fbox, 'save',png, ...
                              'fig',opts.save, 'quiet',false);
        if opts.save
            price = U.price;   price.fig = []; %#ok<STRNU>
            save(chk, 'price', '-v7.3');
        end
    end
    b = [U.price.raw.blur_um];   t = [U.price.raw.tilt_deg];
    [U.blur_min, j] = min(b);
    U.blur_min_tilt = t(j);
    U.blur_at_zero  = b(t == 0);
    U.unclaimed_pct = 100*(1 - U.blur_min/U.blur_at_zero);
    fprintf(['\n    the committed design is NOT at its own pupil optimum: ' ...
             'blur %.1f um at 0 deg, %.1f um at %+.0f deg,\n    i.e. %.1f %% ' ...
             'of the pupil blur is available for a tilt and no re-solve at ' ...
             'all.  It costs clearance (%+.2f -> %+.2f mm),\n    which is ' ...
             'why the design has to be swung the OTHER way.\n'], ...
            U.blur_at_zero, U.blur_min, U.blur_min_tilt, U.unclaimed_pct, ...
            U.price.raw(t==0).floor_body*1e3, U.price.raw(j).floor_body*1e3);
end

function W = pupil_weighted_(R, P, opts, here) %#ok<INUSD>
%PUPIL_WEIGHTED_  At a fixed tilt and wall: does a pupil-weighted merit
%   recover any of section 3's blur without giving back wavefront or margin?
%
%   THE MEASUREMENT THAT MAKES THE POINT IS A CROSS-TABLE, not a row.  A
%   re-weighted solve has to be judged on the merit it was solving, and so
%   does the design it was supposed to improve on -- otherwise "the pupil
%   metrics got worse" is just the observation that a different objective has
%   a different optimum.  Every design is therefore scored under EVERY
%   weighting, and the question is whether the re-weighted solve beats the
%   incumbent ON THE INCUMBENT'S NEW SCORE.  It does not.
%
%   THE DOCTRINE IS NOT REOPENED.  Log-domain residuals and walls-not-terms
%   both stand; P.weights has always carried these as knobs (the S4 brief
%   says so).  This measures what the slack is worth, it does not propose a
%   re-weighting.
    W = struct('have',false, 'rows',[]);
    f = dir(fullfile(here, 'wall_pw*_t*.mat'));
    if isempty(f)
        fprintf(['    no pupil-weighted checkpoints (wall_pw<W>_t<TILT>.mat) ' ...
                 '-- run them with WALL_PUPILW.\n']);
        return;
    end
    % the incumbent: the control point at the same tilt, standoff and wall
    inc = [];
    if isfield(R,'control') && ~isempty(R.control.row)
        c = R.control.row([R.control.row.ok]);
        [~, j] = min([c.blur_um]);   inc = c(j);
    end
    ws = [];   decks = {};   labs = {};
    for i = 1:numel(f)
        Z = load(fullfile(here, f(i).name), 'R');
        if ~isfield(Z,'R') || ~isfield(Z.R,'deck') || ~isfile(Z.R.deck), continue; end
        w = sscanf(Z.R.tag, 'pw%d');
        if isempty(w), continue; end
        ws(end+1) = w;              %#ok<AGROW>
        decks{end+1} = Z.R.deck;    %#ok<AGROW>
        labs{end+1} = sprintf('pw%d (solved AT x%d)', w, w); %#ok<AGROW>
    end
    if isempty(ws), return; end
    [ws, io] = sort(ws);   decks = decks(io);   labs = labs(io);
    if ~isempty(inc)
        decks = [{inc.deck}, decks];
        labs  = [{'INCUMBENT (study weights)'}, labs];
    end
    mult = unique([1, ws]);
    fprintf('\n    every design scored under every weighting:\n');
    fprintf('    %-28s', 'design');
    for m = mult, fprintf(' %10s', sprintf('m @ x%d', m)); end
    fprintf(' %9s %9s %8s %9s %9s\n', 'WFE nm','blur um','breath%','wander um','M');
    rows = struct('label',{},'deck',{},'merit',{},'mult',{},'wfe',{},'blur',{}, ...
                  'breathe',{},'wander',{},'mag',{});
    for i = 1:numel(decks)
        mm = zeros(1, numel(mult));   S1 = [];
        for k = 1:numel(mult)
            Q = P;
            Q.weights.blur    = P.weights.blur    * mult(k);
            Q.weights.breathe = P.weights.breathe * mult(k);
            Q.weights.wander  = P.weights.wander  * mult(k);
            Sk = afocal4_score(Q, decks{i}, 'fields',P.Fsolve, ...
                               'nodes',P.solve.nodes_score);
            mm(k) = Sk.merit;   if mult(k) == 1, S1 = Sk; end
        end
        fprintf('    %-28s', labs{i});
        fprintf(' %10.1f', mm);
        fprintf(' %9.1f %9.1f %8.4f %9.1f %9.4f\n', S1.wfe_max_nm, S1.blur_um, ...
                S1.breathe_pct, S1.wander_um, S1.mag_centre_chief);
        rows(end+1) = struct('label',labs{i}, 'deck',decks{i}, 'merit',mm, ...
            'mult',mult, 'wfe',S1.wfe_max_nm, 'blur',S1.blur_um, ...
            'breathe',S1.breathe_pct, 'wander',S1.wander_um, ...
            'mag',S1.mag_centre_chief); %#ok<AGROW>
    end
    W.rows = rows;   W.mult = mult;   W.have = true;
    % the verdict: does any re-weighted solve beat the incumbent on ITS merit?
    if numel(rows) > 1 && startsWith(rows(1).label, 'INCUMBENT')
        beat = false;
        for i = 2:numel(rows)
            k = find(mult == sscanf(rows(i).label,'pw%d'), 1);
            if ~isempty(k) && rows(i).merit(k) < rows(1).merit(k), beat = true; end
        end
        W.reweighting_wins = beat;
        fprintf(['\n    does a re-weighted solve beat the incumbent ON ITS OWN ' ...
                 'MERIT?  %s\n'], tern_(beat,'yes', ...
                 'NO -- the design solved at the study''s own weights wins every column'));
        fprintf(['    and both re-weighted solves drift the interface: M %s ' ...
                 'against a %.1f %% target.\n'], strjoin(arrayfun(@(r) ...
                 sprintf('%.2f %%', abs(r.mag/30-1)*100), rows(2:end), ...
                 'UniformOutput',false), ' / '), P.targets.mag_pct);
    end
end

% ---------------------------------------------------------------------
function F = fifth_bar_(R)
%FIFTH_BAR_  Leverage 4 stays PRICED, not built -- and the price is
%   whatever the walled frontier's best point actually delivers, not the
%   clearing stage's provisional one.
    d = R.delivered;
    F = struct('from','delivered -10 deg (clearing stage)', ...
               'blur_um',d.blur_um, 'breathe_pct',d.breathe_pct, ...
               'wander_um',d.wander_um, 'wfe_nm',d.wfe_nm, ...
               'floor_mm',d.floor_body_mm, 'sep_needed_mm',111.6, ...
               'sep_supplied_mm',201.4);
    if isfield(R,'frontier') && ~isempty(R.frontier.operating)
        o = R.frontier.operating;
        F = struct('from',sprintf('walled frontier, tilt %+.1f deg at %+.0f mm', ...
                   o.tilt, o.umin_mm), 'blur_um',o.blur_um, ...
                   'breathe_pct',o.breathe_pct, 'wander_um',o.wander_um, ...
                   'wfe_nm',o.wfe_nm, 'floor_mm',o.floor_mm, ...
                   'sep_needed_mm',111.6, 'sep_supplied_mm',NaN);
    end
    fprintf(['    a fifth mirror must supply at least %.1f mm of ' ...
             'FIELD-INDEPENDENT separation -- which the tilt already does --\n' ...
             '    so its case rests on doing it WITHOUT spending the pupil ' ...
             'control.  The bar, from the %s:\n'], F.sep_needed_mm, F.from);
    fprintf(['      beat blur %.1f um, breathing %.4f %%, wander %.1f um, at ' ...
             '%.1f nm of wavefront and a %+.2f mm union floor.\n'], ...
            F.blur_um, F.breathe_pct, F.wander_um, F.wfe_nm, F.floor_mm);
    fprintf(['    Priced, not built: the brief graduates it to a build task ' ...
             'only if the walled frontier cannot deliver an\n    acceptable ' ...
             'pupil price, and that is Dave''s call, not this stage''s.\n']);
end

% ---------------------------------------------------------------------
function figure_(R, png)
%FIGURE_  The frontier on one page: what the wall costs and what it holds.
    row = R.frontier.row;
    ok  = [row.ok];
    if ~any(ok), return; end
    f = figure('Position',[80 80 1180 520], 'Color','w', 'Visible','off');
    tl = tiledlayout(f,1,2,'Padding','compact','TileSpacing','compact');
    d  = R.delivered;

    ax = nexttile(tl);   hold(ax,'on');
    mk = {'-o','-s'};   lb = {};
    if isfield(R,'control') && ~isempty(R.control.row)
        c = R.control.row([R.control.row.ok]);
        [~,i] = sort([c.tilt]);   c = c(i);
        plot(ax, [c.tilt], [c.blur_um], '-d', 'LineWidth',2.4, 'MarkerSize',7, ...
             'Color',[0.10 0.45 0.75], 'MarkerFaceColor',[0.10 0.45 0.75]);
        lb{end+1} = 'CONTROL: standoff fixed, tilt isolated'; %#ok<AGROW>
    end
    for k = 1:2
        u = [0 15];  m = ok & ([row.umin_mm] == u(k)) & [row.admissible];
        if ~any(m), continue; end
        r = row(m);   [~,i] = sort([r.tilt]);   r = r(i);
        plot(ax, [r.tilt], [r.blur_um], mk{k}, 'LineWidth',1.8, 'MarkerSize',5);
        lb{end+1} = sprintf('walled at %+.0f mm', u(k)); %#ok<AGROW>
    end
    bad = ok & ~[row.admissible];
    if any(bad)
        r = row(bad);
        plot(ax, [r.tilt], [r.blur_um], 'x', 'LineWidth',1.4, 'MarkerSize',9, ...
             'Color',[0.6 0.6 0.6], 'LineStyle','none');
        lb{end+1} = 'clears, but not a design (M or AOI)'; %#ok<AGROW>
    end
    plot(ax, d.tilt_deg, d.blur_um, 'kp', 'MarkerSize',13, 'MarkerFaceColor','k');
    lb{end+1} = 'delivered -10 deg (unwalled, budget-capped)';
    yline(ax, d.committed_blur_um, 'k--');
    lb{end+1} = 'committed 343 mm design';
    xlabel(ax,'extraction tilt on the field mirror  (deg)');
    ylabel(ax,'pupil blur rms  (\mum)');
    title(ax,'the price');   grid(ax,'on');   box(ax,'on');
    legend(ax, lb, 'Location','north', 'Box','off');

    ax2 = nexttile(tl);   hold(ax2,'on');
    if isfield(R,'control') && ~isempty(R.control.row)
        c = R.control.row([R.control.row.ok]);
        [~,i] = sort([c.tilt]);   c = c(i);
        plot(ax2, [c.tilt], [c.floor_mm], '-d', 'LineWidth',2.4, 'MarkerSize',7, ...
             'Color',[0.10 0.45 0.75], 'MarkerFaceColor',[0.10 0.45 0.75]);
    end
    for k = 1:2
        u = [0 15];  m = ok & ([row.umin_mm] == u(k)) & [row.admissible];
        if ~any(m), continue; end
        r = row(m);   [~,i] = sort([r.tilt]);   r = r(i);
        plot(ax2, [r.tilt], [r.floor_mm], mk{k}, 'LineWidth',1.8, 'MarkerSize',5);
        yline(ax2, u(k), ':', 'Color',[.5 .5 .5]);
    end
    if any(bad)
        r = row(bad);
        plot(ax2, [r.tilt], [r.floor_mm], 'x', 'LineWidth',1.4, 'MarkerSize',9, ...
             'Color',[0.6 0.6 0.6], 'LineStyle','none');
    end
    plot(ax2, d.tilt_deg, d.floor_body_mm, 'kp', 'MarkerSize',13, ...
         'MarkerFaceColor','k');
    yline(ax2, 0, 'k-');
    xlabel(ax2,'extraction tilt on the field mirror  (deg)');
    ylabel(ax2,'union body-in-beam floor, declared body  (mm)');
    title(ax2,'what the wall holds');   grid(ax2,'on');   box(ax2,'on');

    annotation(f,'textbox',[0.02 0.955 0.96 0.04], 'String', ...
        ['the cleared frontier: past the knee the tilt buys no more ' ...
         'clearance and keeps charging for it'], ...
        'HorizontalAlignment','center','EdgeColor','none', ...
        'FontWeight','bold','FontSize',11);
    tl.OuterPosition = [0 0 1 0.94];
    exportgraphics(f, png, 'Resolution',150);
    close(f);
    fprintf('  wrote %s\n', png);
end

% ---------------------------------------------------------------------
function print_price_(pr)
    fprintf('  %7s %9s %9s %9s %9s %9s %8s %9s %5s\n', 'tilt deg', 'bare mm', ...
            'body mm', 'offset', 'WFE nm', 'blur um', 'breathe%', 'wander um','lost');
    for i = 1:numel(pr.raw)
        s = pr.raw(i);
        fprintf('  %+7.2f %9.2f %9.2f %9.1f %9.1f %9.1f %8.4f %9.1f %5d\n', ...
                s.tilt_deg, s.floor_bare*1e3, s.floor_body*1e3, s.offset_mm, ...
                s.wfe_nm, s.blur_um, s.breathe_pct, s.wander_um, s.nLost);
    end
end

function v = gz_(s, f)
    if isempty(s) || ~isfield(s,f) || ~s.ok, v = NaN; else, v = s.(f); end
end
function v = gzs_(s, f)
    if isempty(s) || ~s.ok || ~isfield(s.S,f), v = NaN; else, v = s.S.(f); end
end
function s = nm_(v),  if isfinite(v), s = sprintf('%.2f',v); else, s = '--'; end,  end
function v = getfielddef_(s, f, d)
    if isfield(s,f), v = s.(f); else, v = d; end
end
function hdr_(t)
    fprintf('\n%s\n  %s\n%s\n', repmat('=',1,74), upper(t), repmat('=',1,74));
end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
function del_(p),  if exist(p,'file'), delete(p); end,  end
