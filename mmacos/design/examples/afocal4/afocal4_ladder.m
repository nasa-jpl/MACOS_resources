function R = afocal4_ladder(opts)
%AFOCAL4_LADDER  The S4 answer ladder: Rodgers' four slides, with M4.
%
%   R = AFOCAL4_LADDER() replays his four-slide ladder on the recommended
%   four-mirror form and scores every rung on BOTH axes -- image quality and
%   the interface-pupil ladder -- which is the pair the study exists to
%   deliver.  His slides carry only the first.
%
%     rung 1  ON AXIS, joint solve                     -- the anchor
%     rung 2  offset +0.6 deg, FROZEN                  -- the collapse
%     rung 3  offset, joint re-solve                   -- the repair
%     rung 4  + M2/FM/M3 tilt and decenter, joint      -- the finish
%
%   The optical DOF set is the conics, the FIELD-MIRROR standoff and the
%   front end (M2's radius, the M1-M2 spacing); rung 4 adds the rigid
%   bodies.  See the OPT_DOFS comment in the code for the measurement that
%   set it -- three conics alone stall 20x above the diffraction limit.
%
%   ONE JOINT DOF SET PER RUNG; each rung seeds from the previous and then
%   solves everything it owns at once.  The first-order conditions are never
%   in the merit -- AFOCAL4_BUILD re-closes them exactly at every iterate, so
%   M = 30.000, collimation and the pupil station at P.iface are identities
%   of every design scored here (see AFOCAL4_BUILD for why that is a ruling
%   and not a convenience).
%
%   THE OPERATING POINT IS A PARAMETER.  P.iface = 140 mm is the flagged
%   default -- where the S3 breathing null sits -- and the standoff-versus-
%   power trade is REPORTED as a curve (section 2), not optimised.  The
%   instrument picks the point later; rung 4 runs at the flagged one.
%
%   Sections:
%     0  the seed, and where the unoptimised four-mirror layout starts
%     1  the four rungs
%     2  the interface-standoff trade curve
%     3  the merit A/B -- log versus linear residuals, measured
%     4  figures: field map + pupil ladder per rung, the trade, the summary
%
%   Name-value:
%     'sections'  which to run (default 0:4; section 3 is the expensive one)
%     'save'      write .in / .png / .mat (true)
%     'max_iter'  solver cap per rung (P.solve.max_iter)
%     'trade_iter' solver cap per trade point (30 -- warm-started from rung 3)
%     'resume'    a previous R to extend instead of re-solving
%     'prefix'    tag prepended to every artifact name (default '').  The
%                 S4b constrained redo runs with 'b_', so it lands BESIDE
%                 the S4 record rather than on top of it: those numbers
%                 stand as the unconstrained reference, labelled NOT
%                 BUILDABLE in RESULTS.md, and are never rewritten.
%     'iface_trade'  override P.iface_trade (the swept operating points)
%
%   Run:  >> afocal4_ladder                 (everything)
%         >> afocal4_ladder('sections',0:1) (the rungs only)
%   Results: RESULTS.md.
%
%   See also AFOCAL4_SOLVE, AFOCAL4_SCORE, AFOCAL4_BUILD, AFOCAL4_MERSENNE.

    arguments
        opts.sections (1,:) double  = 0:4
        opts.save     (1,1) logical = true
        opts.max_iter (1,1) double  = 0
        opts.trade_iter (1,1) double = 30
        opts.resume   struct = struct([])
        opts.prefix   (1,:) char = ''
        opts.iface_trade (1,:) double = []
    end
    here = fileparts(mfilename('fullpath'));
    P = afocal4_params();
    % PREFIX tags every artifact this run writes.  The S4 record is a
    % published reference that the S4b redo must not overwrite (Dave's
    % retract-in-place discipline), so the constrained run writes
    % afocal4_b_*.in / .png / .mat beside the S4 ones rather than on top
    % of them.  Empty = the S4 names, unchanged.
    pfx = opts.prefix;
    if ~isempty(opts.iface_trade), P.iface_trade = opts.iface_trade; end
    if opts.max_iter <= 0, opts.max_iter = P.solve.max_iter; end
    macos.init(P.model_size);
    R = struct('P',P, 'when',datestr(now,31)); %#ok<TNOW1,DATST>

    % THE OPTICAL DOF SET, and the rule that set it.  The brief's rung 1 is
    % "conics + phi4"; with the interface standoff carried as a PARAMETER the
    % closure consumes phi4, which leaves three conics -- and three conics
    % alone STALL at 1391 nm on axis, 19.6x the diffraction limit, with every
    % pupil target already met (measured; the field mirror sits at an image,
    % so its conic buys field aberration and almost no spherical, leaving two
    % effective surfaces against a collimator the closure has made 29%
    % faster).  An anchor rung that misses DL by 20x is not an anchor.  So the
    % optical set is the conics PLUS the field-mirror standoff -- which gives
    % that conic a footprint to act on -- PLUS the front end, exactly the
    % "conics and radii" his own rung 3 re-optimises.
    OPT_DOFS = {'conic','standoff','front'};
    if ~isempty(fieldnames(opts.resume)), R = opts.resume; end

    % =====================================================================
    if any(opts.sections == 0)
    banner('0.  THE SEED');
    % =====================================================================
    D0 = afocal4_seed(P, 'bias_deg', 0);
    b0 = afocal4_build(P, D0, deck_(here,[pfx 'seed'],opts.save), 'quiet',false);
    S0 = afocal4_score(P, b0.file, 'nodes',P.solve.nodes_score, 'grid',P.grid_n);
    afocal4_score_print(P, S0, 'seed, on axis, conics carried');
    fprintf(['\n  The seed is the S3 recommendation UNOPTIMISED: his conics\n' ...
             '  carried onto a collimator whose radius the closure moved %.0f%%\n' ...
             '  to hold the pupil at %.0f mm, and a field mirror at K = 0.\n' ...
             '  That is why the wavefront error starts where it does.  Every\n' ...
             '  first-order property is already exact -- M = %.4f paraxial,\n' ...
             '  %.4f traced -- and stays exact at every iterate below.\n'], ...
            100*abs(b0.R(4)/P.parent.R(3) - 1), P.iface*1e3, ...
            b0.C.fo.mag, b0.traced.mag);
    R.seed = struct('D',D0, 'b',b0, 'S',S0);
    end

    % =====================================================================
    if any(opts.sections == 1)
    banner('1.  THE FOUR RUNGS');
    % =====================================================================
    % Field names and ORDER must match mkrung_ exactly: MATLAB rejects
    % `arr(k) = s` when s has different fields, and it does so only when the
    % assignment is reached -- i.e. after the rung has been solved.
    rung = struct('name',{},'label',{},'D',{},'deck',{},'S',{},'solve',{});
    ckpt = @(R_) save_(here, R_, opts.save, pfx);

    % --- rung 1: on axis, joint solve -------------------------------------
    D = afocal4_seed(P, 'bias_deg', 0);
    s1 = solve_rung_(P, D, OPT_DOFS, 'rung 1  on axis', ...
                     deck_(here,[pfx 'r1_onaxis'],true), opts.max_iter);
    rung(1) = mkrung_('r1_onaxis','1  on axis, joint solve', ...
                      s1.D, s1.deck, s1.S, s1);
    R.rung = rung;   ckpt(R);

    % --- rung 2: the SAME design at the design field, frozen --------------
    % His slide 2, and the number the whole exercise is measured against: how
    % much an on-axis solution loses when the field box moves off axis.
    % Nothing is re-solved.  The interface PLANE is re-posed, because it is a
    % mechanical part that follows the beam -- his coldstop DAR tilt does
    % exactly this (0 deg on axis, 4.289 deg at the offset field).
    D2 = rung(1).D;   D2.bias_deg = P.bias_deg;
    b2 = afocal4_build(P, D2, deck_(here,[pfx 'r2_offset'],true));
    S2 = afocal4_score(P, b2.file, 'nodes',P.solve.nodes_score, 'grid',P.grid_n);
    rung(2) = mkrung_('r2_offset','2  offset +0.6 deg, FROZEN', D2, b2.file, S2, []);
    afocal4_score_print(P, S2, 'rung 2  offset, frozen');
    R.rung = rung;   ckpt(R);

    % --- rung 3: joint re-solve at the bias -------------------------------
    s3 = solve_rung_(P, D2, OPT_DOFS, 'rung 3  offset re-solve', ...
                     deck_(here,[pfx 'r3_resolve'],true), opts.max_iter);
    rung(3) = mkrung_('r3_resolve','3  offset, joint re-solve', ...
                      s3.D, s3.deck, s3.S, s3);
    R.rung = rung;   ckpt(R);

    % --- rung 4: + rigid bodies, joint ------------------------------------
    s4 = solve_rung_(P, rung(3).D, [OPT_DOFS {'rb'}], 'rung 4  + tilt/dec', ...
                     deck_(here,[pfx 'r4_tiltdec'],true), opts.max_iter);
    rung(4) = mkrung_('r4_tiltdec','4  + M2/FM/M3 tilt and decenter', ...
                      s4.D, s4.deck, s4.S, s4);

    R.rung = rung;
    rung_table_(P, R);
    provenance_(P, R);
    % WHAT the wavefront wall is made of, on the anchor and on the delivered
    % rung.  The S4 brief's stall rule: characterise before polishing.
    R.terms = struct('r1', afocal4_wfe_terms(P, rung(1).deck, 'quiet',true), ...
                     'r4', afocal4_wfe_terms(P, rung(end).deck));
    end

    % =====================================================================
    if any(opts.sections == 2) && isfield(R,'rung')
    banner('2.  THE INTERFACE-STANDOFF TRADE');
    % =====================================================================
    fprintf(['  The interface standoff is a PARAMETER, not a spec (S4\n' ...
             '  ruling).  It rides the field mirror''s power -- 343 mm is the\n' ...
             '  three-mirror''s own pupil at phi4 = 0, 140 mm is the S3\n' ...
             '  breathing null -- so what the instrument is owed is the CURVE,\n' ...
             '  with the design re-solved at every point.  Rung-3 DOFs.\n\n' ...
             '  TWO SEEDS AT EVERY POINT, and the pair is reported.  The S4\n' ...
             '  curve warm-started each point from the last and walked into a\n' ...
             '  basin at 220 mm whose wavefront column looked like the best on\n' ...
             '  the curve while its pupil blur read 16.7 mm (RESULTS 3.1).  A\n' ...
             '  finite-difference Jacobian over six DOFs makes the basin part\n' ...
             '  of the answer, so it is measured rather than hoped for: WARM\n' ...
             '  from the delivered rung, FRESH from the compliant seed at that\n' ...
             '  operating point, and the lower worst-miss wins.\n\n']);
    Tr = struct('iface',{},'phi4',{},'R_fm',{},'R_col',{},'s_fm',{},'S',{}, ...
                'D',{},'behind_m1',{},'pack',{},'seeds',{},'seed_used',{});
    for q = P.iface_trade
        % One operating point that will not solve must not take the curve
        % down with it -- a missing point is a reported gap, an aborted
        % section is four solved points thrown away.
        seeds = struct('name',{},'S',{},'D',{},'deck',{});
        warm = R.rung(3).D;   warm.iface = q;
        cold = afocal4_seed(P, 'iface',q, 'bias_deg',P.bias_deg);
        for sd = [struct('name','warm','D',warm), struct('name','fresh','D',cold)]
            try
                dk = deck_(here, sprintf('%strade_%03.0fmm_%s', pfx, q*1e3, ...
                                         sd.name), opts.save);
                sq = solve_rung_(P, sd.D, OPT_DOFS, ...
                        sprintf('trade  iface %.0f mm  (%s seed)', q*1e3, sd.name), ...
                        dk, opts.trade_iter, true);
                seeds(end+1) = struct('name',sd.name, 'S',sq.S, 'D',sq.D, ...
                                      'deck',sq.deck); %#ok<AGROW>
                fprintf(['    %-5s seed | WFE %8.1f nm  blur %7.1f  breathe %6.3f%%  ' ...
                         'wander %7.1f um  worst %6.2fx  anchor %5.1f um\n'], ...
                        sd.name, sq.S.wfe_max_nm, sq.S.blur_um, sq.S.breathe_pct, ...
                        sq.S.wander_um, sq.S.worst, sq.S.anchor_resid_um);
            catch ME
                fprintf('    %-5s seed | NO POINT: %s\n', sd.name, ME.message);
            end
        end
        if isempty(seeds)
            fprintf('  iface %6.1f mm  NO POINT from either seed\n', q*1e3);
            R.trade = Tr;   ckpt2_(here, R, opts.save, pfx);
            continue;
        end
        [~, kw] = min(arrayfun(@(s) s.S.worst, seeds));
        win = seeds(kw);
        % the winner is what gets committed under the plain trade name
        dk = deck_(here, sprintf('%strade_%03.0fmm', pfx, q*1e3), opts.save);
        bq = afocal4_build(P, win.D, dk, 'verify',false);
        % ANCHORING RESIDUAL is a validity check, not a metric: a design
        % whose cones will not anchor is not a design with a large residual,
        % it is a number nobody should quote (RESULTS 3.1).
        if win.S.anchor_resid_um > 0.1*win.S.blur_um
            fprintf(['    !! anchoring residual %.1f um is %.0f%% of the blur ' ...
                     '-- this point is NOT trustworthy\n'], ...
                    win.S.anchor_resid_um, 100*win.S.anchor_resid_um/win.S.blur_um);
        end
        K = afocal4_pack(P, dk, 'quiet',true);
        Tr(end+1) = struct('iface',q, 'phi4',bq.phi4, 'R_fm',bq.R(3), ...
                           'R_col',bq.R(4), 's_fm',win.D.fm_standoff, ...
                           'S',win.S, 'D',win.D, 'behind_m1',bq.behind_m1, ...
                           'pack',K, 'seeds',seeds, 'seed_used',win.name); %#ok<AGROW>
        fprintf(['  iface %6.1f mm  phi4 %+6.3f /m  R_FM %7.4f m  s_FM %5.0f mm  ' ...
                 '| WFE %8.1f nm  blur %6.1f  breathe %6.3f%%  wander %7.1f um  ' ...
                 'worst %6.2fx  | %s behind %4.0f mm, fold %+5.1f mm  %s  [%s]\n'], ...
                q*1e3, bq.phi4, bq.R(3), win.D.fm_standoff*1e3, ...
                win.S.wfe_max_nm, win.S.blur_um, win.S.breathe_pct, ...
                win.S.wander_um, win.S.worst, bq.names{end}, bq.behind_m1*1e3, ...
                pick_(K,'gap')*1e3, yn_(K.ok), win.name);
        R.trade = Tr;   ckpt2_(here, R, opts.save, pfx);
    end
    R.trade = Tr;
    end

    % =====================================================================
    if any(opts.sections == 3) && isfield(R,'rung')
    banner('3.  THE MERIT A/B  (the earned rule, measured)');
    % =====================================================================
    fprintf(['  AFOCAL4_SCORE scores every term by log(m/t), not m/t.  The\n' ...
             '  rejected alternative is run here from the SAME seed with the\n' ...
             '  SAME DOFs so the claim is measured rather than asserted.\n\n']);
    ab = afocal4_solve(P, R.rung(2).D, 'dofs',OPT_DOFS, ...
                       'label','rung 3, LINEAR merit', 'merit','linear', ...
                       'deck',deck_(here,[pfx 'r3_linear_merit'],opts.save), ...
                       'max_iter',opts.max_iter);
    R.merit_ab = ab;
    fprintf('\n  %-22s %10s %10s %10s %10s %10s\n', ...
        'merit', 'WFE nm', 'blur um', 'breathe %', 'wander um', 'worst');
    lin = ab.S;   lg = R.rung(3).S;
    fprintf('  %-22s %10.1f %10.1f %10.3f %10.1f %10.2f\n', 'log (shipped)', ...
        lg.wfe_max_nm, lg.blur_um, lg.breathe_pct, lg.wander_um, lg.worst);
    fprintf('  %-22s %10.1f %10.1f %10.3f %10.1f %10.2f\n', 'linear (rejected)', ...
        lin.wfe_max_nm, lin.blur_um, lin.breathe_pct, lin.wander_um, lin.worst);
    end

    % =====================================================================
    if any(opts.sections == 4) && opts.save && isfield(R,'rung')
    banner('4.  FIGURES');
    % =====================================================================
    for i = 1:numel(R.rung)
        try
            png = fullfile(here, sprintf('afocal4_%s%s_field.png', pfx, R.rung(i).name));
            field_map_(P, R.rung(i), png);   fprintf('  wrote %s\n', png);
        catch ME, fprintf('   field map %d failed: %s\n', i, ME.message); end
        try
            png = fullfile(here, sprintf('afocal4_%s%s_pupil.png', pfx, R.rung(i).name));
            pupil_fig_(P, R.rung(i), png);   fprintf('  wrote %s\n', png);
        catch ME, fprintf('   pupil fig %d failed: %s\n', i, ME.message); end
    end
    if isfield(R,'trade')
        try
            png = fullfile(here, sprintf('afocal4_%strade.png', pfx));
            trade_fig_(P, R, png);   fprintf('  wrote %s\n', png);
        catch ME, fprintf('   trade figure failed: %s\n', ME.message); end
    end
    try
        png = fullfile(here, sprintf('afocal4_%sladder_summary.png', pfx));
        summary_fig_(P, R, png);   fprintf('  wrote %s\n', png);
    catch ME, fprintf('   summary figure failed: %s\n', ME.message); end
    end

    save_(here, R, opts.save, pfx);
    if opts.save, fprintf('\n  saved afocal4_%sladder.mat\n', pfx); end
end

% =====================================================================
function ckpt2_(here, R, dosave, pfx)
    save_(here, R, dosave, pfx);
end

function save_(here, R, dosave, pfx)
%SAVE_  Checkpoint after every rung.  A ladder rung is tens of minutes of
%   machine time and the run is unattended; losing four of them to a figure
%   that threw is not a trade anyone would make deliberately.
    if ~dosave, return; end
    save(fullfile(here, sprintf('afocal4_%sladder.mat', pfx)), 'R', '-v7.3');
end

% =====================================================================
function s = solve_rung_(P, D, dofs, label, deck, max_iter, quiet)
%SOLVE_RUNG_  A rung's joint solve, seeded by a short conics-only pass.
%
%   SEEDS ARE SEEDS, AND THIS IS ONE.  The rung is still a single joint
%   solve over its whole DOF set -- doctrine rule 1, never alternate.  What
%   comes first is a cheap conics-only pass whose only job is to hand the
%   joint solve a starting point that is already in the right basin.
%
%   The rule was earned, not assumed.  Started cold from the carried conics,
%   the six-DOF joint solve wandered: 2323 -> 2308 -> 2516 nm over twenty
%   evaluations, worse than a three-conic solve from the same seed had
%   already reached (1391 nm).  A finite-difference Jacobian over DOFs whose
%   sensitivities differ by orders of magnitude gives the trust region a
%   poor first step, and the carried conics are far enough from any solution
%   that the first step decides the basin.  The conics-only pass costs about
%   a tenth of the rung and removes that.
    if nargin < 7, quiet = false; end
    pre = afocal4_solve(P, D, 'dofs',{'conic'}, 'max_iter',12, 'quiet',true, ...
                        'label',[label ' (seed)']);
    if ~quiet
        fprintf('  %s: conic seed -> WFE %.1f nm, worst %.2fx\n', ...
                label, pre.S.wfe_max_nm, pre.S.worst);
    end
    s = afocal4_solve(P, pre.D, 'dofs',dofs, 'label',label, 'deck',deck, ...
                      'max_iter',max_iter, 'quiet',quiet);
    s.pre = pre;
end

function r = mkrung_(name, label, D, deck, S, sv)
    r = struct('name',name, 'label',label, 'D',D, 'deck',deck, 'S',S, 'solve',sv);
end

function g = pick_(K, f)
%PICK_  A packaging field that may not exist -- a deck with no usable fold
%   station has no .fold_pick, and the trade line still has to print.
    g = NaN;
    if isfield(K,'fold_pick') && ~isempty(K.fold_pick) && isfield(K.fold_pick,f)
        g = K.fold_pick.(f);
    end
end

function s = yn_(b),  if b, s = 'BUILDABLE'; else, s = 'not buildable'; end,  end

function f = deck_(here, tag, dosave)
    if dosave, f = fullfile(here, sprintf('afocal4_%s.in', tag));
    else,      f = [tempname '.in'];
    end
end

function rung_table_(P, R)
%RUNG_TABLE_  The headline: his ladder, ours, and the pupil column he does
%   not have.  The result of this study is the PAIR, so the table prints
%   both and the worst normalised miss beside them.
    T = P.targets;
    fprintf(['\n  THE LADDER.  WFE is the afocal rung 2 (piston + per-field\n' ...
             '  tip/tilt) -- the rung that matches his CODE V field maps --\n' ...
             '  quoted on HIS 3x3 solve set and on the uniform %dx%d grid.\n' ...
             '  Magnification and breathing are CHIEF-NORMAL; wander is at the\n' ...
             '  REFIT interface plane, whose pose is reported below it.\n\n'], ...
            P.grid_n, P.grid_n);
    fprintf('  %-34s %9s %9s | %8s %9s %9s %9s %8s | %7s\n', ...
        'rung','WFE 3x3','WFE grid','blur','breathe','wander','surf','M','worst');
    fprintf('  %-34s %9s %9s | %8s %9s %9s %9s %8s | %7s\n', ...
        '','nm','nm','um','%','um','mm','x','');
    for i = 1:numel(R.rung)
        S = R.rung(i).S;
        g = NaN;  if isfield(S,'wfe_grid_max_nm'), g = S.wfe_grid_max_nm; end
        fprintf('  %-34s %9.1f %9.1f | %8.1f %9.3f %9.1f %9.4f %8.4f | %7.2f\n', ...
            R.rung(i).label, S.wfe_max_nm, g, S.blur_um, S.breathe_pct, ...
            S.wander_um, S.surf_pv_mm, S.mag_centre_chief, S.worst);
    end
    fprintf('  %-34s %9.1f %9s | %8.1f %9.3f %9.1f %9.4f %8.4f | %7s   <- S4 target\n', ...
        'TARGET', T.wfe_rung2_nm, '-', T.blur_um, T.breathe_pct, T.wander_um, ...
        T.surface_pv_mm, T.mag, '1.00');
    fprintf(['\n  His three-mirror ladder, for the same four steps:\n' ...
             '    15 nm on axis -> 430 offset frozen -> 160 re-solved -> 119 + tilt/dec\n' ...
             '  (his numbers, his metric; the rodgers2 baseline reproduces them\n' ...
             '   to 0.952-1.015x at this rung.)  He reports no pupil column.\n']);
    fprintf('\n  interface pose per rung (refit against the as-emitted plane):\n');
    for i = 1:numel(R.rung)
        S = R.rung(i).S;
        fprintf(['    %-34s shift %+7.3f mm  tilt %+7.4f deg   ' ...
                 '(placed-plane wander %8.1f um)\n'], ...
                R.rung(i).label, S.pose.shift_mm, S.pose.tilt_deg, ...
                S.wander_placed_um);
    end
end

function provenance_(P, R)
%PROVENANCE_  The parameter table IS the solution (rodgers1 doctrine 6).
%   Read from the COMMITTED decks, so what is published is what was scored,
%   and every held quantity is visibly held.
    fprintf(['\n  PARAMETER PROVENANCE.  M1 is HELD -- his radius, and his\n' ...
             '  conic of -1, a parabola.  M2''s radius and the M1-M2 spacing are\n' ...
             '  SOLVED (his own rung 3 re-optimises radii too).  The field\n' ...
             '  mirror''s radius and the collimator''s radius and station are\n' ...
             '  CLOSED -- re-derived at every iterate from the afocal condition,\n' ...
             '  the magnification and the pupil station at %.0f mm -- so they\n' ...
             '  move rung to rung without ever being free.\n'], P.iface*1e3);
    for i = 1:numel(R.rung)
        D = R.rung(i).D;
        b = afocal4_build(P, D, [tempname '.in'], 'verify',false);
        fprintf('\n    %s\n', R.rung(i).label);
        fprintf('      %-6s %12s %12s %12s %12s %10s\n', ...
            'elt','R (m)','conic','dec y (mm)','tilt x (mrad)','prov');
        for k = 1:numel(b.R)
            dy = 0;  tx = 0;
            j = find(P.rb_elts == k, 1);
            if ~isempty(j), dy = D.rb(j,1);  tx = D.rb(j,2); end
            prov = 'closed';
            if k == 1, prov = 'HELD (his)';
            elseif k == 2, prov = 'solved'; end
            fprintf('      %-6s %12.6f %12.6f %12.4f %12.4f %10s\n', ...
                b.names{k}, b.R(k), b.conic(k), dy*1e3, tx*1e3, prov);
        end
        fprintf(['      phi4 %+.5f /m (closed by iface = %.1f mm)   ' ...
                 'FM standoff %.1f mm\n'], b.phi4, D.iface*1e3, D.fm_standoff*1e3);
        fprintf('      spacings (m): %s   train %.4f m\n', ...
            strtrim(sprintf('%.6f  ', b.C.t)), sum(b.C.t)+D.iface);
    end
end

% ---------------------------------------------------------------------
%  Figures
% ---------------------------------------------------------------------
function field_map_(P, rg, png)
%FIELD_MAP_  The three afocal rungs over the uniform box, his 3x3 overlaid.
%   The sampling question is made visible rather than argued: his solve set
%   is a third corners, which biases an AREA average (rodgers1, 8%).
    S = rg.S;
    if ~isfield(S,'L_grid'), error('no uniform-grid score on this rung'); end
    n = P.grid_n;
    F = macos.design.field_grid(P.fov_half_deg*60, n, 'units','arcmin');
    x = reshape(F(:,1)*180/pi, n, n);   y = reshape(F(:,2)*180/pi, n, n);
    fig = figure('Visible','off','Position',[100 100 1180 380]);
    tl = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
    rn = {'rung 1  piston','rung 2  + tip/tilt','rung 3  + power'};
    for r = 1:3
        ax = nexttile(tl);
        z = reshape(S.L_grid(:,r), n, n)*1e9;
        contourf(ax, x, y, z, 16, 'LineColor','none');   hold(ax,'on');
        plot(ax, P.Fsolve_deg(:,1), P.Fsolve_deg(:,2), 'w+', 'MarkerSize',8, ...
             'LineWidth',1.2);
        hold(ax,'off');   axis(ax,'square');   colormap(ax, parula);
        cb = colorbar(ax);   cb.Label.String = 'RMS WFE (nm)';
        title(ax, sprintf('%s   max %.1f nm', rn{r}, max(z(:))));
        xlabel(ax,'XAN - box centre (deg)');
        if r == 1, ylabel(ax,'YAN - box centre (deg)'); end
    end
    title(tl, sprintf('afocal4 rung %s   (DL at 1 um = %.0f nm)', ...
          strrep(rg.label,'_','\_'), P.targets.wfe_rung2_nm));
    exportgraphics(fig, png, 'Resolution', 150);   close(fig);
end

function pupil_fig_(P, rg, png)
%PUPIL_FIG_  The four-part pupil ladder for one rung, plus the breathing.
%   Five panels, never one merged residual: the parts are physically
%   different and an instrument cares about them separately.
    m = rg.S.pm;   g = m.good;
    u = m.nodes(1,g)*1e3;   v = m.nodes(2,g)*1e3;
    % Five panels need HEIGHT, not just width: at 330 px the layout title
    % lands on the panel titles and the figure is unreadable.  The layout
    % title is also split over two lines so it cannot run under them.
    fig = figure('Visible','off','Position',[100 100 1560 460]);
    tl = tiledlayout(fig,1,5,'TileSpacing','compact','Padding','compact');
    % RESERVE the title band explicitly.  title(tl,...) on a 1x5 layout whose
    % panels carry two-line titles of their own lands ON them -- the layout
    % gives the title whatever vertical space is left, and with five square
    % panels there is none.  Shrinking the layout and drawing the heading as
    % an annotation in the freed strip cannot collide by construction.
    tl.OuterPosition = [0 0 1 0.88];

    ax = nexttile(tl);
    scatter(ax, u, v, 26, m.blur.waist_rms(g)*1e6, 'filled');
    lbl_(ax, {'(1) blur, cone waist', ...
              sprintf('rms %.1f um  (target %.0f)', 1e6*m.blur.rms, ...
                      P.targets.blur_um)}, 'um');
    ylabel(ax,'M1 y (mm)');

    ax = nexttile(tl);
    % The MAP is the distance of each convergence point from the flat placed
    % plane -- that is the per-node quantity pupil_map returns.  The number
    % QUOTED is the residual against the ideal image of the primary's own
    % sag, which is the one that is a pupil defect; the two are different
    % statements and the panel says which is which.
    scatter(ax, u, v, 26, m.surface.flat.dist(g)*1e6, 'filled');
    lbl_(ax, {'(2) surface, vs the flat plane', ...
              sprintf('net of imaged sag: %.4f mm', rg.S.surf_pv_mm)}, 'um');

    ax = nexttile(tl);
    scatter(ax, u, v, 26, m.map.distortion(g)*1e6, 'filled');
    lbl_(ax, {'(3) pupil distortion', ...
              sprintf('%.3f%% of the pupil radius', ...
                      100*m.map.distortion_frac_max)}, 'um');

    ax = nexttile(tl);
    scatter(ax, u, v, 26, m.wander.per_node_rms(g)*1e6, 'filled');
    lbl_(ax, {'(4) wander, as-emitted plane', ...
              sprintf('%.0f um  (refit: %.0f um)', rg.S.wander_placed_um, ...
                      rg.S.wander_um)}, 'um');

    ax = nexttile(tl);
    c = m.map.mag_per_field_chief;
    fx = m.fields(:,1)*180/pi;   fy = m.fields(:,2)*180/pi;
    scatter(ax, fx, fy, 90, c, 'filled');   axis(ax,'square');  box(ax,'on');
    colormap(ax, parula);  cb = colorbar(ax);  cb.Label.String = 'M (chief-normal)';
    title(ax, {'(5) magnification breathing', ...
               sprintf('%.3f%%  (target %.1f%%)', rg.S.breathe_pct, ...
                       P.targets.breathe_pct)});
    xlabel(ax,'XAN (deg)');   ylabel(ax,'YAN (deg)');

    % PLAIN ASCII, interpreter 'none'.  A TeX escape inside a sprintf FORMAT
    % is read by sprintf first: '{\times}' becomes '{<TAB>imes}' and the rest
    % of the string is mangled.  That is what put "{ imes}" in the first
    % render.  Either escape the backslash for sprintf or do not use TeX
    % here at all; not using it is the version that cannot regress.
    annotation(fig, 'textbox', [0.02 0.885 0.96 0.10], 'EdgeColor','none', ...
        'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
        'FontWeight','bold', 'FontSize',13, 'Interpreter','none', ...
        'String', {sprintf('afocal4 rung %s  --  the interface pupil', rg.label), ...
                   sprintf(['M = %.4fx chief-normal;  wander %.1f um at the ' ...
                            'refit plane, %.0f um as emitted'], ...
                           rg.S.mag_centre_chief, rg.S.wander_um, ...
                           rg.S.wander_placed_um)});
    exportgraphics(fig, png, 'Resolution', 150);   close(fig);
end

function lbl_(ax, t, un)
    axis(ax,'equal');  axis(ax,'tight');  box(ax,'on');
    colormap(ax, parula);  cb = colorbar(ax);  cb.Label.String = un;
    title(ax, t);  xlabel(ax,'M1 x (mm)');
end

function trade_fig_(P, R, png)
%TRADE_FIG_  Targets versus the interface standoff, each on its own target-
%   normalised axis, so "which operating point meets what" is readable
%   without unit arithmetic.  The flagged default is marked, not privileged.
    Tr = R.trade;   q = [Tr.iface]*1e3;
    S = [Tr.S];
    fig = figure('Visible','off','Position',[100 100 1200 460]);
    tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

    ax = nexttile(tl);
    plot(ax, q, [S.wfe_max_nm]/P.targets.wfe_rung2_nm, 'o-','LineWidth',1.6); hold(ax,'on');
    plot(ax, q, [S.blur_um]/P.targets.blur_um,          's-','LineWidth',1.6);
    plot(ax, q, [S.breathe_pct]/P.targets.breathe_pct,  'd-','LineWidth',1.6);
    plot(ax, q, [S.wander_um]/P.targets.wander_um,      '^-','LineWidth',1.6);
    yl = yline(ax, 1, 'k--', 'target');
    yl.Annotation.LegendInformation.IconDisplayStyle = 'off';
    xl = xline(ax, P.iface*1e3, ':', 'flagged default');
    xl.Annotation.LegendInformation.IconDisplayStyle = 'off';
    set(ax,'YScale','log');   grid(ax,'on');
    legend(ax, {'WFE','blur','breathing','wander'}, 'Location','best');
    xlabel(ax,'interface standoff (mm)');   ylabel(ax,'metric / target');
    title(ax,'what each operating point delivers, re-solved');

    ax = nexttile(tl);
    yyaxis(ax,'left');
    plot(ax, q, [Tr.phi4], 'o-','LineWidth',1.6);
    ylabel(ax,'field-mirror power \phi_4 (1/m)');
    yyaxis(ax,'right');
    plot(ax, q, [Tr.R_fm], 's-','LineWidth',1.6);
    ylabel(ax,'field-mirror radius (m)');
    xlabel(ax,'interface standoff (mm)');   grid(ax,'on');
    title(ax,'the parameter behind the trade: the standoff rides \phi_4');

    title(tl, ['afocal4 -- interface standoff carried as a PARAMETER, ' ...
               'design re-solved at every point'], 'FontWeight','bold');
    exportgraphics(fig, png, 'Resolution', 150);   close(fig);
end

function summary_fig_(P, R, png)
%SUMMARY_FIG_  The ladder as target-normalised bars: the one picture that
%   says whether the PAIR was delivered.
    n = numel(R.rung);
    lbl = cell(1,n);   M = zeros(n,5);
    T = P.targets;
    for i = 1:n
        S = R.rung(i).S;   lbl{i} = R.rung(i).name;
        M(i,:) = [S.wfe_max_nm/T.wfe_rung2_nm, S.blur_um/T.blur_um, ...
                  S.breathe_pct/T.breathe_pct, S.wander_um/T.wander_um, ...
                  S.mag_pct/T.mag_pct];
    end
    fig = figure('Visible','off','Position',[100 100 980 460]);
    ax = axes(fig);
    hb = bar(ax, M);   set(ax,'XTickLabel',strrep(lbl,'_','\_'), ...
                             'XTickLabelRotation',15,'YScale','log');
    yl = yline(ax, 1, 'k--','target');
    yl.Annotation.LegendInformation.IconDisplayStyle = 'off';
    legend(ax, hb, {'WFE','blur','breathing','wander','M error'}, ...
           'Location','northeastoutside');
    ylabel(ax,'metric / target');   grid(ax,'on');
    title(ax, 'afocal4 S4 -- the answer ladder against its targets', ...
          'FontWeight','bold');
    exportgraphics(fig, png, 'Resolution', 150);   close(fig);
end

function banner(s)
    fprintf('\n=================================================================\n');
    fprintf('  %s\n', s);
    fprintf('=================================================================\n');
end
