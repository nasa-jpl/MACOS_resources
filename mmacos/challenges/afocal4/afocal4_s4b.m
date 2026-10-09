function R = afocal4_s4b(opts)
%AFOCAL4_S4B  The BUILDABLE trade: S4 redone under the packaging constraint.
%
%   R = AFOCAL4_S4B() answers the S4b brief.  The S4 study delivered a
%   trade curve for an object that cannot be built: at every point on it
%   the collimator, the field mirror, the interface pupil and the whole
%   instrument behind the pupil sit IN FRONT of the primary, inside the
%   incoming beam (Dave, 2026-08-03).  This driver re-derives the same
%   curve with the constraint enforced, and demonstrates -- rather than
%   asserts -- that the delivered design folds into a real package.
%
%   Sections:
%     0  THE CONSTRAINT, IN NUMBERS.  His parent, the delivered S4 rung and
%        a compliant seed through the same engine-truth gate.  This is the
%        retraction: the S4 curve is not amended, it is labelled.
%     1  THE ANCHOR.  The brief's first thing to try -- hold his M1->M2 and
%        M2->image geometry verbatim and let the conics, phi4 and the field-
%        mirror standoff do the work -- against the free-but-constrained
%        solve and against the free-and-unconstrained S4 number.  What the
%        anchor COSTS is the deliverable, not whether it wins.
%     2  THE BUILDABLE TRADE.  Runs (or loads) AFOCAL4_LADDER with prefix
%        'b_': the four rungs and the standoff sweep, every point solved
%        from two seeds and gated for packaging.  Tabulated beside the S4
%        curve.
%     3  THE FOLDED DEMONSTRATION.  Inserts the flat on the collimator's
%        exit leg, emits the folded prescription, re-scores it (a nominally
%        null fold is not a free fold -- e2e2 S3), and measures where the
%        pupil and a stated instrument envelope actually end up.
%     5  THE SECOND BASIN.  Section 2's sweep stays where its seeder puts
%        it -- his front end, field mirror far past the image -- and that
%        basin cannot reach the pupil targets at any standoff.  The
%        front-end variant can, so its basin is swept too.
%     4  FIGURES.  yz + xz layouts of the unfolded and folded designs with
%        the instrument volume drawn, and the S4-vs-S4b trade comparison.
%
%   Name-value:
%     'sections'  which to run (default 0:5)
%     'save'      write .in / .png / .mat (true)
%     'trade'     'run' (default; solve it) | 'load' (use afocal4_b_ladder.mat)
%     'iface'     the operating point for sections 1 and 3 (P.iface)
%     'fold_from' which design section 3 folds: 'ladder' (default, the
%                 delivered rung), 'anchor', 'variant', or 'best' -- the
%                 lowest worst-miss among the trade points that actually
%                 PASS the packaging gate; or 'basin2' -- the same, over
%                 the image-behind-M1 basin only.  'basin2' is what the
%                 study folds: 'best' lands on the point where the fourth
%                 mirror's power has gone to zero, which is his
%                 three-mirror wearing a flat, and folding that
%                 demonstrates nothing about a four-mirror package
%
%   Run:  >> afocal4_s4b                       (everything; hours)
%         >> afocal4_s4b('sections',[0 1])     (the constraint + the anchor)
%         >> afocal4_s4b('sections',[3 4], 'trade','load')
%   Results: RESULTS.md, section S4b.
%
%   See also AFOCAL4_PACK, AFOCAL4_LADDER, AFOCAL4_BUILD, AFOCAL4_PARAMS.

    arguments
        opts.sections (1,:) double  = 0:5
        opts.save     (1,1) logical = true
        opts.trade    (1,:) char {mustBeMember(opts.trade,{'run','load'})} = 'run'
        opts.iface    (1,1) double  = 0
        opts.fold_from (1,:) char {mustBeMember(opts.fold_from, ...
                       {'ladder','anchor','variant','best','basin2'})} = 'ladder'
    end
    here = fileparts(mfilename('fullpath'));
    P = afocal4_params();
    if opts.iface > 0, P.iface = opts.iface; end
    macos.init(P.model_size);
    R = struct('P',P, 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    matf = fullfile(here,'afocal4_s4b.mat');
    % RESUME BY DEFAULT.  The sections cost very different amounts -- the
    % anchor is half an hour of solving, the folded demonstration two
    % minutes -- so they are meant to be run apart, and a driver that threw
    % away the expensive one every time the cheap one was re-run would make
    % that impossible.
    if isfile(matf)
        q = load(matf);
        if isfield(q,'R')
            fn = fieldnames(q.R);
            for i = 1:numel(fn)
                if ~any(strcmp(fn{i}, {'P','when'})), R.(fn{i}) = q.R.(fn{i}); end
            end
            fprintf('  resuming from %s [%s]\n', matf, strjoin(fn(:).', ' '));
        end
    end
    ck = @(R_) ckpt_(matf, R_, opts.save);

    % =====================================================================
    if any(opts.sections == 0)
    banner('0.  THE CONSTRAINT, IN NUMBERS');
    % =====================================================================
    fprintf(['  Sky is at -z, so BEHIND the primary is +z.  Three decks\n' ...
             '  through the same engine-truth gate: his parent (the\n' ...
             '  existence proof Dave names), the delivered S4 rung (the\n' ...
             '  retraction), and a compliant seed.\n']);
    G = struct('name',{},'deck',{},'K',{});
    for c = {{'his 3-mirror parent','afocal4_parent3.in'}, ...
             {'S4 rung 4, delivered','afocal4_r4_tiltdec.in'}}
        f = fullfile(here, c{1}{2});
        if ~isfile(f), fprintf('  (%s missing)\n', c{1}{2}); continue; end
        K = afocal4_pack(P, f);
        G(end+1) = struct('name',c{1}{1}, 'deck',f, 'K',K); %#ok<AGROW>
    end
    Dc = afocal4_seed(P);
    fc = deck_(here, 'b_gate_seed', opts.save);   % not 'b_seed': the
                                  % prefixed ladder owns that name

    bc = afocal4_build(P, Dc, fc, 'quiet',false);
    G(end+1) = struct('name','S4b compliant seed', 'deck',fc, ...
                      'K',afocal4_pack(P, fc));
    R.gate = G;   R.seed = struct('D',Dc,'b',bc);
    fprintf('\n  %-26s %10s %12s %12s %s\n', ...
            'deck','behind M1','fold gap','instr clear','verdict');
    for i = 1:numel(G)
        K = G(i).K;
        fprintf('  %-26s %8.0f mm %9.1f mm %9.1f mm  %s\n', G(i).name, ...
                K.behind_m1*1e3, 1e3*pick_(K,'gap'), 1e3*pick2_(K), ...
                yn_(K.ok));
    end
    ck(R);
    end

    % =====================================================================
    if any(opts.sections == 1)
    banner('1.  THE ANCHOR -- his front end held verbatim');
    % =====================================================================
    fprintf(['  The brief''s first thing to try: hold his M1->M2 spacing and\n' ...
             '  his M2->image geometry (which is to say M2''s radius), so the\n' ...
             '  front end is EXACTLY the benchmark''s, and let the conics, the\n' ...
             '  field-mirror standoff and phi4 (consumed by the closure) carry\n' ...
             '  the whole job.  His back end already satisfies the constraint,\n' ...
             '  so if this closes, the constraint costs only what the lost\n' ...
             '  front-end freedom is worth -- and that is the number to report.\n\n' ...
             '  Operating point: iface = %.0f mm, at the +%.1f deg bias.\n\n'], ...
            P.iface*1e3, P.bias_deg);
    if ~isfield(R,'anchor')
        D = afocal4_seed(P, 'bias_deg', P.bias_deg);
        a = solve_(P, D, {'conic','standoff'}, 'anchor  conics + standoff', ...
                   deck_(here,'b_anchor',opts.save));
        R.anchor = a;
        R.anchor_build = afocal4_build(P, a.D, a.deck, 'verify',false);
        R.anchor_pack  = afocal4_pack(P, a.deck, 'quiet',true);
        ck(R);
    else
        fprintf('  (anchor already solved -- kept)\n');
    end

    % --- and the variant the anchor's own failure mode points at ---------
    % THE CONSTRAINT MOVES THE FIELD MIRROR OFF THE FIELD CONJUGATE, and
    % that is the whole cost.  With his front end held, compliance needs the
    % fourth mirror 250-600 mm PAST the intermediate image: it gains a
    % footprint (so its conic finally works on the wavefront) and loses the
    % property the S3 form study chose it for (at the image it moves pupil
    % imaging at first order and leaves image quality alone).  So the anchor
    % reads much BETTER in wavefront and much WORSE in pupil than the
    % unbuildable S4 design -- the exchange rate runs the other way.
    %
    % There is one way to have both, and it is a front-end change rather
    % than a back-end one: a SLOWER SECONDARY pushes the intermediate image
    % further behind the primary, and once the image itself is ~1.25 m
    % behind M1 the field mirror can sit ON it and the collimator still
    % lands 500+ mm behind.  That costs telescope length.  The solver can
    % reach it -- M2's radius is a DOF -- but only if something starts it
    % there, because his radius is 6% away and the basin in between is not
    % obviously downhill.  So it is seeded explicitly and reported beside
    % the anchor rather than left to luck.
    if ~isfield(R,'front_variant')
        [Dv, iv] = image_behind_seed_(P);
        if ~iv.ok
            fprintf(['  no front end in bounds puts the intermediate image ' ...
                     'far enough back -- variant skipped\n']);
        else
            fprintf(['\n  front-end variant seed: R_M2 %.1f mm (his %.1f), ' ...
                     's_FM %+.0f mm, image %.3f m behind M1, %s %.0f mm behind\n'], ...
                    Dv.R2*1e3, P.parent.R(2)*1e3, Dv.fm_standoff*1e3, ...
                    iv.z_img, iv.name, iv.behind_m1*1e3);
            v = solve_(P, Dv, {'conic','standoff','front'}, ...
                       'variant  image behind M1', ...
                       deck_(here,'b_frontvar',opts.save));
            R.front_variant = v;
            R.front_variant_build = afocal4_build(P, v.D, v.deck, 'verify',false);
            R.front_variant_pack  = afocal4_pack(P, v.deck, 'quiet',true);
            ck(R);
        end
    end

    if isfield(R,'anchor') && isfield(R,'front_variant')
        fprintf('\n  %-28s %9s %9s %9s %9s %9s %8s\n', ...
                'at iface 140 mm, +0.6 deg', 'WFE nm','blur um','breathe %', ...
                'wander um','M','worst');
        for c = {{'anchor (his front end)',R.anchor.S}, ...
                 {'variant (image behind M1)',R.front_variant.S}}
            S = c{1}{2};
            fprintf('  %-28s %9.1f %9.1f %9.3f %9.1f %9.4f %8.2f\n', c{1}{1}, ...
                    S.wfe_max_nm, S.blur_um, S.breathe_pct, S.wander_um, ...
                    S.mag_centre_chief, S.worst);
        end
    end
    end

    % =====================================================================
    if any(opts.sections == 2)
    banner('2.  THE BUILDABLE TRADE');
    % =====================================================================
    lm = fullfile(here,'afocal4_b_ladder.mat');
    if strcmp(opts.trade,'load') && isfile(lm)
        Q = load(lm);   R.ladder = Q.R;
        fprintf('  loaded %s (%d rungs, %d trade points)\n', lm, ...
                numel(R.ladder.rung), numel(R.ladder.trade));
    else
        R.ladder = afocal4_ladder('prefix','b_', 'sections',[0 1 2 4], ...
                                  'save',opts.save);
    end
    ck(R);
    end

    % =====================================================================
    if any(opts.sections == 3)
    banner('3.  THE FOLDED DEMONSTRATION');
    % =====================================================================
    [D, src] = fold_subject_(R, opts.fold_from);
    if isempty(D)
        fprintf('  no design to fold (asked for "%s")\n', opts.fold_from);
    else
        fprintf('  folding the %s design\n', src);
        R.fold = fold_demo_(P, here, D, opts.save);
        R.fold.source = src;
        ck(R);
    end
    end

    % =====================================================================
    if any(opts.sections == 5) && isfield(R,'front_variant')
    banner('5.  THE SECOND BASIN -- the trade with the image behind M1');
    % =====================================================================
    fprintf(['  Section 2''s sweep stays in the basin its seeder hands it: his\n' ...
             '  front end, field mirror far PAST the intermediate image.  That\n' ...
             '  basin never gets the pupil blur below 759 um anywhere on the\n' ...
             '  curve, while the front-end variant of section 1 reads 149 um at\n' ...
             '  the same operating point.  A trade curve swept in the wrong\n' ...
             '  basin is not the trade, so the second basin is swept too -- same\n' ...
             '  machinery, same merit, same gates, seeded at each point by the\n' ...
             '  shortest front end that puts the image far enough behind M1.\n\n']);
    Tv = struct('iface',{},'phi4',{},'R_fm',{},'R_col',{},'s_fm',{},'S',{}, ...
                'D',{},'behind_m1',{},'pack',{},'seed',{});
    warm = [];
    for q = P.iface_trade
        cand = struct('name',{},'D',{});
        if ~isempty(warm), cand(end+1) = struct('name','warm','D',setf_(warm,'iface',q)); end
        [Dv, iv] = image_behind_seed_(P, q);
        if iv.ok, cand(end+1) = struct('name','fresh','D',Dv); end
        best = [];
        for j = 1:numel(cand)
            try
                dk = deck_(here, sprintf('b2_trade_%03.0fmm_%s', q*1e3, ...
                                         cand(j).name), opts.save);
                sq = solve_(P, cand(j).D, {'conic','standoff','front'}, ...
                            sprintf('basin-2  iface %.0f mm (%s)', q*1e3, ...
                                    cand(j).name), dk);
                if isempty(best) || sq.S.worst < best.S.worst
                    best = sq;   best.seed = cand(j).name;
                end
            catch ME
                fprintf('    %-5s seed | NO POINT: %s\n', cand(j).name, ME.message);
            end
        end
        if isempty(best)
            fprintf('  iface %6.1f mm  NO POINT from either seed\n', q*1e3);
            continue;
        end
        warm = best.D;
        dk = deck_(here, sprintf('b2_trade_%03.0fmm', q*1e3), opts.save);
        bq = afocal4_build(P, best.D, dk, 'verify',false);
        K  = afocal4_pack(P, dk, 'quiet',true);
        Tv(end+1) = struct('iface',q, 'phi4',bq.phi4, 'R_fm',bq.R(3), ...
                           'R_col',bq.R(4), 's_fm',best.D.fm_standoff, ...
                           'S',best.S, 'D',best.D, 'behind_m1',bq.behind_m1, ...
                           'pack',K, 'seed',best.seed); %#ok<AGROW>
        fprintf(['  iface %6.1f mm  R_M2 %6.1f mm  phi4 %+6.3f  s_FM %+5.0f mm | ' ...
                 'WFE %8.1f nm  blur %7.1f  breathe %6.3f%%  wander %7.1f um  ' ...
                 'worst %6.2fx | behind %4.0f mm, instr dia max %4.0f mm  %s\n'], ...
                q*1e3, best.D.R2*1e3, bq.phi4, best.D.fm_standoff*1e3, ...
                best.S.wfe_max_nm, best.S.blur_um, best.S.breathe_pct, ...
                best.S.wander_um, best.S.worst, bq.behind_m1*1e3, ...
                K.instr.dia_max*1e3, yn_(K.ok));
        R.trade2 = Tv;   ck(R);
    end
    R.trade2 = Tv;   ck(R);
    end

    % =====================================================================
    if any(opts.sections == 4) && opts.save
    banner('4.  FIGURES');
    % =====================================================================
    try
        png = fullfile(here,'afocal4_b_trade_compare.png');
        trade_compare_(P, here, R, png);   fprintf('  wrote %s\n', png);
    catch ME, fprintf('  trade comparison failed: %s\n', ME.message); end
    if isfield(R,'fold') && ~isempty(R.fold)
        for c = {{'unfolded',R.fold.deck_plain}, {'folded',R.fold.deck}}
            try
                png = fullfile(here, sprintf('afocal4_b_layout_%s.png', c{1}{1}));
                layout_fig_(P, c{1}{2}, c{1}{1}, png, R.fold);
                fprintf('  wrote %s\n', png);
            catch ME
                fprintf('  %s layout failed: %s\n', c{1}{1}, ME.message);
            end
        end
    end
    end

    ck(R);
    if opts.save, fprintf('\n  saved %s\n', matf); end
end

% =====================================================================
function [D, info] = image_behind_seed_(P, iface)
%IMAGE_BEHIND_SEED_  A compliant seed with the FIELD MIRROR BACK ON THE
%   INTERMEDIATE IMAGE, bought by slowing the secondary.
%
%   The packaging constraint says the collimator sits >= 500 mm behind M1.
%   The collimator sits a focal length back toward the primary FROM the
%   field mirror, and the field mirror sits at (or near) the intermediate
%   image, so the requirement passes straight through to the image: put it
%   far enough behind M1 and the whole back end follows.  M2's radius is
%   what moves it -- a slower secondary lengthens the back focus, at a
%   price in overall length that is the variant's whole cost.
%
%   Searched rather than derived: the image station is a steep function of
%   M2's radius near the point where the secondary stops converging the
%   beam at all, so the useful range is narrow and worth finding
%   numerically instead of linearising.
    if nargin < 2, iface = P.iface; end
    D = afocal4_seed(P, 'bias_deg', P.bias_deg, 'iface', iface);
    info = struct('ok',false, 'z_img',NaN, 'behind_m1',NaN, 'name','');
    % DESCENDING from his radius, because the variant's cost IS length: the
    % image station runs away steeply as the secondary slows, so the first
    % radius that clears the bound is the shortest telescope that does, and
    % scanning the other way would seed the solver at a 3.6 m envelope when
    % a 2.3 m one is compliant.
    for R2 = 0.465:-0.0025:0.425
        Q = P;   Q.parent.R(2) = R2;
        p = afocal_first_order(Q.parent.R, Q.parent.t, Q.parent.convex, ...
                               'D',P.D, 'stop_ahead',P.stop_ahead);
        a0 = -p.y_marginal(2)/p.u_marginal(2);
        if ~(a0 > 0) || a0 > 6, continue; end
        z_img = -Q.parent.t(1) + a0;
        % keep the field mirror AT the image (|s| small) and take the
        % SHORTEST telescope that still clears the bound with margin
        for s = [0 0.05 -0.05 0.10 -0.10]
            [phi4, C, found] = afocal4_phi4(Q, s, iface);
            if ~found || C.behind_m1 < P.pack.m3_behind_min + 0.05, continue; end
            D.R2 = R2;   D.fm_standoff = s;
            info = struct('ok',true, 'z_img',z_img, ...
                          'behind_m1',C.behind_m1, 'name',C.names{end}, ...
                          'phi4',phi4, 'R2',R2, 'standoff',s);
            return;
        end
    end
end

function s = solve_(P, D, dofs, label, deck)
%SOLVE_  A rung's joint solve, seeded by the short conics-only pass the S4
%   runs earned (RESULTS rule 4).  Same recipe as AFOCAL4_LADDER's, kept
%   here so this driver does not reach into its private helpers.
    pre = afocal4_solve(P, D, 'dofs',{'conic'}, 'max_iter',12, 'quiet',true, ...
                        'label',[label ' (seed)']);
    fprintf('  %s: conic seed -> WFE %.1f nm, worst %.2fx\n', ...
            label, pre.S.wfe_max_nm, pre.S.worst);
    s = afocal4_solve(P, pre.D, 'dofs',dofs, 'label',label, 'deck',deck, ...
                      'max_iter',P.solve.max_iter);
    s.pre = pre;
end

% ---------------------------------------------------------------------
function [D, src] = fold_subject_(R, want)
%FOLD_SUBJECT_  Which design the folded demonstration is built from.  The
%   fold is a packaging statement about a PARTICULAR design, so the caller
%   names it rather than the driver guessing.
    D = [];   src = want;
    switch want
    case 'ladder'
        if isfield(R,'ladder') && isfield(R.ladder,'rung')
            D = R.ladder.rung(end).D;   src = 'delivered ladder rung';
        end
    case 'anchor'
        if isfield(R,'anchor'), D = R.anchor.D;   src = 'anchor'; end
    case 'variant'
        if isfield(R,'front_variant')
            D = R.front_variant.D;   src = 'front-end variant';
        end
    case 'best'
        % Every trade point from both basins, filtered to the ones that
        % clear the whole gate -- station, fold daylight AND instrument
        % volume -- then the lowest worst-miss.  A folded picture of a
        % design that does not pass the gate would be arguing the opposite
        % of what the figure claims.
        C = struct('D',{},'worst',{},'tag',{});
        for f = {'trade','trade2'}
            if ~isfield(R,'ladder') && strcmp(f{1},'trade'), continue; end
            if strcmp(f{1},'trade')
                if ~isfield(R.ladder,'trade'), continue; end
                T = R.ladder.trade;   nm = 'his front end';
            else
                if ~isfield(R,'trade2'), continue; end
                T = R.trade2;         nm = 'image behind M1';
            end
            for i = 1:numel(T)
                if ~T(i).pack.ok, continue; end
                C(end+1) = struct('D',T(i).D, 'worst',T(i).S.worst, ...
                    'tag',sprintf('%s, iface %.0f mm', nm, T(i).iface*1e3)); %#ok<AGROW>
            end
        end
        if ~isempty(C)
            [~,k] = min([C.worst]);
            D = C(k).D;   src = ['best gated (' C(k).tag ')'];
        end
    case 'basin2'
        if isfield(R,'trade2')
            T = R.trade2;   ok = arrayfun(@(t) t.pack.ok, T);
            if any(ok)
                Tk = T(ok);   [~,k] = min(arrayfun(@(t) t.S.worst, Tk));
                D = Tk(k).D;
                src = sprintf('image behind M1, iface %.0f mm', Tk(k).iface*1e3);
            end
        end
    end
end

function F = fold_demo_(P, here, D, dosave)
%FOLD_DEMO_  Insert the flat, emit it, and check that nothing moved.
%
%   The fold is NOMINALLY NULL -- Telescope/add_fold maps everything
%   downstream by the fold plane's reflection isometry, which preserves
%   path lengths and angles exactly -- but "nominally null" has been wrong
%   before (e2e2 s3: a fold's hole and clearances interact with the bias
%   field).  So the folded deck is re-scored on the same kernel and the two
%   score blocks are printed side by side.  If they differ by more than
%   round-off, the fold is not null and the design is re-solved with it in
%   place; that is a decision this function REPORTS rather than hides.
    plain = deck_(here, 'b_final', dosave);
    b0 = afocal4_build(P, D, plain, 'verify',false);
    S0 = afocal4_score(P, plain, 'nodes',P.solve.nodes_score, 'grid',P.grid_n);
    K0 = afocal4_pack(P, plain);

    % the fold station the gate chose, as a distance along the exit leg:
    % the vertices of a coaxial train lie on the axial chief path, so the
    % beam direction after the last mirror is +-z and the two are the same
    % number.
    dist = abs(K0.fold_pick.z - b0.z(end));
    Df = D;
    Df.fold = struct('dist',dist, 'to',P.pack.fold_to, ...
                     'ap_r', 3*K0.fold_pick.hw_out);
    fold = deck_(here, 'b_final_folded', dosave);
    b1 = afocal4_build(P, Df, fold, 'quiet',false);
    S1 = afocal4_score(P, fold, 'nodes',P.solve.nodes_score, 'grid',P.grid_n);

    fprintf(['\n  THE FOLD IS NULL -- MEASURED, NOT ASSERTED.  Same design, ' ...
             'flat inserted\n  %.0f mm past %s, beam turned into [%s].\n\n'], ...
            dist*1e3, b0.names{end}, strtrim(sprintf('%g ', P.pack.fold_to)));
    fprintf('  %-22s %12s %12s %12s %12s %12s\n', ...
            '', 'WFE nm', 'blur um', 'breathe %', 'wander um', 'M');
    fprintf('  %-22s %12.2f %12.2f %12.4f %12.2f %12.5f\n', 'unfolded', ...
            S0.wfe_max_nm, S0.blur_um, S0.breathe_pct, S0.wander_um, ...
            S0.mag_centre_chief);
    fprintf('  %-22s %12.2f %12.2f %12.4f %12.2f %12.5f\n', 'folded', ...
            S1.wfe_max_nm, S1.blur_um, S1.breathe_pct, S1.wander_um, ...
            S1.mag_centre_chief);
    rel = @(a,b) abs(b/a - 1);
    d = [rel(S0.wfe_max_nm,S1.wfe_max_nm), rel(S0.blur_um,S1.blur_um), ...
         rel(S0.wander_um,S1.wander_um), rel(S0.mag_centre_chief,S1.mag_centre_chief)];
    fprintf('  %-22s %12.1e %12.1e %12s %12.1e %12.1e\n', 'relative', ...
            d(1), d(2), '-', d(3), d(4));
    if max(d) > 1e-3
        fprintf(['\n  !! the fold is NOT null at the 1e-3 level.  Re-solve with ' ...
                 'it in place\n     before quoting the folded numbers.\n']);
    else
        fprintf(['\n  Every column agrees to better than 1e-3, so the folded ' ...
                 'prescription is\n  the same telescope and the S4b numbers ' ...
                 'carry over to it unchanged.\n']);
    end

    % ---- where everything ended up, from the folded deck ---------------
    V  = grab3_(fileread(fold),'VptElt');   nm = grab_names_(fileread(fold));
    cs = V(:,end);
    ex = cs - V(:,end-1);   ex = ex/norm(ex);         % fold -> pupil
    tip = cs + ex*P.pack.instr_len;                   % far end of the envelope
    hw  = 0.5*P.pack.instr_dia;
    F = struct('deck',fold, 'deck_plain',plain, 'dist',dist, 'D',Df, ...
               'S_plain',S0, 'S',S1, 'pack_plain',K0, 'b',b1, ...
               'pupil',cs.', 'tip',tip.', 'dir',ex.', ...
               'z_min',min(cs(3),tip(3)) - hw, ...
               'z_max',max(cs(3),tip(3)) + hw, 'names',{nm});
    F.behind_m1 = F.z_min;
    F.clear = clearance_(fold, F, P);
    fprintf(['\n  FOLDED PACKAGE.  Interface pupil at [%+.3f %+.3f %+.3f] m, ' ...
             'envelope\n  running to [%+.3f %+.3f %+.3f] m.  Its z-slab is ' ...
             '%+.3f .. %+.3f m --\n  entirely %s the primary.  Closest ' ...
             'approach of any other bundle: %.0f mm.\n'], ...
            cs, tip, F.z_min, F.z_max, ...
            ternary_(F.z_min >= 0, 'BEHIND', 'IN FRONT OF'), F.clear*1e3);
end

function c = clearance_(deck, F, P)
%CLEARANCE_  Engine-truth: how close does any traced bundle other than the
%   folded exit beam come to the instrument envelope's axis, inside the
%   z-slab the envelope occupies?  Read from the XZ and YZ fans of the
%   FOLDED deck, so it is the real geometry and not a projection of the
%   unfolded one.
    macos.load_rx(deck);
    nE = macos.num_elt();
    c  = Inf;
    a0 = F.pupil(:);   ad = F.dir(:);
    for pl = {'YZ','XZ'}
        b = macos.draw_rays(pl{1}, 0, nE);
        for r = 1:b.nray
            for i = 1:b.nper(r)-1
                ea = b.elt(i,r);  eb = b.elt(i+1,r);
                if ea < 1 || eb ~= ea+1 || ea >= nE-1, continue; end
                for q = 0:0.05:1
                    z = b.U(i,r) + q*(b.U(i+1,r)-b.U(i,r));
                    v = b.V(i,r) + q*(b.V(i+1,r)-b.V(i,r));
                    if z < F.z_min || z > F.z_max, continue; end
                    if strcmp(pl{1},'YZ'), pt = [0; v; z]; else, pt = [v; 0; z]; end
                    w = pt - a0;   t = dot(w, ad);
                    if t < 0, d = norm(w); else, d = norm(w - t*ad); end
                    c = min(c, d - 0.5*P.pack.instr_dia);
                end
            end
        end
    end
end

% ---------------------------------------------------------------------
function trade_compare_(P, here, R, png)
%TRADE_COMPARE_  The buildable curve against the S4 one it retracts.  Both
%   are drawn; the S4 curve is dashed and labelled NOT BUILDABLE, because a
%   retraction that deletes the numbers cannot be checked.
    S4 = [];
    f4 = fullfile(here,'afocal4_ladder.mat');
    if isfile(f4), q = load(f4);  if isfield(q.R,'trade'), S4 = q.R.trade; end, end
    Tb = R.ladder.trade;
    T  = P.targets;
    met = {'wfe_max_nm','blur_um','breathe_pct','wander_um'};
    tgt = [T.wfe_rung2_nm, T.blur_um, T.breathe_pct, T.wander_um];
    lab = {'WFE rung 2 (nm)','pupil blur (\mum)','breathing (%)','wander (\mum)'};

    f = figure('Visible','off','Position',[100 100 1240 800]);
    tl = tiledlayout(f,2,2,'Padding','compact','TileSpacing','compact');
    for k = 1:4
        ax = nexttile(tl);  hold(ax,'on');  grid(ax,'on');
        if ~isempty(S4)
            plot(ax, [S4.iface]*1e3, arrayfun(@(s) s.S.(met{k}), S4), ...
                 '--o', 'Color',[0.65 0.65 0.65], 'MarkerFaceColor','w', ...
                 'DisplayName','S4 (NOT BUILDABLE)');
        end
        plot(ax, [Tb.iface]*1e3, arrayfun(@(s) s.S.(met{k}), Tb), ...
             '-o', 'Color',[0.15 0.35 0.75], 'LineWidth',1.6, ...
             'MarkerFaceColor',[0.15 0.35 0.75], ...
             'DisplayName','S4b buildable, his front end');
        if isfield(R,'trade2') && ~isempty(R.trade2)
            T2 = R.trade2;
            plot(ax, [T2.iface]*1e3, arrayfun(@(s) s.S.(met{k}), T2), ...
                 '-s', 'Color',[0.85 0.40 0.10], 'LineWidth',1.6, ...
                 'MarkerFaceColor',[0.85 0.40 0.10], ...
                 'DisplayName','S4b buildable, image behind M1');
        end
        yline(ax, tgt(k), ':', 'target', 'Color',[0.8 0.2 0.2], ...
              'LabelHorizontalAlignment','left', 'HandleVisibility','off');
        set(ax,'YScale','log');
        xlabel(ax,'interface standoff (mm)');   ylabel(ax, lab{k});
        if k == 1, legend(ax,'Location','best','Box','off'); end
    end
    title(tl, ['The interface-standoff trade, buildable and not  ' ...
               '(z_{M3} \geq ' sprintf('%.0f', P.pack.m3_behind_min*1e3) ' mm behind M1)'], ...
          'FontWeight','bold');
    exportgraphics(f, png, 'Resolution', 140);   close(f);
end

function layout_fig_(P, deck, tag, png, F)
%LAYOUT_FIG_  yz and xz of one deck, from the engine's own ray history,
%   with the instrument envelope drawn as the box it is.  The point of the
%   figure is the VOLUME, so M1's keep-out cylinder is drawn too: a package
%   is buildable when the box is behind that line and the beams are not in
%   the box.
    macos.load_rx(deck);
    nE  = macos.num_elt();
    txt = fileread(deck);
    V   = grab3_(txt,'VptElt');   nm = grab_names_(txt);
    f = figure('Visible','off','Position',[100 100 1250 620]);
    tl = tiledlayout(f,2,1,'Padding','compact','TileSpacing','compact');
    for pl = {'YZ','XZ'}
        ax = nexttile(tl);  hold(ax,'on');  axis(ax,'equal');  grid(ax,'on');
        b = macos.draw_rays(pl{1}, 0, nE);
        for r = 1:4:b.nray
            n = b.nper(r);
            plot(ax, b.U(1:n,r), b.V(1:n,r), '-', 'Color',[0.25 0.5 0.9 0.35], ...
                 'LineWidth',0.4);
        end
        % Labels alternate above/below and step outward: the back end of a
        % folded afocal packs four elements into 200 mm, and a fixed offset
        % stacks "M3 ColdStop Fold" on top of each other.
        for k = 1:numel(nm)
            if strcmp(pl{1},'YZ'), yv = V(2,k); else, yv = V(1,k); end
            plot(ax, V(3,k), yv, 'ko', 'MarkerFaceColor','k', 'MarkerSize',4);
            up = 1 - 2*mod(k,2);
            text(ax, V(3,k), yv + up*0.055*(1 + 0.6*mod(k,3)), nm{k}, ...
                 'FontSize',9, 'FontWeight','bold', ...
                 'HorizontalAlignment','center', 'Clipping','on');
            plot(ax, [V(3,k) V(3,k)], [yv, yv + up*0.05*(1 + 0.6*mod(k,3))], ...
                 '-', 'Color',[0.6 0.6 0.6], 'LineWidth',0.6);
        end
        xline(ax, 0, '-', 'M1 plane', 'Color',[0.7 0.2 0.2], 'FontSize',8);
        if ~isempty(F) && strcmp(tag,'folded')
            drawbox_(ax, pl{1}, F, P);
        end
        xlabel(ax,'z (m)');   ylabel(ax, [pl{1}(1) ' (m)']);
        title(ax, sprintf('%s  --  %s', tag, pl{1}));
    end
    kcol = find(~ismember(nm, {'Fold','ColdStop'}), 1, 'last');
    title(tl, sprintf(['afocal4 S4b %s layout:  the collimator sits %.0f mm ' ...
                       'behind M1 and the instrument is out of the beam'], ...
                      tag, 1e3*V(3,kcol)), 'FontWeight','bold');
    exportgraphics(f, png, 'Resolution', 140);   close(f);
end

function drawbox_(ax, pl, F, P)
    hw = 0.5*P.pack.instr_dia;
    a = F.pupil(:);   b = F.tip(:);
    if strcmp(pl,'YZ'), i1 = 2; else, i1 = 1; end
    zz = [a(3) b(3)];   vv = [a(i1) b(i1)];
    patch(ax, 'XData',[min(zz)-hw max(zz)+hw max(zz)+hw min(zz)-hw], ...
              'YData',[min(vv)-hw min(vv)-hw max(vv)+hw max(vv)+hw], ...
              'FaceColor',[0.95 0.75 0.2], 'FaceAlpha',0.18, ...
              'EdgeColor',[0.7 0.5 0.1], 'LineStyle','--');
    text(ax, mean(zz), max(vv)+hw, ' instrument envelope', 'FontSize',8, ...
         'Color',[0.6 0.4 0.05], 'VerticalAlignment','bottom');
end

% ---------------------------------------------------------------------
function ckpt_(matf, R, dosave)
    if ~dosave, return; end
    save(matf, 'R', '-v7.3');
end

function f = deck_(here, tag, dosave)
    if dosave, f = fullfile(here, sprintf('afocal4_%s.in', tag));
    else,      f = [tempname '.in'];
    end
end

function g = pick_(K, f)
    g = NaN;
    if isfield(K,'fold_pick') && ~isempty(K.fold_pick) && isfield(K.fold_pick,f)
        g = K.fold_pick.(f);
    end
end

function g = pick2_(K)
    g = NaN;
    if isfield(K,'instr') && ~isempty(K.instr) && isfield(K.instr,'clear_m')
        g = K.instr.clear_m;
    end
end

function s = yn_(b),  if b, s = 'BUILDABLE'; else, s = 'not buildable'; end,  end

function D = setf_(D, f, v),  D.(f) = v;  end
function v = ternary_(c,a,b),  if c, v = a; else, v = b; end,  end

function M = grab3_(txt, key)
    t = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens');
    M = zeros(3, numel(t));
    for i = 1:numel(t), M(:,i) = sscanf(strrep(t{i}{1},'D','E'), '%f', 3); end
end

function n = grab_names_(txt)
    t = regexp(txt, '(?m)^\s*EltName=\s*(\S*)', 'tokens');
    n = cellfun(@(c) c{1}, t, 'UniformOutput', false);
end

function banner(s)
    fprintf('\n%s\n%s\n%s\n', repmat('=',1,74), s, repmat('=',1,74));
end
