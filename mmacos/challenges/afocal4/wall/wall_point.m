function R = wall_point(opts)
%WALL_POINT  One point of the walled, converged tilt-vs-price frontier.
%
%   R = WALL_POINT('tilt',-9, 'union_min',0.015) seeds a compliant design at
%   that extraction tilt, re-solves the conics, the field-mirror standoff and
%   the front end around it WITH the union body-in-beam wall in force and
%   with CENTRAL differences, restarting until the merit plateaus, and then
%   gates and scores the answer at reporting sampling.
%
%   ONE POINT, ONE PROCESS.  Each call is self-contained -- it initialises
%   the engine, recovers the parent design and writes its own checkpoint --
%   so the frontier is run as one MATLAB process per point in parallel and
%   assembled afterwards by AFOCAL4_WALL.  That is the AFOCAL4_BASIN2 'tag'
%   pattern, and the reason is the same: a re-solve is a multi-hour artifact
%   and must survive the process that produced it (the
%   save-resumable-workspaces rule).
%
%   TWO THINGS THIS DOES THAT THE CLEARING STAGE'S RE-SOLVES DID NOT, and
%   they are the whole point of the slice:
%
%   1  THE WALL IS IN FORCE.  The clearing stage measured that a re-solve
%      SPENDS clearance -- at -8 and -9 deg it walked +23.3 and +42.3 mm of
%      margin down to +2.3 and +0.7 mm -- because AFOCAL4_SCORE cannot see
%      it.  Here the union floor is a wall in the builder
%      (P.pack.union_enforce), never a merit term, seeded compliantly by
%      WALL_SEED.  'wall',false reproduces the clearing stage's own
%      treatment and is what makes the non-vacuity A/B possible.
%
%   2  IT IS CONVERGED, NOT BUDGET-CAPPED.  The delivered clearing numbers
%      came out of 427 evaluations at exitflag 0, on the study's default
%      3e-3 FORWARD difference -- the setting S4c MEASURED as reading this
%      merit's gradient 17 % low on an objective smooth to 1e-5.  The stalls
%      that produced were misread as convergence once already.  Here:
%      central differences at 1e-4, FunctionTolerance 1e-8, StepTolerance
%      1e-9, and RESTART rounds (a fresh trust region from the converged
%      point) until the solver returns exitflag 1 or a round buys less than
%      'plateau' of the merit.  Exitflags and evaluation counts are
%      reported per round, so "converged" and "plateau" are told apart by a
%      number and not by an exit code.
%
%   A SAMPLING NOTE THAT HAS TO BE STATED.  The wall is evaluated inside the
%   solver at SOLVE sampling (ngrid 21) and the frontier table quotes the
%   gate at REPORTING sampling (ngrid 41).  More rays make a bigger union
%   hull, so the solve-sampling floor is the optimistic one -- measured at
%   about 1 mm on this design, and R.sampling_bias_mm carries it per point.
%   The reported floor is the REPORTING one, because that is the number the
%   standing gate produces; the seeder's 10 mm margin is what keeps the
%   difference from mattering.
%
%   Name-value:
%     'tilt'       extraction tilt, deg (required in practice)
%     'union_min'  the wall's floor, m (0)
%     'wall'       apply it at all (true)
%     'rounds'     restart rounds (3)
%     'evals'      function evaluations per round (300)
%     'plateau'    relative merit gain below which a round is a plateau (1e-6)
%     'seed_margin' m of clearance the seed must hold over union_min (0.010)
%     'seed_standoff' m -- FORCE the field-mirror standoff to this value and
%                  skip the seeder search.  The control for the frontier's own
%                  confound: with 'standoff' in the DOF set the solver walks
%                  it from the parent's -38.6 mm out to +230..+536 mm and the
%                  wavefront falls monotonically along the way, so points at
%                  different tilts differ mostly by HOW FAR EACH SOLVE GOT.
%                  Fixing the standoff and dropping it from 'dofs' makes the
%                  tilt the only difference between points, which is what a
%                  tilt-vs-price curve has to mean.
%     'pupil_w'    multiply the blur / breathing / wander merit WEIGHTS by
%                  this (1 = the study's own).  The addendum's question --
%                  at a fixed tilt and wall, does a pupil-weighted merit
%                  recover blur without giving back wavefront or margin?  The
%                  merit DOCTRINE is untouched: log-domain residuals and
%                  walls-not-terms both stand, and P.weights has always
%                  carried these as knobs.  It measures the slack; it does
%                  not propose a re-weighting.
%     'dofs'       ({'conic','standoff','front'} -- the clearing stage's set)
%     'axis'       tilt axis ([1 0 0])
%     'deck'       parent deck (../afocal4_b2long_343mm.in)
%     'tag'        artifact suffix (default built from tilt and union_min)
%     'out'        directory for the checkpoint and the deck (this one)
%     'model'      engine model size (256)
%     'save'       write the .mat and the .in (true)
%
%   Returns R with .tilt_deg .union_min .wall .seed .rounds (per-round
%   exitflag / nfev / merit) .D .S (the score) .gate_body .gate_bare
%   .floor_body_m .floor_bare_m .offset_mm .deck .seconds.
%
%   See also WALL_SEED, AFOCAL4_WALL, CLEAR_SOLVE, AFOCAL4_UNION_WALL.

    arguments
        opts.tilt        (1,1) double = -10
        opts.union_min   (1,1) double = 0
        opts.wall        (1,1) logical = true
        opts.rounds      (1,1) double = 3
        opts.evals       (1,1) double = 300
        opts.plateau     (1,1) double = 1e-6
        opts.seed_margin (1,1) double = 0.010
        opts.dofs        (1,:) cell = {'conic','standoff','front'}
        opts.seed_standoff (1,1) double = NaN
        opts.pupil_w     (1,1) double = 1
        opts.axis        (1,3) double = [1 0 0]
        opts.deck        (1,:) char = 'afocal4_b2long_343mm.in'
        opts.tag         (1,:) char = ''
        opts.out         (1,:) char = ''
        opts.model       (1,1) double = 256
        opts.save        (1,1) logical = true
    end
    t0 = tic;
    here = fileparts(mfilename('fullpath'));
    up   = fileparts(here);
    addpath(here);  addpath(up);  addpath(fullfile(up,'clearing'));
    addpath(fullfile(up,'packaging'));
    if isempty(opts.out), opts.out = here; end
    tag = opts.tag;
    if isempty(tag)
        tag = sprintf('t%+03.0f_u%02.0f%s', opts.tilt*10, opts.union_min*1e3, ...
                      tern_(opts.wall,'','_nowall'));
    end
    src = fullfile(up, opts.deck);
    if ~isfile(src), error('wall_point:deck','no such deck: %s', src); end

    macos.init(opts.model);
    P = afocal4_params();
    F = P.Fsolve;

    % ---- the parent, recovered and VERIFIED against its own file ---------
    D0 = wall_recover(P, src);
    fprintf('\n==== WALL POINT  tilt %+.1f deg, wall %s at %+.1f mm  [%s] ====\n', ...
            opts.tilt, tern_(opts.wall,'ON','OFF'), opts.union_min*1e3, tag);
    fprintf('  parent %s recovered and rebuilt byte-for-byte\n', opts.deck);

    % ---- the solver's P: wall, and the converged settings ----------------
    Q = P;
    Q.pack.union_enforce = opts.wall;
    Q.pack.union_min     = opts.union_min;
    Q.solve.fd_step = 1e-4;      Q.solve.fd_type = 'central';
    Q.solve.tol_fun = 1e-8;      Q.solve.tol_x   = 1e-9;
    Q.solve.tol_opt = 1e-8;      Q.solve.max_fev = opts.evals;
    if opts.pupil_w ~= 1
        Q.weights.blur    = P.weights.blur    * opts.pupil_w;
        Q.weights.breathe = P.weights.breathe * opts.pupil_w;
        Q.weights.wander  = P.weights.wander  * opts.pupil_w;
        fprintf('  pupil merit weights x%g (blur, breathing, wander)\n', opts.pupil_w);
    end

    % ---- the compliant seed ----------------------------------------------
    if isfinite(opts.seed_standoff)
        Ds = D0;   Ds.tilt_deg = opts.tilt;
        Ds.fm_standoff = opts.seed_standoff;
        W = afocal4_union_wall(Q, seedcheck_(Q, Ds, opts.axis), 'fields',F, ...
                               'throw',false, 'bare',true, 'quiet',true);
        seed = struct('ok',true, ...
            'source',sprintf('standoff FORCED to %+.0f mm', opts.seed_standoff*1e3), ...
            'floor_m',W.floor_m, 'bare_m',W.bare_m, 'need_m',opts.union_min, ...
            'phi4',NaN, 'standoff',Ds.fm_standoff, 'R2',Ds.R2, 'd_m',NaN, ...
            'used_own_front_end',true, 'n_closed',0, 'n_gated',1, 'tried',[], ...
            'best_floor_m',W.floor_m, 'seconds',0);
        fprintf(['  standoff FORCED to %+.0f mm (parent %+.1f): floor %+.2f mm ' ...
                 '(bare %+.2f) against a %+.1f mm wall\n'], ...
                opts.seed_standoff*1e3, D0.fm_standoff*1e3, W.floor_m*1e3, ...
                W.bare_m*1e3, opts.union_min*1e3);
        if opts.wall && W.floor_m < opts.union_min
            error('wall_point:forced', ...
                  ['the forced standoff does not clear the wall (%+.2f mm ' ...
                   'against %+.1f) -- this is a SEEDING failure, not a ' ...
                   'design verdict.'], W.floor_m*1e3, opts.union_min*1e3);
        end
    elseif opts.wall
        [Ds, seed] = wall_seed(Q, D0, opts.tilt, 'margin',opts.seed_margin, ...
                               'fields',F, 'axis',opts.axis, 'quiet',false);
        if ~seed.ok
            R = struct('tilt_deg',opts.tilt, 'union_min',opts.union_min, ...
                       'wall',opts.wall, 'seed',seed, 'ok',false, ...
                       'why','no compliant seed', 'tag',tag, ...
                       'seconds',toc(t0));
            fprintf(['\n  NO COMPLIANT SEED at tilt %+.1f deg with a %+.1f mm ' ...
                     'wall.  This is a SEEDING result, not a design verdict:\n' ...
                     '  the best floor reachable by re-posing was %+.2f mm.\n'], ...
                     opts.tilt, opts.union_min*1e3, seed.best_floor_m*1e3);
            if opts.save
                save(fullfile(opts.out, ['wall_' tag '.mat']), 'R', '-v7.3');
            end
            return;
        end
    else
        Ds = D0;   Ds.tilt_deg = opts.tilt;
        seed = struct('ok',true, 'source','tilt alone (wall off)', ...
                      'floor_m',NaN, 'bare_m',NaN, 'need_m',-Inf, 'phi4',NaN, ...
                      'standoff',Ds.fm_standoff, 'R2',Ds.R2, 'd_m',NaN, ...
                      'used_own_front_end',true, 'n_closed',0, 'n_gated',0, ...
                      'tried',[], 'best_floor_m',NaN, 'seconds',0);
        fprintf('  wall OFF: seeded from the committed design, tilt only\n');
    end

    % ---- restart rounds, until the merit plateaus ------------------------
    deck = fullfile(opts.out, sprintf('afocal4_wall_%s.in', tag));
    D = Ds;   rounds = struct('k',{},'exitflag',{},'nfev',{},'merit',{}, ...
                              'gain',{},'seconds',{});
    mprev = Inf;   nfev_tot = 0;
    for k = 1:opts.rounds
        % A ROUND THAT THROWS MUST NOT COST THE POINT.  Each round is over an
        % hour; if one dies the design from the previous round is still a
        % converged answer and is reported as such, with the failure named.
        % (The known way this happened -- the report build being judged by
        % the wall at a sampling the wall was not enforced at -- is fixed in
        % CLEAR_SOLVE, but the guard stays: this is the pattern that turns a
        % five-hour run into nothing.)
        try
            S = clear_solve(Q, D, 'dofs',opts.dofs, 'deck',deck, ...
                    'axis',opts.axis, 'max_iter',400, ...
                    'label',sprintf('round %d', k), 'quiet',true);
        catch ME
            fprintf('  round %d FAILED (%s): %s\n', k, ME.identifier, ME.message);
            if k == 1, rethrow(ME); end
            fprintf('  -> keeping round %d''s design and reporting it.\n', k-1);
            break;
        end
        m = S.S.merit;
        if isfinite(mprev), gain = (mprev - m)/max(abs(mprev), eps);
        else,               gain = Inf;   end
        nfev_tot = nfev_tot + S.nfev;
        rounds(end+1) = struct('k',k, 'exitflag',S.exitflag, 'nfev',S.nfev, ...
                'merit',m, 'gain',gain, 'seconds',S.seconds); %#ok<AGROW>
        fprintf(['  round %d: %4d evals, %6.1f s, exitflag %d, merit %.6f ' ...
                 '(gain %.2e)\n'], k, S.nfev, S.seconds, S.exitflag, m, gain);
        D = S.D;   mprev = m;
        if S.exitflag == 1
            fprintf('  -> first-order optimality reached; stopping.\n');   break;
        end
        if k > 1 && gain < opts.plateau
            fprintf('  -> plateau (round bought %.2e < %.0e); stopping.\n', ...
                    gain, opts.plateau);   break;
        end
    end

    % ---- the answer, at REPORTING sampling -------------------------------
    Dr = D;   Dr.ngrid = P.ngrid;
    Pr = P;   Pr.pack.union_enforce = false;    % measure, do not judge
    clear_build(Pr, Dr, deck, 'axis',opts.axis, 'verify',false);
    Sc = afocal4_score(P, deck, 'fields',F, 'nodes',P.solve.nodes_score, ...
                       'grid',P.grid_n);
    Kb = afocal4_union(deck, 'fields',F, 'body_k',1.0, 'body_pad',0.0, ...
                       'quiet',true);
    Km = afocal4_union(deck, 'fields',F, 'body_k',P.pack.union_body_k, ...
                       'body_pad',P.pack.union_body_pad, 'init',false, 'quiet',true);
    L  = clear_law(deck, 'fields',F, 'leg',2, 'elt',Km.nElt-1, 'init',false, ...
                   'quiet',true);
    % ... and the same gate at SOLVE sampling, so the bias the wall was
    % judged under is a number in the record rather than a caveat.
    tsolve = [tempname '.in'];
    Dq = D;   Dq.ngrid = P.solve.ngrid;
    clear_build(Pr, Dq, tsolve, 'axis',opts.axis, 'verify',false);
    Ks = afocal4_union(tsolve, 'fields',F, 'body_k',P.pack.union_body_k, ...
                       'body_pad',P.pack.union_body_pad, 'quiet',true);
    if exist(tsolve,'file'), delete(tsolve); end
    aoi = aoi_chief_(deck);
    tr  = traced_(deck, P.D);
    % the deck is the artifact every later reader re-gates; stamp it so an
    % assembler can prove the file it opens is the one these numbers were
    % measured on (census artifacts by embedded stamps, not by filenames).
    stamp = deck_stamp_(deck);

    R = struct('tilt_deg',opts.tilt, 'union_min',opts.union_min, ...
        'wall',opts.wall, 'ok',true, 'why','', 'tag',tag, 'seed',seed, ...
        'rounds',rounds, 'nfev',nfev_tot, 'exitflag',rounds(end).exitflag, ...
        'D',D, 'S',Sc, 'deck',deck, 'deck_stamp',stamp, 'traced',tr, ...
        'floor_body_m',Km.floor_m, 'floor_bare_m',Kb.floor_m, ...
        'floor_solve_m',Ks.floor_m, ...
        'sampling_bias_mm',(Ks.floor_m - Km.floor_m)*1e3, ...
        'worst_pair',Km.worst_name, 'nLost',Km.nLost, ...
        'offset_mm',L.offset_m*1e3, 'ratio',L.ratio, ...
        'aoi_max_deg',aoi.max_deg, 'aoi_fm_deg',aoi.fm_deg, ...
        'gate_body',Km, 'gate_bare',Kb, 'seconds',toc(t0));

    fprintf(['\n  RESULT tilt %+.1f, wall %s %+.1f mm: floor %+.2f mm ' ...
             '(bare %+.2f; solve-sampling %+.2f, bias %+.2f)\n'], ...
            opts.tilt, tern_(opts.wall,'ON','OFF'), opts.union_min*1e3, ...
            Km.floor_m*1e3, Kb.floor_m*1e3, Ks.floor_m*1e3, R.sampling_bias_mm);
    fprintf(['  WFE %.1f nm | blur %.1f um | breathe %.4f %% | wander %.1f um ' ...
             '| M %.4f | lost %d\n'], Sc.wfe_max_nm, Sc.blur_um, ...
            Sc.breathe_pct, Sc.wander_um, Sc.mag_centre_chief, Km.nLost);
    fprintf(['  exit beam %.3f mm | collimation %.1f urad | traced M %.4f ' ...
             '| max chief AOI %.2f deg (FM %.2f)\n'], tr.exit_dia*1e3, ...
            tr.collimation_urad, tr.mag, aoi.max_deg, aoi.fm_deg);
    fprintf('  %d evaluations over %d round(s), %.1f min\n', nfev_tot, ...
            numel(rounds), R.seconds/60);

    if opts.save
        save(fullfile(opts.out, ['wall_' tag '.mat']), 'R', '-v7.3');
        fprintf('  wrote %s\n', fullfile(opts.out, ['wall_' tag '.mat']));
    end
end

% =====================================================================
function A = aoi_chief_(deck)
%AOI_CHIEF_  Chief-ray incidence on every mirror, from the traced chief
%   alone: a mirror turns the beam by 180 - 2*AOI, so AOI = 90 -
%   acos(d_in . d_out)/2 and no surface normal is needed.  VERBATIM the
%   construction AFOCAL4_CLEARING uses, including the station offset -- the
%   frontier's AOI column has to be comparable with the delivered row's, and
%   two slightly different readings of the same quantity is exactly how a
%   table stops meaning one thing.  The interface plane's entry is NaN
%   rather than a small number that could be mistaken for one.
    macos.load_rx(deck);
    nE = macos.num_elt();
    macos.ray_hist('on');   t = macos.trace();   h = macos.ray_hist(t.nRays);
    macos.ray_hist('off');
    Pc  = squeeze(h.P(:,1,:));   off = size(h.P,3) - nE;
    a = nan(1,nE);
    for k = 1:nE-1
        di = Pc(:,k+off)   - Pc(:,k+off-1);
        do = Pc(:,k+off+1) - Pc(:,k+off);
        if norm(di) < eps || norm(do) < eps, continue; end
        a(k) = 90 - rad2deg(acos(max(-1,min(1, ...
                    dot(di/norm(di), do/norm(do))))))/2;
    end
    A = struct('per_elt_deg',a, 'max_deg',max(a(1:nE-1)), 'fm_deg',a(3));
end

function f = seedcheck_(P, D, ax)
%SEEDCHECK_  Build the forced-standoff seed at solve sampling and hand back
%   the deck so the wall can be read on it.  Wall OFF here: a seed is
%   measured, then judged by the caller, so a failure is reported as a
%   SEEDING failure rather than raised out of the builder.
    Q = P;   Q.pack.union_enforce = false;
    Dq = D;  Dq.ngrid = P.solve.ngrid;
    f = [tempname '.in'];
    clear_build(Q, Dq, f, 'axis',ax, 'verify',false);
end

function s = traced_(deck, Dap)
%TRACED_  Exit beam, collimation and traced M of a committed deck.  Same
%   construction AFOCAL4_BUILD and CLEAR_BUILD use, so the frontier's
%   interface columns are comparable with the delivered row's.
    macos.load_rx(deck);
    tr = macos.trace(macos.num_elt());   ri = macos.get_ray_info(tr.nRays);
    ok = ri.ok_trace(:) & ri.ok_pass(:);   ok(1) = false;
    dd = ri.dir(:,ok);   dd = dd ./ vecnorm(dd);
    dm = mean(dd,2);     dm = dm/norm(dm);
    q  = ri.pos(:,ok) - mean(ri.pos(:,ok),2);
    q  = q - dm*(dm.'*q);
    dia = 2*max(vecnorm(q));
    s = struct('exit_dia',dia, 'mag',Dap/max(dia,realmin), ...
               'collimation_urad', max(acos(min(1, dm.'*dd)))*1e6, ...
               'nrays', nnz(ok));
end

function h = deck_stamp_(deck)
%DECK_STAMP_  A content hash of the emitted prescription.  Not decoration:
%   the frontier is assembled from checkpoints written by other processes,
%   and an assembler that reads a deck by FILENAME can silently pick up a
%   file some later run overwrote.  Census artifacts by their embedded
%   stamp, never by their name.
    b = uint8(fileread(deck));
    m = java.security.MessageDigest.getInstance('SHA-256');
    m.update(b(:));
    d = typecast(m.digest(), 'uint8');
    h = lower(reshape(dec2hex(d(1:8), 2).', 1, []));
end

function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
