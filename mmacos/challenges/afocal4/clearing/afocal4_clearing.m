function R = afocal4_clearing(opts)
%AFOCAL4_CLEARING  BRIEF_afocal4_clear: get the collimator out of its own beam.
%
%   R = AFOCAL4_CLEARING() runs the whole clearing stage on the committed
%   343 mm family-2 design, WITHOUT overwriting anything under
%   challenges/afocal4: every deck it writes is new and lives here.
%
%     0  THE DEFECT, in engine truth.  The union body-in-beam gate on the
%        committed deck, the field-walk law that explains it, and the
%        per-field number that made it invisible.
%     1  LEVERAGE 1 -- one flat extraction fold in the feed leg.  RETIRED,
%        with the two-sided squeeze measured (CLEAR_FOLD).
%     2  LEVERAGE 2 -- the collimator station, and the interface standoff
%        beside it.  RETIRED, with the figure the brief asks for first
%        (CLEAR_SCAN).  Also measures every committed trade point.
%     3  LEVERAGE 3 -- the extraction TILT.  The raw exchange rate
%        (CLEAR_PRICE), then the delivered design: the same tilt with the
%        conics, the field-mirror standoff and the front end re-solved
%        around it (CLEAR_SOLVE).
%     4  NULLS.  Zero tilt must rebuild the committed deck byte-for-byte;
%        CLEAR_BUILD must equal AFOCAL4_BUILD; and the tilt must leave the
%        chief ray's path up to the swung mirror untouched.
%     5  PACKAGING.  Path A re-run on the cleared design -- does the
%        four-fold depth answer survive the swing?
%     6  THE ROW.  The cleared design's trade quantities beside the
%        committed ones, and the gate's NON-VACUITY: it must FAIL on the
%        committed deck and PASS on this one.
%     7  LEVERAGE 4 -- what a fifth mirror would have to deliver, priced
%        from the law rather than built.
%
%   Name-value:
%     'sections'  which of 0:7 (default all)
%     'tilt'      the delivered extraction tilt, deg (default -10)
%     'price'     tilt angles for the raw sweep
%     'resolve'   'run' re-solves at 'price_resolve'; 'load' reads finished
%                 CLEAR_SOLVE checkpoints from 'resolved_mat' (a re-solve is
%                 a 45-minute artifact and is checkpointed, not re-run every
%                 time the report is rebuilt); 'skip'
%     'price_resolve'  angles to re-solve (default: the delivered one and
%                 its two neighbours, so the price curve has a shape)
%     'resolved_mat'   cell of checkpoint paths for 'load'
%     'max_fev'   evaluation budget per re-solve
%     'deck'      the parent (default ../afocal4_b2long_343mm.in)
%     'x_step_try' / 'x_out_try' / 'z_front_try' / 'm3_gap_try'
%                 the Path-A route quantities to search on the cleared deck.
%                 SEARCHED, not carried over: a route that fits the parent's
%                 2.93 m feed leg need not fit a shorter one, and the swing
%                 shortens it.
%     'save'      write decks, figures and the .mat (true)
%
%   Model size 256, one MATLAB process per model size,
%   MACOS_HOME=~/dev/macos/macos_f90.  Batch:  run('run_clear.m').
%
%   See also CLEAR_LAW, CLEAR_SCAN, CLEAR_FOLD, CLEAR_TILT, CLEAR_PRICE,
%   CLEAR_SOLVE, AFOCAL4_UNION.

    arguments
        opts.sections (1,:) double = 0:7
        opts.tilt     (1,1) double = -10
        opts.price    (1,:) double = [-14 -12 -10 -9 -8 -6 -4 -2 0 2 4 6 8]
        opts.resolve  (1,:) char {mustBeMember(opts.resolve,{'run','load','skip'})} = 'run'
        opts.price_resolve (1,:) double = []
        opts.resolved_mat  (1,:) cell = {}
        opts.max_fev  (1,1) double = 420
        opts.deck     (1,:) char = 'afocal4_b2long_343mm.in'
        opts.model    (1,1) double = 256
        opts.x_step_try  (1,:) double = [0.30 0.34 0.375 0.42]
        opts.x_out_try   (1,:) double = [0.13 0.19 0.24]
        opts.z_front_try (1,:) double = [0.02 0.08 0.15 0.23]
        opts.m3_gap_try  (1,:) double = [0.05 0.10]
        opts.save     (1,1) logical = true
    end

    here = fileparts(mfilename('fullpath'));
    up   = fileparts(here);
    addpath(here);  addpath(up);  addpath(fullfile(up,'packaging'));
    src  = fullfile(up, opts.deck);
    if ~isfile(src), error('afocal4_clearing:deck','no such deck: %s', src); end
    if isempty(opts.price_resolve)
        opts.price_resolve = opts.tilt + [2 1 0];
    end

    P = afocal4_params();
    % Every clearance in this stage is measured over the WHOLE field box.
    % The whole defect lives in the difference between one field and all of
    % them: per field the collimator has 10.8 mm of daylight, over the box
    % it is 79.9 mm inside the beam.
    Fbox = P.Fsolve;
    macos.init(opts.model);
    matf = fullfile(here, 'afocal4_clearing.mat');
    R = struct('parent',src, 'P',P, 'opts',opts, 'when',[]);

    % ---- 0  THE DEFECT ---------------------------------------------------
    if any(opts.sections == 0)
        hdr_('0  the defect, in engine truth');
        R.defect.gate_body = afocal4_union(src, 'fields',Fbox);
        fprintf('\n    the same measurement against BARE LIT GLASS (1.00x, 0 mm):\n');
        R.defect.gate_bare = afocal4_union(src, 'fields',Fbox, 'body_k',1.0, ...
                                'body_pad',0.0, 'init',false, 'quiet',true);
        fprintf('      floor %+.2f mm on %s -- %s\n', ...
                R.defect.gate_bare.floor_m*1e3, R.defect.gate_bare.worst_name, ...
                tern_(R.defect.gate_bare.ok,'clear', ...
                      'the interference is the design''s, not the body model''s'));
        nE = R.defect.gate_body.nElt;
        R.defect.law = clear_law(src, 'fields',Fbox, 'leg',2, 'elt',nE-1, ...
                                 'M',P.M, 'init',false);
        R.defect.legs = pack_legs(src, 'instr',P.pack.instr_len, 'quiet',true);
    end

    % ---- 1  LEVERAGE 1: a flat fold ---------------------------------------
    if any(opts.sections == 1)
        hdr_('1  leverage 1 -- one flat extraction fold in the feed leg');
        R.fold = clear_fold(P, src, 'fields',Fbox);
        b = max([R.fold.pt.floor_body]);
        fprintf(['\n    VERDICT: the best station in either turn direction ' ...
                 'leaves %+.2f mm.  A flat cannot do it, and the reason is\n' ...
                 '    exact: the collimator sits ON the flat at %.3f of the ' ...
                 'leg, and the two failure modes are complementary about\n' ...
                 '    that station -- before it the fold has not separated ' ...
                 'the partners, after it the return beam comes back\n' ...
                 '    through the flat.\n'], b*1e3, R.fold.crit_frac);
    end

    % ---- 2  LEVERAGE 2: the station scan (the brief's first deliverable) --
    if any(opts.sections == 2)
        hdr_('2  leverage 2 -- the collimator station, and the standoff beside it');
        D0 = recover_(P, src);
        R.D0 = D0;
        dks = arrayfun(@(q) fullfile(up, sprintf('afocal4_b2long_%03.0fmm.in', q*1e3)), ...
                       P.iface_trade, 'UniformOutput',false);
        dks = dks(cellfun(@isfile, dks));
        png = '';
        if opts.save, png = fullfile(here,'afocal4_clear_scan.png'); end
        R.scan = clear_scan(P, D0, 'decks',dks, 'save',png);
        fprintf(['\n    VERDICT: over %.2f m of collimator travel the ratio ' ...
                 'reaches %.3f against the %.3f the field box demands,\n' ...
                 '    and the demanded footprint never comes down to the ' ...
                 'available one.  The station cannot clear it.\n'], ...
                range_([R.scan.standoff.z_body]), ...
                max([R.scan.standoff.ratio]), R.scan.need);
    end

    % ---- 3  LEVERAGE 3: the extraction tilt --------------------------------
    if any(opts.sections == 3)
        hdr_('3  leverage 3 -- the extraction tilt: what it buys, what it costs');
        if ~isfield(R,'D0'), R.D0 = recover_(P, src); end
        png = '';   dd = '';
        if opts.save
            png = fullfile(here,'afocal4_clear_price.png');   dd = here;
        end
        res = [];   rmat = {};
        switch opts.resolve
        case 'run',  res  = opts.price_resolve;
        case 'load', rmat = opts.resolved_mat(cellfun(@isfile, opts.resolved_mat));
        end
        R.price = clear_price(P, R.D0, 'tilt',opts.price, 'resolve',res, ...
                    'resolved_mat',rmat, 'max_fev',opts.max_fev, ...
                    'deck_dir',dd, 'save',png);
        % the delivered design is the re-solved point at the delivered tilt
        k = find(abs([R.price.resolved.tilt_deg] - opts.tilt) < 1e-9, 1);
        if isempty(k)
            k = find(abs([R.price.raw.tilt_deg] - opts.tilt) < 1e-9, 1);
            R.cleared = R.price.raw(k);
            fprintf(['    NOTE: no re-solved point at %+.1f deg -- the RAW ' ...
                     'tilt is being carried as the cleared design.\n'], opts.tilt);
        else
            R.cleared = R.price.resolved(k);
        end
        if opts.save && ~isempty(R.cleared.deck)
            dst = fullfile(here, 'afocal4_clear_343mm.in');
            copyfile(R.cleared.deck, dst);
            R.cleared.deck = dst;
            fprintf('    cleared design written to %s\n', dst);
        end
    end

    % ---- 4  NULLS ----------------------------------------------------------
    if any(opts.sections == 4)
        hdr_('4  nulls -- the machinery must not move anything it does not mean to');
        if ~isfield(R,'D0'), R.D0 = recover_(P, src); end
        a = [tempname '.in'];  b = [tempname '.in'];
        Dz = R.D0;   Dz.tilt_deg = 0;
        afocal4_build(P, Dz, a, 'verify',false);
        clear_build(P, Dz, b, 'verify',false);
        R.null.rebuild_is_committed = isequal(fileread(a), fileread(src));
        R.null.clear_build_is_build = isequal(fileread(a), fileread(b));
        fprintf(['    the design struct rebuilds the committed deck ' ...
                 'byte-for-byte : %d\n'], R.null.rebuild_is_committed);
        fprintf(['    clear_build(tilt 0) == afocal4_build, byte-for-byte  ' ...
                 '        : %d\n'], R.null.clear_build_is_build);
        % and the tilt leaves the chief ray alone up to the swung mirror
        R.null.chief = chief_null_(src, opts.tilt);
        fprintf(['    tilt %+.1f deg: chief-ray path to the swung mirror ' ...
                 'moves %.3e m\n'], opts.tilt, R.null.chief.d_upstream);
        fprintf(['    ... the pivot is still ON the mirror: the chief lands ' ...
                 '%.3e m from it\n'], R.null.chief.d_pivot);
        fprintf(['    ... and the beam turns %.4f deg for a %.4f deg tilt ' ...
                 '(the axis is normal to the plane of incidence)\n'], ...
                R.null.chief.turn_deg, 2*abs(opts.tilt));
        delete(a); delete(b);
    end

    % ---- 5  PACKAGING: Path A on the cleared design -------------------------
    if any(opts.sections == 5) && isfield(R,'cleared')
        hdr_('5  packaging -- what the swing did, and Path A on the cleared deck');
        R.pack = path_a_(P, src, R.cleared.deck, Fbox, here, opts);
        % The gate's other three clauses, on the cleared deck: is there
        % still a fold station for the instrument, and does the instrument
        % volume clear the incoming beam?  Those are AFOCAL4_PACK's parts
        % 1-3 and they are not what the swing was aimed at.
        R.pack.gate = afocal4_pack(P, R.cleared.deck, 'fields',Fbox, 'quiet',true);
        fprintf(['\n    the rest of the packaging gate on the cleared deck: ' ...
                 '%s behind M1 %+.0f mm %s, fold daylight %.1f mm %s,\n' ...
                 '      instrument %.0f mm off axis, largest that fits ' ...
                 '%.0f mm dia %s;  union %s\n'], ...
                R.pack.gate.names{numel(R.pack.gate.z)-1}, ...
                R.pack.gate.behind_m1*1e3, tick2_(R.pack.gate.ok_station), ...
                R.pack.gate.fold_pick.gap*1e3, tick2_(R.pack.gate.ok_fold), ...
                R.pack.gate.instr.r_min*1e3, R.pack.gate.instr.dia_max*1e3, ...
                tick2_(R.pack.gate.instr.clears_beams), tick2_(R.pack.gate.ok_union));
        % The layouts, on ONE scale.  A clearance is a hardware claim and
        % the reader has to be able to see the bodies, not just read a
        % floor -- and the decks only compare if they are drawn to the same
        % rule (the packaging stage's own figure convention).
        if opts.save
            dks = {src};   lbs = {'committed 343 mm'};
            if isfile(R.pack.parent_pack_deck)
                dks{end+1} = R.pack.parent_pack_deck;
                lbs{end+1} = 'committed + Path A (4 flats)';
            end
            dks{end+1} = R.cleared.deck;
            lbs{end+1} = sprintf('cleared (tilt %+.0f deg, re-solved)', ...
                                 R.cleared.tilt_deg);
            if R.pack.ok
                dks{end+1} = R.pack.deck;   lbs{end+1} = 'cleared + Path A';
            end
            R.pack.fig = pack_view(dks, lbs, ...
                'r_env',P.pack.m1_keepout, 'instr_len',P.pack.instr_len, ...
                'instr_dia',P.pack.instr_dia, ...
                'save',fullfile(here,'afocal4_clear_layouts.png'), ...
                'title','getting the collimator out of its own feed beam');
        end
    end

    % ---- 6  THE ROW, and the gate's non-vacuity -----------------------------
    if any(opts.sections == 6) && isfield(R,'cleared')
        hdr_('6  the trade row, beside the committed one');
        R.row = row_(P, src, R.cleared, Fbox);
        hdr_('6b  the gate''s non-vacuity');
        R.nonvac = nonvacuity_(P, src, R.cleared.deck, Fbox);
    end

    % ---- 7  LEVERAGE 4: what a fifth mirror would have to deliver -----------
    if any(opts.sections == 7)
        hdr_('7  leverage 4 -- a fifth mirror, priced from the law');
        R.fifth = fifth_(P, R);
    end

    R.when = datestr(now, 'yyyy-mm-dd HH:MM:SS'); %#ok<TNOW1,DATST>
    if opts.save
        % Strip the figure HANDLES before saving.  A saved handle drags the
        % whole graphics object into the .mat (MATLAB warns about it), and
        % the figures are already on disk as PNGs -- the .mat is for the
        % numbers.
        R = strip_figs_(R);
        save(matf, 'R', '-v7.3');
        fprintf('\n  wrote %s\n', matf);
    end
end

function S = strip_figs_(S)
%STRIP_FIGS_  Drop graphics handles, one level down, wherever they sit.
    f = fieldnames(S);
    for i = 1:numel(f)
        v = S.(f{i});
        if isa(v,'matlab.ui.Figure')
            S.(f{i}) = [];
        elseif isstruct(v) && isscalar(v)
            S.(f{i}) = strip_figs_(v);
        end
    end
end

% =====================================================================
function D = recover_(P, deck)
%RECOVER_  The design struct behind a committed afocal4 deck.
%   An afocal4 design is fully recoverable from its own prescription
%   (RESULTS rule 9): the conics and R_M2 and t_M1M2 are read off, and the
%   field-mirror standoff is the intermediate-image distance minus the
%   emitted M2 -> FM spacing.  Recovering it rather than carrying a stored
%   struct means this stage cannot silently drift from the deck it claims to
%   be starting from -- and the NULL section proves the recovery by
%   rebuilding the committed file byte-for-byte.
    txt = fileread(deck);
    Kc  = grab1_(txt,'KcElt');   Kr = grab1_(txt,'KrElt');
    zE  = grab1_(txt,'zElt');
    nM  = numel(Kc) - 1;                       % the last element is the interface
    D = struct('form','field', 'K',Kc(1:nM).', 'bias_deg',P.bias_deg, ...
               'ngrid',P.ngrid, 'rb',zeros(numel(P.rb_elts),2), 'tilt_deg',0);
    D.R2 = abs(Kr(2));
    % READ THE SPACINGS FROM zElt, NOT FROM THE VERTICES.  The builder poses
    % the interface plane on the TRACED CHIEF, so the last mirror's vertex
    % is 359 mm from the interface vertex on a deck whose interface standoff
    % is 343 -- the difference is the chief's own offset from the axis, and
    % taking the vertex distance recovers a design that is 16 mm wrong and
    % rebuilds to a deck that is not this one.  zElt carries the emitted
    % spacing verbatim (the pose edit does not touch it), so it is exact.
    D.t1    = zE(1);
    D.iface = zE(nM);
    % the intermediate image, from the paraxial marginal ray of the emitted
    % front end -- the same kernel the closure uses, so the round trip is
    % exact rather than approximate.
    fo = afocal_first_order([abs(Kr(1)) abs(Kr(2))], D.t1, ...
                            [false true], 'D',P.D, 'stop_ahead',P.stop_ahead);
    a0 = -fo.y_marginal(2)/fo.u_marginal(2);
    D.fm_standoff = a0 - zE(2);
end

function N = chief_null_(deck, a_deg)
%CHIEF_NULL_  The tilt turns the mirror about the point the chief actually
%   strikes, so on a RE-TRACE of the written deck: nothing upstream of that
%   mirror may move, the chief must still land ON the pivot (which is the
%   statement that the pivot is still on the surface), and the beam must
%   turn by exactly 2*alpha -- the last only because the tilt axis is normal
%   to this design's plane of incidence.
%
%   NOT reported as a check: the unsigned incidence angle moving by alpha.
%   The field mirror is worked at a few degrees, so any larger tilt carries
%   the SIGNED angle through zero and the unsigned one folds back.
    t = [tempname '.in'];
    cu = onCleanup(@() del_(t)); %#ok<NASGU>
    o = clear_tilt(deck, struct('elt','FM','alpha',deg2rad(a_deg), ...
                                'axis',[1 0 0]), t);
    P0 = chief_(deck);   P1 = chief_(t);
    k  = o.elt;
    N.d_upstream = max(vecnorm(P0(:,1:k+1) - P1(:,1:k+1)));
    N.d_pivot    = norm(P1(:,k+1) - o.Q(:));
    N.turn_deg   = o.turn_deg;
    N.aoi_before = 90 - rad2deg(acos(max(-1,min(1, dot(o.din, o.dout)))))/2;
    N.aoi_after  = 90 - rad2deg(acos(max(-1,min(1, dot(o.din, o.dout_new)))))/2;
end

function P = chief_(deck)
    macos.load_rx(deck);
    macos.ray_hist('on');   t = macos.trace();   h = macos.ray_hist(t.nRays);
    macos.ray_hist('off');
    P = squeeze(h.P(:,1,:));
end

function A = path_a_(P, src, deck, Fbox, here, opts)
%PATH_A_  Does the packaging stage's four-fold depth answer survive the swing?
%
%   Two questions, in this order, because the second only matters if the
%   first is still open:
%     1  WHAT THE SWING ALONE DID TO THE PACKAGE.  The tilt is not a
%        packaging move and was not asked to be one, but it re-poses the
%        whole back end, so the envelope is re-measured before anything is
%        folded.  Committed deck, cleared deck, and the packaging stage's
%        own four-fold deck side by side.
%     2  PATH A ON THE CLEARED DECK.  The route is re-derived from the
%        cleared deck's OWN spacings -- nothing is carried over from the
%        parent's -- and searched over the four stated quantities, because
%        a route that fits a 2.93 m feed leg need not fit a shorter one.
%        Every candidate is gated: union floor, fold-induced floor, rays
%        lost, and the primary's keep-out radius.
%
%   A route that LOSES RAYS is not a candidate at any margin.  The
%   packaging study emits its flats with ApType None precisely so that an
%   honestly-sized flat does not turn a packaging study into a ray-loss
%   study -- so rays lost here mean the geometry itself failed, not a stop.
    macos.load_rx(deck);
    A.before = pack_legs(deck,  'instr',P.pack.instr_len, 'quiet',true);
    A.parent = pack_legs(src,   'instr',P.pack.instr_len, 'quiet',true);
    A.parent_pack = [];
    pp = fullfile(fileparts(here), 'packaging', 'afocal4_b2pack_343mm.in');
    if isfile(pp)
        A.parent_pack = pack_legs(pp, 'instr',P.pack.instr_len, 'quiet',true);
    end
    A.parent_pack_deck = pp;

    % TWO RATIOS, NAMED APART, because the packaging stage quotes one and
    % this stage was about to quote the other: "1.81x" there is the
    % DEEPEST OPTIC over the M1-M2 spacing, while overhang/spacing is
    % 0.81x on the same deck.  Printing one under the other's name makes
    % an improvement look 3x bigger than it is (frame before angle).
    sf = A.parent.span_front_m;
    fprintf('    %-38s %12s %12s %12s\n', 'quantity', 'committed', ...
            'committed+4f', 'cleared');
    row3_('M1-M2 spacing, the yardstick (m)', sf, sf, A.before.span_front_m);
    row3_('deepest optic behind M1 (m)', A.parent.span_back_m, ...
          gz2_(A.parent_pack,'span_back_m'), A.before.span_back_m);
    row3_('... as a multiple of the yardstick', A.parent.span_back_m/sf, ...
          gz2_(A.parent_pack,'span_back_m')/sf, A.before.span_back_m/sf);
    row3_('overhang (deepest - yardstick) (m)', A.parent.overhang_m, ...
          gz2_(A.parent_pack,'overhang_m'), A.before.overhang_m);
    row3_('... as a multiple of the yardstick', A.parent.overhang_m/sf, ...
          gz2_(A.parent_pack,'overhang_m')/sf, A.before.overhang_m/sf);
    % The radial envelope of ANY body is the primary's own 0.500 m on
    % every deck, which says nothing; what the packaging stage means by
    % "optics radius" is the girth of the structure BEHIND the primary.
    row3_('radius of the optics behind M1 (m)', rback_(A.parent), ...
          rback_(A.parent_pack), rback_(A.before));
    row3_('radial extent of any body incl. M1 (m)', A.parent.r_env_m, ...
          gz2_(A.parent_pack,'r_env_m'), A.before.r_env_m);
    row3_('back focal path (m)', A.parent.path_back_m, ...
          gz2_(A.parent_pack,'path_back_m'), A.before.path_back_m);
    fprintf('    %-38s %12d %12d %12d\n','extra flats', 0, 4, 0);

    % ---- 2. the route, searched on the cleared deck's own geometry -------
    A.try = struct('x_step',{},'x_out',{},'z_front',{},'m3_gap',{}, ...
                   'deepest',{},'overhang',{},'r_env',{},'union_m',{}, ...
                   'fold_m',{},'nLost',{},'why',{});
    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>
    best = [];
    for xs = opts.x_step_try
      for xo = opts.x_out_try
        for zf = opts.z_front_try
          for mg = opts.m3_gap_try
            if xo >= xs, continue; end
            t = struct('x_step',xs,'x_out',xo,'z_front',zf,'m3_gap',mg, ...
                       'deepest',NaN,'overhang',NaN,'r_env',NaN, ...
                       'union_m',NaN,'fold_m',NaN,'nLost',NaN,'why','');
            try
                [f, ~] = pack_route(deck, 'init',true, 'x_step',xs, ...
                          'x_out',xo, 'z_front',zf, 'm3_gap',mg, ...
                          'fields',Fbox, 'quiet',true);
                o  = pack_fold(deck, f, tmp, 'quiet',true);
                Lg = pack_legs(tmp, 'instr',P.pack.instr_len, 'quiet',true);
                U  = afocal4_union(tmp, 'fields',Fbox, 'init',false, 'quiet',true);
                C  = pack_clear(tmp, 'init',false, 'fields',Fbox, ...
                                'body_pad',P.pack.fold_margin, ...
                                'fold_elts',find(o.isfold), 'quiet',true);
                t.deepest = Lg.span_back_m;   t.overhang = Lg.overhang_m;
                t.r_env   = Lg.r_env_m;       t.union_m  = U.floor_m;
                t.fold_m  = C.floor_fold_body_m;  t.nLost = Lg.nLost;
                if U.ok && Lg.nLost == 0 && Lg.r_env_m < P.pack.m1_keepout
                    if isempty(best) || U.floor_m > best.union_m
                        best = t;   copyfile(tmp, fullfile(here, ...
                            'afocal4_clear_343mm_pack.in'));
                    end
                end
            catch ME
                t.why = one_line_(ME.message);
            end
            A.try(end+1) = t; %#ok<AGROW>
          end
        end
      end
    end
    n = numel(A.try);   nc = nnz(~cellfun(@isempty, {A.try.why}) == 0);
    A.n_tried = n;   A.n_closed = nc;
    A.best = best;
    A.ok = ~isempty(best);
    if A.ok
        A.deck = fullfile(here,'afocal4_clear_343mm_pack.in');
        fprintf(['\n    Path A on the cleared deck: %d routes tried, %d closed, ' ...
                 'best union floor %+.2f mm\n'], n, nc, best.union_m*1e3);
        fprintf(['      x_step %.3f  x_out %.3f  z_front %.3f  m3_gap %.3f  ' ...
                 '-> deepest %.4f m, overhang %+.4f m, %d rays lost\n'], ...
                best.x_step, best.x_out, best.z_front, best.m3_gap, ...
                best.deepest, best.overhang, best.nLost);
    else
        cl = A.try(cellfun(@isempty, {A.try.why}));
        fprintf(['\n    Path A does NOT close on the cleared deck: %d routes ' ...
                 'tried, %d satisfied the route algebra and the\n' ...
                 '    plane-intersection bound, and every one of those lost ' ...
                 'rays or left a body in a beam.\n'], n, numel(cl));
        if ~isempty(cl)
            [~,ib] = max([cl.union_m]);
            fprintf(['      best of them: x_step %.3f x_out %.3f z_front %.3f ' ...
                     'm3_gap %.3f -> union %+.2f mm, %d rays lost\n'], ...
                    cl(ib).x_step, cl(ib).x_out, cl(ib).z_front, ...
                    cl(ib).m3_gap, cl(ib).union_m*1e3, cl(ib).nLost);
        end
        fprintf(['      And it does not need to.  The four-fold trombone was ' ...
                 'invented to remove %+.3f m of overhang; the swing\n' ...
                 '      leaves %+.3f m, so the route has nothing left to ' ...
                 'fold -- its own algebra runs out of leg.\n'], ...
                A.parent.overhang_m, A.before.overhang_m);
    end
end

function row3_(nm, a, b, c)
    fprintf('    %-38s %12.4f %12.4f %12.4f\n', nm, a, b, c);
end

function r = rback_(L)
%RBACK_  The girth of the structure BEHIND the primary -- what the
%   packaging stage calls the optics radius.  Every deck's r_env_m is the
%   primary's own 0.500 m, so the whole-train number cannot distinguish
%   them; this one can.
    r = NaN;
    if isempty(L) || ~isfield(L,'body_r'), return; end
    m = L.z > 0;
    if any(m), r = max(L.body_r(m)); end
end
function v = gz2_(S, f)
    v = NaN;   if ~isempty(S) && isfield(S,f), v = S.(f); end
end

function T = row_(P, src, cleared, Fbox)
%ROW_  The cleared design's trade quantities beside the committed row.  The
%   committed row is RE-MEASURED here rather than quoted, so the two columns
%   come from one run of one scorer and a difference cannot be a transcription.
    S0 = afocal4_score(P, src, 'fields',P.Fsolve, 'nodes',P.solve.nodes_score, ...
                       'grid',P.grid_n);
    % Both columns scored the same way, including the UNIFORM grid.  Solve
    % set is not scoring set (his 3x3 is a third corners), and quoting one
    % column with the grid and the other without would compare two
    % different questions.  CLEAR_PRICE omits the grid because it scores 16
    % designs; here there are two.
    S1 = cleared.score;
    if ~isfield(S1,'wfe_grid_max_nm') || isempty(S1.wfe_grid_max_nm)
        S1 = afocal4_score(P, cleared.deck, 'fields',P.Fsolve, ...
                           'nodes',P.solve.nodes_score, 'grid',P.grid_n);
    end
    K0 = afocal4_union(src, 'fields',Fbox, 'quiet',true);
    K1 = afocal4_union(cleared.deck, 'fields',Fbox, 'quiet',true);
    % The chief-ray incidence on every powered mirror.  A swung field mirror
    % is worked further off normal than an unswung one, and the design
    % drivers' standing rule is AOI < 15 deg -- so this column is part of
    % the price, not a footnote.
    A0 = aoi_chief_(src);   A1 = aoi_chief_(cleared.deck);
    % The CUSTOMER INTERFACE, re-measured on both decks.  The tilt preserves
    % the chief exactly but is not an isometry of the light, so the exit
    % beam's collimation and diameter are re-traced rather than inherited
    % from a closure that was exact before the swing.
    I0 = iface_(src, P.D);   I1 = iface_(cleared.deck, P.D);
    nm = {'WFE rung2 max (nm)','WFE uniform grid max (nm)','pupil blur rms (um)', ...
          'breathing chief-normal (%)','wander at refit plane (um)', ...
          'surface vs imaged sag (mm)','M at box centre','anchoring residual (um)', ...
          'traced M (exit beam)','exit beam diameter (mm)','collimation (urad)', ...
          'chief AOI on the field mirror (deg)','max chief AOI, any mirror (deg)', ...
          'union body-in-beam floor (mm)','rays lost over the box'};
    v0 = [S0.wfe_max_nm, gz_(S0,'wfe_grid_max_nm'), S0.blur_um, S0.breathe_pct, ...
          S0.wander_um, S0.surf_pv_mm, S0.mag_centre_chief, S0.anchor_resid_um, ...
          I0.mag, I0.exit_dia*1e3, I0.collimation_urad, ...
          A0(3), max(A0), K0.floor_m*1e3, K0.nLost];
    v1 = [S1.wfe_max_nm, gz_(S1,'wfe_grid_max_nm'), S1.blur_um, S1.breathe_pct, ...
          S1.wander_um, S1.surf_pv_mm, S1.mag_centre_chief, S1.anchor_resid_um, ...
          I1.mag, I1.exit_dia*1e3, I1.collimation_urad, ...
          A1(3), max(A1), K1.floor_m*1e3, K1.nLost];
    fprintf('    %-32s %14s %14s %10s\n', '', 'committed 343', ...
            sprintf('cleared %+.0f deg', cleared.tilt_deg), 'ratio');
    for i = 1:numel(nm)
        r = v1(i)/v0(i);
        fprintf('    %-32s %14.4f %14.4f %10.3f\n', nm{i}, v0(i), v1(i), r);
    end
    T = struct('names',{nm}, 'committed',v0, 'cleared',v1, 'S0',S0, 'S1',S1, ...
               'K0',K0, 'K1',K1, 'aoi_committed',A0, 'aoi_cleared',A1, ...
               'iface_committed',I0, 'iface_cleared',I1);
end

function s = iface_(deck, Dap)
%IFACE_  The customer boundary, traced: exit beam diameter, the angular
%   magnification it implies, and how collimated the beam actually is.
    macos.load_rx(deck);
    tr = macos.trace(macos.num_elt());
    ri = macos.get_ray_info(tr.nRays);
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

function A = aoi_chief_(deck)
%AOI_CHIEF_  Chief-ray incidence on every element, from the traced chief
%   alone: a mirror turns the beam by 180 - 2*AOI, so AOI = 90 - acos(d_in .
%   d_out)/2 with no surface normal needed (the AOI_REPORT identity).  The
%   interface plane's entry is meaningless and is returned as NaN rather
%   than as a small number that could be mistaken for one.
    macos.load_rx(deck);
    nE = macos.num_elt();
    macos.ray_hist('on');   t = macos.trace();   h = macos.ray_hist(t.nRays);
    macos.ray_hist('off');
    Pc  = squeeze(h.P(:,1,:));   off = size(h.P,3) - nE;
    A = nan(1,nE);
    for k = 1:nE-1
        di = Pc(:,k+off)   - Pc(:,k+off-1);
        do = Pc(:,k+off+1) - Pc(:,k+off);
        if norm(di) < eps || norm(do) < eps, continue; end
        A(k) = 90 - rad2deg(acos(max(-1,min(1, ...
                    dot(di/norm(di), do/norm(do))))))/2;
    end
end

function N = nonvacuity_(P, src, deck, Fbox)
%NONVACUITY_  A gate nobody can fail is not a gate.  The union check must
%   FAIL on the committed 343 mm deck -- the very design that shipped -- and
%   PASS on the cleared one, at the SAME declared allowance.  Both halves
%   are asserted here rather than left to a reader's inspection.
    N.fail = afocal4_union(src,  'fields',Fbox, 'quiet',true);
    N.pass = afocal4_union(deck, 'fields',Fbox, 'quiet',true);
    N.ok = (~N.fail.ok) && N.pass.ok;
    fprintf('    committed 343 mm deck: %s  (floor %+.2f mm on %s)\n', ...
            tern_(N.fail.ok,'PASSES  <-- the gate is vacuous','FAILS, as it must'), ...
            N.fail.floor_m*1e3, N.fail.worst_name);
    fprintf('    cleared design       : %s  (floor %+.2f mm on %s)\n', ...
            tern_(N.pass.ok,'PASSES, as it must','FAILS  <-- not cleared'), ...
            N.pass.floor_m*1e3, N.pass.worst_name);
    fprintf('    => the gate is %s\n', tern_(N.ok,'NON-VACUOUS','NOT sound'));
end

function F = fifth_(P, R)
%FIFTH_  What a fifth mirror would have to deliver, in the law's own terms.
%   Not built: priced.  The law says a body clears a beam when
%       c_beam*(B-A) - r_beam  >  c_body*(B+A) + r_body,
%   and that the collimator's own walk is PINNED at c_body = M * iface.  So a
%   fifth mirror is worth adding only if it changes one of those three
%   things, and there are exactly three ways it can:
%     (a) take the collimator off the feed axis with POWER instead of a
%         swung field mirror -- the same field-independent offset a tilt
%         buys, but bought by an element designed for it rather than by
%         spending the pupil control the fourth mirror was added for;
%     (b) relay the beam to a SECOND intermediate image, so the collimator
%         no longer lives inside the M2 -> field-mirror cone at all -- this
%         is the 'relay' form of AFOCAL4_CLOSE, eliminated in S3 on pupil
%         grounds with four mirrors and worth re-opening with five;
%     (c) hold the pupil station itself, freeing the field mirror's power
%         (which the four-mirror closure spends entirely on that condition)
%         to put the collimator at the internal chief crossing, where its
%         union footprint collapses to the beam radius.
%   The number each has to beat is printed below.
    F = struct();
    if ~isfield(R,'defect') || ~isfield(R.defect,'law')
        fprintf('    (section 0 not run -- nothing to price against)\n');
        return;
    end
    L = R.defect.law;
    F.need_offset_m = -(L.gap_prop_m) + L.r_beam + L.r_body;
    F.have_ratio = L.ratio;   F.need_ratio = L.need;
    F.c_body_pinned = L.M_iface;
    fprintf(['    the collimator''s walk is pinned at M * iface = %.4f m/rad ' ...
             '(measured %.4f, %+.2f%%)\n'], L.M_iface, L.c_body_abs, ...
            100*L.M_iface_err);
    fprintf(['    to clear WITHOUT a field-independent offset the feed''s ' ...
             'walk would have to reach %.4f m/rad;\n      it is %.4f, and ' ...
             'its ceiling is the intermediate image height itself\n'], ...
            L.need*L.c_body_abs, L.c_beam_abs);
    fprintf(['    a fifth mirror therefore has to supply at least %.1f mm ' ...
             'of FIELD-INDEPENDENT separation -- which is\n      exactly ' ...
             'what the %+.1f deg tilt supplies (%.1f mm measured), so its ' ...
             'case rests on doing it\n      WITHOUT spending the pupil ' ...
             'control, not on doing it at all.\n'], F.need_offset_m*1e3, ...
            R.opts.tilt, cl_(R, 'offset_mm'));
    if isfield(R,'cleared')
        fprintf(['    the bar it must beat: blur %.1f um, breathing %.4f %%, ' ...
                 'wander %.1f um at %.0f nm of wavefront\n'], ...
                R.cleared.blur_um, R.cleared.breathe_pct, R.cleared.wander_um, ...
                R.cleared.wfe_nm);
    end
end

function v = cl_(R, f)
    v = NaN;
    if isfield(R,'cleared') && isfield(R.cleared,f), v = R.cleared.(f); end
end
function v = gz_(S, f)
    v = NaN;   if isfield(S,f), v = S.(f); end
end
function r = range_(v), r = max(v) - min(v); end
function hdr_(t)
    fprintf('\n%s\n  %s\n%s\n', repmat('=',1,74), t, repmat('=',1,74));
end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
function s = tick2_(b), if b, s = 'OK'; else, s = '<-- FAILS'; end, end
function s = one_line_(m)
    s = regexprep(m, '\s+', ' ');   if numel(s) > 100, s = [s(1:100) '...']; end
end
function M = grab3_(txt, key)
    t = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens');
    M = zeros(3, numel(t));
    for i = 1:numel(t), M(:,i) = sscanf(strrep(t{i}{1},'D','E'), '%f', 3); end
end
function v = grab1_(txt, key)
    t = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens');
    v = zeros(1, numel(t));
    for i = 1:numel(t), v(i) = sscanf(strrep(t{i}{1},'D','E'), '%f', 1); end
end
function del_(p),  if exist(p,'file'), delete(p); end,  end
