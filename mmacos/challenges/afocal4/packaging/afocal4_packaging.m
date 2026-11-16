function R = afocal4_packaging(opts)
%AFOCAL4_PACKAGING  BRIEF_r2_packaging: measure the gap, then fold it down.
%
%   R = AFOCAL4_PACKAGING() runs the whole study on the committed 343 mm
%   family-2 winner, WITHOUT touching anything under challenges/afocal4:
%
%     0  MEASURE.  Engine-truth legs, stations, footprints, clearances and
%        envelope of the parent deck.  This is step 1 of the brief and it is
%        delivered whatever else happens (Path C).
%     1  THE COMMITTED RECIPE, on this deck.  One fold after the last
%        mirror, at the station AFOCAL4_PACK picks -- the S4b demonstration's
%        own recipe, rebuilt here so the comparison is three columns of the
%        SAME design instead of a comparison across designs.
%     2  PATH A.  Four folds inside the M2 -> field-mirror leg, wrapping the
%        back end into the stated envelope.
%     3  NULL.  Every deck re-scored on AFOCAL4_SCORE, and every clearance
%        re-measured: a fold is an isometry, so both must agree with the
%        parent to round-off.  Asserted, not assumed.
%     4  FIGURES + the verdict table.
%     5  SENSITIVITY of the route to where it comes back to -- x_out, which
%        is the girth the instrument inherits, swept at a fixed return step
%        (the plane-intersection bound fixes the step; only where the train
%        lands is free).
%
%   Name-value:
%     'sections'  which of 0:4 to run (default all)
%     'x_step' 'x_out' 'z_front' 'm3_gap'   the route's stated quantities
%     'r_env'    stated envelope radius, m (0.560 = P.pack.m1_keepout)
%     'deck'     parent prescription (default afocal4_b2long_343mm.in)
%     'save'     write decks/figures (true)
%
%   See also PACK_LEGS, PACK_CLEAR, PACK_FOLD, PACK_ROUTE, PACK_VIEW.

    arguments
        opts.sections (1,:) double = 0:5
        opts.sweep    (1,:) double = [0.150 0.190 0.225 0.260 0.300]
        opts.x_step   (1,1) double = 0.375
        opts.x_out    (1,1) double = 0.190
        opts.z_front  (1,1) double = 0.230
        opts.m3_gap   (1,1) double = 0.100
        opts.r_env    (1,1) double = 0.560
        opts.deck     (1,:) char = 'afocal4_b2long_343mm.in'
        opts.model    (1,1) double = 256
        opts.save     (1,1) logical = true
    end

    here = fileparts(mfilename('fullpath'));
    up   = fileparts(here);
    addpath(here);  addpath(up);
    src  = fullfile(up, opts.deck);
    if ~isfile(src), error('afocal4_packaging:deck','no such deck: %s', src); end

    P = afocal4_params();
    % Every clearance in this study is measured over the WHOLE field box, not
    % the deck's own field: the box corners walk the beam nearly twice as far
    % off the fold axis as the centre field does, and a mirror has to carry
    % all of them.
    Fbox = P.Fsolve;
    macos.init(opts.model);
    R = struct('deck',src, 'P',P, 'opts',opts);
    tag = erase(opts.deck, {'afocal4_','.in'});

    % ---- 0  MEASURE the parent ------------------------------------------
    if any(opts.sections == 0)
        hdr_('0  the gap, from engine truth');
        R.parent.legs  = pack_legs(src, 'instr', P.pack.instr_len);
        R.parent.clear = pack_clear(src, 'init',false, 'fields',Fbox, ...
                                    'body_pad',P.pack.fold_margin);
        R.parent.env   = envelope_(R.parent.legs, R.parent.clear, P, opts);
        % The same measurement with NO body inflation at all -- bare lit
        % glass, no 1.15, no edge allowance.  If an interference survives
        % that, it is the design's and not the body model's.
        fprintf('\n    the same clearances against BARE LIT GLASS (body_k 1, pad 0):\n');
        R.parent.bare = pack_clear(src, 'init',false, 'fields',Fbox, ...
                                   'body_k',1.0, 'body_pad',0.0, 'quiet',true);
        [~,ob] = sort([R.parent.bare.pair.d_m]);
        for i = ob(1:min(4,numel(ob)))
            q = R.parent.bare.pair(i);
            fprintf('      %-26s %9.2f mm  %s\n', q.name, q.d_m*1e3, ...
                    tern2_(q.pierced,'<-- PIERCED',''));
        end
    end

    % ---- 1  the committed recipe, one fold after the last mirror ---------
    if any(opts.sections == 1)
        hdr_('1  the committed S4b recipe on this deck: ONE fold after the last mirror');
        K = afocal4_pack(P, src, 'quiet', true);
        R.one.gate = K;
        nM = R.parent.legs.nElt - 1;
        d1 = abs(K.fold_pick.z - R.parent.legs.vpt(3,nM));
        f1 = struct('name','Fold', 'after',nM, 'dist',d1, 'to',P.pack.fold_to);
        deck1 = fullfile(here, sprintf('afocal4_%s_1fold.in', tag));
        R.one.fold  = pack_fold(src, f1, deck1, 'quiet',true);
        R.one.deck  = deck1;
        R.one.legs  = pack_legs(deck1, 'instr', P.pack.instr_len);
        R.one.clear = pack_clear(deck1, 'init',false, 'fields',Fbox, ...
                                 'body_pad',P.pack.fold_margin, ...
                                 'fold_elts', find(R.one.fold.isfold));
        R.one.env   = envelope_(R.one.legs, R.one.clear, P, opts);
        fprintf(['    fold %.4f m past %s, into [%s]; AFOCAL4_PACK measured ' ...
                 '%.1f mm of beam-to-beam daylight there (margin %.1f mm)\n'], ...
                d1, R.parent.legs.names{nM}, num2str(P.pack.fold_to), ...
                K.fold_pick.gap*1e3, P.pack.fold_margin*1e3);
        kf = find(R.one.fold.isfold, 1);
        fprintf(['      the flat that has to sit there: footprint radius %.1f mm, ' ...
                 'body radius %.1f mm; its clearance to the feed beam is %.2f mm\n'], ...
                R.one.clear.foot_r(kf)*1e3, R.one.clear.body_r(kf)*1e3, ...
                R.one.clear.floor_fold_m*1e3);
    end

    % ---- 2  PATH A: four folds in the M2 -> field-mirror leg -------------
    if any(opts.sections == 2)
        hdr_('2  Path A: four folds in the collimator feed leg');
        macos.load_rx(src);
        [folds, plan] = pack_route(src, 'init',false, 'x_step',opts.x_step, ...
                        'x_out',opts.x_out, 'z_front',opts.z_front, ...
                        'm3_gap',opts.m3_gap, 'fields',Fbox);
        deck4 = fullfile(here, sprintf('afocal4_%s.in', strrep(tag,'b2long','b2pack')));
        R.pack.route = plan;
        R.pack.fold  = pack_fold(src, folds, deck4, 'quiet',true);
        R.pack.deck  = deck4;
        R.pack.legs  = pack_legs(deck4, 'instr', P.pack.instr_len);
        R.pack.clear = pack_clear(deck4, 'init',false, 'fields',Fbox, ...
                                  'body_pad',P.pack.fold_margin, ...
                                  'fold_elts', find(R.pack.fold.isfold));
        R.pack.env   = envelope_(R.pack.legs, R.pack.clear, P, opts);
    end

    % ---- 3  NULL ---------------------------------------------------------
    if any(opts.sections == 3)
        hdr_('3  the folds are null -- asserted, not assumed');
        R.null = null_(P, R, here);
    end

    % ---- 4  figures + verdict --------------------------------------------
    if any(opts.sections == 4)
        hdr_('4  figures and the verdict table');
        R.fig = figs_(R, here, opts);
        verdict_(R, opts);
    end

    % ---- 5  the route's one free quantity ---------------------------------
    if any(opts.sections == 5)
        hdr_('5  sensitivity to the lateral fold step');
        R.sweep = sweep_(P, src, here, opts);
    end

    if opts.save
        save(fullfile(here, 'afocal4_packaging.mat'), 'R');
        fprintf('\n  saved %s\n', fullfile(here,'afocal4_packaging.mat'));
    end
end

% =====================================================================
function E = envelope_(L, K, P, opts, quiet)
%ENVELOPE_  Where the design actually sits, against the STATED envelope:
%   a cylinder of radius r_env about the telescope axis, and a slab behind
%   the primary no deeper than the M1-M2 spacing.  The instrument is scored
%   SEPARATELY, because it is an interface volume the telescope does not own
%   -- but its reach is reported in both directions, since that is exactly
%   what a fold trades.
    if nargin < 5, quiet = false; end
    nE = L.nElt;
    behind = 3:nE;                                % everything after the front end
    % body radii from the FIELD-UNION footprints (K), not the single-field
    % ones: a mirror has to carry the whole box.
    br = vecnorm(K.foot_c([1 2],:)) + K.foot_r;
    r_opt  = max(br(behind));
    E = struct('r_env',opts.r_env, 'z_slab',L.span_front_m, ...
               'z_optics',[min(L.z(behind)) max(L.z(behind))], ...
               'r_optics',r_opt, ...
               'optics_in_slab', max(L.z(behind)) <= L.span_front_m, ...
               'optics_in_r',    r_opt <= opts.r_env);

    % the instrument: from the interface plane along the exit chief
    P0 = L.foot_c(:,nE);
    a  = L.leg(end).d;
    P1 = P0 + a*P.pack.instr_len;
    hw = 0.5*P.pack.instr_dia;
    E.instr = struct('p0',P0, 'p1',P1, 'dir',a, ...
        'r_max', max(hypot(P0(1),P0(2)), hypot(P1(1),P1(2))) + hw, ...
        'z_max', max(P0(3),P1(3)) + hw, 'z_min', min(P0(3),P1(3)) - hw, ...
        'aft_frac', a(3));
    E.instr.in_r = E.instr.r_max <= opts.r_env;
    E.shroud_dia = 2*max([r_opt, E.instr.r_max, br(1), P.pack.m1_keepout]);
    E.total_len  = E.instr.z_max - min(L.z);

    if quiet, return; end
    fprintf(['    ENVELOPE  optics behind the front end: z %+.3f..%+.3f m ' ...
             '(slab %.3f)  r %.3f m\n'], E.z_optics, E.z_slab, r_opt);
    fprintf('      optics inside the M1-M2 slab: %s   inside r_env %.3f m: %s\n', ...
            yn_(E.optics_in_slab), opts.r_env, yn_(E.optics_in_r));
    fprintf(['      instrument %.2f m x %.0f mm runs [%+.3f %+.3f %+.3f] -> ' ...
             '[%+.3f %+.3f %+.3f]\n'], P.pack.instr_len, P.pack.instr_dia*1e3, ...
            P0, P1);
    fprintf(['        reach %.3f m radial (inside r_env: %s), %.3f m behind ' ...
             'M1;  axial fraction %.2f\n'], E.instr.r_max, yn_(E.instr.in_r), ...
            E.instr.z_max, E.instr.aft_frac);
    fprintf('      observatory envelope: %.3f m dia x %.3f m long\n', ...
            E.shroud_dia, E.total_len);
end

function N = null_(P, R, here) %#ok<INUSD>
%NULL_  Score every deck on the same kernel and compare.  A nominally null
%   fold is not a free fold (the e2e2 s3 lesson, and the S4b fold check
%   caught a real defect of its own this way), so the folded decks are
%   re-scored rather than assumed identical -- and the clearance floor is
%   compared too, which is the sharper test: a merit column can agree while
%   the geometry moved, but the clearance model reads the geometry directly.
    names = {'parent'};   decks = {R.deck};
    if isfield(R,'one'),  names{end+1} = 'one fold';   decks{end+1} = R.one.deck;  end
    if isfield(R,'pack'), names{end+1} = 'four folds'; decks{end+1} = R.pack.deck; end
    S = cell(1,numel(decks));
    for i = 1:numel(decks)
        S{i} = afocal4_score(P, decks{i}, 'nodes',P.solve.nodes_score, ...
                             'grid',P.grid_n);
    end
    fprintf('    %-12s %12s %10s %11s %10s %11s %12s %12s\n', 'deck','WFE nm', ...
            'blur um','breathe %','wander um','M','pre-fold mm','new mm');
    fl = [R.parent.clear.floor_pre_body_m];
    if isfield(R,'one'),  fl(end+1) = R.one.clear.floor_pre_body_m;  end
    if isfield(R,'pack'), fl(end+1) = R.pack.clear.floor_pre_body_m; end
    ff = NaN;
    if isfield(R,'one'),  ff(end+1) = R.one.clear.floor_fold_body_m;  end
    if isfield(R,'pack'), ff(end+1) = R.pack.clear.floor_fold_body_m; end
    for i = 1:numel(decks)
        fprintf('    %-12s %12.2f %10.2f %11.4f %10.2f %11.5f %12.4f %12.2f\n', ...
                names{i}, S{i}.wfe_max_nm, S{i}.blur_um, S{i}.breathe_pct, ...
                S{i}.wander_um, S{i}.mag_centre_chief, fl(i)*1e3, ff(i)*1e3);
    end
    d = @(f) max([0, arrayfun(@(i) abs(S{i}.(f) - S{1}.(f)), 2:numel(S))]);
    N = struct('names',{names}, 'S',{S}, 'floor_m',fl, ...
        'dWFE_nm', d('wfe_max_nm'), 'dblur_um', d('blur_um'), ...
        'dbreathe', d('breathe_pct'), 'dwander_um', d('wander_um'), ...
        'dmag', d('mag_centre_chief'), ...
        'dfloor_m', max([0, abs(fl(2:end) - fl(1))]));
    fprintf(['    NULL  max |dWFE| %.3e nm, |dblur| %.3e um, |dbreathe| %.3e %%, ' ...
             '|dwander| %.3e um, |dM| %.3e\n'], N.dWFE_nm, N.dblur_um, ...
            N.dbreathe, N.dwander_um, N.dmag);
    fprintf('          max |d PRE-EXISTING clearance floor| %.3e mm\n', N.dfloor_m*1e3);
    tol = 1e-9;
    N.ok = N.dWFE_nm < 1e-3 && N.dblur_um < 1e-6 && N.dfloor_m < tol;
    if N.ok
        fprintf('          => every added flat is null to round-off.\n');
    else
        fprintf(['          => NOT null at the stated tolerance -- do not quote ' ...
                 'the folded numbers until this is understood.\n']);
    end
end

function F = figs_(R, here, opts)
    decks = {R.deck};   labs = {'as committed (unfolded)'};
    if isfield(R,'one')
        decks{end+1} = R.one.deck;   labs{end+1} = 'S4b recipe: one fold after the last mirror';
    end
    if isfield(R,'pack')
        decks{end+1} = R.pack.deck;  labs{end+1} = 'Path A: four folds in the feed leg';
    end
    F.compare = fullfile(here, 'afocal4_pack_compare.png');
    pack_view(decks, labs, 'r_env',opts.r_env, 'save',F.compare, 'title', ...
        'afocal4 343 mm, family 2 -- packaging behind the primary (x-z elevation, engine truth)');
    if isfield(R,'pack')
        macos.load_rx(R.pack.deck);
        F.std = fullfile(here, 'afocal4_b2pack_343mm_view_std.png');
        try
            macos.view_std('save',F.std, 'visible',false, 'title', ...
                'afocal4 b2pack 343 mm -- four-fold package');
        catch ME
            fprintf('    view_std skipped: %s\n', ME.message);
            F.std = '';
        end
    end
end

function verdict_(R, opts)
    fprintf('\n  VERDICT\n');
    fprintf('    %-26s %10s %10s %10s\n', '', 'unfolded', '1 fold', '4 folds');
    rows = {'deepest optic behind M1 (m)', @(x) max(x.legs.z), ...
            'overhang vs M1-M2 (m)',       @(x) max(x.legs.z) - x.legs.span_front_m, ...
            'optics slab depth (m)',       @(x) diff(x.env.z_optics), ...
            'optics radius (m)',           @(x) x.env.r_optics, ...
            'instrument radial reach (m)', @(x) x.env.instr.r_max, ...
            'instrument z max (m)',        @(x) x.env.instr.z_max, ...
            'shroud diameter (m)',         @(x) x.env.shroud_dia, ...
            'body floor, pre-fold (mm)',   @(x) x.clear.floor_pre_body_m*1e3, ...
            'body floor, new flats (mm)',  @(x) x.clear.floor_fold_body_m*1e3, ...
            'back focal path (m)',         @(x) x.legs.path_back_m};
    have = {R.parent};
    if isfield(R,'one'),  have{end+1} = R.one;  else, have{end+1} = []; end
    if isfield(R,'pack'), have{end+1} = R.pack; else, have{end+1} = []; end
    for i = 1:2:numel(rows)
        fprintf('    %-26s', rows{i});
        for j = 1:3
            if isempty(have{j}), fprintf(' %10s','--');
            else, fprintf(' %10.3f', rows{i+1}(have{j}));
            end
        end
        fprintf('\n');
    end
    fprintf('    stated envelope: r <= %.3f m, optics slab 0 < z <= %.3f m\n', ...
            opts.r_env, R.parent.legs.span_front_m);
end

function W = sweep_(P, src, here, opts)
%SWEEP_  The route fixes the DEPTH from z_front and m3_gap alone -- the
%   deepest optic lands at z_front + L_next + m3_gap whatever the lateral
%   geometry is.  The steps are fixed by the plane-intersection bound.  What
%   is left free is X_OUT, where the train comes back to: it buys daylight
%   against the outbound axial leg and costs girth the instrument inherits.
%   Both measured, on real decks.
    tmp = fullfile(here, 'sweep_tmp.in');
    c = onCleanup(@() delete_if_(tmp));
    W = struct('x_out',{},'z_deep',{},'nLost',{},'floor_fold_mm',{}, ...
               'floor_pre_mm',{},'instr_r',{},'shroud',{},'ok_r',{});
    fprintf('    %9s %10s %14s %14s %12s %10s %8s\n', 'x_out m','deepest m', ...
            'new floor mm','pre floor mm','instr r m','shroud m','lost');
    for x = opts.sweep
        try
            macos.load_rx(src);
            f = pack_route(src, 'init',false, 'x_step',x + 0.185, 'x_out',x, ...
                           'z_front',opts.z_front, 'm3_gap',opts.m3_gap, ...
                           'fields',P.Fsolve, 'quiet',true);
            o = pack_fold(src, f, tmp, 'quiet',true);
            L = pack_legs(tmp, 'instr',P.pack.instr_len, 'quiet',true);
            K = pack_clear(tmp, 'init',false, 'fields',P.Fsolve, ...
                           'body_pad',P.pack.fold_margin, ...
                           'fold_elts', find(o.isfold), 'quiet',true);
            E = envelope_(L, K, P, opts, true);
            W(end+1) = struct('x_out',x, 'z_deep',max(L.z), 'nLost',K.nLost, ...
                'floor_fold_mm',K.floor_fold_body_m*1e3, ...
                'floor_pre_mm',K.floor_pre_body_m*1e3, ...
                'instr_r',E.instr.r_max, 'shroud',E.shroud_dia, ...
                'ok_r', E.instr.r_max <= opts.r_env); %#ok<AGROW>
            fprintf('    %9.3f %10.4f %14.2f %14.2f %12.3f %10.3f %8d  %s\n', ...
                    x, max(L.z), K.floor_fold_body_m*1e3, K.floor_pre_body_m*1e3, ...
                    E.instr.r_max, E.shroud_dia, K.nLost, ...
                    tern_(E.instr.r_max <= opts.r_env, 'instr inside r_env', ''));
        catch ME
            fprintf('    %9.3f  route failed: %s\n', x, ME.message);
        end
    end
end

function delete_if_(f), if isfile(f), delete(f); end, end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
function s = tern2_(c,a,b), if c, s = a; else, s = b; end, end

function hdr_(s), fprintf('\n============ %s ============\n', s); end
function s = yn_(b), if b, s = 'YES'; else, s = 'no'; end, end
