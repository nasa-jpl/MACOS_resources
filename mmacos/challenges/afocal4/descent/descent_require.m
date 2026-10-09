function Q = descent_require(P, deck, opts)
%DESCENT_REQUIRE  The ladder's requirement set, scored on one footing.
%
%   Q = DESCENT_REQUIRE(P, DECK) judges one rung against the FULL requirement
%   set the descent brief fixes -- every row, with its MARGIN, on every rung,
%   so that "what the removed mirror bought" is a subtraction between two
%   tables rather than a comparison of two prose paragraphs.
%
%   THE 71 nm WAVEFRONT TARGET IS IN THE SET.  That is the ruling that makes
%   this stage different from S4: the 4-mirror family never met it (best
%   ~7.5 um at the operating points, 100x off) and the arc lived with a
%   requirement PAIR of which only half was ever on the table.  The descent
%   starts where the whole set is met and walks down to find where it breaks,
%   so the target has to be scored, not carried as an aspiration.
%
%   TARGETS, WALLS AND GATES ARE THREE DIFFERENT THINGS and the table says
%   which is which:
%
%     TARGET  a number the design is trying to beat.  Reported as a margin;
%             missing one is what ENDS the ladder, and which one broke first
%             is the ladder's answer.
%     WALL    a constraint the builder refuses to violate at all.  It cannot
%             be "missed" -- a design that violates it does not exist -- so
%             its column reports how much room is left, never a ratio.
%     GATE    a fact that must hold for the numbers to mean anything (rays
%             lost, the solver-integrity residual).  A rung that fails a gate
%             is not a worse rung, it is a rung that was not measured.
%
%   THE INTERFACE SURFACE IS SCORED RIM-ANCHORED.  S4c's spec rule, and it is
%   not cosmetic: the convergence surface reads 0.0174 mm surface-anchored
%   and 0.1853 mm rim-anchored on the same design -- the difference between
%   12x inside the 0.2 mm target and 1.08x inside it.  A flat object's ideal
%   image is a plane and the powered design does not deliver one; the
%   curved-object reference was absorbing that.  Scoring it the loose way up
%   a seven-rung ladder would hide the one row that nearly binds.
%
%   Name-value:
%     'fields'  field set (default P.Fsolve -- his 3x3 box)
%     'nodes'   pupil_map lattice (default P.solve.nodes_score)
%     'grid'    uniform WFE scoring grid (default P.grid_n)
%     'union'   also run the union body-in-beam gate (true).  It is a
%               nine-field re-trace and the ladder wants it on every rung,
%               but a quick pass can turn it off.
%     'quiet'   (false)
%
%   Returns Q with .rows (name, kind, value, target, margin, ok), .ok
%   (every TARGET met), .walls_ok, .gates_ok, .worst (the largest normalized
%   miss), .S (the full score), .K (the union gate), and .slack -- the
%   minimum margin over the target rows, which is what "the top rung must
%   have SLACK everywhere" is measured by.
%
%   See also AFOCAL4_SCORE, AFOCAL4_UNION, DESCENT_BUILD, DESCENT_LADDER.

    arguments
        P (1,1) struct
        deck (1,:) char
        opts.fields (:,2) double = []
        opts.nodes  (1,1) double = 0
        opts.grid   (1,1) double = 0
        opts.union  (1,1) logical = true
        opts.quiet  (1,1) logical = false
    end
    F = opts.fields;   if isempty(F), F = P.Fsolve; end
    nd = opts.nodes;   if nd == 0, nd = P.solve.nodes_score; end
    gr = opts.grid;    if gr == 0, gr = P.grid_n; end

    % ---- the score, and the interface surface RIM-anchored ---------------
    S  = afocal4_score(P, deck, 'fields',F, 'nodes',nd, 'grid',gr);
    Sr = afocal4_score(P, deck, 'fields',F, 'nodes',nd, 'anchor','rim');
    T  = P.targets;

    % ---- the union gate + the traced interface ---------------------------
    K = [];   floor_mm = NaN;   bare_mm = NaN;
    if opts.union
        Kb = afocal4_union(deck, 'fields',F, 'body_k',1.0, 'body_pad',0.0, ...
                           'quiet',true);
        K  = afocal4_union(deck, 'fields',F, ...
                           'body_k',getf_(P.pack,'union_body_k',1.15), ...
                           'body_pad',getf_(P.pack,'union_body_pad',0.015), ...
                           'init',false, 'quiet',true);
        floor_mm = K.floor_m*1e3;   bare_mm = Kb.floor_m*1e3;
    end
    tr  = traced_(deck, P.D);
    aoi = aoi_chief_(deck);
    zz  = stations_(deck);

    r = struct('name',{},'kind',{},'value',{},'target',{},'unit',{}, ...
               'margin_pct',{},'ok',{});
    % --- TARGETS: less is better, margin = how far inside ------------------
    r = add_(r,'WFE rung-2 max','target', S.wfe_max_nm,  T.wfe_rung2_nm, 'nm');
    r = add_(r,'pupil blur',    'target', S.blur_um,     T.blur_um,      'um');
    r = add_(r,'wander (refit)','target', S.wander_um,   T.wander_um,    'um');
    r = add_(r,'breathing',     'target', S.breathe_pct, T.breathe_pct,  '%');
    r = add_(r,'iface surface (rim)','target', Sr.surf_pv_mm, T.surface_pv_mm, 'mm');
    r = add_(r,'M error',       'target', abs(S.mag_centre_chief/T.mag - 1)*100, ...
                                 T.mag_pct, '%');
    % --- WALLS: room left, never a ratio -----------------------------------
    r = add_(r,'union floor (declared)','wall', floor_mm, 0, 'mm');
    r = add_(r,'last powered behind M1','wall', zz.behind_m1*1e3, ...
              getf_(P.pack,'m3_behind_min',0.5)*1e3, 'mm');
    r = add_(r,'min spacing',   'wall', zz.tmin*1e3, 20, 'mm');
    r = add_(r,'max chief AOI', 'wall', aoi.max_deg, 15, 'deg');
    % --- GATES: facts, not scores ------------------------------------------
    r = add_(r,'rays lost',     'gate', S_lost_(K), 0, '');
    r = add_(r,'anchoring resid','gate', S.anchor_resid_um, 100, 'um');

    tgt = strcmp({r.kind},'target');
    wal = strcmp({r.kind},'wall');
    gat = strcmp({r.kind},'gate');
    Q = struct('deck',deck, 'rows',r, 'S',S, 'S_rim',Sr, 'K',K, ...
               'floor_mm',floor_mm, 'bare_mm',bare_mm, 'traced',tr, ...
               'aoi',aoi, 'z',zz, ...
               'ok',all([r(tgt).ok]), 'walls_ok',all([r(wal).ok]), ...
               'gates_ok',all([r(gat).ok]));
    Q.worst = max([r(tgt).value] ./ [r(tgt).target]);
    Q.slack = min([r(tgt).margin_pct]);
    Q.all_ok = Q.ok && Q.walls_ok && Q.gates_ok;
    if ~opts.quiet, report_(Q); end
end

% =====================================================================
function r = add_(r, name, kind, value, target, unit)
%ADD_  One row.  A WALL's margin is room left in its own unit, never a
%   percentage of a target it is not trying to approach; a TARGET's margin is
%   how far INSIDE it the design sits, which is what "slack" means.
    switch kind
    case 'target'
        m  = 100*(1 - value/target);          % + = inside
        ok = value <= target;
    case 'wall'
        m  = value - target;                  % room left, in the row's unit
        ok = value >= target;
        if strcmp(name,'max chief AOI'), m = target - value; ok = value <= target; end
        if strcmp(name,'min spacing'),   m = value - target; ok = value >= target; end
    case 'gate'
        m  = target - value;
        ok = value <= target;
    end
    r(end+1) = struct('name',name, 'kind',kind, 'value',value, ...
                      'target',target, 'unit',unit, 'margin_pct',m, 'ok',ok);
end

function report_(Q)
    fprintf('\n  REQUIREMENT SET  %s\n', Q.deck);
    fprintf('    %-24s %6s %12s %12s %12s  %s\n', 'row','kind','value','target', ...
            'margin','');
    for i = 1:numel(Q.rows)
        r = Q.rows(i);
        if strcmp(r.kind,'target')
            mg = sprintf('%+8.1f %%', r.margin_pct);
        else
            mg = sprintf('%+8.2f %s', r.margin_pct, r.unit);
        end
        fprintf('    %-24s %6s %9.4f %s %9.4f %s %12s  %s\n', r.name, r.kind, ...
                r.value, pad_(r.unit), r.target, pad_(r.unit), mg, ...
                tern_(r.ok,'','<-- MISSED'));
    end
    fprintf(['    => targets %s, walls %s, gates %s;  worst miss %.2fx, ' ...
             'slack %+.1f %%\n'], yn_(Q.ok), yn_(Q.walls_ok), yn_(Q.gates_ok), ...
            Q.worst, Q.slack);
end

function z = stations_(deck)
    macos.load_rx(deck);
    nE = macos.num_elt();
    V = zeros(3,nE);
    for k = 1:nE, V(:,k) = macos.get_elt_vpt(k); end
    zv = V(3,1:nE-1);                       % powered mirrors only
    z = struct('z',zv, 'behind_m1',zv(end)-zv(1), ...
               'tmin',min(abs(diff(zv))));
end

function A = aoi_chief_(deck)
%AOI_CHIEF_  Chief incidence on every mirror from the traced chief alone --
%   the AFOCAL4_CLEARING construction, kept verbatim so the ladder's AOI
%   column is comparable with every other table in this study.
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
    A = struct('per_elt_deg',a, 'max_deg',max(a(1:nE-1)));
end

function s = traced_(deck, Dap)
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

function n = S_lost_(K)
    if isempty(K), n = 0; else, n = K.nLost; end
end
function v = getf_(s, f, d),  if isfield(s,f), v = s.(f); else, v = d; end,  end
function s = pad_(u),  s = sprintf('%-3s', u);  end
function s = yn_(b),   s = tern_(b,'MET','MISSED');  end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
