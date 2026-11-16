function W = afocal4_union_wall(P, deck, opts)
%AFOCAL4_UNION_WALL  The union body-in-beam floor, applied as a WALL.
%
%   W = AFOCAL4_UNION_WALL(P, DECK) measures AFOCAL4_UNION's floor on a
%   COMMITTED deck and, if P.pack.union_enforce is true, ERRORS when that
%   floor is below P.pack.union_min.  It is the clearing stage's follow-on
%   to the S4b packaging wall in AFOCAL4_BUILD, and it is deliberately the
%   same SHAPE of object: a wall the solver turns back on, never a term in
%   the merit.
%
%   WHY A WALL AND NOT A MERIT TERM.  The clearing stage delivered a -10 deg
%   extraction tilt whose raw union floor was +57.4 mm, and measured that at
%   -8 and -9 deg the re-solve walks +23.3 and +42.3 mm of margin down to
%   +2.3 and +0.7 mm -- because AFOCAL4_SCORE cannot see clearance, so the
%   solver spends it on the wavefront for free.  That is not a design
%   choosing a trade; it is a quantity nobody is holding.  The log-domain
%   merit doctrine (RESULTS section 5 rule 1) stands untouched: a body
%   standing in a beam is not a worse telescope, it is not a telescope.
%
%   OFF BY DEFAULT, AND THAT IS LOAD-BEARING.  P.pack.union_enforce defaults
%   to FALSE, so every committed S4 / S4b / S4c / clearing artifact rebuilds
%   BIT-IDENTICALLY and no number in the record moves.  Turning it on is an
%   explicit act of the study that wants it -- and the committed 343 mm deck
%   would fail it at -79.9 mm, i.e. AFOCAL4_BUILD could not even re-emit the
%   design the trade shipped.  (The S4b wall had the same problem in reverse
%   and solved it the same way: P.pack.enforce = false reproduces S4.)
%
%   THE FLOOR IS READ ON THE DECLARED BODY MODEL (1.15 x union footprint +
%   15 mm), because that is what the gate's verdict means and a wall must
%   hold the same quantity the gate reports.  BARE LIT GLASS is reported
%   beside it wherever the answer is PRINTED -- 'bare',true here, and
%   unconditionally in every frontier table -- but not inside the solver
%   loop, where it would double a cost that is already the dominant one.
%
%   COST, MEASURED, because a wall that is too slow to sit in the loop has
%   to be said so out loud: on this design at solve sampling (ngrid 21,
%   nodes 11) one evaluation is 1.6 s of CLEAR_BUILD + 6.6 s of
%   AFOCAL4_SCORE = 8.2 s, and the wall adds 4.2 s -- +51 %.  Nearly all of
%   that is the nine-field RE-TRACE inside AFOCAL4_UNION (the probe count
%   barely matters: 314 probes cost 4.18 s and 65 cost 3.52 s), which is a
%   trace AFOCAL4_SCORE has already paid for once.  Sharing it would mean
%   restructuring the committed scorer, so it is not done: the wall is
%   evaluated INSIDE the build, every iterate the solver sees is compliant,
%   and the frontier costs half again as much per evaluation.
%
%   P.pack fields (all optional; a P that predates them gets the wall OFF):
%     .union_enforce   apply the wall at all (false)
%     .union_min       the floor it must hold, m (0 -- the gate's own pass
%                      condition; +0.015 is the declared allowance's own pad)
%     .union_body_k    body = this x union footprint (1.15)
%     .union_body_pad  ... grown by this, m (0.015)
%
%   Name-value:
%     'fields'  K x 2 field offsets, rad (default P.Fsolve -- the field BOX;
%               the whole defect lives in the difference between one field
%               and all of them)
%     'throw'   error when the floor is below the wall (true).  False
%               measures without judging, which is what a report wants.
%     'bare'    also measure bare lit glass, 1.00 x + 0 mm (false -- it is a
%               second nine-field trace)
%     'init'    load the deck (true);  'quiet' (true)
%
%   Returns W with .enforced .ok .floor_m .min_m .bare_m .worst_name
%   .nLost .body_k .body_pad .K (the full AFOCAL4_UNION result) .Kbare.
%
%   See also AFOCAL4_UNION, AFOCAL4_BUILD, CLEAR_BUILD, WALL_SEED.

    arguments
        P (1,1) struct
        deck (1,:) char
        opts.fields (:,2) double = []
        opts.throw  (1,1) logical = true
        opts.bare   (1,1) logical = false
        opts.init   (1,1) logical = true
        opts.quiet  (1,1) logical = true
    end

    U = union_spec_(P);
    F = opts.fields;
    if isempty(F) && isfield(P,'Fsolve'), F = P.Fsolve; end

    W = struct('enforced',U.enforce, 'ok',true, 'floor_m',NaN, ...
               'min_m',U.min, 'bare_m',NaN, 'worst_name','', 'nLost',NaN, ...
               'body_k',U.body_k, 'body_pad',U.body_pad, 'K',[], 'Kbare',[]);
    % OFF and asked to JUDGE: return without tracing a ray.  OFF and asked to
    % MEASURE ('throw',false -- what a report does) still measures, because a
    % report wants the number whether or not a wall is in force.
    if ~U.enforce && opts.throw
        return;
    end

    K = afocal4_union(deck, 'fields',F, 'body_k',U.body_k, ...
                      'body_pad',U.body_pad, 'init',opts.init, 'quiet',true);
    W.K = K;   W.floor_m = K.floor_m;   W.worst_name = K.worst_name;
    W.nLost = K.nLost;
    W.ok = K.floor_m >= U.min;
    if opts.bare
        Kb = afocal4_union(deck, 'fields',F, 'body_k',1.0, 'body_pad',0.0, ...
                           'init',false, 'quiet',true);
        W.Kbare = Kb;   W.bare_m = Kb.floor_m;
    end
    if ~opts.quiet
        fprintf(['    union wall: floor %+.2f mm (body %.2fx + %.0f mm) ' ...
                 'against %+.1f mm%s -- %s\n'], W.floor_m*1e3, U.body_k, ...
                U.body_pad*1e3, U.min*1e3, bare_(W), tern_(W.ok,'CLEAR','IN BEAM'));
    end

    if U.enforce && ~W.ok && opts.throw
        error('macos:design:afocal4_build:union', ...
              ['a body stands in a beam: %s at %+.2f mm against a %+.1f mm ' ...
               'floor (body %.2fx footprint + %.0f mm).  The clearance is a ' ...
               'WALL, not a merit term -- this layout is not a telescope ' ...
               'anyone can build.'], K.worst_name, K.floor_m*1e3, U.min*1e3, ...
              U.body_k, U.body_pad*1e3);
    end
end

% =====================================================================
function U = union_spec_(P)
%UNION_SPEC_  The wall's spec, with the pre-clearing behaviour (no wall) as
%   the default for a P that predates it.  Same construction -- and same
%   reason -- as PACK_SPEC_ inside AFOCAL4_BUILD: a parameter struct saved
%   inside an older .mat must still rebuild its own decks, and it must
%   rebuild them under the rules it was scored under.
    U = struct('enforce',false, 'min',0.0, 'body_k',1.15, 'body_pad',0.015);
    if isfield(P,'pack')
        m = struct('union_enforce','enforce', 'union_min','min', ...
                   'union_body_k','body_k', 'union_body_pad','body_pad');
        f = fieldnames(m);
        for i = 1:numel(f)
            if isfield(P.pack, f{i}), U.(m.(f{i})) = P.pack.(f{i}); end
        end
    end
end

function s = bare_(W)
    if isnan(W.bare_m), s = ''; else, s = sprintf(' (bare %+.2f mm)', W.bare_m*1e3); end
end

function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
