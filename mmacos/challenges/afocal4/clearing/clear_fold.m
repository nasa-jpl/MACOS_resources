function R = clear_fold(P, deck, opts)
%CLEAR_FOLD  Can a FLAT get the collimator out of the feed beam?  No -- here
%   is the measurement, and the two-sided squeeze that makes it structural.
%
%   R = CLEAR_FOLD(P, DECK) inserts a single flat extraction fold into the
%   M2 -> field-mirror feed leg at a range of stations, turns the post-focus
%   train aside, and gates every result with AFOCAL4_UNION.  It is leverage 1
%   of BRIEF_afocal4_clear -- the cheapest physics on the list, and the one
%   that has to be tried first.
%
%   WHY IT LOOKS LIKE IT SHOULD WORK.  A fold inserted BETWEEN the two
%   conflict partners re-routes one of them, so the usual "an isometry
%   carries every clearance across unchanged" objection does not apply.
%   Near the internal focus the beam is at its smallest, so the flat is
%   cheap; and everything past the focus -- field mirror, collimator, cold
%   stop -- goes with it.
%
%   WHY IT DOES NOT, AND THE NUMBER IS EXACT.  Let LEG be the M2 -> field
%   mirror vertex spacing and B the field-mirror -> collimator spacing.  Put
%   the flat DIST along the leg and everything downstream lies on the folded
%   axis at
%       fold 0,   field mirror (LEG - DIST),   collimator (LEG - DIST - B),
%   so the collimator sits exactly ON the flat when DIST = LEG - B.  On
%   either side of that station one of two things is true:
%
%     DIST > LEG - B   the collimator is still DOWNSTREAM of the fold, so
%                      the fold has not separated it from the piece of feed
%                      beam that pierces it -- both moved together, and the
%                      floor is the parent's, unchanged;
%     DIST < LEG - B   the collimator is now BEHIND the fold, so the
%                      field-mirror -> collimator beam has to travel back
%                      through the flat's own station, and the flat -- which
%                      is union-sized, ~250 mm across, because the field
%                      walk at the intermediate image dominates the beam --
%                      is what is in the way.
%
%   The two conditions are complementary and meet at a single station, so
%   there is no window: ONE flat cannot do it.  That is what this routine
%   measures, station by station, and it reports which pair binds at each
%   one so the mechanism is visible and not just the verdict.
%
%   The obvious escape -- put the flat far enough before the focus that the
%   returning beam misses it sideways -- is closed by the same field-walk
%   law CLEAR_LAW measures: the outgoing and returning bundles at that
%   station are two scaled copies of the field box whose ratio only reaches
%   the required 2.43 about a metre past the field mirror, which is well
%   beyond where the collimator is allowed to be.
%
%   Name-value:
%     'dist'    fold stations, as a FRACTION of the M2 -> FM leg.  The
%               default is AUTO: a spread of stations placed relative to the
%               critical one, so the squeeze is resolved on both sides of it
%               whatever deck this is run on.  A fixed list would straddle
%               the crossing on one deck and miss it on the next.
%     'to'      cell of 1x3 turn directions (default {+x, -y})
%     'leg'     which leg to fold (default 2, M2 -> field mirror)
%     'fields'  field set (default P.Fsolve)
%     'body_k'/'body_pad'/'quiet'
%
%   Returns R with .crit_frac (the critical station), .leg_m, .b_m and
%   .pt (per station: .dist_m .frac .to .floor_bare .floor_body .worst
%   .nLost .deepest_m).
%
%   See also CLEAR_LAW, CLEAR_TILT, AFOCAL4_UNION, PACK_FOLD.

    arguments
        P (1,1) struct
        deck (1,:) char
        opts.dist     (1,:) double = []
        opts.to       (1,:) cell = {[1 0 0], [0 -1 0]}
        opts.leg      (1,1) double = 2
        opts.fields   (:,2) double = []
        opts.body_k   (1,1) double = 1.15
        opts.body_pad (1,1) double = 0.015
        opts.quiet    (1,1) logical = false
    end
    F = opts.fields;   if isempty(F), F = P.Fsolve; end

    macos.load_rx(deck);
    nE = macos.num_elt();
    V  = zeros(3,nE);
    for k = 1:nE, V(:,k) = macos.get_elt_vpt(k); end
    k   = opts.leg;
    leg = norm(V(:,k+1) - V(:,k));
    b   = norm(V(:,k+2) - V(:,k+1));
    R = struct('deck',deck, 'leg_m',leg, 'b_m',b, ...
               'crit_m', leg - b, 'crit_frac',(leg-b)/leg, 'pt',[]);
    fr = opts.dist;
    if isempty(fr)
        fr = R.crit_frac + [-0.60 -0.35 -0.15 -0.06 -0.015 0.015 0.06 0.11];
        fr = [0.05, fr(fr > 0.02 & fr < 0.99), 0.98];
    end
    if ~opts.quiet
        fprintf(['\n  --- LEVERAGE 1: one flat extraction fold in the ' ...
                 'feed leg ---\n']);
        fprintf(['    leg %d->%d is %.4f m, next spacing %.4f m -> the ' ...
                 'collimator lands ON the flat at dist %.4f m (%.3f of the ' ...
                 'leg)\n'], k, k+1, leg, b, leg-b, (leg-b)/leg);
        fprintf('  %9s %7s %10s %9s %9s %9s %5s  %s\n', 'dist m','frac', ...
                'turn','bare mm','body mm','deepest','lost','binding pair');
    end

    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>
    pts = struct('dist_m',{},'frac',{},'to',{},'floor_bare',{}, ...
                 'floor_body',{},'worst',{},'nLost',{},'deepest_m',{});
    for j = 1:numel(opts.to)
        for f = fr
            d = f*leg;
            fs = struct('name','XF', 'after',k, 'dist',d, 'to',opts.to{j});
            try
                pack_fold(deck, fs, tmp, 'quiet',true);
                Kb = afocal4_union(tmp, 'fields',F, 'body_k',1.0, ...
                                   'body_pad',0.0, 'quiet',true);
                Km = afocal4_union(tmp, 'fields',F, 'body_k',opts.body_k, ...
                                   'body_pad',opts.body_pad, 'init',false, ...
                                   'quiet',true);
            catch ME
                % Report the failure whatever the verbosity: a station that
                % cannot be measured is a hole in the sweep, and a sweep
                % with holes must not read as a sweep that found nothing.
                fprintf('  %9.4f %7.3f %10s   FAILED: %s\n', d, f, ...
                        v_(opts.to{j}), one_line_(ME.message));
                lasterr(ME.message); %#ok<LERR>
                continue;
            end
            zz = zeros(1,Km.nElt);
            for e = 1:Km.nElt, zz(e) = Km.vpt(3,e); end
            p = struct('dist_m',d, 'frac',f, 'to',opts.to{j}, ...
                       'floor_bare',Kb.floor_m, 'floor_body',Km.floor_m, ...
                       'worst',Km.worst_name, 'nLost',Km.nLost, ...
                       'deepest_m',max(zz));
            pts(end+1) = p; %#ok<AGROW>
            if ~opts.quiet
                fprintf('  %9.4f %7.3f %10s %9.2f %9.2f %9.4f %5d  %s\n', ...
                        d, f, v_(opts.to{j}), p.floor_bare*1e3, ...
                        p.floor_body*1e3, p.deepest_m, p.nLost, p.worst);
            end
        end
    end
    % Not one station produced a measurement.  That is not a datum -- a
    % swept study whose every point failed reads as "nothing clears", which
    % is the answer this routine is looking for, so it must be an error and
    % not a quiet empty result.  (It was, once: PACK_FOLD lives in
    % ../packaging and a caller that had not put it on the path got a clean
    % "a flat cannot clear it" out of nine undefined-function exceptions.)
    if isempty(pts)
        error('macos:design:clear_fold:none', ...
              ['no fold station produced a measurement -- every one errored.  ' ...
               'Last error: %s'], one_line_(lasterr)); %#ok<LERR>
    end
    R.pt = pts;
    if ~opts.quiet
        [best, ib] = max([pts.floor_body]);
        fprintf(['    best over every station and both turn directions: ' ...
                 '%+.2f mm (%s, dist %.3f of the leg) -- %s\n'], best*1e3, ...
                v_(pts(ib).to), pts(ib).frac, ...
                tern_(best >= 0, 'CLEARS', 'still a body in a beam'));
    end
end

% =====================================================================
function s = v_(t),  s = sprintf('[%g %g %g]', t(1), t(2), t(3));  end
function del_(p),  if exist(p,'file'), delete(p); end,  end
function s = one_line_(m)
    s = regexprep(m, '\s+', ' ');   if numel(s) > 80, s = [s(1:80) '...']; end
end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
