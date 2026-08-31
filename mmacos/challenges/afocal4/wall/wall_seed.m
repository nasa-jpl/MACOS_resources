function [D, info] = wall_seed(P, D0, tilt_deg, opts)
%WALL_SEED  A design that already CLEARS, to start a walled solve from.
%
%   [D, INFO] = WALL_SEED(P, D0, TILT_DEG) returns a CLEAR_BUILD design
%   struct at the given extraction tilt whose union body-in-beam floor is at
%   least P.pack.union_min plus a margin -- i.e. a point INSIDE the feasible
%   region the union wall bounds.
%
%   WHY THIS EXISTS, AND IT IS NOT A NEW LESSON.  S4b earned it: *a wall
%   needs a compliant seed or it is a cage.*  Dropped in at a non-compliant
%   point, every finite-difference direction is an error, the Jacobian is
%   walls all the way round, and lsqnonlin hands back the seed it was given
%   -- which reads as "this operating point has no design" and is actually a
%   seeding failure.  Measured then: warm-starting the S4 trade lost FOUR OF
%   ITS FIVE points that way, every one of which AFOCAL4_PACK_SEED closed.
%   This is that function's clearing-stage sibling.
%
%   PROBE, THEN BISECT -- AND DO NOT TRUST THE LAW AS A PREDICTOR HERE.
%   AFOCAL4_PACK_SEED can rank its candidates by preference alone, because
%   every closure it looks at either satisfies its constraint or does not by
%   pure algebra.  This wall is not algebra -- it costs a nine-field trace,
%   ~6 s -- so the search has to be short.  The obvious short search is to
%   predict each candidate from the field-walk law: a tilt separates the two
%   bundles by a field-INDEPENDENT 2*alpha*d, d the field-mirror ->
%   collimator spacing, so
%
%       f_hat = f(tilt alone) + 2*|alpha| * (d - d_0)      <-- WRONG HERE
%
%   MEASURED, IT IS 5x OPTIMISTIC.  At -6 deg, moving the standoff from the
%   parent's -38.6 mm to +250 mm takes d from 0.563 to 0.680 m and the law
%   predicts the floor going -13.00 -> +11.45 mm; it measures -8.25.  Over
%   the whole probed range the realised slope is 40.6 mm per metre of d
%   against the law's 209.  The law is not wrong -- the tilt really does
%   supply 2*alpha*d -- but a STANDOFF change moves the FIELD-PROPORTIONAL
%   part at the same time, and the two very nearly cancel.  That is leverage
%   2 (the station) showing up again inside leverage 3: the clearing stage
%   retired the station as "nearly powerless" and it is nearly powerless
%   here too.  Predicting from the law alone would filter the candidate list
%   down to points that do not clear.
%
%   So the search MEASURES the slope instead of assuming it: gate the
%   parent's own standoff, gate the extreme of the admitted range (the most
%   lever the closure allows), and if the extreme clears, BISECT back toward
%   the parent for the SMALLEST departure that still clears.  Four to six
%   gate evaluations, no prediction relied on, and INFO.slope_mm_per_m
%   carries the realised sensitivity beside the law's -- which makes the
%   seeder one more measurement rather than a black box.
%
%   WHAT IT PREFERS, in order:
%     1  THE TILT ALONE.  If the parent design swung by TILT_DEG already
%        clears, nothing else is moved.  That is the answer worth having --
%        it says the clearance came from the tilt and not from a re-posing
%        the price table would then have to carry -- and it is what keeps
%        the frontier comparable with the delivered -10 deg row, which was
%        seeded exactly that way.
%     2  THE SMALLEST FIELD-MIRROR STANDOFF CHANGE THAT CLEARS, on the
%        parent's own front end.  Minimal departure keeps the frontier point
%        in the basin the delivered row lives in; jumping to the largest
%        available lever would clear just as well and would silently be a
%        different design.
%     3  A DIFFERENT FRONT END (M2's radius), because the S4b anchor is that
%        this study changes the front end last.  Tried only if the standoff
%        alone cannot reach.
%     4  THE DELIVERED -10 deg CLEARED DESIGN's own DOFs, as a last resort.
%        It is a KNOWN-feasible point (BRIEF_afocal4_wall names it as one),
%        so a frontier point that can only be seeded from it is still a
%        frontier point -- but it is a different basin from the parent's,
%        and INFO.source says so rather than leaving the reader to assume
%        the whole curve was seeded the same way.
%
%   THE BISECTION ASSUMES THE FLOOR IS MONOTONE IN THE STANDOFF over the
%   probed interval, which the probe points confirm on this design (-8.25,
%   -7.94, -7.64, -7.35, -7.08, -6.82 mm at s = +250..+375).  If it were
%   not, the bisection would return a compliant point that is not the
%   smallest compliant departure -- suboptimal, never wrong.
%
%   AND THE CLOSURE MUST BE RE-POSED ON THE PARENT'S OWN FRONT END, NOT ON
%   P.parent.  Cost one debugging cycle: P.parent carries MIKE's raw
%   secondary (R_M2 468.8 mm, t_M1M2 1.0492 m) while the committed 343 mm
%   deck has a re-solved front end (448.4 mm, 1.0420 m).  Filtering
%   candidates through P.parent admitted 21 standoffs of 57 and none of them
%   was the parent design's own; carrying D.R2 and D.t1 into the closure --
%   which is exactly what AFOCAL4_BUILD does -- admits 54, spanning
%   d = 0.255 .. 0.821 m against the parent's 0.563.
%
%   MARGIN OVER THE BOUND, NOT THE BOUND ITSELF.  A seed sitting exactly on
%   the wall has half its finite-difference stencil outside it, so the
%   default asks for 10 mm more clearance than the wall demands.  That
%   margin also absorbs the SAMPLING difference between the wall (evaluated
%   at solve sampling) and the gate (quoted at reporting sampling), which is
%   about 1 mm on this design and always in the optimistic direction.
%
%   A FAILURE HERE IS REPORTED AS A FAILURE TO SEED, never as a design
%   verdict: INFO.ok is false, INFO.best_floor_m carries the closest it got,
%   and the caller must say "no compliant seed at this tilt" rather than
%   "this tilt has no design".
%
%   Name-value:
%     'margin'    m of clearance over P.pack.union_min (0.010)
%     'standoff'  standoff grid to search (default P.bounds.fm_standoff in
%                 25 mm steps)
%     'R2'        M2 radii to try IN ORDER (default: the parent's own, then
%                 his, then 0.470..0.430)
%     'max_gate'  most build-and-gate evaluations to spend (18)
%     'fallback'  the delivered cleared deck for preference 4 ('' = the
%                 stage's own ../clearing/afocal4_clear_343mm.in; 'none')
%     'fields'    field set (default P.Fsolve)
%     'axis'      tilt axis handed to CLEAR_BUILD ([1 0 0])
%     'quiet'     (true)
%
%   Returns D (ready for CLEAR_SOLVE, tilt_deg set) and INFO with .ok
%   .source .floor_m .bare_m .need_m .phi4 .standoff .R2 .d_m
%   .used_own_front_end .n_closed .n_gated .slope_mm_per_m (the realised
%   sensitivity of the floor to the lever) .tried (every candidate gated,
%   with its floor) .best_floor_m .seconds.
%
%   See also AFOCAL4_PACK_SEED, AFOCAL4_UNION_WALL, CLEAR_BUILD, WALL_POINT.

    arguments
        P (1,1) struct
        D0 (1,1) struct
        tilt_deg (1,1) double
        opts.margin   (1,1) double = 0.010
        opts.standoff (1,:) double = []
        opts.R2       (1,:) double = []
        opts.max_gate (1,1) double = 18
        opts.fallback (1,:) char = ''
        opts.fields   (:,2) double = []
        opts.axis     (1,3) double = [1 0 0]
        opts.quiet    (1,1) logical = true
    end
    t0 = tic;
    F = opts.fields;   if isempty(F), F = P.Fsolve; end
    ss = opts.standoff;
    if isempty(ss), ss = P.bounds.fm_standoff(1):0.025:P.bounds.fm_standoff(2); end
    R2s = opts.R2;
    if isempty(R2s)
        R2s = unique_stable_([D0.R2, P.parent.R(2), 0.470, 0.460, 0.450, ...
                              0.440, 0.430]);
    end
    fb = opts.fallback;
    if isempty(fb)
        fb = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
                      'clearing', 'afocal4_clear_343mm.in');
    end

    U = wall_spec_(P);
    need = U.min + opts.margin;
    al2  = 2*abs(deg2rad(tilt_deg));         % the law's own coefficient

    % measure, never judge, while seeding: a candidate that fails the wall is
    % information, not an exception.
    Q = P;   Q.pack.union_enforce = false;

    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>

    tried = struct('source',{},'standoff',{},'R2',{},'phi4',{},'d_m',{}, ...
                   'pred_mm',{},'floor_m',{},'bare_m',{});
    info = struct('ok',false, 'source','', 'floor_m',NaN, 'bare_m',NaN, ...
                  'need_m',need, 'phi4',NaN, 'standoff',NaN, 'R2',NaN, ...
                  'd_m',NaN, 'used_own_front_end',true, 'n_closed',0, ...
                  'n_gated',0, 'tried',tried, 'best_floor_m',-Inf, ...
                  'slope_mm_per_m',NaN, 'law_slope_mm_per_m',al2*1e3, ...
                  'seconds',0);
    D = D0;   D.tilt_deg = tilt_deg;

    % ---- preference 1: the tilt alone ------------------------------------
    [f, b, d0] = gate_(Q, D, tmp, F, opts.axis);
    tried(end+1) = rec_('tilt alone', D.fm_standoff, D.R2, NaN, d0, NaN, f, b);
    info.n_gated = 1;   info.best_floor_m = f;
    say_(opts.quiet, ['    seed 1  tilt alone (parent untouched): d %.3f m -> ' ...
         'floor %+.2f mm (need %+.1f)\n'], d0, f*1e3, need*1e3);
    if f >= need
        info = finish_(info, tried, 'tilt alone', D, f, b, NaN, d0, D0, t0);
        say_(opts.quiet, '    -> the tilt alone complies; nothing else moved\n');
        return;
    end
    if ~isfinite(f)
        error('macos:design:wall_seed:parent', ...
              ['the parent design does not even BUILD at tilt %+.1f deg -- ' ...
               'there is nothing to seed from.'], tilt_deg);
    end

    % ---- preferences 2 and 3: probe the standoff, then bisect ------------
    % Closure validity and the S4b packaging station are pure arithmetic on
    % AFOCAL4_PHI4's output, so the admitted list is built by them before a
    % single ray is traced.  THE CLOSURE IS RE-POSED ON THE PARENT'S OWN
    % FRONT END -- D.R2 and D.t1, exactly as AFOCAL4_BUILD does it.
    for R2 = R2s
        if info.n_gated >= opts.max_gate, break; end
        Qc = Q;   Qc.parent.R(2) = R2;   Qc.parent.t(1) = D.t1;
        adm = struct('s',{},'phi4',{},'d',{});
        for s = ss
            [phi4, C, found] = afocal4_phi4(Qc, s, D.iface);
            if ~found || any(C.t < 0.02), continue; end
            if P.pack.enforce && C.behind_m1 < P.pack.m3_behind_min, continue; end
            adm(end+1) = struct('s',s, 'phi4',phi4, 'd',C.t(3)); %#ok<AGROW>
        end
        info.n_closed = info.n_closed + numel(adm);
        own = (R2 == D0.R2);
        [~, io] = sort([adm.d]);   adm = adm(io);       % ascending lever
        % ON THE PARENT'S OWN FRONT END, only the candidates with MORE lever
        % than the parent are useful, and over that half the floor is
        % monotone in d -- which is what makes the bisection sound and what
        % makes "the smallest d that clears" mean "the smallest departure in
        % the direction that helps".  On a DIFFERENT front end the parent's
        % own d is not a reference for anything, so the whole family is
        % searched.
        if own, adm = adm([adm.d] >= d0); end
        if numel(adm) < 2, continue; end

        % probe the extreme: the most lever this closure family allows
        hi = numel(adm);
        [fh, bh, dh] = gate_(Q, setsr_(D,adm(hi).s,R2), tmp, F, opts.axis);
        info.n_gated = info.n_gated + 1;
        info.best_floor_m = max(info.best_floor_m, fh);
        pred = f + al2*(adm(hi).d - d0);
        tried(end+1) = rec_('probe', adm(hi).s, R2, adm(hi).phi4, dh, ...
                            pred*1e3, fh, bh); %#ok<AGROW>
        if own && adm(hi).d - d0 > 0.02
            % the realised sensitivity, reported only where it MEANS
            % something: within the parent's own front end, over a lever
            % change big enough to divide by.  On another front end the base
            % geometry has moved too and the ratio is not a slope.
            info.slope_mm_per_m = (fh - f)/(adm(hi).d - d0)*1e3;
            say_(opts.quiet, ['    seed %-2d probe  s %+.0f mm, R_M2 %.1f mm%s' ...
                 ', d %.3f m -> floor %+.2f mm  [law said %+.2f; realised ' ...
                 'slope %.1f mm/m vs %.1f]\n'], info.n_gated, adm(hi).s*1e3, ...
                 R2*1e3, own_(own), dh, fh*1e3, pred*1e3, ...
                 info.slope_mm_per_m, al2*1e3);
        else
            say_(opts.quiet, ['    seed %-2d probe  s %+.0f mm, R_M2 %.1f mm%s' ...
                 ', d %.3f m -> floor %+.2f mm\n'], info.n_gated, ...
                 adm(hi).s*1e3, R2*1e3, own_(own), dh, fh*1e3);
        end
        if fh < need
            say_(opts.quiet, ['            -> this front end cannot reach the ' ...
                 'wall even at full lever\n']);
            continue;
        end

        % bisect back toward the parent for the SMALLEST departure that clears
        lo = 1;   best = struct('i',hi, 'f',fh, 'b',bh, 'd',dh);
        while hi - lo > 1 && info.n_gated < opts.max_gate
            mid = floor((lo+hi)/2);
            [fm, bm, dm] = gate_(Q, setsr_(D,adm(mid).s,R2), tmp, F, opts.axis);
            info.n_gated = info.n_gated + 1;
            info.best_floor_m = max(info.best_floor_m, fm);
            tried(end+1) = rec_('bisect', adm(mid).s, R2, adm(mid).phi4, dm, ...
                                (f + al2*(adm(mid).d - d0))*1e3, fm, bm); %#ok<AGROW>
            say_(opts.quiet, ['    seed %-2d bisect s %+.0f mm, phi4 %+.3f /m, ' ...
                 'd %.3f m -> floor %+.2f mm\n'], info.n_gated, adm(mid).s*1e3, ...
                 adm(mid).phi4, dm, fm*1e3);
            if fm >= need
                hi = mid;   best = struct('i',mid, 'f',fm, 'b',bm, 'd',dm);
            else
                lo = mid;
            end
        end
        j = best.i;
        D = setsr_(D, adm(j).s, R2);
        info = finish_(info, tried, 're-pose', D, best.f, best.b, ...
                       adm(j).phi4, best.d, D0, t0);
        info.used_own_front_end = own;
        say_(opts.quiet, ['    -> compliant seed by re-posing: s %+.0f mm ' ...
             '(parent %+.0f), R_M2 %.1f mm%s, phi4 %+.3f /m, floor %+.2f mm\n'], ...
             adm(j).s*1e3, D0.fm_standoff*1e3, R2*1e3, own_(own), ...
             adm(j).phi4, best.f*1e3);
        return;
    end

    % ---- preference 4: the delivered cleared design, if there is one -----
    if ~strcmpi(fb,'none') && isfile(fb)
        Df = wall_recover(P, fb, 'verify',false);
        Df.tilt_deg = tilt_deg;   Df.ngrid = D0.ngrid;
        [ff, bf, dd] = gate_(Q, Df, tmp, F, opts.axis);
        info.n_gated = info.n_gated + 1;
        info.best_floor_m = max(info.best_floor_m, ff);
        tried(end+1) = rec_('delivered -10 deg DOFs', Df.fm_standoff, Df.R2, ...
                            NaN, dd, NaN, ff, bf); %#ok<AGROW>
        say_(opts.quiet, ['    seed %-2d delivered -10 deg DOFs at this tilt: ' ...
             's %+.0f mm, d %.3f m -> floor %+.2f mm\n'], info.n_gated, ...
             Df.fm_standoff*1e3, dd, ff*1e3);
        if ff >= need
            info = finish_(info, tried, 'delivered -10 deg DOFs', Df, ff, bf, ...
                           NaN, dd, D0, t0);
            say_(opts.quiet, ['    -> compliant only from the delivered ' ...
                 'design''s basin -- noted in the record\n']);
            D = Df;   return;
        end
    end

    info.tried = tried;   info.seconds = toc(t0);
    say_(opts.quiet, ['    -> NO COMPLIANT SEED at tilt %+.1f deg: best floor ' ...
         '%+.2f mm against a %+.1f mm need, over %d closures and %d gated ' ...
         'candidates\n'], tilt_deg, info.best_floor_m*1e3, need*1e3, ...
         info.n_closed, info.n_gated);
end

% =====================================================================
function [f, b, d] = gate_(P, D, deck, F, ax)
%GATE_  Build the swung design and read its union floor.  Both body models:
%   the DECLARED one is what the wall holds, and bare lit glass rides along
%   so a seed can never be reported without the number that says whether an
%   interference is the design's or the body model's.  D is the measured
%   field-mirror -> collimator spacing, i.e. the lever the tilt works on.
    Dq = D;   Dq.ngrid = P.solve.ngrid;
    f = -Inf;   b = -Inf;   d = NaN;
    try
        o = clear_build(P, Dq, deck, 'axis',ax, 'verify',false);
    catch
        return;                              % a wall upstream: not a candidate
    end
    d = o.C.t(3);
    W = afocal4_union_wall(P, deck, 'fields',F, 'throw',false, 'bare',true, ...
                           'quiet',true);
    f = W.floor_m;   b = W.bare_m;
end

function r = rec_(src, s, R2, phi4, d, pred, f, b)
    r = struct('source',src, 'standoff',s, 'R2',R2, 'phi4',phi4, 'd_m',d, ...
               'pred_mm',pred, 'floor_m',f, 'bare_m',b);
end

function info = finish_(info, tried, src, D, f, b, phi4, d, D0, t0)
    info.ok = true;      info.source = src;
    info.floor_m = f;    info.bare_m = b;    info.phi4 = phi4;
    info.standoff = D.fm_standoff;           info.R2 = D.R2;
    info.d_m = d;        info.tried = tried;
    info.used_own_front_end = (D.R2 == D0.R2);
    info.best_floor_m = max(info.best_floor_m, f);
    info.seconds = toc(t0);
end

function v = unique_stable_(v)
    [~, i] = unique(v, 'stable');   v = v(i);
end

function D = setsr_(D, s, R2)
    D.fm_standoff = s;   D.R2 = R2;
end

function U = wall_spec_(P)
    U = struct('min',0.0);
    if isfield(P,'pack') && isfield(P.pack,'union_min'), U.min = P.pack.union_min; end
end

function say_(quiet, varargin),  if ~quiet, fprintf(varargin{:}); end,  end
function s = own_(t),  if t, s = ' (parent''s)'; else, s = ''; end,  end
function del_(p),  if exist(p,'file'), delete(p); end,  end
