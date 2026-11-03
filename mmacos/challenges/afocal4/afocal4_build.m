function out = afocal4_build(P, D, deck, opts)
%AFOCAL4_BUILD  One evaluable afocal4 design: close it, emit it, pose it.
%
%   out = AFOCAL4_BUILD(P, D, DECK) turns a design struct D into a committed
%   MACOS prescription at DECK and returns what was built.  This is the
%   INNER LOOP of the S4 joint solve (PLAN_AFOCAL4 S4): the outer solver
%   moves conics, the field-mirror standoff and rigid bodies; everything
%   first-order is RE-CLOSED here, exactly, at every iterate.
%
%   WHY THE CLOSURE IS INNER AND NOT A MERIT TERM.  The two first-order
%   conditions -- recollimate at M = 30, and land the image of the stop on
%   the interface plane -- are not aspirations to be traded against wavefront
%   error; they are what makes the thing an afocal telescope with an
%   interface pupil.  A mirror AT the intermediate image cannot change them
%   for any power (S3 FORM_STUDY section 2), but the standoff that gives it a
%   footprint breaks that exactness, and so does every radius.  So each
%   iterate re-derives the collimator's power and station from the marginal
%   ray and the field mirror's power from the chief ray, and M = 30.000 and
%   collimation and the pupil station are IDENTITIES of every design the
%   solver ever sees -- never penalties it has to buy.
%
%   The consequence, and it is the S4 ruling: the field mirror's POWER IS NOT
%   AN OUTER DOF.  The closure consumes it to hold the pupil station at
%   D.iface.  "phi4, equivalently P.iface" (the brief) makes them one knob,
%   and the ruling carries that knob as a PARAMETER -- the operating point --
%   with the trade curve reported rather than a value optimised.  Sweeping
%   D.iface IS sweeping phi4; afocal4_ladder('trade') does exactly that.
%
%   AND ONE THING IS NOT NEGOTIABLE AT ALL: the design has to be BUILDABLE.
%   The S4b constraint (Dave, 2026-08-03) says the last powered mirror sits
%   at least P.pack.m3_behind_min BEHIND the primary, so that a fold on its
%   exit leg can take the interface pupil and the instrument out of the
%   incoming beam -- which is what Mike's own parent does.  It enters here
%   as a WALL (an error the solver turns back on), never as a merit term:
%   an unbuildable layout is not a worse design to be traded against
%   wavefront error, it is not a design.  P.pack.enforce = false reproduces
%   the unconstrained S4 reference.
%
%   D fields (AFOCAL4_SEED builds a default one):
%     .form         'field' (the S3 ruling) | 'mersenne' (the hedge)
%     .K            1x4 conics in BUILD order.  'field': [M1 M2 FM M3] with
%                   M1 HELD at -1 (a parabola, as in his study); 'mersenne':
%                   [M1 M2 M3 M4], all four relaxed.
%     .iface        interface standoff, m (the operating point)
%     .fm_standoff  'field': field-mirror standoff before the intermediate
%                   image, m
%     .gap          'mersenne': inter-stage spacing, m
%     .rb           numel(P.rb_elts) x 2 rigid body, [y-decenter m, x-tilt
%                   rad], applied to the elements P.rb_elts names
%     .fold         optional struct('dist',m,'to',[1x3],'ap_r',m): insert a
%                   flat fold that far along the LAST mirror's exit leg,
%                   turning the beam into 'to'.  The S4b packaging
%                   demonstration; omit for the unfolded deck
%     .bias_deg     field-box bias in +Y, deg
%     .ngrid        ray grid (default P.ngrid)
%
%   THE INTERFACE PLANE IS POSED ON THE TRACED CHIEF, ALWAYS.  A terminal
%   resolved on the folded paraxial axis is exact only for an unbiased
%   coaxial train; at a 0.6 deg bias the exit chief leaves at 30x that, and a
%   plane normal to the axis is oblique to the beam it interfaces -- which
%   puts a 1/cos frame term into every footprint read on it (rodgers2 PACKET
%   section 4).  Telescope/align_exit_reference does this for an unperturbed
%   build; a rigid-bodied one is no longer a Telescope, so the pose is
%   recomputed here from the trace, by the same rule, on ONE code path for
%   both.  TAFOCAL4 pins the two against each other at 1e-12.
%
%   AND A SECOND WALL, OFF BY DEFAULT: no BODY may stand in a BEAM.  The
%   clearing stage (BRIEF_afocal4_clear) found the collimator of the
%   committed 343 mm design inside its own M2 -> field-mirror feed cone, and
%   then measured that a re-solve SPENDS whatever clearance a remedy wins,
%   because the merit cannot see it.  P.pack.union_enforce turns the union
%   body-in-beam floor into a wall exactly like the packaging one above --
%   see AFOCAL4_UNION_WALL for the cost it carries and why it is default
%   OFF (with it on, this function could not re-emit the deck the S4b/S4c
%   trade shipped, which reads -79.9 mm).
%
%   Name-value:
%     'verify'  trace the emitted deck and report M / collimation (true)
%     'defer_union'  skip the union wall here because the CALLER is about to
%               move the deck and will apply it afterwards (false).
%               CLEAR_BUILD sets it: it tilts the emitted deck after this
%               function returns, so a wall applied here would judge the
%               untilted train -- i.e. exactly the design the tilt exists
%               to get away from.
%     'quiet'   (true)
%
%   Returns .file .D .C (the closure) .phi4 .R .t .conic .iface_pred
%   .coldstop (.Vpt .psi .chief .moved_m) and, with 'verify', .traced
%   (.mag .exit_dia .collimation_urad .nrays) and .paraxial_ok.
%
%   See also AFOCAL4_CLOSE, AFOCAL4_SCORE, AFOCAL4_SOLVE, AFOCAL4_LADDER,
%   AFOCAL4_UNION_WALL.

    arguments
        P (1,1) struct
        D (1,1) struct
        deck (1,:) char
        opts.verify (1,1) logical = true
        opts.defer_union (1,1) logical = false
        opts.quiet  (1,1) logical = true
    end

    D = fill_(P, D);

    % ---- 1. first order, re-closed --------------------------------------
    switch D.form
    case 'field'
        % The FRONT END is a DOF group, not a constant: D.R2 / D.t1 are M2's
        % radius and the M1-M2 spacing, and everything downstream -- the
        % intermediate image, the field mirror's power, the collimator's
        % radius and station -- is re-derived from them by the closure.  His
        % own rung 3 re-optimises "conics and radii"; holding the whole front
        % end would be answering a smaller question than he asked.
        Q = P;
        Q.parent.R(2) = D.R2;   Q.parent.t(1) = D.t1;
        [C, phi4] = close_field_(Q, D.fm_standoff, D.iface);
    case 'mersenne'
        % Confocal SPACINGS held -- that is where the pupil relay lives, and
        % relaxing them would make this a different form rather than the
        % same form with its conics freed.  Nothing to re-close: each pair is
        % afocal by construction and a conic is a higher-order term, so M and
        % the afocal condition are identities for every member.
        C = afocal4_close(P, 'mersenne', 'm1',P.mersenne.m1, ...
                          'stage2_fnum',P.mersenne.stage2_fnum, ...
                          'stage2_type',P.mersenne.stage2_type, 'gap',D.gap);
        phi4 = NaN;
        D.iface = C.fo.pupil_dist;
    otherwise
        error('macos:design:afocal4_build:form', 'unknown form "%s".', D.form);
    end
    C.K = D.K(:).';
    % A closure can return an arithmetically valid layout that is not a
    % telescope: a collimator behind the mirror that feeds it, or two
    % surfaces on top of each other.  Reject it HERE so the solver sees a
    % wall rather than building and scoring nonsense.
    if any(C.t < 0.02)
        error('macos:design:afocal4_build:degenerate', ...
              ['the closure put a spacing at %.4f m (minimum 0.02): this ' ...
               'layout stacks two mirrors or runs the beam backwards.'], ...
              min(C.t));
    end
    % ---- 1b. PACKAGING: the last mirror must sit BEHIND the primary -------
    % The S4b constraint (Dave 2026-08-03; PLAN_AFOCAL4 BUILDABILITY
    % CONSTRAINT), and it is a WALL, not a merit term -- see the P.pack
    % comment in AFOCAL4_PARAMS for why.  An arithmetically valid closure
    % that puts the collimator, the field mirror and the interface pupil in
    % front of M1 is not a worse telescope; it is a telescope whose
    % instrument sits in its own incoming beam.
    pk = pack_spec_(P);
    if pk.enforce && C.behind_m1 < pk.m3_behind_min
        error('macos:design:afocal4_build:packaging', ...
              ['the closure puts %s at z = %+.3f m, %.0f mm behind M1 ' ...
               '(minimum %.0f): the back end and the instrument that ' ...
               'follows the pupil would sit in the incoming beam.'], ...
              C.names{end}, C.z(end), C.behind_m1*1e3, pk.m3_behind_min*1e3);
    end

    % ---- 2. emit ---------------------------------------------------------
    t = macos.design.Telescope('family','tma', 'aperture_diameter_m',P.D, ...
            'wavelength_m',P.lambda, 'grid_npts',D.ngrid, ...
            'model_size',P.model_size);
    tt = [C.t, D.iface];
    for k = 1:numel(C.R)
        t.add_mirror(C.names{k}, 'radius_m',C.R(k), 'spacing_after_m',tt(k), ...
                     'convex',logical(C.convex(k)), 'conic',C.K(k));
    end
    % THE FOLD, when there is one.  A flat on the last mirror's exit leg is
    % the S4b demonstration (task 3): it takes the interface pupil and the
    % instrument out of the beam and into the x-y plane behind the primary,
    % which is what "buildable" means here.  Telescope/add_fold maps
    % everything downstream by the fold plane's reflection isometry, so it
    % is EXACT -- spacings, angles and the trace are preserved and the flat
    % contributes zero aberration.  Nominally null is not the same as free,
    % though: the pupil ladder is re-scored on the folded deck rather than
    % asserted (e2e2 S3 lesson).
    if isfield(D,'fold') && ~isempty(D.fold)
        t.add_fold('Fold', 'after', C.names{end}, 'dist_m', D.fold.dist, ...
                   'to', D.fold.to, 'ap_r', D.fold.ap_r);
    end
    t.add_exit_reference('ColdStop', 'dist_m', D.iface);
    if D.bias_deg ~= 0, t.set_field_bias(D.bias_deg*60); end
    t.build(deck);

    % ---- 3. rigid bodies, then the interface pose ------------------------
    txt = fileread(deck);
    if any(D.rb(:) ~= 0)
        for i = 1:numel(P.rb_elts)
            txt = rigid_body_(txt, P.rb_elts(i), D.rb(i,1), D.rb(i,2));
        end
        write_(deck, txt);
    end
    % The interface plane is posed IFACE past the last mirror ALONG THE
    % CHIEF's own path, so a fold in between costs exactly the path it
    % adds and nothing else.
    cs = place_coldstop_(deck, D.iface, numel(C.R));
    out = struct('file',deck, 'D',D, 'C',C, 'phi4',phi4, ...
                 'R',C.R, 't',C.t, 'conic',C.K, 'names',{C.names}, ...
                 'iface',D.iface, 'iface_pred',C.fo.pupil_dist, ...
                 'z',C.z, 'behind_m1',C.behind_m1, ...
                 'coldstop',cs, 'traced',[], 'paraxial_ok',true);

    % ---- 3b. the EMITTED stations must be the ones the wall judged --------
    % The packaging wall is applied to the closure's predicted stations,
    % because it has to be cheap enough to sit inside the solve.  That is
    % only sound if the emitter puts the mirrors where the closure says --
    % so read them back off the committed deck and pin the two together.  A
    % constraint checked against a prediction the artifact does not honour
    % is the "closes on paper" failure the S3 builder bug already cost this
    % study once.
    zi = grab_all3_(fileread(deck),'VptElt');
    dz = max(abs(zi(3,1:numel(C.z)) - C.z));
    if dz > 1e-9
        error('macos:design:afocal4_build:stations', ...
              ['the emitted vertex stations differ from the closure''s by ' ...
               '%.3e m: the packaging check judged a layout that was not ' ...
               'built.'], dz);
    end

    % ---- 3c. UNION: no BODY may stand in a BEAM ---------------------------
    % The clearing stage's wall, and the LAST thing checked because it is by
    % far the most expensive (a nine-field re-trace, +51% on an evaluation)
    % -- a degenerate closure or an unbuildable station must be turned back
    % on by the cheap algebra above before a ray is traced for this one.
    % Default OFF: with it on, AFOCAL4_BUILD cannot re-emit the committed
    % 343 mm deck (-79.9 mm), which is the design the S4b/S4c trade shipped.
    if ~opts.defer_union
        out.union = afocal4_union_wall(P, deck, 'quiet',opts.quiet);
    else
        out.union = [];
    end

    % ---- 4. verify against the closure it was built from -----------------
    if opts.verify
        out.traced = traced_(size(zi,2), P.D);
        out.paraxial_ok = abs(out.traced.mag/P.M - 1) < 0.05;
        if ~opts.quiet
            fprintf(['  built %s: phi4 %+.4f /m, R_FM %.4f m, R_col %.4f m, ' ...
                     'iface %.1f mm\n'], deck, phi4, C.R(3), C.R(4), D.iface*1e3);
            fprintf(['        traced M %.4fx (paraxial %.4f), exit %.3f mm, ' ...
                     'collimation %.2f urad\n'], out.traced.mag, C.fo.mag, ...
                     out.traced.exit_dia*1e3, out.traced.collimation_urad);
            fprintf('        %s at z %+.3f m = %.0f mm behind M1\n', ...
                    C.names{end}, C.z(end), C.behind_m1*1e3);
        end
    end
end

% =====================================================================
function pk = pack_spec_(P)
%PACK_SPEC_  The packaging spec, with the S4 (pre-constraint) behaviour as
%   the default for a P that predates it.  A parameter struct saved inside
%   an older afocal4_ladder.mat must still rebuild its own decks -- and it
%   must rebuild them UNCONSTRAINED, because that is what it was scored
%   under.  Silently applying today's wall to yesterday's record would make
%   the S4 reference unreproducible, which is the opposite of retract-in-
%   place.
    pk = struct('enforce',false, 'm3_behind_min',0.500);
    if isfield(P,'pack')
        f = fieldnames(P.pack);
        for i = 1:numel(f), pk.(f{i}) = P.pack.(f{i}); end
    end
end

function D = fill_(P, D)
%FILL_  Defaults, and the shape checks that catch a mis-packed DOF vector
%   before it becomes a silently wrong design.
    if ~isfield(D,'form'),        D.form = P.form;              end
    if ~isfield(D,'iface'),       D.iface = P.iface;            end
    if ~isfield(D,'fm_standoff'), D.fm_standoff = P.fm_standoff;end
    if ~isfield(D,'gap'),         D.gap = P.mersenne.gap;       end
    if ~isfield(D,'R2'),          D.R2 = P.parent.R(2);         end
    if ~isfield(D,'t1'),          D.t1 = P.parent.t(1);         end
    if ~isfield(D,'bias_deg'),    D.bias_deg = P.bias_deg;      end
    if ~isfield(D,'ngrid'),       D.ngrid = P.ngrid;            end
    if ~isfield(D,'K') || numel(D.K) ~= 4
        error('macos:design:afocal4_build:K', 'D.K must be 1x4 (build order).');
    end
    nrb = numel(P.rb_elts);
    if ~isfield(D,'rb') || isempty(D.rb), D.rb = zeros(nrb,2); end
    if ~isequal(size(D.rb), [nrb 2])
        error('macos:design:afocal4_build:rb', ...
              'D.rb must be %dx2 ([decenter_m tilt_rad] per P.rb_elts).', nrb);
    end
end

function [C, phi4] = close_field_(P, s, iface)
%CLOSE_FIELD_  The field mirror's power that puts the exit pupil at IFACE,
%   at field-mirror standoff S -- and then the whole closed layout.
%
%   The root selection (and why a sign change is not a root) lives in
%   AFOCAL4_PHI4, so the SEEDER uses the same rule the builder does.  A
%   seeder that picked roots differently would hand the solver designs the
%   builder rejects.
    [phi4, C, found] = afocal4_phi4(P, s, iface);
    if ~found
        error('macos:design:afocal4_build:noRoot', ...
              ['no field-mirror power in [-0.9, 9] /m puts the exit pupil at ' ...
               '%.1f mm with a %.0f mm standoff and closes a real layout.  ' ...
               'Pick an interface distance the standoff can reach.'], ...
              iface*1e3, s*1e3);
    end
end

% ---------------------------------------------------------------------
%  Deck surgery.  The emitted prescription is the single artifact every
%  scorer reads, so a rigid body is applied to IT and not to a MATLAB-side
%  shadow that could drift out of step with what was committed.
% ---------------------------------------------------------------------
function txt = rigid_body_(txt, k, dy, alpha)
%RIGID_BODY_  Decenter element K by DY in +y, then tilt it by ALPHA about
%   its (decentered) vertex, x axis.  Decenter first, tilt second: CODE V's
%   YDE-then-ADE order, which is what the rodgers2 transcription decodes and
%   therefore what "his tilt/dec" means when the two are compared.
    [psi, Vpt] = elt_pose_(txt, k);
    Rx  = [1 0 0; 0 cos(alpha) -sin(alpha); 0 sin(alpha) cos(alpha)];
    psi = Rx*psi(:);
    Vpt = Vpt(:) + [0; dy; 0];
    txt = set_elt_pose_(txt, k, psi, Vpt);
end

function cs = place_coldstop_(deck, iface, nmir)
%PLACE_COLDSTOP_  Put the interface plane on the TRACED exit chief, IFACE
%   past the LAST MIRROR, normal to that chief -- and then map it through
%   any flat between the two.  Idempotent: a flat Reference does not bend
%   rays, so the chief line it is placed from does not depend on where it
%   currently is.
%
%   THE FOLD IS APPLIED AS AN ISOMETRY, NOT RE-DERIVED.  The unfolded rule
%   is "project the last mirror's vertex onto the exit chief, step IFACE
%   along it"; with a fold in between, the plane is that same plane
%   REFLECTED through the fold, which is exactly what a flat does to
%   everything downstream of it.  Re-deriving it from the fold's own vertex
%   instead is wrong by the vertex-to-chief offset -- measured, at a 343 mm
%   standoff that moved the pupil breathing from 3.9% to 23% across a fold
%   that cannot change anything.  The fold-is-null check in AFOCAL4_S4B
%   caught it, which is what that check is for.
%
%   With no fold the loop below is empty and the construction is the
%   original one to the last digit, so every committed deck re-emits
%   unchanged.
    txt = fileread(deck);
    Vs  = grab_all3_(txt, 'VptElt');
    Ps  = grab_all3_(txt, 'psiElt');
    nE  = size(Vs, 2);
    macos.load_rx(deck);
    if ~macos.has_rx()
        error('macos:design:afocal4_build:load', 'deck failed to load: %s', deck);
    end
    macos.ray_hist('on');
    tr = macos.trace(nE);
    h  = macos.ray_hist(tr.nRays);
    macos.ray_hist('off');
    Pc = squeeze(h.P(:,1,:));              % chief polyline; Pc(:,k+1) = elt k
    pm = Pc(:, nmir+1);                    % chief hit on the last mirror
    dm = Pc(:, nmir+2) - pm;   dm = dm/norm(dm);
    V  = pm + dm*(dot(Vs(:,nmir) - pm, dm) + iface);
    d  = dm;
    for k = nmir+1 : nE-1                  % every flat in between
        nf = Ps(:,k)/norm(Ps(:,k));
        Mr = eye(3) - 2*(nf*nf.');
        V  = Vs(:,k) + Mr*(V - Vs(:,k));
        d  = Mr*d;
    end
    old = Vs(:, nE);
    txt = set_elt_pose_(txt, nE, -d, V);
    write_(deck, txt);
    cs = struct('Vpt',V.', 'psi',(-d).', 'chief',d.', ...
                'dist_m',iface, 'moved_m',norm(V - old));
end

function [psi, Vpt] = elt_pose_(txt, k)
    Ps  = grab_all3_(txt,'psiElt');   Vs = grab_all3_(txt,'VptElt');
    psi = Ps(:,k);   Vpt = Vs(:,k);
end

function txt = set_elt_pose_(txt, k, psi, Vpt)
%SET_ELT_POSE_  Rewrite element K's psiElt / VptElt / RptElt and its TElt
%   frame block.  Line-based rather than regex-over-the-whole-file: the
%   element blocks are delimited by `iElt=` and a global regexp would hit
%   every element at once.
    psi = psi(:)/norm(psi);   Vpt = Vpt(:);
    L   = strsplit(txt, newline, 'CollapseDelimiters', false);
    ie  = find(~cellfun('isempty', regexp(L, '^\s*iElt=', 'once')));
    if k > numel(ie)
        error('macos:design:afocal4_build:elt', ...
              'element %d requested, deck has %d.', k, numel(ie));
    end
    lo = ie(k);
    if k < numel(ie), hi = ie(k+1) - 1; else, hi = numel(L); end
    R  = frame_(psi);
    v3 = @(a) sprintf('%.16E  %.16E  %.16E', a(1), a(2), a(3));
    v6 = @(u,w) sprintf('%.16E  %.16E  %.16E  %.16E  %.16E  %.16E', ...
                        u(1),u(2),u(3),w(1),w(2),w(3));
    for i = lo:hi
        s = L{i};
        if     ~isempty(regexp(s,'^\s*psiElt=','once'))
            L{i} = ['           psiElt=  ' v3(psi)];
        elseif ~isempty(regexp(s,'^\s*VptElt=','once'))
            L{i} = ['           VptElt=  ' v3(Vpt)];
        elseif ~isempty(regexp(s,'^\s*RptElt=','once'))
            L{i} = ['           RptElt=  ' v3(Vpt)];
        elseif ~isempty(regexp(s,'^\s*TElt=','once'))
            L{i}   = ['             TElt=  ' v6(R(:,1),[0;0;0])];
            L{i+1} = ['                    ' v6(R(:,2),[0;0;0])];
            L{i+2} = ['                    ' v6(R(:,3),[0;0;0])];
            L{i+3} = ['                    ' v6([0;0;0],R(:,1))];
            L{i+4} = ['                    ' v6([0;0;0],R(:,2))];
            L{i+5} = ['                    ' v6([0;0;0],R(:,3))];
        end
    end
    txt = strjoin(L, newline);
end

function R = frame_(psi)
%FRAME_  Element TElt frame: z along psi, x/y tangent.  Trace-neutral, and
%   the SAME construction Telescope/surf_frame_ and rodgers2_deck/frame_ use,
%   so a deck this function edits stays consistent with one it did not.
    z = psi(:)/norm(psi);
    yh = [0;1;0];   if abs(z(2)) > 0.95, yh = [1;0;0]; end
    y  = yh - (yh.'*z)*z;   y = y/norm(y);
    x  = cross(y, z);
    R  = [x, y, z];
end

function s = traced_(nE, Dap)
%TRACED_  Exit beam and collimation of the deck currently loaded.
    tr = macos.trace(nE);   ri = macos.get_ray_info(tr.nRays);
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

function M = grab_all3_(txt, key)
%   ^-anchored: an unanchored 'iElt=' matches inside 'psiElt=' too.
    t = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens');
    M = zeros(3, numel(t));
    for i = 1:numel(t), M(:,i) = sscanf(strrep(t{i}{1},'D','E'), '%f', 3); end
end

function write_(f, txt)
    fid = fopen(f,'w');   fprintf(fid,'%s',txt);   fclose(fid);
end
