function out = descent_build(P, D, deck, opts)
%DESCENT_BUILD  One evaluable N-mirror afocal design: close it, emit it, swing it.
%
%   OUT = DESCENT_BUILD(P, D, DECK) is AFOCAL4_BUILD generalized to any
%   number of powered mirrors: the three first-order conditions are re-closed
%   EXACTLY at every iterate by DESCENT_CLOSE, the layout is emitted through
%   the same MACOS.DESIGN.TELESCOPE path, the interface plane is posed on the
%   TRACED exit chief, and the same walls turn the solver back.
%
%   WHY THE CLOSURE IS INNER AND NOT A MERIT TERM -- unchanged from the S4
%   ruling, and it is the reason this stage can climb to seven mirrors
%   without the specification drifting: recollimating at M = 30 and landing
%   the stop's image on the interface plane are not aspirations to be traded
%   against wavefront error, they are what makes the thing an afocal
%   telescope with an interface pupil.  Every design the solver ever sees has
%   them as IDENTITIES (residuals at 1e-12), never as penalties it has to buy.
%
%   EXTRACTION TILTS ARE IN THE DOF SET FROM THE START, per mirror.  That is
%   the wall slice's lesson, paid for once already: the committed 4-mirror
%   design missed 35 % of its own pupil blur because a field-mirror tilt was
%   never among {conic, standoff, front}, and the merit -- which would have
%   taken the move -- was never offered it.  A tilt is a PUPIL knob that
%   costs a little wavefront; the descent does not get to rediscover that.
%
%   TILTS ARE APPLIED TO THE EMITTED DECK, UPSTREAM FIRST.  CLEAR_TILT swings
%   one mirror about the point the CHIEF actually strikes it (engine truth,
%   read from the ray history) and re-poses everything downstream by the
%   rotation carrying the old outgoing chief onto the new one -- so the chief
%   path is preserved exactly and each swing composes with the ones before
%   it.  Applying them downstream-first would re-pose mirrors that have not
%   been swung yet and the composition would not be the same design.
%
%   THE WALLS, in the order they are cheap:
%     1  degenerate closure     any spacing < 20 mm             (algebra)
%     2  packaging             z(last powered) - z(M1) >= P.pack.m3_behind_min
%     3  union body-in-beam    AFOCAL4_UNION_WALL, DEFERRED past the tilts
%   Walls run on ITERATES and never on reports (the wall slice's rule), and
%   the union wall is the expensive one so it goes last and only if asked.
%
%   D fields:
%     .N        powered mirrors
%     .R        1 x (N-2) free radius magnitudes  |  .convex 1 x (N-2)
%     .t        1 x (N-2) free spacings           |  .K      1 x N conics
%     .tilt_deg 1 x N extraction tilts, deg (0 = untilted)
%     .iface    interface standoff, m
%     .ngrid    ray grid (default P.ngrid)
%     .bias_deg field-box bias in +Y, deg (default P.bias_deg)
%
%   Name-value:
%     'axis'          tilt axis, global ([1 0 0] -- the bias plane, which
%                     CLEAR_SCAN measured to be the cheaper of the two axes
%                     per mm of clearance won)
%     'names'         override the generic M1..MN element names.  Only the
%                     N = 4 identity check needs this: the committed decks
%                     call their mirrors M1/M2/FM/M3, and an EltName is the
%                     one thing in an emitted prescription that carries a
%                     FORM's vocabulary rather than a layout's.
%     'window'/'npts' the penultimate-power scan, handed to DESCENT_CLOSE.
%                     The N = 4 identity check passes AFOCAL4_PHI4's own
%                     window: FZERO lands on a bracket, so a different scan
%                     grid converges to a root 2e-16 away and the emitted
%                     KrElt differs in its last digit.  Same recipe, same
%                     bits -- and saying so is more honest than widening a
%                     tolerance until the check passes.
%     'defer_union'   skip the union wall here (false)
%     'oa_fields'     field points the OFF-AXIS aperture fit must span, rad
%                     (default: the deck's bias point alone).  Pass P.Fsolve
%                     for a design that will be QUOTED -- apertures fitted on
%                     one field vignette the corners of the box.
%     'verify'        re-trace and report M / collimation (true)
%     'quiet'         (true)
%
%   Returns .file .D .C (the closure) .R .t .conic .z .behind_m1 .tilt
%   .coldstop .union and, with 'verify', .traced and .paraxial_ok.
%
%   See also DESCENT_CLOSE, AFOCAL4_BUILD, CLEAR_TILT, AFOCAL4_UNION_WALL.

    arguments
        P (1,1) struct
        D (1,1) struct
        deck (1,:) char
        opts.axis        (1,3) double = [1 0 0]
        opts.names       (1,:) cell = {}
        opts.window      (1,2) double = [-1.5 9]
        opts.npts        (1,1) double = 241
        opts.defer_union (1,1) logical = false
        opts.oa_fields   (:,2) double  = []
        opts.verify      (1,1) logical = true
        opts.quiet       (1,1) logical = true
    end

    D = fill_(P, D);
    N = D.N;

    % ---- 1. first order, re-closed EXACTLY ------------------------------
    S = struct('N',N, 'R',D.R, 'convex',D.convex, 't',D.t, 'iface',D.iface, ...
               'K',D.K);
    C = descent_close(P, S, 'window',opts.window, 'npts',opts.npts);
    if ~isfield(C,'found') || ~C.found
        error('macos:design:descent_build:noRoot', ...
              ['no penultimate power closes the exit pupil at %.1f mm for ' ...
               'this %d-mirror front end.  The interface condition has no ' ...
               'root here -- pick a front end that can reach it.'], ...
              D.iface*1e3, N);
    end
    if ~C.closed
        error('macos:design:descent_build:notClosed', ...
              ['the closure did not close: residuals [%.2e %.2e %.2e] on ' ...
               '(u_out, M/30-1, pupil-iface).  A first-order identity that ' ...
               'is only nearly true is not an identity.'], C.residual);
    end
    % A closure can be arithmetically valid and not a telescope: two mirrors
    % on top of each other, or a beam running backwards.  Reject HERE so the
    % solver sees a wall rather than building and scoring nonsense.
    if any(C.t < 0.02)
        error('macos:design:descent_build:degenerate', ...
              ['the closure put a spacing at %.4f m (minimum 0.02): this ' ...
               'layout stacks two mirrors or runs the beam backwards.'], ...
              min(C.t));
    end
    % ---- 1b. PACKAGING: the last powered mirror sits BEHIND the primary ---
    pk = pack_spec_(P);
    if pk.enforce && C.behind_m1 < pk.m3_behind_min
        error('macos:design:descent_build:packaging', ...
              ['the closure puts %s at z = %+.3f m, %.0f mm behind M1 ' ...
               '(minimum %.0f): the back end and the instrument that ' ...
               'follows the pupil would sit in the incoming beam.'], ...
              C.names{end}, C.z(end), C.behind_m1*1e3, pk.m3_behind_min*1e3);
    end

    % ---- 2. emit ---------------------------------------------------------
    if ~isempty(opts.names)
        if numel(opts.names) ~= N
            error('macos:design:descent_build:names', ...
                  '''names'' must be 1x%d.', N);
        end
        C.names = opts.names;
    end
    t = macos.design.Telescope('family','tma', 'aperture_diameter_m',P.D, ...
            'wavelength_m',P.lambda, 'grid_npts',D.ngrid, ...
            'model_size',P.model_size);
    tt = [C.t, D.iface];
    for k = 1:N
        t.add_mirror(C.names{k}, 'radius_m',C.R(k), 'spacing_after_m',tt(k), ...
                     'convex',logical(C.convex(k)), 'conic',C.K(k));
    end
    t.add_exit_reference('ColdStop', 'dist_m', D.iface);
    if D.bias_deg ~= 0, t.set_field_bias(D.bias_deg*60); end
    t.build(deck);

    % ---- 2b. OFF-AXIS: displace the pupil, re-fit the apertures ----------
    % Ordered here for a reason.  It must follow the emit (there is no deck
    % before it) and PRECEDE the interface pose and the tilts, both of which
    % read the TRACED chief: posing the interface on the coaxial chief and
    % then decentering would leave the plane off the beam it is meant to
    % receive.  A decenter of zero does nothing, not even a file read.
    out_oa = [];
    if D.decenter ~= 0
        out_oa = offaxis_decenter(deck, D.decenter, 'fields',opts.oa_fields, ...
                                  'quiet',opts.quiet);
    end

    % ---- 3. the interface pose, then the tilts ---------------------------
    cs  = place_coldstop_(deck, D.iface, N);
    out = struct('file',deck, 'D',D, 'C',C, 'R',C.R, 't',C.t, 'conic',C.K, ...
                 'names',{C.names}, 'iface',D.iface, 'z',C.z, ...
                 'behind_m1',C.behind_m1, 'coldstop',cs, 'tilt',[], ...
                 'union',[], 'traced',[], 'paraxial_ok',true, ...
                 'offaxis',out_oa);

    % ---- 3b. the EMITTED stations must be the ones the wall judged --------
    zi = grab_all3_(fileread(deck), 'VptElt');
    dz = max(abs(zi(3,1:numel(C.z)) - C.z));
    if dz > 1e-9
        error('macos:design:descent_build:stations', ...
              ['the emitted vertex stations differ from the closure''s by ' ...
               '%.3e m: the packaging check judged a layout that was not ' ...
               'built.'], dz);
    end

    % ---- 3c. the extraction tilts, UPSTREAM FIRST ------------------------
    tl = D.tilt_deg(:).';
    if any(tl ~= 0)
        rec = cell(1, N);
        for k = 1:N
            if tl(k) == 0, continue; end
            tmp = [tempname '.in'];
            cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>
            copyfile(deck, tmp);
            rec{k} = clear_tilt(tmp, struct('elt',k, ...
                        'alpha',deg2rad(tl(k)), 'axis',opts.axis), deck);
            clear cu;
        end
        out.tilt = rec;
    end

    % ---- 3d. UNION: no BODY may stand in a BEAM --------------------------
    % Deferred past the tilts by construction -- applied before them it would
    % judge the untilted train, i.e. exactly the design the tilts exist to
    % get away from.  That is the wall slice's cage lesson, and it is why
    % this call sits here and not beside the packaging wall above.
    if ~opts.defer_union
        out.union = afocal4_union_wall(P, deck, 'quiet',opts.quiet);
    end

    % ---- 4. verify against the closure it was built from -----------------
    if opts.verify
        out.traced = traced_(deck, P.D);
        out.paraxial_ok = abs(out.traced.mag/P.M - 1) < 0.05;
        if ~opts.quiet
            fprintf(['  built %s: N %d, phi [%s] /m, iface %.1f mm\n'], deck, ...
                    N, strjoin(arrayfun(@(x) sprintf('%+.3f',x), C.phi, ...
                    'UniformOutput',false), ' '), D.iface*1e3);
            fprintf(['        traced M %.4fx (paraxial %.4f), exit %.3f mm, ' ...
                     'collimation %.2f urad\n'], out.traced.mag, C.fo.mag, ...
                    out.traced.exit_dia*1e3, out.traced.collimation_urad);
            fprintf('        %s at z %+.3f m = %.0f mm behind M1\n', ...
                    C.names{end}, C.z(end), C.behind_m1*1e3);
        end
    end
end

% =====================================================================
function D = fill_(P, D)
    if ~isfield(D,'N'),        error('macos:design:descent_build:N','D.N required.'); end
    if ~isfield(D,'iface'),    D.iface = P.iface;    end
    if ~isfield(D,'bias_deg'), D.bias_deg = P.bias_deg; end
    if ~isfield(D,'ngrid'),    D.ngrid = P.ngrid;    end
    if ~isfield(D,'tilt_deg') || isempty(D.tilt_deg), D.tilt_deg = zeros(1,D.N); end
    % .decenter: pupil displacement off the parent axis, m.  ZERO is the
    % coaxial train this stage was written for and takes the IDENTICAL code
    % path -- no deck edit is attempted at all -- so every committed descent
    % result is unaffected, and the N = 4 byte-identity check guards it.
    if ~isfield(D,'decenter') || isempty(D.decenter), D.decenter = 0; end
    if numel(D.tilt_deg) ~= D.N
        error('macos:design:descent_build:tilt', ...
              'D.tilt_deg must be 1x%d (one per powered mirror).', D.N);
    end
end

function pk = pack_spec_(P)
%PACK_SPEC_  As AFOCAL4_BUILD's, with the pre-constraint behaviour as the
%   default for a P that predates it.
    pk = struct('enforce',false, 'm3_behind_min',0.500);
    if isfield(P,'pack')
        f = fieldnames(P.pack);
        for i = 1:numel(f)
            if isfield(pk, f{i}), pk.(f{i}) = P.pack.(f{i}); end
        end
    end
end

function cs = place_coldstop_(deck, iface, nmir)
%PLACE_COLDSTOP_  Put the interface plane on the TRACED exit chief, IFACE
%   past the LAST MIRROR, normal to that chief.  Ported from AFOCAL4_BUILD,
%   whose own comment records why: a terminal resolved on the folded
%   paraxial axis is exact only for an unbiased coaxial train, and at a
%   0.6 deg bias the exit chief leaves at 30x that, putting a 1/cos frame
%   term into every footprint read on the plane.  NMIR is already a
%   parameter there, so the construction is N-generic as written.
    txt = fileread(deck);
    Vs  = grab_all3_(txt, 'VptElt');
    nE  = size(Vs, 2);
    macos.load_rx(deck);
    if ~macos.has_rx()
        error('macos:design:descent_build:load', 'deck failed to load: %s', deck);
    end
    macos.ray_hist('on');
    tr = macos.trace(nE);
    h  = macos.ray_hist(tr.nRays);
    macos.ray_hist('off');
    Pc = squeeze(h.P(:,1,:));              % chief polyline; Pc(:,k+1) = elt k
    pm = Pc(:, nmir+1);                    % chief hit on the last mirror
    dm = Pc(:, nmir+2) - pm;   dm = dm/norm(dm);
    V  = pm + dm*(dot(Vs(:,nmir) - pm, dm) + iface);
    old = Vs(:, nE);
    txt = set_elt_pose_(txt, nE, -dm, V);
    write_(deck, txt);
    cs = struct('Vpt',V.', 'psi',(-dm).', 'chief',dm.', ...
                'dist_m',iface, 'moved_m',norm(V - old));
end

function txt = set_elt_pose_(txt, k, psi, Vpt)
%SET_ELT_POSE_  Rewrite element K's psiElt / VptElt / RptElt and its TElt
%   frame.  Line-based, not regex-over-the-file: the element blocks are
%   delimited by `iElt=` and a global regexp would hit every element at once.
    psi = psi(:)/norm(psi);   Vpt = Vpt(:);
    L   = strsplit(txt, newline, 'CollapseDelimiters', false);
    ie  = find(~cellfun('isempty', regexp(L, '^\s*iElt=', 'once')));
    lo  = ie(k);
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
    z = psi(:)/norm(psi);
    yh = [0;1;0];   if abs(z(2)) > 0.95, yh = [1;0;0]; end
    y  = yh - (yh.'*z)*z;   y = y/norm(y);
    x  = cross(y, z);
    R  = [x, y, z];
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

function M = grab_all3_(txt, key)
    t = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens');
    M = zeros(3, numel(t));
    for i = 1:numel(t), M(:,i) = sscanf(strrep(t{i}{1},'D','E'), '%f', 3); end
end

function write_(f, txt)
    fid = fopen(f,'w');   fprintf(fid,'%s',txt);   fclose(fid);
end

function del_(p),  if exist(p,'file'), delete(p); end,  end
