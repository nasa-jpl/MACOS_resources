function pf = pupil_find(rx, Ffield, opts)
%PUPIL_FIND  Locate the exit pupil as a cone-convergence SURFACE and place its
%   best-fit reference sphere into the engine's internal Rx -- FEX generalized
%   from one chief ray to a field-cone bundle.
%
%   pf = PUPIL_FIND(RX, FFIELD) runs the cone-convergence pupil model
%   (design/src/pupil_map) on the prescription file RX over the field set
%   FFIELD (K x 2 rad, box-relative, K>=3), fits the exit-pupil reference
%   SPHERE to the ray-crossing cloud, and -- exactly as the engine's FEX does
%   -- WRITES that sphere into the loaded Rx via macos.set_xp (VptElt/psiElt/
%   KrElt at the exit-pupil element).  It returns the pupil metrics; it does
%   NOT touch the filesystem, print, or plot.  The template wrapper pupil_id.m
%   adds the report, the figures, and the revised-Rx write; sensitivity
%   drivers (run_dwd*.m) call PUPIL_FIND directly and continue with the
%   improved XP in the internal Rx.
%
%   THE TWO XP RADII (kept distinct, like FEX):
%     - the reference-sphere radius WRITTEN to the Rx is the chief-ray
%       distance from the WRITTEN vertex to the next plane (the far-field
%       propagation target).  For FEX's own vertex that is FEX's radius;
%       when 'fit_chief' slides the vertex t along the chief, the radius
%       follows (fex.rad - t) so the sphere CENTER -- which sits on the
%       propagation target plane -- is invariant.
%     - the cone-crossing sag curvature (pf.conv_radius) is the curvature of
%       the pupil-imaging convergence surface -- a QUALITY diagnostic, a
%       different quantity, NOT written to the Rx.
%
%   On return the engine has RX loaded, the stop set at ep_elt, and the
%   exit-pupil element carrying the bundle-fit sphere (get_xp reads it back).
%
%   Name-value:
%     'ep_elt'     entrance-pupil (stop) element (default 1).  Only used
%                  to SET a stop when neither 'stop_elt' is given nor the
%                  deck declares its own ApStop= -- a deck-declared stop
%                  (including the object-space header 3-vector, the
%                  segmented-primary idiom) is left in force.
%     'xp_elt'     exit-pupil Return/Reference element (default nElt-1); the
%                  element PUPIL_FIND writes, must be Return(8)/Reference(3).
%     'anchor'     'rim' (beam rim, default) | 'surface' | 'stop' (see pupil_map).
%     'nodes'      entrance-surface node lattice (default 21).
%     'min_fields' fewest cone rays to score a node (default 3).
%     'model_size' engine grid (default 256).
%     'init'       call macos.init first (default false -- the caller owns init).
%     'place'      apply set_xp to the internal Rx (default true).  false =
%                  measure only, leave the Rx untouched (still returns metrics).
%     'vertex'     written-vertex mode: 'fit_chief' (default; the fitted
%                  convergence surface's crossing with the chief ray --
%                  on-chief, carries the measured pupil station),
%                  'bundle' (raw fit vertex), 'chief' (FEX crossing).
%
%   Returns pf with fields:
%     .ep_elt .xp_elt .nElt        the resolved elements
%     .fex        FEX baseline .vpt/.rad (the one-chief-ray sphere beaten)
%     .vtx                        the BUNDLE fit vertex (pupil-wander diagnostic)
%     .vtx_written .rad .psi      the sphere WRITTEN ('vertex' option: 'bundle'
%                                 default = .vtx, or 'chief' = FEX's crossing;
%                                 FEX radius; FEX normal -- the fitted cone
%                                 normal is only the fit frame, see the
%                                 sign/normal note below)
%     .conv_radius                convergence-surface curvature (quality, not Rx)
%     .dep_rms                    departure-from-sphere RMS (frame terms removed)
%     .uv .dep                    per-node (u,v) and departure (for a map)
%     .blur .surface .map .wander the pupil_map four-part ladder (rim anchor)
%     .o_rim                      the full pupil_map struct (cloud .X, frame)
%     .placed                     whether set_xp was applied
%
%   See also: PUPIL_MAP, MACOS.FEX, MACOS.SET_XP, MACOS.PUPIL_QUALITY.
    arguments
        rx (1,:) char
        Ffield (:,2) double
        opts.ep_elt      (1,1) double = 1
        opts.xp_elt      (1,1) double = 0        % 0 -> nElt-1
        opts.anchor      (1,:) char {mustBeMember(opts.anchor,{'rim','surface','stop'})} = 'rim'
        opts.nodes       (1,1) double = 21
        opts.min_fields  (1,1) double = 3
        opts.model_size  (1,1) double = 256
        opts.init        (1,1) logical = false
        opts.place       (1,1) logical = true
        opts.vertex      (1,:) char {mustBeMember(opts.vertex,{'fit_chief','bundle','chief'})} = 'fit_chief'
        opts.stop_elt    (1,1) double {mustBeInteger} = 0
        opts.stop_pos    double = []
    end
    assert(isfile(rx), 'Rx not found: %s', rx);
    if opts.init, macos.init(opts.model_size); end

    % resolve + validate the exit-pupil element
    macos.load_rx(rx);  nE = macos.num_elt();
    EP = opts.ep_elt;   XP = opts.xp_elt;  if XP <= 0, XP = nE - 1; end
    % SEGMENTED ENTRANCE (Luis/Dave 2026-08-26).  The cone binning must
    % identify rays through the same STOP point, and the stop of a
    % segmented primary is the deck's OBJECT-SPACE ApStop -- so the
    % anchor is the STOP PLANE itself (Dave's construction: the cones'
    % vertices live on the virtual plane the header ApStop defines),
    % and the entrance positions come from the ray-position HISTORY at
    % obj_elt (macos.ray_hist -- one exit trace, legal at Segment
    % elements where trace-to-element is refused, and the recorded
    % positions lie on each ray's INCIDENT line, which is what the
    % plane anchor's source-direction back-projection requires).
    %   History of this branch, each step measured on e5hex1 against
    % the FEX chief crossing (bundle-vs-chief vertex): plane anchor at
    % the first non-Segment element = 52.7 mm (cones binned by M2
    % points fit the IMAGE OF M2 -- Dave's catch); source-grid index
    % grouping = 23 mm; stop-plane anchor + history entrance = 23 mm
    % with dep_rms 0.9 um (vs 4.5 um) -- two independent correct
    % binnings agreeing on a CLEAN surface.  ATTRIBUTION (measured):
    % the 23 mm is the DECK's pupil, not the machinery -- e5hex1's
    % two-singlet relay images the pupil badly: differential chief
    % (FEX) 1133.3 / finite +-1e-4 chief-pair 1142.3 / annular cone
    % zones 1156.2 (flat across zones rho 200-1300), a ~23 mm pupil
    % smear.  The zoom deck, well-imaged, agrees to 0.6 um.  The
    % WRITTEN vertex is the fit-surface/chief-ray crossing ('fit_chief',
    % Dave 2026-08-26): on the chief (no bundle tilt) at the MEASURED
    % pupil station -- it rides the smear where the beam actually
    % crosses, and collapses to the FEX point on a well-imaged pupil.
    seg_entrance = macos.get_elt_info(EP).elt_id == 11;  % Segment
    if seg_entrance
        fprintf(['[pupil_find] segmented entrance (elt %d) -- anchor ' ...
                 'on the stop plane (deck ApStop), entrance positions ' ...
                 'from the ray history\n'], EP);
    end
    xi = macos.get_elt_info(XP);
    assert(any(xi.elt_id == [3 8]), ['xp_elt %d is a %s; the exit pupil must be a ' ...
        'Return or Reference surface (pass ''xp_elt'').'], XP, xi.type);

    % FEX baseline -- the one-chief-ray sphere we improve on (and its
    % radius), run at the CONE CENTER.  The written sphere's axis is
    % f0.psi, so for an off-axis cone (a pf_scope='field' mini-cone
    % centered on one field of a wide set) the baseline chief must be
    % THAT combo's own: with the deck's nominal chief instead, the placed
    % sphere keeps the nominal field's axis and the field tilt is NOT
    % absorbed (measured on the zoom fixture: 0.46 mm RMS at the +-1'
    % fields vs 1.2e-5 re-aimed; center field identical either way).  A
    % centered cone (mean offset 0 -- every symmetric field-set-wide
    % fit) is bit-unchanged.
    set_stop_(rx, opts.stop_elt, EP);
    ctr = mean(Ffield, 1);
    if any(abs(ctr) > 0)
        s0 = macos.get_src_fov();
        dirc = s0.src_dir(:) + [ctr(1); ctr(2); 0];
        macos.set_src_fov('src_pos', s0.src_pos, ...
            'src_dir', dirc / norm(dirc), 'zSrc', s0.zSrc);
        % stop-enforced chief (Dave 2026-08-28): re-issue the stop at the
        % cone-center field so the FEX baseline below uses the field's
        % stop-aimed chief -- keeps pupil_find's written sphere consistent
        % with the supervisors' per-field fex convention.
        set_stop_(rx, opts.stop_elt, EP);
        macos.modify();
    end
    macos.trace(nE);
    f0 = macos.fex(1);

    % cone-convergence surface (pupil_map re-emits temp decks; needs the path)
    anch = opts.anchor;
    if seg_entrance, anch = 'stop'; end
    o = pupil_map(rx, Ffield, 'anchor',anch, 'obj_elt',EP, 'img_elt',XP, ...
                  'nodes',opts.nodes, 'min_fields',opts.min_fields, 'init',false, ...
                  'stop_elt',opts.stop_elt, 'stop_pos',opts.stop_pos, ...
                  'obj_hist',seg_entrance);

    % best-fit sphere to the crossing cloud, in the exit frame.  Fit
    % w = c0 + b1*u + b2*v + a*rho^2 : the sphere may be positioned (c0) and
    % TILTED (b1,b2 -- a frame term from the field bias, not aberration); the
    % residual after removing all three is the true departure-from-sphere.
    X  = o.X(:, o.good);
    X0 = o.exit_frame.origin(:);  nn = o.exit_frame.n(:);
    e1 = o.exit_frame.e1(:);      e2 = o.exit_frame.e2(:);
    d  = X - X0;
    uu = (e1.'*d).';  vv = (e2.'*d).';  ww = (nn.'*d).';  rho = hypot(uu,vv);
    A   = [ones(numel(ww),1) uu vv rho.^2];
    sol = A \ ww;
    c0  = sol(1);  a = sol(4);
    conv_radius = 1/(2*a);
    dep = ww - A*sol;  dep_rms = sqrt(mean(dep.^2));
    vtx = X0 + c0*nn;                                    % bundle XP vertex
    % Sphere NORMAL: FEX's own psi, verbatim -- pupil_find improves the
    % VERTEX only (the doc block above: "bundle vertex, FEX radius"); the
    % fitted cone normal nn is only the FIT FRAME and must not be written.
    % Two measured failure modes from writing nn (zoom fixture, 0.5' FSM
    % tilt at the pupil):
    %   1. the old "if psi(3)>0, psi=-psi" hemisphere rule NEGATED the
    %      engine's stored normal, reflecting the sphere center to the
    %      wrong side -- every field carried the full sag as a ~0.45 mm
    %      RMS bias;
    %   2. even sign-corrected, nn sits ~18 mrad off the chief-based FEX
    %      psi (it is the cone-fit axis, not the chief), which swings the
    %      center ~R*18e-3 = 55 mm and yields a reference whose OPD is
    %      BLIND to a pupil-mirror tilt (moved the map 3e-7 mm where the
    %      deck's own EP sphere moves 3e-2 mm).
    % Gated by tPupilFindMethod/
    % test_placed_sphere_keeps_the_reference_tilt_sensitive.
    psi = f0.psi(:);

    % WRITTEN vertex, three modes (Dave 2026-08-26):
    %   'fit_chief' (default) -- the FIT-SURFACE / CHIEF-RAY crossing:
    %      intersect the chief line (through the FEX crossing, along
    %      psi) with the fitted quadric.  ON the chief by construction,
    %      so none of the bundle vertex's lateral offset (which injects
    %      a PURE TILT frame term -- 0.384 mm -> 4.4e-3 mm RMS on the
    %      zoom fixture), yet it carries the MEASURED pupil station
    %      (e5hex1: 23 mm from the paraxial FEX point, where the
    %      annular beam's rays actually cross).  On a well-imaged
    %      pupil it collapses to the FEX point (zoom: sub-um).
    %   'bundle' -- the raw fit vertex (lateral offset and all).
    %   'chief'  -- FEX's own crossing, the pure paraxial anchor; the
    %      dw_d* supervisors' pf_scope='field' passes this (mini-cone
    %      fits are noisier, and its zoom gates pin bit-equality).
    % psi is the chief direction (FEX's, signed per the Return
    % convention) in all three modes.  The RADIUS follows the vertex
    % (Dave 2026-08-26): FEX's radius is the chief-ray distance from
    % ITS vertex to the next plane -- the propagation target, where
    % the sphere CENTER sits.  When 'fit_chief' slides the vertex t
    % along the chief, that plane does not move, so rad = fex.rad - t:
    % the center is INVARIANT.  'chief' (t=0) and 'bundle' (no defined
    % along-chief t; lateral offset) keep FEX's radius unchanged.
    vsel = opts.vertex;
    rad_w = f0.rad;
    switch vsel
        case 'chief'
            vw = f0.vpt(:);
        case 'bundle'
            vw = vtx(:);
        otherwise   % fit_chief
            % chief line in the exit frame: p(t) = f0.vpt + t*psi
            p0 = f0.vpt(:) - X0;
            u0 = e1.'*p0;   v0 = e2.'*p0;   w0 = nn.'*p0;
            du = e1.'*psi;  dv = e2.'*psi;  dw = nn.'*psi;
            b1 = sol(2);  b2 = sol(3);
            Aq = a*(du^2 + dv^2);
            Bq = b1*du + b2*dv + 2*a*(u0*du + v0*dv) - dw;
            Cq = c0 + b1*u0 + b2*v0 + a*(u0^2 + v0^2) - w0;
            if abs(Aq) < 1e-15 * max(abs(Bq), 1)
                t = -Cq / Bq;
            else
                disc = Bq^2 - 4*Aq*Cq;
                if disc < 0
                    t = 0;      % no real crossing: fall back to FEX
                else
                    tt = (-Bq + [1 -1]*sqrt(disc)) / (2*Aq);
                    [~, kmin] = min(abs(tt));
                    t = tt(kmin);
                end
            end
            vw = f0.vpt(:) + t*psi;
            rad_w = f0.rad - t;          % center stays on the next plane
    end
    macos.load_rx(rx);  set_stop_(rx, opts.stop_elt, EP);
    if opts.place
        macos.set_xp(vw, psi, rad_w);
    end

    pf = struct('ep_elt',EP, 'xp_elt',XP, 'nElt',nE, 'xp_type',xi.type, ...
        'fex',struct('vpt',f0.vpt(:).', 'rad',f0.rad), ...
        'vtx',vtx(:).', 'vtx_written',vw(:).', 'vertex',vsel, ...
        'rad',rad_w, 'psi',psi(:).', ...
        'conv_radius',conv_radius, 'dep_rms',dep_rms, 'uv',[uu vv], 'dep',dep(:), ...
        'blur',o.blur, 'surface',o.surface, 'map',o.map, 'wander',o.wander, ...
        'anchor',opts.anchor, 'o_rim',o, 'placed',opts.place);
end

function set_stop_(rx, stop_elt, EP)
%  Stop resolution, in priority order: an EXPLICIT element stop wins;
%  else a deck-declared ApStop= GOVERNS -- and since the stop-enforced-
%  chief ruling (Dave 2026-08-28) its object-space form is RE-ISSUED via
%  stop_obj rather than merely left in force, so a call AFTER a source
%  re-aim re-anchors the chief through the stop at the CURRENT field
%  (overriding with macos.stop(EP) would still be wrong -- it would
%  replace e.g. a segmented-primary object-space stop with ONE segment's
%  aperture, Luis 2026-08-26); else the legacy default, stop at ep_elt.
if stop_elt > 0
    macos.stop(stop_elt);
    return
end
tok = regexp(fileread(rx), '^\s*ApStop=\s*([^\n%]*)', 'tokens', ...
             'once', 'lineanchors');
if isempty(tok)
    macos.stop(EP);
    return
end
v = sscanf(tok{1}, '%f');
if numel(v) >= 3
    macos.stop_obj(v(1), v(2), v(3));
elseif isscalar(v) && v == round(v) && v >= 1
    macos.stop(int32(v));
end
end
