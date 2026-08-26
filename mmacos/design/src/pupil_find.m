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
%     - the reference-sphere radius WRITTEN to the Rx is FEX's XP->next-plane
%       FAR-FIELD radius (what the diffraction propagator uses).  PUPIL_FIND
%       improves the sphere's VERTEX (the cone-bundle apex) over FEX's single
%       chief-ray vertex, but keeps FEX's propagation radius.
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
        opts.vertex      (1,:) char {mustBeMember(opts.vertex,{'bundle','chief'})} = 'bundle'
        opts.stop_elt    (1,1) double {mustBeInteger} = 0
        opts.stop_pos    double = []
    end
    assert(isfile(rx), 'Rx not found: %s', rx);
    if opts.init, macos.init(opts.model_size); end

    % resolve + validate the exit-pupil element
    macos.load_rx(rx);  nE = macos.num_elt();
    EP = opts.ep_elt;   XP = opts.xp_elt;  if XP <= 0, XP = nE - 1; end
    % Segmented entrance: the cone-lattice anchor (pupil_map's obj_elt)
    % cannot be a Segment element -- the probe traces evaluate OPD at
    % the anchor and the engine refuses OPD there ("OPD: cannot
    % evaluate at a Segment element").  Advance to the first
    % non-Segment element.  The STOP is unaffected: a segmented
    % primary's stop is the deck's object-space ApStop, which
    % set_stop_ leaves in force (Luis 2026-08-26).
    if macos.get_elt_info(EP).elt_id == 11               % Segment
        E0 = EP;
        while EP < nE && macos.get_elt_info(EP).elt_id == 11
            EP = EP + 1;
        end
        fprintf(['[pupil_find] elt %d is a Segment -- cone-lattice ' ...
                 'anchor advanced to elt %d (stop unaffected)\n'], E0, EP);
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
        macos.modify();
    end
    macos.trace(nE);
    f0 = macos.fex(1);

    % cone-convergence surface (pupil_map re-emits temp decks; needs the path)
    o = pupil_map(rx, Ffield, 'anchor',opts.anchor, 'obj_elt',EP, 'img_elt',XP, ...
                  'nodes',opts.nodes, 'min_fields',opts.min_fields, 'init',false, ...
                  'stop_elt',opts.stop_elt, 'stop_pos',opts.stop_pos);

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

    % WRITTEN vertex: 'bundle' (default; the fit vertex above) or 'chief'
    % (FEX's own crossing).  The bundle vertex sits laterally off the
    % chief by the cone's coma asymmetry (0.384 mm on the zoom fixture),
    % and referencing OPD to a sphere centered there injects a PURE TILT
    % frame term (4.4e-3 mm RMS at the center field) with zero aberration
    % content -- the map agrees with fex's to 1.2e-8 mm RMS once
    % tip/tilt+focus are removed.  'chief' keeps the bundle fit as the
    % pupil-wander DIAGNOSTIC (pf.vtx, pf.dep_rms, pf.wander) and writes
    % the chief crossing, so w_nom carries no bundle tilt.  The dw_d*
    % supervisors' pf_scope='field' passes 'chief'.
    if strcmp(opts.vertex, 'chief'), vw = f0.vpt(:); else, vw = vtx(:); end
    macos.load_rx(rx);  set_stop_(rx, opts.stop_elt, EP);
    if opts.place
        macos.set_xp(vw, psi, f0.rad);
    end

    pf = struct('ep_elt',EP, 'xp_elt',XP, 'nElt',nE, 'xp_type',xi.type, ...
        'fex',struct('vpt',f0.vpt(:).', 'rad',f0.rad), ...
        'vtx',vtx(:).', 'vtx_written',vw(:).', 'vertex',opts.vertex, ...
        'rad',f0.rad, 'psi',psi(:).', ...
        'conv_radius',conv_radius, 'dep_rms',dep_rms, 'uv',[uu vv], 'dep',dep(:), ...
        'blur',o.blur, 'surface',o.surface, 'map',o.map, 'wander',o.wander, ...
        'anchor',opts.anchor, 'o_rim',o, 'placed',opts.place);
end

function set_stop_(rx, stop_elt, EP)
%  Stop resolution, in priority order: an EXPLICIT element stop wins;
%  else a deck-declared ApStop= (header object-space 3-vector or the
%  element form) GOVERNS and is left alone -- FEX handles object-space
%  stops, and overriding with macos.stop(EP) would replace e.g. a
%  segmented-primary object-space stop with ONE segment's aperture
%  (Luis 2026-08-26); else the legacy default, stop at ep_elt.
if stop_elt > 0
    macos.stop(stop_elt);
elseif isempty(regexp(fileread(rx), '^\s*ApStop\s*=', 'once', 'lineanchors'))
    macos.stop(EP);
end
end
