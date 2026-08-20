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
%     'ep_elt'     entrance-pupil (stop) element (default 1).
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
%     .vtx .rad .psi              the sphere WRITTEN (bundle vertex, FEX radius)
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
    end
    assert(isfile(rx), 'Rx not found: %s', rx);
    if opts.init, macos.init(opts.model_size); end

    % resolve + validate the exit-pupil element
    macos.load_rx(rx);  nE = macos.num_elt();
    EP = opts.ep_elt;   XP = opts.xp_elt;  if XP <= 0, XP = nE - 1; end
    xi = macos.get_elt_info(XP);
    assert(any(xi.elt_id == [3 8]), ['xp_elt %d is a %s; the exit pupil must be a ' ...
        'Return or Reference surface (pass ''xp_elt'').'], XP, xi.type);

    % FEX baseline -- the one-chief-ray sphere we improve on (and its radius)
    macos.stop(EP);  macos.trace(nE);
    f0 = macos.fex(1);

    % cone-convergence surface (pupil_map re-emits temp decks; needs the path)
    o = pupil_map(rx, Ffield, 'anchor',opts.anchor, 'obj_elt',EP, 'img_elt',XP, ...
                  'nodes',opts.nodes, 'min_fields',opts.min_fields, 'init',false);

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
    psi = nn;  if psi(3) > 0, psi = -psi; end            % normal toward the image

    % place the sphere into the internal Rx (FEX radius, bundle vertex) -- as FEX does
    macos.load_rx(rx);  macos.stop(EP);
    if opts.place
        macos.set_xp(vtx, psi, f0.rad);
    end

    pf = struct('ep_elt',EP, 'xp_elt',XP, 'nElt',nE, 'xp_type',xi.type, ...
        'fex',struct('vpt',f0.vpt(:).', 'rad',f0.rad), ...
        'vtx',vtx(:).', 'rad',f0.rad, 'psi',psi(:).', ...
        'conv_radius',conv_radius, 'dep_rms',dep_rms, 'uv',[uu vv], 'dep',dep(:), ...
        'blur',o.blur, 'surface',o.surface, 'map',o.map, 'wander',o.wander, ...
        'anchor',opts.anchor, 'o_rim',o, 'placed',opts.place);
end
