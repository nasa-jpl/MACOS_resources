function out = pupil_map(deck, Ffield, opts)
%PUPIL_MAP  Interface-pupil quality by the CONE-CONVERGENCE model.
%
%   out = PUPIL_MAP(DECK, FFIELD) measures how well DECK images its
%   ENTRANCE pupil onto its exit/interface pupil, over the field set FFIELD
%   (K x 2 rad, BOX-RELATIVE like every other design/src scorer).
%
%   THE MODEL (Dave, 2026-08-02).  A point on the entrance-pupil surface is
%   imaged to the exit pupil by the CONE of rays that pass through it -- one
%   ray per field angle.  So the imaging bundle's APERTURE IS THE FIELD SET.
%   Never trace a wider or a symmetric cone: rays outside the used field
%   range do not exist in operation, and a symmetric cone would report a
%   pupil the instrument never sees.  This is the finite-field
%   generalisation of the engine's XPS field-differential crossing, which is
%   its two-ray limit -- TPUPILMAP gates the two against each other.
%
%   SCOPE -- THE MODEL REQUIRES A FIELD.  A testbed or interferometer fed by
%   a POINT SOURCE has one ray per entrance-pupil point and therefore no
%   ray-cone pupil image at all; its pupil quality is a Fresnel/Talbot
%   wavefront question and is handled separately.  Nothing here applies to
%   that case.
%
%   FAST-PRIMARY ANCHORING -- MANDATORY, NOT A REFINEMENT.  The cones must be
%   anchored at FIXED points on the CURVED entrance-pupil surface.  Grouping
%   rays by source-grid INDEX across field traces is not that: with the stop
%   offset from the mirror vertex and the mirror's own sag, a fixed grid ray
%   walks across the surface by ~(stop offset + sag)*dtheta as the field
%   changes, growing like r^2 toward the rim.  On the Rodgers2 f/1.25 primary
%   that is ~0.9 mm at the rim, which divided by the 30x pupil demagnification
%   masquerades as ~29 um of exit-pupil blur -- several times the diffraction
%   floor, and entirely fictitious.  This routine therefore REGRIDS: each
%   field's exit-side ray data is interpolated as a function of that field's
%   own entrance-surface hit point, and evaluated at common nodes.  The
%   anchoring residual is ESTIMATED (linear vs natural interpolant) and
%   returned as .anchor.resid_max; it must be small against the blur being
%   measured, and TPUPILMAP gates that it is.
%
%   RIM PARTIAL CONES ARE KEPT, NOT REPAIRED.  A node whose ray at some field
%   angle falls outside the stop simply has no contribution from that field;
%   the asymmetric partial cone IS the physical pupil-edge behaviour.  Nodes
%   with fewer than 'min_fields' contributions are dropped and counted.
%
%   THE FOUR-PART LADDER, never one merged residual:
%
%     (1) .blur      per-node convergence waist = the pupil-image BLUR.
%     (2) .surface   the SURFACE of convergence points -- tilt (a frame
%                    term) / power / astigmatism, against TWO references:
%                      .zern      low-order fit in normalised entrance
%                                 coordinates (the same basis and frame
%                                 macos.pupil_quality uses, so the two are
%                                 directly comparable);
%                      .ideal     residual against the IDEAL IMAGE of the
%                                 primary's own sag.  A curved entrance
%                                 pupil images to a curved exit pupil at
%                                 longitudinal magnification m^2 (here
%                                 50 mm of sag -> ~56 um), which is the same
%                                 order as the residuals.  Charging a fast
%                                 primary's faithfully-imaged sag as "pupil
%                                 curvature" is the error this reference
%                                 exists to prevent.  .ideal.beta_over_m2
%                                 is the measured ratio: 1 means the sag is
%                                 imaged exactly as it should be.
%                      .flat      residual against the FLAT placed plane =
%                                 the operational number.
%     (3) .map       the transverse map entrance -> exit: magnification
%                    (.mag; 30.000 is the Rodgers2 target, and his 28.7x
%                    slip lands here), anamorphism, rotation, and
%                    .distortion -- the nonlinear residual, i.e. the SHAPE
%                    of the wander across the pupil.  Distortion is a
%                    first-class metric here, not a leftover.
%                    THE PER-FIELD MAGNIFICATION COMES IN TWO FRAMES and
%                    they are not interchangeable: .mag_per_field is read on
%                    the deck's PLACED plane (operational -- what a fixed
%                    coldstop samples) and therefore carries a 1/cos
%                    obliquity term as each field's exit chief swings away
%                    from that plane's normal; .mag_per_field_chief reads
%                    the same station perpendicular to THAT FIELD'S OWN exit
%                    chief and is the true pupil-imaging scale.  Only the
%                    chief-normal spread is a pupil-imaging defect.  On a
%                    30x afocal over a 0.5 deg box the frame term alone is
%                    ~0.8% of apparent magnification.
%     (4) .wander    where each cone's rays actually pierce the PLACED
%                    plane.  Non-convergence there IS the footprint shear
%                    the instrument feels.  Reported for the deck's own
%                    plane AND for the best-fit plane (position + tilt),
%                    which is the analogue of tuning a coldstop's tilt.
%
%   .image (with 'image',true) is the simulated pupil image: the cone spot
%   function convolved with the entrance-pupil amplitude map.  This is legal
%   because extended-field illumination makes pupil imaging INCOHERENT --
%   field angles are mutually incoherent and the field IS the pupil-imaging
%   aperture.  Caveats, all encoded in the returned struct: a single kernel
%   assumes isoplanatism ('zones' > 1 gives a piecewise-radial kernel for the
%   rim, where coldstop margins are decided); the kernel is only as smooth as
%   the FIELD SAMPLING (.image.n_field_samples -- a 3x3 solve set gives a
%   nine-point kernel, so pass a denser grid over the SAME box for a picture);
%   and vignetted rim cones have asymmetric kernels the convolution misses,
%   though the traced .wander data does contain them.
%
%   .diffraction quotes the geometric-treatment floor lambda/(2*NA_field)
%   once -- ~4 um on the Rodgers2 33 mm exit pupil, i.e. negligible against
%   everything measured here.
%
%   WHERE THE CONES ARE ANCHORED IS A CHOICE, AND IT IS NOT A DETAIL.
%     'anchor','surface' (DEFAULT) anchors on the curved OBJ_ELT SURFACE --
%          Dave's ruled model, because what an instrument cares about is the
%          image of the PRIMARY (its segment gaps, its spider, its hole),
%          and on a segmented primary that surface is the thing with
%          structure on it.
%     'anchor','stop' anchors on the flat plane through the prescription's
%          ApStop normal to the box-centre chief -- the classical ENTRANCE
%          PUPIL.  This is what the engine's XPS computes (it rotates the
%          chief about StopPos by 5e-6 rad along xGrid and crosses each ray
%          with its own differential partner), so it is the setting in which
%          PUPIL_MAP and MACOS.PUPIL_QUALITY measure the same surface.
%          TPUPILMAP gates that identity.  Requires a collimated source.
%     'anchor',struct('Vpt',1x3,'psi',1x3) anchors on any fixed flat plane.
%     'anchor','index' groups rays by SOURCE-GRID INDEX across the field
%          traces.  THIS IS THE WRONG ANSWER and exists only to measure how
%          wrong: it is what a naive implementation does, and on a fast
%          primary it manufactures blur out of the ray's walk across the
%          surface.  Never quote a result from it.
%   The two differ by the object plane's own longitudinal offset and sag,
%   imaged at m^2 -- 50 mm of stop offset and 50 mm of primary sag on the
%   Rodgers2 deck, i.e. ~0.11 mm at the exit pupil, which is comparable to
%   everything else on the ladder.  Quote which anchor a number came from.
%
%   Name-value:
%     'anchor'      'surface' (default) | 'stop' | a plane struct (above)
%     'obj_elt'     entrance-pupil element for 'surface' anchoring (default 1)
%     'img_elt'     element at which the exit ray lines are read, and whose
%                   plane is the PLACED plane (default: the last element)
%     'nodes'       n for the n x n node lattice on the entrance surface (21)
%     'min_fields'  fewest cone rays for a node to be scored (3)
%     'strip_ap'    rewrite every ApType= to None (true)
%     'model_size'  (256)   'init'  call macos.init first (true)
%     'placed'      struct('Vpt',1x3,'psi',1x3) overriding img_elt's plane
%     'image'       build the simulated pupil image (false)
%     'zones'       radial zones for the piecewise kernel (1)
%     'image_npix'  raster size (256)
%     'quiet'       (true)
%
%   See also AFOCAL_LADDER_DECK, MACOS.PUPIL_QUALITY, MACOS.XPS.

    arguments
        deck (1,:) char
        Ffield (:,2) double
        opts.anchor     = 'surface'
        opts.obj_elt    (1,1) double = 1
        opts.img_elt    (1,1) double = 0
        opts.nodes      (1,1) double {mustBeInteger,mustBePositive} = 21
        opts.min_fields (1,1) double = 3
        opts.strip_ap   (1,1) logical = true
        opts.model_size (1,1) double = 256
        opts.init       (1,1) logical = true
        opts.placed     struct = struct([])
        opts.image      (1,1) logical = false
        opts.zones      (1,1) double {mustBeInteger,mustBePositive} = 1
        opts.image_npix (1,1) double = 256
        opts.quiet      (1,1) logical = true
    end

    K = size(Ffield,1);
    if K < 3
        error('macos:design:pupil_map:fields', ...
              ['the cone aperture IS the field set: at least 3 field ' ...
               'angles are needed to converge a cone (%d given)'], K);
    end

    txt = fileread(deck);
    if opts.strip_ap
        txt = regexprep(txt, '(ApType=\s*)\S+', '$1None');
    end
    [cdir0, cpos0, apst, lam] = deck_src_(txt);
    stand = dot(apst - cpos0, cdir0);
    bx0 = asin(cdir0(1));   by0 = asin(cdir0(2));
    Vs = grab_all3_(txt,'VptElt');   Ps = grab_all3_(txt,'psiElt');
    io = opts.obj_elt;
    ii = opts.img_elt;   if ii <= 0, ii = size(Vs,2); end
    V1 = Vs(:,io);   n1 = Ps(:,io)/norm(Ps(:,io));
    plane = struct('Vpt', Vs(:,ii).', 'psi', (Ps(:,ii)/norm(Ps(:,ii))).');
    if ~isempty(fieldnames(opts.placed)), plane = opts.placed; end
    npl = plane.psi(:)/norm(plane.psi);

    if opts.init, macos.init(opts.model_size); end

    % ---- the anchor: a curved element surface, or a fixed flat plane ------
    [~, kc] = min(sum(Ffield.^2, 2));
    anc = opts.anchor;
    if ischar(anc) && strcmpi(anc,'stop')
        cbx = bx0 + Ffield(kc,1);   cby = by0 + Ffield(kc,2);
        nc  = [sin(cbx); sin(cby); sqrt(max(0,1-sin(cbx)^2-sin(cby)^2))];
        anc = struct('Vpt', apst(:).', 'psi', nc(:).');
    end
    index_group = ischar(anc) && strcmpi(anc,'index');
    if index_group, anc = 'surface'; end
    if isstruct(anc)
        A0 = anc.Vpt(:);   nA = anc.psi(:)/norm(anc.psi);
        zsrc = regexp(txt,'zSource\s*=\s*([-\d.EeD+]+)','tokens','once');
        if isempty(zsrc) || str2double(strrep(zsrc{1},'D','E')) < 1e10
            error('macos:design:pupil_map:anchor', ...
                  ['plane anchoring back-projects each ray along the SOURCE ' ...
                   'direction, which is only exact for a collimated source ' ...
                   '(zSource >= 1e10); this deck is not collimated']);
        end
        anchor_mode = 'plane';
    else
        A0 = V1;   nA = n1;   anchor_mode = 'surface';
    end
    [a1,a2] = perp_(nA);

    % ---- trace every field, twice (entrance then exit) --------------------
    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>
    S = struct('m',{},'sag',{},'pe',{},'de',{},'chief',{});
    for k = 1:K
        bx = bx0 + Ffield(k,1);   by = by0 + Ffield(k,2);
        [Pm, Pe, De, ok, chief] = trace_field_(txt, tmp, apst, stand, ...
                                              bx, by, io, ii);
        Q = Pm(:,ok);
        if strcmp(anchor_mode,'plane')
            % back-project each ray along the (collimated) source direction
            % onto the fixed anchor plane
            cd = [sin(bx); sin(by); sqrt(max(0,1-sin(bx)^2-sin(by)^2))];
            t  = (nA.'*(A0 - Q)) / (nA.'*cd);
            Q  = Q + cd*t;
        end
        d = Q - A0;
        S(k).m    = [a1.'*d; a2.'*d];
        S(k).sag  = (nA.'*d).';
        S(k).pe   = Pe(:,ok);
        S(k).de   = De(:,ok) ./ vecnorm(De(:,ok));
        S(k).idx  = find(ok).';
        S(k).chief = chief;
    end

    if index_group
        % THE WRONG ANSWER, on purpose.  Group by source-grid index: no
        % regrid, no anchoring, the node coordinates taken from the centre
        % field.  Every ray's walk across the curved surface between fields
        % turns straight into apparent pupil blur.
        idx = S(1).idx;
        for k = 2:K, idx = intersect(idx, S(k).idx); end
        J = numel(idx);
        PE = nan(3,J,K);   DE = nan(3,J,K);
        for k = 1:K
            [~, loc] = ismember(idx, S(k).idx);
            PE(:,:,k) = S(k).pe(:,loc);
            DE(:,:,k) = S(k).de(:,loc);
        end
        [~, loc] = ismember(idx, S(kc).idx);
        nod  = S(kc).m(:,loc);
        sagn = S(kc).sag(loc).';
        ares = zeros(1,J);
        anchor_mode = 'index';
    else
        % ---- the node lattice on the entrance surface ---------------------
        % Anchored on the field nearest the box centre; the lattice is
        % REGULAR in the anchor's transverse frame, so it does not inherit
        % the ray lattice's own sampling.
        rr  = vecnorm(S(kc).m);
        rout = max(rr);   rin = min(rr);
        if rin < 0.05*rout, rin = 0; end          % no central obscuration
        g = linspace(-rout, rout, opts.nodes);
        [NX,NY] = meshgrid(g,g);
        rn = hypot(NX,NY);
        keep = rn <= rout & rn >= rin;
        nod = [NX(keep).'; NY(keep).'];
        J = size(nod,2);

        % ---- regrid each field's exit-side data onto the nodes ------------
        PE = nan(3,J,K);   DE = nan(3,J,K);   ares = zeros(1,J);
        for k = 1:K
            [pe, de, res] = regrid_(S(k).m, S(k).pe, S(k).de, nod);
            PE(:,:,k) = pe;   DE(:,:,k) = de;
            ares = max(ares, res);
        end
        sagn = regrid_scalar_(S(kc).m, S(kc).sag(:).', nod);
    end
    nfld = squeeze(sum(all(isfinite(PE),1), 3)).';
    if K == 1, nfld = nfld(:).'; end

    % ---- (1) converge each cone ------------------------------------------
    X = nan(3,J);   waist = nan(1,J);   wmax = nan(1,J);  halfang = nan(1,J);
    for j = 1:J
        sel = find(all(isfinite(PE(:,j,:)),1));
        if numel(sel) < opts.min_fields, continue; end
        p = squeeze(PE(:,j,sel));   d = squeeze(DE(:,j,sel));
        if numel(sel) == 1, p = p(:); d = d(:); end
        A = zeros(3); b = zeros(3,1);
        for i = 1:size(d,2)
            M = eye(3) - d(:,i)*d(:,i).';
            A = A + M;   b = b + M*p(:,i);
        end
        if rcond(A) < 1e-12, continue; end
        x = A\b;
        r = vecnorm((x - p) - d.*sum(d.*(x-p),1));
        X(:,j) = x;   waist(j) = rms(r);   wmax(j) = max(r);
        dm = mean(d,2); dm = dm/norm(dm);
        halfang(j) = max(atan2(vecnorm(cross(repmat(dm,1,size(d,2)), d)), dm.'*d));
    end
    good = all(isfinite(X),1);
    if nnz(good) < 6
        error('macos:design:pupil_map:converge', ...
              'only %d nodes converged -- the field set or the node lattice is degenerate', nnz(good));
    end

    % ---- exit-pupil frame -------------------------------------------------
    ne = mean([S.chief], 2);  ne = ne/norm(ne);
    X0 = mean(X(:,good), 2);
    [b1,b2] = perp_(ne);
    dX = X - X0;
    w  = [b1.'*dX; b2.'*dX];
    sg = (ne.'*dX).';

    % ---- (3) the transverse map: affine fit + distortion -------------------
    Amat = [nod(1,good).', nod(2,good).', ones(nnz(good),1)];
    cx = Amat\w(1,good).';   cy = Amat\w(2,good).';
    Lmat = [cx(1) cx(2); cy(1) cy(2)];
    sv = svd(Lmat);
    [U0,~,V0] = svd(Lmat);   Rot = U0*V0.';
    pred = nan(2,J);
    pred(:,good) = [Amat*cx, Amat*cy].';
    dist = vecnorm(w - pred);
    Rex  = max(vecnorm(w(:,good)));
    % PER-FIELD magnification: the affine map each field's OWN rays make from
    % the anchor plane to the exit pupil.  This is a different number from
    % the cone-convergence .mag above and both belong in the report: the
    % cone number is the field-AVERAGED pupil magnification (what an
    % extended-field instrument's pupil image actually shows), while the
    % per-field number is what a single field angle delivers -- and it is the
    % one his slides quote when they say the magnification slips from 30x to
    % 28.7x.  On a perfect afocal they coincide; the gap between them IS
    % pupil aberration.
    %
    % AND IT IS MEASURED IN TWO FRAMES, because one of them contains a term
    % that is not pupil imaging at all (Fable review, 2026-08-03; the
    % frame-before-angle rule).  A footprint read on the deck's PLACED plane
    % -- which a coldstop holds FIXED while each field's exit chief swings by
    % M*theta -- is stretched by 1/cos(incidence) in the plane of incidence.
    % On a 30x afocal over a 0.5 deg box the exit chief reaches ~10.6 deg off
    % the box centre, i.e. ~1.6% of areal stretch, which reads as ~0.8% of
    % apparent magnification change ON A PERFECT SYSTEM.  The chief-normal
    % number reads the SAME STATION in the plane perpendicular to THAT
    % FIELD'S OWN exit chief and carries no such term.
    %   .mag_per_field         placed-plane frame -- what a fixed detector or
    %                          coldstop at the interface actually samples
    %                          (operational; keep it, it is what the
    %                          instrument feels)
    %   .mag_per_field_chief   chief-normal frame -- the true pupil-imaging
    %                          scale, and the ONLY one whose spread across
    %                          the field is a pupil-imaging defect
    %   .obliquity_per_field   the ratio between them; sqrt(cos(incidence))
    %                          to first order, and .incidence_deg is that
    %                          incidence
    % Quote which frame a magnification came from.  Note that .mag (the
    % cone-convergence magnification) is already chief-normal: it projects
    % 3-D convergence POINTS onto the plane normal to the mean exit chief,
    % and never intersects the placed plane at all.
    magf = nan(1,K);   magc = nan(1,K);
    anaf = nan(1,K);   anac = nan(1,K);
    incd = nan(1,K);
    for k = 1:K
        nk = S(k).chief(:)/norm(S(k).chief);
        [c1,c2] = perp_(nk);
        [magf(k), anaf(k)] = field_mag_(S(k), plane.Vpt(:), npl, b1, b2);
        [magc(k), anac(k)] = field_mag_(S(k), plane.Vpt(:), nk,  c1, c2);
        incd(k) = acosd(min(1, abs(npl.'*nk)));
    end

    map = struct('linear',Lmat, 'offset',[cx(3) cy(3)], ...
                 'sigma',sv(:).', 'mag_per_field',magf, ...
                 'mag_per_field_chief',magc, ...
                 'anamorph_per_field',anaf, ...
                 'anamorph_per_field_chief',anac, ...
                 'incidence_deg',incd, ...
                 'obliquity_per_field',magc./magf, ...
                 'mag_centre',magf(kc), ...
                 'mag_centre_chief',magc(kc), ...
                 'mag', 1/sqrt(prod(sv)), ...
                 'mag_axes', 1./sv(:).', ...
                 'anamorph', sv(1)/sv(2), ...
                 'rotation_deg', atan2d(Rot(2,1), Rot(1,1)), ...
                 'distortion_rms', rms(dist(good)), ...
                 'distortion_max', max(dist(good)), ...
                 'distortion_frac_rms', rms(dist(good))/Rex, ...
                 'distortion_frac_max', max(dist(good))/Rex, ...
                 'distortion', dist, 'pupil_radius', Rex, ...
                 'predicted', pred);

    % ---- (2) the convergence SURFACE ---------------------------------------
    Rm  = max(vecnorm(nod(:,good)));
    un  = nod(1,good).'/Rm;   vn = nod(2,good).'/Rm;
    [zc, znames] = zfit_(un, vn, sg(good));
    zres = sg(good) - zbasis_(un,vn)*zc;
    gi = @(nm) zc(strcmp(znames,nm));
    % ideal image of the entrance sag: fit sag_exit = piston + tilt + beta*sag_in
    % A FLAT anchor has no sag, so this reference does not exist there -- it
    % is the curved-primary correction and nothing else.  Report NaN rather
    % than a number fitted to round-off.
    m2  = prod(sv);                        % longitudinal magnification
    sag_in_pv = max(sagn(good)) - min(sagn(good));
    if strcmp(anchor_mode,'surface') && sag_in_pv > 1e-9
        Ai = [ones(nnz(good),1), un, vn, sagn(good).'];
        ci = Ai\sg(good);
        ires = sg(good) - Ai*ci;
        beta = ci(4);
    else
        beta = NaN;   ires = nan(nnz(good),1);
    end
    % flat placed plane: signed distance of each convergence point from it
    dfl = (npl.'*(X - plane.Vpt(:))).';
    surface = struct('zern',zc(:).', 'names',{znames}, ...
        'defocus',gi('defocus'), 'astig',[gi('astig0') gi('astig45')], ...
        'coma',[gi('coma_x') gi('coma_y')], 'spherical',gi('spherical'), ...
        'tilt',[gi('tilt_x') gi('tilt_y')], 'piston',gi('piston'), ...
        'fit_resid',rms(zres), 'sag_rms',rms(sg(good)), ...
        'sag_pv',max(sg(good))-min(sg(good)), 'sag',sg(:).', ...
        'norm_radius',Rm, ...
        'ideal',struct('beta',beta, 'm2',m2, 'beta_over_m2',beta/m2, ...
                       'resid_rms',rms(ires), 'resid_max',max(abs(ires)), ...
                       'sag_in_pv',sag_in_pv, ...
                       'sag_imaged_pv',abs(m2)*sag_in_pv), ...
        'flat',struct('resid_rms',rms(dfl(good)), ...
                      'resid_pv',max(dfl(good))-min(dfl(good)), ...
                      'dist',dfl));

    % ---- (4) wander at the placed plane ------------------------------------
    wander = wander_(PE, DE, plane.Vpt(:), npl, b1, b2, good, Rex);
    bestpl = best_plane_(PE, DE, plane.Vpt(:), npl, ne, b1, b2, good, Rex);

    % ---- diffraction floor --------------------------------------------------
    na = sin(max(halfang(good)));
    diffraction = struct('NA_field',na, 'floor_m', lam/(2*max(na,realmin)), ...
                         'lambda',lam, 'half_angle_deg', max(halfang(good))*180/pi);

    out = struct('deck',deck, 'fields',Ffield, 'obj_elt',io, 'img_elt',ii, ...
        'anchor_mode',anchor_mode, 'anchor_plane',struct('Vpt',A0.','psi',nA.'), ...
        'lambda',lam, 'placed',plane, 'nodes',nod, 'node_sag',sagn, ...
        'n_fields_per_node',nfld, 'good',good, ...
        'X',X, 'exit_frame',struct('origin',X0.','n',ne.','e1',b1.','e2',b2.'), ...
        'w',w, ...
        'blur',struct('waist_rms',waist, 'waist_max',wmax, ...
                      'rms',rms(waist(good)), 'max',max(wmax(good)), ...
                      'median',median(waist(good))), ...
        'surface',surface, 'map',map, ...
        'wander',wander, 'best_plane',bestpl, ...
        'anchor',struct('resid',ares, 'resid_max',max(ares(good)), ...
                        'blur_ratio', max(ares(good))/max(rms(waist(good)),realmin)), ...
        'diffraction',diffraction, 'image',[]);

    if opts.image
        out.image = pupil_image_(PE, DE, plane.Vpt(:), npl, b1, b2, ...
                                 w, good, Rex, opts.zones, opts.image_npix, K);
    end

    if ~opts.quiet, report_(out); end
end

% =====================================================================
function report_(o)
    fprintf('pupil_map: %s\n', o.deck);
    fprintf('  anchor %s (elt %d) -> img elt %d, %d fields, %d/%d nodes\n', ...
        o.anchor_mode, o.obj_elt, o.img_elt, size(o.fields,1), ...
        nnz(o.good), numel(o.good));
    fprintf('  (0) anchoring   resid %8.3f um   = %.1f%% of the blur\n', ...
        o.anchor.resid_max*1e6, 100*o.anchor.blur_ratio);
    fprintf('  (1) blur        rms %8.3f um   max %8.3f um   (diffraction floor %.2f um)\n', ...
        o.blur.rms*1e6, o.blur.max*1e6, o.diffraction.floor_m*1e6);
    fprintf(['  (2) surface     defocus %+8.3f mm  astig [%+7.3f %+7.3f] mm  ' ...
             'sag %7.3f mm rms\n'], o.surface.defocus*1e3, o.surface.astig*1e3, ...
             o.surface.sag_rms*1e3);
    fprintf('      vs ideal image of the entrance sag: beta/m2 %+7.4f, resid %8.3f um rms\n', ...
        o.surface.ideal.beta_over_m2, o.surface.ideal.resid_rms*1e6);
    fprintf('      vs the flat placed plane:            resid %8.3f um rms (%7.3f mm P-V)\n', ...
        o.surface.flat.resid_rms*1e6, o.surface.flat.resid_pv*1e3);
    fprintf(['  (3) map         mag %9.4fx  anamorph %7.5f  rot %+7.4f deg  ' ...
             'distortion %6.3f%% of R\n'], o.map.mag, o.map.anamorph, ...
             o.map.rotation_deg, 100*o.map.distortion_frac_max);
    fprintf('      per-field mag, PLACED plane:  centre %9.4fx  range %9.4f .. %9.4fx (%+.2f%%)\n', ...
        o.map.mag_centre, min(o.map.mag_per_field), max(o.map.mag_per_field), ...
        100*(max(o.map.mag_per_field)-min(o.map.mag_per_field))/o.map.mag_centre);
    fprintf('      per-field mag, CHIEF-normal:  centre %9.4fx  range %9.4f .. %9.4fx (%+.2f%%)\n', ...
        o.map.mag_centre_chief, min(o.map.mag_per_field_chief), ...
        max(o.map.mag_per_field_chief), ...
        100*(max(o.map.mag_per_field_chief)-min(o.map.mag_per_field_chief))/o.map.mag_centre_chief);
    fprintf('      exit-chief incidence on the placed plane %.3f .. %.3f deg -> obliquity %.4f .. %.4f\n', ...
        min(o.map.incidence_deg), max(o.map.incidence_deg), ...
        min(o.map.obliquity_per_field), max(o.map.obliquity_per_field));
    fprintf('  (4) wander      placed plane  rms %8.3f um  max %8.3f um  (%.3f%% of R)\n', ...
        o.wander.rms*1e6, o.wander.max*1e6, 100*o.wander.frac_max);
    fprintf('                  best plane    rms %8.3f um  max %8.3f um  (shift %+7.3f mm, tilt %+7.4f deg)\n', ...
        o.best_plane.rms*1e6, o.best_plane.max*1e6, o.best_plane.shift*1e3, ...
        o.best_plane.tilt_deg);
end

% =====================================================================
function [Pm, Pe, De, ok, chief] = trace_field_(txt, tmp, apst, stand, bx, by, io, ii)
%TRACE_FIELD_  One field: entrance hits, exit ray lines, exit chief.
%   TWO traces, no reload between them -- macos.trace(k) reports at element
%   k, and the second call overwrites the first's buffer, so the entrance
%   data is read out before the exit trace runs.
    cdir = [sin(bx); sin(by); sqrt(max(0, 1 - sin(bx)^2 - sin(by)^2))];
    cpos = apst - stand*cdir;
    s = regexprep(txt, '(ChfRayDir=\s*)[^\n]*', ['$1' v3_(cdir)]);
    s = regexprep(s,   '(ChfRayPos=\s*)[^\n]*', ['$1' v3_(cpos)]);
    s = regexprep(s,   '(yGrid=\s*)[^\n]*', ['$1' v3_([0; cos(by); -sin(by)])]);
    fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
    macos.load_rx(tmp);
    if ~macos.has_rx()
        error('macos:design:pupil_map:load','deck failed to load: %s', tmp);
    end
    tr = macos.trace(io);   ro = macos.get_ray_info(tr.nRays);
    tr = macos.trace(ii);   re = macos.get_ray_info(tr.nRays);
    ok = ro.ok_trace(:) & ro.ok_pass(:) & re.ok_trace(:) & re.ok_pass(:);
    ok(1) = false;                       % the chief is the frame, not a node
    Pm = ro.pos;   Pe = re.pos;   De = re.dir;
    chief = re.dir(:,1)/norm(re.dir(:,1));
end

function [mg, an] = field_mag_(Sk, Vp, np, e1, e2)
%FIELD_MAG_  One field's affine anchor->plane map, and its scale.
%   NP is the evaluation plane's normal and (E1,E2) an orthonormal basis of
%   that plane; the station VP is the same whichever normal is passed, so
%   the only difference between the placed-plane and chief-normal calls is
%   the OBLIQUITY of the plane the footprint is read on.  MG is the
%   geometric-mean demagnification (1/sqrt of the areal Jacobian) and AN the
%   anamorphic ratio -- which is where an oblique read shows up first, since
%   the 1/cos stretch acts only in the plane of incidence.
    t  = (np.'*(Vp - Sk.pe)) ./ (np.'*Sk.de);
    q  = Sk.pe + Sk.de.*t;
    wq = [e1.'*(q - Vp); e2.'*(q - Vp)];
    A  = [Sk.m.', ones(size(Sk.m,2),1)];
    L  = [A\wq(1,:).', A\wq(2,:).'].';
    sv = svd(L(:,1:2));
    mg = 1/sqrt(prod(sv));
    an = sv(1)/sv(2);
end

function [pe, de, res] = regrid_(m, pe_s, de_s, nod)
%REGRID_  Interpolate the exit-side ray data onto common entrance nodes.
%   The independent variable is the ray's OWN hit point on the entrance
%   surface, so the result is anchored there by construction.  RES is the
%   anchoring residual estimate: the spread between two independent
%   interpolants (linear and natural) on the same data.  Nodes outside a
%   field's convex hull come back NaN -- that is a rim PARTIAL cone, kept.
    pe = nan(3, size(nod,2));   de = nan(3, size(nod,2));
    res = zeros(1, size(nod,2));
    for c = 1:3
        Fl = scatteredInterpolant(m(1,:).', m(2,:).', pe_s(c,:).', 'linear','none');
        Fn = scatteredInterpolant(m(1,:).', m(2,:).', pe_s(c,:).', 'natural','none');
        a = Fl(nod(1,:).', nod(2,:).');   b = Fn(nod(1,:).', nod(2,:).');
        pe(c,:) = a.';
        res = max(res, abs(a-b).');
        Gl = scatteredInterpolant(m(1,:).', m(2,:).', de_s(c,:).', 'linear','none');
        de(c,:) = Gl(nod(1,:).', nod(2,:).').';
    end
    n = vecnorm(de);   de = de ./ n;
    res(~isfinite(res)) = 0;
end

function s = regrid_scalar_(m, val, nod)
    F = scatteredInterpolant(m(1,:).', m(2,:).', val(:), 'linear','none');
    s = F(nod(1,:).', nod(2,:).').';
end

function W = wander_(PE, DE, Vp, np, b1, b2, good, Rex)
%WANDER_  Where each cone's rays pierce a plane, and how far apart.
    J = size(PE,2);
    r  = nan(1,J);  mx = nan(1,J);   ctr = nan(2,J);
    for j = 1:J
        if ~good(j), continue; end
        sel = find(all(isfinite(PE(:,j,:)),1));
        p = reshape(PE(:,j,sel), 3, []);   d = reshape(DE(:,j,sel), 3, []);
        t = (np.'*(Vp - p)) ./ (np.'*d);
        q = p + d.*t;
        c = mean(q,2);
        e = vecnorm(q - c);
        r(j) = rms(e);  mx(j) = max(e);
        ctr(:,j) = [b1.'*(c - Vp); b2.'*(c - Vp)];
    end
    W = struct('per_node_rms',r, 'per_node_max',mx, 'centre',ctr, ...
               'rms',rms(r(good)), 'max',max(mx(good)), ...
               'frac_rms',rms(r(good))/Rex, 'frac_max',max(mx(good))/Rex);
end

function B = best_plane_(PE, DE, Vp, np, ne, b1, b2, good, Rex)
%BEST_PLANE_  Slide and tilt the interface plane to minimise the wander.
%   Three parameters (one shift along the chief, two tilts) -- the analogue
%   of tuning a coldstop's position and DAR tilt.  Reported ALONGSIDE the
%   deck's own placement, never instead of it.
    f = @(p) plane_cost_(PE, DE, Vp + ne*p(1), tiltn_(np, b1, b2, p(2), p(3)), good);
    p0 = [0 0 0];
    p  = fminsearch(f, p0, optimset('Display','off','TolX',1e-9,'TolFun',1e-14));
    Vb = Vp + ne*p(1);   nb = tiltn_(np, b1, b2, p(2), p(3));
    W  = wander_(PE, DE, Vb, nb, b1, b2, good, Rex);
    B  = W;
    B.shift    = p(1);
    B.tilt_deg = acosd(min(1, abs(dot(nb, np))));
    B.Vpt = Vb.';   B.psi = nb.';
end

function c = plane_cost_(PE, DE, Vp, np, good)
    J = size(PE,2);   s = 0;  n = 0;
    for j = 1:J
        if ~good(j), continue; end
        sel = find(all(isfinite(PE(:,j,:)),1));
        p = reshape(PE(:,j,sel), 3, []);   d = reshape(DE(:,j,sel), 3, []);
        t = (np.'*(Vp - p)) ./ (np.'*d);
        q = p + d.*t;
        e = q - mean(q,2);
        s = s + sum(sum(e.^2));   n = n + numel(sel);
    end
    c = sqrt(s/max(n,1));
end

function n = tiltn_(np, b1, b2, t1, t2)
    n = np + b1*t1 + b2*t2;   n = n/norm(n);
end

function I = pupil_image_(PE, DE, Vp, np, b1, b2, w, good, Rex, nz, npix, K)
%PUPIL_IMAGE_  |cone spot|^2 convolved with the entrance amplitude map.
%   Incoherent by construction: the field angles are mutually incoherent, so
%   the cone's piercings add in INTENSITY.  The kernel is built per radial
%   ZONE when nz > 1 -- a single kernel assumes isoplanatism, which fails
%   exactly at the rim where the coldstop margin is decided.
    ext = 1.25*Rex;
    ax  = linspace(-ext, ext, npix);
    dx  = ax(2)-ax(1);
    rw  = vecnorm(w);
    edges = linspace(0, max(rw(good))*1.0001, nz+1);
    img = zeros(npix);
    kern = cell(1,nz);
    for z = 1:nz
        inz = good & rw >= edges(z) & rw <= edges(z+1);
        if ~any(inz), kern{z} = []; continue; end
        % kernel: piercing offsets about each cone's own centre
        off = [];
        for j = find(inz)
            sel = find(all(isfinite(PE(:,j,:)),1));
            p = reshape(PE(:,j,sel), 3, []);   d = reshape(DE(:,j,sel), 3, []);
            t = (np.'*(Vp - p)) ./ (np.'*d);
            q = p + d.*t;   c = mean(q,2);
            off = [off, [b1.'*(q-c); b2.'*(q-c)]]; %#ok<AGROW>
        end
        Kk = hist2_(off, ax, ax);
        if sum(Kk(:)) > 0, Kk = Kk/sum(Kk(:)); end
        kern{z} = Kk;
        % amplitude map for this zone: the mapped entrance transmission
        A = zone_mask_(ax, w(:,inz), dx);
        img = img + conv2(A, Kk, 'same');
    end
    I = struct('x',ax, 'y',ax, 'img',img, 'kernel',{kern}, ...
               'zone_edges',edges, 'n_zones',nz, 'n_field_samples',K, ...
               'pixel_m',dx, 'pupil_radius',Rex);
end

function H = hist2_(pts, ax, ay)
    H = zeros(numel(ay), numel(ax));
    if isempty(pts), return; end
    ix = round((pts(1,:)-ax(1))/(ax(2)-ax(1))) + 1;
    iy = round((pts(2,:)-ay(1))/(ay(2)-ay(1))) + 1;
    ok = ix >= 1 & ix <= numel(ax) & iy >= 1 & iy <= numel(ay);
    for i = find(ok), H(iy(i),ix(i)) = H(iy(i),ix(i)) + 1; end
end

function A = zone_mask_(ax, wz, dx)
%ZONE_MASK_  1 where the zone's mapped entrance nodes fall.  Built from the
%   TRACED map, so the hole, the gaps and the rim shape come from the deck
%   rather than from an analytic aperture model.
    A = zeros(numel(ax));
    ix = round((wz(1,:)-ax(1))/dx) + 1;
    iy = round((wz(2,:)-ax(1))/dx) + 1;
    ok = ix >= 1 & ix <= numel(ax) & iy >= 1 & iy <= numel(ax);
    for i = find(ok), A(iy(i),ix(i)) = 1; end
    % fill between nodes: the node lattice is coarser than the raster
    du = diff(sort(unique(wz(1,:))));
    if isempty(du) || ~isfinite(median(du)), du = dx; end
    rad = max(1, ceil(0.5*median(du)/dx));
    A = conv2(A, ones(2*rad+1), 'same') > 0;
    A = double(A);
end

% =====================================================================
function [a1,a2] = perp_(n)
    n = n(:)/norm(n);   t = [1;0;0];
    if abs(n.'*t) > 0.9, t = [0;1;0]; end
    a1 = t - (n.'*t)*n;   a1 = a1/norm(a1);   a2 = cross(n,a1);
end

function B = zbasis_(u,v)            % macos.pupil_quality's basis, verbatim
    r2 = u.^2 + v.^2;
    B = [ones(size(u)), u, v, 2*r2-1, u.^2-v.^2, 2*u.*v, ...
         (3*r2-2).*u, (3*r2-2).*v, 6*r2.^2-6*r2+1];
end

function [c,names] = zfit_(u,v,s)
    names = {'piston','tilt_x','tilt_y','defocus','astig0','astig45', ...
             'coma_x','coma_y','spherical'};
    c = zbasis_(u,v) \ s;
end

function s = v3_(v),  s = sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));  end

function [cdir, cpos, apst, lam] = deck_src_(txt)
    cdir = grab3_(txt,'ChfRayDir');   cpos = grab3_(txt,'ChfRayPos');
    apst = grab3_(txt,'ApStop');
    t = regexp(txt,'Wavelen=\s*([-\d.EeD+]+)','tokens','once');
    lam = str2double(strrep(t{1},'D','E'));
end

function v = grab3_(txt, key)
    t = regexp(txt,[key '=\s*([^\n]*)'],'tokens','once');
    v = sscanf(strrep(t{1},'D','E'),'%f',3);
end

function M = grab_all3_(txt, key)
    t = regexp(txt,[key '=\s*([^\n]*)'],'tokens');
    M = zeros(3,numel(t));
    for i = 1:numel(t), M(:,i) = sscanf(strrep(t{i}{1},'D','E'),'%f',3); end
end

function del_(p),  if exist(p,'file'), delete(p); end,  end
