classdef Telescope < handle
%MACOS.DESIGN.TELESCOPE  De-novo two-mirror telescope builder (Sprint 2A-ii).
%   The fixed-topology builder front-end of the design layer
%   (PLAN_DESIGN_LAYER §1.0/§2/§5).  The user states design intent
%   (family + first-order parameters); the builder derives the full
%   first-order layout and conic constants in closed form (Schroeder
%   (m,β) convention, optical_design/TELESCOPE_DESIGN_REFERENCE.md),
%   emits a MACOS prescription, and validates it by loading through
%   SMACOS.  Everything downstream (vary / evaluate / optimize) is the
%   shared analysis core — import the emitted Rx with
%   macos.design.System.from_rx(t.build()).
%
%   Families (2-mirror): Cassegrain, RC, Gregorian, Dall-Kirkham.
%
%   Example (PLAN_DESIGN_LAYER §2, Stage 1-2):
%       t  = macos.design.Telescope('family','RC', ...
%               'aperture_diameter_mm',6000, 'primary_fnum',2.0, ...
%               'system_fnum',20.0, 'BFD_mm',1000, 'model_size',256);
%       rx = t.build();        % derive -> emit .in -> validate-by-load
%       t.describe();          % every derived value + provenance
%
%   Convention (validated 2026-06-16 against the shared fixtures to RMS
%   WFE ~1e-15 m on-axis, see reference memory):  KcElt = K directly;
%   KrElt = -|R|;  psiElt -> centre of curvature (one rule, all
%   surfaces: concave M1 and convex Cass secondary point -z, concave
%   Gregorian secondary points +z); the trailing nOutCord/Tout block is
%   REQUIRED for the SMACOS load.  Light travels +z, source at -z.
%
%   See also: macos.design.System, macos.load_rx.

    properties (SetAccess = private)
        spec   % plain struct — the design spec (state-as-data, §3)
    end

    properties (Constant, Access = private)
        FAMILIES = {'cassegrain','ritchey_chretien','gregorian','dall_kirkham','tma'}
        NMIRROR_FAMILIES = {'tma'}     % built via add_mirror (vs auto 2-mirror)
        ALIASES  = struct('cass','cassegrain', 'classicalcassegrain','cassegrain', ...
                          'rc','ritchey_chretien', 'ritchey','ritchey_chretien', ...
                          'ritcheychretien','ritchey_chretien', ...
                          'greg','gregorian', 'classicalgregorian','gregorian', ...
                          'dk','dall_kirkham', 'dallkirkham','dall_kirkham', ...
                          'tma','tma', 'threemirror','tma', 'korsch','tma', ...
                          'threemirroranastigmat','tma')
    end

    methods
        function obj = Telescope(opts)
        %TELESCOPE  Construct a two-mirror telescope from design intent.
        %   Name-value (SI canonical; mm sugar accepted, §10 Made #11):
        %     'family'              one of Cassegrain / RC / Gregorian /
        %                           Dall-Kirkham (aliases ok).  Required.
        %     'aperture_diameter_m' | 'aperture_diameter_mm'  (one req.)
        %     'system_fnum'         system f/# (= EFL/D).  Required.
        %     'primary_fnum'        primary f/# (= f1/D).   Required.
        %     'BFD_m' | 'BFD_mm'    back focal distance (vertex->focus).
        %                           One required.
        %     'optical_axis'        default [0 0 1] (only +z in MVP).
        %     'model_size'          engine model size (default 256).
        %     'wavelength_m'        layout/eval wavelength (default 633e-9).
            arguments
                opts.family              (1,:) char
                opts.aperture_diameter_m  (1,1) double = NaN   % validated in body
                opts.aperture_diameter_mm (1,1) double = NaN   % (NaN default can't
                opts.system_fnum         (1,1) double = NaN    %  carry mustBePositive)
                opts.primary_fnum        (1,1) double = NaN
                opts.BFD_m               (1,1) double = NaN
                opts.BFD_mm              (1,1) double = NaN
                opts.optical_axis        (1,3) double = [0 0 1]
                opts.model_size          (1,1) double {mustBeInteger,mustBePositive} = 256
                opts.wavelength_m        (1,1) double {mustBePositive} = 633e-9
                opts.grid_npts           (1,1) double {mustBeInteger,mustBePositive} = 41
            end
            if ~isfield(opts,'family') || isempty(opts.family)
                error('macos:design:Telescope:family', ...
                    'family is required (Cassegrain/RC/Gregorian/Dall-Kirkham/TMA).');
            end
            fam = obj.canon_family_(opts.family);
            D   = obj.pick_len_(opts.aperture_diameter_m, opts.aperture_diameter_mm, ...
                                'aperture_diameter');
            if ~isequal(opts.optical_axis, [0 0 1])
                error('macos:design:Telescope:axis', ...
                    'MVP supports optical_axis [0 0 1] only (got [%g %g %g]).', ...
                    opts.optical_axis);
            end

            sp = struct();
            sp.source      = 'builder';
            sp.family      = fam;
            sp.model_size  = opts.model_size;
            sp.wavelength  = opts.wavelength_m;          % SI metres
            sp.field_points = [0 0];                     % on-axis (rad); set_field_points overrides
            sp.field_bias   = 0;                         % nominal +y field-bias half-angle (rad); set_field_bias overrides
            sp.aperture_decenter = 0;                    % +y beam/stop offset from the on-axis vertex (m); set_aperture_decenter overrides
            sp.sampling    = opts.grid_npts;             % circular grid (geometric default)
            sp.in.D        = D;

            if any(strcmp(fam, obj.NMIRROR_FAMILIES))
                % N-mirror (TMA...): mirrors come via add_mirror; the
                % layout + Seidel-seeded conics resolve at build() time.
                sp.is_nmirror = true;
                sp.mirrors    = obj.empty_mirror_list_();
                sp.fp_name    = 'FP';
                sp.elt        = [];                      % unresolved until build()
                obj.spec      = sp;
            else
                % 2-mirror families: full closed form from the intent numbers.
                BFD = obj.pick_len_(opts.BFD_m, opts.BFD_mm, 'BFD');
                if isnan(opts.system_fnum) || isnan(opts.primary_fnum)
                    error('macos:design:Telescope:fnum', ...
                        'both system_fnum and primary_fnum are required (2-mirror).');
                end
                if ~(opts.system_fnum > 0) || ~(opts.primary_fnum > 0)
                    error('macos:design:Telescope:fnumSign', ...
                        'system_fnum and primary_fnum must be positive.');
                end
                sp.in.system_fnum  = opts.system_fnum;
                sp.in.primary_fnum = opts.primary_fnum;
                sp.in.BFD          = BFD;
                obj.spec = sp;
                obj.resolve_();                          % derive layout + conics + elements
            end
        end

        function add_mirror(obj, name, opts)
        %ADD_MIRROR  Append a mirror to an N-mirror (TMA) telescope.
        %   t.add_mirror(NAME, 'radius_m',R, 'spacing_after_m',T) appends a
        %   coaxial mirror of vertex radius R (magnitude; emitted as
        %   KrElt=-|R|, psiElt=(0,0,-1)) at vertex spacing T from the
        %   previous mirror.  The LAST mirror's spacing is the derived
        %   paraxial focus -- give it 'spacing_after','derive'.  Conics are
        %   Seidel-seeded (null S_I/II/III) at build().  Radius accepts
        %   'radius_m'/'radius_mm', spacing 'spacing_after_m'/'_mm'.
        %
        %   RADIUS IS A MAGNITUDE (> 0) for ALL mirrors, convex included.
        %   A convex secondary is NOT a sign-flipped radius -- in MACOS it is
        %   KrElt=-|R| (like any mirror) made convex by GEOMETRY: the secondary
        %   sits BEFORE the M1 focus (Cassegrain spacing, t1 < f1), so the beam
        %   reflects away from the centre of curvature (j18mono's convex SM).
        %   The Seidel seed's n-flip paraxial model also wants magnitudes.
        %   In the TILTED-fold path, pass 'convex',true for a secondary whose
        %   centre of curvature is downstream of the vertex (e.g. a Cassegrain
        %   SM): resolve_nmirror_fold_ then emits psiElt pointing to that
        %   downstream CoC (the coaxial path infers convex from geometry).
            arguments
                obj
                name (1,:) char
                opts.radius_m         (1,1) double = NaN
                opts.radius_mm        (1,1) double = NaN
                opts.spacing_after_m  (1,1) double = NaN
                opts.spacing_after_mm (1,1) double = NaN
                opts.spacing_after    (1,:) char   = ''
                opts.tilt_deg         (1,1) double = 0     % fold tilt about x (Bauer)
                opts.convex           (1,1) logical = false % convex: psi->downstream CoC
                opts.conic            (1,1) double = NaN   % explicit Kc seed (skips seidel)
            end
            if ~obj.is_nmirror_()
                error('macos:design:Telescope:add_mirror:family', ...
                    'add_mirror is for N-mirror families (family=%s).', obj.spec.family);
            end
            R = obj.pick_len_(opts.radius_m, opts.radius_mm, 'radius', true);
            derive = strcmpi(strtrim(opts.spacing_after), 'derive');
            if derive
                t = NaN;
            else
                t = obj.pick_len_(opts.spacing_after_m, opts.spacing_after_mm, ...
                                  'spacing_after');
            end
            % tilt_deg folds the chief ray at this mirror (Bauer/Schiesser/Rolland
            % unobscuring -- TILT minimally to clear, not decenter).  0 = coaxial
            % (the legacy path, byte-identical).  See resolve_nmirror_fold_.
            % opts.conic: an EXPLICIT Kc seed for this mirror (NaN = let
            % seidel_seed provide it).  Use when carrying optimized conics
            % into a longer chain -- e.g. the 3+1: seidel cannot seed a
            % relay-past-focus 4-mirror chain (the degenerate-reimager
            % regime), so seed M1-M3 from the optimized TMA and M4 = 0.
            obj.spec.mirrors(end+1) = struct('name',name, 'R',R, 't',t, ...
                'derive',derive, 'tilt_deg',opts.tilt_deg, 'convex',opts.convex, ...
                'conic',opts.conic);
            obj.spec.elt = [];                           % invalidate -> re-resolve
        end

        function set_base_sphere(obj, tf)
        %SET_BASE_SPHERE  Hold all mirror base surfaces as spheres (Kc=0).
        %   When true, resolve_nmirror_/_fold_ skip the Seidel conic seed and
        %   emit pure spheres (KcElt=0); ALL aberration correction is then
        %   carried by the Zernike departures (set_freeform + optimize_freeform).
        %   This is the e5mono sphere+Zernike model -- the 0th-order layout
        %   (radii + fold) sets first-order f/# and packaging, the Zernikes do
        %   the rest, so geometry and correction fully decouple.
            arguments, obj, tf (1,1) logical = true, end
            obj.spec.base_sphere = tf;
            obj.spec.elt = [];                           % invalidate -> re-resolve
        end

        function add_focal_plane(obj, name, opts)
        %ADD_FOCAL_PLANE  Name the terminal focal plane of an N-mirror
        %   telescope (default 'FP'); placed at the derived focus at build.
        %   'ap_r' sets the PHYSICAL body radius (m) used by check_clipping
        %   -- default 0.3*D is a generous placeholder; for an honest
        %   clearance verdict size it to the real detector + structure
        %   (image extent = EFL * field radius, plus housing).
            arguments
                obj
                name (1,:) char = 'FP'
                opts.ap_r (1,1) double = NaN
            end
            if ~obj.is_nmirror_()
                error('macos:design:Telescope:add_focal_plane:family', ...
                    'add_focal_plane is for N-mirror families.');
            end
            obj.spec.fp_name = name;
            obj.spec.fp_ap_r = opts.ap_r;                % NaN -> 0.3*D default
            obj.spec.elt     = [];
        end

        function add_fold(obj, name, opts)
        %ADD_FOLD  Insert a FLAT fold mirror into an N-mirror chain.
        %   t.add_fold('FM','after','M2','dist_m',0.5)  puts a flat 0.5 m
        %   along the beam after mirror M2 and folds the beam into 'to'
        %   (default [0 1 0]: 90 deg into +y).  Everything DOWNSTREAM of the
        %   fold (mirrors, focal plane, their psi) is mapped by the fold
        %   plane's reflection isometry, so path lengths and angles are
        %   EXACTLY preserved -- a flat fold adds zero aberration (image
        %   parity flips).  Folds compose: chain a second fold by naming the
        %   first as 'after'.
        %
        %   THE use case (the centered/Korsch family): the coaxial TMA's
        %   focal plane lands ON AXIS in the middle of the incoming beam.
        %   A field bias separates the M2->M3 feed and the M3->FP return
        %   bundles laterally near the exit pupil; a small fold there picks
        %   off ONE of them and moves the back end into the x-y plane
        %   behind the primary.  Use design/src/fold_station_report to find
        %   a station where the two bundles have daylight between them, and
        %   set_hole to stop check_clipping charging the through-the-hole
        %   passes to a perforated primary.
        %
        %   Name-value:
        %     'after'   (required) name of the element the fold follows
        %     'dist_m'  (required) station distance (m) along the beam after
        %               that element; must be < the spacing to the next one
        %     'to'      outgoing beam direction, global (default [0 1 0])
        %     'ap_r'    fold body radius (m); default 0.1*D.  Size it to the
        %               local beam footprint -- check_clipping judges it.
        %
        %   PLANNED (Dave 2026-07-05): a 'radius_m' option for WEAK POWER on
        %   the fold (slow focus/pupil trim without a dedicated powered
        %   mirror).  Until then folds are strictly flat (Kr sentinel -1e22);
        %   a powered fold also needs enrollment in optimize's ROC DOF and
        %   an astigmatism warning at large fold angles.
            arguments
                obj
                name (1,:) char
                opts.after  (1,:) char
                opts.dist_m (1,1) double
                opts.to     (1,3) double = [0 1 0]
                opts.ap_r   (1,1) double = NaN
            end
            if ~obj.is_nmirror_()
                error('macos:design:Telescope:add_fold:family', ...
                    'add_fold is for N-mirror families (family=%s).', obj.spec.family);
            end
            if ~isfield(opts,'after') || ~isfield(opts,'dist_m')
                error('macos:design:Telescope:add_fold:args', ...
                    'add_fold requires ''after'' (element name) and ''dist_m''.');
            end
            if ~(opts.dist_m > 0)
                error('macos:design:Telescope:add_fold:dist', ...
                    'dist_m must be positive (got %g).', opts.dist_m);
            end
            if norm(opts.to) < eps
                error('macos:design:Telescope:add_fold:to', ...
                    '''to'' direction must be non-zero.');
            end
            f = struct('name',name, 'after',opts.after, 'dist',opts.dist_m, ...
                       'to',opts.to./norm(opts.to), 'ap_r',opts.ap_r);
            if ~isfield(obj.spec,'folds') || isempty(obj.spec.folds)
                obj.spec.folds = f;
            else
                obj.spec.folds(end+1) = f;
            end
            obj.spec.elt = [];                           % invalidate -> re-resolve
        end

        function d = center_focal_plane(obj)
        %CENTER_FOCAL_PLANE  Move the focal-plane BODY to the traced beam
        %   centroid at the FP -- body placement only: a lateral shift of a
        %   flat focal plane within its own plane is trace-neutral.  Use
        %   after set_field_bias and/or add_fold, where the image walks off
        %   the derived on-axis FP center and check_clipping would judge
        %   the body in the wrong place (and a real detector would sit in
        %   the wrong place).  Returns the offset applied (m).
            obj.ensure_loaded_();
            if ~macos.has_rx(), obj.build(); else, obj.build('','init',false); end
            e  = obj.spec.elt;  nE = numel(e);
            fk = find(strcmp({e.kind},'FocalPlane'), 1, 'last');
            if isempty(fk)
                error('macos:design:Telescope:center_fp:none', ...
                    'no FocalPlane element.');
            end
            % Beam centroid at the FP from engine-truth GLOBAL ray
            % landings (trace to fk + get_ray_info) -- the previous
            % draw_rays('YZ'/'XZ') route read the DRAW plot projection,
            % whose axis signs follow the source-grid handedness (the
            % 2026-07-18 heritage xGrid=(-1,0,0) emission flipped its U
            % axis and sent the FP a metre off).
            sc = macos.trace(fk);
            b  = macos.get_ray_info(sc.nRays);
            ok = logical(b.ok_trace) & logical(b.ok_pass);
            if ~any(ok)
                error('macos:design:Telescope:center_fp:trace', ...
                    'no rays reach the focal plane.');
            end
            ctr = mean(b.pos(:, ok), 2);
            d = norm(ctr.' - e(fk).Vpt);
            obj.spec.elt(fk).Vpt = ctr.';
            macos.trace(nE);                    % leave a full trace behind
        end

        function res = align_focal_plane(obj, opts)
        %ALIGN_FOCAL_PLANE  Place AND tilt the focal plane from multi-field
        %   best foci.  center_focal_plane translates the detector body
        %   only; on a field-biased design the TRUE focal plane is TILTED
        %   with respect to the chief ray, and at least 3 non-collinear
        %   field points are needed to identify that tilt (Dave 2026-07-06).
        %   This traces a small field set about the (biased) field center,
        %   finds each field's best-focus point in 3-D -- the least-squares
        %   closest point to that field's arriving ray bundle, no scan --
        %   fits a plane through the foci, and sets the FocalPlane Vpt
        %   (center-field focus, projected onto the plane) + psi (plane
        %   normal, oriented along the arriving chief) from it.
        %
        %   Run BEFORE add_pupil: the inserted FP_return/ExitPupil are
        %   derived from the FP station and would go stale.
        %
        %   Name-value:
        %     'grid'         N -> map the foci on an N x N field grid over
        %                    +/- span (Dave 2026-07-06: start 2x2 for
        %                    prelim analysis, refine to 5x5 / 7x7 for the
        %                    final design).  Default 0 = the minimal
        %                    4-point cross.  The field CENTER is always
        %                    traced too, as the anchor (field 1).
        %     'span_arcmin'  half-span of the grid / cross radius
        %                    (default 0.25').
        %     'fields'       explicit (N,2) rad offsets (>=2, plus the
        %                    auto-anchor; must not be collinear with it);
        %                    supersedes grid/span.
        %     'allow_pupil'  run even when add_pupil has already inserted the
        %                    FP_return/ExitPupil pair (default false, which
        %                    keeps the original hard error).  Needed for the
        %                    exit-pupil-referenced optimisation loop, which must
        %                    ALTERNATE solve <-> re-fit the detector with the
        %                    pupil in place.  Safe there: CALIB re-runs FEX per
        %                    field, re-deriving the ExitPupil pose from scratch,
        %                    and FP_return's station CANCELS out of the
        %                    Return-pair OPL (the ray retraces the same line, so
        %                    the leg it adds is the leg the second Return
        %                    subtracts).  Only the terminal FocalPlane -- the
        %                    frozen detector the metric is tied to -- is updated.
        %
        %   Returns res: .fields (N,2 rad, row 1 = center anchor), .foci
        %   (3xN), .fp_vpt, .psi, .tilt_deg (plane normal vs arriving
        %   chief), .defocus_m (along-chief shift of the center-field
        %   focus from the old FP station), .sag_m (1xN signed
        %   out-of-plane residual per focus -- the FIELD-CURVATURE map
        %   over the grid), .spot_rms_m (per-field residual blur at best
        %   focus), .fit_rms_m (plane-fit residual = rms(sag_m)).
            arguments
                obj
                opts.grid (1,1) double {mustBeInteger,mustBeNonnegative} = 0
                opts.span_arcmin (1,1) double {mustBePositive} = 0.25
                opts.fields (:,2) double = zeros(0,2)
                opts.allow_pupil (1,1) logical = false
            end
            obj.ensure_loaded_();
            if ~macos.has_rx(), obj.build(); else, obj.build('','init',false); end
            e  = obj.spec.elt;
            if any(strcmp({e.name}, 'FP_return')) && ~opts.allow_pupil
                error('macos:design:Telescope:align_fp:afterPupil', ...
                    'align_focal_plane must run BEFORE add_pupil.');
            end
            fk = find(strcmp({e.kind},'FocalPlane'), 1, 'last');
            if isempty(fk)
                error('macos:design:Telescope:align_fp:none', ...
                    'no FocalPlane element.');
            end
            F = opts.fields;
            if isempty(F)
                if opts.grid > 0
                    F = macos.design.field_grid(opts.span_arcmin, ...
                            opts.grid, 'units','arcmin');
                    F = F(any(abs(F) > 1e-12, 2), :);  % center re-added below
                else
                    r = opts.span_arcmin * pi/180/60;
                    F = [r 0; -r 0; 0 r; 0 -r];
                end
            end
            F = [0 0; F];                    % field 1 = the center anchor
            if size(F,1) < 3
                error('macos:design:Telescope:align_fp:nfields', ...
                    'need >= 3 field points to identify the FP tilt.');
            end
            nF   = size(F,1);
            foci = zeros(3,nF);  blur = zeros(1,nF);
            dch  = zeros(3,1);
            for f = 1:nF
                obj.trace_at_field(F(f,:));
                s = macos.trace(fk-1);  a = macos.get_ray_info(s.nRays);
                macos.trace(fk);        b = macos.get_ray_info(s.nRays);
                ok = logical(a.ok_trace(:)) & logical(a.ok_pass(:)) & ...
                     logical(b.ok_trace(:)) & logical(b.ok_pass(:));
                if nnz(ok) < 10
                    obj.trace_at_field([]);
                    error('macos:design:Telescope:align_fp:rays', ...
                        'field [%g %g]: only %d rays reach the FP.', ...
                        F(f,1), F(f,2), nnz(ok));
                end
                P = b.pos(:,ok);
                D = b.pos(:,ok) - a.pos(:,ok);  D = D ./ vecnorm(D);
                % least-squares closest point to the ray bundle:
                % min_X sum ||(I - d d')(X - p)||^2
                A = nnz(ok)*eye(3) - D*D.';
                bb = sum(P,2) - D*sum(D.*P,1).';
                X = A \ bb;
                foci(:,f) = X;
                R = X - P;                           % ray->focus offsets
                R = R - D .* sum(D.*R,1);            % transverse part
                blur(f) = sqrt(mean(sum(R.^2,1)));
                if f == 1, dch = mean(D,2); dch = dch/norm(dch); end
            end
            obj.trace_at_field([]);                  % restore nominal
            % plane through the foci
            C = mean(foci,2);
            [U,S,~] = svd(foci - C, 'econ');
            sv = diag(S);
            if sv(2) < 1e3*sv(3) || sv(2) < 1e-12
                warning('macos:design:Telescope:align_fp:collinear', ...
                    ['foci are near-collinear (sv2/sv3 = %.1e); the ' ...
                     'fitted tilt is poorly determined.'], sv(2)/max(sv(3),eps));
            end
            nrmv = U(:,3);
            nrmv = nrmv * sign(dot(nrmv, dch));      % FP psi convention:
            sag  = nrmv.' * (foci - C);              % along arriving chief
            Vnew = foci(:,1) - nrmv*dot(nrmv, foci(:,1)-C);  % center focus on plane
            oldV = e(fk).Vpt(:);
            res = struct('fields',F, 'foci',foci, 'fp_vpt',Vnew.', ...
                'psi',nrmv.', ...
                'tilt_deg', acosd(min(1,abs(dot(nrmv,dch)))), ...
                'defocus_m', dot(Vnew - oldV, dch), ...
                'sag_m', sag, 'spot_rms_m', blur, ...
                'fit_rms_m', sqrt(mean(sag.^2)));
            if opts.grid > 0
                % ready-to-plot N x N field-curvature map (arcmin axes).
                % Odd grids: the (0,0) slot was deduped into the anchor;
                % put its sag back so the matrix is complete.
                gm = F(2:end,:);  sv = sag(2:end);
                if mod(opts.grid,2) == 1
                    gm = [gm; 0 0];  sv = [sv, sag(1)];
                end
                [gm, io] = sortrows(gm);  sv = sv(io);
                n = opts.grid;
                res.map = struct( ...
                    'thx_arcmin', reshape(gm(:,1),n,n)*180*60/pi, ...
                    'thy_arcmin', reshape(gm(:,2),n,n)*180*60/pi, ...
                    'sag_m',      reshape(sv,n,n));
            end
            obj.spec.elt(fk).Vpt = Vnew.';
            obj.spec.elt(fk).psi = nrmv.';
            obj.build('', 'init', false);            % re-emit the tilted FP
        end

        function set_hole(obj, name, r_m)
        %SET_HOLE  Declare a central perforation (hole) of radius R_M in the
        %   named element's body.  Geometry/trace are UNCHANGED -- this only
        %   informs check_clipping: a foreign-beam crossing within R_M of the
        %   body center passes THROUGH the hole and is not an obstruction.
        %   The centered (Korsch/Cassegrain) families need this for their
        %   perforated primary -- without it every through-the-hole pass of
        %   the M2->M3 beam is charged to M1 as a false body-in-beam hit.
        %   r_m = 0 removes the hole.  The hole is ALSO emitted into the Rx
        %   as a real ObsType=Circle obscuration centered on the vertex
        %   (2026-07-18): the trace clips the central rays (no glass there
        %   -- physically honest) and layout views (macos.view_rx /
        %   view_std) render the hole.
            arguments, obj, name (1,:) char, r_m (1,1) double {mustBeNonnegative}, end
            h = struct('name',name, 'r',r_m);
            if ~isfield(obj.spec,'holes') || isempty(obj.spec.holes)
                obj.spec.holes = h;
            else
                i = find(strcmp({obj.spec.holes.name}, name), 1);
                if isempty(i), obj.spec.holes(end+1) = h;
                else,          obj.spec.holes(i)     = h; end
            end
        end

        function set_field_points(obj, fp)
        %SET_FIELD_POINTS  Field points (Nx2, radians) for evaluation.
        %   Per-eval state (not emitted into geometry); the on-axis
        %   layout is what build() writes.
            arguments, obj, fp (:,2) double, end
            obj.spec.field_points = fp;
        end

        function set_bandwidth(obj, wvl)
        %SET_BANDWIDTH  Wavelength list (SI metres).  nλ=1 default is the
        %   all-reflective policy (§1.3.6); the first λ is the layout λ.
            arguments, obj, wvl (1,:) double {mustBePositive}, end
            obj.spec.wavelength = wvl(1);
            obj.spec.bandwidth  = wvl;
        end

        function set_field_bias(obj, bias_arcmin)
        %SET_FIELD_BIAS  Take the on-axis design OFF-AXIS by biasing the
        %   nominal chief ray in +y by BIAS_ARCMIN (a half-angle).  The
        %   element vertices stay PINNED on-axis and psi stays axis-aligned
        %   -- only the source chief ray tilts, so the beam runs through a
        %   different OFF-AXIS part of the same on-axis parents (the
        %   e5mono/dmt6mono "design on-axis, then move off-axis" recipe;
        %   PLAN_DESIGN_LAYER §8).  build() emits the biased ChfRayDir;
        %   optimize() then re-derives the conics for the biased field.
        %   bias_arcmin = 0 restores the on-axis design exactly.
            arguments, obj, bias_arcmin (1,1) double, end
            obj.spec.field_bias = deg2rad(bias_arcmin/60);   % store radians
        end

        function set_aperture_decenter(obj, dy_m)
        %SET_APERTURE_DECENTER  Take the design off-axis by offsetting the
        %   beam/aperture-stop center in +y by DY_M (metres) from the
        %   on-axis vertex -- the beam then uses an OFF-AXIS PART of the
        %   same pinned parents (off-axis-parabola style: it converges to
        %   focus from one side, clear of the incoming cone).  Vertices and
        %   psi are unchanged; only the source ApStop + ChfRayPos shift.
        %   Complements set_field_bias (which tilts the chief ray); the two
        %   compose.  dy_m = 0 restores the centered design.
            arguments, obj, dy_m (1,1) double, end
            obj.spec.aperture_decenter = dy_m;
        end

        function d = set_offaxis(obj, clear, opts)
        %SET_OFFAXIS  Build an UNOBSCURED off-axis section: decenter the beam
        %   so a downstream optic clears the incoming cone, then emit each
        %   mirror as a true off-axis SECTION of its (unchanged) parent conic.
        %   This is the engine-true off-axis-parabola / eccentric-pupil
        %   representation -- VptElt = parent VERTEX, psiElt = parent AXIS,
        %   RptElt = the section POLE on the parent surface, TElt = the section
        %   frame (Z = outward surface normal at the pole).  ConSrf (surfsub.F)
        %   measures the conic sag from VptElt only, so RptElt is trace-neutral;
        %   it sets the PERTURB / sensitivity interface frame and the rigid-body
        %   rotation center.  Matches the JWST segmented model (j18sc: segments
        %   share one parent vertex, each carries its own off-axis pole + frame).
        %
        %   For an aplanatic parent (RC/Gregorian) an eccentric sub-aperture at
        %   the axial field is spherical- AND coma-free by construction, so the
        %   off-axis section traces diffraction-limited with NO re-optimization;
        %   the decenter only has to lift the secondary clear of the beam.
        %
        %   The off-axis distance is driven by clearing the optic(s) the
        %   designer is EXTRACTING from the beam -- NOT necessarily every body.
        %   Accepted obscurations stay: a JWST-like TMA keeps the central M2 in
        %   the beam and decenters only until M3 clears ('clear','M3'); an
        %   unobscured 2-mirror clears both mirrors ('clear','all').  For an RC
        %   the BINDING body is M1 (the M2->FP return beam crosses the M1 plane
        %   behind it) -- clearing M2 alone is NOT enough.
        %
        %   CLEAR is REQUIRED -- name the optic(s) the off-axis is FOR, so the
        %   intent is explicit (no presumed "clear everything"):
        %     'M3'          JWST-style: M3 out of the beam, M2 still obscures
        %     'all'         unobscured: every mirror clears
        %     {'M1','M2'}   a specific set
        %     'none'        no clearance solve -- pair with 'dist' (explicit
        %                   decenter) or apply sections at the current decenter
        %
        %   t.set_offaxis('M3')               % JWST: clear M3, M2 still central
        %   t.set_offaxis('all')              % unobscured: every mirror clears
        %   t.set_offaxis({'M1','M2'})        % name a specific set
        %   t.set_offaxis('none','dist',0.6)  % explicit +y decenter (metres)
        %   Name-value:
        %     'dist'     explicit +y decenter (m); omit -> clearance-driven
        %     'margin'   clearance margin as a fraction of D (default 0.05)
        %     'max_dist' bisection upper bound (m); default 1.5*D
        %   Returns the decenter distance used (m).
            arguments
                obj
                clear                              % REQUIRED: name | cellstr | 'all' | 'none'
                opts.dist     (1,1) double = NaN
                opts.margin   (1,1) double = 0.05
                opts.max_dist (1,1) double = NaN
            end
            D = obj.spec.in.D;
            if ~isnan(opts.dist)
                d = opts.dist;                     % explicit decenter
            elseif (ischar(clear) || isstring(clear)) && strcmpi(clear,'none')
                d = obj.spec.aperture_decenter;    % no solve; keep current decenter
            else
                hi = opts.max_dist;  if isnan(hi), hi = 1.5*D; end
                d  = obj.clearance_solve_(clear, opts.margin*D, hi);
            end
            obj.spec.aperture_decenter = d;
            obj.spec.offaxis_section   = true;
            obj.resolve_section_poles_();
        end

        function rx = build(obj, path, opts)
        %BUILD  Emit the prescription and validate it by loading via SMACOS.
        %   rx = t.build()           -> writes a temp .in, returns its path
        %   rx = t.build('foo.in')   -> writes foo.in
        %   Name-value: 'validate' (default true) load-checks the emitted
        %   Rx through SMACOS (the path pymacos/mmacos use); 'init'
        %   (default true) inits the engine at the spec model_size first;
        %   'check' (default false) runs check_clipping() on the loaded
        %   design and warns on any body-in-beam / vignetting conflict.
            arguments
                obj
                path (1,:) char = ''
                opts.validate (1,1) logical = true
                opts.init     (1,1) logical = true
                opts.check    (1,1) logical = false
            end
            if isempty(path), path = [tempname '.in']; end
            if obj.is_nmirror_() && (~isfield(obj.spec,'elt') || isempty(obj.spec.elt))
                obj.resolve_nmirror_();              % derive layout + conics once
            end
            txt = obj.emit_();
            fid = fopen(path, 'w');
            if fid < 0
                error('macos:design:Telescope:write', 'cannot open %s', path);
            end
            fprintf(fid, '%s', txt);
            fclose(fid);
            if opts.validate
                if opts.init, macos.init(obj.spec.model_size); end
                macos.load_rx(path);
                if ~macos.has_rx()
                    error('macos:design:Telescope:loadFailed', ...
                        'emitted Rx failed to load via SMACOS: %s', path);
                end
                if opts.check
                    rep = obj.check_clipping('noload', true, 'quiet', true);
                    if ~all([rep.ok])
                        bad = {rep(~[rep.ok]).name};
                        warning('macos:design:Telescope:clipping', ...
                            ['layout has body-in-beam / vignetting conflicts ' ...
                             'at: %s  (run check_clipping() for the report)'], ...
                            strjoin(bad, ', '));
                    end
                end
            end
            obj.spec.rx_path = path;
            rx = path;
        end

        function rx = save(obj, path)
        %SAVE  Emit the prescription .in (no validation/load).
            arguments, obj, path (1,:) char, end
            rx = obj.build(path, 'validate', false);
        end

        function save_spec(obj, path)
        %SAVE_SPEC  Persist the design spec struct (re-loadable, §2 Stage 6).
            arguments, obj, path (1,:) char, end
            spec = obj.spec; %#ok<NASGU>
            save(path, 'spec');
        end

        function describe(obj)
        %DESCRIBE  Print the resolved design table with provenance (§2).
            if obj.is_nmirror_()
                obj.describe_nmirror_();
                return;
            end
            sp = obj.spec; d = sp.derived;
            fprintf('macos.design.Telescope  (family=%s)\n', sp.family);
            fprintf('  inputs [user]:  D=%.6g m  system f/%.4g  primary f/%.4g  BFD=%.6g m\n', ...
                sp.in.D, sp.in.system_fnum, sp.in.primary_fnum, sp.in.BFD);
            fprintf('  derived(layout): EFL=%.6g m  f1=%.6g m  m=%.6g  beta=%.6g\n', ...
                d.f, d.f1, d.m, d.beta);
            fprintf('  %-8s %14s %14s   [provenance]\n', 'quantity', 'value', 'units');
            rows = {'R1',d.R1,'m'; 'R2',d.R2,'m'; 'M1_M2_sep',d.sep,'m'; ...
                    'BFD',d.bfd,'m'; 'K1',d.K1,''; 'K2',d.K2,''; ...
                    'k_ratio',d.k,''; 'p_ratio',d.p,''};
            for i = 1:size(rows,1)
                fprintf('  %-8s %14.8g %14s   [derived(%s)]\n', ...
                    rows{i,1}, rows{i,2}, rows{i,3}, sp.family);
            end
            fprintf('  %d elements:\n', numel(sp.elt));
            for k = 1:numel(sp.elt)
                e = sp.elt(k);
                fprintf('   %2d  %-10s %-10s Vpt=[% .4g % .4g % .4g]  [%s]\n', ...
                    k, e.name, e.kind, e.Vpt(1), e.Vpt(2), e.Vpt(3), e.provenance);
            end
        end

        function add_pupil(obj, ielt, opts)
        %ADD_PUPIL  Insert exit-pupil + image reference surfaces before a
        %   focal plane (PLAN_DESIGN_LAYER §8 Sprint 2B; Dave 2026-06-18).
        %   t.add_pupil(IELT) inserts, immediately BEFORE the FocalPlane at
        %   element IELT (default: the terminal FocalPlane):
        %     [IELT]    a FLAT Return at the focal-plane location (the
        %               image reference);
        %     [IELT+1]  a SPHERICAL Return at the paraxial exit pupil:
        %               radius = chief-ray distance FP->EP;  psi =
        %               -unit(chief-ray FP->EP) -- i.e. pointing back at
        %               the image, toward the sphere's centre of curvature.
        %   The original FocalPlane is PRESERVED and shifts to IELT+2;
        %   nElt grows by 2 ("don't lose the FP").  The exit pupil is
        %   located by the engine's FEX finder (the off-axis chief ray's
        %   axis crossing), so this also generalises to optimised layouts.
        %
        %   Name-value:
        %     'stop_elt'  aperture-stop element for FEX (default 1 = M1).
        %     'field_rad' off-axis field (rad) used to locate the EP
        %                 (default ~1 arcmin); restored to on-axis after.
        %     'mode'      FEX mode (1 = chief-ray centred, default).
        %
        %   The exit pupil is the DELIVERABLE handle for downstream
        %   instruments; the optimiser does NOT need it -- the FP OPD over
        %   the ray grid is already the exit-pupil-referenced wavefront.
            arguments
                obj
                ielt (1,1) double = -1
                opts.stop_elt  (1,1) double {mustBeInteger,mustBePositive} = 1
                opts.field_rad (1,1) double {mustBePositive} = 2.908882e-4
                opts.mode      (1,1) double {mustBeInteger,mustBePositive} = 1
            end
            n0 = numel(obj.spec.elt);
            if ielt < 0, ielt = n0; end            % default: terminal FocalPlane
            validateattributes(ielt, {'double'}, ...
                {'integer','positive','<=',n0}, 'add_pupil', 'ielt');
            fp = obj.spec.elt(ielt);
            if ~strcmp(fp.kind, 'FocalPlane')
                error('macos:design:Telescope:add_pupil:notFP', ...
                    'element %d is %s, not a FocalPlane.', ielt, fp.kind);
            end
            if ielt < 2
                error('macos:design:Telescope:add_pupil:noOptic', ...
                    'need at least one optic before the focal plane.');
            end
            FP_Vpt = fp.Vpt(:);
            apR    = max([obj.spec.elt.ap_r]);     % generous reference aperture
            prev   = obj.spec.elt(ielt-1);         % last optic before the FP

            % --- insert flat image-return + placeholder EP sphere, keeping
            %     the original FocalPlane (now the detector). FEX recomputes
            %     the EP, so the seed only has to make the Rx loadable. ---
            seed    = prev.Vpt(:);
            rSeed   = norm(seed - FP_Vpt);
            % Seed orientations from the chief line prev->FP.  A literal
            % [0 0 1] here assumes an UNFOLDED axial train: on a folded
            % bench the beam arrives in-plane, the z-facing flat is
            % ray-parallel, and the chief dies at it (the
            % tma_centered_foldfp "undefined after element 4" break).
            % -uIn reproduces (0,0,1) exactly for axial trains; uIn is the
            % same contract the post-FEX assignment uses (psi=-unit(FP->EP)).
            uIn     = (FP_Vpt - seed) / rSeed;     % chief prev -> FP
            flatRet = obj.new_elt_('FP_return', 'Return', FP_Vpt, -uIn, ...
                                   -1.0e22, apR, 'derived(add_pupil)', rSeed);
            sphRet  = obj.new_elt_('ExitPupil', 'Return', seed, uIn, ...
                                   -abs(rSeed), apR, 'derived(fex)', rSeed);
            obj.spec.elt = [obj.spec.elt(1:ielt-1), flatRet, sphRet, ...
                            obj.spec.elt(ielt:end)];
            obj.build();                           % emit + load the augmented Rx

            % --- locate the exit pupil with FEX (axis crossing of an
            %     off-axis chief ray). XP lands at nElt-1 = the EP slot. ---
            iEP = ielt + 1;  iFPnew = ielt + 2;  nE = numel(obj.spec.elt);
            cur = macos.get_src_fov();
            % Probe field = a small offset FROM THE CURRENT chief direction.
            % A literal [sin fr, 0, cos fr] probes about the unbiased +z
            % axis, locating the EP for the wrong field on a field-biased
            % (or folded) design.
            d0 = cur.src_dir(:) / norm(cur.src_dir);
            px = [1;0;0];
            if abs(dot(px,d0)) > 0.9, px = [0;1;0]; end
            px = px - dot(px,d0)*d0;  px = px / norm(px);
            macos.set_src_fov('src_dir', ...      % off-axis field first ...
                d0*cos(opts.field_rad) + px*sin(opts.field_rad));
            macos.stop(opts.stop_elt);            % ... then aim chief ray thru stop
            macos.trace(nE);
            f = macos.fex(opts.mode);
            macos.set_src_fov('src_dir', cur.src_dir);   % restore on-axis
            EP_Vpt = f.vpt(:);

            % --- radius + psi per the contract ---
            d      = EP_Vpt - FP_Vpt;              % FP -> EP
            radius = norm(d);
            psi    = -d / radius;                  % -unit(FP->EP), toward CoC@FP

            obj.spec.elt(iEP).Vpt  = EP_Vpt.';
            obj.spec.elt(iEP).psi  = psi.';
            obj.spec.elt(iEP).Kr   = -radius;      % sphere, CoC at the image
            obj.spec.elt(iEP).zElt = radius;       % EP -> detector
            obj.spec.elt(ielt).zElt   = radius;    % flat image -> EP
            obj.spec.elt(iFPnew).zElt = 1.0e20;    % detector terminal
            obj.spec.pupil = struct('img_elt',ielt, 'ep_elt',iEP, ...
                'fp_elt',iFPnew, 'ep_vpt',EP_Vpt.', 'ep_radius',radius);
            obj.build('', 'init', false);          % re-emit + reload (validate)
        end

        function res = optimize(obj, opts)
        %OPTIMIZE  Multi-field conic optimization of the telescope.
        %   res = t.optimize('fields_arcmin',[1.2 2.4]) refines every mirror
        %   conic to minimise the FoV-weighted RMS wavefront error over the
        %   on-axis field PLUS the given OFF-axis half-angles (+y), using
        %   MACOS's native multi-field design optimizer (CALIB).  Works for
        %   both 2-mirror and N-mirror families (it varies whatever Reflector
        %   conics exist).  Radii and spacings are held FIXED -- one shared
        %   physical system; only the per-mirror conic (DOF 8) varies, so the
        %   field varies without changing any fixed parameter (the structure
        %   constraint).  Two conics (2-mirror) cannot null field astigmatism
        %   -> a WFE "wall" off-axis; three (TMA) can -> the wide-field win.
        %
        %   Name-value:
        %     'engine'        'native' (CALIB, default) | 'fmincon' (TODO).
        %     'fields_arcmin' OFF-axis field half-angles, +y (default [1.2 2.4]).
        %     'fields'        (:,2) [thx thy] OFF-axis field OFFSETS (rad) -- an
        %                     explicit 2-D field set (a CROSS or area GRID);
        %                     supersedes 'fields_arcmin'.  A (0,0) row is
        %                     dropped (on-axis is the implicit field 1), so a
        %                     full grid incl. center is safe.  Build one with
        %                     macos.design.field_cross / field_grid.  NOTE:
        %                     CALIB caps at 12 FoV (a 3x3 area grid = 9).
        %     'max_iters'     CALIB iteration cap (default 60).
        %     'target'        'WFE' (default).
        %     'weights'       FoV weights, length 1+numel(fields) (default equal).
        %     'fpa_dofs'      (1,8) VarElt mask enrolling the terminal
        %                     FocalPlane as a varied element, so the detector
        %                     is solved JOINTLY with the optics instead of
        %                     being re-fitted between solves.  The detector is
        %                     what the exit-pupil merit's reference sphere is
        %                     centred on (CALIB's FEX radius is the chief-ray
        %                     distance from the pupil to the plane of the NEXT
        %                     element), so alternating solve <-> align is a
        %                     two-objective loop that need not contract --
        %                     measured drifting at 0.6-13 mm per round on the
        %                     rodgers1 TMA.  This is also what CODE V does.
        %                     Focus + tilt is [1 0 0 0 0 1 0 0]: DOF 1 = TIP
        %                     (rotation about local x) and DOF 6 = PIST
        %                     (translation along the local z, i.e. the surface
        %                     normal -- the focus/Tz direction).  Only DOFs
        %                     1..6 are accepted; ROC/CONIC on a flat detector
        %                     are meaningless.
        %     'dofs'          VarElt mask [TIP TILT CLOCK DX DY PIST ROC
        %                     CONIC] (default [0 0 0 0 0 0 0 1] = conic only).
        %                     A (1,8) row applies to EVERY varied element; an
        %                     (Nv,8) matrix gives a PER-ELEMENT mask, its rows
        %                     aligned to the varied elements in ascending
        %                     element-index order (or to 'elts' if given) --
        %                     e.g. hold M1 rigid (conic only) while M2/M3 also
        %                     decenter+tilt, without a global-tilt gauge freedom.
        %
        %   Returns: .converged, .n_fov, .fields_xy_arcmin (nfov x 2, absolute
        %   (thx,thy) incl. on-axis row 1), .fields_arcmin (the y-angles, back-
        %   compat), .wfe_before/.wfe_after (per field, metres), .conics
        %   (optimised K), .wavelength.  Optimised conics/geometry are written
        %   back to the spec, so a subsequent save()/add_pupil() emits the
        %   clean optimised design.
            arguments
                obj
                opts.engine        (1,:) char = 'native'
                opts.fields_arcmin (1,:) double = [1.2 2.4]
                opts.max_iters     (1,1) double {mustBeInteger,mustBePositive} = 60
                opts.target        (1,:) char = 'WFE'
                opts.weights       (1,:) double = []
                opts.fields        (:,2) double = []   % explicit (thx,thy) OFF-axis offsets (rad)
                opts.dofs          (:,8) double = [0 0 0 0 0 0 0 1]  % VarElt mask ((1,8) shared or (Nv,8) per-elt)
                opts.elts          (1,:) double = []   % subset of elements to vary
                opts.fpa_dofs      (:,8) double = []   % enrol the detector as a varied element
            end
            if ~all(ismember(opts.dofs(:), [0 1]))
                error('macos:design:Telescope:optimize:dofs', ...
                    'dofs must be a 0/1 mask over [TIP TILT CLOCK DX DY PIST ROC CONIC].');
            end
            if ~strcmp(opts.engine, 'native')
                error('macos:design:Telescope:optimize:engine', ...
                    'only engine=''native'' is implemented (fmincon is a follow-on).');
            end
            if obj.is_nmirror_() && (~isfield(obj.spec,'elt') || isempty(obj.spec.elt))
                obj.resolve_nmirror_();
            end
            % powered Reflectors only: a flat fold (add_fold, Kr sentinel
            % -1e22) has no radius/conic to vary and must not be enrolled
            var_elts = find(arrayfun(@(e) strcmp(e.kind,'Reflector') ...
                                     && abs(e.Kr) < 1e21, obj.spec.elt));
            if ~isempty(opts.elts)
                % subset: vary only the named elements (e.g. the imaging
                % mirrors while a field mirror is held, or vice versa --
                % the 3+1's image-vs-pupil split)
                bad = setdiff(opts.elts, var_elts);
                if ~isempty(bad)
                    error('macos:design:Telescope:optimize:elts', ...
                        'elts must be POWERED Reflector element indices (got %s).', ...
                        mat2str(bad));
                end
                var_elts = intersect(var_elts, opts.elts);
            end
            if isempty(var_elts)
                error('macos:design:Telescope:optimize:noMirror', ...
                    'no Reflector elements to vary.');
            end
            % --- optionally enrol the detector as a varied element --------
            fpa_elt = [];
            if ~isempty(opts.fpa_dofs)
                if size(opts.fpa_dofs,1) ~= 1
                    error('macos:design:Telescope:optimize:fpaRows', ...
                        'fpa_dofs must be a single (1,8) row.');
                end
                if any(opts.fpa_dofs(7:8) ~= 0)
                    error('macos:design:Telescope:optimize:fpaSurfDofs', ...
                        ['fpa_dofs may set only the rigid-body DOFs 1..6; ' ...
                         'ROC/CONIC are meaningless on a flat detector.']);
                end
                fpa_elt = find(strcmp({obj.spec.elt.kind},'FocalPlane'), 1, 'last');
                if isempty(fpa_elt)
                    error('macos:design:Telescope:optimize:noFPA', ...
                        'fpa_dofs given but the design has no FocalPlane.');
                end
                var_elts = [var_elts(:).' fpa_elt];
            end
            Nv     = numel(var_elts);                     % # varied elements
            % Expand the DOF mask to one row per varied element.  A single
            % (1,8) row is shared across all; an (Nv,8) matrix is per-element
            % (rows aligned to var_elts in ascending index order).
            Nopt = Nv - numel(fpa_elt);                   % optics rows expected
            if size(opts.dofs,1) == 1
                dof_rows = repmat(opts.dofs, Nopt, 1);
            elseif size(opts.dofs,1) == Nopt
                dof_rows = opts.dofs;
            else
                error('macos:design:Telescope:optimize:dofsRows', ...
                    ['dofs must be (1,8) [shared] or (%d,8) [per varied ' ...
                     'element] -- got %d rows for %d varied elements.'], ...
                    Nopt, size(opts.dofs,1), Nopt);
            end
            if ~isempty(fpa_elt)
                dof_rows = [dof_rows; opts.fpa_dofs];      % detector row last
            end
            % WHERE THE MERIT IS EVALUATED.  Default: the terminal FocalPlane,
            % which makes CALIB minimise std(OPL) to each ray's OWN intercept
            % on the detector plane.  On a tilted image surface that carries
            % (transverse ray aberration) x tan(tilt) -- an artifact that can
            % dwarf the wavefront error itself (rodgers1 PACKET Addendum 3 A.1).
            % When add_pupil has inserted an ExitPupil, evaluate THERE and turn
            % OptFEX on: CALIB then re-runs FEX per field
            % (smacos_compute.inc:391-397, hard-wired to nElt-1 = the
            % ExitPupil), giving each field a reference sphere centred on ITS
            % chief-ray intercept on the detector.  That is the strict metric;
            % verified numerically equal to it (2.7e-9) by
            % design/rodgers1/gate0_merit_identity.m.
            use_ep = isfield(obj.spec,'pupil') && ~isempty(obj.spec.pupil);
            if use_ep
                fp_elt = obj.spec.pupil.ep_elt;
                if fp_elt ~= numel(obj.spec.elt) - 1
                    error('macos:design:Telescope:optimize:epNotPenultimate', ...
                        ['the exit-pupil merit needs the ExitPupil at nElt-1 ' ...
                         '(CALIB''s FEX call is hard-wired there) -- it is at ' ...
                         '%d of %d.'], fp_elt, numel(obj.spec.elt));
                end
            else
                fp_elt = numel(obj.spec.elt);             % terminal FocalPlane
            end
            % Off-axis eval directions for the OptChfRayDir block.  Field 1 is
            % the nominal (possibly biased) ChfRayDir and is omitted here (it
            % shares the OptChfRayDir parse block).  The OFF-axis fields are
            % OFFSETS from that nominal, given either as a 2-D set or +y only:
            %   'fields'        (:,2) [thx thy] pairs (rad) -- a 2-D field set
            %                   (e.g. a CROSS); takes precedence when non-empty.
            %   'fields_arcmin' (1,:) +y half-angles (arcmin) -- 1-D default.
            % Directions are direction-cosines [sin ax, sin ay, sqrt(1-..)],
            % reducing EXACTLY to the legacy [0,sin,cos] form when ax=0.
            by = 0;  if isfield(obj.spec,'field_bias'), by = obj.spec.field_bias; end
            if ~isempty(opts.fields)
                F  = opts.fields;
                F  = F(any(abs(F) > 1e-12, 2), :);     % drop on-axis (= field 1)
                ax = F(:,1);                           % x offsets (rad)
                ay = by + F(:,2);                      % y offsets about the bias (rad)
            else
                ax = zeros(numel(opts.fields_arcmin),1);
                ay = by + deg2rad(opts.fields_arcmin(:)/60);
            end
            ax = ax(:);  ay = ay(:);
            cz = sqrt(max(0, 1 - sin(ax).^2 - sin(ay).^2));
            dirs = [sin(ax), sin(ay), cz];             % off-axis field directions
            fxy  = [0, by; ax, ay];                    % absolute (thx,thy)/field (rad)
            nfov = 1 + size(dirs,1);
            w = opts.weights;  if isempty(w), w = ones(1,nfov); end
            if numel(w) ~= nfov
                error('macos:design:Telescope:optimize:weights', ...
                    'weights must have 1+numel(fields_arcmin) = %d entries.', nfov);
            end

            obj.spec.opt = struct('target',opts.target, 'wf_elt',fp_elt, ...
                'max_iters',opts.max_iters, 'fields',dirs, 'weights',w, ...
                'var_elts',var_elts, 'dof_mask',opts.dofs, 'dof_rows',dof_rows, ...
                'fex',use_ep);
            obj.build();                                  % emit opt block -> load
            if use_ep
                % design_optim.F:170-180 aborts the solve unless the system
                % stop is set before CALIB is entered; smacos_compute.inc:279
                % then re-issues it per evaluation.
                macos.stop(1);
            end
            r = macos.calib();

            % read back per-element params CALIB may have moved, into the spec
            % (for describe()/view_layout); the deliverable handling differs
            % for conic-only vs geometry-moving runs (see below).
            % Kopt covers the OPTICS only: a detector enrolled via fpa_dofs
            % has no ROC/CONIC DOF, so its Kc/Kr are meaningless in derived.K
            % and in res.conics.  Its psi/Vpt ARE read back (below) -- that is
            % the whole point of enrolling it.
            Kopt = zeros(1, Nopt);
            for j = 1:Nv
                k = var_elts(j);
                if j <= Nopt
                    Kopt(j)         = macos.get_elt_kc(k);
                end
                obj.spec.elt(k).Kc  = macos.get_elt_kc(k);
                obj.spec.elt(k).Kr  = macos.get_elt_kr(k);          % ROC DOF
                obj.spec.elt(k).psi = reshape(macos.get_elt_psi(k), 1, 3); % tilt
                obj.spec.elt(k).Vpt = reshape(macos.get_elt_vpt(k), 1, 3); % decenter
            end
            if isfield(obj.spec.derived,'K'), obj.spec.derived.K = Kopt; end
            if obj.is_nmirror_()
                % write the optimized conic (and radius, for ROC runs) back
                % into the MIRROR LIST too, so a later re-resolve -- e.g.
                % add_fold/add_mirror invalidating spec.elt -- keeps the
                % optimized prescription instead of silently re-seeding
                % from Seidel.  (Rigid-body moves are NOT representable in
                % the coaxial mirror list and are still lost on re-resolve.)
                for j = 1:Nopt
                    nm = obj.spec.elt(var_elts(j)).name;
                    i  = find(strcmp({obj.spec.mirrors.name}, nm), 1);
                    if ~isempty(i)
                        obj.spec.mirrors(i).conic = Kopt(j);
                        obj.spec.mirrors(i).R     = abs(obj.spec.elt(var_elts(j)).Kr);
                    end
                end
            end
            obj.spec = rmfield(obj.spec, 'opt');
            % Clean re-emit from the updated spec.  CALIB bakes the rigid-body
            % result into psiElt/VptElt (verified), and our mirrors are
            % rotationally-symmetric conics, so the moved psi/Vpt fully define
            % each tilted/decentered surface -- the in-plane roll (TElt) is
            % irrelevant for a conic.  The reload-reproduces-WFE test guards
            % this.  (TElt emission is still needed for non-symmetric surfaces
            % and for fold flats -- a later step.)
            obj.build('', 'init', false);

            res = struct('converged',r.converged, 'n_fov',r.n_fov, ...
                'fields_xy_arcmin',rad2deg(fxy)*60, ...      % (thx,thy)/field, incl bias
                'fields_arcmin',rad2deg(fxy(:,2)).'*60, ...  % y-angles only (back-compat)
                'wfe_before',r.old_wfe(:,1).', 'wfe_after',r.new_wfe(:,1).', ...
                'conics',Kopt, 'var_elts',var_elts, 'wavelength',obj.spec.wavelength);
        end

        function set_freeform(obj, elt, modes, coef, opts)
        %SET_FREEFORM  Layer a Zernike-departure (freeform) figure onto a mirror.
        %   The conic base (KrElt/KcElt) is HELD -- preserving the first-order
        %   f/# and EP/layout -- and the Zernike terms add the figure correction,
        %   so the element emits Surface=Zernike (conic + Zernike departure: the
        %   §7.1 NAT-friendly canonical freeform form, e5mono's reflective M2).
        %   Zernike departures break rotational symmetry, so they reach the
        %   field-dependent aberrations conics alone cannot (-> optimize_freeform).
        %
        %     t.set_freeform(2, [4 11], [1e-7 -3e-8])   % focus + spherical on M2
        %     t.set_freeform(3, 5:11)                   % declare modes 5..11 (zeros)
        %     t.set_freeform(2, [4 11], c, 'type','Fringe')
            arguments
                obj
                elt   (1,1) double {mustBeInteger, mustBePositive}
                modes (1,:) double {mustBeInteger, mustBePositive}
                coef  (1,:) double = []
                opts.type (1,:) char = 'ANSI'
                opts.lmon (1,1) double = NaN   % Zernike normalization radius
                % (m); default = the element BODY ap_r.  Set it to the BEAM
                % footprint when the beam underfills the mirror: modes
                % normalized to a much larger radius are nearly degenerate
                % over the lit patch (they all look like tilt/astig there),
                % the solve goes ill-conditioned, and CALIB nulls the target
                % field with huge canceling coefficients that wreck the
                % surrounding field via beam walk.
            end
            if isempty(coef), coef = zeros(1, numel(modes)); end
            if numel(coef) ~= numel(modes)
                error('macos:design:Telescope:freeform:len', ...
                    'modes (%d) and coef (%d) must match in length.', ...
                    numel(modes), numel(coef));
            end
            if obj.is_nmirror_() && (~isfield(obj.spec,'elt') || isempty(obj.spec.elt))
                obj.resolve_nmirror_();
            end
            if ~isfield(obj.spec,'elt') || isempty(obj.spec.elt)
                error('macos:design:Telescope:freeform:noElts', ...
                    'no elements yet -- build the telescope before set_freeform.');
            end
            if elt > numel(obj.spec.elt)
                error('macos:design:Telescope:freeform:range', ...
                    'elt %d > nElt %d.', elt, numel(obj.spec.elt));
            end
            if ~strcmp(obj.spec.elt(elt).kind, 'Reflector')
                error('macos:design:Telescope:freeform:kind', ...
                    'freeform applies to a Reflector; elt %d is %s.', ...
                    elt, obj.spec.elt(elt).kind);
            end
            obj.spec.elt(elt).freeform = struct( ...
                'modes',modes(:).', 'coef',coef(:).', 'type',opts.type, ...
                'lmon',opts.lmon);
        end

        function res = optimize_freeform(obj, elts, opts)
        %OPTIMIZE_FREEFORM  Optimize Zernike-departure coefficients on the given
        %   mirror ELTS over a FIELD, holding every radius and conic FIXED, using
        %   MACOS's native multi-field design optimizer (CALIB, the OptZern DOF
        %   channel).  This is the §7.2 method: fix the geometry (radii set the
        %   EP/layout), then correct the wavefront with freeform departures that
        %   don't move the first-order solution.  Freeform terms break rotational
        %   symmetry, so they reach the field-dependent aberrations (coma/
        %   astigmatism) conics cannot; CALIB balances them over the on-axis
        %   field PLUS the given off-axis half-angles (the same multi-field
        %   machinery optimize() uses for conics -- see rc_unobscured).  Radii/
        %   conics are held by pairing OptZern with an all-zero VarElt mask, so
        %   only the Zernike modes vary.
        %
        %     t.optimize_freeform(2, 'modes',[5 6 7 8], 'fields_arcmin',[2 4])
        %     t.optimize_freeform([2 3], 'modes',5:11, 'fields',F)   % 2-D field set
        %
        %   fields_arcmin are the OFF-axis +y half-angles; the on-axis field is
        %   implicit (field 1).  Returns .wfe_before/.wfe_after (per field,
        %   metres), .fields_arcmin, .modes, .elts, .converged.
            arguments
                obj
                elts (1,:) double {mustBeInteger, mustBePositive}
                opts.modes (1,:) double {mustBeInteger,mustBePositive} = [5 6 7 8]
                opts.fields_arcmin (1,:) double = [2 4]   % OFF-axis (on-axis implicit)
                opts.fields (:,2) double = []             % explicit (thx,thy) offsets (rad)
                opts.max_iters (1,1) double = 80
                opts.target (1,:) char = 'WFE'
                opts.type   (1,:) char = 'ANSI'
                opts.weights (1,:) double = []
                opts.lmon   (1,:) double = NaN   % Zernike normalization radius
                                                 % (m): scalar for all ELTS, or
                                                 % one per ELTS entry -- multi-
                                                 % mirror solves span very
                                                 % different footprints (a
                                                 % near-focus field mirror is
                                                 % ~100x smaller than the
                                                 % stop); see set_freeform
            end
            if obj.is_nmirror_() && (~isfield(obj.spec,'elt') || isempty(obj.spec.elt))
                obj.resolve_nmirror_();
            end
            modes = opts.modes;
            lmon  = opts.lmon;
            if isscalar(lmon), lmon = repmat(lmon, 1, numel(elts)); end
            if numel(lmon) ~= numel(elts)
                error('macos:design:Telescope:optfree:lmon', ...
                    'lmon must be scalar or one per ELTS entry (%d).', numel(elts));
            end
            [elts, iu] = unique(elts(:).', 'stable');
            lmon = lmon(iu);
            for j = 1:numel(elts)
                k = elts(j);
                if k > numel(obj.spec.elt) || ~strcmp(obj.spec.elt(k).kind,'Reflector')
                    error('macos:design:Telescope:optfree:kind', ...
                        'optimize_freeform: elt %d is not a built Reflector.', k);
                end
                % establish/keep a Zernike surface with the requested modes,
                % seeding from any departure already present on those modes
                seed = zeros(1, numel(modes));
                ff = obj.spec.elt(k).freeform;
                if isstruct(ff) && ~isempty(ff) && isfield(ff,'modes')
                    for i = 1:numel(modes)
                        kk = find(ff.modes == modes(i), 1);
                        if ~isempty(kk), seed(i) = ff.coef(kk); end
                    end
                    % lmon continuity: coefficients only mean anything on
                    % the normalization radius they were solved on -- an
                    % omitted lmon INHERITS the stored one rather than
                    % silently reinterpreting the seed on the body radius.
                    if isnan(lmon(j)) && isfield(ff,'lmon') && ...
                            isscalar(ff.lmon) && ~isnan(ff.lmon)
                        lmon(j) = ff.lmon;
                    end
                end
                obj.set_freeform(k, modes, seed, 'type', opts.type, ...
                                 'lmon', lmon(j));
            end

            % off-axis eval directions about any +y bias; on-axis = field 1
            % (replicates optimize()'s OptChfRayDir set-up exactly).
            by = 0;  if isfield(obj.spec,'field_bias'), by = obj.spec.field_bias; end
            if ~isempty(opts.fields)
                F = opts.fields;  F = F(any(abs(F) > 1e-12, 2), :);
                ax = F(:,1);  ay = by + F(:,2);
            else
                ax = zeros(numel(opts.fields_arcmin),1);
                ay = by + deg2rad(opts.fields_arcmin(:)/60);
            end
            ax = ax(:);  ay = ay(:);
            cz   = sqrt(max(0, 1 - sin(ax).^2 - sin(ay).^2));
            dirs = [sin(ax), sin(ay), cz];
            nfov = 1 + size(dirs,1);
            w = opts.weights;  if isempty(w), w = ones(1,nfov); end
            if numel(w) ~= nfov
                error('macos:design:Telescope:optfree:weights', ...
                    'weights must have 1+numel(fields) = %d entries.', nfov);
            end

            % CALIB opt block: OptZern DOF on elts, all-zero rigid/conic mask.
            o = struct('target',opts.target, 'wf_elt',numel(obj.spec.elt), ...
                'max_iters',opts.max_iters, 'fields',dirs, 'weights',w, ...
                'var_elts',elts, 'dof_mask',[0 0 0 0 0 0 0 0], 'zern_elts',elts);
            o.zern_modes  = repmat({modes}, 1, numel(elts));  % cell, assigned after
            obj.spec.opt  = o;
            obj.build();                              % emit Zernike surfaces + OptZern
            r = macos.calib();

            % read the optimized Zernike coefficients back into the spec
            % (lmon rides along -- the coefficients are tied to the
            % normalization radius they were solved on; dropping it would
            % re-emit the surface on the body radius = a different figure)
            for j = 1:numel(elts)
                k = elts(j);
                c = macos.get_elt_zrn_coef(k, modes(:)).';
                obj.spec.elt(k).freeform = struct('modes',modes,'coef',c, ...
                    'type',opts.type, 'lmon',lmon(j));
            end
            obj.spec = rmfield(obj.spec, 'opt');
            obj.build('', 'init', false);             % clean re-emit from updated spec

            res = struct('converged',r.converged, 'n_fov',r.n_fov, ...
                'wfe_before',r.old_wfe(:,1).', 'wfe_after',r.new_wfe(:,1).', ...
                'fields_arcmin',rad2deg(ay).'*60, 'modes',modes, 'elts',elts, ...
                'wavelength',obj.spec.wavelength);
        end

        function res = optimize_aspheres(obj, elts, opts)
        %OPTIMIZE_ASPHERES  Refine even-radial AsphCoef (h^4, h^6, ...) on the
        %   given mirror ELTS to minimise on-axis RMS WFE -- the higher-order
        %   spherical the conic seed cannot reach at a fast primary.  Radii and
        %   conics are HELD (preserving the first-order f/# layout); aspheres
        %   are layered on, so the mirror emits Surface=Aspheric.  Nelder-Mead
        %   over scaled coefficients; each eval re-emits the spec and traces.
        %
        %   t.optimize_aspheres([1 3], 'nterms',3)   % M1 + M3, h^4/h^6/h^8
            arguments
                obj
                elts (1,:) double {mustBeInteger, mustBePositive}
                opts.nterms    (1,1) double = 3      % h^4, h^6, h^8
                opts.max_evals (1,1) double = 300
                opts.xmax      (1,1) double = 25     % |scaled coeff| bound (~waves)
            end
            D = obj.spec.in.D;  lam = obj.spec.wavelength;  h = D/2;
            nt = opts.nterms;  ne = numel(elts);
            scale = lam ./ h.^(2*(1:nt)+2);          % x~O(1) -> ~lam of sag
            function ww = ev(x)
                % bound the search -- extreme aspheres make rays miss/NaN and
                % SIGSEGV the engine (uncatchable); keep fminsearch in range.
                if any(abs(x) > opts.xmax), ww = 1e3; return; end
                xi = reshape(x, nt, ne);
                for j = 1:ne
                    obj.spec.elt(elts(j)).asph = xi(:,j).' .* scale;
                end
                obj.build('', 'init', false);        % re-emit spec + reload
                s  = macos.trace(numel(obj.spec.elt));
                ww = s.rmsWFE / lam;
                if ~isfinite(ww), ww = 1e3; end
            end
            w0 = ev(zeros(nt*ne, 1));                % baseline (asph = 0)
            o  = optimset('MaxFunEvals',opts.max_evals, 'MaxIter',opts.max_evals, ...
                          'TolFun',1e-4, 'TolX',1e-3, 'Display','off');
            [xo, wf] = fminsearch(@ev, zeros(nt*ne,1), o);
            ev(xo);                                  % leave optimum set + loaded
            res = struct('wfe_before',w0, 'wfe_after',wf, 'elts',elts, 'nterms',nt);
        end

        function fig = diagram(obj, opts)
        %DIAGRAM  Side-view (z-y) layout: element bodies + chief ray +
        %   marginal beam (PLAN_DESIGN_LAYER §8 Sprint 4).  Reveals when an
        %   element BODY sits in another element's BEAM -- e.g. a coaxial
        %   TMA where M1 and the focal plane occult the M2->M3 beam (all
        %   vertices on one axis -> physically unbuildable until taken
        %   off-axis).  z is the (folded) optical axis, y the in-plane
        %   transverse coord; x is ignored (planar layouts).
        %   Name-value: 'save' (PNG path), 'visible' (default true).
            arguments
                obj
                opts.save    (1,:) char   = ''
                opts.visible (1,1) logical = true
            end
            if obj.is_nmirror_() && (~isfield(obj.spec,'elt') || isempty(obj.spec.elt))
                obj.resolve_nmirror_();
            end
            e = obj.spec.elt;  n = numel(e);
            z = arrayfun(@(x) x.Vpt(3), e);     % side view: z horizontal
            y = arrayfun(@(x) x.Vpt(2), e);     %            y vertical
            h = obj.paraxial_heights_();         % marginal beam radius at each elt

            vis = 'on';  if ~opts.visible, vis = 'off'; end
            fig = figure('Visible',vis, 'Position',[80 80 980 520]);  hold on;
            % chief ray (vertex path) + marginal-beam envelope (both folded)
            plot(z, y,   'r-', 'LineWidth',1.2, 'DisplayName','chief ray');
            plot(z, y+h, 'b-', 'LineWidth',0.8, 'DisplayName','marginal beam');
            plot(z, y-h, 'b-', 'LineWidth',0.8, 'HandleVisibility','off');
            % element bodies: segment perpendicular to psi, length 2*aperture
            for k = 1:n
                p = e(k).psi;  apr = e(k).ap_r;
                bdir = [-p(2), p(3)];            % perp to psi projected into (z,y)
                nb = hypot(p(2), p(3));  if nb > 0, bdir = bdir/nb; end
                zz = z(k) + apr*[-1 1]*bdir(1);
                yy = y(k) + apr*[-1 1]*bdir(2);
                col = 'k';  if strcmp(e(k).kind,'FocalPlane'), col = 'm'; end
                if strcmp(e(k).kind,'Return'), col = [0 .6 0]; end
                plot(zz, yy, 'Color',col, 'LineWidth',2.5, 'HandleVisibility','off');
                text(z(k), y(k), ['  ' e(k).name], 'FontSize',8, ...
                     'VerticalAlignment','bottom');
            end
            axis equal; grid on; box on;
            xlabel('z  (optical axis)'); ylabel('y  (transverse)');
            title(sprintf('%s layout (side view) -- bodies black, FP magenta', ...
                  obj.spec.family));
            legend('Location','best');
            if ~isempty(opts.save), print(fig, opts.save, '-dpng', '-r150'); end
        end

        function fig = view_layout(obj, plane, opts)
        %VIEW_LAYOUT  Real-ray layout view (engine DRAW bundle) + conic
        %   surfaces -- the revealing beam-train / deconfliction view
        %   (PLAN_DESIGN_LAYER §8 Sprint 4).  Plots the engine's actual
        %   traced ray fan in PLANE ('YZ'|'XZ'|'XY') together with each
        %   element's conic-sag surface profile, so the beam filling the
        %   optics and any body-in-beam obscuration are visible.
        %
        %   Because a 2-D projection collapses depth and can paint FALSE
        %   conflicts (e.g. a fold sends light behind the PM), the view is
        %   sliceable:
        %     'hide'    element indices whose SURFACE to omit (e.g. the PM)
        %     'istart'  first element to draw (0 = from the source)
        %     'iend'    last element to draw   (0 = nElt)
        %     'save'    PNG path;  'visible'  (default true)
            arguments
                obj
                plane   (1,:) char    = 'YZ'
                opts.hide   (1,:) double  = []
                opts.istart (1,1) double  = 0
                opts.iend   (1,1) double  = 0
                opts.nrays  (1,1) double  = 25     % # rays drawn (subsampled)
                opts.save   (1,:) char    = ''
                opts.visible (1,1) logical = true
            end
            obj.ensure_loaded_();                  % current design in the engine
            vis = 'on';  if ~opts.visible, vis = 'off'; end
            fig = figure('Visible',vis, 'Position',[60 60 1000 560]);
            ax  = axes('Parent', fig);
            obj.draw_plane_(ax, plane, opts.hide, opts.istart, opts.iend, opts.nrays);
            if ~isempty(opts.save), print(fig, opts.save, '-dpng', '-r150'); end
        end

        function fig = view_orthoviews(obj, planes, opts)
        %VIEW_ORTHOVIEWS  Multi-panel orthographic layout -- the design-report
        %   figure.  Draws the same real-ray VIEW_LAYOUT in several planes side
        %   by side so the design can be judged from all angles.  PLANES is a
        %   cellstr or a token list ('YZ XZ XY') of 'YZ'|'XZ'|'XY' (default
        %   {'YZ','XZ'} -- add 'XY' for folded / non-planar designs).  Same
        %   'hide'/'istart'/'iend'/'nrays'/'save'/'visible' options as
        %   view_layout, applied to every panel.
        %   'zoom' appends one extra DETAIL panel: {'PLANE',[Ulo Uhi Vlo Vhi]}
        %   redraws that plane cropped to the given plot-axis window (U = the
        %   panel's horizontal axis, V vertical) -- e.g. magnify a folded
        %   back end that a full-train view renders unreadably small.  An
        %   optional third entry {'PLANE',rect,[istart iend]} restricts the
        %   detail panel to that ELEMENT range, so only the legs of interest
        %   are drawn (a folded bench without the front-end beams that
        %   otherwise project through the crop).
            arguments
                obj
                planes                     = {'YZ','XZ'}
                opts.hide    (1,:) double  = []
                opts.istart  (1,1) double  = 0
                opts.iend    (1,1) double  = 0
                opts.nrays   (1,1) double  = 25
                opts.save    (1,:) char    = ''
                opts.visible (1,1) logical = true
                opts.zoom    (1,:) cell    = {}
                opts.fans      (1,:) char  = 'both'   % XY main panels: 'both'|'x'|'y'
                opts.zoom_fans (1,:) char  = 'both'   % XY detail panel
            end
            pl  = obj.plane_list_(planes);
            nz  = ~isempty(opts.zoom);
            if nz && (numel(opts.zoom) < 2 || numel(opts.zoom) > 3 ...
                      || ~isnumeric(opts.zoom{2}) || numel(opts.zoom{2}) ~= 4)
                error('macos:design:Telescope:orthoviews:zoom', ...
                    'zoom must be {''PLANE'',[Ulo Uhi Vlo Vhi]} or {...,[istart iend]}.');
            end
            np  = numel(pl) + nz;
            obj.ensure_loaded_();
            vis = 'on';  if ~opts.visible, vis = 'off'; end
            fig = figure('Visible',vis, 'Position',[40 60 min(520*np,1560) 540]);
            tl  = tiledlayout(fig, 1, np, 'TileSpacing','compact', 'Padding','compact');
            for i = 1:numel(pl)
                ax = nexttile(tl);
                obj.draw_plane_(ax, pl{i}, opts.hide, opts.istart, opts.iend, ...
                                opts.nrays, opts.fans);
            end
            if nz
                zp = obj.plane_list_(opts.zoom{1});
                zi = opts.istart;  ze = opts.iend;
                if numel(opts.zoom) == 3
                    zr = opts.zoom{3};
                    zi = zr(1);  if numel(zr) > 1, ze = zr(2); end
                end
                ax = nexttile(tl);
                obj.draw_plane_(ax, zp{1}, opts.hide, zi, ze, opts.nrays, ...
                                opts.zoom_fans);
                r = opts.zoom{2};
                xlim(ax, r(1:2));  ylim(ax, r(3:4));
                title(ax, sprintf('%s detail', zp{1}), 'Interpreter','none');
            end
            sgtitle(tl, sprintf('%s -- orthographic layout (real rays)', obj.spec.family), ...
                    'Interpreter','none');
            if ~isempty(opts.save), print(fig, opts.save, '-dpng', '-r150'); end
        end

        function B = ray_bundle(obj, opts)
        %RAY_BUNDLE  Full-grid 3-D ray bundle: the position of EVERY grid
        %   ray at EVERY element, per field point -- the primitive behind
        %   slice-selectable layout views (Dave 2026-07-05: engine DRAW
        %   traces only the middle meridian fan per plane, so tricky
        %   folded layouts cannot be sliced usefully from it).  Built on
        %   macos.trace(k) + macos.get_ray_info -- no engine change, no
        %   DRAW ray-count cap.
        %
        %   B = t.ray_bundle()                        % nominal field
        %   B = t.ray_bundle('fields',F)              % (N,2) rad offsets
        %                                             % about the bias
        %   Returns:
        %     .nray, .nelt, .fields (N,2)
        %     .pos {f}  3 x nray x nelt   ray position at each element
        %     .ok  {f}  nray x nelt       traced-AND-passed mask
        %     .pup      2 x nray          pupil coords, normalized to the
        %                                 entrance footprint (slice masks:
        %                                 |pup(1,:)| < tol = the Y-slice,
        %                                 |pup(2,:)| < tol = the X-slice,
        %                                 offsets/annuli at will)
        %
        %   Slice example (the fan at x = +0.5 of the pupil radius):
        %     m = abs(B.pup(1,:) - 0.5) < 0.08;
        %     plot(squeeze(B.pos{1}(3,m,:)).', squeeze(B.pos{1}(2,m,:)).')
            arguments
                obj
                opts.fields (:,2) double = [0 0]
            end
            obj.ensure_loaded_();
            if ~macos.has_rx(), obj.build(); else, obj.build('','init',false); end
            nE = numel(obj.spec.elt);
            nF = size(opts.fields,1);
            B  = struct('nelt',nE, 'fields',opts.fields, ...
                        'pos',{cell(1,nF)}, 'ok',{cell(1,nF)});
            for f = 1:nF
                fxy = opts.fields(f,:);
                if any(abs(fxy) > 1e-15), obj.trace_at_field(fxy); end
                s  = macos.trace(1);
                ri = macos.get_ray_info(s.nRays);
                nR = size(ri.pos, 2);
                P  = nan(3, nR, nE);  OK = false(nR, nE);
                P(:,:,1) = ri.pos;
                OK(:,1)  = logical(ri.ok_trace) & logical(ri.ok_pass);
                if f == 1
                    % pupil coords from the entrance footprint (elt 1),
                    % centered on its mean, normalized to max radius
                    c = mean(ri.pos(1:2, OK(:,1)), 2);
                    q = ri.pos(1:2,:) - c;
                    B.pup  = q / max(hypot(q(1,OK(:,1)), q(2,OK(:,1))));
                    B.nray = nR;
                end
                for k = 2:nE
                    macos.trace(k);
                    ri = macos.get_ray_info(nR);
                    P(:,:,k) = ri.pos;
                    OK(:,k)  = logical(ri.ok_trace) & logical(ri.ok_pass);
                end
                B.pos{f} = P;  B.ok{f} = OK;
                if any(abs(fxy) > 1e-15), obj.trace_at_field([]); end
            end
        end

        function trace_at_field(obj, fxy)
        %TRACE_AT_FIELD  Re-emit + trace the design at one field offset.
        %   t.trace_at_field([thx thy]) rebuilds the emitted Rx with the
        %   chief ray pointed at the given field OFFSET (rad, about any
        %   field bias) and traces it, so macos.opd() / macos.draw_rays /
        %   macos.spot inspect that field.  t.trace_at_field([]) restores
        %   the nominal field (and re-traces).  This is the per-field
        %   mechanism realize_apertures / aperture_full_field use
        %   internally, exposed as a utility -- the sanctioned way to look
        %   at an off-axis field (macos.set_src_fov does NOT move the
        %   field of an emitted design; the biased chief ray must be
        %   re-emitted).
            arguments
                obj
                fxy (:,:) double = []
            end
            if isempty(fxy)
                obj.restore_trace_field_([]);          % rebuilds nominal
            else
                if numel(fxy) ~= 2
                    error('macos:design:Telescope:trace_at_field:fxy', ...
                        'fxy must be [thx thy] (rad) or [] for nominal.');
                end
                by = 0;
                if isfield(obj.spec,'field_bias'), by = obj.spec.field_bias; end
                obj.spec.trace_field = [fxy(1), by + fxy(2)];
                obj.build('', 'init', false);
            end
            macos.trace(numel(obj.spec.elt));
        end

        function fig = view_field_map(obj, scan, opts)
        %VIEW_FIELD_MAP  Map of RMS WFE over the 2-D field -- the design-report
        %   field view.  SCAN is a realize_apertures (or compatible) result
        %   carrying .fields (K x 2 field angles, arcmin) and .wfe (K, waves).
        %   When the samples lie on a rectangular GRID (e.g. macos.design.
        %   field_grid) the WFE is drawn as a filled contour (default) or a
        %   surface; otherwise it falls back to a colored scatter.  Use a fine
        %   grid (7x7+) for a smooth report map.
        %     'kind'    'contour' (default) | 'surf'
        %     'save'    PNG path;  'visible' (default true)
        %
        %   LOST FIELDS (NaN wfe) are rendered VISIBLY (grey) and
        %   NEVER interpolated over (Dave 2026-07-30): a filled contour of a
        %   grid with NaN holes silently paints across them, hiding lost
        %   fields.  We first grey-fill the whole grid extent, then contour
        %   only the finite region on top, so any NaN field shows through as
        %   grey.  The title/caption also states which WFE metric the scan
        %   used (scan.metric).
            arguments
                obj
                scan struct
                opts.kind (1,:) char {mustBeMember(opts.kind,{'contour','surf'})} = 'contour'
                opts.save (1,:) char = ''
                opts.visible (1,1) logical = true
            end
            fx = scan.fields(:,1);  fy = scan.fields(:,2);  w = scan.wfe(:);
            ux = uniquetol(fx, 1e-9);  uy = uniquetol(fy, 1e-9);
            metric = 'global';
            if isfield(scan,'metric') && ~isempty(scan.metric), metric = scan.metric; end
            vis = 'on';  if ~opts.visible, vis = 'off'; end
            fig = figure('Visible',vis, 'Position',[60 60 620 500]);
            isgrid = numel(ux) >= 2 && numel(uy) >= 2 && ...
                     numel(w) == numel(ux)*numel(uy);
            nlost = nnz(~isfinite(w));
            if isgrid
                W = nan(numel(uy), numel(ux));
                for i = 1:numel(w)
                    ix = find(abs(ux - fx(i)) < 1e-9, 1);
                    iy = find(abs(uy - fy(i)) < 1e-9, 1);
                    W(iy, ix) = w(i);
                end
                if strcmp(opts.kind, 'surf')
                    surf(ux, uy, W);  shading interp;  view(40, 30);
                    zlabel('RMS WFE (waves)');
                else
                    % grey underlay over the full grid extent so any NaN field
                    % shows through as grey instead of being interpolated over.
                    if nlost > 0
                        ax = gca;  set(ax,'Color',[0.75 0.75 0.75]);
                        hold(ax,'on');
                        % explicit grey patch behind the axes fill (robust when
                        % the whole grid is finite except a border):
                        patch('XData',[ux(1) ux(end) ux(end) ux(1)], ...
                              'YData',[uy(1) uy(1) uy(end) uy(end)], ...
                              'FaceColor',[0.75 0.75 0.75],'EdgeColor','none', ...
                              'HandleVisibility','off');
                    end
                    % contour only the FINITE region; NaN cells are left blank
                    % (contourf does not fill NaN-cornered cells) -> grey shows.
                    contourf(ux, uy, W, 12, 'LineColor','none');  axis equal tight;
                end
            else
                % scatter: draw finite fields colored, lost fields as grey x.
                fin = isfinite(w);
                scatter(fx(fin), fy(fin), 45, w(fin), 'filled');  hold on;
                if any(~fin)
                    scatter(fx(~fin), fy(~fin), 45, [0.5 0.5 0.5], 'x', 'LineWidth',1.2);
                end
                axis equal tight;
            end
            cb = colorbar;  cb.Label.String = 'RMS WFE (waves)';
            xlabel('\theta_x (arcmin)');  ylabel('\theta_y (arcmin)');
            ttl = sprintf('%s -- RMS WFE over field  [metric: %s]', ...
                          obj.spec.family, metric);
            if nlost > 0
                ttl = sprintf('%s\n(%d/%d fields lost -- grey)', ttl, nlost, numel(w));
            end
            title(ttl, 'Interpreter','none');
            if ~isempty(opts.save), print(fig, opts.save, '-dpng', '-r150'); end
        end

        function rep = check_clipping(obj, opts)
        %CHECK_CLIPPING  3-D body-in-beam obscuration + footprint margin
        %   (PLAN_DESIGN_LAYER §8 Sprint 4).  DRAW (data-only) traces a 1-D
        %   MERIDIAN fan per plane, so the YZ and XZ passes are DIFFERENT rays
        %   (the y-fan and the x-fan) and must NOT be stitched into one bundle.
        %   This uses them as TWO independent 3-D fans -- each plane fixes 2
        %   coords; the off-plane coord is the per-element beam center -- and
        %   tests every PHYSICAL element body (disk: centre = beam center,
        %   normal psi, radius = beam footprint) for piercing a beam segment
        %   between two OTHER elements (the self-obscuration the coaxial TMA
        %   suffers: M1 + FP on the M2->M3 axis).  Judged in 3-D: a single 2-D
        %   projection paints FALSE conflicts (a fold tucks the beam behind PM).
        %
        %   rep = t.check_clipping() returns a struct array, one per element:
        %     .name .kind .ap_r   aperture radius (m)
        %     .foot_r            realised beam-footprint radius at the elt
        %     .margin            ap_r - foot_r  (>=0: beam fits the aperture)
        %     .obstructs         # beam segments this body pierces (0 = clear)
        %     .ok                margin>=0 && obstructs==0
        %   Prints a table + overall verdict unless 'quiet'.  'noload' skips
        %   the build/reload when the design is already loaded in the engine.
            arguments
                obj
                opts.quiet  (1,1) logical = false
                opts.noload (1,1) logical = false
                opts.tol    (1,1) double  = 1e-9   % segment-endpoint exclusion
            end
            if ~opts.noload
                if ~macos.has_rx(), obj.build(); else, obj.build('','init',false); end
            end
            e  = obj.spec.elt;  nE = numel(e);

            % --- per-element disk geometry + physical-body flag + hole
            % (set_hole: a perforated primary passes the through-beam; a
            % foreign crossing within hole_r of the body center is NOT an
            % obstruction and does not count toward clearance)
            Vpt = zeros(3,nE);  psi = zeros(3,nE);  apr = zeros(1,nE);
            isBody = false(1,nE);  hole = zeros(1,nE);
            for k = 1:nE
                Vpt(:,k) = e(k).Vpt(:);
                p = e(k).psi(:);  np = norm(p);  if np > 0, p = p/np; end
                psi(:,k) = p;  apr(k) = e(k).ap_r;
                isBody(k) = any(strcmp(e(k).kind, {'Reflector','FocalPlane'}));
                if isfield(obj.spec,'holes') && ~isempty(obj.spec.holes)
                    i = find(strcmp({obj.spec.holes.name}, e(k).name), 1);
                    if ~isempty(i), hole(k) = obj.spec.holes(i).r; end
                end
            end

            % --- two orthogonal DRAW MERIDIAN fans (data-only).  DRAW traces a
            % 1-D meridian fan per plane (the middle row/col of the ray grid,
            % macos_cmd_loop.inc) -- NOT the full bundle: YZ -> the y-fan, XZ ->
            % the x-fan, which are DIFFERENT rays.  So they must NOT be stitched
            % into one 3-D bundle: pairing an x-fan ray's X with a y-fan ray's Y
            % fills the bounding SQUARE (corner r*sqrt2 -- the old M1 foot=0.707
            % for a 0.5 beam).  Treat them as two INDEPENDENT 3-D fans instead:
            % each plane fixes 2 coords; the off-plane coord is the beam CENTER
            % at that element (the other fan's transverse mean -- exact for a
            % meridian ray, which lies in its plane through the beam center).
            byz = macos.draw_rays('YZ', 0, nE);   % y-fan: V=Y, U=Z  (x ~ center)
            bxz = macos.draw_rays('XZ', 0, nE);   % x-fan: V=X, U=Z  (y ~ center)

            % per-element beam center (off-plane coords from each fan's mean) +
            % footprint radius (max transverse half-extent over both fans).
            ctr = Vpt;  foot = zeros(1,nE);
            for k = 1:nE
                my = (byz.elt == k);  mx = (bxz.elt == k);
                if ~any(my(:)) && ~any(mx(:)), continue; end
                cx = Vpt(1,k);  if any(mx(:)), cx = mean(bxz.V(mx)); end
                cy = Vpt(2,k);  if any(my(:)), cy = mean(byz.V(my)); end
                zz = [byz.U(my); bxz.U(mx)];  cz = mean(zz(:));
                ctr(:,k) = [cx; cy; cz];
                ry = 0;  if any(my(:)), ry = max(abs(byz.V(my) - cy)); end
                rx = 0;  if any(mx(:)), rx = max(abs(bxz.V(mx) - cx)); end
                foot(k) = max(rx, ry);              % beam radius (0.5 for a 0.5 beam)
            end

            % --- body-in-beam: test each fan's REAL 3-D ray segments (off-plane
            % coord = the per-element beam center) against every non-endpoint
            % body disk.  obstructs counts pierced segments; clr tracks the
            % closest foreign-beam approach to the body center -> signed
            % clearance = clr - foot.
            obstructs = zeros(1,nE);
            clr       = inf(1,nE);
            for pass = 1:2
                isy = (pass == 1);
                if isy, bb = byz; else, bb = bxz; end
                for r = 1:bb.nray
                    npr = bb.nper(r);
                    for i = 1:npr-1
                        ea = bb.elt(i,r);  eb = bb.elt(i+1,r);
                        A = obj.fan_pt_(bb, i,   r, isy, ctr, nE);
                        B = obj.fan_pt_(bb, i+1, r, isy, ctr, nE);
                        AB = B - A;
                        for k = 1:nE
                            if ~isBody(k) || k == ea || k == eb, continue; end
                            den = psi(:,k).' * AB;
                            if abs(den) < 1e-30, continue; end       % grazes plane
                            t = (psi(:,k).' * (ctr(:,k) - A)) / den;
                            if t <= opts.tol || t >= 1-opts.tol, continue; end
                            Q   = A + t*AB;
                            rho = norm(Q - ctr(:,k));
                            if rho < hole(k), continue; end   % through the hole
                            clr(k) = min(clr(k), rho);
                            if rho < foot(k), obstructs(k) = obstructs(k) + 1; end
                        end
                    end
                end
            end

            % --- assemble report
            rep = struct('name',{},'kind',{},'ap_r',{},'foot_r',{}, ...
                         'margin',{},'obstructs',{},'clearance',{},'ok',{});
            for k = 1:nE
                margin    = apr(k) - foot(k);            % patch vs nominal aperture (info)
                clearance = clr(k) - foot(k);           % patch edge to nearest foreign beam
                okk       = (obstructs(k) == 0);          % body clears all foreign beams
                rep(k) = struct('name',e(k).name, 'kind',e(k).kind, ...
                    'ap_r',apr(k), 'foot_r',foot(k), 'margin',margin, ...
                    'obstructs',obstructs(k), 'clearance',clearance, 'ok',okk);
            end

            if ~opts.quiet
                fprintf('check_clipping  (family=%s, %d elements)\n', ...
                        obj.spec.family, nE);
                fprintf('  %-10s %-10s %9s %9s %9s %9s %8s  %s\n', ...
                    'name','kind','ap_r','foot_r','margin','clearnce','obstruct','status');
                for k = 1:nE
                    st = 'OK';  if ~rep(k).ok, st = '** CLIP'; end
                    cstr = sprintf('%9.4g', rep(k).clearance);
                    if isinf(rep(k).clearance), cstr = sprintf('%9s','--'); end
                    fprintf('  %-10s %-10s %9.4g %9.4g %9.4g %s %8d  %s\n', ...
                        rep(k).name, rep(k).kind, rep(k).ap_r, rep(k).foot_r, ...
                        rep(k).margin, cstr, rep(k).obstructs, st);
                end
                if all([rep.ok])
                    fprintf(['  => layout is CLEAR ' ...
                             '(no body-in-beam, beams fit apertures)\n']);
                else
                    fprintf(['  => layout has CONFLICTS: margin<0 = own beam ' ...
                             'overfills aperture; clearance<0 = body cuts a ' ...
                             'foreign beam\n']);
                end
            end
        end

        function clear_realized_apertures(obj)
        %CLEAR_REALIZED_APERTURES  Drop the realized per-element clear
        %   apertures (spec.elt.ap / ap_rect) and re-emit with the
        %   design-phase (vertex-centered ap_r) stops.  STOPGAP for the
        %   realize_apertures frame bug (2026-07-06): footprint centers
        %   are measured in GLOBAL XY (draw_rays) but emitted as LOCAL
        %   ApVec offsets -- correct only while the element origin sits
        %   at the global origin (coaxial / eccentric-section parents).
        %   On a tilted-fold design the emitted stops land metres off
        %   the beam and the SAVED .in loses every ray on reload
        %   (sz_tma.in carries this latent).  Call before save() until
        %   the ray_bundle-based aperture-frame fix lands.
            for k = 1:numel(obj.spec.elt)
                obj.spec.elt(k).ap = [];
                obj.spec.elt(k).ap_rect = [];
            end
            obj.build('', 'init', false);
        end

        function rep = aperture_full_field(obj, opts)
        %APERTURE_FULL_FIELD  Per-element clear aperture covering the FULL
        %   FIELD (PLAN_DESIGN_LAYER §8).  Traces a set of field points
        %   spanning the design FoV and, for each element, returns the
        %   smallest centred circle (centre + radius, in the element's local
        %   aperture plane) that contains EVERY field point's beam footprint
        %   -- the essential aperture-sizing output once a design meets its
        %   other requirements.  Directly emit-ready as ApVec=(radius,xc,yc).
        %
        %   Name-value:
        %     'fields'  Kx2 field points [theta_x theta_y] (rad) to span.
        %               Default: the bias point plus set_field_points offsets
        %               (shifted onto the bias), or just the bias alone.
        %     'margin'  fractional radius margin (default 0.05).
        %     'quiet'   suppress the printed table (default false).
        %   rep(k): .name .center [xc yc] .radius .nfield
            arguments
                obj
                opts.fields (:,2) double  = []
                opts.margin (1,1) double  = 0.05
                opts.quiet  (1,1) logical = false
            end
            by = 0;  if isfield(obj.spec,'field_bias'), by = obj.spec.field_bias; end
            F = opts.fields;
            if isempty(F)
                if isfield(obj.spec,'field_points') && any(obj.spec.field_points(:))
                    fp = obj.spec.field_points;          % Kx2 offsets (rad)
                    F  = [fp(:,1), by + fp(:,2)];
                else
                    F = [0, by];
                end
            end
            nE = numel(obj.spec.elt);

            % accumulate per-element footprint bounding box over field points
            lo = inf(2,nE);  hi = -inf(2,nE);
            saved = [];
            if isfield(obj.spec,'trace_field'), saved = obj.spec.trace_field; end
            restore = onCleanup(@() obj.restore_trace_field_(saved)); %#ok<NASGU>
            for i = 1:size(F,1)
                obj.spec.trace_field = F(i,:);
                obj.build('', 'init', false);
                b = macos.draw_rays('XY', 0, nE);        % U=X, V=Y (pinned plane)
                for k = 1:nE
                    m = (b.elt == k);
                    if ~any(m(:)), continue; end
                    lo(1,k) = min(lo(1,k), min(b.U(m)));
                    hi(1,k) = max(hi(1,k), max(b.U(m)));
                    lo(2,k) = min(lo(2,k), min(b.V(m)));
                    hi(2,k) = max(hi(2,k), max(b.V(m)));
                end
            end

            rep = struct('name',{},'center',{},'radius',{},'nfield',{});
            for k = 1:nE
                if ~isfinite(lo(1,k))
                    c = [0 0];  r = 0;
                else
                    c = [(lo(1,k)+hi(1,k))/2, (lo(2,k)+hi(2,k))/2];
                    % half-diagonal of the bounding box covers every footprint
                    r = 0.5*hypot(hi(1,k)-lo(1,k), hi(2,k)-lo(2,k))*(1+opts.margin);
                end
                rep(k) = struct('name',obj.spec.elt(k).name, 'center',c, ...
                                'radius',r, 'nfield',size(F,1));
            end
            if ~opts.quiet
                fprintf('aperture_full_field  (%d field point(s), family=%s)\n', ...
                        size(F,1), obj.spec.family);
                fprintf('  %-10s %12s %12s %12s\n', 'element','radius','xc','yc');
                for k = 1:nE
                    fprintf('  %-10s %12.5g %12.5g %12.5g\n', rep(k).name, ...
                            rep(k).radius, rep(k).center(1), rep(k).center(2));
                end
            end
        end

        function scan = realize_apertures(obj, opts)
        %REALIZE_APERTURES  Field scan -> per-optic clear apertures + WFE(field).
        %   Sweeps the chief-ray direction over the FoV (about any field bias),
        %   traces each field, and records (a) the RMS WFE at each field and
        %   (b) the MAXIMUM beam footprint on every optic across the field.
        %   Sizes a clear aperture to that full-field footprint -- CIRCULAR
        %   (radius,xc,yc) on the mirrors, SQUARE (Rectangular) on the focal
        %   plane -- stores it on the spec (so build() emits the ApVec and
        %   view_layout draws each optic to its real size + center) and returns
        %   the scan.  Footprints use BOTH DRAW meridian fans (YZ -> y-extent,
        %   XZ -> x-extent) so the aperture is the true 2-D beam size.
        %
        %   The FIELD SET (FoV) is telescope-specific -- by default it comes
        %   from the design's field_points (set_field_points), NOT a built-in
        %   list.  Name-value:
        %     'fields_arcmin'  +y field half-angles (arcmin) -- convenience
        %                      override of the design FoV.
        %     'fields'         Kx2 [thx thy] field set (rad) -- explicit override.
        %     'margin'         fractional aperture margin (default 0.05).
        %     'quiet'          suppress the printed table (default false).
        %   The WFE metric is selectable (Dave / Rodgers reconciliation,
        %   2026-07-30):
        %     'metric'  'global'    (default) RMS = std(OPD) at ONE global
        %                           image plane -- leaves the fast anastigmat's
        %                           field-curvature defocus in the off-centre
        %                           corners.  The historical behaviour; committed
        %                           baselines depend on it, so it is UNCHANGED.
        %               'refsphere' CODE V-consistent per-field best-focus
        %                           reference-sphere RMS: each field's OPD has
        %                           piston + tip/tilt + defocus removed (fit over
        %                           the pupil) before the RMS, i.e. referenced to
        %                           that field's own best-focus sphere, and is
        %                           evaluated over the realised CLEAR APERTURE
        %                           (the vignetted pupil, a 2nd pass after the
        %                           apertures are sized) -- exactly CODE V's
        %                           field-map RMS convention.  Reconciles the
        %                           Rodgers on-axis field map to ~3% (rodgers1
        %                           PACKET Sec 4a).  OPT-IN; the returned scan
        %                           records .metric and every rendered field map
        %                           states which metric it used.
        %
        %   TWO FURTHER NAMED REFERENCES live with the strict-metric study
        %   code (mmacos/design/rodgers1/), not here, because they need an
        %   exit-pupil probe and a FROZEN detector plane -- state this
        %   function does not carry:
        %     'strict-centroid'  sphere anchored at the exit pupil, centred on
        %                        the SPOT CENTROID on the frozen detector.
        %                        PRIMARY per Dave's 2026-07-31 ruling -- it is
        %                        what the detector integrates.
        %     'strict-chief'     the same, centred on the CHIEF-RAY intercept.
        %                        Secondary, reported as a labelled column.
        %   Select them with strict_wfe/strict_wfe_deck's 'reference' option;
        %   both are always computed, and view_field_map labels whichever the
        %   scan carries in .metric.
        %
        %   IDEMPOTENCY (Dave 2026-07-30): any previously-realized clear
        %   apertures are dropped at entry, so each call re-measures on the
        %   CLEAN (un-clipped) design.  Without this, a second call -- or a
        %   later per-field trace / metric_ladder -- runs THROUGH the stale
        %   apertures sized for the earlier box; images that walk outside them
        %   are clipped and the field reads NaN (the "top rows lost" / "second
        %   call all-NaN" evaluator findings, PACKET Sec B).  On a fresh
        %   telescope this clear is a no-op (bit-identical first-call numbers).
        %
        %   Returns scan: .fields (Kx2 arcmin) .wfe (waves, per field) .lambda
        %                 .metric ('global'|'refsphere')
        %                 .aperture (struct array: name/shape/radius/center/rect).
            arguments
                obj
                opts.fields_arcmin (1,:) double = []
                opts.fields (:,2) double = []
                opts.margin (1,1) double = 0.05
                opts.quiet  (1,1) logical = false
                opts.metric (1,:) char {mustBeMember(opts.metric,{'global','refsphere'})} = 'global'
            end
            by0 = 0;  if isfield(obj.spec,'field_bias'), by0 = obj.spec.field_bias; end
            nE  = numel(obj.spec.elt);
            lam = obj.spec.wavelength;
            % Field set (Kx2 rad, absolute incl. bias).  Priority: explicit
            % fields_arcmin (+y) > explicit fields > the design's field_points
            % (the user-specified FoV) > on-axis.
            if ~isempty(opts.fields_arcmin)
                F = [zeros(numel(opts.fields_arcmin),1), ...
                     by0 + deg2rad(opts.fields_arcmin(:)/60)];
            elseif ~isempty(opts.fields)
                F = opts.fields;
            elseif isfield(obj.spec,'field_points') && any(obj.spec.field_points(:))
                fp = obj.spec.field_points;             % Kx2 offsets (rad)
                F  = [fp(:,1), by0 + fp(:,2)];
            else
                F = [0, by0];                           % on-axis only
            end
            nF = size(F,1);

            % Idempotency: drop any previously-realized clear apertures so we
            % re-measure on the CLEAN design (see the header note).  No-op on a
            % fresh telescope -> first-call numbers are bit-identical.
            hadAp = false;
            for k = 1:nE
                if ~isempty(obj.spec.elt(k).ap) || ~isempty(obj.spec.elt(k).ap_rect)
                    obj.spec.elt(k).ap = [];  obj.spec.elt(k).ap_rect = [];
                    hadAp = true;
                end
            end
            if hadAp, obj.build('', 'init', false); end

            saved = [];
            if isfield(obj.spec,'trace_field'), saved = obj.spec.trace_field; end
            restore = onCleanup(@() obj.restore_trace_field_(saved)); %#ok<NASGU>

            xlo = inf(1,nE);  xhi = -inf(1,nE);
            ylo = inf(1,nE);  yhi = -inf(1,nE);
            wfe = nan(1, nF);
            for j = 1:nF
                obj.spec.trace_field = F(j,:);
                obj.build('', 'init', false);
                macos.trace(nE);
                W = macos.opd();  v = W(isfinite(W) & W ~= 0);
                % PASS 1 always records the GLOBAL-plane RMS over the clean
                % (un-clipped) geometric pupil -- the historical metric, kept
                % bit-identical.  The refsphere metric is a SECOND pass below,
                % after the clear apertures are installed, so it is evaluated
                % over the design's realised CLEAR APERTURE (the vignetted
                % pupil) -- the CODE V field-map convention (PACKET Sec 4a).
                if ~isempty(v), wfe(j) = std(v) / lam; end
                byz = macos.draw_rays('YZ', 0, nE);        % y-fan: V=Y
                bxz = macos.draw_rays('XZ', 0, nE);        % x-fan: V=X
                for k = 1:nE
                    my = (byz.elt == k);  mx = (bxz.elt == k);
                    if any(my(:))
                        ylo(k)=min(ylo(k),min(byz.V(my))); yhi(k)=max(yhi(k),max(byz.V(my)));
                    end
                    if any(mx(:))
                        xlo(k)=min(xlo(k),min(bxz.V(mx))); xhi(k)=max(xhi(k),max(bxz.V(mx)));
                    end
                end
            end

            ap = struct('name',{},'shape',{},'radius',{},'center',{},'rect',{});
            for k = 1:nE
                e = obj.spec.elt(k);
                if ~isfinite(xlo(k)) && ~isfinite(ylo(k)), continue; end
                xs = max(0, xhi(k)-xlo(k));  ys = max(0, yhi(k)-ylo(k));
                cx = (xlo(k)+xhi(k))/2;      cy = (ylo(k)+yhi(k))/2;
                hw = 0.5*(1+opts.margin)*max(xs, ys);    % half-width / radius
                if strcmp(e.kind,'FocalPlane')
                    rect = [cx-hw, cx+hw, cy-hw, cy+hw];  % SQUARE
                    obj.spec.elt(k).ap_rect = rect;  obj.spec.elt(k).ap = [];
                    ap(end+1) = struct('name',e.name,'shape','rect', ...
                        'radius',hw,'center',[cx cy],'rect',rect);  %#ok<AGROW>
                elseif strcmp(e.kind,'Reflector')
                    obj.spec.elt(k).ap = [hw, cx, cy];  obj.spec.elt(k).ap_rect = [];
                    ap(end+1) = struct('name',e.name,'shape','circ', ...
                        'radius',hw,'center',[cx cy],'rect',[]);  %#ok<AGROW>
                end
            end
            % PASS 2 (refsphere metric only): with the MIRROR clear apertures
            % now installed, re-trace each field and take the per-field best-
            % focus reference-sphere RMS (piston+tip/tilt+defocus removed) over
            % the rays that pass the realised STOPS -- the CODE V-consistent
            % field-map RMS (references each field to its own best-focus sphere
            % and counts only the clear-aperture pupil).  Reconciles the
            % Rodgers on-axis map to ~3% (PACKET Sec 4a).  A field the aperture
            % vignettes to < 6 samples stays NaN -> rendered as a lost field.
            %
            % The FocalPlane rect clip is DROPPED for this pass: on a tilted /
            % offset FP its emitted ApVec carries the known global-XY ->
            % local-ApVec frame bug (see clear_realized_apertures) and would
            % vignette every ray off-axis, making the metric all-NaN.  The
            % physical STOPS are the (coaxial, frame-bug-immune) mirrors; the
            % FP only samples the wavefront.  The returned aperture struct
            % still reports the FP rect (restored below) for downstream use.
            if strcmp(opts.metric, 'refsphere')
                fpsave = cell(nE,2);
                for k = 1:nE
                    if strcmp(obj.spec.elt(k).kind,'FocalPlane')
                        fpsave{k,1} = obj.spec.elt(k).ap;
                        fpsave{k,2} = obj.spec.elt(k).ap_rect;
                        obj.spec.elt(k).ap = [];  obj.spec.elt(k).ap_rect = [];
                    end
                end
                obj.build('', 'init', false);          % mirror aps only
                for j = 1:nF
                    obj.spec.trace_field = F(j,:);
                    obj.build('', 'init', false);
                    macos.trace(nE);
                    r = obj.refsphere_rms_(macos.opd());
                    if ~isnan(r), wfe(j) = r / lam; else, wfe(j) = NaN; end
                end
                for k = 1:nE                            % restore FP clip
                    if strcmp(obj.spec.elt(k).kind,'FocalPlane')
                        obj.spec.elt(k).ap = fpsave{k,1};
                        obj.spec.elt(k).ap_rect = fpsave{k,2};
                    end
                end
            end

            scan = struct('fields', rad2deg(F)*60, 'wfe',wfe, ...
                          'lambda',lam, 'metric',opts.metric, 'aperture',ap);

            if ~opts.quiet
                fa = rad2deg(F(:,2))*60;
                fprintf(['realize_apertures  (%d fields, +y %g..%g arcmin, ' ...
                         'family=%s, metric=%s)\n'], nF, min(fa), max(fa), ...
                         obj.spec.family, opts.metric);
                fprintf('  field WFE (waves):');  fprintf(' %.4f', wfe);  fprintf('\n');
                fprintf('  %-10s %-5s %10s %10s %10s\n','optic','shape','radius','xc','yc');
                for i = 1:numel(ap)
                    fprintf('  %-10s %-5s %10.4g %10.4g %10.4g\n', ap(i).name, ...
                        ap(i).shape, ap(i).radius, ap(i).center(1), ap(i).center(2));
                end
            end
        end
    end

    methods (Static)
        function obj = load_spec(path)
        %LOAD_SPEC  Reconstruct a Telescope from a saved spec (.mat).
            arguments, path (1,:) char, end
            S = load(path, 'spec');
            obj = macos.design.Telescope.from_spec_(S.spec);
        end
    end

    % ===================================================================
    methods (Access = private)
        function ensure_loaded_(obj)
        %ENSURE_LOADED_  Make sure the CURRENT design is loaded in the engine.
            if ~macos.has_rx()
                obj.build();
            else
                obj.build('', 'init', false);
            end
        end

        function pl = plane_list_(~, planes)
        %PLANE_LIST_  Normalize a planes arg (cellstr or token list) -> cellstr.
            if iscell(planes)
                pl = cellfun(@char, planes, 'UniformOutput', false);
            else
                pl = regexp(char(planes), '[A-Za-z][A-Za-z]', 'match');
            end
            if isempty(pl)
                error('macos:design:Telescope:view_orthoviews:planes', ...
                      'planes must be a cellstr or token list of YZ/XZ/XY.');
            end
        end

        function draw_plane_(obj, ax, plane, hide, istart, iend, nrays, fans)
        %DRAW_PLANE_  Draw the real-ray layout for ONE plane into axes AX -- the
        %   shared core of view_layout / view_orthoviews.  Assumes the current
        %   design is already loaded (see ensure_loaded_).  FANS applies to the
        %   XY cross-plane view only: 'both' (default) overlays the x- AND
        %   y-meridian fans (beam extents measurable in both axes); 'x' or 'y'
        %   draws that single fan -- e.g. a folded X-Y bench reads cleanest
        %   with only the rays that LIE IN the bench plane (the x-fan).
            if nargin < 8, fans = 'both'; end
            nE = numel(obj.spec.elt);
            if iend <= 0, iend = nE; end
            b = macos.draw_rays(plane, istart, iend);
            switch upper(plane)            % which 3-D comps map to (U,V)
                case 'YZ', cU = 3; cV = 2;
                case 'XZ', cU = 3; cV = 1;
                case 'XY', cU = 1; cV = 2;
                otherwise
                    error('macos:design:Telescope:view_layout:plane', ...
                          'plane must be YZ, XZ or XY.');
            end
            axn = 'XYZ';
            % per-element beam FOOTPRINT in the (cU,cV) plane: HALF-WIDTH about
            % the beam center (not |offset-from-vertex|) + the center in plane
            % coords, so each optic is drawn to its real beam size AND position
            % -- correct for off-axis sections (FP / exit pupil) whose beam
            % center is offset from the vertex.
            foot   = zeros(1, nE);
            cenU_b = nan(1, nE);
            cenV_b = nan(1, nE);
            for k = 1:nE
                mask = (b.elt == k);
                if ~any(mask(:)), continue; end
                e  = obj.spec.elt(k);
                pu = e.psi(cU);  pv = e.psi(cV);  np = hypot(pu,pv);
                if np > 0, pu = pu/np;  pv = pv/np; end
                tu = -pv;  tv = pu;                          % in-plane transverse
                tperp = (b.U(mask)-e.Vpt(cU))*tu + (b.V(mask)-e.Vpt(cV))*tv;
                tlo = min(tperp);  thi = max(tperp);
                foot(k)   = 0.5*(thi - tlo);
                hc        = 0.5*(thi + tlo);
                cenU_b(k) = e.Vpt(cU) + hc*tu;
                cenV_b(k) = e.Vpt(cV) + hc*tv;
            end
            hold(ax, 'on');
            % --- real ray bundle (subsampled so the beam shape stays legible) ---
            if strcmpi(plane, 'XY')
                % CROSS-PLANE view: draw the TRUE position of every selected
                % ray at every element (ray_bundle), projected onto (X,Y).
                % The earlier beam-center fan reconstruction drew the Return
                % retrace legs visibly separated when physically they
                % overlay exactly (rhat = -ihat retro), and could not show
                % the real XY spread (Dave 2026-07-05: the panel must show
                % the actual ray spread in the XY plane).  'fans' picks the
                % slice: 'y' = the nrays rays nearest the pupil y-meridian,
                % 'x' = nearest the x-meridian, 'both' = both slices.
                B  = obj.ray_bundle();
                i0 = max(1, istart);
                % slice = the full grid COLUMN/ROW within one grid pitch of
                % the pupil meridian, subsampled EVENLY along the fan.  A
                % nearest-N pick has no ray exactly ON the meridian, so it
                % clumps to both sides and draws a visible GAP in the slice
                % (Dave 2026-07-05).
                tol = 1.2 / max(2, obj.spec.sampling - 1);
                sel = [];
                if any(strcmpi(fans, {'y','both'}))     % pupil-x ~ 0 column
                    sel = [sel, slice_(B.pup(1,:), B.pup(2,:), tol, nrays)];
                end
                if any(strcmpi(fans, {'x','both'}))     % pupil-y ~ 0 row
                    sel = [sel, slice_(B.pup(2,:), B.pup(1,:), tol, nrays)];
                end
                for r = unique(sel)
                    P   = squeeze(B.pos{1}(:, r, i0:iend));
                    okr = B.ok{1}(r, i0:iend);
                    P(:, ~okr) = NaN;                   % break lost segments
                    plot(ax, P(1,:), P(2,:), '-', 'Color',[0 .45 .85], ...
                         'LineWidth',0.5, 'HandleVisibility','off');
                end
            else
                step = max(1, floor(b.nray / max(2, nrays)));
                for r = 1:step:b.nray
                    m = b.nper(r);
                    if m >= 2
                        plot(ax, b.U(1:m,r), b.V(1:m,r), '-', 'Color',[0 .45 .85], ...
                             'LineWidth',0.5, 'HandleVisibility','off');
                    end
                end
            end
            % --- conic-sag surfaces, to the MEASURED clear aperture
            % (realize_apertures) when present, else the real beam footprint.
            % cW is the out-of-plane axis; the section's offset along it feeds
            % surface_profile_ so an off-axis slice (e.g. M1 in XZ with the beam
            % decentered in y) is drawn at the right depth, not the y=0 sag. ---
            cW = 6 - cU - cV;                  % the third axis (1+2+3 = 6)
            % Out-of-plane (cW) beam center per element, from an ORTHOGONAL DRAW
            % fan -- so the conic sag uses the FULL transverse radius (the y
            % offset for an XZ view, etc.), not just the in-plane coordinate;
            % otherwise an off-axis section is drawn at the wrong depth.
            cwc = obj.beam_offplane_(plane, cW, istart, iend, nE);
            % label placement: stack labels that would overlap (TEXT-WIDTH
            % aware, not just point distance) and draw a thin leader to the
            % element, so clustered optics (secondary + exit pupil + image
            % return) stay readable.
            Vspan = max(b.V(:)) - min(b.V(:));  if Vspan <= 0, Vspan = 1; end
            gap   = 0.075 * Vspan;              % vertical stack step
            cw    = 0.022 * Vspan;              % approx char width (font 8)
            placed = zeros(0,3);                % [u, v, text-width]
            for k = max(1,istart):iend
                if ismember(k, hide), continue; end
                e = obj.spec.elt(k);
                cen = [];
                if isfield(e,'ap') && ~isempty(e.ap)             % measured circular
                    ext = e.ap(1);   G3 = [e.ap(2), e.ap(3), e.Vpt(3)];
                    cen = [G3(cU), G3(cV)];
                elseif isfield(e,'ap_rect') && ~isempty(e.ap_rect)   % measured rect (FP)
                    rr  = e.ap_rect;  ext = 0.5*max(rr(2)-rr(1), rr(4)-rr(3));
                    G3  = [0.5*(rr(1)+rr(2)), 0.5*(rr(3)+rr(4)), e.Vpt(3)];
                    cen = [G3(cU), G3(cV)];
                elseif foot(k) > 0             % mirror / FP / EP: real beam footprint
                    ext = foot(k)*1.15;
                    cen = [cenU_b(k), cenV_b(k)];
                else
                    ext = e.ap_r;              % no rays here: physical (detector) size
                end
                woff = 0;
                if ~isnan(cwc(k)), woff = cwc(k) - e.Vpt(cW); end  % out-of-plane offset
                [su, sv] = obj.surface_profile_(e, cU, cV, ext, cen, woff);
                col = 'k';
                if strcmp(e.kind,'FocalPlane'), col = 'm';
                elseif strcmp(e.kind,'Return'), col = [0 .6 0]; end
                plot(ax, su, sv, 'Color',col, 'LineWidth',2.4, 'HandleVisibility','off');
                lu0 = e.Vpt(cU);  lv0 = e.Vpt(cV);
                if ~isempty(cen), lu0 = cen(1);  lv0 = cen(2); end  % element point
                lu = lu0;  lv = lv0;
                w  = (numel(e.name)+2) * cw;       % approx rendered text width
                bumped = true;
                while bumped                       % stack up until no overlap
                    bumped = false;
                    for r = 1:size(placed,1)
                        xov = (lu < placed(r,1)+placed(r,3)) && (placed(r,1) < lu+w);
                        if xov && abs(lv-placed(r,2)) < gap
                            lv = lv + gap;  bumped = true;  break;
                        end
                    end
                end
                placed = [placed; lu lv w];        %#ok<AGROW>
                if abs(lv-lv0) > 1e-9              % offset -> thin leader line
                    plot(ax, [lu0 lu], [lv0 lv], '-', 'Color',[.65 .65 .65], ...
                         'LineWidth',0.4, 'HandleVisibility','off');
                end
                text(ax, lu, lv, ['  ' e.name], 'FontSize',8, 'Interpreter','none');
            end
            axis(ax, 'equal');  grid(ax, 'on');  box(ax, 'on');
            xlabel(ax, [axn(cU) ' axis']);  ylabel(ax, [axn(cV) ' axis']);
            ttl = sprintf('%s layout -- %s plane (real rays)', ...
                          obj.spec.family, upper(plane));
            if ~isempty(hide), ttl = [ttl sprintf('  [hidden: %s]', mat2str(hide))]; end
            title(ax, ttl, 'Interpreter','none');
        end

        function cwc = beam_offplane_(~, plane, cW, istart, iend, nE)
        %BEAM_OFFPLANE_  Out-of-plane (axis cW) beam center per element, from a
        %   DRAW fan in a plane ORTHOGONAL to the viewing PLANE.  Lets
        %   draw_plane_ draw each conic at its true transverse radius (so an
        %   off-axis section sits at the right depth in the cross-plane view).
        %   Returns NaN for elements with no rays.
            oplane = 'YZ';  if strcmpi(plane,'YZ'), oplane = 'XZ'; end
            switch upper(oplane)
                case 'YZ', oc = [3 2];        % bo.U = z, bo.V = y
                case 'XZ', oc = [3 1];        % bo.U = z, bo.V = x
            end
            cwc = nan(1, nE);
            bo  = macos.draw_rays(oplane, istart, iend);
            for k = 1:nE
                m = (bo.elt == k);  if ~any(m(:)), continue; end
                if     oc(1) == cW, cwc(k) = 0.5*(min(bo.U(m)) + max(bo.U(m)));
                elseif oc(2) == cW, cwc(k) = 0.5*(min(bo.V(m)) + max(bo.V(m)));
                end
            end
        end

        function R = surf_frame_(~, psi)
        %SURF_FRAME_  Local surface frame [x y z] (columns, in global coords)
        %   for the element TElt: Z along the OUTWARD surface normal (psi) at
        %   the pole, X/Y tangent to the surface, right-handed.  Matches the
        %   dmt6mono convention (psi=(0,0,-1) -> x=(-1,0,0), y=(0,1,0)).
        %   Trace-neutral; this is the interface frame for PERTURB +
        %   MACOS-emitted sensitivities (structures/controls hand-off).
            z = psi(:) / norm(psi);
            yhat = [0;1;0];
            if abs(z(2)) > 0.95, yhat = [1;0;0]; end   % avoid psi ~ y degeneracy
            y = yhat - (yhat.'*z)*z;  y = y / norm(y);
            x = cross(y, z);                           % right-handed: x = y x z
            R = [x, y, z];
        end

        function restore_trace_field_(obj, saved)
        %RESTORE_TRACE_FIELD_  Undo the transient per-field-point source
        %   re-pointing used by aperture_full_field, and re-emit the nominal
        %   design so the engine state matches the design again.
            if isempty(saved)
                if isfield(obj.spec,'trace_field')
                    obj.spec = rmfield(obj.spec, 'trace_field');
                end
            else
                obj.spec.trace_field = saved;
            end
            obj.build('', 'init', false);
        end

        function rms = refsphere_rms_(~, W)
        %REFSPHERE_RMS_  Per-field best-focus reference-sphere RMS of an OPD map.
        %   The CODE V-consistent field-map metric (Dave / Rodgers, 2026-07-30):
        %   fit and remove piston + tip/tilt + defocus over the LIT pupil, i.e.
        %   reference the wavefront to that field's own best-focus sphere, then
        %   take the RMS of the residual.  This strips the field-curvature
        %   defocus a fast anastigmat leaves at a single global image plane
        %   (reconciles the Rodgers on-axis map to ~3%; see rodgers1 PACKET 4a).
        %   Same fit basis as design/src/wfe_field_diag.m's rms_focus rung, so
        %   the two agree.  Returns NaN if too few lit samples to fit.
        %   W is the raw macos.opd() map (metres, 0 / >1e30 = unlit sentinels).
            [ny,nx] = size(W);
            [X,Y] = meshgrid(linspace(-1,1,nx), linspace(-1,1,ny));
            m = isfinite(W) & (W ~= 0) & (abs(W) < 1e30);
            if nnz(m) < 6, rms = NaN; return; end
            x = X(m);  y = Y(m);  w = W(m);
            x = x - mean(x);  y = y - mean(y);
            s = max(hypot(x,y));  if s > 0, x = x/s;  y = y/s; end
            B = [ones(size(x)), x, y, (2*(x.^2+y.^2)-1)];   % piston+tilt+defocus
            c = B \ w;
            rms = std(w - B*c);                             % metres
        end

        function resolve_section_poles_(obj)
        %RESOLVE_SECTION_POLES_  For every mirror, set RptElt = the beam-
        %   footprint center on the parent surface (the section pole) and nrm =
        %   the analytic outward surface normal there, so emit_ writes a true
        %   off-axis section (RptElt!=VptElt + section TElt).  Trace-neutral
        %   (ConSrf uses VptElt only) -- this changes only the interface /
        %   perturbation frame, never the WFE.  The analytic normal
        %   n = (psi - s'(d)*that)/sqrt(1+s'^2) reproduces j18sc's segment TElt
        %   col-3 exactly (s'(d) = (d/R)/sqrt(1-(1+K)(d/R)^2) is the conic-sag
        %   slope at off-axis height d; R=|Kr|, K=Kc, that = transverse unit).
            obj.build('', 'init', false);              % current (decentered) design
            nE = numel(obj.spec.elt);
            b  = macos.draw_rays('XY', 0, nE);         % U=X, V=Y (pinned plane)
            for k = 1:nE
                e = obj.spec.elt(k);
                if ~strcmp(e.kind, 'Reflector'), continue; end
                m = (b.elt == k);
                if ~any(m(:)), continue; end
                xc = mean(b.U(m));  yc = mean(b.V(m));
                Vpt = e.Vpt(:);  psi = e.psi(:)/norm(e.psi);
                off = [xc - Vpt(1); yc - Vpt(2); 0];   % footprint center vs vertex
                off = off - (psi.'*off)*psi;           % perpendicular to parent axis
                d   = norm(off);  R = abs(e.Kr);  K = e.Kc;
                if d < 1e-12 || R >= 1e21
                    obj.spec.elt(k).pole = [];  obj.spec.elt(k).nrm = [];
                    continue;                          % on-axis / flat: no section
                end
                that = off / d;
                u    = min((1+K)*(d/R)^2, 1 - 1e-12);  % guard beyond valid aperture
                sag  = d^2 / (R*(1 + sqrt(1 - u)));     % conic sag at height d
                sp   = (d/R) / sqrt(1 - u);             % d(sag)/d(transverse)
                pole = Vpt + d*that + sag*psi;          % the pole lies ON the parent
                nrm  = psi - sp*that;  nrm = nrm/norm(nrm);
                obj.spec.elt(k).pole = pole(:).';
                obj.spec.elt(k).nrm  = nrm(:).';
            end
            obj.build('', 'init', false);              % re-emit with the section poles
        end

        function d = clearance_solve_(obj, target, margin_m, hi)
        %CLEARANCE_SOLVE_  Smallest +y beam decenter (m) such that element
        %   TARGET's body clears every foreign beam by >= MARGIN_M.  Clearance
        %   grows monotonically with decenter, so bisect on [0, HI].
            saved_d = obj.spec.aperture_decenter;
            restore = onCleanup(@() obj.restore_decenter_(saved_d)); %#ok<NASGU>
            if obj.probe_clearance_(0, target) >= margin_m
                d = 0;  return;                        % already clear (unlikely)
            end
            if obj.probe_clearance_(hi, target) < margin_m
                warning('macos:design:Telescope:offaxis:noclear', ...
                    ['%s does not clear by %.3g m even at decenter %.3g m; ' ...
                     'using the max.'], target, margin_m, hi);
                d = hi;  return;
            end
            lo = 0;
            for it = 1:40 %#ok<NASGU>
                d = 0.5*(lo + hi);
                if obj.probe_clearance_(d, target) >= margin_m, hi = d; else, lo = d; end
                if (hi - lo) < 1e-4*max(1, obj.spec.in.D), break; end
            end
            d = hi;                                    % return the cleared side
        end

        function c = probe_clearance_(obj, d, target)
        %PROBE_CLEARANCE_  WORST signed clearance (m) over the TARGET optic set
        %   at +y decenter D -- negative if any targeted body still pierces a
        %   foreign beam.  TARGET is a name, a cellstr, or 'all' (every mirror).
        %   Bodies with no foreign beam crossing their plane are infinitely
        %   clear and do not constrain the solve.
            obj.spec.aperture_decenter = d;
            obj.build('', 'init', false);
            rep   = obj.check_clipping('noload', true, 'quiet', true);
            names = obj.clear_targets_(target);
            c = inf;
            for k = 1:numel(rep)
                if ~any(strcmp(rep(k).name, names)), continue; end
                ck = rep(k).clearance;
                if rep(k).obstructs > 0, ck = -abs(ck); end     % pierced -> negative
                if isinf(ck) && ck > 0, continue; end           % infinitely clear
                c = min(c, ck);
            end
            if isinf(c), c = 1e9; end                            % all targets clear
        end

        function names = clear_targets_(obj, target)
        %CLEAR_TARGETS_  Resolve a 'clear' spec (name | cellstr | 'all') to a
        %   cellstr of mirror element names.
            if iscell(target)
                names = target;  return;
            end
            if ischar(target) && ~strcmpi(target, 'all')
                names = {target};  return;
            end
            names = {};                                          % 'all' -> every mirror
            for k = 1:numel(obj.spec.elt)
                if strcmp(obj.spec.elt(k).kind, 'Reflector')
                    names{end+1} = obj.spec.elt(k).name; %#ok<AGROW>
                end
            end
        end

        function P = fan_pt_(~, bb, i, r, isy, ctr, nE)
        %FAN_PT_  3-D point of crossing I on ray R of a DRAW meridian fan BB.
        %   y-fan (isy): off-plane x = beam center x at the crossed element;
        %   x-fan: off-plane y = beam center y.  Meridian rays lie in their
        %   plane through the beam center, so this is exact (not a stitch).
            ke = bb.elt(i,r);
            cx = 0;  cy = 0;
            if ke >= 1 && ke <= nE, cx = ctr(1,ke);  cy = ctr(2,ke); end
            if isy        % y-fan: V=Y, U=Z, x = beam center
                P = [cx; bb.V(i,r); bb.U(i,r)];
            else          % x-fan: V=X, U=Z, y = beam center
                P = [bb.V(i,r); cy; bb.U(i,r)];
            end
        end

        function restore_decenter_(obj, d)
        %RESTORE_DECENTER_  Undo the transient decenter probing in the
        %   clearance bisection (the final decenter is set by set_offaxis).
            obj.spec.aperture_decenter = d;
        end

        function resolve_(obj)
        %RESOLVE_  Closed-form first-order layout + conics (§5.1/§5.2).
        %   Ported and validated against the shared fixtures
        %   (optical_design/fixtures/telescope_design_fixtures.json).
            sp = obj.spec;
            D  = sp.in.D;
            f  = sp.in.system_fnum * D;          % EFL
            f1 = sp.in.primary_fnum * D;         % primary focal length
            m  = f / f1;                         % secondary magnification
            beta = sp.in.BFD / f1;               % back-focal-dist parameter
            greg = strcmp(sp.family,'gregorian');

            R1 = 2*f1;
            if greg
                if ~(beta > 0) || ~(m > 1)
                    error('macos:design:Telescope:greg', ...
                        'Gregorian needs m>1 and beta>0 (intermediate focus).');
                end
                sep = f1*(m+beta)/(m-1);         % > f1 (past prime focus)
                R2  = -2*f*(1+beta)/(m^2-1);     % concave secondary (R2<0)
                k   = (1+beta)/(m-1);
            else
                sep = f1*(m-beta)/(m+1);
                R2  = 2*f*(1+beta)/(m^2-1);      % convex secondary (R2>0)
                k   = (1+beta)/(m+1);
            end
            bfd = beta*f1;  p = R2/R1;
            [K1, K2] = obj.conics_(sp.family, m, beta, k, p);

            d = struct('f',f,'f1',f1,'m',m,'beta',beta,'R1',R1,'R2',R2, ...
                       'sep',sep,'bfd',bfd,'k',k,'p',p,'K1',K1,'K2',K2);
            obj.spec.derived = d;

            % --- expand to MACOS elements (light +z, source -z) ---
            psi_M2 = -1;  if greg, psi_M2 = +1; end   % concave secondary -> CoC at +z
            mk = @(name,kind,Vz,psz,Kr,Kc,apr) struct( ...
                'name',name,'kind',kind,'Vpt',[0 0 Vz],'psi',[0 0 psz], ...
                'Kr',Kr,'Kc',Kc,'ap_r',apr,'provenance',['derived(' sp.family ')']);
            e1 = mk('M1','Reflector', 0.0,   -1.0,    -abs(R1), K1, D/2);
            e2 = mk('M2','Reflector', -sep,  psi_M2,  -abs(R2), K2, 0.6*D/2);
            e3 = mk('FP','FocalPlane', bfd,  -1.0,    -1.0e22,  0.0, 0.2*D);
            e1.zElt = sep;  e2.zElt = sep + bfd;  e3.zElt = 1.0e20;
            % pole/nrm/ap/ap_rect complete the canonical schema (empty = on-axis
            % section, no measured aperture)
            [e1.pole,e1.nrm,e1.ap,e1.ap_rect,e1.asph,e1.freeform, ...
             e2.pole,e2.nrm,e2.ap,e2.ap_rect,e2.asph,e2.freeform, ...
             e3.pole,e3.nrm,e3.ap,e3.ap_rect,e3.asph,e3.freeform] = deal([]);
            obj.spec.elt = [e1 e2 e3];
        end

        function [K1,K2] = conics_(~, fam, m, beta, k, p)
        %CONICS_  Family conic constants (§5.1-5.5; β-dependent forms).
            cs = ((m+1)/(m-1))^2;
            switch fam
                case 'cassegrain'
                    K1 = -1.0;  K2 = -cs;
                case 'ritchey_chretien'
                    K1 = -1.0 - 2*(1+beta)/(m^2*(m-beta));
                    K2 = -cs  - 2*m*(m+1)/((m-beta)*(m-1)^3);
                case 'gregorian'
                    K1 = -1.0;  K2 = -((m-1)/(m+1))^2;
                case 'dall_kirkham'
                    K1 = -1.0 + (k^4/p^3)*cs;  K2 = 0.0;   % spherical secondary
                otherwise
                    error('macos:design:Telescope:family','unknown family %s', fam);
            end
        end

        function txt = emit_(obj)
        %EMIT_  Render the spec to MACOS .in text (full double precision).
        %   NOTE: accumulate with L{end+1}=... — an anonymous "append"
        %   helper would capture L by value and silently drop all but the
        %   last line.
            sp = obj.spec;  D = sp.in.D;
            % Source standoff: place the collimated source just ahead of the
            % FRONTMOST optic (the most negative vertex z, usually M2) so the
            % DRAW layout is dominated by the telescope, not the incoming
            % beam -- the old 2.5*D floor drew ~17 m of empty beam on a deep
            % Korsch and made the back end unreadable (Dave 2026-07-05).
            % zSource=1e22 (collimated) makes this WFE-neutral; only the
            % drawn incoming-beam length changes.  Vertex-based so it works
            % for 2-mirror AND N-mirror (no 'sep').
            zmin  = min(arrayfun(@(e) e.Vpt(3), sp.elt));
            stand = max(1.0*D, -zmin + 0.25*D);
            v3 = @(a,b,c) sprintf('%.16E  %.16E  %.16E', a, b, c);
            v6 = @(u,w) sprintf('%.16E  %.16E  %.16E  %.16E  %.16E  %.16E', ...
                                u(1),u(2),u(3),w(1),w(2),w(3));
            % Two off-axis tools, both keeping the parent VERTICES pinned and
            % psi axis-aligned (only the source moves):
            %   field_bias       tilts the chief ray in +y (image off-axis)
            %   aperture_decenter offsets the beam/stop center in +y to an
            %                     off-axis point on the parent (use an
            %                     off-axis patch; off-axis-parabola style)
            % Both zero -> (0,0,1)/(0,0,0), byte-identical to the on-axis emit.
            by    = 0;   if isfield(sp,'field_bias'),       by  = sp.field_bias;       end
            bx    = 0;   % no x field-bias design knob; trace_field overrides below
            apdy  = 0;   if isfield(sp,'aperture_decenter'), apdy = sp.aperture_decenter; end
            if isfield(sp,'trace_field')     % transient: emit for ONE field point
                bx = sp.trace_field(1);  by = sp.trace_field(2);
            end
            cdir  = [sin(bx), sin(by), sqrt(max(0, 1 - sin(bx)^2 - sin(by)^2))];
            apst  = [0, apdy, 0];                  % aperture-stop center (global)
            cpos  = apst - stand*cdir;             % chief ray back-projected through the stop
            L = {};
            L{end+1} = sprintf('%% MACOS prescription emitted by macos.design.Telescope (family=%s)', sp.family);
            L{end+1} = '% Source Definition';
            L{end+1} = ['        ChfRayDir=  ' v3(cdir(1),cdir(2),cdir(3))];
            L{end+1} = ['        ChfRayPos=  ' v3(cpos(1),cpos(2),cpos(3))];
            L{end+1} = '          zSource=1.0E+22';
            L{end+1} = '        BaseUnits=  m';
            L{end+1} = '        WaveUnits=  m';
            L{end+1} = '           IndRef=1.0E+00';
            L{end+1} = '           Extinc=0.0E+00';
            L{end+1} = sprintf('          Wavelen=%.16E', sp.wavelength);
            L{end+1} = '             Flux=1.0E+00';
            L{end+1} = sprintf('         Aperture=%.16E', D);
            L{end+1} = '         Obscratn=0.0E+00';
            L{end+1} = ['         ApStop=  ' v3(apst(1),apst(2),apst(3))];
            L{end+1} = '         GridType=  Circular';
            L{end+1} = sprintf('         nGridpts=  %d', sp.sampling);
            % NOTE (2026-07-18): the heritage corpus (e5mono/dmt6mono)
            % uses xGrid=(-1,0,0), and the SegMirMaker <-> engine
            % segment-tiling contract silently assumes that orientation.
            % The design layer keeps (+1,0,0) -- draw_rays plot-axis
            % signs follow the grid handedness and several consumers
            % (check_clipping legs, realize_apertures, reports) were
            % built on it -- and segment_rx flips the grid line to the
            % heritage orientation in its merged output instead.
            L{end+1} = ['            xGrid=  ' v3(1,0,0)];
            L{end+1} = ['            yGrid=  ' v3(0,1,0)];
            % --- native multi-field optimization block (when configured) ---
            % The nominal ChfRayDir IS field 1 (shares the OptChfRayDir parse
            % block), so OptChfRayDir is emitted for the OFF-axis fields only
            % and OptFOVWt is sized 1+n_off (else a list-directed-read crash).
            if isfield(sp,'opt')
                o = sp.opt;
                L{end+1} = ['        OptTarget=  ' o.target];
                L{end+1} = sprintf('         OptWFElt=  %d', o.wf_elt);
                L{end+1} = sprintf('       OptMaxItrs=  %d', o.max_iters);
                fex = false;
                if isfield(o,'fex') && ~isempty(o.fex), fex = o.fex; end
                if fex
                    L{end+1} = '           OptFEX=  Yes';  %#ok<AGROW>
                else
                    L{end+1} = '           OptFEX=  No';   %#ok<AGROW>
                end
                for j = 1:size(o.fields,1)
                    d  = o.fields(j,:);
                    cp = apst - stand*d;            % through the (decentered) stop
                    L{end+1} = ['     OptChfRayDir=  ' v3(d(1),d(2),d(3))];          %#ok<AGROW>
                    L{end+1} = ['     OptChfRayPos=  ' v3(cp(1),cp(2),cp(3))];       %#ok<AGROW>
                end
                L{end+1} = ['         OptFOVWt=  ' strtrim(sprintf('%.6g  ', o.weights))];
            end
            L{end+1} = '% Element Definitions';
            L{end+1} = sprintf('             nElt=  %d', numel(sp.elt));
            for k = 1:numel(sp.elt)
                e = sp.elt(k);
                L{end+1} = sprintf('             iElt=  %d', k);                  %#ok<AGROW>
                L{end+1} = ['          EltName=  ' e.name];
                L{end+1} = ['          Element=  ' e.kind];
                hasAsph = isfield(e,'asph') && ~isempty(e.asph) && any(e.asph ~= 0);
                % A freeform element emits Surface=Zernike whenever modes are
                % DECLARED -- even with zero coefficients -- so the CALIB OptZern
                % optimizer has a Zernike surface (ZernTypeL/=0) to perturb from
                % a zero seed.  (Clear .freeform to revert a mirror to Conic.)
                hasFree = isfield(e,'freeform') && ~isempty(e.freeform) ...
                          && isstruct(e.freeform) && isfield(e.freeform,'modes') ...
                          && ~isempty(e.freeform.modes);
                if strcmp(e.kind,'FocalPlane') || abs(e.Kr) >= 1e21
                    % focal planes AND flat fold mirrors (add_fold: Reflector
                    % with the flat sentinel Kr=-1e22) emit Surface= Flat
                    L{end+1} = '          Surface=  Flat';
                elseif hasFree
                    % conic base + Zernike departure (§7.1 canonical freeform
                    % representation); KrElt/KcElt held, the Zernike terms add
                    % the figure correction.  This is e5mono's reflective M2 form.
                    L{end+1} = '          Surface=  Zernike';                       %#ok<AGROW>
                elseif hasAsph
                    L{end+1} = '          Surface=  Aspheric';                     %#ok<AGROW>
                else
                    L{end+1} = '          Surface=  Conic';
                end
                L{end+1} = sprintf('            KrElt=%.16E', e.Kr);
                L{end+1} = sprintf('            KcElt=%.16E', e.Kc);
                if hasFree
                    % Zernike-departure block: sparse (modes + coefs), 6/row to
                    % match MACOS's own emit.  lMon = beam footprint radius
                    % (rho=1 at the aperture edge -> standard normalization).
                    % The Mon frame (pMon=Vpt, axes = local surface frame) is the
                    % Zernike evaluation frame; emitted explicitly to match the
                    % engine's known-good round-trip (e5mono/SegDemo3).
                    ft = 'ANSI';
                    if isfield(e.freeform,'type') && ~isempty(e.freeform.type)
                        ft = e.freeform.type;
                    end
                    zm = e.freeform.modes(:).';  zc = e.freeform.coef(:).';
                    nz = numel(zc);
                    L{end+1} = ['         ZernType=  ' ft];                          %#ok<AGROW>
                    L{end+1} = sprintf('        nZernCoef=  %d', nz);               %#ok<AGROW>
                    % ZernModes on ONE line -- the Surface=Zernike parser
                    % (msmacosio.inc:1733) reads all nZernCoef modes from a
                    % single VALUE with no continuation (unlike ZernCoef, which
                    % does wrap).  Matches e5mono M2's 16-on-one-line emit;
                    % wrapping crashed the parser with a list-directed EOF.
                    L{end+1} = ['        ZernModes=  ' strtrim(sprintf('%d ', zm))]; %#ok<AGROW>
                    L{end+1} = ['         ZernCoef=  ' ...                           %#ok<AGROW>
                                strtrim(sprintf('%.16E ', zc(1:min(6,nz))))];
                    for g = 7:6:nz
                        L{end+1} = ['                   ' ...                        %#ok<AGROW>
                                    strtrim(sprintf('%.16E ', zc(g:min(g+5,nz))))];
                    end
                    % Zernike normalization radius: an explicit freeform
                    % lmon (set_freeform) wins over the body ap_r -- for a
                    % beam that underfills the mirror, ap_r-normalized modes
                    % are degenerate over the lit patch (ill-conditioned
                    % OptZern -> huge canceling coefficients)
                    lz = e.ap_r;
                    if isfield(e.freeform,'lmon') && ~isempty(e.freeform.lmon) ...
                            && ~isnan(e.freeform.lmon)
                        lz = e.freeform.lmon;
                    end
                    L{end+1} = sprintf('             lMon=%.16E', lz);              %#ok<AGROW>
                    nrmz = e.psi;
                    if isfield(e,'nrm') && ~isempty(e.nrm), nrmz = e.nrm; end
                    Rz = obj.surf_frame_(nrmz);
                    L{end+1} = ['             pMon=  ' v3(e.Vpt(1),e.Vpt(2),e.Vpt(3))];   %#ok<AGROW>
                    L{end+1} = ['             xMon=  ' v3(Rz(1,1),Rz(2,1),Rz(3,1))];      %#ok<AGROW>
                    L{end+1} = ['             yMon=  ' v3(Rz(1,2),Rz(2,2),Rz(3,2))];      %#ok<AGROW>
                    L{end+1} = ['             zMon=  ' v3(Rz(1,3),Rz(2,3),Rz(3,3))];      %#ok<AGROW>
                end
                if hasAsph && ~hasFree
                    % conic base + AsphCoef (even radial terms h^4,h^6,...).
                    % Parser reads in GROUPS OF 4 per line (crashes otherwise);
                    % count must precede the coefficient array.
                    a = e.asph(:).';  na = numel(a);
                    L{end+1} = sprintf('        nAsphCoef=  %d', na);              %#ok<AGROW>
                    L{end+1} = ['         AsphCoef=  ' ...                          %#ok<AGROW>
                                strtrim(sprintf('%.16E ', a(1:min(4,na))))];
                    for g = 5:4:na
                        L{end+1} = ['                    ' ...                      %#ok<AGROW>
                                    strtrim(sprintf('%.16E ', a(g:min(g+3,na))))];
                    end
                end
                % Off-axis section (engine-true, ConSrf surfsub.F:82): the
                % conic sag is measured from VptElt (parent VERTEX) along
                % psiElt (parent AXIS); RptElt is the section POLE -- the point
                % ON the parent surface at the used sub-aperture center, and the
                % origin of the TElt/perturbation frame.  RptElt is NOT used by
                % the conic intersection, so it never changes the trace; it sets
                % the interface frame + the rigid-body rotation center.  On-axis
                % (no 'pole' field) -> RptElt=VptElt, byte-identical to before.
                pole = e.Vpt;
                if isfield(e,'pole') && ~isempty(e.pole), pole = e.pole; end
                L{end+1} = ['           psiElt=  ' v3(e.psi(1),e.psi(2),e.psi(3))];  %#ok<AGROW>
                L{end+1} = ['           VptElt=  ' v3(e.Vpt(1),e.Vpt(2),e.Vpt(3))];  %#ok<AGROW>
                L{end+1} = ['           RptElt=  ' v3(pole(1),pole(2),pole(3))];      %#ok<AGROW>
                L{end+1} = '           IndRef=1.0E+00';
                L{end+1} = '           Extinc=0.0E+00';
                if isfield(sp,'opt') && any(sp.opt.var_elts == k)
                    % VarElt mask over [TIP TILT CLOCK DX DY PIST ROC CONIC].
                    % Per-element rows (dof_rows) let e.g. M1 stay conic-only
                    % while M2/M3 also decenter+tilt; falls back to the shared
                    % dof_mask row for back-compat.
                    if isfield(sp.opt,'dof_rows')
                        vi   = find(sp.opt.var_elts == k, 1);
                        mask = sp.opt.dof_rows(vi,:);
                    else
                        mask = sp.opt.dof_mask;
                    end
                    L{end+1} = ['           VarElt=  ' ...                        %#ok<AGROW>
                                strtrim(sprintf('%d ', mask))];
                end
                % CALIB Zernike-departure DOF (OptZern= n mode1 .. moden): the
                % freeform figure-correction channel.  Listing modes here adds
                % them to the optimizer (auto-enrolls the elt as VarElt); paired
                % with an all-zero VarElt mask it varies ONLY the Zernike coefs,
                % so radii/conics are held (optimize_freeform).
                if isfield(sp,'opt') && isfield(sp.opt,'zern_elts') ...
                        && any(sp.opt.zern_elts == k)
                    zi  = find(sp.opt.zern_elts == k, 1);
                    zmd = sp.opt.zern_modes{zi};
                    L{end+1} = ['          OptZern=  ' num2str(numel(zmd)) ...    %#ok<AGROW>
                                '  ' strtrim(sprintf('%d ', zmd))];
                end
                % Central perforation (set_hole): emitted as a REAL circular
                % obscuration centered on the vertex -- the trace is
                % physically honest (no glass at the hole; the central rays
                % clip) and the layout views render it (Dave 2026-07-18).
                % xObs defaults from psiElt (ChkDf2); r=0 removes the hole.
                hr = 0;
                if isfield(sp,'holes') && ~isempty(sp.holes)
                    hi = find(strcmp({sp.holes.name}, e.name), 1);
                    if ~isempty(hi), hr = sp.holes(hi).r; end
                end
                if hr > 0
                    L{end+1} = '             nObs=  1';                              %#ok<AGROW>
                    L{end+1} = '          ObsType=  Circle';                         %#ok<AGROW>
                    L{end+1} = ['           ObsVec=  ' v3(hr,0,0)];                  %#ok<AGROW>
                else
                    L{end+1} = '             nObs=  0';                              %#ok<AGROW>
                end
                % Aperture: honor a MEASURED full-field clear aperture when one
                % has been realized (realize_apertures) -- Rectangular on the
                % focal plane, Circular (radius,xc,yc) on the mirrors.  In the
                % off-axis design phase BEFORE apertures are sized, mirrors emit
                % ApType=None (don't clip the decentered/biased beam at a vertex-
                % centered stop -- matches dmt6mono/e5mono).  Otherwise the
                % default vertex-centered circle.
                offaxis = (by ~= 0) || (apst(2) ~= 0);
                hasRect = isfield(e,'ap_rect') && ~isempty(e.ap_rect);
                hasCirc = isfield(e,'ap')      && ~isempty(e.ap);
                if hasRect
                    L{end+1} = '           ApType=  Rectangular';                    %#ok<AGROW>
                    L{end+1} = ['            ApVec=  ' sprintf('%.16E  %.16E  %.16E  %.16E', ...
                                e.ap_rect(1),e.ap_rect(2),e.ap_rect(3),e.ap_rect(4))]; %#ok<AGROW>
                elseif hasCirc
                    L{end+1} = '           ApType=  Circular';                       %#ok<AGROW>
                    L{end+1} = ['            ApVec=  ' v3(e.ap(1),e.ap(2),e.ap(3))]; %#ok<AGROW>
                elseif strcmp(e.kind,'Reflector') && (offaxis || abs(e.Kr) >= 1e21)
                    % off-axis design phase (don't clip the biased beam at a
                    % vertex-centered stop) AND flat folds (add_fold: ap_r is
                    % the check_clipping BODY, not a stop -- a design-phase
                    % fold must not clip silently; realize_apertures sizes
                    % the real aperture later)
                    L{end+1} = '           ApType=  None';                           %#ok<AGROW>
                elseif strcmp(e.kind,'FocalPlane')
                    % same policy for the detector: ap_r is its BODY for
                    % check_clipping, not a field stop.  An honestly-sized
                    % (small) FP emitted as a hard Circular stop makes CALIB
                    % rigid-body trial steps lose EVERY ray the moment the
                    % image walks off it -- no gradient, runaway solve.
                    L{end+1} = '           ApType=  None';                           %#ok<AGROW>
                elseif strcmp(e.kind,'Return')
                    % add_pupil's FP_return + ExitPupil reference surfaces are
                    % REFERENCE geometry, not stops (Dave 2026-07-30): a
                    % Circular ApType at the generous ap_r clips the fast
                    % f/0.86 beam and kills rays at the exit pupil.  No
                    % obscuration on the Return surfaces.
                    L{end+1} = '           ApType=  None';                           %#ok<AGROW>
                else
                    L{end+1} = '           ApType=  Circular';                       %#ok<AGROW>
                    L{end+1} = ['            ApVec=  ' v3(e.ap_r,0,0)];              %#ok<AGROW>
                end
                % PropType: FarField ONLY at the add_pupil exit-pupil
                % sphere (the EP->detector hop is the far-field
                % propagation; enables PSF/Strehl metrics at the FP --
                % Dave 2026-07-06); every other element is Geometric.
                if isfield(obj.spec,'pupil') && ~isempty(obj.spec.pupil) ...
                        && strcmp(e.name,'ExitPupil')
                    L{end+1} = '         PropType=  FarField';                       %#ok<AGROW>
                else
                    L{end+1} = '         PropType=  Geometric';                      %#ok<AGROW>
                end
                L{end+1} = sprintf('             zElt=%.16E', e.zElt);
                % Sensible element coordinate frame (TElt): trace-neutral, but
                % the interface frame MACOS uses for PERTURB + emitted
                % sensitivities (the structures/controls hand-off).  Convention
                % (matches dmt6mono): Z along the OUTWARD SURFACE NORMAL at the
                % pole (RptElt), X/Y tangent to the surface.  For an off-axis
                % section the normal at the pole differs from the parent axis
                % (psi); use e.nrm when present, else psi (on-axis: they
                % coincide).  6x6 block-diagonal [R R]; each line is one COLUMN.
                nrm = e.psi;
                if isfield(e,'nrm') && ~isempty(e.nrm), nrm = e.nrm; end
                R = obj.surf_frame_(nrm);
                L{end+1} = '          nECoord=  6';                              %#ok<AGROW>
                L{end+1} = ['             TElt=  ' v6(R(:,1),[0;0;0])];          %#ok<AGROW>
                L{end+1} = ['                    ' v6(R(:,2),[0;0;0])];          %#ok<AGROW>
                L{end+1} = ['                    ' v6(R(:,3),[0;0;0])];          %#ok<AGROW>
                L{end+1} = ['                    ' v6([0;0;0],R(:,1))];          %#ok<AGROW>
                L{end+1} = ['                    ' v6([0;0;0],R(:,2))];          %#ok<AGROW>
                L{end+1} = ['                    ' v6([0;0;0],R(:,3))];          %#ok<AGROW>
            end
            % REQUIRED trailing block (else SMACOS load -> nElt=0)
            L{end+1} = '% Output Coordinate System Definition';
            L{end+1} = '         nOutCord=  5';
            L{end+1} = ['             Tout=  ' v3(1,0,0) '  ' v3(0,0,0) '  0.0E+00'];
            L{end+1} = ['                    ' v3(0,1,0) '  ' v3(0,0,0) '  0.0E+00'];
            L{end+1} = ['                    ' v3(0,0,0) '  ' v3(1,0,0) '  0.0E+00'];
            L{end+1} = ['                    ' v3(0,0,0) '  ' v3(0,1,0) '  0.0E+00'];
            L{end+1} = ['                    ' v3(0,0,0) '  ' v3(0,0,0) '  1.0E+00'];
            txt = [strjoin(L, newline) newline];
        end

        function f = canon_family_(obj, fam)
        %CANON_FAMILY_  Normalise family name (lowercase + aliases).
            key = lower(regexprep(fam, '[\s_-]', ''));
            if isfield(obj.ALIASES, key)
                f = obj.ALIASES.(key);
            elseif any(strcmp(key, regexprep(obj.FAMILIES,'_','')))
                f = obj.FAMILIES{strcmp(key, regexprep(obj.FAMILIES,'_',''))};
            else
                error('macos:design:Telescope:family', ...
                    ['unknown family ''%s'' (Cassegrain/RC/Gregorian/' ...
                     'Dall-Kirkham).'], fam);
            end
        end

        function L = pick_len_(~, v_m, v_mm, name, allow_signed)
        %PICK_LEN_  Resolve a length given _m and/or _mm forms (SI metres out).
        %   allow_signed (default false) permits a NEGATIVE value -- used for a
        %   mirror RADIUS in the n-flip Seidel convention, where a slowing relay
        %   tertiary (whose beam has crossed an intermediate focus) carries a
        %   negative n-flip radius.  KrElt is still emitted as -|R| (always
        %   negative); the sign only drives the Seidel conic/focus math.  All
        %   other lengths stay positive.
            if nargin < 5, allow_signed = false; end
            has_m = ~isnan(v_m); has_mm = ~isnan(v_mm);
            if has_m && has_mm
                error('macos:design:Telescope:dupUnit', ...
                    'specify %s in metres OR mm, not both.', name);
            elseif has_m,  L = v_m;
            elseif has_mm, L = v_mm * 1e-3;
            else
                error('macos:design:Telescope:missing', ...
                    '%s is required (give %s_m or %s_mm).', name, name, name);
            end
            if allow_signed
                if L == 0
                    error('macos:design:Telescope:sign', '%s must be nonzero.', name);
                end
            elseif ~(L > 0)
                error('macos:design:Telescope:sign', '%s must be positive.', name);
            end
        end

        function e = new_elt_(~, name, kind, Vpt, psi, Kr, ap_r, prov, zElt)
        %NEW_ELT_  Build a spec element struct with the canonical field set
        %   (matches resolve_'s mk()) so it concatenates into spec.elt.
        %   Used by add_pupil for Return surfaces; Kc fixed at 0.  pole/nrm are
        %   part of the canonical schema (empty = on-axis, no off-axis section)
        %   so off-axis and on-axis designs concatenate identically.
            e = struct('name',name, 'kind',kind, 'Vpt',Vpt(:).', ...
                       'psi',psi(:).', 'Kr',Kr, 'Kc',0.0, 'ap_r',ap_r, ...
                       'provenance',prov, 'zElt',zElt, 'pole',[], 'nrm',[], ...
                       'ap',[], 'ap_rect',[], 'asph',[], 'freeform',[]);
            % asph = AsphCoef row; freeform = struct(modes,coef,type) Zernike departure
        end

        function tf = is_nmirror_(obj)
        %IS_NMIRROR_  True for add_mirror-built families (TMA, ...).
            tf = isfield(obj.spec,'is_nmirror') && obj.spec.is_nmirror;
        end

        function m = empty_mirror_list_(~)
        %EMPTY_MIRROR_LIST_  0x0 struct carrying the add_mirror field set.
            m = struct('name',{},'R',{},'t',{},'derive',{},'tilt_deg',{}, ...
                       'convex',{},'conic',{});
        end

        function resolve_nmirror_(obj)
        %RESOLVE_NMIRROR_  Layout + Seidel-seed conics for an N-mirror
        %   coaxial telescope (§5.2 TMA row).  Spacings 1..N-1 are user
        %   values; the last is the derived paraxial focus.  Conics null
        %   3rd-order S_I/II/III (macos.design.seidel_seed).  All mirrors
        %   share psiElt=(0,0,-1); vertices fold along z (the propagation
        %   direction flips each reflection); KrElt=-|R|, KcElt=K.
        %   Validated against the proof_korsch f/8 layout
        %   (R=[8 2 4], t=[3 4.5,derive] -> K~[-0.622 0.148 -3.904]).
            sp  = obj.spec;
            mir = sp.mirrors;
            N   = numel(mir);
            if isfield(mir,'tilt_deg') && any([mir.tilt_deg] ~= 0)
                obj.resolve_nmirror_fold_();     % Bauer tilted-fold unobscuring
                return;
            end
            if N < 3
                error('macos:design:Telescope:nmirror:tooFew', ...
                    'TMA needs >= 3 mirrors via add_mirror (have %d).', N);
            end
            if ~mir(N).derive
                error('macos:design:Telescope:nmirror:lastDerive', ...
                    'the last mirror (%s) spacing must be ''derive'' (the focus).', ...
                    mir(N).name);
            end
            D = sp.in.D;
            R = [mir.R];                         % 1xN radii (magnitudes)
            t_between = zeros(1, N-1);
            for k = 1:N-1
                if mir(k).derive
                    error('macos:design:Telescope:nmirror:midDerive', ...
                        'only the LAST mirror spacing may be ''derive'' (%s is).', ...
                        mir(k).name);
                end
                t_between(k) = mir(k).t;
            end

            cvx = false(1, N);
            if isfield(mir,'convex'), cvx = logical([mir.convex]); end
            [K, t_focus, EFL] = macos.design.seidel_seed(R, t_between, D, cvx);
            if isfield(sp,'base_sphere') && sp.base_sphere
                K = zeros(1, N);            % sphere+Zernike: hold base spheres
            end
            if isfield(mir,'conic')
                % explicit per-mirror Kc seeds (add_mirror 'conic') override
                % the seidel seed -- the carry-optimized-conics path
                for k = 1:N
                    if ~isnan(mir(k).conic), K(k) = mir(k).conic; end
                end
            end
            t = [t_between, t_focus];       % seidel_seed returns the convex-aware
                                            % paraxial focus + K=0 sphere seed

            % fold vertices: propagation dir after mirror k is (-1)^k (the
            % incoming beam travels +z before M1, so z2 = -t1, z3 = -t1+t2, ...).
            z = zeros(1, N+1);
            for k = 1:N
                z(k+1) = z(k) + ((-1)^k) * t(k);
            end
            apr = repmat(0.5*D, 1, N);  apr(1) = 0.55*D;   % generous defaults

            % KrElt=-|R| for EVERY mirror (MACOS convention): convex vs concave
            % is the geometry (a secondary before the M1 focus reflects away
            % from its CoC -> convex), never the radius sign (j18mono's SM).
            %
            % psiElt: the legacy all-(0,0,-1) is kept for mirrors 1..3 -- it
            % is engine-validated for every combination that occurs there
            % (j18mono: concave/+z, convex/-z, concave/+z; note a Korsch M2
            % is often convex BY GEOMETRY with no 'convex' flag, so the flag
            % cannot discriminate at k<=3).  From the 4th mirror on (relay
            % mirrors past a real focus, e.g. the 3+1 M4) the parity rule
            % applies: psi_z = -dir_in for concave (default), +dir_in when
            % flagged convex, where dir_in = (-1)^(k-1) is the beam
            % direction into mirror k.  A 4th mirror CONCAVE to a -z beam
            % needs +1; emitted at -1 it traces CONVEX and diverges the
            % relay.  (The fully general discriminator is the paraxial
            % vergence at each mirror -- follow-on.)
            psiz = -ones(1, N);
            for k = 4:N
                dir_in = (-1)^(k-1);
                if cvx(k), psiz(k) = dir_in; else, psiz(k) = -dir_in; end
            end
            elts = obj.new_elt_(mir(1).name, 'Reflector', [0 0 z(1)], ...
                    [0 0 psiz(1)], -abs(R(1)), apr(1), 'derived(tma+seidel)', t(1));
            elts.Kc = K(1);
            for k = 2:N
                e = obj.new_elt_(mir(k).name, 'Reflector', [0 0 z(k)], ...
                    [0 0 psiz(k)], -abs(R(k)), apr(k), 'derived(tma+seidel)', t(k));
                e.Kc = K(k);
                elts(k) = e;                     %#ok<AGROW>
            end
            fpr = 0.3*D;
            if isfield(sp,'fp_ap_r') && ~isnan(sp.fp_ap_r), fpr = sp.fp_ap_r; end
            elts(N+1) = obj.new_elt_(sp.fp_name, 'FocalPlane', [0 0 z(N+1)], ...
                    [0 0 -1], -1.0e22, fpr, 'derived(tma)', 1.0e20);

            obj.spec.elt     = elts;
            obj.spec.derived = struct('N',N, 'R',R, 'K',K, 't',t, 'z',z, ...
                'EFL',EFL, 'fnum',EFL/D, 't_focus',t_focus);
            obj.apply_folds_();                  % flat folds (no-op if none)
        end

        function resolve_nmirror_fold_(obj)
        %RESOLVE_NMIRROR_FOLD_  Tilted-fold N-mirror layout (Bauer/Schiesser/
        %   Rolland 2018 unobscuring -- TILT each mirror minimally to clear the
        %   beam, NOT decenter the pupil; keeps the system compact, M2 near the
        %   incoming beam).  The chief ray folds M1->M2->M3->FP: each mirror's
        %   normal = the normal-incidence normal (-d_in) rotated by tilt_deg
        %   about x, so the reflection deviates the chief ray by 2*tilt.  Radii
        %   are the user/Bauer values; conics are the seidel n-flip seed (the
        %   rotationally-symmetric starting point -- the tilt-induced field-
        %   dependent aberrations are left for optimize + optimize_freeform per
        %   the Bauer method).  doc/bauer2018_starting_geometry_method.md.
            sp = obj.spec;  mir = sp.mirrors;  N = numel(mir);
            if N < 3
                error('macos:design:Telescope:nmirror:tooFew', ...
                    'TMA needs >= 3 mirrors via add_mirror (have %d).', N);
            end
            if ~mir(N).derive
                error('macos:design:Telescope:nmirror:lastDerive', ...
                    'the last mirror (%s) spacing must be ''derive''.', mir(N).name);
            end
            D = sp.in.D;  R = [mir.R];  tilt = deg2rad([mir.tilt_deg]);
            t_between = zeros(1, N-1);
            for k = 1:N-1
                if mir(k).derive
                    error('macos:design:Telescope:nmirror:midDerive', ...
                        'only the LAST mirror spacing may be ''derive'' (%s is).', mir(k).name);
                end
                t_between(k) = mir(k).t;
            end
            cvx = false(1, N);
            if isfield(mir,'convex'), cvx = logical([mir.convex]); end
            [K, t_focus, EFL] = macos.design.seidel_seed(R, t_between, D, cvx);
            if isfield(sp,'base_sphere') && sp.base_sphere
                K = zeros(1, N);            % sphere+Zernike: hold base spheres
            end
            if isfield(mir,'conic')
                % explicit per-mirror Kc seeds (add_mirror 'conic') -- see
                % resolve_nmirror_
                for k = 1:N
                    if ~isnan(mir(k).conic), K(k) = mir(k).conic; end
                end
            end
            t = [t_between, t_focus];       % seidel_seed: convex-aware focus + K=0

            % fold the chief ray: vertex k on the chief ray, normal = tilted
            % normal-incidence normal; reflect to get the next direction.
            rotx = @(a) [1 0 0; 0 cos(a) -sin(a); 0 sin(a) cos(a)];
            Vpt = zeros(N+1, 3);  psi = zeros(N, 3);
            din = [0 0 1];  v = [0 0 0];
            for k = 1:N
                Vpt(k,:) = v;
                nk = (rotx(tilt(k)) * (-din(:))).';     % psiElt = tilted normal
                nk = nk / norm(nk);
                if isfield(mir,'convex') && mir(k).convex
                    nk = -nk;               % convex: psiElt -> downstream CoC
                end
                psi(k,:) = nk;
                dout = din - 2*(din*nk.')*nk;           % reflect chief ray (sign-
                dout = dout / norm(dout);               % invariant to nk flip)
                v = v + t(k)*dout;  din = dout;
            end
            Vpt(N+1,:) = v;

            apr = repmat(0.5*D, 1, N);  apr(1) = 0.55*D;
            elts = obj.new_elt_(mir(1).name, 'Reflector', Vpt(1,:), psi(1,:), ...
                    -abs(R(1)), apr(1), 'derived(tma+fold)', t(1));
            elts.Kc = K(1);
            for k = 2:N
                e = obj.new_elt_(mir(k).name, 'Reflector', Vpt(k,:), psi(k,:), ...
                    -abs(R(k)), apr(k), 'derived(tma+fold)', t(k));
                e.Kc = K(k);  elts(k) = e;               %#ok<AGROW>
            end
            fpr = 0.3*D;
            if isfield(sp,'fp_ap_r') && ~isnan(sp.fp_ap_r), fpr = sp.fp_ap_r; end
            elts(N+1) = obj.new_elt_(sp.fp_name, 'FocalPlane', Vpt(N+1,:), ...
                    -din, -1.0e22, fpr, 'derived(tma)', 1.0e20);   % FP faces the beam

            obj.spec.elt     = elts;
            obj.spec.derived = struct('N',N, 'R',R, 'K',K, 't',t, 'Vpt',Vpt, ...
                'psi',psi, 'tilt_deg',[mir.tilt_deg], 'EFL',EFL, 'fnum',EFL/D, ...
                't_focus',t_focus, 'folded',true);
            obj.apply_folds_();                  % flat folds (no-op if none)
        end

        function apply_folds_(obj)
        %APPLY_FOLDS_  Insert the queued flat folds (add_fold) into the
        %   resolved element chain.  Each fold: station P sits 'dist' along
        %   the beam after the named element (the beam direction there is
        %   the vertex-to-next-vertex direction -- vertices lie on the axial
        %   chief path); the outgoing direction is 'to'.  Everything
        %   downstream is mapped by the reflection isometry about the fold
        %   plane (M = I - 2nn', n = unit(d_in - d_out)): an EXACT unfold,
        %   so spacings, angles, and the trace are preserved and the flat
        %   contributes zero aberration.  psi of the flat faces the
        %   incoming beam (psi = -n, psi.d_in < 0).  Folds are applied in
        %   add_fold order; to chain, name the previous fold as 'after'.
            sp = obj.spec;
            if ~isfield(sp,'folds') || isempty(sp.folds), return; end
            if isfield(sp,'offaxis_section') && sp.offaxis_section
                error('macos:design:Telescope:fold:section', ...
                    ['add_fold + set_offaxis sections do not compose yet -- ' ...
                     'apply folds to the centered design.']);
            end
            D = sp.in.D;
            for q = 1:numel(sp.folds)
                f = sp.folds(q);
                e = obj.spec.elt;
                k = find(strcmp({e.name}, f.after), 1);
                if isempty(k)
                    error('macos:design:Telescope:fold:after', ...
                        'add_fold ''after'' element ''%s'' not found.', f.after);
                end
                if k == numel(e)
                    error('macos:design:Telescope:fold:last', ...
                        'cannot fold after the last element (%s).', f.after);
                end
                seg  = e(k+1).Vpt - e(k).Vpt;
                slen = norm(seg);
                if ~(f.dist < slen)
                    error('macos:design:Telescope:fold:dist', ...
                        ['fold ''%s'': dist %.4g m >= the %s->%s spacing ' ...
                         '%.4g m -- move the next element back or shorten ' ...
                         'dist.'], f.name, f.dist, f.after, e(k+1).name, slen);
                end
                din  = seg / slen;
                dout = f.to(:).' / norm(f.to);
                if norm(dout - din) < 1e-9
                    error('macos:design:Telescope:fold:straight', ...
                        'fold ''%s'': ''to'' equals the incoming direction.', f.name);
                end
                P = e(k).Vpt + f.dist * din;
                n = (din - dout);  n = n / norm(n);      % reflection normal
                M = eye(3) - 2*(n.'*n);                  % fold-plane isometry
                for j = k+1:numel(e)
                    e(j).Vpt = (P.' + M*(e(j).Vpt.' - P.')).';
                    e(j).psi = (M*e(j).psi.').';
                    if isfield(e(j),'pole') && ~isempty(e(j).pole)
                        e(j).pole = (P.' + M*(e(j).pole.' - P.')).';
                    end
                    if isfield(e(j),'nrm') && ~isempty(e(j).nrm)
                        e(j).nrm = (M*e(j).nrm.').';
                    end
                end
                apr = f.ap_r;  if isnan(apr), apr = 0.1*D; end
                fe = obj.new_elt_(f.name, 'Reflector', P, -n, -1.0e22, ...
                                  apr, 'derived(fold)', slen - f.dist);
                e(k).zElt = f.dist;
                obj.spec.elt = [e(1:k), fe, e(k+1:end)];
            end
        end

        function [bf, EFL] = paraxial_focus_(~, R, t, convex)
        %PARAXIAL_FOCUS_  Back focus after the last mirror + system EFL via the
        %   unfolded thin-lens equivalent (mirror f = R/2; a CONVEX mirror is a
        %   NEGATIVE lens, f = -R/2).  The Seidel seed takes |radii| only, so its
        %   n-flip focus is wrong for a system with a convex secondary (it places
        %   the focus as if the secondary were concave); this recovers the true
        %   paraxial focus.  R = |radii| (1xN), t = inter-mirror spacings (1xN-1),
        %   convex = logical(1xN).  Validated vs the e5mono real-ray intermediate
        %   image (0.585 m beam at M2, image 25 m past M2, f/20.8).
            N = numel(R);
            f = R/2;  f(convex) = -f(convex);     % convex mirror = negative lens
            y = 1.0;  u = 0.0;                     % collimated unit-height marginal
            for k = 1:N
                u = u - y/f(k);                   % thin-lens (mirror) power
                if k < N, y = y + u*t(k); end     % propagate to next vertex
            end
            bf  = -y / u;                          % distance to on-axis crossing
            EFL = 1.0 / abs(u);                    % collimated in, unit input height
        end

        function describe_nmirror_(obj)
        %DESCRIBE_NMIRROR_  Resolved N-mirror design table with provenance.
            if ~isfield(obj.spec,'derived') || ~isfield(obj.spec,'elt') ...
                    || isempty(obj.spec.elt)
                obj.resolve_nmirror_();
            end
            sp = obj.spec; d = sp.derived;
            fprintf('macos.design.Telescope  (family=%s, %d mirrors)\n', sp.family, d.N);
            fprintf('  inputs [user]:  D=%.6g m\n', sp.in.D);
            fprintf('  derived(layout): EFL=%.6g m  (f/%.4g)  focus=%.6g m\n', ...
                d.EFL, d.fnum, d.t_focus);
            fprintf('  %-6s %13s %13s %13s\n', 'mirror','R (m)','conic K','spacing (m)');
            for k = 1:d.N
                fprintf('  %-6s %13.6g %13.6g %13.6g   [seidel]\n', ...
                    sp.mirrors(k).name, d.R(k), d.K(k), d.t(k));
            end
            fprintf('  %d elements:\n', numel(sp.elt));
            for k = 1:numel(sp.elt)
                e = sp.elt(k);
                fprintf('   %2d  %-10s %-10s Vpt=[% .4g % .4g % .4g]  [%s]\n', ...
                    k, e.name, e.kind, e.Vpt(1), e.Vpt(2), e.Vpt(3), e.provenance);
            end
        end

        function h = paraxial_heights_(obj)
        %PARAXIAL_HEIGHTS_  Marginal ray radius at each element from a folded
        %   paraxial trace (collimated full-aperture input; mirror n-flip;
        %   inter-element distance = |dz| between vertices).  Flat surfaces
        %   (FP/Return, |Kr| huge) pass the ray through.  Used by diagram()
        %   and check_clipping() to draw / test the beam vs element bodies.
            e = obj.spec.elt;  n = numel(e);
            z = arrayfun(@(x) x.Vpt(3), e);
            h = zeros(1, n);
            nn = 1.0;  yy = obj.spec.in.D/2;  u = 0.0;
            for k = 1:n
                h(k) = abs(yy);
                R  = abs(e(k).Kr);  c = 1/R;       % flat -> R huge -> c ~ 0
                np = -nn;  phi = (np-nn)*c;
                u  = (nn*u - yy*phi)/np;
                if k < n
                    yy = yy + abs(z(k+1)-z(k))*u;
                end
                nn = np;
            end
        end

        function [su, sv] = surface_profile_(~, e, cU, cV, extent, cenUV, woff)
        %SURFACE_PROFILE_  Conic-sag profile of element e projected onto the
        %   (cU,cV) plane axes (for view_layout).  Sag s(r) along psi at
        %   transverse radius r; a flat surface (huge |Kr|) becomes a straight
        %   segment perpendicular to psi.  Optional CENUV = [u v] is the
        %   USED-section center in the plane: the profile is drawn over
        %   [h0-extent, h0+extent] about it (the off-axis section), not about
        %   the vertex.  Optional WOFF is the OUT-OF-PLANE offset of that
        %   section center from the vertex (along the third axis): the conic sag
        %   uses the FULL transverse radius r = sqrt(h^2 + woff^2), so an
        %   off-axis slice (e.g. M1 in XZ while the beam is decentered in y)
        %   sits at the correct depth instead of the y=0 sag.  (Assumes the
        %   out-of-plane axis is ~perpendicular to psi -- true for pinned-axis
        %   off-axis sections; tilted folds would need a full 3-D slice.)
            Rsig = -e.Kr;  Kc = e.Kc;   % |radius| (Kr=-|R| always, so Rsig > 0):
            %   a convex secondary (convex by geometry, not by Kr sign) is drawn
            %   with the same |R| sphere as a concave one -- a known minor
            %   cosmetic caveat; the trace/conics are unaffected.
            apr = e.ap_r;
            if nargin >= 5 && extent > 0, apr = extent; end
            vu = e.Vpt(cU);  vv = e.Vpt(cV);
            pu = e.psi(cU);  pv = e.psi(cV);
            np = hypot(pu, pv);  if np > 0, pu = pu/np;  pv = pv/np; end
            tu = -pv;  tv = pu;                       % in-plane transverse
            h0 = 0;
            if nargin >= 6 && ~isempty(cenUV)
                h0 = (cenUV(1)-vu)*tu + (cenUV(2)-vv)*tv;   % in-plane section offset
            end
            w = 0;  if nargin >= 7 && ~isempty(woff), w = woff; end   % out-of-plane offset
            h = linspace(h0-apr, h0+apr, 41);
            if abs(Rsig) > 1e15
                s = zeros(size(h));                  % flat (FP / Return)
            else
                c    = 1/Rsig;                       % signed curvature
                r2   = h.^2 + w.^2;                  % full transverse radius^2
                disc = 1 - (1+Kc)*c^2*r2;  disc(disc < 0) = 0;
                s = c*r2 ./ (1 + sqrt(disc));        % signed sag (convex -> s < 0)
            end
            su = vu + h.*tu + s.*pu;                  % vertex + h*t + sag*psi
            sv = vv + h.*tv + s.*pv;
        end
    end

    methods (Static, Access = private)
        function obj = from_spec_(sp)
            npts = 41;  if isfield(sp,'sampling'), npts = sp.sampling; end
            if isfield(sp,'is_nmirror') && sp.is_nmirror
                obj = macos.design.Telescope('family', sp.family, ...
                    'aperture_diameter_m', sp.in.D, ...
                    'model_size', sp.model_size, 'wavelength_m', sp.wavelength, ...
                    'grid_npts', npts);
                obj.spec.mirrors = sp.mirrors;
                if isfield(sp,'fp_name'), obj.spec.fp_name = sp.fp_name; end
                if isfield(sp,'fp_ap_r'), obj.spec.fp_ap_r = sp.fp_ap_r; end
                if isfield(sp,'folds'),   obj.spec.folds   = sp.folds;   end
                if isfield(sp,'holes'),   obj.spec.holes   = sp.holes;   end
                if isfield(sp,'base_sphere')
                    obj.spec.base_sphere = sp.base_sphere;
                end
                if isfield(sp,'offaxis_section')
                    obj.spec.offaxis_section = sp.offaxis_section;
                end
                % keep the RESOLVED elements verbatim: CALIB's rigid-body
                % moves, center_focal_plane's FP position, measured
                % apertures, and freeform/asph departures live only there
                % -- a bare re-resolve from the mirror list loses them
                % (this is what made a saved folded design come back
                % unfolded).  build() re-resolves only when elt is empty.
                if isfield(sp,'elt') && ~isempty(sp.elt)
                    obj.spec.elt = sp.elt;
                    if isfield(sp,'derived'), obj.spec.derived = sp.derived; end
                else
                    obj.spec.elt = [];                   % re-resolve at build
                end
            else
                obj = macos.design.Telescope( ...
                    'family', sp.family, 'aperture_diameter_m', sp.in.D, ...
                    'system_fnum', sp.in.system_fnum, 'primary_fnum', sp.in.primary_fnum, ...
                    'BFD_m', sp.in.BFD, 'model_size', sp.model_size, ...
                    'wavelength_m', sp.wavelength);
            end
            if isfield(sp,'field_points'), obj.spec.field_points = sp.field_points; end
            if isfield(sp,'field_bias'),   obj.spec.field_bias = sp.field_bias; end
            if isfield(sp,'aperture_decenter'), obj.spec.aperture_decenter = sp.aperture_decenter; end
            if isfield(sp,'bandwidth'),    obj.spec.bandwidth = sp.bandwidth; end
        end
    end
end

function sel = slice_(pa, pb, tol, nrays)
%SLICE_ (file-local)  Rays within one grid pitch of a pupil meridian
%   (|pa| < tol), ordered ALONG the fan (by pb) and evenly subsampled to
%   ~nrays -- a contiguous, symmetric fan with no gap at the meridian.
    m = find(abs(pa) < tol);
    if isempty(m), sel = m; return; end
    [~, o] = sort(pb(m));  m = m(o);
    n = max(2, nrays);
    if numel(m) > n, m = m(round(linspace(1, numel(m), n))); end
    sel = m;
end
