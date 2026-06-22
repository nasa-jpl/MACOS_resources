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
            arguments
                obj
                name (1,:) char
                opts.radius_m         (1,1) double = NaN
                opts.radius_mm        (1,1) double = NaN
                opts.spacing_after_m  (1,1) double = NaN
                opts.spacing_after_mm (1,1) double = NaN
                opts.spacing_after    (1,:) char   = ''
            end
            if ~obj.is_nmirror_()
                error('macos:design:Telescope:add_mirror:family', ...
                    'add_mirror is for N-mirror families (family=%s).', obj.spec.family);
            end
            R = obj.pick_len_(opts.radius_m, opts.radius_mm, 'radius');
            derive = strcmpi(strtrim(opts.spacing_after), 'derive');
            if derive
                t = NaN;
            else
                t = obj.pick_len_(opts.spacing_after_m, opts.spacing_after_mm, ...
                                  'spacing_after');
            end
            obj.spec.mirrors(end+1) = struct('name',name, 'R',R, 't',t, 'derive',derive);
            obj.spec.elt = [];                           % invalidate -> re-resolve
        end

        function add_focal_plane(obj, name)
        %ADD_FOCAL_PLANE  Name the terminal focal plane of an N-mirror
        %   telescope (default 'FP'); placed at the derived focus at build.
            arguments, obj, name (1,:) char = 'FP', end
            if ~obj.is_nmirror_()
                error('macos:design:Telescope:add_focal_plane:family', ...
                    'add_focal_plane is for N-mirror families.');
            end
            obj.spec.fp_name = name;
            obj.spec.elt     = [];
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
            flatRet = obj.new_elt_('FP_return', 'Return', FP_Vpt, [0 0 1], ...
                                   -1.0e22, apR, 'derived(add_pupil)', rSeed);
            sphRet  = obj.new_elt_('ExitPupil', 'Return', seed, [0 0 1], ...
                                   -abs(rSeed), apR, 'derived(fex)', rSeed);
            obj.spec.elt = [obj.spec.elt(1:ielt-1), flatRet, sphRet, ...
                            obj.spec.elt(ielt:end)];
            obj.build();                           % emit + load the augmented Rx

            % --- locate the exit pupil with FEX (axis crossing of an
            %     off-axis chief ray). XP lands at nElt-1 = the EP slot. ---
            iEP = ielt + 1;  iFPnew = ielt + 2;  nE = numel(obj.spec.elt);
            cur = macos.get_src_fov();
            macos.set_src_fov('src_dir', ...      % off-axis field first ...
                [sin(opts.field_rad); 0; cos(opts.field_rad)]);
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
        %     'dofs'          (1,8) VarElt mask [TIP TILT CLOCK DX DY PIST ROC
        %                     CONIC] (default [0 0 0 0 0 0 0 1] = conic only).
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
                opts.dofs          (1,8) double = [0 0 0 0 0 0 0 1]  % per-elt VarElt mask
            end
            if ~all(ismember(opts.dofs, [0 1]))
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
            var_elts = find(arrayfun(@(e) strcmp(e.kind,'Reflector'), obj.spec.elt));
            if isempty(var_elts)
                error('macos:design:Telescope:optimize:noMirror', ...
                    'no Reflector elements to vary.');
            end
            Nv     = numel(var_elts);                     % # conic DOFs
            fp_elt = numel(obj.spec.elt);                 % terminal FocalPlane
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
                'var_elts',var_elts, 'dof_mask',opts.dofs);
            obj.build();                                  % emit opt block -> load
            r = macos.calib();

            % read back per-element params CALIB may have moved, into the spec
            % (for describe()/view_layout); the deliverable handling differs
            % for conic-only vs geometry-moving runs (see below).
            Kopt = zeros(1, Nv);
            for j = 1:Nv
                k = var_elts(j);
                Kopt(j)             = macos.get_elt_kc(k);
                obj.spec.elt(k).Kc  = Kopt(j);
                obj.spec.elt(k).Kr  = macos.get_elt_kr(k);          % ROC DOF
                obj.spec.elt(k).psi = reshape(macos.get_elt_psi(k), 1, 3); % tilt
                obj.spec.elt(k).Vpt = reshape(macos.get_elt_vpt(k), 1, 3); % decenter
            end
            if isfield(obj.spec.derived,'K'), obj.spec.derived.K = Kopt; end
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
            arguments
                obj
                planes                     = {'YZ','XZ'}
                opts.hide    (1,:) double  = []
                opts.istart  (1,1) double  = 0
                opts.iend    (1,1) double  = 0
                opts.nrays   (1,1) double  = 25
                opts.save    (1,:) char    = ''
                opts.visible (1,1) logical = true
            end
            pl  = obj.plane_list_(planes);
            np  = numel(pl);
            obj.ensure_loaded_();
            vis = 'on';  if ~opts.visible, vis = 'off'; end
            fig = figure('Visible',vis, 'Position',[40 60 min(520*np,1560) 540]);
            tl  = tiledlayout(fig, 1, np, 'TileSpacing','compact', 'Padding','compact');
            for i = 1:np
                ax = nexttile(tl);
                obj.draw_plane_(ax, pl{i}, opts.hide, opts.istart, opts.iend, opts.nrays);
            end
            sgtitle(tl, sprintf('%s -- orthographic layout (real rays)', obj.spec.family), ...
                    'Interpreter','none');
            if ~isempty(opts.save), print(fig, opts.save, '-dpng', '-r150'); end
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
            arguments
                obj
                scan struct
                opts.kind (1,:) char {mustBeMember(opts.kind,{'contour','surf'})} = 'contour'
                opts.save (1,:) char = ''
                opts.visible (1,1) logical = true
            end
            fx = scan.fields(:,1);  fy = scan.fields(:,2);  w = scan.wfe(:);
            ux = uniquetol(fx, 1e-9);  uy = uniquetol(fy, 1e-9);
            vis = 'on';  if ~opts.visible, vis = 'off'; end
            fig = figure('Visible',vis, 'Position',[60 60 620 500]);
            isgrid = numel(ux) >= 2 && numel(uy) >= 2 && ...
                     numel(w) == numel(ux)*numel(uy);
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
                    contourf(ux, uy, W, 12);  axis equal tight;
                end
            else
                scatter(fx, fy, 45, w, 'filled');  axis equal tight;
            end
            cb = colorbar;  cb.Label.String = 'RMS WFE (waves)';
            xlabel('\theta_x (arcmin)');  ylabel('\theta_y (arcmin)');
            title(sprintf('%s -- RMS WFE over field', obj.spec.family), 'Interpreter','none');
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

            % --- per-element disk geometry + physical-body flag
            Vpt = zeros(3,nE);  psi = zeros(3,nE);  apr = zeros(1,nE);
            isBody = false(1,nE);
            for k = 1:nE
                Vpt(:,k) = e(k).Vpt(:);
                p = e(k).psi(:);  np = norm(p);  if np > 0, p = p/np; end
                psi(:,k) = p;  apr(k) = e(k).ap_r;
                isBody(k) = any(strcmp(e(k).kind, {'Reflector','FocalPlane'}));
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
        %   Returns scan: .fields (Kx2 arcmin) .wfe (waves, per field) .lambda
        %                 .aperture (struct array: name/shape/radius/center/rect).
            arguments
                obj
                opts.fields_arcmin (1,:) double = []
                opts.fields (:,2) double = []
                opts.margin (1,1) double = 0.05
                opts.quiet  (1,1) logical = false
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
                if ~isempty(v), wfe(j) = std(v)/lam; end
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
            scan = struct('fields', rad2deg(F)*60, 'wfe',wfe, ...
                          'lambda',lam, 'aperture',ap);

            if ~opts.quiet
                fa = rad2deg(F(:,2))*60;
                fprintf(['realize_apertures  (%d fields, +y %g..%g arcmin, ' ...
                         'family=%s)\n'], nF, min(fa), max(fa), obj.spec.family);
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

        function draw_plane_(obj, ax, plane, hide, istart, iend, nrays)
        %DRAW_PLANE_  Draw the real-ray layout for ONE plane into axes AX -- the
        %   shared core of view_layout / view_orthoviews.  Assumes the current
        %   design is already loaded (see ensure_loaded_).
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
            step = max(1, floor(b.nray / max(2, nrays)));
            for r = 1:step:b.nray
                m = b.nper(r);
                if m >= 2
                    plot(ax, b.U(1:m,r), b.V(1:m,r), '-', 'Color',[0 .45 .85], ...
                         'LineWidth',0.5, 'HandleVisibility','off');
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
            [e1.pole,e1.nrm,e1.ap,e1.ap_rect, ...
             e2.pole,e2.nrm,e2.ap,e2.ap_rect, ...
             e3.pole,e3.nrm,e3.ap,e3.ap_rect] = deal([]);
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
            % Source standoff: place the collimated source a couple aperture
            % diameters in front of the FRONTMOST optic (the most negative
            % vertex z) so the DRAW layout stays interpretable -- the old
            % 10*f1 put it ~100 m out.  zSource=1e22 (collimated) makes this
            % WFE-neutral; only the drawn incoming-beam length changes.
            % Vertex-based so it works for 2-mirror AND N-mirror (no 'sep').
            zmin  = min(arrayfun(@(e) e.Vpt(3), sp.elt));
            stand = max(2.5*D, -zmin + 0.5*D);
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
                L{end+1} = '           OptFEX=  No';
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
                if strcmp(e.kind,'FocalPlane')
                    L{end+1} = '          Surface=  Flat';
                else
                    L{end+1} = '          Surface=  Conic';
                end
                L{end+1} = sprintf('            KrElt=%.16E', e.Kr);
                L{end+1} = sprintf('            KcElt=%.16E', e.Kc);
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
                    % VarElt mask over [TIP TILT CLOCK DX DY PIST ROC CONIC]
                    L{end+1} = ['           VarElt=  ' ...                        %#ok<AGROW>
                                strtrim(sprintf('%d ', sp.opt.dof_mask))];
                end
                L{end+1} = '             nObs=  0';
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
                elseif offaxis && strcmp(e.kind,'Reflector')
                    L{end+1} = '           ApType=  None';                           %#ok<AGROW>
                else
                    L{end+1} = '           ApType=  Circular';                       %#ok<AGROW>
                    L{end+1} = ['            ApVec=  ' v3(e.ap_r,0,0)];              %#ok<AGROW>
                end
                L{end+1} = '         PropType=  Geometric';
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

        function L = pick_len_(~, v_m, v_mm, name)
        %PICK_LEN_  Resolve a length given _m and/or _mm forms (SI metres out).
        %   All design-layer lengths are POSITIVE magnitudes -- including a
        %   mirror radius, where convexity is encoded by geometry (Cassegrain
        %   spacing), not the radius sign (see add_mirror / the MACOS KrElt=-|R|
        %   convention).
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
            if ~(L > 0)
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
                       'ap',[], 'ap_rect',[]);   % measured clear apertures
        end

        function tf = is_nmirror_(obj)
        %IS_NMIRROR_  True for add_mirror-built families (TMA, ...).
            tf = isfield(obj.spec,'is_nmirror') && obj.spec.is_nmirror;
        end

        function m = empty_mirror_list_(~)
        %EMPTY_MIRROR_LIST_  0x0 struct carrying the add_mirror field set.
            m = struct('name',{},'R',{},'t',{},'derive',{});
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

            [K, t_focus, EFL] = macos.design.seidel_seed(R, t_between, D);
            t = [t_between, t_focus];

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
            elts = obj.new_elt_(mir(1).name, 'Reflector', [0 0 z(1)], ...
                    [0 0 -1], -abs(R(1)), apr(1), 'derived(tma+seidel)', t(1));
            elts.Kc = K(1);
            for k = 2:N
                e = obj.new_elt_(mir(k).name, 'Reflector', [0 0 z(k)], ...
                    [0 0 -1], -abs(R(k)), apr(k), 'derived(tma+seidel)', t(k));
                e.Kc = K(k);
                elts(k) = e;                     %#ok<AGROW>
            end
            elts(N+1) = obj.new_elt_(sp.fp_name, 'FocalPlane', [0 0 z(N+1)], ...
                    [0 0 -1], -1.0e22, 0.3*D, 'derived(tma)', 1.0e20);

            obj.spec.elt     = elts;
            obj.spec.derived = struct('N',N, 'R',R, 'K',K, 't',t, 'z',z, ...
                'EFL',EFL, 'fnum',EFL/D, 't_focus',t_focus);
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
            if isfield(sp,'is_nmirror') && sp.is_nmirror
                obj = macos.design.Telescope('family', sp.family, ...
                    'aperture_diameter_m', sp.in.D, ...
                    'model_size', sp.model_size, 'wavelength_m', sp.wavelength);
                obj.spec.mirrors = sp.mirrors;
                if isfield(sp,'fp_name'), obj.spec.fp_name = sp.fp_name; end
                obj.spec.elt = [];                       % re-resolve at build
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
