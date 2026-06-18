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

        function rx = build(obj, path, opts)
        %BUILD  Emit the prescription and validate it by loading via SMACOS.
        %   rx = t.build()           -> writes a temp .in, returns its path
        %   rx = t.build('foo.in')   -> writes foo.in
        %   Name-value: 'validate' (default true) load-checks the emitted
        %   Rx through SMACOS (the path pymacos/mmacos use); 'init'
        %   (default true) inits the engine at the spec model_size first.
            arguments
                obj
                path (1,:) char = ''
                opts.validate (1,1) logical = true
                opts.init     (1,1) logical = true
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
            flatRet = obj.new_elt_('PupImg', 'Return', FP_Vpt, [0 0 1], ...
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
        %     'max_iters'     CALIB iteration cap (default 60).
        %     'target'        'WFE' (default).
        %     'weights'       FoV weights, length 1+numel(fields) (default equal).
        %
        %   Returns: .converged, .n_fov, .fields_arcmin (incl. on-axis 0),
        %   .wfe_before/.wfe_after (per field, metres), .conics (optimised K),
        %   .wavelength.  Optimised conics are written back to the spec, so a
        %   subsequent save()/add_pupil() emits the clean optimised design.
            arguments
                obj
                opts.engine        (1,:) char = 'native'
                opts.fields_arcmin (1,:) double = [1.2 2.4]
                opts.max_iters     (1,1) double {mustBeInteger,mustBePositive} = 60
                opts.target        (1,:) char = 'WFE'
                opts.weights       (1,:) double = []
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
            off    = deg2rad(opts.fields_arcmin(:).'/60); % off-axis half-angles (rad)
            dirs   = [zeros(numel(off),1), sin(off(:)), cos(off(:))];
            nfov   = 1 + numel(off);
            w = opts.weights;  if isempty(w), w = ones(1,nfov); end
            if numel(w) ~= nfov
                error('macos:design:Telescope:optimize:weights', ...
                    'weights must have 1+numel(fields_arcmin) = %d entries.', nfov);
            end

            obj.spec.opt = struct('target',opts.target, 'wf_elt',fp_elt, ...
                'max_iters',opts.max_iters, 'fields',dirs, 'weights',w, ...
                'var_elts',var_elts);
            obj.build();                                  % emit opt block -> load
            r = macos.calib();

            % read the optimised conics back into the spec
            Kopt = zeros(1, Nv);
            for j = 1:Nv
                k = var_elts(j);
                Kopt(j) = macos.get_elt_kc(k);
                obj.spec.elt(k).Kc = Kopt(j);
            end
            if isfield(obj.spec.derived,'K'), obj.spec.derived.K = Kopt; end
            obj.spec = rmfield(obj.spec, 'opt');          % clean deliverable
            obj.build('', 'init', false);                 % re-emit clean optimised Rx

            res = struct('converged',r.converged, 'n_fov',r.n_fov, ...
                'fields_arcmin',[0, opts.fields_arcmin(:).'], ...
                'wfe_before',r.old_wfe(:,1).', 'wfe_after',r.new_wfe(:,1).', ...
                'conics',Kopt, 'var_elts',var_elts, 'wavelength',obj.spec.wavelength);
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
            L = {};
            L{end+1} = sprintf('%% MACOS prescription emitted by macos.design.Telescope (family=%s)', sp.family);
            L{end+1} = '% Source Definition';
            L{end+1} = ['        ChfRayDir=  ' v3(0,0,1)];
            L{end+1} = ['        ChfRayPos=  ' v3(0,0,-stand)];
            L{end+1} = '          zSource=1.0E+22';
            L{end+1} = '        BaseUnits=  m';
            L{end+1} = '        WaveUnits=  m';
            L{end+1} = '           IndRef=1.0E+00';
            L{end+1} = '           Extinc=0.0E+00';
            L{end+1} = sprintf('          Wavelen=%.16E', sp.wavelength);
            L{end+1} = '             Flux=1.0E+00';
            L{end+1} = sprintf('         Aperture=%.16E', D);
            L{end+1} = '         Obscratn=0.0E+00';
            L{end+1} = ['         ApStop=  ' v3(0,0,0)];
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
                    d = o.fields(j,:);
                    L{end+1} = ['     OptChfRayDir=  ' v3(d(1),d(2),d(3))];          %#ok<AGROW>
                    L{end+1} = ['     OptChfRayPos=  ' ...                            %#ok<AGROW>
                                 v3(-stand*d(1),-stand*d(2),-stand*d(3))];
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
                L{end+1} = ['           psiElt=  ' v3(e.psi(1),e.psi(2),e.psi(3))];
                L{end+1} = ['           VptElt=  ' v3(e.Vpt(1),e.Vpt(2),e.Vpt(3))];
                L{end+1} = ['           RptElt=  ' v3(e.Vpt(1),e.Vpt(2),e.Vpt(3))];
                L{end+1} = '           IndRef=1.0E+00';
                L{end+1} = '           Extinc=0.0E+00';
                if isfield(sp,'opt') && any(sp.opt.var_elts == k)
                    L{end+1} = '           VarElt=  0 0 0 0 0 0 0 1';  % conic DOF
                end
                L{end+1} = '             nObs=  0';
                L{end+1} = '           ApType=  Circular';
                L{end+1} = ['            ApVec=  ' v3(e.ap_r,0,0)];
                L{end+1} = '         PropType=  Geometric';
                L{end+1} = sprintf('             zElt=%.16E', e.zElt);
                L{end+1} = '          nECoord=  -6';
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
        %   Used by add_pupil for Return surfaces; Kc fixed at 0.
            e = struct('name',name, 'kind',kind, 'Vpt',Vpt(:).', ...
                       'psi',psi(:).', 'Kr',Kr, 'Kc',0.0, 'ap_r',ap_r, ...
                       'provenance',prov, 'zElt',zElt);
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
            if isfield(sp,'bandwidth'),    obj.spec.bandwidth = sp.bandwidth; end
        end
    end
end
