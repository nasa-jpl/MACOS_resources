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
        FAMILIES = {'cassegrain','ritchey_chretien','gregorian','dall_kirkham'}
        ALIASES  = struct('cass','cassegrain', 'classicalcassegrain','cassegrain', ...
                          'rc','ritchey_chretien', 'ritchey','ritchey_chretien', ...
                          'ritcheychretien','ritchey_chretien', ...
                          'greg','gregorian', 'classicalgregorian','gregorian', ...
                          'dk','dall_kirkham', 'dallkirkham','dall_kirkham')
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
                    'family is required (Cassegrain/RC/Gregorian/Dall-Kirkham).');
            end
            fam = obj.canon_family_(opts.family);
            D   = obj.pick_len_(opts.aperture_diameter_m, opts.aperture_diameter_mm, ...
                                'aperture_diameter');
            BFD = obj.pick_len_(opts.BFD_m, opts.BFD_mm, 'BFD');
            if isnan(opts.system_fnum) || isnan(opts.primary_fnum)
                error('macos:design:Telescope:fnum', ...
                    'both system_fnum and primary_fnum are required (2-mirror).');
            end
            if ~(opts.system_fnum > 0) || ~(opts.primary_fnum > 0)
                error('macos:design:Telescope:fnumSign', ...
                    'system_fnum and primary_fnum must be positive.');
            end
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
            sp.in.D            = D;
            sp.in.system_fnum  = opts.system_fnum;
            sp.in.primary_fnum = opts.primary_fnum;
            sp.in.BFD          = BFD;
            obj.spec = sp;
            obj.resolve_();                              % derive layout + conics + elements
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
            sp = obj.spec;  D = sp.in.D;  f1 = sp.derived.f1;
            stand = 10*f1;                       % collimated -> position irrelevant
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
    end

    methods (Static, Access = private)
        function obj = from_spec_(sp)
            obj = macos.design.Telescope( ...
                'family', sp.family, 'aperture_diameter_m', sp.in.D, ...
                'system_fnum', sp.in.system_fnum, 'primary_fnum', sp.in.primary_fnum, ...
                'BFD_m', sp.in.BFD, 'model_size', sp.model_size, ...
                'wavelength_m', sp.wavelength);
            if isfield(sp,'field_points'), obj.spec.field_points = sp.field_points; end
            if isfield(sp,'bandwidth'),    obj.spec.bandwidth = sp.bandwidth; end
        end
    end
end
