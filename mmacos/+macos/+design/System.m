classdef System < handle
%MACOS.DESIGN.SYSTEM  Imported optical system — the design-layer analysis front-end.
%   The front-end-agnostic analysis core (PLAN_DESIGN_LAYER §1.0,
%   Sprint 2A-i).  `from_rx` loads a prescription via SMACOS and reads
%   its element parameters back through the existing `+macos` getter
%   surface — engine readback, NOT a MATLAB text parser — into a plain
%   design-spec struct (the §3 state-as-data rule).
%
%   The expected dominant entry point is a CodeV/Zemax-converted Rx:
%       s = macos.design.System.from_rx('Rx_Cass_FarField.in');
%       s.describe();
%
%   One Fortran session per MATLAB process: a System owns the live
%   engine state for its model_size; constructing a second System (or
%   calling interactive macos.* commands) over a running analysis
%   corrupts it (PLAN_DESIGN_LAYER §1.2).
%
%   This is the import spine.  `vary` / `evaluate` / `sensitivities`
%   (the rest of Sprint 2A-i) build on the spec it produces.
%
%   See also: macos.load_rx, macos.num_elt, macos.get_elt_vpt.

    properties (SetAccess = private)
        spec   % plain struct — the design spec (see from_rx for layout)
    end

    methods (Static)
        function s = from_rx(rx_path, opts)
        %FROM_RX  Import a prescription into a design System via engine readback.
        %   s = macos.design.System.from_rx(PATH) inits the engine (at
        %   the default model_size), loads PATH, and reads its element +
        %   source parameters back into s.spec.
        %
        %   Name-value:
        %     'model_size'  engine model size (default 128).  Fixed for
        %                   the life of the System (one size per study,
        %                   PLAN_DESIGN_LAYER §1.2).
        %     'init'        logical, default true.  Pass false to reuse
        %                   an already-initialised session at the same
        %                   model_size (e.g. inside a worker).
        %
        %   s.spec fields:
        %     .source      'import'
        %     .rx_path      char
        %     .model_size   scalar
        %     .units        macos.sys_units() struct + .cbm (BaseUnits→m)
        %     .src          macos.get_src_fov() struct + .wvl + .sampling
        %     .n_elt        element count
        %     .elt(k)       per-element readback (BaseUnits, as the
        %                   engine returns): .vpt .rpt .psi  (each 3×1),
        %                   .provenance = 'imported'
        %
        %   Element geometry is stored in the Rx's BaseUnits (faithful
        %   to the engine); .units.cbm converts to SI metres.  The
        %   forthcoming `vary` layer applies the +macos SI convention at
        %   the user surface.
            arguments
                rx_path (1,:) char
                opts.model_size (1,1) double {mustBeInteger, mustBePositive} = 128
                opts.init (1,1) logical = true
            end

            if opts.init
                macos.init(opts.model_size);
            end
            macos.load_rx(rx_path);
            if ~macos.has_rx()
                error('macos:design:System:loadFailed', ...
                    'from_rx: no prescription loaded after load_rx(%s).', rx_path);
            end

            spec = struct();
            spec.source     = 'import';
            spec.rx_path     = rx_path;
            spec.model_size  = opts.model_size;

            u       = macos.sys_units();
            u.cbm   = macos.cbm();
            spec.units = u;

            src          = macos.get_src_fov();
            src.wvl       = macos.get_src_wvl();
            src.sampling  = macos.get_src_sampling();
            spec.src      = src;

            n = macos.num_elt();
            spec.n_elt = n;
            elt = repmat(struct('vpt',[],'rpt',[],'psi',[],'provenance',''), n, 1);
            for k = 1:n
                elt(k).vpt        = macos.get_elt_vpt(k);
                elt(k).rpt        = macos.get_elt_rpt(k);
                elt(k).psi        = macos.get_elt_psi(k);
                elt(k).provenance = 'imported';
            end
            spec.elt = elt;

            s = macos.design.System();
            s.spec = spec;
        end
    end

    methods
        function n = n_elt(obj)
        %N_ELT  Element count of the imported system.
            n = obj.spec.n_elt;
        end

        function out = sensitivities(obj, opts)
        %SENSITIVITIES  OPD sensitivity Jacobians for the imported system.
        %   out = s.sensitivities() runs the bitwise-verified Phase 7
        %   sensitivity drivers (PLAN_DESIGN_LAYER §1.0 / Sprint 2A-i,
        %   PLAN.md §5.4) and returns their results side by side — it
        %   HARVESTS the existing channel machinery, it does not re-derive
        %   FD.  The two Jacobians are kept as SEPARATE matrices (current
        %   practice); join them yourself if you prefer a single matrix:
        %       J = [out.rigid.dwdx, out.zern.dwdz];   % same Nw rows
        %
        %     'rigid' -> out.rigid = macos.dw_dx        (rigid-body / src / FP)
        %     'zern'  -> out.zern  = macos.dw_dz_zernike (MonZern / Zern,
        %                                       auto-discovered from the Rx)
        %
        %   Name-value pairs:
        %     'families'    cellstr subset of {'rigid','zern'} (default both).
        %                   A family not requested -> [] in the output.
        %                   'zern' yields an empty dwdz if the Rx has no
        %                   Zernike-eligible elements.
        %     'dofs'        rigid DOF NAMES, cellstr subset of
        %                   {'Rx','Ry','Rz','Tx','Ty','Tz'} (default all
        %                   6).  The design layer is name-based; the
        %                   0-based dw_dx index convention is translated
        %                   internally and never exposed here.
        %
        %   FRAME: per-element rigid DOFs are in the element's LOCAL
        %   (EltCoord / TElt) frame — e.g. 'Ty' on a tilted OAP is along
        %   that element's own y-axis, NOT global Y.  This is the mount-
        %   aligned convention tolerancing wants, and it is inherited
        %   from the Phase 7 RigidBodyChannel (which perturbs local).
        %   Global-frame per-element sensitivities are not yet exposed
        %   (would need a RigidBodyChannel frame arg).
        %     'fp_mode'     'track'(default)|'srs'|'sxp'|'none' (dw_dx).
        %     'zern_kinds'  {'monzern','zern'} default (dw_dz_zernike).
        %     'zmode_start' lowest Zernike mode (default 4).
        %     'n_zcoef'     highest Zernike mode (default 15).
        %     'delta_rigid' rigid FD step (default 1e-8).
        %     'delta_zern'  Zernike FD step (default 1e-6).
        %     'verbose'     logical (default false).
        %
        %   Output struct:
        %     rigid       full macos.dw_dx result struct, or [].
        %                 (.dwdx Nw×Nz, .channel_names, .iElt, .dof_idx,
        %                  .w_nom_vec, …)
        %     zern        full macos.dw_dz_zernike result struct, or [].
        %                 (.dwdz Nw×Nz, .channel_names, .iElt, .mode,
        %                  .kind, .w_nom_vec, …)
        %     families, rx_path, model_size.
        %
        %   Until nominal snapshot/restore lands (§9.1 Q9 → PLAN.md §11.7)
        %   each driver reloads the Rx; that is correct (Q5 certified
        %   repeated load/trace is bit-stable) just not yet cheap.
            arguments
                obj
                opts.families   (1,:) cell   = {'rigid','zern'}
                opts.dofs       (1,:) cell   = {'Rx','Ry','Rz','Tx','Ty','Tz'}
                opts.fp_mode    (1,:) char   = 'track'
                opts.zern_kinds (1,:) cell   = {'monzern','zern'}
                opts.zmode_start (1,1) double = 4
                opts.n_zcoef    (1,1) double = 15
                opts.delta_rigid (1,1) double = 1e-8
                opts.delta_zern  (1,1) double = 1e-6
                opts.verbose    (1,1) logical = false
            end
            sp = obj.spec;
            m  = macos.Session(sp.model_size);

            out = struct();
            out.rigid      = [];
            out.zern       = [];
            out.families   = opts.families;
            out.rx_path    = sp.rx_path;
            out.model_size = sp.model_size;

            if any(strcmp('rigid', opts.families))
                out.rigid = macos.dw_dx(m, sp.rx_path, ...
                    'dofs', obj.dofs_to_idx_(opts.dofs), ...
                    'fp_mode', opts.fp_mode, 'delta', opts.delta_rigid, ...
                    'verbose', opts.verbose);
            end
            if any(strcmp('zern', opts.families))
                out.zern = macos.dw_dz_zernike(m, sp.rx_path, ...
                    'kinds', opts.zern_kinds, 'zmode_start', opts.zmode_start, ...
                    'n_zcoef', opts.n_zcoef, 'delta', opts.delta_zern, ...
                    'verbose', opts.verbose);
            end
        end

        function vary(obj, elt, param, opts)
        %VARY  Declare a design variable (an optimizer-driven override).
        %   s.vary(ELT, PARAM, ...) records a free DOF for optimize()
        %   (PLAN_DESIGN_LAYER §5.4: vary shares addressing with override;
        %   a design var IS an optimizer-driven override).  It only edits
        %   the spec — no engine call.
        %
        %   ELT    integer element index (1..n_elt).  (Name addressing is
        %          a builder-path / follow-on concern; import addresses
        %          by index.)
        %
        %   FRAME: rigid DOFs are in the element LOCAL (EltCoord / TElt)
        %   frame — 'Ty' is the element's own y-axis, not global Y (see
        %   sensitivities()).  Local is the only per-element frame today.
        %   PARAM  one of:
        %     rigid single DOF : 'Rx' 'Ry' 'Rz' 'Tx' 'Ty' 'Tz'
        %     rigid aliases    : 'despace'(Tz), 'tilt'(Rx+Ry),
        %                        'decenter'(Tx+Ty)   [expand to 2 vars]
        %     Zernike          : 'zern' with 'mode', N
        %     surface          : 'conic' / 'asphere'  (declared, but no
        %                        Phase 7 sensitivity channel yet — see
        %                        sensitivities(); FD-from-scratch pending)
        %
        %   Name-value:
        %     'bounds' [lo hi]  optimizer bounds (lo<hi).  Default [NaN NaN].
        %     'unit'   char     'm'|'mm'|'rad'|'mrad'|'lambdaD'|''  (sugar;
        %                       bare SI is canonical — §10 Made #11).
        %     'role'   'design'(default) | 'compensator'  (compensator =
        %              solved inner-loop per outer iterate, §5.4).
        %     'mode'   N         Zernike mode index (required for 'zern').
        %
        %   See also: design_vars, clear_vars, override.
            arguments
                obj
                elt   (1,1) double {mustBeInteger, mustBePositive}
                param (1,:) char
                opts.bounds (1,2) double = [NaN NaN]
                opts.unit   (1,:) char = ''
                opts.role   (1,:) char {mustBeMember(opts.role, ...
                              {'design','compensator'})} = 'design'
                opts.mode   (1,1) double = NaN
            end
            if elt > obj.spec.n_elt
                error('macos:design:System:eltRange', ...
                    'vary: element %d out of range (n_elt=%d).', ...
                    elt, obj.spec.n_elt);
            end
            if ~any(isnan(opts.bounds)) && opts.bounds(1) >= opts.bounds(2)
                error('macos:design:System:bounds', ...
                    'vary: bounds must be [lo hi] with lo<hi; got [%g %g].', ...
                    opts.bounds(1), opts.bounds(2));
            end
            recs = obj.resolve_param_(elt, param, opts);
            if ~isfield(obj.spec, 'vars') || isempty(obj.spec.vars)
                obj.spec.vars = recs;
            else
                obj.spec.vars = [obj.spec.vars, recs];
            end
        end

        function v = design_vars(obj)
        %DESIGN_VARS  Struct array of declared design variables (or []).
            if isfield(obj.spec, 'vars'), v = obj.spec.vars; else, v = []; end
        end

        function clear_vars(obj)
        %CLEAR_VARS  Remove all declared design variables.
            obj.spec.vars = [];
        end

        function describe(obj)
        %DESCRIBE  Print the imported element table with provenance.
        %   Everything in an imported System has provenance 'imported'
        %   (read back from the engine); the builder path adds
        %   derived/optimized provenance later (PLAN_DESIGN_LAYER §2).
            sp = obj.spec;
            fprintf('macos.design.System  (source=%s)\n', sp.source);
            fprintf('  rx_path    : %s\n', sp.rx_path);
            fprintf('  model_size : %d\n', sp.model_size);
            fprintf('  wavelength : %g (WaveUnits)\n', sp.src.wvl);
            fprintf('  sampling   : %d\n', sp.src.sampling);
            fprintf('  cbm        : %g (BaseUnits->m)\n', sp.units.cbm);
            fprintf('  %d elements (Vpt, BaseUnits) [provenance]\n', sp.n_elt);
            for k = 1:sp.n_elt
                v = sp.elt(k).vpt;
                fprintf('   %2d  [% .6g % .6g % .6g]  [%s]\n', ...
                    k, v(1), v(2), v(3), sp.elt(k).provenance);
            end
            vars = obj.design_vars();
            if ~isempty(vars)
                fprintf('  %d design variable(s):\n', numel(vars));
                for j = 1:numel(vars)
                    dv = vars(j);
                    if strcmp(dv.family, 'zern')
                        tag = sprintf('zern mode %d', dv.mode);
                    else
                        tag = dv.dof_name;
                    end
                    fprintf('   Elt %2d  %-12s [%s]  bounds=[%g %g] %s {%s}\n', ...
                        dv.elt, tag, dv.family, dv.bounds(1), dv.bounds(2), ...
                        dv.unit, dv.role);
                end
            end
        end
    end

    methods (Access = private)
        function idx = dofs_to_idx_(obj, names)
        %DOFS_TO_IDX_  Translate rigid DOF names -> dw_dx 0-based indices.
        %   The ONE place the design layer touches the 0-based dw_dx
        %   convention; the user surface stays name-based.
            idx = zeros(1, numel(names));
            for i = 1:numel(names)
                k = find(strcmpi(obj.DOF_NAMES, names{i}), 1);
                if isempty(k)
                    error('macos:design:System:dofName', ...
                        ['unknown rigid DOF name ''%s'' (expected one of ' ...
                         'Rx Ry Rz Tx Ty Tz).'], names{i});
                end
                idx(i) = k - 1;            % 1-based MATLAB find -> 0-based dw_dx
            end
        end

        function recs = resolve_param_(obj, elt, param, opts)
        %RESOLVE_PARAM_  Map a vary() (elt,param) to design-var record(s).
        %   Rigid params resolve by NAME (no 0-based index stored);
        %   aliases tilt/decenter expand to two records.
            base = struct('elt', elt, 'param', param, 'family', '', ...
                          'dof_name', '', 'mode', NaN, ...
                          'bounds', opts.bounds, 'unit', opts.unit, ...
                          'role', opts.role);
            switch lower(param)
                case {'rx','ry','rz','tx','ty','tz'}
                    r = base; r.family = 'rigid';
                    k = find(strcmpi(obj.DOF_NAMES, param), 1);
                    r.dof_name = obj.DOF_NAMES{k};      % canonical case
                    recs = r;
                case 'despace'
                    r = base; r.family = 'rigid'; r.dof_name = 'Tz';
                    recs = r;
                case 'tilt'
                    a = base; a.family = 'rigid'; a.dof_name = 'Rx';
                    b = base; b.family = 'rigid'; b.dof_name = 'Ry';
                    recs = [a, b];
                case 'decenter'
                    a = base; a.family = 'rigid'; a.dof_name = 'Tx';
                    b = base; b.family = 'rigid'; b.dof_name = 'Ty';
                    recs = [a, b];
                case 'zern'
                    if isnan(opts.mode)
                        error('macos:design:System:zernMode', ...
                            'vary(...,''zern''): supply ''mode'', N.');
                    end
                    r = base; r.family = 'zern'; r.mode = opts.mode;
                    recs = r;
                case {'conic','asphere'}
                    r = base; r.family = 'surface';
                    warning('macos:design:System:noChannel', ...
                        ['vary(%d,''%s''): surface param has no Phase 7 ' ...
                         'sensitivity channel yet; sensitivities() skips ' ...
                         'it (FD-from-scratch pending).'], elt, param);
                    recs = r;
                otherwise
                    error('macos:design:System:param', ...
                        ['vary: unrecognised param ''%s'' (expected ' ...
                         'Rx/Ry/Rz/Tx/Ty/Tz, despace/tilt/decenter, ' ...
                         'zern, conic/asphere).'], param);
            end
        end
    end

    properties (Constant, Access = private)
        DOF_NAMES = {'Rx','Ry','Rz','Tx','Ty','Tz'}   % dw_dx idx 0..5
    end
end
