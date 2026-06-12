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
        %     'dofs'        rigid DOF indices 0..5 (default all 6).
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
                opts.dofs       (1,:) double = 0:5
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
                out.rigid = macos.dw_dx(m, sp.rx_path, 'dofs', opts.dofs, ...
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
        end
    end
end
