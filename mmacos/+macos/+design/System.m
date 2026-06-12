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
