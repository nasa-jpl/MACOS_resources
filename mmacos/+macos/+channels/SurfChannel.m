classdef SurfChannel < handle
%MACOS.CHANNELS.SURFCHANNEL  One-DOF powered-surface perturbation (Kr or Kc).
%
%   A single (element, param) pair with param in {'Kr','Kc'} -- the base
%   radius of curvature (KrElt, BaseUnits) or conic constant (KcElt,
%   dimensionless) of a POWERED optic (Reflector / Refractor with
%   |Kr| << the flat sentinel 1e22).  Used to build dw/dKr and dw/dKc
%   wavefront sensitivities (macos.dw_dsurf).
%
%   Kr and Kc are ABSOLUTE element parameters (not incremental like
%   CPERTURB), so apply(value) simply sets the parameter to nominal+value
%   via the absolute setter -- no cumulative bookkeeping needed.  restore()
%   is apply(0).  Mirrors the RigidBodyChannel / ZernChannel contract
%   (apply / restore / name / kind) consumed by dwdz_for_current_source.
%
%   Construction:
%     ch = macos.channels.SurfChannel(session, iElt, 'Kr')
%
%   Inputs:
%     session  macos.Session handle (used for modify()).
%     iElt     1-based element id of a powered optic.
%     param    'Kr' (radius) | 'Kc' (conic).

    properties (SetAccess = private)
        iElt    (1,1) double
        param   (1,:) char        % 'Kr' | 'Kc'
        session
        nominal (1,1) double      % nominal parameter value
    end
    properties (Access = protected)
        current (1,1) double = 0
    end

    methods
        function obj = SurfChannel(session, iElt, param)
            arguments
                session
                iElt  (1,1) double {mustBeInteger, mustBePositive}
                param (1,:) char {mustBeMember(param, {'Kr','Kc'})}
            end
            obj.session = session;
            obj.iElt    = iElt;
            obj.param   = param;
            switch param
                case 'Kr', obj.nominal = macos.get_elt_kr(iElt);
                case 'Kc', obj.nominal = macos.get_elt_kc(iElt);
            end
        end

        function apply(obj, value)
            arguments
                obj
                value (1,1) double
            end
            switch obj.param
                case 'Kr', macos.set_elt_kr(obj.iElt, obj.nominal + value);
                case 'Kc', macos.set_elt_kc(obj.iElt, obj.nominal + value);
            end
            % MODIFY clears macos's "trace state is current" cache so the
            % next trace re-derives the surface with the new Kr/Kc -- same
            % gotcha as the Zernike / rigid-body channels (else dw/d* == 0).
            obj.session.modify();
            obj.current = value;
        end

        function restore(obj)
            obj.apply(0);
        end

        function s = name(obj)
            s = sprintf('Elt %d %s', obj.iElt, obj.param);
        end

        function k = kind(~)
            k = 'Surface';
        end
    end
end
