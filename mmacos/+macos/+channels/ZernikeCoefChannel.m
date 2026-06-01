classdef ZernikeCoefChannel < handle
%MACOS.CHANNELS.ZERNIKECOEFCHANNEL  One-mode Zernike sensitivity channel.
%
%   A single (element, mode) pair on either a MonZern, FFZern, or
%   Zern coefficient array.  Snapshots its nominal coefficient on
%   the first apply() so that apply(+d) followed by apply(-d) and
%   then restore() produces a clean +d / -d / 0 sequence for central
%   finite-difference Jacobian construction (see dw_dz_zernike).
%
%   Construction:
%     ch = macos.channels.ZernikeCoefChannel(session, iElt, mode, kind)
%
%   Inputs:
%     session  macos.Session instance (the wavefront engine).
%     iElt     1-based element id (must match the chosen kind).
%     mode     1-based Zernike mode index.
%     kind     'MonZern' | 'FFZern' | 'Zern'.
%
%   Methods:
%     apply(d)   Write snapshot+d to the coefficient.  Reads the
%                current value on first call and caches it as snapshot.
%     restore()  Write snapshot back (no-op before the first apply).
%     name()     Human-readable identifier 'Elt N <Kind><mode>'.
%
%   See also: macos.channels.MonZernChannel,
%             macos.channels.FFZernChannel,
%             macos.channels.ZernChannel,
%             macos.channels.zernike_channels,
%             macos.channels.freeform_monzern_channels,
%             macos.channels.freeform_ffzern_channels.

    properties (SetAccess = private)
        iElt    (1,1) double
        mode    (1,1) double
        kind    (1,:) char
        session                  % macos.Session handle
    end
    properties (Access = private)
        snapshot (1,1) double = 0
        snapshot_taken (1,1) logical = false
    end

    methods
        function obj = ZernikeCoefChannel(session, iElt, mode, kind)
            arguments
                session
                iElt (1,1) double {mustBeInteger, mustBePositive}
                mode (1,1) double {mustBeInteger, mustBePositive}
                kind (1,:) char {mustBeMember(kind, ...
                    {'MonZern','FFZern','Zern'})}
            end
            obj.session = session;
            obj.iElt    = iElt;
            obj.mode    = mode;
            obj.kind    = kind;
        end

        function apply(obj, increment)
            arguments
                obj
                increment (1,1) double
            end
            if ~obj.snapshot_taken
                obj.snapshot = obj.get_current();
                obj.snapshot_taken = true;
            end
            obj.set_value(obj.snapshot + increment);
            % MODIFY invalidates macos's "trace state is current"
            % caches so the next trace re-runs ZerntoMon and re-
            % evaluates the FreeForm / Zernike surface with the new
            % coefficient.  Without this, sensitivities silently
            % return zero -- the trace reuses the cached MonCoef.
            obj.session.modify();
        end

        function restore(obj)
            if obj.snapshot_taken
                obj.set_value(obj.snapshot);
                obj.session.modify();
            end
        end

        function s = name(obj)
            s = sprintf('Elt %d %s%d', obj.iElt, obj.kind, obj.mode);
        end
    end

    methods (Access = private)
        function c = get_current(obj)
            switch obj.kind
                case 'MonZern'
                    c = obj.session.get_elt_mon_zrn_coef(obj.iElt, obj.mode);
                case 'FFZern'
                    c = obj.session.get_elt_ff_zrn_coef(obj.iElt, obj.mode);
                case 'Zern'
                    c = obj.session.get_elt_zrn_coef(obj.iElt, obj.mode);
            end
            c = c(1);
        end
        function set_value(obj, val)
            switch obj.kind
                case 'MonZern'
                    obj.session.set_elt_mon_zrn_coef(obj.iElt, ...
                        obj.mode, val);
                case 'FFZern'
                    obj.session.set_elt_ff_zrn_coef(obj.iElt, ...
                        obj.mode, val);
                case 'Zern'
                    obj.session.set_elt_zrn_coef(obj.iElt, ...
                        obj.mode, val);
            end
        end
    end
end
