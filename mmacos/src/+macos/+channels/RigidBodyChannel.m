classdef RigidBodyChannel < handle
%MACOS.CHANNELS.RIGIDBODYCHANNEL  One-DOF rigid-body perturbation channel.
%
%   A single (element, DOF) pair with DOF in {Rx, Ry, Rz, Tx, Ty, Tz}.
%   Rotations are in radians; translations are passed in SI metres
%   (macos.perturb converts to BaseUnits via CBM internally).
%
%   macos's CPERTURB_PROG is INCREMENTAL -- each call adds to the
%   element's current pose.  To support the central-difference
%   pattern  apply(+d) -> measure -> apply(-d) -> measure -> restore()
%   this class tracks its own cumulative state and passes
%   (value - current) as the increment each call.  Restore is
%   apply(0).
%
%   Construction:
%     ch = macos.channels.RigidBodyChannel(session, iElt, dof_idx)
%
%   Inputs:
%     session  macos.Session handle.
%     iElt     1-based element id (an "actual optic" -- not Reference
%              / Return / FocalPlane; the rigid_body_channels builder
%              handles the eligibility filtering).
%     dof_idx  0..5 -> {Rx, Ry, Rz, Tx, Ty, Tz}.

    properties (SetAccess = private)
        iElt    (1,1) double
        dof_idx (1,1) double
        session
    end
    properties (Access = protected)
        current (1,1) double = 0
    end

    properties (Constant)
        DOF_LABELS = {'Rx','Ry','Rz','Tx','Ty','Tz'}
    end

    methods
        function obj = RigidBodyChannel(session, iElt, dof_idx)
            arguments
                session
                iElt    (1,1) double {mustBeInteger, mustBePositive}
                dof_idx (1,1) double {mustBeInteger, ...
                            mustBeGreaterThanOrEqual(dof_idx, 0), ...
                            mustBeLessThanOrEqual(dof_idx, 5)}
            end
            obj.session = session;
            obj.iElt    = iElt;
            obj.dof_idx = dof_idx;
        end

        function apply(obj, value)
            arguments
                obj
                value (1,1) double
            end
            increment = value - obj.current;
            if increment ~= 0
                obj.do_perturb(increment);
            end
            obj.current = value;
        end

        function restore(obj)
            obj.apply(0);
        end

        function s = name(obj)
            s = sprintf('Elt %d %s', obj.iElt, ...
                obj.DOF_LABELS{obj.dof_idx + 1});
        end

        function k = kind(~)
            k = 'RigidBody';
        end
    end

    methods (Access = protected)
        function do_perturb(obj, increment)
            rot = [0; 0; 0];
            trans = [0; 0; 0];
            if obj.dof_idx < 3
                rot(obj.dof_idx + 1) = increment;
            else
                trans(obj.dof_idx - 2) = increment;
            end
            obj.session.perturb(obj.iElt, ...
                'rotation', rot, ...
                'translation', trans, ...
                'frame', 'local');
            % MODIFY clears macos's "trace state is current" cache so
            % the next trace re-derives the geometry with the new pose.
            % Without it sensitivities silently come back as zero --
            % same gotcha as the Zernike channels.
            obj.session.modify();
        end
    end
end
