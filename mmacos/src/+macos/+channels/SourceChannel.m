classdef SourceChannel < handle
%MACOS.CHANNELS.SOURCECHANNEL  One-DOF rigid-body perturbation on the source.
%
%   Wraps macos.perturb_src (the iElt=0 branch of CPERTURB).  The
%   source rotates ChfRayDir and translates ChfRayPos.  A source
%   perturbation moves the chief ray off the stop centre, so this
%   channel re-enforces the stop after every perturbation:
%
%     stop_mode='obj'  (default): call macos.stop_obj(*stop_obj_pos).
%                                 Right for Rxes that declare
%                                 ApStop= x y z.
%     stop_mode='elt'           : call macos.stop(stop_elt).  Right
%                                 when the system stop is an element.
%     stop_mode='none'          : skip; the chief ray drifts off-stop
%                                 with the source.  Diagnostic only.
%
%   DOF layout matches RigidBodyChannel: 0..5 -> {Rx, Ry, Rz, Tx, Ty, Tz}.
%
%   perturb_src is INCREMENTAL like CPERTURB_PROG, so the
%   cumulative-state pattern (current tracks the running perturbation;
%   each apply sends value - current) mirrors RigidBodyChannel.

    properties (SetAccess = private)
        dof_idx       (1,1) double
        stop_mode     (1,:) char
        stop_obj_pos  (1,3) double = [0 0 0]
        stop_elt      (1,1) double = 0
        session
    end
    properties (Access = private)
        current (1,1) double = 0
    end
    properties (Constant)
        DOF_LABELS = {'Rx','Ry','Rz','Tx','Ty','Tz'}
    end

    methods
        function obj = SourceChannel(session, dof_idx, opts)
            arguments
                session
                dof_idx (1,1) double {mustBeInteger, ...
                            mustBeGreaterThanOrEqual(dof_idx, 0), ...
                            mustBeLessThanOrEqual(dof_idx, 5)}
                opts.stop_mode    (1,:) char {mustBeMember( ...
                    opts.stop_mode, {'obj','elt','none'})} = 'obj'
                opts.stop_obj_pos (1,3) double = [0 0 0]
                opts.stop_elt     (1,1) double {mustBeInteger} = 0
            end
            if strcmp(opts.stop_mode, 'elt') && opts.stop_elt <= 0
                error('macos:channels:SourceChannel:stop_elt', ...
                    'stop_mode=''elt'' requires stop_elt > 0');
            end
            obj.session       = session;
            obj.dof_idx       = dof_idx;
            obj.stop_mode     = opts.stop_mode;
            obj.stop_obj_pos  = opts.stop_obj_pos;
            obj.stop_elt      = opts.stop_elt;
        end

        function apply(obj, value)
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
            s = sprintf('Src %s', obj.DOF_LABELS{obj.dof_idx + 1});
        end

        function k = kind(~)
            k = 'Source';
        end
    end

    methods (Access = private)
        function do_perturb(obj, increment)
            rot = [0; 0; 0];
            trans = [0; 0; 0];
            if obj.dof_idx < 3
                rot(obj.dof_idx + 1) = increment;
            else
                trans(obj.dof_idx - 2) = increment;
            end
            obj.session.perturb_src('rotation', rot, ...
                                     'translation', trans);
            obj.enforce_stop();
            obj.session.modify();
        end

        function enforce_stop(obj)
            switch obj.stop_mode
                case 'obj'
                    obj.session.stop_obj(obj.stop_obj_pos(1), ...
                                          obj.stop_obj_pos(2), ...
                                          obj.stop_obj_pos(3));
                case 'elt'
                    obj.session.stop(obj.stop_elt);
                case 'none'
                    % no-op
            end
        end
    end
end
