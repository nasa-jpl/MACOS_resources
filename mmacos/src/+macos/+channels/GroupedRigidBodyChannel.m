classdef GroupedRigidBodyChannel < handle
%MACOS.CHANNELS.GROUPEDRIGIDBODYCHANNEL  Rigid-body perturbation of a member group.
%
%   Dispatches macos's GPERTURB (CPERTURB_GRP_DVR) to perturb the
%   declared members as a single rigid unit.  Like the per-element
%   RigidBodyChannel this is INCREMENTAL: each apply(value) sends
%   (value - current) as the perturbation increment so the standard
%   apply(+d) -> measure -> apply(-d) -> measure -> restore() central-
%   difference pattern works.
%
%   Why macos-side and not Python/MATLAB-side superposition: per-
%   element rigid-body columns can in principle be linearly combined
%   to synthesize a group column, but combinations involving the
%   Reference/Return surfaces around the exit pupil and the focal-
%   plane element do NOT superimpose linearly -- a rigid camera
%   (lens + FP) motion produces an EP/FP rigid coupling that cancels
%   across the Reference and the FP into a small residual which
%   superposition of two individually-large columns can't reproduce
%   within numerical precision.  Letting macos perturb the members as
%   a rigid unit captures the cancellation directly.
%
%   Group declaration is installed dynamically per apply(): the
%   existing EltGrp on ref_elt is snapshotted, the desired members
%   are written, GPERTURB runs, and the snapshot is restored when
%   restore() is called.  This permits OVERLAPPING groups across
%   separate channels even though macos's EltGrp data structure
%   only allows one group per element at a time.
%
%   Rotations are in radians; translations are in SI metres (converted
%   internally to BaseUnits via CBM to match the prb_grp signature).
%   That is the SAME convention RigidBodyChannel uses -- macos.perturb
%   does the identical division internally -- so a group column and a
%   per-element column are both OPD-per-metre and a single numeric FD
%   'delta' is the same PHYSICAL poke for either.  The 6-vector is
%   [Rx, Ry, Rz, Tx, Ty, Tz] in ref_elt's frame (coords='global' by
%   default; 'local' = ref_elt's TElt frame).
%
%   Optional FP follow-up for groups containing a focal-plane element
%   (fp_elt > 0):
%     'none'  (default if no FP): GPERTURB only.
%     'sxp'   (auto-default if FP in group): trace + sxp to refine
%             EP from post-perturbation chief-ray geometry.
%     'srs'   trace + srs(ep, fp) to slave EP pose to the moved FP.
%   (fex deferred -- no Fortran wrapper.)

    properties (SetAccess = private)
        members      (:,1) double
        dof_idx      (1,1) double
        group_name   (1,:) char
        ref_elt      (1,1) double
        fp_elt       (1,1) double = 0
        fp_mode      (1,:) char   = 'none'
        ep_elt       (1,1) double = -1
        coords       (1,:) char   = 'global'
        stop_mode    (1,:) char   = 'obj'
        stop_obj_pos (1,3) double = [0 0 0]
        stop_elt     (1,1) double = 0
        session
    end
    properties (Access = private)
        current      (1,1) double = 0
        saved_grp                = []   % column vec or [] before install
        saved_taken  (1,1) logical = false
        cbm_         (1,1) double = 0   % metres per BaseUnit; captured at
                                        % construction, resolved lazily if
                                        % no Rx was loaded yet (0 = unknown)
    end
    properties (Constant)
        DOF_LABELS = {'Rx','Ry','Rz','Tx','Ty','Tz'}
    end

    methods
        function obj = GroupedRigidBodyChannel(session, members, dof_idx, opts)
            arguments
                session
                members (:,1) double {mustBeInteger, mustBePositive}
                dof_idx (1,1) double {mustBeInteger, ...
                            mustBeGreaterThanOrEqual(dof_idx, 0), ...
                            mustBeLessThanOrEqual(dof_idx, 5)}
                opts.group_name   (1,:) char = ''
                opts.ref_elt      (1,1) double {mustBeInteger} = 0
                opts.fp_elt       (1,1) double {mustBeInteger} = 0
                opts.fp_mode      (1,:) char {mustBeMember(opts.fp_mode, ...
                                      {'auto','none','sxp','srs'})} = 'auto'
                opts.ep_elt       (1,1) double {mustBeInteger} = -1
                opts.coords       (1,:) char {mustBeMember(opts.coords, ...
                                      {'global','local'})} = 'global'
                opts.stop_mode    (1,:) char {mustBeMember(opts.stop_mode, ...
                                      {'obj','elt','none'})} = 'obj'
                opts.stop_obj_pos (1,3) double = [0 0 0]
                opts.stop_elt     (1,1) double {mustBeInteger} = 0
            end
            if numel(members) < 2
                error('macos:channels:GroupedRigidBodyChannel:size', ...
                    'group needs at least 2 members; got %d', ...
                    numel(members));
            end
            ref = opts.ref_elt;
            if ref == 0
                ref = members(1);
            end
            if ~any(members == ref)
                error('macos:channels:GroupedRigidBodyChannel:ref', ...
                    'ref_elt=%d must be one of members=%s', ...
                    ref, mat2str(members(:).'));
            end
            mode = opts.fp_mode;
            if strcmp(mode, 'auto')
                if opts.fp_elt > 0 && any(members == opts.fp_elt)
                    mode = 'sxp';
                else
                    mode = 'none';
                end
            end
            nm = opts.group_name;
            if isempty(nm)
                nm = sprintf('%d-%d', min(members), max(members));
            end
            if strcmp(opts.stop_mode, 'elt') && opts.stop_elt <= 0
                error('macos:channels:GroupedRigidBodyChannel:stop', ...
                    'stop_mode=''elt'' requires stop_elt > 0');
            end

            obj.session      = session;
            obj.members      = members(:);
            obj.dof_idx      = dof_idx;
            obj.group_name   = nm;
            obj.ref_elt      = ref;
            obj.fp_elt       = opts.fp_elt;
            obj.fp_mode      = mode;
            obj.ep_elt       = opts.ep_elt;
            obj.coords       = opts.coords;
            obj.stop_mode    = opts.stop_mode;
            obj.stop_obj_pos = opts.stop_obj_pos;
            obj.stop_elt     = opts.stop_elt;
            % CBM for the SI-metres -> BaseUnits translation conversion.
            % Captured here (the Rx is loaded by the time the builder
            % runs) but NOT required: a channel constructed before a load
            % keeps 0 and resolves lazily on its first TRANSLATION poke.
            % Rotations never need it.
            try
                obj.cbm_ = session.cbm();
            catch
                obj.cbm_ = 0;
            end
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
            obj.restore_group();
        end

        function s = name(obj)
            s = sprintf('Grp[%s] %s', obj.group_name, ...
                obj.DOF_LABELS{obj.dof_idx + 1});
        end

        function k = kind(~)
            k = 'Group';
        end
    end

    methods (Access = private)
        function install_group(obj)
            if obj.saved_taken
                return;
            end
            cur = obj.session.get_elt_grp(obj.ref_elt);
            obj.saved_grp = cur(:);
            obj.saved_taken = true;
            target = obj.members(:);
            if numel(cur) ~= numel(target) ...
                    || ~isequal(sort(cur), sort(target))
                obj.session.set_elt_grp(obj.ref_elt, target);
            end
        end

        function restore_group(obj)
            if ~obj.saved_taken
                return;
            end
            saved = obj.saved_grp;
            obj.saved_grp = [];
            obj.saved_taken = false;
            if isempty(saved)
                obj.session.del_elt_grp(obj.ref_elt);
            else
                obj.session.set_elt_grp(obj.ref_elt, saved);
            end
        end

        function do_perturb(obj, increment)
            obj.install_group();
            % UNITS.  prb_grp's signature is BaseUnits for translations
            % and rad for rotations, so the SI-metre translation
            % increment is divided by CBM here -- exactly what
            % macos.perturb does internally for RigidBodyChannel.  The
            % two channel classes therefore speak the SAME language: a
            % group translation column is OPD-per-METRE like a
            % per-element one, and one numeric 'delta' is one physical
            % poke for both.
            %
            % This USED to pass the increment straight through, matching
            % a pymacos prototype (tests/sensitivities/channels.py --
            % test-tree only, never in the shipped package).  The cost
            % was silent and sharp: a scalar delta poked a group 1/CBM
            % times smaller than the elements -- 10 pm against 10 nm on
            % a millimetre deck -- which drove group columns to the
            % finite-difference floor.  Measured on e5hex1, group column
            % / frame-resolved member sum ran 1.0000 (delta 1e-5) ->
            % 1.0005 (1e-6) -> 1.012 (1e-7) -> 1.657 (1e-8): error
            % GROWING as the step shrinks, the signature of a step that
            % is too small.  Gated by
            % tDwDxGroups/test_scalar_delta_matches_the_split_step.
            prb6 = zeros(6, 1);
            prb6(obj.dof_idx + 1) = obj.to_base_units(increment);
            ifGlobal = double(strcmp(obj.coords, 'global'));
            obj.session.prb_grp(obj.ref_elt, prb6, ifGlobal);
            obj.enforce_stop();
            obj.session.modify();

            switch obj.fp_mode
                case 'sxp'
                    obj.session.trace(obj.ref_elt);
                    obj.session.sxp();
                case 'srs'
                    if obj.ep_elt > 0
                        ep = obj.ep_elt;
                    else
                        ep = obj.session.num_elt() - 1;
                    end
                    if obj.fp_elt > 0
                        fp = obj.fp_elt;
                    else
                        fp = obj.session.num_elt();
                    end
                    obj.session.trace(obj.ref_elt);
                    obj.session.srs(ep, fp, 'link', true);
                case 'none'
                    % no-op
            end
        end

        function v = to_base_units(obj, increment)
            % Rotations (dof 0..2) are rad on both sides -- untouched.
            if obj.dof_idx <= 2
                v = increment;
                return
            end
            if obj.cbm_ <= 0
                try
                    obj.cbm_ = obj.session.cbm();
                catch
                    obj.cbm_ = 0;
                end
            end
            if obj.cbm_ <= 0
                error('macos:channels:GroupedRigidBodyChannel:noCBM', ...
                    ['CBM unavailable (Rx not loaded?) -- cannot ' ...
                     'convert an SI-metre translation to BaseUnits']);
            end
            v = increment / obj.cbm_;
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
