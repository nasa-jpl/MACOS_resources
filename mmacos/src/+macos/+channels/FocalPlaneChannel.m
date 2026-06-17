classdef FocalPlaneChannel < macos.channels.RigidBodyChannel
%MACOS.CHANNELS.FOCALPLANECHANNEL  Rigid-body perturbation on the FP element.
%
%   macos's standard trace does NOT propagate FP perturbations back
%   to the EP-OPD measurement -- without compensation all six FP DOFs
%   come out as zero sensitivities (physically wrong: the EP is a
%   sphere centered on the FP, so moving the FP must change the EP
%   and thus the wavefront referenced to it).
%
%   fp_mode dispatch:
%     'track' (default) -- perturb FP + drag EP along (DOF-aware:
%       Tx/Ty propagate the same vector; Tz doesn't move EP vpt
%       (SXP refines radius); rotations rotate EP rigidly about
%       FP's RptElt via hand-computed vpt/psi/rpt updates).  After
%       perturbing, trace + SXP + restore the EP vpt/psi/rpt
%       (SXP refines the EP radius without disturbing the
%       track-induced EP pose).  Matches the physics "EP is a
%       sphere centered on the FP".  Doesn't need a Stop set.
%     'srs'             -- perturb FP, then srs(EP, FP) so macos
%       recomputes the EP pose from chief-ray geometry.  Needs a
%       Stop set.
%     'sxp'             -- perturb FP, then sxp() recomputes the EP
%       at nElt-1 with radius = chief-ray distance to FP.  Needs a
%       Stop set.
%     'none'            -- just perturb FP, no EP follow-up
%       (sensitivity is zero by design).

    properties (SetAccess = private)
        mode    (1,:) char
        ep_elt  (1,1) double
    end

    methods
        function obj = FocalPlaneChannel(session, iElt, dof_idx, opts)
            arguments
                session
                iElt    (1,1) double {mustBeInteger, mustBePositive}
                dof_idx (1,1) double {mustBeInteger, ...
                            mustBeGreaterThanOrEqual(dof_idx, 0), ...
                            mustBeLessThanOrEqual(dof_idx, 5)}
                opts.mode   (1,:) char {mustBeMember(opts.mode, ...
                                {'track','srs','sxp','none'})} = 'track'
                opts.ep_elt (1,1) double {mustBeInteger} = -1
            end
            obj@macos.channels.RigidBodyChannel(session, iElt, dof_idx);
            obj.mode    = opts.mode;
            obj.ep_elt  = opts.ep_elt;
        end
    end

    methods (Access = protected)
        function do_perturb(obj, increment)
            % Build the 6-vector for this DOF.
            rot = [0; 0; 0];  trans = [0; 0; 0];
            if obj.dof_idx < 3
                rot(obj.dof_idx + 1) = increment;
            else
                trans(obj.dof_idx - 2) = increment;
            end

            % Always perturb the FP.
            obj.session.perturb(obj.iElt, ...
                'rotation', rot, ...
                'translation', trans, ...
                'frame', 'local');

            if strcmp(obj.mode, 'track')
                ep = obj.resolve_ep();
                if obj.dof_idx == 3 || obj.dof_idx == 4
                    % Lateral translation: EP follows by the same vector.
                    obj.session.perturb(ep, ...
                        'rotation', rot, ...
                        'translation', trans, ...
                        'frame', 'local');
                elseif obj.dof_idx == 5
                    % Axial translation: EP vpt unchanged; SXP refines
                    % radius below.
                else
                    % Rotation DOF: rigid-rotate EP about FP's RptElt.
                    obj.rotate_ep_about_fp_rpt(ep, increment);
                end
            end

            obj.session.modify();

            % Post-perturb EP refinement.
            switch obj.mode
                case 'track'
                    ep = obj.resolve_ep();
                    vpt_save = obj.session.get_elt_vpt(ep);
                    psi_save = obj.session.get_elt_psi(ep);
                    rpt_save = obj.session.get_elt_rpt(ep);
                    obj.session.trace(obj.iElt);
                    obj.session.sxp();
                    obj.session.set_elt_vpt(ep, vpt_save);
                    obj.session.set_elt_psi(ep, psi_save);
                    obj.session.set_elt_rpt(ep, rpt_save);
                    obj.session.modify();
                case 'sxp'
                    obj.session.trace(obj.iElt);
                    obj.session.sxp();
                case 'srs'
                    ep = obj.resolve_ep();
                    obj.session.trace(obj.iElt);
                    obj.session.srs(ep, obj.iElt);
                case 'none'
                    % no-op
            end
        end
    end

    methods (Access = private)
        function ep = resolve_ep(obj)
            if obj.ep_elt > 0
                ep = obj.ep_elt;
            else
                ep = obj.session.num_elt() - 1;
            end
        end

        function rotate_ep_about_fp_rpt(obj, ep, increment)
            % Local rotation vector in FP's frame.
            theta_local = zeros(3, 1);
            theta_local(obj.dof_idx + 1) = increment;

            % FP TElt's upper-left 3x3 = local->global rotation matrix.
            s = obj.session.get_elt_csys(obj.iElt);
            if ndims(s.csys) == 3
                R = s.csys(1:3, 1:3, 1);
            else
                R = s.csys(1:3, 1:3);
            end
            theta_global = R * theta_local;

            fp_rpt = obj.session.get_elt_rpt(obj.iElt);
            ep_vpt = obj.session.get_elt_vpt(ep);
            ep_psi = obj.session.get_elt_psi(ep);
            ep_rpt = obj.session.get_elt_rpt(ep);
            fp_rpt = fp_rpt(:);
            ep_vpt = ep_vpt(:);
            ep_psi = ep_psi(:);
            ep_rpt = ep_rpt(:);

            % Small-angle rigid rotation about FP RptElt:
            %   v_new = v + theta_global x (v - FP_RptElt)
            new_ep_vpt = ep_vpt + cross(theta_global, ep_vpt - fp_rpt);
            new_ep_rpt = ep_rpt + cross(theta_global, ep_rpt - fp_rpt);
            new_ep_psi = ep_psi + cross(theta_global, ep_psi);
            % Use sqrt(sum(v.^2)) instead of norm(v) so the
            % normalization matches NumPy's np.linalg.norm exactly.
            % MATLAB's norm() uses a more sophisticated
            % (dnrm2-style) algorithm that differs by 1 ULP at this
            % scale -- the residual then propagates through the
            % trace and shows up as a ~7e-7 m FP-channel discrepancy
            % vs pymacos.
            n = sqrt(sum(new_ep_psi.^2));
            if n > 0
                new_ep_psi = new_ep_psi / n;
            end

            obj.session.set_elt_vpt(ep, new_ep_vpt(:));
            obj.session.set_elt_psi(ep, new_ep_psi(:));
            obj.session.set_elt_rpt(ep, new_ep_rpt(:));
        end
    end
end
