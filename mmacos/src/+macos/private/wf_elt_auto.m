function wf_elt = wf_elt_auto(session, exit_pupil_elt)
%WF_ELT_AUTO  Choose the wavefront read surface for the dw_d* sensitivity
%   supervisors.  Shared by dw_dx / dw_dz_zernike / dw_dsurf / dw_dgrid
%   (the _multi drivers inherit it by delegating the per-field read to
%   these single-DOF supervisors; the _multi path ALSO preflights this in
%   dw_multi_core before the field loop, raising macos:<front>_multi:noPupil).
%
%   The sensitivity OPD is read at an EXIT-PUPIL reference -- that is the OPD
%   a PSF, or any diffraction calculation, uses (Dave's ruling 2026-09-08,
%   macos/REPORT_ep_dome_review.md).  The focal plane is NOT an alternative
%   reference: the FP OPD is the optical path to each ray's LANDING POINT, so
%   a displaced perfect image has equal paths and the read is BLIND to tilt.
%
%   The read criterion is the ELEMENT TYPE, not "is it powered":
%   `is_powered` answers "would a RESET WRITE clobber a real optic" and
%   returns false for a FLAT fold at nElt-1, which is not a pupil either --
%   its OPD in converging space is tilt-blind, the same class as the FP
%   (CCL 2026-09-08).  So:
%     exit_pupil_elt >= 0                    -> honored verbatim.
%     nElt-1 is a Return/Reference (elt_id 3/8, a placed pupil):
%        curved  -> read at nElt-1 (the exit-pupil sphere);
%        FLAT    -> read at nElt-1 but WARN macos:dw_dx:flatPupil -- valid
%                   ONLY in collimated space (a SharedPupil); in converging
%                   space a flat reference's OPD is tilt-blind.
%     otherwise (a powered/flat Reflector, no placed pupil)
%                -> ERROR macos:dw_dx:noPupil.  Reading here is the
%                   one-signed dome; the FP is never a substitute.
%
%   See also: reset_xp_guard (is_powered -- for the reset WRITE, not this
%   read), Telescope.add_pupil, dw_dx, dw_dz_zernike, dw_dsurf, dw_dgrid.
    n_elt = session.num_elt();
    if exit_pupil_elt >= 0
        wf_elt = exit_pupil_elt;
        return
    end
    ep   = n_elt - 1;
    info = macos.get_elt_info(ep);
    if ~any(info.elt_id == [3, 8])          % not a Reference/Return -> not a pupil
        error('macos:dw_dx:noPupil', ...
            ['nElt-1 (elt %d) is a %s, not an exit-pupil element -- there ' ...
             'is no valid wavefront reference.  Reading the OPD here ' ...
             'references it against a converging/focusing surface (a ' ...
             'one-signed dome), and the FocalPlane is BLIND to tilt so it ' ...
             'is never a substitute.  Remedies: place an exit pupil with ' ...
             'Telescope.add_pupil (or the Return@image -> Return/Conic ' ...
             'ExitPupil recipe), or pass exit_pupil_elt pointing at a ' ...
             'collimated-space pupil Reference (e.g. a SharedPupil).'], ...
            ep, info.type);
    end
    if abs(session.get_elt_kr(ep)) >= 1e22  % a FLAT Return/Reference
        warning('macos:dw_dx:flatPupil', ...
            ['nElt-1 (elt %d) is a FLAT %s -- a valid exit-pupil read ONLY ' ...
             'in collimated space (a SharedPupil).  In converging space a ' ...
             'flat reference''s OPD is the path to a plane and is tilt-blind ' ...
             'like the FocalPlane.  Pass exit_pupil_elt if that is not what ' ...
             'you intend.'], ep, info.type);
    end
    wf_elt = ep;
end
