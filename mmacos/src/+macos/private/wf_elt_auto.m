function wf_elt = wf_elt_auto(session, exit_pupil_elt)
%WF_ELT_AUTO  Choose the wavefront read surface for the dw_d* sensitivity
%   supervisors.  Shared by dw_dx / dw_dz_zernike / dw_dsurf / dw_dgrid
%   (the _multi drivers inherit it by delegating the per-field read to
%   these single-DOF supervisors).
%
%   The wavefront MUST be read at an EXIT-PUPIL reference -- that is the OPD
%   a PSF, or any diffraction calculation, uses (Dave's ruling 2026-09-08,
%   macos/REPORT_ep_dome_review.md).  The focal plane is NOT an alternative
%   reference: the focal-plane OPD is the optical path to each ray's LANDING
%   POINT, so a displaced perfect image has equal paths and the read is
%   BLIND to tilt -- a segment tilt reads as a segment PISTON, a global
%   field tilt reads ~0.  The old "FP-relation doctrine" is retracted.
%
%   exit_pupil_elt >= 0 is honored verbatim (explicit override).  < 0
%   AUTO-selects:
%     * nElt-1 is UNPOWERED -- a placed pupil (Return/Reference, e.g.
%       Telescope.add_pupil) OR a bench pupil at a real, unpowered
%       beam-train point -- read the wavefront there (nElt-1).
%       is_powered is false for both, so both are allowed, no special case.
%     * nElt-1 is a POWERED optic and no exit-pupil element is placed
%       (the bare-focal imaging deck): ERROR macos:dw_dx:noPupil.  Reading
%       at a powered nElt-1 references the OPD against a FOCUSING optic and
%       renders tilt DOFs as one-signed DOMES; the FocalPlane is tilt-blind
%       and is never a substitute.  The caller must place a pupil (or pass
%       exit_pupil_elt at a collimated-space pupil Reference).
%
%   See also: reset_xp_guard (is_powered), Telescope.add_pupil,
%   dw_dx, dw_dz_zernike, dw_dsurf, dw_dgrid.
    n_elt = session.num_elt();
    if exit_pupil_elt >= 0
        wf_elt = exit_pupil_elt;
        return
    end
    if ~reset_xp_guard('is_powered', session)
        wf_elt = n_elt - 1;
        return
    end
    error('macos:dw_dx:noPupil', ...
        ['nElt-1 (elt %d) is a powered optic and no exit-pupil element ' ...
         'is placed -- there is no valid wavefront reference.  Reading ' ...
         'the OPD here references it against a focusing optic (a ' ...
         'one-signed dome), and the FocalPlane is BLIND to tilt so it ' ...
         'is never a substitute.  Remedies: place an exit pupil with ' ...
         'Telescope.add_pupil (or the Return@image -> Return/Conic ' ...
         'ExitPupil recipe), or pass exit_pupil_elt pointing at a ' ...
         'collimated-space pupil Reference (e.g. a SharedPupil).'], ...
        n_elt - 1);
end
