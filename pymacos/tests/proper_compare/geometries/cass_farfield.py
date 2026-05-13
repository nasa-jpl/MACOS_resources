"""
Cass FarField comparison geometry.

Single source of truth for the macos and PROPER renderings of the
Cassegrain-far-field problem in Rx_Cass_FarField.in.  Both engines
should agree (to within stated tolerances) on the focal-plane
intensity pattern when running this geometry.

The macos prescription (Rx_Cass_FarField.in) is the authoritative
description; the PROPER side is constructed to mirror its key
physical parameters (aperture diameter, secondary obscuration,
spider, equivalent focal length, wavelength).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CassFarField:
    rx_filename:       str   = "Rx_Cass_FarField.in"
    macos_detector_srf: int  = 6        # FocalPlane element in the Rx
    macos_size:        int   = 512   # 2x oversampling vs prescription nGridpts

    # Physical parameters extracted from Rx_Cass_FarField.in
    wavelength_m:      float = 1.0e-6   # 1 micron
    pupil_diameter_m:  float = 4.0      # primary aperture (Elt 0 Aperture=4.0)
    sec_obs_radius_m:  float = 0.5      # secondary central obscuration ObsVec
    spider_half_width: float = 3.125e-3 # half-width of each cross-spider strut
    spider_half_length: float = 3.0     # half-length of each strut

    # Macos-reported exit-pupil / focal-plane geometry (FF prop output
    # at model_size=512; doubles the focal sampling vs. prescription
    # nGridpts).
    z_pupil_to_focal_m: float = 5.5601    # exit-pupil to focal-plane distance
    dx_exit_pupil_m:    float = 3.9032e-3 # sampling at exit pupil
    dx_focal_m:         float = 2.7823e-6 # sampling at focal plane (at size=512)

    # PROPER grid
    #   N=512 with beam_ratio=0.5 matches macos at model_size=512:
    #     - dx_pupil_entrance = 4 / 0.5 / 512 = 0.015625 m
    #     - dx_focal = lambda * f_eff / (N * dx_pupil) = 2.78e-6 m
    #     - grid extent at focal plane = 512 * 2.78e-6 = 1.42 mm
    #   Both arrays are 512 x 512 at the same pixel pitch -> direct
    #   pixel-by-pixel comparison is meaningful.
    proper_grid_n:     int   = 512
    proper_beam_ratio: float = 0.5

    @property
    def effective_focal_length_m(self) -> float:
        """Equivalent thin-lens focal length such that PROPER produces
        the same focal-plane pixel pitch as macos.

        Derivation:  PROPER's dx_focal = lambda * f_eff / (N * dx_pupil),
        with dx_pupil = D / (N * beam_ratio).  Substituting:
            dx_focal = lambda * f_eff * beam_ratio / D
            f_eff    = dx_focal * D / (lambda * beam_ratio)

        For lambda=1um, D=4m, beam_ratio=0.5, dx_focal=2.78e-6:
            f_eff = 22.24 m
        """
        return (self.dx_focal_m * self.pupil_diameter_m
                / (self.wavelength_m * self.proper_beam_ratio))


DEFAULT = CassFarField()


def _embed_macos_array_in_proper_grid(macos_arr, geom: 'CassFarField',
                                      allow_resample: bool = False,
                                      fill_value: float = 0.0):
    """Place a macos source-grid array (OPD or amplitude mask) into a
    PROPER-sized array.

    Sampling: macos's source-grid pixel pitch equals PROPER's
    entrance-pupil pitch by construction (proper_grid_n * beam_ratio
    = macos source grid size).  No resampling needed; centre-pad
    only.

    allow_resample=True falls back to bilinear scipy.zoom on a shape
    mismatch.  OFF by default to avoid masking real engine
    disagreements with interpolation error.
    """
    import numpy as np
    macos_n = macos_arr.shape[0]
    inner_n = int(round(geom.proper_grid_n * geom.proper_beam_ratio))

    if macos_n == inner_n:
        n = geom.proper_grid_n
        padded = np.full((n, n), fill_value, dtype=float)
        off = (n - macos_n) // 2
        padded[off:off + macos_n, off:off + macos_n] = macos_arr
        return padded

    if not allow_resample:
        raise ValueError(
            f"macos array size {macos_n} does not match the expected "
            f"PROPER inner-pupil size {inner_n} "
            f"(= proper_grid_n {geom.proper_grid_n} * beam_ratio "
            f"{geom.proper_beam_ratio}).")

    from scipy.ndimage import zoom
    factor = geom.proper_grid_n / macos_n
    return zoom(macos_arr, factor, order=1, mode='constant',
                cval=fill_value)


# Backwards-compatible alias (older call sites or external scripts)
_embed_opd_in_proper_grid = _embed_macos_array_in_proper_grid


def proper_run(geom: CassFarField = DEFAULT,
               include_obscurations: bool = True,
               macos_opd=None,
               macos_amplitude=None,
               allow_resample: bool = False,
               opd_sign_flip: bool = True):
    """Drive PROPER for the Cass-FF geometry.

    Returns (intensity, sampling_m_per_pixel).

    Aperture model
    --------------
    When macos_opd is provided (or macos_amplitude is provided
    explicitly), PROPER's amplitude pattern is taken DIRECTLY from
    macos's mask -- the analytical prop_circular_aperture /
    prop_circular_obscuration / prop_rectangular_obscuration calls
    are SKIPPED.  This guarantees PROPER's illuminated pixels are
    exactly the pixels macos sees as carrying light, which is the
    only physically-defensible choice: putting amplitude where macos
    says there's none introduces a phase-mismatch artefact that
    (in the Tx +1um diagnostic case) halved the apparent PSF shift.

    When no macos input is given the analytical aperture+spider
    model is used (so the "PROPER only" comparison still works).

    Args
    ----
    include_obscurations: only consulted in the analytical-aperture
        path (when macos input is not provided).
    macos_opd: optional 2D OPD map (in metres) from pymacos.opd().
        Placed at PROPER's entrance pupil via prop_add_phase.
    macos_amplitude: optional 2D amplitude mask.  Defaults to
        (macos_opd != 0) when macos_opd is provided.  Override to
        supply a non-binary amplitude (e.g. apodisation, real macos
        amplitude output once that wrapper exists).
    opd_sign_flip: if True (default) multiplies macos OPD by -1
        before adding to PROPER's phase.  Empirically determined:
        macos OPD sign convention is opposite to PROPER's
        prop_add_phase input -- without the flip the focal-plane
        PSF shifts in the wrong direction.
    allow_resample: bilinear-resample if macos array shape doesn't
        match PROPER's inner-pupil size; OFF by default.
    """
    import numpy as np
    import proper

    n = geom.proper_grid_n
    beam_ratio = geom.proper_beam_ratio
    wfo = proper.prop_begin(geom.pupil_diameter_m,
                            geom.wavelength_m, n, beam_ratio)

    use_macos_mask = (macos_opd is not None) or (macos_amplitude is not None)

    if use_macos_mask:
        # Apply macos's amplitude mask directly.  Default: binary mask
        # from |OPD| > 0.
        if macos_amplitude is None:
            macos_amplitude = (np.asarray(macos_opd) != 0).astype(float)
        amp_padded = _embed_macos_array_in_proper_grid(
            macos_amplitude, geom, allow_resample=allow_resample,
            fill_value=0.0)
        proper.prop_multiply(wfo, amp_padded)
    else:
        proper.prop_circular_aperture(wfo, geom.pupil_diameter_m / 2.0)
        if include_obscurations:
            proper.prop_circular_obscuration(wfo, geom.sec_obs_radius_m)
            proper.prop_rectangular_obscuration(
                wfo,
                geom.spider_half_width * 2.0,
                geom.spider_half_length * 2.0,
            )
            proper.prop_rectangular_obscuration(
                wfo,
                geom.spider_half_length * 2.0,
                geom.spider_half_width * 2.0,
            )

    if macos_opd is not None:
        opd_arr = np.asarray(macos_opd, dtype=float)
        if opd_sign_flip:
            opd_arr = -opd_arr
        phase = _embed_macos_array_in_proper_grid(
            opd_arr, geom, allow_resample=allow_resample,
            fill_value=0.0)
        proper.prop_add_phase(wfo, phase)

    proper.prop_define_entrance(wfo)
    proper.prop_lens(wfo, geom.effective_focal_length_m)
    proper.prop_propagate(wfo, geom.effective_focal_length_m)
    psf_field, sampling = proper.prop_end(wfo)

    intensity = (abs(psf_field) ** 2 if psf_field.dtype.kind == "c"
                 else psf_field)
    return intensity, sampling


def macos_run(geom: CassFarField = DEFAULT, pymacos_session=None,
              return_opd: bool = False, exit_pupil_srf: int = 5,
              perturbation=None):
    """Drive macos via pymacos for the same geometry.

    Args:
      pymacos_session: imported pymacos module (after init()).
      return_opd: if True also returns the OPD map at the exit pupil.
      exit_pupil_srf: which prescription element is the exit pupil
        (defaults to 5 = ExitPupil in Rx_Cass_FarField.in).
      perturbation: optional (iElt, [Rx,Ry,Rz,Tx,Ty,Tz]) tuple to
        apply via prb_elt before tracing.  Frame is local element
        (ifGlobal=0).

    Returns:
      (intensity, dx_m)              if return_opd is False
      (intensity, dx_m, opd_at_xp)   if return_opd is True
    """
    if pymacos_session is None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve()
                                .parents[3] / "src"))
        import pymacos.macos as pymacos_session

    import numpy as np
    from pathlib import Path
    rx_path = (Path(__file__).resolve().parents[2]
               / "Rx" / geom.rx_filename)

    pymacos_session.load(str(rx_path))

    if perturbation is not None:
        iElt, prb = perturbation
        prb_arr = np.asarray(prb, dtype=float).reshape(6, 1)
        # Use global coordinate frame: the prescription elements have
        # nECoord=-6 which does not provide a local frame; ifGlobal=True
        # interprets [Rx,Ry,Rz,Tx,Ty,Tz] in world coordinates.
        pymacos_session.prb_elt([int(iElt)], prb_arr, [True])

    opd_map = None
    if return_opd:
        pymacos_session.trace_rays(exit_pupil_srf)
        opd_map = pymacos_session.opd().copy()

    intensity = pymacos_session.intensity(geom.macos_detector_srf)

    if return_opd:
        return intensity, geom.dx_focal_m, opd_map
    return intensity, geom.dx_focal_m
