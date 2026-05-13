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


def _embed_opd_in_proper_grid(macos_opd, geom: 'CassFarField',
                              allow_resample: bool = False):
    """Place macos's OPD array into a PROPER-sized phase array.

    Default mode (no resampling): macos's source-grid pixel pitch
    equals PROPER's entrance-pupil pixel pitch by construction --
    proper_grid_n * beam_ratio = macos source grid size.  We
    zero-pad macos OPD into the centre of a (proper_grid_n,
    proper_grid_n) array.  The aperture stop in PROPER masks the
    surrounding zeros, so they're physically harmless.

    Fallback (allow_resample=True): if the inner dim doesn't match,
    use scipy.ndimage.zoom for a bilinear resample.  This is OFF by
    default because resampling introduces interpolation error that
    can mask real engine disagreement -- prefer to choose grid
    parameters that match exactly.

    Raises ValueError on shape mismatch when allow_resample=False.
    """
    import numpy as np
    macos_n = macos_opd.shape[0]
    inner_n = int(round(geom.proper_grid_n * geom.proper_beam_ratio))

    if macos_n == inner_n:
        # zero-pad centre into PROPER's full grid
        n = geom.proper_grid_n
        padded = np.zeros((n, n), dtype=float)
        off = (n - macos_n) // 2
        padded[off:off + macos_n, off:off + macos_n] = macos_opd
        return padded

    if not allow_resample:
        raise ValueError(
            f"macos OPD size {macos_n} does not match the expected "
            f"PROPER inner-pupil size {inner_n} "
            f"(= proper_grid_n {geom.proper_grid_n} * beam_ratio "
            f"{geom.proper_beam_ratio}).  Either fix the geometry so "
            f"the sampling matches, or call proper_run with "
            f"allow_resample=True (interpolation error will be folded "
            f"into the comparison).")

    from scipy.ndimage import zoom
    factor = geom.proper_grid_n / macos_n
    return zoom(macos_opd, factor, order=1, mode='constant', cval=0.0)


def proper_run(geom: CassFarField = DEFAULT,
               include_obscurations: bool = True,
               macos_opd=None,
               allow_resample: bool = False):
    """Drive PROPER for the Cass-FF geometry.

    Returns (intensity, sampling_m_per_pixel).

    Args:
      include_obscurations: place secondary + cross spider at the
        entrance aperture when True (default).
      macos_opd: optional 2D OPD map (in metres) from pymacos.opd() at
        the exit pupil.  If provided, it is placed at PROPER's
        entrance pupil via prop_add_phase so PROPER carries the same
        aberration content macos traced through its mirrors.  Default
        sampling assumption: macos OPD pixel pitch already matches
        PROPER's entrance-pupil pitch; only zero-padding is applied.
      allow_resample: if True and the OPD shape doesn't match
        PROPER's inner pupil size exactly, bilinearly resample.  Off
        by default to avoid silently introducing interpolation error.
    """
    import proper

    n = geom.proper_grid_n
    beam_ratio = geom.proper_beam_ratio
    wfo = proper.prop_begin(geom.pupil_diameter_m,
                            geom.wavelength_m, n, beam_ratio)

    proper.prop_circular_aperture(wfo, geom.pupil_diameter_m / 2.0)

    if include_obscurations:
        proper.prop_circular_obscuration(wfo, geom.sec_obs_radius_m)
        # cross-spider: two orthogonal rectangular obscurations
        proper.prop_rectangular_obscuration(
            wfo,
            geom.spider_half_width * 2.0,    # x-width
            geom.spider_half_length * 2.0,   # y-height
        )
        proper.prop_rectangular_obscuration(
            wfo,
            geom.spider_half_length * 2.0,
            geom.spider_half_width * 2.0,
        )

    if macos_opd is not None:
        phase = _embed_opd_in_proper_grid(macos_opd, geom,
                                          allow_resample=allow_resample)
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
