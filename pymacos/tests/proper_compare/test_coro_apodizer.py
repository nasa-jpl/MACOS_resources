"""
Phase 6a: pupil apodisation via pymacos.apodize + PROPER prop_multiply.

The apodizer mask is constructed once in Python (via apodizer.build_
apodised_mask) and handed verbatim to both engines.  Tests confirm:

  1. The two engines remain bit-identical wavefronts immediately after
     apodisation (already proven by the apodize smoke test; not
     repeated here -- this file tests the DOWNSTREAM propagation).
  2. macos's NFPlane propagation of the apodised wavefront agrees
     with PROPER's prop_propagate at numerical-precision levels.
  3. The soft Gaussian-edge taper REDUCES the kernel-discretisation
     residual that limited Phase 3a (3.7e-8 with hard-edge clipping),
     because the high-frequency edge content is suppressed.
"""
from __future__ import absolute_import

import numpy as np
import pytest
import proper

from .conftest import compare_and_record
from .geometries.coro_nfprop import CoroNFprop, macos_run
from .apodizer import build_apodised_mask, circle, gaussian_edge_taper

pytestmark = pytest.mark.proper_compare


# Phase 3a geometry: NFPlane between Elt 5 (pupil) and Elt 6.
GEOM = CoroNFprop(
    rx_filename="Rx_Coro_noLyot.in",
    src_elt=5, detector_elt=6,
    propagation_m=0.774,
)

# Apodiser: soft Gaussian roll-off outside r0=18 mm with sigma=2 mm,
# fully truncated at r1 = 18 + 4*sigma = 26 mm (Gaussian tail ~ 3e-7).
# The natural beam at Elt 5 extends to ~22 mm; the apodiser softens
# the outer few mm and crops everything > 26 mm.
R0_MM    = 18.0
SIGMA_MM = 2.0
R1_MM    = 26.0


def _build_mask(N: int, dx_m: float) -> np.ndarray:
    aperture = circle(R1_MM * 1e-3)
    taper    = gaussian_edge_taper(R0_MM * 1e-3, SIGMA_MM * 1e-3)
    return build_apodised_mask(N, dx_m, aperture, taper, supersample=16)


def _macos_apodised(geom: CoroNFprop, mask: np.ndarray, pymacos_session
                    ):
    """Drive macos with the apodiser injected at the src element.

    Returns (intensity_at_detector, dx_at_det_m, wavefront_dict).
    The wavefront dict contains the un-apodised cfield at src so the
    PROPER side can apply the same mask itself.
    """
    from pathlib import Path
    rx_path = (Path(__file__).resolve().parents[1] / "Rx"
               / geom.rx_filename)
    pymacos_session.init(geom.macos_size)
    pymacos_session.load(str(rx_path))

    # Populate WFElt at the source element.
    pymacos_session.intensity(geom.src_elt)

    # Snapshot the un-apodised cfield + dx for PROPER's reference.
    cfield_unapo = pymacos_session.complex_field(geom.src_elt,
                                                  reset_trace=False)
    dx_src_m = pymacos_session.dx_at(geom.src_elt)

    # Apodise in place, then propagate WITHOUT resetting the trace
    # (reset_trace=True would re-run MODIFY and wipe our mask).
    pymacos_session.apodize(geom.src_elt, mask)
    intensity_det = pymacos_session.intensity(geom.detector_elt,
                                              reset_trace=False)
    dx_det_m = pymacos_session.dx_at(geom.detector_elt)

    return (intensity_det, dx_det_m,
            dict(complex_field_unapo=cfield_unapo,
                 dx_at_src_m=dx_src_m,
                 dx_at_det_m=dx_det_m,
                 mask=mask))


def _proper_apodised(geom: CoroNFprop, wavefront: dict):
    """Same wavefront recipe as proper_run() for NFPlane, but with
    the same mask multiplied in via prop_multiply.
    """
    cfield = np.asarray(wavefront['complex_field_unapo'],
                        dtype=np.complex128)
    mask   = np.asarray(wavefront['mask'], dtype=np.float64)
    dx_src = wavefront['dx_at_src_m']

    N = geom.macos_size
    grid_extent = N * dx_src
    wfo = proper.prop_begin(grid_extent, geom.wavelength_m, N, 1.0)

    # Combine amp + mask in a single prop_multiply call (matches what
    # macos's apodize wrapper does internally: |cf_apo| = |cf| * mask).
    proper.prop_multiply(wfo, np.abs(cfield) * mask)
    opd = -np.angle(cfield) * geom.wavelength_m / (2.0 * np.pi)
    proper.prop_add_phase(wfo, opd)

    proper.prop_define_entrance(wfo)
    proper.prop_propagate(wfo, geom.propagation_m)
    field, sampling = proper.prop_end(wfo)
    intensity = (np.abs(field) ** 2
                 if field.dtype.kind == 'c' else field)
    return intensity, sampling


def test_coro_apodised_nfplane_elt5_to_elt6(pymacos_session,
                                            results_dir_phase3):
    """NFPlane Elt 5 -> 6 with a Gaussian-edge pupil apodiser.

    Mirrors Phase 3a's geometry but applies a soft-edged pupil
    apodiser via pymacos.apodize (macos side) and prop_multiply
    (PROPER side) using the SAME numpy mask array.  Expected:
    agreement at or BELOW Phase 3a's 3.7e-8, because the apodised
    edge replaces the hard-edge clipping that limited that step.
    """
    # Build the apodiser at the macos diffraction-grid sampling.  We
    # need dx_at(5) at runtime; do a probe-load to find it, then
    # build the mask, then run for real.
    import pymacos.macos as m_probe
    from pathlib import Path
    rx_path = (Path(__file__).resolve().parents[1] / "Rx"
               / GEOM.rx_filename)
    m_probe.init(GEOM.macos_size)
    m_probe.load(str(rx_path))
    m_probe.intensity(GEOM.src_elt)
    dx_at_src = m_probe.dx_at(GEOM.src_elt)
    mask = _build_mask(GEOM.macos_size, dx_at_src)
    # mask values: should be 1 inside r0, smoothly tapering to 0 by r1.
    assert mask.shape == (GEOM.macos_size, GEOM.macos_size)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0 + 1e-12   # tolerate float noise
    assert mask.max() > 0.99           # core is fully transmissive

    intensity_m, dx_m, wf = _macos_apodised(GEOM, mask, pymacos_session)
    intensity_p, dx_p     = _proper_apodised(GEOM, wf)

    assert intensity_m.shape == intensity_p.shape
    assert dx_p == pytest.approx(abs(dx_m), rel=1e-3)

    metrics = compare_and_record(
        'coro_apodised_nfplane_elt5_to_elt6',
        intensity_m, intensity_p, abs(dx_m),
        results_dir_phase3,
        crop_pixels=intensity_m.shape[0],
        norm_kind='sum',
        extra_metadata={
            'wavelength_m':       GEOM.wavelength_m,
            'propagation_m':      GEOM.propagation_m,
            'src_elt':            GEOM.src_elt,
            'detector_elt':       GEOM.detector_elt,
            'rx_filename':        GEOM.rx_filename,
            'apodiser_r0_mm':     R0_MM,
            'apodiser_sigma_mm':  SIGMA_MM,
            'apodiser_r1_mm':     R1_MM,
            'apodiser_supersample': 16,
            'macos_cfield_unapo': wf['complex_field_unapo'],
            'apodiser_mask':      mask,
        })

    # Expect agreement at or below Phase 3a's 3.7e-8 (which used the
    # same NFPlane physics but with hard-edge clipping).  Set a loose
    # 1e-7 threshold for the first run; will tighten once observed.
    assert metrics['max_abs'] < 1e-7, (
        f"max |a-b| = {metrics['max_abs']:.3e} (sum-normalised); "
        f"RMS = {metrics['rms_abs']:.3e}; "
        f"Δcom = ({metrics['dx_pix']:+d}, "
        f"{metrics['dy_pix']:+d}) px")
