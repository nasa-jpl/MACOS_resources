"""
macos-vs-PROPER comparison on Rx_Cass_FarField.in.

First-pass comparison: focal-plane PSF magnitude/structure.  The two
engines model the same physical problem with different propagation
machinery (macos: ray-traced Cass + scalar FF DFT; PROPER: thin-lens
Fresnel propagation).  They should agree on:

  - PSF peak location at the array center (within 1 pixel)
  - Total intensity (after Strehl normalization)
  - Approximate Airy/diffracted-pattern shape

They are NOT expected to agree to floating-point precision; the
tolerances here are loose by design and should be tightened as the
match is understood and the sampling reconciliation is improved.
"""
from __future__ import absolute_import

import numpy as np
import pytest

from .conftest import compare_and_record
from .geometries.cass_farfield import DEFAULT, proper_run, macos_run

pytestmark = pytest.mark.proper_compare


def test_proper_cass_ff_runs():
    """PROPER side of the Cass-FF problem produces a non-degenerate
    obstructed-aperture PSF at the focal plane.
    """
    intensity, sampling = proper_run(DEFAULT)

    assert intensity.ndim == 2
    n = DEFAULT.proper_grid_n
    assert intensity.shape == (n, n)
    assert np.all(np.isfinite(intensity))
    assert intensity.max() > 0
    peak = np.unravel_index(np.argmax(intensity), intensity.shape)
    center = (n // 2, n // 2)
    assert abs(peak[0] - center[0]) <= 1
    assert abs(peak[1] - center[1]) <= 1
    # Focal-plane sampling must match macos's, by construction.
    assert sampling == pytest.approx(DEFAULT.dx_focal_m, rel=1e-3)


def test_macos_cass_ff_runs(pymacos_session):
    """macos side of the Cass-FF problem produces a 512x512 intensity
    array with the peak at array center.  (At model_size=512, macos
    produces the 2x-oversampled focal-plane grid: dx=2.78e-6 m.)
    """
    intensity, dx = macos_run(DEFAULT, pymacos_session)

    assert intensity.ndim == 2
    n = DEFAULT.macos_size
    assert intensity.shape == (n, n)
    assert np.all(np.isfinite(intensity))
    peak = np.unravel_index(np.argmax(intensity), intensity.shape)
    center = (n // 2, n // 2)
    assert abs(peak[0] - center[0]) <= 1
    assert abs(peak[1] - center[1]) <= 1
    # macos diagnostic at model_size=512 on this prescription:
    #   Peak intensity= 3.2361861D+05; Sum of intensity= 1.8017729D+06
    assert intensity.max() == pytest.approx(3.236e5, rel=1e-3)
    assert intensity.sum() == pytest.approx(1.802e6, rel=1e-3)


def test_compare_cass_ff_psf(pymacos_session, tol, results_dir):
    """Pixel-by-pixel focal-plane PSF comparison, nominal Cass-FF.

    Same prescription for both engines.  No OPD pass-through (PROPER
    uses its ideal thin-lens model; macos uses ray-traced Cass).
    """
    proper_int, dx_p = proper_run(DEFAULT)
    macos_int,  dx_m = macos_run(DEFAULT, pymacos_session)

    assert dx_p == pytest.approx(dx_m, rel=1e-3)
    assert proper_int.shape == macos_int.shape

    metrics = compare_and_record(
        'cass_ff_nominal',
        macos_int, proper_int, dx_m,
        results_dir,
        extra_metadata={
            'wavelength_m':     DEFAULT.wavelength_m,
            'pupil_diameter_m': DEFAULT.pupil_diameter_m,
        })
    assert metrics['max_abs_aligned'] < 0.1, (
        f"max |a-b| = {metrics['max_abs_aligned']:.3e} aligned")


def test_compare_cass_ff_psf_with_opd(pymacos_session, tol, results_dir):
    """As above, but PROPER's wavefront carries macos's OPD at the
    exit pupil.  In the nominal (un-perturbed) case this is ~1 pm RMS
    and shouldn't change the focal-plane PSF much; the test confirms
    the OPD plumbing is silent in the no-aberration limit.
    """
    macos_int, dx_m, opd = macos_run(DEFAULT, pymacos_session,
                                      return_opd=True)
    proper_int, dx_p = proper_run(DEFAULT, macos_opd=opd)

    assert dx_p == pytest.approx(dx_m, rel=1e-3)

    metrics = compare_and_record(
        'cass_ff_nominal_with_opd',
        macos_int, proper_int, dx_m,
        results_dir,
        extra_metadata={
            'wavelength_m':     DEFAULT.wavelength_m,
            'pupil_diameter_m': DEFAULT.pupil_diameter_m,
            'macos_opd_at_xp':  opd,
        })
    # With OPD pass-through, residual disagreement drops because the
    # remaining mismatch is engine architecture rather than wavefront
    # bookkeeping.  Set the bar tighter than the no-OPD test.
    assert metrics['max_abs_aligned'] < 0.02, (
        f"max |a-b| = {metrics['max_abs_aligned']:.3e} aligned")
