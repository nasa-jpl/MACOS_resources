"""
Phase 2: macos vs PROPER comparison on Rx_Coro's first NF
plane-to-plane propagation step (Elt 2 -> Elt 3, 774 mm).

Strategy: feed both engines the SAME wavefront at Elt 2 (macos's
amplitude + OPD), then compare PROPER's propagated intensity to
macos's INT at Elt 3.  This isolates the NF plane-to-plane
propagator from any preceding source / aperture / OAP modelling.

If both engines implement the same Fresnel kernel correctly,
agreement should land at numerical-precision level analogous to
Phase 1.  Disagreement at any larger scale is a real engine
difference in the NF propagator.
"""
from __future__ import absolute_import

import numpy as np
import pytest

from .conftest import compare_and_record
from .geometries.coro_nfprop import DEFAULT, macos_run, proper_run

pytestmark = pytest.mark.proper_compare


def test_coro_nfprop_macos_runs(pymacos_session):
    """Sanity: macos returns sensible intensity at Elt 3 and a
    wavefront at Elt 2.
    """
    intensity_at_3, dx, wf2 = macos_run(DEFAULT, pymacos_session)

    assert intensity_at_3.shape == (DEFAULT.macos_size, DEFAULT.macos_size)
    assert np.all(np.isfinite(intensity_at_3))
    assert intensity_at_3.max() > 0

    assert wf2['amplitude'].shape == intensity_at_3.shape
    assert np.all(np.isfinite(wf2['amplitude']))
    # OPD is captured for the .mat archive but is on macos's source
    # ray grid (smaller than the diffraction grid).  Not used by
    # PROPER in v1; just check it exists and is finite.
    if wf2.get('opd') is not None:
        assert np.all(np.isfinite(wf2['opd']))

    # macos's interactive diagnostic on this prescription:
    #   Peak intensity= 6.305460e-06; Sum of intensity= 0.1962738
    assert intensity_at_3.max() == pytest.approx(6.305e-6, rel=1e-3)
    assert intensity_at_3.sum() == pytest.approx(0.1963,    rel=1e-3)


def test_coro_nfprop_proper_runs(pymacos_session):
    """Sanity: PROPER ingests macos's Elt-2 wavefront and produces
    a finite, non-degenerate intensity at the Elt-3 plane.
    """
    _, _, wf2 = macos_run(DEFAULT, pymacos_session)
    intensity_p, dxp = proper_run(DEFAULT, wavefront_at_elt2=wf2)

    assert intensity_p.shape == (DEFAULT.macos_size, DEFAULT.macos_size)
    assert np.all(np.isfinite(intensity_p))
    assert intensity_p.max() > 0
    assert dxp == pytest.approx(DEFAULT.dx_m, rel=1e-3)


def test_coro_nfprop_compare(pymacos_session, results_dir_phase2):
    """Pixel-level comparison of PROPER vs macos at Elt 3 with both
    starting from the same Elt 2 wavefront.

    First-pass tolerance is loose (0.1 on Strehl-normalised max
    |a-b|) because the NF propagator has more subtle implementation
    differences than the FF case (boundary handling, kernel
    discretisation, sign conventions on the quadratic phase).  The
    report will show whether we land near 1e-11 like Phase 1 or
    somewhere broader; tighten the bound once the actual gap is
    characterised.
    """
    intensity_m, dx_m, wf2 = macos_run(DEFAULT, pymacos_session)
    intensity_p, dx_p      = proper_run(DEFAULT, wavefront_at_elt2=wf2)

    assert intensity_p.shape == intensity_m.shape
    assert dx_p == pytest.approx(dx_m, rel=1e-3)

    metrics = compare_and_record(
        'coro_nfprop_elt2_to_elt3',
        intensity_m, intensity_p, dx_m,
        results_dir_phase2,
        crop_pixels=intensity_m.shape[0],   # full grid; PSF here is
                                            # off-axis and the default
                                            # central crop would hide it
        extra_metadata={
            'wavelength_m':       DEFAULT.wavelength_m,
            'propagation_m':      DEFAULT.propagation_m,
            'rx_filename':        DEFAULT.rx_filename,
            'macos_opd_at_elt2':           wf2['opd'],
            'macos_amp_at_elt2':           wf2['amplitude'],
            'macos_complex_field_at_elt2': wf2['complex_field'],
        })

    # Use raw max_abs (NOT max_abs_aligned).  This Coro PSF is a
    # flat-top pillar, not a sharp peak; the centroid-alignment
    # metric that worked for Cass FF actively mis-aligns flat-tops
    # because the "peak" position within a flat region is noise-
    # dominated.  Raw max_abs over Strehl-normalised arrays is the
    # right measure for extended-source intensity distributions.
    assert metrics['max_abs'] < 1e-4, (
        f"max |a-b| = {metrics['max_abs']:.3e} (Strehl-normalised); "
        f"peak offset = ({metrics['dx_pix']:+d}, "
        f"{metrics['dy_pix']:+d}) px -- but this is a flat-top PSF "
        f"so peak-position is not a meaningful diagnostic")
