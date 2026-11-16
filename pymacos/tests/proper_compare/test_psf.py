"""
PSF comparison: macos INT/PIX vs PROPER prop_end.

Starting point — diffraction-limited circular pupil → on-axis focus.
The Airy pattern is the textbook reference; both engines should
produce numerically-close intensity arrays at the focal plane.
"""
from __future__ import absolute_import

import numpy as np
import pytest

from .geometries.circular_pupil_focus import DEFAULT, proper_run

# Step 1, in this commit: PROPER side runs end-to-end. The macos side
# requires a representative .in prescription that pymacos can `load()`
# and trace through to an INT/PIX detector. Filling that in is the
# next implementation step; for now we mark the comparison part
# xfail-style and assert only on the PROPER side so the harness
# can be validated piecewise.

pytestmark = pytest.mark.proper_compare


def test_proper_circular_psf_runs():
    """Sanity check: PROPER produces a non-degenerate Airy PSF.

    Verifies the PROPER side of the harness is wired correctly
    before we hook in the macos side. The macos comparison test
    is sketched in ``test_compare_circular_psf`` below and skipped
    until ``macos_rx_text()`` is implemented.
    """
    intensity, sampling = proper_run(DEFAULT)

    assert intensity.ndim == 2
    assert intensity.shape == (DEFAULT.grid_n, DEFAULT.grid_n)
    assert np.all(np.isfinite(intensity))
    assert intensity.max() > 0
    # PSF peak should sit at the array center (within 1 pixel)
    peak = np.unravel_index(np.argmax(intensity), intensity.shape)
    center = (DEFAULT.grid_n // 2, DEFAULT.grid_n // 2)
    assert abs(peak[0] - center[0]) <= 1
    assert abs(peak[1] - center[1]) <= 1
    # Sampling at the focal plane: roughly lambda * F / grid_size_at_pupil
    expected = DEFAULT.wavelength_m * DEFAULT.focal_length_m / DEFAULT.grid_size_m
    assert np.isclose(sampling, expected, rtol=0.1), \
        f"focal-plane sampling {sampling:.3e} vs expected {expected:.3e}"


@pytest.mark.skip(reason="macos rx_text + pymacos load not yet wired")
def test_compare_circular_psf(pymacos_session, tol, session_dir):
    """End-to-end macos-vs-PROPER intensity comparison.

    Skipped until ``CircularPupilFocus.macos_rx_text()`` returns a
    valid prescription and the pymacos load/trace path lands intensity
    at the focal plane.
    """
    from .geometries.circular_pupil_focus import macos_rx_text
    from .conftest import resample_to_common_grid

    geom = DEFAULT
    proper_int, _ = proper_run(geom)

    rx_path = session_dir / "circ_pupil_focus.in"
    rx_path.write_text(macos_rx_text(geom))
    pymacos_session.load(str(rx_path))
    pymacos_session.trace_rays(-1)  # detector surface
    # macos_int = ... pull the INT/PIX array from pymacos here ...
    macos_int = np.zeros_like(proper_int)  # placeholder

    a, b = resample_to_common_grid(proper_int, macos_int)
    a /= a.max()
    b /= b.max()
    np.testing.assert_allclose(a, b,
                               atol=tol["intensity_abs"],
                               rtol=tol["intensity_rel"])
