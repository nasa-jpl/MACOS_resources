"""Radial-contrast scoring of the Phase 5 coronagraph baselines.

Computes radially-averaged contrast at the science focal plane (Elt
21) of Rx_Coro for both the no-mask reference (Phase 5.1) and the
FPM=400um + Lyot=14mm coronagraph (Phase 5.2), normalised by the
no-mask on-axis peak.  Saves a single overlaid contrast plot in
``results_phase3/contrast_curves.png``.

The metric:
    contrast(r) = mean(intensity in radial ring at r) / peak_no_mask

is the form used by the coronagraph literature ("dark-zone contrast")
and decouples scoring from the specific peak values that macos and
PROPER each report (peak macos = 0.30 vs peak PROPER = 0.012 from
their different normalisation conventions; sum vs peak normalisation
isn't directly comparable across engines anyway, but Strehl-norm
contrast IS once both are referenced to the same un-coronagraphed
peak).

This test is macos-only (no PROPER comparison); the macos<->PROPER
agreement is already validated to 2.3e-9 (5.1) and 2.6e-6 (5.2) in
test_coro_nfprop_phase3.py.  Here we just exercise the new contrast
machinery on the proven physics.
"""
from __future__ import absolute_import

from pathlib import Path

import numpy as np
import pytest

from .geometries.coro_nfprop import (
    CoroSphereToPlane, macos_run_sphere_to_plane)
from .contrast import (radial_contrast, lambda_over_D_pixels,
                       plot_contrast_curves)

pytestmark = pytest.mark.proper_compare


NO_MASK = CoroSphereToPlane(
    rx_filename="Rx_Coro_noLyot.in",
    src_elt=20, detector_elt=21,
    focal_length_m=0.9514,
)

WITH_MASK = CoroSphereToPlane(
    rx_filename="Rx_Coro_FPM.in",   # FPM=400um + Lyot=14mm
    src_elt=20, detector_elt=21,
    focal_length_m=0.9514,
)


def test_coronagraph_contrast_curves(pymacos_session, results_dir_phase3):
    """Run the no-mask and FPM+Lyot focal-plane PSFs, derive lambda/D
    empirically from the no-mask first Airy null, score both as
    radial contrast vs lambda/D, save an overlay plot.
    """
    # No-mask PSF: Strehl reference
    I_no, _, _ = macos_run_sphere_to_plane(NO_MASK, pymacos_session)
    peak_no = float(I_no.max())

    # Empirical lambda/D from the no-mask PSF's first Airy null
    lam_D = lambda_over_D_pixels(I_no)
    print(f"\n[contrast] lambda/D = {lam_D:.2f} px "
          f"(first null at 1.22 lambda/D = {1.22 * lam_D:.2f} px)")
    print(f"[contrast] no-mask peak = {peak_no:.4e}")

    # With-mask PSF: scoring target
    I_co, _, _ = macos_run_sphere_to_plane(WITH_MASK, pymacos_session)
    peak_co = float(I_co.max())
    print(f"[contrast] with-mask peak = {peak_co:.4e}  "
          f"(suppression factor {peak_no / peak_co:.2e})")

    # Radial contrast curves (normalised to no-mask peak)
    r_no, c_no = radial_contrast(I_no, peak_no, lam_D,
                                 max_lambda_over_D=20.0)
    r_co, c_co = radial_contrast(I_co, peak_no, lam_D,
                                 max_lambda_over_D=20.0)

    # Plot
    out_png = Path(results_dir_phase3) / "contrast_curves.png"
    plot_contrast_curves(
        {'no mask (baseline)':       (r_no, c_no),
         'FPM=400um + Lyot=14mm':    (r_co, c_co)},
        out_png,
        title=("Rx_Coro radial contrast at science focal plane (Elt 21)\n"
               "Strehl-normalised to un-coronagraphed peak"),
        ylim=(1e-14, 2.0),
    )
    print(f"[contrast] wrote {out_png}")

    # Print a dark-zone digest for the report
    print(f"\n[contrast] Radial contrast at key separations:")
    print(f"{'r (lambda/D)':>14s}  {'no-mask':>10s}  {'with-mask':>10s}  "
          f"{'gain':>9s}")
    for r_target in [0, 1, 2, 3, 5, 7, 10, 15]:
        i_no = int(np.argmin(np.abs(r_no - r_target)))
        i_co = int(np.argmin(np.abs(r_co - r_target)))
        gain = c_no[i_no] / max(c_co[i_co], 1e-300)
        print(f"{r_no[i_no]:>14.2f}  {c_no[i_no]:>10.3e}  "
              f"{c_co[i_co]:>10.3e}  {gain:>9.2e}")

    # Smoke checks.  Note: the radial-profile r=0 bin is NaN under
    # even-N grids (no pixel at exact array centre), so use the peak
    # ratio for on-axis contrast and the first FINITE radial bin for
    # the "near on-axis" smoke.
    suppress_factor = peak_no / peak_co
    assert 1e6 < suppress_factor < 1e8, (
        f"suppression factor {suppress_factor:.2e} outside the "
        "expected 1e6..1e8 range for FPM=400um + Lyot=14mm "
        "(Phase 5.2 reports factor ~3.2 million)")
    # First finite bin of the no-mask curve should be near Strehl=1.
    first_finite = int(np.argmax(np.isfinite(c_no)))
    assert c_no[first_finite] > 0.5, (
        f"no-mask near-on-axis contrast at r={r_no[first_finite]:.2f} "
        f"lambda/D = {c_no[first_finite]:.3e} -- lambda/D detection "
        "or peak finding broken")
    # With-mask near-on-axis must be heavily suppressed.
    assert c_co[first_finite] < 1e-4, (
        f"with-mask near-on-axis contrast at r={r_co[first_finite]:.2f} "
        f"lambda/D = {c_co[first_finite]:.3e} -- coronagraph not "
        "suppressing as expected")
