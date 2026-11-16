"""Aberration sweep through the full Rx_Coro_FPM_Zern.in chain
(Phase 5.2 configuration: FPM=400um + Lyot=14mm).

The state vector model (Dave, 2026-05-16):

  - For each real optic i: alignment 6-vector
        x_i = [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
    (radians + SI metres).
  - For each Zernike optic: figure vector z_i (SI metres surface RMS).
  - Full system state: SystemState(layout, x, z).

Real users (control loops, EFC, Monte-Carlo) operate on x and z; the
test harness uses the same data model to make the test reproducible
and informative.

Each parametrised SystemState is run through the stand-alone
``run_chain_with_state`` callable, which produces macos and PROPER
focal-plane intensities at Elt 21 plus the radial-contrast curve.
An overlay plot of all curves is saved.
"""
from __future__ import absolute_import

from pathlib import Path

import numpy as np
import pytest

from .aberrations import (
    CORO_LAYOUT, SystemState, run_chain_with_state)
from .contrast import lambda_over_D_pixels, plot_contrast_curves
from .geometries.coro_nfprop import (
    CoroSphereToPlane, macos_run_sphere_to_plane)

pytestmark = pytest.mark.proper_compare


GEOM = CoroSphereToPlane(
    rx_filename="Rx_Coro_FPM_Zern.in",
    src_elt=20, detector_elt=21,
    focal_length_m=0.9514,
)


# ----------------------------------------------------------------------
# Nominal-reference fixture: un-coronagraphed PSF for Strehl + lambda/D.
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def nominal_reference(pymacos_session):
    geom_no_mask = CoroSphereToPlane(
        rx_filename="Rx_Coro_noLyot.in",
        src_elt=20, detector_elt=21,
        macos_size=GEOM.macos_size,
        focal_length_m=GEOM.focal_length_m,
    )
    I_no, _, _ = macos_run_sphere_to_plane(geom_no_mask, pymacos_session)
    return (float(I_no.max()), float(lambda_over_D_pixels(I_no)))


# ----------------------------------------------------------------------
# Perturbation states.  Each is built by mutating a fresh nominal
# SystemState via set_dof / set_zernike (the user-facing API).
# ----------------------------------------------------------------------

def _make_states():
    states = []
    # 1. Nominal
    states.append(SystemState(layout=CORO_LAYOUT, name="nominal"))

    # 2. Elt 1 tip about y by 10 urad -- a tilt of the M1.  Wavefront
    #    tilt is doubled by reflection (20 urad), which shifts the
    #    focal-plane PSF by f * 20e-6 = 0.9514 * 2e-5 = 19 um =
    #    ~3.3 focal-plane pixels.  CLEAR demonstrator that the
    #    perturbation lands on the wavefront.
    s = SystemState(layout=CORO_LAYOUT, name="Elt1_tip_y_10urad")
    s.set_dof(element=1, dof="roty", value=10e-6)
    states.append(s)

    # 3. Elt 1 Tx +5 um -- pure translation.  For a curved mirror,
    #    this is mostly a wavefront-tilt-equivalent; less visible
    #    radially.
    s = SystemState(layout=CORO_LAYOUT, name="Elt1_Tx_plus_5um")
    s.set_dof(element=1, dof="transx", value=5e-6)
    states.append(s)

    # 4. DM1 mixed Zernike: astigmatism + trefoil + Z33, each 3 nm.
    s = SystemState(layout=CORO_LAYOUT, name="DM1_astig+trefoil+Z33_3nm")
    s.set_zernike(element=4, mode=5,  coef_m=3e-9)
    s.set_zernike(element=4, mode=9,  coef_m=3e-9)
    s.set_zernike(element=4, mode=33, coef_m=3e-9)
    states.append(s)

    # 5. Elt 17 Tx +5 um -- downstream of the coronagraph.  Affects
    #    PSF position but not coronagraph suppression mechanism.
    s = SystemState(layout=CORO_LAYOUT, name="Elt17_Tx_plus_5um")
    s.set_dof(element=17, dof="transx", value=5e-6)
    states.append(s)

    return states


PERTURBATION_STATES = _make_states()


# ----------------------------------------------------------------------
# Focal-plane plot helper -- log-stretched 2D intensity, cropped to a
# window around the PSF centre, in lambda/D units.
# ----------------------------------------------------------------------

def _save_focal_plane_plot(state_name: str,
                           intensity: np.ndarray,
                           lambda_over_D_px: float,
                           out_dir: Path,
                           window_lambda_over_D: float = 20.0):
    """Save a log10 intensity plot for ONE aberration case.

    Crops to +/- window_lambda_over_D around the (N-1)/2 centre.
    Axes labelled in lambda/D so speckle positions are physically
    meaningful.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = intensity.shape[0]
    half_window_px = int(window_lambda_over_D * lambda_over_D_px)
    cy = cx = (N - 1) // 2
    lo_y = max(0, cy - half_window_px)
    hi_y = min(N, cy + half_window_px + 1)
    lo_x = max(0, cx - half_window_px)
    hi_x = min(N, cx + half_window_px + 1)
    crop = intensity[lo_y:hi_y, lo_x:hi_x]

    # Extent in lambda/D from the crop centre.
    ext = (-(cx - lo_x) / lambda_over_D_px,
            (hi_x - cx - 1) / lambda_over_D_px,
           -(cy - lo_y) / lambda_over_D_px,
            (hi_y - cy - 1) / lambda_over_D_px)

    # Log-stretch with a floor at log10(peak) - 10.
    peak = crop.max() if crop.max() > 0 else 1.0
    floor = max(peak * 1e-10, 1e-30)
    log_crop = np.log10(np.maximum(crop, floor))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(log_crop, extent=ext, origin="lower", cmap="viridis",
                   vmin=np.log10(floor), vmax=np.log10(peak))
    ax.set_xlabel(r"x ($\lambda/D$)")
    ax.set_ylabel(r"y ($\lambda/D$)")
    ax.set_title(f"{state_name}\nmacos focal-plane intensity, "
                 r"log$_{10}$;  peak = " f"{peak:.3e}")
    plt.colorbar(im, ax=ax, label=r"log$_{10}$ intensity")
    fig.tight_layout()
    out_path = out_dir / f"aberration_{state_name}_focalplane.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------
# Module-scoped result cache for the aggregate plot.
# ----------------------------------------------------------------------
_results: dict = {}


@pytest.mark.parametrize("state", PERTURBATION_STATES,
                         ids=lambda s: s.name)
def test_coro_aberration(state: SystemState, pymacos_session,
                          nominal_reference, results_dir_phase3):
    """For each state, run macos+PROPER and verify macos<->PROPER
    agreement holds and the result is physically sensible.
    """
    peak_ref, lam_D = nominal_reference

    result = run_chain_with_state(
        state, GEOM, pymacos_session,
        peak_unaberrated=peak_ref,
        lambda_over_D_px=lam_D)

    _results[state.name] = result

    # Per-state 2D focal-plane plot (so we can SEE speckle redistribution
    # that the radial-averaged contrast curve smears out).
    _save_focal_plane_plot(state.name,
                            result['intensity_macos'],
                            result['lambda_over_D_px'],
                            Path(results_dir_phase3))

    assert result['intensity_macos'].max() > 0
    assert result['intensity_proper'].max() > 0
    # macos<->PROPER agreement holds under perturbation.  Slightly
    # looser bound than Phase 5.2 nominal (2.6e-6) -- some
    # perturbations push the wavefront structure around.
    assert result['agreement_max_abs'] < 1e-4, (
        f"{state.name}: macos<->PROPER max|a-b| = "
        f"{result['agreement_max_abs']:.3e} exceeds 1e-4")


def test_aberration_overlay_plot(results_dir_phase3):
    """Assembled at session end: overlay all aberration cases'
    contrast curves on one log-y plot, print a digest table.
    """
    if not _results:
        pytest.skip("no aberration results to plot (parametrised tests "
                    "didn't run yet)")

    curves = {}
    for name, res in _results.items():
        label = name.replace("_", " ")
        curves[label] = (res['contrast_r_lambda_over_D'],
                          res['contrast_values'])

    out_path = Path(results_dir_phase3) / "aberration_contrast.png"
    plot_contrast_curves(
        curves, out_path,
        title=("Rx_Coro_FPM_Zern.in: dark-zone contrast under "
               "perturbations\n"
               "FPM=400um + Lyot=14mm; Strehl-norm to un-coronagraphed peak"),
        ylim=(1e-14, 1e-3),
    )
    print(f"\n[aberrations] wrote {out_path}")

    print(f"\n  {'state':<28s}  {'agreement':>11s}  "
          f"{'peak macos':>11s}  {'@ 3 λ/D':>10s}  {'@ 7 λ/D':>10s}")
    for name, res in _results.items():
        r = res['contrast_r_lambda_over_D']
        c = res['contrast_values']
        c_3 = c[int(np.argmin(np.abs(r - 3.0)))]
        c_7 = c[int(np.argmin(np.abs(r - 7.0)))]
        peak = res['intensity_macos'].max()
        print(f"  {name:<28s}  {res['agreement_max_abs']:>11.3e}  "
              f"{peak:>11.3e}  {c_3:>10.3e}  {c_7:>10.3e}")
