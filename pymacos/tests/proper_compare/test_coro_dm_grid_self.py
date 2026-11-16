"""
Phase 6c: macos self-consistency check -- apodize_complex (post-trace
phase imprint) vs grid-surface deformation (pre-trace, physically
correct).

Setup (mirrors Phase 6b but stays inside macos):
  Method A -- post-trace:  load Rx_Coro_DM.in (grid surface left at
                            zero), trace to Elt 5, apply OPD via
                            apodize_complex(5, exp(i*2pi*OPD/lambda)),
                            propagate to Elt 6.
  Method B -- pre-trace:   load Rx_Coro_DM.in, set Elt 4's grid
                            surface to grid_dz = OPD / 2 (the
                            reflector-OPD factor), trace to Elt 6
                            with no apodize_complex.

Both methods see the SAME OPD function inside the support disk.  The
diff at Elt 6 quantifies the ray-trace-vs-post-add-phase coupling:
  - beam wander: 2 grad(OPD) tilts each ray; over the (Elt 4 -> 5 ->
    6) propagation the rays land at slightly different pixels.
  - downstream-OPD-from-slope: a non-zero surface slope at the DM
    produces an extra OPD contribution at any downstream surface
    that 'apodize_complex' (a flat-mirror phase add) ignores.

Expected magnitude for lambda/20 RMS over 16 mm support: slope ~ 3
microrad, lateral wander at Elt 6 (z ~ 1 m downstream) ~ 3 microm
-- well under dx_at(6) = 334 microm, so the wander effect lives
inside one pixel and only modulates the wavefront via local
diffraction.  Same-pixel apodization is essentially exact for both
methods so the residual should remain near the Phase 3a 3.7e-8
floor; significant departures would localize a real second-order
effect.

Three OPD shapes (same as Phase 6b): defocus, 5-cycle sinusoid,
and filtered-noise (k_max = 12 cycles per pupil radius).
"""
from __future__ import absolute_import

from pathlib import Path

import numpy as np
import pytest

from .conftest import compare_and_record

pytestmark = pytest.mark.proper_compare


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RX_FILENAME    = "Rx_Coro_DM.in"
SRC_ELT        = 5           # NF prop start (same as Phase 6b)
DETECTOR_ELT   = 6           # NF prop end
DM_ELT         = 4           # the grid-surface reflector (Surface=FreeForm)
MACOS_SIZE     = 1024        # diffraction-grid resolution
WAVELENGTH_M   = 8.5e-7
R_NORM_M       = 14.0e-3     # OPD support radius (well inside 21 mm aperture)
RMS_TARGET_M   = WAVELENGTH_M / 20.0   # 42.5 nm RMS

# Grid spec on the DM surface -- matched to Elt 4's prescription.
DM_GRID_NPTS   = 1024
DM_GRID_DX_MM  = 0.33382


# ---------------------------------------------------------------------------
# OPD-shape generators (same recipe as Phase 6b, just regenerated at
# the surface-grid resolution so both methods see identical OPDs).
# ---------------------------------------------------------------------------

def _mesh(N: int, dx_m: float):
    ax = (np.arange(N) - (N - 1) / 2) * dx_m
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)
    return xx, yy, rr


def opd_defocus(N: int, dx_m: float, *, r_norm_m: float, rms_m: float):
    _xx, _yy, rr = _mesh(N, dx_m)
    rho = rr / r_norm_m
    z = 2.0 * rho * rho - 1.0
    inside = rho <= 1.0
    z = np.where(inside, z, 0.0)
    rms = float(np.sqrt(np.mean(z[inside] ** 2)))
    if rms > 0:
        z *= rms_m / rms
    return z


def opd_sinusoid(N: int, dx_m: float, *, r_norm_m: float,
                 cycles_per_pupil: float, rms_m: float):
    xx, _yy, rr = _mesh(N, dx_m)
    pupil_D = 2.0 * r_norm_m
    s = np.sin(2.0 * np.pi * cycles_per_pupil * xx / pupil_D)
    inside = rr <= r_norm_m
    s = np.where(inside, s, 0.0)
    rms = float(np.sqrt(np.mean(s[inside] ** 2)))
    if rms > 0:
        s *= rms_m / rms
    return s


def opd_filtered_noise(N: int, dx_m: float, *, r_norm_m: float,
                       max_cycles_per_pupil_radius: float,
                       rms_m: float, seed: int = 12345):
    _xx, _yy, rr = _mesh(N, dx_m)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((N, N))
    fx = np.fft.fftfreq(N, d=dx_m)
    fy = np.fft.fftfreq(N, d=dx_m)
    kxx, kyy = np.meshgrid(fx, fy, indexing="ij")
    k_mag = np.sqrt(kxx * kxx + kyy * kyy)
    k_cutoff = max_cycles_per_pupil_radius / r_norm_m
    F = np.fft.fft2(noise)
    F[k_mag > k_cutoff] = 0.0
    filt = np.real(np.fft.ifft2(F))
    inside = rr <= r_norm_m
    filt = np.where(inside, filt, 0.0)
    rms = float(np.sqrt(np.mean(filt[inside] ** 2)))
    if rms > 0:
        filt *= rms_m / rms
    return filt


# ---------------------------------------------------------------------------
# Engine drivers
# ---------------------------------------------------------------------------

def _rx_path() -> Path:
    return (Path(__file__).resolve().parents[1] / "Rx" / RX_FILENAME)


def _method_a_apodize(pymacos, opd_wf_m, detector_elt: int):
    """Post-trace phase imprint at Elt 5.  Grid surface stays flat."""
    pymacos.init(MACOS_SIZE)
    pymacos.load(str(_rx_path()))
    # Grid surface defaults to zero (flat DM) -- nothing to set.
    pymacos.intensity(SRC_ELT)
    phi = 2.0 * np.pi * opd_wf_m / WAVELENGTH_M
    mask = np.cos(phi) + 1j * np.sin(phi)
    pymacos.apodize_complex(SRC_ELT, mask)
    inten = pymacos.intensity(detector_elt, reset_trace=False)
    dx_det = pymacos.dx_at(detector_elt)
    return inten, dx_det


def _method_b_grid_surface(pymacos, opd_grid_m, detector_elt: int):
    """Pre-trace grid deformation at Elt 4 (DM).  grid_dz = OPD / 2
    (reflector OPD = 2 * surface displacement at normal incidence).

    Convention note: ``elt_grid()``'s Python wrapper transposes its
    input before handing to Fortran (macos.py: ``grid_dz.T``), whereas
    ``apodize_complex()`` does not.  To make method B sample the
    SAME physical OPD pattern as method A's WFElt-aligned array, we
    transpose the surface OPD here -- gives the two methods a
    consistent (row=y, col=x) interpretation across the comparison.
    Without this, asymmetric OPD shapes (e.g. an x-direction
    sinusoid) produce focal-plane satellite spots rotated by 90
    between the two methods.
    """
    pymacos.init(MACOS_SIZE)
    pymacos.load(str(_rx_path()))
    # grid_dz: surface displacement in BaseUnits (mm).  OPD in metres,
    # convert via 1e3.  Reflector OPD = 2 * surface, so surface = OPD/2.
    surface_mm = (opd_grid_m / 2.0) * 1e3
    pymacos.elt_grid(DM_ELT, DM_GRID_DX_MM, surface_mm.T)
    inten = pymacos.intensity(detector_elt)
    dx_det = pymacos.dx_at(detector_elt)
    return inten, dx_det


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

OPD_SHAPES = [
    ("defocus_Z4", lambda N, dx: opd_defocus(
        N, dx, r_norm_m=R_NORM_M, rms_m=RMS_TARGET_M)),
    ("sinusoid_5cyc", lambda N, dx: opd_sinusoid(
        N, dx, r_norm_m=R_NORM_M, cycles_per_pupil=5.0,
        rms_m=RMS_TARGET_M)),
    ("filtered_noise_k12", lambda N, dx: opd_filtered_noise(
        N, dx, r_norm_m=R_NORM_M, max_cycles_per_pupil_radius=12.0,
        rms_m=RMS_TARGET_M, seed=12345)),
]

# Detectors:
#   Elt 6  = near-field plane immediately after the NF prop pair --
#            isolates the propagation-pipeline residual.
#   Elt 21 = science focal plane (end of chain) -- shows how the
#            method-A-vs-method-B difference propagates through the
#            full coronagraph to the user-visible PSF.
DETECTORS = [
    ("elt6_nf",     6),
    ("elt21_focal", 21),
]


@pytest.mark.parametrize("shape_name, opd_fn",
                         OPD_SHAPES,
                         ids=[s[0] for s in OPD_SHAPES])
@pytest.mark.parametrize("det_name, det_elt",
                         DETECTORS,
                         ids=[d[0] for d in DETECTORS])
def test_coro_dm_grid_self_consistency(pymacos_session,
                                       results_dir_phase6,
                                       shape_name, opd_fn,
                                       det_name, det_elt):
    """Compare post-trace apodize_complex vs pre-trace grid-surface
    deformation for the SAME OPD function.  Residual quantifies the
    ray-trace-vs-post-add-phase difference."""

    # Probe macos's dx_at(SRC_ELT) so the wavefront-resolution OPD
    # array lines up with the WFElt sampling for apodize_complex.
    import pymacos.macos as m_probe
    m_probe.init(MACOS_SIZE)
    m_probe.load(str(_rx_path()))
    m_probe.intensity(SRC_ELT)
    dx_wf_m = m_probe.dx_at(SRC_ELT)

    # Build OPD at both samplings:
    #   wf grid (1024 x 0.334 mm) for apodize_complex
    #   surface grid (128 x 0.33382 mm) for elt_grid
    # The dx values match to 5 sig figs (probed wf vs prescribed
    # surface), so both samplings cover the same physical extent
    # of the OPD function; only the resolution differs.
    opd_wf      = opd_fn(MACOS_SIZE, dx_wf_m)
    opd_surface = opd_fn(DM_GRID_NPTS, DM_GRID_DX_MM * 1e-3)

    # Confirm RMS within 1% of target on the wf-grid (the version
    # apodize_complex uses).
    nz = opd_wf[opd_wf != 0.0]
    rms_actual = float(np.sqrt(np.mean(nz ** 2))) if nz.size else 0.0
    assert rms_actual == pytest.approx(RMS_TARGET_M, rel=1e-2), \
        f"OPD RMS {rms_actual:.3e} != target {RMS_TARGET_M:.3e}"

    # Method A: apodize_complex (post-trace).
    inten_a, dx_a = _method_a_apodize(pymacos_session, opd_wf, det_elt)

    # Method B: grid-surface (pre-trace).
    inten_b, dx_b = _method_b_grid_surface(pymacos_session,
                                            opd_surface, det_elt)

    assert inten_a.shape == inten_b.shape
    # Pixel pitches can differ slightly: the deformed-mirror trace
    # (method B) produces a beam with slightly different downstream
    # geometry than the flat-mirror trace (method A), and macos's
    # dx_at(detector) reflects that.  Capture the dx mismatch as a
    # diagnostic but DON'T equate them -- a non-zero dx difference
    # IS the ray-trace coupling effect we're trying to measure.
    dx_rel_diff = abs(dx_a - dx_b) / abs(dx_a)
    print(f"[6c-self {shape_name}] dx_A = {abs(dx_a):.6e}, "
          f"dx_B = {abs(dx_b):.6e}, rel-diff = {dx_rel_diff:.3e}")

    # Compare via the same harness used for macos<->PROPER (sum-norm
    # for NF intensities), labelled phase6c to keep the report row
    # adjacent to the Phase 6b apodize_complex<->PROPER residual.
    # Norm kind: 'sum' for NF intensity comparison (Elt 6),
    # 'peak' for focal-plane PSF (Elt 21) -- matches the convention
    # in conftest header ("sum for pupil/NF, peak for image-plane").
    norm_kind = 'peak' if det_elt >= 20 else 'sum'

    metrics = compare_and_record(
        f'coro_dm_self_{shape_name}_{det_name}',
        inten_a, inten_b, abs(dx_a),
        results_dir_phase6,
        crop_pixels=inten_a.shape[0],
        norm_kind=norm_kind,
        extra_metadata={
            'wavelength_m':    WAVELENGTH_M,
            'src_elt':         SRC_ELT,
            'detector_elt':    det_elt,
            'dm_elt':          DM_ELT,
            'rx_filename':     RX_FILENAME,
            'opd_shape':       shape_name,
            'opd_rms_m':       rms_actual,
            'opd_support_r_m': R_NORM_M,
            'method_a':        'apodize_complex',
            'method_b':        'elt_grid (surface = OPD/2)',
        },
    )
    # No hard threshold here -- this test characterises a real
    # physical difference rather than a numerical floor.  The
    # report.md row captures the residual for analysis.
    print(f"[6c-self {shape_name}@{det_name}] "
          f"norm={norm_kind} "
          f"max diff = {metrics['max_abs']:.3e}, "
          f"RMS diff = {metrics['rms_abs']:.3e}, "
          f"shift = ({metrics['dx_pix']:+d}, {metrics['dy_pix']:+d}) pix")
