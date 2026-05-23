"""
Phase 6b: DM phase-imprint check -- pure macos vs PROPER comparison
for a known OPD pattern injected at a pupil-like NF plane.

This is the *phase-only* analog of Phase 6a's apodizer test:

  apodizer (6a):  pure amplitude mask via m.apodize / prop_multiply
  DM    (this):   pure phase   mask via m.apodize_complex / prop_add_phase

The test does NOT model DM actuators / influence functions -- both
engines see the SAME OPD array as input and the comparison validates
the downstream propagation of a phase imprint at a pupil plane.
Actuated DM models (macos's UDSrfType / PROPER's prop_dm) differ in
their influence-function families and will be compared separately
once the pure phase-pipe agreement is established here.

Three OPD shapes test the diagnostic ground:
  1. Defocus (low spatial frequency, smooth Z4)
  2. 5-cycle x-direction sinusoid (mid-frequency, the regime that
     matters for DM speckle control)
  3. Filtered white noise capped at ~N/8 cycles/pupil (high-freq
     content near the Nyquist limit for a 100x100 DM)

Geometry mirrors Phase 3a / Phase 6a: Rx_Coro_noLyot.in, NF
propagation Elt 5 -> Elt 6 (774 mm).  Sum-normalised intensity
residual at Elt 6 should match Phase 3a's 3.7e-8 floor (or better,
since softer phase content suppresses the high-frequency edge ring
that limited Phase 3a's hard-edge geometry).
"""
from __future__ import absolute_import

import numpy as np
import pytest
import proper

from .conftest import compare_and_record
from .geometries.coro_nfprop import CoroNFprop

pytestmark = pytest.mark.proper_compare


# Phase 3a / Phase 6a geometry: NFPlane between Elt 5 and Elt 6 of
# the no-Lyot coronagraph chain.
GEOM = CoroNFprop(
    rx_filename="Rx_Coro_noLyot.in",
    src_elt=5, detector_elt=6,
    propagation_m=0.774,
)


# ---------------------------------------------------------------------------
# OPD-shape generators -- all return real-valued (N, N) arrays at
# pitch dx_m.  Amplitudes scaled to a fixed RMS (lambda/20) so the
# three shapes are at comparable wavefront budgets.
# ---------------------------------------------------------------------------

def _radius_mesh(N: int, dx_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_mesh, y_mesh, r_mesh) in metres, centred on the grid."""
    ax = (np.arange(N) - (N - 1) / 2) * dx_m
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)
    return xx, yy, rr


def opd_defocus(N: int, dx_m: float, *, r_norm_m: float, rms_m: float
                ) -> np.ndarray:
    """Z4 defocus: 2 rho^2 - 1 on unit disk r/r_norm <= 1.

    The Z4 form gives a smooth quadratic phase that's the cleanest
    diagnostic for "do my low-order phases propagate identically?"
    Outside r_norm the wavefront is unaffected (mask zeroed).
    """
    _xx, _yy, rr = _radius_mesh(N, dx_m)
    rho = rr / r_norm_m
    z4 = 2.0 * rho * rho - 1.0
    inside = rho <= 1.0
    z4 = np.where(inside, z4, 0.0)
    # Rescale to requested RMS over the support disk.
    rms_actual = float(np.sqrt(np.mean(z4[inside] ** 2)))
    if rms_actual > 0:
        z4 *= rms_m / rms_actual
    return z4


def opd_sinusoid(N: int, dx_m: float, *, r_norm_m: float,
                 cycles_per_pupil: float, rms_m: float) -> np.ndarray:
    """Sinusoidal ripple along x with the given spatial frequency.

    OPD(x, y) = A * sin(2*pi * k * x / (2*r_norm)) on the unit disk;
    A chosen so the disk RMS = rms_m.  k cycles per pupil diameter.
    """
    xx, _yy, rr = _radius_mesh(N, dx_m)
    pupil_D = 2.0 * r_norm_m
    s = np.sin(2.0 * np.pi * cycles_per_pupil * xx / pupil_D)
    inside = rr <= r_norm_m
    s = np.where(inside, s, 0.0)
    rms_actual = float(np.sqrt(np.mean(s[inside] ** 2)))
    if rms_actual > 0:
        s *= rms_m / rms_actual
    return s


def opd_filtered_noise(N: int, dx_m: float, *, r_norm_m: float,
                       max_cycles_per_pupil_radius: float,
                       rms_m: float, seed: int = 12345
                       ) -> np.ndarray:
    """Random low-pass-filtered phase -- spectrum flat to k_max,
    zero above.  Stresses the spatial-frequency band that a 100x100
    DM would actually drive (N/8 cycles per pupil radius ~ 12 for
    a 100x100 DM).  Deterministic via the seed.
    """
    _xx, _yy, rr = _radius_mesh(N, dx_m)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((N, N))
    # FFT-space filter: keep frequencies where |k| / k_pupil <= k_max
    # where k_pupil = 1 / pupil_diameter (cycles per metre).
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
    rms_actual = float(np.sqrt(np.mean(filt[inside] ** 2)))
    if rms_actual > 0:
        filt *= rms_m / rms_actual
    return filt


# ---------------------------------------------------------------------------
# Engine drivers
# ---------------------------------------------------------------------------

def _macos_phase(geom: CoroNFprop, opd_m: np.ndarray, pymacos_session):
    """Drive macos: populate WFElt at src, apply complex phase imprint,
    propagate to detector, return intensity + dx.

    The phase mask is exp(i * 2*pi * OPD / lambda) -- pure phase,
    unit magnitude.  apodize_complex multiplies WFElt by it in place;
    reset_trace=False on the second intensity() preserves the
    imprinted phase through the propagation.
    """
    from pathlib import Path
    rx_path = (Path(__file__).resolve().parents[1] / "Rx"
               / geom.rx_filename)
    pymacos_session.init(geom.macos_size)
    pymacos_session.load(str(rx_path))

    pymacos_session.intensity(geom.src_elt)
    cfield_pre = pymacos_session.complex_field(geom.src_elt,
                                                reset_trace=False)
    dx_src_m = pymacos_session.dx_at(geom.src_elt)

    # Build phase mask.  exp(i k OPD) with k = 2*pi/lambda; the
    # OPD-to-phase sign matches PROPER's prop_add_phase (positive OPD
    # advances phase).
    phi = 2.0 * np.pi * opd_m / geom.wavelength_m
    mask_re = np.cos(phi)
    mask_im = np.sin(phi)
    mask_complex = mask_re + 1j * mask_im

    pymacos_session.apodize_complex(geom.src_elt, mask_complex)
    intensity_det = pymacos_session.intensity(geom.detector_elt,
                                              reset_trace=False)
    dx_det_m = pymacos_session.dx_at(geom.detector_elt)

    return (intensity_det, dx_det_m,
            dict(complex_field_pre=cfield_pre,
                 dx_at_src_m=dx_src_m,
                 dx_at_det_m=dx_det_m,
                 opd_m=opd_m))


def _proper_phase(geom: CoroNFprop, wavefront: dict):
    """PROPER counterpart: amplitude + nominal-OPD from macos's
    pre-imprint cfield, plus the same imprinted OPD via prop_add_phase.
    """
    cfield = np.asarray(wavefront['complex_field_pre'],
                        dtype=np.complex128)
    opd_m  = np.asarray(wavefront['opd_m'],
                        dtype=np.float64)
    dx_src = wavefront['dx_at_src_m']

    N = geom.macos_size
    grid_extent = N * dx_src
    wfo = proper.prop_begin(grid_extent, geom.wavelength_m, N, 1.0)

    # macos cfield = amplitude * exp(i phi_nom); reproduce both pieces.
    proper.prop_multiply(wfo, np.abs(cfield))
    opd_nom = -np.angle(cfield) * geom.wavelength_m / (2.0 * np.pi)
    proper.prop_add_phase(wfo, opd_nom)

    # Now add the DM phase imprint -- the test's whole point.
    proper.prop_add_phase(wfo, opd_m)

    proper.prop_define_entrance(wfo)
    proper.prop_propagate(wfo, geom.propagation_m)
    field, sampling = proper.prop_end(wfo)
    intensity = (np.abs(field) ** 2
                 if field.dtype.kind == 'c' else field)
    return intensity, sampling


# ---------------------------------------------------------------------------
# Test cases (parameterised over OPD shape)
# ---------------------------------------------------------------------------

# Geometry: Rx_Coro_noLyot.in Elt 5 has a natural beam radius of
# ~17-18 mm at the macos diffraction grid.  Anchor the OPD support
# disk at 16 mm (well inside the un-clipped pupil) so the phase
# imprint sits where the wavefront is non-zero.
R_NORM_M = 16.0e-3

# All three shapes scaled to the same RMS phase (lambda/20).
def _rms_target(geom: CoroNFprop) -> float:
    return geom.wavelength_m / 20.0


def _probe_dx_at_src(geom: CoroNFprop) -> float:
    """Probe-load to discover macos's diffraction-grid pitch at src.
    Separate from the full run because the OPD array is needed BEFORE
    we drive the actual measurement.
    """
    import pymacos.macos as m
    from pathlib import Path
    rx_path = (Path(__file__).resolve().parents[1] / "Rx"
               / geom.rx_filename)
    m.init(geom.macos_size)
    m.load(str(rx_path))
    m.intensity(geom.src_elt)
    return m.dx_at(geom.src_elt)


OPD_SHAPES = [
    ("defocus_Z4", lambda N, dx, rms: opd_defocus(
        N, dx, r_norm_m=R_NORM_M, rms_m=rms)),
    ("sinusoid_5cyc", lambda N, dx, rms: opd_sinusoid(
        N, dx, r_norm_m=R_NORM_M, cycles_per_pupil=5.0, rms_m=rms)),
    ("filtered_noise_k12", lambda N, dx, rms: opd_filtered_noise(
        N, dx, r_norm_m=R_NORM_M,
        max_cycles_per_pupil_radius=12.0, rms_m=rms, seed=12345)),
]


@pytest.mark.parametrize("shape_name, opd_fn",
                         OPD_SHAPES,
                         ids=[s[0] for s in OPD_SHAPES])
def test_coro_dm_phase_imprint(pymacos_session, results_dir_phase6,
                                shape_name, opd_fn):
    dx_at_src = _probe_dx_at_src(GEOM)
    opd_m = opd_fn(GEOM.macos_size, dx_at_src, _rms_target(GEOM))

    # Sanity: RMS within 1% of the target.
    nz = opd_m[opd_m != 0.0]
    rms_actual = float(np.sqrt(np.mean(nz ** 2))) if nz.size else 0.0
    assert rms_actual == pytest.approx(_rms_target(GEOM), rel=1e-2), \
        f"OPD RMS {rms_actual:.3e} != target {_rms_target(GEOM):.3e}"

    intensity_m, dx_m, wf = _macos_phase(GEOM, opd_m, pymacos_session)
    intensity_p, dx_p     = _proper_phase(GEOM, wf)

    assert intensity_m.shape == intensity_p.shape
    assert dx_p == pytest.approx(abs(dx_m), rel=1e-3)

    metrics = compare_and_record(
        f'coro_dm_phase_{shape_name}',
        intensity_m, intensity_p, abs(dx_m),
        results_dir_phase6,
        crop_pixels=intensity_m.shape[0],
        norm_kind='sum',
        extra_metadata={
            'wavelength_m':       GEOM.wavelength_m,
            'propagation_m':      GEOM.propagation_m,
            'src_elt':            GEOM.src_elt,
            'detector_elt':       GEOM.detector_elt,
            'rx_filename':        GEOM.rx_filename,
            'opd_shape':          shape_name,
            'opd_rms_m':          rms_actual,
            'opd_support_r_m':    R_NORM_M,
        },
    )
    # Phase 3a's hard-edge baseline was 3.7e-8 RMS.  A pure phase
    # imprint shouldn't be worse than that on a soft-supported OPD.
    assert metrics['rms_abs'] < 1e-7, (
        f"{shape_name}: sum-norm RMS residual "
        f"{metrics['rms_abs']:.3e} > 1e-7 threshold")
