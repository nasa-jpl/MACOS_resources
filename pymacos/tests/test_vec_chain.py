"""Phase 3a Tranche 1 -- vector near-field propagation chain (PLAN_POLARIZATION §3a).

pymacos mirror of the mmacos ``tests/tVecChain.m`` gate.  Both bindings share
one ``libsmacos.a``, so this re-runs the same engine invariants through the
other language surface (and, when pymacos is linked with ifx, doubles as the
standing ifx smoke for the Tranche-1 changes).

What it pins:

  3a.1(1)  the vector far-field leg called ``PFFPROP``, a bare per-component
           FFT missing the Fresnel-integral output factors scalar ``FFPROP``
           applies through ``applyfac2``.  Both legs now share one kernel.
  3a.1(2)  the polarized field assembly RELOADED the grid from ``RayE`` at
           every physical leg -- erasing earlier legs' diffraction and
           resurrecting rays the ray-side aperture masking had already
           extinguished.  It now seeds once, then applies the same
           incremental geometric-phase update the non-polarized branch uses.
  3a.2     every near-field / DFT leg, ``FFObscure`` and the ray-side
           clip/taper sites now cover all three component planes.

The gate prescription ``Rx/Rx_VecChain.in`` is a collimated on-axis source
through flat, normal-incidence, uncoated planes, so the ray E-field direction
is a CONSTANT unit vector: the field factorises as ``E_k = e_k * u(x, y)`` and
propagating the three planes separately then summing ``|E_k|**2`` must
reproduce the scalar intensity to round-off at every leg, for ANY input
polarization state.  On a real off-normal train (Rx_Cass_FarField) vector and
scalar differ by ~2.6e-3, so no exact comparison is possible there -- hence
the dedicated fixture.

UNVERIFIED ATTRIBUTION.  That 2.6e-3 is *believed* to be the off-normal
train's out-of-plane content -- ``|Ez|/|Ex|`` measures ~8.8e-2 at the exit
pupil through ``ray_field``, the right order -- but it is NOT verified: there
is no plane-selectable complex-field getter, so the per-plane contribution to
the propagated intensity cannot currently be measured.  Treat it as a
plausible explanation, not a validated one; if a plane-selectable cfield
getter lands, close this out.  Nothing here DEPENDS on the attribution -- the
assertions bound the difference, they do not explain it.

NON-VACUITY (checked 2026-07-26 against the pre-fix engine): the pre-fix code
fails these at 0.21 .. 0.38 relative error and mis-states total power by 4-7%.
The 45-degree and circular states are load-bearing -- with an x-only source ALL
the energy sits in component plane 1, which the old single-plane propagator
happened to carry correctly, so an x-pol-only gate passes vacuously.
"""
from pathlib import Path

import numpy as np
import pytest

import context  # noqa: F401  -- adds ../src to sys.path
from pymacos import macos as m


RXDIR = Path(__file__).resolve().parent / 'Rx'
RX_CHAIN = str(RXDIR / 'Rx_VecChain.in')
RX_FF = str(RXDIR / 'Rx_Cass_FarField.in')

MODEL = 128      # mWF=3 -> vector diffraction available
LEG1 = 2         # MidStop  -- end of near-field leg 1
LEG2 = 4         # Detector -- end of near-field leg 2
FF_DET = 6       # Rx_Cass_FarField detector (single far-field hop)

TOL = 1e-13      # round-off budget for N=128 through two FFT-pair legs

_STATES = {
    'scalar':   None,
    'polsc':    ((1.0, 0.0), (0.0, 0.0), False),
    'vec_x':    ((1.0, 0.0), (0.0, 0.0), True),
    'vec_45':   ((1.0, 0.0), (1.0, 0.0), True),
    'vec_circ': ((1.0, 0.0), (0.0, 1.0), True),
}


def run_case(rx: str, mode: str, elt: int) -> np.ndarray:
    """Intensity at ``elt`` for one polarization mode, from a clean load."""
    m.init(MODEL)
    m.load(rx)
    spec = _STATES[mode]
    if spec is None:
        m.polarization('off')
    else:
        ex, ey, vec = spec
        m.polarization('on', Ex=ex, Ey=ey)
        m.vector_diffraction(vec)
    return np.asarray(m.intensity(elt))


def relerr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), np.finfo(float).tiny))


# ---- 3a.1(2): polarized-scalar must reduce to the scalar path ------------
@pytest.mark.parametrize('elt', [LEG1, LEG2])
def test_polarized_scalar_is_bit_identical(elt):
    """pol ON + vector OFF must reproduce pol OFF EXACTLY.

    Before the seed-once + LRayPass-vignetting fixes this was wrong by 21%
    after one leg and 38% after two.
    """
    scal = run_case(RX_CHAIN, 'scalar', elt)
    pol = run_case(RX_CHAIN, 'polsc', elt)
    assert np.array_equal(pol, scal)


# ---- 3a.2 + 3a.1(2): vector chain == scalar chain, any input state -------
@pytest.mark.parametrize('elt', [LEG1, LEG2])
@pytest.mark.parametrize('state', ['vec_x', 'vec_45', 'vec_circ'])
def test_vector_equals_scalar_every_state(elt, state):
    scal = run_case(RX_CHAIN, 'scalar', elt)
    vec = run_case(RX_CHAIN, state, elt)
    # Ex=Ey=1 carries twice the flux; compare normalized shapes.
    r = relerr(vec / vec.sum(), scal / scal.sum())
    assert r < TOL, f'{state} at elt {elt}: rel={r:.3e}'


# ---- validation ladder 1: energy conservation per leg --------------------
@pytest.mark.parametrize('elt', [LEG1, LEG2])
def test_energy_conserved_per_leg(elt):
    scal = run_case(RX_CHAIN, 'scalar', elt)
    vec = run_case(RX_CHAIN, 'vec_x', elt)
    assert vec.sum() == pytest.approx(scal.sum(), rel=1e-14)


# ---- 3a.2: the mask really is on the vector path -------------------------
@pytest.mark.parametrize('state', ['vec_x', 'vec_45', 'vec_circ'])
def test_mask_throughput_identical_on_vector_path(state):
    """MidStop's central obscuration must cost the vector run the same
    fraction of the power it costs the scalar run.  If the ray-side masking
    were still single-plane, the stale Ey/Ez planes would keep their share
    of the blocked power.  Compares throughput rather than the shadow
    itself: at Fresnel number 25 the centre of a 1 mm obscuration is an
    Arago BRIGHT spot, not a null.
    """
    ts = [run_case(RX_CHAIN, 'scalar', e).sum() for e in (LEG1, LEG2)]
    assert ts[1] < 0.99 * ts[0], 'fixture broken: obscuration removes no power'
    tv = [run_case(RX_CHAIN, state, e).sum() for e in (LEG1, LEG2)]
    assert tv[1] / tv[0] == pytest.approx(ts[1] / ts[0], rel=1e-14)


# ---- validation ladder 5: far-field normalization A/B --------------------
def test_far_field_vector_matches_scalar_normalization():
    """PFFPROP applied only 1/N per component and skipped applyfac2; FFPROP
    applies 1/(i*lambda*dz)*dx1**2 plus the output quadratic phase.  With one
    shared kernel the vector total power must now equal the scalar total
    exactly (Parseval: the three component planes partition the scalar norm,
    and the extra factor is a common scale times a unimodular phase).

    A/B measured 2026-07-26 on this Rx at model 128:
        pre-fix   sum(vector) = 8.937660518e-01
        post-fix  sum(vector) = 1.815495281e+06 == sum(scalar)
    i.e. the vector far-field leg was low by 2.031e+06 in intensity
    (1.425e+03 in amplitude) and is now normalized identically to scalar.
    """
    scal = run_case(RX_FF, 'scalar', FF_DET)
    vec = run_case(RX_FF, 'vec_x', FF_DET)
    assert vec.sum() == pytest.approx(scal.sum(), rel=1e-12)
    # Sanity bound only: the vector run must neither collapse onto the scalar
    # map nor wander far from it.  The out-of-plane content is the SUSPECTED
    # source of the difference -- see the UNVERIFIED ATTRIBUTION note in the
    # module docstring.  These are empirical brackets on the observed 2.6e-3,
    # not a derived budget.
    r = relerr(vec, scal)
    assert 1e-4 < r < 1e-2, f'vector/scalar far-field difference {r:.3e}'
