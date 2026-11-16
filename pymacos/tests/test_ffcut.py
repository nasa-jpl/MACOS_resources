"""The far-field evanescent cut (engine 2026-10-03, dyson5 addendum 15).

A far-field leg maps spatial frequency f to the output coordinate
x = lambda*dz*f, so |x| > dz is |sin theta| > 1: frequencies above
1/lambda that carry no propagating energy.  A pupil sampled finer than
lambda/2 makes the output window wider than 2*dz and those pixels EXIST;
an energy-fraction metric over the window then counts them.  ``ffcut(True)``
zeroes them.

Two legs, both load-bearing:
  * Rx_FarFieldPinhole.in -- a 20 um pinhole at 1 um on a 1 mm sphere,
    nGridpts 64 (dx1 0.32 um < lambda/2): the cut MUST zero pixels, leave
    every pixel inside |x| < dz bit-identical, and lower the total.
  * Rx_Cass_FarField.in -- dx1 >> lambda: the cut MUST be a no-op, zero
    pixels, bit-identical.  (A gate that only checked the pinhole could
    pass with a cut that zeroes everything.)
"""
import numpy as np
import pytest
from context import pymacos as m
from context import rx_path

MODEL = 256


def _run(name, on):
    m.init(MODEL)
    m.load(str(rx_path(name)))
    m.ffcut(on)
    I = np.asarray(m.intensity(m.num_elt())).copy()
    state, npix = m.ffcut()
    dx = m.dx_at(m.num_elt())
    m.ffcut(False)
    return I, state, npix, dx


@pytest.fixture(scope="module")
def pinhole():
    return _run('Rx_FarFieldPinhole.in', False), _run('Rx_FarFieldPinhole.in', True)


@pytest.fixture(scope="module")
def cass():
    return _run('Rx_Cass_FarField.in', False), _run('Rx_Cass_FarField.in', True)


def _mask_inside(n, dx, dz):
    # applyfac2 / FFEvanCut index convention: pixel n/2+1 (1-based) is x = 0
    i = np.arange(n) - (n // 2)
    X, Y = np.meshgrid(i * dx, i * dx, indexing='ij')
    return (X**2 + Y**2) <= dz**2


def test_pinhole_window_reaches_past_dz(pinhole):
    (I0, _, _, dx), _ = pinhole
    n = I0.shape[0]
    assert n * dx / 2 > 1.0e-3, f"the fixture must have an output window wider than dz (half-width {n*dx/2:.3e} m)"


def test_pinhole_cut_zeroes_only_the_evanescent_pixels(pinhole):
    (I0, s0, n0, dx), (I1, s1, n1, _) = pinhole
    assert s0 is False and s1 is True
    assert n0 == 0 and n1 > 0, (n0, n1)
    inside = _mask_inside(I0.shape[0], dx, 1.0e-3)
    assert np.array_equal(I1[inside], I0[inside]), "pixels with |x| <= dz must be bit-identical"
    assert np.all(I1[~inside] == 0.0), "pixels with |x| > dz must be zero"
    assert n1 == int(np.count_nonzero(~inside)), (n1, int(np.count_nonzero(~inside)))
    assert I0[~inside].sum() > 0.0, "the uncut deck must carry energy outside |x| = dz (else the leg is vacuous)"
    assert I1.sum() < I0.sum()


def test_cass_is_untouched(cass):
    (I0, _, n0, _), (I1, _, n1, _) = cass
    assert n0 == 0 and n1 == 0
    assert np.array_equal(I0, I1), "a pupil sampled coarser than lambda/2 has no evanescent pixels: bit-identical"
