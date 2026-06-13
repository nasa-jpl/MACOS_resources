"""Tests for m.compose() -- multi-wavelength PSF assembly on a fixed
pixel grid (MACOS 'COMPOSE' + 'ADD').

compose() initialises a composite accumulator at an image-plane element,
then for each wavelength propagates and accumulates the intensity onto a
single npix x npix detector grid.  Because the per-wavelength PSFs are
resampled onto the SAME fixed grid and summed incoherently, the result
is exactly linear in the wavelength list -- which is the strong
regression invariant pinned here:

    compose([a, b]) == compose([a]) + compose([b])      (to machine eps)
    compose([a, a]) == 2 * compose([a])
"""
from pathlib import Path

import numpy as np
import pytest

import context  # noqa: F401  -- adds ../src to sys.path
from pymacos import macos as m


RX = str(Path(__file__).resolve().parent / 'Rx' / 'Rx_Cass_FarField.in')
DET = 6          # FocalPlane
MODEL = 128
NPIX = 64


@pytest.fixture()
def loaded():
    m.init(MODEL)
    m.load(RX)
    m.intensity(DET)             # establish a propagation -> dxElt(DET)
    w0 = float(m.src_wvl())
    dx = m.dx_at(DET)            # SI metres
    return w0, dx


def test_compose_shape_and_positive(loaded):
    w0, dx = loaded
    I = m.compose(DET, [w0], npix=NPIX, dx=dx)
    assert I.shape == (NPIX, NPIX)
    assert np.all(np.isfinite(I))
    assert I.min() >= 0.0
    assert I.max() > 0.0


def test_compose_is_linear_in_wavelength_list(loaded):
    w0, dx = loaded
    I1  = m.compose(DET, [w0],          npix=NPIX, dx=dx)
    I2  = m.compose(DET, [w0 * 1.02],   npix=NPIX, dx=dx)
    I12 = m.compose(DET, [w0, w0*1.02], npix=NPIX, dx=dx)
    rel = np.max(np.abs(I12 - (I1 + I2))) / max(I12.max(), 1e-300)
    assert rel < 1e-12, f"compose linearity residual {rel:.3e}"


def test_compose_repeated_wavelength_scales(loaded):
    w0, dx = loaded
    I1  = m.compose(DET, [w0],     npix=NPIX, dx=dx)
    Idd = m.compose(DET, [w0, w0], npix=NPIX, dx=dx)
    rel = np.max(np.abs(Idd - 2.0 * I1)) / max(Idd.max(), 1e-300)
    assert rel < 1e-12, f"compose doubling residual {rel:.3e}"


def test_compose_dx_units_consistent(loaded):
    # 'm' and 'mm' must address the same physical pixel size.
    w0, dx = loaded
    Im  = m.compose(DET, [w0], npix=NPIX, dx=dx,       dx_unit='m')
    Imm = m.compose(DET, [w0], npix=NPIX, dx=dx * 1e3, dx_unit='mm')
    assert np.allclose(Im, Imm, rtol=0, atol=1e-9 * max(Im.max(), 1.0))


def test_src_flux_roundtrip(loaded):
    f0 = float(m.src_flux())
    m.src_flux(0.25)
    assert abs(float(m.src_flux()) - 0.25) < 1e-12
    m.src_flux(f0)
    assert abs(float(m.src_flux()) - f0) <= 1e-12 * max(abs(f0), 1.0) + 1e-15


def test_compose_rejects_bad_args(loaded):
    w0, dx = loaded
    with pytest.raises(Exception):
        m.compose(DET, [], npix=NPIX, dx=dx)              # empty list
    with pytest.raises(Exception):
        m.compose(DET, [w0], npix=NPIX, dx=dx, dx_unit='furlong')
    with pytest.raises(Exception):
        m.compose(DET, [-w0], npix=NPIX, dx=dx)           # negative wvl
