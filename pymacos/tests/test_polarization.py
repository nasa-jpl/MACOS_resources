"""Tests for the Phase-1 polarization exposure (PLAN_POLARIZATION).

Covers the engine polarization physics newly reachable from pymacos:
  m.polarization()      -- pol_set / pol_get (on/off + source Jones state)
  m.vector_diffraction()-- vecdif_set (VECTOR/SCALAR + ordering guard)
  m.coating()           -- coat_set / coat_get (Model A, physical thickness)
  m.ray_field()         -- rayfield_get (RayE + geometry + status)

These are the state / round-trip / geometry gates; the Jones-pupil physics
lands in Phase 2.  init() is one-shot per process (module globals), so a
single MODEL size is used throughout.
"""
from pathlib import Path

import numpy as np
import pytest

import context  # noqa: F401  -- adds ../src to sys.path
from pymacos import macos as m


RX = str(Path(__file__).resolve().parent / 'Rx' / 'Rx_Cass_FarField.in')
DET = 6          # FocalPlane
FOLD = 3         # a reflector to coat
MODEL = 128      # mWF=3 -> vector diffraction available


@pytest.fixture()
def loaded():
    m.init(MODEL)
    m.load(RX)
    yield
    # leave polarization off so other tests in the process are unaffected
    try:
        m.polarization('off')
    except Exception:
        pass


# ---- polarization on/off + source state round-trip -------------------
def test_pol_on_off_roundtrip(loaded):
    m.polarization('on', Ex=(1.0, 0.0), Ey=(0.0, 0.0))
    s = m.polarization()
    assert s['on'] is True
    assert s['Ex'] == pytest.approx(complex(1, 0))
    assert s['Ey'] == pytest.approx(complex(0, 0))

    m.polarization('off')
    assert m.polarization()['on'] is False


def test_pol_circular_state(loaded):
    m.polarization('on', Ex=1 + 0j, Ey=0 + 1j)
    s = m.polarization()
    assert s['Ey'] == pytest.approx(complex(0, 1))


def test_pol_enables_vector(loaded):
    m.polarization('on')
    assert m.polarization()['vector'] is True


# ---- vector / scalar toggle + ordering guard -------------------------
def test_vecdif_toggle(loaded):
    m.polarization('on')
    m.vector_diffraction(False)
    assert m.polarization()['vector'] is False
    m.vector_diffraction(True)
    assert m.polarization()['vector'] is True


def test_vector_requires_polarization(loaded):
    m.polarization('off')
    with pytest.raises(Exception):
        m.vector_diffraction(True)


# ---- coating set/get round-trip (Model A) ----------------------------
def test_coat_roundtrip_identity(loaded):
    n = np.array([1.38, 2.30])
    k = np.array([0.0, 0.10])
    t = np.array([1.0e-7, 5.0e-8])   # physical thickness, BaseUnits = m
    m.coating(FOLD, index=n, extinc=k, thickness=t)
    s = m.coating(FOLD)
    assert s['n_layer'] == 2
    assert s['index'] == pytest.approx(n)
    assert s['extinc'] == pytest.approx(k)
    assert s['thickness'] == pytest.approx(t, rel=1e-9)


def test_coat_query_uncoated(loaded):
    s = m.coating(DET)               # focal plane, no coating
    assert s['n_layer'] == 0


def test_coat_bad_count_errors(loaded):
    with pytest.raises(Exception):
        m.coating(FOLD, index=np.zeros(20), extinc=np.zeros(20),
                  thickness=np.zeros(20))


# ---- ray_field structure + status mask -------------------------------
def test_ray_field_shape_and_status(loaded):
    m.polarization('on', Ex=(1.0, 0.0), Ey=(0.0, 0.0))
    m.trace_rays(DET)
    rf = m.ray_field(DET)
    assert rf['E'].shape == (MODEL, MODEL, 3)
    assert rf['k'].shape == (MODEL, MODEL, 3)
    assert rf['status'].shape == (MODEL, MODEL)
    ok = rf['status'] == 0
    assert np.count_nonzero(ok) > 0
    # OK rays carry a non-zero Ex for an x-polarized source
    assert np.max(np.abs(rf['E'][..., 0][ok])) > 0.0


# ---- physics: a coating changes the polarized throughput -------------
def test_coating_changes_polarized_intensity(loaded):
    m.polarization('on', Ex=(1.0, 0.0), Ey=(0.0, 0.0))
    I0 = m.intensity(DET)
    m.coating(FOLD, index=1.2, extinc=7.0, thickness=0.1)   # absorbing metal
    I1 = m.intensity(DET)
    rel = np.linalg.norm(I1 - I0) / max(np.linalg.norm(I0), np.finfo(float).eps)
    assert rel > 1e-6
