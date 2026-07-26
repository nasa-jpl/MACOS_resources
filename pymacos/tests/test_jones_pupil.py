"""Phase-2 Jones-pupil physics gates (PLAN_POLARIZATION 2a/2b), pymacos side.

Mirrors the mmacos tJonesPupil gates that run on Rx_Cass_FarField:

  * unitarity      -- stock Cass mirrors carry the perfect-conductor idiom
                      (IndRef=1, Extinc=1e22): RP=RS=-1, so the Jones pupil
                      is unitary-to-a-scalar (D, ret, T-nonuniformity ~ 0).
  * basis          -- D is a singular-value invariant (identical between
                      'double-pole' and 'local-sp'); retardance is NOT (the
                      s/p coordinate singularity is the artifact the
                      double-pole basis kills).
  * 2-theta        -- Al on both mirrors: on-axis rotationally-symmetric
                      system; the diattenuation axis locks to the pupil
                      azimuth, no circular component, D grows with radius.
  * synthetic      -- pol_maps recovers a constructed diattenuator*retarder
                      exactly; ambiguity flag fires near delta=pi.

The Bench-emitted flat-fold Fresnel gate is mmacos-only (the Bench builder
is MATLAB); the coated physics itself is engine-shared, pinned there.
"""
from pathlib import Path

import numpy as np
import pytest

import context  # noqa: F401  -- adds ../src to sys.path
from pymacos import macos as m


RX = str(Path(__file__).resolve().parent / 'Rx' / 'Rx_Cass_FarField.in')
DET = 6
PRIM = 2
SEC = 3
MODEL = 128
# Al at 632.8 nm; thickness far beyond the skin depth (Rx BaseUnits = m)
N_AL, K_AL, THK_AL = 1.45, 7.54, 2.0e-7


@pytest.fixture()
def loaded():
    m.init(MODEL)
    m.load(RX)   # load_rx clears any coating state from a prior test
    yield
    try:
        m.polarization('off')
    except Exception:
        pass


def test_unitarity_gate(loaded):
    jp = m.jones_pupil(DET)
    pm = m.pol_maps(jp)
    msk = jp['mask']
    assert np.count_nonzero(msk) > 1000
    assert np.nanmax(pm['D'][msk]) < 1e-12
    assert np.nanmax(pm['ret'][msk]) < 1e-12
    assert np.nanstd(pm['T'][msk]) / np.nanmean(pm['T'][msk]) < 1e-12
    assert jp['leak'] < 1e-12


def test_restores_pol_state(loaded):
    m.polarization('on', Ex=0.6 + 0j, Ey=0.8j)
    m.jones_pupil(DET)
    s = m.polarization()
    assert s['on'] is True
    assert s['Ex'] == pytest.approx(0.6 + 0j)
    assert s['Ey'] == pytest.approx(0.8j)


def test_basis_invariance_and_sp_artifact(loaded):
    m.coating(SEC, index=N_AL, extinc=K_AL, thickness=THK_AL)
    pm_dp = m.pol_maps(m.jones_pupil(DET, basis='double-pole'))
    pm_sp = m.pol_maps(m.jones_pupil(DET, basis='local-sp'))
    msk = pm_dp['mask'] & pm_sp['mask']
    assert np.nanmax(np.abs(pm_dp['D'][msk] - pm_sp['D'][msk])) < 1e-12
    assert (np.nanmax(np.abs(pm_dp['T'][msk] - pm_sp['T'][msk]))
            / np.nanmean(pm_dp['T'][msk])) < 1e-12
    # coordinate singularity inflates s/p retardance variation
    assert pm_sp['var_rms']['ret'] > 10 * pm_dp['var_rms']['ret']


def test_2theta_symmetry(loaded):
    m.coating(PRIM, index=N_AL, extinc=K_AL, thickness=THK_AL)
    m.coating(SEC, index=N_AL, extinc=K_AL, thickness=THK_AL)
    pm = m.pol_maps(m.jones_pupil(DET))
    msk = pm['mask']
    ii, jj = np.nonzero(msk)
    N = msk.shape[0]
    II, JJ = np.mgrid[1:N + 1, 1:N + 1]
    R = np.hypot(II - ii.mean() - 1, JJ - jj.mean() - 1)
    TH = np.arctan2(JJ - jj.mean() - 1, II - ii.mean() - 1)
    rmax = R[msk].max()
    ring = msk & (R > 0.60 * rmax) & (R < 0.75 * rmax)
    inner = msk & (R > 0.25 * rmax) & (R < 0.40 * rmax)

    D1, D2, D3 = (pm['Dvec'][..., i] for i in range(3))
    ang = 0.5 * np.arctan2(D2[ring], D1[ring])
    off = 0.5 * np.angle(np.mean(np.exp(2j * (ang - TH[ring]))))
    resid = np.mod(ang - TH[ring] - off + np.pi / 2, np.pi) - np.pi / 2
    assert np.max(np.abs(resid)) < 1e-10          # axis locks to azimuth
    assert np.nanmax(np.abs(D3[msk])) < 1e-10 * np.nanmax(pm['D'][msk])
    assert np.nanmean(pm['D'][ring]) > 2 * np.nanmean(pm['D'][inner])


def test_pol_maps_synthetic_identity():
    D0, dl = 0.3, 0.8
    Jd = np.diag([np.sqrt(1 + D0), np.sqrt(1 - D0)])
    Jr = np.array([[np.cos(dl / 2), -1j * np.sin(dl / 2)],
                   [-1j * np.sin(dl / 2), np.cos(dl / 2)]])
    Jpt = Jd @ Jr
    J = np.broadcast_to(Jpt, (2, 2, 2, 2)).copy()
    pm = m.pol_maps({'J': J, 'mask': np.ones((2, 2), dtype=bool)})
    assert pm['D'][0, 0] == pytest.approx(D0, abs=1e-12)
    assert pm['ret'][0, 0] == pytest.approx(dl, abs=1e-12)
    assert pm['retvec'][0, 0] == pytest.approx([0, dl, 0], abs=1e-12)
    assert not pm['ambiguous'][0, 0]

    Jr2 = np.array([[np.cos(1.6), -1j * np.sin(1.6)],
                    [-1j * np.sin(1.6), np.cos(1.6)]])   # delta = 3.2
    J2 = np.broadcast_to(Jr2, (2, 2, 2, 2)).copy()
    pm2 = m.pol_maps({'J': J2, 'mask': np.ones((2, 2), dtype=bool)})
    assert pm2['ambiguous'][0, 0]
