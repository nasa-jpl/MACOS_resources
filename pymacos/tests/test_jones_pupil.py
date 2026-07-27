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


# ---- 2b: Zernike expansion of the polarization-aberration maps ----------
def _zern(j, rho, th):
    """ANSI mode on caller polar coords -- independent of macos._ansi_zernike
    on purpose, so the test does not check the fit against its own basis."""
    import math
    jj = j - 1
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * jj)) / 2))
    mm = 2 * jj - n * (n + 2)
    am = abs(mm)
    R = np.zeros_like(rho)
    for s in range((n - am) // 2 + 1):
        c = ((-1) ** s * math.factorial(n - s)
             / (math.factorial(s) * math.factorial((n + am) // 2 - s)
                * math.factorial((n - am) // 2 - s)))
        R = R + c * rho ** (n - 2 * s)
    ang = np.cos(mm * th) if mm >= 0 else np.sin(am * th)
    P = (1, 2, 2, np.sqrt(6), np.sqrt(3), np.sqrt(6), np.sqrt(8), np.sqrt(8),
         np.sqrt(8), np.sqrt(8), np.sqrt(10), np.sqrt(10), np.sqrt(5),
         np.sqrt(10), np.sqrt(10))
    return P[j - 1] * R * ang


def test_pol_zernike_synthetic_recovery():
    """Pure math: build the maps FROM known coefficients, get them back.

    On an ANNULUS on purpose -- circular Zernikes are not orthogonal there,
    so this also pins that the fit is least-squares and not a projection.
    """
    N = 64
    ii, jj = np.indices((N, N)).astype(float)
    c0 = (N - 1) / 2
    rr = np.hypot(ii - c0, jj - c0)
    rad = 0.45 * N
    mask = (rr <= rad) & (rr >= 0.25 * rad)
    rho, th = rr / rad, np.arctan2(jj - c0, ii - c0)
    modes = [1, 4, 5, 6, 9, 13]
    B = np.stack([_zern(j, rho, th) for j in modes], axis=-1)
    ctrue = np.array([[1.0, -0.3, 0.7], [-2.0, 0.5, 0.0], [0.4, 1.1, -0.6],
                      [3.0, -0.2, 0.9], [-0.7, 0.8, 0.1], [0.2, -1.4, 0.3]])
    Dv = np.einsum('ijk,kc->ijc', B, ctrue)
    Dv[~mask] = np.nan
    pm = {'Dvec': Dv, 'retvec': 2 * Dv,
          'D': np.sqrt((Dv ** 2).sum(-1)), 'ret': 2 * np.sqrt((Dv ** 2).sum(-1)),
          'mask': mask}
    pz = m.pol_zernike(pm, modes=modes, center=(c0, c0), radius=rad)
    assert np.allclose(pz['D'], ctrue, atol=1e-10)
    assert np.allclose(pz['ret'], 2 * ctrue, atol=1e-10)
    assert pz['names'][modes.index(6)] == 'astig0'
    assert tuple(pz['nm'][modes.index(6)]) == (2, 2)
    assert tuple(pz['nm'][modes.index(4)]) == (2, -2)


def test_pol_zernike_two_mirror_form(loaded):
    """The published two-mirror form: polarization ASTIGMATISM and nothing else.

    Standard polarization-aberration theory for an on-axis rotationally
    symmetric two-mirror system gives diattenuation and retardance growing
    as rho**2 with a 2*theta azimuth -- which in the Pauli representation is
    exactly astig0 in s1 and astig45 in s2, equal magnitude, no circular
    component, no defocus.

    Tolerances: the astig pair matches to 1.9e-7 at model 128 and 5.8e-8 at
    256, i.e. a pupil-DISCRETIZATION asymmetry that shrinks with sampling,
    not physics (it is the same value for D and for retardance).
    """
    for e in (PRIM, SEC):
        m.coating(e, index=N_AL, extinc=K_AL, thickness=THK_AL)
    pz = m.pol_zernike(m.pol_maps(m.jones_pupil(DET)))
    mo = list(pz['modes'])
    iA0, iA45 = mo.index(6), mo.index(4)
    i2A0, i2A45 = mo.index(14), mo.index(12)
    for key in ('D', 'ret'):
        C = pz[key]
        a0, a45 = abs(C[iA0, 0]), abs(C[iA45, 1])
        assert a0 > 0
        assert abs(a0 - a45) / a0 < 1e-6, f'{key}: astig pair unequal'
        keep = [k for k in range(len(mo)) if k not in (iA0, iA45, i2A0, i2A45)]
        assert np.abs(C[keep, :2]).max() / a0 < 1e-10, \
            f'{key}: only astigmatism (+ its rho^4 companion) may appear'
        assert abs(C[i2A0, 0]) / a0 < 1e-2, f'{key}: companion not sub-dominant'
        assert np.abs(C[:, 2]).max() / a0 < 1e-10, f'{key}: circular present'
    # radial law: |D| is rotationally symmetric (piston + defocus), and its
    # on-axis extrapolation vanishes -- the fit is never told to arrange that
    cm = pz['Dmag']
    keep = [k for k in range(len(mo)) if mo[k] not in (1, 5, 13)]
    assert np.abs(cm[keep]).max() / abs(cm[mo.index(1)]) < 1e-6
    D0 = sum(cm[k] * _zern(mo[k], 0.0, 0.0) for k in range(len(mo)))
    D1 = sum(cm[k] * _zern(mo[k], 1.0, 0.0) for k in range(len(mo)))
    assert abs(D0) / abs(D1) < 1e-3, 'on-axis diattenuation must vanish'


def test_pol_zernike_basis_dependence(loaded):
    """D's expansion is basis-invariant; retardance's is not."""
    m.coating(SEC, index=N_AL, extinc=K_AL, thickness=THK_AL)
    pzd = m.pol_zernike(m.pol_maps(m.jones_pupil(DET, basis='double-pole')))
    pzs = m.pol_zernike(m.pol_maps(m.jones_pupil(DET, basis='local-sp')))
    scale = np.abs(pzd['D']).max()
    assert np.abs(pzd['D'] - pzs['D']).max() / scale < 1e-9
    assert np.abs(pzs['ret']).max() > 10 * np.abs(pzd['ret']).max()
