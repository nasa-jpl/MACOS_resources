"""Gratings on CURVED surfaces are chord-ruled (macos elemsub.F, 2026-09-30).

A straight-ruled concave grating is cut by equidistant parallel planes (normal
s0 = unit(h1HOE) in the vertex plane, spacing d), so at a point with surface
normal N the local grating vector is (m*lambda/d) * (s0 - (s0.N) N) --
UN-normalised: its magnitude falls as the surface tilts.  The pre-fix engine
unitised the projection, a period constant along the SURFACE, which on a
Dyson/Offner grating is a spectral blur ~3 px rms at 2500 nm (dyson5 beat 2).

The gate is the vector grating equation per ray, written from the physics and
not from the engine: the tangential part of (nb*R - na*I) equals the local
grating vector.  Air fixture, so na = nb = 1.  The must-fail leg evaluates the
same equation with the surface model (|G| = m lambda/d everywhere): on this
100 mm radius / 20 mm aperture fixture the two differ by up to ~0.5% of the
kick, and the pre-fix engine satisfies the surface form instead.
"""
import numpy as np
import pytest
from context import pymacos as m
from context import rx_path

MODEL = 256
RX = 'Rx_GratingConcave_air.in'
VPT = np.array([0.0, 0.0, 0.0])        # set from the fixture in the module fixture
TOL = 1e-10


def _read_fixture_geometry():
    """Vertex, axis, curvature radius and rule direction of the grating element (iElt 2)."""
    txt = rx_path(RX).read_text()
    blk = txt.split('iElt=  2')[1].split('iElt=  3')[0]
    def vec(key):
        line = [l for l in blk.splitlines() if l.strip().startswith(key + '=')][0]
        return np.array([float(v) for v in line.split('=')[1].split()])
    kr = vec('KrElt')[0]
    return vec('VptElt'), vec('psiElt'), kr, vec('h1HOE'), float(vec('RuleWidth')[0]), int(vec('OrderHOE')[0])


@pytest.fixture(scope="module")
def rays():
    m.init(MODEL)
    m.load(str(rx_path(RX)))
    lam = 1.0e-6
    m.set_src_wvl(lam) if hasattr(m, 'set_src_wvl') else None
    n1 = m.trace_rays(1)[1]
    _, dir1, _, ok1, _ = m.getRayInfo(int(n1))
    n2 = m.trace_rays(2)[1]
    pos2, dir2, _, ok2, _ = m.getRayInfo(int(n2))
    ok = ok1 & ok2
    vpt, psi, kr, h1, d, order = _read_fixture_geometry()
    psi = psi / np.linalg.norm(psi)
    centre = vpt + abs(kr) * psi                  # KrElt = -|R|; psi points from the vertex toward the beam, the centre lies |R| along it
    I = dir1[:, ok]; P = pos2[:, ok]; R = dir2[:, ok]
    N = P - centre[:, None]; N /= np.linalg.norm(N, axis=0)
    s0 = h1 - (h1 @ psi) * psi; s0 /= np.linalg.norm(s0)
    G = order * lam / d
    return I, R, N, s0, G


def _residuals(I, R, N, s0, G, chord):
    res = np.zeros(I.shape[1])
    for k in range(I.shape[1]):
        n = N[:, k]
        sraw = s0 - (s0 @ n) * n
        s = sraw / np.linalg.norm(sraw)
        g = G * np.linalg.norm(sraw) if chord else G
        res[k] = (R[:, k] @ s) - (I[:, k] @ s) - g
    return res


def test_rays_survive(rays):
    I, *_ = rays
    assert I.shape[1] > 50


def test_vector_grating_equation_chord_model(rays):
    I, R, N, s0, G = rays
    assert np.max(np.abs(_residuals(I, R, N, s0, G, chord=True))) < TOL


def test_surface_model_is_the_must_fail_leg(rays):
    """The pre-fix engine satisfies THIS form; the curved fixture separates the two by ~0.5% of the kick."""
    I, R, N, s0, G = rays
    assert np.max(np.abs(_residuals(I, R, N, s0, G, chord=False))) > 100 * TOL


def test_groove_direction_component_is_conserved(rays):
    I, R, N, s0, G = rays
    for k in range(I.shape[1]):
        n = N[:, k]; sraw = s0 - (s0 @ n) * n; s = sraw / np.linalg.norm(sraw)
        u = np.cross(n, s)
        assert abs((R[:, k] @ u) - (I[:, k] @ u)) < TOL
