"""srt() -- one ray from a normalized source-aperture point (the CLI's SRT through the
api's srt_trace; BRIEF_to_bindings B.2, 2026-10-17); twin of mmacos tSrt.

Deck: the manual's Cassegrain (primary R = 10.8, K = -1 at the origin; detector at z = 1.5).
The CLI numbers are what `srt 5` printed there (cli_tests/cmd/SRAY_SRT.jou's deck; both
compilers, 2026-10-17): the binding must reproduce its own CLI.  The analytic legs are written
from the geometry, not read from the engine: the primary is a paraboloid, so a ray at radius r
meets it at z = -r^2 / (2R); an on-axis Cassegrain focuses every ray on the detector axis.
"""
from pathlib import Path
import os

import numpy as np
import pytest
from context import pymacos as m

DECK = (Path(os.environ.get('MACOS_HOME', str(Path.home() / 'dev' / 'macos' / 'macos_f90')))
        .parent / 'docs' / 'macos-manual' / 'examples' / 'Cassegrain.in')
R_PRIMARY = 10.8
# CLI `srt 5` at (0.3, 0.2): the crossings at elements 2, 3, 5 and the direction leaving 2, 3
CLI_POS = {2: (-6.0e-01, -4.0e-01, -2.40740740740737e-02),
           3: (-1.48921405137844e-01, -9.92809367585626e-02, -4.06568259532358e+00),
           5: (-3.20217463656292e-11, -2.13479789401561e-11, 1.5)}
CLI_DIR = {2: (1.10617958347559e-01, 7.37453055650393e-02, -9.91123250256060e-01),
           3: (2.67432537941554e-02, 1.78288358627703e-02, 9.99483332021240e-01)}


@pytest.fixture
def cass():
    m.init(128)
    m.load(str(DECK))
    return m.num_elt()


def test_srt_reproduces_the_cli(cass):
    r = m.srt(0.3, 0.2)
    assert r['ok'] and r['status'] == 0 and r['fail_elt'] == 0
    for k, p in CLI_POS.items():
        assert np.allclose(r['pos'][:, k - 1], p, rtol=1e-12, atol=1e-15), k
    for k, d in CLI_DIR.items():
        assert np.allclose(r['dir'][:, k - 1], d, rtol=1e-12, atol=1e-15), k
    assert np.array_equal(r['n_prev'], np.ones(cass)) and np.array_equal(r['n_cur'], np.ones(cass))


@pytest.mark.parametrize('xy', [(0.3, 0.2), (0.5, 0.0), (-0.1, 0.45)])
def test_srt_geometry_paraboloid_sag_and_axial_focus(cass, xy):
    r = m.srt(*xy)
    x, y, z = r['pos'][:, 1]                              # the primary, element 2
    assert np.isclose(z, -(x * x + y * y) / (2 * R_PRIMARY), rtol=1e-12, atol=1e-15)
    assert np.hypot(*r['pos'][:2, cass - 1]) < 1e-9      # the focus is on axis
    assert np.isclose(r['pos'][2, cass - 1], 1.5, rtol=0, atol=1e-12)
    assert np.allclose(np.linalg.norm(r['dir'], axis=0), 1.0, rtol=0, atol=1e-12)


def test_srt_obscured_ray_is_traced_and_flagged(cass):
    # (0.5, 0) and the chief both meet the secondary's obscuration (element 1) -- the CLI
    # prints "Ray 2 is obscured by element 1" and still traces it to the detector
    for xy in [(0.5, 0.0), (0.0, 0.0)]:
        r = m.srt(*xy)
        assert r['ok'] and r['status'] == 1 and r['fail_elt'] == 1
        assert np.isclose(r['pos'][2, cass - 1], 1.5, rtol=0, atol=1e-12)


def test_srt_end_element_and_refusal(cass):
    r = m.srt(0.3, 0.2, 3)
    assert r['pos'][:, :3].any(axis=0).all() and not r['pos'][:, 3:].any()
    with pytest.raises(Exception):
        m.srt(0.6, 0.0)                                   # outside the aperture


def test_srt_leaves_the_full_trace_unchanged(cass):
    m.trace_rays(cass)
    w0, k0 = m.opd().copy(), m.opd_mask().copy()
    m.srt(0.3, 0.2)
    m.trace_rays(cass)
    assert np.array_equal(m.opd_mask(), k0)
    assert np.array_equal(m.opd()[k0], w0[k0])
