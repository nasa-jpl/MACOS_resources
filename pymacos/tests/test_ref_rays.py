"""ref_rays() -- the reference rays of the source grid (the CLI's RRAY / PRAY through the api's
ref_rays; BRIEF_to_bindings B.2, 2026-10-17); twin of mmacos tRefRays.

Deck: Rx_Cass_FarField (circular aperture 4 m, collimated along +z, xGrid = -x, yGrid = -y), model
128.  The CLI numbers are what `rray` / `pray` printed (both compilers, 2026-10-17, the fixed
build): the binding must reproduce its own CLI.  The geometric leg is the commands' definition,
from the deck's own lines: RRAY's rays sit at the aperture edge (radius Aperture/2) on the +x, +y,
-x, -y axes of the grid frame, PRAY's on a ring at 0.9 of it every 45 degrees from +x -- each the
nearest grid ray that passes, so within two grid spacings (4 m / 126 = 0.032 m) of its point.
PRAY's first point read 9.0 before the fix (ray 2 was RRAY's edge ray, 0.17 m off the ring).
Each ray's offset is tied to its index through ray(): its crossing at element 1 less the chief
ray's, projected on xGrid / yGrid.
"""
import re

import numpy as np
import pytest
from context import pymacos as m
from context import rx_path

DECK = rx_path('Rx_Cass_FarField.in')
# CLI `rray`, `pray` (Ref. ray XY offsets ... ID = ; printed to 4 decimals)
CLI = {'rray': ([1, 6416, 12443, 6292, 15],
                [(0, 0), (1.9683, 0.0317), (0.0317, 1.9683), (-1.9683, 0.0317),
                 (0.0317, -1.9683)]),
       'pray': ([1, 6411, 10941, 12241, 10861, 6297, 1515, 217, 1595],
                [(0, 0), (1.8095, 0.0317), (1.2698, 1.2698), (0.0317, 1.8095),
                 (-1.2698, 1.2698), (-1.8095, 0.0317), (-1.2698, -1.2698),
                 (0.0317, -1.8095), (1.2698, -1.2698)])}
TWO_SPACINGS = 0.065


def _deck():
    txt = open(DECK).read()
    ap = float(re.search(r'Aperture=\s*(\S+)', txt).group(1))
    xg = np.array(re.search(r'xGrid=\s*(\S+)\s+(\S+)\s+(\S+)', txt).groups(), float)
    yg = np.array(re.search(r'yGrid=\s*(\S+)\s+(\S+)\s+(\S+)', txt).groups(), float)
    return ap / 2, xg, yg


@pytest.fixture
def cass():
    m.init(128)
    m.load(str(DECK))


@pytest.mark.parametrize('kind', ['rray', 'pray'])
def test_ref_rays_reproduce_the_cli(cass, kind):
    r = m.ref_rays(kind)
    ids, off = CLI[kind]
    assert list(r['id']) == ids
    assert np.allclose(r['offset'], np.array(off, float).T, atol=5e-5, rtol=0)


@pytest.mark.parametrize('kind', ['rray', 'pray'])
def test_ref_rays_sit_on_their_points(cass, kind):
    rad, _, _ = _deck()
    if kind == 'rray':
        pts = rad * np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], float)
    else:
        t = np.deg2rad(45.0 * np.arange(8))
        pts = 0.9 * rad * np.c_[np.cos(t), np.sin(t)]
    r = m.ref_rays(kind)
    assert np.all(r['id'] > 0)
    miss = np.linalg.norm(r['offset'][:, 1:].T - pts, axis=1)
    assert np.all(miss < TWO_SPACINGS), miss


def test_pray_ring_is_symmetric(cass):
    o = m.ref_rays('pray')['offset'][:, 1:]
    # +x / -x mirror in x, +y / -y in y; the opposite diagonals in both
    for a, b, ks in ((0, 4, (0,)), (2, 6, (1,)), (1, 5, (0, 1)), (3, 7, (0, 1))):
        for k in ks:
            assert o[k, a] == pytest.approx(-o[k, b], abs=1e-12)


@pytest.mark.parametrize('kind', ['rray', 'pray'])
def test_ref_rays_offset_is_the_rays_own(cass, kind):
    _, xg, yg = _deck()
    r = m.ref_rays(kind)
    p0 = m.ray(1)['pos'][:, 0]
    for i, o in zip(r['id'][1:], r['offset'][:, 1:].T):
        q = m.ray(int(i))
        assert q['ok']
        d = q['pos'][:, 0] - p0
        assert np.allclose([d @ xg, d @ yg], o, atol=1e-9, rtol=0)


def test_rray_clears_the_slots_it_does_not_fill(cass):
    from pymacos import api_gen as g
    m.ref_rays('pray')
    ids, off = g.ref_rays(1, np.zeros(9, dtype=np.int32), np.zeros((2, 9), order='F'))
    assert list(ids[:5]) == CLI['rray'][0]
    assert list(ids[5:]) == [0, 0, 0, 0] and not np.any(off[:, 5:])


def test_ref_rays_refuses_a_bad_kind(cass):
    assert list(m.ref_rays('rray')['id']) == CLI['rray'][0]
    with pytest.raises(ValueError):
        m.ref_rays('tpr')
    from pymacos import api_gen as g
    for k in (0, 3):
        with pytest.raises(g.ApiError):
            g.ref_rays(k, np.zeros(9, dtype=np.int32), np.zeros((2, 9), order='F'))


def test_ref_rays_leave_the_full_trace_unchanged(cass):
    ne = m.num_elt()
    m.trace_rays(ne)
    w0, k0 = m.opd().copy(), m.opd_mask().copy()
    m.ref_rays('pray')
    m.ref_rays('rray')
    # read WITHOUT a re-trace: the source grid RRAY / PRAY rebuild left nGridPts one short,
    # so opd() / opd_mask() returned a crop of the last map
    assert m.opd_mask().shape == k0.shape and np.array_equal(m.opd_mask(), k0)
    assert np.array_equal(m.opd()[k0], w0[k0])
    m.trace_rays(ne)
    assert np.array_equal(m.opd_mask(), k0)
    assert np.array_equal(m.opd()[k0], w0[k0])


def test_ref_rays_leave_ray_printing_off(cass, capfd):
    # FSR (RRAY / PRAY / FSR) left ifRayPrint .TRUE.: the next full trace printed every ray's
    # segments (tens of thousands of lines on this deck)
    ne = m.num_elt()
    m.ref_rays('pray')
    capfd.readouterr()
    m.trace_rays(ne)
    out = capfd.readouterr().out
    assert 'segment from Element' not in out
