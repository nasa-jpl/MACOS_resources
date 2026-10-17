"""segray() -- the ray through a Segment element's centre (the CLI's SEGRAYTRACE through the
api's segray_trace; BRIEF_to_bindings B.2, 2026-10-17); twin of mmacos tSegRay.

Deck: e5hex1 (seven Segment elements 1-7, then m2 = element 8, a collimated source).  The CLI
numbers are what `segraytrace` printed for segments 3 and 7 (cli_tests/cmd/SEGRAYTRACE.jou,
both compilers, 2026-10-17): the binding must reproduce its own CLI.  The analytic leg is the
command's definition, written from the deck's own lines (ChfRayDir=, RptElt=), not from the
engine: for a collimated source the ray is parallel to the chief ray and passes through the
segment's reference point, so its crossing at the segment lies on that line.
"""
import re

import numpy as np
import pytest
from context import pymacos as m
from context import rx_path

DECK = rx_path('e5hex1.in')
# CLI `segraytrace 3`, `7`: the crossing at the segment and at m2 (element 8), the direction
# leaving the segment
CLI = {3: {'seg': (-1.33333333333333e+03, -2.29984379552443e+03, 2.20877392583239e+02),
           'm2': (-1.94630316570224e+02, -5.81602731534171e+03, -2.12818888039439e+04),
           'dir': (5.21907694203786e-02, -1.61159073631132e-01, -9.85547500922033e-01)},
       7: {'seg': (2.66666666666667e+03, -8.65286482234703e+00, -6.90022891415174e+01),
           'm2': (3.90032347074229e+02, -5.47980841189830e+03, -2.13166480632806e+04),
           'dir': (-1.03208762760953e-01, -2.48028939046969e-01, -9.63239117085985e-01)}}


def _deck_vectors():
    txt = open(DECK).read()
    chf = np.array(re.search(r'ChfRayDir=\s*(\S+)\s+(\S+)\s+(\S+)', txt).groups(), float)
    rpt = [np.array(g, float) for g in re.findall(r'RptElt=\s*(\S+)\s+(\S+)\s+(\S+)', txt)]
    return chf / np.linalg.norm(chf), rpt


@pytest.fixture
def e5():
    m.init(128)
    m.load(str(DECK))
    return m.num_elt()


@pytest.mark.parametrize('seg', [3, 7])
def test_segray_reproduces_the_cli(e5, seg):
    r = m.segray(seg)
    assert r['ok'] and r['status'] == 0 and r['fail_elt'] == 0
    c = CLI[seg]
    assert np.allclose(r['pos'][:, seg - 1], c['seg'], rtol=1e-13, atol=1e-12)
    assert np.allclose(r['pos'][:, 7], c['m2'], rtol=1e-13, atol=1e-12)
    assert np.allclose(r['dir'][:, seg - 1], c['dir'], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize('seg', range(1, 8))
def test_segray_crosses_on_the_chief_parallel_through_the_centre(e5, seg):
    d, rpt = _deck_vectors()
    r = m.segray(seg)
    assert r['ok'] and r['status'] == 0
    p = r['pos'][:, seg - 1]
    off = p - rpt[seg - 1]
    assert np.linalg.norm(np.cross(off, d)) < 1e-8          # on the line, to ~1e-12 relative
    assert np.allclose(np.linalg.norm(r['dir'], axis=0), 1.0, rtol=0, atol=1e-12)


def test_segray_refuses_a_non_segment(e5):
    assert m.segray(3)['ok']                                 # the call exists and works
    for bad in (8, 0, e5 + 1):                                # m2 is a Reflector
        with pytest.raises(Exception):
            m.segray(bad)


def test_segray_leaves_the_full_trace_unchanged(e5):
    m.trace_rays(e5)
    w0, k0 = m.opd().copy(), m.opd_mask().copy()
    m.segray(3)
    m.trace_rays(e5)
    assert np.array_equal(m.opd_mask(), k0)
    assert np.array_equal(m.opd()[k0], w0[k0])
