"""A GridData figure on a NON-SEQUENTIAL refracting surface (PLAN_CONSOLIDATION 1.2, macos
2026-10-11).  NSRefractor handed GridSrf a local NULL grid frame, so every ray read the grid's
CENTRE pixel and the figure imaged as a piston.  Twins = macos ZGD tst_nsref_{seq,ns}.in (here
Rx_NSRefGrid{Seq,NS}.in + tst_nsref_grid.txt): a flat glass window whose front face carries one
2e-7 m bump at an off-centre pixel, Face1 a sequential Refractor vs a one-member NSRefractor
group.  The NS face must trace -- and propagate -- exactly like the sequential one.
Pre-fix: the NS twin's OPD is identically 0 (the centre pixel is 0) and its field differs.
GridFile= is a bare name resolved from the cwd: every leg runs from a staged copy.
"""
import shutil

import numpy as np
import pytest
from context import pymacos as m
from context import rx_path

FILES = ('Rx_NSRefGridSeq.in', 'Rx_NSRefGridNS.in', 'tst_nsref_grid.txt')


@pytest.fixture
def staged(tmp_path, monkeypatch):
    for f in FILES:
        shutil.copy(rx_path(f), tmp_path / f)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(deck):
    m.init(128)
    m.load(deck)
    m.trace_rays(4)
    w = np.asarray(m.opd()).copy()
    k = np.asarray(m.opd_mask()).astype(bool)
    f = np.asarray(m.complex_field(4)).copy()
    return w, k, f


def test_the_ns_face_traces_and_propagates_like_the_sequential_one(staged):
    ws, ks, fs = _run(FILES[0])
    wn, kn, fn = _run(FILES[1])
    assert np.array_equal(ks, kn)
    assert np.max(np.abs(ws[ks])) > 5e-8, 'non-vacuity: the sequential face shows the bump'
    assert np.max(np.abs(wn[kn] - ws[ks])) == 0.0
    assert np.max(np.abs(fn - fs)) == 0.0


def test_the_bump_is_local_not_a_piston(staged):
    wn, kn, _ = _run(FILES[1])
    dev = np.abs(wn[kn] - np.median(wn[kn]))
    assert 0 < np.count_nonzero(dev > 1e-9) < 0.1 * np.count_nonzero(kn), \
        'the figure touches a few pupil pixels (a piston would move them all, or none)'
