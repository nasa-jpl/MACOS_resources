"""The IRIS save_rx -> reload SIGSEGV, closed (macos db9236b + the 2026-10-10 SAVE fix).

Twin of mmacos tZrnGrRoundTrip; fixtures = macos ZGD_test_files/tst_zrngr_*.in
(Rx_ZrnGrRoundTrip.in, Rx_ZrnGrOverflow.in, tst_zrngr_grid.txt): a Cassegrain
whose primary is a single-member NSReflector group with a flat 64x64 ZrnGrData
grid and NSCount= 1.  GridFile= is a bare name resolved from the cwd, so every
leg runs from a staged copy.

  * A: load -> trace -> save -> reload -> trace.  The save keeps NSCount= 1
    (pre-fix SAVE dropped it), writes no blank ZernType= (the deck declares
    none; the old writer's blank block made the reload refuse SILENTLY), and
    the reload passes the same rays with the same OPD.
  * B, in a SUBPROCESS: the overflow twin (GridSrfdx= 1e-10, xi ~ 1e10)
    traces to completion with the rejected grid samples COUNTED
    (grid_idx_ovf() > 0) and the grid inert (the same pass count).
  * the control: the finite deck rejects nothing.

Pre-fix: A fails (no NSCount= in the save; the reload refused silently).  B
cannot fail on x86_64 -- an out-of-range double -> INT gives INT_MIN there, so
the old guard already caught it; the crash is ARM-only (see tZrnGrRoundTrip).
The pre-fix .so also has no grid_idx_ovf binding.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from context import pymacos as m
from context import rx_path

HERE = Path(__file__).resolve().parent
MODEL, NPASS = 256, 32168
FILES = ('Rx_ZrnGrRoundTrip.in', 'Rx_ZrnGrOverflow.in', 'tst_zrngr_grid.txt')


@pytest.fixture
def staged(tmp_path, monkeypatch):
    for f in FILES:
        shutil.copy(rx_path(f), tmp_path / f)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _trace(deck):
    m.init(MODEL)
    m.load(str(deck))
    _, nrays, _ = m.trace_rays(m.num_elt())
    _, _, _, ok, passed = m.getRayInfo(int(nrays))
    npass = int(np.count_nonzero(np.asarray(ok)[1:] & np.asarray(passed)[1:]))
    return npass, np.asarray(m.opd()).copy(), m.opd_mask()


def test_save_keeps_nscount_and_the_reload_traces_the_same(staged):
    n0, w0, k0 = _trace(staged / FILES[0])
    assert n0 == NPASS
    m.save(str(staged / 'saved.in'))
    txt = (staged / 'saved.in').read_text()
    assert len(re.findall(r'^\s*NSCount=\s*1\s*$', txt, re.M)) == 1
    blk = txt[txt.index('EltName=  Primary'):txt.index('EltName=  Secondary')]
    assert 'ZernType' not in blk, 'no Zernike block for an element that declares none'
    n1, w1, k1 = _trace(staged / 'saved.in')
    assert n1 == n0
    assert np.array_equal(k1, k0)
    assert np.allclose(w1[k1], w0[k0], rtol=0, atol=1e-15)


def test_overflow_deck_completes_and_counts_in_a_subprocess(staged):
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import numpy as np, pymacos.macos as m\n"
        "m.init(%d); m.load(%r)\n"
        "_, nr, _ = m.trace_rays(m.num_elt())\n"
        "_, _, _, ok, p = m.getRayInfo(int(nr))\n"
        "print('NPASS', int(np.count_nonzero(np.asarray(ok)[1:] & np.asarray(p)[1:])), flush=True)\n"
        "print('NOVF', m.grid_idx_ovf(), flush=True)\n"
    ) % (str(HERE.parent / 'src'), MODEL, FILES[1])
    r = subprocess.run([sys.executable, '-c', code], cwd=staged, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, 'the overflow deck must not kill the host:\n' + r.stdout[-2000:] + r.stderr[-2000:]
    assert int(re.search(r'NPASS (\d+)', r.stdout).group(1)) == NPASS
    assert int(re.search(r'NOVF (\d+)', r.stdout).group(1)) > 0


def test_finite_deck_rejects_nothing(staged):
    _trace(staged / FILES[0])
    assert m.grid_idx_ovf() == 0
