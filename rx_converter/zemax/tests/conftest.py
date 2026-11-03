"""Fixtures for the zmx2macos tests.

The engine is pymacos from MACOS_resources/pymacos/src (env PYMACOS_SRC overrides).  The
program decks (DST, AFTA) are not committed: they are read from ZMX2MACOS_DECKS (default
~/dev/MACOS_sandbox/zemax2macos) and their tests skip with a message when absent.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ZDIR = HERE.parent
RES = ZDIR.parent.parent                        # MACOS_resources
sys.path.insert(0, str(ZDIR))
import zmx2macos as zm                          # noqa: E402

DECKS = Path(os.environ.get('ZMX2MACOS_DECKS',
                            Path.home() / 'dev' / 'MACOS_sandbox' / 'zemax2macos'))
DST = DECKS / 'dst2v2_aox_v5_di_2025_05_07_fiximgtilt_stop_at_dm1_20260507_Design-Locked.zmx'
AFTA = DECKS / 'AFTA COR V5B-2-3-5-Testbed-O tmp 1 60inch oap source_FSS99-removed.zmx'
DST_CODEV = RES / 'mmacos' / 'templates' / '30_instruments' / 'bench_ctb' / 'dst2' / 'dst2_F01_norm.in'


def need(path):
    if not Path(path).is_file():
        pytest.skip(f'{path} is absent (the program decks are not committed; '
                    'set ZMX2MACOS_DECKS)')
    return path


@pytest.fixture(scope='session')
def engine():
    src = os.environ.get('PYMACOS_SRC', str(RES / 'pymacos' / 'src'))
    sys.path.insert(0, src)
    try:
        import pymacos
    except ImportError as e:
        pytest.skip(f'pymacos not importable from {src}: {e}')
    pymacos.init(128)
    return pymacos


def convert(path, *args):
    opts = zm.options([str(path), *args])
    return zm.convert(str(path), opts)


class Loaded:
    """A converted deck loaded in the engine, traced to the focal plane."""

    def __init__(self, m, rx):
        self.m = m
        self.n = m.load(str(rx))
        self.names = [l.split('=', 1)[1].strip() for l in open(rx)
                      if l.strip().startswith('EltName=')]
        rms, nr, _ = m.trace_rays(-1)
        p, d, o, ok, ps = m.getRayInfo(int(nr))
        self.rms, self.nrays = float(rms), int(nr)
        self.pos_fp = np.asarray(p).reshape(3, -1) if np.asarray(p).shape[0] == 3 else np.asarray(p).T
        self.ok, self.passed = np.asarray(ok, bool).ravel(), np.asarray(ps, bool).ravel()
        src = m.src_info()
        self.zsource = float(np.ravel(src[0])[0])
        self.chf_pos, self.chf_dir = np.ravel(src[1]), np.ravel(src[2])

    def elt(self, i):
        m = self.m
        return dict(v=np.ravel(m.elt_vpt(i)), p=np.ravel(m.elt_psi(i)),
                    kr=float(np.ravel(m.elt_kr(i))[0]), kc=float(np.ravel(m.elt_kc(i))[0]))

    def by_name(self):
        return {nm: self.elt(i) for i, nm in enumerate(self.names, 1)}

    def rays_at(self, i):
        rms, nr, _ = self.m.trace_rays(i)
        p, d, o, ok, ps = self.m.getRayInfo(int(nr))
        p = np.asarray(p)
        return (p if p.shape[0] == 3 else p.T), np.asarray(ok, bool).ravel()


@pytest.fixture
def load(engine, tmp_path):
    def _load(text, name='deck.in'):
        rx = tmp_path / name
        rx.write_text(text)
        return Loaded(engine, rx)
    return _load


def cross_trace(cv, L, nmax=400):
    """Re-trace the engine's rays with the converter's own Zemax-frame tracer.

    The engine's hit points on element 1 fix each ray (the source geometry fixes its
    direction); the converter traces it through every Zemax surface.  Returns the largest
    distance (base units) between the two image points over the rays both trace."""
    s = cv.scale
    p1, ok1 = L.rays_at(1)
    pf, okf = L.rays_at(L.n)
    idx = np.flatnonzero(ok1 & okf)
    idx = idx[np.linspace(0, len(idx) - 1, min(nmax, len(idx))).astype(int)]
    worst, n = 0.0, 0
    for k in idx:
        hit = p1[:, k] / s
        if cv.obj_inf:
            d = cv.chief_d
            start = hit - 10.0 * d
        else:
            start = cv.obj_pos
            d = hit - start
        hs = zm.trace(cv, start, d)
        if not hs:
            continue
        worst = max(worst, float(np.linalg.norm(hs[-1][2] * s - pf[:, k])))
        n += 1
    return worst, n
