"""Short / re-wrapped coefficient blocks load instead of killing the host
(PLAN_CONSOLIDATION item 1, 2026-10-09; the pymacos surface of the shared
msmacosio.inc parser -- the mmacos gate is tRxShortCoef, the CLI deck
macos/ZGD_test_files/tst_short_coef.in).

The parser read ZernCoef= with a bare list-directed internal READ: a line with
fewer values than nZernCoef= was an uncaught end-of-file, a Fortran runtime
abort that takes the Python process with it.  Now ReadCoefBlock (elt_mod)
pads with zero and prints one note.

  * THE MUST-FAIL LEG runs in a SUBPROCESS: Rx_ShortCoef.in (nZernCoef= 4,
    a ZernCoef= line of 3) loads and reads back [2e-4, 1e-4, 5e-5, 0]; the
    pre-fix .so aborts at the load, so the subprocess exits nonzero.
  * the control: the same deck with the 4th value given reads it unchanged.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from context import pymacos as m
from context import rx_path

MODEL = 128
HERE = Path(__file__).resolve().parent


def test_short_zerncoef_loads_in_a_subprocess():
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import pymacos.macos as m\n"
        "m.init(%d); m.load(%r)\n"
        "print('COEF', *('%%.10e' %% c for c in m.elt_zrn_coef(1, [1, 2, 3, 4])))\n"
    ) % (str(HERE.parent / 'src'), MODEL, str(rx_path('Rx_ShortCoef.in')))
    r = subprocess.run([sys.executable, '-c', code], cwd=HERE, capture_output=True, text=True, timeout=300)
    out = r.stdout + r.stderr
    assert r.returncode == 0, 'the short ZernCoef= deck must not kill the process (pre-fix: the engine aborts at load)\n' + out
    line = [l for l in out.splitlines() if l.startswith('COEF')]
    assert line, out
    np.testing.assert_allclose([float(t) for t in line[0].split()[1:]], [2e-4, 1e-4, 5e-5, 0.0], rtol=1e-12)
    assert out.count('ZernCoef (elt') == 1, 'exactly ONE note line'


def test_full_zerncoef_control(tmp_path):
    s = Path(rx_path('Rx_ShortCoef.in')).read_text()
    old = '         ZernCoef=  2.0D-04  1.0D-04  5.0D-05\n'
    assert s.count(old) == 1
    p = tmp_path / 'full.in'
    p.write_text(s.replace(old, old.rstrip('\n') + '  3.0D-05\n'))
    m.init(MODEL)
    m.load(str(p))
    np.testing.assert_allclose(m.elt_zrn_coef(1, [1, 2, 3, 4]), [2e-4, 1e-4, 5e-5, 3e-5], rtol=1e-12)
