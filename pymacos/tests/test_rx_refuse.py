"""A value with no natural zero that cannot be read REFUSES the load; the
process lives (PLAN_CONSOLIDATION item 1b, 2026-10-09; Dave's ruling 4a).
The pymacos surface of the shared msmacosio.inc parser -- the mmacos gate is
tRxRefuse, the CLI decks macos/ZGD_test_files/tst_short_vec.in and
tst_bad_scalar.in.

THE MUST-FAIL LEG runs in a SUBPROCESS: Rx_ShortVec.in (psiElt= with 2 of 3
values) and Rx_BadScalar.in (KrElt= minus200) are each refused with an
exception naming nothing in Python but one engine line naming the key and the
element; a good deck then loads and traces in the same process.  The pre-fix
.so aborts at the first load (nonzero exit).
"""
import subprocess
import sys
from pathlib import Path

from context import rx_path

HERE = Path(__file__).resolve().parent


def test_refusals_name_the_key_and_the_process_lives():
    bad = [str(rx_path('Rx_ShortVec.in')), str(rx_path('Rx_BadScalar.in'))]
    good = str(rx_path('Rx_ShortCoef.in'))
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import pymacos.macos as m\n"
        "m.init(128)\n"
        "for k, d in enumerate(%r, 1):\n"
        "    try:\n"
        "        m.load(d); print('LOADED', k, flush=True)\n"
        "    except Exception:\n"
        "        print('CAUGHT', k, flush=True)\n"
        "m.load(%r); print('GOOD', m.num_elt(), flush=True)\n"
    ) % (str(HERE.parent / 'src'), bad, good)
    r = subprocess.run([sys.executable, '-c', code], cwd=HERE, capture_output=True, text=True, timeout=300)
    out = r.stdout + r.stderr
    assert r.returncode == 0, 'a refused load must not kill the process (pre-fix: the engine aborts)\n' + out
    assert 'CAUGHT 1' in out and 'CAUGHT 2' in out, out
    assert out.count('Rx load refused: psiElt (elt   1)') == 1, out
    assert out.count('Rx load refused: KrElt (elt   1)') == 1, out
    assert 'GOOD 3' in out, 'a good deck loads after the refusals, in the same process\n' + out
