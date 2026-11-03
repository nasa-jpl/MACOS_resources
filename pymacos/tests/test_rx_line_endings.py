"""Line endings at LOAD (PLAN_CONSOLIDATION item 5, 2026-10-09; Dave's ruling
4a).  The pymacos surface of the shared loader -- the mmacos gate is
tRxLineEndings, the CLI decks macos/ZGD_test_files/tst_eol_{lf,crlf,cr}.in.

A deck with any CR is loaded from an LF copy written to the system temp dir
and deleted on every exit of the load.  pymacos links ifx, and ifx reads a
CR-ONLY (classic Mac) deck as one record: pre-fix the CR twin did not load.
In a SUBPROCESS with TMPDIR pointed at an empty directory: the LF / CRLF / CR
twins of one deck load and trace to the identical wavefront; the CRLF and CR
loads each print one note naming the ORIGINAL deck, the LF load none; the
temp directory is empty afterwards and the temp name never appears.
"""
import os
import subprocess
import sys
from pathlib import Path

from context import rx_path

HERE = Path(__file__).resolve().parent


def test_three_line_endings_one_answer(tmp_path):
    decks = [str(rx_path('Rx_Eol_%s.in' % t)) for t in ('lf', 'crlf', 'cr')]
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import pymacos.macos as m\n"
        "m.init(128)\n"
        "for d in %r:\n"
        "    m.load(d); w = m.traceWavefront(m.num_elt())\n"
        "    print('WFE %%s %%.12e' %% (d[-7:], w[0]), flush=True)\n"
    ) % (str(HERE.parent / 'src'), decks)
    tmpd = tmp_path / 'tmpdir'
    tmpd.mkdir()
    env = dict(os.environ, TMPDIR=str(tmpd))
    r = subprocess.run([sys.executable, '-c', code], cwd=HERE, env=env, capture_output=True, text=True, timeout=300)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    w = [float(l.split()[2]) for l in out.splitlines() if l.startswith('WFE')]
    assert len(w) == 3, 'all three twins load (pre-fix: the CR twin does not on ifx)\n' + out
    assert w[0] > 0 and w[1] == w[0] and w[2] == w[0], w
    assert out.count('CR line endings normalized for this load') == 2, out
    for d in decks[1:]:
        assert d + ': CR line endings normalized' in out, 'the note names the ORIGINAL deck'
    assert 'macos_rx_' not in out, 'the temp name never appears'
    assert list(tmpd.iterdir()) == [], 'the LF copy is deleted'
