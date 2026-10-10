"""Tracing a deck must not change the prescription SAVE writes
(PLAN_CONSOLIDATION item 3, 2026-10-09; the mmacos gate is tTraceKeepsRx).

CTRACE's Return branch wrote the RUNNING medium index into the Return's IndRef
after every ray; after a LensArray in glass, load -> trace -> SAVE wrote the
lenslet index (1.51242597) onto the next Return where load -> SAVE wrote the
deck's 1.0 (Rx_SaveKeys.in = macos ZGD_test_files/tst_save_keys.in, elt 9).
SAVE after load -> trace(nElt) must equal SAVE after load, byte for byte.
"""
import os
import shutil
from pathlib import Path

from context import pymacos as m
from context import rx_path


def test_save_after_a_trace_equals_save_after_load(tmp_path):
    for f in ('Rx_SaveKeys.in', 'tst_save_ampl.dat'):
        shutil.copy(rx_path(f), tmp_path / f)
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        m.init(256)
        m.load('Rx_SaveKeys.in')
        m.save(str(tmp_path / 'a.in'))
        m.load('Rx_SaveKeys.in')
        m.traceWavefront(m.num_elt())
        m.save(str(tmp_path / 'b.in'))
    finally:
        os.chdir(old)
    a = (tmp_path / 'a.in').read_text().splitlines()
    b = (tmp_path / 'b.in').read_text().splitlines()
    assert len(a) == len(b)
    d = [(i + 1, b[i]) for i in range(len(a)) if a[i] != b[i]]
    assert d == [], 'SAVE after a trace differs: %r' % d
