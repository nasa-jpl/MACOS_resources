"""api save_rx on a NEW file says nothing stray (PLAN_CONSOLIDATION 1.2 (v), 2026-10-22; mmacos twin tSaveNewFileQuiet).

The wrapper passed CARG(2) = 'YES' as the standing answer to CSAVE's 'Replace?' prompt; for a file that did
not exist CSAVE never asked, the token fell through to the SMACOS loop as a command, and the loop printed
'** Unknown command = YES'.  Now the wrapper answers only when the file exists (the first word of the name +
'.in', as CSAVE forms it), and a blank second token is not pushed.  Legs: a save to a new file and a second save
over it (the Replace? path) both write the deck and print no 'Unknown command'.  The pre-fix .so prints the
line on the first save -> red.  capfd reads the Fortran stdout at the descriptor level (none on macOS, where
capfd captures no Fortran stdout: skipped there, as test_rx_parse_warnings).
"""
import sys

import pytest

from context import pymacos as m
from context import rx_path

pytestmark = pytest.mark.skipif(sys.platform == 'darwin',
                                reason='capfd captures no Fortran stdout on macOS (Dave 2026-10-19: do not block)')


def test_save_new_then_existing_prints_no_stray_command(tmp_path, capfd):
    m.init(128)
    m.load(str(rx_path('Rx_SaveKeys.in')))
    capfd.readouterr()
    out = tmp_path / 'saved.in'
    m.save(str(out))
    first = capfd.readouterr().out
    assert out.is_file() and 'iElt=' in out.read_text()
    out.write_text('% overwritten by the next save\n')
    m.save(str(out))
    second = capfd.readouterr().out
    assert 'iElt=' in out.read_text()              # the Replace? path took its YES
    assert 'Unknown command' not in first, first[-400:]
    assert 'Unknown command' not in second, second[-400:]
