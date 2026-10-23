"""api save_rx on a NEW file says nothing stray (PLAN_CONSOLIDATION 1.2 (v), 2026-10-22; mmacos twin tSaveNewFileQuiet).

The wrapper passed CARG(2) = 'YES' as the standing answer to CSAVE's 'Replace?' prompt; for a file that did
not exist CSAVE never asked, the token fell through to the SMACOS loop as a command, and the loop printed
'** Unknown command = YES'.  Now the wrapper answers only when the file exists (the first word of the name +
'.in', as CSAVE forms it), and a blank second token is not pushed.  Legs: a save to a new file and a second save
over it (the Replace? path) both write the deck and print no 'Unknown command'.  The pre-fix .so prints the
line on the first save -> red.

The SMACOS loop's 'Unknown command' line is read from the engine's message buffer (messages(), PLAN_CONSOLIDATION
1.2 (ee)), which carries it on every platform; this test read it through capfd and was skipped on macOS, where
the Fortran stdout reaches no Python-visible stream.  Must-pass leg: the load's own 'being loaded' line IS in the
buffer, so the channel works here and the absence after the saves means something.
"""
from context import pymacos as m
from context import rx_path


def test_save_new_then_existing_prints_no_stray_command(tmp_path):
    m.init(128)
    m.messages()
    m.load(str(rx_path('Rx_SaveKeys.in')))
    loaded = m.messages()
    assert [l for l in loaded if 'Rx_SaveKeys.in' in l and 'being loaded' in l], loaded[-10:]
    out = tmp_path / 'saved.in'
    m.save(str(out))
    first = m.messages()
    assert out.is_file() and 'iElt=' in out.read_text()
    out.write_text('% overwritten by the next save\n')
    m.save(str(out))
    second = m.messages()
    assert 'iElt=' in out.read_text()              # the Replace? path took its YES
    assert not [l for l in first if 'Unknown command' in l], first[-10:]
    assert not [l for l in second if 'Unknown command' in l], second[-10:]
