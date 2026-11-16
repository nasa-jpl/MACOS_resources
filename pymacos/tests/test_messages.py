"""The engine's message buffer (pymacos messages(); PLAN_CONSOLIDATION 1.2 (ee); mmacos twin tMessages).

The load path's notes, warnings and refusals, CALIB's cause lines and the SMACOS loop's 'Unknown command' print
to Fortran unit 6, which reaches no Python-visible stream on macOS; the engine (emsg_mod) also keeps each line,
and messages() reads it.  Legs:
  * a refused load is named in the buffer AND in load()'s exception, with no Rx loaded afterwards;
  * clear=False reads without emptying, the default empties;
  * past 500 kept lines the first entry counts the dropped ones;
  * Linux: the buffer holds exactly the lines the engine printed (capfd) -- the two channels cannot drift.
The pre-(ee) .so has no messages_count: every leg is red.
"""
import sys

import pytest

from context import pymacos as m
from context import rx_path

import pymacos.api_gen as api_gen


def _short_psi_deck(tmp_path):
    src = rx_path('Rx_SaveKeys.in').read_text().split('\n')
    k = next(i for i, l in enumerate(src) if l.strip().startswith('psiElt='))
    src[k] = '           psiElt=  0.0E+00  0.0E+00'          # two of three values
    p = tmp_path / 'short_psi.in'
    p.write_text('\n'.join(src))
    return p


def test_a_refused_load_is_named_in_the_buffer_and_the_exception(tmp_path):
    m.init(128)
    m.messages()
    with pytest.raises(Exception) as ei:
        m.load(str(_short_psi_deck(tmp_path)))
    assert 'Rx load refused: psiElt (elt' in str(ei.value)
    got = m.messages()
    assert [l for l in got if 'Rx load refused: psiElt (elt' in l], got[-10:]
    assert [l for l in got if 'short_psi.in' in l and 'being loaded' in l], got[-10:]


def test_clear_false_keeps_and_the_default_empties():
    m.init(128)
    m.messages()
    m.load(str(rx_path('Rx_HdrUnknown.in')))
    a = m.messages(clear=False)
    assert [l for l in a if 'Unidentified header string = OptFooKey' in l], a[-10:]
    assert m.messages() == a
    assert m.messages() == []


def test_the_buffer_counts_what_it_dropped():
    m.init(128)
    m.messages()
    for _ in range(400):                       # each load keeps several lines; 500 are kept
        m.load(str(rx_path('Rx_HdrUnknown.in')))
        n_lines, n_chars, n_dropped = api_gen.messages_count()
        if n_dropped:
            break
    assert n_dropped > 0 and n_lines == 500
    got = m.messages()
    assert got[0] == f'({n_dropped} earlier messages dropped)' and len(got) == 501
    assert m.messages() == []


@pytest.mark.skipif(sys.platform == 'darwin',
                    reason='the parity leg reads the Fortran stdout through capfd, which sees none of it on macOS')
def test_the_buffer_holds_exactly_what_was_printed(capfd):
    m.init(128)
    m.messages()
    capfd.readouterr()
    m.load(str(rx_path('Rx_OptAsphOrder.in')))
    printed = capfd.readouterr().out.split('\n')
    kept = m.messages()
    assert kept, 'the load kept lines'
    # every kept line was printed, in order (the load also prints lines that are not routed)
    it = iter(l.rstrip() for l in printed)
    assert all(any(k == p for p in it) for k in kept), (kept[:5], printed[:10])
