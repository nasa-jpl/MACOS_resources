"""sint() -- the interpolated-surface setup (the CLI's SINT through the api's sint_cmd;
BRIEF_to_bindings B.2, 2026-10-23); twin of mmacos tSint.

Deck: Rx_TraceRewrites.in (= macos ZGD tst_trace_rewrites.in), the f = 1 m paraboloid whose element 1
is `Surface= Interpolated`; with no data it traces as its base conic.  The data file is the deck's name
+ '.srf1.txt', with or without the deck's '.in' (the CLI's spelling keeps it, the bindings' drops it; each
front end tries its own first, then the other), written here on the 15 x 15 grid of
cli_tests/cmd/SINT_short_file.jou: a
1e-6 piston + 1e-4 r^2 deviation, 225 points.  The CLI numbers are what that journal's `opd 2` printed
(both compilers, 2026-10-23): the binding must reproduce its own CLI.  Legs: no file -> element 1 not
set, both names in messages(); the file under either name -> set, the CLI's RMS; a header of 226 for the 225 rows -> the
same RMS and the warning in messages() (the pre-fix engine traced an unread row: RMS 6.17e-7).
The pre-B.2 .so has no sint_cmd: every leg is red.
"""
import shutil

import numpy as np
import pytest
from context import pymacos as m
from context import rx_path

CLI_RMS = 5.630563199e-07          # SINT_short_file.jou, `opd 2`, every leg


def _grid(n_header):
    rows = [f'{n_header} 1']
    for i in range(-7, 8):
        for j in range(-7, 8):
            x, y = i * 0.0175, j * 0.0175
            rows.append(f'{x:g} {y:g} {1e-6 + 1e-4 * (x * x + y * y):.12g}')
    return '\n'.join(rows) + '\n'


@pytest.fixture
def deck(tmp_path):
    p = tmp_path / 'Rx_TraceRewrites.in'
    shutil.copy(rx_path('Rx_TraceRewrites.in'), p)
    m.init(256)
    m.load(str(p))
    m.messages()
    return p


def test_no_data_file_is_not_set_and_says_why(deck):
    r = m.sint()
    assert r['n_set'] == 0
    assert list(r['set']) == [-1] + [0] * (m.num_elt() - 1)
    got = m.messages()
    for name in ('Rx_TraceRewrites.srf1.txt', 'Rx_TraceRewrites.in.srf1.txt'):
        assert [l for l in got if l.rstrip().endswith('/' + name + ' does not exist.')], (name, got)
    assert [l for l in got if 'not correctly set for element   1' in l], got


@pytest.mark.parametrize('name', ['Rx_TraceRewrites.in.srf1.txt', 'Rx_TraceRewrites.srf1.txt'])
def test_the_data_file_sets_the_surface_and_traces_as_the_cli(deck, name):
    rms0 = m.trace_rays(2)[0]
    deck.with_name(name).write_text(_grid(225))
    r = m.sint()
    assert r['n_set'] == 1 and r['set'][0] == 1 and not any(r['set'][1:])
    rms = m.trace_rays(2)[0]
    assert rms == pytest.approx(CLI_RMS, rel=1e-5)
    assert rms > 1e3 * rms0                       # the conic alone is at the round-off floor
    assert not [l for l in m.messages() if 'declares' in l]


def test_a_short_file_uses_the_points_read_and_warns(deck):
    deck.with_name(deck.name + '.srf1.txt').write_text(_grid(226))
    r = m.sint()
    assert r['set'][0] == 1
    assert m.trace_rays(2)[0] == pytest.approx(CLI_RMS, rel=1e-5)
    got = m.messages()
    assert [l for l in got if 'element   1' in l and 'declares 226 points, 225 read' in l], got
