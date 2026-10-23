"""The api surface of Dave's three loader / SAVE rulings of 2026-10-18 (the CLI gates are
cli_tests/cmd/OUTCORD_missing.jou, PROPTYPE_numeric.jou, CALIB_save_roundtrip.jou; the mmacos
twin is tRxLoadRulings1018).  The reader is shared, so each ruling is gated here too.

1. A deck with no nOutCord=/Tout= block loads (it failed silently before): the loader proposes
   the beam frame at the last element and the api takes it -- the Cassegrain's OPD comes back,
   save writes the identity Tout the deck used to carry, and the saved deck re-saves identically.
2. A numeric PropType= (0..15 in element order) loads; save writes the names, 0 as Geometric.
3. The CALIB setup transfers through save: the deck's keys and a cap set by calib_set_iter come
   back in the saved file, and a reload + re-save is byte-identical.
"""
from context import pymacos as m
from context import rx_path

NAMES = ['Geometric', 'Geometric', 'NFS1surf', 'FarField', 'NFP1surf', 'NFSpherical', 'NFPlane',
         'SF1surf', 'SpatialFilter', 'GeomUpdate', 'NF1', 'NF2', 'Fresnel', 'NFdftPlane',
         'NFdftSphere', 'FFdft']


def _resave_identical(first, second):
    m.load(first)
    m.save(second)
    assert open(second).read() == open(first).read()


def test_no_outcord_deck_loads_with_the_beam_frame(tmp_path):
    m.init(256)
    n = m.load(str(rx_path('tst_no_outcord.in')))
    assert n == 5
    rms, _, _ = m.traceWavefront(n)
    assert abs(rms - 1.2938e-12) < 3e-15, rms        # the proposal does not touch the trace
    f1 = str(tmp_path / 'noc1.in')
    m.save(f1)
    lines = open(f1).read().splitlines()
    k = next(i for i, l in enumerate(lines) if l.strip().startswith('nOutCord='))
    assert lines[k].split('=')[1].split() == ['5']
    assert lines[k + 1].split('=')[1].split() == ['1.0E+00'] + ['0.0E+00'] * 6   # x row = global x
    assert lines[k + 2].split() == ['0.0E+00', '1.0E+00'] + ['0.0E+00'] * 5      # y row = global y
    _resave_identical(f1, str(tmp_path / 'noc2.in'))


def test_numeric_proptype_saves_as_names(tmp_path):
    m.init(256)
    assert m.load(str(rx_path('tst_proptype_numeric.in'))) == 16
    f1 = str(tmp_path / 'pt1.in')
    m.save(f1)
    names = [l.split('=')[1].strip() for l in open(f1).read().splitlines()
             if l.strip().startswith('PropType=')]
    assert names == NAMES
    _resave_identical(f1, str(tmp_path / 'pt2.in'))


def test_calib_setup_transfers_through_save(tmp_path):
    m.init(256)
    m.load(str(rx_path('Rx_AsphCalib.in')))   # SPOT target, OptMaxItrs 10, OptAsph on element 1
    m.calib_set_iter(7)                            # a setting made by the binding
    f1 = str(tmp_path / 'cs1.in')
    m.save(f1)
    txt = open(f1).read()
    assert 'OptTarget=  SPOT' in txt
    assert 'OptMaxItrs=   7' in txt
    assert 'OptAsph=' in txt
    _resave_identical(f1, str(tmp_path / 'cs2.in'))
