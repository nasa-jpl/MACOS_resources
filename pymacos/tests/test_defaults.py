"""One default per option (Dave 2026-10-10; BRIEF_to_bindings A.6) -- twin of mmacos
tDefaults.  Every session/user option the api exposes has ONE default, held by the
engine; a veneer inherits it (reads the api getter) instead of restating it.  Each
test reads the same deck as tDefaults and asserts the SAME numbers (measured
2026-10-16, pymacos and mmacos alike) for the default and for its override.

The table of options (engine default, how each front end reads / sets it, the
override) is cmdref 25_defaults.md, generated with the binding generator.
"""
from pathlib import Path
import os

import numpy as np
import pytest
from context import pymacos as m
from context import rx_path

import pymacos.api_gen as api_gen

MODEL = 128
ZGD = Path(os.environ.get('MACOS_HOME', str(Path.home() / 'dev' / 'macos' / 'macos_f90'))
           ).parent / 'ZGD_test_files'

# measured 2026-10-16 (e5hex1, model 128); tDefaults carries the same constants
# spot shifts on a FRESH load + trace (a FEX first moves the exit-pupil surface and
# with it the FP beam frame: -6242.98 / -6305.26 after fex())
E5_VALID_PX = 10245
MEAN_DECK_CHIEF_MINUS_MEAN = -2.534833896820362e-12   # opd() units; constant over the pupil
MEAN_DECK_VALID_PX = 1756


@pytest.fixture
def e5():
    m.init(MODEL)
    m.load(str(rx_path('e5hex1.in')))
    ne = m.num_elt()
    m.trace_rays(ne)
    return ne


def test_session_modes_default():
    m.init(MODEL)
    m.load(str(rx_path('e5hex1.in')))
    md = m.modes()
    assert (md['pol'], md['vecdif'], md['centroid'], md['lneg'], md['cir']) == \
        (False, False, False, True, False)
    assert md['in_force'] == []


def test_opd_reference_default_and_deck_override():
    m.init(MODEL)
    m.load(str(rx_path('e5hex1.in')))
    assert m.opd_ref() == 'chief'                       # the engine default
    m.load(str(ZGD / 'tst_opdref_mean.in'))             # UseChfRay4OPD= N
    assert m.opd_ref() == 'mean'
    ne = m.num_elt()
    m.trace_rays(ne)
    w_mean, mask = m.opd(), m.opd_mask()
    m.opd_ref('chief')                                  # the api override
    assert m.opd_ref() == 'chief'
    m.trace_rays(ne)
    d = (m.opd() - w_mean)[mask]
    assert mask.sum() == MEAN_DECK_VALID_PX
    assert abs(d.mean()) < 1e-9, 'the chief-minus-mean constant is a numerical zero (bounded, not pinned: 2026-10-19)'
    assert d.std() < 1e-20, 'the two references differ by a constant'


def test_fex_axis_default_is_the_session_axis(e5):
    f = np.asarray(m.fex()).ravel()
    assert np.array_equal(f, np.asarray(m.fex(1)).ravel()), 'default = chief (session)'
    assert np.abs(f - np.asarray(m.fex(0)).ravel()).max() > 1e-8, 'centroid override moves it'
    s = np.asarray(m.sxp()).ravel()
    assert np.array_equal(s, np.asarray(m.sxp(1)).ravel())


def test_spot_centre_default_is_the_element(e5):
    s, s_elt, s_chf = m.spot(e5), m.spot(e5, vpt_center=True), m.spot(e5, vpt_center=False)
    assert api_gen.spc_option_get() == 1, \
        'a per-call chief centre does not change the session option (engine default: vertex)'
    assert np.allclose(m.spot(e5)[0], s_elt[0], rtol=0, atol=1e-9), \
        'so the next default call is the element again'
    # each spot() re-traces; a re-trace is not bit-idempotent (a few ulp), so the
    # points agree to round-off
    assert np.allclose(s[0], s_elt[0], rtol=0, atol=1e-9)
    # the centre moves by the chief-vs-vertex offset of the api's shift vector (the centroid difference is
    # an unstable near-zero quantity: it flipped sign on arm64; Dave 2026-10-19)
    assert np.linalg.norm(np.asarray(s[2])[:2] - np.asarray(s[2])[2:]) > 1, 'the chief override moves the centre'
    # shift = the api's 4-vector [chief xy, vertex xy] (the mmacos convention)
    assert np.asarray(s[2]).shape == (4,)
    # no exact pin on the vector: the second component flips sign on arm64 (6243 vs -5877: a near-zero
    # denominator, not round-off) and the first reads 7.2e-7 off there (CCMac's gate 6; Dave 2026-10-19:
    # assert a different quantity -- the norm above is it)


def test_obscuration_option_default_is_positive(e5):
    assert m.obs_set() == 'positive'                    # the query (obs_opt_get)
    assert m.opd_mask().sum() == E5_VALID_PX
    m.obs_set('negative')
    try:
        assert m.obs_set() == 'negative'
        m.trace_rays(e5)
        assert m.opd_mask().sum() == 0
    finally:
        m.obs_set('positive')
    m.trace_rays(e5)
    assert m.opd_mask().sum() == E5_VALID_PX


def test_ffcut_default_off(e5):
    assert m.ffcut() == (False, 0)


def test_opd_orientation_default_raw(e5):
    w = m.opd()
    assert np.array_equal(w, m.opd(orient='raw')) and np.array_equal(w.T, m.opd(orient='xy'))
    k = m.opd_mask()
    assert np.array_equal(k, m.opd_mask(orient='raw'))


def test_model_size_default_256():
    # Dave 2026-10-16: 256 everywhere (the engine's default for SMACOS with no
    # size; mmacos init.m the same) -- pymacos was 512
    import inspect
    assert inspect.signature(m.init).parameters['model_size'].default == 256



def test_calib_beam_rows_leave_the_spot_centre_alone():
    # CALIB's SPOT rows (macos_ops.F) set the session spcOption to the chief ray and
    # never restored it: every later SPOT -- CLI and the bindings' default -- was
    # chief-centred (found 2026-10-16: the full suite ran test_calib_matrix before
    # this file).  Fixed: the session option is restored after the row.
    m.init(MODEL)
    m.load(str(rx_path('Rx_AsphCalib.in')))     # a SPOT target
    assert api_gen.spc_option_get() == 1
    m.calib_set_iter(2)
    m.calib()
    assert api_gen.spc_option_get() == 1
