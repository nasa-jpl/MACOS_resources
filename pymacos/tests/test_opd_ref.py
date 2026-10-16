"""The OPD reference: the CHIEF ray by default (engine 2026-10-10), the mean the option.

Dave's ruling (BRIEF_to_opd_reference): the engine matches the manual -- each
ray's OPD is its optical path minus the chief ray's.  Until 2026-10-10 every
path subtracted the aperture MEAN (the per-load reset put the flag back after
the LOAD handler set it), which on a segmented pupil pistons every untouched
segment when one segment moves.  Twin of mmacos tOpdRef.

Legs, on e5hex1.in (7 hex segments, OPD at the exit-pupil Return, elt 12,
Tz = 1e-8 m on segment elt 3):
  * the DEFAULT after a plain load is 'chief' and the untouched segments
    read EXACTLY 0 (pre-change engine: 'mean', median 2.85e-6 -- and no
    opd_ref binding at all);
  * opd_ref('mean') brings the cross-segment piston back (non-vacuity);
  * a `UseChfRay4OPD= N` twin deck gives the mean map bit for bit, and
    opd_ref('chief') after that load overrides it;
  * the obscured-chief deck Rx_Cass_FarField defaults to the chief map.
"""
import numpy as np
import pytest
from context import pymacos as m
from context import rx_path

MODEL, NGRID, EVAL, POKE, DZ = 128, 63, 12, 3, 1e-8


def _poke(deck, ref=None):
    m.init(MODEL)
    m.load(str(deck))
    m.src_sampling(NGRID)
    m.modify()
    if ref is not None:
        m.opd_ref(ref)
    state = m.opd_ref()
    m.trace_rays(EVAL)
    w0 = np.asarray(m.opd()).copy()
    m.perturb(POKE, translation_m=(0.0, 0.0, DZ))
    m.modify()
    m.trace_rays(EVAL)
    w1 = np.asarray(m.opd()).copy()
    v = w0 != 0
    return state, w0, w1 - w0, v


def _twin(tmp_path, yn):
    t = rx_path('e5hex1.in').read_text().split('\n')
    k = next(i for i, l in enumerate(t) if l.strip().startswith('nElt'))
    p = tmp_path / f'e5hex1_{yn}.in'
    p.write_text('\n'.join(t[:k] + [f'   UseChfRay4OPD=  {yn}'] + t[k:]))
    return p


def test_default_is_the_chief_ray():
    state, _, d, v = _poke(rx_path('e5hex1.in'))
    assert state == 'chief'
    assert np.median(d[v]) == 0.0, 'untouched segments must read 0 by default'
    assert np.max(np.abs(d[v])) > 1e-9, 'non-vacuity: the poke responds'


def test_mean_option_restores_the_cross_segment_piston():
    state, _, d, v = _poke(rx_path('e5hex1.in'), 'mean')
    assert state == 'mean'
    assert abs(np.median(d[v])) > 0.05 * np.max(np.abs(d[v]))


def test_deck_keyword_n_and_the_override(tmp_path):
    deck = _twin(tmp_path, 'N')
    s_n, w_n, _, _ = _poke(deck)
    _, w_m, _, v = _poke(rx_path('e5hex1.in'), 'mean')
    s_o, w_o, _, _ = _poke(deck, 'chief')
    _, w_c, _, _ = _poke(rx_path('e5hex1.in'))
    assert s_n == 'mean' and s_o == 'chief'
    assert np.array_equal(w_n, w_m), 'the N deck == the mean map'
    assert np.array_equal(w_o, w_c), "opd_ref('chief') overrides the deck"
    assert np.max(np.abs(w_m[v] - w_c[v])) > 0, 'non-vacuity: the maps differ'


def test_obscured_chief_defaults_to_the_chief_map():
    m.init(MODEL)
    m.load(str(rx_path('Rx_Cass_FarField.in')))
    ie = m.num_elt() - 1
    m.trace_rays(ie); wd = np.asarray(m.opd()).copy()
    m.opd_ref('mean'); m.trace_rays(ie); wm = np.asarray(m.opd()).copy()
    m.opd_ref('chief'); m.trace_rays(ie); wc = np.asarray(m.opd()).copy()
    v = wm != 0
    assert np.array_equal(wd, wc)
    d = wc[v] - wm[v]
    assert np.max(np.abs(d)) > 0 and np.std(d) < 1e-9 * np.max(np.abs(d))


def test_bad_mode_raises():
    m.init(MODEL)
    m.load(str(rx_path('e5hex1.in')))
    with pytest.raises(ValueError):
        m.opd_ref('centroid')


def test_opd_mask_is_the_ray_set():
    """opd_mask() is True exactly where the map holds a ray: every passing ray
    (ray 1, the chief, is never written), a superset of opd() != 0 -- a valid
    ray at the chief's path reads exactly 0 under the default reference --
    and the same under either reference."""
    m.init(MODEL)
    m.load(str(rx_path('e5hex1.in')))
    m.src_sampling(NGRID)
    m.modify()
    _, nrays, _ = m.trace_rays(EVAL)
    W = np.asarray(m.opd()).copy()
    M = m.opd_mask()
    _, _, _, ok, passed = m.getRayInfo(int(nrays))
    assert M.sum() == np.count_nonzero(np.asarray(ok)[1:] & np.asarray(passed)[1:])
    assert M[W != 0].all()
    m.opd_ref('mean')
    m.trace_rays(EVAL)
    assert np.array_equal(m.opd_mask(), M)
