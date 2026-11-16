"""CALIB's LM stopping rule and its report (PLAN_CONSOLIDATION 3.3, 2026-10-11); twin of
mmacos tCalibStop.  Rx_AsphCalib.in (SPOT target, h^4 spoiled) converges to the round-off
floor by itr 3-4 -- QUADRATICALLY, so no accepted step is flat: 'floor' at itr 14 (T=20),
'cap' at T=10.  'converged' needs a non-quadratic tail (macos
cli_tests/records/calib_loop_old_vs_new.md: the tDesignTelescope fourth_mirror designs).

The old loop exited on itr_done, a logical never set before the loop and static under
nls_optim_dvr's blanket Save: a T=4 CALIB exited on whatever the previous CALIB left (CLI:
4 iterations fresh, 3 after a converged T=20 CALIB).  Now LM judges rms = sqrt(chisq/rows)
against OptTol (absolute): 'converged' (two flat accepted steps), 'floor' (ten rejected
trials), 'cap'.  calib() carries n_iter / merit / stop_reason; calib_stats() reads them.
Pre-fix .so: no calib_stats_get binding, every test red.
"""
from context import pymacos as m
from context import rx_path


def _run(n_iter, tol=None):
    m.init(128)
    m.load(str(rx_path('Rx_AsphCalib.in')))
    m.calib_set_iter(n_iter)
    if tol is not None:
        m.calib_set_tol(tol)
    return m.calib()


def test_the_count_does_not_depend_on_the_previous_calib():
    r1 = _run(4)
    _run(20)
    r3 = _run(4)
    assert r3['n_iter'] == r1['n_iter']
    assert r3['merit'] == r1['merit']


def test_each_stop_reason_is_reached_and_reported():
    r = _run(20)
    assert r['stop_reason'] == 'floor' and 11 <= r['n_iter'] < 20
    r = _run(10)
    assert r['stop_reason'] == 'cap' and r['n_iter'] == 10
    r = _run(2)
    assert r['stop_reason'] == 'cap' and r['n_iter'] == 2


def test_calib_stats_matches_the_result():
    r = _run(20)
    st = m.calib_stats()
    assert (st['n_iter'], st['merit'], st['stop_reason']) == (r['n_iter'], r['merit'], r['stop_reason'])
    assert r['merit'] >= 0
