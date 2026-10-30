"""ZernType= ExtFringe traces (macos PLAN_CONSOLIDATION 1.2, 2026-10-12; the CLI twin is
cli_tests/cmd/ZernType_extfringe.jou).  Extended Fringe continues Fringe's ordering past 37; the
converter (surfsub ZerntoMon_ZFE) existed but no dispatch chain called it, so an ExtFringe surface
traced as its bare conic.  Twins (= macos ZGD tst_zerndef_{fringe,extfringe,ansi}.in): an NS
Cassegrain whose ZrnGrData primary carries mode 9 = 2e-7 m -- spherical in Fringe and ExtFringe,
trefoil in ANSI.  Pre-fix .so: the ExtFringe rms is the bare conic's (~3e-12 m) -- red.
"""
import numpy as np
from context import pymacos as m
from context import rx_path


def _rms(name):
    m.init(128)
    m.load(str(rx_path(name)))
    rms, _, _ = m.traceWavefront(m.num_elt())
    return float(rms)


def test_extfringe_traces_as_fringe_on_shared_modes():
    f, e, a = _rms('Rx_ZernFringe9.in'), _rms('Rx_ZernExtFringe9.in'), _rms('Rx_ZernAnsi9.in')
    assert f > 1e-7, 'the Fringe twin carries the figure (else the gate is vacuous)'
    assert e == f, 'ExtFringe mode 9 == Fringe mode 9 (the same polynomial): %r vs %r' % (e, f)
    assert not np.isclose(e, a, rtol=1e-6), 'and differs from ANSI mode 9 (trefoil)'
