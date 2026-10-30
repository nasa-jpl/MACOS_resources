"""SAVE -> load -> SAVE reaches a fixed point: the source frame is no longer
rebuilt on every load (PLAN_CONSOLIDATION 6a, 2026-10-09; the mmacos gate is
tSaveFixedPoint).  Three SAVE generations of Rx_Roll180Frame.in -- a 180-deg
roll whose sin(pi) residues (xGrid(2) = 1.22e-16) walked an ulp per load
before the fix -- must be byte-identical.
"""
from context import pymacos as m
from context import rx_path


def test_three_save_generations_are_identical(tmp_path):
    m.init(256)
    src = str(rx_path('Rx_Roll180Frame.in'))
    gens = []
    for k in range(3):
        m.load(src)
        out = str(tmp_path / ('g%d.in' % k))
        m.save(out)
        gens.append(open(out).read().splitlines())
        src = out
    for k in (1, 2):
        d = [(i + 1, gens[k][i]) for i in range(len(gens[0])) if gens[k][i] != gens[0][i]]
        assert d == [], 'generation %d differs: %r' % (k + 1, d)
