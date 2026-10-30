"""A grid FILE larger than the model's mGridMat is declined by name and COUNTED (engine
2026-10-10); twin of mmacos tGridDropped.  e5hex1's eight segment grids are a 256-point
flat.txt: at model 128 the load proceeds with all eight declined (GridInit used to write
them past their GridMat slots); at model 256 nothing is declined."""
from context import pymacos as m
from context import rx_path


def test_too_small_a_model_declines_and_counts():
    m.init(128)
    m.load(str(rx_path('e5hex1.in')))
    assert m.grid_dropped() == 8
    _, nrays, _ = m.trace_rays(m.num_elt() - 1)
    assert nrays > 0


def test_a_model_that_fits_declines_nothing():
    m.init(256)
    m.load(str(rx_path('e5hex1.in')))
    assert m.grid_dropped() == 0
