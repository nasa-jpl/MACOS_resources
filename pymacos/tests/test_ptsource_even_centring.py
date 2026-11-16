"""Even-grid POINT-source centring (engine macos 6d13d07, PR #70 completed; PLAN 1.1b).

PtSource mapped grid point j to diffraction-grid row jwf = j+(mdttl-nGridPts)/2+1, which
for an EVEN grid puts the seeded field one row high (and at nGridPts == mdttl wrote
RayID(., mdttl+1)); ColSource had been fixed in PR #70.  Now both use
(mdttl-nGridPts+mod(nGridPts,2))/2.

Measured on Rx_Coro.in (a point source, zSource = -100 mm) at model 1024, the field the
first physical leg seeds at Elt 2 (Prop_1_start):
  * odd grid (511): pre-fix and post-fix BIT-identical -- the formula is unchanged there;
  * even grid (510): the pre-fix field is the post-fix field shifted by exactly ONE ROW
    in j (max diff 0 after the roll), centroid (512.5, 513.5) pre vs (512.5, 512.5) post;
    total power identical.
Gate: the seeded field's intensity centroid is symmetric in i and j on the even grid,
as on the odd one (the deck is centred in both axes at this plane).  Red on a pre-fix
.so (built in a side tree against build_release_prefix9 + resources a35ad6e, 2026-10-10):
the even-grid j centroid is one pixel off.
"""
import numpy as np
import pytest
from context import pymacos as m
from context import rx_path


def _centroid(n):
    m.init(1024)
    m.load(str(rx_path('Rx_Coro.in')))
    m.src_sampling(n)
    m.modify()
    intensity = np.asarray(m.intensity(2)).copy()
    tot = intensity.sum()
    ii, jj = np.indices(intensity.shape) + 1
    return (intensity * ii).sum() / tot, (intensity * jj).sum() / tot, tot


@pytest.mark.parametrize('n', [511, 510])
def test_point_source_seed_is_centred_on_both_grid_parities(n):
    ci, cj, tot = _centroid(n)
    assert tot > 0
    assert cj == pytest.approx(ci, abs=1e-6), 'the seeded field must sit in j where it sits in i'


def test_even_grid_sits_half_a_pixel_below_the_odd_one():
    ci_o, cj_o, _ = _centroid(511)
    ci_e, cj_e, _ = _centroid(510)
    assert ci_o - ci_e == pytest.approx(0.5, abs=1e-6)
    assert cj_o - cj_e == pytest.approx(0.5, abs=1e-6)
