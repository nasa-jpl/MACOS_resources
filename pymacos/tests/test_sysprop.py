"""first_order_properties (the SYSPROP command) -- the pixel-based fields come only from a
DIFFRACTION pitch (PLAN_CONSOLIDATION 1.2 (d), 2026-10-17); twin of mmacos
tSysProp/test_px_not_from_an_OPD and cli_tests/cmd/SYSPROP_after_OPD.jou.

An OPD at the element re-sets dxElt to the ray-grid spacing (~0 at a focus): on
Rx_Cass_FarField (model 128) the fields read lamD_px 3.3130544e6, pitch 1.6958903e-12 after
an OPD alone and after intensity() then opd().  Now 0 there; intensity()'s numbers between.
"""
import numpy as np
from context import pymacos as m
from context import rx_path

INT_LAMD_PX = 1.0177264e+00          # measured 2026-10-17, model 128
INT_PLATE_ARCSEC_PX = 5.0668040e-02
INT_DX_FOCAL = 5.5207145e-06
PX = ('lamD_px', 'plate_arcsec_px', 'dx_focal_baseunits')


def test_px_not_from_an_opd():
    m.init(128)
    m.load(str(rx_path('Rx_Cass_FarField.in')))
    ne = m.num_elt()
    m.trace_rays(ne)
    m.opd()
    p0 = m.first_order_properties(ne)
    assert [p0[k] for k in PX] == [0.0, 0.0, 0.0]
    m.intensity(ne)
    p1 = m.first_order_properties(ne)
    assert np.allclose([p1[k] for k in PX], [INT_LAMD_PX, INT_PLATE_ARCSEC_PX, INT_DX_FOCAL],
                       rtol=1e-7, atol=0)
    m.trace_rays(ne)
    m.opd()
    p2 = m.first_order_properties(ne)
    assert [p2[k] for k in PX] == [0.0, 0.0, 0.0], \
        'an opd() after the intensity() replaces the pitch with the ray spacing'
