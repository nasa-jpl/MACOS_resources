"""
    -------------------------------------------------------------------------------------------
[ ] Grating on Conic Base Srf.
    -------------------------------------------------------------------------------------------
    [ ] elt_grating_any       Checks if any Grating Srfs. are defined in Rx
    [ ] elt_grating_fnd       Find all elements with Grating Srfs. types
    [ ] elt_grating_params    Linear grating (h1HOE, RuleWidth, Trans. or Refl.)
    [ ] elt_grating_type      Transmission or Reflective Grating
    [ ] elt_grating_order     Linear Grating Order (OrderHOE)
    [ ] elt_grating_spacing   RuleWidth
    [ ] elt_grating_dir       h1HOE vector perpendicular to the ruling dir and psiElt.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

import rx_data
from context import pymacos as _lib
from context import rx_path
from test_settings import _Tol

RX_SETS = (rx_data.data_Rx_003(),
           )


# https://www.codesdope.com/course/python-subclass-of-a-class/
# can also be placed into "conftest.py" containing all the fixtures
class RxData(rx_data.RxDataBase):
    """pytest manual: 2.3.12 Modularity: using fixtures from a fixture function"""

    def __init__(self, data, macos_size):
        super().__init__(data)  # initialise Base Class first

        self.tol = {"abs": 1e-15, "rel": 1e-15}   # Testing: abs. & rel. error

        self.macos_size = macos_size
        self.srfs = np.arange(1, self.n_elt+1)

        # chk if Rx has Grid Srfs in DB
        #    SrfTypeName:  01) Flat      05) Interpolated  09) GridData
        #                  02) Conic     06) Anamorphic    10) Toric
        #                  03) Aspheric  07) UserDefined   11) AsGrData
        #                  04) Monomial  08) Zernike
        self.grid_srf_types = (9, 11)
        if self.rx_settings.get('EltSrfType', None) is not None:
            srf_types = self.rx_settings['EltSrfType']
            self.grid_defined = len(set(srf_types).intersection(self.grid_srf_types)) > 0
            self.csys = self.rx_settings['EltSrfCFrame']
            self.grid = self.rx_settings.get('GridInfo', None)
            if self.grid is not None:
                self.grid_srf = [srf for (srf, _, _) in self.grid]
            else:
                self.grid_srf = None


@pytest.fixture(scope="module",   # session, module, class, function,
                params=(128, ))   # check if changing the macos size is affecting computation:
def macos_setup(request):
    _lib.init(request.param)

    assert _lib._MODELSIZE == request.param
    assert _lib._SYSINIT
    return request.param


class TestGratingAPI:
    """Check Grating API"""

    rx_sets = (rx_data.Rx_Grating_001(),  # with Grating
            #    rx_data.jwst_Rx_OO_monoPM(),        # no Grating
            #    rx_data.freeform_4mirrors(),
               )

    @pytest.fixture(scope="class", params=rx_sets)
    def rx_setup(self, macos_setup, request) -> RxData:
        # load test Rx
        data_set = request.param

        # # chk if Rx has Zernike Srfs in DB
        # if data_set.get('Zern', None) is None:
        #     n_elt = _lib.load(rx_path(data_set['Rx']))
        #     assert  not _lib.elt_zrn_any()
        #     pytest.skip('Require Zrn. entry in DB')

        n_elt = _lib.load(rx_path(data_set['Rx']))
        assert n_elt == _lib.num_elt()
        assert n_elt == data_set['nElt']
        return RxData(data_set, macos_setup)

    # -------------------------
    # read
    # -------------------------
    def test_grating_any(self, rx_setup) -> None:
        """Grating def. existence check"""
        # print("=======> ", _lib.elt_grating_any())
        # print(rx_setup.rx_settings['Grating'])
        assert _lib.elt_grating_any() == (rx_setup.rx_settings['Grating'] is not None)

    def test_grating_fnd(self, rx_setup) -> None:
        """Grating def. on Srfs  against DB check"""

        srfs = rx_setup.rx_settings['Grating'].keys()
        assert set(srfs) == set(_lib.elt_grating_fnd()[0])

    def test_grating_params(self, rx_setup) -> None:
        """Cmp. Grating def. on Srfs  against DB check"""

        tol = rx_setup.tol
        for srf, (gr_refl, gr_order, gr_rw, gr_dir) in rx_setup.rx_settings['Grating'].items():
            refl, rule_width, diff_order, rule_dir = _lib.elt_grating_params(srf)

            assert gr_refl == refl         # reflective Grating
            assert gr_order == diff_order
            assert gr_rw == rule_width
            np.testing.assert_allclose(rule_dir, np.asarray(gr_dir), tol["rel"], tol["abs"])

    def test_grating_order(self, rx_setup) -> None:
        """Cmp. Grating def. on Srfs  against DB check -- Order"""

        for srf, (gr_refl, gr_order, gr_rw, gr_dir) in rx_setup.rx_settings['Grating'].items():
            assert _lib.elt_grating_fnd(srf)[0]
            assert _lib.elt_grating_params(srf)[2] == gr_order
            assert _lib.elt_grating_order(srf).item() == gr_order

    def test_grating_rulewidth(self, rx_setup) -> None:
        """Cmp. Grating def. on Srfs  against DB check -- rule width"""

        for srf, (gr_refl, gr_order, gr_rw, gr_dir) in rx_setup.rx_settings['Grating'].items():
            assert _lib.elt_grating_fnd(srf)[0]
            assert _lib.elt_grating_params(srf)[1] == gr_rw
            assert _lib.elt_grating_rulewidth(srf) == gr_rw

    def test_grating_type(self, rx_setup) -> None:
        """Cmp. Grating def. on Srfs  against DB check -- Type"""

        for srf, (gr_refl, gr_order, gr_rw, gr_dir) in rx_setup.rx_settings['Grating'].items():
            assert _lib.elt_grating_fnd(srf)[0]
            assert _lib.elt_grating_params(srf)[0] == gr_refl
            assert _lib.elt_grating_type(srf) == gr_refl

    def test_grating_dir(self, rx_setup) -> None:
        """Cmp. Grating def. on Srfs  against DB check -- direction"""

        tol = rx_setup.tol
        for srf, (gr_refl, gr_order, gr_rw, gr_dir) in rx_setup.rx_settings['Grating'].items():
            assert _lib.elt_grating_fnd(srf)[0]

            gr_dir_ = _lib.elt_grating_params(srf)[3]
            np.testing.assert_allclose(gr_dir_, np.asarray(gr_dir), tol["rel"], tol["abs"])

            gr_dir_ = _lib.elt_grating_dir(srf)
            np.testing.assert_allclose(gr_dir_, np.asarray(gr_dir), tol["rel"], tol["abs"])

    # -------------------------
    # read/write
    # -------------------------
    @pytest.mark.parametrize("gr_order_", [-4, 4])
    def test_grating_order_rw(self, rx_setup, gr_order_) -> None:
        """Cmp. Grating def. on Srfs  against DB check -- Order"""

        for srf, (gr_refl, gr_order, gr_rw, gr_dir) in rx_setup.rx_settings['Grating'].items():
            refl, rule_width, diff_order, rule_dir = _lib.elt_grating_params(srf)
            assert _lib.elt_grating_order(srf) == gr_order

            # write
            _lib.elt_grating_order(srf, gr_order_)
            assert _lib.elt_grating_order(srf) == gr_order_

            # restore
            _lib.elt_grating_order(srf, gr_order)

    @pytest.mark.parametrize("rule_width_", [0.05, 1.00])
    def test_grating_rulewidth_rw(self, rx_setup, rule_width_) -> None:
        """Cmp. Grating def. on Srfs  against DB check -- Rule Width"""

        for srf, (gr_refl, gr_order, gr_rw, gr_dir) in rx_setup.rx_settings['Grating'].items():
            refl, rule_width, diff_order, rule_dir = _lib.elt_grating_params(srf)
            assert _lib.elt_grating_rulewidth(srf) == gr_rw

            # write/read back
            _lib.elt_grating_rulewidth(srf, rule_width_)
            assert _lib.elt_grating_rulewidth(srf) == rule_width_

            # restore
            _lib.elt_grating_rulewidth(srf, gr_rw)

    @pytest.mark.parametrize("reflective", [True, False])
    def test_grating_type_rw(self, rx_setup, reflective) -> None:
        """Cmp. Grating def. on Srfs  against DB check -- Rule Width"""

        for srf, (gr_refl, gr_order, gr_rw, gr_dir) in rx_setup.rx_settings['Grating'].items():
            refl, rule_width, diff_order, rule_dir = _lib.elt_grating_params(srf)
            assert _lib.elt_grating_type(srf) == gr_refl

            # write/read
            _lib.elt_grating_type(srf, reflective)
            assert _lib.elt_grating_type(srf) == reflective

            # restore
            _lib.elt_grating_type(srf, gr_refl)

    @pytest.mark.parametrize("direction", [(1, 0, 0), (1, 1, 1) ])
    def test_grating_dir_rw(self, rx_setup, direction) -> None:
        """Cmp. Grating def. on Srfs  against DB check -- Direction"""

        tol = rx_setup.tol
        for srf, (gr_refl, gr_order, gr_rw, gr_dir) in rx_setup.rx_settings['Grating'].items():
            gr_dir = np.asarray(gr_dir, dtype=float)
            refl, rule_width, diff_order, rule_dir = _lib.elt_grating_params(srf)
            np.testing.assert_allclose(rule_dir, gr_dir, tol["rel"], tol["abs"])

            # write/read
            _lib.elt_grating_dir(srf, direction)
            direction = np.asarray(direction, dtype=float)
            direction /= np.linalg.norm(direction)
            gr_dir_ = _lib.elt_grating_dir(srf)
            np.testing.assert_allclose(gr_dir_, direction, tol["rel"], tol["abs"])

            # restore
            _lib.elt_grating_dir(srf, gr_dir)

    @pytest.mark.parametrize("refl_in, order_in, rule_width_in, dir_in",
                             [(True, -2, 5., (1, 0, 0)),
                              (False, 3, 2.5, (0, 1, 0))])
    def test_grating_param_rw(self, rx_setup, refl_in, order_in, rule_width_in, dir_in) -> None:
        """Cmp. Grating def. on Srfs  against DB check -- Param"""

        tol = rx_setup.tol
        for srf, (gr_refl, gr_order, gr_rw, gr_dir) in rx_setup.rx_settings['Grating'].items():

            # cmp against db ()
            refl, rule_width, diff_order, rule_dir = _lib.elt_grating_params(srf)  # refl, rule_width, diff_order, rule_dir =

            assert gr_refl == refl
            assert gr_rw == rule_width
            assert gr_order == diff_order
            gr_dir = np.array(gr_dir, dtype=float)
            gr_dir /= np.linalg.norm(gr_dir)
            np.testing.assert_allclose(rule_dir, gr_dir, tol["rel"], tol["abs"])


            # write
            _lib.elt_grating_params(srf, reflective=refl_in, rule_width=rule_width_in,
                                    diff_order=order_in, rule_dir=dir_in)
            refl_cmp, rule_width_cmp, diff_order_cmp, dir_cmp = _lib.elt_grating_params(srf)

            assert refl_cmp == refl_in
            assert rule_width_cmp == rule_width_in
            assert diff_order_cmp == order_in
            dir_in = np.array(dir_in, dtype=float)
            dir_in /= np.linalg.norm(dir_in)
            np.testing.assert_allclose(dir_cmp, dir_in, tol["rel"], tol["abs"])

            # restore
            _lib.elt_grating_params(srf, reflective=refl, rule_width=rule_width,
                                    diff_order=diff_order, rule_dir=rule_dir)

