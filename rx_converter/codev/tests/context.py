# from __future__ import absolute_import

# https://docs.python-guide.org/writing/structure/
# https://dev.to/codemouse92/dead-simple-python-project-structure-and-imports-38c6
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import win32com.client

sys.path.insert(0, str(Path(".").absolute().parent.parent.parent / 'pymacos' / 'src'))
# import pymacos.macos as pymacos
import pymacos

PRJ_PATH = Path('.').absolute()
os.chdir(str(PRJ_PATH))


# ---------------------------------------------------------------------
#  define pass / fail tolerances
# ---------------------------------------------------------------------
# pass/fail threshold for data comparisons: abs(actual - desired) > atol + rtol * abs(desired)
TOL = {'P': (1e-10, 1e-10),          # max Positional  rel. & abs. error
       'r': (1e-13, 1e-13),          # max Directional rel. & abs. error
       'L': (1e-11, 1e-11),          # max Path-Length rel. & abs. error
       'v': (1e-15, 1e-15),          # max value       rel. & abs. error
       'eps': (np.finfo(float).eps)*2} # eps value : 2.2204460492503131e-16 for float64


# --------------------------------------------------------
# CODE V
# --------------------------------------------------------

class CodeVSession:
    cvv = None

    def __init__(self, wrk_path:[str, Path], setup_cmd:str='', version_id:str='.115_SR1'):
        print("Start CodeV Session")
        self.cv = win32com.client.Dispatch(f"CodeV.Application{version_id}")
        self.cv.StartCodeV()
        self.cv.Wait(200)
        # work directory
        # self.cv.StartingDirectory = wrk_path
        self.cv.Command(f"cd '{wrk_path}'")
        # disables (REC No) recording of data in the CODE V recovery file
        self.cv.Command('rec no')
        self.cv.Command('DDM M; SEE 0')   # set lens units to "mm" and set seed 0 for random number generator
        print(f'Version     : {self.cv.CodeVVersion}')
        print(f'working dir : {self.cv.EvaluateExpression("(cd)")}')
        self.cvv = self.cv.EvaluateExpression("(cvv)")
        # initialise
        # self.cv.Command(f'cd "{wrk_path}"')
        if not (setup_cmd == ''):
            self.cv.Command(setup_cmd)

    def wait(self, time):
        self.cv.Wait(time)

    def load(self, rx):
        self.cmd(f"in '{rx}'")

    def cmd(self, cv_cmd):
        return self.cv.Command(cv_cmd)

    def eval(self, expression):
        return self.cv.EvaluateExpression(expression)

    def rms_wfe(self, izoo=1, iwl=1, nrd=51, focus_pos=0):
        # rms @ nom. Pos. (no focus shift) for all FoV at given zoom Pos
        com_return = self.cv.RMSWE(izoo, iwl, nrd, focus_pos)  # fct with dictionary Item:
        return com_return("Output")[0][:-1]  # (rm last entry, which is avg. over all FoV)

    def mtf_fov(self, izoo=1, ifov=1, frq=12.5, azimuth=0., nrd=221, geo=0, sqw=0, through_focus=0):
        """
        # (0) =='DIF', (1) == 'GEO'
        # (0) == 'SIN',  (1) == 'SQW'  Sin-/Square-Wave Response
        # azimuth: 0 for Y/Radial, 90 for X/Tangential
         Output(1): Modulation
         Output(2): Phase (degrees)
         Output(3): Analytic diffraction limit value
         Output(4): Actual diffraction limit value
         Output(5): Illumination for unit brightness
         Output(6): Number of rays traced (in convolved pupil for diffraction
        """
        com_return = self.cv.MTF_1FLD(izoo, ifov, frq, azimuth, nrd, geo, sqw, through_focus)
        return com_return("Output")

    def stop(self):
        self.cv.StopCodeV()

    def sag(self, isrf, izoo, x, y):
        return self.cv.sagf(isrf, izoo, x, y)

    def transform(self, srf, gsrf, zoo=1):
        com_return = self.cv.Transform(srf, gsrf, zoo)
        return np.asarray(com_return("Output"), dtype=np.float64).reshape(4, -1)

    def transform_raw(self, srf, gsrf, zoo=1):
        com_return = self.cv.Transform(srf, gsrf, zoo)
        return np.asarray(com_return("Output"), dtype=np.float64)

    def buffer_to_array(self, buf:int, start_row:int, end_row:int, start_col:int, end_col:int, transpose=0):
        """BUFFER_TO_ARRAY(bufNum As Long, startRow As Long, endRow As
            Long, startCol As Long, endCol As Long, transpose As Long) As Object"""
        com_return = self.cv.BUFFER_TO_ARRAY(buf, start_row, end_row, start_col, end_col, transpose)
        return np.asarray(com_return("Output"), dtype=np.float64)

    def normradius(self, wvl_id:int, fov_id:int, zoo_id:int,
                   n_pol_type:int=0, fit_type:int=1, zrn_type:int=0):
        return self.cv.NORMRADIUS(wvl_id, fov_id, zoo_id, n_pol_type,
                                  fit_type, zrn_type)

    def zernikegq(self, wvl_id:int, fov_id:int, zoo_id:int, zcoef_id:int,
                n_coefs:int, pupil_type:int=2, n_pol_type:int=0,
                fit_type:int=1, zrn_type:int=0):
        """Returns the computed Zernike Coefficient
        :param wvl_id:
        :param fov_id:
        :param zoo_id:
        :param n_coef_num:
        :param n_num terms: Number of Zernike terms to use in computing the coefficient
        :param pupil_type: (0) ENP, (1) EXP,  (2) EXS
        :param n_pol_type: int = 0
        :param fit_type: (0) intensity, (1) Phase
        :param zrn_type: (0) ZFR,  (1) ZRN
        :return:
        """
        return self.cv.ZERNIKEGQ(wvl_id, fov_id, zoo_id, zcoef_id,
                n_coefs, pupil_type, n_pol_type, fit_type, zrn_type)

    def zernike(self, wvl_id:int, fov_id:int, zoo_id:int, zcoef_id:int,
                n_rays:int,
                n_coefs:int, pupil_type:int=2, n_pol_type:int=0,
                fit_type:int=1, zrn_type:int=0):
        """Returns the computed Zernike Coefficient
        :param wvl_id:
        :param fov_id:
        :param zoo_id:
        :param n_coef_num:
        :param n_num terms: Number of Zernike terms to use in computing the coefficient
        :param pupil_type: (0) ENP, (1) EXP, (2) EXS
        :param n_pol_type: int = 0
        :param fit_type: (0) intensity, (1) Phase, (2) PMA
        :param zrn_type: (0) ZFR,  (1) ZRN
        :return:
        """
        return self.cv.ZERNIKE(wvl_id, fov_id, zoo_id, zcoef_id, n_rays,
                n_coefs, pupil_type, n_pol_type, fit_type, zrn_type)

    @property
    def n_zoo(self):
        return self.cv.ZoomCount

    @property
    def n_fov(self):
        return self.cv.FieldCount

    @property
    def n_srf(self):
        return self.cv.SurfaceCount

    @property
    def n_wvl(self):
        return self.cv.WavelengthCount

    def test_ray_trace(self,
                       fname: Path | str,
                       n_sampling: int = 5,
                       i_wl: int = 1,
                       i_fov: int = 0,
                       i_zoo: int = 1,
                       i_buffer: int = 100) -> tuple[Any, int, int, int]:

        # compare rays that traced to Exit Port
        #    BLS � For last single ray, returns surface number at
        #          which ray is first blocked by an aperture or
        #          obscuration
        #        <0 = Ray is blocked by obscuration
        #        >0 = Ray is blocked by aperture
        #         0 = Ray is not blocked
        #
        #    RER � For last single ray, return ray error flag
        #         0 = Trace successful
        #        >0 = Value is surface number where
        #             failure occurred
        #
        #        <0 = Ray failed when corresponding
        #             chief ray (RSI only) failed to
        #             trace through the center of the
        #             stop; |RER| is surface number
        #             where failure occurred

        # load CV: Rx
        if len(fname)>0:
            self.cmd(f"in {fname}")

        # extract ray trace info.
        origin     = int(self.eval("(slb s'origin')"))  # Glb ref (0,0,0) location
        entry_port = int(self.eval("(slb s'SrcSrf')"))  # Entry Port Srf. ID
        exit_port  = int(self.eval("(slb s'RefSrf')"))  # Exit  Port Srf. ID

        self.cmd((f"in trace_Fov {origin} {entry_port} {exit_port} "
                      f"{n_sampling} {i_wl} {i_fov} {i_zoo} {i_buffer}"))
        end_row = int(self.eval(f"(buf.lst B{i_buffer})"))

        # max_col = int(cv_setup("(BUF.MXJ B100)"))
        data = self.buffer_to_array(buf=100, start_row=4, end_row=end_row,
                                    start_col=1, end_col=15, transpose=0)

        rx_data = {'n_rays': data.shape[0],
                   'ray_in': data[:, :6],
                   'ray_out': data[:, 6:12],
                   'opl': data[:, 12],
                   'BLS': data[:, 13],
                   'RER': data[:, 14],
                   'origin': origin,
                   'entry_port': entry_port,
                   'exit_port': exit_port
                   }

        return rx_data

    def cv2macos(self,
                 file_macos: Path,
                 i_origin: int = 1,
                 i_wl: int = 1,
                 i_fov: int = 1,
                 i_zoo: int = 1,
                 ) -> None:
        # ---------- conversion
        if file_macos.exists():
            file_macos.unlink()
        # [glb srf, zoo, fov, wvl, fid, dummy, macos srf]
        self.cmd(fr"in cv2macos {i_origin} {i_zoo} {i_fov} {i_wl} '{file_macos}' 1 1")


cv_session = CodeVSession(PRJ_PATH, setup_cmd='in setup.seq', version_id='12')

@pytest.fixture(scope="session")
def cv_setup(request):
    # print("Setting up resource")
    # note: placing "CodeVSession" here will cause a crash

    def finalizer():
        print("\nRunning finalizer cleanup")
        try:
            print(">>> stopping codev ")
            cv_session.stop()

        except Exception as e:
            print(f"Error in finalizer: {e}")

    request.addfinalizer(finalizer)

    # The request.addfinalizer() method within a fixture allows you
    # to register a function to be called at the end of the fixture's
    # scope, regardless of whether the test passed or failed.
    # This can be useful for ensuring critical cleanup occurs.
    return cv_session


# @pytest.fixture(scope="session", autouse=True)
# def cv_session():
#     return CodeVSession(prj_path, setup_cmd='', version_id='12')





# # decorator: change ray-tracing wavelength for ray-trace comparisons
# def set_wavelength(func):
#     """ changes the source wavelength """
#     def set_trace_wavelength(*args, **kwargs):
#         pymacos.src_wvl(args[0]*1e-6)  # upd. trace wavelength [nm] => [mm]
#         func(*args, **kwargs)                          # call test definition (no args needed)

#     return set_trace_wavelength


# # decorator: change ray-tracing wavelength for ray-trace comparisons
# def set_wavelength_no_args(func):
#     """ changes the source wavelength """
#     def set_trace_wavelength(*args, **kwargs):
#         pymacos.src_wvl(args[0]*1e-6)  # upd. trace wavelength [nm] => [mm]
#         func()                         # call test definition (no args needed)
#     return set_trace_wavelength


# ## decorator: change ray-tracing wavelength
# #def set_wavelength(func): #, *args, **kwargs):
# #    """ changes the source wavelength """
# #    def set_trace_wavelength(*args, **kwargs):
# #        pymacos.src_wvl(args[0]*1e-6)  # upd. trace wavelength [nm] => [mm]
# #        func(*args, **kwargs)          # call test definition (no args needed)
# #
# #    return set_trace_wavelength


def rx_path(rx: Path, as_str:bool = False) -> Path | str:
    """Create abs. path to Rx and check if it exists

    Args:
        rx (Path): path to Rx

    Returns:
        Path: abs. path to Rx
    """
    rx_ = (Path('.') / 'Rx' / rx).resolve()
    if not rx_.is_file():
        raise FileExistsError(f"{rx} was not found in ./Rx/ dir.")

    return str(rx_) if as_str else rx_



def rmTTP(opd):
    """
        [mx,my] = size(OPD);
        i       = find(OPD~=0);   % replaced find(...) with logical <<< faster but does not work sometimes WHY?
        n       = length(i);
        o       = OPD(i);
        %
        % remove tip/tilt & piston
        %
        [X,Y] = meshgrid(-(mx-1)/2:(mx-1)/2,-(my-1)/2:(my-1)/2);
        Tx    = X(i)/norm(X(i));
        Ty    = Y(i)/norm(X(i));
        mTx   = mean(Tx);
        mTy   = mean(Ty);

        sTx = sum((Tx-mTx).^2);
        sTy = sum((Ty-mTy).^2);

    """
    mx, my = opd.shape
    i = opd != 0e0
    o = opd[i]
    n = np.size(o)

    # remove tip/tilt & piston
    X, Y = np.mgrid[-(mx - 1) / 2:(mx - 1) / 2 + 1, -(my - 1) / 2:(my - 1) / 2 + 1]
    Tx = X[i] / np.sqrt(np.sum(X[i] ** 2))
    Ty = Y[i] / np.sqrt(np.sum(Y[i] ** 2))
    mTx = np.mean(Tx)
    mTy = np.mean(Ty)

    sTx = np.sum((Tx - mTx) ** 2)
    sTy = np.sum((Ty - mTy) ** 2)

    fx = (np.dot(Tx, o) - n * np.mean(o) * mTx) / sTx
    o = o - fx * Tx
    fy = (np.dot(Ty, o) - n * np.mean(o) * mTy) / sTy
    o = o - fy * Ty
    o = o - np.mean(o)

    opd_ = opd.copy()
    opd_[i] = o
    return opd_
