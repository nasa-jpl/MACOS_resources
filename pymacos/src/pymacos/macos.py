# ------------------------------------------------------------------------------
#
# Provides a Python API to SMACOS
#
# ------------------------------------------------------------------------------
import functools
import math
import warnings
from pathlib import Path
from typing import Any, List, NewType, Tuple, TypeVar, TypeVarTuple
from dataclasses import dataclass

import numpy as np
from numpy._typing import ArrayLike, NDArray

from . import pymacosf90 as lib

_T = TypeVar("_T", bound=np.generic, covariant=True)
Vector = np.ndarray[Tuple[int], np.dtype[_T]]
Matrix = np.ndarray[Tuple[int, int], np.dtype[_T]]
Tensor = np.ndarray[Tuple[int, ...], np.dtype[_T]]

Integer = int | np.integer[Any] | np.ndarray[int, np.dtype[_T]]
Integers = Integer | Tuple[int] | Vector[np.int32]

Floats = float | np.float64 | Tuple[float] | Vector[np.float64]

Surface = int | Tuple[int] | np.int32 | Vector[np.int32]                       # int | Tuple[int] | np.ndarray[int]
Position = Tuple[float] | Vector[np.float64] | Matrix[np.float64]
Direction = Tuple[float] | Vector[np.float64] | Matrix[np.float64]
Parameter = float | np.float64 | Tuple[float] | Vector[np.float64]


# Sentinel used by zrn_freeform (and any other set-some-leave-others
# wrappers) to distinguish "preserve current value" (default) from
# "explicitly clear/wipe" (None).
_PRESERVE = object()
Index = int | np.int32 | Tuple[int] | Vector[np.int32]

_floatType = (float, np.float32, np.float64)
_integerType = (int, np.intp, np.int8, np.int16, np.int32, np.int64)

# ------------------------------------------------------------------------------
# pymacos status information
# ------------------------------------------------------------------------------

_SYSINIT   = False    # status if pymacos was loaded
_isRx      = False    # status if a Rx is loaded
_NELT      = np.nan   # contains number of elements after Rx is loaded
# _n_srfs    = np.nan   # ditto
_MODELSIZE = np.nan   # defines system parameters linked to model size


# ------------------------------------------------------------------------------
# external tracking
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#
# internal functions
#
# ------------------------------------------------------------------------------


def _chk_macos_and_rx_loaded() -> None:
    """entry point check: PASS if both MACOS and Rx are loaded"""
    if not _SYSINIT:
        raise Exception('MACOS is not yet initialised')
    elif not _isRx or np.isnan(_NELT):
        raise Exception('MACOS Rx not loaded')


def _chk_if_macos_and_rx_loaded(func):
    """Decorator for checking if MACOS is initialised and Rx loaded
    """
    @functools.wraps(func)
    def wrapper_check(*args, **kwargs):
        if not _SYSINIT:
            raise Exception('MACOS is not yet initialised')
        elif not _isRx or np.isnan(_NELT):
            raise Exception('MACOS Rx not loaded')

        return func(*args, **kwargs)
    return wrapper_check


def isinteger(iElt: _integerType) -> bool:
    return isinstance(iElt, _integerType)


def isfloat(x: _floatType) -> bool:
    return isinstance(x, _floatType)


def isNumeric(x) -> bool:
    return isinstance(x, (*_floatType, *_integerType))


def _map_Elt(iElt, exclude=False, multi=False, max_rows=None):
    """
    Check & remap input parameter "iElt" for loaded system (checked)

    :param     iElt:   [MxN,D] (tuple,list,ndarray): Elt.ID   (Range: -nElt < iElt[j] <= nElt)
    :param  exclude:   [1x1,L] (default = False) if set, last 3 elements (detector) are excluded
    :param    multi:   [1x1,L] (default = False) is set, allow multiple occurrences

    :return    iElt:   [MxN,I]            (ndarray): Elt. ID  (Range: 0 < iElt <= nElt)
    """

    # basic checks
    if not isinstance(iElt, (*_floatType, *_integerType, tuple, list, np.ndarray)):
        raise TypeError("iElt must be a finite scalar, tuple, list or numpy.ndarray")

    elif isinstance(iElt, (tuple, list)):
        if not np.all([isNumeric(i) for i in iElt]):
            raise TypeError("iElt values in 'tuple' or 'list' must be finite scalar")

    if np.size(iElt) == 0:
        raise ValueError("iElt cannot be empty")

    # warn for float to int conversion
    if np.any(np.int32(iElt) != iElt):
        warnings.warn("\n => iElt contains a non-integer type ===> type convert from 'float' to 'int32'")

    # type conversion
    jElt = np.asarray_chkfinite(iElt, dtype=np.int32)

    if jElt.shape == ():
        jElt = jElt.reshape(1)

    # check row dimension
    if max_rows is not None:
        if jElt.ndim > 1:
            if jElt.shape[0] > max_rows:
                raise TypeError(f'iElt exceeded max. row dimension of {max_rows}')

    # Chk iElt range
    if np.any((jElt < -_NELT) | (jElt > _NELT)):   # note: use bitwise or operator (|) not logical 'or'
        raise ValueError("iElt contains value(s) larger than permitted: Elt range: -nElt < iElt <= nElt")

    # remapping to positive range values:  0 < iElt <= nElt
    #            1,  2,  3,  4,  5,  6,  7,  8,  9, 10
    #          -10, -9, -8, -7, -6, -5, -4, -3, -2, -1
    jElt[jElt < 0] += _NELT+1

    # check if in exclude
    if exclude:
        if np.any(jElt > _NELT-3):
            raise ValueError("iElt contains excluded detector elements")

    # check for multiple occurrence: see https://www.peterbe.com/plog/fastest-way-to-uniquify-a-list-in-python-3.6
    if not multi:
        if jElt.size != len(set(jElt.flatten())):
            raise ValueError("Multiple occurrence of the same Element")

    return jElt



# ------------------------------------------------------------------------------------------- ToDo: tests it
def _chk_values_pos(srf: list[int] | int | np.ndarray[int],
                    val_xyz: list[float] | np.ndarray[float],
                    K: int = 3
                    ) -> tuple[list[int] | int | np.ndarray[int],
                               list[float] | np.ndarray[float]]:
    """Checks the position (orientation & location) values (RptElt, VptElt, psiElt)
       against the element IDs given in 'srf'.

    Args:
        srf (list[int] | int | np.ndarray):  [M x N] array
            Element ID, 1D-array (Range: -nElt < srf[j] <= nElt)
            where neg. values are referenced with respect to the last
            surface.

        val_xyz (list[float] | np.ndarray[float]): [K x N] array
            defines the (x,y,z) values per element (finite)

        K (int, optional): Defaults to 3.
            defines required axis=0 size

    Raises:
        ValueError: - not finite
                    - Dimension miss-match between values and srfs

    Returns:
        srf (tuple[list[int] | int | np.ndarray[int])

        vpt: (list[float] | np.ndarray[float]])
    """

    val_xyz = np.asarray_chkfinite(val_xyz, dtype=float)
    srf = np.asarray_chkfinite(srf, dtype=np.int32)

    if val_xyz.size == 0 or srf.size == 0:
        raise ValueError('assigned values cannot be empty must be a [KxN] array')

    elif srf.size == 1 and val_xyz.size == K:                          # iElt [1,1]  Vpt [K,1]
        srf.shape, val_xyz.shape = (1, 1), (K, 1)

    elif 1 in srf.shape:
        if val_xyz.size == K:                                          # iElt [M,1]  Vpt [K,1]
            srf.shape, val_xyz.shape = (-1, 1), (K, 1)
        elif val_xyz.size == K * srf.size and val_xyz.ndim == 2:       # iElt [1,N]  Vpt [K,N]
            srf.shape, val_xyz.shape = (1, -1), (K, -1)

    elif not (srf.ndim == 2 and val_xyz.shape == (K, srf.shape[1])):   # iElt [1,N]  Vpt [K,N]
        raise ValueError('Mismatch between srf and value data structure')

    return srf, val_xyz

# ------------------------------------------------------------------------------------------- ToDo: test it
def _chk_values_2d(array_2d, M=3, N=1):

    array_2d = np.asarray_chkfinite(array_2d, dtype=float)

    # if not isinstance(array_2d, np.ndarray):
    #     raise TypeError("expected a 2D ndarray")

    if array_2d.ndim != 2:
        raise ValueError("expected a 2D ndarray")

    # elif not (np.all(np.isfinite(array_2d)) and np.all(np.isreal(array_2d))):
    #     raise ValueError("array contains non-finite or complex values")

    elif array_2d.shape[0] != M or array_2d.shape[1] != N:
        raise ValueError(f"expected a 2D ndarray with dimension {M:}x{N:}")


# ------------------------------------------------------------------------------------------- ToDo: test it
def _chk_values_1d(vector, size=3, row=True):
    """
    Checks that the float vector is a 1D-array of type (tuple, list, numpy.ndarray) with finite
    values and length 'size'. If needed, the vector is converted to numpy.ndarray with row (default) or
    column shape unless vector is a scalar.

    :param   vector:     [Nx1,D] float vector to check and format
    :param   size:       [1x1,I] expected length (N) of array, throws an error if not
    :param   row:        [1x1,L] if true (=1), shape of output vector is [1xN] else [Nx1]

    :return: vector              2D-array [Nx1] (if row else) [1xN]
    """
    # scalar
    if np.isscalar(vector):
        if isNumeric(vector):
            if np.isreal(vector) and np.isfinite(vector):
                return np.atleast_2d(np.float64(vector))
        else:
            raise ValueError("provided scalar is not valid")

    # tuple / list
    elif isinstance(vector, (tuple, list)):
        if size > 0:
            if len(vector) != size:
                raise ValueError(f"list or tuple has an incorrect length {len(vector)} <> {size}")

        if np.any([(not isNumeric(v)) for v in vector]) == True:
            raise ValueError("list or tuple contains non-numeric values")

        elif not (np.all(np.isfinite(vector)) and np.all(np.isreal(vector))):
            raise ValueError("list or tuple contains non-finite values")

        elif size==1:
            return vector[0]
            #if not(isNumeric(vector) and np.isreal(vector) and np.isfinite(vector)):
            #    raise ValueError("single entry in 'list' or 'tuple' is not a real, finite numeric value")
        elif row:
            return np.asarray_chkfinite(vector, dtype=np.float64).reshape((1, -1))
        else:
            return np.asarray_chkfinite(vector, dtype=np.float64).reshape((-1, 1))

    # numpy.ndarray
    elif isinstance(vector, np.ndarray):
        if size > 0:
            if vector.shape not in ((size,), (1, size), (size, 1)):
                raise ValueError(f"vector must be a 1D-array of length {size}")

        if not (np.all(np.isfinite(vector)) and np.all(np.isreal(vector))):
            raise ValueError("array contains non-finite or complex value(s)")

        elif row:
            return vector.reshape(1, -1).copy()
        else:
            return vector.reshape(-1, 1).copy()

    else:
        raise TypeError("input vector is not a 1D-array of type tuple, list or numpy.ndarray")


# ------------------------------------------------------------------------------
#     [ ] model_size
#     [ ] init
#     [ ] load
#     [ ] n_srfs
#     [ ] has_rx
#     [ ] rx_modified
# ------------------------------------------------------------------------------


def model_size() -> int:
    """Returns the MACOS model size

    Returns:
        int: model size -1, 128, 256, 512, 1024, 2048 or 4096
             where -1 indicates that MACOS has not yet been initialised
    """
    # currrent_macos_model_size is a FUNCTION and was never wrapped for
    # f2py (imported but no shim), so call the model_size_get subroutine
    # wrapper instead.
    return int(lib.api.model_size_get())


def init(model_size: int = 512) -> None:
    """Initialises MACOS and defines model size

    Args:
        model_size (int, optional): must be: 128, 256, 512, 1024, 2048 or 4096.
                                    Defaults to 512.

    Raises:
        ValueError: if model_size was invalid
        Exception:  when MACOS failed to initialise

    Returns:
        bool: True if initialisation was successful; False, otherwise

    Note:
        MACOS configuration linked to "macos_param.txt"
    """

    global _SYSINIT, _MODELSIZE

    # input checks
    # if not(isinteger(modelsize) or isfloat(modelsize)):
    #     raise ValueError("param 'modelsize' must be float or integer")

    msize = np.int32(model_size)
    if msize not in (128, 256, 512, 1024, 2048, 4096):
        raise ValueError("the model size must be member of (128, 256, 512, 1024, 2048, 4096)")

    # initialise
    if not lib.api.init(msize):
        raise Exception('unable to initialize MACOS')

    _SYSINIT = True
    _MODELSIZE = msize


def load(macos_rx: Path | str) -> int:
    """Load an optical prescription (Rx) into MACOS.

    Args:
        macos_rx (Path | str):
          file (path + file name) where total length cannot exceed
          128 characters (MACOS limit). The MACOS extension '.in'
          is inherently assumed, i.e., can be left out.

    Raises:
        FileExistsError: if Rx is not found
        ValueError:      if length of file name > 128 Characters
        Exception:       MACOS internal or MACOS was not yet initialised

    Returns:
        int: Number of optical elements defined in Rx (n_srf)
    """
    global _NELT, _isRx

    # status & input check
    if not _SYSINIT:
        raise Exception("MACOS is not yet initialised")

    macos_rx = Path(macos_rx).with_suffix('.in')
    macos_rx_str = macos_rx.with_suffix('').__str__()

    if not macos_rx.is_file():
        raise FileExistsError(f"Rx file '{macos_rx}' not found")
    elif len(macos_rx_str) > 128:
        raise ValueError("Path + FileName greater than max. permitted by MACOS (>128)")

    # MACOS Bug Tmp Fix: ensure file name does NOT contain extension '.in'
    #                    ==> otherwise we exercise a MACOS Bug (endless loop)

    _NELT, _isRx = np.nan, False   # reset
    ok, n_srf = lib.api.load_rx(macos_rx_str)

    if (not ok) or n_srf == 0:
        raise Exception('MACOS was unable to load Rx')

    # update status
    _NELT = n_srf
    _isRx = True

    return n_srf


def save(rx: Path | str,
         overwrite: bool = True) -> None:
    """Save current optical prescription (Rx) state to file.

    Args:
        rx (Path | str):
          file (path + file name) where total length cannot exceed
          128 characters (MACOS limit). The MACOS extension '.in'
          is inherently assumed, i.e., can be left out.

        overwrite (bool):
          if set, it will overwrite existing file (default).

    Raises:
        FileExistsError: if Rx path is not found
        ValueError:      if length of file name > 128 Characters
        Exception:       MACOS was not yet initialised or no Rx loaded
    """
    _chk_macos_and_rx_loaded()

    rx = Path(rx)
    if not rx.parent.is_dir():
        raise FileExistsError(f"path '{str(rx)}' not found")

    if not overwrite:
        if rx.with_suffix('.in').is_file():
            raise FileExistsError(f"{rx.with_suffix('.in')} exists")

    rx = Path(rx).with_suffix('')  # macos will add the suffix
    if len(str(rx)) > 128:
        raise ValueError("Path + FileName greater than >128 (MACOS))")

    if not lib.api.save_rx(str(rx)):
        raise Exception("MACOS threw an Exception")


def has_rx() -> bool:
    """returns Rx load status

    Returns:
        bool: (True) if Rx loaded; (False) otherwise
    """
    return _isRx


def num_elt() -> int:
    """returns number of elements loaded (if Rx is present)

    Raises:
        Exception: if Rx is not present

    Returns:
        int: Number of elements loaded
    """
    _chk_macos_and_rx_loaded()  # pymacos & Rx loaded

    return lib.api.n_elt()


def rx_modified():
    """Reset ray-tracing state

    Submits a "Rx modified" cmd. to MACOS to reset ray-trace dependent
    parameters, which is recommended after a Rx modification, i.e.,
    perturbElt, define VptElt, ...

    Raises:
        Exception: MACOS execution failure

    Note:
        traceChiefRay contains already a "Rx modified" cmd.
    """
    _chk_macos_and_rx_loaded()  # pymacos & Rx loaded

    if not lib.api.modified_rx():
        raise Exception("failed to reset MACOS status")


# ------------------------------------------------------------------------------
# [ ] Source
# ------------------------------------------------------------------------------
#     [ ] src_info             get Src. Information
#     [ ] src_sampling     set/get Src. Sampling
#     [ ] src_size         set/get Src. Size (Aperture & Abscuration)
#     [ ] src_wvl          set/get Src. Wavelength
#     [ ] src_fov          set/get Src. FoV
#     [ ] src_finite           get Src. type (Point or Collimated)
#     [ ] src_csys         set/get Src. Coord. Sys Pose
#
#     [ ] getActivePointSrc    [ ] setActivePointSrc      [Source]: set/get ray bundle origin information (for Point Source)
# ------------------------------------------------------------------------------


def src_info() -> Tuple:
    """Retrieve Source def. information: shape, position & wavelength

    Raises:
        Exception: MACOS Triggered

    Returns:
        tuple: (src_dist, src_pos, src_dir, is_finite, wvl, src_ape, src_obs, base_unit, wave_unit)

        src_dist: np.float64
                Distance from Src. Position to Spherical wavefront pos.
                - 0 < src_dist <=  1e10: converging wave (to   Pt. Src.)
                - 0 > src_dist >= -1e10: diverging  wave (from Pt. Src.)
                with src_dist = 1e22: collimated Beam (Col. Src.)

        src_pos: np.ndarray
                Source Position = [x, y, z]
                - if Pt. Src.: src_pos <= ChfRayPos + src_dist*src_dir
                - if Col.Src.: src_pos <= ChfRayPos

        src_dir: np.ndarray
                Source Pointing == ChfRayDir = [L, M, N])

        is_finite: bool
                (True) if |src_dist| < 1d10, i.e., finite object

        wvl: np.float64
                Source Wavelength in 'WaveUnits'

        src_ape:  np.float64
                Source Aperture were:  src_obs < src_ape > 0.0
                - if Pt. Src. => N.A. of beam
                - if Col.Src. => Beam Diameter in BaseUnits

        src_obs:  np.float64
                Source Obscuration were:  0.0 <= src_obs < src_ape
                - if Pt. Src. => N.A. of beam
                - if Col.Src. => Beam Diameter in BaseUnits

      base_unit: str
            Length Unit as defined in Rx:  ('m', 'cm', 'mm', 'in')

      wave_unit: str
            Wavefront Unit as defined in Rx:
                ('m', 'cm', 'mm', 'um', 'nm', 'A', 'in')

    """
    _chk_macos_and_rx_loaded()

    (ok, src_dist, src_pos, src_dir, is_finite, wl, src_ape,
     src_obs, BaseUnitID, WaveUnitID) = lib.api.src_info()

    if not ok:
        raise Exception("MACOS failed to retrieve Source Information")
    else:
        base_units = ('none', 'm', 'cm', 'mm', 'in')[BaseUnitID]
        wave_units = ('none', 'm', 'cm', 'mm', 'um', 'nm', 'A', 'in')[WaveUnitID]

        return (src_dist, src_pos.reshape((1,3)), src_dir.reshape((1,3)),
                is_finite, wl, src_ape, src_obs, base_units, wave_units)


def src_sampling(n_gridpts: int | np.int32 | None = None) -> None | np.int32:
    """Get / Set Source Sampling (nGridPts) Grid Pts.

    Args:
        n_gridpts (int | None, optional): Defaults to None.
                Source Sampling Points where the max. is defined by the
                MACOS model size and may/may not be limited within
                'macos_param.txt', the config. file.

    Raises:
        Exception:  MACOS Triggered
        ValueError: (n_gridpts < 3) or (n_gridpts > model_size)

    Returns:
        None | np.int32: Source sampling
    """

    """
    Parameters
    ----------
    n_gridpts : None or int, optional
                Source Sampling Points where the max. is defined by the
                MACOS model size and may/may not be limited within
                'macos_param.txt', the config. file.

    Returns
    -------
    None or int
          Returns Number of Grid Points to sample the source

    """
    _chk_macos_and_rx_loaded()

    if n_gridpts is None:
        ok, n_gridpts = lib.api.get_src_sampling()
        if not ok:
            raise Exception("failure occurred in 'src_sampling'")
        return int(n_gridpts)

    # define source sampling
    n_gridpts = np.asarray_chkfinite(n_gridpts, dtype=np.int32)
    # np.int32(_chk_values_1d(n_gridpts, size=1)[0])

    if (n_gridpts < 3) or (n_gridpts > _MODELSIZE):
        raise ValueError("'nGridPts' must be an integer within range [3, ... , {_MODELSIZE}]")

    elif not lib.api.set_src_sampling(n_gridpts):
        raise Exception("MACOS: exception arose")

    else:
        n_gridpts_ = lib.api.get_src_sampling()[1]
        if n_gridpts_ != n_gridpts:
            warnings.warn(f"\n => 'nGridPts' was set to {n_gridpts_}")


def src_size(ape: None | Parameter = None,
             obs: None | Parameter = None) -> Tuple[np.float64, np.float64] | None:
    """Set / get Source Aperture and/or Source Obscuration

    If no input is provided the values are returned; otherwise, the
    Aperture and/or Obscuration is defined.

    Args:
        ape (float | None, optional): Defaults to None.
            Source Aperture must be: Obscuration < Aperture > 0.0
                - if Pt. Src. => N.A. of beam
                - if Col.Src. => Beam Diameter in BaseUnits

        obs (float | None, optional): Defaults to None.
            Source Obscuration were:  0.0 <= Obscuration < Aperture
                - if Pt. Src. => N.A. of beam
                - if Col.Src. => Beam Diameter in BaseUnits

    Raises:
        Exception:   MACOS Triggered
        ValueError:  0e0 <= ape <= obs

    Returns:
        tuple[float] | None:

            ape : None or tuple of float
                Aperture value per above definition

            obs : None or tuple of float
                Obscuration value per above definition
    """
    _chk_macos_and_rx_loaded()

    if ape is None and obs is None:
        ape = np.array(0e0)
        obs = np.array(0e0)
        if not lib.api.src_size(ape, obs, 0):
            raise Exception("MACOS: Communication failed")
        return ape, obs

    else:

        if ape is None or obs is None:
            ape_rx, obs_rx = src_size()    # get current values

        if ape is None:
            ape, obs = ape_rx, np.asarray_chkfinite(obs, dtype=float)
        elif obs is None:
            ape, obs = np.asarray_chkfinite(ape, dtype=float), obs_rx
        else:
            ape = np.asarray_chkfinite(ape, dtype=float)
            obs = np.asarray_chkfinite(obs, dtype=float)

        if ape <= 0e0:
            raise ValueError("'Aperture' is less than or equal to 0")
        if ape <= obs:
            raise ValueError("'Aperture' is less than or equal to Obscuration")
        if obs < 0e0:
            raise ValueError("'Obscuration' is less than 0")

        if not lib.api.src_size(ape, obs, 1):
            raise Exception("MACOS: Communication failed")


def src_csys(x_dir: None | Direction = None,
             y_dir: None | Direction = None,
             z_rot: Parameter | np.float64 = 0.,
             threshold: bool = True
             ) -> None | Tuple[Vector[np.float64], Vector[np.float64], Vector[np.float64]]:

    """set / get Source Coordinate Frame

    In MACOS: xGrid, yGrid, zGrid, where the coordinate frame changes
    with ChfRayDir = zGrid. The x- & y-axis is adjusted accordingly.

    The values will be updated / defined AFTER the first ray-trace!

    If both axes are given, the x-axis is used.

    Args:
        x_dir (Tuple[float] | Vector[np.float64] | None, optional): Defaults to None.
            [L,M,N] => x-Axis expressed in Global CSYS (1 = L^2+M^2+N^2 )

        y_dir (Tuple[float] | Vector[np.float64] | None, optional): Defaults to None.
            [L,M,N] => y-Axis expressed in Global CSYS (1 = L^2+M^2+N^2 )

        z_rot (float | Tuple[float] | np.ndarray[float], optional): Defaults to Rz = 0.0
            [rad] Rot. mag. for post. rot. about zDir = zGrid = ChfRayDir

        threshold (bool, optional): _description_. Defaults to True.

    Returns:
        x_dir : np.ndarray[Tuple[int], dtype=np.float64)
                [Lx, Ly, Lz] => Src. Coord. Frame: x-axis expressed in GCF
                MACOS: xGrid

        y_dir : np.ndarray[Tuple[int], dtype=np.float64)
                [Mx, My, Mz] => Src. Coord. Frame: y-axis expressed in GCF
                MACOS: yGrid

        z_dir : np.ndarray[Tuple[int], dtype=np.float64)
                [Nx, Ny, Nz] => Src. Coord. Frame: z-axis expressed in GCF

    Note:
        - will be re-calculated internally when Chf. Ray. changes !!!

        - will orthonormalize:
            if xAxis yDir <= cross(zDir,xDir)  else  xDir <= cross(yDir,zDir)
                     xDir <= cross(yDir,zDir)        yDir <= cross(zDir,xDir)

        - rotation will be applied afterwards about zGrid = ChfRayDir, i.e.,
                     xDir <= Rot(Rz)*xDir   and
                     yDir <= Rot(Rz)*yDir

    """
    _chk_macos_and_rx_loaded()

    if x_dir is None and y_dir is None:
        ok, xDir, yDir, zDir = lib.api.get_src_csys()

        if not ok:
            raise Exception("MACOS: failed to get Source Coord. Frame value using 'get_src_csys'")
        else:
            return xDir.reshape((3,1)), yDir.reshape((3,1)), zDir.reshape((3,1))

    elif ((x_dir is not None) or (y_dir is not None)):

        # parameter checks
        axis = x_dir if x_dir is not None else y_dir
        is_x_axis = x_dir is not None

        axis = np.asarray_chkfinite(axis, dtype=np.float64)
        z_rot = np.asarray_chkfinite(z_rot, dtype=np.float64)

        if not isinstance(threshold, bool):
            raise ValueError("'set_src_csys' requires param 'threshold' to be boolean")

        # calling Fortran f90 function
        ok, x_dir, y_dir, z_dir = lib.api.set_src_csys(axis, is_x_axis, z_rot, threshold)

        if not ok:
            raise Exception("failed to set Source Coordinate Frame value")
        else:
            return x_dir.reshape((3, 1)), y_dir.reshape((3, 1)), z_dir.reshape((3, 1))


def src_wvl(wvl: float | None = None) -> None | float:
    """set / get Source Wavelength in 'WaveUnits'

    Args:
        wvl (float | None, optional): _description_. Defaults to None.

    Raises:
        Exception:  MACOS failure
        ValueError: if Wavelength is not finite and/or <= 0

    Returns:
        float: Source Wavelength expressed in Units of 'WaveUnits'
    """
    _chk_macos_and_rx_loaded()

    if wvl is None:
        wvl_ = np.array(0, dtype=float)
        if not lib.api.src_wvl(wvl_, 0):
            raise Exception("MACOS: failed to set wavelength")
        return wvl_

    # define wavelength
    wvl_ = np.asarray_chkfinite(wvl, dtype=np.float64).squeeze()

    if wvl_.shape not in ( (), (1,), (1,1) ):
        raise ValueError("Wavelength must be a scalar")

    if wvl_ <= 0:
        raise ValueError("'wavelength' must be real, > 0 and finite")

    if not lib.api.src_wvl(wvl_, 1):
        raise Exception("failed to set wavelength")


def src_flux(flux: float | None = None) -> None | float:
    """set / get the source FLUX.

    The propagated intensity scales linearly with the source flux
    (sourcsub seeds each ray amplitude as sqrt(flux)).  Set a small
    flux on an off-axis source to inject a faint "planet" alongside the
    on-axis star, then COMPOSE the two scenes onto one detector image.
    Takes effect on the next trace/propagation.

    Args:
        flux: source flux (>= 0).  None (default) -> get current value.

    Returns:
        float current flux when called as a getter; None when setting.

    Raises:
        Exception:  MACOS failure
        ValueError: if flux is not finite and/or < 0
    """
    _chk_macos_and_rx_loaded()

    if flux is None:
        flux_ = np.array(0, dtype=float)
        if not lib.api.src_flux(flux_, 0):
            raise Exception("MACOS: failed to get source flux")
        return flux_

    flux_ = np.asarray_chkfinite(flux, dtype=np.float64).squeeze()
    if flux_.shape not in ((), (1,), (1, 1)):
        raise ValueError("flux must be a scalar")
    if flux_ < 0:
        raise ValueError("'flux' must be real, >= 0 and finite")

    if not lib.api.src_flux(flux_, 1):
        raise Exception("failed to set source flux")


def first_order_properties(srf: int | np.int32 = -1) -> dict:
    """First-order optical system properties (the SYSPROP command).

    Runs the engine EFL analysis (traces chief + marginal rays; the
    source must be at infinity) and returns the system's first-order /
    diffraction properties at focal-plane element ``srf`` (default -1 =
    last element).  This is the SAME engine computation the interactive
    SYSPROP command prints and the design layer can target.

    The pixel-based entries (``lamD_px``, ``plate_arcsec_px``,
    ``plate_px_rad``, ``dx_focal_baseunits``) require a prior
    propagation to ``srf`` (e.g. ``intensity(srf)``) to have set the
    diffraction grid pitch; they are 0.0 otherwise.

    Returns:
        dict with keys:
          'efl_baseunits'      effective focal length (BaseUnits)
          'fno'                F-number (EFL / entrance-pupil diameter)
          'dpup_m'             entrance-pupil diameter (metres)
          'obscuration'        central obscuration ratio
          'lambda_m'           wavelength (metres)
          'lamD_rad'           lambda/D (radians) -- also the source-tilt
                               offset for 1 lambda/D ("planet" placement)
          'lamD_arcsec'        lambda/D (arcsec)
          'lamD_px'            lambda/D (detector pixels; 0 pre-INT)
          'plate_arcsec_px'    plate scale (arcsec/pixel; 0 pre-INT)
          'plate_px_rad'       source tilt -> focal shift (px/rad; 0 pre-INT)
          'nyquist_baseunits'  Nyquist focal sampling (BaseUnits)
          'dx_focal_baseunits' detector pixel pitch (BaseUnits; 0 pre-INT)

    Raises:
        Exception: MACOS-side failure (e.g. EFL needs source at infinity).
    """
    _chk_macos_and_rx_loaded()

    iElt = _map_Elt(srf)
    if hasattr(iElt, '__len__'):
        iElt = int(iElt[0])

    (ok, efl, fno, dpup_m, obsc, lam_m, lamD_rad, lamD_arcsec, lamD_px,
     plate_arcsec_px, plate_px_rad, nyquist_bu,
     dx_focal_bu) = lib.api.sysprop(int(iElt))
    if not ok:
        raise Exception("MACOS: SYSPROP failed (EFL needs source at "
                        f"infinity?) at Elt {iElt}")

    return {
        'efl_baseunits':      float(efl),
        'fno':                float(fno),
        'dpup_m':             float(dpup_m),
        'obscuration':        float(obsc),
        'lambda_m':           float(lam_m),
        'lamD_rad':           float(lamD_rad),
        'lamD_arcsec':        float(lamD_arcsec),
        'lamD_px':            float(lamD_px),
        'plate_arcsec_px':    float(plate_arcsec_px),
        'plate_px_rad':       float(plate_px_rad),
        'nyquist_baseunits':  float(nyquist_bu),
        'dx_focal_baseunits': float(dx_focal_bu),
    }


def src_fov(src_pos: np.ndarray | None = None,
            src_dir: np.ndarray | None = None,
            src_dist: float | None = None) -> tuple[float, np.ndarray, np.ndarray, bool] | None:

    """Set / get active source Field-of-View (FoV) Information

    Args: To define the Source ALL must be defined

        src_pos (np.ndarray | None, optional): Defaults to None.
            Src. Position:  if Col.Src, src_pos = ChfRayPos

        src_dir (np.ndarray | None, optional): Defaults to None.
            Src. Beam Direction (= ChfRayDir) (will be normalized)

        src_dist (float | None, optional): Defaults to None.
            Distance from wavefront position to Src. Pos. (= zSource)
            note: Finite Source if |src_dist| (=|zSource|) <= 1e10

    Raises:
        Exception: MACOS Triggered
        ValueError: If not ALL parameters are defined when defining the Source

    Returns:
        tuple[float | np.ndarray | bool] | None:

        For Get Src. Information:
            src_dist      (float)            : Distance from Wave Pos to Src. Pos. (= zSource)
            src_pos  (np.ndarray) [1x3 array]: Src. Pos.:  if Col.Src, src_pos = ChfRayPos
            src_dir  (np.ndarray) [1x3 array]: Src. Beam Direction (= ChfRayDir)
            src_finite     (bool)            : if |zSource = src_dist| <= 1d10

    Note:
     - zSrc:  0 < |zSrc| <= 1d10: Pt. Src.: if zSrc<0 -> converging wave (to   Pt. Src.)
                                            if zSrc>0 -> diverging  wave (from Pt. Src.)
                  |zSrc|  > 1d10: Col.Src.
     - SrcPos: if Pt. Src.: SrcPos = ChfRayPos + zSource*ChfRayDir
               if Col.Src.: SrcPos = ChfRayPos
     - ChfRayDir = SrcDir
     - ChfRayPos = SrcPos - zSource*SrcDir
    """

    _chk_macos_and_rx_loaded()

    if (src_pos is None) and (src_dir is None) and (src_dist is None):
        ok, src_dist, src_pos, src_dir, src_finite = lib.api.get_src_fov()

        if not ok:
            raise Exception("failed to get FoV values using 'get_src_fov'")
        else:
            return src_dist, src_pos.ravel(), src_dir.ravel(), src_finite > 0

    elif (src_pos is None) or (src_dir is None) or (src_dist is None):
        raise ValueError("all parameters must be defined for Source def.")

    else:
        src_dist = np.asarray_chkfinite(src_dist, dtype=float).squeeze()
        if np.abs(src_dist == 0):
            raise ValueError("Source Distance cannot be zero")

        src_pos = np.asarray_chkfinite(src_pos, dtype=float)
        src_dir = np.asarray_chkfinite(src_dir, dtype=float)
        src_dir /= np.linalg.norm(src_dir)

        if not lib.api.set_src_fov(src_dist, src_pos, src_dir):
            raise Exception("MACOS execution error")


def getActivePointSrc():   #ToDo
    pass


def setActivePointSrc():   #ToDo
    pass


def src_finite() -> bool:
    """Returns if Source Position is finite

        0 < |zSrc| <= 1d10: Pt. Src.:
                if zSrc<0 -> converging wave (to   Pt. Src.)
                if zSrc>0 -> diverging  wave (from Pt. Src.)
            |zSrc|  > 1d10: Col.Src.

    Raises:
        Exception: MACOS triggered (Rx not loaded)

    Returns:
        bool: True if Pt. Source is finite
    """

    ok, src_finite = lib.api.src_finite()
    if not ok:
        raise Exception("MACOS: failed executing 'is_src_finite'")
    return src_finite


def sys_units() -> Tuple[str, str]:
    """Returns BaseUnits & WaveUnits as defined in Rx

    Raises:
        Exception: MACOS not initialised or Rx is not loaded

    Returns:
        list[str, str]: BaseUnits, WaveUnits
    """

    ok, base_unit_id, wave_unit_id = lib.api.sys_units()
    if not ok:
        raise Exception("MACOS: failed executing 'sys_units'")

    base_units = ('none', 'm', 'cm', 'mm', 'in')[base_unit_id]
    wave_units = ('none', 'm', 'cm', 'mm', 'um', 'nm', 'A', 'in')[wave_unit_id]
    return base_units, wave_units


# ------------------------------------------------------------------------------
# [ ] Element Pose
# ------------------------------------------------------------------------------
#     [x] elt_vpt      [Position]: set/get Elt. Vertex   Point
#     [x] elt_psi      [Position]: set/get Elt. Surface Normal
#     [x] elt_rpt      [Position]: set/get Elt. Rotation Point
#
#     [x] elt_csys     [CSYS]: set/get/del Local Coord. System
# ------------------------------------------------------------------------------


def elt_vpt(srf: Surface, vpt: None | Position = None) -> None | Position:
    """Set/Get Element vertex position(s) for specified elements

    Args:
        srf (int | Tuple[int] | np.ndarray[int], dtype=np.int32]):
           Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        vpt (list[float] | np.ndarray[float], optional):
            If provided, the Vertex Positions at the elements srf will be
            replaced. 'vpt' must have the shape [3xN] where vpt is defined
            as [ [x, y, z]_1, ..., [x, y, z]_N ] expressed in the
            global coordinate frame and N is the number of surfaces
            specified in srf. Defaults to None.

    Raises:
        Exception: MACOS triggered error

    Returns:
        elt_vpt (None | np.ndarray[float]):
            If no Option was given, it will return the Vertex Positions of
            at the surfaces defined by 'srf'; otherwise, nothing.
    """

    _chk_macos_and_rx_loaded()
    srf = _map_Elt(srf)  # iElt check => 1D array

    if vpt is None:
        vpt_ = np.zeros((3, len(srf)), dtype=float, order='F')
        if not lib.api.elt_vpt(srf, vpt_, 0):
            raise Exception("MACOS: failed to get VptElt values")
        return vpt_

    # define parameter values
    vpt_ = np.asarray_chkfinite(vpt, order='F')
    if vpt_.shape != (3, len(srf)):
        raise ValueError("vpt shape is invalid")

    if not lib.api.elt_vpt(srf, vpt_, 1):
        raise Exception("MACOS: failed to set VptElt values")


def elt_rpt(srf:Surface, rpt: None | Position = None) -> None | Position:
    """Set/Get Element Rotation position(s) for specified elements

    Args:
        srf (int | Tuple[int] | np.ndarray[int], dtype=np.int32]):
           Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        rpt (list[float] | np.ndarray[float], optional):
            If provided, the Rotation Positions at the elements srf will
            be replaced. 'rpt' must have the shape [3xN] where rpt is
            defined as = [ [x, y, z]_1, ..., [x, y, z]_N ].T  expressed in
            the global coordinate frame where N is the number of surfaces
            specified in srf. Defaults to None.

    Raises:
        Exception: MACOS triggered error

    Returns:
        elt_rpt (None | np.ndarray[float]):
            If no Option was given, it will return the Rotation Positions of
            at the surfaces defined by 'srf'; otherwise, nothing.

    """
    _chk_macos_and_rx_loaded()
    srf = _map_Elt(srf)

    if rpt is None:
        rpt_ = np.zeros((3, len(srf)), dtype=float, order='F')
        if not lib.api.elt_rpt(srf, rpt_, 0):
            raise Exception("MACOS: failed to get RptElt values")
        return rpt_

    # define parameter values
    rpt_ = np.asarray_chkfinite(rpt, order='F')
    if rpt_.shape != (3, len(srf)):
        raise ValueError("Rotation Point shape is invalid: must be 3xN")

    if not lib.api.elt_rpt(srf, rpt_, 1):
        raise Exception("MACOS: failed to set RptElt values")


def elt_psi(srf: Surface, psi: None | Direction = None) -> None | Direction:
    """Set/Get surface normals at vertex locations for specified elements.

    Args:
        srf (int | Tuple[int] | np.ndarray[int], dtype=np.int32]):
           Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        psi (list[float] | np.ndarray[float], optional): Defaults to None.
            If provided, the Surface Normal at the elements srf will be
            replaced. 'psi' must have the shape [3xN] where psi is
            defined as = [ [l, m, n]_1, ..., [l, m, n]_N ]
            expressed in the global coordinate frame where N is the number
            of surfaces specified in srf.
            where [l,m,n] are the direction cosine values with the property
            1 = l^2 + m^2 + n^2
            Defaults to None.

    Raises:
        Exception: MACOS triggered error

    Returns:
        None | list[float] | np.ndarray[float]:
            If no Option was given, it will return the Surface Normal(s) at
            the Vertex Position of the surface(s); otherwise, None. The
            return has the same format as psi input, i.e., [3xN] where N
            is the number of surfaces specified in srf.

    """
    _chk_macos_and_rx_loaded()
    srf = _map_Elt(srf)                # iElt check => [1xN] array

    if psi is None:
        # retrieve param values of the defined elements
        psi_ = np.zeros((3, len(srf)), dtype=float, order='F')
        if not lib.api.elt_psi(srf, psi_, 0):
            raise Exception("MACOS: failed to get PsiElt values")
        return psi_

    # define parameter values
    # srf, psi_ = _chk_values_pos(srf, psi)  # iElt and PsiElt data structure alignment & checks
    psi_ = np.asarray_chkfinite(psi, order='F')
    if psi_.shape != (3, len(srf)):
        raise ValueError("Psi shape is invalid")

    if not lib.api.elt_psi(srf, psi_, 1):
        raise Exception("MACOS: failed to set PsiElt values")


def elt_csys(srf: Surface,
             xdir: None | Direction = None,
             ydir: None | Direction = None,
             zdir: None | Direction = None,
             upd: bool = True,
             glb: bool = False
             ) -> None | tuple[np.ndarray, np.ndarray, np.ndarray]:

    """Set, Get or Delete Local Coordinate System (LCS) Information (Rx:TElt)

    Args:
        srf (int | Tuple[int] | np.ndarray[int], dtype=np.int32]):
           Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        xdir (ndarray | list[float] | None, optional): Defaults to None
            A [3x1] Vector =[Lx,Ly,Lz] defining the x-axis expressed in Global CS

        ydir (ndarray | list[float] | None, optional): Defaults to None
            A [3x1] Vector =[Lx,Ly,Lz] defining the y-axis expressed in Global CS

        zdir (ndarray | list[float] | None, optional): Defaults to None
            A [3x1] Vector =[Lx,Ly,Lz] defining the z-axis expressed in Global CS

        upd (bool, optional): Defaults to True
            If set, LCS (TElt) will be updated with element perturbations.

        gbl (bool, optional): Defaults to False
            If set, removes LCS if defined on all identified surfaces.

    Returns:
        None: when defining or deleting the LCS

        TElt: (np.ndarray[float])
            Returns a [6x6xN] Tensor where TElt is the [6x6xj] matrix for a
            given surface 'j' expressed in the Global CS. Eye(6) is returned
            for Surfaces having a Global Coord. System defined.

        upd: (tuple[bool])
            Returns a Boolean tuple where 'True' identifies the surfaces on
            which the LCS is setup to update its orientation with element
            perturbations where 'False'

        lcs: (tuple[bool])
            Returns a [1xN] tuple where 'True' is defined if a LCS is
            defined on a particular surface and the tuple defines the state
            over all identified surfaces 'srf'.
            nECoord/=-6 or 0 (Global CS) or 6 (Local CS).

    Note:
        When defining the LCS, two of the three LCS axes are needed. The
        order to define the LCS is:
            ToDo

    Examples:
       csys, csys_lcs, csys_upd = elt_csys(srf):
            Get LCS information at defined surfaces

       elt_csys(srf, glb=True):
            Removes LCS if defined on all identified surfaces

       elt_csys(srf, xdir, ydir, zdir, upd):
            Defines a new LCS using the provided x/y/z-axis (defined in global CS)
            and define if LCS is changing with element prbs. (upd)ate option
            on all identified surfaces (same LCS).
    """

    # Entry check and re-mapping iElt
    _chk_macos_and_rx_loaded()

    srf = _map_Elt(srf, max_rows=1)     # iElt check & re-mapping
    n_srf = srf.argmax()+1

    axes = (xdir is not None, ydir is not None, zdir is not None)

    if glb:
        # Delete LCS on defined surfaces
        if not lib.api.elt_csys_rm(srf):
            raise Exception("elt_csys: MACOS API execution failed")

    elif np.all(np.logical_not(axes)):
        # LCS state extraction

        # csys == TElt
        csys = np.zeros((6, 6, n_srf), dtype=float, order='F')
        csys_lcs = np.zeros(n_srf, dtype=np.int32)
        csys_upd = np.zeros(n_srf, dtype=np.int32)
        if not lib.api.elt_csys_get(srf, csys, csys_lcs, csys_upd):
            raise Exception("elt_csys: MACOS API execution failed")

        # print(f"=====> {csys_lcs=}")
        # print(f"=====> {csys_lcs>0=}")
        # if (not ok) or (csys.ndim != 3) or (csys.shape != (6, 6, n_srf)) or \
        #    (csys_lcs.size != n_srf) or (csys_upd.size != n_srf):
        #     raise Exception("elt_csys: MACOS API execution failed")

        # csys_lcs.shape = (1, -1)     # note: other option: np.atleast_2d()
        # csys_upd.shape = (1, -1)
        return csys, csys_lcs > 0, csys_upd > 0

    else:
        # LCS definition
        if axes[1] and axes[2]:
            ydir = _chk_values_1d(ydir, 3, True)
            zdir = _chk_values_1d(zdir, 3, True)

            ydir /= np.linalg.norm(ydir)
            zdir /= np.linalg.norm(zdir)

            xdir = np.cross(ydir, zdir)
            xdir /= np.linalg.norm(xdir)

            ydir = np.cross(zdir, xdir)
            ydir /= np.linalg.norm(ydir)

        elif axes[0] and axes[2]:
            xdir = _chk_values_1d(xdir, 3, True)
            zdir = _chk_values_1d(zdir, 3, True)

            xdir /= np.linalg.norm(xdir)
            zdir /= np.linalg.norm(zdir)

            ydir = np.cross(zdir, xdir)
            ydir /= np.linalg.norm(ydir)

            xdir = np.cross(ydir, zdir)
            xdir /= np.linalg.norm(xdir)

        elif axes[0] and axes[1]:
            xdir = _chk_values_1d(xdir, 3, True)
            ydir = _chk_values_1d(ydir, 3, True)

            xdir /= np.linalg.norm(xdir)
            ydir /= np.linalg.norm(ydir)

            zdir = np.cross(xdir, ydir)
            zdir /= np.linalg.norm(zdir)

            ydir = np.cross(zdir, xdir)
            ydir /= np.linalg.norm(ydir)

        else:
            raise ValueError("elt_csys: require at least 2 axes to define LCS")

        # ToDo: check type size -- or let Python handle it
        upd = np.asarray(upd, dtype=np.int32)

        if not lib.api.elt_csys_set(srf, xdir.T, ydir.T, zdir.T, upd):
            raise Exception("elt_csys: MACOS API execution failed")


""" -------------------------------------------------------------------------------------------
[ ] Element Surface Properties
    -------------------------------------------------------------------------------------------
    ! [ ] Pose
    !     [x] elt_vpt     : set/get Element Vertex   Point
    !     [x] elt_rpt     : set/get Element Rotation Point
    !     [x] elt_psi     : set/get Element Surface Normal
    !
    ! [ ] Base Srf. Shape
    !     [x] elt_kc      : set/get Element Conic Constant
    !     [x] elt_kr      : set/get Element Base Radius
    !
    ! [ ] Material
    !     [ ] IndRef_     : set/get Refractive Index
    !     [ ] Glass_      : set/get Material Specification and read data from Glass Tbl.
    !     [ ] GlassModel_ : set/get Material Specification based on Glass Properties
    !
    ! [ ] Local CSYS (TElt)
    !     [ ] set/get/rm  EltCFrame
    !
    ! [Srf. Shape] elt_srf_csys      : set Srf. Coordinate Frame
    ! [Srf. Shape] getEltGridInfo    : get Grid Srf. Settings
    ! [Srf. Shape] setEltGrid        : set element surface grid data
    !  [Pos/Shape] xp_set            : set XP parameters (Kr, Psi, Vpt, Rpt & zElt)
    !=============================================================================================
"""


def elt_kc(srf: Surface,
           conic_constant: None | Parameter = None
           ) -> None | Vector[np.float64]:
    """Set/Get Conic Constant(s) (Rx:KcElt) for specified elements.

    Args:
        srf (int | Tuple[int] | np.ndarray[int], dtype=np.int32]):
           Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        conic_constant (float | list[float] | np.ndarray[float], optional):
           If provided, the Conic Constant(s) at the elements 'srf'
           will be replaced by the conic values. 'conic' has the same
           length as 'srf'. Defaults to None.

    Raises:
        Exception: MACOS Triggered error

    Returns:
        None | float | list[float] | np.ndarray[float]:
            Conic Constants at the specified surface(s); otherwise,None.
    """
    _chk_macos_and_rx_loaded()
    srf = _map_Elt(srf) #.reshape(1, -1)  # iElt check => [1xN] array

    if conic_constant is None:
        conic_constant_ = np.zeros_like(srf, dtype=float)
        if not lib.api.elt_kc(srf, conic_constant_, 0):
            raise Exception("failed to get Conic Constant 'KcElt' values")
        return conic_constant_

    # define conic values
    conic_constant_ = np.asarray_chkfinite(conic_constant)

    if not lib.api.elt_kc(srf, conic_constant_, 1):
        raise Exception('KcElt threw an exception')


def elt_kr(srf: Surface,
           radii: None | Parameter = None
           ) -> None | Vector[np.float64]:
    """Set/Get Radii of Curvatures (Rx:KrElt) for specified elements.

    Args:
        srf (int | Tuple[int] | np.ndarray[int], dtype=np.int32]):
           Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        radii (float | list[float] | np.ndarray[float], optional): Defaults to None.
           If provided, the Radii of Curvatures(s) at the elements 'srf'
           will be replaced by the provided values. 'radii' has the same
           length as 'srf'. Defaults to None.

    Raises:
        Exception: MACOS Triggered error

    Returns:
        None | np.ndarray[float]:
            Radius of Curvature at the specified surface(s); otherwise, None.

     Examples
     --------
     - get radius at 1st element:     `pymacos.elt_kr(1)`
     - get radii at the 1st & 5th element: `pymacos.elt_kr((1,5))`
     - set radii at 1st & 2nd Element: `pymacos.elt_kr((1, 2),(1.0, -1.0))`
    """

    # Entry check and re-mapping iElt
    _chk_macos_and_rx_loaded()    # pymacos & Rx loaded
    srf = _map_Elt(srf)           # srf check => [1xN] array

    # retrieve radii values of the defined elements
    if radii is None:
        radii_ = np.zeros_like(srf, dtype=float)
        if not lib.api.elt_kr(srf, radii_, 0):
            raise Exception("failed to get Radius of Curvature 'KrElt' values")
        return radii_

    # define radii values
    radii_ = np.asarray_chkfinite(radii, dtype=float)

    if np.any(np.abs(radii_) <= np.finfo(float).eps):
        raise ValueError("Radius cannot be Zero")

    if not lib.api.elt_kr(srf, radii_, 1):
        raise Exception('MACOS raised an error')


def elt_srf_csys(srf: Surface,
                 origin: None | Position  = None,
                 xdir: None | Direction = None,
                 ydir: None | Direction = None,
                 zdir: None | Direction = None
                ) -> None | tuple[np.ndarray, bool, bool]:

    """Set / Get Local Surface Coordinate System (LCS) on Element

       For querying, multiple surfaces can be defined; otherwise, only 1

    Args:
        srf (int | Tuple[int] | Vector[int]):
           Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        origin (Tuple[float] | Vector[np.float64] | None, optional): Defaults to None
            pMon = [x,y,z] (3x1 array) defining the CS origin expressed in Global CS

        xdir (Tuple[float] | Vector[np.float64] | None, optional): Defaults to None
            xMon = [Lx,Ly,Lz] (3x1 array) defining the x-axis expressed in Global CS

        ydir (Tuple[float] | Vector[np.float64] | None, optional): Defaults to None
            yMon = [Lx,Ly,Lz] (3x1 array) defining the y-axis expressed in Global CS

        zdir (Tuple[float] | Vector[np.float64] | None, optional): Defaults to None
            zMon = [Lx,Ly,Lz] (3x1 array) defining the z-axis expressed in Global CS

    Returns:
        None: when defining LCS

    ToDo
    """
    _chk_macos_and_rx_loaded()

    srf = _map_Elt(srf, max_rows=1)    # iElt check & re-mapping
    n_srf = srf.argmax()+1

    axes = (xdir is not None, ydir is not None, zdir is not None)
    pmon = origin is not None

    if not (any(axes) or pmon):

        origin = np.zeros((3, n_srf), dtype=float, order='F')
        xdir = np.zeros_like(origin, dtype=float)
        ydir = np.zeros_like(origin, dtype=float)
        zdir = np.zeros_like(origin, dtype=float)

        if not lib.api.elt_srf_csys(origin, xdir, ydir, zdir, srf, 0):
            raise Exception("MACOS API execution failed")
        return origin, xdir, ydir, zdir

    else:
        if n_srf != 1:
            raise ValueError("Values can only be set at a single surface a time")

        if sum(axes) == 1:
            raise ValueError("require at least 2 axes to define LCS")

        if pmon:
            origin = np.asarray_chkfinite(origin, dtype=float, order="F")

        if any(axes):
            def a_cross_b(a, b):

                if a.shape != (3, 1) or b.shape != (3, 1):
                    raise ValueError("Vectors must have shape (3 x 1)")

                a = np.asarray_chkfinite(a, dtype=float, order="F")
                b = np.asarray_chkfinite(b, dtype=float, order="F")

                a /= np.linalg.norm(a)
                b /= np.linalg.norm(b)

                c = np.cross(a.T, b.T).T
                c /= np.linalg.norm(c)

                a = np.cross(b.T, c.T).T
                a /= np.linalg.norm(a)
                return a, b, c

            # LCS definition
            if axes[1] and axes[2]:
                ydir, zdir, xdir = a_cross_b(ydir, zdir)   # todo: they can be 3xN  or only 3x1 for setting at srf??

            elif axes[0] and axes[2]:
                zdir, xdir, ydir = a_cross_b(zdir, xdir)

            elif axes[0] and axes[1]:
                xdir, ydir, zdir = a_cross_b(xdir, ydir)

            elif np.all(axes):
                if xdir.shape != (3, 1) or ydir.shape != xdir.shape or \
                   ydir.shape != zdir.shape:
                    raise ValueError("Vectors must have shape [3 x 1]")

                xdir = np.asarray_chkfinite(xdir, dtype=float, order="F")
                ydir = np.asarray_chkfinite(ydir, dtype=float, order="F")
                zdir = np.asarray_chkfinite(zdir, dtype=float, order="F")

        # only pMon is to be updated
        if pmon and not all(axes):
            if not lib.api.elt_srf_csys_pos(origin, srf, 1):
                raise Exception("MACOS API execution failed")

        # only axes are to be updated
        elif not pmon:
            if not lib.api.elt_srf_csys_dir(xdir, ydir, zdir, srf, 1):
                raise Exception("MACOS API execution failed")

        else:
            if not lib.api.elt_srf_csys(origin, xdir, ydir, zdir, srf, 1):
                raise Exception("MACOS API execution failed")


#  ----------------------------------------------------------------------------
#  [ ] Element Surface Properties: Grating
#  ----------------------------------------------------------------------------
#  [x] elt_grating_any       Checks if any Grating Srfs. are defined in Rx
#  [x] elt_grating_fnd       Find all elements with Grating Srfs. attached
#  [x] elt_grating_params    Grating (h1HOE, RuleWidth, Trans. or Refl.)
#  [x] elt_grating_type      Transmission or Reflective Grating
#  [x] elt_grating_order     Grating Order (Param: OrderHOE)
#  [x] elt_grating_rulewidth Rule Width
#  [x] elt_grating_dir       h1HOE vector prpdicular. to the ruling dir and psiElt.
#  ----------------------------------------------------------------------------


def elt_grating_any() -> bool:
    """Checks if Gratings on Srfs. are defined in Rx

    Raises:
        Exception: MACOS and/or Rx not loaded

    Returns:
        bool: True if Grating are defined in Rx; otherwise, False
    """

    _chk_macos_and_rx_loaded()
    return bool(lib.api.elt_srf_grating_any())


def elt_grating_fnd(srf: None | Surface = None
                    ) -> Tuple[Integer, Integer] | None:
    """Find/Check elements with a Grating defined

    Args:
        srf (None | int | Tuple[int] | np.ndarray[int], dtype=np.int32]):
           Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

           When srf is not defined, it uses all surfaces in Rx

    Raises:
        Exception: MACOS Triggered error

    Returns:
        None:
            Empty list when no Grating def. is defined

        (List[Integers], List[Integers]):
            Tuple[0]: Surfaces IDs where Grating def. are defined
            Tuple[1]: Grating ID: (=1) Reflection   Grating
                                  (=2) Transmission Grating
    """
    _chk_macos_and_rx_loaded()

    if lib.api.elt_srf_grating_any() == 0:
        return None

    if srf is None:
        n_elt = lib.api.n_elt()
        srf_ = np.arange(1, n_elt+1, dtype=np.int32)
    else:
        srf_ = np.asarray_chkfinite(srf)

    ok, elt_grating = lib.api.elt_srf_grating_fnd(srf_)
    if not ok:
        raise Exception('MACOS threw an error')

    jsrf = elt_grating.nonzero()[0]
    return jsrf+1, elt_grating[jsrf]


def elt_grating_params(srf: Surface, *,
                       reflective: bool | None = None,
                       rule_width: np.ndarray | float | None = None,
                       diff_order: np.ndarray | Integer | None = None,
                       rule_dir: Vector | None = None
                       ) -> Tuple | None:
    """set/get the Grating Params. of an existing Element with a Grating

    Args:
        srf (None | int | Tuple[int] | np.ndarray[int], dtype=np.int32]):
            Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
            Neg. values are referenced with respect to the last surface
            where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        reflective (bool | None, optional):
            If True, the Grating is Reflective and Transmissive otherwise
            Defaults to "None".

        rule_width (np.ndarray | float | None, optional):
            The fixed distance between rules as projected to a flat plane
            underlying the surface. Distance between rules along the curved
            surface can vary if the surface shape is curved, which will be
            the case with a conic or aspheric surface type.
            Defaults to None.

        diff_order (np.ndarray | Integer | None, optional):
            Diffraction order. Defaults to None.

        rule_dir (Vector | None, optional):
            Direction of the ruling orientation, i.e., perpendicular to the
            ruling direction and to the psiElt vector.
            Defaults to None.

        When ALL optional params are "None", the values at Ele. "srf" are
        returned.

    Raises:
        Exception: MACOS Triggered error

    Returns:
        None:
            when updating Rx with new values

        List[reflective, rule_width, diff_order, rule_dir]:
            bool:   True if reflective
            float:  rule_width
            int:    diff_order
            Vector: rule_dir
    """

    _chk_macos_and_rx_loaded()
    srf = _map_Elt(srf).item()

    params = (rule_width, diff_order, rule_dir, reflective)

    # read
    if all([v is None for v in params]):
        refl = np.array(True, dtype=bool)
        rule_width = np.array(0e0, dtype=float)
        diff_order = np.array(0, dtype=np.int32)
        rule_dir = np.zeros(3, dtype=float)

        if not lib.api.elt_srf_grating_params(srf, rule_width, diff_order,
                                             rule_dir, refl, 0):
            raise Exception('MACOS threw an exception')
        return refl == 1, rule_width.item(), diff_order.item(), rule_dir

    # check params
    if rule_width is not None:
        rule_width_ = np.asarray_chkfinite(rule_width, dtype=float)
        if rule_width_ <= 0:
            raise ValueError("Rule Width must be > 0")

    if diff_order is not None:
        diff_order_ = np.asarray_chkfinite(diff_order, dtype=int)
        if abs(diff_order_) > 5:
            raise ValueError("limit |Diffraction Order| < 6")

    if rule_dir is not None:
        rule_dir_ = np.asarray_chkfinite(rule_dir, dtype=float)
        rule_dir_ /= np.linalg.norm(rule_dir_)

    if reflective is not None:
        reflective_ = np.asarray_chkfinite(reflective, dtype=bool)

    # write all
    if all([v is not None for v in params]):
        if not lib.api.elt_srf_grating_params(srf, rule_width_, diff_order_,
                                             rule_dir_, reflective_, 1):
            raise Exception('MACOS threw an exception')

    # write partial data
    else:
        if rule_width is not None:
            elt_grating_rulewidth(srf, rule_width_)

        if diff_order is not None:
            elt_grating_order(srf, diff_order_)

        if reflective is not None:
            elt_grating_type(srf, reflective_)

        if rule_dir is not None:
            elt_grating_dir(srf, rule_dir_)


def elt_grating_type(srf: Surface,
                     reflective: bool | None = None
                     ) -> Tuple | None:

    """set/get the Grating Params. of an existing Element with a Grating

    Args:
        srf (None | int | Tuple[int] | np.ndarray[int], dtype=np.int32]):
            Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
            Neg. values are referenced with respect to the last surface
            where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        reflective (bool | None, optional):
            If True, the Grating is Reflective and Transmissive otherwise
            Defaults to "None".

        When optional param. is "None", the values at Ele. "srf" are
        returned.

    Raises:
        Exception: MACOS Triggered error

    Returns:
        None:
            when updating Rx with new values
        int:
            diff_order
    """

    _chk_macos_and_rx_loaded()
    srf = _map_Elt(srf).item()

    # read
    if reflective is None:
        refl = np.array(True, dtype=bool)
        if not lib.api.elt_srf_grating_type(srf, refl, 0):
            raise Exception('MACOS threw an exception')
        return refl.item() == 1

    # write
    reflective_ = np.asarray_chkfinite(reflective, dtype=bool)
    if not lib.api.elt_srf_grating_type(srf, reflective_, 1):
        raise Exception('MACOS threw an exception')


def elt_grating_order(srf: Integers,
                      diff_order: Integers | np.ndarray | None = None,
                      ) -> Integers | np.ndarray | None:
    """set/get the Grating Diff. Order on existing Element with a Grating

    Args:
        srf (None | int | Tuple[int] | np.ndarray[int], dtype=np.int32]):
            Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
            Neg. values are referenced with respect to the last surface
            where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        diff_order (np.ndarray | Integer | None, optional):
            Diffraction order. Defaults to None.

        When optional param. is "None", the Diff. Order at Ele. "srf" are
        returned.

    Raises:
        Exception: MACOS Triggered error

    Returns:
        None:
            when updating Rx with new values
        int:
            diff_order
    """

    _chk_macos_and_rx_loaded()
    srf = _map_Elt(srf)

    # read
    if diff_order is None:
        # diff_order = np.array(0, dtype=np.int32)
        diff_order_ = np.zeros_like(srf, dtype=np.int32)
        if not lib.api.elt_srf_grating_order(srf, diff_order_, 0):
            raise Exception('MACOS threw an exception')
        return diff_order_

    # write
    diff_order_ = np.asarray_chkfinite(np.asarray(diff_order, dtype=np.int32))
    # print(f"{diff_order_=}, {len(diff_order_)=}")
    # if abs(diff_order_) > 5:
    #     raise ValueError("limit |Diffraction Order| < 6")
    if not lib.api.elt_srf_grating_order(srf, diff_order_, 1):
        raise Exception('MACOS threw an exception')


def elt_grating_rulewidth(srf: Surface,
                          rule_width: Floats | None = None,
                          ) -> float | None:
    """set/get the Grating Diff. Order on existing Element with a Grating

    Args:
        srf (None | int | Tuple[int] | np.ndarray[int], dtype=np.int32]):
            Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
            Neg. values are referenced with respect to the last surface
            where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        rule_width (np.ndarray | float | None, optional):
            The fixed distance between rules as projected to a flat plane
            underlying the surface. Distance between rules along the curved
            surface can vary if the surface shape is curved, which will be
            the case with a conic or aspheric surface type.
            Defaults to None.

        When optional param. is "None", the Rule Width at Ele. "srf" is
        returned.

    Raises:
        Exception: MACOS Triggered error

    Returns:
        None:
            when updating Rx with new values
        float:
            rule width
    """

    _chk_macos_and_rx_loaded()
    srf = _map_Elt(srf).item()

    # read
    if rule_width is None:
        rule_width = np.array(0, dtype=float)
        if not lib.api.elt_srf_grating_rule_width(srf, rule_width, 0):
            raise Exception('MACOS threw an exception')
        return rule_width.item()

    # write
    rule_width_ = np.asarray_chkfinite(rule_width, dtype=float)
    if not lib.api.elt_srf_grating_rule_width(srf, rule_width_, 1):
        raise Exception('MACOS threw an exception')


def elt_grating_dir(srf: Surface,
                    rule_dir: Vector | None = None
                    ) -> Vector | None:

    """set/get the Grating Direction on existing Grating Element

    Args:
        srf (None | int | Tuple[int] | np.ndarray[int], dtype=np.int32]):
            Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
            Neg. values are referenced with respect to the last surface
            where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        rule_dir (Vector | None, optional):
            Direction of the ruling orientation, i.e., perpendicular to the
            ruling direction and to the psiElt vector.
            Defaults to None.

        When optional param. is "None", the Rule Width at Ele. "srf" is
        returned.

    Raises:
        Exception: MACOS Triggered error

    Returns:
        None:
            when updating Rx with new values

        np.ndarray:
            rule width
    """

    _chk_macos_and_rx_loaded()
    srf = _map_Elt(srf).item()

    # read
    if rule_dir is None:
        rule_dir = np.zeros(3, dtype=float)
        if not lib.api.elt_srf_grating_rule_dir(srf, rule_dir, 0):
            raise Exception('MACOS threw an exception')
        return rule_dir

    # write
    rule_dir_ = np.asarray_chkfinite(rule_dir, dtype=float)
    rule_dir_ /= np.linalg.norm(rule_dir_)
    if not lib.api.elt_srf_grating_rule_dir(srf, rule_dir_, 1):
        raise Exception('MACOS threw an exception')


# ----------------------------------------------------------------------------
# [ ] Element Surface Properties: Zernike
# ----------------------------------------------------------------------------
#
#     elt_zrn_norm_rad          (lMon)  Norm. Radius
#     elt_zrn_coef              Zernike Coefficients
#     elt_zrn_any               Checks if Zernike Srfs. are defined in Rx
#     elt_zrn_fnd               Find all elements with Zernike Srfs.
#     elt_zrn_type              Zernike Ordering Type (B&W, Noll, ...)
#     elt_zrn_annular_ratio
#
#     elt_srf_csys              Set/get Zernike Surface placement
#
#     elt_zrn                   True/False if it is a Zernike Srf
#     getEltSrfZern
#     setEltSrfZern
#     elt_srf_zrn_mode_set
# ----------------------------------------------------------------------------


def elt_zrn_norm_rad(srf: int | np.int32,
                     norm_rad: None | float | np.float64 = None
                     ) -> None | float:

    """Set/Get Zernike Normalisation Radius for specified Surface.

    Zernike Coefficients will _not_ be re-scalled

    Args:
        srf (int | np.int32):
           Element ID in Range: -nElt < srf <= nElt
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        norm_rad (None | float | np.float64):
           Normalisation Radius over which Zernikes are Orthonormal [units Length]

    Raises:
        Exception: when input constrains are not satisfied within MACOS or
                   not a Zernike Surface

    Returns:
                  (None)     when defining the Norm. Radius
        norm_rad: (np.int32) when extracting the Norm. Radius
    """
    # Entry check and re-mapping iElt
    _chk_macos_and_rx_loaded()
    srf = _map_Elt(srf).item()

    # read
    if norm_rad is None:
        # return Zernike Norm Radius
        norm_rad = np.array(0, dtype=np.float64)
        if not lib.api.elt_srf_zrn_norm_radius(srf, norm_rad, 0):
            raise Exception("MACOS threw an exception")
        return norm_rad.item() if norm_rad > 0 else np.nan

    # write
    norm_rad = np.asarray_chkfinite(norm_rad, dtype=np.float64)
    if norm_rad <= 0.:
        raise ValueError("Zernike Norm. Radius cannot be less equal to zero")

    if not lib.api.elt_srf_zrn_norm_radius(srf, norm_rad, 1):
        raise Exception("MACOS threw an exception")


def elt_zrn_coef(srf: int | np.int32,
                 mode: Index,
                 coefs: None | Parameter = None,
                 reset: bool = False
                 ) -> None | Vector[np.float64]:
    """Set/Get Zernike Coefficients for a specific Surface

    Args:
        srf (int | np.int32):
           Element ID in Range: -nElt < srf <= nElt
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        mode (int | Tuple[int] | ndarray[int]):
            Identifies the Zernike Modes (1, ..., 66), where for Fringe
            Zernike it is limited to 37 Modes.

        coefs (None | Parameter, optional): Defaults to None.
            Zernike Coefficients, which are only to be defined for setting
            the values.

        reset (bool, optional): Defaults to False.
            When set and setting values, all modes are set to zero
            first (wipe mode).

    Raises:
        Exception: when input constrains are not satisfied within MACOS

    Returns:
        coefs: (None)               when setting Zernike coefficients
        coefs: (Vector[np.float64]) when extracting Zernike coefficients
    """

    # Entry check and re-mapping iElt
    _chk_macos_and_rx_loaded()
    srf = _map_Elt(srf).squeeze()
    mode = np.asarray_chkfinite(mode, dtype=np.int32)

    # read
    if coefs is None:
        # extract Zernike Coefficients
        # mode = np.asarray_chkfinite(mode, dtype=np.int32)
        coefs = np.zeros_like(mode, dtype=float)
        if not lib.api.elt_srf_zrn_coef(srf, mode, coefs, 0, 0):
            raise Exception("'elt_zrn_coef' threw an exception")
        return coefs

    # write
    coefs = np.asarray_chkfinite(coefs, dtype=float)
    if not lib.api.elt_srf_zrn_coef(srf, mode, coefs, 1, reset):
        raise Exception("'elt_zrn_coef' threw an exception")


def elt_zrn_any() -> bool:
    """Checks if Zernike Srfs. are defined in Rx

    Raises:
        Exception: MACOS and/or Rx not loaded

    Returns:
      found (bool):
        True if Zernike Srfs. is/are defined in Rx; otherwise, False
    """

    _chk_macos_and_rx_loaded()
    return bool(lib.api.elt_srf_zrn_any())


def elt_zrn_fnd(srf: None | Surface = None) -> Tuple[int]:
    """Find all elements with Zernike Srfs.

    Args:
        srf (None | int | Tuple[int] | np.ndarray[int], dtype=np.int32]):
           Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

           When 'None' is defined, it uses all surfaces in Rx

    Raises:
        Exception: MACOS Triggered error

    Returns:
        ([]):
            Empty list when no Zernike Srf. is defined
        (list[int]):
            Surfaces IDs where Zernikes are defined
    """
    _chk_macos_and_rx_loaded()

    if lib.api.elt_srf_zrn_any() == 0:
        return []

    # find elements where Grps. are defined
    if srf is None:
        n_elt = lib.api.n_elt()
        srf_ = np.arange(1, n_elt+1, dtype=np.int32)
    else:
        srf_ = np.asarray_chkfinite(srf)

    ok, n_elt_zrn = lib.api.elt_srf_zrn_fnd(srf_)
    if not ok:
        raise Exception('MACOS threw an error')

    return n_elt_zrn.nonzero()[0]+1


def elt_zrn_type(srf: int | np.int32,
                 zrn_type: None | int | np.int32 = None,
                 reset: bool = False
                 ) -> None | int:

    """Set/Get Zernike Type for specified Surface

    Args:
        srf (int | np.int32):
           Element ID in Range: -nElt < srf <= nElt
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        zrn_type (None | int | np.int32):
            1) ANSI         4) Norm. ANSI         7) Norm. Hex
            2) Born & Wolf  5) Norm. Born & Wolf  8) Norm. Noll
            3) Fringe       6) Norm. Fringe       9) Norm. AnnularNoll
           10) Noll        11) Ext. Fringe

        reset (bool, optional): Defaults to False.
            When reset is set and when setting Zernike Type, all Zernike
            Coefficients are set to zero (wipe values).

    Raises:
        Exception: when input constrains are not satisfied within MACOS or
                   not a Zernike Surface

    Returns:
                  (None)     when setting Zernike Type
        zrn_type: (np.int32) when extracting Zernike Types
                  -1 is returned if the surface is not a Zernike Surface
    """

    _chk_macos_and_rx_loaded()
    srf = _map_Elt(srf).squeeze()

    # read
    if zrn_type is None:
        # return Zernike Type
        zrn_type = np.array(0, dtype=np.int32)
        if not lib.api.elt_srf_zrn_type(srf, zrn_type, 0, 0):
            raise Exception("MACOS threw an exception")
        return int(zrn_type)

    # write
    zrn_type = np.asarray_chkfinite(zrn_type, dtype=np.int32)
    if (zrn_type < 1) or (zrn_type > 11):
        raise ValueError("Zernike Type outside valid range")

    if not lib.api.elt_srf_zrn_type(srf, zrn_type, 1, reset):
        raise Exception("'elt_zrn_coef' threw an exception")


def getEltSrfZern(iElt):
    """
    Retrieve Zernike Srf. settings of Zernike Srf. element(s)

    :param   iElt:               [1xN,I]: Elt. ID ( -nElt < iElt[i] <= nElt )

    :return  lMon:               [1xN,D]: = Zernike Radius
    :return  ZernType:           [1xN,I]: 1) Malacara  4) NormMalacara  7) NormHex
                                          2) BornWolf  5) NormBornWolf  8) NormNoll
                                          3) Fringe    6) NormFringe    9) NormAnnularNoll
    :return  ZernCoef:           [MxN,D]: = [C_1,C_2,...,C_N] Zernike Coefficients  (M=45)
    :return  ZernAnnularRatio:   [1xN,D]: = inner/outer radius ratio (0,...,1)
                                            only important for ZernType = NormAnnularNoll (9)

    Note: ZCF => Zernike Coord. Frame: retrieved via elt_srf_csys(...)
    """

    _chk_macos_and_rx_loaded()       # pymacos & Rx loaded
    iElt = _map_Elt(iElt)            # iElt check

    ok, lmon, zernType, zernCoef, zernAnnularRatio = lib.api.getEltSrfZern(iElt)

    if not ok:
        raise Exception("'getEltSrfType' threw an exception")
    else:
        return lmon, zernType, zernCoef, zernAnnularRatio


def setEltSrfZern(iElt, lMon, zernType, zernMode=45, zernCoef=np.zeros(45), zernAnnularRatio=0):
    """
    Define Zernike Srf. settings of equal Zernike Srf. element(s)

    :param iElt:                [Mx1,I]: Elt. ID ( -nElt < iElt[i] <= nElt )
    :param lMon:                [1x1,D]: = Zernike Radius
    :param zernType:            [1x1,I]: 1) Malacara  4) NormMalacara  7) NormHex
                                         2) BornWolf  5) NormBornWolf  8) NormNoll
                                         3) Fringe    6) NormFringe    9) NormAnnularNoll
    :param zernMode:            [1xN,I]: [Optional] = [Z_1,Z_2,...,Z_N] Zernike Modes
    :param zernCoef:            [1xN,D]: [Optional] = [C_1,C_2,...,C_N] Zernike Coefficients
    :param zernAnnularRatio:    [1x1,D]: [Optional] = inner/outer radius ratio (0,...,1)

       Defaults : ZernMode         [1xN,I]: = 1:45
                  ZernCoef         [1xN,D]: = zeros(1,45)
                  ZernAnnularRatio [1x1,D]: = 0           <= only for ZernType = NormAnnularNoll (9)

       Note     : ZCF => Zernike Coord. Frame: defined via calling elt_srf_csys(...)
                  M of iElt defines identical elements, e.g., iElt = [1;-5]
                     => Element [1;nElt-5] have the same Zernike settings:
    """

    # Entry check and re-mapping iElt
    _chk_macos_and_rx_loaded()              # pymacos & Rx loaded
    iElt = _map_Elt(iElt)            # iElt check

    lMon = _chk_values_1d(lMon, 1)           # scalar
    if lMon < 0e0:
        raise ValueError("Expecting: 0 < lMon")

    zernType = _chk_values_1d(zernType, 1)   # scalar
    if zernType<1 or zernType>9:
        raise ValueError("Expecting: 0 < ZernType <= 9")

    zernMode = _chk_values_1d(zernMode, -1, row=False).flatten()   # vector
    if np.any(zernMode<1) or np.any(zernMode>45):
        raise ValueError("Expecting: 0 < ZernMode values <= 45")

    zernCoef =  _chk_values_1d(zernCoef, zernMode.size, row=False).flatten()   # vector

    zernAnnularRatio = _chk_values_1d(zernAnnularRatio, 1)   # scalar
    if zernAnnularRatio < 0e0 or zernAnnularRatio > 1e0:
        raise ValueError("Expecting: 0 < ZernAnnularRatio <= 1")

    if not lib.api.elt_srf_zrn_set(iElt, lMon, zernType, zernMode, zernCoef, zernAnnularRatio):
        raise Exception("'elt_srf_zrn_set' threw an exception")


def setEltSrfZernMode(iElt, izernMode=None, zernCoef=None) -> None:
    """ToDo"""

    iElt = _map_Elt(iElt)            # iElt check

    if izernMode is None or zernCoef is None:
        raise ValueError("'elt_srf_zrn_mode_set' undefined input")

    zernMode = _chk_values_1d(izernMode, -1, row=False).flatten()   # vector
    if np.any(zernMode<1) or np.any(zernMode>45):
        raise ValueError("'elt_srf_zrn_mode_set 'Expecting: 0 < ZernMode values <= 45")

    if not lib.api.elt_srf_zrn_mode_set(iElt, zernMode, zernCoef):
        raise Exception("'elt_srf_zrn_mode_set' threw an exception")


def getEltSrfZernMode(iElt: int,
                      zernMode: int | np.ndarray) -> np.ndarray:
    """Get current ZernCoef values at the requested modes on a
    Zernike or ZrnGrData element (SrfType=8 or 13).

    Symmetric companion to :func:`setEltSrfZernCoef`.  Used by the
    sensitivity-matrix work (channels.py) to read the nominal value
    before perturbing.  Calls the underlying ``elt_srf_zrn_coef`` in
    get mode -- bypasses the older ``getEltSrfZern`` wrapper, which
    targets a Fortran name (``getEltSrfZern``) that doesn't exist in
    the current build.
    """
    _chk_macos_and_rx_loaded()
    iElt = int(_map_Elt(iElt, max_rows=1).squeeze())
    modes = np.atleast_1d(np.asarray(zernMode, dtype=np.int32)).ravel()
    if modes.size == 0:
        raise ValueError("getEltSrfZernMode(): zernMode is empty")
    coef = np.zeros(modes.size, dtype=np.float64)
    ok = lib.api.elt_srf_zrn_coef(iElt, modes, coef, False, False)
    if not ok:
        raise Exception(
            f"getEltSrfZernMode(): MACOS rejected iElt={iElt} (not a "
            f"Zernike / ZrnGrData surface, or modes out of range)")
    return coef


def setEltSrfZernCoef(iElt: int,
                      zernMode: int | np.ndarray,
                      zernCoef: float | np.ndarray,
                      reset: bool = False) -> None:
    """Set ZernCoef values at the given modes on a Zernike (SrfType=8)
    or ZrnGrData (SrfType=13) element.

    Symmetric setter for :func:`getEltSrfZernMode`.  Routes through
    ``elt_srf_zrn_coef`` in setter mode -- bypasses the older
    :func:`setEltSrfZernMode` which has an f2py-side error path
    that triggers on single-mode calls.

    Args:
        iElt:     Element ID (1..nElt).
        zernMode: 1-based mode indices to write.
        zernCoef: Coefficient values (same length as ``zernMode``).
        reset:    If True, zero ALL modes on this element BEFORE
                  writing the new ones.

    Raises:
        Exception: Rx not loaded, iElt out of range, surface not
                   Zernike/ZrnGrData, or modes out of range.
    """
    _chk_macos_and_rx_loaded()
    iElt = int(_map_Elt(iElt, max_rows=1).squeeze())
    modes = np.atleast_1d(np.asarray(zernMode, dtype=np.int32)).ravel()
    coef  = np.atleast_1d(np.asarray(zernCoef, dtype=np.float64)).ravel()
    if modes.size == 0 or modes.size != coef.size:
        raise ValueError(
            "setEltSrfZernCoef(): zernMode and zernCoef must have the "
            f"same nonzero length (got {modes.size} and {coef.size})")
    ok = lib.api.elt_srf_zrn_coef(iElt, modes, coef, True, bool(reset))
    if not ok:
        raise Exception(
            f"setEltSrfZernCoef(): MACOS rejected iElt={iElt} (not a "
            f"Zernike/ZrnGrData surface, or modes out of range)")


# ----------------------------------------------------------------------------
# Element Surface Properties: FreeForm Mon-Zernike Coefficients
#
# MonZernCoef(:, iElt) is the perturbation channel for FreeForm surfaces
# (SrfType=14).  Used as the state vector for HWO-style wavefront-control
# sensitivity matrices: perturb a coefficient, retrace, capture OPD, build
# dw/dz columns.  Mirrors the GMI pmonzern path.
# ----------------------------------------------------------------------------

def findFreeFormElts() -> np.ndarray:
    """Return 1-based indices of every FreeForm-typed element in the
    loaded prescription (SrfType=14 in macos), in element order.

    Empty array if no FreeForm surfaces are present.
    """
    _chk_macos_and_rx_loaded()
    n = num_elt()
    all_elts = np.arange(1, n + 1, dtype=np.int32)
    ok, mask = lib.api.elt_srf_ff_fnd(all_elts)
    if not ok:
        raise Exception("'elt_srf_ff_fnd' threw an exception")
    return all_elts[mask.astype(bool)]


def getEltSrfMonZern(iElt: int,
                     zernMode: int | np.ndarray) -> np.ndarray:
    """Get current MonZernCoef values at the requested mode indices on
    a FreeForm element (SrfType=14).

    Args:
        iElt:     Element ID (1..nElt).  Must be a FreeForm surface.
        zernMode: 1-based mode indices to read (scalar or 1D array).

    Returns:
        1D float64 array of coefficients at the requested modes.

    Raises:
        Exception: Rx not loaded, iElt out of range, or the surface at
                   iElt isn't FreeForm.
    """
    _chk_macos_and_rx_loaded()
    iElt = int(_map_Elt(iElt, max_rows=1).squeeze())
    modes = np.atleast_1d(np.asarray(zernMode, dtype=np.int32)).ravel()
    if modes.size == 0:
        raise ValueError("getEltSrfMonZern(): zernMode is empty")
    coef = np.zeros(modes.size, dtype=np.float64)
    ok = lib.api.elt_srf_mon_zrn_coef(iElt, modes, coef,
                                      False, False)
    if not ok:
        raise Exception(
            f"getEltSrfMonZern(): MACOS rejected iElt={iElt} (not a "
            f"FreeForm surface, or modes out of [1, mMonCoef])")
    return coef


def setEltSrfMonZern(iElt: int,
                     zernMode: int | np.ndarray,
                     zernCoef: float | np.ndarray,
                     reset: bool = False) -> None:
    """Set MonZernCoef values at the given modes on a FreeForm element.

    The perturbation persists until the next setter call, a reset, or
    a fresh prescription load.  The next trace picks up the new
    coefficients via ZerntoMon in tracesub.F.

    Args:
        iElt:     Element ID (1..nElt).  Must be a FreeForm surface.
        zernMode: 1-based mode indices to write.
        zernCoef: Coefficient values (same shape as ``zernMode``).
        reset:    If True, zero ALL modes on this element BEFORE
                  writing the new ones (gives a clean state).

    Raises:
        Exception: Rx not loaded, iElt out of range, surface not
                   FreeForm, or modes out of [1, mMonCoef].
    """
    _chk_macos_and_rx_loaded()
    iElt = int(_map_Elt(iElt, max_rows=1).squeeze())
    modes = np.atleast_1d(np.asarray(zernMode, dtype=np.int32)).ravel()
    coef  = np.atleast_1d(np.asarray(zernCoef, dtype=np.float64)).ravel()
    if modes.size == 0 or modes.size != coef.size:
        raise ValueError(
            "setEltSrfMonZern(): zernMode and zernCoef must have the "
            f"same nonzero length (got {modes.size} and {coef.size})")
    ok = lib.api.elt_srf_mon_zrn_coef(iElt, modes, coef,
                                      True, bool(reset))
    if not ok:
        raise Exception(
            f"setEltSrfMonZern(): MACOS rejected iElt={iElt} (not a "
            f"FreeForm surface, or modes out of [1, mMonCoef])")


def getEltSrfFFZern(iElt: int,
                    zernMode: int | np.ndarray) -> np.ndarray:
    """Get current FFZernCoef values at the requested modes on a
    FreeForm element (SrfType=14).

    FFZernCoef is the FreeForm surface's *figure-description* Zernike
    component -- the static shape of the freeform itself.  It's a
    distinct array from MonZernCoef, which carries the perturbation
    overlay used for sensitivity / control work.  Both are converted
    to MonCoef at trace time via ZerntoMon.

    Args/Raises: same shape as :func:`getEltSrfMonZern`.
    """
    _chk_macos_and_rx_loaded()
    iElt = int(_map_Elt(iElt, max_rows=1).squeeze())
    modes = np.atleast_1d(np.asarray(zernMode, dtype=np.int32)).ravel()
    if modes.size == 0:
        raise ValueError("getEltSrfFFZern(): zernMode is empty")
    coef = np.zeros(modes.size, dtype=np.float64)
    ok = lib.api.elt_srf_ff_zrn_coef(iElt, modes, coef, False, False)
    if not ok:
        raise Exception(
            f"getEltSrfFFZern(): MACOS rejected iElt={iElt} (not a "
            f"FreeForm surface, or modes out of [1, mFFCoef])")
    return coef


def setEltSrfFFZern(iElt: int,
                    zernMode: int | np.ndarray,
                    zernCoef: float | np.ndarray,
                    reset: bool = False) -> None:
    """Set FFZernCoef values at the given modes on a FreeForm element.

    Use this to edit the FreeForm surface's *figure description*
    (the static shape).  For *perturbations* on top of an unchanged
    figure -- the wavefront-control / sensitivity-matrix channel --
    use :func:`setEltSrfMonZern` instead.

    Args/Raises: same shape as :func:`setEltSrfMonZern`.
    """
    _chk_macos_and_rx_loaded()
    iElt = int(_map_Elt(iElt, max_rows=1).squeeze())
    modes = np.atleast_1d(np.asarray(zernMode, dtype=np.int32)).ravel()
    coef  = np.atleast_1d(np.asarray(zernCoef, dtype=np.float64)).ravel()
    if modes.size == 0 or modes.size != coef.size:
        raise ValueError(
            "setEltSrfFFZern(): zernMode and zernCoef must have the "
            f"same nonzero length (got {modes.size} and {coef.size})")
    ok = lib.api.elt_srf_ff_zrn_coef(iElt, modes, coef,
                                     True, bool(reset))
    if not ok:
        raise Exception(
            f"setEltSrfFFZern(): MACOS rejected iElt={iElt} (not a "
            f"FreeForm surface, or modes out of [1, mFFCoef])")


# ----------------------------------------------------------------------------
# [ ] Element Surface Properties: Grid
# ----------------------------------------------------------------------------
#
# [x] elt_grid            set/get element surface grid data (N x N)
# [x] elt_grid_add        add element grid data (z-displacement)
# [x] elt_grid_scale      Scales element surface grid data.
# [x] elt_grid_npts_max   get     max surface grid sampling points (model dependent)
# [x] elt_grid_npts       get     element surface grid sampling points
#                         (@ def. Surfaces) [Nx == Ny]
# [x] elt_grid_dx         get/set element surface grid sampling spacing (dx==dy)
# [x] elt_grid_any        Determine if any Grid Srf. is defined in Rx
# [x] elt_grid_fnd        Return Srf. IDs where a Grid Srf. is defined
#                         (optional: specific Type (AsGrData, GridData, ...))
# ----------------------------------------------------------------------------

def elt_grid(srf: int,
             sampling_spacing: None | Parameter = None,
             grid_dz: None | Matrix[np.float64] = None
             ) -> None:

    """Get / Set element grid (sampling and displacement)

    Args:
        srf (int | Tuple[int] | np.int32):
           Element ID in Range: -nElt < srf <= nElt
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        sampling_spacing (None | float | Tuple[float] | Vector[float]):
           Grid surface sampling spacing (dx==dy > 0) [Length Units]
           default: None

        grid_dz (None | Matrix[np.float64])
           defines the displacement at the grid nodes for a [Ny x Nx] grid
           along the z-axis with Nx==Ny.
           default: None
           Note: grid_dz[y-axis, x-axis] from -Y to +Y and -X to +X

    Raises:
        Exception: when input constrains are not satisfied within MACOS or
                   not a Grid Surface

    Returns:
        None: when defining the Grid surface
        (dx, grid_dz): (Parameter, Matrix[np.float64]) when extracting the Grid Data

    """

    # Entry check and re-mapping iElt
    _chk_macos_and_rx_loaded()                 # pymacos & Rx loaded
    srf = _map_Elt(srf, max_rows=1).squeeze()  # Srf. check

    # read
    if sampling_spacing is None and grid_dz is None:
        npts = elt_grid_npts(srf).squeeze()
        grid_dz = np.zeros((npts, npts), dtype=float, order='F')
        sampling_spacing = np.array(0, dtype=float)
        if not lib.api.elt_srf_grid_data(srf, sampling_spacing, grid_dz, 0):
            raise Exception("MACOS threw an exception")
        return sampling_spacing, grid_dz.T

    # write
    if sampling_spacing is not None and grid_dz is not None:

        sampling_spacing = np.asarray_chkfinite(sampling_spacing, dtype=np.float64)
        if np.any(sampling_spacing <= 0):
            raise ValueError("Srf. Grid Sampling Spacing must be greater than zero")

        grid_dz = np.asarray_chkfinite(grid_dz.T, dtype=np.float64, order='F')

        if grid_dz.ndim != 2:
            raise ValueError("'grid_dz' must be a 2D numpy ndarray")
        if grid_dz.shape[0] != grid_dz.shape[1]:
            raise ValueError("'grid_dz' must be a square ndarray")
        if (grid_dz.shape[0] < 3) or (grid_dz.shape[0] > lib.api.elt_srf_grid_size_max()):
            raise ValueError("'grid_dz' size must be at least a [3x3] array")

        if not lib.api.elt_srf_grid_data(srf, sampling_spacing, grid_dz, 1):
            raise Exception("mACOS threw an exception")

    else:
        raise ValueError("define none or both grid parameters")


def elt_grid_add(srf: int | np.int32,
                 grid_dz: Matrix[np.float64]
                 ) -> None:
    """add element grid data (z-displacement)

    Args:
        srf (int | Tuple[int] | np.int32):
           Element ID in Range: -nElt < srf <= nElt
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        grid_dz (Matrix[np.float64])
           Adds displacement to the grid nodes for a [N x N] grid
           along the z-axis. with Nx==Ny.

           Note: grid_dz[y-axis, x-axis] from -Y to +Y and -X to +X

    Raises:
        Exception: when input constrains are not satisfied within MACOS or
                   not a Grid Surface
    """

    # Entry check and re-mapping iElt
    _chk_macos_and_rx_loaded()                 # pymacos & Rx loaded
    srf = _map_Elt(srf, max_rows=1).squeeze()  # Srf. check

    # write
    ok, npts = lib.api.elt_srf_grid_size(srf)
    if not ok:
        raise Exception("MACOS threw an exception")

    grid_dz = np.asarray_chkfinite(grid_dz.T, dtype=np.float64, order='F')

    if grid_dz.ndim != 2:
        raise ValueError("'grid_dz' must be a 2D numpy ndarray")
    if grid_dz.shape[0] != grid_dz.shape[1]:
        raise ValueError("'grid_dz' must be a square ndarray")
    if grid_dz.shape[0] != npts:
        raise ValueError("'grid_dz' size not equal to current defined size")

    if not lib.api.elt_srf_grid_data_add(srf, grid_dz):
        raise Exception("mACOS threw an exception")


def elt_grid_scale(srf: Surface, scalar: Parameter) -> None:
    """Scales element surface grid data.

    Args:
        srf (int | Tuple[int] | np.int32 | Vector[np.int32]):
           Element ID in Range: -nElt < srf <= nElt
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        scalar (float | Tuple[float] | Vector[float]):
           Scaling factors for scaling Surface Grid

    Raises:
        Exception: when input constrains are not satisfied within MACOS or
                   not a Grid Surface
    """

    # Entry check and re-mapping iElt
    _chk_macos_and_rx_loaded()              # pymacos & Rx loaded
    srf = _map_Elt(srf, max_rows=1)         # Srf. check

    scalar = np.asarray_chkfinite(scalar, dtype=np.float64)

    if not lib.api.elt_srf_grid_data_scale(srf, scalar):
        raise Exception("MACOS threw an exception")


def elt_grid_npts_max() -> int:

    """Get Max. surface grid sampling Size N (N == Nx == Ny).

    Raises:
        Exception: when MACOS is not initialised

    Returns:
        Max Grid surface sampling Size: (int)
    """

    if not _SYSINIT:
        raise Exception('MACOS is not yet initialised')

    return lib.api.elt_srf_grid_size_max()


def elt_grid_npts(srf: Surface) -> Index:

    """Get element grid surface grid sampling N (N == Nx == Ny).

    Args:
        srf (int | Tuple[int] | np.int32 | Vector[np.int32]):
           Element ID in Range: -nElt < srf <= nElt
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

    Raises:
        Exception: when input constrains are not satisfied within MACOS or
                   not a Grid Surface

    Returns:
        npts: (Index) Grid surface sampling
                      where a value of -1 is defined for non-grid srfs.
    """

    _chk_macos_and_rx_loaded()              # pymacos & Rx loaded
    srf = _map_Elt(srf, max_rows=1)         # Srf. check

    ok, npts = lib.api.elt_srf_grid_size(srf)
    if not ok:
        raise Exception("MACOS threw an exception")
    return npts


def elt_grid_dx(srf: Surface,
                sampling_spacing: None | Parameter = None
                ) -> None | Parameter:

    """Set/Get element grid surface grid sampling spacing (dx==dy).

    Args:
        srf (int | Tuple[int] | np.int32 | Vector[np.int32]):
           Element ID in Range: -nElt < srf <= nElt
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        sampling_spacing (None | float | Tuple[float] | Vector[float]):
           Grid surface sampling spacing (dx==dy > 0) [units Length]

    Raises:
        Exception: when input constrains are not satisfied within MACOS or
                   not a Grid Surface

    Returns:
            (None)      when defining the Grid surface sampling spacing
        dx: (Parameter) when extracting the Grid surface sampling spacing
    """

    # Entry check and re-mapping iElt
    _chk_macos_and_rx_loaded()              # pymacos & Rx loaded
    srf = _map_Elt(srf, max_rows=1)         # Srf. check

    # read
    if sampling_spacing is None:
        sampling_spacing_ = np.zeros_like(srf, dtype=float)
        if not lib.api.elt_srf_grid_spacing(srf, sampling_spacing_, 0):
            raise Exception("MACOS threw an exception")
        return sampling_spacing_

    # write
    sampling_spacing = np.asarray_chkfinite(sampling_spacing, dtype=np.float64)
    if np.any(sampling_spacing <= 0):
        raise ValueError("Srf. Grid Sampling Spacing must be greater than zero")

    if not lib.api.elt_srf_grid_spacing(srf, sampling_spacing, 1):
        raise Exception("MACOS threw an exception")


def elt_grid_any() -> bool:
    """Checks if Grid Srfs. are defined in Rx

    Raises:
        Exception: MACOS and/or Rx not loaded

    Returns:
      found (bool):
        True if Grid Srfs. is/are defined in Rx; otherwise, False
    """

    _chk_macos_and_rx_loaded()
    return bool(lib.api.elt_srf_grid_any())


def elt_grid_fnd(srf: None | Surface = None,
                 grid_srf_type: int | None = None
                 ) -> Tuple[int]:
    """Find all elements with any or specific Grid Srfs.

    Args:
        srf (None | int | Tuple[int] | np.ndarray[int], dtype=np.int32]):
           Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

           When 'None' is defined, it uses all surfaces in Rx

        grid_srf_type (None, int): None (default)
           GridData (ID=9) or AsGrdData (ID=11)

    Raises:
        Exception: MACOS Triggered error

    Returns:
        ([]):
            Empty list when no Grid Srf. is defined
        (list[int]):
            Surfaces IDs where Grid Type Srfs. are defined
    """
    _chk_macos_and_rx_loaded()

    if lib.api.elt_srf_grid_any() == 0:
        return []

    # find elements where Grps. are defined
    if srf is None:
        n_elt = lib.api.n_elt()
        srf_ = np.arange(1, n_elt+1, dtype=np.int32)
    else:
        srf_ = np.asarray_chkfinite(srf)

    if grid_srf_type is None:
        ok, n_elt_grid = lib.api.elt_srf_grid_fnd(srf_)
    else:
        ok, n_elt_grid = lib.api.elt_srf_grid_fnd_type(srf_, grid_srf_type)

    if not ok:
        raise Exception('MACOS threw an error')

    return n_elt_grid.nonzero()[0]+1



# ------------------------------------------------------------------------------
# [ ] Element Surface Properties: FreeForm
# ------------------------------------------------------------------------------
#
# [x] zrn_freeform        set/get FreeForm surface data (Zernike + Grid)
#
# FreeForm surfaces combine multiple surface descriptions:
# - FF (FreeForm): First Zernike polynomial definition with coordinate system
# - Mon (Monolithic): Second Zernike polynomial definition with coordinate system
# - Grid: Grid data displacement map with coordinate system
# ------------------------------------------------------------------------------


@dataclass(slots=True)
class LocalCSYS:
    """3D Local Coordinate System definition.

    Attributes:
        pos: Position vector [3] (origin)
        x: X-direction unit vector [3]
        y: Y-direction unit vector [3]
        z: Z-direction unit vector [3]
    """
    pos: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def __post_init__(self):
        """Validate and convert to numpy arrays."""
        self.pos = np.asarray(self.pos, dtype=np.float64).ravel()
        self.x = np.asarray(self.x, dtype=np.float64).ravel()
        self.y = np.asarray(self.y, dtype=np.float64).ravel()
        self.z = np.asarray(self.z, dtype=np.float64).ravel()

        if self.pos.size != 3:
            raise ValueError("Position must be 3-element vector")

        if self.x.size != 3 or self.y.size != 3 or self.z.size != 3:
            raise ValueError("Direction vectors must be 3-element vectors")


@dataclass(slots=True)
class ZernikeData:
    """Zernike surface data structure.

    Attributes:
        norm_rad: Zernike normalization radius (lmon)
        zern_type: Zernike type (1-9)
            1) Malacara (ANSI)     4) Norm. Malacara     7) Norm. Hex
            2) Born & Wolf         5) Norm. Born & Wolf  8) Norm. Noll
            3) Fringe              6) Norm. Fringe       9) Norm. AnnularNoll
        modes: Zernike mode indices [N] (1-66, Fringe limited to 37)
        coefs: Zernike coefficients [N] corresponding to modes
        csys:  Local Coord. System (Pose)
        annular_ratio: Inner/outer radius ratio (0-1), only for type 9
    """
    zern_type: int
    modes: np.ndarray
    coefs: np.ndarray
    norm_rad: float
    csys: LocalCSYS
    annular_ratio: float = 0.0

    def __post_init__(self):
        """Validate Zernike data after initialization."""
        self.norm_rad = np.asarray_chkfinite(self.norm_rad, dtype=np.float64)
        self.modes = np.asarray_chkfinite(self.modes, dtype=np.int32)
        self.coefs = np.asarray_chkfinite(self.coefs, dtype=np.float64)

        if self.norm_rad <= 0:
            raise ValueError("Normalization radius must be > 0")

        if not (1 <= self.zern_type <= 9):
            raise ValueError("Zernike type must be in range [1-9]")

        if self.modes.shape != self.coefs.shape:
            raise ValueError("Modes and coefficients must have same shape")

        if np.any(self.modes < 1) or np.any(self.modes > 66):
            raise ValueError("Zernike modes must be in range [1-66]")

        if self.zern_type in [3, 6] and np.any(self.modes > 37):
            raise ValueError("Fringe Zernike types limited to 37 modes")

        if not (0 <= self.annular_ratio <= 1):
            raise ValueError("Annular ratio must be in range [0-1]")

    def to_tuple(self) -> Tuple:
        """Convert to tuple format matching pymacos.elt.zernike.get() output.

        Returns:
            Tuple of (type, modes, coefs, norm_rad, pos, x_dir, y_dir, z_dir)

        Example:
            >>> zdata = ZernikeData(10.0, 6, np.arange(1, 11), np.zeros(10))
            >>> lmon, ztype, modes, coefs, ratio = zdata.to_tuple()
        """
        return (
            self.zern_type,
            self.modes,
            self.coefs,
            self.norm_rad,
            self.csys.pos,
            self.csys.x,
            self.csys.y,
            self.csys.z
        )


@dataclass(slots=True)
class GridData:
    """Grid displacement data structure.

    Attributes:
        dx: Grid sampling spacing (dx == dy)
        mat: Grid displacement matrix [Ny x Nx] (must be square, min 3x3)
    """
    dx: float
    mat: np.ndarray
    csys: LocalCSYS

    def __post_init__(self):
        """Validate grid data."""
        self.dx = np.asarray_chkfinite(self.dx, dtype=np.float64)

        if self.dx.ndim != 0:
            raise ValueError("Grid spacing must be a scalar")

        if self.dx <= 0:
            raise ValueError("Grid spacing must be > 0")

        self.mat = np.asarray_chkfinite(self.mat, dtype=np.float64, order='F')

        if self.mat.ndim != 2:
            raise ValueError("Grid matrix must be 2D")

        ny, nx = self.mat.shape
        if nx != ny:
            raise ValueError("Grid matrix must be square (Nx == Ny)")
        if nx < 3:
            raise ValueError("Grid matrix must be at least 3x3")


def zrn_freeform(
    srf: int,
    zrn_1=_PRESERVE,
    zrn_2=_PRESERVE,
    grid=_PRESERVE,
) -> tuple[ZernikeData | None, ZernikeData | None, GridData | None] | None:
    """Get or set FreeForm surface data using dataclass structures.

    FreeForm surfaces combine up to three independent surface descriptions:
    1. FF (FreeForm) Zernike terms with coordinate system
    2. Mon (Monolithic) Zernike terms with coordinate system
    3. Grid data displacement map with coordinate system

    **Getter Mode:** call ``zrn_freeform(srf)`` with no other arguments
    to retrieve the current (FF, Mon, Grid) triple.

    **Setter Mode:** pass any of ``zrn_1`` / ``zrn_2`` / ``grid`` to
    update the corresponding component.  Three sentinel semantics:

      - default (``_PRESERVE``)  -> leave that component unchanged;
        the function reads the current state via an internal get and
        substitutes it so the setter call carries the existing values
        forward.
      - ``None``                  -> EXPLICITLY clear/wipe that
        component (its activity flag goes to FAIL, all data zeroed).
      - ``ZernikeData`` / ``GridData`` -> write the new value.

    This avoids the old behaviour where ``zrn_freeform(srf, zrn_1=X)``
    would silently wipe ``zrn_2`` and ``grid`` -- callers no longer
    need to ``get`` first and pass the unchanged components back in.

    Args:
        srf: Element ID (single surface only)
        zrn_1: First Zernike component (FF) as ZernikeData or None
        zrn_2: Second Zernike component (Mon) as ZernikeData or None
        grid: Grid displacement data as GridData or None

    Returns:
        Tuple of (zrn_1, zrn_2, grid) when getting (all params None):
            - zrn_1: ZernikeData or None (FF component)
            - zrn_2: ZernikeData or None (Mon component)
            - grid: GridData or None (Grid component)

        None when setting (any param provided)

    Raises:
        RuntimeError: If MACOS is not initialized, Rx not loaded, or API call fails
        ValueError: If srf is not a single surface ID

    See Also:
        LocalCSYS: 3D coordinate system dataclass
        ZernikeData: Zernike polynomial data with validation
        GridData: Grid displacement map data
    """
    _chk_macos_and_rx_loaded()

    srf_mapped = _map_Elt(srf)
    if srf_mapped.size != 1:
        raise ValueError("Only accepting a single surface ID")
    srf = srf_mapped.item()

    n_zrn_coef_max = 66

    def _init_zernike_arrays(ncoef: int) -> tuple:
        """Initialize empty Zernike arrays for Fortran getter."""
        return (
            np.array(0, dtype=np.int32),
            np.array(0, dtype=np.int32),
            np.zeros(ncoef, dtype=np.int32),
            np.zeros(ncoef, dtype=float),
            np.array(0.0, dtype=float),
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
        )

    def _init_grid_arrays(srf_id: int | None = None) -> tuple:
        """Initialize empty grid arrays for Fortran getter."""
        if srf_id is None:
            g_mat = np.zeros((1, 1), dtype=float, order='F')
        else:
            ok, n_grid = lib.api.elt_srf_grid_size(srf_id)
            n_grid = n_grid.item()
            g_mat = np.zeros((max(n_grid, 1), max(n_grid, 1)), dtype=float, order='F')
        return (
            np.array(0, dtype=np.int32),
            np.array(0.0, dtype=float),
            g_mat,
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
        )

    def _zrn_from_api(if_active, ztype, modes, coefs, norm_rad, pos, x, y, z):
        """Convert Fortran output arrays to ZernikeData dataclass.

        Returns ``modes = [1..last_nonzero]`` with the full prefix of
        coefficients (intermediate zeros PRESERVED) so a user-set
        ``Z4 = 0.0`` is reported on get and survives set/get round-
        trips.  Trailing zeros beyond the last non-zero entry are
        dropped because the Fortran getter pads the buffer out to
        ``mZernCoef = 66`` regardless of how many modes the Rx
        actually declared.
        """
        if not if_active:
            return None
        nz = np.flatnonzero(coefs)
        if nz.size == 0:
            # active flag set but every coefficient is zero (e.g., the
            # component is declared in the Rx but currently un-perturbed).
            # Return a single-slot placeholder so the round-trip is
            # explicit rather than silently producing None.
            last = 0
        else:
            last = int(nz[-1])
        active_modes = np.arange(1, last + 2, dtype=np.int32)
        active_coefs = np.asarray(coefs[:last + 1], dtype=float).copy()
        return ZernikeData(
            norm_rad=float(norm_rad),
            zern_type=int(ztype),
            modes=active_modes,
            coefs=active_coefs,
            csys=LocalCSYS(pos, x, y, z)
        )

    def _grid_from_api(if_active, dx, mat, pos, x, y, z):
        """Convert Fortran output arrays to GridData dataclass."""
        if not if_active:
            return None
        return GridData(
            dx=float(dx),
            mat=np.ascontiguousarray(mat),
            csys=LocalCSYS(pos, x, y, z)
        )

    def _zrn_to_api(zdata, ncoef: int) -> tuple:
        """Convert ZernikeData dataclass to Fortran arrays for setter."""
        if zdata is None or len(zdata.modes) == 0:
            return _init_zernike_arrays(ncoef)
        return (
            1,
            zdata.zern_type,
            zdata.modes,
            zdata.coefs,
            zdata.norm_rad,
            zdata.csys.pos,
            zdata.csys.x,
            zdata.csys.y,
            zdata.csys.z
        )

    def _grid_to_api(gdata, srf_id: int) -> tuple:
        """Convert GridData dataclass to Fortran arrays for setter."""
        if gdata is None or gdata.dx == 0:
            return _init_grid_arrays(srf_id)
        return (
            1,
            gdata.dx,
            np.asfortranarray(gdata.mat),
            gdata.csys.pos,
            gdata.csys.x,
            gdata.csys.y,
            gdata.csys.z
        )

    def _do_get():
        z1_if, z1_t, z1_m, z1_c, z1_r, z1_p, z1_x, z1_y, z1_z = \
                                      _init_zernike_arrays(n_zrn_coef_max)
        z2_if, z2_t, z2_m, z2_c, z2_r, z2_p, z2_x, z2_y, z2_z = \
                                      _init_zernike_arrays(n_zrn_coef_max)
        g_if, g_d, g_m, g_p, g_x, g_y, g_z = _init_grid_arrays(srf)

        ok = lib.api.elt_srf_zrn_freeform(
            srf,
            z1_if, z1_t, z1_m, z1_c, z1_r, z1_p, z1_x, z1_y, z1_z,
            z2_if, z2_t, z2_m, z2_c, z2_r, z2_p, z2_x, z2_y, z2_z,
            g_if, g_d, g_m, g_p, g_x, g_y, g_z, 0)

        if not ok:
            raise RuntimeError("Failed to retrieve FreeForm surface data")

        return (
          _zrn_from_api(z1_if, z1_t, z1_m, z1_c, z1_r, z1_p, z1_x, z1_y, z1_z),
          _zrn_from_api(z2_if, z2_t, z2_m, z2_c, z2_r, z2_p, z2_x, z2_y, z2_z),
          _grid_from_api(g_if, g_d, g_m, g_p, g_x, g_y, g_z)
        )

    is_getter = (zrn_1 is _PRESERVE) and (zrn_2 is _PRESERVE) \
                                    and (grid is _PRESERVE)
    if is_getter:
        return _do_get()

    # Setter: preserve un-specified components by reading current state.
    # None stays explicit "clear" semantics; _PRESERVE -> substitute.
    if (zrn_1 is _PRESERVE) or (zrn_2 is _PRESERVE) or (grid is _PRESERVE):
        cur_z1, cur_z2, cur_g = _do_get()
        if zrn_1 is _PRESERVE: zrn_1 = cur_z1
        if zrn_2 is _PRESERVE: zrn_2 = cur_z2
        if grid  is _PRESERVE: grid  = cur_g

    z1_if, z1_t, z1_m, z1_c, z1_r, z1_p, z1_x, z1_y, z1_z = \
                                    _zrn_to_api(zrn_1, n_zrn_coef_max)
    z2_if, z2_t, z2_m, z2_c, z2_r, z2_p, z2_x, z2_y, z2_z = \
                                    _zrn_to_api(zrn_2, n_zrn_coef_max)
    g_if, g_d, g_m, g_p, g_x, g_y, g_z = _grid_to_api(grid, srf)

    ok = lib.api.elt_srf_zrn_freeform(
        srf,
        z1_if, z1_t, z1_m, z1_c, z1_r, z1_p, z1_x, z1_y, z1_z,
        z2_if, z2_t, z2_m, z2_c, z2_r, z2_p, z2_x, z2_y, z2_z,
        g_if, g_d, g_m, g_p, g_x, g_y, g_z, 1
    )

    if not ok:
        raise RuntimeError("Failed to set FreeForm surface data")



# ------------------------------------------------------------------------------
# [ ] Element Group Management
# ------------------------------------------------------------------------------
#     [ ] elt_grp_any   ! Check if Rx has any Elt Grp. defined
#     [ ] elt_grp_fnd   : returns 1/0 if srf has a Grp defined
#
#     [ ] elt_grp       : set/get elt Grp information
#     [ ] elt_grp_rm    : remove  elt Grp   (single, set or all)
# ------------------------------------------------------------------------------


def elt_grp_max_size(srf=None) -> np.int32:
    # chk: pymacos & Rx loaded
    _chk_macos_and_rx_loaded()

    # find elements where Grps. are defined
    if srf is None:
        srf = np.arange(1,_NELT+1, dtype=np.int32)
    else:
        srf = np.asarray_chkfinite(srf)

    ok, elt_grp_max = lib.api.elt_grp_max(srf)
    if not ok:
        raise Exception('MACOS threw an error')
    return elt_grp_max


def elt_grp_any() -> bool:
    """Check if Rx has any Elt Grp. defined

    Args:
        None

    Returns:
        bool: True if Rx is loaded and Elt. Grp. is defined
    """
    return lib.api.elt_grp_any() == 1


def elt_grp_fnd(srf: None | Surface = None
                ) -> Tuple[int] | Surface:
    """Find all elements where element Grps. are defined.

    Args:
        srf (None | int | Tuple[int] | np.ndarray[int], dtype=np.int32]):
           Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

           When 'None' is defined, it uses all surfaces in Rx

    Raises:
        Exception: MACOS Triggered error

    Returns:
        ([]):
            Empty list when no Elt. Grp. is defined
        (list[int]):
            Surfaces IDs where Element Grps. are defined
    """
    _chk_macos_and_rx_loaded()

    # quick check
    if lib.api.elt_grp_any() == 0:
        return None

    # find elements where Grps. are defined
    if srf is None:
        srf = np.arange(lib.api.n_elt(), dtype=np.int32)+1
    else:
        srf = np.asarray_chkfinite(srf)

    ok, n_elt_grp = lib.api.elt_grp_fnd(srf)
    if not ok:
        raise Exception('MACOS threw an error')

    return n_elt_grp.nonzero()[0]+1


def elt_grp(srf: Surface,
            srfs_in_grp:None | Surface = None
            ) -> None | Tuple[Tuple[int]]:
    """set / get Element Grp. Definitions

    Args:
        srf (int | Tuple[int] | np.ndarray[int], dtype=np.int32]):
           Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

           Note: when defining an Elt. Grp. ONLY a single Surface is accepted

        srfs_in_grp (None | int | Tuple[int] | np.ndarray[int], dtype=np.int32], optional):
           Default to None
           Element IDs, 1D-array (Range: -nElt < srf[j] <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

           When not specified (= None), Ele. Grp. Information will be retrieved.

    Raises:
        Exception: MACOS Triggered error

    Returns:
        None:
            when Ele. Grp. definitions were defined

        Tuple[Tuple[int]]:
            Surface IDs defined in Element Grps. at specified Surfaces
    """
    _chk_macos_and_rx_loaded()
    srf = _map_Elt(srf)

    # retrieve Ele. Grp. information
    if srfs_in_grp is None:

        n_elt_grp_max = lib.api.elt_grp_max_all()
        ok, grp_srfs, n_grp_srfs = lib.api.elt_grp_get(srf, n_elt_grp_max)

        if not ok :
            raise Exception("MACOS threw an exception")

        # export list of the surface IDs defined within the Ele. Grp.
        return [grp_srfs[:n, i].tolist() for i, n in enumerate(n_grp_srfs)]

    # define Ele. Grp. information
    if srf.size != 1:
        raise ValueError("Setting an Elt. Grp. only for single surface")

    srfs_in_grp = _map_Elt(srfs_in_grp)
    if len(srfs_in_grp) != len(set(srfs_in_grp)):
        raise ValueError("Cannot define same surface multiple times")

    if not lib.api.elt_grp_set(srf[0], srfs_in_grp):
        raise Exception("MACOS threw an exception")


def elt_grp_rm(srf: Surface) -> None:
    """Remove Element-Grp. Settings at specified Element(s)

    Args:
        srf (int | Tuple[int] | np.ndarray[int], dtype=np.int32]):
           Element IDs, 1D-array (Range: -nElt < srf <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

    Raises:
        Exception: MACOS Triggered error
    """
    _chk_macos_and_rx_loaded()
    srf = _map_Elt(srf)

    if not lib.api.elt_grp_del(srf):
        raise Exception("MACOS threw an exception")


def elt_grp_wipe() -> None:
    """Wipes out all Element-Grp. Settings from Rx
    """
    _chk_macos_and_rx_loaded()
    if not lib.api.elt_grp_del_all():
        raise Exception("MACOS threw an exception")


# ------------------------------------------------------------------------------
# [ ] System Perturbation
#     --------------------------------------------------------------------------
#     [ ] prb_src     perturbSrc
#     [x] prb_elt
#     [x] prb_grp

#     [ ] perturbElt_METROLOGY_NODES
# ------------------------------------------------------------------------------


@_chk_if_macos_and_rx_loaded
def prb_elt(srf: Surface,
            prb: Matrix[np.float64],
            glb_csys: bool | Tuple[bool] | Vector[bool]) -> None:
    """Apply 6-DoF rigid body perturbations to elements defined by srf

    Args:
        srf (int | Tuple[int] | np.ndarray[int], dtype=np.int32]): []
           Element IDs, 1D-array (Range: -nElt < srf <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        prb (np.ndarray[Tuple[int, int], np.dtype[float]]): [6 x N]
            Rigid Body Perturbation Vector: (column Vector)
                   = [[Rx,Ry,Rz,Tx,Ty,Tz]_1;...;[Rx,Ry,Rz,Tx,Ty,Tz]_N]
                        Rotation    Vector: R = [Rx,Ry,Rz]_i
                        Translation Vector: T = [Tx,Ty,Tz]_i

        glb_csys (bool | np.ndarray[Tuple[int], np.dtype[np.bool_]]): [1 x N]
            (1=True ):Global Coordinate Frame
            (0=False):Local Element Coordinate Frame  (must be defined in Rx)

    Note:
        N defines the number of defined surfaces in 'srf'

    Raises:
        ValueError: if prb is not a [6xN] array that is finite
        Exception:  MACOS Triggered and/or not init. or Rx loaded

    """
    srf = _map_Elt(srf)  # iElt check => 1D array
    n_srf = len(srf)

    prb = np.asarray_chkfinite(prb, dtype=float)
    if prb.shape != (6, n_srf):
        raise ValueError("'prb array' must be a [6 x N] ndarray")
    elif not np.all(np.isreal(prb)):
        raise ValueError("'prb array' values must be real and finite")

    glb_csys = np.asarray_chkfinite(glb_csys, dtype=np.int32).reshape(-1)
    # np.int32(_chk_values_1d(glb_csys, n_srf))
    if len(glb_csys) != n_srf:
        raise ValueError("'glb_csys' vector must be a [1 x N] ndarray")

    # ToDo
    # # call external tracking (before state is modified)
    # if _METROLOGY_NODES is not None:
    #     _perturbElt_METROLOGY_NODES(srf, prb, glb_csys.ravel(), n_srf)

    if not lib.api.prb_elt(srf, prb, glb_csys):
        raise Exception("MACOS threw an exception")


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Optical System Analysis
# ------------------------------------------------------------------------------
# [ ] TraceWavefront
# [ ] ray_info_get
# [ ] setRayInfo
# ------------------------------------------------------------------------------
@_chk_if_macos_and_rx_loaded
def prb_grp(srf: Surface,
            prb: Matrix[np.float64],
            glb_csys: bool | Tuple[bool] | Vector[np.bool_]) -> None:
    """Apply 6-DoF rigid body grp. prb. to selected elements defined by elt

    Apply 6-DoF rigid body perturbation to selected elements identified by
    "EltGrp" keyword at a given element. If no EltGrp is defined at the Elt.,
    the element will be skipped and no warning message given.

    Args:
        srf (int | Tuple[int] | np.ndarray[int], dtype=np.int32]): []
           Element IDs, 1D-array (Range: -nElt < srf <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        prb (np.ndarray[float]): [6 x N]
            Rigid Body Perturbation Vector: (column Vector)
                   = [[Rx,Ry,Rz,Tx,Ty,Tz]_1;...;[Rx,Ry,Rz,Tx,Ty,Tz]_N]
                        Rotation    Vector: R = [Rx,Ry,Rz]_i
                        Translation Vector: T = [Tx,Ty,Tz]_i

        global_csys (bool): [1 x N]
            (1=True ):Global Coordinate Frame
            (0=False):Local Element Coordinate Frame  (must be defined in Rx)

    Note:
        N defines the number of defined Elt IDs

    Raises:
        ValueError: if prb is not a [6xN] array that is finite
        Exception:  MACOS Triggered and/or not init. or Rx loaded
    """
    srf = _map_Elt(srf)  # iElt check => 1D array
    n_srf = len(srf)

    prb = np.asarray_chkfinite(prb, dtype=float)
    if prb.shape != (6, n_srf):
        raise ValueError("'prb array' must be a [6 x N] array")

    glb_csys = np.asarray_chkfinite(glb_csys, dtype=np.int32).reshape(1, -1)
    if glb_csys.shape != (1, n_srf):
        raise ValueError("'glb_csys array' must be a [1 x N] array")

    # call external tracking (before state is modified)
    # if _METROLOGY_NODES is not None:
    #     _perturbEltGrp_METROLOGY_NODES(srf, prb, glb_csys.flatten())

    if not lib.api.prb_elt_grp(srf, prb, glb_csys):
        raise Exception("'SMACOS' threw an exception")


# ------------------------------------------------------------------------------
# [ ] System requests / queries / Analysis / Tools / ...
#     --------------------------------------------------------------------------
#     [x] getRayInfo  Trace all rays from Src. to Srf at current grid sampling.
#     [x] modify      reset ray-trace state to trace from source
#     [x] opd         Get OPD at last ray-traced state.
#
#     [x] fex         set XP  (FEX cmd) --- set based on wavefront state
#     [x] xp          set/get XP parameters (Kr, Psi(L,M,N), Vpt(x,y,z))
#     [x] stop        set/get stop information
#     --------------------------------------------------------------------------


def getRayInfo(nRays):   # ToDo -- testing
    """
    Retrieve Ray-Trace Data (Pos & Dir) from previous call to traceWavefront(...)

    :param    nRays: [1x1,I]: Number of traced rays (obtained via pymacos.trace_rays(...) )

    :return  rayPos:   [3xnRays,D]: = [[x1,y1,z1],...] Ray-Srf. Intersection Point
    :return  rayDir:   [3xnRays,D]: = [[L1,M1,N1],...] Ray Direction before surface
    :return     opl:   [1xnRays,D]: Optical Path Length from Src. Srf to last traced Srf. (trace_rays)
    :return   rayOK:   [1xnRays,L]: (True=1) if valid ray; otherwise, (False=0)
    :return   RayPass: [1xnRays,L]: (True) if ray is not blocked; (False) otherwise
    """

    _chk_macos_and_rx_loaded()

    # chk: param
    if not isinteger(nRays):
        raise TypeError("Number of rays must be an integer value")
    elif nRays < 0:
        raise ValueError("Number of rays must be greater than zero")

    ok, rayPos, rayDir, opl, rayOK, rayPass = lib.api.ray_info_get(np.int32(nRays))

    if not ok:
        raise Exception("'getRayInfo' threw an exception")

    return rayPos, rayDir, opl, rayOK.astype(bool), rayPass.astype(bool)


def setRayInfo(rayPos, rayDir, opl, rayOK):
    """
    Replace Ray-Trace Data at current trace location.

    :param  rayPos:  [3xK,D]: = [[x1,y1,z1],...] Ray-Srf. Intersection Point
    :param  rayDir:  [3xK,D]: = [[L1,M1,N1],...] Ray Direction before surface
    :param     opl:  [1xK,D]: Optical Path Length from Src. Srf to last traced Srf. (trace_rays)
    :param   rayOK:  [1xK,I]: (True=1) if valid ray; otherwise, (False=0)
                        where  K => # of rays to be traced

    Note: Used for Rx validation where CodeV rays are injected at the Entry Port (Source Srf.)
           ==> WARNING -- you really must understand how to use this functionality correctly
    """

    _chk_macos_and_rx_loaded()

    # chk: input parameters
    opl = _chk_values_1d(opl, -1)
    K = opl.shape[1]
    rayOK = _chk_values_1d(rayOK, K)
    _chk_values_2d(rayPos, 3, K)
    _chk_values_2d(rayDir, 3, K)

    if not lib.api.ray_info_set(rayPos, rayDir, opl, rayOK):
        raise Exception("'setRayInfo' threw an exception")


def traceWavefront(srf) -> tuple[np.float64, int, int]:
    return trace_rays(srf)


def trace_rays(srf: int | Tuple[int] | np.int32
               ) -> tuple[np.float64, int, int]:
    """Trace all rays from source to surface 'srf' at current grid sampling.

    Args:
        srf (int | Tuple[int] | np.ndarray[int] | np.int32):
           Element IDs, 1D-array (Range: -nElt < srf <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

    Raises:
        Exception: MACOS Triggered error

    Returns:
        np.ndarray[float]:
            rms_wfe -- rms Wavefront Error

        np.ndarray[int]:
            n_rays -- Number of traced rays

        np.ndarray[int]:
            n_pts -- Wavefront sampling (nGridPts x nGridPts)
    """

    _chk_macos_and_rx_loaded()
    srf = _map_Elt(srf).squeeze()

    ok, rms_wfe, n_rays, n_pts = lib.api.trace_rays(srf)
    if not ok:
        raise Exception('MACOS" trace_rays failure')

    return rms_wfe, n_rays, n_pts


def modify() -> None:
    """Reset ray-trace state to trace from source

    Executes a "MODIFY" cmd to reset ray-trace dependent
    parameters, which is recommended after a Rx modification.

    Raises:
        Exception: MACOS execution failure
    """
    _chk_macos_and_rx_loaded()  # pymacos & Rx loaded

    if not lib.api.modified_rx():
        raise Exception("failed to reset MACOS status")


def opd_val(nGridPts) -> Matrix[np.float64]:
    return opd()


def opd(orient: str = "raw", sign: str = "opl") -> Matrix[np.float64]:
    """Retrieve Optical Path Difference (OPD) at last ray-traced state.

    Requirement:
        The OPD can be obtained _after_ running a trace_rays(srf).
        For the OPD at the Exit Pupil, run a 'trace_rays(-2)'.

    Conventions (see mmacos/doc/opd_conventions.md for the full story):
        The raw array is OPD[i, j] with FIRST index i = global X and
        SECOND index j = global Y -- identical in the CLI, mmacos and
        pymacos.  Sign: a ray LONGER than the reference is POSITIVE
        (optical path difference).  The reference is the chief ray
        when it survives the trace, else the bundle mean (the map
        comes back mean-removed).

    Args:
        orient: "raw" (default) -- the engine array as stored,
            [i, j] = (X, Y).  "xy" -- transposed so ROWS run along Y
            and COLUMNS along X (standard image convention); display
            with plt.imshow(W, origin="lower") for an x-right / y-up
            view matching the CLI plot.  (Equivalent to W_raw.T)
        sign: "opl" (default) -- engine convention, longer path
            positive.  "wavefront" -- negated: the interferometer-
            style wavefront-error map, and the sign PROPER's
            prop_add_phase expects.

    Raises:
        Exception: MACOS Triggered error
        ValueError: bad orient / sign value

    Returns:
        Matrix[np.float64]:
            opd: Optical Path Difference where size (nGridPts x nGridPts)

    Example:
        __ = pymacos.trace_rays(-2)  # Trace rays to XP
        pymacos.opd()                # get OPD map (raw, OPL sign)
        pymacos.opd(orient="xy", sign="wavefront")
    """
    if orient not in ("raw", "xy"):
        raise ValueError(f"opd: orient must be 'raw' or 'xy', got {orient!r}")
    if sign not in ("opl", "wavefront"):
        raise ValueError(f"opd: sign must be 'opl' or 'wavefront', got {sign!r}")
    _chk_macos_and_rx_loaded()

    # OPD map
    npts = lib.api.get_src_sampling()[1]   # == nGridPts
    ok, opd = lib.api.opd_val(npts)

    if not ok:
        raise Exception("MACOS: 'opd' threw an exception")

    if orient == "xy":
        opd = opd.T
    if sign == "wavefront":
        opd = -opd

    return opd


def complex_field(srf: int | np.int32,
                  reset_trace: bool = True,
                  plane: int = 0) -> Matrix[np.complex128]:
    """Retrieve the diffraction-grid complex field at element `srf`.

    Triggers macos's full propagation chain (via the 'INT' command),
    then returns WFElt(:,:, iEltToiWF(srf)) as a NumPy complex128
    array of shape (mdttl, mdttl).  This is the same complex
    amplitude macos's downstream propagation routines (FFPROP,
    NFPROP, PPPROP, ...) operate on; useful for handing off the
    wavefront to an external physical-optics engine like PROPER.

    Args:
        srf: Element ID (Range: -nElt < srf <= nElt).  Negative
             values index from the end.
        reset_trace: If True (default), runs MODIFY first so the trace
             starts from the source.
        plane: 0 (default) returns the element's own wavefront -- the
             historical behaviour.  1, 2 or 3 returns a single Cartesian
             FIELD COMPONENT Ex, Ey or Ez.  In vector-diffraction mode
             the three wavefront storage planes are repurposed as the
             components of one wavefront, so this is the only way to see
             how they contribute to a propagated intensity (which sums
             them).  Requesting 1..3 with vector diffraction OFF raises:
             in scalar mode plane k is an unrelated wavefront, not a
             field component, and returning it would invite exactly the
             misreading the option exists to prevent.

             The planes add in INTENSITY, not amplitude:
             ``sum(abs(complex_field(s, plane=k))**2 for k in (1,2,3))``
             equals ``intensity(s)``.

    Returns:
        2D complex128 ndarray (N x N) of the complex amplitude at the
        diffraction-grid sampling.  N equals macos's mdttl, which for
        the current model_size is typically equal to model_size.

    Raises:
        Exception: MACOS not loaded, invalid srf, or element has no
                   diffraction wavefront slot (purely geometric path).

    Example:
        pymacos.load('Rx_Coro.in')
        field = pymacos.complex_field(2)        # at Elt 2
        amp    = np.abs(field)
        phase  = np.angle(field)                # radians
        # opd_metres = phase / (2*np.pi) * wavelength_m
    """
    _chk_macos_and_rx_loaded()

    iElt = _map_Elt(srf)
    if hasattr(iElt, '__len__'):
        if len(iElt) != 1:
            raise Exception("complex_field() takes a single srf, got "
                            f"{len(iElt)}")
        iElt = int(iElt[0])

    ok, n = lib.api.cfield_cmd(int(iElt), int(bool(reset_trace)))
    if not ok or n == 0:
        raise Exception("MACOS: 'complex_field' propagation failed at "
                        f"Elt {iElt}")

    if plane not in (0, 1, 2, 3):
        raise ValueError(f"complex_field: plane must be 0..3, got {plane}")
    ok, re_arr, im_arr = lib.api.cfield_plane_get(n, int(iElt), int(plane))
    if not ok:
        raise Exception(
            f"MACOS: complex-field buffer retrieval failed at Elt "
            f"{iElt} (element may have no diffraction wavefront slot; "
            f"plane 1..3 additionally requires vector diffraction ON)")

    return re_arr + 1j * im_arr


def intensity(srf: int | np.int32,
              reset_trace: bool = True) -> Matrix[np.float64]:
    """INT: Compute intensity (modulus squared of complex amplitude) at
    the given element. Equivalent to MACOS interactive 'INT <srf>'.

    The result is the diffraction-grid (mdttl x mdttl) intensity at the
    wavefront's native sampling. For a pixelated detector (configurable
    pixel size + count) use a future 'pix()' wrapper; intensity() takes
    no pixelization arguments.

    Args:
        srf: Element ID (Range: -nElt < srf <= nElt). Negative values
             index from the end; -1 is the last surface (image plane).
        reset_trace: If True (default), runs MODIFY first so the trace
             starts from the source.  Set False to keep the prior trace
             state (e.g. after a perturbation sequence).

    Returns:
        2D ndarray (N x N) of intensity values, float64. N equals
        macos's diffraction grid size (param_mod.mdttl), which for the
        active model_size is typically equal to model_size.

    Raises:
        Exception: MACOS-side failure (system not loaded, invalid srf,
                   or unallocated intensity buffer).

    Example:
        pymacos.load('Rx_Cass_FarField.in')
        psf = pymacos.intensity(6)     # focal plane
    """
    _chk_macos_and_rx_loaded()

    iElt = _map_Elt(srf)
    if hasattr(iElt, '__len__'):
        if len(iElt) != 1:
            raise Exception("intensity() takes a single srf, got "
                            f"{len(iElt)}")
        iElt = int(iElt[0])

    ok, n = lib.api.int_cmd(int(iElt), int(bool(reset_trace)))
    if not ok or n == 0:
        raise Exception("MACOS: 'INT' command failed at "
                        f"Elt {iElt}")

    ok, arr = lib.api.int_get(n)
    if not ok:
        raise Exception("MACOS: intensity buffer retrieval failed")

    return arr


def compose(srf: int | np.int32,
            wavelengths,
            npix: int,
            dx: float,
            dx_unit: str = 'm') -> Matrix[np.float64]:
    """COMPOSE: assemble a multi-wavelength (or multi-field) PSF on a
    FIXED pixel grid at element ``srf`` (MACOS 'COMPOSE' + 'ADD').

    For each wavelength in ``wavelengths`` the source wavelength is set,
    the field is propagated to ``srf``, and its intensity is accumulated
    onto a single ``npix`` x ``npix`` detector grid of pitch ``dx``.  The
    result is the **incoherent** sum -- a broadband PSF.  Each wavelength
    has a different native diffraction sampling (focal dx is proportional
    to lambda), and COMPOSE resamples each onto the common pixel grid, so
    this is the right primitive for broadband scoring rather than summing
    raw intensity() arrays.

    Note: only incoherent intensity composition is supported; the
    engine's coherent complex-amplitude path ('CADD') is unimplemented.

    Args:
        srf: element ID where the composite image is formed (image plane).
        wavelengths: iterable of source wavelengths, in the prescription's
             WaveUnits (e.g. micron).  May be a single broadband sampling
             or a set of field/spectral points.
        npix: pixels per side of the composite grid.
        dx: detector pixel size, in ``dx_unit``.
        dx_unit: 'm' (default), 'mm', or 'native' (prescription BaseUnits).

    Returns:
        2D ndarray (npix x npix) of accumulated intensity, float64.

    Raises:
        Exception: MACOS-side failure or bad arguments.

    Example:
        pymacos.load('Rx_Coro_FPM.in')
        lams = [0.80, 0.85, 0.90]              # micron
        psf  = pymacos.compose(21, lams, npix=128, dx=5e-6)   # 5 um pixels
    """
    _chk_macos_and_rx_loaded()

    iElt = _map_Elt(srf)
    if hasattr(iElt, '__len__'):
        if len(iElt) != 1:
            raise Exception("compose() takes a single srf, got "
                            f"{len(iElt)}")
        iElt = int(iElt[0])

    wl = [float(w) for w in wavelengths]
    if len(wl) == 0:
        raise Exception("compose(): wavelengths is empty")
    if any(w <= 0 for w in wl):
        raise Exception("compose(): wavelengths must be positive")
    npix = int(npix)

    # Engine COMPOSE wants the pixel size in BaseUnits; dx_at() and this
    # API speak SI metres by default.  Convert metres -> BaseUnits via CBM.
    if dx_unit == 'native':
        dxpix = float(dx)
    else:
        to_m = {'m': 1.0, 'mm': 1.0e-3}.get(dx_unit)
        if to_m is None:
            raise Exception(f"compose(): bad dx_unit {dx_unit!r}; "
                            "use 'm', 'mm', or 'native'")
        ok, cbm = lib.api.base_unit_to_metres()
        if not ok or cbm <= 0:
            raise Exception("compose(): base-unit conversion failed")
        dxpix = float(dx) * to_m / cbm

    ok = lib.api.compose_start(int(iElt), npix, dxpix)
    if not ok:
        raise Exception(f"MACOS: 'COMPOSE' failed at Elt {iElt}")

    for lam in wl:
        src_wvl(lam)
        # re-MODIFY + propagate to iElt at this wavelength, then accumulate
        ok, n = lib.api.int_cmd(int(iElt), 1)
        if not ok or n == 0:
            raise Exception(f"MACOS: propagation failed at Elt {iElt} "
                            f"for wavelength {lam}")
        ok = lib.api.compose_add(0)        # do_plot = False
        if not ok:
            raise Exception(f"MACOS: 'ADD' failed at wavelength {lam}")

    ok, arr = lib.api.compose_get(npix)
    if not ok:
        raise Exception("MACOS: composite image retrieval failed")

    return arr


_DX_AT_UNIT_FACTORS = {
    'm':      1.0,            # SI metres (default)
    'mm':     1.0e3,          # millimetres
    'native': None,           # whatever the prescription's BaseUnits are
}


def dx_at(srf: int | np.int32, unit: str = 'm') -> float:
    """Diffraction-grid pixel pitch at element ``srf``.

    Reads macos's per-element dxElt(iElt) at full double precision.
    Returns the same value macos prints as 'dx2=' in propagation
    diagnostics, but without the 5-sig-fig display truncation -- useful
    for cross-codes that must seed their own grids at exactly macos's
    sampling (e.g. PROPER's prop_begin).

    Requires that a propagation has reached ``srf`` first (TRACE or
    INT/CFIELD/etc.); a slot that has never been touched by diffraction
    propagation has dxElt(iElt)=0 and this call raises.

    Args:
        srf:  Element id (1..nElt, or -k from end).
        unit: Output unit. One of:
              - 'm'      (default) -- SI metres.
              - 'mm'     -- millimetres.
              - 'native' -- the prescription's own BaseUnits (mm, cm,
                            m, in, ...).  No conversion applied -- this
                            is dxElt(iElt) verbatim.  Callers using
                            this option are responsible for knowing
                            what BaseUnits= the loaded prescription
                            declared.

    Returns:
        Pixel pitch as a float, in the requested unit.

    Raises:
        Exception: Rx not loaded, srf out of range, dxElt unallocated
                   / zero at that slot, or invalid ``unit`` string.
    """
    if unit not in _DX_AT_UNIT_FACTORS:
        raise Exception(
            f"dx_at(): unit={unit!r} not recognised; expected one of "
            f"{sorted(_DX_AT_UNIT_FACTORS)}")

    _chk_macos_and_rx_loaded()

    iElt = _map_Elt(srf)
    if hasattr(iElt, '__len__'):
        if len(iElt) != 1:
            raise Exception("dx_at() takes a single srf, got "
                            f"{len(iElt)}")
        iElt = int(iElt[0])

    # Fortran side returns SI metres (dxElt * CBM); for 'native' we
    # divide back out by querying CBM separately.
    ok, dx_m = lib.api.elt_dx_get(int(iElt))
    if not ok:
        raise Exception(f"MACOS: dxElt({iElt}) unavailable -- check that "
                        "an Rx is loaded and a propagation has reached "
                        "this element")
    if dx_m == 0.0:
        raise Exception(f"MACOS: dxElt({iElt}) is zero -- this slot "
                        "hasn't been populated by a diffraction "
                        "propagation yet")

    if unit == 'native':
        ok, cbm = lib.api.base_unit_to_metres()
        if not ok or cbm == 0.0:
            raise Exception("MACOS: base-units conversion factor "
                            "unavailable; can't return native dx")
        return float(dx_m) / float(cbm)

    return float(dx_m) * _DX_AT_UNIT_FACTORS[unit]


def apodize(srf: int | np.int32, mask: np.ndarray) -> None:
    """Multiply macos's diffraction-grid complex field at ``srf`` by
    a user-supplied real-valued amplitude transmission map, in place.

    Companion to PROPER's ``prop_multiply(wfo, mask)``.  Pass the SAME
    NxN array to both engines and the apodisation is bit-identical:
    no parametric-reconstruction drift, no FITS round-tripping.

    macos must already have propagated to ``srf`` (e.g. via a prior
    ``intensity(srf)``, ``complex_field(srf)``, or ``trace_rays(srf)``
    call) so that ``WFElt(:,:, iEltToiWF(srf))`` is populated.
    Subsequent calls (``intensity``, ``complex_field``, downstream
    propagations) see the apodised wavefront.

    Caveat (important): this modifies ONLY the diffraction-grid
    wavefront, not the geometric ray channel.  Macos's prescription
    aperture stops (``Element=Reference`` with ``ApType=Circular``,
    etc.) do TWO things during the trace -- mask the WFElt AND clip
    the rays.  Geometric props between elements carry rays + per-ray
    OPD, and the next diffractive plane reconstructs WFElt from those
    rays.  An ``apodize`` call only handles the WFElt half; the rays
    at ``srf`` are not clipped to the mask support.

    Honest use:
      - Smooth apodisers (Gaussian taper, super-Gaussian, etc.).
      - Hard-edged masks applied immediately before a diffractive
        prop with no following geometric prop -- so the masked
        WFElt is the only thing that matters.

    For hard-edge aperture stops in a chain with intervening
    geometric props, use macos's prescription ``ApType=Circular``
    instead (which clips rays too).  See
    ``tests/proper_compare/README.md`` "pymacos.apodize limitation"
    for the full story.

    Args:
        srf:  Element id (1..nElt, or -k from end).
        mask: Real-valued (N, N) array.  N must equal macos's
              diffraction-grid size ``mdttl`` (= ``model_size`` arg
              to ``init()``).  Typical values are amplitude
              transmissions in [0, 1], but the wrapper does not
              clamp -- caller may pass values outside [0, 1] for
              e.g. inverted apodisers or sign-flipped masks.

    Raises:
        Exception: Rx not loaded, srf out of range, mask shape
                   mismatch, or no diffraction wavefront populated
                   at ``srf`` (``iEltToiWF(srf) <= 0``).
    """
    _chk_macos_and_rx_loaded()

    iElt = _map_Elt(srf)
    if hasattr(iElt, '__len__'):
        if len(iElt) != 1:
            raise Exception("apodize() takes a single srf, got "
                            f"{len(iElt)}")
        iElt = int(iElt[0])

    mask = np.asarray(mask, dtype=np.float64)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise Exception(f"apodize(): mask must be square 2D, got "
                        f"shape {mask.shape}")
    N = mask.shape[0]
    if N != _MODELSIZE:
        raise Exception(f"apodize(): mask shape ({N}, {N}) does not "
                        f"match macos's diffraction-grid size "
                        f"(mdttl = {_MODELSIZE}); call init({N}) and "
                        "reload the Rx, or resize the mask")

    ok = lib.api.cfield_apodize(mask, int(iElt))
    if not ok:
        raise Exception(
            f"MACOS: cfield_apodize failed at Elt {iElt} -- check "
            "that a propagation has populated WFElt at this element "
            "(call intensity(srf) or complex_field(srf) first)")


def apodize_complex(srf: int | np.int32, mask: np.ndarray) -> None:
    """Complex-valued sibling of :func:`apodize`.

    Multiplies macos's diffraction-grid complex field at ``srf`` by a
    user-supplied COMPLEX-valued NxN mask, in place.  Used for phase
    masks (vortex coronagraphs, PIAA-like apodisers, sub-wavelength
    gratings) where the mask carries both amplitude and phase.

    Same WFElt-only caveat as :func:`apodize` applies: ray-channel
    information at ``srf`` is not modified.  For apodisers immediately
    upstream of a DIFFRACTIVE propagation step (NFPlane / NF1 / NF2 /
    FarField), the effect propagates correctly to the next plane via
    WFElt; subsequent geometric propagations may dilute the effect
    when WFElt is reconstructed from rays.

    Args:
        srf:  Element id (1..nElt).
        mask: Complex (N, N) array.  Real/imag pair handed to the
              Fortran wrapper as separate float64 arrays.

    Raises:
        Exception: Rx not loaded, srf out of range, mask shape
                   mismatch, or no diffraction wavefront at ``srf``.
    """
    _chk_macos_and_rx_loaded()

    iElt = _map_Elt(srf)
    if hasattr(iElt, '__len__'):
        if len(iElt) != 1:
            raise Exception("apodize_complex() takes a single srf, got "
                            f"{len(iElt)}")
        iElt = int(iElt[0])

    mask = np.asarray(mask)
    if not np.iscomplexobj(mask):
        # Allow real-valued input but warn -- this is usually a sign
        # the caller meant apodize() instead.
        mask = mask.astype(np.complex128)

    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise Exception(f"apodize_complex(): mask must be square 2D, "
                        f"got shape {mask.shape}")
    N = mask.shape[0]
    if N != _MODELSIZE:
        raise Exception(f"apodize_complex(): mask shape ({N}, {N}) does not "
                        f"match macos's diffraction-grid size "
                        f"(mdttl = {_MODELSIZE}); call init({N}) and "
                        "reload the Rx, or resize the mask")

    mask_re = np.ascontiguousarray(mask.real, dtype=np.float64)
    mask_im = np.ascontiguousarray(mask.imag, dtype=np.float64)
    ok = lib.api.cfield_apodize_complex(mask_re, mask_im, int(iElt))
    if not ok:
        raise Exception(
            f"MACOS: cfield_apodize_complex failed at Elt {iElt} -- check "
            "that a propagation has populated WFElt at this element")


def sxp(mode: int = 1) -> Tuple[np.float64,
                                Vector[np.float64], Vector[np.float64]]:
    """SXP -- Set eXit Pupil.  FEX variant with one geometry fix:
    the EP radius is set to the chief-ray distance from the EP
    (element ``nElt-1``) to the FP (element ``nElt``), instead of
    FEX's legacy distance from ``iEm1=nElt-2`` to the EP.

    Makes the EP geometry sensitive to FP-Tz perturbations (focus),
    which the original FEX is not (because ``iEm1`` is upstream of
    the EP and doesn't move when the FP moves).  Lateral FP shifts
    (Tx/Ty) and FP rotations are still NOT captured by SXP -- they
    would need the EP vertex / orientation to also follow, not just
    the radius (use the dw/dx FocalPlaneChannel "track" mode for
    full coverage).

    See tracesub.F SUBROUTINE SXP for the algorithm; the dispatch
    path is identical to FEX (SMACOS('SXP', ...)) -- added in
    macos_ops.F.

    Args:
        mode: 1 (default) -> chief-ray-centered; 0 -> centroid-centered
              (same convention as :func:`fex`).

    Returns:
        (rad, psi, vpt) of the updated XP surface at ``nElt-1``.

    Raises:
        Exception: Rx not loaded, Stop not set, or fewer than 3
                   elements (so no XP slot at ``nElt-1``).
    """
    _chk_macos_and_rx_loaded()

    if lib.api.n_elt() <= 3:
        raise Exception("'sxp': not more than 3 surfaces defined")

    ok, xp = lib.api.sxp_fnd(np.int32(mode))
    if not ok:
        raise Exception("'sxp' threw an exception - stop set?")
    return xp


def ors(srf: int | np.int32) -> None:
    """ORS -- Optimize Reference Surface.

    Traces the chief ray from the source to ``srf-1`` and then runs
    macos's ``CRSOPTIMIZE`` to fit an optimal reference sphere at
    element ``srf`` against the current ray geometry.  Used in
    set-up to derive a clean EP-element pose from the unperturbed
    design (typically follows :func:`fex` and precedes
    :func:`srs`).

    NOT typically part of a sensitivity / dw/dx loop -- ORS would
    re-optimize away from the perturbation being measured.

    Args:
        srf: element id to optimize.  macos requires the element
             to be Reference / Return / Zernike-typed.

    Raises:
        Exception: Rx not loaded, srf out of range, or macos's ORS
                   rejected the request (wrong element type).
    """
    _chk_macos_and_rx_loaded()
    s = int(_map_Elt(srf, max_rows=1).squeeze())
    ok = lib.api.ors_run(s)
    if not ok:
        raise Exception(
            f"MACOS: ors(srf={s}) rejected -- valid element types are "
            "Reference / Return / Zernike-typed")


def srs(slave: int | np.int32,
        master: int | np.int32,
        link: bool = True) -> None:
    """SRS -- slave one optical surface to another.

    Wraps macos's interactive SRS (Slave Reference Surface) command.
    Recomputes ``slave``'s pose from the chief ray traced through
    ``master``, so the slave tracks the master across subsequent
    traces.

    Typical setup pattern (from an unperturbed design):
        FEX     - find the pupil
        ORS  N  - set element N as the exit pupil
        srs(FP, EP)  - lock the focal plane behind the EP

    For the dw/dx FocalPlaneChannel "srs" mode, the channel calls
    ``srs(EP_elt, FP_elt, link=True)`` after each FP perturbation,
    recomputing the EP's pose against the new chief-ray geometry
    from the moved FP.

    Args:
        slave:  element id whose pose is computed from the chief ray.
        master: element id the chief ray is traced through.
                Must differ from ``slave``.
        link:   if True (default), establish a persistent SRS link
                that re-evaluates on subsequent traces.  If False,
                the slave's pose is set once but not maintained.

    Raises:
        Exception: Rx not loaded, either id out of range, slave ==
                   master, or macos's SRS rejected the request.
    """
    _chk_macos_and_rx_loaded()
    s = int(_map_Elt(slave, max_rows=1).squeeze())
    M = int(_map_Elt(master, max_rows=1).squeeze())
    if s == M:
        raise Exception(f"srs(): slave and master must differ (both = {s})")
    ok = lib.api.srs_run(s, M, bool(link))
    if not ok:
        raise Exception(
            f"MACOS: srs(slave={s}, master={M}) rejected -- check that "
            "both elements are valid types for SRS (Reference / Return / "
            "Zernike-typed)")


# --- Optimization-target sentinels (mirror dopt_mod's *_TARGET) -----
CALIB_TARGET_WFE       = 1
CALIB_TARGET_WFE_ZMODE = 2
CALIB_TARGET_BEAM      = 3
CALIB_TARGET_SPOT      = 4
CALIB_TARGET_OPL       = 5

_CALIB_TARGET_NAMES = {
    'WFE':       CALIB_TARGET_WFE,
    'WFE_ZMODE': CALIB_TARGET_WFE_ZMODE,
    'ZWF':       CALIB_TARGET_WFE_ZMODE,
    'BEAM':      CALIB_TARGET_BEAM,
    'SPOT':      CALIB_TARGET_SPOT,
    'OPL':       CALIB_TARGET_OPL,
}

# Position order matches dopt_mod's DOF_NameList:
#   [TIP, TILT, CLOCK, DX, DY, PIST, ROC, CONIC]
_CALIB_DOF_NAMES = ('TIP', 'TILT', 'CLOCK', 'DX', 'DY', 'PIST', 'ROC', 'CONIC')


def _calib_dof_mask(dofs) -> np.ndarray:
    """Normalize a ``dofs`` argument into the 8-int mask CALIB expects.

    Accepts:
      * iterable of 8 ints: nonzero = vary, zero = freeze (positional).
      * iterable of strings: names from {TIP, TILT, CLOCK, DX, DY,
        PIST, ROC, CONIC}; case-insensitive; everything not listed
        is frozen.
    """
    mask = np.zeros(8, dtype=np.int32)
    items = list(dofs)
    if not items:
        return mask
    if all(isinstance(x, str) for x in items):
        name_to_idx = {n: i for i, n in enumerate(_CALIB_DOF_NAMES)}
        for s in items:
            key = s.strip().upper()
            if key not in name_to_idx:
                raise ValueError(
                    f"calib_set_var_elt: unknown DOF name {s!r}; "
                    f"valid: {_CALIB_DOF_NAMES}")
            mask[name_to_idx[key]] = 1
    else:
        if len(items) != 8:
            raise ValueError(
                f"calib_set_var_elt: positional dofs must be length 8 "
                f"(got {len(items)}); use a name list or pad with zeros")
        mask[:] = [int(bool(v)) for v in items]
    return mask


def calib_clear_var_elts() -> None:
    """Wipe all CALIB variable-element state (DOFs + Zernike modes).

    Use before defining a fresh variable-element list; otherwise
    subsequent :func:`calib_set_var_elt` calls accumulate on top of
    whatever the prescription's ``VarDOF=`` keywords already set.
    """
    _chk_macos_and_rx_loaded()
    if not lib.api.calib_clear_var_elts():
        raise Exception("MACOS: calib_clear_var_elts() failed")


def calib_set_var_elt(srf: int | np.int32,
                      dofs,
                      zern_modes=None) -> None:
    """Mark element ``srf`` as a CALIB variable.

    Programmatic equivalent of the interactive AVAR command.  If
    ``srf`` was already a variable element, this call REPLACES its
    DOF + Zernike configuration (MVAR semantics).

    Args:
        srf:        element id (1..nElt).
        dofs:       iterable selecting free DOFs.  Either:
                      - 8-element positional mask [TIP, TILT, CLOCK,
                        DX, DY, PIST, ROC, CONIC]; nonzero = vary, or
                      - list of name strings from the set above
                        (case-insensitive).
        zern_modes: optional iterable of Zernike mode indices (1..45)
                    to also optimize on this element; None / empty =
                    rigid-body-only.

    Raises:
        Exception: Rx not loaded, srf out of range, mode out of
                   range, or macos rejected the configuration.

    Example:
        >>> m.calib_set_var_elt(7, dofs=['TIP', 'TILT'])
        >>> m.calib_set_var_elt(3, dofs=['DX', 'DY', 'PIST'],
        ...                    zern_modes=[4, 5, 11])
    """
    _chk_macos_and_rx_loaded()
    s = int(_map_Elt(srf, max_rows=1).squeeze())
    mask = _calib_dof_mask(dofs)
    if zern_modes is None:
        modes = np.array([], dtype=np.int32)
    else:
        modes = np.asarray(zern_modes, dtype=np.int32).ravel()
    ok = lib.api.calib_set_var_elt(s, mask, modes)
    if not ok:
        raise Exception(
            f"MACOS: calib_set_var_elt(srf={s}) rejected -- check that "
            "srf is in range 1..nElt and any zern_modes are in 1..45")


def calib_set_iter(n_iter: int) -> None:
    """Set the CALIB optimizer iteration cap (``nitrs_dopt``)."""
    n = int(n_iter)
    if n < 1:
        raise ValueError(f"calib_set_iter: n_iter must be >= 1 (got {n})")
    if not lib.api.calib_set_iter(n):
        raise Exception(f"MACOS: calib_set_iter({n}) failed")


def calib_set_tol(tol: float) -> None:
    """Set the CALIB convergence tolerance (``dopt_tol``)."""
    t = float(tol)
    if t <= 0.0:
        raise ValueError(f"calib_set_tol: tol must be > 0 (got {t})")
    if not lib.api.calib_set_tol(t):
        raise Exception(f"MACOS: calib_set_tol({t}) failed")


def calib_set_target(target, wf_zern_modes=None) -> None:
    """Set the CALIB optimization target.

    Args:
        target: either an integer (1..5 mapping to the dopt_mod
                *_TARGET constants) OR a name string from
                {'WFE', 'WFE_ZMODE', 'ZWF', 'BEAM', 'SPOT', 'OPL'}
                (case-insensitive).
        wf_zern_modes: required for the WFE_ZMODE target -- list of
                Zernike mode indices (1..45) the optimizer should
                drive to zero.  Ignored for other targets.

    Raises:
        ValueError: unknown target name.
        Exception:  macos rejected the configuration.

    Example:
        >>> m.calib_set_target('WFE')
        >>> m.calib_set_target('WFE_ZMODE', wf_zern_modes=[4, 5, 11])
    """
    if isinstance(target, str):
        key = target.strip().upper()
        if key not in _CALIB_TARGET_NAMES:
            raise ValueError(
                f"calib_set_target: unknown target name {target!r}; "
                f"valid: {sorted(_CALIB_TARGET_NAMES)}")
        t = _CALIB_TARGET_NAMES[key]
    else:
        t = int(target)
    if wf_zern_modes is None:
        modes = np.array([], dtype=np.int32)
    else:
        modes = np.asarray(wf_zern_modes, dtype=np.int32).ravel()
    if not lib.api.calib_set_target(t, modes):
        raise Exception(
            f"MACOS: calib_set_target(target={target!r}) failed -- check "
            "wf_zern_modes is set for WFE_ZMODE target")


def calib() -> dict:
    """CALIB -- run the macos design optimizer.

    Wraps the SMACOS ``CALIB`` command.  Reads the optimization
    configuration from current state -- variable elements (set via the
    prescription's ``VarDOF=`` keyword or future programmatic setters),
    field-of-view list, wavelength list, target (``OptTarget=``),
    iteration cap (``OptMxItrs=``), tolerance, and the FEX
    pre-optimization flag (``OptFEX=``).  The simplest workflow is to
    bake the config into the .in file and call :func:`calib` after
    loading::

        m.load('opt_example.in')
        m.perturb(1, rotation_rad=(1e-3, 0, 0), in_local_coords=False)
        m.stop(7, [0.0, 0.0])
        result = m.calib()
        if result['converged']:
            print(f"old WFE = {result['old_wfe']}")
            print(f"new WFE = {result['new_wfe']}")

    The constrained (NPSOL) optimizer is selected automatically when
    the prescription enables it AND macos was built with
    ``-DUSE_NPSOL=ON``; otherwise CALIB falls back to an unconstrained
    Levenberg-Marquardt step.

    Returns:
        dict with keys:

        ``converged`` (bool):
            True if rtn_flag == 0 (optimizer ran to completion).
        ``rtn_flag`` (int):
            Optimizer return code; 0 = converged, nonzero = failure.
        ``n_fov`` (int):
            Number of field-of-view points used.
        ``n_wavelength`` (int):
            Number of wavelengths used.
        ``old_wfe`` (np.ndarray, shape=(n_fov, n_wavelength)):
            RMS wavefront error per (FOV, wavelength) BEFORE the
            optimization.  Only populated for ``OptTarget=WFE``;
            BEAM / SPOT / OPL / ZWF targets leave this zero (their
            equivalent metrics aren't yet exposed -- Phase 1b).
        ``new_wfe`` (np.ndarray, shape=(n_fov, n_wavelength)):
            Same, AFTER the optimization.

    Raises:
        Exception: Rx not loaded, no variable elements defined, no
                   FOVs defined, or the optimizer rejected the run.
    """
    _chk_macos_and_rx_loaded()
    maxFov, maxWl = lib.api.calib_buffer_dims()
    old_wfe = np.zeros((maxFov, maxWl), order='F', dtype=np.float64)
    new_wfe = np.zeros((maxFov, maxWl), order='F', dtype=np.float64)
    ok, rtn_flag, n_fov, n_wl = lib.api.calib_run(old_wfe, new_wfe)
    if not ok:
        raise Exception(
            f"MACOS: calib() failed -- rtn_flag={rtn_flag}.  Common "
            "causes: no variable elements defined (set VarDOF= on at "
            "least one element in the prescription), no FOVs defined "
            "(set OptFOVWt=), or CALIB's iteration cap was exhausted.")
    return {
        'converged':     rtn_flag == 0,
        'rtn_flag':      int(rtn_flag),
        'n_fov':         int(n_fov),
        'n_wavelength':  int(n_wl),
        'old_wfe':       old_wfe[:n_fov, :n_wl].copy(),
        'new_wfe':       new_wfe[:n_fov, :n_wl].copy(),
    }


def perturb(srf: int | np.int32,
            rotation_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
            translation_m: Tuple[float, float, float] = (0.0, 0.0, 0.0),
            in_local_coords: bool = True) -> None:
    """Apply a rigid-body coordinate-frame perturbation to element ``srf``.

    Wraps macos's ``CPERTURB_PROG`` (the programmatic sibling of the
    interactive ``CPERTURB``).  Unlike a raw ``elt_vpt`` setter, this
    performs the full PERTURB bookkeeping: position + orientation,
    the element's local TElt frame matrix, aperture vector xObs, HOE
    points, the Mon / pData / FF figure-error coordinate frames,
    metrology surface points, and any linked-element children.

    For an optic that has a Monomial / FreeForm figure error riding
    on its surface, ``elt_vpt`` alone would shift the vertex but
    leave the figure error stuck at the original location -- a
    silent bug.  ``perturb`` moves everything together, as it would
    physically.

    Args:
        srf:            Element id (1..nElt).
        rotation_rad:   (x, y, z) rotation perturbation vector
                        in radians.
        translation_m:  (x, y, z) translation in SI metres.  Converted
                        to the prescription's BaseUnits internally
                        via the CBM factor.
        in_local_coords: True (default) -> the rotation + translation
                        are expressed in the element's LOCAL coordinate
                        frame; False -> already in GLOBAL coords.

    Raises:
        Exception: Rx not loaded, srf out of range, or macos's
                   CPERTURB_PROG signalled failure.
    """
    _chk_macos_and_rx_loaded()

    iElt = _map_Elt(srf)
    if hasattr(iElt, '__len__'):
        if len(iElt) != 1:
            raise Exception(f"perturb() takes a single srf, got {len(iElt)}")
        iElt = int(iElt[0])

    # SI metres -> BaseUnits via CBM.
    ok_cbm, cbm = lib.api.base_unit_to_metres()
    if not ok_cbm or cbm == 0.0:
        raise Exception("MACOS: base-units conversion factor unavailable "
                        "(perturb needs CBM to convert SI metres "
                        "translation to BaseUnits)")
    si_to_base = 1.0 / float(cbm)

    th  = np.asarray(rotation_rad, dtype=np.float64).reshape(3)
    del_ = np.asarray(translation_m, dtype=np.float64).reshape(3) * si_to_base

    ok = lib.api.perturb_elt(int(iElt), th, del_, bool(in_local_coords))
    if not ok:
        raise Exception(
            f"MACOS: perturb failed at Elt {iElt} -- check that the "
            "Rx is loaded and the element index is valid")


def perturb_src(rotation_rad: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                translation_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
                ) -> None:
    """Apply a rigid-body perturbation to the SOURCE (iElt=0 in macos).

    Routes through macos's ``CPERTURB`` source-perturb branch
    (funcsub.F:41-92) via ``SMACOS('PERTURB', IARG(1)=0, DARG=6vec)``.
    macos updates ``ChfRayDir``, ``ChfRayPos``, ``xGrid``, ``yGrid``,
    and ``SegXGrid`` if the prescription uses a segmented xGrid.

    Coordinate frame:
        - GLOBAL by default.
        - LOCAL when the prescription declares source local frames
          (``SrcXAxis=`` / ``SrcYAxis=`` keywords -> ``SrcLF_FLG``
          set inside macos).  There is no per-call switch because
          macos itself has none -- the frame choice is baked into
          the Rx.

    For a rotation-only perturbation (point source), the chief ray
    is REDIRECTED (its direction rotates).  For a translation, the
    source position shifts; chief ray direction stays the same.
    Note that this changes where the chief ray hits the system Stop
    -- callers that want to KEEP the chief ray passing through the
    Stop after a source motion should re-call :func:`stop` (for
    element-based stops) or otherwise re-enforce the chief-ray
    aiming after this call.

    Args:
        rotation_rad:    (x, y, z) source rotation in radians.
        translation_m:   (x, y, z) source translation in SI metres
                         (converted to prescription BaseUnits via
                         CBM, matching :func:`perturb`).

    Raises:
        Exception: Rx not loaded, or macos signalled failure.
    """
    _chk_macos_and_rx_loaded()

    ok_cbm, cbm = lib.api.base_unit_to_metres()
    if not ok_cbm or cbm == 0.0:
        raise Exception("MACOS: base-units conversion factor unavailable "
                        "(perturb_src needs CBM to convert SI metres "
                        "translation to BaseUnits)")
    si_to_base = 1.0 / float(cbm)

    th  = np.asarray(rotation_rad, dtype=np.float64).reshape(3)
    del_ = np.asarray(translation_m, dtype=np.float64).reshape(3) * si_to_base

    ok = lib.api.perturb_src(th, del_)
    if not ok:
        raise Exception(
            "MACOS: perturb_src failed -- check that the Rx is loaded")


def spot(srf: int | Tuple[int] | np.int32,
         vpt_center: bool | int = True,
         beam_csys: int = 1,
         reset_trace: bool = True
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """SPOT: Retrieve Ray-Surface Intersections points at Surface.

    Args:
        srf (int | Tuple[int] | np.ndarray[int] | np.int32):
           Element ID, Scalar (Range: -nElt < srf <= nElt)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        vpt_center (bool):
            If (True or 1) Element Center (default) is ref. pos.;
            otherwise, (False, 0) Chief-Ray Pos.

        beam_csys (int):
            defining the Coord. Sys. for Spot where
              =1: Beam Coordinate Frame (default);
              =2: TOUT (relevant for the last surface only)
              =3: TELT (Element Coord. Frame, if not def. Glb used)

        reset_trace (bool):
            If (True or 1), the ray-trace is restarted from the source
            (default); otherwise, (False, 0), it is continued from the
            last ray-trace state.

    Raises:
        Exception: MACOS Triggered error

    Returns:
        pts (Matrix[np.float64]):
            Ray-Surface Intersection Pts. expressed in defined local coordinate
            system where 'Shift' is already subtracted where pts(iray,[x,y]).

        centroid (Vector[np.float64]):
            Centroid Location (x,y) of all extracted Spot pts assuming
            uniform intensity, i.e., pts.mean(axis=0)

        shift (Vector[np.float64]):
            shift = nd.array([dx,dy]) where [dx,dy] is shift from
             (a) Element Vertex Position in local CSYS (if vpt_center=1)
             (b) Chief-Ray Intersection point in local CSYS (if vpt_center=0)

        csys (Matrix[np.float64]):
            csys = [x_axis, y_axis, z_axis] a 3x3 matrix where
              x_axis = csys[:, 0], y_axis = csys[:, 1], z_axis = csys[:, 2]
    """
    _chk_macos_and_rx_loaded()

    # spot_cmd signature is (iElt, ref_csys, ref_pos, res_trace): ref_csys
    # is the coord frame (1=BEAM/2=TOUT/3=TELT), ref_pos is the centering
    # logical (1=ELT vertex / 0=chief ray).  beam_csys -> ref_csys and
    # vpt_center -> ref_pos (previously swapped, which made vpt_center=False
    # send ref_csys=0 and spot_cmd reject every call -- no spot test caught it).
    ok, npts = lib.api.spot_cmd(_map_Elt(srf).squeeze(),
                                np.int32(beam_csys),
                                np.int32(vpt_center),
                                np.int32(reset_trace))
    if not ok:
        raise Exception('MACOS threw an exception')

    ok, pts, shift, centre, csys = lib.api.spot_get(npts)
    if not ok:
        raise Exception('MACOS threw an exception')

    return pts, centre, shift[2:] if vpt_center else shift[:2], csys




def window(frame: str,
           siz_pix: float,
           elt_pix: Tuple[float, float] = (0.0, 0.0),
           win_cen: Tuple[float, float] = (0.0, 0.0)) -> None:
    """WINDOW -- place diffraction images at their TRUE sky offset.

    Turns on the pixel-location option so PIX / COMPOSE place each
    source's image at its real chief-ray offset on the grid instead of
    re-centred -- required to COMPOSE an off-axis source (e.g. a planet)
    at its true position relative to an on-axis star.  Needs a prior trace.

    Args:
        frame: output coordinate frame the placement references --
               'tout' (prescription output frame) or 'beam' (local beam
               frame at the output element).
        siz_pix: window pixel size in BaseUnits (match the COMPOSE pitch).
        elt_pix: (x, y) element reference pixel (default (0, 0)).
        win_cen: (x, y) window centre pixel (default (0, 0)).

    Raises:
        Exception: Rx not loaded, bad frame, or MACOS rejected the call.
    """
    _chk_macos_and_rx_loaded()
    codes = {'tout': 1, 'beam': 2}
    if frame not in codes:
        raise Exception(f"window: frame must be 'tout' or 'beam', got {frame!r}")
    ok = lib.api.window_set(np.int32(codes[frame]),
                            np.float64(siz_pix),
                            np.float64(elt_pix[0]), np.float64(elt_pix[1]),
                            np.float64(win_cen[0]), np.float64(win_cen[1]))
    if not ok:
        raise Exception('MACOS: window_set failed')


def window_off() -> None:
    """Turn the WINDOW pixel-location option back off (re-centre images)."""
    _chk_macos_and_rx_loaded()
    if not lib.api.window_off():
        raise Exception('MACOS: window_off failed')


_BEAM_CODES = {'uniform': 1, 'gaussian': 2, 'cos': 3, 'dipole': 4}
_BEAM_NAMES = {v: k for k, v in _BEAM_CODES.items()}


def beam(kind: str | None = None,
         waist: float | Tuple[float, float] | None = None,
         radius: float | None = None,
         power: float | None = None) -> dict | None:
    """BEAM -- shape the source amplitude (apodization) profile.

    Sets the source beam profile macos applies to the aperture amplitude
    before tracing.  Setting the beam resets the trace, so re-trace after.

    Args:
        kind: 'uniform' (flat-top, no params), 'gaussian' (needs ``waist``),
              'cos' (cosine**power, needs ``radius`` + ``power``), or
              'dipole' (no params).  If None (default), QUERY the current
              profile instead of setting it.
        waist: GAUSSIAN x/y waist radii in source BaseUnits -- a scalar is
               broadcast to (rx, ry).  Required for 'gaussian'.
        radius: COS cosine beam radius in source BaseUnits.  Required for 'cos'.
        power: COS cosine exponent.  Required for 'cos'.

    Returns:
        None when setting.  When called with no args (query), a dict:
        ``{'kind': str, 'waist': (rx, ry), 'power': float}``.

    Raises:
        Exception: Rx not loaded, bad kind / missing params, or MACOS
            rejected the call.
    """
    _chk_macos_and_rx_loaded()

    # query mode -----------------------------------------------------
    if kind is None:
        ok, code, rx, ry, cos_pwr = lib.api.beam_get()
        if not ok:
            raise Exception('MACOS: beam_get failed')
        code = int(round(code))
        return {'kind': _BEAM_NAMES.get(code, 'unset'),
                'waist': (float(rx), float(ry)),
                'power': float(cos_pwr)}

    # set mode -------------------------------------------------------
    kind = kind.lower()
    if kind not in _BEAM_CODES:
        raise Exception(f'MACOS: unknown beam kind {kind!r} '
                        f'(expected one of {list(_BEAM_CODES)})')
    p1, p2 = 0.0, 0.0
    if kind == 'gaussian':
        if waist is None:
            raise Exception("MACOS: gaussian beam requires 'waist'")
        w = (waist, waist) if np.isscalar(waist) else tuple(waist)
        if len(w) != 2:
            raise Exception("MACOS: 'waist' must be a scalar or (rx, ry)")
        p1, p2 = float(w[0]), float(w[1])
    elif kind == 'cos':
        if radius is None or power is None:
            raise Exception("MACOS: cos beam requires 'radius' and 'power'")
        p1, p2 = float(radius), float(power)

    if not lib.api.beam_set(_BEAM_CODES[kind], p1, p2):
        raise Exception('MACOS: beam_set failed')
    return None


# --- Polarization (PLAN_POLARIZATION Phase 1) --------------------------
def _single_elt(srf) -> int:
    """Map a user srf to a single 1-based element ID (raise if multiple)."""
    iElt = _map_Elt(srf)
    if hasattr(iElt, '__len__'):
        if len(iElt) != 1:
            raise Exception(f'expected a single element, got {len(iElt)}')
        iElt = iElt[0]
    return int(iElt)


def polarization(state: str | None = None,
                 Ex: complex | Tuple[float, float] = (1.0, 0.0),
                 Ey: complex | Tuple[float, float] = (0.0, 0.0)) -> dict | None:
    """POLARIZATION -- enable/disable polarized ray tracing + source state.

    With ``state='on'`` enables polarized tracing (rays carry a complex
    3-vector E-field, surface coatings become active, and vector
    diffraction turns on when the model supports it, mWF>=3).  With
    ``state='off'`` restores scalar tracing.  With ``state=None`` (default)
    QUERIES the current state.

    Args:
        state: 'on', 'off', or None (query).
        Ex, Ey: source Jones amplitudes, complex or (re, im).  Used only
            when ``state='on'``.  Defaults to x-polarized (Ex=1, Ey=0).

    Returns:
        None when setting.  When querying, a dict:
        ``{'on': bool, 'vector': bool, 'Ex': complex, 'Ey': complex}``.

    Raises:
        Exception: Rx not loaded, bad state, mWF<3 when enabling, or the
            engine rejected the call.
    """
    _chk_macos_and_rx_loaded()

    if state is None:
        ok, on, vec, exre, exim, eyre, eyim = lib.api.pol_get()
        if not ok:
            raise Exception('MACOS: pol_get failed')
        return {'on': bool(on), 'vector': bool(vec),
                'Ex': complex(exre, exim), 'Ey': complex(eyre, eyim)}

    s = state.lower()
    if s == 'off':
        if not lib.api.pol_set(False, 0.0, 0.0, 0.0, 0.0):
            raise Exception('MACOS: pol_set (off) failed')
        return None
    if s != 'on':
        raise Exception(f"MACOS: polarization state must be 'on'/'off'/None, "
                        f"got {state!r}")

    exc = Ex if isinstance(Ex, complex) else complex(*np.atleast_1d(Ex)[:2]) \
        if hasattr(Ex, '__len__') else complex(Ex)
    eyc = Ey if isinstance(Ey, complex) else complex(*np.atleast_1d(Ey)[:2]) \
        if hasattr(Ey, '__len__') else complex(Ey)
    if not lib.api.pol_set(True, exc.real, exc.imag, eyc.real, eyc.imag):
        raise Exception('MACOS: pol_set (on) failed -- model may have mWF<3')
    return None


def vector_diffraction(on: bool) -> None:
    """VECTOR / SCALAR -- toggle 3-component (vector) diffraction.

    ``on=True`` propagates Ex/Ey/Ez as three independent fields.  Since
    PLAN_POLARIZATION Phase 3a Tranche 1 this covers the WHOLE chain --
    every near-field, plane-to-plane, spherical, Fresnel and DFT leg, plus
    ``FFObscure`` and the ray-side aperture masking -- not just the
    far-field FFT leg.  Intensity / complex-field readouts sum the three
    components.  ``on=False`` restores single-field scalar diffraction.

    VECTOR requires polarization ON already (:func:`polarization`) and a
    model with mWF>=3, else raises (the engine would otherwise silently
    revert to scalar).

    Constraint: vector mode repurposes the model's mWF=3 wavefront planes
    as Ex/Ey/Ez of ONE wavefront, so only a single wavefront can be in
    flight -- do not combine it with multi-WF / COMPOSE work.

    Chains with COATED or reflective surfaces BETWEEN physical propagation
    legs still need Tranche 2 (per-ray running Jones); Tranche 1 is exact
    when the elements between legs are non-polarizing (Obscuring /
    Reference / FocalPlane) -- the coronagraph pupil->FPM->Lyot->focal case.
    """
    _chk_macos_and_rx_loaded()
    if not lib.api.vecdif_set(bool(on)):
        raise Exception('MACOS: vecdif_set failed -- turn polarization on '
                        'first (and need mWF>=3)')


_MCOAT = 10   # mCoat in elt_mod.F


def coating(srf: int | np.int32,
            index: ArrayLike | None = None,
            extinc: ArrayLike | None = None,
            thickness: ArrayLike | None = None) -> dict | None:
    """Set or query the thin-film coating stack on an element (Model A).

    The polarization-path coating (active only under
    ``polarization('on')``).  Layers are ordered OUTERMOST -> INNERMOST.

    Args:
        srf: element ID.
        index, extinc, thickness: equal-length per-layer vectors.  ``index``
            is the real refractive index n, ``extinc`` the extinction kappa
            (index = n - i*kappa), ``thickness`` the PHYSICAL layer thickness
            in element BaseUnits (NOT waves).  Omit all three to QUERY.

    Returns:
        None when setting.  When querying, a dict:
        ``{'n_layer': int, 'index': ndarray, 'extinc': ndarray,
        'thickness': ndarray}`` (physical thickness).

    Raises:
        Exception: Rx not loaded, bad layer count/length, or engine error.
    """
    _chk_macos_and_rx_loaded()
    iElt = _single_elt(srf)

    if index is None and extinc is None and thickness is None:
        ok, n_layer, idx, ext, thk = lib.api.coat_get(iElt, _MCOAT)
        if not ok:
            raise Exception(f'MACOS: coat_get failed at Elt {iElt}')
        n = int(n_layer)
        return {'n_layer': n,
                'index': np.asarray(idx[:n], dtype=float),
                'extinc': np.asarray(ext[:n], dtype=float),
                'thickness': np.asarray(thk[:n], dtype=float)}

    n_arr = np.atleast_1d(np.asarray(index, dtype=float))
    k_arr = np.atleast_1d(np.asarray(extinc, dtype=float))
    t_arr = np.atleast_1d(np.asarray(thickness, dtype=float))
    L = n_arr.size
    if L < 1 or L > _MCOAT:
        raise Exception(f'MACOS: coating needs 1..{_MCOAT} layers (got {L})')
    if k_arr.size != L or t_arr.size != L:
        raise Exception("MACOS: 'index','extinc','thickness' must match length")
    # f2py derives nlayer from the array shapes (optional trailing arg).
    if not lib.api.coat_set(iElt, n_arr, k_arr, t_arr):
        raise Exception(f'MACOS: coat_set failed at Elt {iElt}')
    return None


def ray_field(srf: int | np.int32) -> dict:
    """Per-ray complex E-field + geometry + status at an element.

    Harvests, on the N x N ray grid at element ``srf`` (N = model size),
    the per-ray complex field RayE(3,:), the ray direction cosines, the
    element surface normal, and the per-ray status.  Requires a polarized
    trace: call :func:`polarization` ('on') and trace first.

    Returns:
        dict of ndarrays: ``E`` (complex, shape (N, N, 3) = Ex/Ey/Ez),
        ``k`` (float, (N, N, 3) direction cosines), ``n`` (float, (N, N, 3)
        surface normal), and ``status`` (int, (N, N); 0=OK..5=Undef).

    Raises:
        Exception: Rx not loaded or engine error.
    """
    _chk_macos_and_rx_loaded()
    iElt = _single_elt(srf)
    N = model_size()
    (ok, exre, exim, eyre, eyim, ezre, ezim,
     kx, ky, kz, nx, ny, nz, status) = lib.api.rayfield_get(N, iElt)
    if not ok:
        raise Exception(f'MACOS: rayfield_get failed at Elt {iElt}')
    E = np.stack([exre + 1j * exim, eyre + 1j * eyim, ezre + 1j * ezim],
                 axis=-1)
    k = np.stack([kx, ky, kz], axis=-1)
    n = np.stack([nx, ny, nz], axis=-1)
    return {'E': E, 'k': k, 'n': n, 'status': status.astype(int)}


def jones_pupil(srf: int | np.int32, basis: str = 'double-pole',
                axis: ArrayLike | None = None,
                xref: ArrayLike | None = None) -> dict:
    """2x2 Jones pupil at an element from two orthogonal polarized traces.

    Traces the loaded prescription twice with source states (Ex0,Ey0) =
    (1,0) and (0,1), harvests the per-ray vector field at ``srf``
    (:func:`ray_field`), and assembles the 2x2 complex Jones matrix per
    ray-grid point: [E_out in exit basis] = J [source state].

    INPUT basis (fixed by the engine): collimated sources launch every
    ray with E = S*(Ex0*xGrid + Ey0*yGrid) -- the source-frame pair,
    uniform over the grid; point sources use the per-ray frame
    yray = unit(RayDir x xGrid), xray = yray x RayDir (ssrcray.inc).
    S is the engine flux normalization (~1/sqrt(nRays)); J carries this
    common real scalar (ratios/metrics unaffected).

    Args:
        srf: element at which to harvest.
        basis: output transverse basis --
            'double-pole' (default): Chipman double-pole coordinates,
            smooth over any physical pupil; use for budget numbers.
            'local-sp': per-ray s/p relative to the exit axis;
            coordinate-singular on axis -- diagnostic only.
            'global': project onto (xref, yref) ignoring ray direction.
        axis: exit-axis 3-vector (default: mean unit ray direction).
        xref: reference x direction (default: global +x projected out of
            the axis; +y fallback).

    Returns:
        dict: ``J`` (N,N,2,2) complex, NaN at vignetted points; ``mask``
        (N,N) bool; ``basis``, ``axis``, ``xref``, ``yref``; ``k``
        (N,N,3) exit unit ray directions; ``leak`` max longitudinal
        residual |E.k|/|E|; ``singular`` (N,N) bool where the requested
        basis is coordinate-singular (local-sp only, else all-False).

    The two traces are geometry-identical by construction (asserted
    bitwise).  The pre-call polarization state is restored on exit.
    """
    if basis not in ('double-pole', 'local-sp', 'global'):
        raise Exception(f"MACOS: jones_pupil basis must be 'double-pole', "
                        f"'local-sp' or 'global', got {basis!r}")
    s0 = polarization()

    polarization('on', Ex=1 + 0j, Ey=0 + 0j)
    trace_rays(srf)
    rfx = ray_field(srf)
    polarization('on', Ex=0 + 0j, Ey=1 + 0j)
    trace_rays(srf)
    rfy = ray_field(srf)

    if s0['on']:
        polarization('on', Ex=s0['Ex'], Ey=s0['Ey'])
    else:
        polarization('off')

    if (not np.array_equal(rfx['status'], rfy['status'])
            or not np.array_equal(rfx['k'], rfy['k'])):
        raise Exception('MACOS: jones_pupil -- x/y-state traces disagree in '
                        'ray geometry (engine bug?)')
    mask = (rfx['status'] == 0) & (rfy['status'] == 0)
    if not mask.any():
        raise Exception(f'MACOS: jones_pupil -- no unvignetted rays at {srf}')
    N = mask.shape[0]

    k = rfx['k'].copy()
    kmag = np.linalg.norm(k, axis=-1)
    kmag[~mask] = 1.0
    k /= kmag[..., None]

    if axis is not None:
        ax = np.asarray(axis, dtype=float)
        ax = ax / np.linalg.norm(ax)
    else:
        ax = k[mask].mean(axis=0)
        ax = ax / np.linalg.norm(ax)

    if xref is not None:
        xr = np.asarray(xref, dtype=float)
    else:
        xr = np.array([1.0, 0.0, 0.0])
        if abs(xr @ ax) > 1 - 1e-8:
            xr = np.array([0.0, 1.0, 0.0])
    xr = xr - (xr @ ax) * ax
    xr = xr / np.linalg.norm(xr)
    yr = np.cross(ax, xr)                      # right-handed: xr x yr = ax

    singular = np.zeros((N, N), dtype=bool)
    if basis == 'double-pole':
        # Rodrigues rotation carrying ax -> khat, applied to (xr, yr)
        cth = k @ ax
        r = np.cross(np.broadcast_to(ax, k.shape), k)
        sth = np.linalg.norm(r, axis=-1)
        onax = sth < 1e-12
        u = r / np.where(sth[..., None] == 0, 1, sth[..., None])

        def _rotc(v0):
            cxv = np.stack([u[..., 1] * v0[2] - u[..., 2] * v0[1],
                            u[..., 2] * v0[0] - u[..., 0] * v0[2],
                            u[..., 0] * v0[1] - u[..., 1] * v0[0]], axis=-1)
            ud = u @ v0
            v = (v0[None, None, :] * cth[..., None] + cxv * sth[..., None]
                 + u * (ud * (1 - cth))[..., None])
            v[onax] = v0
            return v
        e1 = _rotc(xr)
        e2 = _rotc(yr)
    elif basis == 'local-sp':
        s_ = np.cross(k, np.broadcast_to(ax, k.shape))
        smag = np.linalg.norm(s_, axis=-1)
        singular = mask & (smag < 1e-9)
        s_ /= np.where(smag[..., None] < 1e-9, 1, smag[..., None])
        e1 = np.cross(s_, k)                   # p (meridional); e1 x e2 = k
        e2 = s_
        e1[singular] = xr
        e2[singular] = yr
    else:                                      # 'global'
        e1 = np.broadcast_to(xr, k.shape).copy()
        e2 = np.broadcast_to(yr, k.shape).copy()

    J = np.full((N, N, 2, 2), np.nan + 1j * np.nan, dtype=complex)
    J[..., 0, 0] = (e1 * rfx['E']).sum(axis=-1)
    J[..., 1, 0] = (e2 * rfx['E']).sum(axis=-1)
    J[..., 0, 1] = (e1 * rfy['E']).sum(axis=-1)
    J[..., 1, 1] = (e2 * rfy['E']).sum(axis=-1)
    J[~mask] = np.nan + 1j * np.nan

    leak = 0.0
    for rf in (rfx, rfy):
        Edotk = (rf['E'] * k).sum(axis=-1)
        Emag = np.linalg.norm(rf['E'], axis=-1)
        lk = np.abs(Edotk) / np.where(Emag == 0, 1, Emag)
        leak = max(leak, float(lk[mask].max()))

    return {'J': J, 'mask': mask, 'basis': basis, 'axis': ax,
            'xref': xr, 'yref': yr, 'k': k, 'leak': leak,
            'singular': singular}


def pol_maps(jp: dict) -> dict:
    """Polarization-aberration decomposition of a Jones pupil.

    Decomposes ``jp`` (from :func:`jones_pupil`, or any dict with ``J``
    (N,N,2,2) complex and ``mask`` (N,N) bool) via the per-point polar
    decomposition J = H W (H hermitian >= 0, W unitary), closed-form 2x2
    Pauli algebra, fully vectorized.

    Pauli ordering: s1 = x/y (0/90 linear), s2 = +/-45 linear,
    s3 = circular.

    Returns:
        dict of (N,N) maps (NaN off-mask): ``T`` intensity transmission
        (carries the engine source normalization -- ratios only), ``D``
        diattenuation magnitude, ``Dvec`` (N,N,3) components, ``ret``
        retardance magnitude in [0, pi], ``retvec`` (N,N,3), ``phase``
        unitary-part global phase (mod pi), ``ambiguous`` bool (ret
        within 0.2 rad of pi -- branch unresolved, flagged not chosen),
        ``mask``; plus ``mean`` and ``var_rms`` dicts (T, D, Dvec, ret,
        retvec), the pupil mean and the RMS of the variation SEPARATELY
        -- uniform retardance/diattenuation is a state change (and,
        after folds, the system's geometric rotation), not an
        aberration; only the variation drives a contrast floor or a PSI
        systematic.

    Basis dependence: ``D``/``T`` are singular-value invariants
    (basis-independent); ``ret``/``retvec`` are exactly what the
    double-pole basis makes artifact-free.
    """
    J = jp['J']
    mask = np.asarray(jp['mask'], dtype=bool)
    J11, J12 = J[..., 0, 0], J[..., 0, 1]
    J21, J22 = J[..., 1, 0], J[..., 1, 1]
    with np.errstate(invalid='ignore', divide='ignore'):
        return _pol_maps_body(J11, J12, J21, J22, mask)


def _pol_maps_body(J11, J12, J21, J22, mask):
    # off-mask points are NaN by construction (errstate-silenced caller)

    # hermitian product M = J^dag J (Pauli coefficients, real)
    M11 = np.abs(J11) ** 2 + np.abs(J21) ** 2
    M22 = np.abs(J12) ** 2 + np.abs(J22) ** 2
    M12 = np.conj(J11) * J12 + np.conj(J21) * J22
    t0 = (M11 + M22) / 2
    t1 = (M11 - M22) / 2
    t2 = M12.real
    t3 = -M12.imag
    tm = np.sqrt(t1 ** 2 + t2 ** 2 + t3 ** 2)

    tiny = np.finfo(float).tiny
    T = t0
    D = tm / np.maximum(t0, tiny)
    Dvec = np.stack([t1, t2, t3], axis=-1) / np.maximum(t0, tiny)[..., None]

    # polar: H = sqrtm(M) closed form; W = J H^-1
    detM = t0 ** 2 - tm ** 2
    sq = np.sqrt(np.maximum(detM, 0))
    den = np.sqrt(np.maximum(2 * t0 + 2 * sq, tiny))
    H11, H22, H12 = (M11 + sq) / den, (M22 + sq) / den, M12 / den
    detH = np.maximum(sq, tiny)
    Hi11, Hi22 = H22 / detH, H11 / detH
    Hi12, Hi21 = -H12 / detH, -np.conj(H12) / detH
    W11 = J11 * Hi11 + J12 * Hi21
    W12 = J11 * Hi12 + J12 * Hi22
    W21 = J21 * Hi11 + J22 * Hi21
    W22 = J21 * Hi12 + J22 * Hi22

    # retardance from the unitary part: strip psi (det W = e^{2i psi}),
    # W' = cos(d/2) I - i sin(d/2) (nhat . sigma)
    detW = W11 * W22 - W12 * W21
    psi = np.angle(detW) / 2
    ph = np.exp(-1j * psi)
    W11p, W12p, W21p, W22p = W11 * ph, W12 * ph, W21 * ph, W22 * ph
    c = (W11p + W22p).real / 2
    sn1 = (W22p - W11p).imag / 2
    sn2 = -(W12p + W21p).imag / 2
    sn3 = (W21p - W12p).real / 2
    neg = c < 0                                # canonicalize d in [0, pi]
    c = np.where(neg, -c, c)
    sn1 = np.where(neg, -sn1, sn1)
    sn2 = np.where(neg, -sn2, sn2)
    sn3 = np.where(neg, -sn3, sn3)
    psi = np.where(neg, psi + np.pi, psi)
    psi = np.mod(psi + np.pi / 2, np.pi) - np.pi / 2
    snm = np.sqrt(sn1 ** 2 + sn2 ** 2 + sn3 ** 2)
    delta = 2 * np.arctan2(snm, c)
    snmn = np.where(snm < tiny, 1, snm)
    retvec = np.stack([delta * sn1 / snmn, delta * sn2 / snmn,
                       delta * sn3 / snmn], axis=-1)
    ambiguous = mask & (delta > np.pi - 0.2)

    off = ~mask
    for a in (T, D, delta, psi):
        a[off] = np.nan
    Dvec[off] = np.nan
    retvec[off] = np.nan

    def _stat(x):
        v = x[mask] if x.ndim == 2 else x[mask]
        mu = v.mean(axis=0)
        return mu, np.sqrt(((v - mu) ** 2).mean(axis=0))

    mT, vT = _stat(T)
    mD, vD = _stat(D)
    mR, vR = _stat(delta)
    mDv, vDv = _stat(Dvec)
    mRv, vRv = _stat(retvec)

    return {'T': T, 'D': D, 'Dvec': Dvec, 'ret': delta, 'retvec': retvec,
            'phase': psi, 'ambiguous': ambiguous, 'mask': mask,
            'mean': {'T': mT, 'D': mD, 'Dvec': mDv, 'ret': mR,
                     'retvec': mRv},
            'var_rms': {'T': vT, 'D': vD, 'Dvec': vDv, 'ret': vR,
                        'retvec': vRv}}


_ANSI_NORM_RMS = (1.0, 2.0, 2.0, np.sqrt(6), np.sqrt(3), np.sqrt(6),
                  np.sqrt(8), np.sqrt(8), np.sqrt(8), np.sqrt(8),
                  np.sqrt(10), np.sqrt(10), np.sqrt(5), np.sqrt(10),
                  np.sqrt(10))

_ANSI_NAMES = ('piston', 'tilt-y', 'tilt-x', 'astig45', 'defocus', 'astig0',
               'trefoil-y', 'coma-y', 'coma-x', 'trefoil-x', 'quadrafoil-y',
               'astig45-2', 'spherical', 'astig0-2', 'quadrafoil-x')


def _ansi_nm(j: int) -> tuple:
    """Radial and azimuthal order of MACOS ANSI mode ``j`` (1-based)."""
    jj = j - 1
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * jj)) / 2))
    return n, 2 * jj - n * (n + 2)


def _ansi_zernike(j: int, rho, th):
    """MACOS ANSI Zernike mode ``j`` (1-based, as in MonZernModes=).

    Mirrors mmacos ``+macos/private/ansi_zernike_eval.m`` exactly --
    ZerntoMon1 convention, m < 0 -> sin(|m|th), RMS-normalized by
    NORM_RMS_PARAM_ANSI (elt_mod.F:288-299).  The two must agree: a mode
    index that means different things in the two bindings is a silent
    cross-language trap.
    """
    if j > len(_ANSI_NORM_RMS):
        raise ValueError(f'NORM_RMS_PARAM_ANSI tabulated to mode '
                         f'{len(_ANSI_NORM_RMS)} here (got {j})')
    n, m = _ansi_nm(j)
    am = abs(m)
    R = np.zeros_like(rho)
    for s in range((n - am) // 2 + 1):
        c = ((-1) ** s * math.factorial(n - s)
             / (math.factorial(s) * math.factorial((n + am) // 2 - s)
                * math.factorial((n - am) // 2 - s)))
        R = R + c * rho ** (n - 2 * s)
    ang = np.cos(m * th) if m >= 0 else np.sin(am * th)
    return _ANSI_NORM_RMS[j - 1] * R * ang


def pol_zernike(pm: dict, modes=None, center=None, radius=None,
                orthonormalize: bool = False) -> dict:
    """Low-order Zernike expansion of polarization-aberration maps.

    Expands the diattenuation and retardance maps in ``pm`` (from
    :func:`pol_maps`) onto a Zernike basis over the unvignetted pupil,
    giving the standard polarization-aberration terms -- piston, tilt,
    defocus, astigmatism and up -- in each Pauli component.  This is what
    makes a MACOS result comparable term-by-term with the published
    polarization-aberration literature, which is written in aberration
    terms rather than as maps.

    For an on-axis rotationally symmetric two-mirror system the theory
    predicts a specific answer: diattenuation and retardance grow as
    rho**2 with a 2*theta azimuthal dependence, so the expansion must be
    dominated by ASTIGMATISM in the two linear Pauli components (mode 6 =
    astig0 in s1, mode 4 = astig45 in s2, equal magnitude), with no
    circular (s3) content and no defocus -- "polarization astigmatism".

    The fit is least-squares, which is the correct estimator whether or
    not the basis is orthogonal over the actual mask; on an obscured
    (annular) pupil circular Zernikes are NOT orthogonal, so the
    conditioning is reported in ``cond``.

    Args:
        pm: dict from :func:`pol_maps` (needs Dvec, retvec, D, ret, mask;
            uses ``ambiguous`` if present).
        modes: MACOS ANSI 1-based mode indices.  Default 1..15.
        center: (i, j) pupil centre in grid indices.  Default: mask
            centroid.
        radius: normalization radius in pixels.  Default: the largest
            mask-point distance from the centre, so rho <= 1.
        orthonormalize: Gram-Schmidt the basis over the ACTUAL mask before
            fitting.  Coefficients become mutually independent but are no
            longer standard Zernike coefficients -- use for energy
            bookkeeping, not for literature comparison.

    Returns:
        dict with ``modes``, ``names``, ``nm`` (K,2), ``D`` (K,3),
        ``ret`` (K,3), ``Dmag`` (K,), ``retmag`` (K,), ``resid_rms``,
        ``frac`` (fraction of each map's mean square explained),
        ``cond``, ``recon`` (reconstructed maps, NaN off-mask),
        ``center``, ``radius``, ``npts``, ``npts_ret``,
        ``orthonormalized``, ``mask``.

    Mode 1 (piston) is the pupil MEAN; every other mode is variation
    about it.  Keep the separation -- a uniform diattenuation or
    retardance is a state change, not an aberration, and only the
    variation drives a contrast floor or a PSI systematic.

    Retardance caveat: points flagged ``pm['ambiguous']`` (retardance
    within 0.2 rad of pi, where the branch is unresolved) are EXCLUDED
    from the retardance fits and counted in ``npts_ret``.
    """
    modes = np.arange(1, 16) if modes is None else np.asarray(modes, int)
    mask = np.asarray(pm['mask'], dtype=bool)
    if not mask.any():
        raise ValueError('pupil mask is empty')
    ii, jj = np.indices(mask.shape).astype(float)   # first index = +x
    ctr = ((ii[mask].mean(), jj[mask].mean()) if center is None
           else (float(center[0]), float(center[1])))
    dx, dy = ii - ctr[0], jj - ctr[1]
    rr = np.hypot(dx, dy)
    rad = float(rr[mask].max()) if radius is None else float(radius)
    if rad <= 0:
        raise ValueError('pupil radius must be positive')
    rho, th = rr / rad, np.arctan2(dy, dx)

    basis = np.stack([_ansi_zernike(int(j), rho, th) for j in modes], axis=-1)

    def _design(sel):
        A = basis[sel]
        if orthonormalize:
            A, _ = np.linalg.qr(A)
            A = A * np.sqrt(A.shape[0])
        return A

    A = _design(mask)
    cond = float(np.linalg.cond(basis[mask]))

    retmask = mask
    amb = pm.get('ambiguous')
    if amb is not None:
        a = np.asarray(amb)
        retmask = mask & ~np.where(np.isnan(a.astype(float)), False,
                                   a.astype(bool))
    Ar = A if np.array_equal(retmask, mask) else _design(retmask)

    def _fit(A_, sel, Map):
        y = np.asarray(Map)[sel]
        ok = np.isfinite(y)
        Au, yu = (A_[ok], y[ok]) if not ok.all() else (A_, y)
        c, *_ = np.linalg.lstsq(Au, yu, rcond=None)
        r = yu - Au @ c
        den = ((yu - yu.mean()) ** 2).mean()
        frac = 1.0 if den <= 0 else 1 - ((r - r.mean()) ** 2).mean() / den
        recon = np.full(np.asarray(Map).shape, np.nan)
        idx = np.where(sel)
        if not ok.all():
            idx = tuple(v[ok] for v in idx)
        recon[idx] = Au @ c
        return c, float(np.sqrt((r ** 2).mean())), float(frac), recon

    cD = np.zeros((len(modes), 3))
    cR = np.zeros((len(modes), 3))
    rD = np.zeros(3)
    rR = np.zeros(3)
    fD = np.zeros(3)
    fR = np.zeros(3)
    reconD = np.full(mask.shape + (3,), np.nan)
    reconR = np.full(mask.shape + (3,), np.nan)
    for c in range(3):
        cD[:, c], rD[c], fD[c], reconD[..., c] = _fit(
            A, mask, pm['Dvec'][..., c])
        cR[:, c], rR[c], fR[c], reconR[..., c] = _fit(
            Ar, retmask, pm['retvec'][..., c])
    cDm, rDm, fDm, reconDm = _fit(A, mask, pm['D'])
    cRm, rRm, fRm, reconRm = _fit(Ar, retmask, pm['ret'])

    nm = np.array([_ansi_nm(int(j)) for j in modes])
    names = [(_ANSI_NAMES[j - 1] if j <= len(_ANSI_NAMES)
              else f'n{n} m{m:+d}') for j, (n, m) in zip(modes, nm)]
    return {'modes': modes, 'names': names, 'nm': nm,
            'D': cD, 'ret': cR, 'Dmag': cDm, 'retmag': cRm,
            'resid_rms': {'D': rD, 'ret': rR, 'Dmag': rDm, 'retmag': rRm},
            'frac': {'D': fD, 'ret': fR, 'Dmag': fDm, 'retmag': fRm},
            'cond': cond,
            'recon': {'Dvec': reconD, 'retvec': reconR,
                      'D': reconDm, 'ret': reconRm},
            'center': ctr, 'radius': rad,
            'npts': int(mask.sum()), 'npts_ret': int(retmask.sum()),
            'orthonormalized': orthonormalize, 'mask': mask}


def ffp(place_elt: int | np.int32,
        offset: Tuple[float, float]) -> None:
    """FFP -- place an off-axis field point by DIRECTION COSINES (sky angle).

    Tilts the source so the image at element ``place_elt`` lands at the
    off-axis field point ``offset = (dx, dy)`` given as direction cosines
    (normalized; ~= field angle in rad for small angles).  This is the
    "angle on the sky" placement; :func:`pfp` is the focal-pixel sibling
    (the two differ by the plate scale).  Requires the system stop set
    first (:func:`stop`); FFP changes the source pointing, so reset the
    Return / exit-pupil reference surfaces afterwards (:func:`ors`,
    :func:`fex`).

    Args:
        place_elt: element at which to position the off-axis image.
        offset: (dx, dy) direction cosines.

    Raises:
        Exception: Rx not loaded, element out of range, or MACOS rejected.
    """
    _chk_macos_and_rx_loaded()
    s = int(_map_Elt(place_elt, max_rows=1).squeeze())
    ok = lib.api.ffp(s, np.float64(offset[0]), np.float64(offset[1]))
    if not ok:
        raise Exception(f'MACOS: ffp(place_elt={s}) failed -- stop set?')


def pfp(place_elt: int | np.int32,
        pix_size: float,
        offset: Tuple[float, float]) -> None:
    """PFP -- place an off-axis field point in focal-plane PIXELS.

    Focal-plane-pixel sibling of :func:`ffp` (which places by direction
    cosines / sky angle); the two differ by the plate scale.  Positions
    the image at element ``place_elt`` at pixel ``offset = (dx, dy)`` on a
    grid of pitch ``pix_size`` (BaseUnits) -- match the COMPOSE pitch.
    Requires the system stop set first (:func:`stop`); reset reference
    surfaces afterwards (:func:`ors`, :func:`fex`).

    Args:
        place_elt: element at which to position the off-axis image.
        pix_size: pixel size in BaseUnits.
        offset: (dx, dy) image position in pixels.

    Raises:
        Exception: Rx not loaded, element out of range, or MACOS rejected.
    """
    _chk_macos_and_rx_loaded()
    s = int(_map_Elt(place_elt, max_rows=1).squeeze())
    ok = lib.api.pfp(s, np.float64(pix_size),
                     np.float64(offset[0]), np.float64(offset[1]))
    if not ok:
        raise Exception(f'MACOS: pfp(place_elt={s}) failed -- stop set?')


def obs_set(option: str = 'positive') -> None:
    """OBS -- set the ray-trace obscuration option for spot diagrams.

    Controls which rays :func:`spot` plots (iObsOpt):
      'all'      -- every ray, regardless of obscuration (option 0)
      'positive' -- unobscured rays only (default; option 1)
      'negative' -- obscured rays only (option 2)
    Session state -- persists until changed.  Set 'all' to draw a spot at
    a focal plane where every ray is obscured (a coronagraph FP), then
    restore 'positive' so the diffraction trace stays masked.

    Args:
        option: 'all' | 'positive' | 'negative'.

    Raises:
        Exception: Rx not loaded, bad option, or MACOS rejected the call.
    """
    _chk_macos_and_rx_loaded()
    codes = {'all': 0, 'positive': 1, 'negative': 2}
    if option not in codes:
        raise Exception(
            f"obs_set: option must be 'all'|'positive'|'negative', got {option!r}")
    if not lib.api.obs_set(np.int32(codes[option])):
        raise Exception('MACOS: obs_set failed')


def fex(mode=1) -> Tuple[np.float64, Vector[np.float64], Vector[np.float64]]:
    """Find Exit Pupil (XP) and sets parameters at Srf. nElt-1

    Args:
        mode (int | np.int32):
            to centre Ref.Srf.: w.r.t. (=1): Chief Ray  (default) OR
                                       (=0): Centroid

    Requirement:
        - The Stop of the Optical System must be set beforehand
        - The XP surface at srf nElt-1 will be updated.

    Raises:
        Exception: MACOS Triggered error

    Returns:
        rad: (np.float64)
            Radius of Curvature of Reference Sphere

        psi: (Vector[np.float64])
            Psi(L,M,N) -- Surface Direction Cosine in Global CSYS

        vpt: (Vector[np.float64])
            Vpt(x,y,z) -- Surface Position in Global CSYS
    """
    _chk_macos_and_rx_loaded()

    if lib.api.n_elt() <= 3:
        raise Exception("'fex': not more than 3 surfaces defined")

    ok, xp = lib.api.xp_fnd(np.int32(mode))

    if not ok:
        raise Exception("'fex' threw an exception - stop set?")
    return xp


def xp(vpt=None, psi=None, ref_rad=None) -> None | Tuple:
    """Set/Get Exit Pupil (XP) parameters at XP Srf. @ nElt-1

    Args:
        vpt: (None | Vector[np.float64]) default: None
            Vpt(x,y,z) -- Surface Position in Global CSYS

        psi: (None | Vector[np.float64]) default: None
            Psi(L,M,N) -- Surface Direction Cosine in Global CSYS

        rad: (None | np.float64) default: None
            Radius of Curvature of Reference Sphere

    Raises:
        Exception: MACOS Triggered error

    Returns:
        None:
            XP parameters were set

        Tuple[vpt, psi, rad]:

            vpt: (Vector[np.float64])
                Vpt(x,y,z) -- Surface Position in Global CSYS

            psi: (Vector[np.float64])
                Psi(L,M,N) -- Surface Direction Cosine in Global CSYS

            rad: (np.float64)
                Radius of Curvature of Reference Sphere
    """
    _chk_macos_and_rx_loaded()

    # read XP parameters
    params = vpt is None and psi is None and ref_rad is None
    if (params):
        ok, vpt, psi, rad = lib.api.xp_get()
        if not ok:
            raise ValueError("MACOS threw an exception")
        return vpt, psi, rad

    # write XP parameters
    if not (params):
        vpt = np.asarray_chkfinite(vpt)
        psi = np.asarray_chkfinite(psi)
        psi /= np.linalg.norm(psi)
        ref_rad = np.asarray_chkfinite(ref_rad)

        if not lib.api.xp_set(vpt, psi, ref_rad):
            raise ValueError("MACOS threw an exception")

    # invalid input
    else:
        raise ValueError("define either all or none of the parameters")


def stop(srf: None | int | Tuple[int] | np.int32 = None,
         offset: None | Tuple[float] | Vector[np.float64] = None
         ) -> None | Tuple:
    """Set/Get Optical System Stop Information

    The stop surface cannot be defined at
        -- the image plane or at the XP or at the object
        -- None-Sequential or Segment Surface

    Args:
        srf (None | int | Tuple[int] | np.int32, optional): Defaults to None.
           Element ID (Range: 0 < srf < nElt-2)
           Neg. values are referenced with respect to the last surface
           where -1 (== # of Elements) is the last surface, i.e., Img. Srf.

        offset (None | Tuple[float] | Vector[np.float64], optional): Defaults to None.
            [dx,dy]: Offset from Srf. Vertex Pos.
                     [0e0,0e0] if not defined when defining stop Srf.

    Raises:
        Exception: MACOS triggered error

    Returns:
        None:
            when defining the stop at a Surface

        Tuple[srf, offset]:
            Srf (int):
                Element ID where Stop Element is defined

            Offset (np.ndarray[np.float64]):
                [dx,dy]: Offset from Srf. Vertex Pos.

    """

    _chk_macos_and_rx_loaded()

    # read stop information
    if srf is None:
        ok, srf, offset = lib.api.stop_info_get()
        if not ok:
            raise Exception('MACOS threw an Exception')
        return srf, offset

    # set stop information
    else:
        srf = _map_Elt(srf).squeeze()
        offset = np.array((0., 0.), dtype=float) if offset is None \
                             else np.asarray_chkfinite(offset, order='F')
        if not lib.api.stop_info_set(srf, offset):
            raise Exception('MACOS threw an Exception')


def stop_obj(x: float, y: float, z: float) -> None:
    """Define / re-enforce an object-space stop.

    The OBJ branch of macos's ``STOP`` command: given a 3D point in
    object space, redirect the chief ray from the current source
    position through that point.  Cheap (no iterative element solve,
    just the geometric chief-ray-aim math in
    ``macos_cmd_loop.inc:2957-3005``).

    Use cases:
        - Prescriptions that declare an object-space stop via
          ``ApStop= x y z`` (e.g. Rx_e5hex1.in, where the segmented
          primary is the natural stop and pymacos's :func:`stop`
          refuses Segment surfaces).
        - Re-aiming the chief ray after a source perturbation
          (``perturb_src``) when an object-space stop is in use --
          for an element-based stop, re-call :func:`stop` instead.

    Args:
        x, y, z: object-space stop position in prescription
                 BaseUnits.  Conventionally ``(0, 0, 0)`` for
                 telescopes whose source is at infinity and whose
                 chief ray nominally passes through the global
                 origin.

    Raises:
        Exception: MACOS triggered error.
    """
    _chk_macos_and_rx_loaded()
    if not lib.api.stop_obj_set(float(x), float(y), float(z)):
        raise Exception('MACOS threw an Exception')


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
