# ------------------------------------------------------------------------------
#
# Provides a Python API to SMACOS
#
# ------------------------------------------------------------------------------
import functools
import warnings
from pathlib import Path
from typing import Any, List, NewType, Tuple, TypeVar, TypeVarTuple

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
    return lib.api.currrent_macos_model_size()


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


"""
  ----------------------------------------------------------------------------
  [ ] Element Surface Properties: Grating
  ----------------------------------------------------------------------------
  [x] elt_grating_any       Checks if any Grating Srfs. are defined in Rx
  [x] elt_grating_fnd       Find all elements with Grating Srfs. attached
  [x] elt_grating_params    Grating (h1HOE, RuleWidth, Trans. or Refl.)
  [x] elt_grating_type      Transmission or Reflective Grating
  [x] elt_grating_order     Grating Order (Param: OrderHOE)
  [x] elt_grating_rulewidth Rule Width
  [x] elt_grating_dir       h1HOE vector prpdicular. to the ruling dir and psiElt.
  ----------------------------------------------------------------------------
"""


def elt_grating_any() -> bool:
    """Checks if Gratings on Srfs. are defined in Rx

    Raises:
        Exception: MACOS and/or Rx not loaded

    Returns:
      found (bool):
        True if Grating are defined in Rx; otherwise, False
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

           When 'None' is defined, it uses all surfaces in Rx

    Raises:
        Exception: MACOS Triggered error

    Returns:
        ([]):
            Empty list when no Grating def. is defined
        (List[Integers]):
            List[0]: Surfaces IDs where Grating def. are defined
            List[1]: Grating ID: (=1) Reflection   Grating
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
           defines the displacement at the grid nodes for a [N x N] grid
           along the z-axis.
           default: None

    Raises:
        Exception: when input constrains are not satisfied within MACOS or
                   not a Grid Surface

    Returns:
            (None)      when defining the Grid surface
        (dx,grid_dz): (Parameter, Matrix[np.float64]) when extracting the Grid Data

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
        return sampling_spacing, grid_dz

    # write
    if sampling_spacing is not None and grid_dz is not None:

        sampling_spacing = np.asarray_chkfinite(sampling_spacing, dtype=np.float64)
        if np.any(sampling_spacing <= 0):
            raise ValueError("Srf. Grid Sampling Spacing must be greater than zero")

        grid_dz = np.asarray_chkfinite(grid_dz, dtype=np.float64)

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
           along the z-axis.

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

    grid_dz = np.asarray_chkfinite(grid_dz, dtype=np.float64)

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


def opd() -> Matrix[np.float64]:
    """Retrieve Optical Path Difference (OPD) at last ray-traced state.

    Requirement:
        The OPD can be obtained _after_ running a trace_rays(srf).
        For the OPD at the Exit Pupil, run a 'trace_rays(-2)'.

    Raises:
        Exception: MACOS Triggered error

    Returns:
        Matrix[np.float64]:
            opd: Optical Path Difference where size (nGridPts x nGridPts)

    Example:
        __ = pymacos.trace_rays(-2)  # Trace rays to XP
        pymacos.opd()                # get OPD map
    """
    _chk_macos_and_rx_loaded()

    # OPD map
    npts = lib.api.get_src_sampling()[1]   # == nGridPts
    ok, opd = lib.api.opd_val(npts)

    if not ok:
        raise Exception("MACOS: 'opd' threw an exception")

    return opd


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

    ok, npts = lib.api.spot_cmd(_map_Elt(srf).squeeze(),
                                np.int32(vpt_center),
                                np.int32(beam_csys),
                                np.int32(reset_trace))

    ok, pts, shift, centre, csys = lib.api.spot_get(npts)
    if not ok:
        raise Exception('MACOS threw an exeption')

    return pts, centre, shift[2:] if vpt_center else shift[:2], csys




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




# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass