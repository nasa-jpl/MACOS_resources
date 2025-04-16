# from __future__ import absolute_import

# https://docs.python-guide.org/writing/structure/
# https://dev.to/codemouse92/dead-simple-python-project-structure-and-imports-38c6
import sys
from pathlib import Path

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, str(Path(".").absolute().parent / 'Src'))

import pymacos.macos as pymacos
import pymacos.pymacosf90 as pymacosf90


# decorator: change ray-tracing wavelength for ray-trace comparisons
def set_wavelength(func):
    """ changes the source wavelength """
    def set_trace_wavelength(*args, **kwargs):
        pymacos.src_wvl(args[0]*1e-6)  # upd. trace wavelength [nm] => [mm]
        func(*args, **kwargs)                          # call test definition (no args needed)

    return set_trace_wavelength


# decorator: change ray-tracing wavelength for ray-trace comparisons
def set_wavelength_no_args(func):
    """ changes the source wavelength """
    def set_trace_wavelength(*args, **kwargs):
        pymacos.src_wvl(args[0]*1e-6)  # upd. trace wavelength [nm] => [mm]
        func()                         # call test definition (no args needed)
    return set_trace_wavelength


## decorator: change ray-tracing wavelength
#def set_wavelength(func): #, *args, **kwargs):
#    """ changes the source wavelength """
#    def set_trace_wavelength(*args, **kwargs):
#        pymacos.src_wvl(args[0]*1e-6)  # upd. trace wavelength [nm] => [mm]
#        func(*args, **kwargs)          # call test definition (no args needed)
#
#    return set_trace_wavelength


def rx_path(rx: Path, as_str:bool = False) -> Path | str:
    """Create abs. path to Rx and check if it exists

    Args:
        rx (Path): path to Rx

    Returns:
        Path: abs. path to Rx
    """
    rx_ = (Path('.').absolute() / 'Rx' / rx).resolve()
    if not rx_.is_file():
        raise FileExistsError(f"{rx} was not found in ./Rx/ dir.")

    return str(rx_) if as_str else rx_
