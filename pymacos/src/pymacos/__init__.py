"""
==============================
MACOS via Python
==============================

A Module for Analyzing complex Controlled Optical Systems.

"""
from __future__ import absolute_import

# -------------------
# needed when non-anaconda Python version is used; otherwise, pymacos DLL
# cannot be loaded (missing Intel & MS DLL's)
#  => requires Intel Redistributable to be installed (or add to search path?)
#  => must be placed before importing the macos API
#
# Windows:
# (1) Install the Intel oneAPI compilers-redistributable-libraries for Fortran and C/C++
#     https://www.intel.com/content/www/us/en/developer/articles/tool/compilers-redistributable-libraries-by-version.html
# (2) open a new cmd window (to include any path changes)

try:
    from .macos import *
    from .version import __version__

except ImportError:

    # ONLY on Windows  (can be ignored with Anaconda Environments)
    dll_path = "C:\\Program Files (x86)\\Intel\\oneAPI\\2025.3\\bin"
    shared_libs = r"C:\Program Files (x86)\Common Files\Intel\Shared Libraries\bin"

    import os
    if hasattr(os, 'add_dll_directory'):  # only on Win OS
        if dll_path.strip() and os.path.exists(dll_path):
            os.add_dll_directory(dll_path)
        else:
            if os.path.exists(shared_libs):
                os.add_dll_directory(shared_libs)
            else:
                raise FileExistsError("No valid Intel Path to Libs found")

    # -------------------
    from .macos import *
    from .version import __version__


# Note: Python's MACOS DLL (pymacosf90.cp313-win_amd64.pyd) is
#       to be placed in this folder.
