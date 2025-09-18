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

try: 
    from .macos import *
    from .version import __version__

except (ImportError):

    # ONLY on Windows  (can be ignored with Anaconda Environments)
    dll_path = "C:\\Program Files (x86)\\Intel\\oneAPI\\2025.2\\bin"

    import os
    if hasattr(os, 'add_dll_directory') and dll_path.strip():
        if not os.path.exists(dll_path):
            raise FileExistsError(f"DLL Path {dll_path} not found")
        os.add_dll_directory(dll_path)
    # -------------------
    from .macos import *
    from .version import __version__


# Note: Python's MACOS DLL (pymacosf90.cp313-win_amd64.pyd) is 
#       to be placed in this folder.
