"""
==============================
MACOS via Python
==============================

A module for analyzing / designing optical systems.

"""
from __future__ import absolute_import

from .macos import *
from .version import __version__

# -------------------
# needed when non-anaconda Python version is used; otherwise, pymacos DLL
# cannot be loaded (missing Intel & MS DLL's)
#  => requires Intel Redistributable to be installed (or add to search path?)
#
# import os
# os.add_dll_directory("C:\\Program Files (x86)\\Intel\\oneAPI\\2025.0\\bin")
# -------------------

# Note: Python's MACOS DLL (pymacosf90.cp313-win_amd64.pyd) is to placed in
#       this folder.
