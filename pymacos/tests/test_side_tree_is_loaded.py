"""A side-tree run tests the side tree (2026-10-22).  With PYMACOS_SRC set, the extension in use must be
the one under it: proper_compare/conftest.py once put this tree's src/ first and imported pymacos at
collection, so every whole-suite side-tree run silently tested the shared .so.  Without PYMACOS_SRC the
test only records which .so was loaded."""
import os
from pathlib import Path

from context import pymacosf90


def test_the_extension_is_the_requested_tree():
    so = Path(pymacosf90.__file__).resolve()
    side = os.environ.get('PYMACOS_SRC')
    print('pymacosf90 loaded from', so)
    if side:
        assert Path(side).resolve() in so.parents, f'PYMACOS_SRC={side} but the loaded .so is {so}'
