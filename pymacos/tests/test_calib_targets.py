"""The CALIB targets WFE_ZMODE (works) and OPL (removed) -- PLAN 3.1, 2026-10-10; twin of
mmacos tCalibTargets.  Rx_AsphCalib.in: a metre paraboloid whose h^4 term is spoiled to 5e-3.

* WFE_ZMODE used to STOP the host ('nerror: svd_decomp: no convergence'): every load reset
  the Zernike-fit basis zcoType to 0, so zern_fit read ZernAnnuRatio(0) out of bounds.  Now
  [11] (Noll primary spherical) drives h^4 out and [12] leaves it -- in a SUBPROCESS.
* OPL had no objective; the name is gone and the engine refuses target 5.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest
from context import pymacos as m
from context import rx_path

HERE = Path(__file__).resolve().parent
SEED = 5e-3


def test_wfe_zmode_converges_on_the_named_mode_in_a_subprocess():
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import pymacos.macos as m\n"
        "for md in (11, 12):\n"
        "    m.init(128); m.load(%r)\n"
        "    m.calib_set_target('WFE_ZMODE', wf_zern_modes=[md]); r = m.calib()\n"
        "    print('MODE%%d %%d %%.6e' %% (md, int(bool(r['converged'])), m.get_elt_asph(1, 4)[0]), flush=True)\n"
    ) % (str(HERE.parent / 'src'), str(rx_path('Rx_AsphCalib.in')))
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, 'WFE_ZMODE must not kill the host:\n' + r.stdout[-2000:] + r.stderr[-2000:]
    c11, a11 = re.search(r'MODE11 (\d) (\S+)', r.stdout).groups()
    c12, a12 = re.search(r'MODE12 (\d) (\S+)', r.stdout).groups()
    assert int(c11) == 1 and abs(float(a11)) < 0.1 * SEED
    assert abs(float(a12)) > 0.5 * SEED, 'non-vacuity: mode 12 does not see the h^4 term'


def test_opl_is_refused_by_name():
    with pytest.raises(ValueError):
        m.calib_set_target('OPL')
    m.init(128)
    m.load(str(rx_path('Rx_AsphCalib.in')))
    with pytest.raises(Exception):
        m.calib_set_target(5)
