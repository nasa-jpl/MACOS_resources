"""CALIB's target x DOF-type matrix (PLAN_CONSOLIDATION 3.4, 2026-10-11); twin of mmacos
tCalibMatrix.  Four targets (WFE, SPOT, WFE_ZMODE on Noll [4 7 8 11], BEAM = the chief ray to the
focal-plane vertex) x four spoiled fixtures of Rx_AsphCalib's f = 1 m paraboloid: rigid
(Rx_CalibRigid, tilted 1 mrad, TIP free), zernike (Rx_CalibZern, ANSI Z12 = 2e-7 m), aspheric
(Rx_AsphCalib, h^4 = 5e-3), mixed (Rx_CalibMixed, tilt + h^4).

* 13 WELL-POSED cells: the solve ends converged / floor / cap with the spoil removed.
* 3 ILL-POSED cells, BEAM x {zernike, aspheric, mixed}: a beam-position row cannot see a shape
  term -- the solve must raise, 'failed', the optics restored.
* HOST SURVIVAL, in a SUBPROCESS: SPOT with TIP AND TILT free on the rigid deck (the unspoiled
  TILT is first-order null) -- LM's first trial lost the chief ray and MACOS_OPS STOPPED the
  process (the mex: rc=139).  Now a failed trial is a REJECTED trial; this solve accepts none,
  and zero accepted steps with failures is a named failure.
* STALLED: Rx_CalibOptimal (no spoil, h^4 free) under SPOT -- 'stalled', and converged (an
  already-optimal start is the best outcome).
"""
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from context import pymacos as m
from context import rx_path

HERE = Path(__file__).resolve().parent
FIX = {'rigid': 'Rx_CalibRigid.in', 'zernike': 'Rx_CalibZern.in',
       'aspheric': 'Rx_AsphCalib.in', 'mixed': 'Rx_CalibMixed.in', 'optimal': 'Rx_CalibOptimal.in'}
WELL = ['rigid_WFE', 'rigid_SPOT', 'rigid_WFE_ZMODE', 'rigid_BEAM',
        'zernike_WFE', 'zernike_SPOT', 'zernike_WFE_ZMODE',
        'aspheric_WFE', 'aspheric_SPOT', 'aspheric_WFE_ZMODE',
        'mixed_WFE', 'mixed_SPOT', 'mixed_WFE_ZMODE']
ILL = ['zernike_BEAM', 'aspheric_BEAM', 'mixed_BEAM']


def _load(dof, tgt):
    m.init(128)
    m.load(str(rx_path(FIX[dof])))
    if tgt == 'WFE_ZMODE':
        m.calib_set_target('WFE_ZMODE', wf_zern_modes=[4, 7, 8, 11])
    elif tgt == 'BEAM':
        m.calib_set_target('BEAM')
        m.calib_set_beam('pos', 2, [0.0, 0.0, -1.0])
    else:
        m.calib_set_target(tgt)


def _spoil(dof):
    psi = np.asarray(m.elt_psi(1), dtype=float).ravel()
    tilt = math.acos(min(1.0, -psi[2] / np.linalg.norm(psi)))
    h4 = float(m.get_elt_asph(1, 1)[0]) if dof in ('aspheric', 'mixed') else float('nan')
    z12 = float(np.asarray(m.getEltSrfZernMode(1, 12)).ravel()[0]) if dof == 'zernike' else float('nan')
    return tilt, h4, z12


@pytest.mark.parametrize('cell', WELL)
def test_a_well_posed_cell_removes_its_spoil(cell):
    dof, tgt = cell.split('_', 1)
    _load(dof, tgt)
    r = m.calib()
    assert r['stop_reason'] in ('converged', 'floor', 'cap'), r['stop_reason']
    tilt, h4, z12 = _spoil(dof)
    assert tilt < 1e-8
    if not math.isnan(h4):
        assert abs(h4) < 1e-6
    if not math.isnan(z12):
        assert abs(z12) < 1e-12
    rms, _, _ = m.trace_rays(2)
    assert rms < 1e-10


@pytest.mark.parametrize('cell', ILL)
def test_an_ill_posed_cell_fails_by_name_with_the_optics_restored(cell):
    dof, tgt = cell.split('_', 1)
    _load(dof, tgt)
    s0 = _spoil(dof)
    with pytest.raises(Exception):
        m.calib()
    assert m.calib_stats()['stop_reason'] == 'failed'
    s1 = _spoil(dof)
    assert np.allclose(np.nan_to_num(s1), np.nan_to_num(s0), rtol=0, atol=1e-15)


def test_an_already_optimal_problem_stalls_without_an_error():
    m.init(128)
    m.load(str(rx_path(FIX['optimal'])))
    m.calib_set_target('SPOT')
    r = m.calib()
    assert r['stop_reason'] == 'stalled'
    assert r['converged'] is True
    assert r['n_failed'] == 0 and r['n_rejected'] == r['n_iter']


def test_a_trial_that_loses_the_chief_ray_does_not_kill_the_host():
    code = (
        "import sys, math; sys.path.insert(0, %r)\n"
        "import numpy as np, pymacos.macos as m\n"
        "m.init(128); m.load(%r)\n"
        "m.calib_set_var_elt(1, ['TIP', 'TILT']); m.calib_set_target('SPOT')\n"
        "try:\n    m.calib(); print('CALIB_RETURNED', flush=True)\n"
        "except Exception as e:\n    print('CALIB_FAILED', e, flush=True)\n"
        "p = np.asarray(m.elt_psi(1), dtype=float).ravel()\n"
        "print('TILT %%.6e' %% math.acos(-p[2] / np.linalg.norm(p)), flush=True)\n"
    ) % (str(HERE.parent / 'src'), str(rx_path(FIX['rigid'])))
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, 'the host must survive (pre-fix: MACOS_OPS STOPPED):\n' + r.stdout[-2000:] + r.stderr[-2000:]
    assert 'CALIB_FAILED' in r.stdout
    assert re.search(r'no trial step was accepted -- \d+ of \d+ trials lost the chief ray', r.stdout)
    tilt = float(re.search(r'TILT (\S+)', r.stdout).group(1))
    assert abs(tilt - 1e-3) < 1e-9
