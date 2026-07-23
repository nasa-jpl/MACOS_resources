"""Tests for m.calib() (Phase 1a: bare CALIB wrapper).

Uses opt_example.in -- an 11-element Cassegrain-style design with
optimization config (OptTarget=BeamOnly, OptMxItrs=20, VarDOF on
Elt 7) baked into the prescription.  The workflow under test is:

  load -> perturb upstream optic -> set stop -> calib -> assert converged
"""
import shutil
from pathlib import Path

import pytest

import context  # noqa: F401  -- adds ../src to sys.path
from pymacos import macos as m


RX_SRC = Path(__file__).resolve().parent / 'Rx' / 'opt_example.in'


@pytest.fixture()
def rx_path(tmp_path):
    """Copy the opt_example.in fixture to a writable tmp dir.

    macos sometimes writes a sibling Rx.out/StateIn file on CALIB
    success; running from tmp keeps the repo clean.
    """
    dst = tmp_path / 'opt_example.in'
    shutil.copy(RX_SRC, dst)
    return dst


def test_calib_runs_baseline(rx_path):
    """Bare wrap: load -> calib (with no perturbation), expect convergence
    on the already-optimized baseline."""
    m.init(256)
    m.load(str(rx_path))
    m.stop(7, [0.0, 0.0])

    result = m.calib()
    assert result['converged'], (
        f"baseline calib should converge; rtn_flag={result['rtn_flag']}")
    assert result['n_fov'] >= 1
    assert result['n_wavelength'] >= 1


def test_calib_converges_after_perturbation(rx_path):
    """Perturb Elt 1, then run CALIB; expect successful convergence."""
    m.init(256)
    m.load(str(rx_path))
    m.perturb(1, rotation_rad=(1e-3, 0, 0), translation_m=(0, 0, 0),
              in_local_coords=False)
    m.stop(7, [0.0, 0.0])

    result = m.calib()
    assert result['converged'], (
        f"calib should converge after small perturb; "
        f"rtn_flag={result['rtn_flag']}")


def test_calib_returns_expected_schema(rx_path):
    """Schema check on the returned dict (Phase 1a contract)."""
    m.init(256)
    m.load(str(rx_path))
    m.stop(7, [0.0, 0.0])

    result = m.calib()
    expected_keys = {
        'converged', 'rtn_flag', 'n_fov', 'n_wavelength',
        'old_wfe', 'new_wfe',
    }
    assert set(result.keys()) == expected_keys
    assert isinstance(result['converged'], bool)
    assert isinstance(result['rtn_flag'], int)
    n_fov = result['n_fov']
    n_wl = result['n_wavelength']
    assert result['old_wfe'].shape == (n_fov, n_wl)
    assert result['new_wfe'].shape == (n_fov, n_wl)


# --------------------------------------------------------------------
# Phase 1b: programmatic setters
# --------------------------------------------------------------------

def test_calib_set_var_elt_via_name_list(rx_path):
    """Configure CALIB entirely from Python: clear the prescription's
    VarDOF, define Elt 7 as TIP+TILT, run, expect convergence."""
    m.init(256)
    m.load(str(rx_path))
    m.stop(7, [0.0, 0.0])
    m.perturb(1, rotation_rad=(1e-3, 0, 0), translation_m=(0, 0, 0),
              in_local_coords=False)

    m.calib_clear_var_elts()
    m.calib_set_var_elt(7, dofs=['TIP', 'TILT'])

    result = m.calib()
    assert result['converged'], (
        f"calib after programmatic config did not converge; "
        f"rtn_flag={result['rtn_flag']}")


def test_calib_set_var_elt_via_positional_mask(rx_path):
    """Same as above but using the 8-int positional mask interface."""
    m.init(256)
    m.load(str(rx_path))
    m.stop(7, [0.0, 0.0])
    m.perturb(1, rotation_rad=(1e-3, 0, 0), translation_m=(0, 0, 0),
              in_local_coords=False)

    m.calib_clear_var_elts()
    # Position 1 = TIP, 2 = TILT.  The mask says "vary TIP and TILT".
    m.calib_set_var_elt(7, dofs=[1, 1, 0, 0, 0, 0, 0, 0])

    result = m.calib()
    assert result['converged']


def test_calib_set_iter_and_tol(rx_path):
    """Verify the iter + tol setters don't fail and the optimizer
    respects the cap (1 iter -> may not fully converge but mustn't
    crash)."""
    m.init(256)
    m.load(str(rx_path))
    m.stop(7, [0.0, 0.0])
    m.calib_set_iter(1)
    m.calib_set_tol(1e-12)
    # Just exercise the path; convergence with 1 iter is not guaranteed.
    m.calib()


def test_calib_set_target_accepts_names(rx_path):
    """All five named targets dispatch without errors."""
    m.init(256)
    m.load(str(rx_path))
    for name in ('WFE', 'BEAM', 'SPOT', 'OPL'):
        m.calib_set_target(name)
    # WFE_ZMODE requires modes -- exercise that path too.
    m.calib_set_target('WFE_ZMODE', wf_zern_modes=[4, 5, 11])
    # Aliases
    m.calib_set_target('ZWF', wf_zern_modes=[4])


def test_calib_set_target_rejects_unknown_name(rx_path):
    """Unknown target name -> ValueError before reaching Fortran."""
    m.init(256)
    m.load(str(rx_path))
    with pytest.raises(ValueError):
        m.calib_set_target('not_a_target')


def test_calib_set_var_elt_rejects_unknown_dof_name(rx_path):
    """Unknown DOF name -> ValueError, doesn't reach Fortran."""
    m.init(256)
    m.load(str(rx_path))
    with pytest.raises(ValueError):
        m.calib_set_var_elt(7, dofs=['TIP', 'not_a_dof'])
