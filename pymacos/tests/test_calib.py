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


RX_SRC = Path(
    '/home/dcr/dev/macos/ZGD_test_files/opt_example.in')


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
