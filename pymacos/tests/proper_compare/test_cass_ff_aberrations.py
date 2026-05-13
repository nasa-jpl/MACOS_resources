"""
macos-vs-PROPER comparison under small secondary-mirror perturbations.

Workflow per case:
  1. Load Rx_Cass_FarField.in
  2. Apply a 6-DoF perturbation to the secondary (Elt 3)
  3. macos: trace to the exit pupil, capture OPD; trace to detector,
     capture intensity (INT)
  4. PROPER: thin-lens model + macos OPD as a phase map
  5. Compare focal-plane PSFs pixel-by-pixel after Strehl
     normalisation; save a 3-panel plot and append a row to the
     comparison report.

The perturbation magnitudes are deliberately small (microns to tens
of microns translation) so that:
  - the resulting OPD is a few hundred nm to a few um RMS
  - the PSF stays well-sampled by the 2.78 um/pix focal grid
  - both engines remain in their linear, well-conditioned regime
"""
from __future__ import absolute_import

import numpy as np
import pytest

from .conftest import compare_and_record
from .geometries.cass_farfield import DEFAULT, proper_run, macos_run

pytestmark = pytest.mark.proper_compare


# Each entry: (name, [Rx, Ry, Rz, Tx, Ty, Tz]).  Rotations in
# radians, translations in metres.  Secondary mirror is Elt 3.
SECONDARY_ELT = 3
PERTURBATIONS = [
    ("nominal",      [0, 0, 0, 0,      0,      0     ]),
    ("Tx_plus_1um",  [0, 0, 0, 1e-6,   0,      0     ]),
    ("Tx_minus_1um", [0, 0, 0, -1e-6,  0,      0     ]),
    ("Ty_plus_1um",  [0, 0, 0, 0,      1e-6,   0     ]),
    ("Tz_plus_5um",  [0, 0, 0, 0,      0,      5e-6  ]),
    ("Tz_minus_5um", [0, 0, 0, 0,      0,      -5e-6 ]),
]


@pytest.mark.parametrize('name, prb', PERTURBATIONS,
                         ids=[p[0] for p in PERTURBATIONS])
def test_compare_secondary_perturbation(name, prb,
                                         pymacos_session, results_dir_phase1):
    """One macos/PROPER comparison per secondary-mirror perturbation
    in the PERTURBATIONS table.
    """
    perturbation = (SECONDARY_ELT, prb) if any(prb) else None

    macos_int, dx_m, opd = macos_run(DEFAULT, pymacos_session,
                                      return_opd=True,
                                      perturbation=perturbation)
    proper_int, dx_p = proper_run(DEFAULT, macos_opd=opd)

    assert dx_p == pytest.approx(dx_m, rel=1e-3), \
        (f"focal-plane sampling mismatch: PROPER {dx_p:.3e}, "
         f"macos {dx_m:.3e}")
    assert macos_int.shape == proper_int.shape

    metrics = compare_and_record(
        f'cass_ff_perturb_SM_{name}',
        macos_int, proper_int, dx_m,
        results_dir_phase1,
        extra_metadata={
            'wavelength_m':     DEFAULT.wavelength_m,
            'pupil_diameter_m': DEFAULT.pupil_diameter_m,
            'perturbation_elt': SECONDARY_ELT,
            'perturbation_RxRyRzTxTyTz': np.array(prb, dtype=float),
            'macos_opd_at_xp':  opd,
        })

    # With macos's mask applied as PROPER's amplitude (via prop_multiply)
    # and the OPD sign reconciled, the two engines now agree to
    # numerical-precision level (~1e-11) on Strehl-normalised PSFs.
    # Keep a generous margin (1e-6) to swallow FFT round-off and
    # platform-dependent intel-MKL drift; large deviations indicate
    # a real regression in either pymacos or the harness.
    assert metrics['max_abs'] < 1e-6, (
        f"{name}: max |a-b| = {metrics['max_abs']:.3e} "
        f"(Strehl-norm); PSF shift = "
        f"({metrics['dx_pix']:+d}, {metrics['dy_pix']:+d}) px")
    rms_opd = float(opd[opd != 0].std()) if (opd != 0).any() else 0.0
    assert np.isfinite(rms_opd)
