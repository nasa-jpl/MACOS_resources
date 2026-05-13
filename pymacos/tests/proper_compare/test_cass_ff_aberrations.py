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
                                         pymacos_session, results_dir):
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
        results_dir,
        extra_metadata={
            'wavelength_m':     DEFAULT.wavelength_m,
            'pupil_diameter_m': DEFAULT.pupil_diameter_m,
            'perturbation_elt': SECONDARY_ELT,
            'perturbation_RxRyRzTxTyTz': np.array(prb, dtype=float),
            'macos_opd_at_xp':  opd,
        })

    # Use the centroid-aligned metric: macos chief ray re-aims under
    # tilt-class perturbations and the PSF shifts on the detector,
    # but PROPER's prop_add_phase doesn't reproduce that shift (the
    # OPD as reported by macos is referenced to the chief ray).  The
    # *shape* of the PSF is what each engine actually computes from
    # the wavefront aberration, and that's what we want to check.
    assert metrics['max_abs_aligned'] < 0.15, (
        f"{name}: max |a-b| aligned = "
        f"{metrics['max_abs_aligned']:.3e} (Strehl-norm); "
        f"PSF shift = ({metrics['dx_pix']:+d}, "
        f"{metrics['dy_pix']:+d}) pixels")
    # OPD scale sanity: report ties RMS OPD to perturbation magnitude
    rms_opd = float(opd[opd != 0].std()) if (opd != 0).any() else 0.0
    assert np.isfinite(rms_opd)
