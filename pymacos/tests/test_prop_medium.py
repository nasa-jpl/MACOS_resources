"""Medium-aware physical-optics kernels (macos propsub.F, 2026-09-30).

Every propagation kernel (NFPROP, PPPROP, SFPROP, FRPROP, the DFT legs,
FFPROP, SPH2PL/PL2SPH and FnCalc) used to be handed WaveBU, the VACUUM
wavelength, whatever medium the leg ran in.  The inter-leg geometric phase
was right (CumRayL is an optical path), but a leg inside glass ran at the
wrong Fresnel number by n.  Now each leg gets WaveBU / n_leg, n_leg the index
of the medium at the leg's start element.

The gate is an identity: Fresnel diffraction depends on lambda*z only, so a
leg of length z in index n must equal the same leg of length z/n in vacuum.
Rx_PropMedium_glass.in is Rx_VecChain.in with IndRef 1.5 everywhere and the
geometry unchanged; Rx_PropMedium_vac.in is Rx_VecChain.in with every axial
distance scaled by 1/1.5.  Intensities are compared (the complex fields differ
by a global piston, OPL n*z vs z/n).

Non-vacuity (measured on the pre-fix engine, 2026-09-30): the glass twin
matched the UNSCALED vacuum deck to 1.1e-15 and differed from the scaled one
by 74%.  Post-fix: glass == scaled to 1.4e-15, glass != unscaled by 71-73%.
Found while scoping the dyson5 challenge, whose Dyson block puts every leg
inside silica.
"""
import numpy as np
import pytest
from context import pymacos as m
from context import rx_path

MODEL = 256
ELTS = (2, 4)          # after leg 1 (PupilStop -> MidStop) and leg 2 (Prop2Start -> Detector)


def _intensities(name):
    m.init(MODEL)
    m.load(str(rx_path(name)))
    return {e: np.asarray(m.intensity(e)).copy() for e in ELTS}


def _rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


@pytest.fixture(scope="module")
def twins():
    return (_intensities('Rx_PropMedium_glass.in'),
            _intensities('Rx_PropMedium_vac.in'),
            _intensities('Rx_VecChain.in'))


@pytest.mark.parametrize("elt", ELTS)
def test_glass_leg_equals_scaled_vacuum_leg(twins, elt):
    """z in index n == z/n in vacuum: the kernels use lambda/n."""
    glass, vac, _ = twins
    assert _rel(glass[elt], vac[elt]) < 1e-12


@pytest.mark.parametrize("elt", ELTS)
def test_glass_leg_differs_from_unscaled_vacuum(twins, elt):
    """The must-fail leg: the pre-fix engine matched THIS deck to 1e-15."""
    glass, _, unscaled = twins
    assert _rel(glass[elt], unscaled[elt]) > 0.5


def test_energy_is_the_same_in_both_media(twins):
    glass, vac, _ = twins
    for e in ELTS:
        assert abs(glass[e].sum() - vac[e].sum()) / vac[e].sum() < 1e-10
