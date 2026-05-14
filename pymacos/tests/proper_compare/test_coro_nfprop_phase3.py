"""
Phase 3: per-step NF-prop comparisons further down the Rx_Coro
optical chain.

Phase 2 tested NFPlane between Elt 2 (Prop_1_start) and Elt 3
(Prop_1_end) -- the propagation just after the first OAP.  Phase 3
adds the analogous test between Elt 5 (Prop_2_start) and Elt 6
(Prop_2_end) -- the propagation just after the DM in the second
relay.  Same NFPlane physics, different wavefront content (includes
DM phase + intermediate geometric stages); a clean cross-check that
the engines agree across the chain rather than just at one location.

The NF1/NF2 spherical-to-flat propagation from Elt 8 (1stPropStart)
to Elt 9 (CorMask) is a separate kind of test -- macos's output
sampling at Elt 9 (1.93 um/pix) is 170x finer than the input
(0.333 mm/pix), so PROPER needs to be told about the beam's
convergence (prop_lens / beam-state API) to produce comparable
output.  Tackled as a separate iteration once 5->6 is in place.
"""
from __future__ import absolute_import

import numpy as np
import pytest

from .conftest import compare_and_record
from .geometries.coro_nfprop import (
    CoroNFprop, macos_run, proper_run,
    CoroSphereToPlane, DEFAULT_SPHERE_TO_PLANE,
    macos_run_sphere_to_plane, proper_run_sphere_to_plane)

pytestmark = pytest.mark.proper_compare


# Phase 3 geometry: same dataclass machinery, different element pair.
# dx_m for Elt 5 / Elt 6 is the same 0.33382 mm as Elt 2 / Elt 3 because
# NFPlane propagation preserves the diffraction-grid sampling.
NFPLANE_5_TO_6 = CoroNFprop(
    src_elt=5, detector_elt=6,
    propagation_m=0.774,
)


def test_coro_nfplane_elt5_to_elt6(pymacos_session, results_dir_phase3):
    """NFPlane propagation between Elt 5 (Prop_2_start) and Elt 6
    (Prop_2_end) of Rx_Coro.in.  Mirror of the Phase 2 test at Elt
    2 -> Elt 3, but with the wavefront having gone through the
    intermediate DM (Elt 4) and the geometric/lens transitions.
    """
    intensity_m, dx_m, wf2 = macos_run(NFPLANE_5_TO_6, pymacos_session)
    intensity_p, dx_p      = proper_run(NFPLANE_5_TO_6,
                                        wavefront_at_elt2=wf2)

    assert intensity_m.shape == intensity_p.shape
    assert dx_p == pytest.approx(dx_m, rel=1e-3)

    metrics = compare_and_record(
        'coro_nfplane_elt5_to_elt6',
        intensity_m, intensity_p, dx_m,
        results_dir_phase3,
        crop_pixels=intensity_m.shape[0],
        norm_kind='sum',
        extra_metadata={
            'wavelength_m':       NFPLANE_5_TO_6.wavelength_m,
            'propagation_m':      NFPLANE_5_TO_6.propagation_m,
            'src_elt':            NFPLANE_5_TO_6.src_elt,
            'detector_elt':       NFPLANE_5_TO_6.detector_elt,
            'rx_filename':        NFPLANE_5_TO_6.rx_filename,
            'macos_complex_field_at_src':  wf2['complex_field'],
        })

    # Tolerance: 1e-7 max, 1e-8 RMS (sum-normalised).  Looser than
    # Phase 2 (1e-8) because the wavefront at Elt 5 carries the
    # accumulated propagation history of OAP_1 + NFPlane (Elt 2-3) +
    # geometric (3-4) + DM (4) + geometric (4-5), so its phase
    # structure is richer than the post-OAP_1 wavefront tested in
    # Phase 2.  Richer phase content -> larger kernel-discretisation
    # residual when independently propagated by the two engines.
    # Observed: ~2e-8 max, ~5e-10 RMS -- still ~9 sig-fig agreement.
    assert metrics['max_abs'] < 1e-7, (
        f"max |a-b| = {metrics['max_abs']:.3e} (sum-normalised); "
        f"RMS = {metrics['rms_abs']:.3e}; "
        f"Δcom = ({metrics['dx_pix']:+d}, "
        f"{metrics['dy_pix']:+d}) px")
    assert metrics['rms_abs'] < 1e-8, (
        f"RMS |a-b| = {metrics['rms_abs']:.3e} (sum-normalised); "
        f"max = {metrics['max_abs']:.3e}")


def test_coro_sphere_to_plane_elt8_to_elt9(pymacos_session,
                                            results_dir_phase3):
    """Sphere-to-plane propagation from Elt 8 (1stPropStart, spherical
    reference KrElt=-774) to Elt 9 (CorMask, plane).  Geometrically a
    far-field calc just like Phase 1's Cass FF: macos's
    PropType=NF1/NF2 reflects the local frame ("near-field of the
    focus"), but the physics is sphere-to-plane FF, so PROPER uses
    prop_lens(f=774) + prop_propagate(f=774).

    This is the first comparison step where macos's diffraction-grid
    pixel pitch CHANGES between source (0.333 mm/pix) and destination
    (1.93 um/pix, 170x finer) -- PROPER's FF kernel rebins exactly
    the same way, by construction (focal-plane dx = lambda * f /
    grid_extent matches macos's reported dx2).
    """
    intensity_m, dx_m, wf = macos_run_sphere_to_plane(
        DEFAULT_SPHERE_TO_PLANE, pymacos_session)
    intensity_p, dx_p     = proper_run_sphere_to_plane(
        DEFAULT_SPHERE_TO_PLANE, wavefront_at_pupil=wf)

    assert intensity_m.shape == intensity_p.shape
    assert dx_p == pytest.approx(dx_m, rel=1e-3), (
        f"focal-plane sampling mismatch: macos dx={dx_m:.3e}, "
        f"PROPER dx={dx_p:.3e}")

    metrics = compare_and_record(
        'coro_sphere_to_plane_elt8_to_elt9',
        intensity_m, intensity_p, dx_m,
        results_dir_phase3,
        crop_pixels=intensity_m.shape[0],
        norm_kind='peak',                 # image-plane (focal): Strehl form
        extra_metadata={
            'wavelength_m':            DEFAULT_SPHERE_TO_PLANE.wavelength_m,
            'focal_length_m':          DEFAULT_SPHERE_TO_PLANE.focal_length_m,
            'src_elt':                 DEFAULT_SPHERE_TO_PLANE.src_elt,
            'detector_elt':            DEFAULT_SPHERE_TO_PLANE.detector_elt,
            'rx_filename':             DEFAULT_SPHERE_TO_PLANE.rx_filename,
            'macos_cfield_at_pupil':   wf['complex_field'],
        })

    # First-cut tolerance.  Will tighten or adjust use_cfield_phase
    # after the actual residual is observed.
    assert metrics['max_abs'] < 0.1, (
        f"max |a-b| = {metrics['max_abs']:.3e} (Strehl-normalised); "
        f"RMS = {metrics['rms_abs']:.3e}; "
        f"Δcom = ({metrics['dx_pix']:+d}, "
        f"{metrics['dy_pix']:+d}) px")
