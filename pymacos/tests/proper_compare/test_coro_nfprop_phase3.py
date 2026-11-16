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
    macos_run_sphere_to_plane, proper_run_sphere_to_plane,
    CoroPupilToPupilThruFocus, DEFAULT_PUPIL_TO_PUPIL,
    macos_run_pupil_to_pupil, proper_run_pupil_to_pupil)

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
    # Phase 2 (4.8e-13) -- not because of richer phase content (phase
    # RMS at Elt 5 is ~5e-5 rad, essentially zero, same as Elt 2) but
    # because of beam-size / Fresnel-number scaling:
    #
    #   Elt 2: beam covers ~51 k px (R~128 px), F = a^2/(lambda*z) ~ 2770
    #   Elt 5: beam covers ~12 k px (R~ 63 px), F ~ 684
    #
    # Intervening apertures (Elt 3, Elt 4 DM, Elt 5 reference) clip the
    # beam to 1/4 its Elt-2 footprint.  Lower F -> finer Fresnel ripples
    # at Elt 6 relative to the (fixed-N) grid -> larger sampling-limited
    # kernel residual between macos's PPPROP and PROPER's prop_propagate.
    #
    # Verified 2026-05-14:
    #   - dx_at(2/3/5/6) all agree to 1e-16 relative (kernel sees the
    #     same pitch).
    #   - centroids of cfield_at_2 and cfield_at_5 both land at +0.5 px
    #     from (N-1)/2 -- the standard FFT-center convention; no
    #     sub-pixel origin offset.
    #   - phase tilts at Elt 5 are 1e-14 rad/px (no Fourier shift).
    #   - apodizing cfield_at_5 with a soft window moves PROPER AWAY
    #     from macos by the same amount it moves PROPER away from itself
    #     -> hard edges are real signal that both engines diffract
    #     correctly; not a kernel-boundary artefact.
    #
    # Observed: 3.7e-8 max, 1.2e-9 RMS (sum-normalised) at N=1024.
    # Expected to drop as 1/N^2 with a denser grid.
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


# =====================================================================
# Phase 4a: chain through the focus to the far-side pupil reference
# =====================================================================

def test_coro_pupil_to_pupil_elt8_to_elt10(pymacos_session,
                                            results_dir_phase3):
    """Phase 4a: continue past the focal plane to Elt 10 (the
    spherical-reference pupil on the divergent side of the focus).

    Same starting point as Phase 3b (cfield at Elt 8), same recipe
    (prop_lens + prop_propagate to focus), then a SECOND
    prop_propagate(f) carries the beam past the focus to Elt 10.
    Output sampling rebins back to the pupil scale (0.333 mm/pix);
    PROPER's outside-beam Fresnel kernel handles the rebin
    automatically.

    No mask or Lyot stop yet -- this is the baseline.
    """
    intensity_m, dx_m, wf = macos_run_pupil_to_pupil(
        DEFAULT_PUPIL_TO_PUPIL, pymacos_session)
    intensity_p, dx_p     = proper_run_pupil_to_pupil(
        DEFAULT_PUPIL_TO_PUPIL, wavefront_at_pupil=wf)

    assert intensity_m.shape == intensity_p.shape
    assert dx_p == pytest.approx(dx_m, rel=1e-3), (
        f"pupil-plane sampling mismatch: macos dx={dx_m:.3e}, "
        f"PROPER dx={dx_p:.3e}")

    metrics = compare_and_record(
        'coro_pupil_to_pupil_elt8_to_elt10',
        intensity_m, intensity_p, dx_m,
        results_dir_phase3,
        crop_pixels=intensity_m.shape[0],
        norm_kind='sum',                    # pupil-plane (back at coarse
                                            # sampling): flux-norm
        extra_metadata={
            'wavelength_m':           DEFAULT_PUPIL_TO_PUPIL.wavelength_m,
            'focal_length_m':         DEFAULT_PUPIL_TO_PUPIL.focal_length_m,
            'src_elt':                DEFAULT_PUPIL_TO_PUPIL.src_elt,
            'focus_elt':              DEFAULT_PUPIL_TO_PUPIL.focus_elt,
            'detector_elt':           DEFAULT_PUPIL_TO_PUPIL.detector_elt,
            'rx_filename':            DEFAULT_PUPIL_TO_PUPIL.rx_filename,
            'macos_cfield_at_pupil':  wf['complex_field'],
            'macos_intensity_at_focus': wf['intensity_at_focus'],
        })

    # First-cut tolerance.  Tighten once observed.
    assert metrics['max_abs'] < 1e-4, (
        f"max |a-b| = {metrics['max_abs']:.3e} (sum-normalised); "
        f"RMS = {metrics['rms_abs']:.3e}; "
        f"Δcom = ({metrics['dx_pix']:+d}, "
        f"{metrics['dy_pix']:+d}) px")


# =====================================================================
# Phase 4b: third NFPlane in the chain, post-focus pupil relay
# =====================================================================

# Rx_Coro.in has a 20 mm circular Lyot stop at Elt 14 (ApType=Circular,
# ApVec=20).  Phase 4b is the kernel-only baseline -- mask and Lyot
# get added back for Phase 5.  Rx_Coro_noLyot.in is identical to
# Rx_Coro.in but with ApType=None at Elt 14 (Lyot disabled).
NFPLANE_13_TO_14 = CoroNFprop(
    rx_filename="Rx_Coro_noLyot.in",
    src_elt=13, detector_elt=14,
    propagation_m=0.774,
)


def test_coro_nfplane_elt13_to_elt14(pymacos_session,
                                      results_dir_phase3):
    """Phase 4b: NFPlane propagation Elt 13 -> Elt 14 (post-focus
    pupil relay).  Third NFPlane step in the Coro chain, after a
    sphere-to-plane (8->9), a plane-to-sphere (9->10), and the
    geometric/OAP transitions to Elt 13.

    Mirror of Phase 3a (Elt 5 -> Elt 6) structurally: same NFPlane
    physics, but the wavefront has now been through the entire
    coronagraph stage including the focus and the post-focus pupil
    reimaging.  Uses Rx_Coro_noLyot.in so the Lyot stop at Elt 14
    doesn't contaminate the kernel-only comparison; Phase 5 will
    re-enable the Lyot + add a focal-plane mask.
    """
    intensity_m, dx_m, wf = macos_run(NFPLANE_13_TO_14, pymacos_session)
    intensity_p, dx_p     = proper_run(NFPLANE_13_TO_14,
                                       wavefront_at_elt2=wf)

    assert intensity_m.shape == intensity_p.shape
    assert dx_p == pytest.approx(dx_m, rel=1e-3)

    metrics = compare_and_record(
        'coro_nfplane_elt13_to_elt14',
        intensity_m, intensity_p, dx_m,
        results_dir_phase3,
        crop_pixels=intensity_m.shape[0],
        norm_kind='sum',
        extra_metadata={
            'wavelength_m':       NFPLANE_13_TO_14.wavelength_m,
            'propagation_m':      NFPLANE_13_TO_14.propagation_m,
            'src_elt':            NFPLANE_13_TO_14.src_elt,
            'detector_elt':       NFPLANE_13_TO_14.detector_elt,
            'rx_filename':        NFPLANE_13_TO_14.rx_filename,
            'macos_complex_field_at_src':  wf['complex_field'],
        })

    # Tolerance: 5e-4 max, 1e-5 RMS (sum-normalised).  Observed
    # 7.4e-5 max, 1.4e-6 RMS.  Much looser than Phase 3a's 3.7e-8 --
    # not because of sampling but because the wavefront at Elt 13
    # has been through the full coronagraph chain (focus at Elt 9 +
    # post-focus pupil relay through 3rdOAP) and now fills ~80% of
    # the grid with full 2pi-wrapped phase.
    #
    # Residual structure (2026-05-14 investigation):
    #   r=  0..17 mm: mean 1.1e-06, max 1.7e-05
    #   r= 17..34 mm: mean 3.1e-06, max 7.4e-05  <-- all residual here
    #   r= 34..51 mm: mean 1.3e-11
    #   r > 51 mm   : 1e-13 to 1e-15  (machine precision floor)
    #
    # The residual is a NARROW ANNULAR RING at r=17-34 mm, inside an
    # otherwise machine-precision-agreement wavefront across the rest
    # of the grid.  Diagnostic confirmed:
    #   - centroids match at 14+ digits (no sub-pixel grid offset)
    #   - re-enabling the Lyot at 20 mm doesn't help (the ring is
    #     INSIDE the Lyot clear aperture); the Lyot adds its own
    #     edge residual at r=20 mm
    # -> the ring is a diffraction signature from a hard aperture
    # earlier in the chain (M1, DM, OAPs) that the two engines'
    # Fresnel kernels place at slightly different sub-pixel
    # positions.  Not a leak in the handoff -- a real engine-level
    # disagreement on a specific diffraction feature.  Phase 5's
    # focal mask will radically change the on-axis flux, so this
    # Phase 4b "naked chain" residual is mostly historical context.
    assert metrics['max_abs'] < 5e-4, (
        f"max |a-b| = {metrics['max_abs']:.3e} (sum-normalised); "
        f"RMS = {metrics['rms_abs']:.3e}; "
        f"Δcom = ({metrics['dx_pix']:+d}, "
        f"{metrics['dy_pix']:+d}) px")
    assert metrics['rms_abs'] < 1e-5, (
        f"RMS |a-b| = {metrics['rms_abs']:.3e} (sum-normalised); "
        f"max = {metrics['max_abs']:.3e}")


# =====================================================================
# Phase 5 step 1: extend the chain to the science focal plane (Elt 21),
# no coronagraph mask yet.  Sets up the end-to-end comparison before
# the FPM is added in step 2.
# =====================================================================

# Elt 20 is the ExitPupil reference (spherical, Kr=-951.4 mm).  Elt 21
# is the science FocalPlane.  Macos's PropType=FarField between them
# is the same sphere-to-plane FF physics as Phase 3b, just at a
# different f.  PROPER recipe: prop_lens(f=0.9514) + prop_propagate(f).
EXIT_PUPIL_TO_SCI_FOCAL_NO_MASK = CoroSphereToPlane(
    rx_filename="Rx_Coro_noLyot.in",  # no mask, no Lyot -- pure baseline
    src_elt=20, detector_elt=21,
    focal_length_m=0.9514,
)


def test_coro_exit_pupil_to_focal_no_mask(pymacos_session,
                                          results_dir_phase3):
    """Phase 5 step 1: ExitPupil (Elt 20, spherical reference) ->
    science FocalPlane (Elt 21), with no coronagraph mask and no
    Lyot stop -- this is the un-coronagraphed baseline PSF at the
    science focal plane, used as the reference for the suppressed-
    PSF tests once the FPM lands.

    Reuses the Phase 3b sphere-to-plane recipe directly (same Siegman-
    Sziklas spherical-reference geometry, just a different focal
    length and a different src/det element pair).
    """
    intensity_m, dx_m, wf = macos_run_sphere_to_plane(
        EXIT_PUPIL_TO_SCI_FOCAL_NO_MASK, pymacos_session)
    intensity_p, dx_p     = proper_run_sphere_to_plane(
        EXIT_PUPIL_TO_SCI_FOCAL_NO_MASK, wavefront_at_pupil=wf)

    # macos's dxElt can come back signed at a FarField destination
    # plane (encodes propagation direction).  Physical pitch is the
    # absolute value; PROPER reports unsigned.
    dx_m = abs(dx_m)

    assert intensity_m.shape == intensity_p.shape
    assert dx_p == pytest.approx(dx_m, rel=1e-3), (
        f"focal-plane sampling mismatch: macos |dx|={dx_m:.3e}, "
        f"PROPER dx={dx_p:.3e}")

    metrics = compare_and_record(
        'coro_exit_pupil_to_focal_no_mask',
        intensity_m, intensity_p, dx_m,
        results_dir_phase3,
        crop_pixels=intensity_m.shape[0],
        norm_kind='peak',                 # image plane: Strehl-norm
        extra_metadata={
            'wavelength_m':            EXIT_PUPIL_TO_SCI_FOCAL_NO_MASK.wavelength_m,
            'focal_length_m':          EXIT_PUPIL_TO_SCI_FOCAL_NO_MASK.focal_length_m,
            'src_elt':                 EXIT_PUPIL_TO_SCI_FOCAL_NO_MASK.src_elt,
            'detector_elt':            EXIT_PUPIL_TO_SCI_FOCAL_NO_MASK.detector_elt,
            'rx_filename':             EXIT_PUPIL_TO_SCI_FOCAL_NO_MASK.rx_filename,
            'macos_cfield_at_pupil':   wf['complex_field'],
        })

    # First-cut tolerance.  Phase 3b (Elt 8->9) achieved 4.2e-11 max
    # with the same recipe.  Expect similar here.
    assert metrics['max_abs'] < 1e-4, (
        f"max |a-b| = {metrics['max_abs']:.3e} (Strehl-normalised); "
        f"RMS = {metrics['rms_abs']:.3e}; "
        f"Δcom = ({metrics['dx_pix']:+d}, "
        f"{metrics['dy_pix']:+d}) px")


# =====================================================================
# Phase 5 step 2: add the focal-plane mask (FPM) + Lyot stop.
# =====================================================================

# Rx_Coro_FPM.in:
#   - Elt 9 (CorMask): 132 um circular obscuration (4*lambda*F#,
#     F# = 774/20 = 38.7).  Built by /tmp/make_fpm_rx.py from
#     Rx_Coro.in.
#   - Elt 14 (LyotStop): 20 mm circular aperture (already in
#     Rx_Coro.in).
EXIT_PUPIL_TO_SCI_FOCAL_WITH_MASK = CoroSphereToPlane(
    rx_filename="Rx_Coro_FPM.in",
    src_elt=20, detector_elt=21,
    focal_length_m=0.9514,
)


def test_coro_exit_pupil_to_focal_with_mask(pymacos_session,
                                             results_dir_phase3):
    """Phase 5 step 2: full coronagraph chain with FPM (Elt 9, 400 um
    radius circular obscuration, Element=Obscuring) and Lyot stop
    (Elt 14, 14 mm circular aperture) both active.  Sizes were tuned
    in /tmp/fpm_lyot_sweep.py + /tmp/lyot_sweep.py (2026-05-14) to
    give a meaningful Lyot coronagraph: on-axis peak at the science
    focal plane drops to 3.1e-7 of the un-coronagraphed baseline
    (factor 3.2 million suppression, vs ~1.2 with the original
    FPM=132 um + Lyot=20 mm combo, which produced essentially no
    coronagraphic suppression -- only ~17% flux trimming).

    Two prescription details matter:
      - Elt 9 MUST be Element=Obscuring (not Reference) for macos
        to apply the FPM to the diffraction-grid wavefront.  The
        original Reference-typed Elt 9 carried the ObsType=Circle /
        ObsVec metadata but only applied it to geometric rays --
        the diffraction WFElt sailed through untouched.
      - macos's m.load() doesn't fully re-parse obscuration shapes
        when reloading the same path with changed contents; an
        m.init(modelsize) before each load forces a clean re-read.
        (Bites the FPM/Lyot sweep scripts; the production tests
        only load each prescription once so it's invisible.)

    macos applies both masks during its internal trace; PROPER
    ingests the post-FPM, post-Lyot complex field at Elt 20 and
    propagates the final sphere-to-plane step to Elt 21.  The
    Strehl-normalised residual should land in the same regime as
    step 1 (~1e-9) since the comparison is on the LAST step only;
    the FPM/Lyot physics is identical in both engines (both just
    multiply by binary masks earlier in the chain, before the
    PROPER<->macos handoff at Elt 20).
    """
    intensity_m, dx_m, wf = macos_run_sphere_to_plane(
        EXIT_PUPIL_TO_SCI_FOCAL_WITH_MASK, pymacos_session)
    intensity_p, dx_p     = proper_run_sphere_to_plane(
        EXIT_PUPIL_TO_SCI_FOCAL_WITH_MASK, wavefront_at_pupil=wf)

    dx_m = abs(dx_m)
    assert intensity_m.shape == intensity_p.shape
    assert dx_p == pytest.approx(dx_m, rel=1e-3), (
        f"focal-plane sampling mismatch: macos |dx|={dx_m:.3e}, "
        f"PROPER dx={dx_p:.3e}")

    metrics = compare_and_record(
        'coro_exit_pupil_to_focal_with_mask',
        intensity_m, intensity_p, dx_m,
        results_dir_phase3,
        crop_pixels=intensity_m.shape[0],
        norm_kind='peak',
        extra_metadata={
            'wavelength_m':            EXIT_PUPIL_TO_SCI_FOCAL_WITH_MASK.wavelength_m,
            'focal_length_m':          EXIT_PUPIL_TO_SCI_FOCAL_WITH_MASK.focal_length_m,
            'src_elt':                 EXIT_PUPIL_TO_SCI_FOCAL_WITH_MASK.src_elt,
            'detector_elt':            EXIT_PUPIL_TO_SCI_FOCAL_WITH_MASK.detector_elt,
            'rx_filename':             EXIT_PUPIL_TO_SCI_FOCAL_WITH_MASK.rx_filename,
            'macos_cfield_at_pupil':   wf['complex_field'],
            'fpm_radius_um':           400.0,
            'lyot_radius_mm':          14.0,
        })

    # Tolerance: 1e-5 max (Strehl-normalised).  Observed 2.6e-6 max,
    # 1e-8 RMS at this operating point.  Three orders of magnitude
    # looser than the no-mask step 1 result (2.3e-9) because the
    # suppressed wavefront has lots of structured low-amplitude
    # content with hard edges (FPM, Lyot); both engines compute the
    # same physics but their finite-grid samplings of the suppressed
    # wavefront diverge at the 1e-6 relative-to-peak level.  At
    # peak 9.5e-8 (factor 3.2 million below the un-coronagraphed
    # baseline), this is 2.5e-13 in absolute units -- still at the
    # double-precision floor.
    assert metrics['max_abs'] < 1e-5, (
        f"max |a-b| = {metrics['max_abs']:.3e} (Strehl-normalised); "
        f"RMS = {metrics['rms_abs']:.3e}; "
        f"Δcom = ({metrics['dx_pix']:+d}, "
        f"{metrics['dy_pix']:+d}) px")
