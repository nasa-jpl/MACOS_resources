"""
Coro NF plane-to-plane propagation geometry (Phase 2).

Scope: isolate the near-field plane-to-plane propagator between
Elt 2 (Prop_1_start) and Elt 3 (Prop_1_end) of Rx_Coro.in -- the
774 mm Fresnel step right after OAP_1.  Skip the source-to-OAP_1
modelling entirely by feeding the SAME wavefront into both engines
at Elt 2 (macos's amplitude + OPD), then comparing PROPER's
propagated intensity to macos's INT at Elt 3.

This isolates the NF plane-to-plane physics without confounding
issues from:
  - the divergent point-source geometry (zSource = -100 mm),
  - source-aperture sub-pixel sampling (0.22 mm aperture inside a
    342 mm grid extent),
  - OAP_1 thin-lens vs parabola approximation,
  - any preceding propagation step.

Macos's interactive INT 3 diagnostic on this prescription reports:
  Wavelength = 8.50e-04 mm  (850 nm)
  z1 = -774 mm   dx1 = 3.3382e-01 mm   (Elt 2 plane)
  z2 =    0 mm   dx2 = 3.3382e-01 mm   (Elt 3 plane)
The two dx values are equal: NF plane-to-plane preserves the grid
pitch (angular-spectrum Fresnel), so PROPER's natural grid setup
(beam_diam == grid_extent) matches without resampling.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CoroNFprop:
    rx_filename:      str   = "Rx_Coro.in"
    src_elt:          int   = 2          # Prop_1_start (NF prop start)
    detector_elt:     int   = 3          # Prop_1_end   (NF prop end)
    macos_size:       int   = 1024       # nGridpts in the prescription
    wavelength_m:     float = 8.5e-7     # 850 nm (= 8.5e-4 mm)
    propagation_m:    float = 0.774      # 774 mm Elt 2 -> Elt 3
    # dx is queried at runtime via pymacos.dx_at(src_elt) -- macos's
    # dxElt(iElt) carries the full double-precision value, removing the
    # 5-sig-fig display truncation that bit Phase 2/3a earlier.  macos
    # uses "samples at corners" (dx = extent/(N-1)), but for PROPER's
    # grid setup we feed N * dx as the beam_diam -- PROPER's pitch =
    # beam_diam/N = dx, matching macos bit-for-bit.


DEFAULT = CoroNFprop()


@dataclass(frozen=True)
class CoroSphereToPlane:
    """NF1/NF2 spherical-to-plane propagation (Rx_Coro Elt 8 -> Elt 9).

    macos's NF1/NF2 is mathematically the same as the Cass FF case:
    a propagation from a spherical reference surface (KrElt=-774 at
    Elt 8) to a plane (Elt 9), exactly what PROPER's
    prop_lens(f) + prop_propagate(f) recipe is designed for.  Use
    the Phase 1 template (mask-matched amplitude + sign-flipped OPD
    + prop_lens + prop_propagate) but parameterized for this step.

    Output sampling: macos's NF1/NF2 rebins to dx2 = 1.928e-6 m at
    Elt 9 (170x finer than the Elt-8 pupil sampling).  PROPER's FF
    propagator does the same rebinning: focal-plane dx = lambda *
    f_eff / grid_extent = 8.5e-7 * 0.774 / 0.341 = 1.929e-6 m.
    Matches macos by construction (the macos diffraction-grid extent
    IS what determines the FF focal sampling on both sides).
    """
    rx_filename:        str   = "Rx_Coro.in"
    src_elt:            int   = 8          # 1stPropStart (spherical ref)
    detector_elt:       int   = 9          # CorMask (plane)
    macos_size:         int   = 1024
    wavelength_m:       float = 8.5e-7     # 850 nm
    focal_length_m:     float = 0.774      # |KrElt| at Elt 8
    # dx_pupil and dx_focal queried at runtime via pymacos.dx_at()
    # at src_elt / detector_elt; see CoroNFprop note on precision.


DEFAULT_SPHERE_TO_PLANE = CoroSphereToPlane()


def macos_run_sphere_to_plane(geom: CoroSphereToPlane = DEFAULT_SPHERE_TO_PLANE,
                              pymacos_session=None):
    """Drive macos for the spherical-to-plane step.

    Returns (intensity_at_detector, dx_focal_m, dict with amplitude
    and complex_field at the spherical reference, plus dx_pupil_m /
    dx_focal_m queried at runtime from macos for PROPER to consume).
    """
    if pymacos_session is None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
        import pymacos.macos as pymacos_session

    import numpy as np
    rx_path = (Path(__file__).resolve().parents[2]
               / "Rx" / geom.rx_filename)
    pymacos_session.init(geom.macos_size)
    pymacos_session.load(str(rx_path))

    cfield_at_pupil = pymacos_session.complex_field(geom.src_elt)
    intensity_pupil = pymacos_session.intensity(geom.src_elt)
    intensity_focal = pymacos_session.intensity(geom.detector_elt)
    amplitude_pupil = np.sqrt(np.clip(intensity_pupil, 0, None))

    dx_pupil_m = pymacos_session.dx_at(geom.src_elt)
    dx_focal_m = pymacos_session.dx_at(geom.detector_elt)

    return (intensity_focal, dx_focal_m,
            dict(complex_field=cfield_at_pupil,
                 amplitude=amplitude_pupil,
                 dx_pupil_m=dx_pupil_m,
                 dx_focal_m=dx_focal_m))


def proper_run_sphere_to_plane(geom: CoroSphereToPlane = DEFAULT_SPHERE_TO_PLANE,
                               wavefront_at_pupil=None,
                               opd_sign_flip: bool = True,
                               use_cfield_phase: bool = True):
    """Drive PROPER for the spherical-to-plane step, Phase 1-style.

    Args:
      wavefront_at_pupil: dict with 'complex_field' and 'amplitude'
        from macos_run_sphere_to_plane().
      opd_sign_flip: same sign reconciliation as Phase 1 / Phase 2 v2.
      use_cfield_phase: if True (default), pass macos's cfield phase
        as PROPER's OPD (the residual phase content beyond the
        spherical reference, since cfield at a spherical-reference
        element is referenced to that sphere).  Then prop_lens
        re-applies the convergence.  If macos's cfield turns out to
        contain the convergence too (rather than being in the reference
        frame), set False to use amplitude only and let prop_lens
        supply the full convergence.
    """
    if wavefront_at_pupil is None:
        raise ValueError("macos wavefront at the pupil is required")

    import numpy as np
    import proper

    N = geom.macos_size
    dx_pupil_m   = wavefront_at_pupil['dx_pupil_m']
    grid_extent  = N * dx_pupil_m  # match macos's pitch bit-for-bit
    wfo = proper.prop_begin(grid_extent, geom.wavelength_m, N, 1.0)

    cfield = np.asarray(wavefront_at_pupil['complex_field'],
                        dtype=np.complex128)
    proper.prop_multiply(wfo, np.abs(cfield))

    if use_cfield_phase:
        opd = np.angle(cfield) * geom.wavelength_m / (2.0 * np.pi)
        if opd_sign_flip:
            opd = -opd
        proper.prop_add_phase(wfo, opd)

    proper.prop_define_entrance(wfo)
    # prop_lens supplies the convergence; prop_propagate uses
    # PROPER's Sphere-to-plane FF kernel (same as Cass FF Phase 1).
    proper.prop_lens(wfo, geom.focal_length_m)
    proper.prop_propagate(wfo, geom.focal_length_m)
    field, sampling = proper.prop_end(wfo)
    intensity = abs(field) ** 2 if field.dtype.kind == "c" else field
    return intensity, sampling


@dataclass(frozen=True)
class CoroPupilToPupilThruFocus:
    """Phase 4a: chain through the coronagraph focal plane to the
    far-side pupil reference.

    macos's Elt 10 (1stPropEnd) is the conjugate of Elt 8: another
    spherical reference of radius 774 mm, on the divergent side of
    the focus.  macos reports dx ~ 0.333 mm/pix at Elt 10 -- back
    at the pupil-like sampling, having rebinned twice (8->9 via
    Siegman-Sziklas to the fine focal plane, then 9->10 back to the
    coarse pupil).

    PROPER's equivalent: prop_lens(f) + prop_propagate(f) lands us
    at the focus (Elt 9); a further prop_propagate(f) carries the
    beam past focus to Elt 10.  PROPER's outside-beam Fresnel kernel
    auto-rebins the sampling back to the pupil-scale grid.
    """
    rx_filename:    str   = "Rx_Coro.in"
    src_elt:        int   = 8       # 1stPropStart (sphere)
    focus_elt:      int   = 9       # CorMask (plane)
    detector_elt:   int   = 10      # 1stPropEnd  (sphere, far side)
    macos_size:     int   = 1024
    wavelength_m:   float = 8.5e-7
    focal_length_m: float = 0.774
    # dx_pupil and dx_focal queried at runtime via pymacos.dx_at().


DEFAULT_PUPIL_TO_PUPIL = CoroPupilToPupilThruFocus()


def macos_run_pupil_to_pupil(geom: CoroPupilToPupilThruFocus
                              = DEFAULT_PUPIL_TO_PUPIL,
                              pymacos_session=None):
    """Drive macos through Elt 8 -> 9 -> 10.  Returns (intensity_at_10,
    dx_m_at_10, dict with cfield_at_pupil for PROPER to ingest).
    """
    if pymacos_session is None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
        import pymacos.macos as pymacos_session

    import numpy as np
    rx_path = (Path(__file__).resolve().parents[2]
               / "Rx" / geom.rx_filename)
    pymacos_session.init(geom.macos_size)
    pymacos_session.load(str(rx_path))

    cfield_at_pupil   = pymacos_session.complex_field(geom.src_elt)
    intensity_focus   = pymacos_session.intensity(geom.focus_elt)
    intensity_at_10   = pymacos_session.intensity(geom.detector_elt)

    dx_pupil_m = pymacos_session.dx_at(geom.src_elt)
    dx_focal_m = pymacos_session.dx_at(geom.focus_elt)
    dx_at_10_m = pymacos_session.dx_at(geom.detector_elt)

    return (intensity_at_10, dx_at_10_m,
            dict(complex_field=cfield_at_pupil,
                 intensity_at_focus=intensity_focus,
                 dx_pupil_m=dx_pupil_m,
                 dx_focal_m=dx_focal_m,
                 dx_at_10_m=dx_at_10_m))


def proper_run_pupil_to_pupil(geom: CoroPupilToPupilThruFocus
                               = DEFAULT_PUPIL_TO_PUPIL,
                               wavefront_at_pupil=None,
                               opd_sign_flip: bool = True):
    """Drive PROPER through Elt 8 -> 9 -> 10.  Same setup as the
    sphere-to-plane case for the first leg, then a second
    prop_propagate(f) past the focus.
    """
    if wavefront_at_pupil is None:
        raise ValueError("macos wavefront at the pupil is required")

    import numpy as np
    import proper

    N  = geom.macos_size
    dx_pupil_m  = wavefront_at_pupil['dx_pupil_m']
    grid_extent = N * dx_pupil_m
    wfo = proper.prop_begin(grid_extent, geom.wavelength_m, N, 1.0)

    cfield = np.asarray(wavefront_at_pupil['complex_field'],
                        dtype=np.complex128)
    proper.prop_multiply(wfo, np.abs(cfield))
    opd = np.angle(cfield) * geom.wavelength_m / (2.0 * np.pi)
    if opd_sign_flip:
        opd = -opd
    proper.prop_add_phase(wfo, opd)

    proper.prop_define_entrance(wfo)
    proper.prop_lens(wfo, geom.focal_length_m)
    proper.prop_propagate(wfo, geom.focal_length_m)  # to Elt 9 (focus)
    proper.prop_propagate(wfo, geom.focal_length_m)  # to Elt 10

    field, sampling = proper.prop_end(wfo)
    intensity = abs(field) ** 2 if field.dtype.kind == "c" else field
    return intensity, sampling


# ---------------------------------------------------------------------
# macos side
# ---------------------------------------------------------------------
def macos_run(geom: CoroNFprop = DEFAULT, pymacos_session=None):
    """Drive macos via pymacos for the Coro NF prop step.

    Returns:
      (intensity_at_elt3, dx_m, wavefront_at_elt2)
        wavefront_at_elt2 is a dict with keys
          'amplitude' (sqrt of macos intensity at Elt 2, sign chosen
                       to be the positive square root -- the wavefront
                       there is post-aperture / pre-mask and the real
                       amplitude is non-negative)
          'opd'       (macos OPD at Elt 2, metres, sign convention as
                       returned by pymacos.opd())
    """
    if pymacos_session is None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve()
                                .parents[3] / "src"))
        import pymacos.macos as pymacos_session

    import numpy as np

    rx_path = (Path(__file__).resolve().parents[2]
               / "Rx" / geom.rx_filename)
    # Each macos_run owns its model-size init (cheap when unchanged).
    pymacos_session.init(geom.macos_size)
    pymacos_session.load(str(rx_path))

    intensity_at_2 = pymacos_session.intensity(geom.src_elt)

    # Diffraction-grid complex field at Elt 2 (Phase 3 wrapper):
    # WFElt(:,:, iEltToiWF(2)) as np.complex128.  This is the
    # quantity macos's own propagation routines operate on, so it's
    # the right thing to hand off to PROPER for a faithful Phase 2 v2
    # NF-prop comparison.  Lives on the SAME grid as intensity (1024
    # x 1024 at 0.334 mm pix) -- no resampling.
    cfield_at_2 = pymacos_session.complex_field(geom.src_elt)

    # Source-ray-grid OPD (legacy / for archive) -- kept for the .mat
    # but no longer the primary phase source for PROPER.
    pymacos_session.trace_rays(geom.src_elt)
    opd_at_2 = pymacos_session.opd().copy()

    intensity_at_3 = pymacos_session.intensity(geom.detector_elt)

    dx_at_2 = pymacos_session.dx_at(geom.src_elt)
    dx_at_3 = pymacos_session.dx_at(geom.detector_elt)

    amplitude_at_2 = np.sqrt(np.clip(intensity_at_2, 0, None))
    return (intensity_at_3, dx_at_3,
            dict(amplitude=amplitude_at_2,
                 complex_field=cfield_at_2,
                 opd=opd_at_2,
                 dx_at_src_m=dx_at_2,
                 dx_at_det_m=dx_at_3))


# ---------------------------------------------------------------------
# PROPER side
# ---------------------------------------------------------------------
def proper_run(geom: CoroNFprop = DEFAULT, wavefront_at_elt2=None,
               opd_sign_flip: bool = True):
    """Drive PROPER for the same NF prop step.

    Args:
      wavefront_at_elt2: dict from macos_run with 'amplitude' and
        'opd' arrays.  Both are (macos_size, macos_size) at dx_m.
      opd_sign_flip: same convention reconciliation discovered in
        Phase 1 -- macos OPD's positive sign is opposite to PROPER's
        prop_add_phase input.

    Returns:
      (intensity_at_elt3_proper, sampling_m)
    """
    if wavefront_at_elt2 is None:
        raise ValueError(
            "PROPER side needs macos's wavefront at Elt 2; "
            "call macos_run() first and pass its third return.")

    import numpy as np
    import proper

    N = geom.macos_size
    # beam_diam = grid_extent -> PROPER dx = macos dx exactly.
    dx_at_src   = wavefront_at_elt2['dx_at_src_m']
    grid_extent = N * dx_at_src
    wfo = proper.prop_begin(grid_extent, geom.wavelength_m, N, 1.0)

    # Preferred path (Phase 2 v2): use macos's diffraction-grid
    # complex field at Elt 2 -- both amplitude and phase on the
    # matching 1024 x 1024 / 0.334 mm grid.
    cfield = wavefront_at_elt2.get('complex_field', None)
    if cfield is not None:
        cfield = np.asarray(cfield, dtype=np.complex128)
        amp = np.abs(cfield)
        # Phase in radians; convert to OPD in metres for prop_add_phase.
        # exp(+i phi) macos vs PROPER convention: same Phase 1 sign
        # flip applies (the OPD-extracted-from-cfield is positive when
        # macos's wavefront is delayed, opposite to PROPER's).
        phase_rad  = np.angle(cfield)
        opd_metres = phase_rad * geom.wavelength_m / (2.0 * np.pi)
        if opd_sign_flip:
            opd_metres = -opd_metres
        proper.prop_multiply(wfo, amp)
        proper.prop_add_phase(wfo, opd_metres)
    else:
        # Fallback (Phase 2 v1): amplitude-only from sqrt(intensity);
        # source-grid OPD is captured but not used (grid mismatch).
        amp = np.asarray(wavefront_at_elt2['amplitude'], dtype=float)
        proper.prop_multiply(wfo, amp)
        opd = wavefront_at_elt2.get('opd', None)
        if opd is not None and opd.shape == amp.shape:
            opd = np.asarray(opd, dtype=float)
            if opd_sign_flip:
                opd = -opd
            proper.prop_add_phase(wfo, opd)

    proper.prop_define_entrance(wfo)

    # Plane-to-plane Fresnel propagation.  PROPER selects the
    # propagator (angular-spectrum vs convolution Fresnel) based on
    # the geometry; for a near-field step inside the beam this is
    # angular-spectrum.
    proper.prop_propagate(wfo, geom.propagation_m)

    field, sampling = proper.prop_end(wfo)
    intensity = abs(field) ** 2 if field.dtype.kind == "c" else field
    return intensity, sampling
