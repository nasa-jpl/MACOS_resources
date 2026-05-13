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
    dx_m:             float = 3.3382e-4  # 0.33382 mm pixel pitch at both
                                         # ends of the NF prop step
    propagation_m:    float = 0.774      # 774 mm Elt 2 -> Elt 3

    @property
    def grid_extent_m(self) -> float:
        return self.macos_size * self.dx_m  # 0.342 m

    @property
    def proper_beam_ratio(self) -> float:
        """beam_ratio = beam_diam / grid_extent.  With beam_diam ==
        grid_extent the PROPER grid pitch equals macos's dx_m exactly,
        so no resampling is needed for the OPD or amplitude transfer.
        """
        return 1.0


DEFAULT = CoroNFprop()


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

    amplitude_at_2 = np.sqrt(np.clip(intensity_at_2, 0, None))
    return (intensity_at_3, geom.dx_m,
            dict(amplitude=amplitude_at_2,
                 complex_field=cfield_at_2,
                 opd=opd_at_2))


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
    wfo = proper.prop_begin(geom.grid_extent_m, geom.wavelength_m,
                            N, geom.proper_beam_ratio)

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
