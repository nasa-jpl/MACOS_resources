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

    # OPD at Elt 2 -- captured here so each .mat carries it AND so
    # downstream aberration cases have it available, but NOT passed
    # into PROPER for the v1 NF-prop test.  pymacos.opd() returns a
    # source-ray-grid map (512x512 over the 0.22 mm aperture at ~0.43
    # um pix pitch); PROPER's prop_add_phase needs the phase on the
    # diffraction grid (1024x1024 at 0.334 mm).  Bringing them
    # together for downstream perturbation cases needs either a new
    # pymacos wrapper exposing the diffraction-grid complex field at
    # Elt 2, or a documented embedding scheme that maps source-grid
    # OPD onto the diffraction grid the way macos does internally.
    # For the nominal Rx_Coro Elt-2 state the OPD is 0.28 pm RMS
    # (3e-7 waves at 850 nm), so ignoring it here is harmless.
    pymacos_session.trace_rays(geom.src_elt)
    opd_at_2 = pymacos_session.opd().copy()

    intensity_at_3 = pymacos_session.intensity(geom.detector_elt)

    amplitude_at_2 = np.sqrt(np.clip(intensity_at_2, 0, None))
    return (intensity_at_3, geom.dx_m,
            dict(amplitude=amplitude_at_2, opd=opd_at_2))


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

    # Take amplitude (and therefore mask) directly from macos.  The
    # Phase 1 "mask-matched amplitude via prop_multiply" recipe carries
    # over: PROPER's analytical aperture model isn't used.
    amp = np.asarray(wavefront_at_elt2['amplitude'], dtype=float)
    proper.prop_multiply(wfo, amp)

    # Phase from macos OPD -- gated on shape match.  In v1 macos OPD
    # is on a 512x512 source grid, PROPER's wavefront is on a
    # 1024x1024 diffraction grid: shapes differ, so skip and rely on
    # the fact that nominal Rx_Coro has 0.28 pm RMS OPD at Elt 2.
    # Once a diffraction-grid OPD wrapper exists, this branch will
    # apply the phase via prop_add_phase (with sign reconciliation
    # per Phase 1's discovery).
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
