"""
Single-source-of-truth geometry: circular unobstructed pupil at the
entrance, single ideal lens, focal plane.

Both PROPER and macos render their version of this problem from the
constants here, so the comparison test cannot drift apart on
geometry parameters.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CircularPupilFocus:
    """One representative on-axis monochromatic problem."""
    wavelength_m: float = 633e-9        # HeNe
    pupil_diameter_m: float = 0.010     # 10 mm
    focal_length_m: float = 0.500       # 500 mm
    grid_n: int = 256                   # square sampling per side
    grid_size_m: float = 0.020          # physical extent at pupil plane

    @property
    def f_number(self) -> float:
        return self.focal_length_m / self.pupil_diameter_m

    @property
    def airy_first_zero_m(self) -> float:
        return 1.22 * self.wavelength_m * self.f_number


DEFAULT = CircularPupilFocus()


def proper_run(geom: CircularPupilFocus):
    """Drive PROPER directly (bypassing prop_run/IDL prescription
    dispatch, which expects a module name string). Returns
    (intensity, sampling_m_per_pixel).
    """
    import proper

    beam_diam = geom.pupil_diameter_m
    # beam_ratio: pupil diameter relative to the grid extent
    beam_ratio = beam_diam / geom.grid_size_m
    wfo = proper.prop_begin(beam_diam, geom.wavelength_m,
                            geom.grid_n, beam_ratio)
    proper.prop_circular_aperture(wfo, beam_diam / 2.0)
    proper.prop_define_entrance(wfo)
    proper.prop_lens(wfo, geom.focal_length_m)
    proper.prop_propagate(wfo, geom.focal_length_m)
    psf_field, sampling = proper.prop_end(wfo)

    intensity = (abs(psf_field) ** 2 if psf_field.dtype.kind == "c"
                 else psf_field)
    return intensity, sampling


def macos_rx_text(geom: CircularPupilFocus) -> str:
    """Generate a minimal macos .in prescription for this geometry.

    Returned as a string so the test can write it to session_dir's
    temp .in file. NOTE: this is a stub — needs to be filled in with
    a valid Source + Reflector/Lens + FocalPlane block once the test
    is wired up against pymacos's load() expectations.
    """
    raise NotImplementedError(
        "macos_rx_text(): fill in once the source/lens/detector "
        "prescription convention is settled."
    )
