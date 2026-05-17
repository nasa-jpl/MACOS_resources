"""System-state representation + applier + runner.

Data model (Dave, 2026-05-16):

  - For each REAL OPTIC i (Reflector elements; not References, Returns,
    or Obscuring), an alignment state vector
        x_i = [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
    with rotations in radians and translations in SI metres.

  - The SYSTEM alignment state is the concatenation
        x = [x_1; x_2; ...; x_N]
    a flat vector of length 6 * N_real_optics.

  - For each optic that carries Zernike figure errors, a figure
    state vector
        z_i = [z_m1; z_m2; ...; z_mk]
    of k Zernike coefficients (SI metres of surface RMS), indexed by
    the mode numbers declared in the prescription.

  - The SYSTEM figure state is the concatenation
        z = [z_1; z_2; ...; z_M]
    of length sum_i(k_i).

A ``SystemLayout`` describes which elements have which DOFs (i.e.,
sets up the indexing).  A ``SystemState`` carries the layout plus the
two flat vectors ``x`` and ``z``.  A nominal state is
``SystemState(layout)`` with both vectors zero.  Real users (control
loops, Monte-Carlo, sensitivity studies) operate directly on ``x``
and ``z``.

Apply via ``apply_system_state(session, state)`` -- it walks the
vectors and dispatches per-element pymacos calls:
  - alignment DOFs -> pymacos.perturb (which calls macos's
    CPERTURB_PROG with the full Mon/FF/pData frame bookkeeping)
  - figure DOFs    -> pymacos.elt_zrn_coef

Run via ``run_chain_with_state(state, geom, session)`` -- the
stand-alone callable that real users will adapt for their workflow.
"""
from __future__ import absolute_import

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# pymacos.elt_zrn_type integer codes (per its docstring).
_ZERNIKE_TYPE_CODES = {
    "ANSI":            1,
    "BornWolf":        2,
    "Fringe":          3,
    "NormANSI":        4,
    "NormBornWolf":    5,
    "NormFringe":      6,
    "NormHex":         7,
    "NormNoll":        8,
    "NormAnnularNoll": 9,
    "Noll":            10,
    "ExtFringe":       11,
}

# Order of DOF labels within an xelt 6-vector.
DOF_LABELS = ("rotx", "roty", "rotz", "transx", "transy", "transz")
DOF_INDEX  = {label: i for i, label in enumerate(DOF_LABELS)}


# ----------------------------------------------------------------------
# Layout: which elements have which DOFs
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SystemLayout:
    """Describes the DOF indexing for a particular prescription.

    Args:
        real_optics: tuple of element indices that carry alignment DOFs
            (the "real optics" -- reflectors / refractors; not
            Reference, Return, Obscuring, or FocalPlane elements).
            Each contributes 6 entries to the x vector.
        zernike_optics: mapping from element index to a tuple of
            Zernike mode indices that element supports.  Each mode
            contributes 1 entry to the z vector.  Concatenation order
            is sorted by element index, then mode-tuple order.
        zernike_type: macos Zernike basis label (one of the keys in
            ``_ZERNIKE_TYPE_CODES``).  Same for every Zernike optic
            for simplicity; can be split later if needed.
    """
    real_optics:    Tuple[int, ...] = ()
    zernike_optics: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    zernike_type:   str = "BornWolf"

    @property
    def n_alignment_dofs(self) -> int:
        return 6 * len(self.real_optics)

    @property
    def n_zernike_dofs(self) -> int:
        return sum(len(modes) for modes in self.zernike_optics.values())

    def _z_offset(self, element: int) -> Tuple[int, int]:
        """Return (start, stop) into z for one Zernike optic's coefs.
        Raises ``KeyError`` if ``element`` has no Zernike DOFs.
        """
        ofs = 0
        for e in sorted(self.zernike_optics):
            n = len(self.zernike_optics[e])
            if e == element:
                return ofs, ofs + n
            ofs += n
        raise KeyError(f"element {element} not in zernike_optics")


# ----------------------------------------------------------------------
# State: layout + flat x and z vectors
# ----------------------------------------------------------------------

@dataclass
class SystemState:
    """Full alignment + figure state.

    Mutable so real users can hand the same SystemState to a loop and
    keep updating ``x`` / ``z``.  Use ``set_dof`` / ``set_zernike``
    for index-safe writes, or write into the underlying vectors
    directly.
    """
    layout: SystemLayout
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    name: str = "unnamed"

    def __post_init__(self):
        expected_x = self.layout.n_alignment_dofs
        expected_z = self.layout.n_zernike_dofs
        # If the caller passed an empty array, allocate now.
        if self.x is None or np.asarray(self.x).size == 0:
            self.x = np.zeros(expected_x)
        else:
            self.x = np.asarray(self.x, dtype=float).reshape(-1).copy()
            if self.x.size != expected_x:
                raise ValueError(
                    f"x size {self.x.size} != expected {expected_x} "
                    f"(6 * {len(self.layout.real_optics)} real optics)")
        if self.z is None or np.asarray(self.z).size == 0:
            self.z = np.zeros(expected_z)
        else:
            self.z = np.asarray(self.z, dtype=float).reshape(-1).copy()
            if self.z.size != expected_z:
                raise ValueError(
                    f"z size {self.z.size} != expected {expected_z}")

    def is_nominal(self) -> bool:
        return not (np.any(self.x) or np.any(self.z))

    # -- views ----------------------------------------------------
    def xelt(self, element: int) -> np.ndarray:
        """View of the 6-vector for one real optic. Mutating mutates x."""
        if element not in self.layout.real_optics:
            raise KeyError(f"element {element} not in real_optics")
        i = self.layout.real_optics.index(element)
        return self.x[6*i : 6*(i+1)]

    def zelt(self, element: int) -> np.ndarray:
        """View into z for one Zernike optic's coefs. Mutating mutates z."""
        lo, hi = self.layout._z_offset(element)
        return self.z[lo:hi]

    # -- setters --------------------------------------------------
    def set_dof(self, element: int, dof: str, value: float) -> "SystemState":
        """Set one alignment DOF on one element.  Returns self
        (for chaining)."""
        if element not in self.layout.real_optics:
            raise KeyError(f"element {element} not in real_optics")
        i = self.layout.real_optics.index(element)
        self.x[6*i + DOF_INDEX[dof]] = value
        return self

    def set_zernike(self, element: int, mode: int,
                    coef_m: float) -> "SystemState":
        """Set one Zernike coefficient (SI metres surface RMS) on one
        element.  Returns self (for chaining)."""
        modes = self.layout.zernike_optics.get(element)
        if modes is None:
            raise KeyError(f"element {element} not in zernike_optics")
        if mode not in modes:
            raise ValueError(
                f"mode {mode} not in layout's zernike_optics[{element}]="
                f"{modes}")
        lo, _ = self.layout._z_offset(element)
        self.z[lo + modes.index(mode)] = coef_m
        return self


# ----------------------------------------------------------------------
# Applier: walk the state vectors, dispatch pymacos setters
# ----------------------------------------------------------------------

def apply_system_state(session, state: SystemState) -> None:
    """Apply the alignment + figure DOFs in ``state`` to the loaded
    macos prescription.

    Alignment: each element's 6-vector x_i is split into rotation
        (rad) and translation (SI metres); applied via
        ``pymacos.perturb`` (which wraps macos's CPERTURB_PROG, doing
        the full Mon/FF/pData/aperture/linked-element bookkeeping).
    Figure: Zernike coefficients are set via ``pymacos.elt_zrn_coef``
        after a one-shot ``elt_zrn_type`` call to set the basis.

    Caller must have already loaded the Rx; the apply accumulates on
    the loaded state.  Revert by reloading the prescription.
    """
    # Alignment
    for elt in state.layout.real_optics:
        xelt = state.xelt(elt)
        rotation    = tuple(xelt[0:3])
        translation = tuple(xelt[3:6])
        if not (any(rotation) or any(translation)):
            continue
        session.perturb(elt, rotation_rad=rotation,
                        translation_m=translation,
                        in_local_coords=False)

    # Figure / Zernike
    if state.layout.n_zernike_dofs > 0 and np.any(state.z):
        type_code = _ZERNIKE_TYPE_CODES[state.layout.zernike_type]
        ok_cbm, cbm = session.lib.api.base_unit_to_metres()
        if not ok_cbm or cbm == 0.0:
            raise RuntimeError("apply_system_state: CBM unavailable")
        si_to_base = 1.0 / float(cbm)

        for elt in sorted(state.layout.zernike_optics):
            zelt = state.zelt(elt)
            if not np.any(zelt):
                continue
            session.elt_zrn_type(elt, type_code)
            modes = np.asarray(state.layout.zernike_optics[elt],
                                dtype=np.int32)
            coefs_base = np.asarray(zelt, dtype=float) * si_to_base
            session.elt_zrn_coef(elt, modes, coefs_base, reset=False)


# ----------------------------------------------------------------------
# Stand-alone runner: state -> macos + PROPER + scoring
# ----------------------------------------------------------------------

def run_chain_with_state(state: SystemState,
                          geom,
                          pymacos_session,
                          peak_unaberrated: Optional[float] = None,
                          lambda_over_D_px: Optional[float] = None,
                          ) -> dict:
    """Run macos + PROPER through ``geom`` with ``state`` applied,
    return joint intensities + metrics + radial-contrast curve.

    Args:
      state: SystemState (alignment + figure) to apply.
      geom:  Geometry dataclass (CoroSphereToPlane for the Phase 5/6
             chain to Elt 21).
      pymacos_session: a pymacos.macos module reference.
      peak_unaberrated: optional Strehl reference for the contrast.
             If None, uses macos's own peak at this state (good only
             when scoring the un-coronagraphed PSF).
      lambda_over_D_px: optional lambda/D in focal-plane pixels.  If
             None, derived empirically from the macos PSF's first
             Airy null.

    Returns a dict (see source for the full key list).
    """
    from .contrast import lambda_over_D_pixels, radial_contrast
    from .geometries.coro_nfprop import (
        macos_run_sphere_to_plane, proper_run_sphere_to_plane)

    def _hook(session):
        apply_system_state(session, state)

    I_m, dx_m, wf = macos_run_sphere_to_plane(geom, pymacos_session,
                                               post_load_hook=_hook)
    I_p, dx_p     = proper_run_sphere_to_plane(geom, wavefront_at_pupil=wf)

    a = I_m / I_m.max() if I_m.max() > 0 else I_m
    b = I_p / I_p.max() if I_p.max() > 0 else I_p
    d = np.abs(a - b)

    if lambda_over_D_px is None:
        lambda_over_D_px = float(lambda_over_D_pixels(I_m))
    if peak_unaberrated is None:
        peak_unaberrated = float(I_m.max())
    r_ld, c = radial_contrast(I_m, peak_unaberrated, lambda_over_D_px,
                              max_lambda_over_D=20.0)

    return dict(
        state=state,
        intensity_macos=I_m,
        intensity_proper=I_p,
        dx_macos=float(abs(dx_m)),
        cfield_at_pupil=wf['complex_field'],
        lambda_over_D_px=float(lambda_over_D_px),
        peak_unaberrated=float(peak_unaberrated),
        contrast_r_lambda_over_D=r_ld,
        contrast_values=c,
        agreement_max_abs=float(d.max()),
        agreement_rms_abs=float(np.sqrt(np.mean(d * d))),
    )


# ----------------------------------------------------------------------
# Standard layout for Rx_Coro_FPM_Zern.in (test prescription).
# ----------------------------------------------------------------------
#
# Real optics (Reflector elements in Rx_Coro.in):
#   Elt 1  : 1stOAP (M1, primary)
#   Elt 4  : DM1   (flat reflector, with Zernike schema)
#   Elt 7  : 2ndOAP
#   Elt 12 : 3rdOAP
#   Elt 15 : 4thOAP
#   Elt 17 : 5thOAP (with Zernike schema)
#   Elt 18 : 6thOAP
#
# Zernike-capable optics (Surface=Zernike in Rx_Coro_FPM_Zern.in):
#   Elt 4  (DM1)
#   Elt 17 (downstream optic; NOT a true DM2)
#
# Standard Zernike mode list: low-order astigmatism + trefoil + one
# high-order radial polynomial.  Extend as needed for sensitivity
# studies.

CORO_LAYOUT = SystemLayout(
    real_optics=(1, 4, 7, 12, 15, 17, 18),
    # Zernike modes by Noll index: Z5 (oblique astig), Z7 (vertical
    # coma), Z9 (vertical trefoil), Z33 (high-order radial).  Extend
    # as needed for new sensitivity studies -- adding modes here is
    # backward-compatible with existing tests (modes default to zero).
    zernike_optics={4: (5, 7, 9, 33), 17: (5, 7, 9, 33)},
    zernike_type="BornWolf",
)
