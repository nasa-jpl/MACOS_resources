"""Perturbation channels for sensitivity-matrix work.

A SensitivityChannel encapsulates one degree of freedom in the
perturbation state vector ``z``.  Each channel knows how to:

  - report its human-readable name (used as a column label),
  - apply a perturbation of magnitude ``value`` to the macos state,
  - restore the macos state to nominal.

A SensitivityChannel is responsible ONLY for the perturbation; the
wavefront extraction (the ``w`` side of ``dw/dz``) is separate and
defined by the calling script.  This lets the same engine compute
``dw/dz`` for Zernike-form coefficient perturbations today and
``dw/dx`` for rigid-body, ``dw/ddm`` for DM actuators,
``dw/dsource`` etc. tomorrow -- just add a new channel class.

Convention: ``apply(value)`` sets the absolute perturbation to
``value`` (i.e. NOT incremental); the channel remembers its nominal
state and writes ``nominal + value``.  ``restore()`` writes the
nominal back.  Symmetric finite differences are therefore
``apply(+delta) -> w_plus; apply(-delta) -> w_minus; restore()`` and
``J_col = (w_plus - w_minus) / (2 * delta)``.

ZernikeCoefChannel covers the three Zernike-form coefficient arrays
in macos:

  - ZernCoef on SrfType=Zernike (8) or ZrnGrData (13)
  - MonZernCoef on SrfType=FreeForm (14)     -- perturbation overlay
  - FFZernCoef  on SrfType=FreeForm (14)     -- figure description

Each variant is a thin subclass that names its target array, its
SrfType filter for ``findFreeFormElts``-like helpers, and the macos
getter / setter pair.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class SensitivityChannel:
    """Abstract one-DOF perturbation channel."""

    name: str  # column label

    def apply(self, value: float) -> None:
        raise NotImplementedError

    def restore(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Zernike-form coefficient channels (the three flavors in macos)
# ---------------------------------------------------------------------------

# Channel-kind metadata: name (for labels), pymacos getter / setter.
# All three setters / getters have the same signature
# ``f(iElt, modes, [coefs])`` -- the macos-side gating on SrfType is
# inside pymacos itself.
@dataclass(frozen=True)
class _ZernKind:
    label: str                       # e.g. "Zern", "MonZern", "FFZern"
    getter_name: str                 # macos module attr name
    setter_name: str                 # macos module attr name


_ZERN_KINDS: dict[str, _ZernKind] = {
    "Zern":    _ZernKind("Zern",    "getEltSrfZernMode",
                                     "setEltSrfZernCoef"),
    "MonZern": _ZernKind("MonZern", "getEltSrfMonZern",
                                     "setEltSrfMonZern"),
    "FFZern":  _ZernKind("FFZern",  "getEltSrfFFZern",
                                     "setEltSrfFFZern"),
}


@dataclass
class ZernikeCoefChannel(SensitivityChannel):
    """Perturbation of one Zernike-form coefficient on one element.

    ``kind`` picks which macos array the channel writes:
      - ``"Zern"``    : ZernCoef       (SrfType=Zernike or ZrnGrData)
      - ``"MonZern"`` : MonZernCoef    (SrfType=FreeForm overlay)
      - ``"FFZern"``  : FFZernCoef     (SrfType=FreeForm figure)

    macos-side type gating happens inside the pymacos setter/getter;
    this class is a thin dispatcher.
    """
    macos: object
    iElt: int
    mode: int
    kind: str          # "Zern" | "MonZern" | "FFZern"
    nominal: float

    def __post_init__(self) -> None:
        if self.kind not in _ZERN_KINDS:
            raise ValueError(
                f"ZernikeCoefChannel: unknown kind {self.kind!r}; "
                f"expected one of {list(_ZERN_KINDS)}")

    @property
    def name(self) -> str:  # type: ignore[override]
        # Format used as the plot title and the MATLAB cell-array entry
        # for this DOF; e.g. "Elt 4 MonZern15".
        return f"Elt {self.iElt} {_ZERN_KINDS[self.kind].label}{self.mode}"

    @property
    def _setter(self) -> Callable:
        return getattr(self.macos, _ZERN_KINDS[self.kind].setter_name)

    def apply(self, value: float) -> None:
        self._setter(self.iElt, [self.mode], [self.nominal + value])
        # MODIFY clears macos's "trace state is current" caches so the
        # next trace_rays() re-runs ZerntoMon and re-evaluates the
        # FreeForm / Zernike surface with the new coefficient.  Without
        # this, sensitivities silently come back as zero -- the trace
        # reuses the cached MonCoef/FFCoef/etc.
        self.macos.modify()

    def restore(self) -> None:
        self._setter(self.iElt, [self.mode], [self.nominal])
        self.macos.modify()


# Concrete shims so calling code can read intent at the call site.
def MonZernChannel(macos, iElt: int, mode: int) -> ZernikeCoefChannel:
    """Channel for MonZernCoef[mode] on FreeForm element ``iElt``."""
    nominal = float(macos.getEltSrfMonZern(iElt, [mode])[0])
    return ZernikeCoefChannel(macos=macos, iElt=iElt, mode=mode,
                              kind="MonZern", nominal=nominal)


def FFZernChannel(macos, iElt: int, mode: int) -> ZernikeCoefChannel:
    """Channel for FFZernCoef[mode] on FreeForm element ``iElt``."""
    nominal = float(macos.getEltSrfFFZern(iElt, [mode])[0])
    return ZernikeCoefChannel(macos=macos, iElt=iElt, mode=mode,
                              kind="FFZern", nominal=nominal)


def ZernChannel(macos, iElt: int, mode: int) -> ZernikeCoefChannel:
    """Channel for ZernCoef[mode] on a Zernike or ZrnGrData element."""
    nominal = float(macos.getEltSrfZernMode(iElt, [mode])[0])
    return ZernikeCoefChannel(macos=macos, iElt=iElt, mode=mode,
                              kind="Zern", nominal=nominal)


# ---------------------------------------------------------------------------
# Auto-discovery: build channels from a loaded Rx
# ---------------------------------------------------------------------------

def freeform_monzern_channels(macos,
                               rx_path: str,
                               elts: Iterable[int] | None = None,
                               modes_per_elt: dict[int, Sequence[int]] | None
                                   = None,
                               ) -> list[ZernikeCoefChannel]:
    """Discover FreeForm MonZernCoef channels from a loaded Rx.

    Loop bounds come from the Rx itself, NOT hardcoded:
      - Eligible elements: every FreeForm-typed surface (SrfType=14)
        unless restricted by ``elts``.
      - Per-element mode list: parsed from the Rx file
        (``nMonZernCoef`` / ``MonZernModes`` per element block).  If
        a per-element ``MonZernModes=`` line is absent, defaults to
        [1..nMonZernCoef] -- matching msmacosio.inc.
        Override with ``modes_per_elt={iElt: [modes]}``.
    """
    return _build_zern_channels(
        macos, rx_path, kind="MonZern",
        rx_n_key="nMonZernCoef", rx_modes_key="MonZernModes",
        eligibility=set(int(i) for i in macos.findFreeFormElts()),
        elts=elts, modes_per_elt=modes_per_elt)


def freeform_ffzern_channels(macos,
                              rx_path: str,
                              elts: Iterable[int] | None = None,
                              modes_per_elt: dict[int, Sequence[int]] | None
                                  = None,
                              ) -> list[ZernikeCoefChannel]:
    """Discover FreeForm FFZernCoef channels from a loaded Rx.

    Like :func:`freeform_monzern_channels` but for the FFZernCoef
    array (the FreeForm surface's *figure description*, distinct from
    the MonZernCoef perturbation overlay).
    """
    return _build_zern_channels(
        macos, rx_path, kind="FFZern",
        rx_n_key="nFFZernCoef", rx_modes_key="FFZernModes",
        eligibility=set(int(i) for i in macos.findFreeFormElts()),
        elts=elts, modes_per_elt=modes_per_elt)


def zernike_channels(macos,
                      rx_path: str,
                      elts: Iterable[int] | None = None,
                      modes_per_elt: dict[int, Sequence[int]] | None = None,
                      ) -> list[ZernikeCoefChannel]:
    """Discover ZernCoef channels from a loaded Rx (SrfType in {8, 13}).

    Like :func:`freeform_monzern_channels` but for the ZernCoef array
    on Zernike (SrfType=8) and ZrnGrData (SrfType=13) elements.
    """
    n = macos.num_elt()
    # No pymacos helper for "find Zernike-typed elements" yet; iterate.
    # The setter rejects wrong SrfType, so we filter by trying a zero
    # poke -- but cheaper to ask the Rx parser.  Use the Rx text as the
    # source of truth.
    eligibility = _parse_rx_zern_elts(rx_path)
    return _build_zern_channels(
        macos, rx_path, kind="Zern",
        rx_n_key="nZernCoef", rx_modes_key="ZernModes",
        eligibility=eligibility,
        elts=elts, modes_per_elt=modes_per_elt)


# ---------------------------------------------------------------------------
# Shared builder + Rx parsing helpers
# ---------------------------------------------------------------------------

def _build_zern_channels(macos, rx_path, *, kind, rx_n_key, rx_modes_key,
                          eligibility, elts, modes_per_elt):
    targets = eligibility
    if elts is not None:
        targets = targets & set(int(i) for i in elts)
    if not targets:
        return []

    rx_modes = _parse_rx_modes(rx_path, rx_n_key, rx_modes_key)

    factory = {
        "MonZern": MonZernChannel,
        "FFZern":  FFZernChannel,
        "Zern":    ZernChannel,
    }[kind]

    channels: list[ZernikeCoefChannel] = []
    for iElt in sorted(targets):
        if modes_per_elt is not None and iElt in modes_per_elt:
            modes = list(modes_per_elt[iElt])
        else:
            modes = list(rx_modes.get(iElt, [1]))
        if not modes:
            continue
        for mode in modes:
            channels.append(factory(macos, iElt, int(mode)))
    return channels


def _parse_rx_modes(rx_path: str,
                     n_key: str,
                     modes_key: str) -> dict[int, list[int]]:
    """Parse a macos .in file for per-element ``n_key=`` and
    ``modes_key=`` and return ``{iElt: [active_modes]}``.

    Mirrors msmacosio.inc semantics:
      - If ``modes_key=`` is present, use the explicit list.
      - If absent (but ``n_key=K`` is present), default to [1..K].
      - If both absent, the element is omitted from the result.
    Handles ``modes_key=`` continuation lines (groups of 6 bare ints
    after the first line, same as msmacosio.inc's Grp parameter).
    """
    out: dict[int, list[int]] = {}
    cur_elt: int | None = None
    n_active: int | None = None
    explicit_modes: list[int] | None = None
    pending_continuation: int = 0  # remaining ints to read

    def _flush():
        if cur_elt is None or n_active is None:
            return
        if explicit_modes is not None:
            out[cur_elt] = list(explicit_modes[:n_active])
        else:
            out[cur_elt] = list(range(1, n_active + 1))

    with open(rx_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if pending_continuation > 0:
            toks = ln.replace(",", " ").split()
            ints: list[int] = []
            for t in toks:
                try:
                    ints.append(int(float(t)))
                except ValueError:
                    break
            if ints:
                assert explicit_modes is not None
                take = min(len(ints), pending_continuation)
                explicit_modes.extend(ints[:take])
                pending_continuation -= take
            i += 1
            continue

        if s.startswith("iElt="):
            _flush()
            try:
                cur_elt = int(s.split("=", 1)[1].strip())
            except ValueError:
                cur_elt = None
            n_active = None
            explicit_modes = None
        elif s.startswith(n_key + "="):
            try:
                n_active = int(s.split("=", 1)[1].strip())
            except ValueError:
                n_active = None
            explicit_modes = None
        elif s.startswith(modes_key + "="):
            if n_active is None:
                # Out-of-order; skip silently and let macos error.
                i += 1
                continue
            payload = s.split("=", 1)[1]
            toks = payload.replace(",", " ").split()
            explicit_modes = []
            for t in toks:
                try:
                    explicit_modes.append(int(float(t)))
                except ValueError:
                    break
            remaining = n_active - len(explicit_modes)
            if remaining > 0:
                pending_continuation = remaining
        i += 1
    _flush()
    return out


def _parse_rx_zern_elts(rx_path: str) -> set[int]:
    """Scan an Rx for elements declared as Surface=Zernike OR
    Surface=ZrnGridData (= ZrnGrData, SrfType 13).  Returns the set
    of element IDs.  (No pymacos query for this exists yet; the Rx
    text is the source of truth.)
    """
    out: set[int] = set()
    cur_elt: int | None = None
    with open(rx_path) as f:
        for ln in f:
            s = ln.strip()
            if s.startswith("iElt="):
                try:
                    cur_elt = int(s.split("=", 1)[1].strip())
                except ValueError:
                    cur_elt = None
            elif s.startswith("Surface=") and cur_elt is not None:
                surf = s.split("=", 1)[1].strip().split()[0]
                if surf in ("Zernike", "ZrnGridData", "ZrnGrData"):
                    out.add(cur_elt)
    return out


# ---------------------------------------------------------------------------
# Rigid-body perturbation channels (CPERTURB_PROG on real optics)
# ---------------------------------------------------------------------------

# Element kinds that the Rx parser accepts but which are NOT "actual optics"
# in the sense of "things a sensitivity matrix should perturb": reference
# planes and return-type elements only carry their pose for the trace's
# bookkeeping, not for a real piece of glass / mirror / detector.
_NON_OPTIC_ELEMENT_KINDS = frozenset({"Reference", "Return"})

# DOF ordering of the per-optic 6-vector x.  Locked by the user spec:
# x rotation, y rotation, z rotation, x translation, y translation,
# z translation -- stacked as [Elt 1 x_6; Elt 2 x_6; ...].
_RB_DOF_LABELS: tuple[str, ...] = ("Rx", "Ry", "Rz", "Tx", "Ty", "Tz")


@dataclass
class RigidBodyChannel(SensitivityChannel):
    """One-DOF rigid-body perturbation on an actual optic.

    ``dof_idx`` in 0..5 selects (Rx, Ry, Rz, Tx, Ty, Tz).  Rotations are
    in radians, translations are passed in SI metres (matches the
    :func:`pymacos.macos.perturb` API; macos converts internally to
    BaseUnits via CBM).

    macos's ``CPERTURB_PROG`` is INCREMENTAL -- each call adds to the
    element's current pose.  To support the sensitivity engine's
    central-difference pattern ``apply(+d) -> measure -> apply(-d) ->
    measure -> restore()`` this class tracks its own cumulative state
    and passes ``(value - current)`` as the increment each call.
    Restore is just ``apply(0.0)``.

    Single-axis perturbations are first-order accurate regardless of
    rotation ordering, so the rotation-noncommutativity error is at
    O(delta^2) -- below numerical noise at the deltas used for
    sensitivity work (1e-9 rad / 1e-9 m).
    """
    macos: object
    iElt: int
    dof_idx: int
    current: float = 0.0  # accumulated perturbation in this DOF

    def __post_init__(self) -> None:
        if not 0 <= self.dof_idx <= 5:
            raise ValueError(
                f"RigidBodyChannel: dof_idx must be in 0..5, got "
                f"{self.dof_idx}")

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"Elt {self.iElt} {_RB_DOF_LABELS[self.dof_idx]}"

    @property
    def kind(self) -> str:
        # Match the ZernikeCoefChannel `kind` field so callers can
        # sort / filter heterogeneous channel lists uniformly.
        return "RigidBody"

    def _do_perturb(self, increment: float) -> None:
        rot = [0.0, 0.0, 0.0]
        trans = [0.0, 0.0, 0.0]
        if self.dof_idx < 3:
            rot[self.dof_idx] = increment
        else:
            trans[self.dof_idx - 3] = increment
        self.macos.perturb(self.iElt,
                           rotation_rad=tuple(rot),
                           translation_m=tuple(trans),
                           in_local_coords=True)
        # MODIFY clears the cached trace state so the next trace_rays()
        # picks up the new pose.  Same gotcha as the Zernike channels.
        self.macos.modify()

    def apply(self, value: float) -> None:
        increment = value - self.current
        if increment != 0.0:
            self._do_perturb(increment)
        self.current = value

    def restore(self) -> None:
        self.apply(0.0)


@dataclass
class FocalPlaneChannel(RigidBodyChannel):
    """Rigid-body perturbation on the focal-plane element.

    macos's standard trace does NOT propagate FP perturbations back
    to the EP-OPD measurement, because the EP definition's dependency
    on the FP position is implicit.  Without compensation, all six FP
    DOFs come out as zero sensitivities -- physically wrong: the EP
    is a sphere centered on the FP, so moving the FP must change the
    EP and thus the wavefront referenced to it (lateral FP shift ->
    tilt; along-axis shift -> defocus).

    Three compensation modes:

    - ``"track"`` (default) : perturb the FP AND the EP element by
      the same 6-vector, so the existing XP sphere is dragged with
      the moving FP.  Matches the physics that "the EP is a sphere
      centered on the FP".  Gives non-zero sensitivities for FP
      Tx/Ty (tilt), Tz (focus), and the rotations.  Doesn't need a
      Stop set.
    - ``"srs"`` : perturb the FP, then ``srs(ep_elt, FP_elt)`` --
      slave the EP to the moved FP via macos's SRS, recomputing the
      EP pose from the new chief-ray geometry.  More principled than
      track (which just drags the EP through the same 6-vector
      naively); the EP-FP geometric link is re-derived from the
      trace each call.  Needs a Stop set for the chief-ray trace
      that SRS rides on.
    - ``"fex"`` : perturb the FP, then ``fex()`` to recompute the
      XP at nElt-1.  Diagnostic-only on this Rx: macos's FEX
      computes the EP as the optical conjugate of the Stop, which
      is unchanged by an FP shift, so FEX returns the same EP and
      FP DOFs come out as zero sensitivities.  Kept because GMI's
      ``ifFEX`` flag uses the same call for general EP recomputation
      in multi-element perturbation loops -- might be useful in
      other workflows.

    Caveat: only meaningful when the wavefront is evaluated at the
    XP surface ``nElt-1`` -- a different wf_elt won't carry the
    FP-tracking EP geometry into the OPD measurement.
    """
    mode: str = "track"
    ep_elt: int = -1   # FP-mode target EP element; -1 -> nElt-1 (auto)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mode not in ("track", "srs", "sxp", "fex"):
            raise ValueError(
                f"FocalPlaneChannel: mode must be 'track', 'srs', "
                f"'sxp', or 'fex'; got {self.mode!r}")

    def _do_perturb(self, increment: float) -> None:
        # Build the 6-vector for this DOF.
        rot = [0.0, 0.0, 0.0]
        trans = [0.0, 0.0, 0.0]
        if self.dof_idx < 3:
            rot[self.dof_idx] = increment
        else:
            trans[self.dof_idx - 3] = increment
        rot_t = tuple(rot)
        trans_t = tuple(trans)

        # Always perturb the FP.
        self.macos.perturb(self.iElt,
                           rotation_rad=rot_t,
                           translation_m=trans_t,
                           in_local_coords=True)

        if self.mode == "track":
            # DOF-dependent EP move per Dave's spec:
            #   - lateral translations (Tx, Ty): EP follows FP by the
            #     same vector in their local frames (which are
            #     approximately aligned for an imager EP/FP pair)
            #   - axial translation (Tz): EP does NOT move axially --
            #     its location is determined by upstream chief-ray
            #     geometry, not by the FP's axial position; only the
            #     EP radius changes (handled by the SXP block below)
            #   - rotations (Rx, Ry, Rz): EP rotates rigidly about
            #     FP's RptElt, NOT about its own RptElt; computed
            #     directly via vpt/psi/rpt setters so the rotation
            #     center is correct (m.perturb always rotates about
            #     the perturbed element's own RptElt)
            ep = (self.ep_elt if self.ep_elt > 0
                  else self.macos.num_elt() - 1)

            if self.dof_idx == 3 or self.dof_idx == 4:
                # Lateral translation -- propagate to EP unchanged.
                self.macos.perturb(ep,
                                   rotation_rad=rot_t,
                                   translation_m=trans_t,
                                   in_local_coords=True)
            elif self.dof_idx == 5:
                # Axial translation -- EP vpt should NOT move; SXP
                # below will refine the radius.
                pass
            else:
                # Rotation DOF -- rotate EP rigidly about FP's RptElt.
                self._rotate_ep_about_fp_rpt(ep, increment)

        self.macos.modify()

        if self.mode == "track":
            # SXP-refine the EP radius without disturbing the track-
            # induced EP vpt/psi.  SXP overwrites VptElt/psiElt/RptElt
            # (it sets them from the chief-ray crossing point), which
            # would undo the lateral / rotational EP move -- so we
            # snapshot them, run SXP, then write them back.
            ep = (self.ep_elt if self.ep_elt > 0
                  else self.macos.num_elt() - 1)
            vpt_save = np.asarray(self.macos.elt_vpt(ep)).copy()
            psi_save = np.asarray(self.macos.elt_psi(ep)).copy()
            rpt_save = np.asarray(self.macos.elt_rpt(ep)).copy()
            self.macos.trace_rays(self.iElt)
            self.macos.sxp()
            self.macos.elt_vpt(ep, vpt_save)
            self.macos.elt_psi(ep, psi_save)
            self.macos.elt_rpt(ep, rpt_save)
            self.macos.modify()
        elif self.mode == "fex":
            # FEX needs a current chief-ray trace; run one to update.
            self.macos.trace_rays(self.iElt)
            # Rebuild the XP at nElt-1 centered on the (now-moved) FP.
            self.macos.fex()
        elif self.mode == "sxp":
            # SXP -- FEX variant with EP radius set to chief-ray
            # distance EP-to-FP (captures FP-Tz focus perturbations
            # that FEX misses).  See macos tracesub.F SUBROUTINE SXP.
            self.macos.trace_rays(self.iElt)
            self.macos.sxp()
        elif self.mode == "srs":
            # Slave the EP element to the moved FP -- macos's SRS
            # recomputes the EP pose from the chief ray traced through
            # the FP.  Needs a current trace first.
            ep = (self.ep_elt if self.ep_elt > 0
                  else self.macos.num_elt() - 1)
            self.macos.trace_rays(self.iElt)
            self.macos.srs(ep, self.iElt, link=True)

    def _rotate_ep_about_fp_rpt(self, ep_elt: int,
                                 increment: float) -> None:
        """Rigid-body rotate the EP element about the FP's RptElt by
        ``increment`` radians, about the rotation axis indexed by
        ``self.dof_idx`` in the FP's local frame.

        m.perturb rotates an element about its OWN RptElt; for a
        rigid pair (EP, FP) being rotated as a unit about FP's
        RptElt the EP's vpt/psi/rpt must be updated by hand.

        Small-angle approximation -- the delta in dw/dx is typically
        1e-9 rad, well into the linear regime.
        """
        # Local rotation vector in FP's frame.
        theta_local = np.zeros(3, dtype=np.float64)
        theta_local[self.dof_idx] = increment

        # Convert to global via FP's TElt (upper-left 3x3 = rotation
        # local->global, columns are local axes in global coords).
        csys = self.macos.elt_csys(self.iElt)
        TElt_FP = csys[0] if isinstance(csys, tuple) else csys
        if TElt_FP.ndim == 3:
            R = TElt_FP[:3, :3, 0]
        else:
            R = TElt_FP[:3, :3]
        theta_global = R @ theta_local

        # Get FP RptElt and EP vpt/psi/rpt (all in macos BaseUnits).
        fp_rpt = np.asarray(self.macos.elt_rpt(self.iElt)).ravel()
        ep_vpt = np.asarray(self.macos.elt_vpt(ep_elt)).ravel()
        ep_psi = np.asarray(self.macos.elt_psi(ep_elt)).ravel()
        ep_rpt = np.asarray(self.macos.elt_rpt(ep_elt)).ravel()

        # Small-angle rigid rotation about FP RptElt:
        #   new_v = v + theta_global x (v - FP_RptElt)
        # Points (vpt, rpt) translate per the cross-product; psi (a
        # direction) just rotates: new_psi = normalize(psi + theta x psi).
        new_ep_vpt = ep_vpt + np.cross(theta_global, ep_vpt - fp_rpt)
        new_ep_rpt = ep_rpt + np.cross(theta_global, ep_rpt - fp_rpt)
        new_ep_psi = ep_psi + np.cross(theta_global, ep_psi)
        n = np.linalg.norm(new_ep_psi)
        if n > 0:
            new_ep_psi = new_ep_psi / n

        # Setters take shape (3, N); reshape from (3,) -> (3, 1).
        self.macos.elt_vpt(ep_elt, new_ep_vpt.reshape(3, 1))
        self.macos.elt_psi(ep_elt, new_ep_psi.reshape(3, 1))
        self.macos.elt_rpt(ep_elt, new_ep_rpt.reshape(3, 1))


def rigid_body_channels(macos,
                         rx_path: str,
                         elts: Iterable[int] | None = None,
                         dofs: Iterable[int] | None = None,
                         fp_mode: str = "track",
                         ep_elt: int = -1,
                         ) -> list[RigidBodyChannel]:
    """Build rigid-body channels for every actual optic in the Rx.

    "Actual optic" = any Element= kind except Reference / Return.
    Source isn't an Element entry in the .in file -- if you want
    source perturbations, add a dedicated SourceChannel class.

    FocalPlane elements get a :class:`FocalPlaneChannel` instead of
    the plain :class:`RigidBodyChannel`; everything else (Reflector,
    Refractor, Segment, HOE, Grating, ...) uses the base class.

    Args:
        macos:     pymacos.macos module (already init+load'd).
        rx_path:   .in file currently loaded.
        elts:      Optional restriction to a subset of element IDs.
        dofs:      Optional subset of DOFs as indices in 0..5
                   (default: all six, in Rx,Ry,Rz,Tx,Ty,Tz order).
        fp_mode:   FocalPlaneChannel mode: "fex" or "track" (default
                   "fex", which uses FEX to rebuild the XP after the
                   FP perturbation; requires a Stop to be set).
        ep_elt:    EP element id for "track" mode (default -1 ->
                   nElt-1).  Ignored in "fex" mode (FEX always
                   updates nElt-1).

    Returns:
        List of channels, element-major then DOF-minor (so the full
        state vector concatenates as Dave's spec: Elt 1 x_6, Elt 2
        x_6, ..., Elt n x_6).
    """
    if dofs is None:
        dofs = range(6)
    dofs = list(dofs)
    kinds = _parse_rx_actual_optic_elts_with_kinds(rx_path)
    if elts is not None:
        wanted = set(int(i) for i in elts)
        kinds = {k: v for k, v in kinds.items() if k in wanted}
    channels: list[RigidBodyChannel] = []
    for iElt in sorted(kinds):
        if kinds[iElt] == "FocalPlane":
            for dof in dofs:
                channels.append(FocalPlaneChannel(
                    macos=macos, iElt=iElt, dof_idx=int(dof),
                    mode=fp_mode, ep_elt=ep_elt))
        else:
            for dof in dofs:
                channels.append(RigidBodyChannel(
                    macos=macos, iElt=iElt, dof_idx=int(dof)))
    return channels


def _parse_rx_actual_optic_elts(rx_path: str) -> set[int]:
    """Scan an Rx for actual-optic elements (Reflector, Refractor,
    Segment, FocalPlane, HOE, Grating, ...).  Excludes Reference and
    Return (bookkeeping-only elements).

    Returns set of 1-based element IDs.
    """
    return set(_parse_rx_actual_optic_elts_with_kinds(rx_path))


def _parse_rx_actual_optic_elts_with_kinds(rx_path: str) -> dict[int, str]:
    """Like :func:`_parse_rx_actual_optic_elts` but also returns the
    element-kind label (Reflector / Refractor / FocalPlane / Segment
    / HOE / Grating / ...) per element, so callers can pick a
    specialized channel subclass (currently used for FocalPlane).
    """
    out: dict[int, str] = {}
    cur_elt: int | None = None
    cur_kind: str | None = None

    def _flush():
        if cur_elt is not None and cur_kind is not None:
            if cur_kind not in _NON_OPTIC_ELEMENT_KINDS:
                out[cur_elt] = cur_kind

    with open(rx_path) as f:
        for ln in f:
            s = ln.strip()
            if s.startswith("iElt="):
                _flush()
                try:
                    cur_elt = int(s.split("=", 1)[1].strip())
                except ValueError:
                    cur_elt = None
                cur_kind = None
            elif s.startswith("Element=") and cur_elt is not None:
                cur_kind = s.split("=", 1)[1].strip().split()[0]
    _flush()
    return out
