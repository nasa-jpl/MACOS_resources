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

from dataclasses import dataclass, field
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
class SourceChannel(SensitivityChannel):
    """One-DOF rigid-body perturbation on the SOURCE (iElt=0 in macos).

    Wraps :func:`pymacos.macos.perturb_src`, the iElt=0 branch of
    macos's CPERTURB.  The source rotates ChfRayDir (point source)
    and translates ChfRayPos (collimated or point); macos updates
    xGrid/yGrid/SegXGrid alongside.

    Stop re-enforcement: a source perturbation moves the chief ray
    off the stop center.  For dw/dx work we want the wavefront
    referenced to the same chief-ray-through-stop geometry as the
    nominal, so the channel re-enforces the stop after every
    perturbation.  Two modes:

    - ``stop_mode='obj'`` (default): call
      ``m.stop_obj(*stop_obj_pos)`` -- the OBJ branch of macos's
      STOP command (deterministic chief-ray-aim math, no iterative
      solve).  Use this for Rxes that declare an object-space stop
      via ``ApStop= x y z``.
    - ``stop_mode='elt'``: call ``m.stop(stop_elt)`` -- the ELT
      branch.  Use this when the system stop is an element.
    - ``stop_mode='none'``: skip; the chief ray drifts off-stop
      with the source.  Diagnostic only; the resulting columns
      include a large "chief ray missed stop" component that is
      not physically meaningful for sensitivity work.

    DOF layout matches :class:`RigidBodyChannel`:
    (Rx, Ry, Rz, Tx, Ty, Tz).  ``perturb_src`` follows macos's
    source-perturb frame convention (GLOBAL by default, LOCAL when
    the Rx declares ``SrcXAxis/SrcYAxis``); since macos itself has
    no per-call switch this channel passes the user vector through
    verbatim.

    perturb_src is incremental like CPERTURB_PROG, so the
    cumulative-state pattern (``current`` tracks the running
    perturbation; each apply sends ``value - current``) mirrors
    RigidBodyChannel.
    """
    macos: object
    dof_idx: int
    stop_mode: str = "obj"        # 'obj' | 'elt' | 'none'
    stop_obj_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stop_elt: int = 0             # required when stop_mode='elt'
    current: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.dof_idx <= 5:
            raise ValueError(
                f"SourceChannel: dof_idx must be in 0..5, got "
                f"{self.dof_idx}")
        if self.stop_mode not in ("obj", "elt", "none"):
            raise ValueError(
                f"SourceChannel: stop_mode must be 'obj', 'elt' or "
                f"'none'; got {self.stop_mode!r}")
        if self.stop_mode == "elt" and self.stop_elt <= 0:
            raise ValueError(
                "SourceChannel: stop_mode='elt' requires stop_elt > 0")

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"Src {_RB_DOF_LABELS[self.dof_idx]}"

    @property
    def kind(self) -> str:
        return "Source"

    def _enforce_stop(self) -> None:
        if self.stop_mode == "obj":
            self.macos.stop_obj(*self.stop_obj_pos)
        elif self.stop_mode == "elt":
            self.macos.stop(self.stop_elt)
        # 'none': no-op

    def _do_perturb(self, increment: float) -> None:
        rot = [0.0, 0.0, 0.0]
        trans = [0.0, 0.0, 0.0]
        if self.dof_idx < 3:
            rot[self.dof_idx] = increment
        else:
            trans[self.dof_idx - 3] = increment
        self.macos.perturb_src(rotation_rad=tuple(rot),
                                translation_m=tuple(trans))
        self._enforce_stop()
        self.macos.modify()

    def apply(self, value: float) -> None:
        increment = value - self.current
        if increment != 0.0:
            self._do_perturb(increment)
        self.current = value

    def restore(self) -> None:
        self.apply(0.0)


def source_channels(macos,
                    dofs: Iterable[int] | None = None,
                    stop_mode: str = "obj",
                    stop_obj_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
                    stop_elt: int = 0,
                    ) -> list[SourceChannel]:
    """Build SourceChannel instances for the requested DOFs.

    Args:
        macos:        pymacos.macos module.
        dofs:         DOF subset (default all six 0..5).
        stop_mode:    'obj' (default) -> use stop_obj_pos;
                      'elt' -> use stop_elt; 'none' -> no
                      re-enforcement.
        stop_obj_pos: object-space stop coordinates for stop_mode
                      'obj'.  Defaults to the origin -- matches Rx
                      conventions like ``ApStop= 0 0 0`` in
                      Rx_e5hex1.in.
        stop_elt:     element index for stop_mode='elt'.

    Returns:
        One :class:`SourceChannel` per DOF, in DOF order.
    """
    if dofs is None:
        dofs = range(6)
    return [SourceChannel(macos=macos, dof_idx=int(d),
                          stop_mode=stop_mode,
                          stop_obj_pos=stop_obj_pos,
                          stop_elt=stop_elt)
            for d in dofs]


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
        if self.mode not in ("track", "srs", "sxp", "fex", "none"):
            raise ValueError(
                f"FocalPlaneChannel: mode must be 'track', 'srs', "
                f"'sxp', 'fex', or 'none'; got {self.mode!r}")

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
                         include_non_optics: bool = False,
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
        include_non_optics:  if True, also build plain
                   :class:`RigidBodyChannel` entries for Reference /
                   Return surfaces (kinds normally excluded as
                   bookkeeping-only).  Needed when the per-element
                   block is meant to drive
                   :func:`predict_global_rigid_response` --
                   reconstructing a rigid global perturbation of a
                   group containing Ref/Return surfaces needs their
                   per-element columns to capture the rigid-coupling
                   cancellations they participate in.

    Returns:
        List of channels, element-major then DOF-minor (so the full
        state vector concatenates as Dave's spec: Elt 1 x_6, Elt 2
        x_6, ..., Elt n x_6).
    """
    if dofs is None:
        dofs = range(6)
    dofs = list(dofs)
    kinds = _parse_rx_actual_optic_elts_with_kinds(
        rx_path, include_non_optics=include_non_optics)
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


# ---------------------------------------------------------------------------
# Grouped rigid-body channels (rigid-body perturbation of a sub-assembly)
# ---------------------------------------------------------------------------

@dataclass
class GroupedRigidBodyChannel(SensitivityChannel):
    """Rigid-body perturbation applied to a group of elements as a
    single rigid unit, dispatched through macos's ``GPERTURB``
    (``CPERTURB_GRP_DVR``).

    Why macos-side and not Python-side: the per-element rigid-body
    columns can be linearly combined to synthesize a group column
    under most conditions, but combinations involving the
    Reference/Return surfaces around the exit pupil and the
    focal-plane element do NOT superimpose linearly -- a rigid
    camera (lens + FP) motion produces an EP/FP rigid coupling that
    cancels across the Reference and the FP into a small residual,
    which superposition of two individually-large columns can't
    reproduce within numerical precision.  Letting macos perturb the
    members as a rigid unit captures the cancellation directly.

    Group declaration is installed dynamically: the existing
    ``EltGrp`` state on ``ref_elt`` is snapshotted, the desired
    members are written via :func:`macos.elt_grp`, ``GPERTURB`` runs
    (via :func:`macos.prb_grp`), and the snapshot is restored on
    ``restore()``.  This permits OVERLAPPING groups across separate
    channels -- the same element can belong to multiple Python-side
    groups even though macos's ``EltGrp(0:N, iElt)`` data structure
    only allows one group per element at a time.

    DOF layout matches :class:`RigidBodyChannel`:
    (Rx, Ry, Rz, Tx, Ty, Tz) interpreted in ``ref_elt``'s LOCAL
    frame.  Rotation pivot is ``ref_elt``'s RptElt (per
    ``CPERTURB_GRP_DVR`` in funcsub.F).  Macos converts the local
    6-vector to global via ``TElt`` and applies the same rigid
    motion to every member.

    Optional FP follow-up for groups containing a focal-plane
    element (``fp_elt > 0``):

    - ``fp_mode='none'`` (default if no FP in group): GPERTURB only.
    - ``fp_mode='sxp'`` (default if FP in group): after GPERTURB,
      run macos's SXP to recompute the EP from the post-
      perturbation chief-ray geometry.  Captures the EP radius
      change driven by FP motion (the upstream optic motion in the
      group already imprints its tilt/defocus on the wavefront via
      its own pose change).
    - ``fp_mode='fex'``: same as 'sxp' but with FEX (EP as the
      Stop's optical conjugate; insensitive to FP motion -- mostly
      diagnostic).
    - ``fp_mode='srs'``: ``srs(ep_elt, fp_elt)`` to slave EP pose
      to the post-perturbation FP.
    """
    macos: object
    members: tuple[int, ...]
    dof_idx: int
    group_name: str = ""
    ref_elt: int = 0          # 0 -> first member; rotation pivot
    fp_elt: int = 0           # 0 -> no FP follow-up
    fp_mode: str = "auto"     # 'auto'|'none'|'sxp'|'fex'|'srs'
    ep_elt: int = -1          # for srs follow-up; -1 -> nElt-1
    coords: str = "global"    # 'global' (default) | 'local' (ref_elt's TElt frame)
    stop_mode: str = "obj"    # 'obj' | 'elt' | 'none' -- re-aim chief ray after GPERTURB
    stop_obj_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    stop_elt: int = 0         # required when stop_mode='elt'
    current: float = 0.0
    _saved_grp: tuple[int, ...] | None = field(
        default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.dof_idx <= 5:
            raise ValueError(
                f"GroupedRigidBodyChannel: dof_idx must be in 0..5, "
                f"got {self.dof_idx}")
        if len(self.members) < 2:
            raise ValueError(
                f"GroupedRigidBodyChannel: need at least 2 members, "
                f"got {self.members!r}")
        if self.ref_elt == 0:
            self.ref_elt = int(self.members[0])
        if self.ref_elt not in self.members:
            raise ValueError(
                f"GroupedRigidBodyChannel: ref_elt={self.ref_elt} "
                f"must be one of members={self.members!r}")
        if not self.group_name:
            self.group_name = f"{min(self.members)}-{max(self.members)}"
        if self.fp_mode not in ("auto", "none", "sxp", "fex", "srs"):
            raise ValueError(
                f"GroupedRigidBodyChannel: fp_mode must be one of "
                f"'auto','none','sxp','fex','srs'; got {self.fp_mode!r}")
        if self.fp_mode == "auto":
            self.fp_mode = ("sxp" if (self.fp_elt > 0
                                       and self.fp_elt in self.members)
                            else "none")
        if self.coords not in ("global", "local"):
            raise ValueError(
                f"GroupedRigidBodyChannel: coords must be 'global' or "
                f"'local'; got {self.coords!r}")
        if self.stop_mode not in ("obj", "elt", "none"):
            raise ValueError(
                f"GroupedRigidBodyChannel: stop_mode must be 'obj', "
                f"'elt' or 'none'; got {self.stop_mode!r}")
        if self.stop_mode == "elt" and self.stop_elt <= 0:
            raise ValueError(
                "GroupedRigidBodyChannel: stop_mode='elt' requires "
                "stop_elt > 0")

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"Grp[{self.group_name}] {_RB_DOF_LABELS[self.dof_idx]}"

    @property
    def kind(self) -> str:
        return "Group"

    def _install_group(self) -> None:
        """Snapshot ref_elt's current EltGrp and write desired members.
        Idempotent across multiple apply() calls -- the snapshot is
        taken only on the first install and held until restore_group().
        """
        if self._saved_grp is not None:
            return
        existing = self.macos.elt_grp(self.ref_elt)
        cur = list(existing[0]) if existing else []
        self._saved_grp = tuple(int(x) for x in cur)
        target = [int(m) for m in self.members]
        if sorted(cur) != sorted(target):
            self.macos.elt_grp(self.ref_elt, target)

    def _restore_group(self) -> None:
        if self._saved_grp is None:
            return
        saved = self._saved_grp
        self._saved_grp = None
        if not saved:
            self.macos.elt_grp_rm(self.ref_elt)
        else:
            self.macos.elt_grp(self.ref_elt, list(saved))

    def _enforce_stop(self) -> None:
        # Defensive re-aim of the chief ray through the stop after
        # the GPERTURB.  GPERTURB itself only changes element poses
        # and leaves ChfRayDir / ChfRayPos untouched, so for the
        # standard apply-from-nominal / measure / restore channel
        # pattern this is a no-op: the source wasn't perturbed, the
        # chief ray is still nominal, the stop is trivially still
        # hit.  But if a caller stacks channels (e.g. SourceChannel
        # followed by GroupedRigidBodyChannel before measuring) the
        # source-side perturbation moved the chief ray; re-aiming
        # here keeps the OPD reference geometry consistent across
        # the stack.  Cost is one cheap macos call per apply().
        if self.stop_mode == "obj":
            self.macos.stop_obj(*self.stop_obj_pos)
        elif self.stop_mode == "elt":
            self.macos.stop(self.stop_elt)

    def _do_perturb(self, increment: float) -> None:
        self._install_group()
        prb_vec = np.zeros((6, 1), dtype=np.float64)
        prb_vec[self.dof_idx, 0] = increment
        # glb_csys = 1 (global, default) or 0 (ref_elt's local frame).
        # Global is the right default for telescopes: the source's
        # ChfRayDir lives in global coords, so for the source-vs-
        # all-optics-group cross-check to give cos=-1 the group
        # rotation axes must be in the same global frame.
        glb = 1 if self.coords == "global" else 0
        self.macos.prb_grp(
            [self.ref_elt], prb_vec,
            np.array([glb], dtype=np.int32))
        self._enforce_stop()
        self.macos.modify()

        if self.fp_mode == "sxp":
            self.macos.trace_rays(self.ref_elt)
            self.macos.sxp()
        elif self.fp_mode == "fex":
            self.macos.trace_rays(self.ref_elt)
            self.macos.fex()
        elif self.fp_mode == "srs":
            ep = (self.ep_elt if self.ep_elt > 0
                  else self.macos.num_elt() - 1)
            fp = self.fp_elt if self.fp_elt > 0 else self.macos.num_elt()
            self.macos.trace_rays(self.ref_elt)
            self.macos.srs(ep, fp, link=True)

    def apply(self, value: float) -> None:
        increment = value - self.current
        if increment != 0.0:
            self._do_perturb(increment)
        self.current = value

    def restore(self) -> None:
        # Drive the pose back to nominal via GPERTURB, then release
        # the temporary EltGrp install so other channels' state is
        # not surprised by lingering group definitions.
        self.apply(0.0)
        self._restore_group()


def predict_global_rigid_response(
        macos,
        dwdx: np.ndarray,
        col_map: dict[tuple[int, int], int],
        members: Sequence[int],
        prb_global: Sequence[float],
        pivot_global: Sequence[float] | None = None,
        ) -> np.ndarray:
    """Predict the dw response of a global rigid-body perturbation
    applied to a group of members, USING per-element dw/dx columns
    measured in each element's LOCAL frame.

    The per-element columns in ``dwdx`` were measured by perturbing
    each element about its own RptElt in its own LOCAL frame (the
    default ``RigidBodyChannel`` and ``perturb()`` semantics).  To
    use them to predict the response of a rigid-body perturbation
    of a group (rotation+translation applied at a GLOBAL reference
    pivot), each member needs its own equivalent local-frame
    perturbation derived from the rigid-body kinematics:

        For pivot P (global) and global rotation theta_g (rad) and
        global translation T_g, a member m at RptElt p_m with
        local-to-global rotation R_m experiences:
            - local rotation:    theta_m_loc = R_m^T @ theta_g
            - local translation: T_m_loc = R_m^T @ (T_g + theta_g × (p_m - P))

    The predicted response is then the linear combination

        dw_pred = Σ_m  Σ_j  theta_m_loc[j] * dwdx[:, (m, j)]
                + Σ_m  Σ_j  T_m_loc[j]      * dwdx[:, (m, 3+j)]

    This is a self-consistency check on the measured per-element
    columns: comparing ``dw_pred`` against a separately-measured
    GPERTURB-driven group column tests whether the per-element
    sensitivities correctly reconstruct the rigid-body response, and
    catches per-element frame/pivot bookkeeping bugs that would
    silently bias group predictions.

    Args:
        macos:         pymacos.macos module (Rx loaded), used to read
                       ``elt_csys``, ``elt_rpt`` and the BaseUnits
                       conversion factor.
        dwdx:          (Nw, Nz) Jacobian holding per-element columns.
        col_map:       ``{(iElt, dof_idx): column_index}`` mapping.
                       Every (m, 0..5) referenced by ``members`` must
                       be present.
        members:       member element IDs of the group.
        prb_global:    6-vector (Rx, Ry, Rz, Tx, Ty, Tz) in GLOBAL
                       frame.  Rotations in radians, translations in
                       SI METRES (matches the per-element columns
                       measured by ``RigidBodyChannel`` which uses
                       ``m.perturb(..., translation_m=...)``).
        pivot_global:  optional global rotation pivot (3-vector), in
                       prescription BaseUnits (the same units macos
                       returns from ``elt_rpt``).  Default: RptElt of
                       the first member.  For cross-checking against
                       a GPERTURB group, set this to the GPERTURB
                       ref_elt's RptElt.

    Returns:
        dw_pred:       (Nw,) predicted OPD response vector.
    """
    members_i = [int(m) for m in members]
    if len(members_i) == 0:
        raise ValueError("predict_global_rigid_response: empty members")
    prb = np.asarray(prb_global, dtype=np.float64).reshape(6)
    theta_g = prb[:3]
    T_g     = prb[3:]
    if pivot_global is None:
        pivot = np.asarray(macos.elt_rpt(members_i[0])).ravel()
    else:
        pivot = np.asarray(pivot_global, dtype=np.float64).ravel()

    # base_units -> metres scale (e.g. 1e-3 if Rx is in mm; 1.0 if Rx
    # is in metres).  Per-element translation columns were measured
    # with perturb(translation_m=...) so the column weight for an
    # offset-induced translation has to be in METRES too -- but
    # offset = rpt_m - pivot is in BaseUnits.  Convert.
    ok_cbm, cbm = macos.lib.api.base_unit_to_metres()
    if not ok_cbm or cbm == 0.0:
        raise Exception("predict_global_rigid_response: macos returned "
                        "no BaseUnits->metres factor (CBM)")
    base_to_m = float(cbm)

    # Cache R_m and rpt_m for each member.
    R_by: dict[int, np.ndarray] = {}
    rpt_by: dict[int, np.ndarray] = {}
    for m_elt in members_i:
        csys_m = macos.elt_csys(m_elt)
        TElt_m = csys_m[0] if isinstance(csys_m, tuple) else csys_m
        R_by[m_elt] = (TElt_m[:3, :3, 0] if TElt_m.ndim == 3
                       else TElt_m[:3, :3])
        rpt_by[m_elt] = np.asarray(macos.elt_rpt(m_elt)).ravel()

    Nw = dwdx.shape[0]
    dw_pred = np.zeros(Nw, dtype=np.float64)

    for m_elt in members_i:
        R_m  = R_by[m_elt]
        offset_m_base = rpt_by[m_elt] - pivot          # global, BaseUnits
        offset_m_m    = offset_m_base * base_to_m      # global, metres
        # Rigid-body local-frame perturbation at this member.
        # theta_g: rad (frame-independent under rotation -> still rad
        # in local frame).  T_g + theta_g × offset: metres.
        theta_m_loc = R_m.T @ theta_g
        T_m_loc     = R_m.T @ (T_g + np.cross(theta_g, offset_m_m))

        for j in range(3):
            if theta_m_loc[j] != 0.0:
                key = (m_elt, j)
                if key not in col_map:
                    raise KeyError(
                        f"predict_global_rigid_response: missing "
                        f"column for ({m_elt}, R{('x','y','z')[j]}) "
                        f"in col_map")
                dw_pred += theta_m_loc[j] * dwdx[:, col_map[key]]
            if T_m_loc[j] != 0.0:
                key = (m_elt, j + 3)
                if key not in col_map:
                    raise KeyError(
                        f"predict_global_rigid_response: missing "
                        f"column for ({m_elt}, T{('x','y','z')[j]}) "
                        f"in col_map")
                dw_pred += T_m_loc[j] * dwdx[:, col_map[key]]

    return dw_pred


def group_synthesis_matrix(
        macos,
        members: Sequence[int],
        dofs: Sequence[int] | None = None,
        pivot_global: Sequence[float] | None = None,
        units: str = "SI",
        ) -> np.ndarray:
    """Build the (N_rows, 6) weight matrix W that maps a global
    rigid-body 6-vector applied at ``pivot_global`` to per-element
    LOCAL-frame perturbations of the group ``members``.

    Layout:
        N_rows = len(members) * len(dofs).
        Rows are ordered ELEMENT-MAJOR, DOF-MINOR -- matching the
        per-element channel order in dw_dx.py:
            [(members[0], dofs[0]), (members[0], dofs[1]), ...,
             (members[1], dofs[0]), ...]
        Columns are indexed 0..5 = global (Rx, Ry, Rz, Tx, Ty, Tz).

    For each member m with local-to-global rotation R_m and RptElt
    offset p_m from the global pivot P:
        local rotation:    theta_m_loc = R_m^T @ theta_g
        local translation: T_m_loc     = R_m^T @ (T_g + theta_g × (p_m - P))

    Usage (consistency check):
        dwdx_group_pred[:, dof_g] = dwdx_perelt[:, member_dofs] @ W[:, dof_g]
    where ``member_dofs`` is the list of dwdx column indices for the
    same (member, dof) order used to construct W.

    Units: rotation rows are unitless (rad/rad).  Translation rows
    coming from theta_g × offset are in METRES (units='SI', for use
    against per-element columns measured in OPD/m), or in the Rx's
    BaseUnits (units='base', for OPD/BaseUnits columns).  Match the
    convention used to build the dwdx matrix you'll multiply with.

    Args:
        macos:         pymacos.macos module (Rx loaded), used for
                       elt_csys, elt_rpt, and (only for units='SI')
                       base_unit_to_metres.
        members:       member element IDs of the group.
        dofs:          subset of local DOFs to include per member
                       (default: all six, in 0..5 order matching
                       ``_RB_DOF_LABELS``).
        pivot_global:  global rotation pivot (3-vector, in macos
                       BaseUnits).  Default: RptElt of the first
                       member.
        units:         'SI' (default) -- W's translation entries
                       have units of metres, matching dwdx columns
                       measured with m.perturb(translation_m=...).
                       'base' -- W's translation entries in
                       BaseUnits, matching dwdx columns measured
                       with translation perturbations in BaseUnits.

    Returns:
        W:             (len(members)*len(dofs), 6) float64 ndarray.
    """
    members_i = [int(m) for m in members]
    if len(members_i) == 0:
        raise ValueError("group_synthesis_matrix: empty members")
    if dofs is None:
        dofs = list(range(6))
    dofs = list(dofs)
    if units not in ("SI", "base"):
        raise ValueError(
            f"group_synthesis_matrix: units must be 'SI' or 'base'; "
            f"got {units!r}")

    if units == "SI":
        ok_cbm, cbm = macos.lib.api.base_unit_to_metres()
        if not ok_cbm or cbm == 0.0:
            raise Exception("group_synthesis_matrix: BaseUnits->m "
                            "factor (CBM) unavailable")
        base_to_m = float(cbm)         # offset BaseUnits -> metres
    else:
        # units='base': offset stays in BaseUnits, no conversion.
        base_to_m = 1.0

    if pivot_global is None:
        pivot = np.asarray(macos.elt_rpt(members_i[0])).ravel()
    else:
        pivot = np.asarray(pivot_global, dtype=np.float64).ravel()

    Nrows = len(members_i) * len(dofs)
    W = np.zeros((Nrows, 6), dtype=np.float64)

    # Per-global-DOF column: a unit perturbation of that DOF.
    for col, g_dof in enumerate(range(6)):
        prb = np.zeros(6, dtype=np.float64)
        prb[g_dof] = 1.0
        theta_g = prb[:3]
        T_g     = prb[3:]
        for mi, m_elt in enumerate(members_i):
            csys_m = macos.elt_csys(m_elt)
            TElt_m = csys_m[0] if isinstance(csys_m, tuple) else csys_m
            R_m = (TElt_m[:3, :3, 0] if TElt_m.ndim == 3
                   else TElt_m[:3, :3])
            rpt_m = np.asarray(macos.elt_rpt(m_elt)).ravel()
            offset_m = (rpt_m - pivot) * base_to_m   # global, metres
            theta_m_loc = R_m.T @ theta_g
            T_m_loc     = R_m.T @ (T_g + np.cross(theta_g, offset_m))
            # Fill the 6 rows for this member, only keeping the
            # requested dofs.
            local6 = np.concatenate([theta_m_loc, T_m_loc])  # (Rx,Ry,Rz,Tx,Ty,Tz)
            for di, dof_idx in enumerate(dofs):
                W[mi * len(dofs) + di, col] = local6[dof_idx]
    return W


def grouped_rigid_body_channels(
        macos,
        groups: dict[str, tuple[int, ...]],
        ref_elt_by_group: dict[str, int] | None = None,
        dofs: Iterable[int] | None = None,
        rx_path: str | None = None,
        fp_mode: str = "auto",
        ep_elt: int = -1,
        fp_elt_by_group: dict[str, int] | None = None,
        coords: str = "global",
        stop_mode: str = "obj",
        stop_obj_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        stop_elt: int = 0,
        ) -> list[GroupedRigidBodyChannel]:
    """Build group channels for each named (name -> members) group.

    Args:
        macos:             pymacos.macos module.
        groups:            mapping from group name to a tuple of
                           member element IDs.
        ref_elt_by_group:  optional mapping name -> reference
                           element id (default: first member).
        dofs:              optional subset of DOF indices 0..5
                           (default: all six).
        rx_path:           if given, parse the Rx to auto-detect a
                           FocalPlane-typed element among each
                           group's members; that element becomes the
                           group's ``fp_elt``.  Without it (None),
                           no FP follow-up is performed unless
                           ``fp_elt_by_group`` overrides per group.
        fp_mode:           propagated to each group channel
                           ('auto' / 'none' / 'sxp' / 'fex' / 'srs').
        ep_elt:            propagated to each group channel.
        fp_elt_by_group:   optional explicit per-group FP element id
                           override; supersedes Rx-based detection.

    Returns:
        List of :class:`GroupedRigidBodyChannel`, ordered as the
        ``groups`` dict iterates, with DOF-minor inside each group.
    """
    if dofs is None:
        dofs = range(6)
    dofs = list(dofs)
    ref_elt_by_group = ref_elt_by_group or {}
    fp_elt_by_group = dict(fp_elt_by_group or {})

    if rx_path is not None:
        kinds = _parse_rx_actual_optic_elts_with_kinds(str(rx_path))
        fp_elts = {iElt for iElt, k in kinds.items() if k == "FocalPlane"}
    else:
        fp_elts = set()

    channels: list[GroupedRigidBodyChannel] = []
    for name, members in groups.items():
        members_t = tuple(int(m) for m in members)
        if len(members_t) < 2:
            continue
        ref = int(ref_elt_by_group.get(name, members_t[0]))
        fp = int(fp_elt_by_group.get(name, 0))
        if fp == 0 and fp_elts:
            fp_in_group = [e for e in members_t if e in fp_elts]
            if fp_in_group:
                fp = fp_in_group[0]
        for dof in dofs:
            channels.append(GroupedRigidBodyChannel(
                macos=macos, members=members_t, dof_idx=int(dof),
                group_name=name, ref_elt=ref,
                fp_elt=fp, fp_mode=fp_mode, ep_elt=ep_elt,
                coords=coords,
                stop_mode=stop_mode,
                stop_obj_pos=stop_obj_pos,
                stop_elt=stop_elt))
    return channels


def parse_rx_groups(rx_path: str) -> dict[str, tuple[int, ...]]:
    """Parse ``EltGrp=`` declarations from a macos .in file.

    Multiple elements typically declare the same group (all members
    of a group repeat ``EltGrp= N m1 m2 ... mN`` in their per-element
    blocks).  Dedups by the sorted member tuple and emits one entry
    per unique group, named ``"min-max"`` of its member IDs.

    Returns ``{name: (m1, m2, ...)}`` in stable order.
    """
    seen: dict[tuple[int, ...], str] = {}
    cur_elt: int | None = None

    with open(rx_path) as f:
        for ln in f:
            s = ln.strip()
            if s.startswith("iElt="):
                try:
                    cur_elt = int(s.split("=", 1)[1].strip())
                except ValueError:
                    cur_elt = None
            elif s.startswith("EltGrp=") and cur_elt is not None:
                # "EltGrp= N m1 m2 ... mN"  (positive N -> explicit list)
                # Macos also supports negative N (range form) and
                # MrEltGrp (multi-range) -- defer those until needed.
                toks = s.split("=", 1)[1].split()
                try:
                    nums = [int(t) for t in toks]
                except ValueError:
                    continue
                if not nums or nums[0] <= 0:
                    continue
                n = nums[0]
                if len(nums) < n + 1:
                    continue
                members = tuple(sorted(nums[1:n + 1]))
                if members not in seen:
                    seen[members] = f"{members[0]}-{members[-1]}"
    # Preserve insertion order (dict in Python 3.7+ does this).
    return {seen[k]: k for k in seen}


def _parse_rx_actual_optic_elts_with_kinds(
        rx_path: str, include_non_optics: bool = False
        ) -> dict[int, str]:
    """Like :func:`_parse_rx_actual_optic_elts` but also returns the
    element-kind label (Reflector / Refractor / FocalPlane / Segment
    / HOE / Grating / ...) per element, so callers can pick a
    specialized channel subclass (currently used for FocalPlane).

    ``include_non_optics=True`` retains Reference / Return elements
    too (needed when measuring per-element rigid-body columns to
    drive ``predict_global_rigid_response`` -- a global rigid motion
    of a group containing Ref/Return surfaces needs those per-
    element columns to reconstruct the rigid-coupling cancellations).
    """
    out: dict[int, str] = {}
    cur_elt: int | None = None
    cur_kind: str | None = None

    def _flush():
        if cur_elt is not None and cur_kind is not None:
            if include_non_optics or cur_kind not in _NON_OPTIC_ELEMENT_KINDS:
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
