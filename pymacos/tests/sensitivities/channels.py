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
