"""Finite-difference Jacobian engine for sensitivity matrices.

Given a list of :class:`SensitivityChannel`, a wavefront-extraction
function ``wf_func() -> ndarray``, and a step size ``delta``, computes
``dw/dz`` column-by-column.  Defaults to central differences (more
accurate, 2x macos traces per column).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .channels import SensitivityChannel


def sensitivity_jacobian(
    channels: Sequence[SensitivityChannel],
    wf_func: Callable[[], np.ndarray],
    delta: float,
    *,
    method: str = "central",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build the Jacobian dw/dz by finite differences.

    Args:
        channels: Perturbation channels (one column each).
        wf_func:  Callable returning the wavefront vector as ndarray
                  (any shape; gets flattened for the Jacobian columns).
        delta:    Perturbation magnitude.  Same units as the channel
                  expects (for MonZern: macos BaseUnits, e.g. mm).
        method:   "central" (default, 2 traces per column) or
                  "forward" (1 trace per column, less accurate).
        verbose:  Print per-column progress.

    Returns:
        J:             (Nw, Nz) Jacobian.
        w_nom:         (Nw,)    flattened nominal wavefront.
        channel_names: list of column labels matching the Nz columns.
    """
    if method not in ("central", "forward"):
        raise ValueError(f"method must be 'central' or 'forward', got "
                         f"{method!r}")

    if verbose:
        print(f"[jacobian] {len(channels)} channels, method={method}, "
              f"delta={delta:.3e}")

    w_nom_arr = wf_func()
    w_nom = np.asarray(w_nom_arr, dtype=np.float64).ravel()
    Nw = w_nom.size
    Nz = len(channels)

    J = np.zeros((Nw, Nz), dtype=np.float64)
    names: list[str] = []

    for k, ch in enumerate(channels):
        if method == "central":
            ch.apply(+delta)
            w_plus = np.asarray(wf_func(), dtype=np.float64).ravel()
            ch.apply(-delta)
            w_minus = np.asarray(wf_func(), dtype=np.float64).ravel()
            ch.restore()
            J[:, k] = (w_plus - w_minus) / (2.0 * delta)
        else:  # forward
            ch.apply(+delta)
            w_plus = np.asarray(wf_func(), dtype=np.float64).ravel()
            ch.restore()
            J[:, k] = (w_plus - w_nom) / delta
        names.append(ch.name)
        if verbose:
            col_rms = float(np.sqrt(np.mean(J[:, k] ** 2)))
            print(f"[jacobian]   {k+1:2d}/{Nz}  {ch.name:20s}  "
                  f"RMS dw/dz = {col_rms:.3e}")

    return J, w_nom, names
