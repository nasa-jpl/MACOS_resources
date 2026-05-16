"""Sampling-density (N) sweep utilities for macos<->PROPER tests.

Runs a phase's test at multiple diffraction-grid sizes
N in {256, 512, 1024, [2048]}, recording macos<->PROPER agreement,
dark-zone contrast, and wall-clock runtime.

The diffraction grid (mdttl) is set by ``pymacos.init(N)``; the
source ray-grid (nGridpts) is in the prescription file.  These two
are independent in macos but should scale together for consistent
sampling -- we patch ``nGridpts`` per-N (odd, near N/2 to keep
the source aperture pixel-centred).

Memory caveat: at N=2048 the diffraction-grid arrays are 32 MB each
in single precision, 64 MB in double; combined with macos's
intermediate buffers and PROPER's working arrays peak resident gets
to a few GB.  This has crashed VS Code once.  Default sweep is
{256, 512, 1024}; 2048 must be opted in explicitly.
"""
from __future__ import absolute_import

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np


# ---- prescription patching ------------------------------------------

def patch_ngridpts(src_rx: Path, dst_rx: Path, new_n: int) -> None:
    """Copy ``src_rx`` to ``dst_rx`` with ``nGridpts`` replaced.

    The keyword ``nGridpts=<integer>`` may appear with arbitrary
    whitespace between '=' and the value (e.g. ``nGridpts=  511``);
    we match the whole token and substitute the new integer back in.
    """
    text = src_rx.read_text()
    new_text, n_subs = re.subn(
        r"(nGridpts=\s*)\d+",
        lambda m: f"{m.group(1)}{new_n}",
        text)
    if n_subs == 0:
        raise ValueError(
            f"patch_ngridpts: no nGridpts directive found in {src_rx}")
    if n_subs > 1:
        raise ValueError(
            f"patch_ngridpts: multiple nGridpts directives in {src_rx} "
            f"({n_subs}) -- ambiguous, aborting")
    dst_rx.write_text(new_text)


def n_to_ngridpts(N: int) -> int:
    """Map a diffraction-grid size N to a sensible odd source nGridpts.

    Convention used throughout the test harness: ``nGridpts = N//2 - 1``
    forced odd.  Keeps the source aperture grid pixel-centred and
    matches the production setup (N=1024 -> nGridpts=511).
    """
    g = N // 2 - 1
    if g % 2 == 0:
        g -= 1
    return g


# ---- sweep driver ---------------------------------------------------

@dataclass
class SweepRow:
    N: int
    nGridpts: int
    runtime_s: float
    peak_macos: float           # raw macos peak intensity
    peak_proper: float          # raw PROPER peak (different conv.)
    max_abs: float              # sum/peak-norm pairwise |a-b|
    rms_abs: float
    norm_kind: str              # 'sum' or 'peak'
    contrast_at: dict = field(default_factory=dict)
                                # {lambda/D: contrast value}


def sweep_n(
    base_rx: Path,
    N_list: Sequence[int],
    run_one: Callable[[Path, int], dict],
    output_dir: Path,
    norm_kind: str = "peak",
    contrast_separations: Optional[Sequence[float]] = None,
    ) -> List[SweepRow]:
    """Run ``run_one`` at each N in ``N_list``, return per-N records.

    Args:
        base_rx: source prescription path; patched per-N to set
            nGridpts.
        N_list: diffraction-grid sizes to sweep.
        run_one: callable(rx_path, N) -> dict with keys
            'intensity_m', 'intensity_p', 'peak_unaberrated'.
            'peak_unaberrated' is for Strehl normalisation of the
            contrast curve -- pass it through from the FIRST N's
            no-mask reference if scoring a coronagraphed PSF, or
            just intensity_m.max() if scoring the un-coronagraphed
            PSF itself.
        output_dir: where to write patched-prescription temp files
            and any artefacts the caller saves.
        norm_kind: 'sum' or 'peak' for the pairwise residual.
        contrast_separations: list of lambda/D values to score.
            Default: (1, 3, 5, 7, 10).

    Returns:
        List of SweepRow, one per N, in input order.
    """
    if contrast_separations is None:
        contrast_separations = (1.0, 3.0, 5.0, 7.0, 10.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[SweepRow] = []
    for N in N_list:
        ngridpts = n_to_ngridpts(N)
        tmp_rx = output_dir / f"_sweep_{base_rx.stem}_N{N}.in"
        patch_ngridpts(base_rx, tmp_rx, ngridpts)

        t0 = time.time()
        result = run_one(tmp_rx, N)
        runtime = time.time() - t0

        I_m = np.asarray(result["intensity_m"])
        I_p = np.asarray(result["intensity_p"])

        if norm_kind == "sum":
            a = I_m / I_m.sum()
            b = I_p / I_p.sum()
        elif norm_kind == "peak":
            a = I_m / I_m.max()
            b = I_p / I_p.max()
        else:
            raise ValueError(f"unknown norm_kind={norm_kind!r}")
        d = np.abs(a - b)

        # Optional contrast scoring
        contrast = {}
        if "contrast_curve" in result:
            # Caller provides (r_lambda_over_D, contrast) arrays.
            r_ld, c = result["contrast_curve"]
            for sep in contrast_separations:
                i = int(np.argmin(np.abs(r_ld - sep)))
                contrast[float(sep)] = float(c[i])

        rows.append(SweepRow(
            N=N,
            nGridpts=ngridpts,
            runtime_s=runtime,
            peak_macos=float(I_m.max()),
            peak_proper=float(I_p.max()),
            max_abs=float(d.max()),
            rms_abs=float(np.sqrt((d * d).mean())),
            norm_kind=norm_kind,
            contrast_at=contrast,
        ))
    return rows


# ---- digest printing / saving ---------------------------------------

def digest_table(rows: List[SweepRow],
                 contrast_separations: Optional[Sequence[float]] = None,
                 ) -> str:
    """Format sweep rows as a markdown-style table.

    Two-stage layout: a "core" table with agreement + runtime, and
    a "contrast" table with the contrast values at requested
    separations (if the rows carry contrast_at data).
    """
    if not rows:
        return "(empty sweep)"

    norm = rows[0].norm_kind

    lines: List[str] = []
    lines.append(
        f"| N | nGridpts | runtime (s) | peak macos | peak PROPER | "
        f"max |a-b| ({norm}) | RMS |a-b| ({norm}) |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r.N} | {r.nGridpts} | {r.runtime_s:.2f} | "
            f"{r.peak_macos:.3e} | {r.peak_proper:.3e} | "
            f"{r.max_abs:.3e} | {r.rms_abs:.3e} |")

    if any(r.contrast_at for r in rows):
        if contrast_separations is None:
            contrast_separations = sorted(
                {s for r in rows for s in r.contrast_at})
        lines.append("")
        lines.append(
            "Radial contrast (Strehl-normalised) at key separations:")
        lines.append("")
        header = "| N | " + " | ".join(
            f"{s} λ/D" for s in contrast_separations) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(contrast_separations) + 1))
        for r in rows:
            cells = " | ".join(
                f"{r.contrast_at.get(float(s), float('nan')):.3e}"
                for s in contrast_separations)
            lines.append(f"| {r.N} | {cells} |")

    return "\n".join(lines)
