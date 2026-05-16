"""N-sweep of Phase 5 (focal-plane PSF), forked-process driver.

Standalone script -- NOT pytest-discovered.  Each N value runs in a
fresh subprocess to dodge pymacos's known state-leakage across
``init(N)`` transitions within one process.  After all subprocesses
complete, this driver reads their JSON sidecars and writes the
digest + overlay plots to ``results_phase3/``.

Usage:
    python -m proper_compare.run_n_sweep_phase5            # 256, 512, 1024
    PROPER_COMPARE_NSWEEP_HIGH_N=1 python -m \
        proper_compare.run_n_sweep_phase5                  # + 2048

Run from the tests/ directory or anywhere the proper_compare package
is importable.  Artefacts:
    results_phase3/n_sweep_phase5.md
    results_phase3/n_sweep_phase5_no_mask_contrast.png
    results_phase3/n_sweep_phase5_with_mask_contrast.png

Mode of operation: if invoked with `--worker N RX_PATH OUT_JSON`, runs
ONE configuration in this process (init -> macos run -> PROPER run ->
contrast -> JSON dump).  Otherwise: forks one subprocess per N x
{no_mask, with_mask} combination, then aggregates.
"""
from __future__ import absolute_import

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


# ---- worker mode ----------------------------------------------------

def worker(N: int, rx_path: Path, mode: str, out_json: Path,
           reference_no_mask_psf_npy: Optional[Path] = None) -> None:
    """Run a single (N, rx, mode) configuration; dump results to JSON.

    mode in {'no_mask', 'with_mask'} -- controls whether we use the
    rx_path's own peak as the Strehl reference or a no-mask reference
    PSF loaded from reference_no_mask_psf_npy.
    """
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import pymacos.macos as m
    from proper_compare.contrast import (lambda_over_D_pixels,
                                          radial_contrast)
    from proper_compare.geometries.coro_nfprop import (
        CoroSphereToPlane, macos_run_sphere_to_plane,
        proper_run_sphere_to_plane)

    m.init(N)
    geom = CoroSphereToPlane(
        rx_filename=str(rx_path),     # absolute path
        src_elt=20, detector_elt=21,
        macos_size=N,
        focal_length_m=0.9514,
    )

    t0 = time.time()
    I_m, dx_m, wf = macos_run_sphere_to_plane(geom, m)
    I_p, dx_p     = proper_run_sphere_to_plane(geom, wavefront_at_pupil=wf)
    runtime = time.time() - t0

    # Strehl-norm pairwise residual
    a = I_m / I_m.max()
    b = I_p / I_p.max()
    d = np.abs(a - b)

    # Contrast curve.  For mode='no_mask', lambda/D + Strehl from this
    # PSF itself.  For 'with_mask', load the no-mask reference PSF
    # (same N) and derive lambda/D + peak_ref from it.
    if mode == "no_mask":
        peak_ref = float(I_m.max())
        lam_D = lambda_over_D_pixels(I_m)
    elif mode == "with_mask":
        if reference_no_mask_psf_npy is None or \
           not reference_no_mask_psf_npy.exists():
            raise RuntimeError(
                f"with_mask worker at N={N} needs a no-mask reference "
                f"PSF .npy (got {reference_no_mask_psf_npy})")
        I_ref = np.load(reference_no_mask_psf_npy)
        peak_ref = float(I_ref.max())
        lam_D = lambda_over_D_pixels(I_ref)
    else:
        raise ValueError(f"unknown mode={mode!r}")

    r_ld, c = radial_contrast(I_m, peak_ref, lam_D, max_lambda_over_D=20.0)

    # Save the no-mask PSF as a sidecar .npy for the with_mask pass.
    if mode == "no_mask":
        psf_npy = out_json.with_suffix(".psf.npy")
        np.save(psf_npy, I_m)
    else:
        psf_npy = None

    result = {
        "N": N,
        "mode": mode,
        "runtime_s": runtime,
        "peak_macos": float(I_m.max()),
        "peak_proper": float(I_p.max()),
        "peak_ref": peak_ref,
        "lambda_over_D_px": float(lam_D),
        "max_abs": float(d.max()),
        "rms_abs": float(np.sqrt((d * d).mean())),
        "contrast_r_lambda_over_D": r_ld.tolist(),
        "contrast_values": c.tolist(),
        "no_mask_psf_npy": str(psf_npy) if psf_npy else None,
    }
    out_json.write_text(json.dumps(result))
    print(f"[worker N={N} mode={mode}] runtime={runtime:.2f}s  "
          f"max|a-b|={d.max():.3e}  peak_macos={I_m.max():.3e}")


# ---- driver mode ----------------------------------------------------

def driver() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import numpy as np

    from proper_compare.contrast import plot_contrast_curves
    from proper_compare.n_sweep import (SweepRow, digest_table,
                                         patch_ngridpts, n_to_ngridpts)

    rx_dir = Path(__file__).resolve().parents[1] / "Rx"
    no_mask_rx   = rx_dir / "Rx_Coro_noLyot.in"
    with_mask_rx = rx_dir / "Rx_Coro_FPM.in"

    if os.environ.get("PROPER_COMPARE_NSWEEP_HIGH_N", "0") == "1":
        N_list = [256, 512, 1024, 2048]
    else:
        N_list = [256, 512, 1024]

    out_dir = (Path(__file__).resolve().parent / "results_phase3")
    out_dir.mkdir(parents=True, exist_ok=True)

    contrast_seps = (1.0, 3.0, 5.0, 7.0, 10.0)

    rows_no: List[SweepRow] = []
    rows_co: List[SweepRow] = []
    no_mask_curves = {}
    with_mask_curves = {}

    for N in N_list:
        # Patch nGridpts per-N for both prescriptions.
        ng = n_to_ngridpts(N)
        no_mask_tmp   = out_dir / f"_sweep_no_mask_N{N}.in"
        with_mask_tmp = out_dir / f"_sweep_with_mask_N{N}.in"
        patch_ngridpts(no_mask_rx,   no_mask_tmp,   ng)
        patch_ngridpts(with_mask_rx, with_mask_tmp, ng)

        # ----- Fork: no_mask worker -----
        no_mask_json = out_dir / f"_sweep_no_mask_N{N}.json"
        no_mask_psf  = no_mask_json.with_suffix(".psf.npy")
        _spawn_worker(N, no_mask_tmp, "no_mask", no_mask_json)

        no_mask_res = json.loads(no_mask_json.read_text())
        rows_no.append(SweepRow(
            N=N, nGridpts=ng,
            runtime_s=no_mask_res["runtime_s"],
            peak_macos=no_mask_res["peak_macos"],
            peak_proper=no_mask_res["peak_proper"],
            max_abs=no_mask_res["max_abs"],
            rms_abs=no_mask_res["rms_abs"],
            norm_kind="peak",
            contrast_at={
                float(s): _sample(no_mask_res["contrast_r_lambda_over_D"],
                                  no_mask_res["contrast_values"], s)
                for s in contrast_seps},
        ))
        no_mask_curves[f"N={N} (λ/D={no_mask_res['lambda_over_D_px']:.2f}px)"] = (
            np.asarray(no_mask_res["contrast_r_lambda_over_D"]),
            np.asarray(no_mask_res["contrast_values"]))

        # ----- Fork: with_mask worker (needs no_mask PSF for ref) -----
        with_mask_json = out_dir / f"_sweep_with_mask_N{N}.json"
        _spawn_worker(N, with_mask_tmp, "with_mask", with_mask_json,
                      no_mask_psf)
        with_mask_res = json.loads(with_mask_json.read_text())
        rows_co.append(SweepRow(
            N=N, nGridpts=ng,
            runtime_s=with_mask_res["runtime_s"],
            peak_macos=with_mask_res["peak_macos"],
            peak_proper=with_mask_res["peak_proper"],
            max_abs=with_mask_res["max_abs"],
            rms_abs=with_mask_res["rms_abs"],
            norm_kind="peak",
            contrast_at={
                float(s): _sample(with_mask_res["contrast_r_lambda_over_D"],
                                  with_mask_res["contrast_values"], s)
                for s in contrast_seps},
        ))
        with_mask_curves[f"N={N} (λ/D={with_mask_res['lambda_over_D_px']:.2f}px)"] = (
            np.asarray(with_mask_res["contrast_r_lambda_over_D"]),
            np.asarray(with_mask_res["contrast_values"]))

    # ----- Aggregate: plots + markdown digest -----
    plot_contrast_curves(
        no_mask_curves,
        out_dir / "n_sweep_phase5_no_mask_contrast.png",
        title="Phase 5.1 (no mask): radial contrast vs N",
        ylim=(1e-10, 2.0),
    )
    plot_contrast_curves(
        with_mask_curves,
        out_dir / "n_sweep_phase5_with_mask_contrast.png",
        title="Phase 5.2 (FPM=400um + Lyot=14mm): radial contrast vs N",
        ylim=(1e-14, 1e-4),
    )

    md = "\n".join([
        "# Phase 5 N-sweep digest",
        "",
        "Driver: `run_n_sweep_phase5.py`.  Each (N, mode) pair runs "
        "in its own subprocess to dodge pymacos state-leakage across "
        "`init(N)` transitions.  Source ray-grid scaled per-N: "
        "`nGridpts = N//2 - 1` (forced odd).",
        "",
        "## Phase 5.1 -- no mask (un-coronagraphed PSF)",
        "",
        digest_table(rows_no, contrast_separations=contrast_seps),
        "",
        "## Phase 5.2 -- FPM=400um + Lyot=14mm",
        "",
        digest_table(rows_co, contrast_separations=contrast_seps),
        "",
    ])
    md_path = out_dir / "n_sweep_phase5.md"
    md_path.write_text(md)
    print(f"\n[driver] wrote {md_path}")

    # Clean up per-N scratch (patched prescriptions + JSON sidecars
    # + reference .npy).  The aggregated .md + .png are the keepers.
    for p in out_dir.glob("_sweep_*"):
        try:
            p.unlink()
        except OSError:
            pass
    print("\n=== Phase 5.1 (no mask) ===")
    print(digest_table(rows_no, contrast_separations=contrast_seps))
    print("\n=== Phase 5.2 (FPM + Lyot) ===")
    print(digest_table(rows_co, contrast_separations=contrast_seps))
    return 0


def _sample(r_list, c_list, sep):
    import numpy as np
    r = np.asarray(r_list)
    c = np.asarray(c_list)
    return float(c[int(np.argmin(np.abs(r - sep)))])


def _spawn_worker(N: int, rx_path: Path, mode: str, out_json: Path,
                  ref_psf_npy: Optional[Path] = None) -> None:
    args = [sys.executable, "-m", "proper_compare.run_n_sweep_phase5",
            "--worker", str(N), str(rx_path), mode, str(out_json)]
    if ref_psf_npy is not None:
        args.append(str(ref_psf_npy))
    print(f"[driver] spawn N={N} mode={mode} ...")
    # Run from tests/ so 'proper_compare' is the package root.
    cwd = Path(__file__).resolve().parents[1]
    res = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError(
            f"worker failed (N={N}, mode={mode}, code={res.returncode})")
    # Pass through any worker print so we see it in the driver output.
    for line in res.stdout.splitlines():
        if line.startswith("[worker"):
            print(line)


def _entry() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        # --worker N RX_PATH MODE OUT_JSON [REF_PSF_NPY]
        N        = int(sys.argv[2])
        rx_path  = Path(sys.argv[3])
        mode     = sys.argv[4]
        out_json = Path(sys.argv[5])
        ref_psf  = Path(sys.argv[6]) if len(sys.argv) > 6 else None
        worker(N, rx_path, mode, out_json, ref_psf)
        return 0
    return driver()


if __name__ == "__main__":
    sys.exit(_entry())
