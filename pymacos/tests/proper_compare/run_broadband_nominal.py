"""Cycle 4: broadband nominal coronagraph score, 7 wavelengths over a
10% bandpass.

Configuration:
  - center wavelength  : 850 nm  (matches the monochromatic baselines)
  - bandwidth fraction : 10%      (i.e., +/- 5% of centre)
  - wavelength samples : 7 (linearly spaced)
  - cases              : (1) no mask, no Lyot       (Rx_Coro_noLyot.in)
                         (2) FPM=400um + Lyot=14mm  (Rx_Coro_FPM.in)
  - per-wavelength run : macos + PROPER through the full Phase 5
                         sphere-to-plane chain to Elt 21
  - aggregation        : incoherent sum of intensities across the band
                         (rectangular filter; could be Gaussian-weighted
                         later if needed)

Outputs (in tests/Rx/results_cycle4/):
  - {case}_broadband_focalplane.png   log10 intensity, summed
  - {case}_broadband_contrast.png     radial contrast curve
  - both_cases_contrast.png           BL vs SS-equivalent overlay (here
                                       no_mask vs with_mask)
  - per-wavelength PSF previews are not saved separately by default;
    intensities are kept only as in-memory accumulators.
  - {case}_broadband.npz              saved arrays for downstream use:
                                       I_macos_sum, I_proper_sum,
                                       I_macos_per_wave (3D),
                                       wavelengths_m, dx_focal_m

Each (case, wavelength) runs in its own subprocess (mirrors the
N-sweep driver pattern) to dodge pymacos state-leakage when changing
the wavelength inside the same process.
"""
from __future__ import absolute_import

import json
import subprocess
import sys
from pathlib import Path
from typing import List


# ----------------------------------------------------------------------
# Worker mode (called as: python -m proper_compare.run_broadband_nominal
# --worker RX_PATH WAVELENGTH_M OUT_JSON)
# ----------------------------------------------------------------------

def worker(rx_path: Path, wavelength_m: float, out_json: Path) -> None:
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import pymacos.macos as m
    from proper_compare.geometries.coro_nfprop import (
        CoroSphereToPlane, macos_run_sphere_to_plane,
        proper_run_sphere_to_plane)

    N = 1024

    # macos_run_sphere_to_plane re-inits and re-loads inside.  The
    # wavelength override has to fire AFTER the load.  Pass a hook.
    def _set_wavelength(session):
        # src_wvl() takes WaveUnits (which = BaseUnits for Rx_Coro.in).
        # Use the CBM factor to convert SI metres -> BaseUnits.
        ok_cbm, cbm = session.lib.api.base_unit_to_metres()
        if not ok_cbm or cbm == 0.0:
            raise RuntimeError("worker: CBM unavailable")
        session.src_wvl(wavelength_m / float(cbm))

    geom = CoroSphereToPlane(
        rx_filename=str(rx_path),
        src_elt=20, detector_elt=21,
        macos_size=N,
        wavelength_m=wavelength_m,
        focal_length_m=0.9514,
    )

    I_m, dx_m, wf = macos_run_sphere_to_plane(geom, m,
                                               post_load_hook=_set_wavelength)
    I_p, dx_p     = proper_run_sphere_to_plane(geom, wavefront_at_pupil=wf)

    out_npz = out_json.with_suffix(".npz")
    np.savez(out_npz,
             I_macos=I_m, I_proper=I_p,
             dx_macos=float(abs(dx_m)),
             dx_proper=float(dx_p),
             wavelength_m=float(wavelength_m))

    out_json.write_text(json.dumps({
        "wavelength_m":  float(wavelength_m),
        "dx_macos":      float(abs(dx_m)),
        "dx_proper":     float(dx_p),
        "peak_macos":    float(I_m.max()),
        "peak_proper":   float(I_p.max()),
        "npz":           str(out_npz),
    }))
    print(f"[worker lambda={wavelength_m*1e9:.1f}nm rx={rx_path.name}] "
          f"peak_macos={I_m.max():.4e}  dx={abs(dx_m):.4e}")


# ----------------------------------------------------------------------
# Driver mode (no args, or --driver explicitly)
# ----------------------------------------------------------------------

def driver() -> int:
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

    from proper_compare.contrast import (
        lambda_over_D_pixels, plot_contrast_curves, radial_contrast)

    # Center wavelength + bandpass
    LAM_C    = 850e-9          # 850 nm
    BAND_FRAC = 0.10           # 10%
    N_WAVE   = 7
    fractions = np.linspace(-BAND_FRAC/2, BAND_FRAC/2, N_WAVE)
    wavelengths = LAM_C * (1.0 + fractions)
    print(f"[driver] center λ = {LAM_C*1e9:.1f} nm; band = "
          f"{BAND_FRAC*100:.1f}%; {N_WAVE} samples:")
    for i, lam in enumerate(wavelengths):
        print(f"  {i}: {lam*1e9:.2f} nm")

    rx_dir = Path(__file__).resolve().parents[1] / "Rx"
    out_dir = Path(__file__).resolve().parent / "results_cycle4"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        "no_mask":   rx_dir / "Rx_Coro_noLyot.in",
        "with_mask": rx_dir / "Rx_Coro_FPM.in",
    }

    case_outputs = {}
    for case_name, rx_path in cases.items():
        print(f"\n[driver] === case: {case_name} ({rx_path.name}) ===")
        I_macos_sum = None
        I_proper_sum = None
        I_macos_per_wave = []
        I_proper_per_wave = []
        per_wave_info = []

        for i, lam in enumerate(wavelengths):
            out_json = out_dir / f"_bb_{case_name}_wave{i}.json"
            _spawn(rx_path, lam, out_json)
            info = json.loads(out_json.read_text())
            arrs = np.load(info["npz"])
            I_m = arrs["I_macos"]
            I_p = arrs["I_proper"]
            if I_macos_sum is None:
                I_macos_sum  = np.zeros_like(I_m)
                I_proper_sum = np.zeros_like(I_p)
            I_macos_sum  += I_m
            I_proper_sum += I_p
            I_macos_per_wave.append(I_m)
            I_proper_per_wave.append(I_p)
            per_wave_info.append(info)

        dx_focal = per_wave_info[N_WAVE // 2]["dx_macos"]

        # Save broadband stack + sums
        np.savez(out_dir / f"{case_name}_broadband.npz",
                 I_macos_sum=I_macos_sum,
                 I_proper_sum=I_proper_sum,
                 I_macos_per_wave=np.stack(I_macos_per_wave),
                 I_proper_per_wave=np.stack(I_proper_per_wave),
                 wavelengths_m=wavelengths,
                 dx_focal_m=dx_focal)

        case_outputs[case_name] = dict(
            I_macos_sum=I_macos_sum,
            I_proper_sum=I_proper_sum,
            dx_focal_m=dx_focal,
            peak_per_wave=[w["peak_macos"] for w in per_wave_info],
        )

        # Per-case focal-plane plot (log10 intensity, broadband sum)
        _save_focal_plane(I_macos_sum, dx_focal,
                          out_dir / f"{case_name}_broadband_focalplane.png",
                          title=(f"{case_name} broadband (Σ across "
                                 f"{N_WAVE} λ in 10% band centred at "
                                 f"{LAM_C*1e9:.0f} nm)"))

    # Contrast curves: use no-mask broadband peak for Strehl ref,
    # and derive lambda/D from the no-mask broadband PSF.
    I_no  = case_outputs["no_mask"]["I_macos_sum"]
    I_co  = case_outputs["with_mask"]["I_macos_sum"]
    peak_ref = float(I_no.max())
    lam_D    = float(lambda_over_D_pixels(I_no))
    print(f"\n[driver] broadband no-mask peak = {peak_ref:.4e}; "
          f"λ/D at centre = {lam_D:.2f} px")
    print(f"[driver] broadband with-mask peak = {I_co.max():.4e}; "
          f"suppression = {peak_ref / max(I_co.max(), 1e-30):.2e}")

    r_no, c_no = radial_contrast(I_no, peak_ref, lam_D, max_lambda_over_D=20.0)
    r_co, c_co = radial_contrast(I_co, peak_ref, lam_D, max_lambda_over_D=20.0)

    # Per-case contrast PNG
    plot_contrast_curves(
        {"no mask (broadband sum)": (r_no, c_no)},
        out_dir / "no_mask_broadband_contrast.png",
        title=(f"No mask, broadband ({N_WAVE} λ across {BAND_FRAC*100:.0f}% "
               f"band centred at {LAM_C*1e9:.0f} nm)"),
        ylim=(1e-10, 2.0),
    )
    plot_contrast_curves(
        {"FPM=400um + Lyot=14mm (broadband sum)": (r_co, c_co)},
        out_dir / "with_mask_broadband_contrast.png",
        title=(f"FPM=400um + Lyot=14mm, broadband ({N_WAVE} λ across "
               f"{BAND_FRAC*100:.0f}% band, centred at "
               f"{LAM_C*1e9:.0f} nm)"),
        ylim=(1e-14, 1e-4),
    )

    # Both-cases overlay
    plot_contrast_curves(
        {"no mask (broadband)":  (r_no, c_no),
         "FPM+Lyot (broadband)": (r_co, c_co)},
        out_dir / "both_cases_contrast.png",
        title=(f"Cycle 4: nominal coronagraph contrast at broadband "
               f"({BAND_FRAC*100:.0f}% band, {N_WAVE} λ summed)\n"
               "Strehl-norm to no-mask broadband peak"),
        ylim=(1e-14, 2.0),
    )

    # Clean up worker scratch
    for p in out_dir.glob("_bb_*"):
        try:
            p.unlink()
        except OSError:
            pass

    print(f"\n[driver] wrote results to {out_dir}/")
    for case_name in cases:
        print(f"  {case_name}_broadband_focalplane.png")
        print(f"  {case_name}_broadband_contrast.png")
        print(f"  {case_name}_broadband.npz")
    print(f"  both_cases_contrast.png")
    return 0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _spawn(rx_path: Path, wavelength_m: float, out_json: Path) -> None:
    args = [sys.executable, "-m", "proper_compare.run_broadband_nominal",
            "--worker", str(rx_path), f"{wavelength_m:.6e}", str(out_json)]
    cwd = Path(__file__).resolve().parents[1]
    print(f"[driver] spawn λ={wavelength_m*1e9:.2f}nm rx={rx_path.name}")
    res = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError(
            f"worker failed (rx={rx_path.name}, "
            f"λ={wavelength_m*1e9:.1f}nm, code={res.returncode})")
    for line in res.stdout.splitlines():
        if line.startswith("[worker"):
            print(line)


def _save_focal_plane(intensity, dx_m, out_path, title=""):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = intensity.shape[0]
    cy = cx = (N - 1) // 2
    # Crop window: +/- 25 lambda/D approx
    half_window_um = 25 * 50.0  # ~25 lambda/D in micron, rough
    half_window_px = int(half_window_um * 1e-6 / dx_m)
    half_window_px = min(half_window_px, N // 2)
    lo_y = max(0, cy - half_window_px); hi_y = min(N, cy + half_window_px + 1)
    lo_x = max(0, cx - half_window_px); hi_x = min(N, cx + half_window_px + 1)
    crop = intensity[lo_y:hi_y, lo_x:hi_x]

    peak = crop.max() if crop.max() > 0 else 1.0
    floor = max(peak * 1e-10, 1e-30)
    log_crop = np.log10(np.maximum(crop, floor))

    ext = ((-(cx - lo_x) * dx_m * 1e6),
            ((hi_x - cx - 1) * dx_m * 1e6),
           (-(cy - lo_y) * dx_m * 1e6),
            ((hi_y - cy - 1) * dx_m * 1e6))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(log_crop, extent=ext, origin="lower", cmap="viridis",
                   vmin=np.log10(floor), vmax=np.log10(peak))
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.set_title(f"{title}\npeak = {peak:.3e}")
    plt.colorbar(im, ax=ax, label=r"log$_{10}$ intensity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[driver] wrote {out_path}")


def _entry() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        # --worker RX_PATH WAVELENGTH_M OUT_JSON
        rx_path      = Path(sys.argv[2])
        wavelength_m = float(sys.argv[3])
        out_json     = Path(sys.argv[4])
        worker(rx_path, wavelength_m, out_json)
        return 0
    return driver()


if __name__ == "__main__":
    sys.exit(_entry())
