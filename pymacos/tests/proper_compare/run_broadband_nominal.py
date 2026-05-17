"""Cycle 4 / broadband nominal coronagraph score.

For each case (no_mask, with_mask), runs macos + PROPER through the
full Phase-5 sphere-to-plane chain at 7 wavelengths uniformly spaced
across a 10% band centred on 850 nm.  All 7 PSFs are then resampled
onto a common pixel pitch (the centre-wavelength's dx_focal) before
incoherent summation -- focal-plane dx scales linearly with λ, so a
naive pixel-by-pixel sum mixes physical positions and is wrong by
~5% radially across a 10% band.

Per-case run lives in ONE subprocess that loops over wavelengths
internally (vs the previous 7 subprocesses per case).  Saves
14 × Python+pymacos startup costs ≈ ~2 min vs ~3.5 min wall time.

Outputs (in results_cycle4/):
  - {case}_broadband_focalplane.png   log10 intensity, summed
  - {case}_broadband_contrast.png     radial contrast curve
  - both_cases_contrast.png           overlay
  - {case}_broadband.npz              full arrays: native per-wave,
                                       resampled per-wave, summed,
                                       wavelengths, dx values
"""
from __future__ import absolute_import

import json
import subprocess
import sys
from pathlib import Path
from typing import List


# ----------------------------------------------------------------------
# Resampling: NxN at dx_native -> NxN at dx_ref, centered.
# ----------------------------------------------------------------------

def resample_to_grid(I, dx_native: float, dx_ref: float):
    """Interpolate ``I`` (NxN, pitch ``dx_native``) onto an NxN grid of
    pitch ``dx_ref``, centred on the array centre.  Outside the native
    grid, values fall to zero.

    Note: linear interpolation of intensity samples.  We're NOT
    rescaling for flux conservation -- the operation is "evaluate the
    underlying continuous I(x, y) at the ref grid's pixel centres"
    and we trust that the input was already a properly-normalised
    intensity-per-pixel quantity at its native dx.  For a broadband
    sum across closely-spaced wavelengths (10% band), the small dx
    change doesn't introduce significant flux distortion at the pixel
    level.
    """
    from scipy.interpolate import RegularGridInterpolator
    import numpy as np

    N = I.shape[0]
    coords_native = (np.arange(N) - (N - 1) / 2.0) * dx_native
    coords_ref    = (np.arange(N) - (N - 1) / 2.0) * dx_ref

    interp = RegularGridInterpolator(
        (coords_native, coords_native), I,
        bounds_error=False, fill_value=0.0, method="linear")

    yy, xx = np.meshgrid(coords_ref, coords_ref, indexing="ij")
    pts = np.stack([yy, xx], axis=-1).reshape(-1, 2)
    return interp(pts).reshape(N, N)


# ----------------------------------------------------------------------
# Worker: one case, all wavelengths in one process.
# ----------------------------------------------------------------------

def worker(rx_path: Path, wavelengths_si: List[float],
           out_npz: Path, case_name: str) -> None:
    """Run macos + PROPER at every wavelength in ``wavelengths_si``
    on the loaded prescription, resample each PSF to the centre-λ
    pixel pitch, and write the full bundle to ``out_npz``.

    Single subprocess, single ``init(N)`` and ``load(rx)``; loops
    wavelengths internally via ``src_wvl()``.
    """
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import pymacos.macos as m
    from proper_compare.geometries.coro_nfprop import (
        CoroSphereToPlane, proper_run_sphere_to_plane)

    N = 1024
    m.init(N)
    m.load(str(rx_path))

    # SI metres -> WaveUnits
    ok_cbm, cbm = m.lib.api.base_unit_to_metres()
    if not ok_cbm or cbm == 0.0:
        raise RuntimeError("worker: CBM unavailable")
    si_to_wave_units = 1.0 / float(cbm)

    n_wave = len(wavelengths_si)
    I_macos_native  = np.zeros((n_wave, N, N))
    I_proper_native = np.zeros((n_wave, N, N))
    dx_macos        = np.zeros(n_wave)
    dx_proper       = np.zeros(n_wave)
    peak_macos_native = np.zeros(n_wave)

    for i, lam in enumerate(wavelengths_si):
        m.src_wvl(lam * si_to_wave_units)

        # Manually replicate macos_run_sphere_to_plane's interior
        # without the per-iteration init/load.
        cfield_pupil = m.complex_field(20)            # re-traces (reset)
        I_pupil      = m.intensity(20, reset_trace=False)
        I_focal      = m.intensity(21, reset_trace=False)
        amp_pupil    = np.sqrt(np.clip(I_pupil, 0, None))
        d_pupil      = m.dx_at(20)
        d_focal      = m.dx_at(21)

        wf = dict(complex_field=cfield_pupil,
                  amplitude=amp_pupil,
                  dx_pupil_m=d_pupil,
                  dx_focal_m=d_focal)

        geom = CoroSphereToPlane(
            rx_filename=str(rx_path),
            src_elt=20, detector_elt=21,
            macos_size=N,
            wavelength_m=lam,
            focal_length_m=0.9514)

        I_proper_i, dx_proper_i = proper_run_sphere_to_plane(
            geom, wavefront_at_pupil=wf)

        I_macos_native[i]  = I_focal
        I_proper_native[i] = I_proper_i
        dx_macos[i]        = abs(d_focal)
        dx_proper[i]       = dx_proper_i
        peak_macos_native[i] = float(I_focal.max())

        print(f"[worker {case_name} λ={lam*1e9:.2f}nm] "
              f"peak_macos={I_focal.max():.4e}  "
              f"dx_macos={abs(d_focal):.4e}  "
              f"peak_proper={I_proper_i.max():.4e}",
              flush=True)

    # Resample each PSF onto the centre-λ's pitch.
    ref_idx = n_wave // 2
    ref_dx_macos  = dx_macos[ref_idx]
    ref_dx_proper = dx_proper[ref_idx]

    print(f"[worker {case_name}] resampling to ref dx (centre λ) = "
          f"{ref_dx_macos:.4e} m", flush=True)

    I_macos_resampled  = np.zeros_like(I_macos_native)
    I_proper_resampled = np.zeros_like(I_proper_native)
    for i in range(n_wave):
        I_macos_resampled[i] = resample_to_grid(
            I_macos_native[i], dx_macos[i], ref_dx_macos)
        I_proper_resampled[i] = resample_to_grid(
            I_proper_native[i], dx_proper[i], ref_dx_proper)

    I_macos_sum  = I_macos_resampled.sum(axis=0)
    I_proper_sum = I_proper_resampled.sum(axis=0)

    np.savez(out_npz,
             # Per-wavelength native arrays + dx
             I_macos_native=I_macos_native,
             I_proper_native=I_proper_native,
             dx_macos_per_wave=dx_macos,
             dx_proper_per_wave=dx_proper,
             # Per-wavelength resampled to centre-λ dx
             I_macos_resampled=I_macos_resampled,
             I_proper_resampled=I_proper_resampled,
             # Summed broadband
             I_macos_sum=I_macos_sum,
             I_proper_sum=I_proper_sum,
             # Common reference dx (used for resampling target)
             ref_dx_macos=ref_dx_macos,
             ref_dx_proper=ref_dx_proper,
             # Wavelengths
             wavelengths_m=np.array(wavelengths_si),
             # Misc digest
             peak_macos_native=peak_macos_native)

    print(f"[worker {case_name}] wrote {out_npz}  "
          f"broadband peak_macos = {I_macos_sum.max():.4e}",
          flush=True)


# ----------------------------------------------------------------------
# Driver: spawn one worker per case, aggregate at the end.
# ----------------------------------------------------------------------

def driver() -> int:
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

    from proper_compare.contrast import (
        lambda_over_D_pixels, plot_contrast_curves, radial_contrast)

    LAM_C    = 850e-9
    BAND_FRAC = 0.10
    N_WAVE   = 7
    fractions = np.linspace(-BAND_FRAC / 2, BAND_FRAC / 2, N_WAVE)
    wavelengths = (LAM_C * (1.0 + fractions)).tolist()
    print(f"[driver] centre λ = {LAM_C*1e9:.1f} nm; band = "
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

    npz_paths = {}
    for case_name, rx_path in cases.items():
        npz_path = out_dir / f"{case_name}_broadband.npz"
        npz_paths[case_name] = npz_path
        print(f"\n[driver] === case: {case_name} ({rx_path.name}) ===")
        _spawn_case(rx_path, wavelengths, npz_path, case_name)

    # ------------- aggregate + plot -------------
    no_mask  = np.load(npz_paths["no_mask"])
    with_mask = np.load(npz_paths["with_mask"])

    I_no  = no_mask["I_macos_sum"]
    I_co  = with_mask["I_macos_sum"]
    dx_ref = float(no_mask["ref_dx_macos"])

    peak_ref = float(I_no.max())
    lam_D    = float(lambda_over_D_pixels(I_no))
    print(f"\n[driver] broadband no-mask peak    = {peak_ref:.4e}  "
          f"(λ/D at centre = {lam_D:.2f} px)")
    print(f"[driver] broadband with-mask peak  = {I_co.max():.4e}")
    print(f"[driver] suppression               = "
          f"{peak_ref / max(I_co.max(), 1e-30):.2e}")

    r_no, c_no = radial_contrast(I_no, peak_ref, lam_D,
                                 max_lambda_over_D=20.0)
    r_co, c_co = radial_contrast(I_co, peak_ref, lam_D,
                                 max_lambda_over_D=20.0)

    plot_contrast_curves(
        {"no mask (broadband sum)": (r_no, c_no)},
        out_dir / "no_mask_broadband_contrast.png",
        title=(f"Cycle 4: no mask, broadband ({N_WAVE} λ across "
               f"{BAND_FRAC*100:.0f}% band centred at "
               f"{LAM_C*1e9:.0f} nm)"),
        ylim=(1e-10, 2.0),
    )
    plot_contrast_curves(
        {"FPM=400um + Lyot=14mm (broadband sum)": (r_co, c_co)},
        out_dir / "with_mask_broadband_contrast.png",
        title=(f"Cycle 4: FPM=400um + Lyot=14mm, broadband"),
        ylim=(1e-14, 1e-4),
    )
    plot_contrast_curves(
        {"no mask (broadband)":  (r_no, c_no),
         "FPM+Lyot (broadband)": (r_co, c_co)},
        out_dir / "both_cases_contrast.png",
        title=(f"Cycle 4: nominal coronagraph broadband contrast "
               f"({BAND_FRAC*100:.0f}% band, {N_WAVE} λ summed,\n"
               f"resampled to centre-λ pitch before sum)"),
        ylim=(1e-14, 2.0),
    )

    _save_focal_plane(I_no, dx_ref,
                      out_dir / "no_mask_broadband_focalplane.png",
                      title=(f"no_mask broadband sum ({N_WAVE} λ in "
                             f"{BAND_FRAC*100:.0f}% band centred at "
                             f"{LAM_C*1e9:.0f} nm)"))
    _save_focal_plane(I_co, dx_ref,
                      out_dir / "with_mask_broadband_focalplane.png",
                      title=(f"with_mask broadband sum ({N_WAVE} λ in "
                             f"{BAND_FRAC*100:.0f}% band centred at "
                             f"{LAM_C*1e9:.0f} nm)"))

    print(f"\n[driver] wrote results to {out_dir}/")
    print(f"  both_cases_contrast.png  "
          "{no_mask,with_mask}_broadband_{focalplane,contrast}.png  "
          "{no_mask,with_mask}_broadband.npz")
    return 0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _spawn_case(rx_path: Path, wavelengths: list, out_npz: Path,
                case_name: str) -> None:
    args = [sys.executable, "-m", "proper_compare.run_broadband_nominal",
            "--worker", str(rx_path),
            ",".join(f"{lam:.10e}" for lam in wavelengths),
            str(out_npz), case_name]
    cwd = Path(__file__).resolve().parents[1]
    res = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError(
            f"worker failed (case={case_name}, code={res.returncode})")
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
    # Crop window roughly +/- 25 λ/D
    half_window_um = 25 * 50.0
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
        # --worker RX_PATH WAVELENGTHS_COMMA_SEP OUT_NPZ CASE_NAME
        rx_path        = Path(sys.argv[2])
        wavelengths_si = [float(s) for s in sys.argv[3].split(",")]
        out_npz        = Path(sys.argv[4])
        case_name      = sys.argv[5]
        worker(rx_path, wavelengths_si, out_npz, case_name)
        return 0
    return driver()


if __name__ == "__main__":
    sys.exit(_entry())
