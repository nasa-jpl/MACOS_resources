"""Cycle 4: broadband aberration sweep.

Combines the broadband summation (run_broadband_nominal pattern) with
the SystemState perturbation framework (test_coro_aberrations
pattern).  For each aberration case, runs macos + PROPER through the
full Phase-5 chain at 7 wavelengths uniformly spaced across a 10%
band, resamples per-wavelength PSFs to the centre-λ pitch, sums
incoherently, and saves:

  - 3-panel focal-plane PNG  (macos / PROPER / signed Strehl-norm
                              difference)
  - per-case .npz             (full per-wavelength stacks +
                              resampled stacks + summed)
  - overlay contrast curves   (all states on one log-y plot)

One subprocess per (state) -- the state is serialised through the
CLI as JSON; the worker reconstructs and applies it via
apply_system_state.  Within a subprocess, wavelengths loop without
re-init/load (matches run_broadband_nominal).

Aberration magnitudes are larger here than test_coro_aberrations.py
so the perturbation signal is visible against the broadband-sum
contrast floor.
"""
from __future__ import absolute_import

import json
import subprocess
import sys
from pathlib import Path
from typing import List


# ----------------------------------------------------------------------
# SystemState JSON serialisation (so we can pass it across subprocesses)
# ----------------------------------------------------------------------

def state_to_json(state) -> str:
    import numpy as np
    return json.dumps({
        "name": state.name,
        "layout": {
            "real_optics": list(state.layout.real_optics),
            "zernike_optics": {str(k): list(v)
                               for k, v in state.layout.zernike_optics.items()},
            "zernike_type": state.layout.zernike_type,
        },
        "x": np.asarray(state.x, dtype=float).tolist(),
        "z": np.asarray(state.z, dtype=float).tolist(),
    })


def state_from_json(s: str):
    import numpy as np
    from proper_compare.aberrations import SystemLayout, SystemState

    d = json.loads(s)
    layout = SystemLayout(
        real_optics=tuple(d["layout"]["real_optics"]),
        zernike_optics={int(k): tuple(v)
                        for k, v in d["layout"]["zernike_optics"].items()},
        zernike_type=d["layout"]["zernike_type"],
    )
    return SystemState(
        layout=layout,
        x=np.array(d["x"], dtype=float),
        z=np.array(d["z"], dtype=float),
        name=d["name"],
    )


# ----------------------------------------------------------------------
# Resample helper (copied from run_broadband_nominal.py; same routine).
# ----------------------------------------------------------------------

def resample_to_grid(I, dx_native: float, dx_ref: float):
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
# Worker: one case, one SystemState, loop wavelengths.
# ----------------------------------------------------------------------

def worker(rx_path: Path, wavelengths_si: List[float], state_json: str,
           out_npz: Path) -> None:
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import pymacos.macos as m
    from proper_compare.aberrations import apply_system_state
    from proper_compare.geometries.coro_nfprop import (
        CoroSphereToPlane, proper_run_sphere_to_plane)

    N = 1024
    state = state_from_json(state_json)
    print(f"[worker {state.name}] applying state with "
          f"{(state.x != 0).sum()} nonzero x-DOFs and "
          f"{(state.z != 0).sum()} nonzero z-DOFs", flush=True)

    m.init(N)
    m.load(str(rx_path))
    apply_system_state(m, state)        # apply ONCE; persists across λ

    ok_cbm, cbm = m.lib.api.base_unit_to_metres()
    if not ok_cbm or cbm == 0.0:
        raise RuntimeError("worker: CBM unavailable")
    si_to_wave_units = 1.0 / float(cbm)

    n_wave = len(wavelengths_si)
    I_macos_native   = np.zeros((n_wave, N, N))
    I_proper_native  = np.zeros((n_wave, N, N))
    dx_macos         = np.zeros(n_wave)
    dx_proper        = np.zeros(n_wave)

    for i, lam in enumerate(wavelengths_si):
        m.src_wvl(lam * si_to_wave_units)

        cfield_pupil = m.complex_field(20)
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

        I_macos_native[i]   = I_focal
        I_proper_native[i]  = I_proper_i
        dx_macos[i]         = abs(d_focal)
        dx_proper[i]        = dx_proper_i

        print(f"[worker {state.name} λ={lam*1e9:.2f}nm] "
              f"peak_macos={I_focal.max():.4e}  "
              f"dx_macos={abs(d_focal):.4e}  "
              f"peak_proper={I_proper_i.max():.4e}", flush=True)

    # macos<->PROPER scale reconciliation:  PROPER's per-call
    # prop_define_entrance normalises total flux per wavelength to ~1,
    # while macos's intensities reflect absolute physical flux (which
    # falls off slowly with λ for fixed grid).  Their per-λ peaks
    # therefore live on different scales (~1e-1 vs ~1e-5 monochromatic
    # nominal).  When summed to broadband each engine sees DIFFERENT
    # relative weights across wavelengths, so the summed PSF shapes
    # diverge.  Scale each PROPER per-λ PSF by peak_macos[i] /
    # peak_proper[i] BEFORE the sum -- this puts the per-λ shapes on
    # the same scale and gives a broadband PROPER sum directly
    # comparable in Strehl-norm to macos's.
    peak_m_per_wave = np.array([I_macos_native[i].max()
                                for i in range(n_wave)])
    peak_p_per_wave = np.array([I_proper_native[i].max()
                                for i in range(n_wave)])
    scale = np.where(peak_p_per_wave > 0,
                     peak_m_per_wave / peak_p_per_wave, 0.0)
    I_proper_rescaled = I_proper_native * scale[:, None, None]

    # Resample to centre-λ pitch
    ref_idx = n_wave // 2
    ref_dx_macos  = dx_macos[ref_idx]
    ref_dx_proper = dx_proper[ref_idx]
    I_macos_resampled  = np.zeros_like(I_macos_native)
    I_proper_resampled = np.zeros_like(I_proper_native)
    for i in range(n_wave):
        I_macos_resampled[i] = resample_to_grid(
            I_macos_native[i], dx_macos[i], ref_dx_macos)
        I_proper_resampled[i] = resample_to_grid(
            I_proper_rescaled[i], dx_proper[i], ref_dx_proper)
    I_macos_sum  = I_macos_resampled.sum(axis=0)
    I_proper_sum = I_proper_resampled.sum(axis=0)

    np.savez(out_npz,
             I_macos_native=I_macos_native,
             I_proper_native=I_proper_native,
             dx_macos_per_wave=dx_macos,
             dx_proper_per_wave=dx_proper,
             I_macos_resampled=I_macos_resampled,
             I_proper_resampled=I_proper_resampled,
             I_macos_sum=I_macos_sum,
             I_proper_sum=I_proper_sum,
             ref_dx_macos=ref_dx_macos,
             ref_dx_proper=ref_dx_proper,
             wavelengths_m=np.array(wavelengths_si),
             state_json=state_json)
    print(f"[worker {state.name}] wrote {out_npz}  "
          f"broadband peak_macos = {I_macos_sum.max():.4e}", flush=True)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def _make_states():
    """Aberration cases for this driver.  Magnitudes are LARGER than
    test_coro_aberrations.py's so the perturbation signal is visible
    against the broadband contrast floor.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from proper_compare.aberrations import CORO_LAYOUT, SystemState

    states = []

    # 1. Nominal
    states.append(SystemState(layout=CORO_LAYOUT, name="nominal"))

    # 2. Elt 1 tip y 50 urad (clear PSF shift; focus stays within
    #    the 400 μm FPM, ~77 μm displaced from centre).
    s = SystemState(layout=CORO_LAYOUT, name="Elt1_tip_y_50urad")
    s.set_dof(element=1, dof="roty", value=50e-6)
    states.append(s)

    # 2b. Elt 1 tip y 500 urad -- focus displacement at the FPM is
    #     ~770 μm, well OUTSIDE the 400 μm FPM.  Coronagraph should
    #     "fail" here: most of the starlight bypasses the FPM and
    #     reaches the science focal plane unsuppressed.  Compare to
    #     the 50 μrad case for the difference.
    s = SystemState(layout=CORO_LAYOUT, name="Elt1_tip_y_500urad")
    s.set_dof(element=1, dof="roty", value=500e-6)
    states.append(s)

    # 3. Elt 1 Tx +20 um
    s = SystemState(layout=CORO_LAYOUT, name="Elt1_Tx_plus_20um")
    s.set_dof(element=1, dof="transx", value=20e-6)
    states.append(s)

    # 4. DM1 mixed Zernike: astig + coma + trefoil + Z33, each 10 nm
    #    surface RMS (so ~20 nm wavefront on reflection).
    s = SystemState(layout=CORO_LAYOUT,
                    name="DM1_astig+coma+trefoil+Z33_10nm")
    s.set_zernike(element=4, mode=5,  coef_m=10e-9)
    s.set_zernike(element=4, mode=7,  coef_m=10e-9)
    s.set_zernike(element=4, mode=9,  coef_m=10e-9)
    s.set_zernike(element=4, mode=33, coef_m=10e-9)
    states.append(s)

    # 5. Elt 17 Tx +20 um (downstream of the coronagraph)
    s = SystemState(layout=CORO_LAYOUT, name="Elt17_Tx_plus_20um")
    s.set_dof(element=17, dof="transx", value=20e-6)
    states.append(s)

    return states


def driver() -> int:
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from proper_compare.aberrations import CORO_LAYOUT, SystemState
    from proper_compare.contrast import (
        lambda_over_D_pixels, plot_contrast_curves, radial_contrast)

    LAM_C    = 850e-9
    BAND_FRAC = 0.10
    N_WAVE   = 7
    fractions = np.linspace(-BAND_FRAC / 2, BAND_FRAC / 2, N_WAVE)
    wavelengths = (LAM_C * (1.0 + fractions)).tolist()
    print(f"[driver] centre λ = {LAM_C*1e9:.1f} nm; band = "
          f"{BAND_FRAC*100:.1f}%; {N_WAVE} samples")

    rx_dir = Path(__file__).resolve().parents[1] / "Rx"
    rx_path = rx_dir / "Rx_Coro_FPM_Zern.in"
    out_dir = Path(__file__).resolve().parent / "results_cycle4"
    out_dir.mkdir(parents=True, exist_ok=True)

    states = _make_states()
    npz_paths = {}
    for state in states:
        npz_path = out_dir / f"aberration_{state.name}_broadband.npz"
        npz_paths[state.name] = npz_path
        print(f"\n[driver] === case: {state.name} ===")
        _spawn_state(rx_path, wavelengths, state, npz_path)

    # ----- Strehl reference: use the un-coronagraphed broadband PSF
    #       so contrast values are reported as "fraction of the star's
    #       un-suppressed light".  Loaded from no_mask_broadband.npz
    #       (run_broadband_nominal must have run first).  If it isn't
    #       present, run the no-mask broadband sweep inline.
    no_mask_ref = out_dir / "no_mask_broadband.npz"
    if no_mask_ref.exists():
        ref = np.load(no_mask_ref)
        I_ref_macos = ref["I_macos_sum"]
    else:
        print(f"\n[driver] no_mask broadband reference not found at "
              f"{no_mask_ref}; running it now ...")
        ref_npz = out_dir / "no_mask_broadband.npz"
        rx_no_mask = rx_dir / "Rx_Coro_noLyot.in"
        # Use a nominal SystemState (no perturbation) on the no-mask Rx.
        nominal_state = SystemState(layout=CORO_LAYOUT, name="no_mask_ref")
        _spawn_state(rx_no_mask, wavelengths, nominal_state, ref_npz)
        I_ref_macos = np.load(ref_npz)["I_macos_sum"]

    peak_ref = float(I_ref_macos.max())
    lam_D    = float(lambda_over_D_pixels(I_ref_macos))
    print(f"\n[driver] broadband no-mask peak (Strehl ref) = {peak_ref:.4e}; "
          f"λ/D = {lam_D:.2f} px")

    # ----- 3-panel focal-plane plot + per-state contrast curve.
    contrast_curves = {}
    digest = []
    for state in states:
        arrs = np.load(npz_paths[state.name])
        I_m = arrs["I_macos_sum"]
        I_p = arrs["I_proper_sum"]
        ref_dx = float(arrs["ref_dx_macos"])

        _save_four_panel(state.name, I_m, I_p, ref_dx,
                         out_dir / f"aberration_{state.name}"
                                   f"_broadband_focalplane.png")

        r_ld, c = radial_contrast(I_m, peak_ref, lam_D,
                                  max_lambda_over_D=20.0)
        contrast_curves[state.name.replace("_", " ")] = (r_ld, c)

        # Strehl-normed pairwise agreement on the broadband sum
        a = I_m / I_m.max() if I_m.max() > 0 else I_m
        b = I_p / I_p.max() if I_p.max() > 0 else I_p
        d = np.abs(a - b)
        digest.append((state.name, float(I_m.max()),
                       float(d.max()),
                       float(c[int(np.argmin(np.abs(r_ld - 3.0)))]),
                       float(c[int(np.argmin(np.abs(r_ld - 7.0)))])))

    plot_contrast_curves(
        contrast_curves,
        out_dir / "aberration_broadband_contrast.png",
        title=("Cycle 4: broadband contrast under perturbations\n"
               "10% band, 7 λ, resampled-then-summed at centre-λ pitch"),
        ylim=(1e-14, 1e-2),
    )

    print(f"\n  {'state':<35s}  {'peak macos':>11s}  "
          f"{'macos-PROPER':>13s}  {'@ 3 λ/D':>10s}  {'@ 7 λ/D':>10s}")
    for name, peak, agree, c3, c7 in digest:
        print(f"  {name:<35s}  {peak:>11.3e}  {agree:>13.3e}  "
              f"{c3:>10.3e}  {c7:>10.3e}")
    print(f"\n[driver] wrote results to {out_dir}/")
    return 0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _spawn_state(rx_path: Path, wavelengths: list, state, out_npz: Path):
    args = [sys.executable, "-m",
            "proper_compare.run_broadband_aberrations",
            "--worker", str(rx_path),
            ",".join(f"{lam:.10e}" for lam in wavelengths),
            state_to_json(state),
            str(out_npz)]
    cwd = Path(__file__).resolve().parents[1]
    res = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError(
            f"worker failed (state={state.name}, "
            f"code={res.returncode})")
    for line in res.stdout.splitlines():
        if line.startswith("[worker"):
            print(line)


def _save_four_panel(state_name: str,
                     I_macos: 'np.ndarray',
                     I_proper: 'np.ndarray',
                     dx_m: float,
                     out_path: Path,
                     window_lambda_over_D: float = 30.0):
    """4-panel diagnostic: macos / PROPER (top); Δ / radial profile
    overlay (bottom).

    Top row: log10 intensity, each on its own peak scale (same
    dynamic range).  Bottom-left: signed Strehl-normed difference
    (macos − PROPER) with a FIXED color range (±1e-3 Strehl) so
    genuine agreement reads as nearly-uniform colour -- avoids the
    auto-stretch lie where sub-precision noise looks like big
    structure.  Bottom-right: radial-profile overlay (log y) plus
    signed difference on a twin axis.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from proper_compare.contrast import radial_profile

    N = I_macos.shape[0]
    # Use N//2 (not (N-1)//2) as the crop center so the clamp
    # half_window_px=N//2 yields lo=0 instead of lo=-1 (which would
    # negative-index slice into the last row and silently miss the
    # PSF core).
    cy = cx = N // 2
    half_window_um = window_lambda_over_D * 50.0
    half_window_px = int(half_window_um * 1e-6 / dx_m)
    half_window_px = min(half_window_px, N // 2)
    lo = cy - half_window_px
    hi = cy + half_window_px + 1
    crop_m = I_macos[lo:hi, lo:hi]
    crop_p = I_proper[lo:hi, lo:hi]

    peak_m = max(crop_m.max(), 1e-30)
    peak_p = max(crop_p.max(), 1e-30)
    floor_dec = 10
    log_m = np.log10(np.maximum(crop_m, peak_m * 10**(-floor_dec)))
    log_p = np.log10(np.maximum(crop_p, peak_p * 10**(-floor_dec)))

    diff = crop_m / peak_m - crop_p / peak_p
    diff_max_abs = float(np.abs(diff).max())
    # Auto-stretch the diff colormap to the data range (per Dave's
    # preference: see the actual data; numerical readout in the title
    # tells you the absolute magnitude).
    diff_lim = max(diff_max_abs, 1e-30)

    ext_um = (-half_window_px * dx_m * 1e6,
               (hi - cy - 1) * dx_m * 1e6,
              -half_window_px * dx_m * 1e6,
               (hi - cy - 1) * dx_m * 1e6)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    im0 = axes[0, 0].imshow(log_m, extent=ext_um, origin="lower",
                             cmap="viridis",
                             vmin=np.log10(peak_m) - floor_dec,
                             vmax=np.log10(peak_m))
    axes[0, 0].set_title(f"macos broadband\npeak = {peak_m:.3e}")
    axes[0, 0].set_xlabel("x (μm)"); axes[0, 0].set_ylabel("y (μm)")
    plt.colorbar(im0, ax=axes[0, 0], label=r"log$_{10}$ I",
                  fraction=0.046)

    im1 = axes[0, 1].imshow(log_p, extent=ext_um, origin="lower",
                             cmap="viridis",
                             vmin=np.log10(peak_p) - floor_dec,
                             vmax=np.log10(peak_p))
    axes[0, 1].set_title(f"PROPER broadband\npeak = {peak_p:.3e}")
    axes[0, 1].set_xlabel("x (μm)"); axes[0, 1].set_ylabel("y (μm)")
    plt.colorbar(im1, ax=axes[0, 1], label=r"log$_{10}$ I",
                  fraction=0.046)

    im2 = axes[1, 0].imshow(diff, extent=ext_um, origin="lower",
                             cmap="RdBu_r",
                             vmin=-diff_lim, vmax=diff_lim)
    axes[1, 0].set_title(
        f"macos/peak − PROPER/peak\n"
        f"max|Δ| = {diff_max_abs:.3e} (auto-stretch ±max|Δ|)")
    axes[1, 0].set_xlabel("x (μm)"); axes[1, 0].set_ylabel("y (μm)")
    plt.colorbar(im2, ax=axes[1, 0], label="Δ (Strehl-norm)",
                  fraction=0.046)

    a = I_macos / I_macos.max() if I_macos.max() > 0 else I_macos
    b = I_proper / I_proper.max() if I_proper.max() > 0 else I_proper
    r_px, mean_a, _, _ = radial_profile(a, max_radius=half_window_px,
                                         bin_size=1.0)
    _,    mean_b, _, _ = radial_profile(b, max_radius=half_window_px,
                                         bin_size=1.0)
    r_um = r_px * dx_m * 1e6

    ax_rad = axes[1, 1]
    ax_rad.semilogy(r_um, np.maximum(mean_a, 1e-18),
                    color="C0", lw=1.5, label="macos / peak")
    ax_rad.semilogy(r_um, np.maximum(mean_b, 1e-18),
                    color="C1", lw=1.5, ls="--", label="PROPER / peak")
    ax_rad.set_xlabel("radius (μm)")
    ax_rad.set_ylabel("Strehl-normed radial mean (log)")
    ax_rad.legend(loc="upper right")
    ax_rad.grid(alpha=0.3, which="both")
    ax_rad.set_title("Radial profile overlay + signed Δ (twin axis)")

    ax_diff = ax_rad.twinx()
    signed = mean_a - mean_b
    ax_diff.plot(r_um, signed, color="C3", lw=1.0, alpha=0.7)
    ax_diff.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax_diff.set_ylabel("signed Δ macos − PROPER (linear)", color="C3")
    ax_diff.tick_params(axis="y", labelcolor="C3")
    sd_lim = float(np.nanmax(np.abs(signed))) if np.isfinite(
        np.nanmax(np.abs(signed))) else 1e-12
    sd_lim = max(sd_lim, 1e-12)
    ax_diff.set_ylim(-2 * sd_lim, 2 * sd_lim)

    fig.suptitle(f"{state_name}  --  broadband (10% band, 7 λ summed)",
                 y=1.00, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[driver] wrote {out_path}")


def _entry() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        # --worker RX_PATH WAVELENGTHS_COMMA STATE_JSON OUT_NPZ
        rx_path        = Path(sys.argv[2])
        wavelengths_si = [float(s) for s in sys.argv[3].split(",")]
        state_json     = sys.argv[4]
        out_npz        = Path(sys.argv[5])
        worker(rx_path, wavelengths_si, state_json, out_npz)
        return 0
    return driver()


if __name__ == "__main__":
    sys.exit(_entry())
