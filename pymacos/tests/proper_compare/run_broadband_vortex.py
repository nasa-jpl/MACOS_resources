"""Cycle 5: broadband vortex coronagraph (oversized-rays scheme).

Runs nominal + comparative aberration cases through a vortex
coronagraph at the focal plane (Elt 9 of
Rx_Coro_FPM_Zern_vortex_oversized.in, which has Element=Reference /
nObs=0 so the vortex is the only focal-plane element).

The "oversized" Rx doubles the source aperture and removes the front-
end ray clips on Elts 1/4/7/12 so the focal-plane vortex's pupil-
domain ring-of-fire isn't zeroed out by macos's geometric prop chain
between Elt 10 and Elt 14.  The 21 mm design entrance pupil is
enforced as a WFElt apodization at Elt 4 via apodize_complex, applied
each wavelength inside the worker.  Lyot is enlarged to 20 mm (the
sweet spot from the radius sweep against a charge-4 vortex).

The mask is applied via the new pymacos.apodize_complex wrapper
(sibling of apodize, takes a complex NxN array).  Two vortex flavors:

  - VECTOR (default): the Pancharatnam-Berry geometric phase mask.
    Wavelength-independent -- the SAME exp(i*charge*theta) at every
    wavelength.  Models multi-twist LCP / sub-wavelength-grating
    devices that hold half-wave behaviour over 20%+ bandwidths.

  - SCALAR: a refractive helical-thickness mask with effective charge
    that varies with wavelength as ℓ_eff = ℓ_design · (λ_design / λ).
    Different mask at each wavelength; non-integer effective charges
    away from λ_design produce chromatic leakage.

For each (case, vortex_mode), runs the full Phase-5 chain at 7
wavelengths across 10% band, resamples each PSF to centre-λ pitch,
sums, and saves 4-panel + contrast curves into results_cycle5/.

PROPER side just propagates macos's post-vortex cfield from Elt 20 to
Elt 21 -- the vortex itself is purely a macos-side operation (via
apodize_complex at Elt 9), and PROPER consumes the result.  Same
handoff pattern as Phase 5.2.

NOTE: the oversized-rays scheme is what makes this work end-to-end.
The standard Rx_Coro_FPM_Zern_vortex.in gives only ~6.7e3 peak
suppression because the post-FPM ring-of-fire (the bulk of the
vortex's pupil-domain effect) falls outside the geometric ray bundle
and gets zeroed at the first geometric prop after Elt 10.  Oversize
plus the WFElt-only entrance-pupil clip keeps the ring-of-fire on
alive-ray pixels through the relay; the Lyot at Elt 14 does the
actual coronagraph clipping.  Result at nominal, charge-4 vector:
1.84e4 peak suppression.
"""
from __future__ import absolute_import

import json
import subprocess
import sys
from pathlib import Path
from typing import List


# ----------------------------------------------------------------------
# SystemState JSON serialisation (copied from run_broadband_aberrations)
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
# Worker: one (case, vortex_mode, charge), all wavelengths
# ----------------------------------------------------------------------

def worker(rx_path: Path, wavelengths_si: List[float], state_json: str,
           vortex_mode: str, charge: int, lambda_design_m: float,
           out_npz: Path) -> None:
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    import pymacos.macos as m
    from proper_compare.aberrations import apply_system_state
    from proper_compare.apodizer import (
        vortex_phase_mask, scalar_vortex_phase_mask)
    from proper_compare.geometries.coro_nfprop import (
        CoroSphereToPlane, proper_run_sphere_to_plane)

    N = 1024
    PUPIL_R_MM = 21.0  # design entrance pupil radius enforced at Elt 4
    state = state_from_json(state_json)
    print(f"[worker {state.name} vortex={vortex_mode} ℓ={charge}] start",
          flush=True)

    m.init(N)
    m.load(str(rx_path))
    apply_system_state(m, state)

    ok_cbm, cbm = m.lib.api.base_unit_to_metres()
    if not ok_cbm or cbm == 0.0:
        raise RuntimeError("worker: CBM unavailable")
    si_to_wave_units = 1.0 / float(cbm)

    n_wave = len(wavelengths_si)
    I_macos_native   = np.zeros((n_wave, N, N))
    I_proper_native  = np.zeros((n_wave, N, N))
    dx_macos         = np.zeros(n_wave)
    dx_proper        = np.zeros(n_wave)

    # Pre-build the 21-mm hard-edge entrance-pupil mask on the Elt 4 grid.
    # macos's pupil-domain dx is wavelength-independent, so this mask
    # works at every wavelength.
    _ = m.complex_field(4)
    dx4_mm = m.dx_at(4) * 1e3
    yy, xx = np.indices((N, N))
    cy = cx = (N - 1) / 2.0
    r_mm = np.sqrt((xx - cx)**2 + (yy - cy)**2) * dx4_mm
    entrance_mask = np.where(r_mm <= PUPIL_R_MM, 1.0 + 0j, 0.0 + 0j)

    # Vector vortex: build the mask once outside the wavelength loop.
    if vortex_mode == "vector":
        vortex_mask_static = vortex_phase_mask(N, charge=charge)
    else:
        vortex_mask_static = None

    for i, lam in enumerate(wavelengths_si):
        m.src_wvl(lam * si_to_wave_units)

        # Re-trace to Elt 4 (resets state) and apply entrance apodization
        # so the design 21 mm pupil amplitude is enforced before the
        # focal-plane vortex.
        _ = m.complex_field(4)
        m.apodize_complex(4, entrance_mask)

        # Continue to Elt 9 with reset_trace=False to preserve the
        # entrance apodization.
        m.intensity(9, reset_trace=False)

        # Apply the vortex mask at the focal plane.
        if vortex_mode == "vector":
            mask = vortex_mask_static
        elif vortex_mode == "scalar":
            mask = scalar_vortex_phase_mask(
                N, charge=charge,
                lambda_design_m=lambda_design_m,
                lambda_m=lam)
        else:
            raise ValueError(f"unknown vortex_mode={vortex_mode!r}")
        m.apodize_complex(9, mask)

        # Continue downstream (reset_trace=False to keep the apodised
        # WFElt(9)).
        cfield_pupil = m.complex_field(20, reset_trace=False)
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

        print(f"[worker {state.name} {vortex_mode}ℓ{charge} "
              f"λ={lam*1e9:.2f}nm] "
              f"peak_macos={I_focal.max():.4e}  "
              f"peak_proper={I_proper_i.max():.4e}", flush=True)

    # Per-λ peak match (PROPER side) + resample to centre-λ pitch
    peak_m_per = np.array([I_macos_native[i].max() for i in range(n_wave)])
    peak_p_per = np.array([I_proper_native[i].max() for i in range(n_wave)])
    scale = np.where(peak_p_per > 0, peak_m_per / peak_p_per, 0.0)
    I_proper_rescaled = I_proper_native * scale[:, None, None]

    ref_idx = n_wave // 2
    ref_dx_m = dx_macos[ref_idx]
    ref_dx_p = dx_proper[ref_idx]
    I_macos_resampled  = np.zeros_like(I_macos_native)
    I_proper_resampled = np.zeros_like(I_proper_native)
    for i in range(n_wave):
        I_macos_resampled[i] = resample_to_grid(
            I_macos_native[i], dx_macos[i], ref_dx_m)
        I_proper_resampled[i] = resample_to_grid(
            I_proper_rescaled[i], dx_proper[i], ref_dx_p)
    I_macos_sum  = I_macos_resampled.sum(axis=0)
    I_proper_sum = I_proper_resampled.sum(axis=0)

    np.savez(out_npz,
             I_macos_native=I_macos_native,
             I_proper_native=I_proper_native,
             I_macos_resampled=I_macos_resampled,
             I_proper_resampled=I_proper_resampled,
             I_macos_sum=I_macos_sum,
             I_proper_sum=I_proper_sum,
             dx_macos_per_wave=dx_macos,
             dx_proper_per_wave=dx_proper,
             ref_dx_macos=ref_dx_m,
             ref_dx_proper=ref_dx_p,
             wavelengths_m=np.array(wavelengths_si),
             charge=charge,
             vortex_mode=vortex_mode,
             lambda_design_m=lambda_design_m,
             state_json=state_json)
    print(f"[worker {state.name} {vortex_mode}ℓ{charge}] wrote {out_npz}  "
          f"broadband peak = {I_macos_sum.max():.4e}", flush=True)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def _make_states():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from proper_compare.aberrations import CORO_LAYOUT, SystemState

    states = [SystemState(layout=CORO_LAYOUT, name="nominal")]
    return states


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
    print(f"[driver] centre λ = {LAM_C*1e9:.1f} nm, band = "
          f"{BAND_FRAC*100:.0f}%, {N_WAVE} samples")

    rx_dir = Path(__file__).resolve().parents[1] / "Rx"
    rx_path = rx_dir / "Rx_Coro_FPM_Zern_vortex_oversized.in"
    out_dir = Path(__file__).resolve().parent / "results_cycle5"
    out_dir.mkdir(parents=True, exist_ok=True)

    charge = 4
    states = _make_states()

    # Per-case run for each vortex_mode
    npz_paths = {}   # (state_name, vortex_mode) -> path
    for state in states:
        for vortex_mode in ("vector", "scalar"):
            key = f"{state.name}_{vortex_mode}"
            npz = out_dir / f"vortex_{key}_charge{charge}_broadband.npz"
            npz_paths[key] = npz
            print(f"\n[driver] === {state.name} / {vortex_mode} vortex "
                  f"(ℓ={charge}) ===")
            _spawn(rx_path, wavelengths, state, vortex_mode, charge,
                   LAM_C, npz)

    # Also need a no_mask broadband reference for Strehl normalisation.
    no_mask_npz = out_dir / "no_mask_broadband.npz"
    if not no_mask_npz.exists():
        print(f"\n[driver] no_mask reference missing -- building")
        rx_no_mask = rx_dir / "Rx_Coro_FPM_Zern_vortex_oversized_noLyot.in"
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from proper_compare.aberrations import CORO_LAYOUT, SystemState
        nominal_state = SystemState(layout=CORO_LAYOUT, name="no_mask")
        _spawn(rx_no_mask, wavelengths, nominal_state, "vector",
               0, LAM_C, no_mask_npz)
        # charge=0 + vector mode -> mask is exp(i*0*θ) = 1 everywhere.
        # That's a no-op vortex on the noLyot oversized Rx, so the
        # result is the un-vortexed, un-Lyotted broadband sum -- the
        # right Strehl reference for the oversized scheme (same
        # entrance-pupil apodization as the coronagraph cases).

    ref = np.load(no_mask_npz)
    peak_ref = float(ref["I_macos_sum"].max())
    lam_D    = float(lambda_over_D_pixels(ref["I_macos_sum"]))
    print(f"\n[driver] broadband no-mask peak (Strehl ref) = {peak_ref:.4e}; "
          f"λ/D = {lam_D:.2f} px")

    # Per-case contrast + 4-panel plot
    curves = {}
    digest = []
    for state in states:
        for vortex_mode in ("vector", "scalar"):
            key = f"{state.name}_{vortex_mode}"
            arrs = np.load(npz_paths[key])
            I_m = arrs["I_macos_sum"]; I_p = arrs["I_proper_sum"]
            ref_dx = float(arrs["ref_dx_macos"])
            _save_four_panel(key, I_m, I_p, ref_dx,
                              out_dir / f"vortex_{key}_broadband_focalplane.png")

            r_ld, c = radial_contrast(I_m, peak_ref, lam_D,
                                      max_lambda_over_D=20.0)
            label = f"{state.name}  {vortex_mode} vortex ℓ{charge}"
            curves[label] = (r_ld, c)

            a = I_m / I_m.max() if I_m.max() > 0 else I_m
            b = I_p / I_p.max() if I_p.max() > 0 else I_p
            d = np.abs(a - b)
            digest.append((key, float(I_m.max()), float(d.max()),
                           float(c[int(np.argmin(np.abs(r_ld - 3.0)))]),
                           float(c[int(np.argmin(np.abs(r_ld - 7.0)))])))

    plot_contrast_curves(
        curves,
        out_dir / f"vortex_charge{charge}_broadband_contrast.png",
        title=(f"Cycle 5: vortex coronagraph contrast (charge {charge})\n"
               f"10% band, {N_WAVE} λ summed, vector vs scalar "
               f"flavours"),
        ylim=(1e-14, 1e-2),
    )

    print(f"\n  {'case':<35s}  {'peak macos':>11s}  "
          f"{'macos-PROPER':>13s}  {'@ 3 λ/D':>10s}  {'@ 7 λ/D':>10s}")
    for name, peak, agree, c3, c7 in digest:
        print(f"  {name:<35s}  {peak:>11.3e}  {agree:>13.3e}  "
              f"{c3:>10.3e}  {c7:>10.3e}")
    print(f"\n[driver] wrote results to {out_dir}/")
    return 0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _spawn(rx_path: Path, wavelengths: list, state, vortex_mode: str,
           charge: int, lambda_design_m: float, out_npz: Path):
    args = [sys.executable, "-m", "proper_compare.run_broadband_vortex",
            "--worker", str(rx_path),
            ",".join(f"{lam:.10e}" for lam in wavelengths),
            state_to_json(state),
            vortex_mode, str(int(charge)),
            f"{lambda_design_m:.10e}",
            str(out_npz)]
    cwd = Path(__file__).resolve().parents[1]
    res = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError(
            f"worker failed (state={state.name}, "
            f"vortex={vortex_mode}, code={res.returncode})")
    for line in res.stdout.splitlines():
        if line.startswith("[worker"):
            print(line)


def _save_four_panel(state_name, I_macos, I_proper, dx_m, out_path,
                     window_lambda_over_D=30.0):
    """Same 4-panel layout as run_broadband_aberrations._save_four_panel.
    Inlined here for module independence.
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
    # PSF core, capping crop_m.max() at the wing brightness).
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
    plt.colorbar(im0, ax=axes[0, 0], label=r"log$_{10}$ I", fraction=0.046)

    im1 = axes[0, 1].imshow(log_p, extent=ext_um, origin="lower",
                             cmap="viridis",
                             vmin=np.log10(peak_p) - floor_dec,
                             vmax=np.log10(peak_p))
    axes[0, 1].set_title(f"PROPER broadband\npeak = {peak_p:.3e}")
    axes[0, 1].set_xlabel("x (μm)"); axes[0, 1].set_ylabel("y (μm)")
    plt.colorbar(im1, ax=axes[0, 1], label=r"log$_{10}$ I", fraction=0.046)

    im2 = axes[1, 0].imshow(diff, extent=ext_um, origin="lower",
                             cmap="RdBu_r",
                             vmin=-diff_lim, vmax=diff_lim)
    axes[1, 0].set_title(f"macos/peak − PROPER/peak\n"
                         f"max|Δ| = {diff_max_abs:.3e}  "
                         f"(auto-stretch)")
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
        # --worker RX_PATH WAVELENGTHS STATE_JSON VORTEX_MODE CHARGE LAMBDA_DESIGN OUT_NPZ
        rx_path        = Path(sys.argv[2])
        wavelengths_si = [float(s) for s in sys.argv[3].split(",")]
        state_json     = sys.argv[4]
        vortex_mode    = sys.argv[5]
        charge         = int(sys.argv[6])
        lambda_design  = float(sys.argv[7])
        out_npz        = Path(sys.argv[8])
        worker(rx_path, wavelengths_si, state_json,
               vortex_mode, charge, lambda_design, out_npz)
        return 0
    return driver()


if __name__ == "__main__":
    sys.exit(_entry())
