"""Compute dw/dz of the exit-pupil OPD wavefront vs Zernike-form
coefficient perturbations on every Zrn-equipped optic in an Rx.

Output is a MATLAB .mat file ``dwdz_{rx_stem}.mat`` containing the
Jacobian, the nominal wavefront, and the ``indx`` struct that
matches ``~/matlab/m2v.m``'s convention so a downstream MATLAB
workflow can call ``w = m2v(opd, indx)`` on a fresh measurement and
have it line up row-for-row with the columns of ``dwdz``.

Default DOF set: each Zernike-form coefficient mode ``4..n_zcoef`` on
every Zrn-equipped element (skipping piston / tip / tilt which are
degenerate for wavefront control).  An element is "Zrn-equipped" if
it carries any of the perturbation channels:

  * MonZernCoef on FreeForm  (SrfType=14)  -- default kind
  * ZernCoef    on Zernike   (SrfType=8)   -- default kind
  * ZernCoef    on ZrnGrData (SrfType=13)  -- default kind
  * FFZernCoef  on FreeForm  (SrfType=14)  -- opt-in via --kinds ffzern

Channels are emitted element-major (then kind, then mode) so the
ordering matches the order optics appear in the Rx.

Use as a template: dw/dx (rigid-body), dw/ddm (DM actuators),
dw/dsource etc. plug in by adding new channel classes in channels.py
and calling sensitivity_jacobian with them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rx", type=Path,
                   default=Path(__file__).resolve().parents[1] /
                           "Rx" / "e5hex1.in",
                   help="prescription path (default: "
                        "pymacos/tests/Rx/e5hex1.in)")
    p.add_argument("--model-size", type=int, default=128,
                   choices=(128, 256, 512, 1024),
                   help="pymacos diffraction grid (default: 128)")
    p.add_argument("--exit-pupil-elt", type=int, default=None,
                   help="element id at which to evaluate the wavefront; "
                        "default: second-to-last element")
    p.add_argument("--kinds", type=str, default="monzern,zern",
                   help="comma-separated subset of {monzern,ffzern,zern}; "
                        "default: monzern,zern (the perturbation channels; "
                        "ffzern is the figure-description and is rarely a "
                        "control DOF -- opt in if you want it)")
    p.add_argument("--zmode-start", type=int, default=4,
                   help="lowest Zernike mode to perturb (default 4 -- "
                        "skips piston/tip/tilt)")
    p.add_argument("--n-zcoef", type=int, default=15,
                   help="highest Zernike mode to perturb (default 15)")
    p.add_argument("--delta", type=float, default=1e-6,
                   help="finite-difference step in Rx BaseUnits (default "
                        "1e-6; for BaseUnits=mm that's 1 nm)")
    p.add_argument("--method", choices=("central", "forward"),
                   default="central",
                   help="finite-difference method (default central)")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parent / "results",
                   help="output directory for .mat + .png")
    p.add_argument("--tag", type=str, default=None,
                   help="suffix for output filenames (default: Rx stem)")
    p.add_argument("--no-plot", action="store_true",
                   help="skip the panel figure")
    args = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pymacos.macos as m  # noqa: E402

    from sensitivities.channels import (                  # noqa: E402
        freeform_monzern_channels, freeform_ffzern_channels,
        zernike_channels)

    # --- Setup ---------------------------------------------------------
    print(f"[setup] init({args.model_size}); load {args.rx}")
    m.init(args.model_size)
    m.load(str(args.rx))

    n_elt = m.num_elt()
    wf_elt = (args.exit_pupil_elt if args.exit_pupil_elt is not None
              else n_elt - 1)
    print(f"[setup] nElt={n_elt}; wavefront evaluated at Elt {wf_elt}")

    requested_kinds = {k.strip().lower() for k in args.kinds.split(",")}
    unknown = requested_kinds - {"monzern", "ffzern", "zern"}
    if unknown:
        raise SystemExit(f"unknown --kinds entries: {sorted(unknown)}")
    if args.zmode_start < 1 or args.n_zcoef < args.zmode_start:
        raise SystemExit(
            f"need 1 <= --zmode-start ({args.zmode_start}) <= --n-zcoef "
            f"({args.n_zcoef})")

    # Override per-element mode list to [zmode_start..n_zcoef] on
    # every Zrn-equipped element.
    target_modes = list(range(args.zmode_start, args.n_zcoef + 1))
    print(f"[setup] perturbing modes {args.zmode_start}..{args.n_zcoef} "
          f"({len(target_modes)} per element)")

    ff_elts = [int(i) for i in m.findFreeFormElts()]
    monzern_modes = {i: target_modes for i in ff_elts}
    ffzern_modes  = {i: target_modes for i in ff_elts}
    # zernike_channels parses the Rx itself for Zernike/ZrnGrData elt list;
    # passing modes_per_elt={iElt: ...} overrides per-element on each.
    # We can pass for any iElt; the discovery filters to eligibles.
    zern_modes    = {i: target_modes for i in range(1, n_elt + 1)}

    channels = []
    if "monzern" in requested_kinds:
        ch = freeform_monzern_channels(m, str(args.rx),
                                       modes_per_elt=monzern_modes)
        print(f"[setup] MonZern  : {len(ch)} channels")
        channels += ch
    if "ffzern" in requested_kinds:
        ch = freeform_ffzern_channels(m, str(args.rx),
                                      modes_per_elt=ffzern_modes)
        print(f"[setup] FFZern   : {len(ch)} channels")
        channels += ch
    if "zern" in requested_kinds:
        ch = zernike_channels(m, str(args.rx),
                              modes_per_elt=zern_modes)
        print(f"[setup] ZernCoef : {len(ch)} channels")
        channels += ch

    if not channels:
        print("[setup] no channels found; nothing to do")
        return 1

    # Sort element-major (then kind, then mode) so the column order
    # mirrors the Rx's element sequence.
    kind_order = {"MonZern": 0, "Zern": 1, "FFZern": 2}
    channels.sort(key=lambda c: (c.iElt, kind_order.get(c.kind, 99),
                                  c.mode))
    print(f"[setup] {len(channels)} channels: "
          + ", ".join(ch.name for ch in channels))

    # --- Wavefront function -------------------------------------------
    def wf_func() -> np.ndarray:
        m.trace_rays(wf_elt)
        return m.opd()

    # --- Nominal + m2v-compatible index struct -----------------------
    w_nom_2d = wf_func()
    indx, w_nom_vec, nz_flat = _m2v_first_call(w_nom_2d)
    Nw = w_nom_vec.size
    Nz = len(channels)
    print(f"[m2v] mask: {Nw} non-zero pixels in {w_nom_2d.shape} OPD")

    # --- Jacobian (central differences in m2v-vector space) ----------
    dwdz = np.zeros((Nw, Nz), dtype=np.float64)
    names: list[str] = []
    for k, ch in enumerate(channels):
        if args.method == "central":
            ch.apply(+args.delta)
            w_plus = wf_func().flatten(order="F")[nz_flat]
            ch.apply(-args.delta)
            w_minus = wf_func().flatten(order="F")[nz_flat]
            ch.restore()
            dwdz[:, k] = (w_plus - w_minus) / (2.0 * args.delta)
        else:  # forward
            ch.apply(+args.delta)
            w_plus = wf_func().flatten(order="F")[nz_flat]
            ch.restore()
            dwdz[:, k] = (w_plus - w_nom_vec) / args.delta
        col_rms = float(np.sqrt(np.mean(dwdz[:, k] ** 2)))
        names.append(ch.name)
        print(f"[dwdz] {k+1:3d}/{Nz}  {ch.name:24s}  "
              f"RMS dw/dz = {col_rms:.3e}")

    # --- Save .mat -----------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag if args.tag is not None else args.rx.stem
    mat_path = args.out_dir / f"dwdz_{tag}.mat"
    _save_mat(mat_path, dwdz, w_nom_vec, indx, names,
              args.rx, args.delta, args.method, wf_elt,
              args.model_size, args.zmode_start, args.n_zcoef,
              sorted(requested_kinds), w_nom_2d.shape)
    print(f"[save] wrote {mat_path}  (dwdz shape {dwdz.shape})")

    # --- Plot ----------------------------------------------------------
    if not args.no_plot:
        _plot_jacobian_panels(
            dwdz, indx, names, w_nom_2d.shape, args.delta,
            args.out_dir / f"dwdz_{tag}.png", tag,
            ",".join(sorted(requested_kinds)))

    return 0


# ---------------------------------------------------------------------------
# m2v.m equivalents (column-major ordering to match MATLAB find())
# ---------------------------------------------------------------------------

def _m2v_first_call(mat: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """Replicate MATLAB [vec, indx] = m2v(mat) on a 2D OPD matrix.

    MATLAB ``find(mat)`` iterates column-major and returns positions
    of non-zero entries.  Mirror that exactly.

    Returns:
        indx:     dict with 'i' (1-based row), 'j' (1-based col),
                  'size' ([nrows, ncols]).  Compatible with m2v's
                  subsequent calls when loaded into MATLAB.
        vec:      1D array of non-zero values in column-major order.
        nz_flat:  0-based positions in mat.flatten('F') (for reuse in
                  the Python-side dwdz loop).
    """
    nrows, ncols = mat.shape
    flat_F = mat.flatten(order="F")
    nz_flat = np.nonzero(flat_F)[0]            # 0-based positions
    vec = flat_F[nz_flat].astype(np.float64)
    # MATLAB 1-based; save as float64 because MATLAB prefers all loaded
    # variables in double type (sub2ind etc. take doubles fine).
    i_row = (nz_flat %  nrows).astype(np.float64) + 1.0
    j_col = (nz_flat // nrows).astype(np.float64) + 1.0
    indx = {
        "i":    i_row.reshape(-1, 1),    # column vectors (MATLAB style)
        "j":    j_col.reshape(-1, 1),
        "size": np.array([[nrows, ncols]], dtype=np.float64),
    }
    return indx, vec, nz_flat


# ---------------------------------------------------------------------------
# Save / plot
# ---------------------------------------------------------------------------

def _save_mat(mat_path, dwdz, w_nom, indx, names, rx, delta, method,
              wf_elt, model_size, zmode_start, n_zcoef, kinds, mat_shape):
    from scipy.io import savemat

    # MATLAB cellstr -> savemat wants an object ndarray of strings.
    name_arr = np.empty(len(names), dtype=object)
    for k, n in enumerate(names):
        name_arr[k] = n

    # MATLAB strongly prefers double-typed scalars/arrays for loaded
    # variables (avoids surprise int promotions in arithmetic).  Cast
    # everything numeric to float64 here.
    savemat(str(mat_path), {
        "dwdz":          np.asarray(dwdz, dtype=np.float64),
        "w_nom":         np.asarray(w_nom.reshape(-1, 1),
                                     dtype=np.float64),
        "indx":          indx,
        "channel_names": name_arr.reshape(-1, 1),
        "rx":            str(rx),
        "delta":         np.float64(delta),
        "method":        method,
        "wf_elt":        np.float64(wf_elt),
        "model_size":    np.float64(model_size),
        "zmode_start":   np.float64(zmode_start),
        "n_zcoef":       np.float64(n_zcoef),
        "kinds":         np.array(kinds, dtype=object).reshape(-1, 1),
        "mat_shape":     np.array(mat_shape, dtype=np.float64).reshape(
                              -1, 1),
        # Convenience for verify_dwdz.m's pad(..., nGridPts) call;
        # equals mat_shape(1) since macos's OPD grid is always square.
        "nGridPts":      np.float64(mat_shape[0]),
    }, do_compression=True, oned_as="column")


def _plot_jacobian_panels(dwdz, indx, names, mat_shape, delta,
                           out_path, tag, kinds_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Nz = dwdz.shape[1]
    cols = int(np.ceil(np.sqrt(Nz)))
    rows = int(np.ceil(Nz / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.8 * rows),
                              squeeze=False)
    vmax = float(np.max(np.abs(dwdz)))
    if vmax == 0.0:
        vmax = 1e-30

    # indx i/j are stored as float64 (MATLAB-friendly); cast back to int
    # for numpy fancy indexing.
    i_row = (indx["i"].ravel() - 1).astype(np.int64)
    j_col = (indx["j"].ravel() - 1).astype(np.int64)
    nrows, ncols = mat_shape

    for k in range(rows * cols):
        ax = axes[k // cols][k % cols]
        if k >= Nz:
            ax.axis("off")
            continue
        img = np.zeros((nrows, ncols))
        img[i_row, j_col] = dwdz[:, k]
        im = ax.imshow(img, origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=+vmax)
        ax.set_title(f"{names[k]}\nRMS={np.sqrt((dwdz[:, k]**2).mean()):.3e}",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(f"dwdz : exit-pupil OPD vs {kinds_label}  ({tag})\n"
                 f"fd step = {delta:.3e}  --  shared scale "
                 f"[{-vmax:.2e}, {+vmax:.2e}]",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
