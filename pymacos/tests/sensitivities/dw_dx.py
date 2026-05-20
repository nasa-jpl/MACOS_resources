"""Compute dw/dx of the exit-pupil OPD wavefront vs rigid-body
perturbations on every actual optic in an Rx.

Per Dave's spec, each optic's perturbation vector x is 6x1 ordered as

    (Rx, Ry, Rz, Tx, Ty, Tz)

  -- x rotation, y rotation, z rotation about the element's local
     frame (radians);
  -- x, y, z translation along its local frame (SI metres at the
     Python boundary; pymacos.perturb converts to BaseUnits internally).

The full x vector stacks Elt 1 ... Elt n in element order, so a
prescription with N actual optics yields a 6N-DOF Jacobian.

"Actual optic" follows the same convention GMI's `prb` channel uses:
every Element= kind EXCEPT Reference and Return.  Source is a special
case (not an Element entry in the .in file) and is skipped here.

DOF layout matches GMI.F's `DARG(1:6) = prb(k+1:k+6)` (rotations
1..3, translations 4..6) via CPERTURB_PROG / pymacos.perturb().

Output:
  dwdx_<rx_stem>.mat  -- m2v.m-compatible Jacobian + indx + w_nom
  dwdx_<rx_stem>.png  -- panel figure: rows = optics, cols = 6 DOFs
"""
from __future__ import absolute_import

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

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
                        "default: second-to-last element (the XP convention)")
    p.add_argument("--stop-elt", type=int, default=None,
                   help="if set, declare this element as the system Stop "
                        "before the Jacobian sweep (required for "
                        "FocalPlane channels in --fp-mode=fex, unless "
                        "the Rx already declares ApStop=).  Object-space "
                        "stops aren't supported via CLI -- add "
                        "ApStop= x y z to the Rx source section instead.")
    p.add_argument("--fp-mode", choices=("track", "sxp", "srs", "fex"),
                   default="track",
                   help="how to handle focal-plane perturbations.  "
                        "track (default): perturb FP AND the EP element "
                        "together so the XP sphere is dragged with the "
                        "moving FP -- matches the physics that the EP "
                        "is a sphere centered on the FP, gives non-zero "
                        "sensitivities for FP Tx/Ty (tilt), Tz (focus), "
                        "and the rotations.  srs: perturb FP, then SRS "
                        "EP-to-FP so macos recomputes the EP pose from "
                        "the new chief-ray geometry (more principled "
                        "than track; needs a Stop set).  fex: perturb "
                        "FP then FEX -- diagnostic only, gives zero "
                        "sensitivities on this Rx because macos's FEX "
                        "EP definition doesn't move with the FP.")
    p.add_argument("--ep-elt", type=int, default=-1,
                   help="EP element id (for --fp-mode=track or srs); "
                        "default -1 = nElt-1 (the standard XP convention).")
    p.add_argument("--update-ep", choices=("none", "sxp", "fex"),
                   default="none",
                   help="If not 'none', run SXP or FEX before each OPD "
                        "measurement so the EP element is re-derived "
                        "from the (possibly perturbed) upstream chief-"
                        "ray geometry.  Captures EP shifts caused by "
                        "perturbations to upstream optics that the FP-"
                        "only modes miss.  Requires a Stop to be set.  "
                        "Off by default because SXP/FEX can be unstable "
                        "for some Rxes.  NOTE: SXP/FEX overwrite the EP "
                        "vpt, so this conflicts with --fp-mode=track on "
                        "FP channels (the wf-side update undoes the "
                        "lateral EP motion that track relies on).  Pair "
                        "with --fp-mode=sxp for consistent EP handling "
                        "across both FP and upstream channels.")
    p.add_argument("--dofs", type=str, default="Rx,Ry,Rz,Tx,Ty,Tz",
                   help="comma-separated subset of {Rx,Ry,Rz,Tx,Ty,Tz} "
                        "(default: all 6, element-major then DOF-minor)")
    p.add_argument("--delta", type=float, default=1e-9,
                   help="finite-difference step.  Interpreted as radians "
                        "for rotation DOFs and SI metres for translation "
                        "DOFs (default 1e-9 -- gives ~nm-scale wavefront "
                        "response on typical reflectors)")
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
        rigid_body_channels, _RB_DOF_LABELS)

    # --- Setup ---------------------------------------------------------
    print(f"[setup] init({args.model_size}); load {args.rx}")
    m.init(args.model_size)
    m.load(str(args.rx))

    n_elt = m.num_elt()
    wf_elt = (args.exit_pupil_elt if args.exit_pupil_elt is not None
              else n_elt - 1)
    print(f"[setup] nElt={n_elt}; wavefront evaluated at Elt {wf_elt}")

    if args.stop_elt is not None:
        m.stop(int(args.stop_elt))
        print(f"[setup] Stop set to Elt {args.stop_elt}")

    # Parse --dofs into integer indices.
    dof_label_to_idx = {lab: i for i, lab in enumerate(_RB_DOF_LABELS)}
    dofs_requested: list[int] = []
    for tok in args.dofs.split(","):
        lab = tok.strip()
        if lab not in dof_label_to_idx:
            raise SystemExit(
                f"unknown --dofs entry {lab!r}; expected one of "
                f"{list(_RB_DOF_LABELS)}")
        dofs_requested.append(dof_label_to_idx[lab])
    n_dof = len(dofs_requested)

    channels = rigid_body_channels(m, str(args.rx),
                                    dofs=dofs_requested,
                                    fp_mode=args.fp_mode,
                                    ep_elt=int(args.ep_elt))
    if not channels:
        print("[setup] no actual-optic elements found in Rx; nothing to do")
        return 1

    n_elt_actual = len(channels) // n_dof
    print(f"[setup] {n_elt_actual} actual optics, {n_dof} DOFs each "
          f"-> {len(channels)} channels")
    print(f"[setup] channel order: "
          + ", ".join(ch.name for ch in channels[:min(12, len(channels))])
          + (", ..." if len(channels) > 12 else ""))

    # --- Wavefront function -------------------------------------------
    # If --update-ep is set, re-derive the EP element (nElt-1) before
    # each OPD measurement.  Upstream-optic perturbations can shift
    # the EP location; without an update the OPD is referenced to
    # the unperturbed EP and that contribution to dw/dx is missed.
    update_ep = args.update_ep
    if update_ep != "none":
        print(f"[setup] EP update mode: {update_ep} (re-derive Elt "
              f"{n_elt - 1} before each OPD)")

    def wf_func() -> np.ndarray:
        if update_ep == "sxp":
            m.sxp()
        elif update_ep == "fex":
            m.fex()
        m.trace_rays(wf_elt)
        return m.opd()

    # --- Nominal + m2v-compatible index struct -----------------------
    w_nom_2d = wf_func()
    indx, w_nom_vec, nz_flat = _m2v_first_call(w_nom_2d)
    Nw = w_nom_vec.size
    Nz = len(channels)
    print(f"[m2v] mask: {Nw} non-zero pixels in {w_nom_2d.shape} OPD")

    # --- Jacobian (central differences in m2v-vector space) ----------
    dwdx = np.zeros((Nw, Nz), dtype=np.float64)
    names: list[str] = []
    for k, ch in enumerate(channels):
        if args.method == "central":
            ch.apply(+args.delta)
            w_plus = wf_func().flatten(order="F")[nz_flat]
            ch.apply(-args.delta)
            w_minus = wf_func().flatten(order="F")[nz_flat]
            ch.restore()
            dwdx[:, k] = (w_plus - w_minus) / (2.0 * args.delta)
        else:  # forward
            ch.apply(+args.delta)
            w_plus = wf_func().flatten(order="F")[nz_flat]
            ch.restore()
            dwdx[:, k] = (w_plus - w_nom_vec) / args.delta
        col_rms = float(np.sqrt(np.mean(dwdx[:, k] ** 2)))
        names.append(ch.name)
        print(f"[dwdx] {k+1:3d}/{Nz}  {ch.name:24s}  "
              f"RMS dw/dx = {col_rms:.3e}")

    # --- Save .mat -----------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag if args.tag is not None else args.rx.stem
    mat_path = args.out_dir / f"dwdx_{tag}.mat"
    _save_mat(mat_path, dwdx, w_nom_vec, indx, names,
              args.rx, args.delta, args.method, wf_elt,
              args.model_size,
              [_RB_DOF_LABELS[d] for d in dofs_requested],
              w_nom_2d.shape, n_elt_actual)
    print(f"[save] wrote {mat_path}  (dwdx shape {dwdx.shape})")

    # --- Plot ----------------------------------------------------------
    if not args.no_plot:
        _plot_jacobian_panels(
            dwdx, indx, names, w_nom_2d.shape, args.delta,
            args.out_dir / f"dwdx_{tag}.png", tag,
            n_elt_actual, n_dof, dofs_requested)

    return 0


# ---------------------------------------------------------------------------
# m2v.m equivalents (column-major; matches the dw/dz driver)
# ---------------------------------------------------------------------------

def _m2v_first_call(mat: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """Replicate MATLAB [vec, indx] = m2v(mat) on a 2D OPD matrix."""
    nrows, ncols = mat.shape
    flat_F = mat.flatten(order="F")
    nz_flat = np.nonzero(flat_F)[0]
    vec = flat_F[nz_flat].astype(np.float64)
    # All numeric fields saved as float64 -- MATLAB prefers loaded
    # scalars/arrays as doubles (see CLAUDE.md).
    i_row = (nz_flat %  nrows).astype(np.float64) + 1.0
    j_col = (nz_flat // nrows).astype(np.float64) + 1.0
    indx = {
        "i":    i_row.reshape(-1, 1),
        "j":    j_col.reshape(-1, 1),
        "size": np.array([[nrows, ncols]], dtype=np.float64),
    }
    return indx, vec, nz_flat


# ---------------------------------------------------------------------------
# Save / plot
# ---------------------------------------------------------------------------

def _save_mat(mat_path, dwdx, w_nom, indx, names, rx, delta, method,
              wf_elt, model_size, dof_labels, mat_shape, n_elt_actual):
    from scipy.io import savemat

    name_arr = np.empty(len(names), dtype=object)
    for k, n in enumerate(names):
        name_arr[k] = n
    dof_arr = np.empty(len(dof_labels), dtype=object)
    for k, n in enumerate(dof_labels):
        dof_arr[k] = n

    savemat(str(mat_path), {
        "dwdx":          np.asarray(dwdx, dtype=np.float64),
        "w_nom":         np.asarray(w_nom.reshape(-1, 1),
                                     dtype=np.float64),
        "indx":          indx,
        "channel_names": name_arr.reshape(-1, 1),
        "dof_labels":    dof_arr.reshape(-1, 1),
        "n_dof":         np.float64(len(dof_labels)),
        "n_elt":         np.float64(n_elt_actual),
        "rx":            str(rx),
        "delta":         np.float64(delta),
        "method":        method,
        "wf_elt":        np.float64(wf_elt),
        "model_size":    np.float64(model_size),
        "mat_shape":     np.array(mat_shape, dtype=np.float64).reshape(
                              -1, 1),
        "nGridPts":      np.float64(mat_shape[0]),
    }, do_compression=True, oned_as="column")


def _plot_jacobian_panels(dwdx, indx, names, mat_shape, delta,
                           out_path, tag, n_elt_actual, n_dof,
                           dofs_requested):
    """Row-per-element, column-per-DOF layout (much more readable than
    sqrt-grid when the channel set has a uniform optic x DOF
    structure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sensitivities.channels import _RB_DOF_LABELS

    Nz = dwdx.shape[1]
    vmax = float(np.max(np.abs(dwdx)))
    if vmax == 0.0:
        vmax = 1e-30

    # indx i/j are float64 (MATLAB convention); cast for numpy fancy
    # indexing.
    i_row = (indx["i"].ravel() - 1).astype(np.int64)
    j_col = (indx["j"].ravel() - 1).astype(np.int64)
    nrows, ncols = mat_shape

    fig, axes = plt.subplots(n_elt_actual, n_dof,
                              figsize=(2.4 * n_dof, 2.3 * n_elt_actual),
                              squeeze=False)
    for k in range(Nz):
        r = k // n_dof
        c = k %  n_dof
        ax = axes[r][c]
        img = np.zeros((nrows, ncols))
        img[i_row, j_col] = dwdx[:, k]
        im = ax.imshow(img, origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=+vmax)
        rms = float(np.sqrt((dwdx[:, k] ** 2).mean()))
        ax.set_title(f"{names[k]}\nRMS={rms:.2e}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    # Hide any leftover axes (shouldn't happen with the uniform grid).
    for k in range(Nz, n_elt_actual * n_dof):
        r = k // n_dof; c = k % n_dof
        axes[r][c].axis("off")

    dof_label_str = ",".join(_RB_DOF_LABELS[d] for d in dofs_requested)
    fig.suptitle(f"dwdx : exit-pupil OPD vs rigid-body x  ({tag})\n"
                 f"DOFs per optic: {dof_label_str}   "
                 f"fd step = {delta:.3e}   "
                 f"shared scale [{-vmax:.2e}, {+vmax:.2e}]",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    # One colourbar on the right
    cax = fig.add_axes([0.92, 0.05, 0.015, 0.9])
    fig.colorbar(im, cax=cax)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
