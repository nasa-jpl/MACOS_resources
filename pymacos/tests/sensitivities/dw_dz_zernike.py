"""Compute dw/dz of the exit-pupil OPD wavefront vs Zernike-form
coefficient perturbations on every applicable optic in an Rx.

Driver:
  - Loads the prescription (default: e5hex1.in, 7 hex segments + 1
    FreeForm lens + an m2 Zernike reflector).
  - Auto-discovers loop bounds from the Rx itself (no hardcoding).
  - Builds channels for any combination of the three Zernike-form
    coefficient arrays:
      * MonZernCoef on FreeForm  (--kinds monzern)  [default]
      * FFZernCoef  on FreeForm  (--kinds ffzern)
      * ZernCoef    on Zernike / ZrnGridData  (--kinds zern)
    or any union (``--kinds monzern,zern`` etc.).
  - Wavefront vector: exit-pupil OPD (default: second-to-last
    element, override with --exit-pupil-elt).
  - Computes the Jacobian by central finite differences and saves a
    .npz plus a panel figure.

Use as a template: dw/dx (rigid-body), dw/ddm (DM actuators),
dw/dsource etc. can be plugged in by adding new channel classes in
channels.py and calling sensitivity_jacobian with them.
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
                   default=Path("/home/dcr/dev/macos/ZGD_test_files/e5hex1.in"),
                   help="prescription path (default: e5hex1.in)")
    p.add_argument("--model-size", type=int, default=128,
                   choices=(128, 256, 512, 1024),
                   help="pymacos diffraction grid (default: 128)")
    p.add_argument("--exit-pupil-elt", type=int, default=None,
                   help="element id at which to evaluate the wavefront; "
                        "default: second-to-last element")
    p.add_argument("--kinds", type=str, default="monzern",
                   help="comma-separated subset of {monzern,ffzern,zern}; "
                        "default: monzern")
    p.add_argument("--delta", type=float, default=1e-6,
                   help="perturbation step in Rx BaseUnits (default 1e-6; "
                        "for BaseUnits=mm that's 1 nm)")
    p.add_argument("--method", choices=("central", "forward"),
                   default="central",
                   help="finite-difference method (default central)")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parent / "results",
                   help="output directory for .npz + .png")
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
    from sensitivities.jacobian import sensitivity_jacobian  # noqa: E402

    # --- Setup ---------------------------------------------------------
    print(f"[setup] init({args.model_size}); load {args.rx}")
    m.init(args.model_size)
    m.load(str(args.rx))

    n_elt = m.num_elt()
    wf_elt = (args.exit_pupil_elt if args.exit_pupil_elt is not None
              else n_elt - 1)
    print(f"[setup] nElt={n_elt}; wavefront evaluated at Elt {wf_elt}")

    # --- Channel discovery --------------------------------------------
    requested = {k.strip().lower() for k in args.kinds.split(",")}
    unknown = requested - {"monzern", "ffzern", "zern"}
    if unknown:
        raise SystemExit(f"unknown --kinds entries: {sorted(unknown)}")

    channels = []
    if "monzern" in requested:
        ch = freeform_monzern_channels(m, str(args.rx))
        print(f"[setup] MonZern  : {len(ch)} channels")
        channels += ch
    if "ffzern" in requested:
        ch = freeform_ffzern_channels(m, str(args.rx))
        print(f"[setup] FFZern   : {len(ch)} channels")
        channels += ch
    if "zern" in requested:
        ch = zernike_channels(m, str(args.rx))
        print(f"[setup] ZernCoef : {len(ch)} channels")
        channels += ch

    if not channels:
        print("[setup] no channels found for requested kinds; nothing to do")
        return 1
    print(f"[setup] {len(channels)} total channels: "
          + ", ".join(ch.name for ch in channels))

    # --- Wavefront function -------------------------------------------
    def wf_func():
        m.trace_rays(wf_elt)
        return m.opd()

    # --- Jacobian -----------------------------------------------------
    J, w_nom, names = sensitivity_jacobian(
        channels, wf_func, args.delta, method=args.method, verbose=True)

    # --- Save ----------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag if args.tag is not None else args.rx.stem
    kinds_tag = "_".join(sorted(requested))
    npz_path = args.out_dir / f"dw_dz_{kinds_tag}_{tag}.npz"
    n_grid = int(round(np.sqrt(w_nom.size)))
    np.savez(npz_path,
             J=J, w_nom=w_nom, channel_names=np.array(names),
             n_grid=n_grid, model_size=args.model_size,
             rx=str(args.rx), wf_elt=wf_elt, delta=args.delta,
             method=args.method, kinds=sorted(requested))
    print(f"[save] wrote {npz_path}  (J shape {J.shape})")

    # --- Plot ----------------------------------------------------------
    if not args.no_plot:
        _plot_jacobian_panels(
            J, w_nom, names, n_grid, args.delta,
            args.out_dir / f"dw_dz_{kinds_tag}_{tag}.png", tag, kinds_tag)

    return 0


def _plot_jacobian_panels(J, w_nom, names, n_grid, delta, out_path,
                           tag, kinds_tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Nz = J.shape[1]
    cols = int(np.ceil(np.sqrt(Nz)))
    rows = int(np.ceil(Nz / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows),
                              squeeze=False)
    vmax = float(np.max(np.abs(J)))
    if vmax == 0.0:
        vmax = 1e-30

    for k in range(rows * cols):
        ax = axes[k // cols][k % cols]
        if k >= Nz:
            ax.axis("off")
            continue
        col = J[:, k].reshape(n_grid, n_grid)
        im = ax.imshow(col, origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=+vmax)
        ax.set_title(f"{names[k]}\nRMS={np.sqrt((col**2).mean()):.3e}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(f"dw/dz : exit-pupil OPD vs {kinds_tag}  ({tag})\n"
                 f"finite-diff step = {delta:.3e}  "
                 f"-- columns share colour scale [{-vmax:.2e}, "
                 f"{+vmax:.2e}]",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
