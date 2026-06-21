"""Compute dw/dKr and dw/dKc of the exit-pupil OPD wavefront vs the base
radius (KrElt) and conic constant (KcElt) of every POWERED optic in an Rx.

A POWERED optic is an ``Element= Reflector`` or ``Refractor`` whose base
radius is real (``|Kr| << 1e22``); flats (fold mirrors, FocalPlane,
Reference, Return) carry the ~1e22 sentinel and are excluded.

Companion to ``dw_dz_zernike.py`` (Zernike-form coefficients): same channel
+ finite-difference machinery, a different DOF set.  One column per
(optic, param) with param in {Kr, Kc}, element-major then param-minor.

Output is a MATLAB ``.mat`` file ``dwds_{rx_stem}.mat`` containing the
Jacobian ``dwds`` (Nw x Ns), the nominal wavefront, and the ``indx`` struct
matching ``m2v.m``'s convention so a downstream MATLAB workflow can call
``w = m2v(opd, indx)`` on a fresh measurement and have it line up
row-for-row with the columns of ``dwds``.
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
                           "Rx" / "Rx_Cass_FarField.in",
                   help="prescription path (default: "
                        "pymacos/tests/Rx/Rx_Cass_FarField.in)")
    p.add_argument("--model-size", type=int, default=256,
                   choices=(128, 256, 512, 1024),
                   help="pymacos diffraction grid (default: 256)")
    p.add_argument("--exit-pupil-elt", type=int, default=None,
                   help="element id at which to evaluate the wavefront; "
                        "default: second-to-last element")
    p.add_argument("--params", type=str, default="Kr,Kc",
                   help="comma-separated subset of {Kr,Kc} (default Kr,Kc)")
    p.add_argument("--delta", type=float, default=1e-6,
                   help="finite-difference step (Kr in BaseUnits, Kc "
                        "dimensionless; default 1e-6)")
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

    from sensitivities.channels import surf_channels         # noqa: E402
    # Reuse the shared FD driver + panel plotter from the Zernike template.
    from dw_dz_zernike import (dwdz_for_current_source,       # noqa: E402
                               _plot_jacobian_panels)

    params = [s.strip() for s in args.params.split(",") if s.strip()]
    bad = [q for q in params if q not in ("Kr", "Kc")]
    if bad:
        raise SystemExit(f"unknown --params entries: {bad} (want Kr/Kc)")

    # --- Setup ---------------------------------------------------------
    print(f"[setup] init({args.model_size}); load {args.rx}")
    m.init(args.model_size)
    m.load(str(args.rx))

    n_elt = m.num_elt()
    wf_elt = (args.exit_pupil_elt if args.exit_pupil_elt is not None
              else n_elt - 1)
    print(f"[setup] nElt={n_elt}; wavefront evaluated at Elt {wf_elt}")

    # BaseUnits sanity (Kr lives in BaseUnits; record the unit context so
    # the saved .mat is self-describing -- consistent with dw_dx/dw_dz).
    base_units, _wave_units = m.sys_units()
    ok_cbm, cbm = m.lib.api.base_unit_to_metres()
    cbm = float(cbm) if ok_cbm else 0.0
    if base_units == "none" or cbm == 0.0:
        raise SystemExit(
            f"** {args.rx}: BaseUnits not declared (sys_units returned "
            f"'{base_units}').  Add a 'BaseUnits=' line to the Rx header.")
    print(f"[setup] Rx BaseUnits = {base_units!r}, CBM = {cbm:g} m/BaseUnit")

    channels = surf_channels(m, str(args.rx), params=params)
    if not channels:
        print("[setup] no powered optics (Reflector/Refractor, |Kr|<<1e22) "
              "found; nothing to do")
        return 1
    # element-major, param-minor (Kr before Kc).
    channels.sort(key=lambda c: (c.iElt, 0 if c.param == "Kr" else 1))
    print(f"[setup] {len(channels)} channels: "
          + ", ".join(ch.name for ch in channels))

    # --- Wavefront function -------------------------------------------
    def wf_func() -> np.ndarray:
        m.trace_rays(wf_elt)
        return m.opd()

    dwds, w_nom_2d, w_nom_vec, indx, nz_flat, names = (
        dwdz_for_current_source(channels, wf_func, args.delta,
                                method=args.method, verbose=True))

    # --- Save .mat -----------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag if args.tag is not None else args.rx.stem
    mat_path = args.out_dir / f"dwds_{tag}.mat"
    _save_mat(mat_path, dwds, w_nom_vec, indx, names, args.rx, args.delta,
              args.method, wf_elt, args.model_size, params, w_nom_2d.shape,
              base_units=base_units, cbm=cbm)
    print(f"[save] wrote {mat_path}  (dwds shape {dwds.shape})")

    # --- Plot ----------------------------------------------------------
    if not args.no_plot:
        _plot_jacobian_panels(
            dwds, indx, names, w_nom_2d.shape, args.delta,
            args.out_dir / f"dwds_{tag}.png", tag, ",".join(params))

    return 0


def _save_mat(mat_path, dwds, w_nom, indx, names, rx, delta, method,
              wf_elt, model_size, params, mat_shape,
              base_units="none", cbm=0.0):
    from scipy.io import savemat

    name_arr = np.empty(len(names), dtype=object)
    for k, n in enumerate(names):
        name_arr[k] = n

    savemat(str(mat_path), {
        "dwds":          np.asarray(dwds, dtype=np.float64),
        "w_nom":         np.asarray(w_nom.reshape(-1, 1), dtype=np.float64),
        "indx":          indx,
        "channel_names": name_arr.reshape(-1, 1),
        "rx":            str(rx),
        "delta":         np.float64(delta),
        "method":        method,
        "wf_elt":        np.float64(wf_elt),
        "model_size":    np.float64(model_size),
        "params":        np.array(params, dtype=object).reshape(-1, 1),
        "mat_shape":     np.array(mat_shape, dtype=np.float64).reshape(-1, 1),
        "nGridPts":      np.float64(mat_shape[0]),
        # dw/dKr is OPD-in-BaseUnits per BaseUnit-of-radius; dw/dKc is
        # OPD-in-BaseUnits per (dimensionless) conic.  Record the unit
        # context like dw_dx/dw_dz; no rescaling is performed.
        "rot_output":    "natural",
        "base_units":    base_units,
        "cbm":           np.float64(cbm),
    }, do_compression=True, oned_as="column")


if __name__ == "__main__":
    sys.exit(main())
