"""Multi-field dw/dx rigid-body supervisor.

Mirror of dw_dz_zernike_multi.py but for the per-element / source /
group rigid-body Jacobian rather than the per-element Zernike one.
Loads the Rx once, builds channels once, then loops field points --
each iteration sets ChfRayDir ABSOLUTELY via src_fov, recomputes
dwdx for the current source, and the per-field results are tiled
into one big OPDall / scattered into one big dwdxall.

REQUIRED USER INPUTS (no safe default; user must supply per Rx):
  --rx           prescription path
  --field-x-rad  test field offset along x (direction cosine added to
                 ChfRayDir).  Pick this < the extreme field-x so the
                 test point has margin.  Typical: 1e-6 .. 1e-3 rad.
  --field-y-rad  test field offset along y (may differ from x).

CHANNEL-SETUP INPUTS (see dw_dx.py for full descriptions; the same
flags are accepted here):
  --stop-elt         element id to declare as the system Stop
  --fp-mode          track | sxp | srs | fex | none  (default: track)
  --ep-elt           EP element (default -1 = nElt-1)
  --update-ep        none | sxp | fex  (re-derive EP before each OPD)
  --include-source   prepend a source-perturbation channel block
  --src-stop-mode    obj | elt | none  (default: obj)
  --src-stop-pos     x,y,z              (default: 0,0,0)
  --include-non-optics    include Reference/Return elements
  --dofs             comma-separated subset of {Rx,Ry,Rz,Tx,Ty,Tz}
                     (default: all 6)
  --rot-output       natural | base-per-rad  (default: natural)

OPTIONAL FIELD / OUTPUT INPUTS:
  --model-size       (default 128)      diffraction grid size
  --src-sampling     (default Rx's nGridPts)
  --exit-pupil-elt   (default nElt-1)
  --delta            (default 1e-8)     finite-difference step (SI)
  --method           central | forward  (default central)
  --fields FILE      override: rows of 'name dx_rad dy_rad tile_row tile_col'
  --grid NxM         auto-generate a uniform N×M grid covering
                     ±field-x-rad × ±field-y-rad.  Center field
                     computed exactly once when both N and M are odd.
  --out-dir          (default tests/sensitivities/results/)
  --tag              (default Rx stem)  suffix for output filenames
  --no-plot          skip the OPDall + difference panels

Outputs (saved to <out-dir>/dwdxall_<tag>.mat):
  dwdxall       Nw × Nz state-vector Jacobian (canonical name).
  w0_stacked    Nw × 1 stacked nominal OPDs (m2v of OPDall).
  indxall       struct with i / j / size — m2v.m round-trip metadata.
  field_table   Nfields × 4: dx_rad, dy_rad, tile_row, tile_col.
  field_names   Nfields × 1 cell array.
  channel_names Nz × 1 cell array.
  chfraydir_nom 3 × 1 nominal ChfRayDir.
  opdall_shape  2 × 1 tiled OPD canvas dimensions.

Plus:
  opdall_<tag>.png         tiled per-field nominal OPDs.
  opdall_diff_<tag>.png    (per-field OPD) - (center OPD); standard.

Field convention: see dw_dz_zernike_multi.py (direction-cosine offset,
unit-renormalised, per-axis x/y).

v1 note: groups (--groups) are NOT yet supported by the supervisor.
Per-Rx use of group channels still works through the single-field
dw_dx.py driver.  Adding group support requires per-field bookkeeping
of the group W synthesis matrices and is deferred.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# Field-set helpers are shared verbatim with dw_dz_zernike_multi.
def make_5field_set(field_x_rad: float, field_y_rad: float):
    return [
        ("C",  0.0,           0.0,           1, 1),
        ("UL", -field_x_rad,  +field_y_rad,  2, 0),
        ("UR", +field_x_rad,  +field_y_rad,  2, 2),
        ("LL", -field_x_rad,  -field_y_rad,  0, 0),
        ("LR", +field_x_rad,  -field_y_rad,  0, 2),
    ]


def make_grid_field_set(nx: int, ny: int,
                         field_x_rad: float, field_y_rad: float):
    fields = []
    dx_axis = (np.linspace(-field_x_rad, +field_x_rad, nx) if nx > 1
                else np.array([0.0]))
    dy_axis = (np.linspace(-field_y_rad, +field_y_rad, ny) if ny > 1
                else np.array([0.0]))
    for ir, dy in enumerate(dy_axis):
        for ic, dx in enumerate(dx_axis):
            is_center = (abs(dx) < 1e-30) and (abs(dy) < 1e-30)
            name = "C" if is_center else f"F_r{ir}_c{ic}"
            fields.append((name, float(dx), float(dy), ir, ic))
    return fields


def _find_center_field_index(fields):
    for k, (_name, dx, dy, _tr, _tc) in enumerate(fields):
        if abs(dx) < 1e-30 and abs(dy) < 1e-30:
            return k
    return None


def field_to_chfraydir(dir_nom: np.ndarray, dx_rad: float, dy_rad: float
                        ) -> np.ndarray:
    candidate = np.asarray(dir_nom, dtype=float) + np.array(
        [dx_rad, dy_rad, 0.0], dtype=float)
    norm = np.linalg.norm(candidate)
    if norm == 0:
        raise ValueError("zero-magnitude direction after field offset")
    return candidate / norm


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Rx + field-set inputs (REQUIRED).
    p.add_argument("--rx", type=Path, required=True)
    p.add_argument("--field-x-rad", type=float, required=True)
    p.add_argument("--field-y-rad", type=float, required=True)
    p.add_argument("--fields", type=Path, default=None,
                   help="override field set with a file (rows of "
                        "'name dx_rad dy_rad tile_row tile_col')")
    p.add_argument("--grid", type=str, default=None,
                   help="auto-generate an N×M grid (e.g. '3x3').  "
                        "Center field is computed exactly once when "
                        "both N and M are odd.")
    # Macos init / OPD-evaluation knobs.
    p.add_argument("--model-size", type=int, default=128,
                   choices=(128, 256, 512, 1024))
    p.add_argument("--src-sampling", type=int, default=None)
    p.add_argument("--exit-pupil-elt", type=int, default=None)
    # Channel-setup knobs (mirrored from dw_dx.py).
    p.add_argument("--stop-elt", type=int, default=None,
                   help="declare element ID as the system Stop "
                        "(STOP <iElt>).  Mutually exclusive with "
                        "--stop-obj-pos.")
    p.add_argument("--stop-obj-pos", type=str, default=None,
                   help="declare an OBJECT-SPACE Stop at position "
                        "'x,y,z' (STOP obj <x>,<y>,<z>).  Use this "
                        "for Rx files without an ApStop= declaration "
                        "when you want SXP/FEX-based EP follow-up to "
                        "work.  Re-applied per field so the new "
                        "chief-ray direction is re-aimed through the "
                        "object-space stop.  Mutually exclusive with "
                        "--stop-elt.")
    p.add_argument("--fp-mode",
                   choices=("track", "sxp", "srs", "fex", "none"),
                   default="track")
    p.add_argument("--ep-elt", type=int, default=-1)
    p.add_argument("--update-ep", choices=("none", "sxp", "fex"),
                   default="none")
    p.add_argument("--include-source", action="store_true")
    p.add_argument("--src-stop-mode", choices=("obj", "elt", "none"),
                   default="obj")
    p.add_argument("--src-stop-pos", type=str, default="0,0,0")
    p.add_argument("--include-non-optics", action="store_true")
    p.add_argument("--dofs", type=str, default="Rx,Ry,Rz,Tx,Ty,Tz")
    p.add_argument("--rot-output",
                   choices=("natural", "base-per-rad"), default="natural")
    # FD knobs.
    p.add_argument("--delta", type=float, default=1e-8)
    p.add_argument("--method", choices=("central", "forward"),
                   default="central")
    # Output knobs.
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parent / "results")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pymacos.macos as m  # noqa: E402
    from sensitivities.channels import (                  # noqa: E402
        rigid_body_channels, source_channels, _RB_DOF_LABELS)
    from sensitivities.dw_dx import (                     # noqa: E402
        dwdx_for_current_source, _m2v_first_call)

    # ---- Load + initial setup --------------------------------------
    print(f"[setup] init({args.model_size}); load {args.rx}")
    m.init(args.model_size)
    m.load(str(args.rx))
    n_elt = m.num_elt()
    wf_elt = (args.exit_pupil_elt if args.exit_pupil_elt is not None
              else n_elt - 1)

    if args.src_sampling is not None:
        m.src_sampling(args.src_sampling)

    base_units, _ = m.sys_units()
    ok_cbm, cbm = m.lib.api.base_unit_to_metres()
    cbm = float(cbm) if ok_cbm else 0.0
    if base_units == "none" or cbm == 0.0:
        raise SystemExit(
            f"** {args.rx}: BaseUnits not declared.  Add a "
            f"'BaseUnits=' line to the Rx header.")
    print(f"[setup] BaseUnits={base_units!r}, CBM={cbm:g} m/BaseUnit; "
          f"--rot-output={args.rot_output}")

    if args.stop_elt is not None and args.stop_obj_pos is not None:
        raise SystemExit("--stop-elt and --stop-obj-pos are mutually exclusive")
    sys_stop_obj_pos: tuple[float, float, float] | None = None
    if args.stop_elt is not None:
        m.stop(int(args.stop_elt))
        print(f"[setup] Stop set to Elt {args.stop_elt}")
    elif args.stop_obj_pos is not None:
        try:
            sys_stop_obj_pos = tuple(
                float(x) for x in args.stop_obj_pos.split(","))
            if len(sys_stop_obj_pos) != 3:
                raise ValueError
        except ValueError:
            raise SystemExit(
                f"--stop-obj-pos must be 'x,y,z'; "
                f"got {args.stop_obj_pos!r}")
        m.stop_obj(*sys_stop_obj_pos)
        print(f"[setup] Object-space Stop set at "
              f"{sys_stop_obj_pos}")

    # ---- Parse --dofs -----------------------------------------------
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

    # ---- Snapshot nominal source ------------------------------------
    src_dist_nom, src_pos_nom, src_dir_nom, _ = m.src_fov()
    print(f"[setup] nominal ChfRayDir = {src_dir_nom}; "
          f"ChfRayPos = {src_pos_nom}; zSrc = {src_dist_nom:.3e}")

    # ---- Build channels (source + per-element) ----------------------
    src_channels: list = []
    if args.include_source:
        try:
            src_stop_pos = tuple(
                float(x) for x in args.src_stop_pos.split(","))
            if len(src_stop_pos) != 3:
                raise ValueError
        except ValueError:
            raise SystemExit(
                f"--src-stop-pos must be 'x,y,z'; "
                f"got {args.src_stop_pos!r}")
        src_stop_elt = int(args.stop_elt) if args.stop_elt else 0
        if args.src_stop_mode == "elt" and src_stop_elt <= 0:
            raise SystemExit(
                "--src-stop-mode=elt requires --stop-elt to be set")
        src_channels = source_channels(
            m, dofs=dofs_requested,
            stop_mode=args.src_stop_mode,
            stop_obj_pos=src_stop_pos,
            stop_elt=src_stop_elt)
        # Establish chief-ray-through-stop baseline so the nominal OPD
        # in the first field matches the source channel's apply()
        # measurements.
        if args.src_stop_mode == "obj":
            m.stop_obj(*src_stop_pos)
        elif args.src_stop_mode == "elt":
            m.stop(src_stop_elt)
        print(f"[setup] source channels ({len(src_channels)}): "
              f"stop_mode={args.src_stop_mode}")

    channels = src_channels + rigid_body_channels(
        m, str(args.rx),
        dofs=dofs_requested,
        fp_mode=args.fp_mode,
        ep_elt=int(args.ep_elt),
        include_non_optics=bool(args.include_non_optics))
    if not channels:
        raise SystemExit("no channels found")
    n_src = len(src_channels)
    n_elt_actual = (len(channels) - n_src) // n_dof
    print(f"[setup] {n_elt_actual} actual optics × {n_dof} DOFs "
          f"= {n_elt_actual * n_dof} per-element channels"
          + (f"; +{n_src} source DOFs" if n_src else ""))

    # ---- Per-column unit rescaler -----------------------------------
    def _output_scale(ch_) -> float:
        if (hasattr(ch_, "dof_idx")
                and 0 <= int(ch_.dof_idx) <= 2
                and args.rot_output == "base-per-rad"):
            return 1.0
        return cbm

    # ---- Wavefront function -----------------------------------------
    update_ep = args.update_ep

    def wf_func() -> np.ndarray:
        if update_ep == "sxp":
            m.sxp()
        elif update_ep == "fex":
            m.fex()
        m.trace_rays(wf_elt)
        return m.opd()

    # ---- Field set --------------------------------------------------
    if args.fields is not None:
        fields = []
        with open(args.fields) as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                fields.append((parts[0], float(parts[1]), float(parts[2]),
                               int(parts[3]), int(parts[4])))
    elif args.grid is not None:
        try:
            nx_str, ny_str = args.grid.lower().split("x")
            nx, ny = int(nx_str), int(ny_str)
        except Exception:
            raise SystemExit(f"--grid must be 'NxM' (e.g. '3x3'); got "
                             f"{args.grid!r}")
        fields = make_grid_field_set(nx, ny, args.field_x_rad,
                                      args.field_y_rad)
    else:
        fields = make_5field_set(args.field_x_rad, args.field_y_rad)
    n_fields = len(fields)
    tile_rows = max(f[3] for f in fields) + 1
    tile_cols = max(f[4] for f in fields) + 1
    print(f"[setup] {n_fields} field points, tile grid "
          f"{tile_rows}x{tile_cols}")
    for (name, dx, dy, tr, tc) in fields:
        print(f"  field {name:8s}: dir-offset=({dx:+.3e},{dy:+.3e}) rad  "
              f"tile=({tr},{tc})")

    # ---- Per-field loop ---------------------------------------------
    per_field_dwdx = []
    per_field_w_nom_2d = []
    names = None
    for (name, dx, dy, tr, tc) in fields:
        new_dir = field_to_chfraydir(src_dir_nom, dx, dy)
        m.src_fov(src_pos=src_pos_nom, src_dir=new_dir,
                  src_dist=src_dist_nom)
        # Re-enforce stop for the new chief-ray direction (per-channel
        # apply() does this too, but doing it here once gives a clean
        # nominal-state measurement for w_nom_2d).
        if sys_stop_obj_pos is not None:
            m.stop_obj(*sys_stop_obj_pos)
        elif args.stop_elt is not None:
            m.stop(int(args.stop_elt))
        if args.include_source:
            if args.src_stop_mode == "obj":
                m.stop_obj(*src_stop_pos)
            elif args.src_stop_mode == "elt":
                m.stop(int(args.stop_elt))
        print(f"[field {name}] ChfRayDir = {new_dir}")

        dwdx_f, w_nom_2d_f, w_nom_vec_f, indx_f, nz_flat_f, names_f = \
            dwdx_for_current_source(channels, wf_func, args.delta,
                                     method=args.method,
                                     output_scale_fn=_output_scale,
                                     verbose=False)
        if names is None:
            names = names_f
        per_field_dwdx.append(dwdx_f)
        per_field_w_nom_2d.append(w_nom_2d_f)
        col_rms_mean = float(np.mean(np.sqrt((dwdx_f ** 2).mean(axis=0))))
        print(f"[field {name}] dwdx shape {dwdx_f.shape}, "
              f"mean col-RMS {col_rms_mean:.3e}")

    # Restore source back to nominal.
    m.src_fov(src_pos=src_pos_nom, src_dir=src_dir_nom,
              src_dist=src_dist_nom)

    # ---- Tile OPDall + scatter dwdxall ------------------------------
    N = per_field_w_nom_2d[0].shape[0]
    OPDall = np.zeros((tile_rows * N, tile_cols * N), dtype=np.float64)
    for (name, _dx, _dy, tr, tc), w_nom in zip(fields, per_field_w_nom_2d):
        OPDall[tr*N:(tr+1)*N, tc*N:(tc+1)*N] = w_nom

    indxall, w0_stacked, nz_flat_all = _m2v_first_call(OPDall)
    Nw = w0_stacked.size
    Nz = len(channels)
    print(f"[stack] OPDall {OPDall.shape}; non-zero pixels = {Nw}")

    dwdxall = np.zeros((Nw, Nz), dtype=np.float64)
    indx_i = indxall["i"].ravel().astype(np.int64) - 1
    indx_j = indxall["j"].ravel().astype(np.int64) - 1
    for (name, _dx, _dy, tr, tc), dwdx_f, w_nom_f in zip(
            fields, per_field_dwdx, per_field_w_nom_2d):
        in_tile = ((indx_i >= tr*N) & (indx_i < (tr+1)*N)
                   & (indx_j >= tc*N) & (indx_j < (tc+1)*N))
        i_local = indx_i[in_tile] - tr * N
        j_local = indx_j[in_tile] - tc * N
        _, _, nz_flat_field = _m2v_first_call(w_nom_f)
        flat_local = j_local * N + i_local  # column-major
        row_in_field = np.searchsorted(nz_flat_field, flat_local)
        assert (row_in_field < dwdx_f.shape[0]).all(), \
            f"field {name}: indxall references pixels outside dwdx_f mask"
        assert (nz_flat_field[row_in_field] == flat_local).all(), \
            f"field {name}: row-index lookup mismatch"
        global_rows = np.where(in_tile)[0]
        dwdxall[global_rows, :] = dwdx_f[row_in_field, :]
        print(f"[stack] field {name}: scattered {len(global_rows)} rows "
              f"into dwdxall")

    print(f"[stack] dwdxall shape {dwdxall.shape}; "
          f"|dwdxall| max = {np.max(np.abs(dwdxall)):.3e}")

    # Sanity check: center-tile rows of dwdxall must equal the per-field
    # dwdx for the center field exactly.
    center_idx = _find_center_field_index(fields)
    if center_idx is not None:
        ctr_tr, ctr_tc = fields[center_idx][3], fields[center_idx][4]
        in_ctr = ((indx_i >= ctr_tr*N) & (indx_i < (ctr_tr+1)*N)
                  & (indx_j >= ctr_tc*N) & (indx_j < (ctr_tc+1)*N))
        dwdxall_ctr_rows = dwdxall[in_ctr, :]
        dwdx_C = per_field_dwdx[center_idx]
        max_diff = float(np.max(np.abs(dwdxall_ctr_rows - dwdx_C)))
        print(f"[check] dwdxall@center-tile vs per_field_dwdx[center]: "
              f"max|diff| = {max_diff:.3e} ({dwdxall_ctr_rows.shape})")
        assert max_diff == 0.0, (
            f"scatter logic bug: dwdxall@center-tile differs from "
            f"per_field_dwdx[center] by {max_diff:.3e}")
    else:
        print(f"[check] no (0,0)-offset field — skipping center-tile check")

    # ---- Field table + save -----------------------------------------
    field_table = np.array(
        [[fld[1], fld[2], fld[3], fld[4]] for fld in fields],
        dtype=np.float64)
    field_names = np.empty(n_fields, dtype=object)
    for k, fld in enumerate(fields):
        field_names[k] = fld[0]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag if args.tag is not None else args.rx.stem
    mat_path = args.out_dir / f"dwdxall_{tag}.mat"
    _save_mat(mat_path, dwdxall, w0_stacked, indxall, names,
              field_table, field_names, src_dir_nom, args.rx, args.delta,
              args.method, wf_elt, args.model_size,
              [_RB_DOF_LABELS[d] for d in dofs_requested],
              n_elt_actual, n_src, OPDall.shape,
              args.fp_mode, args.update_ep, args.rot_output,
              base_units=base_units, cbm=cbm)
    print(f"[save] wrote {mat_path}  (dwdxall {dwdxall.shape}, "
          f"OPDall {OPDall.shape})")

    if not args.no_plot:
        _plot_opdall(OPDall, fields, args.out_dir / f"opdall_{tag}.png",
                     tag)
        _plot_opdall_diff(per_field_w_nom_2d, fields,
                           args.out_dir / f"opdall_diff_{tag}.png", tag)

    return 0


def _save_mat(mat_path, dwdxall, w0_stacked, indxall, names,
              field_table, field_names, src_dir_nom, rx, delta, method,
              wf_elt, model_size, dof_labels, n_elt_actual, n_src,
              opdall_shape, fp_mode, update_ep, rot_output,
              base_units="none", cbm=0.0):
    from scipy.io import savemat

    name_arr = np.empty(len(names), dtype=object)
    for k, n in enumerate(names):
        name_arr[k] = n
    dof_arr = np.empty(len(dof_labels), dtype=object)
    for k, n in enumerate(dof_labels):
        dof_arr[k] = n

    savemat(str(mat_path), {
        "dwdxall":       np.asarray(dwdxall, dtype=np.float64),
        "w0_stacked":    np.asarray(w0_stacked.reshape(-1, 1),
                                     dtype=np.float64),
        "indxall":       indxall,
        "channel_names": name_arr.reshape(-1, 1),
        "field_table":   field_table,
        "field_names":   field_names.reshape(-1, 1),
        "chfraydir_nom": np.asarray(src_dir_nom, dtype=np.float64
                                     ).reshape(-1, 1),
        "dof_labels":    dof_arr.reshape(-1, 1),
        "n_dof":         np.float64(len(dof_labels)),
        "n_src":         np.float64(n_src),
        "n_elt":         np.float64(n_elt_actual),
        "rx":            str(rx),
        "delta":         np.float64(delta),
        "method":        method,
        "wf_elt":        np.float64(wf_elt),
        "model_size":    np.float64(model_size),
        "fp_mode":       fp_mode,
        "update_ep":     update_ep,
        "rot_output":    rot_output,
        "opdall_shape":  np.array(opdall_shape, dtype=np.float64
                                   ).reshape(-1, 1),
        "base_units":    base_units,
        "cbm":           np.float64(cbm),
    }, do_compression=True, oned_as="column")


def _plot_opdall(OPDall, fields, out_path, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    vmax = float(np.max(np.abs(OPDall)))
    if vmax == 0:
        vmax = 1e-30
    im = ax.imshow(OPDall, origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=+vmax, interpolation="nearest")
    n_tile_rows = max(f[3] for f in fields) + 1
    N_actual = OPDall.shape[0] // n_tile_rows
    for (name, _dx, _dy, tr, tc) in fields:
        cy = tr * N_actual + N_actual // 2
        cx = tc * N_actual + N_actual // 2
        ax.text(cx, cy + N_actual * 0.4, name,
                ha="center", va="bottom", fontsize=10,
                color="black", weight="bold")
    ax.set_title(f"OPDall tiled layout — {tag}\n"
                 f"per-field nominal OPD tiled at (tile_row, tile_col); "
                 f"zeros between\n"
                 f"shared scale [{-vmax:.2e}, {+vmax:.2e}]",
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def _plot_opdall_diff(per_field_w_nom_2d, fields, out_path, tag):
    """Tiled (per-field − center) field-response plot.

    Same N×M layout as OPDall but each tile shows the FIELD-DEPENDENT
    DIFFERENCE: per_field_w_nom_at_tile minus per_field_w_nom_at_center.
    Reveals the chief-ray-shift signature isolated from the underlying
    nominal aberration.  Center tile = 0 by construction.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    center_idx = _find_center_field_index(fields)
    if center_idx is None:
        print(f"[plot-diff] no (0,0)-offset field — skipping")
        return
    w_C = per_field_w_nom_2d[center_idx]
    N = w_C.shape[0]

    n_tile_rows = max(f[3] for f in fields) + 1
    n_tile_cols = max(f[4] for f in fields) + 1
    DiffAll = np.zeros((n_tile_rows * N, n_tile_cols * N), dtype=np.float64)
    per_tile_rms = []
    for k, (name, _dx, _dy, tr, tc) in enumerate(fields):
        diff = per_field_w_nom_2d[k] - w_C
        DiffAll[tr*N:(tr+1)*N, tc*N:(tc+1)*N] = diff
        per_tile_rms.append((name, tr, tc,
                              float(np.sqrt(np.mean(diff ** 2)))))

    fig, ax = plt.subplots(figsize=(8, 8))
    vmax = float(np.max(np.abs(DiffAll)))
    if vmax == 0:
        vmax = 1e-30
    im = ax.imshow(DiffAll, origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=+vmax, interpolation="nearest")
    for (name, tr, tc, rms) in per_tile_rms:
        cy = tr * N + N // 2
        cx = tc * N + N // 2
        ax.text(cx, cy + N * 0.4,
                f"{name}\nRMS={rms:.2e}",
                ha="center", va="bottom", fontsize=9,
                color="black", weight="bold")
    ax.set_title(f"OPDall field response — {tag}\n"
                 f"(per-field nominal OPD) − (center nominal OPD); "
                 f"center tile = 0 by construction\n"
                 f"shared scale [{-vmax:.2e}, {+vmax:.2e}]",
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
