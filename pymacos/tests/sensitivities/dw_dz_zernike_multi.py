"""Multi-field dw/dz Zernike supervisor.

REQUIRED USER INPUTS (must be supplied per Rx — no safe default):
  --rx           prescription path
  --field-x-rad  test field offset along x (direction cosine added to
                 ChfRayDir).  Pick this < the extreme field-x so the
                 test point has margin.  Typical: 1e-6 .. 1e-3 rad.
  --field-y-rad  test field offset along y (may differ from x).

OPTIONAL INPUTS (sensible defaults for dev-loop speed):
  --model-size       (default 128)      diffraction grid size
  --src-sampling     (default Rx's nGridPts)  smaller speeds inner loop
  --exit-pupil-elt   (default nElt-1)   element id for OPD evaluation
  --n-zcoef          (default 15)       highest Zernike mode to perturb
  --zmode-start      (default 4)        lowest mode (skips piston/tip/tilt)
  --kinds            (default monzern,zern)  channel-kind subset
  --delta            (default 1e-6)     finite-difference step
  --method           (default central)  central | forward
  --fields           (default 5-corner) override: file with N rows of
                     'name dx_rad dy_rad tile_row tile_col'
  --out-dir          (default tests/sensitivities/results/)
  --tag              (default Rx stem)  suffix for output filenames
  --no-plot          skip the OPDall + difference panels

Workflow (per Dave's spec, 2026-05-31):
  load Rx -> STOP+FEX -> snapshot nominal ChfRayDir ->
  for each field point:
    set ChfRayDir = unit(ChfRayDir_nom + (dx, dy, 0))  via src_fov
    -> single-field dwdz
  -> stack dwdz blocks -> stack nominal w -> build tiled OPDall ->
     m2v(OPDall) -> .mat with dwdxall / w0_stacked / indxall / field_table

Field-point convention: each field is parameterised by a
DIRECTION-COSINE OFFSET (dx, dy) added to the nominal ChfRayDir then
renormalised — NOT a rigid-frame rotation.  The new chief-ray
direction is:

    ChfRayDir_new = unit(ChfRayDir_nom + (dx, dy, 0))

This matches macos's "tip the source pointing" intuition for a fixed
source position.  Per-axis x and y are independent (`--field-x-rad`
and `--field-y-rad` may differ).  Both are Rx-specific — the user
picks them to land at a useful TEST point inset from the extreme
field corner.

Default field set: 5 points (center + 4 corners) at +/- field offsets.

Tiled OPDall layout (5-field default, 3x3 grid):

    [UL]   [zeros]   [UR]
    [zeros]  [C]   [zeros]
    [LL]   [zeros]   [LR]

Each tile is the per-field nominal OPD (N x N) embedded at its 3x3
position.  Spacers are zeros.  ``plot(OPDall)`` displays the layout
directly — no GridSpec / subplot positioning.

Canonical output names (state-vector control form): the .mat file
records ``dwdxall``, ``w0_stacked``, ``indxall``, ``field_table``
so ``wall = dwdxall * x + w0_stacked`` works straight out of MATLAB.
The Zernike-specific alias ``dwdzall`` is also stored.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def make_5field_set(field_x_rad: float, field_y_rad: float):
    """Default 5-field set: center + 4 corners.

    Each entry: (name, dx_rad, dy_rad, tile_row, tile_col).
    dx/dy are direction-cosine offsets added to the nominal ChfRayDir.
    Tile coords are positions in a 3x3 grid (0..2 each).

    Tile-row convention: 0 = bottom (matplotlib imshow origin="lower"
    places array row 0 at the bottom of the plot, so "upper" fields
    map to row 2 and "lower" to row 0).  Plot then shows UL/UR at the
    TOP and LL/LR at the BOTTOM, matching field-of-view intuition.
    """
    return [
        ("C",  0.0,           0.0,           1, 1),
        ("UL", -field_x_rad,  +field_y_rad,  2, 0),
        ("UR", +field_x_rad,  +field_y_rad,  2, 2),
        ("LL", -field_x_rad,  -field_y_rad,  0, 0),
        ("LR", +field_x_rad,  -field_y_rad,  0, 2),
    ]


def make_grid_field_set(nx: int, ny: int,
                         field_x_rad: float, field_y_rad: float):
    """NxM grid of field points covering [-field_x_rad..+field_x_rad] x
    [-field_y_rad..+field_y_rad].

    nx = # columns (along x-field), ny = # rows (along y-field).
    Each row/col is uniformly spaced; the central row/col passes
    through (0, 0) when nx / ny is odd.

    Names: tile (tr, tc) with the (0,0)-offset point named "C" if
    present, otherwise "F_r{tr}_c{tc}".  Tile placement matches the
    matplotlib origin="lower" convention so "upper" fields appear at
    the top of OPDall.
    """
    fields = []
    # ColumnIdx 0..nx-1 maps to dx = -field_x_rad ..+field_x_rad
    # (uniformly).  nx=1 -> single point at 0.
    if nx > 1:
        dx_axis = np.linspace(-field_x_rad, +field_x_rad, nx)
    else:
        dx_axis = np.array([0.0])
    if ny > 1:
        dy_axis = np.linspace(-field_y_rad, +field_y_rad, ny)
    else:
        dy_axis = np.array([0.0])

    for ir, dy in enumerate(dy_axis):
        for ic, dx in enumerate(dx_axis):
            # Pretty-name the center if it lands at exactly (0, 0).
            is_center = (abs(dx) < 1e-30) and (abs(dy) < 1e-30)
            name = "C" if is_center else f"F_r{ir}_c{ic}"
            fields.append((name, float(dx), float(dy), ir, ic))
    return fields


def _find_center_field_index(fields):
    """Locate the (0,0)-offset entry in a field list.  Returns its
    index, or None if no entry has both dx and dy zero.
    """
    for k, (_name, dx, dy, _tr, _tc) in enumerate(fields):
        if abs(dx) < 1e-30 and abs(dy) < 1e-30:
            return k
    return None


def field_to_chfraydir(dir_nom: np.ndarray, dx_rad: float, dy_rad: float
                        ) -> np.ndarray:
    """Apply a (dx, dy) direction-cosine field offset to ChfRayDir.

    new_dir = unit(dir_nom + (dx, dy, 0))

    Small-angle: for |dx|, |dy| << 1 this matches the linearised
    field-angle response.  No frame-rotation gymnastics — the source
    POSITION stays fixed, only the pointing direction shifts.
    """
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
    p.add_argument("--rx", type=Path,
                   default=Path(__file__).resolve().parents[1] /
                           "Rx" / "FFSegDemoAll.in")
    p.add_argument("--model-size", type=int, default=128,
                   choices=(128, 256, 512, 1024),
                   help="default 128 for dev-cycle speed")
    p.add_argument("--src-sampling", type=int, default=None,
                   help="override macos source-grid sampling (smaller "
                        "than nGridPts speeds the inner loop)")
    p.add_argument("--exit-pupil-elt", type=int, default=None)
    p.add_argument("--field-x-rad", type=float, required=True,
                   help="Rx-specific test field offset along x "
                        "(direction cosine added to ChfRayDir).  "
                        "User picks this < the extreme field-x so the "
                        "test point has margin.  Typical values: 1e-6 "
                        "to 1e-3 rad.")
    p.add_argument("--field-y-rad", type=float, required=True,
                   help="Rx-specific test field offset along y; may "
                        "differ from --field-x-rad.")
    p.add_argument("--fields", type=Path, default=None,
                   help="optional file with N rows of "
                        "'name dx_rad dy_rad tile_row tile_col'.  "
                        "Overrides --grid and the 5-field default.")
    p.add_argument("--grid", type=str, default=None,
                   help="auto-generate an N×M grid (e.g. '3x3' or '5x5') "
                        "covering ±field-x-rad × ±field-y-rad.  "
                        "Overrides the 5-field default.  Center field "
                        "is computed exactly once when both N and M "
                        "are odd.")
    p.add_argument("--n-zcoef", type=int, default=15)
    p.add_argument("--zmode-start", type=int, default=4)
    p.add_argument("--kinds", type=str, default="monzern,zern")
    p.add_argument("--delta", type=float, default=1e-6)
    p.add_argument("--method", choices=("central", "forward"),
                   default="central")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parent / "results")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pymacos.macos as m  # noqa: E402
    from sensitivities.channels import (                  # noqa: E402
        freeform_monzern_channels, freeform_ffzern_channels,
        zernike_channels)
    from sensitivities.dw_dz_zernike import (             # noqa: E402
        dwdz_for_current_source, _m2v_first_call)

    # ---- Load + initial setup --------------------------------------
    print(f"[setup] init({args.model_size}); load {args.rx}")
    m.init(args.model_size)
    m.load(str(args.rx))
    n_elt = m.num_elt()
    wf_elt = (args.exit_pupil_elt if args.exit_pupil_elt is not None
              else n_elt - 1)

    if args.src_sampling is not None:
        print(f"[setup] src sampling -> {args.src_sampling}")
        m.src_sampling(args.src_sampling)

    base_units, _ = m.sys_units()
    ok_cbm, cbm = m.lib.api.base_unit_to_metres()
    cbm = float(cbm) if ok_cbm else 0.0
    if base_units == "none" or cbm == 0.0:
        raise SystemExit(f"** {args.rx}: BaseUnits not declared")

    # Snapshot nominal source FoV BEFORE any field perturbations.
    src_dist_nom, src_pos_nom, src_dir_nom, _src_finite = m.src_fov()
    print(f"[setup] nominal ChfRayDir = {src_dir_nom}; "
          f"ChfRayPos = {src_pos_nom}; zSrc = {src_dist_nom:.3e}")

    # ---- Field set -------------------------------------------------
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
        # "NxM" -> (N, M) where N is columns (x-axis), M is rows (y-axis).
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
        print(f"  field {name:4s}: dir-offset=({dx:+.3e},{dy:+.3e}) rad  "
              f"tile=({tr},{tc})")

    # ---- Build channels --------------------------------------------
    target_modes = list(range(args.zmode_start, args.n_zcoef + 1))
    ff_elts = [int(i) for i in m.findFreeFormElts()]
    requested_kinds = {k.strip().lower() for k in args.kinds.split(",")}
    channels = []
    if "monzern" in requested_kinds:
        channels += freeform_monzern_channels(
            m, str(args.rx),
            modes_per_elt={i: target_modes for i in ff_elts})
    if "ffzern" in requested_kinds:
        channels += freeform_ffzern_channels(
            m, str(args.rx),
            modes_per_elt={i: target_modes for i in ff_elts})
    if "zern" in requested_kinds:
        channels += zernike_channels(
            m, str(args.rx),
            modes_per_elt={i: target_modes for i in range(1, n_elt + 1)})
    if not channels:
        raise SystemExit("no channels found")
    # Canonical order: kind-major (MonZern block, then FFZern block,
    # then Zern block), element-minor within each kind, mode-minor
    # within each element.  Users work with en-bloc Jacobians, so the
    # natural block layout is what the saved .mat ships.  The kind
    # blocks above already appear in that fixed order; sort the
    # (iElt, mode) within each block.  kind_order maps each kind to
    # its appearance index so the existing channel list stays grouped.
    kind_order = {"MonZern": 0, "FFZern": 1, "Zern": 2}
    channels.sort(key=lambda c: (kind_order.get(c.kind, 99),
                                  c.iElt, c.mode))
    Nz = len(channels)
    print(f"[setup] {Nz} channels")

    def wf_func():
        m.trace_rays(wf_elt)
        return m.opd()

    # ---- Per-field loop --------------------------------------------
    # Each iteration uses set_src_fov ABSOLUTELY (not perturb), so no
    # round-trip undo is needed — each field starts from the nominal
    # source state by construction.
    per_field_dwdz = []
    per_field_w_nom_2d = []
    names = None
    for (name, dx, dy, tr, tc) in fields:
        new_dir = field_to_chfraydir(src_dir_nom, dx, dy)
        m.src_fov(src_pos=src_pos_nom, src_dir=new_dir,
                  src_dist=src_dist_nom)
        print(f"[field {name}] ChfRayDir = {new_dir}")

        dwdz_f, w_nom_2d_f, w_nom_vec_f, indx_f, nz_flat_f, names_f = \
            dwdz_for_current_source(channels, wf_func, args.delta,
                                     method=args.method, verbose=False)
        if names is None:
            names = names_f
        per_field_dwdz.append(dwdz_f)
        per_field_w_nom_2d.append(w_nom_2d_f)
        col_rms_mean = float(np.mean(np.sqrt((dwdz_f ** 2).mean(axis=0))))
        print(f"[field {name}] dwdz shape {dwdz_f.shape}, "
              f"mean col-RMS {col_rms_mean:.3e}")

    # Restore source back to nominal for any caller post-conditions.
    m.src_fov(src_pos=src_pos_nom, src_dir=src_dir_nom,
              src_dist=src_dist_nom)

    # ---- Tile OPDall + stack dwdzall -------------------------------
    N = per_field_w_nom_2d[0].shape[0]
    OPDall = np.zeros((tile_rows * N, tile_cols * N), dtype=np.float64)
    for (name, _dx, _dy, tr, tc), w_nom in zip(fields, per_field_w_nom_2d):
        OPDall[tr*N:(tr+1)*N, tc*N:(tc+1)*N] = w_nom

    indxall, w0_stacked, nz_flat_all = _m2v_first_call(OPDall)
    Nw = w0_stacked.size
    print(f"[stack] OPDall {OPDall.shape}; non-zero pixels = {Nw}")

    dwdzall = np.zeros((Nw, Nz), dtype=np.float64)
    indx_i = indxall["i"].ravel().astype(np.int64) - 1
    indx_j = indxall["j"].ravel().astype(np.int64) - 1
    for (name, _dx, _dy, tr, tc), dwdz_f, w_nom_f in zip(
            fields, per_field_dwdz, per_field_w_nom_2d):
        in_tile = ((indx_i >= tr*N) & (indx_i < (tr+1)*N)
                   & (indx_j >= tc*N) & (indx_j < (tc+1)*N))
        i_local = indx_i[in_tile] - tr * N
        j_local = indx_j[in_tile] - tc * N
        _, _, nz_flat_field = _m2v_first_call(w_nom_f)
        flat_local = j_local * N + i_local  # column-major
        row_in_field = np.searchsorted(nz_flat_field, flat_local)
        assert (row_in_field < dwdz_f.shape[0]).all(), \
            f"field {name}: indxall references pixels outside dwdz_f mask"
        assert (nz_flat_field[row_in_field] == flat_local).all(), \
            f"field {name}: row-index lookup mismatch"
        global_rows = np.where(in_tile)[0]
        dwdzall[global_rows, :] = dwdz_f[row_in_field, :]
        print(f"[stack] field {name}: scattered {len(global_rows)} rows "
              f"into dwdzall")

    print(f"[stack] dwdzall shape {dwdzall.shape}; "
          f"|dwdzall| max = {np.max(np.abs(dwdzall)):.3e}")

    # Sanity check: the center-tile rows of dwdzall must equal the
    # per-field dwdz for the center field exactly (no field perturb,
    # so the scatter should be the identity into the C-tile rows).
    # Catches bugs in the searchsorted/scatter logic.  Locates "C" by
    # (dx, dy) ≈ 0 rather than by name so the check works for both the
    # 5-field default and the --grid N×M case where the center field
    # is auto-named.
    center_idx = _find_center_field_index(fields)
    if center_idx is not None:
        ctr_tr, ctr_tc = fields[center_idx][3], fields[center_idx][4]
        in_ctr = ((indx_i >= ctr_tr*N) & (indx_i < (ctr_tr+1)*N)
                  & (indx_j >= ctr_tc*N) & (indx_j < (ctr_tc+1)*N))
        dwdzall_ctr_rows = dwdzall[in_ctr, :]
        dwdz_C = per_field_dwdz[center_idx]
        max_diff = float(np.max(np.abs(dwdzall_ctr_rows - dwdz_C)))
        print(f"[check] dwdzall@center-tile vs per_field_dwdz[center]: "
              f"max|diff| = {max_diff:.3e} ({dwdzall_ctr_rows.shape})")
        assert max_diff == 0.0, (
            f"scatter logic bug: dwdzall@center-tile differs from "
            f"per_field_dwdz[center] by {max_diff:.3e}")
    else:
        print(f"[check] no (0,0)-offset field in the set — skipping "
              f"center-tile sanity check")

    # ---- Field table -----------------------------------------------
    field_table = np.array(
        [[fld[1], fld[2], fld[3], fld[4]] for fld in fields],
        dtype=np.float64)
    field_names = np.empty(n_fields, dtype=object)
    for k, fld in enumerate(fields):
        field_names[k] = fld[0]

    # ---- Save .mat -------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag if args.tag is not None else args.rx.stem
    mat_path = args.out_dir / f"dwdzall_{tag}.mat"
    _save_mat(mat_path, dwdzall, w0_stacked, indxall, names,
              field_table, field_names, src_dir_nom, args.rx, args.delta,
              args.method, wf_elt, args.model_size, args.zmode_start,
              args.n_zcoef, sorted(requested_kinds), OPDall.shape,
              base_units=base_units, cbm=cbm)
    print(f"[save] wrote {mat_path}  (dwdzall {dwdzall.shape}, "
          f"OPDall {OPDall.shape})")

    if not args.no_plot:
        _plot_opdall(OPDall, fields, args.out_dir / f"opdall_{tag}.png",
                     tag)
        _plot_opdall_diff(per_field_w_nom_2d, fields,
                           args.out_dir / f"opdall_diff_{tag}.png", tag)

    return 0


def _save_mat(mat_path, dwdzall, w0_stacked, indxall, names,
              field_table, field_names, src_dir_nom, rx, delta, method,
              wf_elt, model_size, zmode_start, n_zcoef, kinds,
              opdall_shape, base_units="none", cbm=0.0):
    from scipy.io import savemat

    name_arr = np.empty(len(names), dtype=object)
    for k, n in enumerate(names):
        name_arr[k] = n

    savemat(str(mat_path), {
        # Canonical state-vector form: wall = dwdxall * x + w0_stacked
        "dwdxall":      np.asarray(dwdzall, dtype=np.float64),
        # Kind-specific alias (Zernike here):
        "dwdzall":      np.asarray(dwdzall, dtype=np.float64),
        "w0_stacked":   np.asarray(w0_stacked.reshape(-1, 1),
                                    dtype=np.float64),
        "indxall":      indxall,
        "channel_names": name_arr.reshape(-1, 1),
        "field_table":  field_table,   # Nfields x 4: dx_rad, dy_rad, tr, tc
        "field_names":  field_names.reshape(-1, 1),
        "chfraydir_nom": np.asarray(src_dir_nom, dtype=np.float64
                                     ).reshape(-1, 1),
        "rx":           str(rx),
        "delta":        np.float64(delta),
        "method":       method,
        "wf_elt":       np.float64(wf_elt),
        "model_size":   np.float64(model_size),
        "zmode_start":  np.float64(zmode_start),
        "n_zcoef":      np.float64(n_zcoef),
        "kinds":        np.array(kinds, dtype=object).reshape(-1, 1),
        "opdall_shape": np.array(opdall_shape, dtype=np.float64
                                  ).reshape(-1, 1),
        "base_units":   base_units,
        "cbm":          np.float64(cbm),
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
    """Tiled (corner − center) field-response plot.

    Same N×M layout as OPDall but each tile shows the FIELD-DEPENDENT
    DIFFERENCE: per_field_w_nom_at_tile minus per_field_w_nom_at_center.
    Reveals what the per-field nominal OPD response looks like spatially
    — the center tile is zero by construction, and the off-center
    tiles isolate the chief-ray-shift signature from the underlying
    nominal aberration.  If the field set has no (0,0)-offset entry,
    the difference plot is skipped (no reference field to subtract).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    center_idx = _find_center_field_index(fields)
    if center_idx is None:
        print(f"[plot-diff] no (0,0)-offset field — skipping difference plot")
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
