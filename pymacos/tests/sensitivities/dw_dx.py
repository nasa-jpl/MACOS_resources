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
    p.add_argument("--fp-mode",
                   choices=("track", "sxp", "srs", "fex", "none"),
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
                        "EP definition doesn't move with the FP.  "
                        "none: no EP follow-up -- FocalPlane channels "
                        "give zero for all DOFs (FP-only motion leaves "
                        "the EP-referenced OPD unchanged), and group "
                        "channels measure only the rigid-body OPD "
                        "change with the EP element passively dragged "
                        "by GPERTURB (the rigid-camera-motion view, "
                        "without re-deriving the EP from chief ray).")
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
    p.add_argument("--include-non-optics", action="store_true",
                   help="include Reference / Return surfaces in the "
                        "per-element rigid-body block (normally "
                        "excluded as bookkeeping-only).  Needed when "
                        "you want to drive the post-processing tool "
                        "predict_global_rigid_response with a group "
                        "spec that contains Refs/Returns -- without "
                        "their per-element columns the synthesis "
                        "can't reconstruct the rigid-coupling "
                        "cancellations those surfaces participate "
                        "in.  Skip when you just want a 'real "
                        "moveable optics' sensitivity matrix.")
    p.add_argument("--group-coords", choices=("global", "local"),
                   default="global",
                   help="frame in which the group 6-vector is "
                        "interpreted by macos's GPERTURB.  global "
                        "(default): rotations are about the ref "
                        "element's RptElt using GLOBAL axes -- right "
                        "for cross-checking against source "
                        "perturbations (which live in the source's "
                        "global frame).  local: about RptElt using "
                        "ref_elt's TElt frame -- legacy semantics "
                        "from the per-element RigidBodyChannel.")
    p.add_argument("--group-stop-mode",
                   choices=("obj", "elt", "none"), default="obj",
                   help="how the group perturbation re-enforces the "
                        "stop after each GPERTURB.  GPERTURB itself "
                        "doesn't touch the chief ray, so unless the "
                        "group's rotation pivot equals StopPos the "
                        "chief ray drifts off the stop after the "
                        "perturbation.  obj (default): m.stop_obj at "
                        "--group-stop-pos (defaults to 0,0,0).  elt: "
                        "m.stop() at --stop-elt.  none: no re-aim -- "
                        "only safe when ref_elt's RptElt coincides "
                        "with StopPos (e.g. central segment of a "
                        "primary with ApStop= 0 0 0).")
    p.add_argument("--group-stop-pos", type=str, default="0,0,0",
                   help="object-space stop position used when "
                        "--group-stop-mode=obj.  Default '0,0,0' "
                        "matches ApStop= 0 0 0.")
    p.add_argument("--groups", type=str, default="auto",
                   help="Element-group declarations.  "
                        "'auto' (default): parse EltGrp= lines from the "
                        "Rx and emit 6 group channels per declared "
                        "group, AFTER the per-element channels.  'none': "
                        "skip group channels.  Otherwise: a semicolon-"
                        "separated list of 'name=m1,m2,...' specs added "
                        "to whatever was auto-discovered, e.g. "
                        "--groups 'L1L2=9,10;CamA=9,10,11,12,13'.  "
                        "Group membership can overlap (the same element "
                        "in multiple groups) -- the grouping is "
                        "Python-side and doesn't go through macos's "
                        "EltGrp/GPERTURB.")
    p.add_argument("--dofs", type=str, default="Rx,Ry,Rz,Tx,Ty,Tz",
                   help="comma-separated subset of {Rx,Ry,Rz,Tx,Ty,Tz} "
                        "(default: all 6, element-major then DOF-minor)")
    p.add_argument("--include-source", action="store_true",
                   help="prepend a SourceChannel block (iElt=0 in "
                        "macos; perturbs ChfRayDir/ChfRayPos via "
                        "perturb_src).  The chief ray drifts off the "
                        "stop after each source perturbation, so the "
                        "channel re-enforces the stop -- by default "
                        "via stop_obj (object-space ApStop), set "
                        "--src-stop-mode=elt + --stop-elt to use the "
                        "element-stop path instead.")
    p.add_argument("--src-stop-mode", choices=("obj", "elt", "none"),
                   default="obj",
                   help="how the SourceChannel re-enforces the stop "
                        "after a perturb_src.  obj (default): "
                        "stop_obj at --src-stop-pos (defaults to "
                        "0,0,0, matching ApStop=).  elt: stop() at "
                        "--stop-elt.  none: no re-enforcement "
                        "(diagnostic only).")
    p.add_argument("--src-stop-pos", type=str, default="0,0,0",
                   help="comma-separated x,y,z object-space stop "
                        "position used when --src-stop-mode=obj.  "
                        "Default '0,0,0' matches ApStop= 0 0 0.")
    p.add_argument("--delta", type=float, default=1e-8,
                   help="finite-difference step.  Interpreted as radians "
                        "for rotation DOFs and SI metres for translation "
                        "DOFs.  Default 1e-8 -- gives ~10 nm-scale "
                        "wavefront response on typical reflectors, with "
                        "central-diff noise floor ~5e-4 in OPD/unit, "
                        "well below the small-signal Group rotations "
                        "(~1 OPD/rad) on e5hex1.  Drop to 1e-9 if the "
                        "Rx has very high leverage and you see "
                        "nonlinearity; raise to 1e-7 to halve the noise "
                        "floor in group plots when leverage is modest.")
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
        rigid_body_channels, parse_rx_groups,
        grouped_rigid_body_channels,
        source_channels,
        group_synthesis_matrix,
        _RB_DOF_LABELS)

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

    # --- Source channels (optional, prepended before per-element) -----
    # Source perturbations rotate ChfRayDir / translate ChfRayPos.
    # After each perturbation the chief ray needs re-aiming through
    # the system stop to keep the wavefront referenced consistently
    # with the nominal trace.  Default re-enforcement: stop_obj at
    # the prescription's nominal object-space stop (e.g. ApStop=
    # 0 0 0).
    src_channels: list = []
    if args.include_source:
        try:
            src_stop_pos = tuple(
                float(x) for x in args.src_stop_pos.split(","))
            if len(src_stop_pos) != 3:
                raise ValueError
        except ValueError:
            raise SystemExit(
                f"--src-stop-pos must be three floats 'x,y,z'; "
                f"got {args.src_stop_pos!r}")
        # ELT mode needs an explicit stop element (the existing
        # --stop-elt arg is repurposed here when src-stop-mode='elt'
        # since the source channel needs to know WHICH element to
        # re-aim at).
        src_stop_elt = int(args.stop_elt) if args.stop_elt else 0
        if args.src_stop_mode == "elt" and src_stop_elt <= 0:
            raise SystemExit(
                "--src-stop-mode=elt requires --stop-elt to be set")
        src_channels = source_channels(
            m, dofs=dofs_requested,
            stop_mode=args.src_stop_mode,
            stop_obj_pos=src_stop_pos,
            stop_elt=src_stop_elt)
        # Establish the same chief-ray-through-stop baseline before
        # measuring the nominal wavefront, so the source's plus/minus
        # measurements are consistent with the nominal reference.
        if args.src_stop_mode == "obj":
            m.stop_obj(*src_stop_pos)
        elif args.src_stop_mode == "elt":
            m.stop(src_stop_elt)
        print(f"[setup] source channels ({len(src_channels)}): "
              f"stop_mode={args.src_stop_mode}"
              + (f", pos={src_stop_pos}" if args.src_stop_mode == "obj"
                 else f", elt={src_stop_elt}" if args.src_stop_mode == "elt"
                 else ""))

    channels = src_channels + rigid_body_channels(
        m, str(args.rx),
        dofs=dofs_requested,
        fp_mode=args.fp_mode,
        ep_elt=int(args.ep_elt),
        include_non_optics=bool(args.include_non_optics))

    # --- Group channel discovery + extras -----------------------------
    # Per Dave's spec: per-element channels first, then per-group
    # channels.  Auto-discover from Rx EltGrp= lines, then append any
    # extras given via --groups.
    groups_spec = (args.groups or "auto").strip().lower()
    groups: dict[str, tuple[int, ...]] = {}
    if groups_spec == "none":
        pass
    else:
        # 'auto' or 'auto;extras' or just 'name=...,...;name=...'
        if groups_spec == "auto" or groups_spec.startswith("auto"):
            groups.update(parse_rx_groups(str(args.rx)))
            extras_str = (args.groups or "")[4:].lstrip(",;")
        else:
            extras_str = args.groups or ""
        for tok in extras_str.split(";"):
            tok = tok.strip()
            if not tok:
                continue
            if "=" not in tok:
                raise SystemExit(
                    f"--groups entry {tok!r} must be 'name=m1,m2,...'")
            name, body = tok.split("=", 1)
            try:
                members = tuple(int(x) for x in body.split(",")
                                if x.strip())
            except ValueError:
                raise SystemExit(
                    f"--groups: bad member list in {tok!r}")
            if len(members) < 2:
                raise SystemExit(
                    f"--groups: group {name!r} needs >=2 members")
            groups[name.strip()] = members

    if not channels:
        print("[setup] no actual-optic elements found in Rx; nothing to do")
        return 1

    n_src = len(src_channels)
    n_elt_actual = (len(channels) - n_src) // n_dof
    print(f"[setup] {n_elt_actual} actual optics, {n_dof} DOFs each "
          f"-> {n_elt_actual * n_dof} per-element channels"
          + (f"; +{n_src} source DOFs" if n_src else ""))

    # --- Group channels (macos GPERTURB) -----------------------------
    # Groups are perturbed as rigid units by macos's GPERTURB
    # (CPERTURB_GRP_DVR) -- the EP/FP rigid coupling that defeats
    # linear superposition is handled inside macos.  Each group
    # channel snapshots and re-writes the ref element's EltGrp so
    # OVERLAPPING groups work even though EltGrp itself is single-
    # group per element.  When a group contains the FocalPlane
    # element, fp_mode='sxp' (default in 'auto') runs SXP after the
    # GPERTURB to capture the EP-radius update driven by FP motion.
    n_group = len(groups)
    group_channels: list = []
    if groups:
        # 'track' is a FocalPlaneChannel-specific DOF-aware EP-follow
        # mode that has no group analog (a rigid camera assembly
        # already moves its FP via GPERTURB; the EP follow-up question
        # is just "recompute the EP radius/pose from the new chief
        # ray").  Map 'track' -> 'auto' so groups containing the FP
        # default to SXP follow-up; pass other modes through verbatim.
        group_fp_mode = ("auto" if args.fp_mode == "track"
                         else args.fp_mode)
        try:
            grp_stop_pos = tuple(
                float(x) for x in args.group_stop_pos.split(","))
            if len(grp_stop_pos) != 3:
                raise ValueError
        except ValueError:
            raise SystemExit(
                f"--group-stop-pos must be three floats 'x,y,z'; "
                f"got {args.group_stop_pos!r}")
        grp_stop_elt = int(args.stop_elt) if args.stop_elt else 0
        if args.group_stop_mode == "elt" and grp_stop_elt <= 0:
            raise SystemExit(
                "--group-stop-mode=elt requires --stop-elt to be set")
        group_channels = grouped_rigid_body_channels(
            m, groups, dofs=dofs_requested,
            rx_path=str(args.rx),
            fp_mode=group_fp_mode, ep_elt=int(args.ep_elt),
            coords=args.group_coords,
            stop_mode=args.group_stop_mode,
            stop_obj_pos=grp_stop_pos,
            stop_elt=grp_stop_elt)
        print(f"[setup] {n_group} group(s): "
              + ", ".join(f"{n}={list(v)}"
                          for n, v in groups.items()))
        fp_modes_seen = sorted({ch.fp_mode for ch in group_channels})
        print(f"[setup] group FP-mode(s): {fp_modes_seen}")
    channels = channels + group_channels

    print(f"[setup] measuring {len(channels)} total channels "
          + (f"({n_src} source + " if n_src else "(")
          + f"{n_elt_actual * n_dof} individual + "
          f"{n_group * n_dof} group)")
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

    # Per-element + per-group columns were measured in one pass;
    # rename for the save/plot path below to make the intent explicit.
    user_dwdx = dwdx
    user_names = names

    # --- Group synthesis weights (consistency check tool) ------------
    # For each group, build the (N_members * n_dof, 6) matrix W such
    # that the predicted group response for global DOF g is
    #     dwdx_group_pred[:, g] = dwdx_perelt[:, member_dofs] @ W[:, g]
    # The member_dofs vector picks the per-element columns in the same
    # (member, dof) order W is built with.  Saved alongside the
    # Jacobian so verify_dwdx.m can recompute the synthesis directly
    # from the .mat without re-running pymacos.
    group_W_list: list[np.ndarray] = []
    group_member_dof_idx_list: list[np.ndarray] = []
    group_member_idx_list: list[np.ndarray] = []
    if groups:
        # Map (iElt, dof_idx) -> column index in user_dwdx (per-element block).
        per_elt_col_map: dict[tuple[int, int], int] = {}
        for k, n in enumerate(user_names):
            if n.startswith("Elt "):
                parts = n.split()
                ie = int(parts[1])
                d_idx = {"Rx":0,"Ry":1,"Rz":2,"Tx":3,"Ty":4,"Tz":5}[parts[2]]
                per_elt_col_map[(ie, d_idx)] = k

        for gname, gmembers in groups.items():
            members_i = [int(x) for x in gmembers]
            # pivot = RptElt(ref_elt); ref_elt defaults to first member.
            pivot_g = np.asarray(m.elt_rpt(members_i[0])).ravel()
            W = group_synthesis_matrix(
                m, members_i, dofs=dofs_requested, pivot_global=pivot_g)
            group_W_list.append(W)
            # 1-based MATLAB-friendly column indices for the per-element
            # block (column j in MATLAB = user_dwdx[:, j-1] in Python).
            mdofs_python = np.array(
                [per_elt_col_map.get((m_elt, d), -1)
                 for m_elt in members_i for d in dofs_requested],
                dtype=np.int64)
            if (mdofs_python < 0).any():
                missing = [(m_elt, _RB_DOF_LABELS[d])
                           for m_elt in members_i
                           for d in dofs_requested
                           if (m_elt, d) not in per_elt_col_map]
                print(f"[group-W] {gname}: per-element columns missing "
                      f"for {missing} -- W is well-defined but the .mat "
                      f"can't reconstruct dwdx_group from per-element "
                      f"columns alone (rerun with --include-non-optics)")
            group_member_dof_idx_list.append(mdofs_python + 1)  # 1-based for MATLAB
            group_member_idx_list.append(
                np.array(members_i, dtype=np.float64))

            # Quick on-screen sanity: ||W|| per global DOF column.
            col_norms = np.linalg.norm(W, axis=0)
            print(f"[group-W] {gname}: W shape={W.shape}  "
                  f"||W[:, j]|| = "
                  + " ".join(f"{_RB_DOF_LABELS[d]}={col_norms[d]:.3e}"
                             for d in range(6)))

    # --- Save .mat -----------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag if args.tag is not None else args.rx.stem
    mat_path = args.out_dir / f"dwdx_{tag}.mat"
    _save_mat(mat_path, user_dwdx, w_nom_vec, indx, user_names,
              args.rx, args.delta, args.method, wf_elt,
              args.model_size,
              [_RB_DOF_LABELS[d] for d in dofs_requested],
              w_nom_2d.shape, n_elt_actual, n_group, groups,
              n_src=n_src,
              group_W=group_W_list,
              group_member_dof_idx=group_member_dof_idx_list,
              group_member_idx=group_member_idx_list)
    print(f"[save] wrote {mat_path}  (dwdx shape {user_dwdx.shape})")

    # --- Plot ----------------------------------------------------------
    if not args.no_plot:
        # n_src source channels collapse into 1 panel row (DOFs are
        # in columns); per-element and group blocks each contribute
        # one row per element/group.
        n_src_rows = 1 if n_src else 0
        _plot_jacobian_panels(
            user_dwdx, indx, user_names, w_nom_2d.shape, args.delta,
            args.out_dir / f"dwdx_{tag}.png", tag,
            n_src_rows + n_elt_actual + n_group,
            n_dof, dofs_requested)

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
              wf_elt, model_size, dof_labels, mat_shape, n_elt_actual,
              n_group, groups, n_src=0,
              group_W=None, group_member_dof_idx=None,
              group_member_idx=None):
    from scipy.io import savemat

    name_arr = np.empty(len(names), dtype=object)
    for k, n in enumerate(names):
        name_arr[k] = n
    dof_arr = np.empty(len(dof_labels), dtype=object)
    for k, n in enumerate(dof_labels):
        dof_arr[k] = n

    # Group metadata: cell arrays for names + per-group member lists,
    # padded into a (n_group, max_members) numeric matrix.
    if n_group > 0:
        group_names_arr = np.empty(n_group, dtype=object)
        max_members = max(len(v) for v in groups.values())
        group_members = np.zeros((n_group, max_members), dtype=np.float64)
        for k, (gn, gm) in enumerate(groups.items()):
            group_names_arr[k] = gn
            group_members[k, :len(gm)] = gm
        group_n_members = np.array(
            [len(v) for v in groups.values()], dtype=np.float64
        ).reshape(-1, 1)
    else:
        group_names_arr = np.empty(0, dtype=object)
        group_members = np.zeros((0, 0), dtype=np.float64)
        group_n_members = np.zeros((0, 1), dtype=np.float64)

    # Per-group synthesis weights: cell arrays (varying shapes per
    # group), padded into MATLAB-friendly object ndarrays.
    n_g_W = len(group_W) if group_W else 0
    group_W_cell = np.empty(n_g_W, dtype=object)
    group_member_dof_idx_cell = np.empty(n_g_W, dtype=object)
    group_member_idx_cell = np.empty(n_g_W, dtype=object)
    for k in range(n_g_W):
        group_W_cell[k] = np.asarray(group_W[k], dtype=np.float64)
        group_member_dof_idx_cell[k] = np.asarray(
            group_member_dof_idx[k], dtype=np.float64).reshape(-1, 1)
        group_member_idx_cell[k] = np.asarray(
            group_member_idx[k], dtype=np.float64).reshape(-1, 1)

    savemat(str(mat_path), {
        "dwdx":          np.asarray(dwdx, dtype=np.float64),
        "w_nom":         np.asarray(w_nom.reshape(-1, 1),
                                     dtype=np.float64),
        "indx":          indx,
        "channel_names": name_arr.reshape(-1, 1),
        "dof_labels":    dof_arr.reshape(-1, 1),
        "n_dof":         np.float64(len(dof_labels)),
        "n_src":         np.float64(n_src),
        "n_elt":         np.float64(n_elt_actual),
        "n_group":       np.float64(n_group),
        "group_names":   group_names_arr.reshape(-1, 1),
        "group_members": group_members,
        "group_n_members": group_n_members,
        "group_W":       group_W_cell.reshape(-1, 1),
        "group_member_dof_idx": group_member_dof_idx_cell.reshape(-1, 1),
        "group_member_idx":     group_member_idx_cell.reshape(-1, 1),
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
