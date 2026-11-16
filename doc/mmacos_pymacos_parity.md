# mmacos ↔ pymacos capability parity audit

**Date:** 2026-07-02 (supersedes the 2026-06-12 parity note in
`mmacos/CLAUDE.md` §"Veneer parity with pymacos").
**Method:** mechanical extraction + diff of the three layers from the
sources themselves — no reliance on prior audit notes:

| Layer | Source | Count |
|---|---|---|
| Shared backbone | `subroutine` names in `macos/macos_f90/macos_api_mod.F90` | 121 routines |
| pymacos raw | `subroutine` names in `pymacos/src/cmake/source/pymacos_f2py.f90` | 116 wrappers |
| mmacos raw | `mmacos/src/mmacos_gen_cmds.txt` + `CASE('…')` in `mmacos_mex.F` | 131 commands |
| pymacos user surface | `def` names in `pymacos/src/pymacos/macos.py` + `tests/sensitivities/*.py` | ~100 public fns |
| mmacos user surface | `mmacos/src/+macos/*.m` + `+channels/` + `+design/` + `sensitivities/` | ~109 fns + drivers |

To re-run the audit: repeat the greps above and `comm -23` the
lowercased name lists (mmacos cmd names are lowercase; a few api
routines are mixed-case, e.g. `elt_srf_zrn_FreeForm`).

---

## Headline

- **Raw engine layer: mmacos is a strict superset of pymacos by 5
  routines** (all post-2026-06-12 additions). Everything else at the
  raw layer is at full parity — the apparent name mismatches are the
  documented hand-written renames (`opd_val`→`opd`,
  `int_cmd/int_get`→`intensity`, `cfield_*`→`complex_field`/`apodize`,
  `elt_dx_get`→`dx_at`).
- **User-facing layer: each side has real one-way capabilities.**
  mmacos's unique list is much larger (design layer, dW/dgrid
  sensitivities, segment tooling, pupil evaluator); pymacos's is two
  analysis tools plus a set of typed veneers over commands mmacos
  exposes raw-only.

---

## pymacos capabilities missing from mmacos

### Real capability gaps (no mmacos equivalent at any layer)

| Capability | Where in pymacos | Status / tracker |
|---|---|---|
| `predict_global_rigid_response` — predict a global rigid-body group response from per-element local-frame columns + W kinematics matrix | `tests/sensitivities/channels.py` | Deferred as mmacos Phase 7.b.2 (`macos/PLAN.md` §5.4) |
| `group_synthesis_matrix` — column-combination weights testing per-element → group reconstruction | same | same |
| `test_coro_dm_grid_self` (apodize_complex vs elt_grid self-consistency) | `tests/proper_compare/` | mmacos Phase 5 slice 7 pending (needs `+macos.elt_grid` veneer; MATLAB is already column-major — transpose convention needs care) |
| `test_coro_aberrations` chain runner (`SystemState`/`CORO_LAYOUT`/`run_chain_with_state`) | `tests/proper_compare/aberrations.py` | mmacos Phase 5 slice 7 pending (~500 LOC port) |
| `run_broadband_*` report generators; `contrast.py` as reusable scoring module | `tests/proper_compare/` | not ports of record — report generators, not tests |

### Typed-veneer gaps only

Capability exists in mmacos via raw `mmacos('cmd', …)` but has no
`+macos/*.m` wrapper. Matches the "still raw-only" tracker in
`mmacos/CLAUDE.md` §11.6 cross-ref — verified still accurate 2026-07-02.

| pymacos typed fn | mmacos raw cmd(s) backing it |
|---|---|
| `obs_set` | `obs_set` |
| `ors` | `ors_run` |
| `elt_grid` (full-grid setter) | `elt_srf_grid_data` |
| `elt_grid_dx`, `elt_grid_npts(_max)`, `elt_grid_scale`, `elt_grid_any` | `elt_srf_grid_*` family (`find_grid_elts.m` covers `_fnd`; `elt_grid_add.m` covers `_data_add`) |
| `src_csys`, `src_size`, `src_info`, `src_finite` | `get/set_src_csys`, `src_size`, `src_info`, … |
| `elt_srf_csys` | `elt_srf_csys(_get/_set/_pos/_dir)` |
| `elt_zrn_type`, `elt_zrn_norm_rad` | `elt_srf_zrn_type`, `elt_srf_zrn_norm_radius` |
| `setRayInfo` | `ray_info_set` |
| `elt_grp_any/_fnd/_max_size/_wipe` | `elt_grp_any/_fnd/_max(_all)/_del_all` (`del_elt_grp.m` covers `_del`) |
| `model_size` query, `traceWavefront`, `opd_val` (raw variant) | (session state / `trace_rays` / `opd_val`) |

---

## mmacos capabilities missing from pymacos

### Engine-wrapper gaps (the only true one-way holes at the raw layer)

Five `macos_api_mod` routines with **no f2py wrapper at all** — all
added on the mmacos side after the 2026-06-12 audit. Closing each is a
mechanical add to `pymacos_f2py.f90` + `macos.py`:

| api_mod routine | mmacos veneer | What it does |
|---|---|---|
| `draw_rays_cmd` / `draw_rays_get` | `macos.draw_rays` | data-only DRAW ray-bundle capture (backs `Telescope.view_layout`) |
| `xps_cmd` | `macos.xps` | full-grid exit-pupil sample (backs `pupil_quality`) |
| `ray_status_get` | `get_ray_status` | per-category ray-failure counts (complements binary `get_ray_info`) |
| `elt_z` | `get_elt_z` | per-element z position |

### User-surface capabilities with no pymacos analog

- **Grid-figure sensitivities:** `dw_dgrid` / `dw_dgrid_multi` +
  `GridChannel` / `grid_channels` — pymacos has **no dW/d(grid) driver
  at all**. Also `dw_dsurf_multi` (pymacos has only single-field
  `dw_dsurf.py`) and `dwdx/dwdz_for_current_source`. mmacos drivers are
  packaged in `+macos/`; pymacos's live as scripts under
  `tests/sensitivities/`.
- **Segment/grid tooling:** `segment_grid_basis` (per-segment
  Voronoi+hull mode stacks), `gs_zernike_segment_basis`,
  `zernike_grid_basis`, `write_grid_file` (engine GridFile writer),
  `opd_psf`, in-package `m2v`/`v2m`.
- **Pupil evaluator:** `pupil_quality` (XPS-based pupil-surface
  Zernike fit).
- **Design layer** (MATLAB-only by plan, `PLAN_DESIGN_LAYER.md`):
  `macos.design.System` (from_rx / vary / sensitivities / optimize),
  `Telescope` builder (`seidel_seed`, `tma_layout`, field grids),
  `diagram` / `view_layout`.
- Minor: `find_powered_elts` as a public fn, the `Session` OO veneer,
  plotting helpers (`plot_dw_per_element`, `plot_dw_channels`,
  `plot_opd_canvas`).

---

## Caveats on pymacos's nominal surface

- `getEltSrfZern` is **broken** (wrong f2py symbol → AttributeError);
  use `getEltSrfZernMode`.
- `setEltSrfZernMode` errors on single-mode calls; use
  `setEltSrfZernCoef`.
- `getActivePointSrc` / `setActivePointSrc` are `#ToDo` stubs.

(All three documented in `pymacos/CLAUDE.md`; left in place because
downstream code may call them.)

---

## Cheapest closure order (if/when parity is wanted)

1. pymacos f2py wrappers for the 5 engine routines (mechanical).
2. mmacos `+macos` veneers for `obs_set` / `ors` (one-liners over
   wired commands).
3. `dw_dgrid` + `grid_channels` port to pymacos (real work; brings the
   grid-figure Jacobian to Python).
4. `predict_global_rigid_response` / `group_synthesis_matrix` port to
   mmacos (Phase 7.b.2).
5. `elt_grid` veneer + `test_coro_dm_grid_self` port (Phase 5 slice 7).
