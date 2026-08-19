# PLAN — e2e2: the improved TMA design-flow worked example

> **Status: PLANNED (Dave, 2026-08-01).  Not started.**
> A from-scratch, comprehensive worked example that folds in everything
> the Rodgers offset-field reproduction taught us
> (`design/rodgers1/PACKET.md`, Addenda 8–10).  Written for cold
> implementation by Opus, Sonnet, or a user.  Read this file, the root
> `macos/CLAUDE.md`, `mmacos/CLAUDE.md`, and the referenced runner
> sources before writing any code.

## What this example is

An end-to-end TMA design flow, staged like a real program and like the
Rodgers study: **input parameters → Korsch axial starting point →
move off-axis → add fold → add relay + focal plane**, each stage a
gated step that emits a committed `.in`, a parameter-delta table, a
thorough text report, and standard views.  The product is the FLOW —
reusable stage drivers a user can re-parameterize — not one telescope.

Home: `templates/80_end_to_end/e2e2/` (house rules: save `.in`+`.mat`, README,
figures in the dir, **no `exit(0)` in example scripts**).  One
parameter block (`e2e2_params.m`) drives everything, like
`e2e/e2e_params.m`.

## Standing doctrine (from the Rodgers arc — apply at every stage)

1. **Joint solve, never alternate.**  Conics + rigid DOFs + FPA
   tip/focus in ONE CALIB DOF set under the per-field exit-pupil merit
   (OptFEX — reachable from the Rx since macos PR #68).
   `align_focal_plane` is the SEED, not a co-optimizer: alternating
   solve-then-refit-FPA chases two objectives and does not converge
   (rodgers1 `xp_optimize.m` is the reference implementation).
2. **State the wavefront reference.**  Centroid-primary, chief
   secondary (`strict_wfe_deck` computes both; Dave's 2026-07-31
   ruling).  For any EXTERNAL comparison also score the
   best-focus + LS-tilt rung — reporting conventions differ by 1.3–1.7×
   on comatic fields, and per-field focus is <3% of that; the tilt
   treatment is the whole game.
3. **Solve set ≠ scoring set.**  Solve on the program's field points;
   SCORE statistics on a uniform grid (`macos.design.field_grid`) — an
   edge-weighted solve set biases the average ~8% at identical max
   (rodgers1 `dense_field_check.m`).
4. **Pupil gate.**  After every source/aperture edit: greatest chord of
   `macos.spot(1)` vs declared `Aperture=`, and zero rays outside the
   declared radius (`tests/tPupilAperture.m` pattern).  The engine is
   correct since macos PR #70; the gate stays because the defect hid
   for decades by reading exactly right along the axes.
5. **Frame before angle.**  Every reported tilt names its reference
   (vs arriving chief ≠ vs axis; the "14.3° tilted detector" was a
   frame artifact).  Report BOTH the beam incidence on the detector
   (hardware driver) and the mechanical tilt about the axis.
6. **Parameter provenance table.**  Each stage emits the key-parameter
   delta vs the previous stage: per mirror Kc, Kr, position, angle;
   FP pose (tilt about axis + beam incidence).  Radii/spacings that a
   solve holds are STATED as held.  (The Rodgers slide-5 lesson: the
   parameter table IS the solution; WFE numbers only score it.)
7. **Report + views per stage.**  `design_report` (identity,
   first-order, field performance, stage metrics) + `view_std`/
   `view_rx` + `view_field_map`, saved beside the `.in`.  Coma
   (centroid-displacement map) and f·θ distortion are standard
   diagnostics at every stage with a detector (rodgers1
   `seq_centroid.m` is the template).
8. **Mechanics.**  model_size ≥ Rx `nGridpts`; one MATLAB process per
   model size; `macos.modify()` after programmatic pokes; makems from
   `~/dev/macos` root; matlab -batch wrappers supply `exit(0)`.

## The stages

### S0 — Input parameters (`e2e2_params.m`)
Single `P` struct, no computation: aperture D, system f/#, primary
f/#, m2, field box (size + bias), λ, obscuration/clearance rules
(AOI < 15°, M1 keep-out, shroud), back-end spec (relay magnification,
final focal ratio, detector envelope), scoring grids, model_size.
Every later stage reads P and nothing else.
**Default design point (Dave, 2026-08-01): the Rodgers geometry
scaled to D = 3 m (lengths × 3/5), f/20 (EFL 60 m), λ = 500 nm,
0.2° × 0.2° box; target = diffraction-limited at 500 nm across the
used box (RMS ≤ λ/14 ≈ 36 nm, Strehl ≥ 0.8).**

### S1 — Korsch axial starting point (`s1_axial.m`)
- Driver: `macos.design.Telescope` (family TMA; closed forms per
  `MACOS_resources/optical_design/TELESCOPE_DESIGN_REFERENCE.md`).
- Gate: fixtures (`optical_design/fixtures/tma_fixture.json`) — conic
  mismatch >1e-6 or nonzero S_I/II/III is stop-and-fix, never widen.
- On-axis joint solve (conics + FPA) → the ≤ few-nm anchor, the
  Rodgers stage-1 analogue.  Emit `s1_axial.in`, report, views,
  parameter table (the baseline column).
- Trap: the paraxial seeder can be wildly off on long-EFL designs
  (96% on the Rodgers deck) — it only seeds; never quote
  `spec.derived.EFL`, always report the traced EFL.

### S2 — Move off-axis (`s2_offaxis.m`)
The Rodgers process, replayed as OUR flow:
- (a) Freeze S1, apply the field bias from P, re-fit FPA only → the
  collapse measurement (the "why we must re-solve" number).
- (b) Re-solve conics + FPA jointly at the bias field.
- (c) Add M2/M3 tilt + decenter to the SAME joint DOF set.
- Bias selection is WFE-based (scan; the small-bias conic basin is
  path-dependent — solve AT the bias point, `project_fold_extraction`
  rule), with the field-center auto-scan from e2e s2 ([4g] pattern).
- Emit one `.in` + parameter-delta column per sub-stage; score
  centroid + chief + LS-tilt rung on solve set AND uniform grid.
- Expect and document the conic↔rigid trade (equal wavefront from
  different DOF magnitudes on one compensation branch — a property,
  not a disagreement).

### S3 — Add fold (`s3_fold.m`)
- Driver: `Telescope.add_fold` + `set_hole` (hole = REAL ObsType
  obscuration, carried through views and traces) +
  `center_focal_plane`.
- Fold rules (memory `project_fold_extraction`): feed into +x, psi in
  the X-Z plane, M1 keep-out honored; extraction tilt sized so the
  return clears at the P bias (the e2e VIS lesson: extraction tilt
  beats brute bias).
- Gates: engine-truth clearance (ray-to-body margins via `ray_bundle`
  / `check_clipping`), AOI < 15° at every surface, pupil gate,
  re-solve after the fold (a fold is nominally null but the hole and
  clearances are not).

### S4 — Add relay + focal plane (`s4_relay.m`)
- Driver: `offner_layout` (ring-field, pure spheres — DL over ±2′ in
  e2e s2) or the 3-mirror bench relay (M4 corrector / M5 collimator /
  M6 camera, radii from the collimator condition) — P selects.
- JOINT refinement: as relay optics are added, keep refining M1–M3
  (the stage-2(e) doctrine from e2e) — one DOF set, not sequential.
- Distortion is tracked, not optimized (rodgers1 finding: no
  wavefront DOF targets mapping; it calibrates) — report the local
  plate-scale slope and the nonlinear residual.
- Final FPA: solved jointly; report detector beam incidence +
  mechanical tilt; verify the fitted plane sits at the per-field
  best-focus floor (the bound any single pose can reach).

### S5 — Final scoring + documentation (`s5_score.m`)
- Full ladder (chief / centroid / +bestfoc / +LS-tilt) on the uniform
  grid; coma + distortion maps; parameter provenance table across all
  stages (the deck slide-5 format); `design_report` + view set;
  README with the design procedure rules actually exercised (the e2e
  11-rule pattern — every rule earned by a failed run).
- Optional handoff: the emitted `.in` feeds the EXISTING pipeline —
  `run_segmentation` → `run_sensitivities` → `run_met` →
  `run_compare` → `run_simulator` — unchanged.  State this in the
  README; do not duplicate those stages here.

## Implementation notes for the executing agent

- Reuse, don't rewrite: `Telescope`, `align_focal_plane`, CALIB via
  `macos.calib*`, `strict_wfe_deck`/`strict_refs` (hoist from
  rodgers1 into `design/src/` when first reused — they are currently
  example-local), `field_grid`, `view_*`, `design_report`,
  `add_fold`/`set_hole`, `offner_layout`, the runners.
- The rodgers1 dir is the reference for S2's solve mechanics
  (`xp_optimize.m`), the metric ladder, and the coma/distortion
  deliverables (`seq_centroid.m`, `seq_spot_example.m`).
- Every stage ships a matlab.unittest test (suite registration per
  model size in `run_mmacos_tests.sh`); fast suite green between
  stages, full suite before commit.  Work lands on `dev`, both repos,
  engine first; push only when Dave asks.
- CALIB caps: ≤12 FoV with the on-axis field IMPLICIT (11 explicit).
- Expect basin path-dependence: record solve ORDER in the README as a
  rule with its failed alternative, Rodgers/e2e style.
