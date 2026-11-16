# e5_seg — segmented primary + edge sensors + laser-MET truss

> **STALE ARTIFACTS (2026-07-19):** the committed `e5_seg` Hx/MET
> artifacts predate the SegMirMaker edge-sensor rework (per shared
> edge: 2 sensor locations x 3 axes piston/gap/shear, no
> absolute-piston anchor row) — regeneration queued; the edge+MET
> numbers will improve as on e2e (in-plane DOFs become
> edge-observable).

One modifiable runner (`e5_seg.m`): `e5mono.in` (monolithic parent) →
segmented `.in` **including the MET points** → `dedx` (SegMirMaker Hx),
`dldx` (FD over the engine METcalc), `dwdx` → **MET metric
performance** = post-control wavefront residual
`trace(dwdx·P_δx·dwdxᵀ)` vs the prior baseline `W = dwdx·X·dwdxᵀ`,
with a Monte-Carlo estimate/control cross-check.

Pipeline: `macos.design.segment_rx` → `macos.design.add_met` (Stewart
trusses: 6 launchers/segment + 6 around the aft body → hub fiducials)
→ `macos.design.edge_sensors` + `macos.design.dmet_dx` + a local dwdx
FD block (column convention [rot|trans] per body in its LOCAL triad,
SI units, identical across all three Jacobians by construction).

Knobs at the top: rings/grid/gap, edge-sensor config, truss geometry
(nf, radii), prior X (deploy tolerances), sensor noises R, MC draws.

Reference output (rings=1 Hex, 7 segs + m2 + fpa = 54 DOF, 20 edge +
48 MET measurements, 1 µrad/1 µm priors, 1 nm sensors) under the
PHYSICAL mounting constraints (Dave 2026-07-16): launchers at the
segment edges + 5 mm clearance, 6 fiducials mounted ON M2 ~25 mm
inside its 615 mm rim, aft ("M3") launcher ring hugging that body
(100 mm — it has no structure at larger radius):

| sensing        | w_post rms |
|----------------|-----------:|
| prior          |    9577 nm |
| edge only      |    3641 nm |
| MET only       |     450 nm |
| edge + MET     |     232 nm |

Monte-Carlo (200 draws) 229 nm vs analytic 232 nm (1.3%).

The dominant residual is the AFT-BODY (fpa) uncertainty: its ring
shrank to the physically-mountable 100 mm radius, so its rotational
lever arms are weak.  (An earlier, unphysical configuration with
free-floating 300 mm fiducial/aft rings scored 4.3 nm — mounting
reality costs real performance and is exactly what the layout
optimizer must work within.)  The tier-3 optimizer
(`e5_seg_metopt.m`) currently optimizes the SEGMENT truss only (its
merit covers the 42 segment DOFs; hub/aft analytic rows are the
queued follow-on) and reaches 3.36 nm on that sub-merit,
engine-validated to 0.00%.

Artifacts land beside the script: `e5_seg_met.in` (+ `flat.txt` it
needs in the cwd at load), `e5_segHx.m`, `e5_seg.mat` (all Jacobians),
`e5_seg_metric.png`, and two layout views:

- `e5_seg_met_layout.png` — `macos.design.met_view`: 3-D MET scene
  (segment hex tiles, launchers, hub fiducials, gauge beams, hub disc,
  real-ray envelope from the engine trace) + face-on launcher layout
  with per-segment radial centerlines.
- `e5_seg_view_rx.png` — the same system through the GENERAL viewer
  `macos.view_rx` (works on ANY loaded Rx: traced beam + per-element
  ray-footprint patches + MET paths via `macos.met_geom`).

The optimizer runner (`e5_seg_metopt.m`) additionally writes
`e5_seg_metopt_layout.png`: the optimized launcher/fiducial layout with
the baseline edge ring overlaid as open circles and the 5 mm
edge-clearance hex dashed.

Notes: the engine MET model is straight-line point-to-point length
(no LOS/obscuration check); `macos.modify()` after each poke is
load-bearing (cached trace/OPD otherwise reads unchanged); OPD is in
WaveUnits (mm here).  Next step (recorded in the plan): optimize the
MET layout against this metric — see PLAN_DESIGN_LAYER §6.6 tier 3.
