# e5_seg — segmented primary + edge sensors + laser-MET truss

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

Reference output (rings=1 Pie, 7 segs + m2 + fpa = 54 DOF, 20 edge +
48 MET measurements, 1 µrad/1 µm priors, 1 nm sensors):

| sensing        | w_post rms |
|----------------|-----------:|
| prior          |    9566 nm |
| edge only      |    3653 nm |
| MET only       |    10.0 nm |
| edge + MET     |    5.98 nm |

Monte-Carlo (200 draws) 5.85 nm vs analytic 5.98 nm (2.2%).

Artifacts land beside the script: `e5_seg_met.in` (+ `flat.txt` it
needs in the cwd at load), `e5_segHx.m`, `e5_seg.mat` (all Jacobians),
`e5_seg_metric.png`.

Notes: the engine MET model is straight-line point-to-point length
(no LOS/obscuration check); `macos.modify()` after each poke is
load-bearing (cached trace/OPD otherwise reads unchanged); OPD is in
WaveUnits (mm here).  Next step (recorded in the plan): optimize the
MET layout against this metric — see PLAN_DESIGN_LAYER §6.6 tier 3.
