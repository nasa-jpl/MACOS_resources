# e2e — the complete end-to-end worked example

Design → instrument → segmentation → sensitivities → MET → compare →
simulator, built from the parameterized design-layer runners and
utilities, **for users to hack for their own systems** (Dave
2026-07-17).  Every stage runner produces a **thorough design report**
(saved `sN_report.txt`) **and graphics** (`macos.view_std` standard
views + the stage's metric figures) beside this file.

**The product is the general stage runners** (Dave 2026-07-19), which
live in `design/runners/` (see its README for the pipeline and the
`.in`-plus-declared-sidecars handoff contract); the `sN_*.m` scripts
here are thin narrative drivers over them.  s5 already works this way
(`run_met`); s1–s4 promote to `run_design`/`run_segmentation`/
`run_sensitivities` in a recast pass after s6.

All user knobs live in **`e2e_params.m`** — one file, commented.  Each
stage consumes the previous stage's saved artifacts (`.in` + `.mat` +
sidecars), so a knob change re-runs from the first stage it affects.

## The telescope case

An **on-axis Korsch TMA taken slightly off-axis, back end folded
behind the primary**: D = 4 m, **f/1.75 primary**, **f/18 system**,
**m2 = 8** (f/14 intermediate focus), 500 nm — a VIS imager.  The
first-order layout comes from `macos.design.tma_layout` with the f/#s
as *free inputs* (change them in `e2e_params.m` and the radii
re-derive).  A 90° **fold after M2** turns the feed into +x just
behind M1, so M3, the image, and the focal plane sit on a flat bench
BEHIND the primary; a small **extraction tilt on M3** (1.2°) takes the
return off the feed axis so the fold clears at a small field bias (the
sweep settles at 2′).  Conics are solved at the biased field, then
refined with freeform Zernike departures under the sphere+Zernike
solve doctrine (per-mirror field-zone `lMon` via `field_zone_lmon`,
coefficient-sanity verification, guarded common-mode null).  The
telescope alone is DIFFRACTION-LIMITED over ±1′ at 500 nm (0.016
waves −tilt worst).  History worth knowing: the first design point
(f/1.25, m2 = 16, no extraction tilt) forced a 5′ bias and could not
reach VIS blur — bias was the killer (aberration ~ bias²); the fix
was geometry (extraction tilt + gentler primary/feed), not more
correction DOFs.

## Stages

| runner | consumes | produces |
|---|---|---|
| `s1_telescope.m` | `e2e_params.m` | telescope design: `s1_telescope.in/.mat`, `s1_views.png`, `s1_wfe_field.png`, `s1_fpmap.png`, `s1_report.txt` |
| `s2_instrument.m` | s1 artifacts | **OFFNER ring-field 1:1 relay** (`P.inst.type="offner"`, geometry from `design/src/offner_layout`): concave R=2 m used twice + convex R/2 at the stop, ALL CONCENTRIC — no tilted powered surfaces, so the tilt-astig floor is deleted at the root and the odd aberrations (coma, distortion) cancel by symmetry.  M4 = flat routing fold + corrector station; the convex stop mirror is held out of every solve (pupil-conjugate).  Result: **DL over the FULL ±2′ field at 500 nm** — −tilt ladder 0.015/0.017/0.027/0.037/0.043, Strehl 0.99→0.93, on pure spheres (all K=0), sub-mm figures.  The ring field hosts several small-field instrument pickoffs.  The tilted-sphere `"zigzag"` variant remains selectable (DL through 1.5′; its tilt astig scales ~f₅·θ²).  The solve is JOINT (CALIB on M2/M3/M5/M6, SVD-engine stages on the dense 5×5 field grid, M1 common mode over the full set); M2/M3 keep refining with the instrument, their field-zone lMon growing with the field.  The BEST DETECTOR PLANE (tilt + despace) is re-fit through a 5×5 grid of measured field foci (`align_focal_plane`) and the ladder is scored on it.  The SCIENCE FIELD CENTER is found AUTOMATICALLY (`P.inst.field_center="auto"`, Dave 2026-07-18): sections [2]–[4e] run in a 2-pass loop — pass 1 solves at the starting guess (`field_dy_arcmin` −0.7′ off the s1 bias, from reading the WFE map), then [4g] maps ±3′ (13×13), takes the centroid of the raw-WFE<0.02 region, and if the chief sits >0.15′ off it pass 2 RE-SOLVES with the chief there.  The loop moved the chief another −0.71′ (total bias 0.59′): worst ±2′ −tilt 0.043 → 0.023 → **0.0215**, Strehl floor **0.965**, <0.02 region 31→52/169 points; the [4f] scan confirms the adopted center is the optimum (both ±0.35′ neighbors worse).  Lesson encoded in the loop: re-SCORING a shifted center undersells re-SOLVING there.  Hand-picked variants preserved in `s2_variants/dy-0.70|dy-1.05/`.  The M1 central hole is now a REAL Rx obscuration (`set_hole` emission): the trace clips the central rays and the layout views render the hole.  The remaining raw−(−tilt) gap is relay DISTORTION — not correctable by detector angle; M4 near the focus is the reflective field-corrector for it (distortion-merit solve, follow-on).  Probed and rejected: per-field patch corrector at a focus (SVD rank collapse) and a 4th weak mirror near the relayed pupil (common-mode, conditions the joint solve worse).  `s2_instrument.in/.mat`, views + field maps + `s2_report.txt` |
| `s3_segmentation.m` | s2 | THIN DRIVER over **`design/runners/run_segmentation.m`** (2026-07-19 recast).  Segmented primary, TWO variants both kept: **pie** (1-ring PIE, 7 segments: center hexagon + 6 chorded wedges → `e2e_pie.in`) and **hex2** (2-ring HEX, 19 segments → `e2e_hex2.in`), each with PHYSICAL polygonal apertures (`emit_apertures`), the parent's solved M1 figure carried onto every segment (FF channel), and the M1 hole riding on the center segment.  Segment size auto-scales (`Aperture/(2·rings+1)`); gap 25 mm; 128-pt source grid.  `P.seg.variant` (default `"pie"`) picks the one stages 4–6 consume.  Footprint/aperture overlays + `view_std` + `s3_report.txt`.  Gotchas this stage flushed out (fixed in `segment_rx`): a Surface=Zernike parent's figure is DROPPED by SegMirMaker (now carried via FFZern), and the SegMirMaker↔engine tiling contract needs the heritage `xGrid=(−1,0,0)` + `SegXgrid=(−1,0,0)` in the merged Rx (else the ray-to-segment map is 180° off the emitted frames and every off-center segment's aperture clips its own rays) |
| `s4_jacobians.m` | s3 (`e2e_pie.in`) | the LINEAR-MODEL substrate for s5/s6, harvested by the production `macos.dw_d*_multi` supervisors over the ±2′ science field (center + 4 corners), each in the canonical stacked form `wall = J·x + w0`: **dwdx** rigid-body 6-DOF per element (60412×90, rank 90 — the spectrum shows exactly 21 strong modes = 7 segments × piston/tip/tilt, then a 2-decade cliff), **dwdz** segment figure via the SEGMENT-LOCAL MonZernike basis, modes 4–11 (×56, cond 5.4 — NOT the FF channel, which carries the parent-aperture basis and paints full-aperture modes), **dwdgrid** per-segment grid pokes on a grid-augmented variant (`s4_grid.in`, flat 256 grids in each segment's CLOCKED Mon frame) through the G-S orthonormal `macos.segment_grid_basis` influence stack (×42, cond 1.26).  Per-segment column-norm table + SV spectra + segment-only conditioning in `s4_report.txt` (= `s4_sens_report.txt`).  THIN DRIVER over **`design/runners/run_sensitivities.m`** (2026-07-19 recast): the grid augmentation is the product `macos.design.grid_augment_rx` (grids in each segment's CLOCKED Mon frame, REPLACING any stale parent-frame lines; span = the dxGrid convention `Aperture/(ng−1)` — never lMon, whose hex-heritage value clips wedge corners), artifacts are `s4_grid.in` + `flat256.txt`, and the figures are the e5-style piston-removed per-element pages in `s4_pages/` (center + multi) plus `s4_<ch>_channels.png` overviews.  `s4_jacobians.mat` carries the three outputs (incl. per-field blocks, which s5 uses for the tilt/non-tilt metric split) |
| `s5_met.m` | s3 (`e2e_pie.in` + `e2e_pieHx.m`) + s4 (`s4_jacobians.mat`) | THIN DRIVER over the general stage runner **`design/runners/run_met.m`**.  As-built Stewart truss: 6 launchers per segment on its TRUE wedge/hexagon boundary (5 mm clearance), 6 fiducials on the M2 rim (25 mm inset), **aft ring on the bench-side FOLD (elt 9) at the SM RADIUS (0.232 m — the stable aft structure extends to the SM radius, Dave 2026-07-19), beaming to M2 through the central hole** (a ring at M3's 2.1 m lateral has no hole line of sight; clearance is checked against the SM-SHADOW radius as the effective hole, since the real hole will be enlarged to match — queued `P.tel` change).  Sensed bodies = 7 segments + M2 + FM = 54 DOFs.  `dedx` from the SegMirMaker Hx **as-is** (edge-sensor placement is never optimized; the layout figures draw the sensors as gray dots).  **Edge-sensor model (2026-07-19, Dave's spec): per shared edge 2 sensor locations at ±0.25·width along the edge × 3 axes — piston (normal), gap (in-plane ⊥ edge), shear (in-plane ∥ edge) — purely differential, no absolute-piston anchor row; the pie = 72 rows = 24 locations × 3 axes;** `dldx` engine-FD **== analytic to 1e-7** (the gate).  **Headline metrics are DIMENSIONLESS WEMs** (Dave: MET tracks the post-WFSC drift state — HWO drift ≪ 1 nm between updates, gauge roadmap ~1 pm; the configured scenario sigmas in `P.met` only set the absolute-forecast column): each configuration is scored **full / tilt / non-tilt** (per-field piston+tilt split of the s4 blocks — tilt control absorbs tilt; note the s4 channels are harvested with focal-plane tracking, so global tilt is already out at the source and WEM_tilt ≡ 0 by convention), MET-only and edge+MET, plus the unobserved floor as a fraction of prior.  The layout optimizer (`macos.design.met_layout_opt`) solves ONE launcher pattern per segment SHAPE CLASS (polar-profile congruence in the pattern frame → center-hexagon + wedge classes) on the **MET-only, non-tilt merit** (the truss must stand alone), with the **aft↔SM leg solved FIRST** as its own coordinate block (clocking + its own fiducial map) and **6 fiducials with the ROTATIONAL assignment** (each wedge's map shifted by its clocking → congruent beam geometry, segments ~interchangeable — and it SCORES BETTER than the asymmetric nf=3 raw-merit winner).  **Result (72-row edge model): as-built WEM 15.6 MET-only / 3.92 edge+MET → optimized 3.77 MET-only / 1.78 edge+MET (≈2–4 pm sensing-limited wavefront at 1 pm gauges), unobserved floor ~0.3% of prior — the in-plane gap/shear axes make the in-plane DOFs edge-observable, cutting edge+MET from the 4.59 of the old normal-height-only model.**  Dave's manual **corner-pairs benchmark** (pairs at each class's two outermost corners + a pair on the inside edge) scores 3.97 — ties the machine — but its 15 mm corner-junction separation violates the 50 mm hardware rule (reported gate-bypassed).  Winner engine-FD-validated 0.00%; Monte-Carlo 0.7%.  Estimator products `dxde`/`dxdl`/`dwde`/`dwdl` (full-suite MMSE) saved in `e2e_pie_met.mat` for the compare/simulate stages.  **The winner is exported as a scale-free PRESET** (`e2e_pie_met_preset.mat`, promoted to `design/runners/presets/pie_met.mat`): future pie builds pass it to `run_met(..., 'preset', ...)` and their as-built truss starts as this optimized configuration realized on their own boundaries.  Artifacts: `e2e_pie_met.in`, `e2e_pie_metopt.in`, layout/metric/view figures, `s5_report.txt` |
| `s6_compare.m` | s3–s5 | PLANNED `run_compare` (spec Dave 2026-07-19): linear model `w = dwdx·x + dwdz·z + dwdgrid·g + dwdu·u + w0` (u = control = segment + SM rigid DOFs), measurement `m = [l; e]`, `dmdx = [dldx; dedx]`.  Poke each DOF in turn (default 100 nm / 100 nrad); per poke show TWO graphics — mmacos vs the linear model — each an OPD map above stacked bar charts of `l`, `e_piston`, `e_gap`, `e_shear`; settable dwell (default 0.25 s); saved frames + per-poke agreement report |
| `s7_simulator.m` | s6 | PLANNED `run_simulator`: PSF simulator — mmacos engine OR linear model (user switch) driven by an x/z/grid time history, with an estimator/controller so UNCONTROLLED and CONTROLLED outputs both emerge |

Run order: `s1` → `s6`, each with `run('.../sN_*.m')` after building
mmacos (`mmacos_setup.m` or the addpath lines at the top of each
runner).  House rules: figures + reports land in this directory; no
`exit(0)` inside example scripts (batch wrappers supply it).

## The design procedure — why the solves are staged this way

Changing the parameters re-derives the whole design, but the SOLVE
STAGING is load-bearing: each rule below was established by a failed
run (the failures are documented where they happened, in the runner
comments).  The runners implement all of them, so a parameter change
should just work — if you restructure the chain, keep the rules:

1. **First-order layout from the free f/#s** (`tma_layout`); packaging
   stations are fractions of D, so everything scales with aperture.
2. **Fold, extraction tilt, then bias.**  Fold the feed behind the
   primary; a small EXTRACTION TILT on M3 (about the bench normal)
   takes the return off the feed axis so the fold clears without
   relying on field bias (bias was the VIS killer: aberration ~
   bias²).  Sweep the bias and take the CLEARING candidate with the
   BEST solved WFE -- not simply the least bias: the tilt/bias
   interplay is non-monotonic (the walk cancels the tilt separation
   at some bias, and a too-small bias lands the conic solve in a bad
   basin).
3. **Conics solve AT the bias point, with the detector at the
   biased-field focus**: `align_focal_plane` BETWEEN two solves — the
   biased field focuses mm from the paraxial focal plane, and that
   defocus (1.6 waves here) otherwise poisons the conic solve.  At a
   SMALL bias (tilt astigmatism dominant) the solve repeatably lands
   in a bad LM basin (K3 → −3..−4): seed it by CONTINUATION — solve
   first at the sweep's best-conditioned bias, then walk the bias to
   the picked value and re-solve.
4. **Hold weak/near-focus correctors OUT of ROC+conic solves** — the
   optimizer abuses conic as high-order sag (K ~ −2000) and drags the
   first order off spec.
5. **Freeform prerequisites**: per-mirror FIELD-ZONE `lMon`
   (`field_zone_lmon`) fixed ONCE before the first solve; ONE Zernike
   type per mirror for the life of its coefficients.
6. **Joint CALIB field solve on the strong set** (≤12 fields — CALIB's
   FOV cap).  M1 stays OUT: it is pupil-degenerate with the relayed
   stop.
7. **SVD engine (`zern_jacobian_solve`) for anything degenerate**:
   deep bases, weak correctors, M1's common mode — scored on a DENSE
   field grid (the SVD engine has no FOV cap; more field points smooth
   the OPD between solve samples).  NEVER give CALIB a degenerate
   basis: the FD-LM normal matrix goes singular, figures diverge, and
   the engine SIGSEGVs.
8. **M1 common mode over the FULL field set**, never at a single field
   — the tilt mode is a gauge there and the solve wanders
   geometrically.
9. **Order matters**: joint field solve FIRST, common-mode cleanup
   AFTER.  Nulling the bias point first lands the joint solve in a
   worse LM basin.
10. **Detector plane last**: re-fit tilt + despace through a 5×5 grid
    of measured field foci AFTER the solves, and score the field
    ladder on that best plane.
11. **Distortion is not blur.**  The raw−(−tilt) gap is chief-ray
    mapping error; no detector angle corrects it.  Solve the
    near-focus corrector (M4) against the affine-projected
    chief-displacement metric — its per-field-tilt channel is exactly
    the distortion knob (and exactly why it is useless for blur).

12. **The relay wants to be an Offner.**  A concentric 1:1 ring-field
    relay (spheres held — a ROC/conic solve would un-Offner the
    concentricity; convex stop mirror out of the solves) beats any
    tilted-sphere arrangement outright: the biased patch is a
    ring-field arc, which is exactly the field shape the Offner
    serves aberration-free.

13. **Let the field center find itself — and RE-SOLVE there.**  After
    the joint solve, map the WFE over a field patch wider than the
    science field, take the centroid of the best region, and if the
    chief is off it, re-solve WITH the chief there (the s2 [4g]
    2-pass loop).  Merely re-scoring candidate centers on the frozen
    design undersells them — the re-solve moved the optimum well past
    the scan's pick.

Dead ends, kept on record in `s2_instrument.m` so nobody re-walks
them: a per-field Zernike patch corrector at a focus (SVD rank
collapse — not a static solution); a 4th weak mirror near the relayed
pupil (common-mode, conditions the joint solve worse); deepening
BornWolf to the full 3:25 list (~13% — the basis, not the solver, is
the wall).  Next real levers if the field must go wider or flatter: a
second re-imaging stage (doubles the distinct conjugates), an ANSI
re-solve of the relay mirrors, or the secondary-magnification trade
(smaller m2 relaxes M2's curvature but raises the clearing bias and
lengthens the bench).

**Which stages re-run after a change**: anything in the telescope
block of `e2e_params.m` → from `s1`; the `P.inst` block → from `s2`;
`P.seg` → from `s3` (s4/s5 follow); `P.met` → `s5` only; `P.sim` →
`s6`.  The bias sweep, field-center loop, lMon measurement, hole
sizing, and detector-plane fit are all re-derived inside the runners —
none of them are hand-tuned numbers.  The MET scenario sigmas in
`P.met` deliberately do NOT cascade: the s5 headline metrics are
dimensionless and scenario-invariant.
