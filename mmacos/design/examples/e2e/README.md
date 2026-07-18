# e2e — the complete end-to-end worked example

Design → instrument → segmentation → linear model → MET → simulator,
built entirely from the parameterized design-layer runners and
utilities, **for users to hack for their own systems** (Dave
2026-07-17).  Every stage runner produces a **thorough design report**
(saved `sN_report.txt`) **and graphics** (`macos.view_std` standard
views + the stage's metric figures) beside this file.

All user knobs live in **`e2e_params.m`** — one file, commented.  Each
stage consumes the previous stage's saved artifacts (`.in` + `.mat`),
so a knob change re-runs from the first stage it affects.

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
| `s2_instrument.m` | s1 artifacts | 3-mirror bench relay (M4 corrector / M5 collimator / M6 camera) widening the corrected field to ±2′ at ~0.3–0.6 −tilt waves (was 0.03→1.9 across the same span for the telescope alone).  The solve is JOINT (CALIB on M2/M3/M5/M6, SVD-engine stages on the dense 5×5 field grid, M1 common mode over the full set); M2/M3 keep refining with the instrument, their field-zone lMon growing with the field.  The BEST DETECTOR PLANE (tilt + despace) is re-fit through a 5×5 grid of measured field foci (`align_focal_plane`) and the ladder is scored on it.  The remaining raw−(−tilt) gap is relay DISTORTION — not correctable by detector angle; M4 near the focus is the reflective field-corrector for it (distortion-merit solve, follow-on).  Probed and rejected: per-field patch corrector at a focus (SVD rank collapse) and a 4th weak mirror near the relayed pupil (common-mode, conditions the joint solve worse).  `s2_instrument.in/.mat`, views + field maps + `s2_report.txt` |
| `s3_segmentation.m` | s2 | segmented primary (`segment_rx` + physical apertures); new views |
| `s4_jacobians.m` | s3 | `dwdx`, `dwdz`, `dwdgrid` sensitivity channels + condition/rank report |
| `s5_met.m` | s4 | MET truss (`add_met` + layout optimizer), `dedx`/`dldx`, estimator gains `dxdl`/`dxde`, `dwdl`/`dwde`, MET-optimized performance report, views with MET |
| `s6_simulator.m` | s5 | PSF simulator — mmacos engine OR linear model (user switch) driven by an x/z/grid time history |

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
`P.seg`/`P.met`/`P.sim` → from `s3`/`s5`/`s6`.  The bias sweep,
lMon measurement, hole sizing, and detector-plane fit are all
re-derived inside the runners — none of them are hand-tuned numbers.
