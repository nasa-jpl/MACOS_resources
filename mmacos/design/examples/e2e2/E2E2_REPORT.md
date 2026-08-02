# e2e2 — status report

**Scope.** The improved TMA design-flow worked example specified in
`design/PLAN_TMA_E2E2.md` and `macos/BRIEF_e2e2_implementation.md`.
Telescope flow complete; relay parked as follow-on. Eight commits, local
on `MACOS_res_dev` `dev`, **not pushed**.

**Engine.** `mmacos.mexa64` is byte-identical (md5 `1f14bd2c…`) to the
`mmacos_FIXED.mexa64` preserved from the macos PR #70 A/B; the macos tree
sits on `colsource-pupil-fix` `36a5f5a`. The PR #70 precondition the
brief sets is therefore met without a rebuild.

---

## 1. The delivered design

D = 3 m, f/20 (EFL 60 m traced 60.333), λ = 500 nm, 0.4° × 0.4° used box,
field bias 13′, M1 perforation 0.2331 m (2.4 % of the area). Scored on a
uniform 13 × 13 grid, all four references:

| rung | max RMS | avg | min Strehl | vs 35.71 nm / 0.80 |
|---|---|---|---|---|
| strict-chief | 49.220 nm | 28.351 | 0.679 | FAIL |
| **strict-centroid** *(primary)* | **30.001** | 20.047 | 0.867 | **PASS** |
| + best focus | 29.380 | 18.691 | 0.872 | PASS |
| + LS tip/tilt | **20.380** | 12.176 | 0.936 | **PASS** |

Diffraction-limited at the primary reference and everything more
permissive; missed at the strictest. The four rungs spread **2.42×**,
which is why every number in this flow names its rung. The 13 × 13
reproduces the 9 × 9 used for the stage verdicts, so the results are not
sampling-limited.

Reported alongside, never inside, the wavefront table: coma
(centroid-minus-chief) 0.11–5.33 µm; chief-ray mapping **f·θ to
−0.0075 %** — the traced EFL correct to 75 ppm — with 630 µm rms
departure, of which 522 µm is genuinely nonlinear and ~110 µm anamorphic.

Clearance passes on M2's accepted central obscuration alone; AOI spread
13.90° against the 15° standing rule.

## 2. Structure

`e2e2_params.m` → `s1_axial.m` → `s2_fold.m` (**the delivered
telescope**) → `s3_score.m` (designs nothing; scores and documents).
`s1_fov_sweep.m` is a stage-1 diagnostic. `relay_followon/` is parked
work, outside the flow.

Two stage reorders, both on Dave's call, both driven by measurement:
fold **before** bias (geometry buys clearance free; bias costs
`^1.80`), then the off-axis stage folded into the relay stage, then the
relay parked once the telescope met the target without it.

## 3. Findings

Each is recorded in-source at the point it applies, with the numbers that
produced it. None was fixed silently.

1. **The brief's hoist gate is unattainable.** It requires `his_designs()`
   to reproduce `rodgers1_epd4060_his_designs.mat` to 0.00e+00, but that
   baseline was committed 2026-07-30, two days before the ColSource fix
   the same brief requires building against. Post-fix ratios are
   0.886/0.883/0.886 across three designs and both statistics — one
   common scale, the pupil-area signature. Substituted a same-engine
   pre/post-hoist A/B: **max|Δ| = 0.000e+00** over 60 ladder and 15
   `strict_wfe_deck` entries. The baseline was **not** regenerated.

2. **The best-focus rung was losing to the rung below it.** `fminbnd` at
   default `TolX` (0.1 mm of focus slide ≈ 31 nm of wavefront at f/20)
   cannot resolve its own minimum on a design corrected to the nm level:
   rung 3 returned 2.386 nm where rung 2 read 0.890. Invisible on the
   rodgers1 deck (1.7e-4) where the metric was developed; ruinous where
   it is now used. Fixed by a scale-relative tolerance plus an explicit
   `ff(0)` evaluation, so the ordering holds by construction.
   **Consequence: `rodgers1_pupil_audit.mat`, `rodgers1_dense_field.mat`
   and PACKET Addendum 10's rung-3/rung-4 columns are stale by 2–3e-4.
   Rungs 1–2, including the centroid ruling, are unchanged. Regeneration
   is a reviewed step and has not been done.**

3. **The offset does not scale with the field.** Measured RMS ~
   `bias^1.80`. At a 0.6° box even the reference design's own 0.5° offset
   reads 117 nm, 3.3× the bar, because field and bias draw on the same
   wavefront budget. `P.offset_ratio` is retired in place with the price
   curve that refuted it.

4. **Freeform on pupil-conjugate mirrors cannot reach a field-varying
   residual.** The stage-2 residual decomposes raw 4.02 → −tilt 3.99 →
   −focus 1.74 → −astig 0.66 waves, and the astigmatism reverses sign
   across the field (z5 spread/mean **4.48**). A fixed figure subtracts
   the same map at every field; all three Korsch mirrors are near the
   pupil. Two independent solvers (CALIB, then the SVD engine) both
   failed to help, which is what made the diagnosis worth doing rather
   than blaming the solver.

5. **A relay is sized by the image it accepts, not by the aperture.**
   image half-height = EFL·tan(half-field): e2e 0.042 m, e2e2 0.314 m at
   0.6°. Scaling e2e's Offner by aperture gave a ring radius 0.48× the
   image and killed the trace (318 surface-miss rays).

6. **Bias beat extraction tilt by ~100× here — the reverse of e2e VIS.**
   On the clearance frontier, tilt 0° + bias 13′ scores 20.4 nm against
   tilt 2.5° + bias 0′ at 2600 nm. 2.5° on a *powered* f/20 M3 is far
   more expensive than 13′ of bias. The e2e lesson does not transfer
   between architectures; only pricing both knobs together exposes it.

### Method notes worth carrying forward

- **Clearance is geometry and cheap; wavefront needs a solve and is not.**
  Sweep clearance densely, take the Pareto-minimal combinations, spend
  solves only on those.
- **Do not prune a frontier under a DOF set that suits one branch.**
  Freeform helps a tilt branch (astigmatism) and hurts a bias branch;
  ranking on conics alone would have discarded the winner had the
  branches been within ~6× rather than 117×.
- **More DOFs must never make a reported design worse** — the previous
  solution is always available. Take `min` per branch.
- **CALIB overfits a deep basis.** 45 coefficients against 8 solve fields
  improved the merit while the uniform 81-point score went 56.6 → 127.6
  nm. e2e rule 7 already said so; the SVD engine on a dense grid is the
  route.

## 4. Shared library added

In `design/src`, all general-purpose: `strict_rungs`,
`strict_ladder_deck` (hoisted kernel, plus the 4-rung ladder that was
duplicated in two rodgers1 files), `pupil_gate`, `param_table`,
`stage_score`, `footprint_radius`, `through_hole_radius`.
`mmacos_setup` now puts `design/src` on the path — it did not before,
contrary to the brief's assumption.

## 5. Verification

| | |
|---|---|
| fast suite | 236 pass, 0 fail |
| `tE2E2Axial` | 7 pass, 0 fail |
| `tStrictKernel` | 4 pass, 0 fail |

The Incomplete rows in the fast suite are pre-existing `assumeTrue`
filters (SegMirMaker binary absent), unchanged by this work.

## 6. Open items

1. **`s3_score.m`'s `distortion_` wants fresh eyes.** Three successive
   errors occurred there: the wrong quantity (centroid-minus-chief, µm,
   where distortion needs the centroid position, ±0.2 m); a forced parity
   on a mapping whose parity the fit must discover; and a transposed `R`
   in the scale formula. All were caught, and the third is now an
   assertion (a uniform-scale fit cannot exceed a scale-held fit), but
   the section should be reviewed before its numbers are quoted
   externally.
2. **The relay's field-corrector hypothesis is untested.** A
   focus-conjugate mirror should have the field authority the
   pupil-conjugate ones lack, but it could only be tried on a relay that
   was itself 77× off the bar. Nothing measured argues for or against it.
   See `relay_followon/README.md`.
3. **The rodgers1 rung-3/rung-4 regeneration** (finding 2) is pending.
4. **Nothing is pushed.**

## 7. Resuming

Read `README.md` in this directory — the design point, the FOV sweep
table, and the numbered solve-order rules, each recorded with the failed
run that earned it — then `s2_report.txt` and `s3_report.txt`. The
`.in` that stage 2 emits feeds the existing `run_segmentation` →
`run_sensitivities` → `run_met` → `run_compare` → `run_simulator`
pipeline unchanged; that chain is worked end-to-end in `../e2e/` and is
deliberately not duplicated here.
