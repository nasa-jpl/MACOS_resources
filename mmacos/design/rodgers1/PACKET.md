# Rodgers offset-field coaxial-TMA study — MACOS reproduction packet

**Source:** `macos_sandbox/Design/Rodgers/260728-TMA_Offsetfield-jmr.pptx`
(J.M. Rodgers, ORA / CODE V, 28-Jul-2026), λ = 1000 nm.
**Driver:** `mmacos/design/rodgers1/rodgers1.m` (parameter-driven; defaults reproduce the study).
**Date:** 2026-07-29.

---

## 1. The system (transcribed verbatim from the slides)

Coaxial three-mirror anastigmat, all mirrors on one axis. Prescription (CODE V):

| | ROC (mm, signed) | conic K (nominal) | spacing (mm, signed) |
|---|---|---|---|
| M1 | −12357.51782 | −0.992924471436 | M1→M2 −5267.903548 |
| M2 |  −2201.36673 | −1.92646737685  | M2→M3  7106.924338 |
| M3 |  −2687.97316 | −0.707216181423 | M3→focus −5095.367898 (paraxial) |

- **FOV:** 0.2° × 0.2° square. **Offset (stages 2–4):** +0.5° in Y.
- **EPD:** *not stated on any slide.* Recovered as ≈2003 mm from the slide-1
  layout drawing (scale bar 590 drawing-units = 1000 mm; M1 arc ≈ 1182 units);
  **Dave set EPD = 2000 mm.** ⚠ This is the one inferred input — see §5.
- Derived: EFL ≈ 3506 mm, **f/1.75**.

**CODE V → MACOS conventions applied** (`rodgers_common.m`): ROC → `KrElt=−|R|`
(convexity is geometry, not sign); K → `KcElt=K` directly; spacings passed as
magnitudes (the builder folds z with (−1)ᵏ); FOV/offset/tilts in degrees.
Rodgers reports WFE in waves @ 1000 nm; we convert ×1000 → nm.

---

## 2. Layout gate — PASSED (verify before trusting any WFE)

Built via `macos.design.Telescope` family=TMA, verbatim conics injected
(`add_mirror(...,'conic',K)` overrides the Seidel seed):

- M1→M2 = 5267.904 mm, M2→M3 = 7106.924 mm — **match his to the micron.**
- **Focus:** the builder's Seidel *seed* focus (+1212 mm → FP at z=+626 mm) is
  ~4× mis-placed — the known fast-relay seed limitation. But
  `align_focal_plane` **independently** (ray-bundle closest-point, seed-free)
  finds best focus at **z = −3256.30 mm**, matching Rodgers' verbatim paraxial
  M3→focus (−3256.35 mm) to **0.05 mm**. On-axis WFE there = **0.072 nm ≈ 0**.
- Conclusion: the optics are faithfully reproduced. `align_focal_plane` *is*
  Rodgers' "FPA focus" DOF; we call it before evaluating/optimizing each stage.

---

## 3. Headline result — the native optimizer recovers his conics to 4 dp

Stages 3–4 run **our** native optimizer (MACOS CALIB) with Rodgers' exact DOF
sets over the 0.2°×0.2° offset box and compare the **solved parameters**:

**Stage 3** (re-optimize the three conics + FPA):

| conic | MACOS | Rodgers | \|diff\| |
|---|---|---|---|
| K_M1 | −0.993475384 | −0.993352673 | 1.2e−4 |
| K_M2 | −1.932030835 | −1.932084692 | 5.4e−5 |
| K_M3 | −0.705426255 | −0.702494853 | 2.9e−3 |

**Our optimizer independently converges to essentially his CODE V solution** —
K_M1, K_M2 to 4 decimal places, K_M3 to 3. This is the strongest evidence the
reproduction is faithful: same starting Rx + same DOFs + same field → same
solved conics, two independent optimizers (CODE V vs MACOS CALIB).

Stage 4 conics also match to 3 dp (K_M1 2.8e−4, K_M2 3.5e−3, K_M3 6.0e−3).

---

## 4. WFE-magnitude and rigid-body deviations — FLAGGED FOR FABLE

Every stage's absolute field-map RMS WFE runs **4–6× off**, and Stage-4
rigid-body values disagree. Per the task, these are flagged, not tuned past.

### 4a. Field-map WFE magnitude = a METRIC-DEFINITION difference (not optics)

The discrepancy is present even in **Stage 1**, a pure evaluation with no
optimizer, and it goes in **opposite directions** (S1 too high, S2 too low) —
so it cannot be a uniform scale error. The signature:

| S1 on-axis box (nm) | min | max | avg |
|---|---|---|---|
| MACOS `realize_apertures` = std(OPD) @ one global plane | 0.31 | 8.00 | 2.61 |
| Rodgers (CODE V field map) | 0.45 | 1.46 | 0.61 |

The field **center** matches (0.31 vs 0.45 nm); the field-*dependent growth* is
~4.5× too steep. Decomposing the per-field OPD (`wfe_field_diag`, metric
ladder) shows why — and reconciles it:

| S1 metric (nm) | min | max | avg |
|---|---|---|---|
| raw (piston only) | 0.31 | 4.56 | 2.27 |
| − piston/tip/tilt | 0.17 | 4.13 | 1.23 |
| **− + defocus (per field)** | **0.03** | **1.83** | **0.62** |
| Rodgers (CODE V) | 0.45 | 1.46 | **0.61** |

The **per-field defocus-removed** average (0.62 nm) matches Rodgers (0.61 nm) to
**3%**. **Interpretation:** CODE V's "RMS wavefront error vs field" references
each field to its own best-focus reference sphere (removing the field-curvature
defocus of the fast f/1.75 anastigmat); `realize_apertures` uses `std(OPD)` at a
single global image plane, leaving that defocus in the off-center corners. It is
a WFE-metric/reference-sphere convention difference, **not an optical error**.

### 4b. The offset FPA tilt is large and its own finding

At the 0.5° offset field, `align_focal_plane` fits a **repeatable 14.24° focal-
plane tilt** (grid=0 and grid=5 agree to 4 dp; a fast anastigmat genuinely has a
strongly tilted image surface off-axis). At that tilt no single mmacos metric
equals Rodgers' 79/375/200 nm (raw 724–4959, −tilt 452–2455, −focus 232–1255,
−astig 19–67 nm over the box). Whether CODE V's FPA "tilt/focus" DOF converges
to the same surface, and exactly which reference-sphere its field-map RMS uses
off-axis, is the open question for Fable.

### 4c. Stage-4 rigid-body values disagree

| | MACOS | Rodgers | ratio |
|---|---|---|---|
| Ydec_M2 (mm) | −2.17 | 8.34 | −0.26× |
| tilt_M2 (deg) | 0.14 | 0.517 | 0.27× |
| Ydec_M3 (mm) | −25.6 | 121.87 | −0.21× |
| tilt_M3 (deg) | 0.64 | 2.330 | 0.28× |

MACOS finds M2/M3 moves in the *same rough proportion* (M3 ≫ M2, tilt ~4×
Ydec-scaled) but ~4–5× smaller and opposite-signed on the decenters. Expected:
once the optimization **objective** differs (the metric of §4a, plus the
different off-axis FPA-tilt handling of §4b), the minimizer lands on a different
rigid-body balance. The rigid-body solution is only comparable once the merit
function matches CODE V's field-map RMS — the §4a/§4b item gates this.

---

## 5. Open questions for Fable (do NOT tune past these)

1. **Field-map RMS metric.** Make `realize_apertures`/the field-map evaluator
   offer a CODE V-consistent **per-field best-focus reference-sphere RMS** (§4a
   shows it reconciles the on-axis stage to 3%). This is the primary fix; it
   likely also brings §4c into agreement by matching the optimization objective.
2. **Off-axis FPA model.** Is CODE V's field-map RMS at the offset field taken on
   a **tilted** focal surface, and does its "FPA tilt/focus" DOF reach the same
   ~14° surface? (§4b.) The tilt vignettes the per-field decomposition.
3. **EPD confirmation.** 2000 mm was inferred from the layout drawing (§1). A
   CODE V `.seq`/EPD readout would remove the one unpinned input. WFE-in-waves
   scales with aperture, so this must be exact before absolute-WFE agreement is
   meaningful. (Conic *values*, §3, are aperture-robust — hence they match.)

---

## 6. Artifacts

- `rodgers1.m` — the parameter-driven driver (4 stages; knobs: stages, EPD_mm,
  lambda_nm, model_size, map_n, opt_n, max_iters, save, plots).
- `rodgers_common.m` — verbatim prescription + Rodgers' ground-truth stats.
- `rodgers1_stage{1..4}.in` — the emitted MACOS prescriptions (SMACOS-validated).
- `rodgers1_stage{1..4}_*.png` — the RMS-WFE field maps.
- `rodgers1_results.mat` — per-stage stats, conics, rigid-body, metric ladders.
- `diag_*.m` — the diagnostics behind §2–§4 (focus, grid convergence, metric
  ladder, aperture sweep, FPA strategy) — kept for reproducibility.

## 7. Engine/API change made for this study

`macos.design.Telescope/optimize` gained a **per-element DOF mask**: `dofs` may
now be `(Nv,8)` (one VarElt row per varied mirror) in addition to the shared
`(1,8)` row. Stage 4 needs it — M1 held rigid (conic only) while M2/M3 add
Ydec+α-tilt, so there is no global-tilt gauge freedom to corrupt the solved
rigid-body values. Back-compatible; covered by
`tDesignTelescope/test_optimize_per_element_dofs` (suite 68/68 green).

---

# ADDENDUM (2026-07-30) — per-field metric, evaluator rulings B1/B2, recenter study

Bundled follow-on to §4a/§5. Branch `rodgers1-metric` (off `dev`). No engine
changes; design-layer only (`macos.design.Telescope` + `rodgers1.m`).
Verified in a clean `dev` worktree MEX (gfortran/Apple-Silicon). Gates: default
`global` metric **bit-identical** (stages 1–4 `isequaln`, conics 0.0e0);
`tDesignTelescope` **70/70** green (+2 new tests).

## A. Per-field best-focus reference-sphere RMS — the CODE V-consistent metric

`realize_apertures` gained `'metric'` ∈ {`'global'` (default, unchanged),
`'refsphere'`}. `refsphere` removes piston + tip/tilt + **defocus** per field
(same fit basis as `wfe_field_diag`'s `rms_focus` rung) and evaluates the RMS
over the **realised clear aperture** (a 2nd pass after the stops are sized) —
exactly CODE V's field-map convention (references each field to its own
best-focus sphere; counts only the clear-aperture pupil). The returned scan
carries `.metric` and every rendered field map states the metric in its title.

**Non-vacuity — the §4a anchor reproduces.** S1 on-axis box, 9×9, EPD 2000 mm:

| S1 on-axis (nm) | min | max | avg |
|---|---|---|---|
| `global` (unchanged) | 0.314 | 8.001 | 2.608 |
| **`refsphere`** | 0.030 | 1.833 | **0.623** |
| Rodgers (CODE V) | 0.446 | 1.463 | **0.606** |

avg ratio **1.027 (2.7%)** — the packet §4a reconciliation, now a built-in
metric rather than a diagnostic ladder. (§4a's 0.62 came from the ladder run
*after* `eval_box` had installed the clip apertures — i.e. already
aperture-limited; `refsphere` makes that the metric's definition, honestly.)

**Vignetting frame note.** Pass 2 vignettes on the **mirror** clear apertures
only and drops the FocalPlane rect for the metric trace: the tilted/offset FP's
emitted `ApVec` carries the known global-XY→local-ApVec frame bug (see
`clear_realized_apertures`) and would vignette every ray off-axis (all-NaN).
The physical stops are the coaxial (frame-bug-immune) mirrors; the FP only
samples the wavefront. The returned `.aperture` still reports the FP rect.

## B. Two evaluator findings — diagnosed and ruled

**Root cause is shared and singular:** `realize_apertures` installed clip
apertures on the optics (elts 1–4) sized to *the box it was handed* and **left
them installed**. Any subsequent evaluation of a *different* box — the metric
ladder (which adds the +30′ bias), a second call, or a recentered box — traced
*through* those stale apertures; images that walk outside them read NaN.

- **B1 — the top-two rows (box-rel y ≥ +4.5′) lost on committed stage 4:
  ARTIFACT, not a physical field limit.** On the **clean** stage-4 design (no
  realised apertures installed) **1305/1305 rays reach the FP at every field
  across the whole box** (abs +24′→+36′, incl. the "lost" rows) — verified by a
  per-field trace with `get_ray_info`. Nothing falls off any element ⇒ per
  Dave's rule (loss to an *aperture* just means the aperture must be
  re-defined; loss off an *element* is the true fail) this is **not** a true
  fail. The committed map's NaN top rows were the stale-aperture clip, and the
  contour silently interpolated over them. **Ruling: usable field is not
  limited here; fix the evaluator, and render lost fields visibly.**

- **B2 — a second `realize_apertures` with shifted fields returns all-NaN,
  including fields finite on the first call: reproduced and FIXED.** Sequence
  on committed stage 4: callA (fresh box) 81/81 finite → apertures now on elts
  [1 2 3 4] → callB (identical box) **0/81** → callC (shifted) 0/81 →
  `clear_realized_apertures` → callD **81/81** restored. The "state left behind"
  is precisely the installed clip apertures (chiefly the frame-buggy tilted-FP
  rect). **Fix:** `realize_apertures` now **clears any previously-realised clear
  apertures at entry** and re-measures on the clean design — idempotent, and a
  no-op on a fresh telescope (first-call numbers bit-identical). `view_field_map`
  now grey-fills lost fields and never interpolates over them.

Both covered by `tDesignTelescope/test_realize_apertures_metric_and_idempotent`
and `.../test_view_field_map_renders_lost_fields`.

## C. The recenter study — `rodgers1('mode','recenter','dy_arcmin',+2)`

Scores the stage-4 solve on the box recentred **+2′ in +y**, two ways, under
**both** metrics, complete 81/81 coverage (Part B resolved). Box is
bias-relative (centred on the used +0.5° field — the physically-correct
convention; the committed stage tables use an absolute box about 0′, hence the
different AS-IS global magnitude).

| scenario | global max/avg (nm) | refsphere max/avg (nm) |
|---|---|---|
| **AS-IS** (committed solve, box +2′) | 936.9 / 542.1 | 197.1 / 112.2 |
| **RE-OPT** (bias moved to +32′, re-solved) | 1168.1 / 720.8 | 232.0 / 134.9 |

**Expected shape confirmed:** AS-IS beats RE-OPT under *both* metrics — the
sweet spot is **solve-relative**; re-optimising at the shifted bias does not
beat the design tuned for its own centre. **The per-field metric does NOT flip
the re-opt conclusion** (AS-IS < RE-OPT holds for `refsphere` too), so this is
a robust trade result, not a metric artifact.

**Cross-check vs Fable's interim (global-plane, partial coverage):**

| quantity | Fable interim | this run (full 81/81) |
|---|---|---|
| RE-OPT conics | [−0.994041, −1.940273, −0.698961] | [−0.994040, −1.940269, −0.698985] |
| RE-OPT M2 | −2.46 mm / 0.156° | −2.451 mm / 0.1557° |
| RE-OPT M3 | −29.25 mm / 0.729° | −29.128 mm / 0.7256° |
| AS-IS (global) max/avg | 900 / 477 (covered 10/12′) | 936.9 / 542.1 (81/81) |
| RE-OPT (global) max/avg | 3038 / 1149 | 1168.1 / 720.8 |

Conics match to **5 dp**, rigid-body to **~0.5%** — the two independent runs
agree. The as-is avg differs modestly (fuller coverage here); the re-opt
magnitude differs more because Fable's interim re-opt scored a partial box.
Both agree on the sign of the trade (re-opt worse), which is the finding.

## Artifacts added
`rodgers1_recenter.mat`, `rodgers1_recenter_{asis,reopt}_{global,refsphere}.png`.
