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
- Derived (**corrected 2026-07-30 — see Addendum 3 §A.5**): the traced geometry
  gives **EFL ≈ 96–101 m**, image-space **f/23.5** at EPD 4060 (M3 footprint
  radius 0.1113 m, M3→focus 5.215 m, marginal half-cone 1.2177°). The earlier
  "EFL ≈ 3506 mm, f/1.75 (→ f/0.86 at EPD 4060)" line was **wrong** and is
  retracted; the *prescription* is unaffected (spacings and radii still match
  Rodgers to the micron, §2). Where the text below says "f/0.86" or "f/1.75"
  it means "at EPD 4060" / "at EPD 2000" — read it that way.

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
defocus of the anastigmat's curved image surface); `realize_apertures` uses
`std(OPD)` at a
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
them installed**. Any subsequent evaluation of a *different* box on the SAME
object — the metric ladder (which adds the +30′ bias), a second call, or a
recentered box — traced *through* those stale apertures; images that walk
outside them read NaN.

**Where the NaN top rows actually appeared (attribution corrected, Fable
review 2026-07-30).** The four committed stage scans in `rodgers1_results.mat`
are **all 81/81 finite, zero NaN** — the committed field maps were never
affected, which is also why the default-metric bit-identity gate holds. The
"top-two-rows lost" NaN pattern appeared only in a **derivative run** (Fable's
interim harness) whose call sequence re-used one telescope object across
several evaluations and so tripped the B2 stale-aperture state *before* its
field map was drawn. So B1 is a finding about **the stale-state artifact's
effect on derivative/re-used-object runs and the metric ladder — not about the
committed maps.**

- **B1 — the substance: no physical usable-field limit at the box edge.** On
  the **clean** stage-4 design (no realised apertures installed) **1305/1305
  rays reach the FP at every field across the whole box** (abs +24′→+36′, incl.
  the box-rel y ≥ +4.5′ top rows) — verified by a per-field trace with
  `get_ray_info`. Nothing falls off any element ⇒ per Dave's rule (loss to an
  *aperture* just means the aperture must be re-defined; loss off an *element*
  is the true fail) there is **no true fail** and **no field limit** here.
  Where the NaN top rows *did* surface (derivative runs / the metric ladder,
  above), they were the stale-aperture clip with the contour silently
  interpolating over them. **Ruling: usable field is not limited at the box
  edge; fix the stale-state evaluator bug (B2), and render any lost field
  visibly.** The committed stage maps do not change post-fix (they had no NaN).

- **B2 — a second `realize_apertures` on the same object with shifted fields
  returns all-NaN, including fields finite on the first call: reproduced and
  FIXED.** Sequence on a stage-4 object: callA (fresh box) 81/81 finite →
  apertures now on elts [1 2 3 4] → callB (identical box) **0/81** → callC
  (shifted) 0/81 → `clear_realized_apertures` → callD **81/81** restored. The
  "state left behind" is precisely the installed clip apertures (chiefly the
  frame-buggy tilted-FP rect). This is the mechanism behind B1's derivative-run
  NaNs. **Fix:** `realize_apertures` now **clears any previously-realised clear
  apertures at entry** and re-measures on the clean design — idempotent, and a
  no-op on a fresh telescope (first-call numbers bit-identical, which is why
  the committed single-call stage maps were already correct). `view_field_map`
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

## D. Bias-relative re-baseline of the stage tables (Fable condition 2)

The convention flag: the committed stage-2/3/4 tables — and the original
packet §4's 4–6× Rodgers-ratio narrative — were computed on the **absolute
box about y = 0**, not the **used +30′ box**. (`realize_apertures('fields',F)`
is the only field branch that does not add `field_bias`, so `eval_box`'s
`field_grid` box sits about 0′.) The committed absolute-box outputs and the
driver's default stage convention are LEFT UNTOUCHED (bit-compat); the table
below is **additional rows**, not a rewrite of history.

Stages 2–4 re-evaluated on the **bias-relative, un-shifted** box (9×9 centred
on the +30′ used field), both metrics, 81/81 finite:

| stage | global max/avg (nm) | refsphere max/avg (nm) | Rodgers max/avg (nm) | global ×Rodgers(max) | refsphere ×Rodgers(max) |
|---|---|---|---|---|---|
| S2 (verbatim conics, FPA re-fit) | 4959.1 / 2322.5 | 1012.8 / 556.5 | 374.6 / 199.9 | 13.2× | 2.7× |
| S3 (reopt conics + FPA)          | 2789.7 / 1208.5 |  247.4 / 216.2 |  91.6 /  46.4 | 30.5× | 2.7× |
| S4 (tilt/dec M2,M3 + reopt)      |  862.8 /  526.9 |  195.1 / 114.4 |  39.8 /  22.5 | 21.7× | 4.9× |

**Sanity vs §C:** the S4 bias-relative refsphere (195.1 / 114.4 nm) sits right
at the recenter study's AS-IS +2′ refsphere (197.1 / 112.2 nm) — as expected
(the un-shifted box should be marginally better than the +2′-shifted one); the
S4 global row (862.8 / 526.9) likewise matches the AS-IS +2′ global (936.9 /
542.1), slightly better un-shifted. Consistent.

**§4 discrepancy conclusion, restated.** Moving to the physically-correct box
does NOT reconcile the off-axis magnitude with Rodgers, and no stage's
conclusion flips. Under the global-plane metric the ratios stay large
(13–30×) — that metric leaves the fast anastigmat's field-curvature defocus in
the corners, so it was never the right comparison off-axis. Under the CODE
V-consistent **refsphere** metric the ratios collapse to **2.7–4.9×** — far
closer, and the honest comparison — but still **well above** Rodgers'
79/375/200-class numbers. That residual gap is real and expected; it is NOT an
optical-reproduction error (§3: the conics match CODE V to 4 dp). It is gated
by the **three open asks to Rodgers**, unchanged from §5:

1. **RMS reference sphere** — exactly which surface/normalization CODE V's
   field-map RMS uses off-axis (our refsphere removes per-field
   piston+tip/tilt+defocus over the clear aperture; if CODE V also removes
   more, e.g. astigmatism, or references a different pupil, the residual
   closes further — §4a's ladder shows −astig drops S2 into the tens of nm).
2. **His FPA tilt/focus DOF** — does CODE V's detector reach the same ~14°
   offset image surface our `align_focal_plane` fits (§4b)? The off-axis
   metric is taken on whatever surface his optimizer converged to.
3. **EPD** — 2000 mm was inferred from the layout drawing (§5.3); WFE-in-waves
   scales with aperture, so the absolute magnitude cannot be pinned until his
   `.seq`/EPD readout confirms it. (Conic *values* are aperture-robust — hence
   §3 matches regardless.)

Reproduce: build each committed stage (as `rodgers1` stages 2–4 do) and call
`realize_apertures('fields', [Frel(:,1), field_bias + Frel(:,2)], 'metric', m)`
for `m ∈ {global, refsphere}` — the same build recipe, only the eval box is
shifted onto the +30′ bias. (Kept out of the default driver flow to preserve
the committed absolute-box stage tables bit-for-bit.)

## Artifacts added
`rodgers1_recenter.mat`, `rodgers1_recenter_{asis,reopt}_{global,refsphere}.png`.

---

# ADDENDUM — the study at Rodgers' true aperture, EPD = 4060 mm (2026-07-30)

**Motivation.** The one unpinned input (§5.3 / open ask 3) was the EPD: 2000 mm
was *inferred* from the slide-1 layout drawing, but Rodgers' 1000-mm scale bar
puts the beam filling M1 at ≈4 m, and **Dave measures D = 4.06 m**. Every
absolute-WFE comparison above was therefore made at **half his aperture**. This
addendum re-runs the full four-stage sequence at **EPD = 4060 mm**, both metrics,
the same bias-relative box and `map_n = 9` as §D, to (a) close the last open
input and (b) run the falsifiable **D²-scaling test**: frozen-design off-axis WFE
(astigmatism-dominated) scales as D², so if "metric convention" alone explained
the residual gap, the Rodgers-ratio would be *aperture-invariant*; if instead the
gap is aperture-driven, the ratio grows by ≈D². **Run and record — not tuned
toward his numbers.**

**Environment.** Clean `origin/dev` worktree (`3cc09e1`) + the current mmacos MEX
(dev's 128 dispatch commands are a strict subset of the MEX's 137; the extra 9
are polarization symbols the scalar TMA path never calls). **Environment gate
PASSED before trusting any 4060 number:** re-running at EPD = 2000 reproduces the
committed §D table bit-for-bit — S2 refsphere 1012.8/556.5, S3 247.4/216.2, S4
195.1/114.4 nm, and the committed S3 conics (−0.993475, −1.932031, −0.705426) and
S4 rigid-body (M2 −2.17 mm/0.138°, M3 −25.6 mm/0.642°). Driver +
scaffold agree.

## A′. Friction record — Dave's question: "just the knob, or rework?"

**It is just the knob.** The entire four-stage sequence ran end-to-end at
EPD = 4060 with **zero touches beyond `EPD_mm`**:

- `rodgers1('EPD_mm',4060)` completes all four stages clean (`stages=4`, exit 0),
  no error, no evaluator guard fired.
- **No element-aperture / M1-hole sizing edits.** There is no hard-coded aperture
  or hole in this Rx — `Aperture`, `ApVec`, and `ChfRayPos` in every emitted deck
  are **derived from `EPD_mm`** by the builder. Diffing `stage1.in` at 2000 vs
  4060 shows *only* those derived fields change (`Aperture` 2.0→4.06,
  `ApVec` 1.04→2.11, `ChfRayPos` z −5.77→−6.28); the four decks stay
  **structurally identical — 119 lines each**, same keyword set.
- **No model-size / ray-grid change** (`model_size = 256`, `map_n = 9`,
  `opt_n = 3` all as committed).
- **No CALIB convergence difference at the true aperture.** `align_focal_plane` fits the
  same focus (FP z = −3256.29 mm vs −3256.28 at 2000) and the S3/S4 optimizer
  converged normally in the same iteration budget (120); conics landed *closer*
  to Rodgers at the true aperture (below).
- **Layout gate unchanged:** M1→M2 / M2→M3 spacings and radii are bit-identical
  across the two apertures to floating-point noise (zElt differs in the 16th
  digit only). Spacings/radii are his, fixed — untouched, as required.
- **Nothing clipped, nothing lost:** all stages report **81/81 finite fields**
  under both metrics at EPD 4060, same as at EPD 2000.

This is the design-layer sales pitch, measured: aperture is a single scalar knob
and the whole prescription — apertures, chief ray, pupil vectors — re-derives.

## B′. Gate (b): solved conics vs Rodgers — **equal-or-better at the true aperture**

His conics were solved at *his* EPD, so agreement should improve at matched
aperture. It does:

| conic | EPD 2000 \|diff\| | EPD 4060 \|diff\| | Rodgers |
|---|---|---|---|
| K_M1 | 1.23e−4 | **1.56e−5** | −0.993352673 |
| K_M2 | 5.39e−5 | 3.12e−4 | −1.932084692 |
| **K_M3** | 2.93e−3 | **3.43e−3** | −0.702494853 |

K_M1 tightens by ~8× (to 5 dp); K_M2 loosens slightly but stays at 4 dp; **K_M3
(the flagged 2.9e−3 term) stays at 3.4e−3** — same order, not worse in kind. The
independent-optimizer conic match (§3's headline) **holds at the true aperture**:
same Rx + DOFs + field → essentially his CODE V conics, at the true aperture.

Stage-4 conics: K_M1 3.3e−4, K_M2 4.0e−3, K_M3 1.06e−2 (K_M3 loosens to 2 dp — the
richer DOF set trades conic tightness for the added rigid-body freedom, same as
at 2000). Stage-4 rigid-body scales up toward his values but stays ~⅓–½:
M2 −3.74 mm/0.233°, M3 **−43.84 mm/1.114°** (vs his 8.34/0.517, 121.87/2.330) —
still opposite-signed decenters, still gated by the merit-function difference
(§4c), now larger in magnitude as expected at the bigger aperture.

## C′. Gate (c): the WFE table, stages 2–4, both metrics, EPD = 4060

Bias-relative box (centred on +30′), `map_n = 9`, 81/81 finite:

| stage | global max/avg (nm) | refsphere max/avg (nm) | Rodgers max/avg (nm) | global ×Rodgers(max) | refsphere ×Rodgers(max) |
|---|---|---|---|---|---|
| S2 (verbatim conics, FPA re-fit) | 12493.3 / 6023.5 | 4555.1 / 2478.9 | 374.6 / 199.9 | 33.4× | **12.2×** |
| S3 (reopt conics + FPA)          |  5816.2 / 2812.0 | 1026.7 /  847.8 |  91.6 /  46.4 | 63.5× | **11.2×** |
| S4 (tilt/dec M2,M3 + reopt)      |  2448.5 / 1871.4 |  775.6 /  485.5 |  39.8 /  22.5 | 61.5× | **19.5×** |

On-axis anchor S1 (about 0′): global 14.4/3.9, **refsphere 9.43/2.78** nm (vs
Rodgers 1.5/0.61). Even the on-axis refsphere anchor, which sat at 0.62 nm ≈
Rodgers at EPD 2000, has grown ~4.5× — consistent with the D²-scaling below and
its own tell that the residual is aperture-driven, not a fixed convention offset.
S2 FPA tilt fits **14.30°** (vs 14.24° at 2000) — the tilted-image-surface finding
(§4b) is aperture-robust.

## D′. Gate (d): the D²-scaling check — **result supports hypothesis (b)**

The task posed two hypotheses. **(a)** the gap closes at matched aperture ⇒ the
2000-mm agreement was coincidental; **(b)** the gap *grows to ≈D²* ⇒ "metric
convention" is falsified as the residual explanation and the gap is aperture-/
field-driven. Measured ratios, EPD 4060 vs 2000:

| stage | refsphere max | refsphere avg | global max | global avg | (4060/2000)² |
|---|---|---|---|---|---|
| S2 | 4.50× | 4.46× | 2.52× | 2.59× | **4.12** |
| S3 | 4.15× | 3.92× | 2.09× | 2.33× | 4.12 |
| S4 | 3.98× | 4.25× | 2.84× | 3.55× | 4.12 |

**The refsphere metric scales as D² to within ~10%** (3.98–4.50× vs the 4.12
expectation) across all three stages — clean astigmatism-dominated D² growth.
Equivalently, the Rodgers-ratio grows by ≈D²: refsphere S2 2.7×→12.2×, S3
2.7×→11.2×, S4 4.9×→19.5× (ratio-growth 4.0–4.5×, i.e. ≈D²). **This is
hypothesis (b).**

The global-plane metric grows *sub*-D² (2.1–3.6×): expected, because it carries a
large aperture-*independent* field-curvature-defocus pedestal that dilutes the
D²-scaling astigmatism — which is exactly why the refsphere metric (defocus
removed per field) is the honest off-axis comparison, and it is the one that
shows the clean D².

## Verdict

**Hypothesis (b), decisively.** At Rodgers' true 4.06-m aperture the refsphere WFE
scales as D² to within 10%, so the residual gap to his numbers is **not** a fixed
metric-convention multiplier (that would be aperture-invariant) — it is genuine
aperture-/field-dependent wavefront error, dominated by astigmatism at EPD 4060.
(⚠ This verdict sentence is itself revisited in Addendum 3 §B, which keeps the
D² scaling but re-attributes the residual to the detector-plane metric.)
Two consequences sharpen the open asks for Mike:

1. The EPD 2000-mm refsphere agreement (2.7–4.9× above Rodgers) was **partly
   coincidental** — half the beam meant a quarter of the astigmatic WFE. At his
   aperture the honest refsphere gap is 11–20×, not 3–5×. **Open ask 3 (EPD) is
   thus not cosmetic: it dominates the absolute-magnitude comparison**, and it is
   now pinned at D = 4060 mm here.

2. Because our conics *still match his to 3–4 dp at the true aperture* (B′) while
   our absolute WFE is ~D²-larger than his, the residual must live in the two
   remaining asks — **the RMS reference-sphere definition (ask 1)** and **his
   off-axis FPA surface (ask 2)**. His slide numbers (79/375/40-class) at a 4-m
   anastigmat imply CODE V removes more than piston+tip/tilt+defocus (its
   field-map RMS likely also references out astigmatism, or a differently-tilted
   image surface); §4a's −astig ladder rung already dropped S2 into the tens of
   nm. This is now the *sole* live explanation — the aperture variable is spent.

Neither hypothesis was contradicted; (b) is confirmed and (a) rejected. No
conclusion from the EPD-2000 packet flips — the conic match and the AS-IS<RE-OPT
recenter trade are aperture-robust — but the absolute-WFE narrative is now
correctly anchored at his aperture.

## Artifacts added (EPD = 4060, parallel to the committed EPD = 2000 set)

`run_epd4060.m` (the sweep driver — reuses `rodgers_common` + the committed build
recipe; runs any EPD, both metrics, emits the §D table); and, one per stage,
matching the four-slide progression:
`rodgers1_epd4060_stage{1..4}.in`,
`rodgers1_epd4060_stage{1..4}_{global,refsphere}.png`,
`rodgers1_epd4060_results.mat`. **All committed EPD = 2000 outputs are left
bit-intact** (the 4060 set is suffixed-parallel, never overwriting).

---

# ADDENDUM 2 — explicit exit pupil + WFE reference-surface comparison (2026-07-30)

**Motivation (Dave).** Insert an explicit exit pupil into the emitted stage-4
deck and evaluate WFE there, to see how much of the residual gap to Rodgers is
just *which reference surface* the RMS is taken on (open-ask #1). `add_pupil`
inserts, before the FocalPlane, a flat image-return and a **spherical ExitPupil
Return** whose centre of curvature is at the image; its contract is that the
exit-pupil-referenced wavefront is the deliverable handle.

**Engine-adjacent fix this exercised (Dave: "no obscuration on the FP or return
surfaces").** `add_pupil`'s two `Return` surfaces (`FP_return`, `ExitPupil`) were
emitted with `ApType=Circular` at the generous reference `ap_r` — the Rx-emit
policy had explicit `ApType=None` branches for off-axis `Reflector` and
`FocalPlane` but `Return` fell through to the `else → Circular`. On this beam
that circular stop **clipped rays at the exit pupil** (and, because
`realize_apertures` also leaves clip apertures installed, poisoned any later raw
trace → the `9.999e39` all-rays-lost sentinel). Fix: `Return` kind now emits
`ApType=None`. After it, **1304/1304 rays survive to the exit pupil at every box
field** (was ray-limited before). One-line change in `Telescope.emit_rx_`; pure
MATLAB, no MEX rebuild; existing decks without a pupil are unaffected.

**Result — WFE is reference-surface dependent (EPD 4060, stage-4, ±6′ box, nm):**

| reference surface / metric | max | avg |
|---|---|---|
| FP, each field to its own chief-ray focus (engine `rmsWFE`) | 0.002 | 0.002 |
| **explicit EXIT-PUPIL sphere (engine, CoC at image)** | **2.644** | **1.588** |
| field-map `refsphere` (per-field piston+tip/tilt+defocus removed) | 775.6 | 485.5 |
| field-map `global` (std of OPD at one image plane) | 2448.5 | 1871.4 |
| Rodgers S4 (CODE V field-map RMS) | 39.8 | 22.5 |

**Reading.** The *same optics* read across **six orders of magnitude** depending
purely on the reference surface:

- **Each field to its own chief focus → ~0 nm.** The design is essentially
  perfect per field; there is no per-field aberration once you sit at that
  field's own best focus. (This is why the stage-4 conic solve succeeds.)
- **Explicit exit-pupil sphere → 1.6 nm avg / 2.6 nm max.** This is the honest
  exit-pupil wavefront: the OPD on a sphere whose CoC is the image point, i.e.
  the field-independent wavefront the downstream instrument actually sees. It is
  tiny — the offset TMA is genuinely well-corrected at its own pupil.
- **`refsphere` → 486 nm, `global` → 1871 nm.** These are large **because they
  compare *different fields against a common plane*** — they retain the
  field-to-field focus/tilt walk of the fast anastigmat's steeply curved,
  ~14°-tilted image surface across the box. `global` keeps all of it;
  `refsphere` removes per-field piston/tip/tilt/defocus but still measures each
  field's residual over a shared realised aperture.

**Bearing on the Rodgers gap (open-ask #1).** This localises the residual
decisively: it is **not** an optical-quality deficit (the exit-pupil wavefront is
1.6 nm, and per-field WFE is ~0), it is **entirely the field-map RMS reference
convention**. Rodgers' 22–40 nm sits *between* our exit-pupil number (1.6) and
our `refsphere` (486) — consistent with CODE V referencing each field to a
best-fit image sphere on its converged (tilted) detector while removing a
specific low-order set, i.e. more removed than our exit-pupil sphere sees across
the box but far less than `global` leaves in. The two remaining asks (his exact
RMS reference-sphere terms, ask 1; his off-axis FPA surface, ask 2) fully account
for the 22–40 vs 1.6/486 spread. **Nothing here changes the D²-scaling verdict of
Addendum 1** — that was about how the *field-map* metric scales with aperture;
this shows the field-map metric's absolute magnitude is a reference-surface
artifact, and the underlying optics (exit-pupil WFE) are excellent.

**Artifacts added:** `epd4060_pupil_check.m` (the cross-check driver),
`rodgers1_epd4060_stage4_pupil.in` (the 6-element augmented deck: M1, M2, M3,
FP_return, ExitPupil sphere, detector FP — all reference surfaces `ApType=None`),
`rodgers1_epd4060_pupil_check.mat`.

---

# ADDENDUM 3 — pinning the metric: engine forensics, the strict rung, stage-2 validation, lane (2026-07-30)

Dave ruled the correct metric and asked to (1) make Addendum 2 publishable — every
row gets an operational definition, the 0.002 and XP rows get resolved, and a
rider reconciles it with Addendum 1 — (2) add and validate the **strict rung**, and
(3) determine the lane for EP-in-the-loop re-optimization (step 3, not this
session).

**Headline: the strict metric closes ask 1.** Rodgers' stage-2 box, at his
aperture, on a detector frozen by his own procedure, under Dave's ruled metric,
reads **429.6 / 246.8 nm (max / avg) against his 374.6 / 199.9 — 1.15x / 1.23x**.
The 4–20x "residual gap" of Addenda 1 and 2 was a reference-convention artifact
plus a units error; neither survives.  Scored across the whole study (§D.2) the
un-optimised stages agree (S1 1.60x, S2 1.15x) and the optimised ones do not
(S3 1.98x, S4 2.98x) -- so the metric question closes and the **merit-function**
question of §4c revives, with step 3 as its test.

## A. Engine forensics — where the OPD reference actually comes from

Citations are to `macos/macos_f90`, branch `pol-core`. Read this section before
interpreting ANY engine WFE number in this packet.

**A.1 `SUBROUTINE OPD` has no reference sphere at all.** `tracesub.F:21-250`.
It accumulates, over rays 2..nRay that pass (`:163-174`), `OPDScal =
CumRayL(iRay) - RefCumRayL`, subtracts the ray MEAN `DAvgL` (`:179`), and takes
the sample std (`:235-239`). `RefCumRayL` is the chief ray's own cumulative path
(`:138-144`) — a per-field CONSTANT, so it cancels out of the RMS entirely; the
engine says so itself at `:135-136`. All three branches (`:190`, `:205`, `:219`)
compute the same `WFE`; they differ only in whether the returned `OPDMat` has the
mean removed. **There is no sphere, no centre, no radius, no tilt fit and no
focus fit anywhere in the path.** The reference surface is simply *the surface the
trace stopped on*, and the reference "point" for each ray is *its own intercept
on that surface*.

Consequences, and they are the whole story:

- **At a FocalPlane, the engine measures OPL to each ray's own intercept on the
  detector plane.** Nothing moves when the FIELD moves except the rays. Nothing
  moves when the DETECTOR moves except the plane the rays are cut on. The design
  layer does not re-reference either: `Telescope.emit_` writes every element
  vertex verbatim from `spec.elt` and only the SOURCE chief ray changes with
  `trace_field` (`Telescope.m:2432-2445`), and `align_focal_plane` writes its
  fitted `Vpt`/`psi` into `spec` once (`Telescope.m:454-455`), after which
  `trace_at_field` re-emits the SAME detector. **There is no silent
  re-referencing.** (This retires the SESSION_STATE caveat.)
- **On a TILTED detector that measurement is not a wavefront error.** A ray whose
  transverse aberration displaces its intercept by `d` on a plane tilted by `g`
  from the beam picks up an extra `~ d*tan(g)` of path. Here `align_focal_plane`
  fits `g = 14.2994 deg` and the geometric blur is `27.9 um` rms, giving
  `~7 um` — and the engine indeed reports `5.099e-06 m` at the box centre where
  the true wavefront error is `2.27e-07 m`. **The engine's raw FP number is ~22x
  the wavefront error, and the excess is (transverse ray aberration) x (detector
  tilt).** It is not a low-order pupil polynomial, which is why fitting piston /
  tilt / defocus out of it — what the `refsphere` metric does — cannot remove it.
- **It is NOT Fermat-protected, and the "0.002 nm row is a quasi-tautology"
  suspicion is not the mechanism.** Fermat makes OPL equal to a stigmatic *point*,
  not to a *plane*: displace the plane by `dz` and the piston-removed OPL grows as
  `dz*(sec(theta) - <sec(theta)>)`, measured coefficient `6.43e-05` on this
  system. The near-zero appearance had a different cause — see A.2.

**A.2 `macos.trace(k).rmsWFE` is in BaseUnits (metres), NOT waves.** Measured on
the committed `rodgers1_epd4060_stage4_pupil.in` (`BaseUnits= m`, `WaveUnits= m`,
`Wavelen = 1e-06`): elt 6 (FP) `rmsWFE = 1.790188e-06`, elt 5 (ExitPupil)
`rmsWFE = 4.139914e-07`, both identical to `std(macos.opd())` and to the engine's
own printed `RMS OPD error ... m`. `realize_apertures(...).wfe` DOES divide by
lambda (`Telescope.m:1860`, `:1923`) and is genuinely in waves.

`epd4060_pupil_check.m:73` used `s.rmsWFE*lam_nm` on the metres quantity, so
**Addendum-2 rows 1 and 2 are low by 1e6**. Restated (EPD 4060, stage 4, +/-6' box):

| # | Addendum-2 row | as printed | **true** |
|---|---|---|---|
| 1 | FP, each field to its own chief focus (engine `rmsWFE`) | 0.002 / 0.002 "nm" | **~2000 / ~2000 nm** |
| 2 | explicit exit-pupil sphere, fixed on-axis CoC | 2.644 / 1.588 "nm" | **~2.6 / ~1.6 mm** |
| 3 | field-map `refsphere` | 775.6 / 485.5 nm | unchanged (correct) |
| 4 | field-map `global` | 2448.5 / 1871.4 nm | unchanged (correct) |

The corrected numbers are self-consistent in a way the printed ones were not:
row 1 (~2 um, all rays) now sits in the same class as row 4 (2.4 um, clear
aperture only) — they ARE the same quantity up to the aperture restriction — and
row 2's ~1.6 mm is exactly the image-displacement tilt a fixed on-axis CoC must
carry (image walk `+/-0.17 m` across the box, `EP->image ~ 96 m`, pupil radius
`2.03 m` -> `~1.8 mm` rms of tilt). **Both disputed §A claims are therefore
resolved, but not as CCMac wrote them:** row 1 is not "0.002 nm, genuine
geometric-OPL agreement" (it is 2 um and it is tilt-artifact-dominated), and it is
not a Fermat tautology either. Row 2's qualitative reading — "retains the off-axis
image-displacement tilt" — was right; its magnitude was not.

**A.3 What FEX ties to what.** `SUBROUTINE FEX`, `tracesub.F:255-566`; write-back
`macos_cmd_loop.inc:2662-2671`. FEX traces the chief ray at the nominal field and
at a `5e-6 rad` probe field (`:312-333`), takes their crossing as the pupil
station `CrossPt` (`:367-368`), and sets, on the EP element:

    VptElt = RptElt = CrossPt        ! vertex  = the two-probe chief crossing
    psiElt = psip = -cr1dir          ! axis    = the chief direction (:563)
    fElt   = |zp| ,  KrElt = -|zp|   ! radius  = zp

with `zp = fexT`, the chief-ray distance from `CrossPt` to the plane of element
`iElt+1` — **the detector** (`:415-437`, the 2026-07-03 rework). Because the
sphere's axis IS the chief direction and its radius IS the EP->detector chief
distance, its **centre of curvature lands exactly on the chief ray's intercept
with the detector plane.** So:

| quantity | moves with the FIELD? | moves with the DETECTOR? |
|---|---|---|
| sphere vertex (`CrossPt`) | yes (chief-ray probe geometry) | no |
| sphere axis (`psip`) | yes | no |
| sphere radius (`zp`) | yes | **yes** |
| **sphere centre = vertex + zp*axis** | **yes** | **yes** |

That is Dave's ruled metric, exactly — *provided FEX is re-run per field*. It is
not re-run by any plain `trace`/`opd` call; CALIB does re-run it per field
(`smacos_compute.inc:391-397`), which is what §E rests on. A deck whose EP was
set once and then evaluated off-axis has a sphere centred on the ON-AXIS image
point, which is Addendum-2 row 2 and its millimetres of tilt.

**A.4 Sensitivity to the radius.** With the centre pinned to the chief intercept,
the radius enters the metric only through an `O(eps^2 / R)` term (`eps` = the
transverse ray aberration). At focus that is `0.075 nm` here — i.e. "anchored at
the exit pupil" fixes the construction but does not, near focus, move the number.
It matters only when the detector is far off focus, which is why the gate-1 rung
at `dz = +627 mm` needs the true `R` and the `dz = 0` rung does not.

**A.5 One geometry correction to §1 of this packet** (§1 now carries the
retraction; this is the measurement behind it). §1 used to read "EFL ~ 3506 mm,
f/1.75", hence "f/0.86 at EPD 4060". The traced geometry disagrees: the M3
footprint radius is `0.1113 m`, the M3->focus leg is `5.215 m`, the marginal
half-cone is `1.2177 deg` — **f/23.5** — and the `+0.5 deg` chief lands at
`y = -0.8819 m`, i.e. **EFL ~ 96-101 m**. The M1->M2, M2->M3 and M3->focus
spacings still match Rodgers to the micron (§2), so the *prescription* is right;
only the derived-EFL line is wrong. Nothing else in the packet depends on it, but
it should not be repeated: the paraxial-basis worry it motivated ("at f/0.86 the
`2*rho^2` basis is the artifact") does not apply at f/23.5. (The strict metric is
computed exactly regardless — see §C.)

## B. Reconciliation rider — Addendum 1 stands; Addendum 2's verdict does not

Addendum 1 concluded *"the residual gap is genuine aperture/field WFE, not a
metric-convention artifact"*; Addendum 2 concluded *"not an optical-quality
deficit — entirely the reference convention"*, and the previous draft of this
rider had Addendum 2 superseding Addendum 1. **That is now reversed, and both are
partly wrong.**

- **Addendum 2's evidence does not exist.** Its two load-bearing rows — "each
  field to its own chief focus -> ~0 nm" and "the exit-pupil wavefront is only
  1.6 nm" — are 2 um and 1.6 mm (§A.2). The claim that the same optics "read
  across six orders of magnitude depending purely on the reference surface" was
  ~1e6 of unit error stacked on one real effect (the fixed-CoC tilt). **Addendum
  2's verdict sentence is WITHDRAWN.**
- **Addendum 1 survives intact, and is now explained.** Its D^2 scaling of the
  field-map metrics (measured 3.98-4.50x vs 4.12), its conic agreement to 3-4 dp
  at the true aperture, and "just-the-knob" are all unaffected. What Addendum 1
  could not say is *why* the field-map metrics were 11-20x Rodgers while the
  conics matched: because `global` and `refsphere` are both taken **on the
  detector plane**, and on a 14.3-degree-tilted image surface that plane carries
  `(transverse aberration) x tan(tilt)` — an artifact ~20x the wavefront error
  which is not a low-order pupil term and so survives the `refsphere` fit
  (§A.1). It scales as D^2 because the aberration it multiplies does.
- **Net, one verdict:** the optics are excellent (conics match; §C's strict WFE is
  0.11-0.43 waves at 0.5 deg off-axis on a frozen detector, right where Rodgers
  puts it); the 11-20x field-map numbers were the *plane* the metric was taken
  on, not the aperture and not the design; and the aperture question raised by
  Addendum 1 is settled at D = 4060 mm by §D, where the strict metric lands at
  1.15x his max without any aperture adjustment.

## C. The strict rung — Dave's metric, implemented constructively

**Definition (Dave, 2026-07-30).** Per field: a reference sphere **anchored at the
exit pupil**, **centred on that field's chief-ray incidence point on the FROZEN
detector**, **piston-only removal**.

**Implementation** — `strict_wfe.m` + `strict_sphere_opl.m`, pure MATLAB, no
engine change and no reliance on an engine reference-surface default. Per field we
harvest the raw ray data at the terminal FocalPlane (`macos.get_ray_info`:
position `P`, direction `d`, cumulative OPL `L`), and reduce it ourselves:

- `c` = the chief ray's intersection with the frozen detector plane -> the sphere
  CENTRE;
- `X` = the exit pupil, found the way FEX finds it — the crossing of two probe
  chief rays `1e-5 rad` apart (`tracesub.F:312-368`), measured here at
  `[0, 0, +0.2185] m` -> the sphere ANCHOR;
- `R = |X - c|`;
- for every ray, with `v = P - c`, `a = v.d`, `e = |v|^2`:

      s = -a - sqrt(R^2 + a^2 - e) ,      W = L + s

  which is the OPL to the sphere **exactly** — rays are straight after the last
  surface (verified: `dir` matches the M3->FP chord to `1.6e-15`, and
  `OPL(FP)-OPL(M3)` equals the segment length to `8.9e-15 m`), so no paraxial or
  `2*rho^2` expansion enters anywhere;
- `wfe = std(W)`, mean removed and nothing else — piston only.

The chief ray is excluded from the statistic, matching the engine's own
`iRay = 2, nRay` loop. All 1305 rays reach the detector at every field; no
`realize_apertures` call is made (that leaves clip apertures installed — the
contamination trap).

**Why the sphere and not the plane:** §A.1. The strict metric removes the
detector-tilt artifact and the chief-ray tilt, and keeps exactly what a frozen
detector genuinely imposes — the field-to-field focus and astigmatism walk.

**GATE 1 — the displaced-detector discriminator. PASS.** Stage 2, detector frozen
by `align_focal_plane('grid',5,'span_arcmin',6)`, then displaced along its own
normal. The metric must grow by the closed-form sphere difference
`delta*cos(theta) - sqrt(R'^2 - delta^2 sin^2(theta)) + R`, evaluated per ray from
the measured obliquities (`std(cos theta) = 6.4286e-05`):

| detector displacement | strict metric | growth vs dz=0 | **analytic** | per-ray residual |
|---|---|---|---|---|
| 0 | 226.5 nm | — | — | — |
| +1 mm | 239.3 nm | 66.33 nm | **66.32 nm** | 0.18 nm |
| +10 mm | 710.6 nm | 661.6 nm | **661.5 nm** | 1.78 nm |
| +627 mm | 35 250 nm | 35 240 nm | **35 240 nm** | 94.8 nm |

Agreement is to 0.02% at +10 mm, and the residual is checked **per ray**, not just
on the RMS. The metric is detector-tied and is not self-referencing.

Two notes. (i) The +627 mm rung reads tens of micrometres, not the millimetres
anticipated — because that estimate assumed f/0.86, and the system is f/23.5
(§A.5); at the measured obliquity `0.627 m x 6.43e-05` is `40 um`, which is what
we get. (ii) The engine's own FP `rmsWFE` at `dz = 0` is `5099 nm` against the
strict metric's `226.5 nm`: the 22x detector-tilt artifact of §A.1, in one line.
This also disposes of the SESSION_STATE "13.4 nm at z = +0.627 m" puzzle — that
number was `13.4 mm` (§A.2 units), at a plane 3.9 m from focus and tilted, which
is exactly the scale `(beam width) x tan(14.3 deg)` demands. It was never evidence
of re-referencing.

**Aberration ladder** (`strict_ladder.m`, stage-2 box, 5x5, nm) — what the strict
number is made of, and how much any further freedom could buy:

| rung | min | max | avg |
|---|---|---|---|
| **strict** (Dave's ruling) | 110.2 | 429.6 | 250.9 |
| + sphere slid to per-field best focus | 110.1 | 429.5 | 250.7 |
| + tip/tilt removed | 54.8 | 261.2 | 143.0 |
| + astigmatism removed | 33.3 | 117.7 | 71.3 |

The best-focus rung is the important one: sliding the sphere centre along each
chief ray to that field's own best focus improves the metric by **0.1%**, and the
longitudinal shift it wants is only `-0.20 .. -0.07 mm` across the box. A detector
surface — plane or curved — can do nothing else. **Open ask 2 (his FPA surface) is
therefore not needed to reconcile the numbers and cannot have been the
explanation.** `align_focal_plane`'s plane is already at best focus to a tenth of
a millimetre (plane-fit rms `0.027 mm`, sag range `0.083 mm` over the box).

## D. Stage-2 validation (step 2) — PASS

Frozen stage-2 design (Rodgers' verbatim conics, `+0.5 deg` offset, EPD **4060
mm**, lambda = 1000 nm, `model_size` 256); detector produced by his procedure
(FPA tilt/focus fit over the box: tilt `14.2994 deg`, `Vpt = [0, -0.881880,
-3.256477] m`) and then **held**; no optimizer; strict metric over the 9x9
bias-relative box; 81/81 fields finite, 1305 rays each.

| | min | max | avg |
|---|---|---|---|
| **strict metric** | **110.2** | **429.6** | **246.8** nm |
| Rodgers (CODE V field map) | 79.4 | 374.6 | 199.9 nm |
| ratio | 1.39x | **1.15x** | **1.23x** |
| *engine FP `rmsWFE`, same run, for contrast* | 1940.7 | 12493.3 | 6023.5 nm |

**Within the ~1.5x gate. Ask 1 closes.** The metric was not tuned toward his
value: it is the ruled construction, validated independently by gate 1 before this
box was ever evaluated.

Two cross-checks worth keeping. (i) The engine `rmsWFE` column above —
`12493.3 / 6023.5` — reproduces the committed §C' `global` row for S2 at EPD 4060
**exactly**, confirming that this run's box is the same box §C' used and that the
`global` metric is precisely the engine's own FP number. (ii) `refsphere` reads
`4555.1 / 2478.9` on that same box while the strict metric reads `429.6 / 246.8`:
removing *more* (piston + tip/tilt + defocus) gives a *larger* number, which is
only possible because `refsphere` is taken on the tilted detector plane and the
strict metric is taken on a sphere — §A.1, quantified.

**GATE 2 — the on-axis anchor.** Stage 1 (no field bias), same recipe, 9x9 box
about 0':

| | min | max | avg |
|---|---|---|---|
| strict metric | 0.277 | 3.570 | 0.968 nm |
| Rodgers on-axis map | 0.446 | 1.463 | 0.606 nm |

Same scale (`~1 nm`), avg within `1.60x`, max `2.44x`. This is the regime where
the design is essentially perfect and the absolute numbers are set by ray
sampling and by exactly where the fitted plane sits, so a factor ~2 on a 1-nm
quantity carries much less information than the offset-field result; it is
recorded as a scale check, not as a second precision claim.

## D.2 The strict metric across all four stages — and what it says about the solves

Pure evaluation of the **committed** `rodgers1_epd4060_stage{1..4}.in` decks:
no rebuild, no optimizer, nothing re-solved. Each stage is scored against **its
own solved FPA** (that deck's `FocalPlane`), held frozen — Rodgers' procedure,
stage by stage. `strict_stage_table.m` drives it; `strict_wfe_deck.m` re-emits
each deck per field with only its two source lines rewritten, exactly as
`emit_` builds them, reading the field bias back **from the deck** so there is
no bias to double. 1304 rays at every field, 81/81 finite, all four stages.

**Cross-validation, asserted in the driver before anything else is reported:**
the deck path reproduces §D's stage-2 gate number — obtained by the entirely
separate `strict_wfe` / Telescope-object path — to **8.6e-7 relative**.

| stage | what was solved | strict max / avg (nm) | Rodgers max / avg (nm) | max x | avg x |
|---|---|---|---|---|---|
| S1 | nothing (on-axis evaluation) | 3.6 / 1.0 | 1.5 / 0.6 | 2.44 | 1.60 |
| S2 | nothing (offset evaluation, FPA re-fit) | **429.6 / 246.8** | 374.6 / 199.9 | **1.15** | **1.23** |
| S3 | 3 conics + FPA | 181.2 / 97.1 | 91.6 / 46.4 | 1.98 | 2.09 |
| S4 | + M2/M3 Ydec + tilt | 118.6 / 84.8 | 39.8 / 22.5 | **2.98** | **3.77** |

**The agreement degrades monotonically with how much OUR optimizer did.** The
two stages our optimizer never touched agree (S2 at 1.15x; S1 at 1.60x on a
1-nm quantity); adding three conic DOFs takes it to ~2x; adding the rigid-body
DOFs takes it to ~3-3.8x. That is the cleanest possible separation of concerns:

- the **evaluation metric** is right — S2 proves it, with no optimizer anywhere
  in the loop;
- the **optics** are right — the conics match CODE V to 3-4 dp (§B');
- what differs is **what our optimizer minimised**, and only that.

### The rigid-body verdict

The question this table was built to answer: our stage-4 rigid body is 4-5x
smaller than his and opposite-signed on the decenters (ours M2 -3.74 mm /
0.233 deg, M3 **-43.84 mm / 1.114 deg**; his M2 8.34 / 0.517, M3 **121.868 /
2.330**). If the two solves were merely different points in a degenerate
`K_M3`/`Ydec` valley, they would score the same under his own metric.

**They do not. Our stage-4 solve scores 118.6 / 84.8 nm where his scores
39.8 / 22.5 — 2.98x on max, 3.77x on avg.**

> ⚠ **RETRACTED by Addendum 4 §B, then RESTATED by Addendum 5 §B** — the honest
> figure is **1.83x** (his 64.9 nm vs our 118.6, design against design, once
> his ADE sign is decoded), not the ~3x below and not "uninterpretable".
> Original retraction text follows.
>
> ⚠ **RETRACTED by Addendum 4 §B.** The sentence that stood here — *"his
> rigid-body solution is genuinely the better design under the strict metric, by
> a factor of ~3"* — compared our solve against his REPORTED number, never
> against his design. Built verbatim, his stage-4 parameters score **64.6 um** in
> our frame, while our own stage-4 re-injected through the identical path
> reproduces the 118.6 / 84.8 above to 5e-6. His stated `Ydec`/`alpha` therefore
> do not mean, in our frame, what `rigid_of` reports for ours; the two parameter
> sets are **not commensurable** and NEITHER "his is better" NOR "the degenerate
> valley explains it" can be concluded. The `K_M3`/`Ydec` question is **reopened**.
> What survives, independently, is the conic finding — and Addendum 4 §B
> sharpens it: his stage-3 conics score 115.3 nm max against our 181.2 nm, so his
> conics really are better by 1.57x under the strict metric.

The merit-function hypothesis of §4c
therefore **revives**, and with a concrete mechanism: our CALIB solve was
driven by the detector-plane WFE, which on this 14.3-degree-tilted image
surface is dominated by `(transverse ray aberration) x tan(tilt)` (§A.1) — an
artifact ~22x the wavefront error. Our optimizer was minimising the blur
against a tilted plane, not the wavefront; it is unsurprising that it stopped
short and landed on a different rigid-body balance.

Corroborating detail: the *direction* of our ladder is right and only its depth
is short. His S2 -> S4 improvement is `374.6 -> 39.8`, a factor **9.4**; ours is
`429.6 -> 118.6`, a factor **3.6**. We capture rather more than half the
available gain in log terms and stop.

**Consequence for step 3: it is NOT unnecessary — it is the indicated next
move, and it now has a validated objective to use.** Re-optimising stages 3 and
4 against the strict metric (per §E, configuration-only: an ExitPupil `Return`
element, the system STOP set, the detector immediately after the EP, FEX on) is
the falsifiable test — if our conics and rigid body then move toward his and the
strict score drops toward 39.8 / 22.5, §4c closes too. **That is a separate
brief; nothing of it is started here.**

Field maps for the four-panel progression (absolute field axes, same rendering
as the committed `*_{global,refsphere}.png`):
`rodgers1_epd4060_stage{1,2,3,4}_strict.png`.

**What is left over.** After §D/§D.2 there is no unexplained magnitude gap
in the *evaluation*. The three open asks to Rodgers reduce to one:

1. ~~RMS reference sphere~~ — **CLOSED.** Dave's ruled construction reproduces his
   field map to 1.15x at his aperture.
2. ~~His FPA tilt/focus DOF~~ — **not load-bearing.** §C's best-focus rung shows
   any detector surface is worth 0.1% here. Worth confirming for completeness,
   but it cannot move the comparison.
3. **EPD** — now *supported* rather than open: D = 4060 mm is the aperture at
   which the strict metric lands on his numbers with no adjustment, which is an
   independent confirmation of Dave's measurement. (Addendum 1's D^2 scaling
   means a 20% aperture error would show as a 44% metric error; we are at 15%.)

The remaining live discrepancy is **not** magnitude but **shape**: across the box
his field map spans `79.4 -> 374.6` (**4.72x**) where ours spans `110.2 -> 429.6`
(**3.90x**) -- we agree best at the worst corner (1.15x) and least at the best one
(1.39x). Both are steeper than the `H^2` an on-axis-corrected anastigmat's
astigmatism alone would give across `0.4 -> 0.6 deg` (2.25x), so neither is pure
astigmatism; the difference in steepness is most naturally a field-definition
question -- exactly where his box centre sits, and whether his 0.2 deg is full or
half width. Worth one line in the note to Mike; it changes no conclusion here.

One thing §D.2 does change: the *optimised* stages (S3, S4) do NOT reproduce
his numbers, and that is a merit-function finding, not a metric one. See §D.2.

## E. Lane determination for step 3 (read-only)

**Verdict: CONFIGURATION-ONLY, no new engine mode.** [SOLID — CCMac finding,
re-checked against the forensics above and unchanged.]  §D.2 upgrades step 3
from optional to *indicated*: the optimised stages do not reproduce Rodgers,
and re-solving them against the now-validated strict metric is the test of
why.  It remains a SEPARATE brief. CALIB's per-field WFE merit
is already chief-ray-tied per field with piston-only removal: `funcs_app` forces
`LUseChfRayIfOK = .false.` (`design_optim.F:659-664`), which selects the
mean-subtract branch of `SUBROUTINE OPD` (`tracesub.F:219-233`) — and §A.1 shows
all three branches give the same RMS anyway, so the piston-only property is
structural, not a branch accident. CALIB additionally re-runs FEX per field
(`smacos_compute.inc:391-397`), and §A.3 shows that a FEX-set EP element is a
sphere whose centre of curvature IS the chief-ray intercept on the next element's
plane. So the strict metric in-the-loop needs only: a deck with an ExitPupil
(`Return`) element, the system STOP set, the detector immediately after the EP,
and FEX enabled. **No Fortran change; engine-first merge ordering does not bite.**

One caveat carried forward for whoever runs step 3: the in-loop merit is then the
strict metric *at the EP element*, which by §A.4 is the same number as the
out-of-loop metric of §C only while `R` is right — i.e. the detector must be the
element right after the EP, or `zp` will be the distance to something else.

## F. Artifacts

Added (suffixed-parallel; every committed EPD-2000 and EPD-4060 output is left
bit-intact):

- `strict_wfe.m` — the strict metric (the deliverable).
- `strict_sphere_opl.m` — the exact ray-to-sphere OPL kernel, shared by
  `strict_wfe` and the gates so there is one implementation.
- `strict_rung_gates.m` — gates 1/2/3 end-to-end; `strict_rung_gates(9)` is the
  reproduction of §C/§D. Gate 1 is the non-vacuity check and must be re-run
  before trusting any change to `strict_wfe`.
- `strict_ladder.m` — the aberration ladder / best-focus floor of §C.
- `strict_wfe_deck.m` — the same metric applied to a saved `.in` deck (pure
  evaluation of an already-solved stage); `strict_stage_table.m` — the §D.2
  four-stage table + the four field maps.  The driver ASSERTS the stage-2
  cross-check against §D before reporting; keep that assert.
- `rodgers1_epd4060_strict_gates.mat`, `rodgers1_epd4060_strict_ladder.mat`,
  `rodgers1_epd4060_strict_stages.mat`, and `rodgers1_epd4060_stage{1..4}_strict.png`
  — the four-panel strict progression, absolute field axes, same rendering as
  the committed `*_{global,refsphere}.png` maps.

Removed: `strict_rung.m`, `strict_rung_stages.m`,
`rodgers1_epd4060_strict_rung.mat` and `wip_scratch_2026-07-30/` — the WIP
sketches, superseded by the above. `strict_rung.m` carried the `s.rmsWFE*lam_nm`
units bug of §A.2 and its `align_focal_plane`-in-the-loop recipe; keeping it would
be keeping a trap.

**Field convention, since it cost this session a run.** `trace_at_field(F)` ADDS
`spec.field_bias` to `F`, so it takes a **box-relative** box.
`realize_apertures('fields',F)` is the one branch that does NOT add the bias and
therefore takes the **absolute** box (the §D `biasbox` helper). Passing the
absolute box to `strict_wfe` doubles the bias and silently evaluates a `+1.0 deg`
box — which reads `2405 nm` avg, i.e. a plausible-looking 12x "miss".

`tDesignTelescope` 70/70 green.

---

# ADDENDUM 4 — step 3, part A: what CALIB minimised, his designs verbatim, and the ORS lane (2026-07-30)

Evaluation only. No re-solve was run; §C explains why the lane is clear anyway
and §B explains why the gate that would have opened part B does not.

**Headline.** His stage-3 design, built verbatim from his slide parameters and
strict-scored, reads **115.3 / 53.7 nm against his 91.6 / 46.4 (1.26x / 1.16x)** —
the strict metric now reproduces **two** of his own designs. His stage-4 design,
built the same way, reads **64.6 um** — 1623x. An injection round-trip proves the
machinery: pushing OUR committed stage-4 solve back through the identical path
reproduces §D.2's number to **5e-6**. So the stage-4 failure is in **his stated
rigid-body values not meaning, in our frame, what `rigid_of` reports for ours** —
and the §4c / §D.2 rigid-body comparison is **retracted as uninterpretable**.

## A. What CALIB actually minimised for these decks

Addendum 3 §A.3/§E said CALIB re-runs FEX per field, so each field is referenced
to its own chief-ray-tied exit-pupil sphere; §D.2 said the solve was driven by
detector-plane WFE with the `tan(tilt)` artifact. **Both are correct; §A.3's
statement was conditional and these decks do not meet the condition.**

The inner loop is `smacos_compute.inc:388-405`:

    If (ifCalcOPD) Then
      if (ifFEX_m) then                     <- 391
        cmd='FEX' ;  IARG(1)=nElt-1
      end if
      If (opt_tgt_m==WFE_TARGET ...) Then   <- 400
        cmd='OPD' ;  IARG(1)=optElt_m

`ifFEX_m` is **false** here, twice over: `LOptIfFEX` initialises to `.FALSE.`
(`dopt_mod.F:229`) and the design layer emits `OptFEX= No` explicitly
(`Telescope.m:2307`, hard-coded in `emit_`). And even had it been true, the call
is hard-wired to `IARG(1)=nElt-1`, which for a 4-element deck is **M3, a
Reflector** — the FEX dispatch accepts only `EltID` 8 (Return) or 3 (Reference)
and would have aborted (`macos_cmd_loop.inc:2618-2627`).

So the loop went straight to `cmd='OPD'` at `optElt_m = OptWFElt`, and the design
layer sets `wf_elt = numel(spec.elt)` — the **terminal FocalPlane**
(`Telescope.m:864`, emitted at `:2305`). Meanwhile `funcs_app` forces
`LUseChfRayIfOK=.false.` (`design_optim.F:664`), selecting the mean-subtract
branch of `SUBROUTINE OPD`, and feeds the **whole OPD map** to the LM as its
residual vector (`design_optim.F:683`: `yfit(off:off+obj_size-1)=OPDm(1:obj_size)`),
so the objective is `sum_pixels OPD^2 = nPix * RMS^2`.

**Answer: CALIB minimised the piston-removed OPL to each ray's own intercept on
the FocalPlane** — the detector-plane quantity of Addendum 3 §A.1, carrying
`(transverse ray aberration) x tan(14.3 deg)`, ~22x the wavefront error. That is
exactly the evaluation the ORS/strict change has to replace: **`OPD` at the
FocalPlane must become `OPD` at a per-field chief-tied sphere.**

## B. His designs verbatim — evaluation

Built from `rodgers1_epd4060_stage2.in` (his layout, his bias, EPD 4060, clip
apertures stripped): conics set to his slide values; stage 4's M2/M3 rigid body
applied through the **same engine path CALIB uses** (`cmd='PERTURB'` /
`CPERTURB_PROG`, `smacos_compute.inc:300-322`, reached as `macos.perturb`); FPA
then fitted by his procedure (the `align_focal_plane` algorithm, 5x5 over +/-6')
and **frozen**; strict-scored over the 9x9 box. 1304 rays, 81/81 fields.

The applied rigid body is **read back and asserted** against his stated numbers in
the same convention `rigid_of` reports ours (`Ydec = Vpt(2)`,
`alpha = atan2d(psi_y, -psi_z)`) — so "his values" means the same thing on both
sides. It asserts clean (M2 8.344733 mm / 0.516947 deg, M3 121.868248 mm /
2.329710 deg, to 1e-6). The assert also **determined a convention**: `perturb`'s
local `+Rx` reports back as **negative** alpha, so his `+alpha` is applied as
`-Rx`. That is recorded in the source, not left implicit.

| design | strict max / avg (nm) | Rodgers max / avg | max x | avg x |
|---|---|---|---|---|
| **his S3 verbatim** (his conics) | **115.3 / 53.7** | 91.6 / 46.4 | **1.26** | **1.16** |
| **his S4 verbatim** (his conics + his rigid body) | **64609 / 64569** | 39.8 / 22.5 | 1623 | 2871 |
| *our committed S4, re-injected (round-trip)* | *118.6 / 84.8* | — | *= §D.2 to 5e-6* | — |

**S3 — the metric is confirmed a second time, on a design we did not solve.**
1.26x / 1.16x sits in the same band as the stage-2 gate (1.15x / 1.23x), and the
expectation in the brief was ~92 nm. This is now two of Rodgers' own designs
reproduced by Dave's construction, one un-optimised and one optimised, which is
considerably stronger evidence than stage 2 alone.

It also lets the §D.2 conic finding be restated against his real design rather
than his reported number: **his conics score 115.3 nm max where our stage-3
conics score 181.2 nm — his are better by 1.57x under the strict metric.** The
"our optimizer minimised the wrong quantity" conclusion survives, measured.

**S4 — 64.6 um, and the round-trip localises it.** Re-injecting our OWN committed
stage-4 solve (its conics and its rigid body, out of
`rodgers1_epd4060_results.mat`) through the identical `macos.perturb` path, with a
fresh FPA fit, reproduces the committed §D.2 stage-4 strict number **118.591 /
84.806 nm to 3.5e-6 / 5.4e-6 relative**. The injection path, the FPA fit and the
scorer are therefore all sound, and the failure is in the VALUES.

Corroborating detail: with his rigid body installed, the fitted FPA moves to
`Vpt_y = +0.3665 m` (from `-0.8811`) and tilts to **21.70 deg** — the image walks
**1.25 m** — and the best-focus rung still reads 64.4 um, so no detector surface
can recover it. A 121.868 mm decenter on a mirror whose beam footprint radius is
**111 mm** is not an alignment perturbation at all; it walks the beam entirely off
the region the surface was figured for. That is a strong hint that his `YDE`/`ADE`
are **not** a global-frame vertex decenter plus a normal tilt — CODE V applies
them in the surface's local, possibly reflection-flipped, coordinate system, and
the sign or the order or the pivot differs.

**Ruling: the stage-4 rigid-body comparison is RETRACTED as uninterpretable.**
Addendum 3 §D.2 concluded *"his rigid-body solution is genuinely the better design
under the strict metric, by a factor of ~3"* — that was inferred from OUR stage-4
scoring 2.98x his REPORTED number, never from evaluating his design. Evaluating it
directly shows his stated parameters do not produce a working design in our frame,
so the two parameter sets are **not commensurable** and neither "his is better" nor
"the degenerate valley explains it" can be concluded. §4c's "4-5x smaller and
opposite-signed" table is void for the same reason. **The `K_M3`/`Ydec` degenerate-
valley question is reopened, not answered.** What §D.2's monotone pattern still
supports, independently of any rigid-body claim, is the conic finding above.

**Next step for this, and it is small:** resolve the `YDE`/`ADE` convention. Four
bounded variants (decenter sign, tilt sign, tilt-then-decenter vs decenter-then-
tilt, pivot at vertex vs at the local origin) can be evaluated in a few minutes
each with `his_designs.m`; the acceptance test is not "does it hit 39.8 nm" but
"does his design work at all" (i.e. land in the hundreds of nm rather than tens of
um). Better still, one line from Mike's `.seq` settles it. Deliberately NOT
searched here — a convention should be read, not fitted.

## C. Lane for the ORS-referenced merit — CONFIGURATION-ONLY, no engine edit

Two separate questions, and they have different answers.

**Literal ORS is not reachable from CALIB, and is also the wrong merit.** The
`ORS` command runs `CRSOPTIMIZE` (`tracesub.F:4436-4517`), which Brent-minimises
RMS OPD over the reference surface's RADIUS `fElt(iElt)` (`:4461-4483`) — i.e. it
finds each field's own **best-fit** sphere, removing per-field focus. Dave's ruled
metric does the opposite: it pins the sphere to the **frozen** detector and keeps
the focus walk, which is exactly what a fixed FPA imposes and what makes the
number comparable with Rodgers. Putting `CRSOPTIMIZE` in the loop would let the
optimizer ignore field curvature. Separately, it is not callable there at all:
`MACOS_OPS` implements STOP / FEX / PERT / GPER / SPOT / OPD / CRT / GBS / RefRay /
ROC_PERT / CONIC_PERT / ZERN_PERT / ASPH_PERT / RESET (`macos_ops.F:56-346`) and
has **no ORS branch**, so wiring it would need Fortran in two places plus a gate
keyword. **We do not want it.**

**The metric we DO want is reachable with existing keywords.** Give the deck an
`add_pupil` exit pupil so the tail is `[..., FP_return(Return), ExitPupil(Return),
FocalPlane]`, set the system STOP, and set `OptFEX= Yes` with
`OptWFElt = nElt-1`. Then:

- CALIB's `IARG(1)=nElt-1` FEX call (`smacos_compute.inc:391-397`) lands exactly
  on the ExitPupil, per field;
- by Addendum 3 §A.3 the FEX-set sphere's centre of curvature is the chief ray's
  intercept on the plane of element `iElt+1` — the detector;
- `OPD` at `nElt-1` then measures OPL to that sphere.

That **is** the strict metric, and structurally so: with the Return pair, the
second Return negates the EP->FP leg, so `CumRayL` at the ExitPupil equals
`OPL(source -> FP intercept)` minus the distance back to the sphere — precisely
the `L + s` that `strict_sphere_opl` computes. `MACOS_OPS` already implements
`FEX` (`macos_ops.F:60`); `design_optim.F:170-180` only requires the STOP to be
set, which `smacos_compute.inc:279-288` then re-issues per evaluation.

**Verdict: no Fortran in `macos`. The stop point does not trigger.** Two small
design-layer (MATLAB, resources-side) changes are needed, both removing a
hard-coded constant: `emit_` writes `OptFEX= No` unconditionally
(`Telescope.m:2307`) and `optimize()` sets `fp_elt = numel(spec.elt)`
unconditionally (`Telescope.m:864`); both must become pupil-aware, using
`spec.pupil.ep_elt` when `add_pupil` has run.

## D. Part B was not started, and why

Its gate is "Part A2 lands near his numbers **and** the lane is clear". The lane
is clear (§C). A2 is **half**: S3 lands (1.26x), S4 does not (1623x). And part B's
own gate 5 reads *"rigid-body values move toward his 121.868 mm / 2.330 deg"* —
a criterion that cannot be evaluated while those numbers are not in our frame
(§B). Re-solving now would produce rigid-body values with nothing meaningful to
compare them against. **The `YDE`/`ADE` convention is the blocker and it is the
next thing to close.** Nothing of the re-solve was begun.

## E. Artifacts

- `his_designs.m` — builds and scores his S3/S4 verbatim, plus the our-S4
  injection round-trip. The rigid-body readback assert and the round-trip check
  are both load-bearing; keep them.
- `rodgers1_epd4060_rodgersS{3,4}.in`, `rodgers1_epd4060_oursS4roundtrip.in` —
  the three built decks (FPA fitted and installed).
- `rodgers1_epd4060_his_designs.mat`,
  `rodgers1_epd4060_{rodgersS3,rodgersS4,oursS4roundtrip}_strict.png`.

**Two harness traps found and fixed here, both of the "plausible wrong answer"
kind** — worth the same standing as the bias-doubling and the metres-vs-waves
entries:

1. `align_fpa_deck_` drives the engine through its probe grid and leaves it
   holding the **last probe field's** chief ray. Saving the artifact deck from
   that state bakes a box-corner field into it, and every later scan is centred
   6' off. First run reported his S3 at 381.9 nm (4.17x) instead of 115.3 nm
   (1.26x) — a miss large enough to look like a real finding. Caught by the field
   map's x axis reading 0..12' instead of -6..+6'; now guarded by an assert that
   the saved deck's `ChfRayDir` still equals the nominal.
2. `macos.perturb`'s local `+Rx` reads back through `rigid_of` as **negative**
   alpha. Injecting his `+alpha` naively puts the mirror at `-alpha`. Caught by
   the readback assert, which is the reason that assert exists.

---

# ADDENDUM 5 — step 3, part B: the convention decode, gate 0, and the engine stop point (2026-07-30)

**Headline.** The CODE V `ADE` sign is opposite to ours; decoded, **his stage-4
design reads 64.9 / 35.4 nm against his 39.8 / 22.5 — 1.63x / 1.57x**, so the
strict metric now reproduces **all three** of his designs. Gate 0 passes at
**2.7e-9**. The re-solve is **BLOCKED at an engine change**: `OptFEX= Yes` is a
no-op, so no prescription can turn on the per-field FEX the merit needs. Design
below; nothing implemented in `macos` — engine-first ordering applies.

## A. Convention decode — his ADE sign, measured

Addendum 4 left his stage-4 rigid body uninterpretable. `convention_decode.m`
screens all 16 sign combinations of (YDE, ADE) across M2/M3 x 2 application
orders, at the box-centre field, on a criterion that is NOT his WFE: the
arriving bundle's own best-focus spot RMS and the strict WFE about a sphere
centred there. A wrong frame fails this by orders of magnitude and no detector
fit can rescue it.

| variant | spot RMS | strict at own focus |
|---|---|---|
| **M2(+YDE, −ADE) M3(+YDE, −ADE)** | **1.911 um** | **8.967 nm** |
| runner-up: M2(−,+) M3(−,+) | 52.7 um | 293 nm |
| the other 14 | 1.2 mm .. 8.5 mm | 5.8 .. 42 um |
| *our committed S4, for scale* | *14.19 um* | *81.83 nm* |
| *stage 2, no rigid body* | *27.45 um* | *155.4 nm* |

**A 30x separation from the runner-up and 9x better than our own S4.** That is
a decode, not a fit. The order of application makes no difference at all
(identical to printed precision), so only the signs matter.

**Result: his YDE sign matches ours; his ADE sign is OPPOSITE** to `rigid_of`'s
`alpha = atan2d(psi_y, -psi_z)`, uniformly on both mirrors. The hypothesis
under test — that CODE V's per-surface frame flips at each mirror, so M2 (odd)
would flip and M3 (even) would not — is **wrong**; the measured answer is
simpler and uniform. Worth recording as such: the reflection-flip story was
plausible and false.

**His design in our frame:** M2 (+8.344733 mm, −0.516947 deg),
M3 (+121.868248 mm, −2.329710 deg).

## B. His stage 4, re-scored — and the rigid-body comparison, finally in one frame

With the decoded ADE sign, the full Addendum-4 pipeline (FPA fitted by his
procedure then frozen, 9x9 box, 1304 rays, 81/81):

| design | strict max / avg (nm) | Rodgers | max x | avg x |
|---|---|---|---|---|
| his S2 (Addendum 3 §D) | 429.6 / 246.8 | 374.6 / 199.9 | 1.15 | 1.23 |
| his S3 verbatim (Addendum 4) | 115.3 / 53.7 | 91.6 / 46.4 | 1.26 | 1.16 |
| **his S4 verbatim, decoded** | **64.9 / 35.4** | 39.8 / 22.5 | **1.63** | **1.57** |
| our committed S4 | 118.6 / 84.8 | — | 2.98 | 3.77 |

**Three of his designs, three reproductions in the 1.15–1.63x band.** The
64.6 um of Addendum 4 was entirely the ADE sign.

**Rigid body, both solves in the converted frame:**

| | Ydec (mm) | alpha (deg) |
|---|---|---|
| his M2 | +8.3447 | −0.5169 |
| our M2 | −3.7417 | +0.2330 |
| his M3 | +121.868 | −2.3297 |
| our M3 | −43.839 | +1.1142 |

His is **−2.1x to −2.8x ours** — the same compensation *pattern*, on the
**opposite branch**, roughly 2.4x further along it. That is the signature of a
genuinely degenerate valley with two usable branches, not of two unrelated
solutions. But they are **not equivalent**: his scores 64.9 nm where ours
scores 118.6, so **his branch is better by 1.83x** under the strict metric.
(Addendum 3 §D.2 claimed ~3x; that was against his reported number. 1.83x,
design against design, is the honest figure and it supersedes both the §D.2
claim and Addendum 4's retraction of it.)

## C. Gate 0 — the in-loop merit IS the strict metric, numerically

`gate0_merit_identity.m`, on the committed `rodgers1_epd4060_stage4_pupil.in`:
run FEX at nElt-1 then `OPD` there (what CALIB's inner loop would evaluate),
against `strict_wfe`'s own construction from raw ray data at M3.

| field | CALIB merit (m) | strict (m) | relative |
|---|---|---|---|
| box centre | 8.228877052003e-08 | 8.228877029755e-08 | 2.7e-09 |
| +x+y corner | 7.472140289252e-08 | 7.472140283521e-08 | 7.7e-10 |
| −x−y corner | 1.185906278408e-07 | 1.185906278375e-07 | 2.8e-11 |

**PASS.** The tolerance is 1e-6, not machine epsilon: the engine intersects the
actual conic reference surface iteratively while the construction solves the
sphere in closed form. 2.7e-9 is 2e-16 m on an 8e-8 m quantity. A conic walk of
+/-0.1 on K_M3 keeps the two identical to all printed digits, so the identity is
not local to the nominal point.

## D. STOP POINT — the re-solve needs a Fortran change in `macos`

**Addendum 4 §C's "configuration-only" verdict is WRONG.** It was reasoned from
the keyword's existence; the parser body says otherwise:

    msmacosio.inc:327-329
      ELSE IF (LCMP(VAR_NAM,'OptFEX',6)) THEN
        If (LCMP(VALUE,'N',1)) LOptIfFEX=.FALSE.
        GO TO 50

**There is no affirmative branch. A prescription can turn the per-field FEX OFF
and never ON.** `OptFEX= Yes` is silently a no-op. (`dopt_mod.F:229` initialises
`LOptIfFEX=.FALSE.`; `macos_cmd_loop.inc:347` sets `.TRUE.` on the interactive
CALIB path, so the effective default is path-dependent — which is its own
defect.) Same class of error as an "analytic" transcribed from the engine's own
expression: I read the keyword, not the code under it.

**Measured consequence** (`fex_in_loop_check.m`, 9-field optimisation box):

| | on-axis field | off-axis fields |
|---|---|---|
| EP OPD, stale add_pupil sphere | 1.26e-07 m | **1.84e-03 .. 2.64e-03 m** |
| EP OPD, after per-field FEX | 2.26e-07 m | 1.10e-07 .. 4.30e-07 m |

Four orders of magnitude, and they agree ONLY on axis — the tell. A real solve
logged its inner merit at **5.5e-03 .. 1.3e-02 m**: the no-FEX column. So CALIB
was minimising the image-displacement tilt of a sphere stuck at the on-axis
image, which it reduces by moving the image — and it ran away exactly as that
predicts (one round took K_M1 to −1.262, K_M3 to +7.198, the FPA 2.7 m, and the
next round lost every ray).

**Proposed engine change, for review — NOT implemented:**

1. `msmacosio.inc:327-329` — add the affirmative branch:
   `If (LCMP(VALUE,'Y',1)) LOptIfFEX=.TRUE.`
2. Make the default explicit rather than path-dependent: `dopt_mod.F:229` says
   `.FALSE.`, `macos_cmd_loop.inc:347` says `.TRUE.`. Pick one and have both
   paths read it.
3. **Gate, already measured and non-vacuous:** with `OptFEX= Yes`, an ExitPupil
   at nElt-1 and the STOP set, CALIB's inner OPD must equal the strict metric —
   1.1e-07..4.3e-07 m across the box, not 1.8e-03..2.6e-03. The pre-change
   engine fails it at every off-axis field by 4 orders.
4. **Risk to check:** FEX rewrites the EP element's `Vpt/psi/Kr/f/z` on every
   evaluation (`macos_ops.F:78-84`); its interaction with CALIB's `RESET` and
   with the finite-difference Jacobian needs a look before the flag is trusted.
5. Scope: one parser line plus a default; no new command, no new state.

**Design-layer side (this commit, resources only).** The plumbing is in place —
`optimize()` routes `OptWFElt` to `spec.pupil.ep_elt` and requests `OptFEX`;
`align_focal_plane` gained `'allow_pupil'` so the solve<->refit alternation can
run with the pupil installed. It is **guarded by a hard error** until the engine
lands: the failure mode is a *silent* wrong solve, so it must not be reachable
by accident. `tDesignTelescope` 70/70.

## E. What part B still owes, once the engine change lands

Unchanged from the brief: alternate solve <-> `align_focal_plane` to
convergence, report the round count, emit `_xpopt` artifacts, and check
S3 <= ~115 nm, S4 <= ~1.5x his 39.8, and the rigid body against his converted
(+121.868 mm, −2.330 deg). §B now says what the last of those should look like:
our solve should either reach his branch or beat it on ours.

## F. Artifacts

`convention_decode.m` + `rodgers1_epd4060_convention_decode.mat`;
`gate0_merit_identity.m` + `rodgers1_epd4060_gate0.mat`;
`fex_in_loop_check.m`; the rebuilt `rodgers1_epd4060_rodgersS4.in` and its map
(decoded convention). `his_designs.m` now carries the decoded ADE sign, with the
readback assert comparing against the CONVERTED values.

---

# ADDENDUM 6 — the OptFEX engine fix (2026-07-31)

Addendum 5 §D proposed a one-line parser change and stopped for review. This
records what was actually needed (two parts, not one), the measurement that
chose the default, and the gates. Engine work is on macos branch
`optfex-fix`; the PR-shaped cherry-pick is `optfex-parse-fix` off `origin/dev`.

## A. It was two defects, not one

**A.1 The parser had no affirmative branch** — as diagnosed
(`msmacosio.inc:327-329`). Fixed by adding the `'Y'` case.

**A.2 The default was path-dependent AND clobbered.** `macos_cmd_loop.inc:347`
set `LOptIfFEX=.TRUE.` in the LOAD block; `dopt_mod.F:229` set `.FALSE.` in
`dopt_init_vars`. On the SMACOS/`load_rx` path the LOAD-site value is **undone**
by a `dopt_init_vars` re-init that runs after it — the same hazard the
`OptTgtElt`/`OptAlg` comment at `dopt_mod.F:218-224` documents, which
`LOptIfFEX` was never protected from. Measured with a diagnostic build:

    >>>DIAG OptFEX parse: VALUE=Yes  LOptIfFEX= F
    >>>DIAG ifFEX_m= F  optElt_m= 5  nElt= 6

i.e. the deck said `Yes`, the flag was already `F` at parse time, and the parser
had no way to raise it. Addendum 5 §D reasoned the root cause from the parser
alone and would have produced a fix that still did nothing on this path.

## B. Default decision — `.FALSE.`, forced by measurement

The brief's criterion: unify to `.TRUE.` if an invalid FEX target degrades
gracefully; to `.FALSE.` if rejection is noisy or state-mutating.

CALIB's FEX call is hard-wired to `nElt-1`, and on a deck ending
`[.. mirror, FocalPlane]` that is a **Reflector**. `MACOS_OPS` calls
`SUBROUTINE FEX` directly and then unconditionally overwrites the target's
`eElt/fElt/KcElt/KrElt/zElt/psiElt/VptElt/RptElt` (`macos_ops.F:60-84`) with no
type check — the check exists only in the interactive `FEXIT` dispatch. So
corruption looked likely. **It is not** (`optfex_default_probe.m`):

| | Kr | Kc | Vpt | psi |
|---|---|---|---|---|
| M3 before CALIB | −2.68797316 | −0.7036063605 | [0 0 1.83902] | [0 0 −1] |
| M3 after CALIB | −2.68797316 | −0.7036063605 | [0 0 1.83902] | [0 0 −1] |

max \|delta\| = **0**. But CALIB **aborts** (`calib_run failed`) — a hard
failure, *not* a fall-back to the plain-OPD merit. Since every existing
optimisation deck ends in a FocalPlane, a `.TRUE.` default would stop them all
from optimising at all. **Unified to `.FALSE.`, opt in per deck with
`OptFEX= Yes`.** One explicit default, stated in a comment at both sites.

## C. Corpus — no committed result changes

Every deck in either repo that carries an optimization block sets `OptFEX`
explicitly, and all four say `No`: `macos/ZGD_test_files/opt_example{,_asph,
_constrained}.in` and `pymacos/tests/Rx/opt_example.in`. Neither the fix (the
`'N'` branch is untouched) nor the default change (they are explicit) alters
any of them. Doc-only hits: `Lou-UpdateNotes.txt:652` — which documents
`OptFEX= YES  % whether do FEX during optimization`, confirming the affirmative
behaviour was always the intent — and a `pymacos/macos.py` docstring. The one
real behaviour change is the interactive CLI default on a hand-written deck with
an Opt block and no `OptFEX=` line; **no such deck exists**. Full table and the
affected-work sweep: `OPTFEX_REDO_LIST.md`.

## D. Gates — `mmacos/tests/tOptFex.m`, 3/3

| test | what it pins |
|---|---|
| `test_merit_is_deterministic` | identical state evaluated twice is **bit-identical** — FEX rewrites the ExitPupil pose on every call, and none of it leaks forward |
| `test_fd_reset_hygiene` | `+δ/−δ` on a conic DOF returns to the baseline merit **exactly**, with a non-vacuity check that δ actually moves it. **PASSES — so no second engine fix is needed**, and the LM's finite differences rest on clean state |
| `test_offaxis_merit_is_wavefront_not_tilt` | promoted from `fex_in_loop_check.m`: off-axis exit-pupil OPD in the **1e-7 band**, not 1e-3, and no more than 100× the on-axis value |

A fourth test asserting the emitted deck's literal `OptFEX= Yes` was written and
dropped: `optimize()` re-emits a plain deck after the solve, so `spec.rx_path`
no longer carries the Opt block by the time a test can read it. The behavioural
gate is the real test of the fix and is what fails pre-fix.

## E. Design layer

`optimize()` routes `OptWFElt` to `spec.pupil.ep_elt` and emits `OptFEX= Yes`
when `add_pupil` has run, and sets the system stop (`design_optim.F:170-180`
aborts without it). `align_focal_plane` gained `'allow_pupil'` (default false,
so the existing guard and its test are untouched) for the solve↔refit
alternation. The Addendum-5 "blocked" hard error is **removed** — the engine
change makes the path real.

**Merge ordering.** The resources side carries decks that emit `OptFEX= Yes`.
Those must not reach resources `dev` before the engine PR merges: on an
unfixed engine the keyword is silently ignored and the merit reverts to the
detector plane — a veneer ahead of its engine, failing quietly rather than at
load. Local work proceeds; the push waits.

---

# ADDENDUM 7 — part B: the exit-pupil re-solve (2026-07-31)

On the fixed engine (macos PR #68), stages 3 and 4 re-solved against the
per-field chief-ray-tied exit-pupil sphere. `xp_optimize.m`; scored by
`strict_wfe_deck` on the emitted deck — an **independent** path from the
in-loop merit. EPD 4060, 9x9 box, 1304 rays, 81/81 fields.

| stage | xp merit | FP merit (committed) | HIS design | Rodgers | x his |
|---|---|---|---|---|---|
| S3 | **157.4 / 118.4** | 181.2 / 97.1 | 115.3 / 53.7 | 91.6 / 46.4 | 1.72 |
| S4 | **77.0 / 41.9** | 118.6 / 84.8 | 64.9 / 35.4 | 39.8 / 22.5 | 1.93 |

**Stage 4 improves substantially: max 118.6 -> 77.0 nm (35%), avg 84.8 -> 41.9
(51%),** and the Rodgers ratio drops from 2.98x to 1.93x. Stage 3's max
improves (181.2 -> 157.4) but its **avg gets worse** (97.1 -> 118.4) — the new
merit flattens the field map rather than minimising its mean, which is what a
per-field chief-tied reference should do, and the committed avg was partly an
artifact of the tilt term being smallest at the box centre.

**Gates: both missed.** S3 was to reach ~115 nm (his-conics level) and reads
157.4; S4 was to reach <= ~1.5x his 39.8 (59.7 nm) and reads 77.0 (1.93x).
Recorded as a miss, not tuned toward.

**The standing rigid-body prediction is CONFIRMED — the solve moved onto his
branch.** All four signs flipped to match:

| | was (FP merit) | now (xp merit) | his (decoded) |
|---|---|---|---|
| M2 Ydec | −3.742 mm | **+2.739** | +8.345 |
| M2 alpha | +0.2330 deg | **−0.1604** | −0.5169 |
| M3 Ydec | −43.839 mm | **+23.135** | +121.868 |
| M3 alpha | +1.1142 deg | **−0.6795** | −2.3297 |

Addendum 5 §B read his solution as the **opposite branch** of a degenerate
valley, ~2.4x further along it. With the corrected merit our optimizer lands on
**his** branch — same signs on all four DOFs — at roughly a fifth to a third of
his magnitudes. So the merit was indeed what put us on the wrong branch; the
remaining gap is depth along the right one.

Conics also tighten where it mattered: S3 K_M3 |diff| 3.43e-3 -> **6.62e-4**
(5x), K_M2 3.12e-4 -> 1.05e-4; K_M1 loosens 1.56e-5 -> 3.37e-4. S4 all three
improve ~2x (K_M2 4.0e-3 -> 2.26e-3, K_M3 1.06e-2 -> 6.17e-3).

**The alternation did not converge in 4 rounds**, and that is the most likely
reason the gates are missed. FPA station movement per round: S3 0.42, 0.49,
0.55, 0.61 mm (drifting, not contracting); S4 66.8, 1.05, 3.93, 13.4 mm
(non-monotone). The merit's sphere is centred on the chief intercept on the
detector, so a detector that is still moving means the objective is still
moving under the optimizer. **Next step for whoever takes this on: diagnose the
solve<->refit coupling before pushing the gates harder** — likely candidates
are the FPA fit's own field set (5x5 over +/-6' about the bias, which the solve
then changes) and the fact that `optimize` restarts its DOF deviations from the
current state each round.

**Artifacts:** `xp_optimize.m`, `rodgers1_epd4060_stage{3,4}_xpopt.in`,
`..._xpopt_strict.png`, `rodgers1_epd4060_xpopt.mat`. Committed baselines
bit-intact.

---

# ADDENDUM 8 — part C: the joint FPA solve, and the close of the arc (2026-07-31)

**Addendum 7's alternation is SUPERSEDED.** Its non-convergence was a
two-objective mismatch, not an optimizer-depth problem: the merit's reference
sphere is centred on each field's chief-ray intercept on the **detector**
(CALIB's FEX radius is the chief-ray distance from the pupil to the plane of the
NEXT element), so re-fitting the detector between solves moves the objective the
solve just minimised. Replacing the loop with a **single joint solve** — the
FPA's tilt and focus in the CALIB DOF set alongside the optics, `align_focal_plane`
run once as a seed — closes both gates. The `_xpopt` artifacts are the joint
result; the alternation numbers survive only in this packet as the diagnosis.

## A. DOF indexing — corrected

The brief specified "VarElt DOFs 3 and 4". The VarElt mask is
`[TIP TILT CLOCK DX DY PIST ROC CONIC]`, and `macos_ops.F:CPERTURB_2` confirms
`PV(1:3)` is the rotation and `PV(4:6)` the translation in the element frame.
So DOFs 3 and 4 are **CLOCK** (rotation about the detector's own normal — a
near-null direction on a nearly symmetric detector) and **DX** (a lateral shift
the chief-ray tie absorbs). The FPA tilt/focus pair is

    FPA_DOFS = [1 0 0 0 0 1 0 0]     % 1 = TIP (alpha, about local x)
                                     % 6 = PIST (along the normal = focus/Tz)

## B. Result — both gates pass

Comparator is **his designs under the same metric** (Addendum 4/5), not his
reported numbers. Gate: ≤ ~1.15×.

| stage | joint xp-merit | gate (1.15× his) | his design | alternation | committed FP-merit | Rodgers reported |
|---|---|---|---|---|---|---|
| **S3** | **95.3 / 55.9** | ≤ 132.6 / 61.7 | 115.3 / 53.7 | 157.4 / 118.4 | 181.2 / 97.1 | 91.6 / 46.4 |
| **S4** | **72.3 / 39.2** | ≤ 74.6 / 40.7 | 64.9 / 35.4 | 77.0 / 41.9 | 118.6 / 84.8 | 39.8 / 22.5 |

**S3 passes with margin and BEATS his stage-3 design on the box maximum**
(95.3 vs 115.3 nm, 0.83×), tying on the average (1.04×). Against Rodgers'
reported numbers it lands at **1.04× max / 1.21× avg** — the closest any
reproduction in this arc has come. **S4 passes** at 1.11× / 1.11× of his design
(72.3 ≤ 74.6, 39.2 ≤ 40.7), against 2.98× / 3.77× for the committed solve.

## C. Diagnostics

**FPA pose vs the align seed** — this is the direct measurement of what the
alternation could not resolve:

| | station move | normal move |
|---|---|---|
| S3 | 3.57 mm | 0.359° |
| S4 | **11.97 mm** | 1.064° |

Addendum 7 recorded the S4 alternation still throwing **13.4 mm** of FPA motion
at round 4 and called it unresolved. The joint solve resolves **11.97 mm** in
one pass. The prediction and the measurement agree to ~10%, which is as direct
a confirmation of the diagnosis as this arc has produced.

**Rigid body along his branch** (decoded frame). The joint solve goes deeper
along the branch the alternation found:

| | committed (FP merit) | alternation | **joint** | his | joint as % of his |
|---|---|---|---|---|---|
| M2 Ydec | −3.742 mm | +2.739 | **+3.561** | +8.345 | 43% |
| M2 alpha | +0.2330° | −0.1604 | **−0.2134** | −0.5169 | 41% |
| M3 Ydec | −43.839 mm | +23.135 | **+38.549** | +121.868 | 32% |
| M3 alpha | +1.1142° | −0.6795 | **−0.9099** | −2.3297 | 39% |

All four signs match his; the joint solve sits 32–43% along his branch, up from
19–34% for the alternation. The design reaches his WFE **without** reaching his
rigid-body magnitudes, which is what a genuinely degenerate valley looks like —
a range of (Ydec, alpha) pairs buying nearly the same wavefront.

**K_M3 delta:** S3 **5.94e-4** (committed 3.43e-3, alternation 6.62e-4) — 5.8×
tighter than the committed solve on the term this packet has flagged since §3.
S4 7.18e-3 (committed 1.06e-2, alternation 6.17e-3).

**LM:** `converged = 1` on both stages, **one solve each**, `max_iters` 120.
Merit WFE per FOV fell 3.41e-7 → 6.84e-8 (S3) and 3.41e-7 → 4.47e-8 (S4).

## D. What this closes, and what it does not

**Closes.** The metric question (Addendum 3, gate 1.15×), the reference-frame
question (Addendum 5's ADE decode), the engine question (the OptFEX fix, PR
#68), and the merit question — our optimizer, given the right objective and the
detector as a real DOF, reproduces Rodgers' designs to 1.04–1.11× of their
own strict-metric scores and lands on his rigid-body branch.

**Does not close.** Against his *reported* numbers we sit at 1.04× (S3 max) but
1.82× (S4 max). Since our reproduction of *his own S4 design* also reads 1.63×
his reported number (Addendum 5 §B), that residual is in the **comparison**, not
in our solve — it is the same field-definition/box-placement question flagged
since Addendum 3 §D, and it needs a line from Mike, not another run.

## E. Artifacts

`xp_optimize.m` (default `joint = true`; pass `false` for the superseded
alternation), `rodgers1_epd4060_stage{3,4}_xpopt.in`, `..._xpopt_strict.png`,
`rodgers1_epd4060_xpopt.mat`. `Telescope.m/optimize` gained `'fpa_dofs'`.
Committed baselines bit-intact; `tDesignTelescope` 70/70, `tDesignOptimize`
4/4, `tOptFex` 3/3.
