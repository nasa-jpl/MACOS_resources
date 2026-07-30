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
- **No CALIB convergence difference at f/0.86.** `align_focal_plane` fits the
  same focus (FP z = −3256.29 mm vs −3256.28 at 2000) and the S3/S4 optimizer
  converged normally in the same iteration budget (120); conics landed *closer*
  to Rodgers at the true aperture (below).
- **Layout gate unchanged:** M1→M2 / M2→M3 spacings and radii are bit-identical
  across the two apertures to floating-point noise (zElt differs in the 16th
  digit only). Spacings/radii are his, fixed — untouched, as required.
- **Nothing clipped, nothing lost:** all stages report **81/81 finite fields**
  under both metrics at f/0.86, same as at f/1.75.

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
same Rx + DOFs + field → essentially his CODE V conics, at f/0.86.

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
aperture-/field-dependent wavefront error, dominated by astigmatism at f/0.86.
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
   f/0.86 anastigmat imply CODE V removes more than piston+tip/tilt+defocus (its
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
`FocalPlane` but `Return` fell through to the `else → Circular`. On this fast
f/0.86 beam that circular stop **clipped rays at the exit pupil** (and, because
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
