# PLAN — afocal4: 4-mirror afocal telescope driver with interface-pupil control

> **Status: S1-S5 DELIVERED, then S4 RETRACTED ON PACKAGING and redone as
> S4b (TO, 2026-08-02/03; local on `MACOS_res_dev` `dev`, not pushed).
> Results: `design/examples/afocal4/RESULTS.md` (S4 as the unconstrained
> reference, §S4b as the buildable trade).**
>
> **THE S4 TRADE IS NOT BUILDABLE.**  Every design on it puts the
> collimator 200-440 mm in FRONT of M1 -- and with it the field mirror,
> the interface pupil and the whole instrument behind the pupil -- inside
> the incoming beam.  One extra mirror flips the parity of the back end:
> his three-mirror parent has M3 640 mm behind M1; the four-mirror child
> built from the same front end has it 200 mm in front.  The numbers stand
> unaltered as the unconstrained reference (retract in place); S4b
> re-derives the curve with the constraint enforced as a solver wall.  See
> the BUILDABILITY CONSTRAINT section below.
>
> **The S4 headline is a PAIR, and only half of it is met.**  The convex
> field mirror at the intermediate image SOLVES the interface pupil: on
> axis, one joint solve puts blur (43.4 um), wander (44.0 um), breathing
> (0.233%), convergence-surface figure and magnification (29.9811x) ALL
> inside their targets -- including the 56 um wander the S3 gate flagged as
> unreachable by any form at first order.  It is paid for in IMAGE QUALITY:
> at the flagged 140 mm operating point the wavefront error floors near
> 8.5 um at the design field, 120x the diffraction limit, and no conic,
> standoff, front-end or rigid-body freedom moves it more than a few
> percent (measured: a wavefront-ONLY re-solve of the same DOFs reaches
> 8467 nm against the frozen design's 8835 -- 4%).
>
> **The two qualities are the same knob.**  The field mirror's power is
> consumed holding the exit pupil at the interface standoff, so more pupil
> control means more power means more field curvature and astigmatism.  The
> interface standoff IS the exchange rate; carrying it as a PARAMETER (the
> S4 ruling) rather than a spec is what makes that reportable.  At the far
> end of the trade -- 343 mm, phi4 -> 0 -- the fourth mirror becomes a flat,
> the wavefront returns to 15.8 nm on axis and the pupil reverts to the
> three-mirror's.
>
> **Rung 4's rigid bodies buy 0.4%** and the solve drives them to 0.2 um /
> 0.1 urad; his three-mirror gains 25% from the same freedom.  The residual
> here is field astigmatism growing across the box (1108 -> 3312 -> 6809 nm
> over YAN -0.25 -> 0 -> +0.25 deg), and a rigid body adds a field-CONSTANT
> term.
>
> **The Mersenne hedge is CLOSED.**  Four conics relaxed with the confocal
> spacings held take its wavefront error from 59.4 um to 35.1 um against a
> 71 nm target, and only to 3955 nm even on axis.  The 59 um was never
> mostly about the parabolas.  The field mirror stands.
>
> **Open for Dave/Mike:** the operating point.  The trade curve is the
> deliverable, not a chosen value -- the instrument's interface standoff
> picks the point, and 140 mm is only the flagged default.
>
> Ground truth: `design/rodgers2/PACKET.md` (S1/S2),
> `design/examples/afocal4/FORM_STUDY.md` (S3),
> `design/examples/afocal4/RESULTS.md` (S4/S5).
> Response to Mike Rodgers' Rodgers2 drop
> (`~/dev/MACOS_sandbox/Design/Rodgers2/`): a 30× afocal 3-mirror
> telescope, 0.5°×0.5° FOV offset 0.6°, with his verbal finding that
> "with 3 mirrors, the pupil quality is not very good; a 4th mirror is
> needed for pupil control."  Written for cold implementation by Opus,
> Sonnet, or a user, one stage per agent run, with Fable/Dave review at
> each stage gate.  Read this file, `macos/CLAUDE.md`, `mmacos/CLAUDE.md`,
> `design/rodgers1/README.md` + `PACKET.md` Addendum 10, and the
> referenced sources before writing any code.

## What this is

Two products, one arc:

1. **A user-accessible design driver** — `macos.design.Telescope` grows
   afocal support (afocal terminal surface, interface-pupil/coldstop
   element, pupil-imaging metrics), so a user can generate an N-mirror
   AFOCAL telescope and score both image quality AND interface pupil
   quality.  The 4-mirror 30× offset-field design is the demonstration.
2. **The Rodgers2 benchmark study** — his four `.seq` decks transcribed,
   audited, and scored under OUR metrics (rodgers1 precedent), producing
   the quantitative "3-mirror pupil is poor" table his deck asserts only
   verbally, and the 4-mirror ladder that answers it.

Homes:
- `design/rodgers2/` — the Mike-facing benchmark record (flat dir,
  rodgers1 shape: transcription `.m`, audit, PACKET.md, README.md,
  committed `.in` + `.png` + `.mat` per variant).
- `design/examples/afocal4/` — the user-facing staged example
  (`afocal4_params.m` + `s1..s4` + README; e2e2 house rules: artifacts
  in the dir, **no `exit(0)` in example scripts**, one `P` struct).
- Shared kernels in `design/src/`; builder changes in
  `src/+macos/+design/Telescope.m`.

## The benchmark (established facts — do not re-derive)

From the deck + `.seq` extraction (2026-08-02):
- EPD 1000 mm, λ = 1000 nm, DIM mm, 30× afocal.  M1 R=−2500 K=−1
  parabola (fixed, hole 130 mm), stop 50 mm ahead of M1.  Intermediate
  image between M2 and M3; M3 recollimates; "recenter" fold
  (ADE 17.7–22.0° ≈ the ×30-magnified field-center angle), then the
  tilted **coldstop** = the interface pupil, then 1000 mm to SI
  (`AFI −1000` evaluation).
- His ladder (max RMS WFE in the used FOV, CodeV field maps):
  **15 nm** on-axis → **430 nm** offset 0.6° unoptimized (mag drops to
  28.7×) → **160 nm** conics+radii reoptimized (30× restored) →
  **119 nm** + M2/M3 tilt/dec.  In-box averages: 4.0 / 154 / 93 / 48 nm.
- **The deck contains no pupil metric.**  The only pupil-adjacent
  evidence: coldstop DAR tilt 0 / 4.289° / 3.577° / **−0.356°** across
  the variants, and the 30→28.7× magnification loss.  The pupil-quality
  definition is OURS to supply (S1c) — state that explicitly in any
  material that goes back to Mike.
- DL at λ=1 µm is RMS ≤ λ/14 ≈ **71 nm** — his best 3-mirror (119 nm)
  misses it; the 4-mirror target is DL in-box PLUS controlled pupil.

## BUILDABILITY CONSTRAINT (Dave, 2026-08-03 — binds every solution; S4b redo)

The S3/S4 solutions place back-end optics (M3, the field mirror, the
interface pupil — and therefore the entire instrument that follows it)
IN FRONT of M1, in the incoming beam.  **Not buildable.**  Mike's
parent is the existence proof of the fix: his M3 sits ~630 mm behind
M1 and the recenter fold takes everything else out of the beam.
Every candidate from S4b on must satisfy:
1. **M3 well behind M1**: `z_M3 − z_M1 ≥ P.pack.m3_behind_min`
   (parameter; default 500 mm).
2. **A fold must be insertable** downstream (after M3 or after the
   field mirror) with engine-truth daylight (`fold_station_report`),
   taking the field mirror + interface pupil + instrument volume into
   the x-y plane behind M1 — fold rules per `project_fold_extraction`
   (M1 keep-out honored, clearances engine-truth).
3. **Demonstrated, not asserted**: the folded variant is emitted and
   rendered (yz + xz), ray-to-body clearance margins reported.
Layout renders are a REVIEW GATE before any result is quoted; the
packaging check includes INSTRUMENT-VOLUME PLACEMENT, not just train
length/AOI/self-obscuration (the S3 gap that let this through).
The unconstrained S4 results stand as the reference trade, labeled
NOT BUILDABLE in RESULTS.md; S4b re-derives the trade curve under
the constraint.

**S4b DELIVERED (2026-08-03).**  `design/examples/afocal4/STATUS_S4B.md`
is the one page; `RESULTS.md` section S4b is the detail.  Headlines: the
constraint does not cost image quality, it SPLITS the S4 design's
performance and forces a choice, because compliance moves the fourth
mirror off the intermediate image -- gaining a footprint (its conic does
wavefront work) and losing the field conjugate (it stops doing pupil
work).  His front end held closes at every operating point and reads
3451 nm against the unbuildable 9600, paying 1141 um of blur against 167.
A 4.5% slower secondary puts the image 900 mm behind M1, the field mirror
back ON it, and recovers the pupil (149 um blur) for 2.5-3x the wavefront
and 240 mm of length -- a second basin the solver does not find unaided,
so both are swept.  At 343 mm, the only standoff where the package closes
around a real instrument, pupil control costs a FACTOR OF 40 in wavefront
and declining it returns his three-mirror.  A second constraint appeared
that S4 never faced: the interface standoff sets the instrument's girth
(0 mm at 90, 464 mm at 343), so the pupil metrics want it short and the
package wants it long.  The fold is demonstrated null to 4.8e-12 and
caught two defects doing it.  `tAfocal4` 7/7.

## Standing doctrine (rodgers1 + e2e2 — apply at every stage)

1. **Joint solve, never alternate** (one CALIB DOF set; seeds are seeds).
2. **State the wavefront reference.**  For afocal decks the reference is
   a PLANE, not a sphere — the afocal rung ladder (S1a) replaces the
   focal one; name the rung on every number.  For comparison with his
   CodeV maps, decode his reference convention first (rodgers1
   Addendum-2 lesson: reporting conventions differ 1.3–1.7×).
3. **Solve set ≠ scoring set** — solve on his 3×3 XAN/YAN points, score
   on `field_grid` uniform over the 0.5°×0.5° box (his quincunx
   edge-weighting biased averages 8% in rodgers1).
4. **Pupil gate after every source/aperture edit** (`pupil_gate.m`;
   engine correct since PR #70, the gate stays).
5. **Frame before angle** — his ADE sign is OPPOSITE ours
   (rodgers1 `convention_decode.m`, 16-combo measurement; reuse, do not
   re-derive).  Every reported tilt names its reference.  DIM mm deck:
   watch the metres↔BaseUnits traps (`rmsWFE` is metres; dwdz-class
   supervisors return BaseUnits).
6. **Parameter provenance table per stage** (`param_table.m`) — the
   parameter table IS the solution; add magnification and pupil rows.
7. **Report + views per stage** — `design_report` + `view_std`/`view_rx`
   + field maps, saved beside the `.in`.
8. **Transcription, not parsing.**  `.seq` → `.in` is a hand
   transcription `.m` with a decoded-syntax header + an audit section
   (rodgers1 `rodgers_seq.m` + PACKET Addendum 10 §10.6 pattern).  No
   seq parser exists and none is to be written.
9. **His designs under our metric** — the comparator is his decks
   scored by OUR kernel (gate ≤1.15×), never his printed numbers.
10. **Mechanics**: model_size ≥ nGridpts; one MATLAB process per model
    size; `macos.modify()` after programmatic pokes; makems from
    `~/dev/macos`; matlab -batch wrappers supply `exit(0)`; work lands
    on `dev` both repos, engine first; push only when Dave asks.

## The stages

### S0 — Parameters (`design/examples/afocal4/afocal4_params.m`)
Pure `P` struct: EPD 1.0 m, angular magnification M = 30 (exit beam
33.33 mm), λ = 1 µm, field box 0.5°×0.5° at +0.6° Y bias, coldstop
interface spec (stop diameter = EPD/M; distance behind last powered
mirror — default from Mike's geometry), solve fields (his 3×3),
scoring grid, model_size, targets:
- **Image**: in-box max RMS WFE ≤ 71 nm (DL @ 1 µm), afocal ladder
  rung stated.
- **Pupil** (defaults; Dave may retarget): traced magnification
  30.000 ± 0.1% at box centre, **chief-normal** (`.mag_centre_chief`;
  the placed-plane read carries the interface plane's own obliquity —
  S3 measurement check, PACKET §4); then the S1c four-part ladder —
  per-node blur,
  convergence-surface power/astig, pupil distortion (wander shape
  across the pupil), and wander at the placed coldstop plane
  (≤ 1% of pupil radius) — each ≥ 10× better than the S2 3-mirror
  baseline.

### S1 — Afocal scoring + pupil-metric infrastructure (`design/src/`)
**DONE** (tAfocalKernel 11/11, tPupilMap 7/7; `afocal_*` + `pupil_map`
in design/src).  Findings that amend this plan: `Element= Return`
REVERSES ray directions — the interface flat must be `Element=
Reference` (OPL unchanged, so the error hides from piston-only
checks); XPS's differential is TANGENTIAL (`th` is a rotation vector
about x) — match the axis before comparing; CALIB loads/converges on
an afocal deck but its merit is the un-referenced OPL spread on the
tilted coldstop (10³–10⁴× the WFE) and the solve is destructive
(K_M2 −1.78→+6.92, 152 nm→288 µm) — MATLAB outer solve confirmed as
the path; an engine-side afocal reference (plane analogue of FEX) is
a recorded follow-on, out of scope.  Anchoring correction measured at
2.9–3.6% of blur here (real but not dominant; keep it — it scales
with primary speed).  The enabler stage; everything else consumes it.
- **(a) Afocal strict kernel**: `afocal_refs.m` / `afocal_rungs.m` /
  `afocal_wfe_deck.m` / `afocal_ladder_deck.m` — RMS OPL to a FLAT
  reference normal to the exit chief at the coldstop, from
  `get_ray_info` position/direction/OPL (rays are straight after the
  last surface; the sphere kernel's R→∞ limit).  Rungs: {piston-only |
  +tip/tilt (= boresight, not error) | +power (= residual divergence,
  reported both as nm and µrad)}.  Gate `tests/tAfocalKernel.m`:
  cross-kernel identity vs `strict_rungs` on a focal deck with a
  long-R sphere, plus rung-ordering invariants (tStrictKernel pattern).
- **(b) Scoring-lens harness**: `afocal_score_psf.m` — append
  `macos.design.ideal_lens` (+`ideal_lens_emit`, K=−n² stigmatic
  singlet, verified ~1e-14 rmsWFE) behind the coldstop in a SEPARATE
  scoring deck so Strehl/PSF/`design_report` work unchanged.  The
  delivered afocal `.in` stays clean.
- **(c) Pupil-imaging metrics — the cone-convergence model (Dave,
  2026-08-02)**: `pupil_map.m` (new, design/src).  Model: points on the
  M1 surface (= entrance pupil here) are imaged through the train to
  the exit pupil.  For each node of an M1 grid, the imaging bundle is
  the CONE of rays through that node — one ray per field angle, so
  **the cone aperture IS the used field set** (0.5°×0.5° at +0.6°
  bias; exit side opens ×30, half-cone ~7.5° centered ~18° off-axis).
  Never trace a wider or symmetric cone: rays outside the field range
  don't exist in operation.  This is the finite-field generalization
  of the engine's XPS field-differential crossing (the two-ray limit).
  - Implementation (no Fortran): trace the full grid at K field
    angles; `ray_pos_hist` gives every ray's position at M1 and at
    the coldstop; per node, closed-form least-squares closest point
    of the K ray-lines.  Cost = K traces.
  - **Fast-primary anchoring (Dave, 2026-08-02 — MANDATORY, not a
    refinement):** cones must be anchored at fixed points ON the
    curved M1 SURFACE.  Same-index grouping across field traces is
    NOT that: a fixed source-grid ray wanders on M1 by ≈ (stop offset
    + sag)·Δθ_field, growing as r² toward the rim — here (f/1.25:
    50 mm sag + 50 mm stop offset) × 8.7 mrad ≈ 0.9 mm at the rim,
    which ÷30 masquerades as ~29 µm of fake pupil blur at the XP —
    several × the diffraction floor.  Fix: regrid/interpolate the K
    traces onto common M1 surface nodes (the wander map is smooth),
    or ray-aim (1–2 Newton corrections on the near-linear source→M1
    map).  Gate: anchoring residual per node ≪ the blur being
    measured.  Do NOT repair rim partial cones — field angles whose
    ray through a rim node falls outside the stop don't exist in
    operation; the asymmetric partial cone is the physical pupil-edge
    behavior.
  - **Curved-object reference:** the anchored object is M1's curved
    surface (50 mm deep here); even perfect pupil imaging maps that
    depth to the XP at longitudinal mag m² = 1/900 (≈56 µm) — same
    order as the residuals.  Ladder part (2) therefore uses TWO
    references: convergence points vs the IDEAL image of M1's sag
    (sag·m² through the fitted transverse map) = true pupil-imaging
    aberration; vs the FLAT placed coldstop = the operational number.
    Never charge a fast primary's own faithfully-imaged sag as
    "pupil curvature".
  - Report the FOUR-PART LADDER, never one merged residual:
    (1) per-node convergence waist = pupil-image BLUR;
    (2) the surface of convergence points → tilt (frame term) /
        power / astig — the "flat coldstop can't be conjugate
        everywhere" numbers;
    (3) the transverse map M1→XP: magnification (target exactly
        1/30 — Mike's 28.7× slip lands here), anamorphism, and
        **pupil DISTORTION** (nonlinear mapping residual = wander
        SHAPE across the pupil; tracked as a first-class metric,
        Dave 2026-08-02);
    (4) residual at the PLACED coldstop plane = operational pupil
        WANDER (where a cone ray pierces the placed plane is where
        that M1 point maps at that field — non-convergence there IS
        the footprint shear the instrument feels).  Plane placement
        is itself a small fit (minimize wander over the used pupil;
        mirrors Mike's coldstop DAR tuning).
  - Simulated pupil image: |cone spot function|² ⊗ M1 amplitude map
    (hole, gaps, spiders).  Valid because extended-field illumination
    makes pupil imaging INCOHERENT (field angles mutually incoherent;
    the field is the pupil-imaging aperture).  Caveats to encode:
    single-kernel convolution assumes isoplanatism — use a few-zone
    piecewise kernel near the pupil edge where coldstop margins are
    decided; quote the diffraction floor λ/(2·NA_field) once
    (~4 µm on the 33 mm exit pupil here — geometric treatment valid);
    vignetted rim cones (hole/obscuration edges) have asymmetric
    kernels the convolution misses — the traced data contains it.
  - SCOPE: this model requires a field.  A testbed/interferometer
    with a point source has no ray-cone pupil image (one ray per M1
    point) — pupil quality there is a Fresnel/Talbot wavefront
    question, handled separately later (Dave).  State this limit in
    the utility's doc block.
  - Cross-check + gate: per-field `macos.pupil_quality` (XPS sag
    Zernikes) must agree with the cone-convergence surface in the
    small-field limit — that identity is the `tPupilMap` test.
- **(d) Optimization path (settled — Dave 2026-08-02): MATLAB outer
  solve** over the afocal kernel + pupil-metric merit
  (`zern_jacobian_solve` SVD pattern / lsqnonlin) is the sanctioned
  baseline for this study.  A CALIB-afocal probe (rodgers1
  `optfex_default_probe.m` style: can CALIB take a deck terminating
  in a flat Return, and what does its FEX-radius merit mean there?)
  is still worth one short run for the record, but nothing gates on
  it.  If the MATLAB path proves out and native speed is wanted,
  modifying CALIB becomes a follow-on engine task — NOT part of this
  plan.  **No engine work is expected in this plan.**

### S2 — Rodgers2 baseline reproduction (`design/rodgers2/`)
**DONE** — see PACKET.md.  Headlines: his CODE V field-map RMS = our
rung 2 (piston + per-field tip/tilt), reproduced 0.952–1.015× max on
a uniform grid, all four variants (bracketed by rungs 1/3, not
fitted); his in-box averages are AREA averages (his 3×3 solve set is
⅓ corners — 2.11× vs 1.04×); ADE decode confirmed at 10⁹ margin.
Baseline pupil table (33 mm XP, 2.7 µm diffraction floor): blur 153 µm
rms on-axis / 775–802 µm offset (0.5% / 2.3–2.7% of pupil dia — pure
geometry), mag 28.686× on the unoptimized offset (his 28.7×) restored
to 30.0015× by the conic re-solve but BREATHING ±3.8% across the box,
convergence surface 1.7–1.9 mm P-V off the flat coldstop (~50 µm of
it = the correctly-imaged M1 sag, matching the m²·sag prediction),
wander 0.9 mm rms with ≤18% recoverable by re-placing the plane.
His coldstop tilt is not wander-optimal (fit wants ~+3° and 2–3 mm) —
question queued for Mike.
- `rodgers2_seq.m`: verbatim transcription of all four `.seq` (decoded
  header + PACKET audit section §-per-surface).  Reuse
  `convention_decode` results; reuse `Telescope`/`Bench` emission per
  `rodgers_seq.m`.
- Score all four under the afocal ladder, solve set AND uniform grid;
  gate: our strict numbers vs his 15/430/160/119 nm ≤ 1.15× at the
  matched rung (rung matching is part of the work — his CodeV afocal
  reference must be decoded once, then pinned by test).
- Run `pupil_map` (full four-part ladder + simulated pupil image) +
  `pupil_quality` on all four → **the baseline pupil table**: the
  first quantitative statement of "3-mirror pupil quality is not very
  good."  Expect the coldstop-tilt story (4.3° → −0.36°) to reappear
  as measured convergence-surface tilt; expect wander, distortion and
  curvature to be the uncontrolled residuals.
- Emit: 4 committed `.in` + field maps + pupil table + PACKET.md with
  the audit; README with run lines.  Artifact naming:
  `rodgers2_<variant>_<stage>[_metric].{in,mat,png}`.

### S3 — 4-mirror afocal form + first-order layout
**FORM STUDY DELIVERED (2026-08-03) — `design/examples/afocal4/FORM_STUDY.md`.
The form CHOICE is a Dave/Fable gate; S4 does not start until it is made.**
Headline: **the 3-mirror already closes BOTH first-order conditions** —
M = 30.000, exit pupil 343.363 mm past M3 against the coldstop Mike placed
by hand at 344.173 mm, i.e. **0.81 mm apart on a 33 mm beam**.  There is no
first-order deficiency for a 4th mirror to repair, and a study that closes
only the first-order conditions returns a **flat** (R = 45–78 km).  So the
4th mirror must be justified on pupil ABERRATION, and the discriminator is
which form can be GIVEN power without spending the first-order solution.
**Recommend (i) the field mirror, convex, ~200 mm standoff** — the only form
that preserves M and the afocal condition IDENTICALLY for any power (the
marginal ray is small where it sits), so its power acts on the chief alone.
One unoptimised K=0 mirror: blur 794 → 286 µm, breathing 3.97% → **0.156%**
at φ4 = +2 /m (INSIDE the ±0.4% target), M held at 29.92, every conic
unspent.  **Runner-up kept alive: (ii) the double Mersenne** — best pupil
ladder measured (blur 175 µm, breathing 0.285%, wander 468 µm, 26% shorter
train) but rung-2 WFE 59 µm and a 53 mm interface pupil; promote it iff a
conic-relaxed version (confocal SPACINGS kept, 4 conics spent on image
quality) reaches DL.  **(iii) the downstream relay is eliminated** — worse
than the parent on every pupil term, 39 µm WFE, 11.7 mm wander, +25% train,
a real internal focus, and it cannot reach the parent's interface distance
without R3 → 0.  Also earned: a collimated exit space has NO first-order
freedom (a powered element there breaks afocality), and the field-mirror
leverage is SIGNED — concave makes everything worse.
Shipped with it: `afocal_first_order` (design/src) + the stop-and-fix
fixture `optical_design/fixtures/afocal_tma_fixture.{json,md}`;
`afocal4_params` / `afocal4_close` / `afocal4_forms` + 7 committed `.in`,
3-view renders and the comparison figure; `tDesignAfocal` 8/8.
Builder bug closed en route: `resolve_nmirror_` emitted a CONVEX third
mirror as concave (parity rule started at k=4) — 17 mrad off collimated,
caught only because the study verifies each trace against its own paraxial
prediction before taking a metric.
- **Form study** (paper section in PACKET + a layout script): candidates
  (i) his TMA + M4 field mirror NEAR the intermediate image (pupil
  relay, minimal disturbance to the imaging solution), (ii) double-
  Mersenne derivative (two confocal-parabola pairs, M = m1·m2,
  anastigmatic, pupil relayed by construction), (iii) TMA + M4
  downstream of M3.  Two first-order constraints close M4: the afocal
  condition and the pupil-imaging condition (stop imaged onto the
  coldstop at pupil mag 1/M).  Pick by: pupil-relay quality at first
  order, packaging off the offset field, DOF count left for the solve.
- **Builder work** (`Telescope.m`): afocal terminal support —
  `resolve_nmirror_` accepts an explicit `'spacing_after'` on the last
  mirror (afocal condition replaces the `'derive'` paraxial focus);
  `add_exit_reference`/coldstop emission — **`Element= Reference`,
  flat, normal to the exit chief** at the interface distance (NOT
  `Return`, which reverses ray directions — S1 finding, PACKET §5.0);
  `add_pupil` sibling for the afocal case (2-pass pattern, flat not
  sphere — the FP-anchored `psi`/`Kr` construction has no afocal
  counterpart).  `seidel_seed` cannot seed this chain
  (known-unreliable for reimagers): follow the `tma_3plus1` recipe —
  solve the 3-mirror parent first, carry conics via
  `add_mirror(...,'conic',K)`, seed M4 at K=0.
- **Numeric targets (gate review, 2026-08-03 — "≥10× baseline" made
  concrete against the BEST 3-mirror, the S4 tilt/dec variant):**
  blur ≤ 47 µm rms (from 469), wander at the placed plane ≤ 56 µm rms
  (from 557), **mag breathing ≤ ±0.4% CHIEF-NORMAL over the box (from
  ±3.63%; see the measurement check below — the pre-check number ±3.8%
  was the placed-plane read)**, convergence-surface P-V ≤ 0.2 mm net of
  imaged sag, M = 30.000 at box center, AND rung-2 WFE ≤ 71 nm DL
  in-box (his best is 119).
- **Pre-S3 measurement check — DONE (2026-08-03), and the headline
  survives.**  `pupil_map` now returns `.mag_per_field_chief` beside
  `.mag_per_field`, measured at the same station in the plane ⊥ that
  field's own exit chief; PACKET §4 carries the refinement in place and
  `tPupilMap/test_chief_normal_mag_carries_no_obliquity` gates it (8/8).
  Findings: the frame term IS exactly 1/√(cos incidence) (matched to
  7e−7 where the box-centre chief is normal to the coldstop); exit
  chiefs reach 10.5–13.6° of incidence; but it moves the corrected
  variants only a few percent of themselves — **S4 ±3.837% → ±3.634%**,
  and **S3 goes the other way, ±3.828% → ±4.071%** (his coldstop tilt
  masks part of the real breathing).  Box-centre M is untouched
  (28.6863 → 28.6848), so his 28.7× stands.  **Where it does bite is the
  PERFECT on-axis variant: 35% of the apparent 29.28–30.00 spread was
  frame, leaving 29.53–30.00 = ±0.78% of genuine pupil distortion** —
  which is the strongest single confirmation that the deficits are
  pupil-IMAGING aberrations, present at 15 nm of image quality.  Every
  magnification number from here on names its frame.
- **Form-study guidance from the baseline:** the deficits are
  pupil-IMAGING aberrations (blur is 0.5% of pupil dia even in the
  perfect on-axis design — the field-set aperture is large), so the
  candidate acting at the field conjugate — M4 near the intermediate
  image (between M2 and M3 in his train) — is the a-priori leader:
  it moves pupil imaging at first order while leaving image quality
  nearly untouched.  Double-Mersenne buys pupil relay by construction
  but repackages everything.  Judge on: the four ladder terms at
  first order, packaging off the 0.6° bias, and DOFs left for the
  joint solve.
- Add an afocal first-order fixture to `optical_design/fixtures/`
  (+ a short afocal section in `TELESCOPE_DESIGN_REFERENCE.md`);
  fixture gate is stop-and-fix, never widen.
- Gates: traced M = 30.000; traced pupil-relay condition (chief pierce
  at coldstop center for the bias field); pupil gate; view set shows a
  buildable train (no self-obscuration beyond M1 hole + M2 shadow).

### S4 — Optimize + the answer ladder (`design/examples/afocal4/`)
- Replay Mike's four-slide ladder WITH the 4th mirror, per doctrine:
  on-axis solve → offset unoptimized (the collapse number) → joint
  conics+radii re-solve at bias → + tilt/dec (M2/M3/M4) joint.  Solve
  = the S1d MATLAB outer loop, ONE joint DOF set per rung, merit =
  afocal WFE + weighted pupil-ladder terms (blur/distortion/wander);
  split image-vs-pupil DOFs via per-element masks where useful (3+1
  pattern); re-derive the pupil placement after each solve.
- Score every rung: afocal ladder (solve set + uniform grid) AND the
  full pupil table.  The headline result is the PAIR:
  in-box WFE ≤ 71 nm DL **and** pupil metrics ≥10× the S2 baseline,
  magnification held at 30.000.
- Expect and document the conic↔rigid compensation branch; record
  solve ORDER as numbered rules earned by failed runs (e2e 11-rule
  pattern).
- Emit per rung: `.in`, param-provenance column (radii, conics,
  tilts/decs, coldstop pose, magnification, pupil metrics), field map,
  pupil figure, report.

### S5 — Package + wrap
- Example README (the design procedure with its earned rules),
  `design_report` + `view_std`/`view_rx` set, PACKET.md finalized.
- Tests registered: `tAfocalKernel`, `tAfocal4` (build + M=30 + pupil
  gates), rodgers2 pins — add to the right model-size group in
  `run_mmacos_tests.sh`; fast suite green between stages, full suite
  before commit.
- Optional (Dave's call): response deck `deck_rodgers2.md` → pptx
  (slides/ precedent: `deck_rodgers_tma.md` + `make_rodgers_layout.m`),
  four-slide ladder mirroring his, plus the pupil table he didn't have.

## Execution model (delegation — save Fable for the gates)

- Each stage = one cold Task/Opus agent run off this file (the e2e2
  contract: names, masks, tolerances, failure modes are literal).
  S1a–S1d are separable agent tasks; S2 one agent; S3 form study and
  builder work may split; S4 one agent per rung if needed.
- Fable/Dave review at stage gates only: S1 kernel + pupil-metric
  gates, S2 baseline table (goes to Mike?), S3 form choice, S4
  headline pair, S5 deck.
- Branch state: stacks on the UNPUSHED local `dev` commits on
  `MACOS_res_dev` (e2e2 + pupil-fix wrap-up).  Nothing pushed until
  Dave says.

## Implementation notes for the executing agent

- Reuse, don't rewrite: `strict_refs`/`strict_rungs`/`strict_sphere_opl`
  (the afocal kernel is their plane limit — share code where possible),
  `stage_score`, `param_table`, `design_report`, `pupil_gate`,
  `field_grid`, `pupil_quality`, `xps`/`fex`, `ideal_lens(_emit)`,
  `convention_decode`, `xp_optimize` (joint-solve reference),
  `his_designs` (verbatim-scoring reference), `aoi_report`,
  `packaging_report`, `view_*`.
- The rodgers1 dir is the reference for: transcription+audit (Addendum
  10 §10.6), metric forensics order-of-battle, gates, artifact naming,
  PACKET discipline (retract in place, never rewrite history).
- `tma_3plus1/` is the reference for: 4-mirror build mechanics, conic
  carry, image-vs-pupil DOF split, `pupil_quality` reporting.
- CALIB caps: ≤12 FoV, on-axis implicit (11 explicit).
- mex relink chain after any engine change: rebuild
  `build_release_gfortran`, `gen_mex_wrappers.py`, `rm src/mmacos.mexa64`,
  `make FC=gfortran` (none expected — S1d is the only candidate).
- Expect basin path-dependence; record solve order as rules.

## Open questions for Dave

1. Response deck to Mike at S5 — produce it?
2. Two lines for Mike meanwhile: (a) what criterion set his coldstop
   DAR tilt (it is not wander-optimal — fit wants ~+3°, 2–3 mm, worth
   15–18%)?  (b) `CIR EDG 0.1` on the stop — decode unconfirmed
   (PACKET §1).
3. S0 pupil targets now carry the S3 review numbers; retarget if an
   instrument spec appears.

Settled (Dave, 2026-08-02): pupil metric = cone-convergence model with
cone aperture = the field set (valid because M1 is the entrance pupil;
point-source/interferometer pupil quality is separate future work);
pupil distortion tracked as a first-class metric; optimization in
MATLAB, CALIB modification only as a possible follow-on after the
approach is proven.
