# e2e6m round 2 — LOG

Round 2 of the 6 m end-to-end example, per `macos/BRIEF_e2e6m_redo.md`
(2026-08-26).  Round 1 (`../e2e6m/`) is FROZEN as the dry-run record;
its ratified decks are consumed read-only by path.

---

## 2026-08-26 — R0 opens: the slide-11 frame discrepancy

Pre-flight from the brief: the CTB tail was already committed by the
CTB session (`48d49b1`, the ctb_study config-driven driver) — clean
tree, nothing to do.

### What reading the code rules OUT before measuring

The round-1 diagnosis said "the channel triads and the perturb-path
TElt resolve x/y differently".  Reading both paths end-to-end says the
MATLAB call is IDENTICAL on both sides:

- S4's Jacobian columns: `run_sensitivities` → `macos.dw_dx_multi` →
  `macos.channels.rigid_body_channels` → `RigidBodyChannel.do_perturb`
  → `session.perturb(iElt,'rotation',r,'translation',t,'frame','local')`
  (`fp_mode` wraps FocalPlane elements only — segments get the plain
  channel; `Session.perturb` is a pure pass-through to `macos.perturb`).
- S5's drift/check: `macos.perturb(ie,'rotation',...,'frame','local')`.

Same veneer, same mex call, same frame flag.  So "two different local
frames" cannot be the whole story at the CALL level; the discrepancy
must be either (a) inside the engine's incremental local resolve as a
function of state/order, (b) in the harvest CONTEXT (per-field FEX
reset, field stacking), or (c) in the stored column ATTRIBUTION
(iElt/dof bookkeeping vs dwdxall column order).  Two facts from the
round-1 fingerprint sharpen (c): the model is x/y symmetric — which is
what the CENTER segment would give — and Tz agrees at 0.6% — piston is
the one column that is nearly THE SAME for every equal-area segment,
so a mis-attributed column would still close on Tz.  The engine side's
own pattern (Ry/Tx active, Rx/Ty null on an outer segment) is
physical: the near-null motions of a segment of a rotationally
symmetric parent are exactly the parent-surface-preserving ones.

Also verified: the deck's segments ARE elements 1–19 (`s3_seg_prop.in`
EltName census), so `P.ts.control_elts = 1:19` and "elt 19" = Seg19
are sound.  `TElt` (via `get_elt_csys`) is a 6×6 — room for a
rotation→translation lever-arm block, i.e. a SCALE riding on a
clocking, which is what the round-1 cross-pairing (~6.5 factor, not
magnitude-preserving) demands.

### The probe

`r0_frame_probe.m`: on the round-1 deck, element 19 —
  [A] the six direct s5-style pokes → engine dW maps + norms
      (expect: reproduce the fingerprint);
  [B] a fresh `dw_dx_multi` replay restricted to elt 19, harvest
      defaults (fex reset, central diff, delta 1e-8) → fresh columns;
  [C] the STORED S4 columns for elt 19 (center field);
  [D] attribution scan: correlate each engine dW map against ALL
      stored center-field columns — if the best match carries a
      different (iElt,dof) label, the defect is bookkeeping, not
      frames;
  [E] the triads: TElt 6×6, psi/vpt/rpt, angles of the local axes to
      the segment's radial / azimuthal / normal / parent-axis
      directions.

### R0.1 RESULT — the diagnosis is REVISED: evaluation surface, not frames

Probe (`r0_frame_probe.m` → `r0_probe_report.txt`), elt 19, round-1 deck:

- [A] the direct s5-style pokes reproduce the round-1 fingerprint
  EXACTLY (Ry 9.257e-10, Tz 4.466e-10, Rx/Ty near-null).
- [B] a fresh `dw_dx_multi` replay reproduces the STORED S4 columns at
  corr +1.0000 on every DOF — the harvest is reproducible; nothing is
  stale and the (iElt,dof) bookkeeping is correct.
- [D] the engine's Ry and Tx responses correlate **1.0000 with Seg19's
  own Tz (piston) column** — the direct-poke response is a pure
  segment-footprint piston with lever 2.07 m = Seg19's pupil radius.

Bisects (`r0_context_bisect{,2}.m`): chief-vs-mean reference, FEX,
forward-vs-central, channel-object-vs-raw-call, set_src_fov — NONE
flips it.  The ONE discriminator is the TRACE TARGET before `opd()`:

    trace(nElt-1) (ExitPupil Return, = the harvest's ox.wf_elt):
        0.1426 m/rad  — tilt about the segment center = the S4 column
    trace(nElt)   (Science focal plane, = what s5 did):
        0.9257 m/rad  — segment-footprint piston, lever 2.07 m

**s5's check, drift series, and WFC all evaluated the OPD at the
Science FOCAL plane while the Jacobian is defined at the coronagraph
exit pupil.**  The physics: at a focal plane a tilted sub-aperture at
pupil radius r shifts its whole sub-beam's OPL nearly uniformly —
piston ∝ θ·r_seg.  Everything in the round-1 fingerprint follows:

- Tz agreed at 0.6% because a segment's normal displacement adds path
  uniformly at ANY evaluation surface — piston is surface-invariant.
- The "consistent ~6.5 factor" = 0.9257/0.1426 = the focal-plane
  piston lever (2.07 m, the segment's pupil radius) over the
  intra-segment tilt lever — identical for rotations and translations,
  which is why the cross-pairing matched at 0.5%.
- The "engine x/y asymmetry" is the focal-plane projection pattern
  (piston ∝ tilt-direction · radial position), not a frame defect.
- "With all six DOFs the corrected leg came out WORSE": control solved
  Science-plane residuals with an EP Jacobian.

The round-1 NOTES' clocked-Mon/TElt hypothesis is REFUTED (the frames
agree; channel-object and raw pokes are bit-identical in every
context), and the prescribed fix ("drive through the same channel
objects") would NOT have fixed it — the channel does not own the
trace.  Dave's principle stands sharpened: the pokes ARE the model,
and "the same basis" includes the evaluation surface.

ERRATUM for round 1 (dir stays frozen; recorded here): s5's WFE
series and check table are Science-plane numbers mislabeled "at the
coronagraph exit pupil".  The contrast series (intensity chain) and
the piston-only control result are unaffected in substance.

Shared machinery audit: `run_compare.m` and `run_simulator.m` are
CLEAN — both use `wf_elt = num_elt()-1` consistently.  The trap lived
only in round-1's bespoke s5 runner (`trace(n)` with n = nElt).

### R0.2 — the fix, landed in shared machinery

`design/src/jacobian_check.m`: the engine-vs-Jacobian closure check as
a REUSABLE function whose one enforced rule is the R0.1 finding — the
engine OPD is evaluated at `ox.wf_elt`, the surface the harvest
measured at.  Null-response pokes (a segment's Rz clocking: 1.7e-13 m
per nrad on e5hex1) report `rel = NaN` per `run_compare`'s
`w_floor = 1e-12 m` convention instead of an FD-noise ratio.

Gate `tests/tJacobianCheck.m` (e5hex1, model 128, Seg2) — 2/2:
- all six DOFs close at the harvest surface (rot ~2e-5, trans ~1e-3);
- the WRONG-surface tripwire: the same check at nElt (FocalPlane) must
  FAIL on rotations while piston still closes — pinning both the
  defect class and the property that made it deceptive.

`run_compare.m` / `run_simulator.m` audited: both already use
`wf_elt = num_elt()-1` throughout — no shared-machinery defect; the
trap lived only in round-1's bespoke s5 runner.

### R0.3 — the gate: all six DOFs close on the round-1 deck

`r0_closure_check.m` → `r0_closure_report.txt`: the round-1 check
table re-run at `wf_elt = 35` against the STORED s4_sens.mat, on
Seg1 / Seg2 / Seg19 / M2 × all six DOFs (the s5 amplitudes, the s5
tol 0.05):

    worst over 24 (elt, DOF) pairs: 0.0126  [PASS]
    rotations ~3e-5, translations ~1.2e-3, piston ~1e-5
    segment Rz = null (below the 1e-12 m floor), as physics says

Round-1 reference at nElt: rel ~1.0 on active axes, ~55x on nulls,
only Tz closed.  **The S5 control basis un-shrinks from piston-only to
the full six rigid-body DOFs** — the R0 gate of the brief, met.

Consequence bound into R4: the round-2 time-series runner evaluates
WFE, the check, and the WFC solve at `art.ox.wf_elt`, making the
"rms OPD at the coronagraph exit pupil" metric tag TRUE.

R0 artifacts committed with this entry; run logs are not committed.

---

## 2026-08-26 — R1: the DM-bearing back end

Dave: "commit the CTB tail and start R0" → R0 closed → "Go!" — R1.

`e2e6m_r2_params.m` + `r1_backend.m`: the CTB topology at observatory
scale — OAP1 collimator (f 1.20 m → the 47 mm pupil) then seven f 0.90
1:1 relays with DM1/DM2 (flat folds, 60 mm clear aperture, 0.15 m
apart) at the collimated pupil, Apodizer / Lyot / Backend pupil
markers, FPM / FieldStop focus stations, Science.  Spliced with
`append_rx` onto BOTH round-1 primaries (read-only from `../e2e6m`).
The pupil is preserved at 47 mm at every pupil station, so masks and
λ/D bookkeeping stay comparable with round 1's committed numbers.

### The fold-convention lesson (shroud 10.33 → 7.451 m)

First build kept round 1's ALTERNATING fold sides and FAILED the
shroud at 10.332 m: a near-retro fold turns the chief by 180−2·AOI,
and because the beam REVERSES at each fold, alternating the geometric
side adds −2·AOI of net direction rotation per fold — over this
chain's 10 folds the accordion fanned ~120° and walked to 5.17 m
radius (position dump in the commit: Science at x −4.26 m).  Round 1
survived the same rule only because it had 5 folds.  The fix is the
opposite convention: the SAME side every fold ping-pongs the beam
between two fixed directions and packs the chain into a leg-sized
pocket.  Re-measured: **7.451 m — identical to round 1's primary-set
envelope; the DM-bearing back end costs nothing in shroud diameter.**
(Known limit, same as round 1: the sequential trace cannot see one
mirror's BODY shadowing a later leg; the gate set is ray-pass +
shroud, not a mechanical clearance model.)

### Gates (r1_seg_report.txt)

37 elements as expected; 983/985 rays (= the telescope alone); beam
radius 23.77 mm at DM1/DM2, 23.75 mm at Apodizer/Lyot/Backend,
0.87 µm spots at FPM/FieldStop/Science — every station in its
conjugate.

### The seed moves to the DM leg (r1_coro)

`r1_coro.m` = round 1's s3_coro on the new train with one deliberate
change: `prop_layout` seeds the near field on the **DM1→DM2 leg**
(CTB's convention) instead of the apodizer-entering gap, so the
complex field exists AT the DM planes — the planes the EFC layer
probes — and `ctb_aplc`'s DM1/DM2 stations are the real planes rather
than stand-ins.  Mask parameters are round 1's exactly, so R1-vs-round-1
measures the topology change alone and seg-vs-mono measures the gaps
alone.  Running.

### R1 RESULTS — the DM-bearing train, gated and scored

`r1_coro` (model 1024, nGridpts 255, round-1 mask parameters):

    seg  DZ mean 4.705e-07  median 2.002e-08  suppression 6.18e+03
    mono DZ mean 3.624e-10  median 5.255e-12  suppression 2.82e+07
    gap cost (seg/mono): 1298x mean, 3809x median
    vs round 1's seg 8.700e-07: topology factor 0.54x -- the DM-bearing
    train scores slightly BETTER open-loop; the DMs are FLAT here and
    R4 closes the loop.

Both diffraction decks verify (PSF CENTRED, chief vs geometric
4.7e-16 / 1.8e-15); the seed sits on the DM1→DM2 leg as designed
(seg station 23, mono station 5).

`r1_dm` — the DMs are REAL: `ctb_dm_rx` rewrites DM1/DM2 (elts 23/26
of the seg diffraction deck) as GridData surfaces in their own frames
(ng 256, dx 0.235 mm on the 60 mm aperture); reload gate PASS
(47 elements, 40357 rays, unchanged).  Poke gate (the e5 lesson,
evaluated at the deck's ExitPupil per R0): a single 32×32-lattice
actuator (880 active in the 47.5 mm beam) poked 20 nm at half a pupil
radius → peak ΔOPD 3.92e-8 m (2× surface = 4.0e-8 expected), 100% of
the response energy within 0.15 R_pupil of the peak, peak at
0.53 R_pupil off center.  PASS — localized, off-center, right
amplitude.  `r1_dm_poke.png`.

`r1_shroud_union` — both instruments (this coronagraph leg + round
1's imager configuration, which is optically unchanged: the pick-off
sits at 0.15 m, DM1 at 0.25 m): union 7.451 m PASS.

Artifact hygiene: `r1_coro_run.mat` PRUNES the derivable heavy arrays
(Phi / I_aplc / I_blc, 32 MB → 16 KB) before saving — the ≥20 MB
derived-binary rule round 1 bent; reports + PNGs carry the content
and the masks rebuild deterministically.

---

## 2026-08-26 — R2: the four graphics (Dave items 2, 3, 5, 6)

All four are committed PNGs, each verified by inspection AND
numerically anchored:

- **`r2_sequence.png`** (item 2) — the light-order schematic, both
  legs.  Node lists are PARSED FROM THE DECKS (r1_seg_full.in +
  round 1's s3_imager_leg.in), so the sketch cannot drift from the
  trains; only the classification (mirror/DM/mask/focus/detector) is
  local.  Layout: shared trunk + imager leg on the top row ("pick-off
  deployed"), coronagraph leg serpentining below, an elbow connector
  routed through clear space — first draft had branch lines crossing
  boxes; reworked until no line crosses anything.
- **`r2_back_plane.png`** (item 3) — the back end as a 2D FOLD-PLANE
  ELEVATION, to scale, elements named and classed.  A 6° near-normal
  accordion is nearly collinear in 3D — every 3D view renders a stick
  (tried the default iso AND the computed face-on [180° 17°]); the
  interpretable picture is the bench drawing in the plane the folds
  live in, plane found by SVD of the element positions.  The OAP1 →
  DM-pocket → W-relay structure reads at a glance; span 5.1 × 0.6 m.
- **`r2_fpm_mask.png`** (item 5) — the occulter itself: opaque disk,
  radius 2.8 λ/D = 27.3 µm at the FPM focus, transmission map with
  the λ/D axis and the physical radius stated; scales MEASURED from
  the engine at the FPM plane (R_EP 0.313 m, 9.75 µm per λ/D at
  500 nm), the coro_setup_ recipe.
- **`r2_apodizer.png`** (item 6) — the apodizer two ways: the
  clear-pupil prolate as manufactured (throughput 0.145 over the
  traced aperture), and the same profile over the TRACED 19-segment
  pupil — gaps visible — axes in mm at the apodizer plane (47 mm
  pupil).  These are the SCORING masks rebuilt with the scoring
  parameters, not lookalikes.

Plus `r2_train_iso.png` — the full-train 3D render for context (the
back end is a label-sized cluster against the 6 m PM, which is
exactly the point item 3 addresses).

---

## 2026-08-26 — R3: sensitivities + the EFC Jacobian + the MET

### The harvest (r3_sensitivities, running)

Round 1's S4 on `r1_seg_prop.in`: dwdx now carries DM1/DM2 and the 8
OAPs as rigid bodies beside the 19 segments + PM group; dwdz/dwdgrid
unchanged in shape.  NEW stage gate: `jacobian_check` (R0's shared
fix) runs on the fresh harvest at its own wf_elt, sampling a segment,
a DM and an OAP over all six DOFs — R0's rule baked into the stage
rather than trusted.

### The EFC Jacobian (r3_dm_jacobian, running — 0.46 s/poke, 1760 cols)

Engine-measured G = dE(dark zone)/d(actuator) through the FULL masked
chain, at N=512 (scales re-measured at that N; one grid for G, EFC and
R4 scoring).  Machinery is the committed CTB core pointed at this
deck: `ctb_chain` (takes 'rx'+'elt'; the prolate rides in via
`run_screened` with its own apodizer disabled) + `ctb_dm` influence
models on r1_dm's GridData surfaces.  The '_mm' fields carry metres
(deck units) — stated in the header.  Artifact gitignored +
fingerprinted (`r3_dmjac.fp.json`).

### The MET (r3_met — waiting on the harvest)

Round 1 skipped MET naming "reconciling run_met's body list with a
full-train Jacobian" as the blocker.  MEASURED, the gap dissolves:
`run_met` selects dwdx columns BY BODY and tolerates extra columns,
and the spliced deck preserves the telescope element ids (segments
1–19, hub M2 = 20).  So the committed runner consumes `r3_sens.mat`
directly, with round 1's committed `s2_segmented.in` +
`s2_segmentedHx.m` as the truss substrate.  No aft ring (stated).

### R4's architecture decision — why NOT run_simulator

`run_simulator` (the committed pipeline) plays its engine legs on the
MET/telescope deck: its per-frame OPD lives at the TELESCOPE exit
pupil, while the round-2 Jacobian lives at the CORONAGRAPH exit pupil
on the full train — pairing them would recreate the R0 defect inside
the committed runner.  R4 therefore stays bespoke ON ONE DECK (the
DM-augmented full-train diffraction deck): drift, MET-gain correction
(x̂ = dxdl·l + dxde·e from run_met's MMSE gain, measurements simulated
from the VALIDATED linear model m = [dldx;dedx]·x + noise — the s6/s7
result that the engine holds met points rigid), six-DOF control (R0),
engine WFE at ox.wf_elt, and an EFC step (real-constrained Tikhonov
normal equations, the ctb_efc idiom, the measured G) re-solved at
every contrast-scored frame of the corrected pass.  Stated bounds:
measurement simulation is the linear model, not a truss-Rx trace;
contrast at N=512 (the Jacobian grid) for series consistency.

### R3 RESULTS

**Harvest** (38.5 min, model 512, 5 fields): dwdxall 170286×192 (31
optics × 6 + PM group — DM1/DM2 and the 8 OAPs now carry rigid-body
columns), dwdzall ×152, dwdgall ×114.  The baked-in `jacobian_check`
gate: **worst rel 0.0035 over 18 (elt,DOF) pairs** (Seg2, DM1=23,
OAP4=30 × six DOFs, 4 clocking nulls) — R0's closure holds on the new
train.  269 MB gitignored behind `r3_sens.fp.json`.

**EFC Jacobian** (18.6 min): 1760 columns (2×880 active actuators),
0.62 s/poke through the full masked chain at N=512; dark zone
3–15 λ/D; column norms median 4.2e5, min 0.081, ZERO null columns.
`r3_dmjac.mat` gitignored behind `r3_dmjac.fp.json`.

**MET** (`run_met` on round 1's s2 substrate + THIS harvest): the
feared body-list integration gap dissolved as predicted — products
dxde 120×252, dxdl 120×114, dwde/dwdl emitted.  Edge+MET WEM 0.864
per unit gauge noise (MET-only 84.1 — the edges carry it, as e2e);
met_layout_opt engine-FD validation 0.00% off; Monte-Carlo 104.3 nm
vs analytic 101.7 (2.6%).  Truss figures land (114 beams, fiducials
on the M2 rim — clustered off-axis as the tilted hub dictates).

Artifact hygiene: `r3_met_run.mat` saves POINTERS only (the verbatim
`art` copy was 478 MB — round 1's exact duplication lesson);
`r3_met.mat` (475 MB, run_met's own artifact) gitignored.

---

## 2026-08-26 — R4: the closed-loop drift series (three mechanization lessons)

The final numbers (`r4_report.txt`, `r4_series.png` — model 512, the
Jacobian grid, one grid for G/EFC/scoring):

    drift:        rigid-body state 0.3 -> 3.0 nm rms over 400 s
    RBCS loop:    residual FLAT at 0.36 nm (matches the pure-linear
                  prediction 0.375 -- the engine leg is faithfully
                  linear at these amplitudes)
    open loop:    contrast 4.65e-07 -> 1.72e-06 (3.7x degradation)
    closed loop:  1.91e-07 -> 2.46e-07 -- the dark zone HELD; 7x
                  better than open at the end of the soak
    EFC dig:      4.65e-07 -> 1.90e-07 in 8 damped iterations, then
                  maintained per scored frame

Three mechanizations were tried and the first two REJECTED BY
MEASUREMENT — the narrative the backup deck should carry:

1. **Held one-shot MET correction (round 1's idiom + real noise):**
   at the round-1 wfc frame the drift (0.3 nm) sits BELOW the 1 nm
   edge noise; the estimate residual exceeded the state and the
   corrected leg ended WORSE (0.056 vs 0.013 waves).  A noise-free
   image-based one-shot was fine in round 1; a real MET is not
   noise-free.
2. **Per-frame loop on the MMSE-gain slice:** run_met's gain
   estimates the FULL body state [segs, hub]; its segment SLICE is
   NON-CONTRACTIVE (spectral radius 1.154) and the loop diverged to
   19 nm on a 3 nm drift — with the pure-linear simulation
   (`r4_loop_diag.m`) predicting the engine's divergence to three
   digits (1.91e-8 both).  The s7 doctrine ("raw estimator loops
   diverge; weighted-LS/BLUE per Tesch") re-derived by measurement.
3. **BLUE + ridge on the segment state, gain 0.5:** spectral radius
   0.9998; converges to the 0.36 nm noise floor.  SHIPPED.

And a fourth, for the EFC: undamped re-solves against the
amplitude-dominated gap speckle DIVERGE (1.8e-7 → 2.9e-6 over 9
re-solves) — the 0.15 m DM spacing gives weak Talbot authority
(z/z_T ≈ 0.4% at 15 λ/D), so ridge solves push strokes along
near-null directions.  Damping (γ=0.7) + a leaky integrator (µ=0.02)
bound the null-space accumulation: the dig converges monotonically
and HOLDS.  The DM strokes' deliberate pupil shaping appears as
closed-loop "WFE" above the uncorrected line — labeled as such in
the payoff figure, middle panel (EFC trades pupil flatness for
contrast; the contrast panel is the payoff).  DM spacing as an
amplitude-authority KNOB (CTB uses 0.5 m in a 21 mm beam) is a
stated future trade, not re-run here.

---

## 2026-08-26 — R5: the deck

`deck_e2e6m_r2.py` + `e2e6m_r2_records.py` → `deck_e2e6m_r2.{md,pptx,pdf}`,
**21 slides** (13 main path, 8 backup behind one plain divider),
DECK_STYLE-governed.  The round-1 telescope/segmentation slides carry
forward through round 1's own frozen records module; every round-2
number parses from the committed r0–r4 reports (two provenance fixes
en route: `r2_masks_fig` now writes `r2_masks_report.txt` so the
occulter's 27.3 µm is parsed, and the R0 lever ratio parses from
`r0_bisect2_report.txt` instead of a constant).  Main path: telescope →
segmented primary → light order (full-width sequence) → two
instruments/shroud → bench plane → masks as objects → DMs are real →
gap cost → error budget closes (worst 0.35%) → metrology truss → the
loop closes → what this demonstrates.  Backup: the R0
evaluation-surface resolution (with the same-element two-surface
table), the control-law selection story, the fold-side packaging
lesson, the round-1 f/# and apodizer carries, what is not in the
model, and how to reproduce.

Visual QA: rendered every slide (libreoffice→pdftoppm contact sheet),
reworked the sequence slide to full width after the first render
buried it in a half column.  STYLE_REPORTS §5 gate: run — **clean**.
DRAFT — Dave signs off; the .pptx is generated, never hand-edited.

---

# Coronagraph-family campaign (CF stages, `macos/BRIEF_e2e6m_coro_families.md`)

## 2026-08-26 — CF0: the config-driven chain runner, gated bit-consistent vs R1

`cf_chain.m` = r1_coro's scoring walk lifted into the ctb_chain pattern:
load once, engine-measured scales once, masks built once (all 8x
supersampled/binned, bench_ctb primitives IMPORTED), `run()` = fresh
trace + masks in place + complex field at the Science plane.  Config
recorded on the struct (`ch.config`, the ctb_jac_check vocabulary) and a
filesystem tag derived from it (`ch.tag`), so families never collide on
disk.  Apodizer kinds: none / clear-disc prolate (R1) / **prolate over
the TRACED gapped pupil** (`ctb_apod_prolate 'support'` -- the
aperture-matched APLC of N'Diaye, Zimmerman & Soummer 2016; support =
`Iap > 0.02*max`, the r2_masks_fig recipe) / supplied.  FPM kinds:
hard / vortex (8x complex-binned) / band-limited (K&T order 4/8).

**The stop-plane lesson (cost one bisect round).**  First build stopped
the walk at the elements NAMED DM1/DM2 (23/26) and missed R1 by 1.9e-8
relative (median 3.5e-7) -- deterministic, load-state-independent
(measured: a fresh `init+load` before the run changes NOTHING; every
intensity read-and-continue perturbs the field at the 1e-10 level, so
the STOP SET is part of the measurement).  R1's walk stops at the
near-field SEED PLANES `Prop23_start/end` (24/25, the planes prop_layout
put ON the DM1->DM2 leg), not at the named DMs.  `cf_chain` now maps its
DM1/DM2 stations to the seed planes (fallback: the named elements) and
carries the named grid elements separately as `ch.dm_elt` for the
control layer; `run_bare` is ctb_aplc's bare walk exactly (one DM1 stop,
then the FPA).

**CF0 gates (`cf0_gates.m` -> `cf0_report.txt`): 10 PASS / 0 FAIL.**
- bare PSF: peak pixel (513,513) = DC both decks; peak_bare rel 0 and
  lambda/D rel 0 vs the committed R1 values; |E|^2 == intensity read to
  2.7e-16.
- pupil sanity: r 127.1/127.2 px; support fill 0.7948 seg (the 25 mm
  gaps + hex corners must show), 0.9884 mono.
- R1 reproduced THROUGH the runner: seg DZ mean 4.704624e-07 rel
  2.25e-16, median rel 1.65e-16; mono mean 3.623558e-10 rel 0, median
  rel 1.54e-16; apodizer throughput rel 0 both.  Bit-consistent.
