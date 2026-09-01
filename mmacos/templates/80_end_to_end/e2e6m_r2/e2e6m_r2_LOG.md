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

## 2026-08-26 — CF1: the families, head-to-head (PRE-CONTROL) — and the APLC surprise, attributed

`cf1_families.m` -> `cf1_report.txt` + `cf1_families.png` /
`cf1_radial.png` (both PRE-CONTROL-labeled, the slide-5 rule).  Model
1024, band center, annulus 3-15 lambda/D, one runner (cf_chain), one
normalisation.  The table:

| family | DZ mean | DZ median | thru | note |
|---|---|---|---|---|
| classical Lyot | 1.625e-06 | 5.131e-07 | 25.0% | hard 2.8 + Lyot 0.50, no apodizer |
| apodized Lyot (R1) | 4.705e-07 | 2.002e-08 | 10.0% | clear-disc prolate (= R1, rel 7e-7 thread) |
| APLC (ap.-matched) | 5.380e-07 | 5.605e-08 | 8.9% | prolate ON the traced gapped pupil |
| band-limited 4th | 1.691e-06 | 3.393e-07 | 13.0% | K&T eps 0.40, Lyot 0.60 |
| vortex chg 4 | 1.818e-05 | 3.219e-06 | 36.0% | gap leak MEASURED: 5 decades off the CTB clear-pupil 8.8e-11 |
| vortex chg 6 | 2.785e-05 | 5.988e-06 | 36.0% | c6 shallower than c4 (as CTB) |
| hybrid Lyot | — | — | — | DEFERRED: FALCO co-design (SESSION-6 ruling), no fabricated row |

**The surprise, attributed before building on it (the brief expected the
aperture-matched APLC to lead; it does NOT):** the segmented-support
prolate CONVERGED (lambda0 0.957346, 758 iterations — not a numerics
artifact) and the eigenvalue IS the attribution.  lambda0 is the energy
fraction the occulter passes for the aperture's dominant prolate:
0.999994 on the clear disc, 0.957 on the gapped aperture — the 25 mm
segment gaps diffract ~4% of any support-confined apodizer's energy
OUTSIDE the 2.8 lambda/D occulter, and the ideal-APLC residual
(1-lambda0)*Phi grows by ~4 decades.  Physically: the segment grid
(pitch ~1.24 m, D/pitch ~ 4.8) puts its diffraction orders at ~5
lambda/D multiples — INSIDE the 3-15 annulus, visible as the blob
lattice in every thumbnail — and those orders are AMPLITUDE structure
no entrance apodizer confined to the pupil support can null.  Both
prolate rows therefore sit on the same gap floor (4.7 vs 5.4e-07, 14%
apart), and the ranking of amplitude-mask families COMPRESSES on a
gapped pupil.  The gap floor is EFC's job — S2 measures what the DMs
recover (Talbot-limited amplitude authority, z/z_T ~ 0.4%).  Round 1's
S3b LP apodizer (the true N'Diaye-class co-design machine) told the
same story from the other side: its engine gate failed at ~5x on this
train (recorded, not relaxed).

The vortex rows are the measured version of the brief's expectation:
charge 4 leaks 1.8e-05 — five decades off its CTB clear-pupil analog
(8.8e-11 at the same Lyot 0.60) — pure segment-gap leak on this
UNOBSCURED train (no secondary; the brief's "gaps + secondary" is half
right here, stated in the report).  Vortex thumbnails carry the
hexagonal star of the gap lattice through the vortex null.

## 2026-08-26 — CF0b: the circular stop (brief amendment, Dave), gated

**Design change (S0b):** on segmented hex apertures the coronagraph's
aperture is a CIRCULAR STOP at the apodizer plane; the telescope's hex
envelope is upstream optics, not the coronagraph pupil.  `cf_chain`
gains `circ_stop_frac` (P.cf default 0.98 x the hex pupil's INSCRIBED
radius): the stop is folded into the apodizer plane of EVERY pass —
bare included, so contrast normalizes to the CIRCULARIZED peak — the
FPM/FPA lambda/D scales and the geometric Lyot radius are RE-MEASURED
through the stop, the prolates are designed over the circularized
pupil ('apl': the circular disc; 'aplc': the circularized gapped
support), the tag gains `_c098`, and the config stamp gains the knob.
The inscribed radius is measured from the CONVEX HULL of the traced
support (min center-to-edge distance) — a polar max-radius scan would
underestimate along the azimuthal gap lines.

**The stamp-guard lesson (supervisor's check, confirmed by measurement):**
`ctb_jac_check` PASSES a pre-stop Jacobian cache against a stop request
— its compare-what-both-have contract skips keys the cached stamp
lacks (deliberate, for CTB's legacy caches).  The campaign therefore
carries a STRICT complement (`cf_efc_lib.stamp_parity`): every
requested config key must exist in the cached stamp; a missing key =
a stale GENERATION and the load refuses loudly.  `cf0b` demonstrates
all three behaviors on the real `cf2_G_hard.mat`: ctb_jac_check passes
it (the documented gap), strict parity refuses it, and it passes
against its own config.

**Sequencing:** cf2 was paused at the family boundary after four
no-stop families completed; their results are PRESERVED as the
"what the circular stop buys" comparison record
(`cf1_nostop_run.mat` + `cf2_nostop_{hard,apl,aplc,blc}_run.mat`):
hard 1.637e-06 -> 5.18e-07 fixed -> 3.18e-07 relin; apl 4.74e-07 ->
1.76e-07 -> 1.29e-07; aplc 5.49e-07 -> 2.88e-07 -> 2.38e-07; blc
1.68e-06 -> 1.43e-06 -> 1.31e-06.  Dark-zone SYMMETRIC fraction stays
0.96-0.99 through control in every family — the amplitude-dominated
gap speckle the Talbot-weak DM pair cannot reach (the S2/S3 readout).

**CF0b gates (`cf0b_report.txt`):**
- the no-stop path is byte-unchanged: cf0 re-ran 10/10 with the stop
  machinery in the file;
- stop sanity, and a geometry CORRECTION en route: the 19-segment
  tiling envelope is ROUNDER than a pure hexagon — the outer ring
  scallops the corners — inscribed/corner 0.9046 (pure hex: 0.866),
  so the stop keeps 0.9202 of the collecting area (the pure-hex
  pi/(2*sqrt(3))*0.98^2 = 0.871 bound does not apply).  The stopped
  bare peak confirms it coherently: peak ratio 0.8488 vs area^2 =
  0.8467.  First gate bounds assumed the pure hexagon and correctly
  FAILED against the better measured geometry; bounds now pin the
  tiling envelope.
- the linear-achievable floor machinery (`cf_efc_lib.linfloor`,
  closed-form rank curve from each cached G's SVD) + the measured S2
  attribution (floor vs linear-achievable at the 50 nm stroke bound)
  land with this commit; the S4 memo lambda-correctness gate is baked
  into cf4's band leg.

## 2026-08-26 — CF1b: S1 re-measured under the stop; two attributions closed by measurement

**The stopped S1 table (primary; no-stop columns preserved as "what the
circular stop buys"):** classical Lyot 3.497e-06 (0.46x -- the stop
COSTS the bare-occulter family: a circular edge paints a uniform
diffraction ring through the annulus where the hex edge concentrated
its energy in six azimuthal spikes, and there is no apodizer to soften
it); apodized Lyot 4.375e-07 (1.08x, its prolate now designed on the
true circular pupil, converged 2558); band-limited 1.645e-06 (1.03x,
neutral); vortex c4 1.641e-05 (1.11x) / c6 1.869e-05 (1.49x) -- the
DECOMPOSITION the amendment asked for: hex-edge leakage is 10-33% of
the vortex total; the remaining 1.6-1.9e-05 is GAP leak, which no
pupil-edge fix touches.

**The stopped aperture-matched APLC is SOLVER-LIMITED, not physics:**
its prolate hit the iteration cap unconverged (5000: lambda0 0.958867,
dz 2.179e-06; probe at 20000: lambda0 0.959417, dz 4.095e-06 -- the
answer MOVES WITH THE CAP, so it is not an eigenfunction and not a
design).  The no-stop seg-support prolate converged in 758 iterations;
the circularized support restores the tiling's near-degenerate mode
pairs and simple power iteration cannot separate them.  Recorded per
the SESSION-6 HLC precedent: the row stands with the unconverged flag,
the apodized-Lyot row is the prolate family's stopped representative,
and the real machines for an aperture-matched stopped design -- a
block/Lanczos eigensolver in ctb_apod_prolate, or the N'Diaye LP
co-design -- are DEFERRED, named, and not faked.  (`cf1b_probe.m`.)

**The no-stop closed-loop record is LINEAR-OPTIMAL at achieved strokes
(`cf2b_linfloor_nostop.m`):** hard 3.18e-07 vs lin-ach 2.67e-07 at its
achieved 10.1 nm (1.19x); apl 1.29e-07 vs 1.19e-07 @ 8.9 nm (1.08x);
aplc 2.38e-07 vs 2.22e-07 @ 4.5 nm (1.07x); blc 1.31e-06 vs 7.85e-07
@ 3.6 nm (1.67x).  ALL within the ~2x criterion: the floors are the
SUBSTRATE speaking -- amplitude-dominated gap speckle (symmetric
fraction 0.96-0.99) against Talbot-weak amplitude authority -- and the
BLC's one-iteration "stall" was the loop already AT its linear optimum.
The 50 nm-bound curve values (e.g. hard 4.6e-10) are the linear model's
claim at strokes where 2 nm pokes extrapolate to 1.3 rad of phase --
NOT physics; the measured line search self-limits at 3.6-10.1 nm, i.e.
it finds the linearity boundary by itself.  The attribution in cf2 now
reads the rank curve AT THE ACHIEVED STROKE (the CTB "4.5e-9 at 11 nm"
pattern).  Consequence for S3: DM spacing is the knob that moves the
linear-achievable curve itself.

## 2026-08-26 — CF1c: the hard family's stop penalty, attributed by measurement

CCL's flag: hard went 1.64e-6 (no stop) -> 3.50e-6 (stop), and the
circularized-peak renormalization looked like only part of the 2.1x.
Measured decomposition (`cf1c_stop_attrib.m`): the stop applied as a
SCREEN on the no-stop chain holds every mask and scale fixed, so
E_rim = E_hex - E_screened is exactly the blocked rim's field, with two
free pins (screened bare peak == stopped bare peak to 0.0e0; both S1
records reproduced to <1%).

**The factorization: 2.16x = [fixed-mask stop edge 2.16] x
[scale-rechain 1.00].**  The lambda/D re-measure through the stop
contributes NOTHING -- the whole penalty is the stop edge itself at
fixed masks.  Inside it: peak renormalization 1.178x (CCL's ~1.3
estimate, measured), dark-zone ENERGY up 1.83x.

**The presumption OVERTURNED: the edge does not paint its own ring --
it interferes with the gap field.**  Babinet on the dark zone:
I_stX 1.996e-5 = I_ns 1.090e-5 + rim 2.53e-6 + cross 6.54e-6
(closure 1e-15).  The rim's own energy is only ~28% of the increase;
the CROSS term (coherent interference between the rim field and the
pre-existing gap-dominated speckle) is 2.6x larger.  Consequence
worth carrying: the stop-edge penalty is not additive light to be
blocked -- it is a coherent modification of the gap speckle, which is
exactly the kind of term EFC can address (S2's stopped-hard floors
are the test).  Radial profile: `cf1c_stop_attrib.png`.

## 2026-08-27 — CF2: EFC floors per family (S2) — every family substrate-limited

**The table (N=512, closed-loop annulus 3–15 λ/D; static → fixed-G →
relin | lin-ach@achieved-stroke | attribution):**
- classical Lyot  3.503e-6 → 7.921e-7 → 3.935e-7 | 2.13e-7 | la 1.8x, sym 0.91
- apodized Lyot   4.485e-7 → 1.623e-7 → 1.081e-7 | 1.09e-7 | la 1.0x, sym 0.96
- APLC (flagged)  1.074e-6 → 9.176e-7 → 7.519e-7 | 7.54e-7 | la 1.0x, sym 0.59
- band-limited    1.636e-6 → 1.347e-6 → 1.257e-6 | 7.24e-7 | la 1.7x, sym 0.98
- vortex c4       1.625e-5 → 1.345e-5 → 5.377e-6 | 5.49e-6 | la 1.0x, sym 0.98
- vortex c6       1.844e-5 → 1.440e-5 → 3.925e-6 | 4.66e-6 | la 0.8x, sym 0.97

ALL six within ~2x of linear-achievable: substrate-limited across the
board, none control-limited.  z/z_T(15 λD) = 3.73e-3 — the S3 DM-
spacing sweep is THE knob.  Readings: (1) the stopped hard family digs
8.9x where its no-stop self dug 5.2x — the loop claws back most of the
CF1c coherent edge term (final 3.94e-7 vs no-stop 3.18e-7, 1.24x apart
despite the 2.16x static penalty); (2) apodized Lyot is the campaign
floor, 1.08e-7, BETTER than its no-stop self (1.29e-7); (3) the
vortices are linear-optimal — relin pays 2.5–3.7x where fixed-G pays
1.2x (large gap-chasing strokes hit the linearity boundary early), and
the residual 4–5e-6 gap leak is uncontrollable at this Talbot
authority; (4) aplc's sym 0.59 is the odd one out (substrate- not
amplitude-limited), consistent with its CF1b solver-limited prolate
injecting its own structure — the flag stands.

**Restart mechanics (the resume that produced this table):** the
2026-08-26 restart killed the run mid-apl; two fixes shipped —
(a) cf_efc_lib's cache-reload a0 assert compared an Nx2 MATRIX
(row-vector max → "condition must be scalar" on EVERY reload; latent,
never exercised in-process) — flattened with a shape guard;
(b) cf2_efc now adopts the dug commands from an existing _r1 cache
(its stored a0 is the authority; line searches are not
bit-deterministic across restarts), re-measuring the two endpoints —
the 'resumed' flag marks affected records (none in this final table:
apl re-ran from its cached forward G cleanly).

## 2026-08-27 — CF3a: the Lyot trade (static) + the DRAFT deck slides

**The sweep (7 fractions x 3 legs, statics under the stop, 8.4 min):**
the vortex legs rise monotonically as the Lyot opens (v4 9.4e-6 @ 23%
thru -> 1.09e-4 @ 88%; v6 similar) -- the CTB dial's shape, with the
depth difference fully explained by gap leak dominating this pupil.
THE FINDING: **the apodized-Lyot leg's contrast is FLAT across the
entire dial** (4.25e-7 -> 4.37e-7) while throughput rises 2.8% ->
10.8% -- the prolate does all the suppression and the Lyot is nearly
free throughput on this train; the S1 operating point (L=0.90) leaves
~19% relative throughput on the table vs L=0.98 at zero contrast
cost.  Pin: the supplied-mask path reproduces the S1 point exactly
(4.375e-7 at L=0.90).  No stop-rule triggers.

**Deck: three CF main slides + one backup drafted into the generator**
(records-parsed from cf1/cf1c/cf2/cf3a reports -- new record fns in
e2e6m_r2_records.py; loud on any miss), rebuilt 25 slides, kicker
carries the DRAFT marker: numbers await Dave's morning review.
Noted for finalization: two derived ratios (8.9x/5.2x dig ratios,
2.16x title) are literals pending a records home; the N=512 vs N=1024
APLC static difference (1.07e-6 vs 2.18e-6) is the cap-limited
prolate moving with N -- one more face of the standing flag.

## 2026-08-27 — phase 2 (BRIEF_e2e6m_cf_s3b_on): R directives + S3b launch

R1 (BLC stall probe, cf2r1_blc.m): THE STALL IS REAL — 13 alphas over
1e-7..1e-1, cap 40: 1.636e-6 -> 1.294e-6 in 2 iters (CF2 fixed-G was
1.347e-6), strokes 1.3/1.5 nm, and the accepted steps are the MOST
DAMPED on the ladder (3e-2, 1e-1) — the model-mismatch signature: the
BLC's residual couples to the DMs more weakly than its G's top modes
suggest.  The la-1.74x gap is bound looseness, not unexploited
margin; "all six substrate-limited" stands.

R2/R4: cf2_report.txt ADDENDUM block (regeneration-safe) + deck
language — "APLC-as-implemented ... family DEFERRED" with the named
machines; lin-ach = rank-curve at achieved stroke, a scale not a
strict inequality (v6's 0.8x reads consistently).

R3 (cf2r3_apl98.m) + S3b (cf3b_spacing.m) LAUNCHED.  ** RESUME NOTE
(power-cycle risk flagged by Dave): if either dies, relaunch the same
driver — every Jacobian is a stamped tag-separated cache (cf2_G_*),
the emitted spacing decks (r1_seg_dNNN_*) survive on disk, and
cf2_efc's r1-cache resume pattern applies: a family with _r1 but no
run state adopts the cached a0.  Launch lines:
  matlab -batch "cf2r3_apl98"     (in e2e6m_r2/, MACOS_HOME set)
  matlab -batch "cf3b_spacing"
S3b spacings 0.15(=CF2 baseline)/0.40/0.70/1.10 m, apl leg all four +
v4 at 0.15/0.70, Lyot held at the CF2 point (isolates spacing). **

## 2026-08-27 — R3 + R3b: the L=0.98 closed loop — static-free is not loop-free

R3 (CF2 protocol at L=0.98): 4.486e-7 -> 1.595e-7 -> 1.595e-7 (relin
flat) at 3.3 nm strokes; lin-ach at that stroke 1.79e-9 — an 89x gap,
the control-limited signature.  R3b (the R1 treatment: 13 alphas,
cap 40): does NOT dig — 1.578e-7 in 2 iters.  So the stall is REAL
(both probes independent), and the la gap is bound optimism at tiny
strokes: the open Lyot's G carries deep top modes the measured chain
cannot follow — the BLC substrate class (the extra light the open
Lyot passes is gap-structured and model-mismatched).

**Supervisor reading for the operating point (overrides the driver's
coded 1.5x "adopt"):** closed loop, L=0.98 costs 1.46x in contrast
(1.578e-7 vs 1.081e-7) for 1.19x in throughput (11.2% vs 9.5%) — a
NET LOSS in a throughput/contrast merit (7.1e5 vs 8.8e5).  CF3a's
"free throughput" was a STATIC statement; the closed-loop dial has a
cost, and L=0.90 dominates.  Recommendation for the gate: keep
L=0.90; an optional 0.94 probe (one Jacobian) would locate the knee
exactly but cannot beat the L=0.90 point already in hand.

## 2026-08-27 — S3b + CF3c + the gate: the knee is the baseline

CF3b (149 min): apl floors 1.081e-7 / 4.977e-7 / 9.670e-7 / 9.005e-7
at 0.15/0.40/0.70/1.10 m — KNEE AT 0.15; v4 cross-checks (5.38e-6 →
8.97e-6 at 0.70).  The Talbot expectation INVERTS: lin-ach deepens
with spacing (6e-11 claimed at 1.10 m) while measured floors worsen —
bound optimism grows, the loop cannot take the authority.  Shroud
7.451 m at every spacing (measured; not the discriminator).
CF3c attribution: static cost is the SAME amplitude-type gap-Fresnel
family (sym fraction 0.986 at EVERY spacing; broadband radial ratio,
median 2.74) evolved over the longer DM1->DM2 leg.  Authority is not
the binding constraint; the static gap-Fresnel level is.
GATE PROPOSAL (cf_gate_proposal.txt): apl, L=0.90, d=0.15 m — the CF2
configuration exactly; no bench change.  S4/S5 follow on Dave's
confirmation.

## 2026-08-27 — S4: the physics layers at the gate point (apl, L=0.90, 0.15 m)

Two stub fixes en route (column-vector coating loop; single-block
Jacobian wrapped with rowoff for efc_multi_).  Results (324 min):
- POL: 31 reflectors coated (MgF2/Al), Jones screens at the exit
  pupil, complex-mean normalized.  Co-pol identity deviation 9.4-9.6%
  (CTB: <0.4% — 31 folds at 6 deg carry a real imprint) yet almost
  entirely COMMON-MODE: static with screens 4.473e-7 (unscreened
  4.485e-7), floor 1.604e-7 == the unscreened fixed-G floor, and the
  UNCONTROLLABLE pol floor is 4.4e-13.  Same verdict as the CTB at 3x
  the mirror count: polarization does not set this train's floor.
- BAND (9-color superset G, per-band block subsets): floors
  1.623e-7 / 1.639e-7 / 1.716e-7 / 2.066e-7 at 0/5/10/20% — 1.27x
  across the full 20% band.  The chromatic penalty is a whisper: the
  floor is gap-speckle-owned, not chromaticity-owned (the inverse of
  the CTB vortex).
- BAND+POL at L=0.90 (rebal pinned to the gate point per R3b; the
  driver's 0.70 default predates CF3a): floor 1.718e-7 == the
  band-only 10% floor to 3 digits; pol floor 4.8e-13.  The layers do
  not interact.
- MEMO GATE (the brief's S4 requirement, cf4_memo_gate.m): memo-hit
  revisit reproduces the cold-build field to 0.0; cross-lambda 2.5e-2
  — the set_lambda memoization is lambda-correct, observed end to end.
Driver note for finalization: when rebal_lyot == the operating point,
the bandpol leg should SUBSET the superset G instead of re-measuring
(1.8 h redundant tonight; redundant, not wrong — deterministic).

## 2026-08-27 — S5: the drift series toned down 10x — the hold is the mechanization's

TWO-TERM LESSON first: the history = random walk + a correlated LINEAR
drift ramp (drift_trans/drift_rot).  The first 0.3 nm attempt scaled
only the walk and the series barely moved (|x| 2.25 vs 2.95 nm) — the
ramp dominated.  "Tone the series down 10x" must scale BOTH terms;
recorded here so the next sweep does not rediscover it.

The corrected series (all four knobs /10; |x| rms 2.95e-10, exactly
10x down; R4's 3 nm artifacts preserved as r4_*_3nm):
- OPEN loop: FLAT — 4.646e-7 -> 4.635e-7; 0.3 nm-class drift is
  invisible over the 41-frame soak.
- CLOSED loop: 1.910e-7 -> 2.464e-7 — IDENTICAL to the 3 nm hold
  (2.462e-7).  The ~2.5e-7 hold level is the MECHANIZATION's floor
  (sensor-noise injection through the gain-0.5 RBCS loop + damped
  EFC), not drift-driven: at this drift the loop's own noise
  injection costs more than the drift does.  Estimator tracks
  cleanly (|x+u| 7.2e-11 vs drift 2.95e-10).  Per note-don't-retune:
  control CADENCE/GAIN is the flagged knob at low drift, not retuned.
- Closed-loop WFE 0.0359 waves is the EFC's deliberate DM shaping
  (contrast-optimal, not WFE-optimal) — not divergence.

## 2026-08-27 — S6: pins

tCfCampaign (tests/, SUITE_CTB_512) 6/6: the CF2 table + la ratios,
the S3b knee at the baseline, the gate operating point, the toned
S5 series, the winner static LIVE through the chain, the lambda-memo
gate LIVE.  Regeneration drift is now loud.

## 2026-08-27 — S6 close: the deck at 27 slides (DRAFT)

Records fns cf3b/cf4/cf5 added (loud-on-miss); r4() re-pointed at the
PRESERVED 3 nm report (the slide's narrative), cf5() parses the toned
series.  Two new main slides — "Both dials lose to the operating
point" (the gate + Lyot stall + spacing inversion, cf3b figure) and
"The physics layers at the operating point" (pol/band/bandpol, cf4
figure) — and the loop slide gains the toned-series finding.  §5
style gate: run, one fix (a doubled word in the gate bullet); the
kicker still carries the DRAFT marker — Dave's sign-off pass pending.

## 2026-08-31 — CF3d: the restart ladder digs d=1.10 three decades (Dave's recipe, measured)

Dave's ask on the R4/S5 slide: "most solutions use multiple EFC
resets... rinse and repeat. Can we try that?" — then, mid-run: "keep
going to try to get below 5e-11 — that's HWO target level.  But no
more than 9 hours more."  CF3d is that ladder, on the spacing study's
best linear substrate: the d=1.10 m two-DM apl leg (L=0.90, c098,
cf3b's decks + cached round-1 G), each round = relinearize about the
dug state, then EFC with a widened Tikhonov schedule.

RESULT (cf3d_report.txt, cf3d_run.mat, cf3d_dig.png): **1.215e-6 →
1.133e-9 — three decades, >1000x, in 10 digging rounds / 4.6 h of
unattended engine time**, DM strokes ending at 32.8 nm rms of the
50 nm bound.  Phase 1 (niter 20, alphas 1e-8..1e-2) walked 8 rounds
to 3.666e-9 in 2.9 h; the extension (Dave's 5e-11 target armed,
budget 9 h) took rounds 9–10 to 1.133e-9; rounds 11–13 accepted NO
step at any alpha down to 1e-10 with niter 30 — the plateau is
proven, not assumed (two fresh G measurements at the unchanged state
agree to 4 digits: la(G) 3.521e-11 both times).

ATTRIBUTION — the plateau is the CONTROLLER, still: the relinearized
substrate holds la(G) = 2.0–3.8e-11 at every dug state, so ~1.5
decades to the HWO-class 5e-11 remain priced-in and unclaimed.  What
the ladder removed was the single-linearization stall (9.0e-7 at this
spacing in cf3b — four decades above bound); what remains is the
monotone full-Tikhonov step rule itself (steps that must never
increase contrast, full-basis, single alpha per round).  This is the
measured case for FALCO-grade step machinery (beta-bumping,
constrained strokes, per-mode regularization) on this same model —
queued post-demo.

BOUND-OPTIMISM TRAJECTORY, now measured rather than argued: la(G)
claims 2.2e-13 at the FLAT state (round 1) and 8.8e-13 at round 2 —
fantasy numbers, far below anything reachable — then settles at
2–4e-11 once the state is dug and the linearization is honest.
CF3b's "bound optimism grows with spacing" lesson is the same
phenomenon; the ladder shows it as a trajectory within one config.

Mechanization lessons (write-at-resolution): (1) `lib.efc` returns
EMPTY alpha when the line search accepts nothing — the first plateau
crashed the driver at `alph(end)`; guarded (log NO ACCEPTED STEP,
alpha=NaN) so a stall is a recorded observation, not a crash.
(2) The resume path resets r0, so the two-stall convergence rule
costs two fresh ~23-min G measurements at an unchanged state before
it fires; correct but slow — a resumed run could seed `stall` from
the checkpoint's trailing no-step rounds.  (3) The r11 G cache
survived the crash, so its round cost 0.2 min — the fp.json-keyed
Jacobian caching pays off exactly here.

Deck fold (Dave pre-authorized "redo the e2e6m_r2 reports and the
deck_keysight* decks"): new slide after the drift-series slide —
"Digging the dark hole: the overnight restart ladder" (cf3d_dig.png
as fig_cf3d_ladder.png); the drift slide's honest-gap bullet now
scopes its 2.5e-7 hold as the loop mechanization's, pointing forward;
the closing slide's restart-ladder claim quantified 1.2e-6 → 1.1e-9.
