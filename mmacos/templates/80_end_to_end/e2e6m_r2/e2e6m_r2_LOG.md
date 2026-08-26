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
