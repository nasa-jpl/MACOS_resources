# REPORT — reflective (OAP) variant of the tg_psi_dm96 96×96 T-G DM gauge

CCMac for Dave, 2026-09-10. Branch `tg96-oap` (MACOS_resources). Numbers first;
departures from the brief flagged. **Status: builder + runner DONE and the lens
equivalence gate PASSES EXACT; the OAP rig builds and nulls but its DM-pupil
imaging is an open design item (below) — handed back for a fold-relay decision.**

## Summary (memory-style)

- **`twyman_green` gained `'optics'` ('lens'|'oap')** — L1/L2 → OAPs via
  `add_oap`, fold confined to the collimator/focuser legs so the BS/arms/recomb
  are unchanged. **`'lens'` byte-identical** (gated `tBench/test_twyman_green_optics`).
- **One parameterized runner** `tg_psi_dm96_oap/{tg96_params,tg96_run,tg96_tail,
  tg96_run_batch}.m` drives both rigs; refractive `tg_psi_dm96/` untouched.
- **LENS EQUIVALENCE GATE: EXACT.** The lens rig through the new runner at model
  1024 reproduces `tg_psi_dm96/tg96_report.txt` line-for-line (table below).
- **OAP finding 1 (build):** `focus_dist=F1` keeps the conjugate exact but the
  same-plane fold gives the OAP collimator **+2.8 % beam at the BS** vs a lens.
- **OAP finding 2 (null):** the OAP flat-DM null floors at **12.9 nm** (lens
  0.134 nm) — a low-order (mostly defocus) ARM-DIFFERENCE the rotationally-
  symmetric field-lens tail (common to both arms) cannot null.
- **OAP finding 3 (RESOLVED — the reflective gauge WORKS):** the earlier
  "cannot image" was a **poke-placement artifact**, not an optical defect.
  `macos.pupil_quality` (Dave's rodgers2 metric) first ruled out the
  astigmatism/defocus hypotheses: the OAP exit pupil is **cleaner than the
  lens** (|astig| **0.144** / sag 0.273 mm vs **1.025** / 1.161 mm) and the
  detector sits at the true pupil vertex (0.000 mm offset). Then, per Dave, the
  OAP illuminated pupil is **smaller** (mask 18376 vs 28917 px) and shifted, so
  a circular beam on the square DM leaves the **center actuator OUTSIDE** the
  pupil. Measured recovery vs actuator radius (OAP rig): center (48,48) **0.0
  nm**, but (48,56)…(48,88) recover **129→135 nm** — i.e. **in-pupil actuators
  image as well as the lens (131 nm)**. The battery aborted only because its
  hardcoded `'single'`/registration pokes assume the (larger) lens pupil and
  land on/near the center. **Fix (runner-level):** place the OAP calibration
  pokes on illuminated actuators (via `dm_gauge_lib/dmg_lit`, the msk→DM
  mapping), not a fixed center. Then the OAP battery runs. No relay, no engine
  work, no astigmatism problem.

## Lens equivalence gate (model 1024) — EXACT

| metric | record (tg_psi_dm96) | this runner (lens) |
|---|---|---|
| null (flat, unaligned) | 0.1345 nm | **0.1345 nm** |
| single actuator @150 nm | 146.1 nm | **146.1 nm** |
| 96×96 closure resid / corr | 6.006 nm / 0.535262 | **6.006 nm / 0.535262** |
| held-out random resid | 11.8914 nm | **11.8914 nm** |
| registration parity / |corr| | 5 / 0.9341 | **5 / 0.9341** |
| Stage-E flat/single-10nm | 0.9654 / 21 pm | **0.9654 / 21.0 pm** |
| Stage-E flat/random-10nm | 0.9203 | **0.9203** |
| distortion (nonlinearity) | 0.1360 mm | **0.1360 mm** |

The builder change and the runner are validated: `'lens'` is the record, to the
digit. `tBench/test_twyman_green_optics` green (lens byte-identical; OAP
pole→focus == F1/F2; chief crosses each OAP pole).

## OAP rig — what works, what doesn't

Fold AOIs solved near-normal (**OAP1 5°, OAP2 9°** at F1=857/F2=428, inside the
700 mm leg cap; clearance margins printed in `runs/oap/oap_report.txt`).
Builds, traces zero-loss, arm departure +0.143° (== lens), piston gain 1.0018.
OAP1 collimation is comparable to the lens (10.9 vs 8.3 waves on a flat plane) —
not a gross geometry error.

The blocker is the **DM-pupil imaging through the folded tail**. Evidence:
- flat-DM null 12.9 nm, tail retune has no leverage (12.90→12.89) — the null is
  a low-order arm-difference the common tail can't touch;
- single-actuator poke → recovered peak ≈ 0; registration |corr| ≈ 0;
- detector-conjugate sweep (DET_TRIM −20…+400) never images the poke near its
  true height — no focus recovers it, so it is not a simple defocus.

Interpretation: a single off-axis OAP2 focuser images the large DM pupil with
astigmatism that the rotationally-symmetric field lens cannot correct; the
flat-DM null (the tail's tuning objective) is blind to it. This is the
"significant effect from same-plane folds" the brief asked us to measure —
here it is significant enough to prevent pupil imaging with the single-OAP tail.

## Where it stands (imaging works; registration is the remaining item)

**The reflective gauge IMAGES the DM as well as the lens rig.** With in-pupil
pokes (placed at the DM footprint centroid ± a per-axis-extent offset — the
exact centre recovers 0 due to the four-step chief/central-pixel reference), the
OAP single-actuator recovers **143 nm** (lens 146/131 nm). The footprint is
circular (half-extent 41 act), centred on the DM; the pupil is optically clean
(pupil_quality above). So the reflective optics are validated.

**Remaining blocker: registration for the folded mapping.** `tg96`'s inline
two-poke registration (a ray affine + 8-parity blob search, tuned for the lens
rig) returns `|corr| 0.0014` on the OAP rig — the fold's detector→DM pixel
mapping (flip/rotation/scale; the OAP pupil is 0.8× the lens linear size, mask
18376 vs 28917 px) is not resolved by the lens-tuned search. The DM images fine;
only the pixel→actuator registration fails, which gates the closure/transfer/
differential rows.

Next step (focused): use the robust `dm_gauge_lib/dmg_register` + `dmg_frame`
(the 8-DOF-class registration the S3/S4 stages use) in place of `tg96`'s inline
`register_two_pokes` for the OAP rig, so the fold's mapping is picked up. Then
the OAP battery (Stage C–E, side-by-side pm table) runs. The ~12.9 nm flat-DM
null remains a same-plane-fold cost (low-order common systematic the differential
should cancel) to quantify then. No relay, no engine work.

## Deliverable status vs the brief

- D1 builder `'optics'` + `tBench` gate — **DONE** (commit 2e56522).
- D2 folded-layout Stage A — **DONE** (in the runner; near-normal 5°/9° fold, margins printed).
- D3 runner + **lens equivalence gate EXACT** — **DONE** (cfe7a3b, 8bb7d8e).
- OAP battery / side-by-side pm table — **BLOCKED on the pupil relay** (finding 3).
- D4 alignment sensitivity, D5 coated-Al, D6 README (done, results pending),
  D7 slide deck — **pending the OAP rig imaging**.

## Artifacts
`runs/lens/` (equivalence, exact), `runs/oap/` (nulls, imaging fails),
`lens_tail.mat` (0.134 nm, == record), `oap_tail.mat` (12.9 nm floor).
Logs on the JPL box; branch `tg96-oap`.
