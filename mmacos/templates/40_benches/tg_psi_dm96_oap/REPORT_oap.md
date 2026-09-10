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
- **OAP finding 3 (OPEN, the blocker):** the OAP rig **cannot image the DM
  pupil** — a single-actuator poke recovers ≈0 nm (lens: 146 nm) and
  registration fails. A detector-conjugate sweep finds **no plane** that images
  the poke faithfully (peak grows monotonically, never near the true poke
  height), so refocusing alone (option 1) does not recover it — consistent with
  off-axis-focuser astigmatism on the large DM pupil. Needs option 2 (a
  reflective pupil relay) or option 3 (field lens as the on-axis pupil imager).

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

## Recommendation / handoff

The pupil relay is a design decision (Dave's option list): **(2)** add a
reflective pupil relay in the tail (a second OAP or an Offner `add_relay`) so
the DM re-images without astigmatism, or **(3)** re-derive the (transmissive)
field lens as the on-axis pupil imager after the OAP focus. Option 1 (retune the
existing tail on sharpness) does not converge — the sweep shows no faithful
pupil plane exists for the single-OAP tail. Recommend option 2 (all-reflective,
matches the intent) as a focused follow-on.

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
