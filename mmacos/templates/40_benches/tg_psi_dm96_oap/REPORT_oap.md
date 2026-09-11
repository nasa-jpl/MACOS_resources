# REPORT — reflective (OAP) variant of the tg_psi_dm96 96×96 T-G DM gauge

CCMac for Dave, 2026-09-11 (updated). Branch `tg96-oap` (MACOS_resources). Numbers
first; departures flagged. **Status: the last-mile items are done. D1 (window
placement from the ray affine) is GREEN on the lens and characterized on the OAP;
D2 (the measured response matrix) beats the record on the lens; D3 (OAP battery,
side-by-side) and D4 (OAP alignment sensitivity) are measured. The reflective
gauge WORKS — its differential reading is clean and robust to the fold's null and
to OAP misalignment; the fold's cost is astigmatism cross-talk on dense patterns.
No engine work; `'lens'` byte-identical (tBench 9/9).**

## The route (what changed)

`register_two_pokes`'s 8-parity blob search cannot express the OAP fold's mapping
(it is a flip + rotation the parity set cannot represent), which is why it returned
|corr| 0.0014 on the OAP. Replaced with:

- **`dmg_frame` extended to return the full ray affine** (`Aaf`/`Lm`/`Linv`/chief
  pixel). The affine carries the fold's flip + scale directly.
- **`tg96_place.m`** — ray-affine bootstrap → field-array parity resolved from TWO
  DIRECTIONAL reference pokes (a diagonal ref cannot disambiguate the transpose) →
  a **robust (MAD-reject) affine refit** from the multiplexed-poke CoMs. The loose
  OAP bootstrap has outlier CoMs the tight lens one did not; robust fitting is what
  the fold needs. Anchor/refs are placed OFF the chief pixel — the exact-centre poke
  reads **exactly 0** (single-poke peak 0.000; the four-step chief-pixel reference,
  confirmed).
- **`battery.calib_mode` 'matrix' | 'kernel'** — Route 2: poke every `matrix_step`-th
  actuator, step through the offsets so every lit actuator is poked once, cut each
  response from its own affine-placed window, assemble J (detector px × lit act),
  estimate by regularized least squares. Every reading mean-referenced over the mask;
  each column carries its own volume as a rank-one term (lifted from ZWFS S10
  `calib_matrix_`/`est_matrix_`, `meas_surface` as the map primitive).

## D1 — window placement gate (CoM within 2 px of the affine-predicted pixel)

| | LENS (control) | OAP |
|---|---|---|
| within 2 px | **100.00%** | **72.77%** (window CONTAINMENT 98.9%) |
| median err | 0.07 px | 0.16 px |
| ray-affine | mag 9.9, −180° (parity) | mag 10.7, det<0 (flip), anamorphism 0.00% |
| robust-affine fit residual | 0.07 px | 0.16 px |

**Lens D1 is GREEN and is the brief's push gate.** Non-vacuity: the best
axis-aligned shear-free map (all a parity+scale search can express) and the old
`register_two_pokes` parity both fail to reproduce the measured CoMs.

**OAP D1 does NOT pass the strict 2 px gate — reported, not thinned.** The
placement *map* is correct (0.16 px robust-affine residual; 98.9% of measurable
responses land inside the calibration window — the criterion the matrix calibration
actually needs). The 2 px shortfall is **physical, not a code artifact**: a single-
poke gate gives the same 75% as multiplexed, and the single off-axis OAP images the
pupil with astigmatism, so ~20–25% of actuators image dark/elongated (worst at
centre). **This is the same-plane-fold effect the brief asked to measure.**

## D2 — measured response matrix, validated on the lens (BEATS the record)

| metric | matrix (this) | kernel record |
|---|---|---|
| Stage-E flat / single-10nm | **0.9916 gain, 2.2 pm** | 0.9654, 21 pm |
| Stage-E flat / random-10nm | **0.9894** | 0.9203 |
| Stage-C single @150nm | 0.9968 gain, 98.8 pm (absolute) | — |
| modal transfer 0.7→68 cyc/pup | **0.96–0.99, cross-talk <6%** | (kernel rolls off) |

The measured matrix has no modal roll-off — flat response across the band, exactly
the ZWFS-S10 behaviour. **Break ladder (base rms sweep):** gain ~1.0 / floor <20 pm
to 120 nm base, then breaks at 240 nm (the flat-calibrated matrix's base limit; the
record's kernel + radial Wiener held to 480 nm — a tradeoff. `calib_surface='base'`
would restore base-robustness; not exercised here).

## D3 — OAP battery, side-by-side with the lens

| | LENS | OAP |
|---|---|---|
| flat-DM null | 0.1345 nm | **12.893 nm** (== record) |
| Stage-C single @150nm | 0.9968 / 98.8 pm | **0.9927 / 22.4 pm** |
| Stage-E flat / **single**-10nm | 0.9916 / 2.2 pm | **0.9948 / 2.2 pm** |
| Stage-E flat / **random**-10nm | 0.9894 / 169 pm | **0.7486 / 4848 pm** |
| modal transfer gain | 0.96–0.99 | **0.63–0.95** |
| modal cross-talk | <0.06 | **~0.42** |

**The reflective gauge measures a LOCALIZED / sparse actuator change as well as the
lens (single differential 0.9948 / 2.2 pm == lens).** The fold's cost lands on
DENSE / high-order patterns: the astigmatic pupil imaging spreads dense responses,
giving ~0.42 modal cross-talk and random-10nm gain 0.75 (vs the lens's 0.99). The
12.9 nm null is a low-order arm difference; its **cancellation in the differential is
measured, not assumed** — see D4.

## D4 — OAP alignment sensitivity (10 µm decenter, 10 µrad tilt, one at a time)

| perturb | null shift | sensitivity | single-diff gain | resid |
|---|---|---|---|---|
| OAP1 decenter | 16.1 nm | **1.61 nm/µm** | 0.9929 | 2.4 pm |
| OAP1 tilt | 94.6 nm | **9.46 nm/µrad** | 0.9929 | 5.8 pm |
| OAP2 decenter | 15.3 nm | **1.53 nm/µm** | 0.9932 | 2.3 pm |
| OAP2 tilt | 102.7 nm | **10.27 nm/µrad** | 0.9915 | 2.5 pm |

**The single-actuator differential gain stays 0.99 (resid 2–6 pm) under every OAP
perturbation** even though the null shifts by tens of nm. The null shift and the
alignment drift are common-mode and **cancel in the differential** — the reflective
gauge's differential reading is robust to the fold's low-order arm difference AND to
OAP misalignment. That is the whole point of the differential and it holds.

## Departures from the brief (numbers first, then why)

1. **OAP D1 2 px gate not met (75%).** Physical astigmatism, not fixable in
   software; reported alongside 98.9% containment (the calibration-relevant
   criterion). Not silently thinned (Dave's ruling: report both, don't claim pass).
2. **The matrix breaks at a lower working state than the kernel** (240 nm vs 480 nm)
   because it is flat-calibrated; `calib_surface='base'` would extend it. Not
   exercised.
3. Cross-talk (0.42) is quoted as the measured fold cost per Dave's steer; it was
   not further decomposed into physical-vs-placement contributions.

## Deliverable status vs the brief

- D1 `dmg_frame` affine + `tg96_place` + gate — **DONE**; lens GREEN, OAP reported.
- D2 `calib_mode` matrix (Route 2) + lens validation — **DONE**, beats the record.
- D3 OAP battery side-by-side + null + break ladder — **DONE**.
- D4 OAP alignment sensitivity — **DONE**.
- D5 coated-Al OAPs — **STRETCH, not done** (deferred).
- D6 README + this report — **DONE**.

## Reproduce

    tg96_run('bench.optics','lens','battery.calib_mode','matrix','tag','lens')
    tg96_run('bench.optics','oap','battery.calib_mode','matrix','battery.d4',true,'tag','oap')
    tg96_run('bench.optics','oap','stages',{'bench','place'},'tag','oap')   % D1 gate only

Runs on the Mac (MATLAB R2024a + `mmacos.mexmaca64`), MODEL 1024 ~11 GB, one at a
time (`tg96_batch.sh <tag> "<args>"`). Evidence in `runs/dev{lens,oap}{,mx}/`
(dev tags; `*_tail.mat` = copies of the tuned `lens_tail`/`oap_tail`, so the bench
is the record bench — nulls reproduce 0.134 / 12.9 nm).

## Artifacts
`dm_gauge_lib/dmg_frame.m` (extended), `tg96_place.m`, `tg96_apply_parity.m`,
`tg96_run.m` (Stage MATRIX/PLACE/D4), `tg96_params.m` (matrix + place + d4 knobs).
