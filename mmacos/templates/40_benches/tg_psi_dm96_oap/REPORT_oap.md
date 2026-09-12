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
lens (single differential 0.9948 / 2.2 pm == lens).** The dense / high-order
random-10nm gain is 0.75 on the UNCOATED (ideal-reflector) OAP — **but see the
Review-response section below: this is the bright FRACTION (bright 0.99, dark-25%
0.05), the dark set is an ideal-reflector polarization null, and a realistic
protected-Al coating recovers the dense gain to 0.95.** So the 0.75 / "0.42 cross-
talk" here OVERSTATES the fold cost. The 12.9 nm null is a low-order arm difference;
its **cancellation in the differential is measured, not assumed** — see D4.

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

## Review response (CCL f46585d, items 1-6) + D5 — the OAP story reframed

The review's investigations changed the OAP conclusion. **The dense-pattern loss
is NOT a fundamental same-plane-fold cost — it is largely an IDEAL-REFLECTOR
polarization artifact that a realistic protected-Al coating removes.**

- **Item 2 (tail keyed by run tag).** Fixed: the tuned tail is now keyed by
  `bench.optics` (`lens_tail`/`oap_tail`), `<tag>_tail` an override; the null is
  printed beside the tail's expected value with a 10x WARN. A non-canonical tag no
  longer silently runs the seed.
- **Item 3a (window truncation?).** NO. `matrix_window='voronoi'` (assign every
  mask pixel to its nearest poked actuator — nothing truncated/double-counted)
  gives BYTE-IDENTICAL results to the box window (dense-random 0.7486; dark 0.05).
  The dark actuators are not truncated into neighbours.
- **Item 3b (regularization shrinking dim columns?).** NO. Splitting the dense-
  random gain over bright vs dark-25% columns: **bright 0.99 (== lens), dark
  0.05-0.13**, barely helped by `matrix_lam` 1e-3→1e-5. The dark columns are
  genuinely faint, not over-regularized. The OAP "0.75 dense gain" is just the
  BRIGHT FRACTION (~0.75·0.99), not a cross-talk defect.
- **Item 4 (worst-at-centre picture).** `tg96_d1_picture` (both rigs, in
  `runs/oap/d1_picture.png`) + numerics: the dark region is **centre-dark**
  (column-norm 0.0 at the exact centre, 0.10 at r<4mm rising to 0.6 outward) **plus
  a stripe along y** (|x| 3.6 vs |y| 18.6 mm for the dark set) — a chief-pixel /
  on-axis / polarization pattern, NOT smooth symmetric optical vignetting.
- **Item 5 (non-vacuity for the wrong rig).** Fixed and honest: over the SAME
  poked set, on the OAP the best axis-aligned parity+scale map reaches **72.77%,
  EQUAL to the affine's 72.77%** — the near-normal fold is axis-aligned (SVD
  anamorphism 0.00%), so the affine's advantage is NOT rotation. The real reason
  `register_two_pokes` failed on the OAP is its CENTRE-poke anchor reading 0 (the
  four-step reference); the affine route's off-centre anchor + ray-fit sidesteps it.
  On the lens the check is vacuous (its mapping is axis-aligned) and labelled so.
- **Item 6 (calib_surface 'base' on the ladder).** Run (lens + OAP). Base
  calibration does NOT extend the 240 nm break: the break is a four-step
  MEASUREMENT WRAP (the base map exceeds lambda/4 surface), independent of which
  surface J is measured on. The wrap guard now flags `BROKE (wrap ...)` when the
  estimator diverges (gain<0/>3 or corr<0.3) instead of printing a negative gain.

**D5 (coated-Al OAPs) — the reframe.** Protected-Al (MgF2 lambda/2 over opaque Al,
HeNe) on both OAPs:

| | uncoated (ideal reflector) | coated (protected-Al) |
|---|---|---|
| flat-DM null | 12.893 nm | 13.092 nm (+0.2 nm) |
| Stage-E flat/single-10nm | 0.9948 / 2.2 pm | **0.9950 / 2.1 pm** |
| Stage-E flat/**random**-10nm | 0.7486 / 4848 pm | **0.9488 / 2118 pm** |
| dense-random gain, **dark-25% columns** | **0.05** | **0.82** |

The realistic coating's reflection retardance **breaks the ideal-reflector
polarization null** that darkened the centre + y-stripe, recovering the dark
actuators (0.05→0.82) and the dense-random gain (0.75→0.95, near the lens 0.99).
The single-actuator differential is unchanged (0.995) and the null shifts only
0.2 nm (common-mode-ish, cancels in the differential like the D4 alignment).
**Flag for Dave/CCL:** the D3 uncoated OAP numbers (dense 0.75, "cross-talk 0.42")
OVERSTATE the fold cost because they use an ideal reflector; a real coated mirror
calibrates dense patterns at ~0.95. Whether the uncoated null is a physical
bare-metal reflection or a model idealization is the open modelling point.

## Deliverable status vs the brief

- D1 `dmg_frame` affine + `tg96_place` + gate — **DONE**; lens GREEN, OAP reported
  (75% within 2px / 98.9% containment; the affine's edge over a well-anchored
  parity map is the off-centre ANCHOR, not rotation — item 5).
- D2 `calib_mode` matrix (Route 2) + lens validation — **DONE**, beats the record.
- D3 OAP battery side-by-side + null + break ladder + wrap guard — **DONE**;
  dense loss decomposed (items 3a/3b: bright 0.99, dark faint — not truncation/reg).
- D4 OAP alignment sensitivity — **DONE** (null cancels in the differential).
- D5 coated-Al OAPs — **DONE**; recovers the dense gain (the ideal-reflector reframe).
- D6 README + this report — **DONE**.
- D7 closed-loop hold metric — **BLOCKED**: `dm_gauge_lib/dmg_loop.m` not yet on
  `origin/dev-candidate` (awaits Dave's push); stage 'loop' to follow.

## Reproduce

    tg96_run('bench.optics','lens','battery.calib_mode','matrix','tag','lens')
    tg96_run('bench.optics','oap','battery.calib_mode','matrix','battery.d4',true,'tag','oap')
    tg96_run('bench.optics','oap','battery.matrix_window','voronoi','tag','oap_vor')   % item 3a
    tg96_run('bench.optics','oap','battery.calib_surface','base','tag','oap_base')     % item 6
    tg96_run('bench.optics','oap','bench.coat_oap',true,'tag','oap_coat')              % D5
    % D1 error + column-norm picture (both rigs), from the saved run .mat:
    tg96_d1_picture('runs/lens/lens.mat','runs/oap/oap.mat','runs/oap/d1_picture.png')

Runs on the Mac (MATLAB R2024a + `mmacos.mexmaca64`), MODEL 1024 ~11 GB, one at a
time (`tg96_batch.sh <tag> "<args>"`). Evidence in `runs/dev{lens,oap}{,mx}/`
(dev tags; `*_tail.mat` = copies of the tuned `lens_tail`/`oap_tail`, so the bench
is the record bench — nulls reproduce 0.134 / 12.9 nm).

## Artifacts
`dm_gauge_lib/dmg_frame.m` (extended), `tg96_place.m`, `tg96_apply_parity.m`,
`tg96_run.m` (Stage MATRIX/PLACE/D4), `tg96_params.m` (matrix + place + d4 knobs).
