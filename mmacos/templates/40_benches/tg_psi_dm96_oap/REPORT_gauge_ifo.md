# REPORT — the interferometer lanes of the DM Surface Gauge Comparison

CCMac for Dave/CCL, 2026-09-13. Branch `dev-candidate`. The IFO's part of the
gauge deck (plan `BRIEF_gauge_deck.md`; brief `BRIEF_ccmac_gauge_deck.md`).
Numbers first; departures flagged; every figure and run tag listed. Companion:
`REPORT_oap.md` (the reflective build-out — D1–D7, item B — cited here, not
repeated). No engine work; `'lens'` byte-identical (tBench 9/9).

## The best configuration, named

**The interferometer's best configuration for the bench is the lens rig read by
the polarization snapshot four-step, calibrated by a PZT four-step.** That is the
hybrid: the polarization snapshot takes the four frames at once — no within-scan
camera or DM drift, no phase-step miscalibration in the change measurement — while
a PZT four-step supplies the absolute step calibration the snapshot's fixed
azimuths do not. The snapshot removes the sequential form's two error classes; the
PZT removes the snapshot's one. The lens rig (not the OAP) because the reflective
front end's same-plane fold leaves a residual cross-talk (~0.18 vs the lens <0.06)
that the open-loop rows tolerate but the closed loop turns into a hard hold-mode
wall (below, and `REPORT_oap.md` D7).

The route there — the three phase-shift forms priced, the OAP front end measured,
the rows and capture range on the working surface — is the backup material below.

## 1. Rows on the 30 nm working surface (matrix measured ON it)

The ZWFS convention: base a 30 nm rms random working surface (seed 7, the field
the ZWFS uses), measure the response matrix on that surface, read differential
rows in actuator space. `g` = gain, `e` = rms error over lit (pm), `flr` = rms of
the unpoked lit actuators (pm), `SNR` = mean(poked)/flr — scored exactly as
`zwfs_run`. Run tags `lens_deck` / `oap_deck` (bare Al); `tg96_run` stage `deck`.

| row (on 30 nm surface) | lens: g / flr / SNR | OAP (bare Al): g / flr / SNR |
|---|---|---|
| single 10 nm at the hold-out site | **0.9911 / 2 pm / 6076** | **0.9956 / 2 pm / 5301** |
| 1 nm on the 52 grid sites (every 8th, 4:8:96) | **0.9895 / 1 pm / 700** | **0.9576 / 1 pm / 683** |
| dense random 10 nm | **0.9895** (err 168 pm) | **0.9497** (err 2099 pm) |

The OAP reads a single localized change as well as the lens (0.9956); the fold
cross-talk appears on the distributed grid (0.958) and dense (0.950) patterns, as
item B found. Tags `lens_deck` / `oap_deck`.

The lit set puts 52 actuators on the every-8th grid (the ZWFS's `4:8:end` grid;
the count is the tg96 pupil's, not 47). Tag `lens_deck` (`runs/lens_deck`).

## 2. Capture range, both ways, with photons

The devices do not operate at null (Dave 2026-09-12): the capture range is the
largest working-surface rms a reading holds to 10% gain error. Two ways —
calibration aging from 30 nm, and the matrix re-measured on the surface — plus the
photons for 1 pm in the S5 noise form. Run tags `lens_deck` / `oap_deck`.

| | lens | OAP (bare Al) |
|---|---|---|
| capture range to 10%, aging from 30 nm — single site / 52-site grid | **322 nm / 480 nm+ (holds)** | **480 nm+ / 480 nm+ (both hold)** |
| capture range, matrix re-measured on 60 / 90 / 120 / 160 nm (1 nm grid) | **holds: gain 0.99, corr 0.9999 to 160 nm** | **holds: gain 0.958, corr 0.980 to 160 nm** |
| photons for 1 pm, S5 form, on 30 nm (gain ≈ 1) | **2.5e14** (stable ~1.4–2.5e14 to 160 nm) | **6.2e14** (~2.5× the lens; stable ~6–8e14) |

**The IFO's small-signal capture range is large and re-measuring holds it — the
earlier "collapse" was a differential bug (CCL 2026-09-13), now fixed.** The deck
differentials and the matrix's own calibration pokes were computing
`measf(base+dev) − measf(base)` — a difference of two SEPARATELY-WRAPPED absolute
maps, which tears at the 2π jumps once the base exceeds λ/4 (≳ 60 nm rms). A small
(1 / 10 nm) differential does not itself wrap, so the correct primitive is the
WRAPPED PHASE DIFFERENCE (the V1 lesson / `fsdiff_`: difference the raw four-step
phases, THEN wrap). Fixed everywhere, byte-identical on a base < λ/4 (the 30 nm
rows reproduce to the digit; the loop is unchanged). With the fix:
- **Aging** (matrix once on 30 nm): the 10 nm gain holds 0.99–1.03 to 480 nm on the
  distributed 52-site grid (capture 480 nm+, holds at the last rung) and 0.99→1.15
  on a single poke (capture 322 nm, the single-site gain drifts high on a large
  base). Both far exceed the ZWFS's 42 nm — a small four-step differential is
  wrap-free even when the absolute base surface wraps.
- **Re-measured** on 60 / 90 / 120 / 160 nm: gain 0.99, floor 1 pm, **corr 0.9999**,
  flat to 160 nm — the same "re-measuring extends it" behaviour the ZWFS has (D1
  windows, placed on the flat, register fine on the larger surfaces).
- **Photons** N(1 pm) is now stable (~1.4–2.5e14 across 30–160 nm), the estimator
  holding gain rather than the dead-estimator artifact the buggy version showed.

The wrap is real only for the ABSOLUTE map and the DESCENT (a ~2 rad differential
from a 100 nm figure to the set point) — item 5, TO's `dmg_unwrap`. Two photon
numbers, both stated (they differ ~1.8×): S5 form **2.5e14** at 30 nm; the loop's
in-run `sig_n 11.97 pm at 1e12` → 1 pm at **~1.4e14** (single-site S5 variance vs
rms-over-lit loop `sig_n`), as the brief anticipated.

## 3. The three phase-shift forms

The four frames are identical in the model; the forms differ in what they get
wrong. Priced with existing machinery on the lens rig.

| form | how the four steps are made | what it gets wrong | pricing |
|---|---|---|---|
| **PZT four-step** | reference-arm phase stepped in time (sequential) | phase-step miscalibration; within-scan camera / DM drift | step error 2% / 5% (`pzt.step_err`); camera 1/f walk (`dmg_loop` `cam`) — below |
| **polarization snapshot** | four analyzer channels at once (simultaneous) | polarization systematics (no within-scan drift) | v1 plate +11.7% (correctable); v2 cube R_p; coating retardance — below |
| **hybrid** | snapshot for the change, PZT for the absolute step | — (each half removes the other's error) | the recommended form |

### PZT four-step — the sequential form's two error classes

- **Phase-step miscalibration** (`P.pzt.step_err`, TO's `pdi.step_err` pattern: the
  four frames step with the error, the atan2 solve assumes the nominal π/2
  quadrature). **A gain error on the absolute rows, ~1:1 with the step size, and
  it self-cancels in the differential hold.** On the 30 nm surface rows (tags
  `lens_deck_se2`/`_se5`): 2% step → single-actuator gain **0.9743** (−2.6%), grid
  0.9905, dense 0.9885, floor 2→5 pm; 5% step → single **0.9543** (−4.6%), grid
  0.9962, dense 0.9915, floor →9 pm. The single (localized) read carries the gain
  error; the distributed grid/dense rows average it out. **In hold mode
  (`loop_lens_se2`) the 2% step error is negligible** — 3 pm noise-only at 5.4e12
  and the 2 pm walk at 1.7e13, essentially the record 5.5e12 / 2.0e13: the fixed
  step error rides both the reference and the measurement, so it is common-mode in
  the closed loop. A known step error is a known (calibratable) gain.
- **Within-scan drift.** The camera 1/f offset (`dmg_loop` `cam`, Dube et al. 2024:
  Roman LOWFS is camera-drift dominated). A zero-sum four-step is exactly immune to
  an offset CONSTANT within a scan; only the fraction that develops frame-to-frame
  (`cam_intra`) breaks it. **A 1e-3-of-signal walk with 25% within-scan
  (`loop_lens_cam`) costs little**: 3 pm held at 6.1e12 photons vs 5.4e12
  noise-only (~13% more light), the held spectrum [<4, 4-12, >12 cyc/ap] = [0.02,
  0.07, 0.93] pm; at `cam_intra 0` it is exactly immune. The **DM's own within-scan
  walk** (`loop.intra`, now landed on origin — TO) is the temporally-stepped
  analogue and is priced next (a walk drift with `loop.intra 0.25`).

### Polarization snapshot — all four frames at once, so no within-scan drift

The systematics are polarization, from the rigs that exist
(`90_polarization/tg_psi_dm`, `tg_psi_dm_v2`):

- **v1 plate rig — BS diattenuation, a GAIN error, correctable.** A plate
  beamsplitter at non-normal incidence is a diattenuator; with each arm's state at
  45° to the s/p axes it rotates the test arm **+7.479°** and the gauge reads a PSI
  scale gain **1.11661 (+11.7% high)**. The analyzer sweep + a +3.768° waveplate
  clock nulls it to gain **1.00000** (fringe visibility 0.996). The residual 4θ term
  (8.9e-4 of the fringe) **cancels in the differential** (1.7e-14 nm) — it survives
  only in a single-shot map.
- **v2 MacNeille cube — diattenuation structurally absent.** Putting each arm's
  state ON a coating eigenaxis (test on p, reference on s) leaves the arms
  **5.3e-6° from orthogonal**, PSI scale gain **0.999999**, fringe visibility
  **1.000000**, no compensator, 2.27× the delivered power. The design lesson (a
  **floor**, not a gain error): the naive odd stack `H(LH)^4` is a **2.11e-2 (2.1%)
  R_p** leaky polarizer; the symmetric period `(½H L ½H)^4` is R_p = 0 (extinction
  T_p/T_s = 2382:1). Use the symmetric stack.
- **Coating retardance (item B).** On the reflective rig, the OAP fold's L1
  retardance VARIATION is 0.55 mrad (bare Al) — a small floor, not a gain error;
  the perfect-conductor 0.00 mrad is the singular idealization (`REPORT_oap.md`
  item B). It has no analogue on the lens rig (transmissive, near-normal).

### Hybrid — which error each half removes

The snapshot's weakness is the absolute step: its four analyzer azimuths are fixed
design constants, so a systematic in their realization is a fixed gain the snapshot
cannot self-calibrate. The PZT's weakness is the sequential scan: step
miscalibration and within-scan drift. Run the change measurement as a snapshot
(immune to within-scan drift, no step error), and calibrate the absolute scale
with an occasional PZT four-step of a known reference (immune to the snapshot's
fixed-azimuth gain, since the PZT step is metrologically traceable). Each half
removes the other's error class; no additional run is needed to state this — the
snapshot's gain (v1 +11.7% → corrected 1.00000; v2 0.999999) and the PZT's step
error (section above) are the two numbers, and the hybrid carries neither.

## 4. Lenses vs OAPs, and the OAP front end under the other gauges

**The IFO on each** (full numbers in `REPORT_oap.md` D3/D4/D7 + item B):

| | lens | OAP (bare Al) |
|---|---|---|
| single 10 nm differential (flat) | 0.9916 / 2.2 pm | 0.9950 / 2.1 pm |
| dense random 10 nm (flat) | 0.9894 | 0.9497 |
| modal cross-talk | <0.06 | ~0.18 |
| flat-DM null | 0.134 nm | 13.09 nm (cancels in the differential; D4) |
| closed-loop 3 pm noise-only / 2 pm walk | 5.5e12 / 2.0e13 | 3.4e13 / never (floors 4.1 pm) |
| thermal floor / noiseless step | 13.1 pm / 87 pm (8.6%) | 39 pm / 276 pm (27.6%) |

**Closing the reflective design (is the 0.18 fold cross-talk reducible?).** The
residual is GEOMETRIC — the same-plane fold's astigmatism cross-talk on dense /
high-order patterns, ~0.18 vs the lens's <0.06 with the identical tail, left after
the ideal-reflector polarization null is retired (item B). It is **not** a coating
or azimuth term: item B settled that the retardance VARIATION (0.55 mrad bare Al)
already breaks the null, and re-solving the waveplate azimuth compensates a fixed
polarization state, not a geometric aberration. So the two levers the brief names
that touch polarization (coating retardance spec, waveplate azimuth) cannot move
0.18; only the geometry can. The fold astigmatism of an off-axis parabola scales
as the fold angle squared, and the fold angles are set by Stage-A body clearance —
OAP1 at 5° (+22 mm margin), OAP2 at **9° with only +6 mm margin**. Halving the
astigmatism (θ→θ/√2, OAP2 9°→6.4°) drops the lateral offset below the 126 mm the
source/camera bodies need, so it costs a **~1.33× longer optical leg** (or a larger
LEG_CAP) — a real packaging price, not free. The alternative is software: a modal-
decoupling correction on the measured matrix (exactly what the ZWFS linear-L
reading needs, and what the shared loop code already takes) flattens the cross-talk
without touching the optics. **Verdict:** the OAP IFO is not intrinsically
open-loop-only — it can hold if the matrix is modally decoupled or the fold is
opened up at a packaging cost — but as a raw measured-matrix gauge on a
clearance-bound bench it is an open-loop / differential-grade instrument, and the
lens rig reaches the on-orbit hold with none of this (which is why the lens is the
recommended IFO configuration).

**The other gauges on the OAP rig** (`zwfs_run` on the OAP test arm; readings
L/S/V/P/PF; stages bench + battery + noise + loop; V arm maps `mask.v_arm
'engine'`). The bench gates G1 (mask-sandwich round trip) and G3 (DM-conjugate
pupil) may fail on an OAP tail; if they do, the numbers are reported and the run
stops — the gates are not forced. Run tag `zwfs_oap`.

**`coat_oap` wired (CCL, `a360faf`); the run STOPS before the gates — the mask
gauges do not set up on the OAP front end (a `twyman_green` OAP-tail issue, for
CCL).** With `bench.optics='oap'`, `bench.coat_oap='bareAl'`, `OAP1_AOI 5` /
`OAP2_AOI 9`, the bareAl coating applies (L1, L2) but the gauge setup fails at
`dmg_zwfs_gauge: mask plane not focused` — the FocalMask does not reach the OAP2
focus. Diagnosed (not forced, per the brief):
- **Not fold astigmatism.** It fails even at `OAP2_AOI 1°` (near-normal, minimal
  astigmatism).
- **Not a reachable defocus.** Scanning `MASK_TRIM` from −200 to +200 (mask sphere
  swept 194→534 mm, ±170 mm around the ~360 mm nominal) never focuses; the lens
  focuses at its nominal trim (−5.582), so the harness is sound.
- **It is the OAP tail's mask-plane geometry in `twyman_green`.** The reflective
  tail (`L2.F − L2.thk + MASK_TRIM` from an OAP2 mirror) does not converge the beam
  to a tight focus at the FocalMask. tg96's IFO tail never checks mask focus (it
  reads the pupil), so this surfaced only under the ZWFS gauge.

**Consequence for slide 15.** The interferometer runs on the OAP front end (the
tg96 OAP deck rows / loop / D4 / item B above). The focal-plane MASK gauges
(ZWFS dimple, vZWFS metasurface, PDI pinhole) cannot be set up on the OAP front
end until `twyman_green` places the FocalMask at the OAP2 focus for the reflective
tail — a builder fix on CCL's side. Flagged; the IFO-on-each comparison (the main
lenses-vs-OAPs content) is complete without it.

## 5. The descent run (capturing the initial figure) — the IFO captures it

Capturing the DM's initial figure is a WRAP problem: the differential from a
100 nm rms surface to the 30 nm set point is ~2 rad, so the four-step's wrapped
difference folds. TO's 2-D least-squares unwrapper (`dm_gauge_lib/dmg_unwrap`, on
the lit mask) removes it — the limit moves from the λ/4 wrap to the pixel gradient,
and the DM is smooth at 4 detector-px/actuator. `tg96_run`'s loop stage mirrors it
(`fsdiff_` → unwrap when `loop.start_rms` is set or `battery.unwrap` is on),
byte-identical with it off (the record loop rows reproduce). The loop stage also
mirrors `loop.start_rms` / `recal_every` / `intra` / `ins.recal` / the
`ins.measure(cmd, aux)` within-scan term.

**Result (lens rig, PZT four-step, unwrapping ON, tag `descent_lens`): the IFO
captures a 60–300 nm initial surface (120–600 nm WFE) and drives it to the ~2 pm
hold floor within 60 cycles, with NO recalibration needed.**

| start rms (WFE) | r(1) | k to 10 nm | k to 3 pm | r(K), 1e13 / 1e15 | ρ |
|---|---|---|---|---|---|
| 60 nm (120) | 29.7 nm | 3 | 16 | 2.15 / 0.22 pm | 0.51 |
| 100 nm (200) | 69.7 nm | 4 | 18 | 2.14 / 0.21 pm | 0.53 |
| 150 nm (300) | 119.7 nm | 6 | 21 | 2.12 / 0.21 pm | 0.57 |
| 200 nm (400) | 169.7 nm | 7 | 25 | 2.11 / 0.21 pm | 0.62 |
| 300 nm (600) | 269.7 nm | 25 | 43 | 2.09 / 0.21 pm | 0.84 |

- **Every start converges** to the hold floor (2.1 pm at 1e13 photons, 0.21 at
  1e15 — the noise floor, not a capture limit). The largest tested start (300 nm
  surface / 600 nm WFE) still reaches 3 pm by cycle 43 (of 60); the convergence
  contraction ρ rises 0.51→0.84 with start size, so 300 nm is near the K=60 edge.
- **Recalibration is not needed:** recal-every-10 and never give identical results
  (e.g. 300 nm: k(3 pm) 45 vs 43, r(K) 2.25 vs 2.09 pm) — the start matrix stays
  valid through the descent, because the loop drives the surface monotonically back
  toward the set point the matrix was measured on.
- The photon level sets the floor, not the capture: 1e13 and 1e15 follow the same
  cycles-to-reach path (deterministic contraction), differing only in the final
  floor (2.1 vs 0.21 pm).

So the route from the DM's initial figure to the hold regime is: unwrap the
four-step differential, close the loop on a start-surface matrix at gain 0.5, and
it converges in tens of cycles with no recal. (OAP descent is backup; the
mask-based gauges' descent is gated on the `twyman_green` OAP-tail fix, §4.)

## 6. Layouts and parts lists

Layout figures redone in the `zwfs_dm96/zwfs_vlayout.m` recipe (`macos.view_rx`
with `labels` off, passive Reference planes hidden, the fold plane seen from above
`view(0,90)` axis-equal, both arms overlaid — test blue, reference + PZT flat
orange — elements named by `text` with leader lines off the beam at 15–17 pt in an
1800-px figure, and the crowded node as a second panel cropped to it):
- **`lens_vlayout.png`** — the lens rig: top, the whole train (collimator L1, 96×96
  DM, beamsplitter, focuser L2, field lens, camera) with the reference arm + PZT
  flat; bottom, the BS / compensator / polarization-tail node (input polarizer,
  compensator, output QWP, beamsplitter, recombination, analyzer).
- **`oap_vlayout.png`** — the reflective rig: OAP1 (collimator) and OAP2 (focuser)
  fold the beam in the BS plane (now seen from above, not edge-on); same node panel.

Regenerate: `tg96_run('stages',{'bench','figs'})` (writes `<tag>_vlayout.png` in
`runs/<tag>/`; the drawing is geometry, so MODEL 512 suffices). CCL QAs at slide
size; the OAP2 label sits close to the title — a candidate tweak.

### Parts list — lens rig (shared front end + interferometer)

Scale s = 96/56 = 1.714 off the 56 mm v1 rig; geometry from `tg96_params` +
Stage A (`runs/lens`).

| part | size / focal length / AOI | coating | count | what it is for |
|---|---|---|---|---|
| source (spatially filtered HeNe, 632.8 nm) | beam radius 51 mm collimated | — | 1 | the illumination |
| collimator L1 | f = 857 mm, ⌀ 103 mm | AR | 1 | collimate the source onto the DM |
| beamsplitter (plate) | AOI 7°, 2.6 mm thick | 50/50 (polarizing) | 1 | split test / reference arms |
| compensator plate | matched to the BS, 171 mm from BS | AR | 1 | balance the BS glass path in the reference arm |
| 96×96 deformable mirror (test object) | 96 mm aperture, 1 mm pitch, 700 mm leg | protected Al | 1 | the surface under test |
| reference flat on a PZT | ⌀ ≥ 103 mm, ~564 mm leg | protected Al | 1 | the reference arm's return + the PZT four-step |
| focuser L2 | f = 429 mm, ⌀ 103 mm | AR | 1 | image the DM pupil to the tail |
| field lens FL | f = 43 mm, ⌀ 21 mm | AR | 1 | pupil-imaging tail (re-tuned per optics) |
| camera | 385 px per pupil | — | 1 | the pupil-image detector |
| **snapshot form only:** input polarizer, arm QWPs (test / reference), output QWP, analyzer | quarter-wave; azimuths 45 / 0 / 45 / 0 / 0° | — | 5 | the polarization four-step (analyzer sweep) |
| **v2 snapshot:** MacNeille cube (replaces plate BS + compensator + ideal polarizers) | 12.7 mm, ZnS/cryolite on n_g 1.655, symmetric stack | MacNeille coating | 1 | structural diattenuation-free split |

### Parts list — OAP rig (reflective front end; replaces L1, L2)

Geometry from Stage A (`runs/oap`): the OAPs fold in the BS plane; off-axis
distance = f·|sin(180−2·AOI)|.

| part | off-axis dist / f / AOI | coating | count | what it is for |
|---|---|---|---|---|
| OAP1 (collimator) | f = 857 mm, AOI 5°, off-axis 149 mm | bare / protected Al | 1 | collimate + fold the source→DM leg |
| OAP2 (focuser) | f = 429 mm, AOI 9°, off-axis 132 mm | bare / protected Al | 1 | image + fold the DM→camera leg |
| (BS, DM, reference flat, field lens, camera, polarization optics as the lens rig) | | | | the tail is re-tuned for the OAP focuser |

The OAP2 fold sits at AOI 9° with only +6 mm lateral clearance margin (Stage A);
OAP1 at 5° has +22 mm. Both are near-normal to keep the fold aberration small.

## Departures / open

- Item 5 (descent) BLOCKED on TO's `loop.start_rms` / `loop.recal_every`.
- Item 3 DM within-scan drift (`loop.intra`) BLOCKED on TO (camera `cam_intra`
  shipped; DM `intra` not yet).
- OAP bench gates G1/G3 may fail on the OAP tail under `zwfs_run` — reported, not
  forced (section 4).

## Reproduce

    ./tg96_batch.sh lens_deck "'stages',{'bench','deck'},'battery.noise',true"
    ./tg96_batch.sh oap_deck  "'bench.optics','oap','bench.coat_oap','bareAl','stages',{'bench','deck'},'battery.noise',true"
    ./tg96_batch.sh lens_deck_se2 "'stages',{'bench','deck'},'pzt.step_err',0.02"   # item 3 PZT rows
    ./tg96_batch.sh loop_lens_cam "'stages',{'bench','loop'},'loop.drifts',{'walk','thermal','cam'}"  # item 3 camera walk
    # the other gauges on the OAP front end (CCL's zwfs_run; tg96 calls it, does not edit it):
    #   zwfs_run('bench.optics','oap','bench.coat_oap','bareAl','readings',{'L','S','V','P','PF'}, ...
    #            'stages',{'bench','battery','noise','loop'},'mask.v_arm','engine')

Runs on the Mac (MATLAB R2024a + `mmacos.mexmaca64`), MODEL 1024 ~11 GB, one at a
time (`tg96_batch.sh`). Pruned run artifacts committed under `runs/<tag>/`.
