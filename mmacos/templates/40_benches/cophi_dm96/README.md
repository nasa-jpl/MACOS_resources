# cophi_dm96 — COPHI, the heterodyne camera interferometer, as the fifth DM gauge

The campaign plan is `macos/BRIEF_cophi_campaign.md` (Dave's rulings in its §7);
Feng Zhao's proposal is `~/dev/MACOS_sandbox/DM_gauge/COPHI_for_DM_gauge.pptx`
(purpose: HOLD — each DM in a local control loop); the first look is
`macos/NOTE_cophi_first_look.md`; the cameras `macos/NOTE_detectors_survey.md`.

## What COPHI is, in this bench's terms

The TG96 interferometer's piezo four-step run CONTINUOUSLY with frequency-set
steps: two AOMs (80 MHz and 80 MHz + 7.5 Hz) on a Rb standard beat at 7.5 Hz,
the camera is triggered from the same clock at 30 fps, each frame integrates a
quarter of the beat, and the phase is the four-bucket `atan2` — the N-frame
least squares is Feng's backup.  The LO is a fiber brought to the recombiner;
there is no analyzer and no QWP, so the polarization gain is absent by
construction (the record's IFO reads at 0.993–1.004).  New error terms: the LO
arm's wavefront and its drift; differential fiber phase WITHIN the four frames;
trigger jitter; the integrating bucket (contrast × 0.900).  Photon-bound like
every camera gauge: the well sets photons per frame.

## Code (shared, not copied)

- `dm_gauge_lib/dmg_cophi_frames.m` — the N integrating-bucket frames from the
  two traced fields (bucket centres returned); jitter / walk / drift knobs.
- `dm_gauge_lib/dmg_cophi_recon.m` — the N-bucket least squares (= the
  four-step `atan2` about the bucket centres for N = 4).
- `dm_gauge_lib/dmg_cophi_camera.m` — well, shot noise, read noise (Dave:
  read noise in EVERY comparison).
- `cophi_params.m` — `tg96_params` with reading 'H' and the `P.cophi` /
  `P.format` sheets (cameras BY NAME from the survey; verify before a slide).
- `cophi_ctx.m` — the H reading as a `tg96_run` reading context (`P.reading_ctx`):
  launches, analyzer, the LO placed by camera position, `measf` / `phasef` / `p_null`.
- `cophi_run.m` — the stages (bench · battery · noise · loop · capture ·
  format · figs) on `tg96_run`'s bones; `bench` and `battery` are in (2026-10-11), the rest
  follow one commit each (`BRIEF_to_cophi.md`, TO).  `cophi_run_batch.m` /
  `cophi_batch.sh` = the -batch wrapper and the memory-capped, serialized
  launcher (it waits on, and is waited on by, the other gauge batches).
- `tests/tCophiBench.m` — the bench gate (registration, null, contrast, and the
  must-fail per-index placement).
- Gate `tests/tCophi.m` (SUITE_FAST, 8): the algebra.

## Run it yourself

```matlab
cophi_run                                              % the sheet (cophi_params): lens rig, stage bench
cophi_run('bench.optics','oap','tag','bench_oap')      % the mirror rig OF RECORD (P.cophi.oap_record)
cophi_run('stages',{'bench','figs','clearance'})       % + tg96_run's layout render and part-by-part clearance
```
```bash
./cophi_batch.sh bench_lens "'stages',{'bench','figs','clearance'}"
./cophi_batch.sh bench_oap  "'bench.optics','oap','stages',{'bench','figs','clearance'}"
```
Outputs in `runs/<tag>/`: `<tag>_cophi_report.txt` (the COPHI rows), tg96_run's
`<tag>_report.txt`, `<tag>_test.in` (the test arm) and `<tag>_ref.in` (**the LO
arm**), `<tag>_vlayout.png` / `<tag>_clearance.png` (LO leg purple).  Gate
`tests/tCophiBench.m` (its own class, ~2 min: both rigs at model 1024).

## Stage bench (2026-10-11): the LO arm, and does it register?

**The LO arm** (`macos.design.twyman_green('ref_arm','lo')`, default `'flat'`
emits byte-identically to every bench before it -- checked on four
configurations against HEAD): a fiber tip behind the reference flat's seat, a
baffle and a plano collimator of L1's design (focal F1, the collimation solve's
figures, fed at its focus), its powered face AT the flat's vertex, so the LO
leaves the seat along the flat's return and the rest of the arm -- the
splitter's internal reflect, Recomb, the whole tail -- is the reference arm's,
element for element.  Fiber tip ~880 mm behind the seat on the old flat's axis;
nothing else is there (clearance worst +38.6 mm lens / +37.2 mm mirror, spec
25, the record's own node parts).  `P.bench.LO_DX` moves the fiber tip (the LO
drift knob: lateral = tilt d/F1, axial = defocus).

**COPHI's bench = the record's with no polarization phase-shifting:** the input
polarizer stays (a source polarizer), the arm QWPs and the output QWP are set
to zero retardance (they are then the record's 2 mm silica windows, so the
glass and the tail of record are unchanged), the analyzer is turned to pass the
test arm (solved from three traces).  The launches are SOLVED, not assumed:
the laser at the input polarizer and the LO's PM-fiber axis each maximize the
power through it (2x2 Gram matrix of the two basis launches, linearity checked
to 6e-16).  Without that the mirror rig's record launch arrives CROSSED with the
input polarizer (the OAP fold turns the source frame): 1e-6 of the power, the
reading pure leakage.  (The record's mirror-rig runs never met it: they all
pass `POL_IN 'source'`; the sheet's default does not, which is why
`P.cophi.oap_record` applies the record's mirror-rig settings -- POL_IN source,
AOIs 20/25, sides +1/-1, D_RC_L2 125 -- whenever `bench.optics` is 'oap'.)

**Finding: the geometric tail places the field BY RAY INDEX.**  Measured: a
passing ray k (k >= 2) sits on the OPD mask's k-1-th pixel in column-major
order, at OPDMat index + the centring offset (320 at N 1024 / 385), field
phase +k W in scalar mode (residual 1e-9 rad) and **-k W in vector mode on these
trains** (the Phase-3a conjugate, measured per deck).  The record's IFO can use
the per-index fields because its reference arm shares the test arm's
collimator -- both decks carry the same index -> camera-position map.  The LO
has its own collimator: on the lens rig the maps agree (the LO's is mirrored in
u, which a symmetric tail cancels -- null 0.0013 nm per index), but on the
mirror rig the test arm's off-axis parabola maps the source grid differently
from the LO's lens (same-index camera points 1.0 px rms apart, 2.4 px max)
and the image space carries ~1.6 um of fold aberration, so per index the null is
**681 nm rms** (by OPL; the field reading wraps it to 87 nm) against **0.45 nm by
camera position**.  So the LO is placed on the test arm's pixels by physical
camera position (`lo_on_test_`: the smooth part E exp(-i sg k W) and the path W
interpolated at the test rays' camera points).  One code path on both rigs.

| rig | chief, LO - test | LO footprint offset (overfill) | LO coverage, flatness | null on the flat DM (surface) | contrast g |
|---|---|---|---|---|---|
| lens | 0.000 px | 0.000 px (33.9 px) | 1.0000, 0.47 % rms | **0.000 nm** | 0.983 |
| mirror (of record) | 0.000 px | 1.625 px (33.9 px) | 1.0000, 0.21 % rms | **0.782 nm** (tilt 0.65, defocus 0.17, rest 0.41) | 0.982 |

REGISTRATION PASS on both.  The mirror rig's 1.6 px is the test pupil (the DM
STOP's image) off the chief, a property of the test arm, well inside the LO's
overfill.  Its 0.78 nm null is a fixed, smooth map subtracted by the frozen null
like the record's (the IFO's mirror-rig null of record is 12.9 nm); the 0.65 nm
of it that is tilt is the LO lens's beam direction against the parabola's.
Records `runs/bench_lens/`, `runs/bench_oap/`.

## Stage battery (2026-10-11): the H reading through the record's deck stage

`cophi_run('stages',{'bench','battery'})` runs `tg96_run`'s OWN placement and
deck stages with `P.reading_ctx = 'cophi_ctx'`: the heterodyne reading
(`dmg_cophi_frames` -> `dmg_cophi_recon`, N = 4, lo_ratio 1, noise-free) stands
where the record's polarization four-step stood, and nothing else changes --
the redo tail of record (`bench.tail_file` = `redo96_<optics>_tail.mat`, md5
identical to `redo96_<optics>ph`'s), the placement and its D1 gate, the lit set
(6948, the Sept-30 aperture cap), the hold-out actuator (62,62), the 112-site
grid, seed 7, the 30 nm working surface, the 64-state matrix ON it, the
estimator and lambda_m.  Mirror-rig placement reproduces `redo96_oapph`'s gate
to the digit (99.53 % within 2 px, median 0.87 px).  ~37 min per rig at model
1024.  Records `runs/battery_{lens,oap}/`.

Two I columns, labelled: **`redo96_<optics>ph` is the SAME surface** (same
lit set, hold-out, grid, tail, seed); `redo96_<optics>` is the deck's headline
rows, run BEFORE the aperture cap (7500 lit, 120 sites) -- the same seed drawn
over a larger set, i.e. a different surface.  gain / err pm / floor pm:

| row (30 nm surface, matrix on it) | H lens | I lens, same surface | H mirror | I mirror, same surface |
|---|---|---|---|---|
| single 10 nm at the hold-out | 0.9947 / 1.7 / 2 | 1.0001 / 1.4 / 1 | 0.9941 / 1.4 / 1 | 0.9832 / 2.7 / 2 |
| 1 nm on the 112-site grid | 0.9964 / 2.4 / 2 | 1.0046 / 2.2 / 2 | 0.9949 / 1.6 / 1 | 0.9838 / 2.9 / 2 |
| dense random 10 nm | 0.9967 / 180 | 1.0053 / 179 | 0.9950 / 118 | 0.9844 / 223 |

Capture range to 10 % (aging, matrix once on 30 nm): 480 nm + on both rigs,
grid and single site (holds at the last rung); re-measured matrix on 60-160
nm: gain 0.9964-0.9965 (lens), 0.9949 (mirror), corr 0.9998-0.9999.

**What it says.**  The heterodyne reading reproduces the IFO's rows on the same
surface: equal floors on the lens rig and half the floor and half the
dense-random error on the mirror rig, where the record's snapshot carries the
mirror rig's polarization cost (I 0.983) and H does not.  **The 0.5 % below unity is the ESTIMATOR'S, named (gain probe, lens rig,
`cophi_gain_probe.m`, `runs/gainprobe_lens/`):**

| probe | single | grid | rand10 | says |
|---|---|---|---|---|
| battery H (lambda_m 1e-3) | 0.9947 | 0.9964 | 0.9967 | -- |
| #1 synthetic 1.000 nm poke, no estimator: H vs the engine's own field change | slope **1.000000** | | | the READING is exact |
| #2 LO := the test arm's own flat field (engine-truth interferometer) | 0.9947 | 0.9964 | 0.9967 | not the LO placement |
| #3 scalar trace (polarization off; contrast 1.0000, +kW) | 0.9947 | 0.9964 | 0.9967 | not the vector sum / polarizers |
| #4 fit on the test beam's support, lit set forced equal | 0.9947 | 0.9962 | 0.9967 | not the mask |
| #5 lambda_m 1e-5 (100x less Tikhonov) | **1.0007** | **1.0022** | **1.0024** | it is the regularization's shrinkage |

So at the record's lambda_m 1e-3 an EXACT reading reads 0.995 through the
matrix estimator -- the brief's "+- the matrix bias", now a number.  Corollary
for the IFO record: the same-surface I column (same estimator, same lambda_m)
reads 1.000-1.005 on the lens rig and 0.983 on the mirror rig, i.e. its
READING carries +0.5..1 % (lens) and -1.1 % (mirror) relative to an exact one --
the snapshot's polarization gain the matrix does not fully absorb.  H is the
exact reading on both rigs.  (Mirror rig: #1-#5 not rerun; the battery's H rows
carry the same 0.994-0.995 signature.)

The deck stage's `flat-DM null` line reads 6.8 nm on H: it is the rms over the
whole mask, and outside the test pupil the H reading is a constant (no test
light), so the line measures the pupil's piston against that annulus.  The
honest in-pupil null is the bench stage's (0.000 / 0.782 nm).

## Stage noise (2026-10-11): pm per reading on every camera, both trains, the electronics

`cophi_run('stages',{'noise'})` re-makes the deck stage's 30 nm matrix and hands
its estimator to `cophi_ctx`'s noise hook (`battery.noise_ext`, opt-in in
`tg96_run`).  Each camera is mapped onto the model grid by BINNING: k = camera
pixels across the pupil (0.95 x format) / model pixels (316), a model pixel holds
k^2 camera pixels, well x k^2, read noise x k; the frame mean at the brightest
pixel sits at fill (0.5) x well.  The per-pixel phase noise matches tCophi's
formula sigma_phi^2 = 2 (I0 + r^2)/(N I0^2 g'^2) to **0.2 % on every camera,
both rigs, both trains** (g' = 0.9003 at N = 4).  Records `runs/noise_{lens,oap}
[_plain]/`; 12 draws.

**pm per reading = the piston-free noise of one actuator**: the reading's lit-set
PISTON is removed first (unobservable, inert on a DM -- the noise map showed it
is most of a raw single-actuator sigma: 165 pm raw vs 76 pm piston-free), then
the robust floor (1.48 MAD over the unpoked lit set, median over draws).  The
distribution is heavy-tailed (per-actuator p99 ~4-6x the median, from a few
actuators in some draws), so the plain std moves +-30 % between benches
(`runs/gated_v1/`, `runs/ungated/`); the robust floor agrees between the two
trains to 1-3 %.  A contrast gate on the reading (amp/dc > 0.2 x the pupil
median in both readings) is in and changes nothing measurable.

| camera (survey, verify before a slide) | px across pupil | e-/px/frame | readings/s | **pm per reading** lens / mirror | **s per pm** lens / mirror |
|---|---|---|---|---|---|
| baseline, as proposed (1k, 60 ke, 30 fps, read 5 e- ASSUMED) | 973 | 3.0e4 | 7.5 | **57.7 / 53.7** | **443 / 385** |
| Photonfocus MV1-D1024E-160 (200 ke, 220 e-) | 973 | 1.0e5 | 37.5 | 39.5 / 36.5 | 42 / 36 |
| pco.edge 5.5, true global shutter (30 ke, 2.2 e-) | 1946 | 1.5e4 | 12.5 | 40.2 / 38.3 | 129 / 117 |
| C-RED 2 InGaAs (1.4 Me-, read 30 ASSUMED) -- AT 632.8 nm, a placeholder for the 1.3 um stage | 608 | 7.0e5 | 150 | 19.6 / 18.7 | 2.6 / 2.3 |

(plain train: lens 57.1 / 39.9 / 40.0 / 19.7, mirror 54.1 / 36.4 / 37.6 / 19.6 pm.)
Feng's baseline: **~7 min of averaging per picometre** (443 s), in the first look's
6-9 min.  The Photonfocus's 220 e- read noise costs little because its well is
deep: 42 s per pm.

**The two trains** (CC's review note): the camera is WELL-bound, so the pm per
reading is the same on both (above).  What the record's polarizing train costs is
LIGHT: source -> camera throughput, test arm 0.271 (lens) / 0.265 (mirror) with
it against 0.408 / 0.392 without (x1.50 laser power for the same well), LO 0.396 /
0.389 against 0.464 / 0.452.  At the baseline the camera takes ~6.6e11 e-/s
(~0.2 uW at 632.8 nm): a mW laser covers either train many times over, so for
COPHI the polarizing optics buy nothing and cost nothing that matters -- Feng's
plain splitter is the layout to price, and the deck can say so.

**Electronics (campaign sec. 9.1), each term alone, noise-free, piston-free floor
of ONE reading (pm, lens / mirror):**

| term | value | pm |
|---|---|---|
| synthesizer / trigger jitter (beat phase per frame) | 1e-3 rad | ~19 / 19 |
| | 1e-2 rad | 197 / 189 |
| AOM drive amplitude ripple (rms fraction per frame) | 1e-3 | 24 / 26 |
| | 1e-2 | 242 / 265 |
| differential fiber walk within the scan | 0.01 rad/frame | 68 / 64 |
| | 0.05 rad/frame | 340 / 321 |

All linear in the term.  So to stay under the baseline's 55 pm photon floor per
reading: beat-phase jitter < ~3e-3 rad per frame (a DDS pair is ~1e-4 class),
drive ripple < ~2e-3, fiber walk < ~8e-3 rad per frame (133 ms): **the fiber is
the term** -- the brief's "row that prices fiber stabilization" -- and a 0.05
rad/frame fiber (unstabilized) is 6x the photon floor.

**The fiber budget in engineering units** (CC, 2026-10-11; the design statement
the campaign brief sec. 9.1 asks for).  The walk is the DIFFERENTIAL phase of the
test-arm and LO fibers (their common part cancels in the beat), single pass:

- 8e-3 rad per frame at 632.8 nm = lambda phi / 2 pi = **0.81 nm of differential
  fiber OPD per 33 ms frame** (the baseline's 30 fps), i.e. a differential OPD
  rate of **~24 nm/s**.  (At 0.05 rad/frame, unstabilized: 5 nm per frame, 150 nm/s.)
- Fiber thermal path coefficient k_T = (1/L) d(OPL)/dT = dn/dT + n alpha ~ **1.0e-5
  /K** for fused silica at 633 nm (dn/dT ~ 0.96e-5 /K, Leviton & Frey 2006; n alpha
  ~ 0.08e-5 /K), i.e. ~10 nm per metre per mK; a coated fiber's effective value is
  ~0.7-0.8e-5 /K (Hocker, Appl. Opt. 18, 1445, 1979).  **Both from memory, not yet
  verified against the sources -- check before a slide.**  We use 1.0e-5 /K (the
  stricter of the two).
- So the budget is a differential temperature RATE of 24 nm/s / (k_T L) = **2.4 mK/s
  per metre of uncommon fiber**: 2.4 mK/s for 1 m, 0.24 mK/s for 10 m (0.08 mK per
  frame per metre).
- The design statement: **the two fibers must be co-routed (the same cable, the
  same thermal path) so that their temperature DIFFERENCE changes by less than
  ~2.4/L mK per second over the L metres where they differ, or the differential
  phase must be servoed** (a reference pick-off of the fiber path, the classical
  heterodyne-metrology fix).  The loop stage's fiber row runs at this budget
  (8e-3 rad/frame) and at 3x it.

**Trains, carried forward:** the plain splitter is the priced layout; the record's
polarizing train costs x1.5 laser power and nothing else.

## What the gate already settled (tCophi, 2026-10-11)

- Four-bucket exact; the LSQ **is** the `atan2` form; N-bucket exact.
- Integrating bucket: contrast × sinc(π/4) = 0.9003 (N = 4), 0.9745 (N = 8).
- Step error: a 1° per-step error is the textbook (ε/2) sin 2φ ripple.
  **More buckets do NOT cure it** — the plain N = 8 LSQ with the same per-step
  error is 1.43× WORSE (the ramp accumulates); **knowing the steps does** (LSQ
  with the true centres is exact), which is exactly what a Rb-clocked
  frequency-set scan provides.  Feng's "N-frame backup" earns its keep against
  harmonics and noise, not against a per-step miscalibration.
- Photons: σ_φ = √(2 (I0 + r²) / (N I0² g'²)) per pixel — shot noise ∝ 1/√I0
  over two decades, read noise r added in quadrature to the shot term; the
  pm-per-photon rows follow from it with h = φ λ/4π.
- Fiber walk and trigger jitter within a scan: a phase-dependent error of a
  fixed fraction of the walk (0.13 w rms for w = 0.05 rad/frame), no better
  with N = 8 — the intra-scan drift has no algorithmic cure; it is a deck row.

## Rulings (Dave 2026-10-09 / 2026-10-11)

LO arm traced as a drop-in at the reference flat's seat; 632.8 nm for the
record with λ a parameter (1.3 µm = a stage: InGaAs camera, refocused tail);
three cameras named + read noise everywhere; **ruling 6 (2026-10-11): full actuator
resolution only — no coarse / low-order forms (EFC probes are commanded at the
full lattice); the second sensor class is a well-free lock-in / ToF / coherent-
array reading at ≥ 1 element per actuator, assumed to exist, scored on cadence and
photons per element** (survey `macos/NOTE_lockin_arrays_survey.md`); campaign
started 2026-10-11 ("then COPHI").
