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
- `tests/tCophiBench.m` — the bench gate, 9 legs (~15 min, its own class):
  registration, null, contrast, the must-fail per-index placement; and the
  battery legs -- a 1 nm poke through the matrix (one-step 0.9949, converged
  1.0010), the LO-tilt absorption MEASURED (a 1 nrad LO tilt reads as a DM SURFACE
  tilt of 1.008 x theta/2, 0.23 pm left after the plane; 0.467 in COMMAND units,
  the influence functions' overlap), and the red legs: no interference and a dark
  camera (well 0) are refused BY NAME (`cophi:noFringe`, amp/dc under the contrast
  floor `P.cophi.contrast_floor`, default 0.05) rather than read as a zero phase.
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

**Electronics, restated under the 'median' annulus fill (RULED 2026-10-12, the COPHI
reading's convention):** budgets (= the baseline camera's photon floor per reading),
lens / mirror: beat jitter **3.0e-3 / 2.8e-3 rad per frame**, AOM ripple **1.9e-3 /
1.8e-3**, differential fiber walk **4.3e-3 / 3.9e-3 rad per frame** (`runs/noiseX_
{lens,oap}_plain_med`; the 'zero' rows below moved <= 9 %).  The fiber budget in
engineering units: **0.39-0.43 nm of differential OPD per frame, ~12-13 nm/s, ~1.2-1.3
mK/s per metre of uncommon fiber** (k_T 1e-5/K, to verify).

**Electronics (campaign sec. 9.1): CORRECTED 2026-10-12 to the exact expectation.**
Each term is linear and enters as a per-frame offset (a phase offset of the
interference for beat/trigger jitter and the differential fiber walk; an
amplitude factor on the LO for AOM ripple), so the noise hook now computes each
frame's response through the estimator once and takes the expectation from the
term's covariance -- no Monte Carlo noise (`runs/noiseX_{lens,oap}_plain/`, the
priced layout).  **The first noise records' walk row (68 / 64 pm at 0.01
rad/frame) read 2.3x LOW:** a reading's whole walk is three random numbers, so a
twelve-draw mean spans +-50 % (2000 draws measured), and that seed set happened
to sit low; the jitter and ripple rows were within sampling.  The loop exposed it
(its fiber row cost 2.5x the prediction) and two independent forms then agreed
with the exact value (one reading vs noiseless 146 pm, differential/sqrt 2 139 pm,
exact 130 pm at 8e-3).  Piston-free error of ONE reading per unit term, lens / mirror:

| term | pm per unit | **budget = the baseline's photon floor per reading** (57 / 54 pm) |
|---|---|---|
| beat / trigger jitter | 1.96e4 / 1.99e4 pm per rad | **2.9e-3 / 2.7e-3 rad per frame** (a DDS pair is ~1e-4 class) |
| AOM drive ripple | 3.1e4 / 3.4e4 pm per unit | **1.8e-3 / 1.6e-3 rms per frame** |
| differential fiber walk | 1.62e4 / 1.60e4 pm per rad/frame | **3.5e-3 / 3.4e-3 rad per frame** (was 8e-3) |

**The fiber budget in engineering units** (CC, 2026-10-11; corrected 2026-10-12).
The walk is the DIFFERENTIAL phase of the test-arm and LO fibers (their common
part cancels in the beat), single pass:

- 3.5e-3 rad per frame at 632.8 nm = lambda phi / 2 pi = **0.35 nm of differential
  fiber OPD per 33 ms frame** (the baseline's 30 fps), a differential OPD rate of
  **~10.6 nm/s**.  (At 0.05 rad/frame, unstabilized: 5 nm per frame, 150 nm/s, 14x.)
- Fiber thermal path coefficient k_T = (1/L) d(OPL)/dT = dn/dT + n alpha ~ **1.0e-5
  /K** for fused silica at 633 nm (dn/dT ~ 0.96e-5 /K, Leviton & Frey 2006; n alpha
  ~ 0.08e-5 /K), i.e. ~10 nm per metre per mK; a coated fiber's effective value is
  ~0.7-0.8e-5 /K (Hocker, Appl. Opt. 18, 1445, 1979).  **Both from memory, not yet
  verified against the sources -- check before a slide.**  We use 1.0e-5 /K (the
  stricter of the two).
- So the budget is a differential temperature RATE of 10.6 nm/s / (k_T L) = **~1.1
  mK/s per metre of uncommon fiber**: 1.1 mK/s for 1 m, 0.11 mK/s for 10 m.
- The design statement: **the two fibers must be co-routed (the same cable, the
  same thermal path) so that their temperature DIFFERENCE changes by less than
  ~1.1/L mK per second over the L metres where they differ, or the differential
  phase must be servoed** (a reference pick-off of the fiber path, the classical
  heterodyne-metrology fix).

**Trains, carried forward:** the plain splitter is the priced layout; the record's
polarizing train costs x1.5 laser power and nothing else.

## Stage loop (2026-10-12): the hold servo at the camera's cadence

`cophi_run('stages',{'loop'})`: the shared `dmg_loop` on the H reading, the
record's loop knobs (g 0.5, K 60, seed 77; walk 2 pm, thermal 5 pm, camera walk
0.13 per CYCLE), the matrix measured on the 30 nm set point, the plain splitter
(the priced layout), one-step AND converged.  Each cycle averages M readings so
the hold reaches ~3 pm (M 120 baseline -> a cycle every 16 s; M 60 Photonfocus ->
1.6 s), an equivalent camera of well x M and read x sqrt M, the intra-scan terms
scaled 1/sqrt M (each reading's error is linear in them -- 0.253 w rms phase for
the walk, measured -- and independent).  New in `dmg_loop`: `opt.delay` (sensor
delay, gated in tDmgLoop G15) and the piston-free `ss_pf / bias_pf / sig_n_pf`
beside the record's metric.  Records `runs/loop_{lens,oap}_plain/` (205 min each).

Hold error ss, pm (the record's metric = piston-free on every row but the fiber's;
one-step -- the converged estimator holds the same to 0.1 pm on every row):

| row | baseline lens / mirror | Photonfocus lens / mirror |
|---|---|---|
| noise only | **2.75 / 2.67** | **2.60 / 2.52** |
| random walk 2 pm/cycle | 3.60 / 3.53 | 3.48 / 3.43 |
| thermal ramp 5 pm/cycle | 10.3 / 10.3 (the lag rate/gG, every gauge) | 10.3 / 10.3 |
| camera offset walk | 2.85 / 2.77 (zero-sum: immune) | 2.66 / 2.60 |
| one cycle of sensor DELAY | 3.65 / 3.54 | 3.44 / 3.36 |
| LO 0.1 nrad tilt + 1 pm rms defocus per cycle | **29.6 / 29.4** | 29.5 / 29.3 |
| fiber walk 8e-3 rad/frame (piston-free; lens only on the fixed feed) | 7.8 / -- | 10.6 / -- |
| fiber walk 2.4e-2 rad/frame (piston-free; lens only on the fixed feed) | 22.0 / -- | 31.0 / -- |
| fiber at the CORRECTED budget 3.5e-3 rad/frame (piston-free) | **4.21 / 3.95** (record metric 6.10 / 5.35) | -- / **4.84** |
| fiber at 3x the corrected budget (piston-free) | 9.94 / 9.16 (record metric 16.6 / 14.2) | -- / 12.65 |

What it says.  Feng's camera holds 3 pm, but at one cycle per 16 s; the
Photonfocus holds it 10x faster.  The thermal ramp's 10 pm is the proportional
loop's lag, the same for every gauge (a faster or higher-gain low-order loop is
the cure, not light).  A full beat of sensor delay costs ~0.9 pm.  **The LO is a
term the loop cannot see past: it reads LO drift as DM drift and drives the DM to
follow it** -- 0.1 nrad of LO tilt per cycle is ~30 pm of hold error, so the LO's
pointing must hold ~10 prad per cycle (a fiber-tip lateral stability of ~10 pm
per cycle at F1 857 mm) or be referenced; the brief's 1 urad per cycle would be
~24 nm of DM tilt per cycle (the run would only show the DM chasing the LO).  The
fiber walk is ~30-40 % piston (common phase), the rest a real figure term; at the
corrected budget (`runs/loopfib_*`) it adds as much noise per cycle as the photons
do, so the hold rises by sqrt 2 (2.7 -> 4.2 pm), exactly the linearity prediction
(4.1 pm) -- the budget is right, and to keep a 3 pm hold with the fiber in, either
the fiber is held ~2x tighter than the budget or the cycle averages ~2x longer.

## Stage capture (2026-10-12): from cold -- temporal unwrapping and two colours

`cophi_run('stages',{'capture'})` (`cophi_capture.m`), plain splitter, both rigs,
noise-free and on each camera; truth = the engine's OPD change per pixel (a ray
path, unwrapped; the H reading equals -dOPD/2 to 3e-7 nm noise-free at 30 nm).

**Two colours, frequency-multiplexed, ONE reading.**  lambda_1 632.8 nm at a 7.5 Hz
beat (4 frames per beat) and lambda_2 652.8 nm at 3.75 Hz (8 per beat) on the SAME 8
frames at 30 fps -- each colour read by `dmg_cophi_recon` at its own bucket centres
(the harmonics are orthogonal over 8 frames: crosstalk 4e-16 rad); lambda_2's fields
are a RE-TRACE of both decks.  Synthetic Lambda = 20.65 um, range +-5.2 um.  On the
battery's capture ladder (30..480 nm rms):

| surface rms | single colour: wrong-order pixels | two colours: wrong orders, rms err (baseline / Photonfocus noise) |
|---|---|---|
| 30 nm | 0 | 0, 0.15 / 0.10 nm |
| 40 nm | 0.03 % | 0 |
| 80 nm | 5.7 % | 0 |
| 160 nm | 34 % | 0 |
| 480 nm | 74 % | **0**, 0.15 / 0.10 nm (both rigs) |

So a 200 nm rms DM figure from cold (peaks ~0.8-1 um) is captured in ONE
multiplexed reading (8 frames, 0.27 s at 30 fps) -- the sec. 9.2 ladder needs ONE
rung, not two, for this job.  The arithmetic (range +-Lambda/4, amplification
Lambda/lambda, fine requirement sqrt2 sigma Lambda/lambda < lambda/8):

| lambda | dlam | Lambda | range | amplification | fine sigma required |
|---|---|---|---|---|---|
| 632.8 nm | 1 nm | 401 um | +-100 um | 634 | 0.088 nm |
| 632.8 nm | 20 nm | 20.7 um | +-5.2 um | 32.6 | 1.7 nm |
| 632.8 nm | 100 nm | 4.6 um | +-1.2 um | 7.3 | 7.6 nm |
| 1319 nm | 20 nm | 88 um | +-22 um | 67 | 1.7 nm |

The reading's per-pixel noise is ~0.1-0.15 nm per multiplexed reading, so any dlam
from 1 to ~100 nm resolves the fringe in one reading; the 1.3 um row (analytic; the
C-RED 2 a placeholder) doubles the single-colour range (+-330 nm) and the synthetic.
**Model caveat:** the bench lenses carry a FIXED index (1.5), so the lambda_2
re-trace cannot show the refractive tail's chromatic refocus; for fused silica (dn
~ -5.6e-4 over 20 nm, to verify) that is ~1.2e-3 of focal length, ~0.5 mm on the
focuser -- its static part is frozen in each colour's own null; its effect on the
pupil image's sharpness is open (the mirror rig is achromatic but for the field lens).

**Temporal unwrapping** (a drifting surface, 6 readings, per pixel): holds at 0.9 x
lambda/4 per reading, fails at 1.1 x (both rigs) -- so it holds to a drift of
**1.19 um/s at the peak pixel at the baseline's 7.5 readings/s** (0.24 um/s of rms
for a 30 nm-class shape) and **5.9 um/s at the Photonfocus's 37.5/s** -- a running
loop never comes near it; a cold start needs the two colours.

**Mirror-rig plain-train fix (2026-10-12):** the record's P.oap.SRC_TRIM (-0.37 mm)
was solved WITH the input polarizer's plate in the diverging leg; the plain train
has no plate, so carried over it fed OAP1 off its focus (an 84 nm DEFOCUS null).
`cophi_run` now sets SRC_TRIM 0 for the plain mirror rig (null 0.746 nm).  Rerun
on the fixed feed: capture identical, the exact-noise floor 54.1 -> 53.2 pm, walk
budget 3.4e-3 -> 3.6e-3 rad/frame (`runs/oap_plain_misfed/` keeps the old).  The
plain-mirror LOOP was RERUN on the fixed feed (2026-10-12, `runs/loop_oap_plain`,
205.7 min, its fiber rows at the corrected budget): every row within ~0.02 pm of the
mis-fed run (`runs/oap_plain_misfed/` keeps it), the table above is the fixed feed.

## Stage format (2026-10-12): the lock-in class (Dave's ruling 6)

`cophi_run('stages',{'format'})`: the sheet's lock-in (CC f179513: the survey's
rank 1, a heliCam C4 class -- 512 sites across the pupil = 5.45 per actuator pitch,
500 ke- well, 1 kHz I/Q, 2.5e9 e-/s/element; unverified), each flux a 'camera' to
the noise hook (one reading per I/Q sample; the per-pixel phase noise matches the
formula to 0.1 %).  pm per I/Q sample per actuator (piston-free robust floor):

| flux e-/s/element | e- per sample | pm per sample | pm in 1 s | s per pm |
|---|---|---|---|---|
| 1e8 | 1e5 | 53 | 1.7 | 2.8 |
| 1e9 and up, WELL-CAPPED at 1 kHz | 2.5e5 (the well's 0.5 fill) | 34 | **1.07** | **1.1** |
| 1e9, well-free (the first run, uncapped) | 1e6 | 17 | 0.54 | 0.29 |
| 1e10, well-free | 1e7 | 5.4 | 0.17 | 0.029 |

So the lock-in class, as the sheet states it, gives **~1 pm per second -- 40x the
Photonfocus (42 s per pm) and 400x Feng's baseline (7 min)** -- and is WELL-bound at
1 kHz from 1e9 e-/s: its per-pixel well, not the laser, sets the time; a faster I/Q
output (or the deeper on-chip integration a lock-in can do) buys the well-free row.
"Well-free" (ruling 6's assumption) is therefore a property to verify on the device
sheet, not a given.  No phasemeter floor is in the sheet; the stage adds one in
quadrature when it is.

**What the deck says (one line):** the lock-in camera as surveyed is well-capped at
its 1 kHz I/Q rate -- **1.1 s per pm, already ~40x faster than the Photonfocus
camera (42 s) and ~400x Feng's baseline (7 min)**; a verified well-free device
(continuous demodulation, or an I/Q output >= 10 kHz) would add another ~4-40x
(0.29 s per pm at 1e9 e-/s/element, 29 ms at 1e10).  [CC's proposed wording had "only a
verified well-free device beats the Photonfocus" -- the record says the capped
device already does, by 40x.]

**The vendor question that decides the well-free row (to Heliotis, through Dave):**
(1) the per-pixel PHASE noise against flux at an integration time tc = 100 ms; and
(2) whether the full-well capacity bounds each INTERNAL demodulation sample (the
1 MSa/s on-chip rate) or the 5.6 kSa/s OUTPUT frame -- the first makes the sensor
effectively well-free at our fluxes, the second caps it as modelled here.

## The lock-in camera as a READING (2026-10-12): every table, a record

`'cophi.iq',true` (`dmg_cophi_frames` opt.iq / `dmg_cophi_recon` 'iq' /
`dmg_cophi_camera` cam.iq -- CC-reviewed): a TRUE I/Q sample per reading, no
integrating bucket, per-sample sigma_phi^2 = 2 (I0 + r^2)/(I0^2 g^2) (the per-pixel
noise matches it to 0.1 %, every run); the heliCam C4-class sheet (512 sites, 500
ke well, 1 kHz I/Q; read 0 = `lockin`, 100 e- per sample ASSUMED = `lockin_r`); the
frame mean capped at 0.5 x the well (one store per sample).  Plain splitter,
'median' annulus fill.  **Footnote on every lock-in row: x1.11 in sigma for
square-wave (two-tap) demodulation** (sqrt(pi^2/8); the heliCam steers charge
between two stores); the vendor question settles it.

**Correction (2026-10-12): the format stage above OVERCOUNTED the lock-in ~3x in
photons** -- it modelled each 1 kHz sample as four bucket frames, each a full well;
a true I/Q sample integrates into ONE store.  Its 34 pm / 1.1 s per pm is superseded
by the rows below.

| table | lock-in, read 0 (read 100 e-) | record |
|---|---|---|
| noise: pm per 1 ms sample, s per pm | mirror **57.9 pm, 3.35 s** (63.1, 4.0); lens 61.1, 3.73 (66.1, 4.37) | `runs/lockin_noise_oap_med`, `lockin_noise_lens` |
| electronics (exact) | jitter / ripple / fiber walk **0** piston-free (a common phase per sample) | same |
| battery (noise-free) | single / grid / rand10: one-step 0.9941 / 0.9950 / 0.9951, converged **1.0000 / 1.0006 / 1.0005** (= the camera reading's: both exact) | `runs/lockin_battery_oap` |
| capture | two colours on ALTERNATE samples: **0 wrong orders to 480 nm rms**, 0.12 nm per pixel; temporal unwrapping to **158 um/s** at the peak pixel (1 kHz) | `runs/lockin_capture_oap` |
| loop, 44 samples per cycle (44 ms) | noise 5.22 (5.32), walk 5.71, thermal 11.4, cam 5.22, LO 29.4, delay (1 cycle = 44 ms, the upper bound) 6.94 | `runs/lockin_loop_oap`, `lockin_loop_oap_r` |
| loop, 125 samples per cycle (125 ms) | **the 3 pm hold: 3.09 pm** (converged 3.14); one cycle of delay 4.12 | `runs/lockin_loop_oap_m125` |

**So the 3 pm hold: the lock-in in 0.125 s cycles (125 samples), the Photonfocus in 1.6
s, Feng's baseline in 16 s.**

**For a lock-in the fiber term is a per-sample piston, inert:** the fiber row at the
budget (3.5e-3 rad per sample) and at 3x reads **5.22 pm = the noise-only hold to the
digit** (piston-free), against the camera four-bucket's 3.95 pm -> 9.2 pm, whose fiber
term exists because its four frames are sequential.  (Under the record's 'zero'
annulus fill the same common phase read 7.3 pm per mrad -- the pupil-edge step;
`runs/lockin_zerofill`, kept as the must-fail evidence.)

**Well-free rows (ANALYTIC, one store per sample, from the 57.9 pm record):** 1e8
e-/s/element 92 pm, 8.4 s per pm; as sheeted (capped from 2.5e8) 58 pm, 3.35 s; 1e9
29 pm, 0.84 s; at the sheet's own 2.5e9 18 pm, 0.34 s; 1e10 9.2 pm, 0.084 s.

## The other gauges on a polarization camera (2026-10-12): "seconds to SNR 1 at 1 pm"

The composite deck's electronics row, through each gauge's OWN reading (CC's request;
Dave: a polarization camera for the IFO and the vector ZWFS).  Opt-in camera modes
`tg96_run` `battery.noise_cam` and `zwfs_run` `noise.cam` (records unchanged):
IMX250MZR class (polarization, 2448 x 2048, 75 fps, **well ~10 ke = the IMX253 pixel,
UNCONFIRMED; read 2.4 e- ASSUMED**); the stepped ZWFS and the pinhole on the same
sensor in MONO (my assumption).  Binned onto the model grid, brightest pixel at 0.5
x well, Poisson + read, the matrix on the 30 nm surface, piston-free robust floor:

| gauge (rig) | reading on the camera | pm per reading | readings/s | **s per pm** | run tag |
|---|---|---|---|---|---|
| IFO snapshot (lens) | 4 analyzer frames in ONE polarization exposure | 119 | 75 | **189** | `tg_psi_dm96_oap/runs/camI_lens` |
| IFO snapshot (mirror) | same | 110 | 75 | **160** | `camI_oap` |
| stepped ZWFS S (mirror) | 4 sequential mono frames | 94 | 18.8 | **475** | `zwfs_dm96/runs/camS_oap` |
| vector ZWFS V (mirror) | the pair in ONE polarization exposure | 197 | 75 | **519** | `camS_oap` |
| pinhole P (mirror) | 4 sequential mono frames | 80 | 18.8 | **342** | `camS_oap` |

The sensors are 10-40x slower than photons-for-1-pm / the sensor's full photon rate:
that arithmetic assumes every pixel full at every frame, where a ZWFS image's bright
pixel sets the exposure (its dark structure then wastes well), the fill is 0.5, and
the polarization camera's two unused orientations are lost.  A first cut of the
sensor rows used a pupil 98 px across (a threshold the dimple structure fools) where
the record's is 157 and read 2.5x too many photons (`runs/camS_oap_wrongpupil`,
withdrawn).

## Stage figs (2026-10-12): the station figure

`cophi_run('stages',{'figs'})` (`cophi_figs.m`): the mirror rig, plain splitter,
noise-free, the deck's format -- the command, the test and LO fields at the camera,
two of the four buckets, the recovered surface and recovered minus the engine's own
OPD.  `runs/figs_oap_plain/figs_oap_plain_stations.png`: the flat DM reads 0; the
30 nm working surface (the DM's full lattice, seed 7) reads **31.95 nm rms, read -
engine 0.31 pm rms**.

## The converged estimator (2026-10-11): every gauge's accuracy row, bias removed

Dave's ruling: the regularization is driven to CONVERGENCE, not dropped.
`dm_gauge_lib/dmg_converge.m` iterates the one-step Tikhonov estimate on its own
residual through the forward model (pupil_blur_demo's iterate_);
`tg96_run` `battery.est_mode` 'onestep' (the record) | 'converged' | 'both' wraps
the measured-matrix estimator with J as the forward model; `dmg_act_fit`
`opts.converge` does the same with its stencil.  Gate `tests/tConvergedEst` (4,
pure MATLAB, SUITE_FAST candidate): one-step on an exact reading < 0.999 (the red
leg), converged 1.0000 +- 1e-4 on both forward models, and the noise trade
MEASURED -- converged to a zero update the white-noise floor is x2.1 the
one-step's; converged to the one-step floor it stops in 2 steps.  Pinned
property (C5): `dmg_act_fit`'s unknowns are ALL actuators while its data are lit-
masked, so with an unlit border its converged gain PLATEAUS (0.962 at w 1.0) --
the stencil estimator's unknown set, not the iteration; it matters for the ZWFS
gauges' re-score.

**Re-score, noise-free, est_mode 'both', rows on the same 30 nm surface** (H =
`runs/conv_H_{lens,oap}`, I = `tg_psi_dm96_oap/runs/conv_I_{lens,oap}ph`; the
one-step rows reproduce `battery_*` and `redo96_*ph` TO THE DIGIT -- the record
path is unchanged; 5-7 steps, last update <= 0.001 pm):

| gain: converged (one-step) | single 10 nm | grid 1 nm | dense random 10 nm |
|---|---|---|---|
| H lens | 1.0008 (0.9947) | 1.0022 (0.9964) | 1.0024 (0.9967) |
| H mirror | **1.0000** (0.9941) | 1.0005 (0.9949) | 1.0005 (0.9950) |
| I lens | **1.0064** (1.0001) | **1.0106** (1.0046) | **1.0112** (1.0053) |
| I mirror | **0.9887** (0.9832) | **0.9892** (0.9838) | **0.9896** (0.9844) |

So with the shrinkage gone **the record interferometer's polarization gain is in
the open: +0.6 to +1.1 % on the lens rig, -1.1 % on the mirror rig** (I/H
converged: lens 1.0056 / 1.0084 / 1.0088, mirror 0.9887 / 0.9887 / 0.9891), where
the one-step rows read 1.000 on the lens rig only because it cancelled the 0.5 %
shrinkage.  H is exact on the mirror rig; on the lens rig a +0.1-0.2 % remains
(the same at lambda_m 1e-5), a lens-rig matrix-model term (the box window's
truncation of the influence wings is the first candidate) -- open, small.
Errors and floors fall too (H mirror dense random 118 -> 27 pm).

## The resolution sweep (2026-10-13): what pixels per actuator buy and cost, per gauge

Brief `macos/BRIEF_to_resolution_sweep.md` (Dave, deck_cophi review: "price resolution
for the IFO, vZWFS and HET").  Runners `ressweep_run` (IFO, COPHI bucket, COPHI lock-in)
and `ressweep_zrun` (stepped ZWFS S, vector ZWFS V, pinhole P -- one zwfs_run per
sampling reads all three); hooks `ressweep_hook` / `ressweep_zhook`; composite figure
`ressweep_figs`; gate `tests/tResolutionSweep` (SUITE_FAST-sized: the operator + the
records).  Records `runs/ressweep_<gauge>/ressweep.txt` (+ the pruned `*_results.mat`).

```matlab
ressweep_run('gauge','ifo')          % or 'het', 'het_iq'; 's',[3.277 2 1.5 1 4] by default
ressweep_zrun                        % S V P at model 2048: 's',[3.30 2 1.5 1 4]
ressweep_zrun('s',1.65)              % the native point-sampled 1.65 at model 1024 -- its OWN process
ressweep_figs                        % runs/ressweep_figs/ressweep_composite.png + .txt
```
```bash
./ressweep_batch.sh ressweep_ifo "'gauge','ifo'"      # serialized, memory-capped (as cophi_batch)
./ressweep_zbatch.sh ressweep_zwfs                    # model 2048 (MemoryMax 20G), ~75 min per sampling
./ressweep_zbatch.sh ressweep_zwfs165 "'s',1.65"      # model 1024 -- NEVER in the same process as a 2048 run
```
(One engine model size per MATLAB process: the first sweep went 3.30 -> 1.65 in one
process and died `free(): invalid next size` -- zwfs_run's documented rule; ressweep_zrun
now refuses the mix.)
```bash
```

**Mechanism -- the camera's PIXELS, not a coarser model.**  The model's detector grid
POINT-samples the pupil image (one ray per model pixel; 3.277 model px per actuator on the
mirror rig at NGRID 385).  A coarser camera INTEGRATES over each pixel.  So for every
sampling at or below the model's, `dm_gauge_lib/dmg_camres` box-integrates every detector
FRAME onto camera pixels of c = 3.277/s model px (area-weighted, fractional overlaps for
a non-integer c) and replicates them back onto the model grid.  It is switched on by
`P.camres` in `tg96_run` / `zwfs_run` and applied where the frames are made (`tg96_run`
`frame()`, `cophi_ctx`'s frames, the ZWFS/PDI factories' frames), so the mask, the
placement, the 64-state matrix and every battery row run THROUGH the resampled reading:
blur and registration are absorbed by calibration, as on a bench.  The ZWFS / PDI solvers'
intensity calibrations (the flat's |E0|^2, the reference's |b|^2, the stepped b2cal) are
taken on the same pixels -- they are images a bench takes on its own camera; the
reference wave's PHASE stays per model pixel (smooth on the cell scale).  c = 1 is the
exact identity (the record's code path bit for bit).  ABOVE the model's sampling (4 px)
there is nothing to integrate: a finer SOURCE grid (NGRID 470 / 467), c = 1.  Why not a
coarser NGRID for s < 3.3: that point-samples (a 1 px/act ray grid sees each actuator at
one point, no pixel blur) -- the ZWFS's native 1.65 (NGRID 193, the camera records' grid)
is run as exactly that contrast.  Replication weights each camera pixel by the model
pixels it covers (uniform to +-1 px at a non-integer c): the estimator is then a mildly
weighted least squares on the camera pixels -- stated, not hidden.

**The other rows, per sampling (the hooks):** the two working-surface states (A0, A0 + the
10 nm single at the hold-out) are traced ONCE unbinned and binned in the hook, so the
noise and the registration rows cost no traces.  NOISE: the gauge's record camera with
the pupil image as recorded, binned to the sampling -- a camera pixel holds (k c)^2 sensor
pixels, so the PHOTONS per reading do not change with the sampling (every sensor pixel
is used; exposure by the brightest model pixel at fill x well x k^2, the record's rule);
read noise two ways: DIGITAL binning (CMOS; read x k c per camera pixel = the same total
read variance per actuator at every sampling; the record's convention) and ON-CHIP charge
binning (read x 1 -- "what binning restores").  Metric = the record's (piston-free robust
floor, one reading = differential / sqrt 2).  LOOP: dmg_loop on a LINEAR instrument (truth
= this sampling's measured matrix, the camera's noise on the binned set-point frames, a
cycle of M readings), noise only and the record's 5 pm/cycle thermal ramp, g 0.5, K 60,
seed 77, at the record's cycle (M: Photonfocus 60 = 1.6 s, Feng's 120 = 16 s, lock-in 125 =
0.125 s; the IFO's 450 = 6 s and the ZWFS's S 330 / V 1435 / P 237 are the cycles that hold
~3 pm at the record's pm per reading -- no camera loop record existed).  On the anchor the
loop sits on dmg_loop's theory line (IFO 3.05 vs 3.05 pm).  REGISTRATION: the camera grid
shifted +-0.1 pitch in x and y AFTER calibration, read through the unshifted matrix.  DATA:
output pixels per frame x 16 bit x frame rate.

**Anchors reproduce the records to the digit** (tResolutionSweep R4): IFO 3.277 =
redo96_oapph / camI_oap (0.9832/2.7/2, 0.9838/2.9/2, 0.9844/222.9; converged 0.9887 /
0.9892 / 0.9896 = conv_I_oapph; pol camera 110 pm, 161 s per pm vs 110 / 160); HET 3.277 =
lockin_battery_oap (0.9941/1.4/1, 0.9950/1.6/1, 0.9951/117.6; converged 1.0000 / 1.0006 /
1.0005); Feng's camera 52.9 pm (record 53.2), lock-in 58.9 pm / 3.47 s per pm (57.9 /
3.35; 12 draws).  4 px = 3.3 px on every row (the finer source grid changes nothing).

### The interferometers (mirror rig of record; records runs/ressweep_{ifo,het,het_iq})

Battery, noise-free (converged gain; dense-random error converged, pm):

| px/act | IFO single / grid / rand err | HET single / grid / rand err |
|---|---|---|
| 4.0 | 0.9884 / 0.9892 / 138 | 0.9999 / 1.0005 / 27 |
| **3.28** | 0.9887 / 0.9892 / 139 | 1.0000 / 1.0006 / 28 |
| 2.0 | 0.9876 / 0.9893 / 149 | 0.9993 / 1.0005 / 33 |
| 1.5 | 0.9883 / 0.9883 / 169 | 0.9991 / 0.9996 / 61 |
| 1.0 | 0.9943 / **0.9562** / **1172** (floor 13 pm one-step) | 1.0029 / **0.9630** / **1123** (floor 13 pm) |

Time and hold (one reading, digital binning; hold = the loop at the record cycle, noise only, pm):

| px/act | IFO pol camera: s per pm / hold (6 s) | HET Photonfocus: s per pm (on-chip) / hold (1.6 s) | HET Feng's: s per pm / hold (16 s) | HET lock-in C4: s per pm / hold (0.125 s) |
|---|---|---|---|---|
| 4.0 | 162 / 3.06 | 36.6 (26.8) / 2.51 | 389 / 2.65 | 3.36 / 3.08 |
| **3.28** | 161 / 3.05 | 36.2 (26.5) / 2.52 | 373 / 2.66 | 3.47 / 3.09 |
| 2.0 | **266** / 3.95 | **68.6** (48.7) / **5.28** | 691 / 3.46 | **5.44** / 4.02 |
| 1.5 | 302 / 4.21 | 73.5 (50.3) / 7.55 | 797 / 3.70 | 6.28 / 4.26 |
| 1.0 | 758 / 8.43 | 292 (212) / 26.5 | 3400 / 7.42 | 15.9 / 8.65 |

Registration, +-0.1 pitch after calibration (worst of four; converged gain, noise-free
floor): IFO 0.939 / 17 pm at 3.28, 0.956 / 16 pm at 2, 0.963 / 18 pm at 1.5, 0.883 / 44
pm at 1; HET 0.950 / 17, 0.967 / 16, 0.974 / 18, 0.890 / 46.  **No shift at any sampling
reads the wrong actuator** (largest neighbour 0.10-0.12 of the poke, 0.22-0.23 at 1 px).

Data (pol camera, all orientations): 0.79 / 0.53 / 0.20 / 0.11 / 0.05 Gb/s at 4 / 3.3 / 2 /
1.5 / 1 px at 75 fps; the lock-in's 1 kHz 1.8 Gb/s at 3.3; pupil pixels per reading
1.2e5 / 7.8e4 / 2.9e4 / 1.6e4 / 7.3e3.

### The Zernike family (stepped S, vector V, pinhole P; records runs/ressweep_zwfs)

The zwfs bench of record (redo96_oapsens: model 2048, NGRID 385 = 3.30 px/act); the
camera's pixels for 2 / 1.5 / 1; NGRID 467 for 4; and the camera records' own grid --
NGRID 193, model 1024, a POINT-sampled 1.65 -- as the control.  Anchor = redo96_oapsens
to the digit on all nine rows (tResolutionSweep R4z).  V and P camera rows: the
polarization camera (V's pair in one exposure) and the mono sensor (S's four frames, P's
four), each image's brightest pixel at fill x well (the record's camera form).

| px/act | single floor, pm (S / V / P; one-step) | s per pm, S / V / P | hold, pm (S 17.6 s / V 19.1 s / P 12.6 s) | +-0.1 pitch: worst gain, floor |
|---|---|---|---|---|
| 4.0 | 5 / 2 / 3 | 456 / 655 / 400 | 3.01 / 3.42 / 3.36 | 0.94-0.95, 21 pm |
| **3.30** | 5 / 2 / 3 | 451 / 639 / 398 | 3.02 / 3.41 / 3.37 | 0.93-0.94, 21 pm |
| 2.0 | 6 / 3 / 4 | 723 / 1040 / 681 | 3.90 / 4.39 / 4.35 | 0.94, 24 pm |
| 1.5 | 6 / 3 / 5 | 812 / 1150 / 734 | 4.11 / 4.71 / 4.58 | 0.95-0.96, 24 pm |
| 1.0 | **33 / 32 / 32** (gain 0.91-0.92) | 2380 / 3090 / 2190 | 8.44 / 9.06 / 9.38 | 0.85, 58 pm |
| 1.65 POINT-sampled | 5 / 2 / 3 | 440 / 629 / 402 | 2.98 / 3.40 / 3.29 | 0.88-0.91, 19 pm |

Converged gains (dmg_converge through the measured J) stay within 1.5 % at every sampling
but 1 px (grid row 0.95-0.96).  No shift reads the wrong actuator.  Data (V, pol camera):
0.78 / 0.53 / 0.19 / 0.11 / 0.05 Gb/s; pupil pixels per reading 1.2e5 / 7.8e4 / 2.9e4 /
1.6e4 / 7.2e3 -- V and P solve every one of them six times (the reference wave's
iterations, two FFTs of the full grid each), so the ZWFS's processing scales with the
pixel count where the interferometers' atan2 is trivial.

**What it says -- the deck's inferred rows, against the records.**
1. **"Above ~2 px the photons and the hold are free" is WRONG.**  The photons per reading
   are the same at every sampling (the same sensor, binned), yet 2 px costs 1.6-1.9x the
   averaging time of 3.3 on EVERY gauge (IFO 266 vs 161 s per pm, Photonfocus 68.6 vs
   36.2, Feng's 691 vs 373, lock-in 5.44 vs 3.47, vector ZWFS 1040 vs 639, stepped 723 vs
   451, pinhole 681 vs 398) and 1.5 px 1.8-2.1x.  It flattens at the model's own 3.3: 4
   px reads as 3.3 everywhere.
1a. **The cost is the PIXEL'S INTEGRATION, not the sample count** -- the ZWFS control
   says so: on a POINT-sampled 1.65 grid every ZWFS row reads as the 3.30 anchor (V 629
   vs 639 s per pm, hold 3.40 vs 3.41 pm), while the camera's pixels at 1.5 and 2 cost
   1.6-1.8x.  A pixel box-averages each actuator's response into its neighbours', the
   matrix's columns lose independence, and the estimator pays for separating them in
   noise.  So a camera with pixels at the actuator scale pays it; a sampled (small
   fill-factor) array would not -- a lenslet or a lock-in array with small sensitive
   areas is the hardware form of the control.
1b. **The deck's ZWFS camera timings are from that point-sampled 1.65 grid** (camS_oap,
   redo96_oapnoise: NGRID 193, their reports print "DM 96x96: 1.65 detector px per
   actuator (min 2) -> NOT MET").  They therefore describe a ~3.3 px CAMERA (V 519 s per
   pm recorded; this sweep's 3.30 gives 639 with the matrix estimator), not a 1.65 px
   one -- a real 1.65 px camera would read ~1.7x slower.  The deck's "every record is at
   3.3-4 px per actuator" holds for the battery rows, not for those camera rows.
2. **There is no collapse at 1-1.5 px.**  Noise-free, the converged single-actuator gain
   holds at 1.5 on every gauge and the floors stay within 1-2 pm of the anchor; no +-0.1
   pitch shift reads the wrong actuator at any sampling, on any gauge.  1 px DEGRADES
   rather than collapses: the single row's floor goes 2 -> 13 pm (IFO, HET) and 2-5 -> 32-33
   pm (S, V, P -- the ZWFS family breaks harder: its one-step gain 0.91), the grid row's
   gain 0.95-0.96, the dense-random error x4-8, the time x4.6-9.  The deck's "1-1.5 collapses"
   row is the blur demo's box-kernel case, not this bench (the blur demo fit with a
   stencil; this is the measured matrix, which absorbs the blur).
3. **Registration is a hold term at EVERY sampling.**  0.1 pitch of camera motion after
   calibration costs 2.5-6 % of gain and lifts the noise-free floor 2 -> 17 pm (ZWFS 21-24
   pm), the same at 3.3 and 1.5 (58 pm at 1 px) -- the pixel count does not buy registration tolerance; the matrix
   re-measure (or a registration servo) does.
4. **Read noise and binning: the Photonfocus pays twice.**  Its 220 e- read is negligible
   per its deep well at 3.3, but binned DIGITALLY to 2 px (1100 e- per camera pixel) and
   amplified by the deconvolution, its 1.6 s hold doubles (2.52 -> 5.28 pm); on-chip
   binning restores ~30 % of the time (68.6 -> 48.7 s per pm).  A 5 e- camera (Feng's) or
   the read-free lock-in barely notice digital binning.
5. **Complexity / data:** the pixel count above 2 costs data linearly (0.2 -> 0.53 Gb/s
   from 2 to 3.3 on the pol camera), and the ZWFS's per-pixel solve (six reference-wave
   iterations per reading) scales with it -- cheap next to item 1: on this bench the right
   camera sampling is ~3 px per actuator for EVERY gauge, not 2, and nothing is gained
   above 3.3.

**The composite figure** (`ressweep_figs`, `runs/ressweep_figs/ressweep_composite.png` +
`.txt`): the hold at each sensor's loop cycle and the seconds per pm against camera
pixels per actuator, one line per gauge, the ZWFS's point-sampled 1.65 as hollow
markers; the band shaded is where the single-actuator row FAILS in the records (noise-free
floor >= 10 pm, i.e. 1 px; the boundary lies between 1 and 1.5) -- not a collapse, which
the records show at no sampling.

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
