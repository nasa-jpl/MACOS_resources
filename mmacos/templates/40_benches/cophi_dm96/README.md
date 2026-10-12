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
| noise only | **2.75 / 2.68** | **2.60 / 2.53** |
| random walk 2 pm/cycle | 3.60 / 3.54 | 3.48 / 3.43 |
| thermal ramp 5 pm/cycle | 10.3 / 10.3 (the lag rate/gG, every gauge) | 10.3 / 10.3 |
| camera offset walk | 2.85 / 2.77 (zero-sum: immune) | 2.66 / 2.59 |
| one cycle of sensor DELAY | 3.65 / 3.55 | 3.44 / 3.35 |
| LO 0.1 nrad tilt + 1 pm rms defocus per cycle | **29.6 / 29.4** | 29.5 / 29.4 |
| fiber walk 8e-3 rad/frame (piston-free) | 7.8 / 7.7 | 10.6 / 10.5 |
| fiber walk 2.4e-2 rad/frame (piston-free) | 22.0 / 21.7 | 31.0 / 30.6 |
| fiber at the CORRECTED budget 3.5e-3 rad/frame (piston-free) | **4.21 / 4.12** (record metric 6.10 / 5.58) | -- |
| fiber at 3x the corrected budget (piston-free) | 9.94 / 9.79 (record metric 16.6 / 14.9) | -- |

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
