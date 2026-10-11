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
- `cophi_run.m` — the stages (bench · battery · noise · loop · capture ·
  format · figs) on `tg96_run`'s bones; `bench` is in (2026-10-11), the rest
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
