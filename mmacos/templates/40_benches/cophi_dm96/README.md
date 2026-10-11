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
  format · figs), to be written on `tg96_run`'s stages once the LO arm is in
  the bench (`BRIEF_to_cophi.md`, TO).
- Gate `tests/tCophi.m` (SUITE_FAST, 8): the algebra.

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
