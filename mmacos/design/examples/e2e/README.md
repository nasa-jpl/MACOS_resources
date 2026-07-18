# e2e — the complete end-to-end worked example

Design → instrument → segmentation → linear model → MET → simulator,
built entirely from the parameterized design-layer runners and
utilities, **for users to hack for their own systems** (Dave
2026-07-17).  Every stage runner produces a **thorough design report**
(saved `sN_report.txt`) **and graphics** (`macos.view_std` standard
views + the stage's metric figures) beside this file.

All user knobs live in **`e2e_params.m`** — one file, commented.  Each
stage consumes the previous stage's saved artifacts (`.in` + `.mat`),
so a knob change re-runs from the first stage it affects.

## The telescope case

An **on-axis Korsch TMA taken slightly off-axis, back end folded
behind the primary**: D = 4 m, **f/1.25 primary**, **f/18 system**,
500 nm.  The first-order layout comes from `macos.design.tma_layout`
with the f/#s as *free inputs* (change them in `e2e_params.m` and the
radii re-derive).  A high feed magnification (m2 = 16 → f/20
intermediate focus, near-unit M3 relay) keeps the package compact: a
~6% M2 obscuration, a cm-scale M1 hole, a short bench.  A 90° **fold
after M2** (Dave 2026-07-17) turns the feed into +x just behind M1, so
M3, the image, and the focal plane sit on a flat bench BEHIND the
primary.  The science field is biased just far enough off the axis
that the folded design fully clears (image walk = bias × EFL separates
the return from the fold body and the detector from the beams) — the
runner sweeps the bias and recommends the least that clears, since
off-axis aberration grows ~quadratically with bias.  Conics are solved
at the biased field (CALIB multi-field), then refined with freeform
Zernike departures under the sphere+Zernike solve doctrine (per-mirror
field-zone `lMon` via `field_zone_lmon`, coefficient-sanity
verification).

## Stages

| runner | consumes | produces |
|---|---|---|
| `s1_telescope.m` | `e2e_params.m` | telescope design: `s1_telescope.in/.mat`, `s1_views.png`, `s1_wfe_field.png`, `s1_fpmap.png`, `s1_report.txt` |
| `s2_instrument.m` | s1 artifacts | 3-mirror bench relay (M4 corrector / M5 collimator / M6 camera) widening the corrected field to ±2′ at ~0.3–0.6 −tilt waves (was 0.03→1.9 across the same span for the telescope alone).  The solve is JOINT (CALIB on M2/M3/M5/M6, SVD-engine stages on the dense 5×5 field grid, M1 common mode over the full set); M2/M3 keep refining with the instrument, their field-zone lMon growing with the field.  The BEST DETECTOR PLANE (tilt + despace) is re-fit through a 5×5 grid of measured field foci (`align_focal_plane`) and the ladder is scored on it.  The remaining raw−(−tilt) gap is relay DISTORTION — not correctable by detector angle; M4 near the focus is the reflective field-corrector for it (distortion-merit solve, follow-on).  Probed and rejected: per-field patch corrector at a focus (SVD rank collapse) and a 4th weak mirror near the relayed pupil (common-mode, conditions the joint solve worse).  `s2_instrument.in/.mat`, views + field maps + `s2_report.txt` |
| `s3_segmentation.m` | s2 | segmented primary (`segment_rx` + physical apertures); new views |
| `s4_jacobians.m` | s3 | `dwdx`, `dwdz`, `dwdgrid` sensitivity channels + condition/rank report |
| `s5_met.m` | s4 | MET truss (`add_met` + layout optimizer), `dedx`/`dldx`, estimator gains `dxdl`/`dxde`, `dwdl`/`dwde`, MET-optimized performance report, views with MET |
| `s6_simulator.m` | s5 | PSF simulator — mmacos engine OR linear model (user switch) driven by an x/z/grid time history |

Run order: `s1` → `s6`, each with `run('.../sN_*.m')` after building
mmacos (`mmacos_setup.m` or the addpath lines at the top of each
runner).  House rules: figures + reports land in this directory; no
`exit(0)` inside example scripts (batch wrappers supply it).
