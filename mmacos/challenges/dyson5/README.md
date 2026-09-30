# dyson5 — the VSWIR Dyson imaging-spectrometer challenge

The fifth design challenge (2026-09-30): an EMIT-class push-broom
imaging spectrometer, **designed FROM the spec** -- no proprietary
prescription is used or requested (Dave's ruling; public papers give
the form, the design conditions and the spec, never surfaces).  It is
therefore scored against the spec, not against a reference design.

## The target (Joe's spec; "made up but EMIT to the digit")

| item | value |
|---|---|
| F-number (image space, air-equivalent) | 1.8 |
| FPA | 3000 spatial x 500 spectral px at 18 um -> slit 54 mm, spectral height 9 mm |
| band | 380–2500 nm -> 4.24 nm/px |
| smile, keystone | < 0.1 px (0.2 "might be ok") |
| SRF FWHM | < 1.5–2.0 px |
| XRF FWHM | < 1.5 px |
| radiometric chain vs wavelength | throughput, grating efficiency, FPA QE, slit loss |

Jim's realism, recorded alongside (not scored): the grating is the
stop; as-built SRF 2.5–3 px; 2-px slits; photon-limited, not
diffraction-limited; smile/keystone are the drivers.  Public comparison
point: Carbon-I (arXiv:2505.22545) -- F/2.2, 2040–2380 nm, 3072 x 512
at 18 um, slit >= 54 mm x 36 um, smile/keystone <= 15 % px.  The 54 mm
slit is 2.3x EMIT's, so the block SCALE is part of the problem.

## The metric -- state it before quoting any number

Every stage names its convention first.  So far (stage s0): block
index n(Silica, 1 um) = 1.450417; F/# = air-equivalent image-space,
in-glass marginal half-angle u = asin(1/(2 n F)); blur = transverse
rms spot of a full F/1.8 cone from a point on the block's flat face,
imaged back to the flat face by the concentric relay with the grating
traced as a mirror (order 0); field height h measured from the Dyson
axis on the flat face; the slit corner is h = hypot(27 mm, 8 mm offset).
The spectrometer metrics (smile, keystone, SRF/XRF, the radiometric
chain) are defined in `optical_design/SPECTROMETER_DESIGN_REFERENCE.md`
§5 and land with stage s2.

## Beat 1 (2026-09-30) -- what is on record

1. **Engine gates, both RED for ONE engine reason (STOP report, see
   `BRIEF_dyson5_beat1.md`):** `GlassElt=` is dead in the current tree
   -- every load blanks the glass catalog before the parser's name
   lookup, so a `GlassElt= Silica` element traces as AIR (measured
   index 1.000 at 0.5/1/2 um; the CLI shows `Default used for
   GlassCoef`).  `tGlassDispersion`'s fixed-index control passes (the
   gate can see), `tGratingImmersed`'s air control passes (the grating
   equation is sound in air) and its immersed residual is exactly
   (n-1) m lambda/d -- the block was air.  Both gates stay in
   SUITE_FAST and go green when CC's fix lands; the CaF2 leg is
   assumption-gated (Incomplete, never a silent pass) until an engine
   built with the new table row is in use.
2. **The Dyson condition R_g = n r/(n-1), verified by exact trace**
   (`design/src/dyson_layout.m`): blur minimum at the condition (0.67 um
   vs 4–5 um at +-5 %), image at -h, residual fifth order (h^4.17,
   r^-3.19).
3. **The scaling law at the spec (pre-registered null, answered):** a
   single concentric silica block needs **r >= 213 mm (R_g = 687 mm)**
   for a quarter-pixel corner blur at F/1.8 with the 54 mm slit -- the
   classical seed does not meet the spec at flight scale, which is why
   the flight forms add an asphere/meniscus/air gap.  Elements are added
   in beat 2, after this is on record.

## Run it yourself

```matlab
run('<path-to>/mmacos/mmacos_setup.m');
addpath('<path-to>/mmacos/challenges/dyson5');
OUT = dyson5_run();                                   % stage s0 at the spec
OUT = dyson5_run(struct('Fno',2.2,'y_offset_m',6e-3)); % another instance
```

All knobs live in `dyson5_params.m` (single source of truth).  Stage
s0 is engine-free; the gates need the mmacos mex:
`./run_mmacos_tests.sh tGratingImmersed` / `tGlassDispersion`.

## Files

- `dyson5_guidance.txt` -- Joe's and Jim's notes (the spec's source).
- `dyson5_params.m`, `dyson5_run.m` -- the runner (stages s0 now; s1
  deck emission, s2 `spectrometer_score`, s3 native optimize queued).
- `dyson5_s0_scaling.{txt,mat,png}` -- the stage-0 record.
- `BRIEF_dyson5_beat1.md` -- the beat-1 report (gate STOP + state).
- `../../design/src/dyson_layout.m`, `dyson_scaling.m` -- the
  closed-form seed with its exact-trace verification.
- `../../tests/tGratingImmersed.m`, `tGlassDispersion.m` and fixtures
  `../../tests/Rx/Rx_GratingImmersed.in`, `Rx_GlassPlate.in`.
- `../../../optical_design/SPECTROMETER_DESIGN_REFERENCE.md` -- forms,
  conventions, the verified condition and scaling, metrics, references.
