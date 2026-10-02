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

## The paper (Mouroulis & Green 2018), folded in 2026-09-30

`NOTE_mg2018_digest.md` (CC) + its correction section (TO).  What it
settles for this challenge: the merit's rules (distortion ~1 % px at
design, > 75 % ensquared, degraded spots OK for uniformity, grating =
stop; pixel-unit smile/keystone in the optimize merit from the FIRST
pass); the scorer's definitions (SRF = slit ⊗ LSF ⊗ pixel, CRF = system
LSF ⊗ pixel; the incoherent chain is honest here, Airy diameter 11 um
< 18 um); and where Joe's spec sits -- ALIS-class (Table 3), and the
paper's own Table 5 freeform prism Dyson at 3200 px / 18 um / F/2 /
57.6 mm slit is 54 cm long and 19 cm across, so beat 1's r ≥ 213 mm
concentric seed is the expected order of this regime.  Correction to
the digest: the quoted spec table is Table 2 = the Fig. 13 long-slit
OFFNER (F/2.8, 48 mm, 10 nm/px), not the Fig. 15 Dyson; it is reported
as the performance CLASS.  The PDF is local and git-ignored (SPIE).

## Beat 2 (2026-09-30) -- emitter, scorer, and a second engine finding

Both engine gates from beat 1 are GREEN on CC's fixed engine
(tGlassDispersion 3/3 incl. CaF2, tGratingImmersed 4/4).  Beat 2 adds:

- `design/src/spectrometer_geom.m` -- ONE chain for both forms (Dyson:
  block + air gap + concave grating in air, the JPL form; Offner: concave
  twice + convex grating at the stop) with the chief aim, the groove
  period (band across the 9 mm FPA) and the FPA focus solved by exact
  3-D trace; `spectrometer_rx.m` emits the MACOS deck; gate
  `tests/tSpectrometerRx` = the engine's chief AND every ray land where
  the chain says (1e-9 m), the band spans the FPA.
- `design/src/spectrometer_score.m` (engine rays) + `_chain.m` (chain
  rays): field-angle and wavelength maps, smile, keystone, SRF/CRF by
  the slit (x) LSF (x) pixel (x) Airy chain, geometric ensquared energy,
  the closed-form radiometric chain.  Runner stages s1 (emit) and s2
  (score); records `dyson5_s1.txt`, `dyson5_s2.txt`.
- **Engine finding #2 (CC's lane): the grating groove model.**  The
  engine holds the period constant along the curved surface; a
  straight-ruled grating has it constant along the chord.  The
  difference is a spectral blur proportional to wavelength, uniform
  over the slit (Offner 2.8 px rms at 2500 nm vs 0.003 px), which sets
  the engine's SRF numbers in s2 until it is fixed; the chain's
  'planes' column is the design's prediction.  Details and the fix
  candidate: reference doc sec. 2.
- Three engine conventions pinned on the way (reference doc sec. 2):
  `ChfRayPos` is where rays start and becomes the physical source at
  load; `macos.stop` aims immediately and is one pass short on its
  first call (declare the stop first, then the chief); a `Return`
  coincident with the `FocalPlane` drops the rays (use a `Reference`
  upstream).

## Beat 2c (2026-10-01) -- the propagation twin, and a third engine finding

`design/src/spectrometer_wave.m` + runner stage `s2w` (opt-in): a
far-field terminal on a reference sphere upstream of the FPA (the
Offner is telecentric, so FEX's exit pupil is unusable), re-posed per
field and wavelength; the complex field at the FPA, PSF centroid vs
ray centroid, SRF/CRF from the propagated PSF.  Validated on the
order-0 Offner relay (Airy spot, 94 % in one pixel, pupil OPD 8e-11 m).
At order -1 the engine's pupil OPD is 4 waves rms while its rays
converge to 0.05 um: **engine finding #3**, the grating's optical-path
jump uses the local-tangent projection of the hit vector where the
groove count needs the chord coordinate (a cubic, 12 waves at this
footprint, zero on a flat grating).  Gate `tests/tGratingOpl` (order 0
passes, order -1 fails today; green on the fix with no test change);
engine-free confirmation in `BRIEF_dyson5_beat2c.md`.  The twin's
order -1 numbers in `dyson5_s2w.txt` are that defect until it lands.

## Beat 3 (2026-10-01) -- the departure ladder

`dyson_ladder.m` + runner stage `s3` (opt-in): rungs solved on the exact
chain with the pixel-unit smile/keystone operands in the merit from the
first pass, each emitted and engine-scored.  At the seed's 220 mm block:
R1 (grating radius factor + face offset) takes keystone 0.095 -> 0.038 px;
R2 (conic + h^4/h^6 asphere on the block face) is inert, and a 1-D scan
proves it a steep bowl at zero; **R3 (the block's centre 0.86 mm off the
grating's along the dispersion) takes keystone to 0.011 px** -- Joe's
0.1 px by 9x, the paper's ~1 % design rule met -- with smile 0.008 px.
The blur (CRF 2.1 px, ensquared 0.48) is the concentric fifth-order
residual at the effective field and is bought only by size (free-radius
variant: EE 0.74 at 341 mm) or by the paper's separate mirror + meniscus
(rung R4, next).  Records `dyson5_s3.txt`, `dyson5_s3free.txt`;
report `BRIEF_dyson5_beat3.md`.

## Beat 3b (2026-10-01) -- R4 and the trade table

**R4, the compact variant (meniscus corrector + everything), at the fixed
220 mm block: keystone 0.0026 px, smile 0.0051 px, CRF 1.327 px (spec
1.5), ensquared 0.759 (paper > 0.75)** -- what the free-radius run bought
at 341 mm, at 63 % of the length and footprint and 27 % of the glass
(`dyson5_s3_trade.txt`).  The meniscus alone is worse than R3; the solve
sits on its bounds and the landscape is multimodal (three solves on
record) -- a global search is beat 4's.  Re-scores on the fixed engine:
engine SRF at the 2-px floor; the twin agrees with the rays to 0.0013 px;
slit diffraction loss measured at < 2.5 % (factor vs sinc^2 open).
Report `BRIEF_dyson5_beat3b.md`; every figure from the runner's stages.

## Beat 3c (2026-10-01) -- the collisions brief on R4

Apertures are declared on every surface from the multi-field, multi-lambda
footprint (+5 mm; not one ray vignetted, gated), `spectrometer_clearance`
scores every leg against every body it does not traverse and FAILS the stage
on a negative entry, the Offner is re-posed at 0.22 R (+10.2 mm clear) and
solved (`offner_solve`: convex radius x 1.0034, second zone x 0.951 --
keystone 0.028, CRF 1.20, SRF 3.69 px), the slit mask and FPA package are
bodies (the Dyson clears its own package by **+1.2 mm with no cold shield**:
the fold-prism item), the trade table carries element sizes, the engine
renders (`dyson5_view_figs`) are the layouts of record, and the twin runs on
R4: **EE with diffraction 0.747 vs geometric 0.759**.  Open for beat 4: the
wave-ray centroid offset (0.001 / 0.030 / 0.122 px on seed / R3 / R4) and
its amplitude-weighting explanation.  Report `BRIEF_dyson5_beat3c.md`.

## Beat 4a (2026-10-01) -- the centroid question, settled

`dyson5_centroid_probe` on R4: the pupil-domain prediction (weighted =
unweighted to 1e-4 px) reproduces the ray centroid, the window/pitch
variants reproduce every digit, and the 0.122 px was the twin reading
its far-field grid in the wrong orientation (the grid is the source
grid inverted by the FFT; now read from the engine's source frame).
**Wave = ray centroid on R4 to 0.0022 px; the detector sees the ray
centroid; keystone 0.0026 px stands.**  Report `BRIEF_dyson5_beat4a.md`.

## Beat 4b (2026-10-01) -- slit-loss factor resolved; global meniscus search

Four one-knob tests (`dyson5_slitloss_tests`): the 0.30x at 380 nm was
aliasing (window 1.11x the acceptance), the 1.36x at 2500 nm was the
planar far field's evanescent energy in the normalisation; with a 2x
window and the propagating-region normalisation the engine reads
0.86-1.14x sinc^2 across the band -- **slit loss 0.3-1.8 %** stands.  The
global meniscus search (12 starts) finds four basins of similar merit
and none with better CRF/EE than the R4 of record: the quadratic
distortion weight was buying distortion already 40x under spec; the
native optimize carries smile/keystone as hinge walls instead.  Every
ladder deck re-emitted with apertures.  Report `BRIEF_dyson5_beat4b.md`.

## Beat 4c (2026-10-01) -- the native optimize, built, gated, blocked on the engine

`dyson_native` + stage `s4`: CALIB (the engine's multi-field least squares)
on the R4 deck -- SPOT target, 5 slit positions x 6 wavelengths, 9 variables
(grating position, block face radius + conic, meniscus faces, focus), the
double-pass copies LINKED so each is one physical surface -- in chunks of
iterations with the smile/keystone WALLS held on the chain between chunks
(CALIB has no distortion operand; the operand is the ask to CC).  Each
chunk is read back from the engine, mapped into the chain and proven by an
identity check before it is scored and gated.  Running it pinned FOUR
engine findings (`BRIEF_dyson5_beat4c.md` section 3): no centroid operand;
the OptAsph slice; `OptRayGrid=` corrupts the heap; and the blocker --
CALIB's SPOT derivative loop steps at the wavefront stride
(`design_optim.F` ~:792), a heap stomp on the second field.  The stage
refuses to run (`native_enabled`) until that fix lands; R4 of record stands.
New emitter options `'links'` and `'opt'`, gated in `tSpectrometerRx`.
**After the fix (macos 0d257ff, same day):** the stage runs end to end.
With the grating's position free, CALIB's first five iterations bought blur
with 15 px of keystone and the wall rejected the chunk; with the blur-only
set (block face, meniscus, focus) every LM step was rejected -- **R4 of
record is a local optimum of CALIB's spot merit too; R4n = R4** (CALIB does
move from a deliberate 0.3 mm defocus: spot 97 -> 21 um in five
iterations).  Two further engine findings: the asphere differential step
is round-off (gaussj singular), and the mex dies on the LM failure path
where the CLI survives (`BRIEF_dyson5_beat4c.md` 3.5-3.6).

## Beat 4d (2026-10-01) -- R5, the fold prism, under the clearance gate

Entrance plate on the slit side, mirror-coated fold prism cemented under
the image (TIR fails at F/1.8 in silica), the FPA folded 27 mm away from
the slit's plane -- the chain carries it (`P.fold_h`, `slit_gap`,
`fpa_gap`, the slit plane's axial position as an R5 variable), every
detector-frame consumer uses the FPA's own frame, the engine lands every
ray where the chain says (gate form `dyson_fold`), and the record forms
reproduce byte for byte.  The clearance gate now also flags a leg INSIDE a
box body and scores the mask and the package against every body (the first
sweep's package stood 1.5 mm inside the block and passed; withdrawn).
Stage `s5` sweeps the COLD-SHIELD HEIGHT (0-5 mm; air gap = height + 1 mm
on both sides, re-solving twelve variables at each) and takes the tallest
closing height as the record: **R5 of record = the fold with no shield
(1 mm gaps): CRF 1.267 px, EE 0.695, smile/keystone 0.005/0.009 px, every
clearance pair positive (+0.90 mm)**; every taller shield fails the CRF
spec through its air gap (1.75 / 2.31 / 2.78 / 3.75 px at 2 / 3 / 4 / 6 mm)
-- physics, not the solver: a plane air/glass boundary ahead of an F/1.8
cone.  A cold shield has to live inside that millimetre or behind a
cemented cold window.  Report `BRIEF_dyson5_beat4d.md`.

## Beat 4e (2026-10-01) -- the closure envelope (addendum 11)

`dyson5_envelope` + stage `s4env`: R4 re-solved from the record one axis at
a time (every point all eleven variables on the chain, engine-scored), a
point CLOSES when smile/keystone < 0.1 px, CRF < 1.5 px, SRF under the
2-px slit floor + 0.1 px and no variable on a bound.  **Designs close for
F/1.8-F/2.2, block radii 220-300 mm, slits to 54 mm, 18 and 30 um pixels,
silica and CaF2; the first metric to fail outside is the CRF** (150 /
180 mm radii at 2.37 / 1.61 px; the 60 mm slit at 1.548 px); F/1.6 and
F/2.8 end on the meniscus bounds with CRF 1.37 / 1.07 px (a bound to
widen, not a form that fails); the F/1.6 + 150 mm corner does not close.
Record `BRIEF_dyson5_beat4e.md`, `dyson5_s4env.{txt,mat,png}`.

## Beat 5 (2026-10-01) -- the telescope, and the instrument end to end

`design/src/telescope_geom.m` + `telescope_seed.m` + `telescope_ladder.m`
+ `telescope_score.m` (engine) / `telescope_score_chain.m`, `e2e_geom.m`,
the shared exact tracer `chain_trace.m` (the Dyson's, lifted verbatim),
`chain_aim.m` / `chain_bundle.m` / `chain_footprints.m`, and runner
stages `t1` (the telescope) and `t2` (end to end).  At EMIT's parameters
(420 km, 60 m -> 0.143 mrad per pixel: f = 126 mm, 70 mm at F/1.8,
24.6 deg across track onto the 54 mm slit) a coaxial three-mirror
anastigmat's off-axis section with a flat fold after M3 feeds the slit.
Three facts first: the R4 spectrometer is TELECENTRIC at the slit to
0.09 deg (its aim lines cross 16.84 m behind it), so the pupil-match
number is the chief's miss of the grating vertex when sent on through
the Dyson's chain; in that telecentric, flat-field limit the three-
mirror first order is a one-parameter family (t2 = f y2, phi3 = 1/t2);
and the push-broom field is the sky line that images onto the straight
slit, not a straight sky line.  Rungs on the exact chain (T0 the layout
under the clearance wall, T1 conics + radii + spacings + bias, T2 + h^4
/ h^6 aspheres, T3 everything), each emitted and ENGINE-scored at the
slit, the clearance gate on the combined chain; then the telescope of
record prepended to R4 and to R5 as ONE prescription each, a collimated
field source with the grating as the stop, scored by the spectrometer's
scorer.  Gate `tests/tTelescopeRx` (both decks, every ray 1e-9 m).
Result: the layout closes (every leg clears every body, +0.75 mm; the
chief lands within 9.4 mm of the grating vertex and the grating admits
the whole beam at every field; EFL 126.4 mm by the map) but the image
does not -- 67 px rms at the slit with a 1.6 mm field swing: conics and
symmetric aspheres on folded mirrors do not image at the pixel at this
field and speed.  End to end with R4: smile 3.1, keystone 0.59, CRF 15.5
px (the telescope's blur through the spectrometer).  Next: the
two-mirror modified Schwarzschild and freeform mirrors.  Records
`dyson5_t1.txt`, `dyson5_t2.txt`; report `BRIEF_dyson5_beat5.md`.

## Block-size trade (2026-10-02, CCMac) -- how small at R4 performance?

`dyson5_size_trade.m` + `dyson5_size_fig.m` (new, independent of the runner):
the R4 block (220 mm, 221 mm thick, 8.2 kg) shrunk by CONTINUATION in the block
radius, each point a full R4 solve warm-started from the next-larger solved
design and ENGINE-scored, closing only when it matches R4 (smile/keystone < 0.1,
CRF <= 1.33, SRF <= 2.05, EE >= 0.76, no variable on a bound).  Three families:
A silica / 54 mm slit, B CaF2 / 54 mm, C silica AND CaF2 / 27 mm slit (two 1500-px
modules share the swath).  **Result: the SLIT is the lever, not the glass.**  At
the full 54 mm slit silica cannot shrink (220 mm floor; CaF2 reaches 180 mm but at
the same 8 kg, being denser).  Split into two 27 mm modules and a **100 mm silica
block (101 mm thick, 1.0 kg, CRF 1.05 / EE 0.96)** matches R4 with margin -- 8x
lighter, 2.2x thinner, ~2x more uniform (the double-pass glass path halves); CaF2
goes to 80 mm.  The size wall is always the concentric fifth-order h^4/r^3 blur,
never clearance (which stays +0.4..1.0 mm throughout).  The solver check settles
beat 4e's 180/150 mm rows: continuation reproduces them (CRF 1.611/2.351 vs
1.607/2.371), so that radius axis was the design's limit, not the solver's.
Records `dyson5_size.{txt,mat,png}`, decks `dyson5_size_<family>_r<mm>.in`; report
`BRIEF_dyson5_size.md`.

**Round 2 (no meniscus, Jim's point): drop it.**  Rung R3 (the de-concentred
block + conic + h^4/h^6, NO meniscus; a `rung` option, two verdict columns
`matches R4` / `meets SPEC`, and a THROUGHPUT column = uncoated Fresnel at 1 um
over the air-glass crossings, R3 = 4 vs R4 = 8).  Families D silica / E CaF2 at
the 27 mm slit, F CaF2 one-module 54 mm, G the thick-meniscus global-search basins.
At the two-module slit the meniscus-free block **matches R4 to the SAME floor**
(D silica 100 mm, E CaF2 80 mm) with **4 crossings not 8 -> +15 % throughput**
(0.87-0.88 vs 0.76) and no 4 mm sliver; the round-1 130 mm block is identical in
image and mass with +15 % throughput.  A *buildable* (28-31 mm) meniscus FAILS
(family G: CRF 1.64 / 1.72) -- only the fragile 4 mm sliver reaches R4, so no
meniscus wins.  A one-module no-meniscus Dyson exists only in CaF2 (F: matches R4
at 240 mm, 14.3 kg) -- silica cannot.  Correction to round 1: CaF2's index is
LOWER than silica's (1.429 vs 1.450); the benefit is CaF2's lower dispersion over
380-2500 nm, not the cone angle.

## Run it yourself

```matlab
run('<path-to>/mmacos/mmacos_setup.m');
addpath('<path-to>/mmacos/challenges/dyson5');
OUT = dyson5_run();                                   % stage s0 at the spec
OUT = dyson5_run(struct('Fno',2.2,'y_offset_m',6e-3)); % another instance
OUT = dyson5_run(struct('stages', {{'t1','t2'}}));     % the telescope, then end to end (after s3 and s5)
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
