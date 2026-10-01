# dyson5 beat 3 -- the Dyson departure ladder (2026-10-01, TO lane on Fable)

Brief: `macos/BRIEF_to_dyson5.md` item 2's "add elements only after the
scaling law is on record" (it is: s0) and the paper's corollary (distortion
operands in the merit from the first pass).  Engine: chord-ruled directions
(799498b); the OPL defect (#3) does not touch this beat -- the ladder is
ray-side and the engine reproduces the chain ray for ray.

## Method (every number re-runnable: `dyson5_run(struct('stages',{{'s3'}}))`)

`dyson_ladder.m`: each rung is solved by lsqnonlin on the EXACT chain
(`spectrometer_geom`, straight-ruled grooves) with residuals in PIXELS over a
5 x 5 (slit x, lambda) grid -- 10 x [smile_ij, keystone_ij] and 1 x [rms spot
u, v] per point, plus a wall on the slit-to-FPA clearance (>= 3 mm); the
groove period and the FPA focus are re-solved inside the chain at every
iterate so the band always spans the 9 mm FPA.  The solved deck is emitted
and scored in the ENGINE (`spectrometer_score`, 7 x 7, 41-pt grid); the
engine rows are the record.  The block radius is HELD at the seed's 220 mm:
freed, it walks to its bound and buys blur by size (s0's h^4/r^3 law), which
is not a departure.  Gate: `tSpectrometerRx` 3/3 incl. the aspheric block
(engine vs chain 1e-12 m per ray; a sphere-only chain misses by 0.26 mm).

## The ladder at r = 220 mm (engine rows; spec smile/keystone < 0.1 px, CRF < 1.5 px; paper: distortion ~1 % px at design, EE > 0.75)

| rung | smile | keystone | CRF FWHM | SRF FWHM | EE (1 px) | what moved |
|---|---|---|---|---|---|---|
| R0 concentric seed | 0.006 | 0.095 | 2.28 | 2.02 | 0.44 | -- |
| R1 R_g factor + face offset | 0.003 | 0.038 | 2.10 | 2.02 | 0.48 | R_g 708 -> 704 mm, face 0.5 -> 0.2 mm |
| R2 + conic + h^4,h^6 on the block face | 0.004 | 0.037 | 2.13 | 2.02 | 0.47 | asphere inert (sags of 10 nm) |
| **R3 + block centre off the grating's** | 0.008 | **0.011** | 2.10 | 2.03 | 0.48 | dC = (dy -0.86, dz +0.11) mm, R_g 698 mm, face 0.71 mm |

Free-radius variant (`dyson5_s3free.txt`): R1 walks r to 341 mm (its bound)
and reaches keystone 0.024 px, CRF 1.34 px, EE 0.74 -- by size; the asphere
is inert there too.

## Findings

1. **Distortion is solved at fixed size by de-concentring the block.**  R3
   takes keystone from 0.095 to 0.011 px -- Joe's 0.1 px by 9x, the paper's
   ~1 % design rule met -- with smile 0.008 px.  The lever is a 0.86 mm
   shift of the block's centre along the dispersion direction (and 0.11 mm
   axially): the Dyson's image of the grating is no longer concentric with
   the slit/FPA plane, which is what trims the field-dependent pupil walk.
2. **The block-face asphere is not a lever on this form.**  R2 converged to
   nothing, and a 1-D scan about R1 confirms a steep bowl centred at zero:
   0.5 um of h^4 sag already costs, 2.5 um doubles the CRF.  At fixed scale,
   with every surface concentric, an axisymmetric figure on the block face
   acts on all field points alike and cannot cancel a residual that grows
   as h^4.  Carbon-I's asphere lives on a design that ALSO breaks
   concentricity (an off-axis block section, F/2.2); R3 is the first half of
   that recipe.
3. **The blur is the concentric fifth-order residual at the EFFECTIVE field
   and is not bought by any single-block departure here.**  CRF 2.1 px and
   EE 0.48 are unchanged from R1 to R3.  The dispersed image sits 12 mm
   from the m = 0 image, so the slit corner's effective height is ~32 mm,
   not 28: s0's law gives ~0.4 px rms there at r = 220 mm, which is what the
   scorer measures (0.7 px at the slit ends).  The ways out are the ones the
   paper names: size (the free-radius run: EE 0.74 at 341 mm) or extra
   surfaces -- a separate mirror near the concentric-aplanatic condition plus
   a meniscus corrector (the compact variant, ~60 % of the size).  That is
   rung R4, a new surface class in the chain (two more refracting spheres),
   for the next slice.
4. **SRF is pinned at the 2-px-slit floor (2.02-2.03 px)** on every rung:
   the spectral blur is already below the slit image; Joe's "< 1.5-2.0 px"
   is at the floor, Jim's 2.5-3 px is the realistic band.

## Engine item (minor, CC): a short `AsphCoef=` line is an uncaught
end-of-file in `msmacosio.inc` (the parser reads `nAsphCoef_Default = 4`
values) that kills the host; the emitter pads to four.  The validator could
count values on multi-value keys.

## Next

R4: separate concave mirror + meniscus (the paper's compact variant) in the
chain; then beat 4, the native optimize with the same pixel-unit operands.
When CC's #3 lands: re-run s2w on the R3 deck (the twin of the ladder's end).
