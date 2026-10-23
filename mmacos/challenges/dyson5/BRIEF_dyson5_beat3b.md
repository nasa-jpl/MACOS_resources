# dyson5 beat 3b -- R4, the compact variant; the trade table; re-scores (2026-10-01, TO lane on Fable)

Brief: `macos/BRIEF_to_dyson5.md` addenda 3 (finding #3 fixed, OPD modulo
lambda) and 4 (R4 = the compact variant; one trade table; re-score on the
current mex; deck-standard figures from the runner).  Engine: chord-ruled
directions (799498b) + chord-coordinate OPL jump (dL = Order*lambda/d *
dot(s0, rho)); mex 04:35.  Gates on it: tGratingOpl 2/2 (wrapped), tSpectrometerRx
3/3 (incl. the aspheric block), tGratingImmersed, tGlassDispersion.

## Re-scores on the fixed engine (addendum 4)

- s2 (ray scorer): engine SRF 2.024 px on BOTH forms (the 2-px-slit floor;
  beat 2's 3.06 was finding #2), CRF 2.28 / 1.03 px, EE 0.44 / 1.000; the
  engine column now equals the chain's 'planes' column.
- s2w (the twin, addendum 3): wave - ray centroid offsets <= 0.0005 px
  (Offner) and 0.0013 px (Dyson) on every (slit x, lambda) point -- the
  propagated PSF sits where the rays say.  Its own products: ensquared
  energy WITH diffraction Offner 0.852 (geometric 1.000), Dyson 0.447; SRF /
  CRF from the propagated PSF 2.04 / 1.04 px (Offner), 2.03 / 2.21 px (Dyson).
  Order-0 validation unchanged (Airy, EE 0.853 at 1.44 um).
- s2l (slit-width diffraction loss, NEW): a 36 um slit, far-field sandwich to
  the grating plane at 0.7 m, loss past the F/1.8 acceptance = 0.08 / 0.54 /
  1.38 / 2.48 % at 380 / 700 / 1440 / 2500 nm vs the sinc^2 closed form
  0.28 / 0.52 / 1.05 / 1.83 %.  Same order and trend; the factor (0.3 to
  1.35) is UNRESOLVED -- suspects: FFT-window aliasing at the short end
  (window = lambda z / dx_in, 450 mm at 380 nm against a +-195 mm
  acceptance) and the engine's aperture-edge treatment.  The design
  statement stands: slit diffraction past the grating is < 2.5 %.

## R4 -- the meniscus corrector (the paper's compact variant), at the fixed 220 mm

Solved in two steps from R3 (meniscus alone as a near-null shell about the
block's centre, then everything), identical operands and scorer:

| rung (engine rows) | keystone | smile | CRF | SRF | EE | length | footprint | nElt | glass |
|---|---|---|---|---|---|---|---|---|---|
| R3 de-concentred block | 0.0111 | 0.0083 | 2.10 | 2.03 | 0.476 | 698 mm | 272 mm | 5 | 22.1 L |
| R4a meniscus alone | 0.0629 | 0.0120 | 2.49 | 2.05 | 0.331 | 698 | 270 | 9 | 22.3 L |
| **R4 meniscus + all** | **0.0026** | **0.0051** | **1.327** | 2.03 | **0.759** | 694 | 269 | 9 | 22.7 L |
| size alone (r free, R1) | 0.0244 | 0.0008 | 1.335 | 2.02 | 0.742 | 1099 | 427 | 5 | 82.8 L |

R4 meets Joe's CRF (< 1.5 px) and the paper's ensquared rule (> 0.75) at
the SEED's size: a 4 mm plate of near-2 m radii 20 mm beyond the block
face, with the block 1.9 mm off the grating's centre along the dispersion
and R_g 694 mm.  It buys what the free-radius run bought (CRF 1.33 / EE 0.74
at 341 mm) at 63 % of the length, 63 % of the grating footprint and 27 % of
the glass -- the compact variant's trade, measured.  Smile and keystone are
at 0.3-0.5 % of a pixel.  The meniscus ALONE is worse than R3 (R4a): the
corrector only pays when the block offset and the grating radius move with
it.

**Caveat, stated:** the R4 solve ends ON its meniscus bounds (c = 0.5 /m,
t = 4 mm, vertex 0.24 m).  Three solves from the same R3 seed mapped a
multimodal landscape -- curvature signs and thickness released: a basin at
vertex 0.44 m with CRF 1.49 / EE 0.66; vertex held 0.235-0.30 m: CRF 1.57 /
EE 0.63.  The record keeps the bounded solve and leaves a global search over
the meniscus (and the native optimize) to beat 4.

## Figures from the runner (addendum 4; `dyson5_run` stages, stable names)

`_s1_layout_{dyson,offner}.png` and `_s3_layout_r{0,1,2,3,4a,4}.png`
(`spectrometer_layout_fig`: mm sections, block body, grating arc, meniscus,
slit/FPA marks, rays at 380/1440/2500 nm from slit centre and ends, scale
bar); `_s2_maps_{dyson,offner}.png`, `_s3_maps_r*.png`
(`spectrometer_maps_fig`: field-angle, keystone, smile, SRF, CRF, EE, each
with its convention on the colorbar); `_s3_trade.{txt,png}` (`dyson5_trade`:
one table over both records, bars with the spec and paper lines);
`_s2w_twin.png` (order-0 PSF with the pixel box, agreement maps, PSF
response functions); `_s2l_slitloss.png`.

## Engine items for CC (minor)

- A short `AsphCoef=` line is an uncaught end-of-file (parser reads 4 values);
  the emitter pads.
- The slit-loss factor above, if it turns out to be the aperture-edge taper.

## Next (beat 4)

Native optimize with the same pixel-unit operands (the paper's rule), a
global search over the meniscus, and the Offner sibling through the ladder.
