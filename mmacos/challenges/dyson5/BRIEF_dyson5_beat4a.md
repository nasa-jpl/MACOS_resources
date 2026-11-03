# dyson5 beat 4a -- the centroid question, settled with discriminators (2026-10-01, TO lane on Fable)

Brief: `macos/BRIEF_to_dyson5.md` addendum 8.  Question: on R4 the twin's
PSF centroid and the engine's ray centroid parted by up to 0.122 px, growing
with wavelength, while the seeds agreed to 0.001 px.  If the detector-seen
keystone were the wave number, R4 read 0.12 px against Joe's 0.1 -- the
headline risk.  Tool: `challenges/dyson5/dyson5_centroid_probe.m` (R4 deck,
slit centre and end, 380 / 1440 / 2500 nm); records
`dyson5_s2w_centroid_{base,window2,pitch2}.txt`.

## Discriminator 1 -- the pupil-domain prediction

From the engine's own complex field on the reference sphere (the seeded
pupil, amplitudes carrying every Fresnel transmission), the wavefront
gradient by finite differences of the wrapped field (modulo lambda, addendum
3), and the far-field centroid theorem: PSF centroid = L lambda/(2 pi) times
the |a|^2-weighted mean gradient; the unweighted mean is the ray centroid.

| x, lambda | ray dv | pupil, unweighted | pupil, weighted | PSF dv |
|---|---|---|---|---|
| 0 mm, 380 nm | -0.0069 | -0.0068 | -0.0068 | -0.0067 |
| 0 mm, 2500 nm | -0.0524 | -0.0505 | -0.0505 | -0.0502 |
| 27 mm, 2500 nm | -0.0619 | -0.0598 | -0.0598 | -0.0598 |

Weighted and unweighted are IDENTICAL to 1e-4 px: **the amplitude-weighting
mechanism is refuted** (the Fresnel apodisation is far too gentle).  Both
reproduce the ray centroid to 4 % of the offset (sampling), and the PSF
centroid reproduces them -- once its grid is read in the right orientation.

## Discriminator 2 -- the wavelength law (numerics null)

The same point on three grids: model 512 / 127 pts (base), model 1024 /
255 pts (window doubled, same pitch), model 1024 / 127 pts (same window,
half pitch).  Every centroid agrees to the printed digit (0.0502 / 0.0508 /
0.0503 px at 2500 nm) -- not the twin's numerics.  The offset's growth with
wavelength is the RAY centroid's own (-0.007 / -0.029 / -0.052 px): the
dispersed geometry's aberration grows with lambda, and the PSF follows it.

## Discriminator 3 -- windows

Full grid, the smallest centred square holding > 99 % of the energy (2 to 9
px wide), and the 1-px box: the first two agree to 0.001 px; the 1-px box
differs (it truncates the spot) and is not how keystone is measured.

## What the 0.122 px was

The twin's far-field grid is not in the FPA's frame.  Raw PSF centroids on
R4 read (+0.0076, +0.0598) px against ray offsets (-0.0078, -0.0619): the
same numbers, both signs flipped.  On the re-posed Offner only the first
axis flipped.  The rule that reproduces both: **the far-field grid carries
the SOURCE grid's (xGrid, yGrid) orientation in index space, inverted by the
FFT** -- index 1 along -xGrid, index 2 along -yGrid.  The emitter writes
xGrid = +X and yGrid = chief x X: +Y for the Dyson's +z chief, -Y for the
Offner's -z chief, hence (-X,-Y) and (-X,+Y).  `spectrometer_wave` now takes
the signs from `macos.get_src_csys` (xDir, yDir) and states them in its
record; the concentric seed had hidden it (no offset to flip), and the
earlier "agreement" on the Dyson seed was vacuous for the same reason.  A
tipping-the-sphere calibration was tried first and discarded: its sign maps
were inconsistent point to point, so it was not measuring what it claimed.

## Result

Wave - ray centroid offsets with the corrected grid: **Dyson R4 <= 0.0022 px
(dispersion), 0.0003 px (slit)** on every (slit x, lambda) point; Offner
<= 0.0058 px.  **The detector sees the ray centroid.  R4's keystone is
0.0026 px, smile 0.0051 px; the deck carries the ray maps as the detector's.**
The twin's own products stand as before: R4 EE with diffraction 0.747
(geometric 0.759), SRF/CRF from the PSF 2.05 / 1.22 px; Offner EE 0.131
(geometric 0.225), SRF/CRF 4.05 / 1.44 px.

## Convention pinned (reference doc sec. 2)

Far-field (and DFT) output grids: index 1 = -xGrid, index 2 = -yGrid of
the source frame, both read back from the engine; never assume the FPA's or
global axes.  The seeds cannot test this (no offset); R3/R4 can and do.

## Next (addendum 8's order)

The global meniscus search; the native optimize with the ray-centroid
operands (unchanged); R5's fold prism under the clearance gate with the cold
shield height as the parameter; beat 5's telescope at the EMIT parameters.
