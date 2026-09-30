# Mouroulis & Green 2018 (Opt. Eng. 57(4) 040901) -- what dyson5 takes from it

Digest by CC, 2026-09-30, from the PDF Dave placed in this folder.  The PDF is
SPIE-copyrighted and is git-ignored here: it stays LOCAL, never in the public
repo.  Cite the paper; do not commit it.

## Design principles (Sec. 5.3) -- the merit function's rules
1. Geometric distortions (smile, keystone) controlled to ~1% of a pixel at the
   DESIGN stage, ~3% after tolerancing.
2. > ~75% of diffraction energy inside the pixel at all wavelengths and fields.
3. Degraded spots are acceptable and DESIRABLE (subject to 2) when they improve
   uniformity -- uniformity is enforced from the start, not added later.
4. The grating (dispersive element) is the preferred stop.
Corollary stated in the paper: optimizing point-imaging first and uniformity
later yields a bad starting point; distortion operands go into the merit from
the first pass.  (This is the strongest argument for the pixel-unit
smile/keystone maps being IN the native optimize merit, beat 4.)

## Response functions (Sec. 4.1) -- the scorer's definitions, verbatim form
- SRF(y)  = rect(slit) (x) LSF_spectrometer(y) (x) DET(y)      [spectral]
- CRF(x)  = LSF_system(x) (x) DET(x)                            [cross-track]
- ARF(y)  = rect(slit) (x) LSF_telescope(y) (x) rect(integration time)
  (the along-track function is the TELESCOPE's; the spectrometer does not
  enter it).  Resolution = FWHM of these.
- Incoherent approximation is adequate when the Airy-disk DIAMETER at the
  longest wavelength is smaller than the pixel and slit width; the full
  (partially coherent slit) calculation then changes things by ~10%.  At
  F/1.8, 2500 nm: 2.44*lambda*F = 11 um < 18 um -- inside the rule, so the
  analytic chain is legitimate, and the propagation twin measures the ~10%.
- Uniformity = invariance of the SRF through field and of the CRF/ARF through
  wavelength; a smooth variation of SRF with wavelength is NOT a uniformity
  concern.  Demonstrated: ~300 nm smile over a 48 mm slit (30 um pixels).

## The public Dyson example (Fig. 15, Sec. 5.4) -- a reference SPEC, not a Rx
380-2500 nm, 5 nm sampling, single CaF2 refractive element, all spherical,
slit up to 38.4 mm (1280 x 30 um), often used with the 640-element half to
avoid detector ghosts.  Its spec table (use as the "flight-class" column):
| Smile | < 0.3% of pixel (< 100 nm) |
| Keystone | < 2% of pixel (< 600 nm) |
| Ensquared energy in 30 um | > 0.76 |
| SRF FWHM | < 1.35 x sampling |
| CRF FWHM | < 1.1 x sampling |
| SRF width variation with field | < 4.5% |
| CRF variation with wavelength | < 2% |
Note the pixel is 30 um there; Joe's is 18 um, so "0.1 pixel" is a tighter
absolute number (1.8 um) than this table's 0.3%/2% (0.1/0.6 um).  State both.

## Where Joe's spec sits (Table 3, JPL instruments)
ALIS: Dyson, 380-2500 nm, 7 nm sampling, 3200 spatial pixels (space/air,
under development in 2018) -- Joe's 3000 x 500 at 380-2500 is ALIS-class
spatially, with 4.2 nm sampling (2120 nm / 500 px), finer than ALIS's 7.
CWIS: Dyson, 380-2500, 7 nm, 1240.  PRISM: Dyson 350-1050, 3 nm, 610.
So the 54 mm slit is not exotic; it is the ALIS regime, which is why the
concentric seed at 2x flight scale (beat 1) is the expected starting point
and the corrections are the work.

## Materials, ghosts, alignment
- Dyson refractive materials in flight: fused silica, CaF2, ZnSe (bonded to
  Ti or Al).  CaF2 is the example's choice at this band.
- Detector ghosts: specular reflections off the detector assembly travel to
  the grating and return in a higher order; more prominent in Dysons.
  Mitigations: order-sorting filters, detector coatings, using half the
  detector.  A ghost-path check belongs in the scorer's "open question".
- Dysons align with a single machined tube; slit-detector proximity is
  solved mechanically (small prism / in-built reflector for clearance).

## What the paper does NOT give
No Dyson prescription, and the concentric condition itself is cited to
Dyson 1959 (ref. 59), not restated -- beat 1's numerical verification plus
Dyson 1959 / Mertz 1977 remains the basis for R_g = n r/(n-1).
