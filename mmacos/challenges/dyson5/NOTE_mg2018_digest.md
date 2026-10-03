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

## Corrections and additions (TO, 2026-09-30, read against the PDF text)
- **The spec table above is Table 2 = the Fig. 13 long-slit OFFNER, not the
  Fig. 15 Dyson.**  The paper's caption says "of Fig. 11" (a thermal-IR lens
  -- a typo in the paper), and the text at the Offner example reads "The
  Offner spectrometer example of Fig. 13 has the following ... performance
  are shown in Table 2".  Its rows: F/2.8, slit 48 mm (1600 x 30 um),
  380-2500 nm, dispersion 10 nm/30 um, smile < 0.3 % px (< 100 nm), keystone
  < 2 % px (< 600 nm), ensquared energy in 30 um > 0.76, SRF FWHM < 1.35 x
  sampling, CRF FWHM < 1.1 x sampling, SRF width variation with field
  < 4.5 %, CRF variation with wavelength < 2 %.  The Fig. 15 CaF2 Dyson
  (5 nm sampling, 38.4 mm slit) has spot diagrams (Fig. 16) but no table of
  its own.  So the "flight-class column" is an OFFNER's; use it as the
  performance CLASS both forms are held to, and say which form it came from.
- **Table 5 / Figs. 19-22 are the paper's design AT JOE'S REGIME:** the
  broad-band PRISM Dyson (BPDS; single CaF2 lens split into a lens + two
  prisms for detector clearance, a curved Fery prism of IR-grade fused
  silica as the disperser, toroidal reflecting rear surface).  All-spherical:
  2160 cross-track px, F/2.5, uniformity > 90 %, optics length 50 cm, prism
  diameter 14 cm.  With a FREEFORM surface on the CaF2 element: **3200 px,
  F/2, 18 x 18 um pixels, slit 57.6 mm, optics length 54 cm, prism diameter
  19.2 cm**, max smile 0.6 um = 3.3 % of an 18-um pixel ("at the upper range
  of acceptability before tolerancing"), keystone-equivalent 0.2 um ~ 1 %,
  SRF variation through field ~1 %, worst CRF variation with wavelength
  ~5.8 %, ensquared energy just over the 75 % rule at both band ends.  A more
  COMPACT variant (Fig. 22) separates the mirror from the prism "so it
  operates closer to the concentric-aplanatic condition" and adds a MENISCUS
  corrector: ~60 % of the size at the same specs, at the cost of six more
  air-glass interfaces.  Two things for dyson5: (a) a 54 cm-long, 19 cm-wide
  instrument at 3200 px / 18 um / F/2 is the published answer to "what does
  this regime cost" -- beat 1's r >= 213 mm (R_g 687 mm) concentric seed is
  the same order, not an anomaly; (b) the paper's own path off the concentric
  seed is freeform-on-the-block, then separate-mirror + meniscus -- the
  departure ladder for beat 2.  The paper also states the 18-um consequence
  directly: a pixel-fraction uniformity spec "becomes significantly tighter
  in absolute terms" -- Joe's 0.1 px = 1.8 um vs the BPDS's achieved 0.6 um
  smile, i.e. Joe's design-stage number is ~3x looser than the paper's own
  achieved value on the same pixel, and 10x looser than rule (1)'s 1 %.
