# Imaging-Spectrometer Design Forms — Reference (Dyson, Offner)

Companion to `TELESCOPE_DESIGN_REFERENCE.md`, same rules: closed forms
the design layer seeds from, conventions the MACOS emitter must honour,
and the engine facts that were VERIFIED rather than assumed.  Started
2026-09-30 for the `dyson5` challenge (`mmacos/challenges/dyson5/`,
brief `macos/BRIEF_to_dyson5.md`).  Dev-only (strip list).

## 0. How to use this file

- §1 forms and when each applies; §2 the MACOS conventions (grating
  keywords, immersion, glass, F/#, dispersion sign); §3 the concentric
  Dyson closed form and its scaling law (verified by exact trace, numbers
  quoted); §4 the Offner (pointer); §5 the metrics and the conventions
  to state before any number; §6 the spec of record; §7 references.
- Public sources only: the papers give the FORM, the design conditions
  and the SPEC.  Surfaces are ours (Dave, 2026-09-30).

## 1. Forms

| Form | Elements | Stop | Character |
|---|---|---|---|
| **Dyson** | plano-convex block (thick lens) + concave grating, concentric | the grating | fast (F/1.8–2.2), compact, slit and FPA on/near the flat face, high uniformity; JPL's flight VSWIR/SWIR form (EMIT, CWIS, Carbon-I) |
| **Offner** | concave sphere (used twice) + convex grating at the stop, concentric | the convex grating | all-reflective, slower (F/3–4 typical), wider slit at the same blur, no glass -- every leg in air |

Both are 1:1 concentric relays of a ring field; the grating replaces the
convex/concave mirror.  In the Dyson the slit sits off the axis by
`y_offset` along the dispersion direction and the FPA at `-y_offset`
(the spectrum unfolds around the image of the slit centre).

## 2. MACOS conventions (verified 2026-09-30)

- **Grating element.**  `Element= Grating` (reflective, EltID 5, any
  conic base), keywords `h1HOE= <3-vector>` (the DISPERSION direction:
  `Snells_Law_Grating` projects it into the tangent plane and it IS the
  grating-vector axis; the groove runs along `N x h1HOE`), `OrderHOE= m`,
  `RuleWidth= d` (groove period in BaseUnits).  `TrGrating` (13) is the
  transmissive twin.  One order per trace: the ray model carries ONLY
  order m; efficiency lives in the radiometric chain.
- **Immersed grating.**  The grating equation is the immersed form,
  `nb (r·s) = na (i·s) + m λ0/d`, λ0 the VACUUM wavelength (`WaveBU` at
  the call site).  `na = CurIndRef` (the medium the ray is in) and
  `nb = IndRef(iElt)` AS WRITTEN on the grating element -- unlike a
  Reflector, the Grating branch never overwrites `IndRef(iElt)` with the
  incident medium.  RULE: a grating that sees glass carries the glass on
  its OWN element (`GlassElt= Silica`); the mirror habit `IndRef= 1`
  silently gives `nb = 1`.  Gate: `mmacos/tests/tGratingImmersed.m`
  (immersed law 1e-10, air control, the trap pinned).
- **Glass -- BROKEN IN THE CURRENT TREE (found 2026-09-30, gate
  `tGlassDispersion`; CC's engine slice).**  `GlassElt= <name>` is meant
  to re-evaluate the Sellmeier index at the CURRENT wavelength at every
  trace (`tracesub.F` CTRACE, gated on `LGlass(iElt)`).  But every load
  runs `reinitialise_variables()` -> `elt_mod_init_vars()` ->
  `GlassName(:) = ''` (elt_mod.F ~:940) BEFORE the parser's name lookup
  (msmacosio.inc ~:2796), and the catalog is (re)loaded only at first
  entry (`smacos.F:177`) or at the CLI's MRESET (`macos_cmd_loop.inc:173`,
  not per load).  So the lookup finds nothing, `LGlass` stays false,
  the written `IndRef` stands, and the deck traces as AIR -- in the CLI
  and in every binding.  Symptom: `Default used for GlassCoef(k)` at
  load; measured index 1.000 at every wavelength.  Until fixed, a
  block is a fixed `IndRef=` and sweeps are achromatic.
  Names come from `macos_f90/macos_glass_list.txt` (built into the
  engine by `tools/gen_glass_builtin.py`): `Silica` (Malitson) and,
  from macos dev-candidate 2026-09-30, `CaF2` (Malitson 1963).  An
  UNKNOWN name is silently ignored (`LGlass` false; the written
  `IndRef` stays) -- `tGlassDispersion`'s CaF2 leg reports Incomplete
  on an engine without the row.  No index getter exists in
  `macos_api_mod`; read the index off a refracted ray.
- **F-number.**  Quoted as the air-equivalent image-space value: the
  cone converging in glass toward the flat face has in-glass marginal
  half-angle `u = asin(1/(2 n F))`; the grating's axial clear diameter is
  `2 R_g sin u` plus the field extent.
- **Stop.**  The grating is the stop (`macos.stop(iGrating)`); the API
  requires `0 < iElt < nElt-2`, so the deck ends `... Grating, ...,
  Return, FocalPlane`.
- **Physical optics.**  The propagation chain's ray re-trace passes a
  Grating (propsub.F has the branch), but every diffraction KERNEL is
  handed the vacuum wavelength -- legs inside glass run at the wrong
  Fresnel number by n until CC's medium-aware kernel slice lands
  (memory `project_dyson5_spectrometer`).  Wave twins run on the Offner
  (all air) first.

## 3. The concentric Dyson (closed form + verified scaling)

Geometry: common centre C; flat face of the block through C; block
radius `r`, index `n`; concave grating radius `R_g`, concentric.

**Dyson condition** (Dyson 1959; Mertz 1977 states it as "the block fills
`(n-1)/n` of the slit-to-grating space"):

    R_g = n r / (n - 1)          gap (air) = R_g - r = r/(n - 1)

Verified 2026-09-30 by exact 3-D ray trace (`dyson_layout.m`, no
paraxial or Seidel shortcut), fused silica n = 1.4585, r = 100 mm,
F/1.8 (u = 10.98 deg in glass), field point h = 10 mm on the flat face:

| R_g / R_dyson | 0.90 | 0.95 | 0.98 | **1.00** | 1.02 | 1.05 | 1.10 |
|---|---|---|---|---|---|---|---|
| rms blur (um) | 9.80 | 4.28 | 1.24 | **0.68** | 2.53 | 5.17 | 9.25 |

Image at exactly `-h` (1:1, inverted) at the condition.  The residual is
FIFTH order: blur ∝ h^4 (measured exponent 4.00–4.06) and ∝ r^-3
(r 100→200 mm: 11.3→1.36 um at h = 20 mm), i.e.

    blur_rms ≈ k · h^4 / r^3,   k ≈ 0.68 um · (100 mm)^3 / (10 mm)^4 at F/1.8, n = 1.4585

**Scaling law for the spec** (runner record `mmacos/challenges/dyson5/
dyson5_s0_scaling.txt`, n = 1.450417 at 1 um, F/1.8, u = 11.04 deg; the
54 mm slit with an 8 mm dispersion offset puts the slit CORNER at
h = 28.16 mm; budget 0.25 px = 4.5 um rms):

| r (mm) | 50 | 75 | 100 | 150 | 200 | 250 | 300 | 400 | 500 |
|---|---|---|---|---|---|---|---|---|---|
| corner blur (um rms) | 602 | 123 | 47.4 | 13.2 | 5.48 | 2.78 | 1.60 | 0.67 | 0.34 |

Fitted r-exponent -3.19; the budget is met at **r = 213 mm (R_g =
687 mm, air gap 474 mm, grating clear diameter >= 263 mm axial plus the
field)** -- an instrument several times larger than the flight Dysons
(EMIT-class blocks are ~100 mm).  The centroid also walks (distortion,
same fifth order): image_y + h = -0.3 um at h = 20 mm and -5.9 um at
h = 30 mm for r = 100 mm.  That is WHY the real ones depart from the
classical seed (an even asphere on the block face and an off-axis block
section in Carbon-I; meniscus/field-flattener and slit air gaps in the
JPL Dyson family), and it is the pre-registered null of the brief:
report the law first, then add elements.  `dyson5_run` stage `s0`
reproduces every number above (`dyson_layout`, `dyson_scaling`).

Dispersion rides on top of the imaging condition: the m-th order chief
from the slit centre lands on the FPA displaced along `h1HOE` by the
grating kick `m λ0/(n d)` in direction, mapped to the flat face through
the concentric geometry -- the emitter solves `d` so the band spans the
FPA's spectral height (beat 2).

## 4. The Offner

`mmacos/design/src/offner_layout.m` already lays out the concentric
Offner relay (concave R used twice, convex R/2 at the stop, ring field
radius h) in the Bauer-chain form the Telescope builder consumes and
asserts closure (image at -h, symmetric path).  The spectrometer form
replaces the convex mirror by a convex grating of the same radius; the
scorer (§5) is shared with the Dyson.

## 5. Metrics -- state the convention, then the number

- **Field-angle map / wavelength map**: FPA centroid `(x_spatial,
  y_spectral)` per (slit position s, wavelength λ) from `macos.spot` at
  the FPA, `set_src_fov` × `set_src_wvl` sweeps; axes and sign: x along
  the slit (spatial), y along `h1HOE` (spectral), in pixels of
  `pixel_m`, origin at the slit-centre / band-centre image.
- **Smile**: variation of the spectral centroid `y` ALONG the slit at
  fixed λ, peak-to-valley over the slit, per λ (max over λ reported).
- **Keystone**: variation of the spatial centroid `x` ACROSS λ at fixed
  slit position, peak-to-valley over the band, per s (max over s).
- **SRF / XRF**: spectral / spatial response FWHM in px: geometric spot
  profile ⊗ slit image (`slit_px`) ⊗ pixel ⊗ Airy (analytic,
  λF/pixel ≈ 0.25 px at 2.5 um, F/1.8) -- replaced by the propagated
  PSF when the wave twin runs.
- **Radiometric chain** vs λ: throughput (Fresnel/coating), grating
  efficiency (scalar blaze closed form -- the engine carries one order),
  FPA QE (table), slit loss (the one MEASURED term: field at the
  grating plane, energy outside its aperture).
- Jim's realism (recorded, not scored): as-built SRF 2.5–3 px; photon
  limited; 2-px slits.

## 6. Spec of record (Joe, 2026-09; "made up but EMIT to the digit")

F/1.8; FPA 3000 × 500 px at 18 um (slit 54 mm, spectral 9 mm);
380–2500 nm (4.24 nm/px); smile/keystone < 0.1 px (0.2 acceptable);
SRF < 1.5–2.0 px FWHM; XRF < 1.5 px FWHM; radiometric gain vs λ.
Public comparison point (Carbon-I, arXiv:2505.22545): F/2.2,
2040–2380 nm at 0.7 nm/px, 3072 × 512 at 18 um, slit ≥ 54 mm × 36 um
(2 px), smile/keystone ≤ 15 % px, SRF ≤ 2.5 nm, fused-silica block with
an even asphere, concave spherical grating on N-BK7 (in AIR, not
immersed), efficiency > 0.84 over the band.

## 7. References

- J. Dyson, "Unit magnification optical system without Seidel
  aberrations," JOSA 49, 713 (1959).
- L. Mertz, "Concentric spectrographs," Appl. Opt. 16, 3122 (1977).
- P. Mouroulis & R. O. Green, "Review of high fidelity imaging
  spectrometer design for remote sensing," Opt. Eng. 57(4), 040901
  (2018).  NOT fetchable from this box (JS wall) -- the condition above
  was verified numerically instead, and matches Dyson/Mertz.
- C. L. Bradley et al., "The Optical Design of the Carbon-I Imaging
  Spectrometer," arXiv:2505.22545 (2025).
- JPL patents US 6,181,418 (concentric spectrometer) and US 8,520,204
  (Dyson-type with improved image quality and low distortion) -- the
  meniscus / field-flattener departures from the classical seed.
