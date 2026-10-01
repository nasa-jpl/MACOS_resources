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
- **Point source at a slit (measured 2026-09-30, gate `tSpectrometerRx`).**
  The deck's `ChfRayPos` is where the engine STARTS its rays, so it must
  lie between the slit and the first surface (a `ChfRayPos` past the
  first surface loses every ray as a "surface miss"); the physical
  source is `ChfRayPos + zSource*ChfRayDir`.  At LOAD the engine folds
  that in once -- afterwards `get_src_fov` reports `ChfRayPos` = the
  physical source and every ray passes through it (common-point fit
  2e-17 m).  `set_src_fov` writes `ChfRayPos` raw, so a slit scan hands
  it the SLIT POINT itself.  `Aperture` for a point source is the FULL
  cone angle in radians.
- **Stop aiming order.**  `macos.stop(k)` aims IMMEDIATELY from the
  source state it finds, and on the Dyson deck its first pass is 3.6 mrad
  short of converged (1.7 mm miss at the grating; a second call finishes
  it).  Declare the stop FIRST, then `set_src_fov` with an exact chief
  (`spectrometer_geom`'s `G.aim`): the engine keeps it to 1e-16.
- **Tail.**  A `Return` COINCIDENT with the `FocalPlane` leaves the FPA
  with zero path length and the engine drops those rays; the emitter
  uses a pass-through `Reference` 1 mm upstream to satisfy the stop
  wrapper's `iElt < nElt-2`.
- **GROOVE MODEL -- engine finding (2026-09-30, CC's lane).**
  `Snells_Law_Grating` normalises the projected rule direction
  (`DPERPUVEC` onto the vertex plane, then unit `shat` in the local
  tangent plane), so the groove period is constant ALONG THE CURVED
  SURFACE.  A straight-ruled concave grating -- the element type's own
  name, and what CODE V/Zemax grating surfaces define (spacing measured
  in the vertex tangent plane) -- has equidistant groove PLANES: the
  tangential kick is `m lambda/d` times the UN-normalised projection
  (magnitude cos of the local tilt).  Measured with the chain
  (`spectrometer_geom`, `P.grating_model`): spectral rms blur at the
  slit centre, 380/1440/2500 nm --
  Offner (R 0.5 m, F/2.8): surface 0.42/1.60/2.78 px vs planes
  0.000/0.000/0.003 px; Dyson (r 220 mm, F/1.8): surface 0.46/1.83/3.28
  px vs planes 0.07/0.03/0.04 px.  A blur proportional to lambda,
  uniform over the slit, that no concentric design can correct.  Fix
  candidate: `Gr_vec = Order*lambda/RuleWidth * (s0 - (s0.Nhat) Nhat)`
  with `s0` = unit(h1HOE), no normalisation.  Gate: the Offner chain
  numbers above (the engine must reproduce the 'planes' column), or the
  Rowland-circle stigmatic property.  Until it lands, engine SRF/CRF
  carry the inflated blur and the record prints both columns.
- **GRATING OPL JUMP -- engine finding #3 (2026-10-01, CC's lane; gate
  `tGratingOpl`).**  With the chord-ruled DIRECTIONS fixed (799498b), the
  engine's rays through the Offner seed at order -1 converge to 0.05 um,
  but the pupil OPD it reports on a reference sphere about that focus is
  **4.06 waves rms** (order 0: 8e-11 m, so the terminal is right).
  Rays and path lengths disagree.  `Snells_Law_Grating` adds
  `dL = (nb r - na i) . rho_prj` = `(m lambda/d)(s0 . rho_prj)` with the
  hit vector projected into the LOCAL tangent plane; the groove-count
  phase of equidistant groove planes is `(m lambda/d)(s0 . rho)`, rho the
  hit vector from the VERTEX along the fixed ruling direction s0.  The
  difference `(m lambda/d)(rho.N)(s0.N)` ~ `(m lambda/d) rho^3/(2R^2)`
  is cubic -- 12 waves at the Offner grating's 45 mm footprint, R =
  250 mm -- and ZERO on a flat grating, which is why the air fixtures
  never saw it.  Engine-free confirmation (chain OPL on the same
  sphere): chord phase 0.0004 waves rms, local-projection phase 4.9
  waves.  Fix candidate: `dL = Order*lambda/RuleWidth * dot(s0, rho)`
  (s0 the unit rule direction in the vertex plane), plus the
  reflection/refraction eikonal part unchanged.  Until it lands the
  propagation twin's order -1 numbers are this defect.
- **Far-field terminal for a spectrometer (the twin's deck).**  The
  Rx_Cass_FarField idiom on a REFERENCE sphere, not FEX's exit pupil:
  the Offner is telecentric (exit pupil at infinity; FEX finds a
  crossing 484 m PAST the focus, where the reversed rays never go).
  `spectrometer_rx(..., 'terminal','farfield','L_ref',L)` writes
  FP_return (Return, flat, at the FPA) -> ExitPupil (Return, sphere
  radius L centred on the chief's focus, vertex L upstream, psi along
  the beam, KrElt = -L, zElt = L, FarField) -> FPA; `spectrometer_wave`
  re-poses it per (field, lambda) with `macos.set_xp` and moves the two
  FPA vertices onto the chief pierce so the PSF grid is centred on the
  chief (centre pixel N/2+1).  Grid index 1 = global X, index 2 =
  global Y (measured: the dispersion offset sits in index 2).  FPA pitch
  `lambda L/(N dx_ep)`; the window is `ngridpts x lambda F`, so
  ngridpts 127 spans +-2.4 px at 380 nm.  Validated on the order-0
  relay: Airy spot, 94 % ensquared in one pixel.
- **Apertures and clearance (addendum 6, 2026-10-01).**  The aperture
  frame is `xObs` AS WRITTEN (the parser's default is the cyclic
  permutation of psi -- `(psi3, psi1, psi2)`, i.e. -x for psi = (0,0,-1)),
  `zObs = psi`, `yObs = psi x xObs`; `ApType Circular` is `ApVec =
  (radius, xc, yc)` in that frame, `Rectangular` is `(x1, x2, y1, y2)`.
  `spectrometer_rx(..., 'apertures', true)` declares every surface's
  aperture from the chain's multi-field, multi-lambda footprint + margin
  and writes `xObs=` explicitly; gate: not one ray vignetted
  (`tSpectrometerRx` 'dyson_apertures').  Obstruction is NOT a ray-trace
  property (the sequential trace never tests a ray against an element it
  is not traversing): `spectrometer_clearance` scores every leg against
  every body it does not traverse (aperture + mount, lifted onto the
  surface; one physical part grouped across its surface records; the
  slit mask and FPA package as mechanical bodies) and the stages FAIL on
  a negative entry.  The Offner must sit at >= 0.22 R to pass the
  grating; the Dyson's slit/FPA mechanics clear by ~1 mm with no cold
  shield (the fold-prism item).
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

**Departure ladder at fixed scale (dyson5 beat 3, r = 220 mm, F/1.8,
54 mm slit; `mmacos/challenges/dyson5/dyson5_s3.txt`):** the concentric
knobs (R_g factor, face offset) take keystone 0.095 -> 0.038 px; an
axisymmetric asphere/conic on the block face is INERT (a 1-D scan is a
steep bowl at zero -- it acts on every field alike and cannot cancel an
h^4 residual); de-concentring the block (centre 0.86 mm off the
grating's along the dispersion) takes keystone to 0.011 px (the ~1 %
design rule).  The blur (CRF 2.1 px, EE 0.48) is the fifth-order
residual at the EFFECTIVE field (the dispersed image adds ~12 mm to the
slit offset) and moves only with size (EE 0.74 at r = 341 mm) or with
the compact variant's separate mirror + meniscus.

**R4, the compact variant (meniscus corrector in the air gap, with the
de-concentred block and the grating radius open), at the same 220 mm:**
keystone 0.0026 px, smile 0.0051 px, CRF 1.33 px, EE 0.76 -- the
free-radius result at 63 % of the length and 27 % of the glass.  The
meniscus alone is worse than R3; the R4 solve ends on its bounds (a thin
weak plate right after the block) and the landscape is multimodal.
Slit diffraction loss past the F/1.8 acceptance: < 2.5 % over the band
(36 um slit; measured 0.08-2.48 %, sinc^2 0.28-1.83 %, factor open).

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

Definitions in Mouroulis & Green 2018's form (Sec. 4.1; digest
`mmacos/challenges/dyson5/NOTE_mg2018_digest.md`):

- **SRF(y) = rect(slit) ⊗ LSF_spectrometer(y) ⊗ DET(y)** (spectral);
  **CRF(x) = LSF_system(x) ⊗ DET(x)** (cross-track; Joe's "XRF");
  **ARF(y) = rect(slit) ⊗ LSF_telescope(y) ⊗ rect(integration)** (along
  track, the telescope's alone -- the spectrometer does not enter it).
  Resolution = FWHM.  The incoherent chain is legitimate when the Airy
  DIAMETER at the longest wavelength is under the pixel and the slit
  width: 2.44 λ F = 11.0 um at F/1.8, 2500 nm, against 18 um -- met;
  the full partially-coherent slit calculation moves things ~10 %,
  which is what the propagation twin measures.
- **Field-angle map / wavelength map**: FPA centroid `(x_spatial,
  y_spectral)` per (slit position s, wavelength λ) from `macos.spot` at
  the FPA, `set_src_fov` × `set_src_wvl` sweeps; x along the slit
  (spatial), y along `h1HOE` (spectral), in pixels of `pixel_m`, origin
  at the slit-centre / band-centre image.
- **Smile**: variation of the spectral centroid `y` ALONG the slit at
  fixed λ, peak-to-valley over the slit, per λ (max over λ reported).
- **Keystone**: variation of the spatial centroid `x` ACROSS λ at fixed
  slit position, peak-to-valley over the band, per s (max over s).
- **Uniformity**: invariance of the SRF through field and of the CRF
  through wavelength (a smooth SRF variation WITH wavelength is not a
  uniformity concern).
- **Radiometric chain** vs λ: throughput (Fresnel/coating), grating
  efficiency (scalar blaze closed form -- the engine carries one order),
  FPA QE (table), slit loss (the one MEASURED term: field at the
  grating plane, energy outside its aperture).  Ghost check (detector
  specular → grating → higher order back to the FPA; "more prominent in
  Dysons") = the scorer's open question.

**Design principles (Sec. 5.3) = the merit function's rules:** (1)
distortions to ~1 % of a pixel at design, ~3 % after tolerancing;
(2) > 75 % of the diffraction energy inside the pixel at every λ and
field; (3) degraded spots are acceptable and desirable when they buy
uniformity; (4) the grating is the stop.  Corollary the paper states:
optimize for point imaging first and uniformity later and you start
from a bad place -- the pixel-unit smile/keystone maps go INTO the
native-optimize merit from the first pass (beat 4).

Jim's realism (recorded, not scored): as-built SRF 2.5–3 px; photon
limited; 2-px slits.

## 6. Spec of record (Joe, 2026-09; "made up but EMIT to the digit")

F/1.8; FPA 3000 × 500 px at 18 um (slit 54 mm, spectral 9 mm);
380–2500 nm (4.24 nm/px); smile/keystone < 0.1 px (0.2 acceptable);
SRF < 1.5–2.0 px FWHM; XRF (= CRF) < 1.5 px FWHM; radiometric gain vs λ.
The spec sits in the paper's ALIS regime (Table 3: Dyson, 380–2500 nm,
7 nm, 3200 spatial px), with finer sampling.

Reference columns reported beside it (never scored against):

| column | source | numbers |
|---|---|---|
| performance class | Table 2 = the Fig. 13 long-slit **Offner** (the paper's caption says "Fig. 11", a typo) | F/2.8, 48 mm slit, 30 um px, 10 nm/px; smile < 0.3 % px, keystone < 2 % px, ensquared > 0.76, SRF FWHM < 1.35× sampling, CRF < 1.1×, SRF var. with field < 4.5 %, CRF var. with λ < 2 % |
| Joe's regime, published | Table 5 / Figs. 19–22, freeform **prism** Dyson (BPDS) | 3200 px, 18 um, F/2, 57.6 mm slit, 54 cm long, 19.2 cm prism; achieved smile 0.6 um (3.3 % px), keystone 0.2 um (~1 %); compact variant: separate mirror near the concentric-aplanatic condition + meniscus, ~60 % size, six more air-glass faces |
| public Dyson | Carbon-I (arXiv:2505.22545) | F/2.2, 2040–2380 nm at 0.7 nm/px, 3072 × 512 at 18 um, slit ≥ 54 mm × 36 um, smile/keystone ≤ 15 % px, SRF ≤ 2.5 nm, fused-silica block with an even asphere, concave spherical grating on N-BK7 in AIR, efficiency > 0.84 |

Pixel fraction is the convention: Joe's 0.1 px on 18 um is 1.8 um, the
Offner table's 0.3 % on 30 um is 0.1 um, the BPDS achieved 0.6 um on
18 um.  The 54 cm × 19 cm BPDS is the published cost of this regime;
beat 1's concentric seed (r ≥ 213 mm, R_g 687 mm) is the same order.

## 7. References

- J. Dyson, "Unit magnification optical system without Seidel
  aberrations," JOSA 49, 713 (1959).
- L. Mertz, "Concentric spectrographs," Appl. Opt. 16, 3122 (1977).
- P. Mouroulis & R. O. Green, "Review of high fidelity imaging
  spectrometer design for remote sensing," Opt. Eng. 57(4), 040901
  (2018).  PDF on disk in `mmacos/challenges/dyson5/` (SPIE copyright:
  git-ignored, cite only); digest `NOTE_mg2018_digest.md`.  It does not
  restate the concentric condition (cites Dyson 1959) -- the numerical
  verification in §3 is the basis.
- C. L. Bradley et al., "The Optical Design of the Carbon-I Imaging
  Spectrometer," arXiv:2505.22545 (2025).
- JPL patents US 6,181,418 (concentric spectrometer) and US 8,520,204
  (Dyson-type with improved image quality and low distortion) -- the
  meniscus / field-flattener departures from the classical seed.
