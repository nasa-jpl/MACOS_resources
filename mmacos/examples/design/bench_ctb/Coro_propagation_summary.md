# CTB diffraction propagation — recipe of record

> Rev 2 (2026-08-04): Dave's draft, reviewed by CCL and rewritten with
> the review folded in.  Rulings: hard-edge masks first (no vortex
> exists in the tree — the pymacos apodize docstring mention is
> aspirational; complex/phase masks are stage 2 and need a
> `cfield_apodize_c` api addition); all masks applied in MATLAB for
> stage 1; per-wavelength focal-plane grids.  Target deck:
> `ctb_planar_stageF.in` (bench_ctb).  Validated precedents cited are
> the pymacos PROPER-compare phases and the CoroExample recipe.

## The scheme

Propagation from source, past the DMs, through the coronagraph masks
and Lyot stop to the FPA runs as multiple planes of diffraction
(CPROPAGATE in propsub.F), starting and ending at reference /
obscuring / focal-plane / return surfaces.  Those surfaces are planes
or spheres, psi aligned to the beam, parameters derived from beam
conditions: ORS sets a sphere's curvature from the beam, SRS slaves a
second sphere to the first, FEX finds the sphere for sphere-to-plane.
Between diffraction legs the trace is geometric, so real-optic OPD
(including DM actuator figures as grid surfaces) accumulates onto the
complex field — the hybrid geometric/diffraction contract.

Reference surfaces may sit where rays run backward to reach them; the
engine permits negative ray lengths at reference-class surfaces
(`ifLNegOK`, default TRUE; commands `LNEG` / `NOLNEG`).  Do not
disable it.  Caution when placing spheres near real optics: ConSrf
picks intersection roots by |L²−mpr| proximity with no flow-of-light
sense — prefer placements with clean daylight to real surfaces.

## Propagator selection (decision rule, per leg — no open "or"s)

- Short collimated plane-to-plane legs: NF p2p (validated 2.4e-14,
  PROPER-compare Phase 2/3a).
- Pupil→pupil through a focus: FF two-part via the reference sphere
  (Phase 4a).
- Genuinely near-focus intermediate planes: the Siegman–Sziklas
  near-focus recipe.
Record the chosen propagator per leg as comments in the deck.

## Mask convention (stage 1 ruling)

ALL masks applied in MATLAB by multiplying the complex field
(`cfield_apodize`; real 0/1 or graded-amplitude arrays).  Mask
elements stay `Element= Reference` — consistent with the deck as
built.  NOTE the documented trap this ruling avoids: an obscuration
declared on a Reference element clips RAYS ONLY; the diffraction
wavefront passes untouched (the Phase-5 FPM lesson).  If any plane is
later moved engine-side it must become `Element= Obscuring` with
declared obscurations — never rely on obscuration metadata on a
Reference.  Stage 2 (phase masks, e.g. a scalar vortex) requires the
complex-mask api `cfield_apodize_c(OK, MASK_RE, MASK_IM, N, iElt)` —
engine-first, deferred until a phase mask exists to apply.

## Per-λ focal grids (hard rule)

The FF focal-plane grid spacing scales with wavelength.  Every
focal-plane mask (FPM, field stop) is evaluated per wavelength on
that λ's grid via runtime `dx_at` at the mask element — never a
cached array (the Phase-2 lesson behind the 1e-13 agreement).  Use
odd nGridpts (511-class precedent).  Center masks on the chief-ray
pierce point, not the array center, where they differ.

## Leg-by-leg (ctb_planar_stageF.in)

1. Plane reference after DM1 (elt 2), Vpt at the chief-ray intercept
   — starts NF p2p leg 1.  (DM1's surface, and its poke figure, is
   traversed geometrically BEFORE this plane — deliberate.)
2. Plane reference before DM2 (elt 3) — ends NF leg 1.  (DM2 figure
   accumulates geometrically after it — deliberate.)
3. Geometric to a sphere return at a pupil before Focus23 (elt 5);
   place via add_pupil referencing Focus23.
4. FF part 1 to the focal plane; FF part 2 backward to the sphere
   (the PROPER pattern).  No mask at Focus23.
5. Geometric to a plane reference after OAP3 (elt 6); NF p2p to the
   Apodizer (elt 7); apply apodizer amplitude mask in MATLAB.
6. Geometric to a sphere return at a pupil before the FPM (elt 9);
   add_pupil referencing the FPM element.
7. FF part 1 to the FPM plane; apply the HARD-EDGE FPM mask in
   MATLAB (per-λ grid); FF part 2 backward to the sphere.
8. Geometric (or NF per the decision rule) to the Lyot (elt 11);
   apply the Lyot amplitude mask in MATLAB (stage-1 ruling — the
   deck's Reference Lyot does NOT mask the wavefront engine-side).
9. Geometric to a sphere return at a pupil before FieldStop
   (elt 13); add_pupil referencing FieldStop.
10. FF part 1 to the field-stop plane; apply the field-stop mask in
    MATLAB (per-λ grid); FF part 2 backward to the sphere.
11. Geometric to the Backend (elt 15); geometric to a sphere return
    at a pupil before the FPA (elt 17); add_pupil referencing FPA.
12. FF part 1 to the FPA — the final propagation; generates the PSF.

## Broadband

Repeat at nwf wavelengths across the science band; sum PSFs
incoherently (MATLAB or COMPOSE ADD via the compose() wrapper).
Start monochromatic; then nwf = 3–5.  Masks re-evaluated per λ
(rule above).

## Validation ladder (acceptance)

1. Per-leg cross-compare against PROPER on the CTB prescription —
   sum-norm at pupil planes, peak-norm at PSF planes, centroid
   alignment (the run_proper_tests pattern).
2. End-to-end hard-edge FPM + Lyot suppression on the CTB layout,
   tied to the Phase-5-class result — this run connects the new
   chain to the validated record.
3. Dark-zone contrast curve at the FPA as the acceptance metric
   (contrast.py analog in MATLAB).
Substrate is the DL-geometric CTB (0.0014 λ); the DST2 reference
(0.5 λ converter artifact) is a geometry reference only, not a
diffraction substrate.

## Stage 2 (deferred, in order)

Complex-mask api (`cfield_apodize_c`, engine-first + relink chain +
pymacos export) → scalar vortex (charge 6) with singular-pixel
treatment documented and a PROPER vortex cross-check at matched
sampling → graded/complex apodizer features.
