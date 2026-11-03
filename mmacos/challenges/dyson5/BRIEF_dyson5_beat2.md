# dyson5 beat 2 -- report (2026-09-30, TO lane on Fable)

Brief: `macos/BRIEF_to_dyson5.md` build-order items 2 (seed) and 3 (scorer),
on CC's fixed engine (b000390 glass catalog, fcd85d7 medium-aware kernels,
aba6350 table precision; build 17:27).  Beat-1 gates on it: tGlassDispersion
3/3 (CaF2 leg now live), tGratingImmersed 4/4.

## Delivered

| item | state |
|---|---|
| `design/src/spectrometer_geom.m` | ONE chain for both forms; chief aim through the grating vertex, groove period (band across the 9 mm FPA) and FPA focus solved by exact 3-D trace; `G.trace`/`G.aim` handles for gates; `P.grating_model` 'surface' (= engine) / 'planes' (straight-ruled) |
| `design/src/spectrometer_rx.m` | MACOS deck emitter (conventions in its header) |
| `tests/tSpectrometerRx.m` (SUITE_FAST) | **2/2 PASS**: engine chief at the grating vertex and at the FPA = chain to 1e-9 m; EVERY engine ray (launch directions from `ray_hist`) re-traced by the chain lands within 1e-9 m; engine source point = the slit; the engine's band spans the FPA height to 1 % |
| `design/src/spectrometer_score.m`, `_chain.m`, `_fwhm.m` | engine-ray and chain-ray scorers, shared FWHM chain |
| runner s1 (emit + layout) and s2 (score + maps + radiometric chain) | records `dyson5_s1.txt`, `dyson5_s2.txt`, figures `_s1_layout.png`, `_s2_maps.png`, `_s2_rad.png` |

## The s2 record (seeds, before any correction; conventions in the file)

| | Dyson seed (r 220 mm, F/1.8) | Offner seed (R 0.5 m, F/2.8) | spec |
|---|---|---|---|
| smile max | 0.006 px | 0.000 px | < 0.1 |
| keystone max | 0.095 px (slit ends) | 0.000 px | < 0.1 |
| SRF FWHM, engine | 3.06 px | 3.03 px | < 1.5-2.0 |
| SRF FWHM, chain straight-ruled | 2.03 px | 2.04 px | (2-px slit floor = 2.0) |
| CRF FWHM | 2.28 px | 1.03 px | < 1.5 |
| ensquared (1 px), engine / straight-ruled | 0.06 / 0.42 | 0.13 / 1.00 | > 0.75 (paper) |

The chain's 'surface' column reproduces the engine (SRF 3.07 vs 3.06, smile/
keystone identical) -- the two scorers agree on the same groove model, which
is what makes the next line a finding and not a discrepancy.

## ENGINE FINDING #2 (CC's lane): the grating groove model

`Snells_Law_Grating` normalises the projected rule direction (DPERPUVEC onto
the vertex plane, then unit `shat` in the local tangent plane), so the groove
period is constant ALONG THE CURVED SURFACE.  A straight-ruled concave
grating (the element type's own name; the CODE V / Zemax convention: spacing
in the vertex tangent plane) has equidistant groove PLANES, i.e. the
tangential kick `m lambda/d` times the UN-normalised projection.  Measured on
the chain, spectral rms blur at the slit centre, 380 / 1440 / 2500 nm:

| form | surface (engine) | planes (straight-ruled) |
|---|---|---|
| Offner | 0.42 / 1.60 / 2.78 px | 0.000 / 0.000 / 0.003 px |
| Dyson | 0.46 / 1.83 / 3.28 px | 0.07 / 0.03 / 0.04 px |

A blur proportional to lambda and uniform over the slit, which no concentric
design can correct -- it is what sets every engine SRF above.  Fix candidate
(one expression): `Gr_vec = Order*lambda/RuleWidth * (s0 - (s0.Nhat)*Nhat)`
with `s0 = unit(h1HOE)`, no normalisation.  Gate: the Offner 'planes' row
above (the engine must reproduce it; `spectrometer_score_chain` is the
reference), or the Rowland-circle stigmatic property.  Note the existing
grating tests (tCodeVGrating / test_api_rx_grating) are self-referential on
the engine's own values, so nothing in the suite would have caught this.

## Engine conventions pinned (reference doc sec. 2)

1. `ChfRayPos` is where rays START (must lie before the first surface -- a
   value past the 0.5 mm Dyson face lost every ray); at LOAD the engine folds
   `ChfRayPos + zSource*ChfRayDir` in once, after which `ChfRayPos` IS the
   physical source and `set_src_fov` must be handed the slit point itself.
2. `macos.stop(k)` aims IMMEDIATELY from the source state it finds; its first
   pass on the Dyson deck is 3.6 mrad short (1.7 mm miss at the grating), a
   second call converges.  Declare the stop first, then write the exact
   chief; the engine keeps it to 1e-16.  (CC may want to look at
   ChiefRayAiming's first-pass convergence through a refracting block.)
3. A `Return` coincident with the `FocalPlane` leaves zero path length and
   the engine drops the rays (1110 of 1258); the emitter's tail is a
   pass-through `Reference` 1 mm upstream.
4. `ray_info_get` returns the OUTGOING direction at the traced element.

## Reading of the seeds (for beat 3)

- Offner: with straight-ruled grooves the seed is diffraction-clean at the
  ring (blur 0.03-0.06 px, EE 1.00); its SRF 2.04 px is the 2-px-slit floor.
  Joe's "SRF < 1.5-2.0 px" with a 2-px slit sits AT the floor; Jim's
  as-built 2.5-3 px is the realistic band.
- Dyson: smile 0.006 and keystone 0.095 px pass; the spatial blur at the
  slit ends (0.7 px rms, CRF 2.3 px, EE 0.42) is the concentric residual plus
  the 0.5 mm face offset and the dispersed geometry -- the paper's departure
  ladder (asphere on the block face; separate mirror + meniscus) is beat 3's
  work, now with a scorer to drive it.
- Radiometric chain: a single blaze at 1 um gives 0.015 at 380 nm -- the
  band needs a multi-blaze / structured grating (as EMIT); the QE table is a
  placeholder.  The slit loss (the one measured term) is the wave twin's.

## Next

Beat 2c: the propagation twin on the Offner (all in air; CC's medium-aware
kernels now also allow the Dyson): complex field at the FPA per (field,
lambda) through NF/DFT legs, PSF centroid maps vs the ray maps, SRF/CRF from
the propagated PSF, slit loss at the grating plane.  Beat 3: the Dyson
departure ladder under the scorer.  Beat 4: native optimize with the
pixel-unit smile/keystone operands in the first pass (the paper's rule).
