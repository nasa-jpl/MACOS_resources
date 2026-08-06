# ctb — coronagraph-testbed Bench example

Two layers over the same 8-OAP / 2-DM bench:

| Layer | Built by | Decks | Answers |
|---|---|---|---|
| Geometry | `example_ctb.m` | `ctb_planar_stage{A..F}.in` | where the optics sit, and whether the bench reaches the diffraction limit |
| Diffraction | `ctb_prop_layout.m` | `ctb_dcr.in`, `ctb_s2s_dcr.in` | what the field does through the coronagraph masks |

The geometry layer is described first; the diffraction layer starts at
[Diffraction layer](#diffraction-layer--the-coronagraph-chain).

A worked `macos.design.Bench` example: build a **DST2R-like, all-reflective
coronagraph relay** sequentially, then optimize it in physically-staged steps.
Teaching example — every block in the runner states its **parameters**, the
**add-optic call**, the **objective** (form a pupil here, focus there), and the
**test** that confirms it.

## Topology (light order)

A classic 2-DM coronagraph relay alternating pupil and focus planes:

```
source → OAP1 → DM1(pupil) → DM2(pupil) → OAP2 →[focus23]→ OAP3
       → apodizer(pupil) → OAP4 → FPM(focus) → OAP5 → Lyot(pupil)
       → OAP6 → field_stop(focus) → OAP7 → backend(pupil) → OAP8 → FPA
```

- 8 off-axis parabolas (Kc=−1). DM1/DM2 are flat fold mirrors (the DMs, probed
  on the collimated pupil). apodizer/FPM/Lyot/field_stop/backend are passive
  `Reference` conjugate markers — the pupil/focus sites a real coronagraph
  populates with masks, and the per-stage optimization targets here.

## OAP geometry — arbitrary fold angle

`add_oap` takes the parent focal length `f` directly and supports any fold. An
off-axis parabola at angle of incidence `AOI` turns the chief by
`θ = 180 − 2·AOI`; the conjugate distance that realizes focal length `f` is

```
r = f / cos²(AOI)   ( = 2f/(1 − cosθ) )
```

DST2R uses **near-normal** folds (AOI ≈ 5°), which keep each OAP only slightly
off-axis (pole ≈ vertex) and minimize off-axis astigmatism. We seed AOI = 5°
and the DST2R parent focal lengths (mm): OAP1 2500, OAP2 1524, OAP3 1143,
OAP4 1350, OAP5 675, OAP6 635, OAP7 635, OAP8 762. DST2R is cited only as the
source of these seeds; the layout is a generic CTB, not the proprietary DST2R
coordinate set.

> **add_oap fix (local, unpushed).** Building this surfaced a bug in
> `MACOS_resources/mmacos/src/+macos/+design/Bench.m`: `add_oap` seeded the
> parent focal length with `(1+cosθ)` instead of `(1−cosθ)` — correct only at
> θ=90° (the sole regime its tests exercised), wrong for every other fold.
> Fixed to `(1−cosθ)` (90° results bit-identical, so `tBench`/`bench_layout`
> stay green) and given an `'f'` option. Also: a circular aperture on an
> off-axis section is applied about the parent **vertex** (far from the beam),
> so it blocks the whole bundle — OAP `aprad` is now metadata only; put
> functional stops on flat marker planes. Flag for a proper PR.

## Source model (MACOS point source)

The point source is a **section of a sphere** centered at the radiating point;
`Aperture` is the **numerical aperture**, sized to just fill the **limiting**
element. The limiting aperture here is the **DM** (radius 22.5 mm, the pupil
stop), not the oversized OAPs (radius 75 mm), so the source NA is set to put a
`R_DM·FILL` beam on the DM. The pupil beam then demagnifies down the relay by
each focus/collimate focal-length ratio (DM 21.4 → apod 16.0 → Lyot 8.0 →
backend 8.0 mm); the runner prints these and asserts none overfills.

## Why staged optimization

Each conjugate (collimated at a pupil, focused at a focus) lives on its own
plane. A single WFE solve at the detector can trade them against each other
(e.g. mis-collimate at the Lyot pupil and hide it as defocus), which ruins
pupil-plane masking. So each conjugate is solved with a **geometric** cost on
its own plane, in light order, freezing upstream optics. RMS WFE is reported
only as a final figure of merit.

- **Stage A solves Kr only** (a sphere / fixed-conic base) — far cheaper to
  fabricate and test than a figured asphere. `optimize_conic(..., 'kr')` vs
  `'kr_kc'` selects the DOF; switch to `'kr_kc'` if the residual is too large.

## Run

```
cd ~/dev/macos_sandbox/ctb
matlab -batch "run('example_ctb.m')"
```

Requires `MACOS_HOME` set (the MEX aborts on engine init without
`macos_param.txt`) and the mmacos MEX built for this platform. Uses the local
(unpushed) `add_oap` fix in `MACOS_resources`.

## Results (current seed, AOI 5°, 500 nm)

- All 3002 rays trace at every element; **zero vignetting** (seed and optimized).
- Builder-vs-engine chief agreement 4.7e-12 mm.
- Each stage improves: OAP1 collimates to 4.6e-4 mrad; foci to ~0.1 µm rms
  spot; pupils to ~1e-4 mrad.
- **Final FPA RMS WFE ≈ 0.0014 waves (0.71 nm)** — diffraction-limited.

## Outputs

- `ctb_planar_seed.in` — as-built (pre-optimization) prescription.
- `ctb_planar_stage{A,B2,B3,C,D,E,F}.in` — per-stage frozen prescriptions.
- `ctb_planar_opt.in` — final optimized source-imaging prescription.
- `ctb_pil150.in`, `ctb_pil75.in` — pupil-imaging configurations (PIL step).
- `ctb_planar_params.png` — annotated input-parameter schematic.
- `ctb_planar_view_rx.png`, `ctb_planar_view_std.png` — beam renders.
- `ctb.mat` — `out` struct (params, element indices, per-stage conics, WFE,
  zone-spot map, polarization metrics, PIL results).

---

# Diffraction layer — the coronagraph chain

`ctb_planar_stageF.in` is geometric everywhere: it places the optics but
propagates no field. The diffraction layer inserts reference and return
surfaces into that same bench so the complex field can be carried from the
source to the detector as a chain of near-field and far-field legs, with the
real optics traced geometrically between them (OPD, including a DM grid
figure, accumulates onto the field along the way).

Two models, same optics:

| Model | Deck | Elements | Propagated legs |
|---|---|---|---|
| compact | `ctb_dcr.in` | 31 | DM1→DM2, the three focal quartets, the terminal |
| full | `ctb_s2s_dcr.in` | 44 | every inter-optic leg as well |

Station indices differ between them, so every driver takes an `'elt'` map:

```
compact: DM1=2 DM2=5 Apodizer=13 FPM=17 Lyot=20 ExitPupil=30 FPA=31
full:    DM1=2 DM2=5 Apodizer=16 FPM=22 Lyot=27 ExitPupil=43 FPA=44
```

Both decks are pre-aligned — the reference spheres are already placed — so
they load and produce a centred PSF with no runtime setup.

## The mask block

Each intermediate focus is a **four-surface quartet**, every surface
`Element=Return`:

| Surface | Type | PropType | Position | zElt |
|---|---|---|---|---|
| `<focus>_FPreturn` | Flat | Geometric | at the focus | 1e22 |
| `<focus>_EPreturn` | Conic, Kr = −R | NF1 | one R before the focus | +R |
| `<focus>` | Flat | NF2 | at the focus — **the mask plane** | 1e22 |
| `<focus>_EPreturn2` | Conic, Kr = −R | Geometric | same sphere | +R |

**Both sphere zElt are +R, identical to every digit.** `propsub.F` builds the
NF1 chirp from `zStart = zElt(sphere)` and `zEnd = zElt(iElt+1)` — the element
*after* the mask, i.e. `EPreturn2`. Equal zElts give chirp argument zero, so
the sandwich is transparent and the round trip is the identity; `EPreturn2` at
−R gives an argument of order 2R, a defocus that no ray check catches. The
mask element's own zElt is not read by the chirp — the two committed decks
disagree on it (1e22 compact, +R full) and both are correct. `tCtbProp` pins
the sphere equality and the round trip.

R is measured, not chosen: it is the exit-pupil conjugate of that focus, found
by running FEX on a truncated deck ending in the same triple. Committed
values (BaseUnits mm): Focus23 7017.8526, FPM 1000.0841, FieldStop 415.9278,
terminal ExitPupil 359.9461.

The terminal is the same pattern with the second sphere dropped: `FP_return`
(Flat, Geometric) → `ExitPupil` (Conic, FarField, Kr = −R, zElt = +R) → `FPA`.
`macos.design.Telescope.add_pupil` emits exactly this, `PropType= FarField`
included.

**Masks are applied in MATLAB, never declared in the deck.** Propagate to the
mask plane, multiply the complex field (`macos.apodize` for amplitude,
`macos.apodize_complex` for phase), and continue with
`'reset_trace', false`. An obscuration declared on a `Reference` element
clips rays only — the diffraction wavefront passes through it untouched.

## Run

Every driver takes `'model_size'`, `'outdir'`, `'visible'`, and an `'elt'`
index map; they default to the compact deck.

```
matlab -batch "run('ctb_prop_layout.m')"                       % regenerate both decks
matlab -batch "ctb_coro_compare('coro',false)"                 % bare, compact vs full
matlab -batch "ctb_coro_compare"                               % apodizer + FPM + Lyot
matlab -batch "ctb_contrast"                                   % dark-zone contrast vs lambda/D
matlab -batch "ctb_optimize_masks"                             % occulter / Lyot radius sweep
matlab -batch "ctb_planet('sep_lamD',6,'flux_ratio',1e-3)"     % off-axis companion
matlab -batch "ctb_bandpass('nwf',5,'band_frac',0.10)"         % finite bandpass
matlab -batch "ctb_vortex('charge',6)"                         % scalar vortex mask
matlab -batch "ctb_proper_compare"                             % PROPER arbiter, FPM leg
matlab -batch "ctb_train_render"                               % bench layout figure
./../../../run_mmacos_tests.sh ctb                             % tCtbProp, 8 checks
```

`ctb_prop_layout` writes `ctb_dcr_gen.in` and `ctb_s2s_dcr_gen.in` beside the
committed decks; the committed ones stay the reference.

## Validated numbers

Bare (no masks), model 512, nGridpts 255, λ = 500 nm. Provenance: the
`tCtbProp` pins, which reproduce the committed decks.

| Quantity | compact | full |
|---|---|---|
| FPA peak intensity | 7.006e-2 | 6.030e-2 |
| peak pixel (of 512) | [257,257] | [257,257] |
| detector pitch | 2.4039e-5 m | 2.4039e-5 m |
| peak-normalised correlation, compact vs full | 0.998863 | |

Model-vs-model agreement is not validation. The arbiter is MATLAB PROPER on
the FPM through-focus leg at matched sampling (`ctb_proper_compare`, needs
`~/dev/proper_matlab`): focal pitch ratio 1.0000, peak-normalised correlation
1.000000, centroid offset 0.000 px.

Coronagraph performance, model 1024, occulter 2.70 λ/D and Lyot 0.50 of the
pupil (`ctb_optimize_masks`, the interior contrast null of a
2.0–3.5 λ/D × 0.45–0.75 sweep): dark-zone mean contrast 2.9e-7 over
3–15 λ/D at 25% throughput. A hard-edge occulter with no apodisation gives
~1e3 suppression; 1e6-class needs band-limited or apodised masks.

Generated versus committed (`ctb_prop_layout`, same sampling): chief ray at
all ten real optics 1.64e-11 mm against the bare deck (the committed decks
give 1.36e-11 mm under the same measure), bare correlation 0.999999 on both
models, peak ratio 1.000000 for the full model.

## Gotchas

- **The focus is the FFT DC pixel, `floor(N/2)+1` (1-based), not `(N-1)/2`.**
  Every mask must be centred there. The half-pixel error leaked starlight
  asymmetrically past the occulter and cost a round to find; the vortex, whose
  singular pixel sits at the centre, suffered most.
- **Focal masks are rebuilt per wavelength.** The focal-plane pitch scales with
  λ, so a cached mask array is wrong at every other wavelength.
- **Broadband sums on one common detector grid.** The FarField FPA re-grids per
  λ, so each monochromatic PSF occupies the same pixel size on the N×N array
  and a naive array sum cancels the chromatic effect exactly. `ctb_bandpass`
  resamples each λ onto the nominal-λ physical pitch, flux-conserving.
- **`model_size` ≥ `nGridpts`.** The decks declare 255, so model 512 or more.
  One MATLAB process per model size — a size transition inside a live process
  can corrupt the engine heap (PLAN.md §0). Batch wrappers end with `exit(0)`;
  example scripts never do.
- **Prescription regexes must be line-anchored** (`^\s*` plus `'lineanchors'`).
  `iElt` is a substring of `psiElt`, so an unanchored `iElt=\s+\d+` eats the
  leading digit of `psiElt`.
- **The generated compact deck differs from the committed one by design.**
  `ctb_dcr.in` propagates the DM1→DM2 leg over 399.94 mm where the chief
  distance between those stations is 499.92 mm. The generator uses the true
  distance, which is the whole of the 0.9977 peak ratio between them. The full
  model, whose committed deck already has the correct length, reproduces
  bit-for-bit. Flagged, not changed.
- **Some installed mexes predate the `'plane'` argument** on
  `macos.complex_field`; the veneer passes three arguments and errors against
  them. `ctb_proper_compare` and `tCtbProp` fall back to the two-argument raw
  dispatch `mmacos('complex_field', iElt, reset)`.

## Pupil image quality — the zone-spot standard

`macos.pupil_zone_map(DM1, FPA)` partitions the DM1 pupil into zones (each a
cone of rays), and measures each zone's spot at the detector. A perfect relay
maps every zone to one FPA point. **Current result: median 0.027 µm, worst
0.05 µm rms** across 25 zones → imaging is essentially perfect. This is a
**reusable package function**
(`macos.pupil_zone_map(PUPIL_ELT, IMAGE_ELT, 'ngrid',N, 'shape',
'square'|'annular')`), added alongside `macos.spot`/`macos.pupil_quality`.

## Polarization diagnostic (fold aberration)

The runner coats the mirrors with aluminium (`macos.coating`), turns on
polarization, and reports the FPA Jones-pupil retardance and diattenuation
(`macos.jones_pupil` → `macos.pol_maps`), separating the pupil **mean** (mostly
geometric frame rotation) from the **variation** (what sets a coronagraph
contrast floor). Current planar bench: retardance mean 0.021 rad (var 4.9e-5),
diattenuation mean 0.0036 (var 8.4e-6).

**Fold polarization compensation is non-trivial** (measured, `diag_pol2`-style):
- In-plane fold *alternation* (−X,+X,…) does **NOT** cancel fold retardance —
  the mean is unchanged and the variation gets worse. DST2R's in-plane ±X
  alternation is for **packaging** (compact beam retrace), not polarization.
- Naive **crossed-plane** folding (a 3-D bench) cuts *diattenuation* ~90× but
  does **not** beat the planar bench on the contrast-relevant *retardance
  variation*, and costs ~10× WFE (out-of-plane folds add astigmatism the
  per-conjugate Kr/Kc solve can't remove).
- Proper compensation pairs *equal-AOI* folds with *exactly*-crossed planes AND
  balances the induced astigmatism — a layout+optimization task, **deferred**.

So we ship the **planar** bench (compact, diffraction-limited) and report
polarization as a diagnostic.

## PIL design step — pupil-imaging lens (additional Rx)

The CTB images the **star** at the FPA (source imaging). A **pupil-imaging
lens** near the star focus switches the camera to view a **pupil** instead. The
runner builds two, emitting an Rx each, reproducing D. Marx's DST2R trade:

| PIL focal length | pupil image | camera position | Rx |
|---|---|---|---|
| 150 mm | 3.29 mm (larger) | reference | `ctb_pil150.in` |
| 75 mm | 1.23 mm (smaller) | −100.7 mm (forward) | `ctb_pil75.in` |

Size ratio 2.7× and camera move ~101 mm — matching Marx's 150 mm/75 mm
(~1000 px/500 px, camera +106 mm) behaviour. Both prescriptions trace clean.
`build_pil(f)` reuses the CTB front end through OAP8 and appends the lens +
camera at the pupil-image conjugate `s_i = 1/(1/f − 1/s_o)`.

### Render niceties

- `macos.view_rx` element labels are offset **perpendicular to the beam, in the
  layout plane** (alternating sides, faint leaders) so they don't print on the
  ray lines.
- `macos.view_std` uses a near-square panel grid (2×2 for 4 views) so each panel
  is ~4× larger at the same paper size.
- (Both are local, unpushed edits in `MACOS_resources`; `tBench` stays green.)

## Parallel faithful reference — `dst2/`

This directory is the **generic, development** CTB. The `dst2/` subdirectory
holds the **faithful DST2R** design (Brandon's actual CODE V layout via
cv2macos) as a parallel reference — same 8 OAP focal lengths and topology, but
Brandon's exact spacings/clocking. Develop here; track the real instrument
there. See `dst2/README.md`.
