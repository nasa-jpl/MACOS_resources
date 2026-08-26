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

**Mask sampling rule (2026-08-25): every mask surface is generated at 8×
sub-pixel resolution and binned to the model grid** — the binned value is
the pixel-averaged transmittance the band-limited field actually
experiences.  Amplitude masks always were (`ctb_mask_disk` /
`ctb_mask_softcircle`, K=8 area-weighted edges).  Phase masks now too:
`ctb_mask_vortex` complex-bins `exp(i·m·θ)` (accumulating the 64
sub-pixel phasor shifts at model resolution, so no 8N grid is ever
built) and `ctb_mask_phase` composes gray zone edges from the
supersampled disks.  This matters most at the vortex core: the
directly-sampled singularity sits on the stellar Airy peak and floored
the dark zone at 2.9e-7 regardless of the bench (ideal-pupil probe
reproduces it); complex-binning zeroes the core pixel smoothly and the
bench vortex reaches **1.41e-8 (21×) at unchanged 81% throughput** —
second-deepest mask after the APLC.  The ideal-pupil binned floor is
3.0e-9, so the bench residual (aberration/amplitude), not the mask, now
sets the vortex floor.

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
matlab -batch "ctb_vortex_lyot_sweep"                          % Lyot trade vs APLC/BLC
matlab -batch "ctb_efc_physics('band',true,'pol',true)"        % EFC under band + polarization
matlab -batch "ctb_phys_summary"                               % physics-campaign figure
matlab -batch "ctb_vortex_stations"                            % 7-station complex fields, +/- pol
matlab -batch "ctb_vortex_bandwidth"                           % floor vs bandwidth (2.5% spacing)
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
| FPA peak intensity | 6.990e-2 | 6.030e-2 |
| peak pixel (of 512) | [257,257] | [257,257] |
| detector pitch | 2.4039e-5 m | 2.4039e-5 m |
| peak-normalised correlation, compact vs full | 0.998895 | |

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
give 1.36e-11 mm under the same measure), bare correlation 0.999999 and peak
ratio 1.000000 on both models.

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
- **A near-field pair's planes go on their stations, not at a fraction along
  the segment.** Until 2026-08-06 `ctb_dcr.in` propagated the DM1→DM2 leg over
  399.94 mm where the chief distance is 499.92 mm — 0.8× it, from a builder
  that placed the two planes at 10% and 90% along the segment, leaving the end
  plane 399.94 mm *behind* DM1 and reached by a negative ray length. The leg is
  collimated pupil-to-pupil, so the only symptom was 0.23% in bare FPA peak;
  nothing failed. Corrected in place from the full deck's `Prop1` pair, so both
  committed decks now agree to all digits and the generator reproduces both.
  Analysis outputs produced before that date carry the 0.23%.
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

## Phase-factor export for external PROPER models

The CTB **full model** (`ctb_s2s_dcr.in`, 44 elements) can be exported as a
self-describing `.mat` so an **external PROPER user** — someone who has never
seen macos — can consume this model's per-plane fields and surfaces in their
own [PROPER](https://sourceforge.net/projects/proper-library/) run and check
their model against ours **plane by plane** (an interface check).

**Generate** (one MATLAB process, ~1 min, needs `MACOS_HOME` + the mmacos MEX):

```matlab
cd mmacos/templates/30_instruments/bench_ctb
ctb_phase_export            % writes ctb_phase_export_N1024.mat (+ preview + .fp.json)
```

**Consume** (needs only the `.mat` + MATLAB PROPER on the path — no macos):

```matlab
addpath ~/dev/proper_matlab
out_s2s       = proper_ctb_check('s2s',       'figure', true);
out_collapsed = proper_ctb_check('collapsed', 'figure', true);
```

### File: `ctb_phase_export_N1024.mat` (`-v7.3`)

All quantities are in **metres** (SI). Fields are on each plane's **own**
diffraction grid; the array centre / focus pixel is `floor(N/2)+1`.

| var | field | meaning |
|---|---|---|
| `meta` | `format_version` | **2** (v1 = stations/legs/spheres/screens; v2 adds `stations.EFL_m` + the `masks` block) |
| | `lambda_m`, `N`, `center_px` | 5.00e-7 m, 1024, 513 |
| | `base_unit_to_metre` | deck is mm → 1e-3 (CBM) |
| | `opd_sign` | **`OPD_m = -angle(E)·λ/(2π)`** — macos OPD is **opposite** PROPER `prop_add_phase` (pymacos `opd_sign_flip=true`). A consumer calls `prop_add_phase(bm, OPD_m)` directly. |
| | `opd_wrapping` | `OPD_m` comes from `angle(E) ∈ [-π,π]`, so it is **wrapped** (`\|OPD\| ≤ λ/2`). `E` is the primary carrier (`AMP·exp(iθ)` reconstructs it exactly); unwrap `OPD_m` before using it as a smooth additive screen. |
| | `grid_orientation` | `E(row,col)`: **row = +Y** (first index), **col = +X**. Verified: a **+X pupil phase ramp** `exp(+i2πk·X/D)` sends the FPA peak to **col < centre** (−X side). |
| | `orientation` | the measured probe result: `k=8` ramp → FPA peak `dcol=-32, drow=0`. A consumer asserts the same handedness from the `.mat` alone. |
| | `convention_focus`, `convention_p2p` | the two propagation conventions — **read these** (summarised below). |
| `stations(k)` | `name, iElt, kind` | one per real optic (8 OAPs + 2 DMs), mask plane (Apodizer/FPM/Lyot/FieldStop), and the FPA. `kind ∈ {optic, pupil, focus}`. |
| | `E, AMP, OPD_m` | complex field, amplitude, wrapped OPD on that plane (`\|E\|² = intensity`). |
| | `dx_m, z_along_chief_m, chief_pos_m` | plane pitch (SI), cumulative chief-ray path, chief 3-vector. |
| | `EFL_m` | **(v2)** focal length `\|Kr\|/2` (SI) of a powered `optic` station (OAP) — what a PROPER `prop_lens` needs; `NaN` for pupils/foci and for the ExitPupil (whose focusing radius is its FarField-sphere `R` in `legs`/`spheres`, not `\|Kr\|/2`). |
| `masks(i)` | `name, station, plane_kind` | **(v2)** the four shipped coronagraph masks as stand-alone arrays: Apodizer / FPM / Lyot / FieldStop. `plane_kind ∈ {amplitude_pupil, amplitude_focus}`. |
| | `M, dx_m, radius_m, active` | the real `N×N` mask array, its plane pitch, its defining radius in **metres** (grid-independent), and whether it is applied in the shipped chain (FieldStop is `active=false`, an all-ones placeholder). |
| | `builder, params, note` | provenance: which builder + parameters produced it (e.g. FPM `r_fpm_lamD=2.70`, Lyot `r_lyot_frac=0.50`). **Pupil masks (Apodizer/Lyot) transfer directly; the FPM is a FOCUS-plane occulter whose array is on the macos focal grid for reference — rebuild it at your own focal `dx` from `radius_m`** (`proper_ctb_run` does this). |
| `legs(j)` | `from, to, chief_len_m, prop_type, sphere_R_m` | the propagation table between consecutive stations: `NFPlane plane-to-plane` \| `through-focus quartet` \| `FarField` \| `geometric jump`. `sphere_R_m` is the reference-sphere radius where the leg is spherical. |
| `spheres(i)` | `feeds_station, sphere_iElt, E, AMP, OPD_m, dx_sphere_m, R_m` | the **feeding reference sphere** for each through-focus / FarField leg — the plane you seed PROPER from to replay that leg (see below). |
| `screens(k)` | `at_station, OPD_add_m, dx_m` | the OPD **added** reaching that plane, as the **difference of consecutive-station OPDs** (a clean per-optic split is not directly readable from the engine; the diff construction is what ships — `meta.screen_method`). |

A committed **preview** (`ctb_phase_export_preview.mat`, 96×-downsampled,
≈3 MB) mirrors the format for inspection without a regen. The full `.mat`
(≈320 MB) is **gitignored**; the committed truth is the fingerprint
`ctb_phase_export_N1024.fp.json` (dims + per-column norms + provenance, via
`jac_fingerprint.m`) — regenerate the `.mat` with the one line above.

### The two propagation conventions you must understand

**Through-focus & FarField legs — replay from the feeding sphere.** A
sphere→focus→sphere (or terminal FarField) leg reproduces macos **exactly**
only when PROPER is seeded at the **feeding reference sphere** (`spheres(i)`),
not the optic-plane field:

```matlab
s  = spheres(i);                                   % feeds e.g. 'FPM'
bm = prop_begin(N*s.dx_sphere_m, lambda_m, N, 'beam_diam_fraction',1.0);
bm = prop_multiply(bm, s.AMP);
bm = prop_add_phase(bm, s.OPD_m);                  % already sign-flipped
bm = prop_define_entrance(bm);
bm = prop_lens(bm, s.R_m);
bm = prop_propagate(bm, s.R_m);                    % -> the focus station
```

This matches macos at the focus at **intensity peak-norm corr 1.000000** (the
`ctb_proper_compare` arbiter class).

**Collimated pupil→pupil (NFPlane) legs — do NOT compare raw complex fields.**
macos's NFPlane p2p propagator reads the field on a **planar reference**
(local-plane curvature re-zeroed); PROPER's `prop_propagate` accumulates the
full Fresnel **quadratic reference-sphere phase**. So across a collimated leg
the **intensity agrees** (corr ≈ 0.95) but the **raw complex fields differ by
a large quadratic reference-phase term** (measured DM1→DM2 raw-field corr
≈ −0.8). This is a **reference-phase convention difference, not an error**
(the NF p2p propagator is validated to 2.4e-14 macos-vs-macos). **Judge
collimated legs by intensity, not by raw complex-field correlation.** If you
need bit-faithful fields at a pupil, **consume our exported `E` directly** at
that plane — which is exactly what `'collapsed'` mode does, and why it is
always valid.

**Powered OAPs are the external user's own `prop_lens`.** A pupil fed *through*
a powered OAP (e.g. `ExitPupil` via OAP8) is **not** replayable by a bare
`prop_propagate` — the OAP's focal-length phase acts between planes and is the
consumer's optic to model (`prop_lens` with its focal length), not ours to
carry. The `optic`-kind stations sit mid-beam through a powered mirror and are
**not valid hand-off planes**; they are reported for completeness, never gated.

### The two use modes

- **`'collapsed'` (recommended for a hand-off).** Ignore the inter-optic legs;
  take our exported `E` at each pupil/focus as the starting field for your own
  downstream propagation. Always valid — no reference-phase ambiguity.
- **`'s2s'` (replicate our propagation).** Replay the diffraction legs: the
  through-focus/FarField legs from `spheres`, the direct NFPlane pupil→pupil
  legs with `prop_propagate`. Model the OAPs as your own `prop_lens`.

### Check-script gate numbers (pinned; MATLAB PROPER, N=1024, 500 nm)

| station kind | `'s2s'` | `'collapsed'` |
|---|---|---|
| **focus** (Focus23, FPM, FieldStop, FPA) | corr_I **1.000000** | corr_I **1.000000** |
| **pupil**, direct NFPlane (Apodizer, Lyot, CheckPoint) | corr_I ≥ **0.9998** | — |
| **pupil**, all (DM1/DM2/Apodizer/Lyot/CheckPoint/ExitPupil) | — | corr_I ≥ **0.961** |
| **optic** (OAPs) & OAP-fed pupils | reported, not gated | reported, not gated |

`proper_ctb_check_s2s.png` / `proper_ctb_check_collapsed.png` render the full
station-by-station bar charts (blue = focus, gated 1.0; green = **gated**
pupil, 0.94; grey = optic / OAP-fed pupil, informational — in `s2s` DM1 and
ExitPupil are OAP-fed and sit in the grey class). Both modes report **focus
PASS / pupil-gate PASS**.

## Hand-off package

A PROPER user receives **three files** and needs nothing else (no macos, no
mmacos, no deck):

| File | What it is | Run |
|---|---|---|
| `ctb_phase_export_N1024.mat` | the model: per-plane fields, feeding spheres, OAP focal lengths, coronagraph masks (`format_version` 2). Gitignored (≈312 MB); regenerate with `ctb_phase_export`. The committed `.fp.json` fingerprint + 96×-downsampled `_preview.mat` stand in for review. | — |
| `proper_ctb_run.m` | **end-to-end model** — reproduces our bare PSF and coronagraph contrast from the `.mat` alone, entirely in PROPER. | `proper_ctb_run('figure',true)` |
| `proper_ctb_check.m` | **per-plane interface check** — verifies the export plane by plane (`'s2s'` replays each leg; `'collapsed'` consumes our `E` as the hand-off). | `proper_ctb_check('s2s'); proper_ctb_check('collapsed')` |

`proper_ctb_run` is the **validation statement**: *a PROPER user, starting
from our data alone, reproduces the coronagraph.* `proper_ctb_check` is the
diagnostic that localises any disagreement to a plane. Both need only
`addpath ~/dev/proper_matlab`.

### What "end-to-end" means (and why it is not a single continuous beam)

A single continuous PROPER beam DM1→FPA does **not** reproduce macos (FPA
pitch ratio 0.71, corr 0.005). macos samples every intermediate focus on the
**system exit-pupil Fraunhofer pitch** (the large EP-sphere radii), not the
local geometric focus set by each OAP's focal length, so one PROPER grid
cannot carry both the pupil pitch and the focal pitch across the f–f relay.
`proper_ctb_run` is therefore a single pure-PROPER **script** that seeds from
our exported fields — not a single beam:

- **Bare PSF** — a *terminal replay*: seed at the exported ExitPupil pupil
  field and focus over its FarField-sphere radius R (`prop_lens(R) +
  prop_propagate(R)`). This is the arbiter recipe.
- **Coronagraph** — a self-contained PROPER **Fourier cascade** seeded at the
  exported Apodizer pupil field: apodizer, then `prop_lens/prop_propagate`
  through the FPM occulter, Lyot stop, and out to the FPA, using the exported
  `EFL_m` of OAP4/5/6 as the relay lenses. It runs in PROPER's own
  self-consistent sampling and is validated by the **dark-zone contrast** it
  produces, not by pixel-matching macos.

### Gate numbers (pinned; MATLAB PROPER, N=1024, 500 nm, v2 export)

| gate | measured | rule | result |
|---|---|---|---|
| **bare** FPA vs exported FPA | pitch ratio **1.0000**, corr_I **1.000000** | `\|ratio−1\|≤1e-3` & corr_I ≥ 0.9999 | **PASS** |
| **coronagraph** dark-zone mean contrast (3–15 λ/D) | **1.4e-8** | ≤ 2× shipped (5.8e-7) **and** ≥ shipped/50 (pathology floor) | **PASS** |
| **mid-chain** Lyot pupil | same-grid corr_I **0.93**, beam-dia ratio **4.3×** | reported, **not gated** | info |

The coronagraph gate is **one-sided-deep**: the idealised Fourier relay
seeded at the Apodizer carries the upstream aberration baked into that field
but omits the downstream OAP4→FPA real-optic figure that scatters extra light
in macos (the export cannot cleanly split per-OAP figure — see
`meta.screen_method`), so the PROPER cascade is legitimately **deeper** than
the shipped macos value (2.9e-7). The upper bound is the real gate; the lower
bound only catches a collapsed FPA. The **mid-chain Lyot is reported, not
gated**: the masks-off cascade forms the Lyot on PROPER's own sampling (its
beam is ~4.3× the exported Lyot diameter, on ~4× the pitch — the same
sampling reason a single beam is ruled out), so a raw correlation across that
scale gap is not a valid gate (README rule 2). The two gated statements are
end-of-chain, where PROPER is self-consistent.

`proper_ctb_run.png` shows the exported (macos) bare PSF beside the
PROPER-chain bare PSF (visually identical, corr 1.000000), the PROPER
coronagraph FPA, and the radial contrast profile with the dark-zone annulus
and the shipped macos level marked.

# DM layer — actuators, engine-measured Jacobian, EFC dark hole

The two flat DMs of the bench become *controllable* here: grid-data
surfaces driven by actuator commands, an EFC Jacobian measured by poking
every actuator through the full masked diffraction chain, and an
electric-field-conjugation loop closed on the engine.  Everything is
engine-in-the-loop — no Fourier model of the bench anywhere, so there is
no model gap for the optimizer to mine (the e2e6m S3b lesson).

## The pieces

| file | what |
|---|---|
| `ctb_dm_rx.m` | emits `ctb_dm.in` from `ctb_dcr.in`: the two DM blocks become `Surface= GridData` with an `nGridMat=256` channel whose frame is the element's own (`pData=VptElt, xData=xObs, zData=psiElt`) — the frame rule that makes pokes localize (the e5 "central dot" lesson). Hand decks untouched; `ctb_dm.in` + `flat256.txt` are derived (the flat grid is gitignored, rewritten on demand). |
| `ctb_dm.m` | influence-function DM model: 32×32 actuator lattice, pitch = beam/32 = 0.666 mm, Gaussian influence with 12 % nearest-neighbor coupling, commands (mm of surface) → 256×256 grid via local stamps. `apply(a)` REPLACES the element grid (`macos.set_elt_grid`) so there is no accumulation state. 880 active actuators per DM (centers within beam radius + 1 pitch). |
| `ctb_chain.m` | reusable masked-chain runner: loads the deck once, sizes apodizer/FPM/Lyot once (the deterministic geometry of `ctb_coro_compare`), then `run()` = fresh trace + masks multiplied in place + complex field at the FPA. 0.4 s per run at N=512. |
| `ctb_dm_jacobian.m` | G = dE(dark zone)/d(actuator): 1760 forward-difference pokes (h = 2 nm surface) through the masked chain, 11.3 min at N=512. Saved to `ctb_dm_jacobian_N512.mat` (37 MB, gitignored) + committed `.fp.json` fingerprint. Regen: `ctb_dm_jacobian()`. |
| `ctb_efc.m` | the EFC loop: Tikhonov least squares on the dark-zone field, α line-searched each iteration against the MEASURED contrast (runs are 0.4 s — a luxury lab EFC needs probing for), stop when no α improves. Sensing assumed perfect (engine field read directly); pairwise probing is the lab-facing extension. |

## The real-stacked solve (the trap this layer documents)

DM commands are REAL.  The complex least squares must be solved in the
stacked form `[Re G; Im G] da = −[Re e; Im e]`.  A complex SVD solve
returns complex `da` — which passes MATLAB's `double` validation and has
its imaginary part **silently dropped by the mex layer** — so the achieved
field decorrelates from the prediction (measured: corr 0.13, magnitude
ratio 0.17) and the line search collapses to ~3 %/iteration crawl.
`ctb_dm.apply` now rejects complex commands; the diagnosis battery is the
pattern to reuse: repeatability (bit-exact), superposition (1e-13),
column reproducibility (2e-8), predicted-vs-achieved correlation.

## Numbers (N=512, 500 nm, shipped mask config: r_fpm 2.70 λ/D, Lyot 0.50)

| quantity | value |
|---|---|
| static coronagraph dark zone (3–15 λ/D mean) | 2.934e-7 |
| EFC floor (19 iterations, fixed G) | **8.055e-9 (36×)** |
| inner band 3–8 λ/D | 9.62e-7 → 2.40e-8 (40×) |
| outer band 8–15 λ/D | 6.85e-8 → 2.69e-9 (25×) |
| DM strokes at the floor (rms) | 9.9 / 8.6 nm |
| DM1-only control (same G, same loop) | 1.30e-7 (2.3×, stalls) |
| DM2-only control | 2.55e-7 (1.1×) |
| linear-achievable floor (top-400 real modes, 11 nm rms) | 4.5e-9 |

The measured floor sits within 2× of the linear-achievable value for a
Jacobian measured ONCE at the flat state — the remaining gap is
regularization + fixed-G error, so relinearization (re-measuring G around
the dug state) is the next depth increment, not a bug hunt.

Both DMs are load-bearing: restricted to either mirror alone the loop
stalls two decades short (DM1 at the stop is phase-only and cannot reach
the symmetric half of the speckle field; DM2, 500 mm out of pupil, is
the amplitude lever but weak alone).  The annular dark zone is a
two-DM product — one DM buys only a one-sided zone.

Physics worth knowing: with Lyot 0.50 the FPA λ/D is a *post-Lyot* unit —
a DM ripple of k cycles across the (full) beam lands at k/2 post-Lyot
λ/D, so 32 actuators control cleanly to ~8 λ/D and weakly beyond.  That
is why the DM1 command map concentrates stroke in a ring at the
Lyot-edge image and why the outer band digs 25× where the inner digs 40×.

Gotcha inherited from gate work: pupil OPD must be read at the EXIT PUPIL
(`macos.trace(30)`); a bare `macos.trace()` traces to the FPA where a DM
bump smears into a global low-order term ~10× its sag — it looks like a
grid-amplitude engine bug and is not (`tCtbDm` pins the 2·cos(AOI) scale).

Gates: `tests/tCtbDm.m` (SUITE_CTB_512) — emitter frame audit, grid
readback, sag→OPD scale/sign/location, speckle-pair symmetry, chain
contrast pin, Jacobian column linearity, and an EFC smoke that must dig
≥2× in 3 iterations.

# The physics layers — bandwidth, polarization, the vector vortex

On the vortex/0.60-Lyot chain (the slide-10 configuration), in order of
addition (drivers: `ctb_efc_physics`, `ctb_vortex_bandwidth`,
`ctb_vvc`; full record CTB_PROP_STATUS SESSIONS 13–14):

| configuration (N=512, 3–15 λ0/D, charge 4, Lyot 0.60) | closed-loop floor |
|---|---|
| scalar vortex, mono | 8.3e-13 (6.8e-15 relinearized) |
| scalar, 5% / 10% / 20% band (3/5/9 colors at 2.5% spacing) | 9.4e-12 / 2.5e-11 / 5.4e-11 |
| polarization floor (coated-train Jones screens, any band) | 1.1e-15 (uncontrollable part) |
| Lyot rebalance under 10%: 0.70 / 0.80 | 6.5e-11 @ 49% / 1.2e-10 @ 64% thru |
| VECTOR vortex, ideal plate, unpolarized | 2.0e-9 (the two-spiral compromise) |
| zero-order plate, unpolarized, 5/10/20% | 7.8e-8 / 2.1e-7 / 6.4e-7 (leakage-pinned) |
| circular sandwich (R in, L analyzed), mono → 20% | 8.3e-13 → 1.5e-10 |
| circular + per-λ stacked control, 5% | **7.9e-12** (= the scalar floor) |
| crossed-linear sandwich, mono / 5 / 10 / 20% | **6.4e-16** / 6.1e-12 / 1.7e-12 / 7.0e-13 |
| crossed-linear full stack (10% + coating screens) | 3.4e-11 (cross-pol re-leak) |

The vector-vortex findings, compressed: the zero-order plate's
retardance leak is UNCORRECTABLE by the DMs (its amplitude flips sign
across band center) but OPTICALLY REMOVABLE by a polarizer/analyzer
sandwich.  The crossed-linear sandwich is the star-side winner (its
analyzed channel is the sin mθ mask chain — an 8-octant-mask analogue)
but carries 2m azimuthal planet nulls (8 blind spots at charge 4, mean
throughput ¼) where the circular sandwich is flat ½.  Verdict figure:
`ctb_vvc_summary.png` (regen `ctb_vvc_summary` after the `ctb_vvc`
tag ladder — run states gitignored, regen lines in the status file).

# Reconstructing the study — ctb_study

The whole slides-9–13 sequence (Jacobian → EFC → relinearization →
physics layers → bandwidth sweep → vector-vortex ladder → verdict
figure) is one config-driven driver:

    >> ctb_study('dry', true);                     % the 25-step plan, no engine
    >> out = ctb_study();                          % audit/resume the shipped study
    >> out = ctb_study('charge', 6, 'r_lyot_frac', 0.70);   % a new parameter point

Every tag, cache file, and figure derives from the config: the shipped
configuration maps onto the historical file names, any other gets an
automatic suffix (`_vc6L070`), so parameter points never collide.
Stages whose run states exist are skipped and their numbers folded into
the returned manifest — a default run over complete states costs
seconds and IS the audit that the deck numbers are reconstructible.
The cache guard: Jacobian files are keyed by name, which cannot encode
the mask geometry, so every cache carries a `chain_opts` stamp and
`ctb_jac_check` refuses a mismatched load loudly (a stale G otherwise
fights the loop with plausible-looking numbers).  `ctb_vvc_summary`
takes a `'suffix'`; `ctb_vortex_bandwidth` takes `'chain'`/`'tag'`;
`ctb_vvc` takes `'r_lyot_frac'` — all default to the shipped study.

# The progress deck — deck_ctb

`deck_ctb.pptx` (23 slides: title + 15 main + 6 backup behind a divider) records
the state of the CTB model: bench + prescriptions, PROPER validation,
mask families head-to-head, planet/bandpass, phase-factor export, the
pure-PROPER hand-off, the DM/EFC dark hole, the vortex/Lyot trade, the
physics layers (polarization, bandwidth), and the vector vortex.
Source is `deck_ctb.md`; regen with `python3 make_brief_slides.py
deck_ctb.md` (figures are the committed PNGs in this directory).
Style: `doc/DECK_STYLE.md` + `doc/STYLE_REPORTS.md` §5 (gate run
2026-08-26, clean).
