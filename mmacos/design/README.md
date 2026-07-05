# `mmacos/design/` — adaptable telescope & instrument design studies

The product of the design layer is **utilities and examples that users
adapt for their own design studies**.  This tree is organized around
that (Dave, 2026-07-04):

```
design/
  src/        shared design UTILITIES the examples call (report helpers,
              field-scan harnesses, pupil-relay solvers, ...)
  examples/   adaptable EXAMPLES -- runnable design studies you copy,
              re-knob, and re-run
```

(The `macos.design.*` classes — `Telescope`, `System`, `seidel_seed`,
`field_grid` — are the underlying API and live with the package in
[`../src/+macos/+design/`](../src/+macos/+design).  `design/src/` holds
the script-level utilities *between* that API and the examples.)

Each example is a runnable script you open, set a handful of design
knobs at the top, and run to produce a *complete, optimized* MACOS
design — a `.in` prescription + a `.mat` spec + report figures.  They
are distinct from [`../examples/`](../examples) (fixed feature demos):
a design example is meant to be **tuned** — change the aperture /
f-numbers / off-axis distance / field / Zernike modes and re-run to
explore the trade.

## The telescope progression

The examples build up in deliberate order — each stage adds one
architectural idea; adapt the stage closest to your problem:

| Stage | Examples | What it adds |
|---|---|---|
| **1. Two-mirror** | [`rc_onaxis/`](examples/rc_onaxis) (obscured aplanat), [`rc_unobscured/`](examples/rc_unobscured) (eccentric-pupil section), [`wf2_freeform/`](examples/wf2_freeform) (freeform on M2) | Conic seeding, the RC astigmatism field limit, the off-axis-section recipe, what one freeform surface buys |
| **2. Three-mirror** | [`tma_centered/`](examples/tma_centered) (**obscured family** + the obscured-vs-unobscured A/B), [`tma_onaxis/`](examples/tma_onaxis) (obscured Korsch, bias sweep), [`tma_offaxis/`](examples/tma_offaxis) (**unobscured family**, conic), [`tma_freeform/`](examples/tma_freeform) (freeform M2+M3), [`sz_tma/`](examples/sz_tma) (all-sphere + Zernike, real intermediate focus) | The third mirror nulls field astigmatism; the two BASIC families — centered/obscured (symmetry intact, wide field, JWST-like) vs unobscured section (clear pupil bought with induced binodal field astigmatism); intermediate focus; convex-secondary convention; sphere+Zernike decoupling |
| **3. Three + one** | [`tma_3plus1/`](examples/tma_3plus1) *(reference design, in progress)* | An M4 **field mirror** just past the science focus relays the exit pupil to an accessible plane and **flattens** it — the coronagraph-ready front end (flat pupil conjugate for DM/apodizer/Lyot, intermediate focus for an FPM) |
| **4. N-mirror** | *(planned)* | Generalization: instrument relays, shared wide fields (HWO pushes toward 10×20 arcmin multi-instrument focal planes) |

After the telescope progression comes the **instrument-building
sequence** (planned): spectrograph (Sprint 2C), coronagraph
(Sprint 3), metrology — each as adaptable examples over the same
utilities.

## Two design strategies

The design layer reaches a corrected design two complementary ways:

1. **Conic** (classical). `macos.design.Telescope` builds the first-order
   layout and **Seidel-seeds the conics**; `optimize` refines the conics
   multi-field via MACOS's native CALIB optimizer. Used by the RC examples.
   *Limit:* conics are rotationally symmetric, so a folded / wide field
   eventually walls on field astigmatism — which is why wide fields go to three
   mirrors and/or freeform.  (Measured on the 6.6 m f/20 unobscured TMA at
   λ=1 µm: the 3-conic system holds ~0.066λ at a 3′ full field and walls
   field-linearly — 0.17λ at 5′; freeform on M2+M3 is the escalation.)

2. **Sphere + Zernike** (freeform). Hold every base surface as a **sphere**
   (`set_base_sphere`, `Kc=0`); put ALL aberration correction in **Zernike
   departures** on each mirror (`set_freeform` + `optimize_freeform`, the CALIB
   OptZern channel). Geometry (sphere radii + fold tilts) and correction
   (Zernikes) **fully decouple** — once the layout meets packaging the optimizer
   can't break it, and the Zernikes reach the field-dependent aberrations conics
   cannot. This is the e5mono model; used by `sz_tma`.

## The pattern

Every example follows the same shape:

1. **User knobs** at the top (aperture, f-numbers, off-axis distance, field,
   modes, …).
2. **Build** via `macos.design.Telescope` (+ `add_mirror` / `add_focal_plane`
   for N-mirror).
3. **Restructure** as the design needs (off-axis eccentric section, decenter,
   fold tilts, base spheres).
4. **Optimize** to diffraction-limited over the field — `optimize` (conics) or
   `optimize_freeform` (Zernike departures, staged center → field).
5. **Verify** — `check_clipping` for an unobscured design (and focal plane out
   of the beam).
6. **Save** the deliverable (`save` → `.in`, `save_spec` → `.mat`) + report
   figures (`view_field_map`, `view_orthoviews`) and place the exit pupil
   (`add_pupil`).

Run from MATLAB with the package on the path:

```matlab
addpath('~/dev/MACOS_resources/mmacos/src');
run('~/dev/MACOS_resources/mmacos/design/examples/tma_offaxis/tma_offaxis.m');
```

Generated artifacts (`.in` / `.mat` / `.png`) are regenerated by re-running.

## Example details

| Folder | Design |
|---|---|
| [`examples/rc_onaxis/`](examples/rc_onaxis) | Parameterized **on-axis Ritchey-Chrétien** — the classical *obscured* Cassegrain form. Aplanatic → diffraction-limited on-axis; reports first-order properties (EFL, plate scale, central obscuration) and the WFE-vs-field **astigmatism limit** two conics cannot beat (the reason wide fields go to a TMA). The obscured baseline for the unobscured example below. |
| [`examples/rc_unobscured/`](examples/rc_unobscured) | Parameterized **unobscured off-axis Ritchey-Chrétien** (eccentric-pupil section). The off-axis distance trades against secondary size — a faster primary and/or slower system f/# shrinks the secondary and lets you go less off-axis (≈0.89·D → 0.64·D, floor ≈0.5·D). Focus behind M1. |
| [`examples/wf2_freeform/`](examples/wf2_freeform) | **Wide-field 2-mirror freeform**: what a Zernike departure on the secondary buys, holding the layout fixed — per-field it NULLS the residual astigmatism (~25×, diffraction-limited); over a symmetric wide field a single off-pupil surface has field-LINEAR control (honest limit shown). |
| [`examples/tma_centered/`](examples/tma_centered) | **Centered (obscured) TMA — the j18 / early-JWST family — and the obscured-vs-unobscured A/B.** Runs BOTH basic families through the shared `tma_conic_recipe` on the same j18mono parent (6.6 m f/20) over a 5′-diameter **circular** field (`field_ring`) at 1 µm: centered holds **0.065λ worst (diffraction-limited, azimuth-uniform ring)** while the eccentric-pupil section pays **0.098λ**, the excess entirely field-dependent (binodal) astigmatism induced by breaking rotational symmetry (`wfe_field_diag` ladder shows it collapse to 0.02λ when astig is removed). The centered price: central obscuration (2/4 optics clear). j18's remaining edge is its 2.3 µm yardstick. |
| [`examples/tma_onaxis/`](examples/tma_onaxis) | **On-axis obscured Korsch designer** (j18mono form): concave M1, geometrically convex M2, concave M3 behind M1, real intermediate focus between M1 and M2; sweeps the off-axis field bias and recommends the least bias that clears the FP while staying diffraction-limited. |
| [`examples/sz_tma/`](examples/sz_tma) | **Sphere + Zernike unobscured three-mirror** (e5mono-derived). 8 m, f/21: three base **spheres** (Kc=0) + Zernike departures, folded unobscured, with a **real intermediate focus between M2 and M3** (metrology-beam injection). The convex secondary forms the real intermediate image; M3 reimages it to a focal plane out of the beam. Diffraction-limited on-axis (0.044 waves) from a 35 700-wave all-sphere start; the central 2-D field is well-corrected, walling at the ±2′ corners (the narrow-field e5mono geometry + 15 modes — see roadmap). Showcases `set_base_sphere`, a `convex` secondary, and `optimize_freeform` staged center → field. |
| [`examples/tma_offaxis/`](examples/tma_offaxis) | **Unobscured off-axis convex-secondary CONIC TMA** (j18mono geometry, 6.6 m, f/20, 2.3 µm). Exercises the `seidel_seed` **convex-secondary fix**: the n-flip `|radii|` model cannot represent a convex secondary that forms a real intermediate image (mis-derives f/20 → f/0.9; brute-forcing all radius signs recovers neither focus nor conics), so the `convex` flag returns the correct *unfolded* paraxial focus + a **K=0** sphere seed and `optimize('engine','native')` over the three conics nulls 3rd-order spherical + coma + astigmatism (53 590 → 0.0015 waves on-axis; conics → j18mono.in's [−0.995, −1.634, −0.854]). Then `set_offaxis('all')` extracts an eccentric-pupil section (0.66·D decenter), refigured + field-balanced over a ±2′ AREA to **0.050 waves worst, fully unobscured (4/4 clear)**. The conic counterpart to `sz_tma`. |
| [`examples/tma_freeform/`](examples/tma_freeform) | **Off-axis unobscured three-mirror freeform**: Korsch seed → eccentric section → 3-conic limit → Zernike departures on M2+M3 (CALIB OptZern, radii/conics held) break the conic field wall. |
| [`examples/tma_3plus1/`](examples/tma_3plus1) | **3+1 coronagraph front end** — three files: `tma_3plus1.m` (the j18-geometry DEMO: unobscured conic TMA + M4 field mirror past the science focus relays a ~30″ coronagraph channel to FP2 and flattens the exit pupil ~10×; wide field stays at the TMA focus — a relay cannot carry ±2.5′, the image walks ±96 mm across M4); `tma_3plus1_aoi_search.m` (CONSTRAINT FINDER: steps the PM–SM separation / primary f/# via `tma_layout` until every mirror's **AOI spread across the beam < 15°** — the coronagraph polarization preference; the j18 f/1.2 parent puts ~21–24° on M1/M2 from beam convergence alone, spread ≈ D/R₁); `tma_3plus1_optimize.m` (full staged optimization at the found geometry → `tma_3plus1_polsafe.in`). Build mechanics that matter: carry optimized conics via `add_mirror(...,'conic',K)` (seidel can't seed a relay-past-focus chain), `DPAST > f₄` for a real conjugate, `optimize(...,'elts',...)` to split image vs pupil DOFs, body-scaled clearance margin for a compact (0.84 D) cut. The Sprint 2D segmentation parent and Sprint 3 coronagraph front end. |

## Utilities (`design/src/` + package helpers)

Script-level utilities shared by the examples (add
`design/src` to the path; examples do it themselves):

- **`tma_conic_recipe(...)`** — the staged conic-TMA recipe for BOTH
  basic families: build → on-axis conic optimize → [eccentric section +
  axial refigure]* → multi-field balance (*`'section',true` only).
  Returns the optimized `Telescope` + per-field results.  Defaults to
  the j18mono-geometry 6.6 m f/20.
- **`wfe_field_diag(t, F)`** — per-field aberration LADDER (raw /
  −tilt / −focus / −astig + Z4/Z5-6 coefficients).  Reading the ladder
  identifies the field wall: focus-dominated → field curvature (fix the
  focal surface / relay conjugates); astig-dominated with field-varying
  orientation → binodal astig (a fixed mirror Zernike cannot fix it —
  smaller field, 4th powered mirror, or field-conjugate freeform).
- **`fold_station_report(t,'mirror',M)`** — where can a fold live?  For
  the legs into/out of a mirror, per-station lateral intervals of the
  two bundles + the daylight GAP between them (from the YZ DRAW fan).
  A fold picking off one leg needs gap > its mount margin or it clips
  the other — the quantitative form of the centered-Korsch focal-plane
  extraction (run on the UNFOLDED biased design, then `add_fold`).

Package-side helpers (in `+macos/+design`, used everywhere):

- **`field_ring(r,'units','arcmin')`** — the CIRCULAR-field set (ring +
  inner samples).  A square `field_grid` of half-field h puts its
  corners at h·√2 — 41 % more field than a circular spec asks — and
  those corners then dominate a balance.  Score round fields on rings.
- **`Telescope.trace_at_field([thx thy])`** — re-emit + trace the
  design at one field offset (the sanctioned per-field inspection;
  `macos.set_src_fov` does NOT move the field of an emitted design).
  `trace_at_field([])` restores the nominal field.

## N-mirror builder capabilities (used by `sz_tma`)

- `add_mirror(NAME,'radius_m',R,'spacing_after_m',T,'tilt_deg',θ,'convex',tf)` —
  append a mirror. `tilt_deg` folds the chief ray about x (Bauer unobscuring —
  tilt minimally to clear, don't decenter). `convex` marks a secondary whose
  centre of curvature is *downstream* (emits psiElt → downstream CoC); for a
  convex secondary `seidel_seed` returns the correct *unfolded* paraxial focus
  plus a **K=0** sphere seed — the n-flip `|radii|` conic seed is unreliable for
  a convex-secondary reimager, so the conics come from `optimize` (see
  `tma_offaxis`).
- `set_base_sphere(true)` — hold all base surfaces as spheres (Kc=0); correction
  is then entirely Zernike.
- `set_freeform(ELT,MODES,COEF,'type','BornWolf'|'ANSI'|…)` — layer a Zernike
  departure on a mirror (emits `Surface=Zernike`).
- `optimize_freeform(ELTS,'modes',M,'fields',F,'weights',W,'max_iters',N)` —
  optimize the Zernike coefficients multi-field (supply 2-D `fields` + area
  `weights`), holding all radii/conics. CALIB OptZern under the hood.
- `add_pupil` — place the exit-pupil reference surface (deliverable + future
  pupil-referenced optimization).
- `add_fold(NAME,'after',M,'dist_m',d,'to',DIR)` — insert a FLAT fold mirror
  d metres after element M; everything downstream is mapped by the fold-plane
  reflection isometry (exactly WFE-neutral, verified to machine precision).
  Folds emit `ApType=None` (the `ap_r` is a check_clipping BODY, not a stop)
  and are excluded from `optimize`'s DOF set. Weak POWER on the fold is a
  planned option (Dave 2026-07-05). See `tma_centered_fold_search`.
- `set_hole(NAME,r)` — declare a perforated element (the centered family's
  primary): through-the-hole crossings stop counting as body-in-beam
  obstructions in `check_clipping` (clearance only; the hole is not yet
  emitted as an inner obscuration).
- `center_focal_plane()` — move the detector BODY to the traced image
  centroid (trace-neutral); use after `set_field_bias`/`add_fold`, where the
  image walks off the derived on-axis FP center.
- `add_focal_plane(NAME,'ap_r',r)` — honest detector body size for the
  clearance judge (default 0.3·D is a generous placeholder).

## Status & roadmap

- **Working:** `rc_onaxis`, `rc_unobscured` (conic); `sz_tma` (sphere+Zernike —
  on-axis + central field diffraction-limited); `tma_offaxis` (convex-secondary
  conic TMA — on-axis + off-axis ±2′ area diffraction-limited, unobscured);
  `wf2_freeform`, `tma_freeform`, `tma_onaxis`.
- **In progress:** `tma_3plus1` — THE reference design (3+1 unobscured,
  conics first + freeform where needed, HWO-ish 6.6 m f/20, sweet spot in
  the ~5 arcmin full field at λ=1 µm).  Downstream consumers: Sprint 2D
  segmentation (SegMirMaker on M1) and coronagraph integration.
- **Later:** the N-mirror stage + the HWO shared wide field (10×20 arcmin,
  multiple instruments — the field-quadratic astigmatism at that scale wants
  a 4th powered imaging mirror, not more freeform on three); the
  instrument-building sequence (spectrograph, coronagraph, metrology).
- **Refit note (Dave 2026-07-04):** migrate the 2-mirror examples
  (`rc_onaxis`, `rc_unobscured`, `wf2_freeform`) onto the utilities
  structure (`tma_conic_recipe`-style shared recipe + `wfe_field_diag` /
  `aoi_report` reporting) when this is further along.
- **Planned optimizer:** `optimize_freeform('engine','jacobian')` — a
  pupil-referenced dW/dZern *linear* solve (the wavefront is ~linear in the
  Zernike coefficients at the pupil, vs. the focal-plane defocus nonlinearity);
  faster + more robust than CALIB's FD, and it reuses the migrated GMI
  sensitivity `dw_dz_zernike`. See [`../doc/GMI_migration_census.md`](../doc/GMI_migration_census.md).

> **Convention note:** the emitter writes `KrElt = -|R|` for every mirror and
> `KcElt = K`; convex vs. concave is the **geometry** (vertex placement + the
> `convex` psi-flip), never the radius sign. Folded multi-mirror designs use
> per-element `psiElt`/`VptElt` (the freeform exception to the coaxial
> all-`(0,0,-1)` rule).
