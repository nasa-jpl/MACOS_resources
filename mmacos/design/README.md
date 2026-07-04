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
| **2. Three-mirror** | [`tma_onaxis/`](examples/tma_onaxis) (obscured Korsch, bias sweep), [`tma_offaxis/`](examples/tma_offaxis) (unobscured conic TMA), [`tma_freeform/`](examples/tma_freeform) (freeform M2+M3), [`sz_tma/`](examples/sz_tma) (all-sphere + Zernike, real intermediate focus) | The third mirror nulls field astigmatism; intermediate focus; convex-secondary convention; sphere+Zernike decoupling |
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
| [`examples/tma_onaxis/`](examples/tma_onaxis) | **On-axis obscured Korsch designer** (j18mono form): concave M1, geometrically convex M2, concave M3 behind M1, real intermediate focus between M1 and M2; sweeps the off-axis field bias and recommends the least bias that clears the FP while staying diffraction-limited. |
| [`examples/sz_tma/`](examples/sz_tma) | **Sphere + Zernike unobscured three-mirror** (e5mono-derived). 8 m, f/21: three base **spheres** (Kc=0) + Zernike departures, folded unobscured, with a **real intermediate focus between M2 and M3** (metrology-beam injection). The convex secondary forms the real intermediate image; M3 reimages it to a focal plane out of the beam. Diffraction-limited on-axis (0.044 waves) from a 35 700-wave all-sphere start; the central 2-D field is well-corrected, walling at the ±2′ corners (the narrow-field e5mono geometry + 15 modes — see roadmap). Showcases `set_base_sphere`, a `convex` secondary, and `optimize_freeform` staged center → field. |
| [`examples/tma_offaxis/`](examples/tma_offaxis) | **Unobscured off-axis convex-secondary CONIC TMA** (j18mono geometry, 6.6 m, f/20, 2.3 µm). Exercises the `seidel_seed` **convex-secondary fix**: the n-flip `|radii|` model cannot represent a convex secondary that forms a real intermediate image (mis-derives f/20 → f/0.9; brute-forcing all radius signs recovers neither focus nor conics), so the `convex` flag returns the correct *unfolded* paraxial focus + a **K=0** sphere seed and `optimize('engine','native')` over the three conics nulls 3rd-order spherical + coma + astigmatism (53 590 → 0.0015 waves on-axis; conics → j18mono.in's [−0.995, −1.634, −0.854]). Then `set_offaxis('all')` extracts an eccentric-pupil section (0.66·D decenter), refigured + field-balanced over a ±2′ AREA to **0.050 waves worst, fully unobscured (4/4 clear)**. The conic counterpart to `sz_tma`. |
| [`examples/tma_freeform/`](examples/tma_freeform) | **Off-axis unobscured three-mirror freeform**: Korsch seed → eccentric section → 3-conic limit → Zernike departures on M2+M3 (CALIB OptZern, radii/conics held) break the conic field wall. |
| [`examples/tma_3plus1/`](examples/tma_3plus1) | **3+1 reference design (in progress)**: unobscured conic TMA + M4 field mirror just past the science focus — relays the exit pupil to an accessible plane and flattens it (nulls the pupil defocus/astig `macos.pupil_quality` measures on the bare TMA). The Sprint 2D segmentation parent and coronagraph front end. |

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
  multiple instruments); the instrument-building sequence (spectrograph,
  coronagraph, metrology).
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
