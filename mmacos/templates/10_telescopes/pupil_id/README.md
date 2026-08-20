# `pupil_id/` — beyond FEX: the exit pupil as a SURFACE

[`tma_onaxis`](../tma_onaxis)'s own note is that *"the exit pupil after M3 is
**ASSESSED (FEX), not constrained**."* This template is the next chapter. FEX
reduces the exit pupil to a **single chief-ray conjugate sphere** — one field
point, forced `Kc=0`, one radius. It gives the XP location and the far-field
propagation distance and **nothing about pupil-imaging quality**: no pupil
spherical aberration, no pupil astigmatism, no pupil **walk** across field.

This driver takes `tma_onaxis` as a representative telescope (M1 **is** the
stop, so the entrance pupil is the M1 beam rim), runs the grid-of-cone-sources
method to find the exit pupil as a fitted **surface**, cross-checks it against
the engine's two-ray XPS, reports how sharply the EP images to the XP, and
tracks the pupil walk vs field.

It **composes existing, test-gated tools** — [`design/src/pupil_map.m`](../../../design/src/pupil_map.m)
(cone convergence), `macos.pupil_quality` (engine XPS), `macos.pupil_zone_map`
(zone spots), `macos.fex`/`macos.set_xp` (the XP sphere). **No new engine or
veneer code.**

## The two XP radii — do not conflate

The prototype surfaced two distinct radii; the example keeps them separate:

- **FEX's `Kr` = the XP→detector far-field radius** (0.147 m here = the
  chief-ray distance from the XP vertex to the next plane). This is the
  **propagation sphere** the diffraction code needs — it is correct, and it is
  what stays in the Rx.
- **The cone-crossing sag-fit curvature** (0.077 m) is the curvature of the
  pupil-imaging **convergence surface** — a pupil-**quality** diagnostic, a
  different quantity. It is **not** the propagation radius and is **not**
  written to the Rx.

So the XP sphere written back with `macos.set_xp` uses the **bundle vertex**
(an improvement on the one-chief-ray vertex) with **FEX's propagation radius**;
the convergence curvature and the departure-from-sphere are reported as the
beyond-FEX pupil quality.

## Three entrance-pupil conventions, one deck

`pupil_map`'s `anchor` names three physically different pupil planes; the
example shows all three:

| `anchor` | the EP is… | when |
|---|---|---|
| `'rim'` | the beam **rim** (M1 rim here) | classic telescope; **the primary view here** |
| `'stop'` | a **point** through `ApStop` | when another element (M2, a DM, a pupil stop) defines the EP; also **exactly what the engine XPS computes**, so it is the cross-check anchor |
| `'surface'` | on the curved entrance **element surface** | pupil imagers that must resolve the EP surface — a segmented primary, or **DM1 in a coronagraph** |

The M1 rim is drawn as an annotation, not inserted as an element: M1 is already
the stop, and `pupil_map` (`strip_ap`) plus the design layer both strip
Return/Reference apertures, so a physical rim stop could not clip a metric ray
anyway.

## Pipeline (`pupil_id.m`)

1. **FEX baseline** — `macos.stop(M1)`, `macos.fex(1)`: the single-chief-ray XP
   sphere we go beyond.
2. **EP = M1 rim** — realized by `anchor='rim'`, no element inserted.
3. **Cone-convergence XP surface** — `pupil_map(deck, field_grid, 'anchor','rim')`
   (and `'surface'`): the four-part ladder (blur / surface Zernikes / map /
   wander).
4. **XP sphere → Rx + pupil-quality departure** — best-fit sphere vertex from
   the crossing cloud (piston+tilt+curvature removed → the residual is the true
   departure), written with FEX's propagation radius via `macos.set_xp`.
5. **Engine two-ray cross-check** — `macos.pupil_quality` (XPS); `pupil_map`
   `anchor='stop'` reproduces it (the headline agreement).
6. **EP→XP imaging sharpness** — `macos.pupil_zone_map(M1, XP)` zone spots.
7. **Pupil walk vs field** — `macos.fex` at a field sweep: XP position and
   radius vs field, the across-field piece FEX gives at only one point.

## Knobs

- `FOV_ARCMIN`, `NGRID` (≥3) — the cone field grid.
- `NODES` — the entrance-surface node lattice.
- `WALK_ARCMIN` — the field sweep for the walk.
- Aperture/conics are **fixed by the deck** (`../tma_onaxis/tma_onaxis.in`) —
  do not edit them here.

## Results (tma_onaxis, D=1 m, f/1.5 primary, 1′ bias, model 256, 500 nm)

| quantity | value |
|---|---|
| FEX baseline: XP z / propagation radius | 0.26358 m / −0.14693 m |
| cone-convergence blur (per-node waist RMS) | 7.65e-6 m |
| convergence-surface curvature (quality, not propagation) | 0.0765 m |
| **departure-from-sphere RMS** (piston+tilt+curvature removed) | **184 nm** |
| pupil aberration (rim anchor): defocus / astig / spherical | 2.66e-4 / −2.2e-8 / 8.6e-8 |
| **XPS cross-check** (pupil_map `stop` vs pupil_quality): defocus / astig | **0.02% / 0.00%** |
| EP→XP imaging: median / worst zone spot (25 zones) | 1.74e-3 / 3.00e-3 m |

## Cross-check statement

`pupil_map` at `anchor='stop'` (tangential field triple, rescaled by
`(Rq/norm_radius)²`) reproduces the engine XPS `defocus`/`astig` to **~0.02%**;
the finite-field XP vertex coincides with FEX; `anchor.blur_ratio = 0.010`
(anchoring error negligible against the blur). This reproduces the
`tests/tPupilMap.m` two-ray-limit identity live.

## ⚠ Caveats — name the anchor on every number

- `'rim'`, `'stop'`, and `'surface'` are **different pupil planes**; a defocus
  or radius is only meaningful with its anchor named.
- **Frame vs aberration:** piston and tilt are removable reference-frame terms
  (the exit axis is tipped ~1′ by the field bias). They are **excluded** from
  the pupil-aberration comparison and absorbed by the best-fit sphere —
  otherwise a 15 µm tilt gradient masquerades as 15 µm of "departure" (the true
  departure is 184 nm).
- The convergence-surface curvature (0.077 m) is **not** the propagation radius
  (0.147 m); only the latter goes in the Rx.

## Run

```matlab
run('.../mmacos/templates/10_telescopes/pupil_id/pupil_id.m')
% headless:  matlab -batch "run('.../pupil_id.m'); exit(0)"
```

Requires `MACOS_HOME` set and the mmacos MEX built. Writes
`pupil_id_results.mat` and `pupil_id_{cloud,zernikes,walk}.png` to this dir.
One MATLAB process, model size 256 (do not transition sizes in a live process).
