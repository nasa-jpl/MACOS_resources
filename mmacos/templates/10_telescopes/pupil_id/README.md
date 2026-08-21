# `pupil_id/` — beyond FEX: the exit pupil as a SURFACE

[`tma_onaxis`](../tma_onaxis)'s own note is that *"the exit pupil after M3 is
**ASSESSED (FEX), not constrained**."* This template is the next chapter. FEX
reduces the exit pupil to a **single chief-ray conjugate sphere** — one field
point, forced `Kc=0`, one radius. It gives the XP location and the far-field
propagation distance and **nothing about pupil-imaging quality**: no pupil
spherical aberration, no pupil astigmatism, no pupil **walk** across field.

It is a **general-purpose driver**: give it any telescope Rx (defaults to
`tma_onaxis`), it runs the grid-of-cone-sources method to find the exit pupil
as a fitted **surface**, writes a **revised Rx** (the XP reference sphere
re-placed at the cone-bundle vertex, an improvement on the one-chief-ray FEX
sphere), cross-checks against the engine's two-ray XPS, reports how sharply the
EP images to the XP, and tracks the pupil walk vs field.

It **composes existing, test-gated tools** — [`design/src/pupil_map.m`](../../../design/src/pupil_map.m)
(cone convergence), `macos.pupil_quality` (engine XPS), `macos.pupil_zone_map`
(zone spots), `macos.fex`/`macos.set_xp` (the XP sphere). **No new engine or
veneer code.**

## Two layers — a finder and a wrapper

| Layer | File | Role |
|---|---|---|
| **core finder** | [`design/src/pupil_find.m`](../../../design/src/pupil_find.m) | Rx in → cone convergence → **`set_xp` into the internal Rx in place, exactly as the engine's FEX modifies engine state** → metrics out. No file I/O, no figures, no printing. |
| **template wrapper** | `pupil_id.m` (here) | calls `pupil_find`, then does the report, the XPS cross-check, the zone + walk metrics, the figures, and writes the **revised Rx** to disk (`macos.save_rx`). |

The split is so **sensitivity drivers (`run_dwd*.m`) call `pupil_find`
directly** — it leaves the engine with the improved XP in the loaded Rx, just
like a FEX call, and returns the metrics — while the wrapper owns everything
file- and figure-facing. `pupil_find(..., 'place',false)` measures without
touching the Rx.

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

## Pipeline

`pupil_find` (steps 1–4, the core) → `pupil_id` adds 5–8:

1. **FEX baseline** — `macos.stop(ep_elt)`, `macos.fex(1)`: the single-chief-ray
   XP sphere we go beyond.
2. **EP anchor** — `'rim'` (beam rim, default), `'stop'`, or `'surface'`; no
   element inserted (the anchor is a `pupil_map` argument).
3. **Cone-convergence XP surface** — `pupil_map` over a `field_grid`: the
   four-part ladder (blur / surface Zernikes / map / wander).
4. **XP sphere → internal Rx** — best-fit sphere vertex from the crossing cloud
   (piston+tilt+curvature removed → the residual is the true departure),
   `macos.set_xp` with **FEX's propagation radius** — modifies the loaded Rx in
   place, as FEX does.
5. **Engine two-ray cross-check** — `macos.pupil_quality` (XPS); `pupil_map`
   `anchor='stop'` reproduces it (the headline agreement).
6. **EP→XP imaging sharpness** — `macos.pupil_zone_map(ep,xp)` zone spots
   (skipped with a message on a deck whose pupil is not fully lit).
7. **Pupil walk vs field** — `macos.fex` at a field sweep: XP position and
   radius vs field, the across-field piece FEX gives at only one point.
8. **Revised Rx → disk** — `macos.save_rx(out_rx)` (default `<rx>_xp.in`).

## Run

```matlab
run_pupil_id                                % the example runner: both bundled cases
run_pupil_id('tma_onaxis')                  %   one case (or 'sz_tma' / 'both')
pupil_id                                    % the driver, template default: tma_onaxis
out = pupil_id('/path/to/my_tel.in');       % your telescope -> my_tel_xp.in
out = pupil_id(rx, 'ep_elt',1, 'xp_elt',5, 'fov_arcmin',3, 'anchor','rim');
% headless:  matlab -batch "run_pupil_id('both'); exit(0)"
% sensitivity drivers call the core finder directly:
pf = pupil_find(rx, field_grid);            % modifies the loaded Rx like FEX
```

`run_pupil_id.m` drives both bundled test telescopes — `tma_onaxis`
(fully-lit pupil) and `sz_tma` (tilted-Zernike M1, so the zone map
auto-skips) — and prints a consolidated table across cases.

## Knobs (name-value)

- `ep_elt` / `xp_elt` — entrance-pupil (stop) / exit-pupil element; `xp_elt`
  defaults to `nElt-1` and must be a Return/Reference surface.
- `fov_arcmin`, `ngrid` (≥3) — the cone field grid; `nodes` — the
  entrance-surface node lattice; `walk_arcmin` — the walk field sweep.
- `anchor` — `'rim'` | `'surface'` | `'stop'` (see the table above).
- `write_rx` / `out_rx` / `outdir` / `figures` — output control.
- Aperture/conics come **from the input Rx** — this driver measures and
  re-places the XP; it does not edit the optics.

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

## Requirements

`MACOS_HOME` set and the mmacos MEX built. Writes `pupil_id_<rx>.mat` and
`pupil_id_{cloud,zernikes,walk}.png` to `outdir` (default the input Rx's dir),
plus the revised `<rx>_xp.in`. One MATLAB process, model size 256 (do not
transition sizes in a live process).
