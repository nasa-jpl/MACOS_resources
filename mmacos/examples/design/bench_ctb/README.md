# ctb — coronagraph-testbed Bench example

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
