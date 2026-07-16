# e5_pie — pie segmentation with physical polygonal apertures

> Companion to `../e5_seg/` (hex).  Manual worked example: run
> `e5_pie.m` after `mmacos_setup` — it regenerates every figure and
> `findings.txt` beside this script.

A segmented prescription built by `macos.design.segment_rx` carries
segment-ness in the SOURCE tiling only: every segment element is a
full mathematical surface with no aperture, and rays are assigned to
segments by the tiling.  That is trace-exact at nominal but
physically dishonest under perturbation — a ray can walk off its
segment and keep tracing against a surface that has no glass there.
This example declares each segment's PHYSICAL boundary as a polygonal
aperture in the Rx, and then uses those declared polygons as the basis
for MET launcher placement — the general case, which also covers
segmentations imported from .in files (Dave 2026-07-16).

## The steps (`e5_pie.m`, one figure each)

1. **Segment** the e5 monolith into a 1-ring PIE (7 segments = center
   + 6 wedges), `gap=50`.
2. **Measure the true footprints** (`e5pie_step2_footprints.png`):
   poke each segment, diff the piston-removed OPD (split on deviation
   from the median).  The central cell of the (X,L,R) hex-coordinate
   tiling is a **HEXAGON, not a disc** — apothem at the tiling midline
   `width/2`, corners toward the wedge–wedge junctions.  The
   `seg_boundary` tiling reconstruction (center hexagon at the
   physical `(width−gap)/2`; ring-1 wedges meet it along straight
   CHORDS — the hexagon's flats plus the gap, not an inner arc — with
   the gap at INTERNAL shared edges only) overlays the traced
   footprints exactly.
3. **Emit the apertures** (`e5pie_step3_apertures.png`,
   `e5pie_polyap.in`): `segment_rx(..., 'emit_apertures', true)` →
   `macos.design.seg_apertures`.  Center = the hexagon (6 vertices;
   `SetCvxPolyApVtx` generates its ApVec — no circular special case);
   wedges = convex chorded sectors + convex `PolyObsVec` obscurations
   (non-convex shapes = convex aperture minus convex obscurations):
   ring-1 wedges obscure with the apex TRIANGLE to the chord facing
   the center hexagon's flat; deeper rings with the inner-sector arc
   (ring-ring boundaries are radial).  Every polygon ships
   with an explicit `xObs` (the ChkDf2 `(psi3,psi1,psi2)` default) so
   parse order never matters.  `ap_pad` knob: 0 = physical edge
   (default), `gap/2` = trace-neutral tiling midline.
4. **Trace parity** (`e5pie_step4_clipped.png`): with `ap_pad=0`, the
   rays the source tiling places in the inter-segment gaps clip —
   physically correct (Dave's OPD review closed this: NOT an engine
   bug).  They report `RayFailElt = nElt+1`, a cosmetic return-leg
   attribution.  The surviving wavefront is unchanged.
5. **Perturbation honesty** (`e5pie_step5_poke.png`): a 100 mm wedge
   decenter.  The aperture-less trace keeps ~400 rays that have no
   glass under them; with the apertures the loss lands at the declared
   edge.
6. **Launchers on the Rx-declared edges** (`e5pie_step6_met.png`):
   with polygons in the Rx, `macos.design.seg_boundary` auto-switches
   to its **`rxpoly`** source — the boundary is the declared aperture
   polygon minus its obscuration (the physical annular band), read
   back from the prescription itself — so `add_met` places launchers
   on the edges the Rx defines, and `met_view` renders the MET scene.

## Assessment history (2026-07-16)

This example grew out of the poly-aperture feasibility assessment
(`assess_e5pie_polyap.m`, retired at commit history; verdict YES).
What it established:

- `PolyApVec` 3-D emission + explicit `xObs` loads clean;
  `SetCvxPolyApVtx` centroids/projects correctly (a first "ApVec never
  set" hypothesis was WRONG); polygons round-trip from the Rx to
  6e-8 mm.
- At the tiling midline (`pad = gap/2`) the apertures clip ZERO rays
  at the segment elements — the polygons match the source tiling.
- The `RayFailElt = nElt+1` ray loss is the gap rays being correctly
  clipped (Dave's OPD review) — not an engine bug.
- Gotchas: `macos.trace().nRays` is the SOURCE ray count, not passing
  — use `get_ray_info().ok_pass` / `get_ray_status().fail_elt` for
  parity claims; SegMirMaker `.presc` segment blocks carry NO
  ApType/nObs lines (ChkDf2 defaults them) — append, don't
  find-and-replace.

## Tests

`tSegmentRx/test_emit_apertures_and_rxpoly` (engine parity + rxpoly
round-trip + forced-source refusals) and
`tSegmentRx/test_seg_apertures_hex` (hex-corner geometry, no engine).
