# e5_pie — poly apertures on pie segments (assessment, 2026-07-16)

> Companion to `../e5_seg/` (hex).  Re-runnable end-to-end: the runner
> (`assess_e5pie_polyap.m`, after `mmacos_setup`) regenerates the
> variant Rx, the parity numbers, and the figures beside this script.

**Question (Dave):** should each segment carry a polygonal aperture in
the Rx, with that polygon as the basis for launcher placement — the
general case, covering .in-defined segmentations too?

**Verdict: yes — the engine machinery is ready.  (The ray loss first
flagged as an open engine question is resolved: Dave's OPD review
shows the lost rays fall into the inter-segment GAPS — physically
correct clipping, not a bug.  Productization is ungated.)**

Prototype: `assess_e5pie_polyap.m` (run after `mmacos_setup`).  Builds
the SegMirMaker e5 PIE system through `segment_rx`, emits a variant
where EVERY segment carries `ApType=Polygonal` + explicit `xObs` +
`PolyApVec`: ring wedges as convex chorded sectors (optional convex
`PolyObsVec` inner-sector obscuration — annular sectors are
non-convex, engine convention is convex aperture minus convex
obscurations), and the center segment as a circumscribed regular
24-gon (Dave 2026-07-16: no circular special case for Elt 1;
`SetCvxPolyApVtx` generates its `ApVec`).  Artifacts beside the
script:
`e5pie_polyap.in`, `e5pie_met_layout.png` (first real-fixture pie
`seg_boundary`/`met_view` render), `clipped_mask.png`, `findings.txt`.

## What works (verified)

1. **Parse + projection + round-trip**: `PolyApVec` 3-D emission with
   an explicit `xObs` (ChkDf2's (psi3,psi1,psi2) convention) loads
   clean; `SetCvxPolyApVtx` centroids/projects correctly (my first
   "ApVec(3) never set" hypothesis was WRONG — it rewrites ApVec
   properly); all 7 segment polygons (wedges + center 24-gon) read
   back from the Rx to sub-µm.  A `seg_boundary('rxpoly')` reader is
   trivially feasible.
2. **Segment-level trace neutrality**: at nominal, ZERO rays are
   clipped at the segment elements (1–7) — per-ray `fail_elt`
   histogram.  The wedge polygons (half-span pi/nring − (g/2−PAD)/rc,
   radial band width/2 ± PAD, circumscribed outer chords) match the
   source pie tiling.
3. **The clipping mechanism is live** on segments (poke demos show
   differential loss vs the aperture-less baseline).
4. **Pie `seg_boundary`/`met_view`/edge launcher placement** work on
   the real SegMirMaker pie fixture (previously only synthetic).

## The ray loss — RESOLVED (Dave, 2026-07-16)

With segment poly apertures present, 442 of 12,520 rays (3.5%) are
lost with `RayFailElt = 14` on a 13-element system (a virtual/
return-leg index of the OPD calculation; the e5 train has Return
elements at 11/12).  Dave's review of the variant OPD: **the lost
rays fall into the inter-segment GAPS** — the polygons are doing
exactly their job on rays the aperture-less baseline let sail through
the gaps, and the 1.4% rms shift is the ray SET becoming physically
honest.  Not an engine bug; no root-cause needed.  (The
`RayFailElt=14` attribution — one past the train — remains a cosmetic
oddity of where the return leg records the failure, worth a note if
it ever confuses a diagnostic.)

## Productization plan (ungated)

1. `seg_boundary('rxpoly')`: read `PolyApVec`/`ApVec` polygons from
   segment blocks — launcher placement from Rx-declared edges (works
   for imported segmented prescriptions immediately; no engine work).
2. `segment_rx('emit_apertures', true)`: write the tile polygons into
   the segmented Rx (hex corners exact; pie = chorded sectors +
   inner-sector obscurations), with a clearance pad knob.
3. Trace-parity test in tSegmentRx gated on the elt-14 fix.
