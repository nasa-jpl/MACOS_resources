# e5pie poly-aperture assessment (2026-07-16)

**Question (Dave):** should each segment carry a polygonal aperture in
the Rx, with that polygon as the basis for launcher placement — the
general case, covering .in-defined segmentations too?

**Verdict: yes — the engine machinery is essentially ready; one
downstream interaction needs a root-cause before productizing.**

Prototype: `assess_e5pie_polyap.m` (run after `mmacos_setup`).  Builds
the SegMirMaker e5 PIE system through `segment_rx`, emits a variant
where every ring wedge carries `ApType=Polygonal` + explicit `xObs` +
`PolyApVec` (convex chorded sector; optional convex `PolyObsVec`
inner-sector obscuration — annular sectors are non-convex, engine
convention is convex aperture minus convex obscurations) and the
center segment `ApType=Circular`.  Artifacts beside the script:
`e5pie_polyap.in`, `e5pie_met_layout.png` (first real-fixture pie
`seg_boundary`/`met_view` render), `clipped_mask.png`, `findings.txt`.

## What works (verified)

1. **Parse + projection + round-trip**: `PolyApVec` 3-D emission with
   an explicit `xObs` (ChkDf2's (psi3,psi1,psi2) convention) loads
   clean; `SetCvxPolyApVtx` centroids/projects correctly (my first
   "ApVec(3) never set" hypothesis was WRONG — it rewrites ApVec
   properly); polygons read back from the Rx to 6e-8 mm.  A
   `seg_boundary('rxpoly')` reader is trivially feasible.
2. **Segment-level trace neutrality**: at nominal, ZERO rays are
   clipped at the segment elements (1–7) — per-ray `fail_elt`
   histogram.  The wedge polygons (half-span pi/nring − (g/2−PAD)/rc,
   radial band width/2 ± PAD, circumscribed outer chords) match the
   source pie tiling.
3. **The clipping mechanism is live** on segments (poke demos show
   differential loss vs the aperture-less baseline).
4. **Pie `seg_boundary`/`met_view`/edge launcher placement** work on
   the real SegMirMaker pie fixture (previously only synthetic).

## The open engine question

With segment poly apertures present, **456 of 12,520 rays (3.6%) are
lost with `RayFailElt = 14` on a 13-element system** — one past the
train, i.e. some virtual/return-leg stage of the OPD calculation (the
e5 train has Return elements at 11/12).  The lost rays form the
OUTERMOST pupil ring (median r ≈ 3933 of 4000 mm) uniformly in
azimuth; rms WFE shifts 1.3% because the ray SET changes.  It is NOT
the inner obscuration (identical with obs disabled) and NOT a
polygon-margin issue (+25 mm pads barely move it).  Root-cause needed:
how does the aperture/obscuration state of Segment elements interact
with the post-train reference-surface / return-leg element index?
Reproducer: this script, `fail_elt` histogram.

## Productization plan (pending the root-cause)

1. `seg_boundary('rxpoly')`: read `PolyApVec`/`ApVec` polygons from
   segment blocks — launcher placement from Rx-declared edges (works
   for imported segmented prescriptions immediately; no engine work).
2. `segment_rx('emit_apertures', true)`: write the tile polygons into
   the segmented Rx (hex corners exact; pie = chorded sectors +
   inner-sector obscurations), with a clearance pad knob.
3. Trace-parity test in tSegmentRx gated on the elt-14 fix.
