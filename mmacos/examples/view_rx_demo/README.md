# view_rx_demo — the general prescription visualizer

`macos.view_rx()` draws the LOADED prescription — **any** .in file, no
design-layer structs: beam, optics, and MET paths if present (Dave
2026-07-16).  It is the modern MATLAB successor of John Lou's 2007
demo 3-D visualizer (Lou-UpdateNotes.txt, never generalized): all data
comes from the engine itself.

- **beam** — a sparse-but-FILLED ray bundle in true global 3-D: a
  rings-and-spokes pattern (chief + 3 rings × 8 spokes by default) cut
  from the engine's per-trace ray-position history (`macos.ray_hist`,
  backed by the engine `RayPosHist` capture — John Lou's Vis3D
  substrate, now API-exposed).  The full traced grid is available, so
  any pattern can be cut; `'rim'` and the legacy dual meridian
  `'fans'` remain as options.  Correct for folded / off-axis systems.
- **optics** — each optic rendered as a **solid body** with lighting:
  the true aperture boundary (Circular/Elliptical/Hexagonal `ApVec`,
  Polygonal `PolyApVtx`, `lMon` disc, or the ray-footprint hull — all
  via `macos.get_elt_info`) lifted onto the real conic sag
  (`KcElt`/`KrElt`, sign calibrated against the actual ray crossings),
  extruded to a plate of thickness aperture/12.  **Consecutive
  Refractor pairs join into one glass solid** (front + back surface +
  barrel).  Passive elements (Reference / Return / FocalPlane /
  Obscuring) draw as outline frames — they are not hardware.  Works
  for every element type the trace crosses, Segment / non-sequential
  included — which per-element `macos.trace(k)` harvesting cannot do;
- **MET** — gauge beams launcher→fiducial via `macos.met_geom`
  whenever the Rx declares `nMetPos`/`tMetElt`/`metBeamFlg`, colored
  per source element.

One runner (`view_rx_demo.m`), three stock cases, PNGs beside it:

| figure | prescription |
|---|---|
| `view_rx_cass.png` | `CassWithExitPupil.in` (manual example) |
| `view_rx_coro.png` | `CoroExample.in` coronagraph train (manual example) |
| `view_rx_met.png`  | `e5mono_met.in` = e5mono + hand-added met keywords — MET paths on a plain Rx that never touched the design layer |

For segmented-primary MET systems the design layer adds annotation on
top of this scene: `macos.design.met_view` (segment tiles via
`seg_boundary`, face-on launcher panel, edge clearance, baseline
overlay) — see `design/examples/e5_seg/`.
