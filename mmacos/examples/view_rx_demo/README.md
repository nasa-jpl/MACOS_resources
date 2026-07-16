# view_rx_demo — the general prescription visualizer

`macos.view_rx()` draws the LOADED prescription — **any** .in file, no
design-layer structs: beam, optics, and MET paths if present (Dave
2026-07-16).  It is the modern MATLAB successor of John Lou's 2007
demo 3-D visualizer (Lou-UpdateNotes.txt, never generalized): all data
comes from the engine itself.

- **beam** — the engine DRAW command's real traced meridian fans in
  true global 3-D (`macos.draw_rays3d`, backed by the new engine
  `Draw3DVec` capture + `draw_rays3d_get`), both fans, correct for
  folded / off-axis systems;
- **optics** — per-element surface cross-section curves through the
  actual beam footprint.  Works for every element type the trace
  crosses, Segment / non-sequential included — which per-element
  `macos.trace(k)` harvesting cannot do (the engine's OPD command
  refuses NSRefractor/Segment/NSReflector targets, and used to
  infinite-loop on them in batch mode; both fixed alongside this
  example);
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
