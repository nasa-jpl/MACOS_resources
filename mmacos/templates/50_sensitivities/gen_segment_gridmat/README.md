# gen_segment_gridmat — per-segment Zernike GridMat generator

Builds, for every grid-bearing segment of a **segmented** prescription, a
bespoke aperture mask plus an array of Zernike *mode* grids
`GridMat(:,:,ii)`, `ii` over the requested range (default Z4..Z15), each in
that segment's own clocked `(xData,yData)` frame.

These per-segment mode grids are the **influence basis** for `run_dwdgrid*` to
build the linear `dW/d(grid)` model (the effect of each mode on each segment).
They are **not** written into the prescription — collapsing the modes into a
single per-segment figure (mode × coefficient) belongs in the engine and is a
separate, later step.

## Run

```matlab
>> mmacos_setup          % once per session (adds src + sensitivities + examples)
>> run('gen_segment_gridmat.m')
```

Everything is driven by the **CONFIG block** at the top of
`gen_segment_gridmat.m`:

| Knob | Meaning |
|---|---|
| `RX` | the segmented prescription (bundled `SegDemo3conic.in`) |
| `PM_REF_ELT` | the near-pupil **Reference** to trace to (a valid trace target carrying the segmented footprint — just *before* the PM for source-defined segmentation, just *after* for non-sequential) |
| `MODES` | Zernike figure modes per segment (default `4:15`) |
| `ORTHOGONALIZE` | `true` = Gram-Schmidt over each segment aperture; `false` = plain circular Zernikes confined to the segment |
| `ZERN_TYPE` | `'ansi'` (engine `ZerntoMon1`/`NormANSI`, default) or `'noll'` (`zernike_mode`) |

## Outputs (written here)

- `gridmat_<rx>_<type>_<gs|circ>.mat` — the per-segment `GridMat` arrays
  (`out.seg(s).B` = `[N×N×numel(MODES)]`), masks, and frames, ready for
  `run_dwdgrid*`.
- `gridmat_<rx>_<type>_<gs|circ>_basis.png` — the per-segment × mode montage
  (each tile is a mode on its segment's mask, so the mask shapes show here too).

## Notes

- `SegDemo3conic.in` here is self-contained with `GridFile= none`: the
  generator needs only the segment frames and the reference-surface trace, and
  the `dW/d(grid)` sensitivity is independent of the nominal grid.
- The masks come out as the segments' true **Voronoi** footprints (macos
  `Element=Segment` with `ApType=None` assigns each ray to the nearest segment
  centre). For SegDemo3conic's symmetric flower these are congruent wedges; the
  per-segment treatment is what makes clipped **edge** segments correct.
- The engine grid library `macos.segment_grid_basis` does the work;
  `macos.write_grid_file` (validated) writes a grid to the engine GridFile
  format for the later Rx-collapse step.

See also: `macos.segment_grid_basis`, `macos.dw_dgrid_multi`,
`../../README.md`.
