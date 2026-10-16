# e5hex2_refzern — conforming Reference → GS Zernike basis → dw/d(grid)

A passive **conforming Reference** surface (`Element=Reference` / `Surface=Zernike`)
on the e5hex2 19-segment telescope, used to **establish the segment shapes** for
Gram-Schmidt Zernike basis development and multi-field `dw/d(grid-data)`
sensitivities — the analytic counterpart of the grid-based `run_dwdgrid*`
workflow.

The Reference is **passive** — the ray passes straight through, so it has no
effect on the light (adding it just records an intermediate point on the same
straight ray). It exists only to give `segment_grid_basis` a valid near-pupil
trace target carrying the segmented footprint (you cannot trace to a `Segment`
element). **Before this engine feature a Reference could not carry a Zernike
surface, so `e5hex2grid.in` could not be loaded at all.**

## Workflow (run after `mmacos_setup`)

```matlab
mmacos_setup

% (0) OPTIONAL: prove the conforming Reference is passive (with-ref == no-ref)
run e5hex2_refzern            % e5hex2grid.in vs e5hex2.in -> identical OPD

% (1) MAKE + SAVE the per-segment GS Zernike basis (the slow step; runs once)
run make_gs_basis_e5hex2      % -> gridmat_e5hex2grid_ansi_gs.mat (+ basis montage)

% (2) USE the saved basis to build the multi-field dw/d(grid) sensitivities
run run_dwdgrid_multi_e5hex2  % -> dwdgrid_multi_e5hex2grid_*.{png,mat}
```

The build is split so the (slow) per-segment basis build runs once and the
`dw/d(grid)` assembly can be re-run cheaply from the saved `.mat`.

## Files

| File | Role |
|---|---|
| `e5hex2grid.in` | 19-hex telescope **with** the conforming `Reference`/`Zernike` at elt 1 |
| `e5hex2.in` | the same system with the reference removed (passivity baseline) |
| `flat.txt` | all-zero grid the segments reference (nominal grid figure) |
| `e5hex2_refzern.m` | (0) passivity check |
| `make_gs_basis_e5hex2.m` | (1) build + save the GS basis |
| `run_dwdgrid_multi_e5hex2.m` | (2) load basis → dw/d(grid) sensitivities |

## Engine + tooling support (what this needed)

- **Engine:** `Element=Reference` now accepts `Surface=Zernike` (8) and
  `Surface=Aspheric` (3) so the reference can **load and carry** the basis
  definition (`EltSurfCompat`, `iosub.inc`).  The reference stays **passive** —
  `RefSrf` is unchanged; the coefficients are stored but never injected into the
  wavefront (deliberately different from `Reference`+`GridData`, which *is* an
  active phase grid).  Two shared-parser bugs were also fixed: `ZernModes=`
  single-line-vs-wrapped read, and the `SrfTypeName(EltID)` mislabel in the
  Element/Surface warning.
- **mmacos:** `segment_grid_basis` and `channels.grid_channels` now exclude
  non-segment grid-bearing elements (a conforming Reference; a downstream
  full-aperture refractor) from the per-segment candidate set — `find_grid_elts`
  keys on `nGridMat>0` alone, so those must be dropped.

## Rx gotcha: the segment grid frame must be the CLOCKED (Mon) frame

For the per-segment poke to localize, each segment's grid coordinate frame
(`pData/xData/yData/zData`) must equal its **clocked monomial frame**
(`pMon/xMon/yMon/zMon`), NOT the un-clocked global frame (`xData = 1 0 0`).
With the global frame the poke cannot land on the segment's footprint and the
`dW` collapses to a central artifact.  The Rx here has `Data == Mon` per
segment; if you build your own segmented Rx, make the grid frame match the
monomial frame.
