# SegMirMaker

`SegMirMaker.f` — Segmented Mirror Prescription Generator. Given a
parent mirror (conic or FreeForm), interactively generates a MACOS
`.in` prescription describing a hexagonally-tiled segmented mirror
that approximates the parent, plus an `Hx.m` edge-sensor measurement
matrix.

Modernized from `SMPGe.for` (dcr, 5/7/1992, VAX Fortran). Original
preserved in `Archive/SMPGe.for`.

## Build

Standalone CMake, outside the macos tree. Links pre-built libraries
from `~/dev/macos/build_release_giza/` (`libsmacos.a` + `libnpsol` +
`lapack` + `blas`) and consumes `.mod` files from
`build_release_giza/mod_smacos/`.

```
cd ~/dev/MACOS_resources/segmirmaker
./makesegmirmaker.sh        # release build, ifx
```

Output: `build_release_ifx/SegMirMaker`. Needs `macos_param.txt` in
the working directory at runtime.

## Quick start

```
cd ~/dev/MACOS_resources/segmirmaker/test_in
../build_release_ifx/SegMirMaker
```

The interactive dialog walks you through:

1. **Parent prescription file** — type a MACOS `.in` filename (e.g.
   `iris_dp_v14`) or hit return for a no-parent canonical conic.
2. **Parent element index** — the element number of the parent
   surface to segment (e.g. `1` for the primary).
3. **Output filename** — produces `<name>.presc` and `<name>Hx.m`.
4. **Number of DOFs per segment** — `3` (piston/tip/tilt) or `6`
   (full rigid-body). Default `6`.
5. **Number of rings** — `1` for 7-segment, `2` for 19-segment, etc.
   (segments per ring = `6·iRing`; total = `1 + 6·sum(1..nRing)`).
6. **Mirror principal axis** (`psi`), eccentricity (`e`), focal length
   (`f`), inter-segment gap, segment definition standoff distance.
7. **Measurement configuration** — `1` (inner edges only) or `2` (all
   adjacencies).
8. **Size option** — `1` specify segment size or `2` specify aperture
   diameter.
9. **Preview / confirm** — segment count, layout, and sizes shown
   before the `.presc` and `Hx.m` files are written.

## Output

| File | Content |
|---|---|
| `<name>.presc` | MACOS prescription with source-section header (`nSeg`, `width`, `gap`, `SegXgrid`, `SegCoord`) followed by one element block per segment |
| `<name>Hx.m` | MATLAB-loadable edge-sensor measurement matrix; rows = sensors, cols = state DOFs |

The `.presc` file is a complete MACOS `.in` once the user prepends
their source/wavelength/aperture and any non-segment elements.

## What gets copied to each segment (FreeForm parent)

- **Conic + position**: `KrElt`, `KcElt`, `psiElt`, `VptElt` are the
  parent's. `RptElt` is the segment center (`pr` from
  `SurfCoordFF`); `TElt` is the segment face frame.
- **FF**: `lFF`, `pFF/xFF/yFF/zFF`, `FFZernType`, `FFZernCoef` —
  replicated verbatim from parent on every segment.
- **Grid**: `nGridMat`, `GridFile`, `GridSrfdx`, `pData/xData/yData/zData`
  — replicated from parent.
- **Mon**: per-segment, **not forwarded** from parent. Each segment
  gets `lMon = L2` (half segment size) but `MonZernCoef = 0` so the
  Mon contribution to the surface is zero.  The Mon coordinate frame
  is set per-segment from the segment face triad (`pMon = RptElt`,
  `xMon/yMon/zMon` = segment-face axes).

The reasoning: the FF stays parent-aligned so each segment shares the
parent's FreeForm shape; the Mon slot is reserved for the user's
per-segment figure error (added later, not by SegMirMaker).

## Conventions / locked design choices

1. **Parent source.** Read from a MACOS `.in` prescription via
   `SMACOS('OLD', filename, ...)`; user names the parent element
   index. Conic-only parents fall through `FreeFormSrf` as a
   degenerate case (same answer as SMPGe's analytical conic solve).
2. **Parent Mon → segment.** NOT forwarded. Each segment's Mon slot
   stays empty. If the user needs parent Mon on the segments, they
   must merge it into FF first.
3. **Segment pose.** Lives in `pElt/RptElt/VptElt` + `TElt`
   (unchanged from SMPGe). Per-segment `pMon = RptElt`,
   `xMon/yMon/zMon` = `TElt` columns.
4. **FF/grid propagation.** Parent `lFF`, FF coefficients, FF coord
   frame, and grid data are replicated verbatim in every segment's
   FF slot.

## Verification

Trace a single ray through the geometric center of each segment and
compare the ray's end point at the segment surface to the segment's
`RptElt`. They should agree to within `SFFZPSolve`'s tolerance
(`~10⁻¹¹` mm).

The `SEGRAYTRACE` command in macos (added alongside the FreeForm
work) automates this:
```
OLD <segmented prescription>.in
SEGRAYTRACE
2     # segment number
0     # quit
```
The end point at the segment's element should match `RptElt(:, 2)`.

A diagonal mismatch by ~180° (Seg2↔Seg5, Seg3↔Seg6, Seg4↔Seg7) means
the psi-flip in SegMirMaker didn't fire when it should have — see
`CLAUDE.md` "Segment placement & psi orientation".

A radial mismatch by ~mm (FF sag scale) means `LoadParent` didn't
populate `FFCoef_p` (the monomial form of the FF Zernikes). The fix
is in `LoadParent` — it now mirrors `tracesub.F`'s `ZerntoMon`
dispatch so `FFCoef_p`/`MonCoef_p` are correct before `SurfCoordFF`
computes `pr`.

## See also

- [`CLAUDE.md`](CLAUDE.md) — full architecture reference: data flow
  (`segmir_parent_mod`, `LoadParent`, `SurfCoordFF`, `WriteSegBlock`),
  output layout, parent-file gotchas, segment placement & psi
  orientation, dialog/defaults flow, math conventions.
- `Archive/SMPGe.for` — the 1992 VAX Fortran ancestor (CR-only line
  terminators preserved).
- `test_in/` — sample parent prescriptions (`ffparent.in`,
  `monparent.in`, `psiparent.in`) plus example outputs and the
  shared `flat.txt` / `zern41em5z155em3.txt` data files.
- `~/dev/macos/macos_f90/surfsub.F:FreeFormSrf` — the surface
  evaluator that backs `SurfCoordFF`.
- `~/dev/macos/macos_f90/macos_cmd_loop.inc` — `SEGRAYTRACE` command
  for verification.
