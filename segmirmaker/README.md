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
from `~/dev/macos/build_release/` (`libsmacos.a` + `libslsqplib` +
`libfitslib` + the giza display stack) and consumes `.mod` files from
`build_release/mod_smacos/`.  Override with `-DMACOS_BUILD_DIR=`.
(NPSOL is gone from the macos tree; SLSQP is the back end libsmacos
references.)

```
cd ~/dev/MACOS_resources/segmirmaker
./makesegmirmaker.sh            # release build, ifx  -> build_release_ifx/
./makesegmirmaker.sh gfortran   # release build, gfortran -> build_release_gfortran/
```

Output: `build_<type>_<fc>/SegMirMaker` (e.g. `build_release_ifx/`).
Needs `macos_param.txt` in the working directory at runtime. The
script auto-selects the matching engine tree from the compiler +
build type (`build_release` / `build_release_gfortran` / debug
variants); override with `MACOS_BUILD_DIR=` in the environment.

The ifx build compiles `-fp-model strict` (no FMA contraction): a
rebuilt binary reproduces the committed `test_in/` reference outputs
**byte-identically**, which is what the batch regression below pins.
gfortran is likewise deterministic run-to-run, but its outputs differ
in the last digits from ifx's, so the byte-identity references are
**per-compiler / per-machine** — they are gitignored and regenerated
locally (see "Batch / scripted use"). The regression compares against
whatever `test_in/*.presc` + `*Hx.m` the local toolchain produced.

### macOS (Apple Silicon)

Fully supported — gfortran only (`ifx` has no arm64 build; the script
forces gfortran on Darwin automatically):

```
cd ~/dev/MACOS_resources/segmirmaker
source ./makesegmirmaker.sh gfortran   # -> build_release_gfortran/SegMirMaker
```

Prereqs mirror the rest of the Mac port ([`../../macos/MAC_PORT.md`]):
a gfortran-built engine tree at `~/dev/macos/build_release_gfortran/`,
Homebrew `readline` (the CMakeLists picks it up via `brew --prefix
readline` — the bundled `macos_f90/readline-8.2` isn't built on Mac),
and `export MACOS_HOME=~/dev/macos/macos_f90` at runtime (a missing
`macos_param.txt` aborts `init`).

**Fixed for the Mac port (2026-07-23):** `CACCEPT`/`PROMPT` declared
their character dummy as fixed-length `CHARACTER*32` while the caller
passes a `CHARACTER*20` (`GridTypeStr`); the `CVAR=' '` blank-fill
overran the actual argument by 12 bytes and clobbered the adjacent
`nRing` on the stack (read back as `0x20202020` = four blanks =
538976288 → a `7.4e-6` segment-size default and `SurfCoordFF: ray
missed parent surface`). ifx's stack layout happened to hide it;
gfortran/arm64 exposed it. The dummies are now assumed-length
`CHARACTER*(*)` — a correctness fix that applies to every platform.

## Quick start

```
cd ~/dev/MACOS_resources/segmirmaker/test_in
../build_release_ifx/SegMirMaker          # or ../build_release_gfortran/SegMirMaker
```

The interactive dialog walks you through:

1. **Parent prescription file** — type a MACOS `.in` filename (e.g.
   `my_telescope`) or hit return for a no-parent canonical conic.
2. **Parent element index** — the element number of the parent
   surface to segment (e.g. `1` for the primary).
3. **Output filename** — produces `<name>.presc` and `<name>Hx.m`.
4. **Number of DOFs per segment** — `3` (piston/tip/tilt) or `6`
   (full rigid-body). Default `3` (a blank answer gives 3; any value
   other than 3 is forced to 6).
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

7b. **Edge-sensor offset** (2026-07-19) — after the size answers, one
more prompt asks for the sensor offset from each shared-edge midpoint
(default `0.25*width`).  The measurement model is now: per SHARED
EDGE, **2 sensor locations** at +/-offset along the edge, **3 axes
per location** — 1 piston (surface normal), 2 gap (in-plane,
perpendicular to the edge), 3 shear (in-plane, along the edge) — all
purely differential (**no absolute-piston anchor row**).  A 1-ring
pie = 12 edges x 6 = 72 rows.  `Hx.m` also carries `MeasToSeg`,
`MeasAxis`, `MeasLoc`, and `SensorPos` (world sensor positions);
`macos.design.edge_sensors` ingests all of it (legacy single-axis
files still parse).  Rows use each segment's own moment arm:
dm/d(del_s) = +/- a'*T_s, dm/d(th_s) = +/- (rho_s x a)'*T_s.

The nine items above are a conceptual grouping — the binary actually
issues ~16–18 individual prompts (item 6 alone is five; ray grid type
and `SegXgrid` are asked too).  The authoritative prompt order is
pinned by the committed answer scripts `test_in/e5pie.stdin` /
`test_in/e5hex2.stdin` (see next section).

## Batch / scripted use (no interactive dialog)

There is no control-file mode; batch operation = feeding the ordered
answer list on stdin (blank line accepts the bracketed default).  Run
in a directory containing `macos_param.txt`, the parent `.in`, and any
`GridFile=` data files the parent references:

```
cd test_in
../build_release_ifx/SegMirMaker < e5pie.stdin     # regenerates e5pie.presc + e5pieHx.m
# (or ../build_release_gfortran/SegMirMaker on a gfortran / macOS build)
```

`test_in/e5pie.stdin` and `test_in/e5hex2.stdin` regenerate the
reference outputs byte-identically (run them in a scratch copy of the
directory — the tool prompts before overwriting an existing output,
which shifts the answer stream off-script). The `.presc`/`Hx.m`
references themselves are gitignored and **regenerated locally** for
the toolchain in use — the first time you build with a new compiler
(e.g. gfortran on the Mac), run both `.stdin` scripts once to populate
`test_in/` before running `tSegMirMaker`. The binary-locate logic in
the driver and the test scans `build_release_ifx` then
`build_release_gfortran` (then the debug variants), so whichever tree
you built is found automatically.

From MATLAB, use the design-layer driver — it assembles the answers,
stages a fresh scratch dir (parent + `macos_param.txt` + grid files),
runs the binary, and returns the output paths:

```matlab
out = macos.design.segmirmaker_run('test_in/e5mono.in', ...
          'name','e5pie', 'grid','Pie', 'rings',1, 'gap',50, ...
          'dofs',6, 'meas_config',1);
% out.presc, out.hx, out.workdir, out.log, out.answers
```

The `tSegMirMaker` class in `mmacos/tests` is the regression: driver
runs must reproduce the committed `e5pie` / `e5hex2` outputs
byte-identically.

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
