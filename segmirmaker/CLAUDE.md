# MACOS_resources/segmirmaker

`SegMirMaker.f` — Segmented Mirror Prescription Generator.
Modernized from `SMPGe.for` (dcr, 5/7/1992, VAX Fortran).
Original preserved in `Archive/SMPGe.for` (CR-only line terminators).

## Build
Standalone cmake, outside the macos tree. Links pre-built libraries
from `~/dev/macos/build_release_giza/` (libsmacos.a + libnpsol + lapack
+ blas) and consumes mod files from `build_release_giza/mod_smacos/`.

- Default build dir: `build_debug_ifx/`
- VS Code: F5 launches Debug SegMirMaker; launch.json cwd = `test_in/`
- Fixed-form `.f` (no CPP). Needs `macos_param.txt` in CWD at runtime.

## Design choices (locked; the plan that drove the modernization)
1. **Parent source.** Read from a MACOS `.in` prescription via
   `SMACOS('OLD', filename, ...)`; user names the parent element index.
2. **Parent Mon → segment.** NOT forwarded. Each segment's Mon slot
   stays empty (`lMon=0`, `MonCoef=0`). If the user needs parent Mon on
   the segments, they must merge it into FF first.
3. **Segment pose.** Lives in `pElt/RptElt/VptElt` + `TElt` (unchanged
   from SMPGe). Per-segment `pMon = RptElt`, `xMon/yMon/zMon` = TElt
   columns (the face-aligned triad), so the Mon frame rides with the
   segment face.
4. **FF/grid propagation.** Parent `lFF`, FF coefficients, FF coord
   frame, and grid data are replicated verbatim in every segment's FF
   slot. Conic-only parents fall through `FreeFormSrf` as a degenerate
   case — same answer as SMPGe's analytical conic solve.

## Key structural elements
- `segmir_parent_mod` (top of SegMirMaker.f) — holds `_p` parent state
  (`Kc_p`, `Kr_p`, `f_p`, `e_p`, `psi_p`, `pv_p`, `pMon_p`, `lFF_p`,
  `FFCoef_p`, `GridMat_p`, …) and the `FmtD` compact formatter.
- `SurfCoordFF` — wraps MACOS `FreeFormSrf` (macos_f90/surfsub.F); one
  call per segment returns surface point, normal, and local coord frame.
- `LoadParent(iParent)` — copies parent's element arrays from `elt_mod`
  into `_p` after SMACOS has loaded the .in file. Sets
  `parentIsFF = .TRUE.` unconditionally (the guard for "did OLD
  succeed?" is in the main program, not here — see gotcha below).
- `WriteSegBlock(iUnit, ...)` — emits one per-segment element block.
  Legacy branch (`parentIsFF=.FALSE.`) is byte-identical to SMPGe's
  EltType=5 + conic output; FF branch emits Surface=FreeForm with full
  FF/grid replication.

## Output layout
- `.presc` starts with a source-section header modeled on SegDemo3.in:
  `nSeg`, `width`, `gap`, `SegXgrid`, then the `SegCoord` table.
- Two scratch units buffer output:
  - Unit 8: segment blocks (appended during the segment loop)
  - Unit 9: source-section header (written after the loop)
  Both are `STATUS='SCRATCH'` (auto-deleted on CLOSE).
- `DumpScratch(iSrc, iDst, maxBlocks)` helper copies a scratch unit
  line-by-line to `iDst` (or stdout if `iDst ≤ 0`). `maxBlocks > 0`
  stops after that many `iElt=` blocks — used to bound the preview.
- `FmtD(x)` formats a REAL*8 with up to 16 sig figs, stripping trailing
  zeros from the mantissa:
  `1.2d0 → 1.2E+00`, `0.866…6d0 → 8.660254037844387E-01`.
  Used for `width`, `gap`, `SegXgrid` to keep round numbers compact
  while preserving precision on parent-derived values.

## Preview / confirm before write
- After the segment loop the program prints the source-section header
  plus the first segment block to stdout (`DumpScratch(8, -1, 1)`),
  followed by a `... (N-1 more segment blocks omitted from preview) ...`
  note when N > 1.
- Prompts `Write prescription to file? [Y/n]:`. Default is Y.
- On `n`/`N`: closes unit 2 (`.presc`) and unit 3 (`Hx.m`) with
  `STATUS='DELETE'` so no partial files are left, then STOPs.
- On Y/blank: `DumpScratch(9, 2, 0)` + `DumpScratch(8, 2, 0)` flushes
  the full content (header + all segments) to the `.presc` file.

## Segment placement & psi orientation
- The in-plane `(xs, ys)` basis comes from
  `DORTHOGANALIZE(psi, SegXgrid, ys)` which sets `ys = psi × xs` (then
  rebuilds `xs` orthogonal to `ys`).  `zs = psi`.
- For nominal mirrors (`psi` roughly along +z, e.g. SegDemo3 with
  `psi=(0,0,1)`), `ys ≈ +y` in world coords and segment numbering
  starts at upper-right (`iSeg=1` at `(+dx, +dy)`), traversing
  counter-clockwise.
- For back-facing mirrors (`psi` with negative z component, e.g. an
  off-axis primary that bulges away from the source so `psi` points
  toward the source), `ys` would naturally come out anti-aligned with
  world +y, placing iSeg=1 at lower-right and reversing the perceived
  rotational direction.  SegMirMaker detects this case (`zs(3) < 0`)
  and rotates the in-plane basis 180° about `zs` (negates both `xs`
  and `ys`).  Net effect: iSeg=1 lands at upper-left and traversal is
  clockwise in the viewer's frame (looking from the source side
  toward the mirror).  Frame stays right-handed.
- Sanity-check the output by tracing a ray to the geometric middle of
  each segment and comparing to the segment's `RptElt` — they should
  agree to within ring-curvature precision.  Mismatch by ~180° means
  the orientation flip didn't fire when it should have (or vice
  versa).

## Parent-file gotchas
- **SMACOS `OLD` appends `.in` internally.** User typing `foo.in`
  causes it to look for `foo.in.in`. SegMirMaker strips a trailing
  `.in`/`.IN` from the user's input before calling SMACOS.
- **SMACOS `OLD` reports failure to stdout but returns no status.**
  SegMirMaker INQUIREs the `.in` file itself; on failure, prints
  `SegMirMaker: parent file X not found; using legacy conic dialog.`
  and falls back. Without this, `LoadParent` silently copies zeros
  from elt_mod and the user sees all-zero defaults in the dialog.

## Dialog / defaults flow
- `_p` vars are loaded by `LoadParent` (or zeroed by `ZeroParent`).
- Before the DACCEPT prompts: `_p` is synced from current working vars
  (handles both parent and no-parent cases); `psi_p`, `f_p`, `e_p`,
  `pv_p` are then passed directly as DACCEPT defaults.
- After the prompts: user's final values are written back to `_p` so
  `SurfCoordFF` and `WriteSegBlock` see whatever the user chose.
- No-parent canonical defaults: `psi=[0,0,1]`, `f=18`, `e=0`,
  `SegXgrid=[1,0,0]`, `gap=0`.
- Segment-size defaults come from MACOS `src_mod::Aperture` when a
  parent was loaded (`USE src_mod, ONLY: Aperture`):
  option 1 (segment width) default = `Aperture/(1+2·nRing)`;
  option 2 (aperture diameter) default = `Aperture`.
  Fall back to legacy `0.03` / `12` when `Aperture ≤ 0`.

## Conventions
- Math helpers (`DORTHOGANALIZE`, `EQUATE`, `ZERO`, `DOT`, `MAG`) are
  linked from `libsmacos` via `USE math_mod` / `USE smacos_mod`. Don't
  inline them — smacos_mod is required regardless for `SMACOS('OLD')`.
- Prompts use `DACCEPT`/`IACCEPT` (original VAX names kept). Prompt
  format is sized dynamically to avoid clipping.
- Statement labels: main program uses 1–21 for CONTINUE, 500–575 for
  FORMAT. Short labels 1–9 are all in use.
