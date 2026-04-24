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
- Segment blocks are written to a scratch unit (unit 8,
  `STATUS='SCRATCH'`) inside the segment loop, then copied to unit 2
  after the header. This lets the header precede blocks even though
  `SegCoord` values are built up inside the generation loop.
- `FmtD(x)` formats a REAL*8 with up to 16 sig figs, stripping trailing
  zeros from the mantissa:
  `1.2d0 → 1.2E+00`, `0.866…6d0 → 8.660254037844387E-01`.
  Used for `width`, `gap`, `SegXgrid` to keep round numbers compact
  while preserving precision on parent-derived values.

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

## Conventions
- Math helpers (`DORTHOGANALIZE`, `EQUATE`, `ZERO`, `DOT`, `MAG`) are
  linked from `libsmacos` via `USE math_mod` / `USE smacos_mod`. Don't
  inline them — smacos_mod is required regardless for `SMACOS('OLD')`.
- Prompts use `DACCEPT`/`IACCEPT` (original VAX names kept). Prompt
  format is sized dynamically to avoid clipping.
- Statement labels: main program uses 1–21 for CONTINUE, 500–575 for
  FORMAT. Scratch-file read loop uses 801/802 (label 7 is taken).
