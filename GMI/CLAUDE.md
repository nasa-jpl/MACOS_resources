# MACOS_resources/GMI

GMI = Generic MACOS Interface. A MATLAB mex (`GMI.mexa64`) that wraps
the SMACOS Fortran model so MATLAB scripts can run a MACOS prescription,
apply per-element perturbations, and pull back OPD/PIX/SPOT/CEF arrays.

## Build
Standalone Makefile, outside the macos cmake tree. Links pre-built
libraries from `~/dev/macos/build_release_giza/`:
- `libsmacos.a`, `libnpsol.a`, plus MATLAB libmex/libmx
- Consumes mod files from `build_release_giza/mod_smacos/`

Build entry points (in `~/dev/macos/`):
- `source ./makegmi.sh` — GMI mex only (requires macos+smacos pre-built)
- `source ./makejoint.sh` — macos + smacos + smacos_dvr + GMI

`make` doesn't see `.inc` includes, so when only `GMI.inc` changes, the
.o files don't rebuild. Force-rebuild via:
```
rm -f /home/dcr/dev/MACOS_resources/GMI/{GMI,GMIG}.o
source ~/dev/macos/makegmi.sh
```

## Files
| File | Role |
|---|---|
| `GMIG.F` | mex gateway (`SUBROUTINE mexfunction`); validates PRHS, copies into Fortran arrays, calls `GMI_DVR` |
| `GMI.F` | `MODULE GMI_mod` containing `GMI_DVR` (the work routine) and a stack of internal subroutines |
| `GMI.inc` | compile-time array sizes (`numseg`, `mzern`, `mpzern`, `mprb`, `mpgrid`, …) |
| `call_GMI.m` | MATLAB wrapper that builds `pflg` from a `param` struct and invokes the mex |
| `optiixInit_jzlou.m` | reference `param` initializer (Optiix Rx). `gmi_Init_*.m` files in user test dirs follow this pattern |
| `test_gmi.m`, `test_gmi_ff.m` | sample driver scripts (rb sensitivity loops, MonZern sensitivity loops) |

## Mex argument layout (14 args, all PRHS positional)
1. `prb` — rigid-body perturbation vector (`mprb`)
2. `pzern` — Zernike-coef perturbations on `zernSrf` elements (`mpzern`)
3. `pgrid` — grid-data perturbations (`mpgrid`)
4. `pdm` — DM/poked-actuator perturbations (`mpdm`)
5. `pfa` — `mpfa`
6. `prad` — `mprad`
7. `pimg` — `[wavelength flux …]` (`mpimg`)
8. `pflg` — packed flags + surface lists (`mpflg`)
9. `InfFcnZern` — 15-vec Zernike influence function
10. `InfFcnGrid` — `mgrid x mgrid` grid influence function
11. `fname` — Rx prescription name (no `.in`)
12. `model_size` — scalar (128/256/512/…)
13. `nProc` — parallelism scalar
14. `pmonzern` — MonZernCoef perturbations on `monzernSrf` elements (`mpzern`)

Old callers passing 13 args fail with `'GMI requires 14 input arguments'`.

## Perturbation channels
The Fortran-side `ApplyPerturbationToOpticalSystem` runs three element-list
× coefficient-vector pairs in sequence:

| Channel | Surface list (in pflg) | Coefficient vector | What it writes | Resets between calls |
|---|---|---|---|---|
| Zernike | `zernSrf` | `pzern` | `ZernCoef(iNode, iElt)` for `iNode = 4..nzern+3`; **forces** `SrfType(iElt)=8` | ELSE branch zeros `ZernCoef` and sets `SrfType=2` |
| MonZern (FreeForm) | `monzernSrf` | `pmonzern` | `MonZernCoef(iNode, iElt)` for `iNode = 4..nmonzern+3`; **does not change SrfType** (caller must already have FreeForm/14) | ELSE branch zeros `MonZernCoef` |
| Rigid body | `rbSrf` | `prb` | calls `PERTURB`/`GPERTURB` via SMACOS; modifies `TElt`, `psiElt`, `vptElt`, `pMon/xMon/yMon/zMon`, and (on FreeForm) `pFF/xFF/yFF/zFF`, `pData/xData/yData/zData` | `SetToNominalSettings` restores frame data at start of each call |

All three loop structures use `DO j=1,jXSrf-1` × `DO i=1,iXSrf` so that the
**last column** of the surface list is reserved for a flag (e.g. for
`rbSrf` it's the global(0)/local(1) frame). A single-column list
(`size(...,2)=1`) makes the outer loop run **zero** iterations and
silently disables the channel — easy to miss.

For `rbSrf` specifically: write
```
param.rbSrf = [(1:N)' zeros(N,1)];   % col 2 = 0 → global frame
```
not `[1:N]'`.

## param.monzernSrf / param.pmonzern (FreeForm Mon perturbations)
- `param.monzernSrf` is the column-vector list of FreeForm element indices
  to perturb (analog of `param.zernSrf`). Encoded into pflg right after
  the `ifFEX` block (with 9999 sentinel when absent).
- `param.pmonzern` is the per-mode coefficient vector consumed by
  `call_GMI.m` and passed as PRHS(14). Length must be
  `length(monzernSrf) * nmonzern`. `nmonzern` defaults to `nzern`
  (= `pflg(16) = param.mzern`).
- The apply loop iterates `iNode = 4..nmonzern+3`, so each segment block
  in `pmonzern` covers Born&Wolf modes 4..(3+nmonzern) — piston/tip/tilt
  are skipped (matching the existing `pzern` convention).
- Set `param.pmonzern` immediately before each `call_GMI` and (optionally)
  `rmfield(param,'pmonzern')` afterwards to leave the struct clean.

## Nominal save/restore
`ObtainNominalSettings` (called once on first entry, after `OLD`) snapshots
per-element nominal state into `*Nom` arrays. `SetToNominalSettings`
(called at the start of every later call) restores from these.

What's snapshotted:
- Scalars: `Wavelen`, `Flux`, `zEltNom(nElt-1)`
- Source: `ChfRayPos`, `ChfRayDir`, `xGrid`, `yGrid`, `zGrid`, `Tout`
- Per element: `KrElt`, `KcElt`, `nObs`, `ObsType(1,*)`, `ObsVec(:,1,*)`,
  `psiElt`, `vptElt`, `rptElt`, `TElt`,
  `pMon/xMon/yMon/zMon`, **`pFF/xFF/yFF/zFF`**, **`pData/xData/yData/zData`**,
  `SrfMetPos` (when `nMetPos>0`)

The bolded entries were added when FreeForm support landed: `PERTURB` on
SrfType=14 modifies `pFF/xFF/yFF/zFF` (always) and `pData/xData/yData/zData`
(when `nGridMat>0`), so leaving them out caused the FreeForm geometry to
drift across calls. Symptom was "the nominal-equivalent OPD differs from
its starting value after a perturbation loop". If you add new state that
PERTURB modifies, save+restore it here too.

What's NOT snapshotted (intentional — the apply blocks reset them on
their ELSE branch instead): `ZernCoef`, `MonZernCoef`, `FFZernCoef`.

## pflg layout
`pflg` is a single REAL*8 vector packing scalars, flags, and surface lists.
Built positionally in `call_GMI.m`, parsed positionally in
`GMI.F:ExtractFlagParameters`. Order matters — adding fields to one side
without the other shifts every later index. The `9999` sentinel marks
"absent" for variable-length blocks (gridSrf, zernSrf, dmSrf, rbSrf,
RptSrf, monzernSrf, …).

Fixed-position scalars at the head (positions 1..29 in the .m, parsed in
`GMI.F` around lines 600-700): `ifFEX`, `ifPupilImg`, `cGrid`, `cPix`,
`DMlim`, `ifOPD`, `ifShotNoise`, `sigReadNoise`, …, `nzern` (pflg(16)),
`QE`, `DBias`, …, `ifPIXSpotDetCheck` (pflg(24)), `ifSysCalib`,
`ifPIXElt`, `ifMetCalc`, `ifSpfCalc`, `ifRetUserSrf`. Then `ipflg = 30`
and the variable-length blocks begin: STOP (4), iFSM/TFSM, iFDP, gridSrf,
zernSrf, dmSrf, rbSrf, RptSrf+RptElt, ifFEX exit-pupil tail (7),
**monzernSrf**, RefSurfs, INTsrf.

Order of variable-length blocks in pflg (must match exactly between .m
and .F): gridSrf → zernSrf → dmSrf → rbSrf → RptSrf → ifFEX(2) tail →
**monzernSrf** → RefSurfs → INTsrf.

`mpflg = 2000` (in `GMI.inc`); a hard cap at the `.m` end aborts with
"PFLG TOO BIG".

## GMI.inc compile-time sizes
`numseg`, `numacf`, `numSAF`, `mzern`, `mgrid` are knobs. Most other
constants derive from these (e.g. `mpzern = numseg*mzern`,
`mprb = (numseg+55)*6`). When a user's runtime `param.mzern` exceeds the
compile-time `mzern`, the mex validation rejects with `'pzern must be
scalar, empty or a mpzern x 1 matrix'` — bump `mzern` (and `numseg` if
needed) and force-rebuild.

`mpzern` is shared by both `pzern` and `pmonzern` (they use the same
mex-side cap).

## Conventions / gotchas
- `call_GMI.m` is duplicated across user test dirs (e.g. `~/dev/tst_GMI/`).
  Updates to the canonical copy in `MACOS_resources/GMI/call_GMI.m` must
  be propagated by hand to active test directories. The mex error
  `'GMI requires 14 input arguments'` from a 13-arg call site means a
  stale .m copy.
- `param.pmonzern` is read inside `call_GMI.m` (default 0), not passed as
  a positional arg to `call_GMI` itself — so callers don't need a new
  function arg.
- `OPDMask_g.mat` is cached in the test cwd. After fixing rb plumbing or
  any state-leak bug, **delete the cache**; otherwise the loop reads back
  the bad mask from the previous run.
- The Rx file (e.g. `ff_pie.in`) must live in the MATLAB cwd; MACOS
  resolves it relative to the working directory, not the script path.
- Fixed-form Fortran (.F preprocessed; .f is not). `IMPLICIT NONE` is
  partial — many of the older subroutines rely on implicit typing.
- Comments inside `GMI.F` reference legacy line numbers (e.g. "lines
  940-979 are the Zernike apply block"). Those drift as edits are made;
  the structure is more durable: each channel has an `IF (ifpX) ... ELSE
  reset ... END IF` pair just before `print*,'GMI-applyPert: check-point N'`.
