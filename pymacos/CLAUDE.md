# MACOS_resources/pymacos

Python interface to MACOS / SMACOS via an f2py wrapper. Three layers:
user Python → `src/pymacos/macos.py` (typed API, ~75 fns + validation)
→ `pymacosf90.*.so` (f2py) → `src/cmake/source/pymacos.f90`
(`MODULE api`) → SMACOS static libs from `~/dev/macos/` build tree.

For the overall layout, build steps, and test suites see `README.md`.
This file is the working-memory cheatsheet of gotchas and shortcuts
not derivable from the code.

## Build (fast path)

After editing `pymacos.f90` or `macos.py`:

```bash
cd /home/dcr/dev/MACOS_resources/pymacos/src/cmake/build
bash -c 'source /opt/intel/oneapi/setvars.sh intel64 >/dev/null 2>&1; make'
```

`macos.py` is pure Python — no rebuild needed for edits there.

Smoke test: `pymacos/tests/sensitivities/run_dw_dz_zernike.sh --help`
exercises the venv + oneAPI runtime setup without touching macos.

## Critical gotchas

### MODIFY required after direct coefficient writes
Direct writes to `ZernCoef`, `MonZernCoef`, or `FFZernCoef` (via the
`setEltSrf{Zern,MonZern,FFZern}…` wrappers) do **not** invalidate
macos's per-element ZerntoMon cache.  Without calling `m.modify()`
before the next `trace_rays()`, ZerntoMon reuses the old `MonCoef` /
`FFCoef` and the trace silently returns nominal results — i.e.
zero sensitivity.

Sensitivity channels (`sensitivities/channels.py`) call `m.modify()`
inside `apply()` and `restore()` for this reason; any new channel
class that mutates coefficient arrays must do the same.

### Pre-existing broken wrappers in `macos.py`
- **`getEltSrfZern(iElt)`** calls `lib.api.getEltSrfZern`, which is
  not the actual f2py name (`elt_srf_zrn_get`).  Will raise
  `AttributeError`.  Use **`getEltSrfZernMode(iElt, modes)`** instead
  (added in dr-dev2; routes through `elt_srf_zrn_coef`).
- **`setEltSrfZernMode(iElt, modes, coefs)`** has an f2py-side error
  path that triggers on single-mode calls ("(len(zernmode)>0) failed
  for hidden ok").  Use **`setEltSrfZernCoef(iElt, modes, coefs)`**
  instead (also added in dr-dev2).

These older wrappers are untouched on purpose — too much downstream
code may already use them.  The new symmetric pair is the safe path
for new sensitivity / control code.

### `getEltSrfZern` *can* match SrfType 8 or 13
`elt_srf_zrn_coef` (the underlying Fortran for the Zern get/set
wrappers) was loosened in dr-dev2 to also accept SrfType=ZrnGrData
(=13).  Both store their Zernike component in `ZernCoef`; the pure
Zernike check (`SrfType==8` only) was too restrictive.

### `init()` is one-shot per process
`_SYSINIT` / `_isRx` are module-level globals.  Changing `model_size`
requires a fresh interpreter.  pymacos tests work around this by
spawning subprocess per model-size when needed (see e.g.
`run_broadband_*.py`'s `_spawn` pattern).

### SMACOS dispatch path (when adding new commands)
SMACOS dispatches via `#include "macos_cmd_loop.inc"` from
`smacos.F:210` -- the SAME command loop the interactive macos uses.
`macos_ops.F:MACOS_OPS` is NOT the SMACOS top-level dispatcher; it's
only called from the inner optimization loop in `smacos_compute.inc`.
When wiring a new SMACOS-callable command:
1. Add the branch to `macos_cmd_loop.inc` (so both interactive and
   SMACOS reach it).
2. Add a `LoadStack` entry in `smacosutil.F` -- pushes args onto the
   STACK that `IACCEPT_S` reads in SMACOS mode.  (Without this entry
   the command's first arg becomes the next-command's `command`
   variable, and you get "** Unknown command = SXP" -- the failure
   I hit when SXP first didn't dispatch.)
3. Pymacos Fortran wrapper that sets `command='X'`, fills `IARG`/
   `CARG`, calls `SMACOS(...)`.  No need to do anything in
   `macos_ops.F` unless you also want it invokable from optimization
   loops.

### SXP vs FEX for FP-perturbation dw/dx
`m.fex()` defines the EP as the optical conjugate of the Stop --
upstream-only chief-ray geometry, INSENSITIVE to FP motion.  Use
`m.sxp()` instead when you want the EP radius to track the FP
position (FEX clone with EP radius = chief-ray distance EP-to-FP,
added on macos joint-dev).  `FocalPlaneChannel` modes in
`sensitivities/channels.py`:
- `track` (default): DOF-aware EP follow per the "EP sphere
  centered on FP" physics.  Internal SXP-with-vpt-restore refines
  radius without undoing the lateral/rotational EP motion.
  Rotations rotate EP rigidly about FP's RptElt, NOT EP's own
  RptElt -- direct vpt/psi/rpt setter geometry (m.perturb always
  rotates about the element's own RptElt).
- `sxp`: simple FP-perturb-then-SXP.  Captures FP-Tz (focus); FP-
  Tx/Ty come out 0.
- `srs`: macos's SRS sphere<-plane case is "not implemented" --
  no-op for now.
- `fex`: diagnostic only; gives 0 by construction of FEX.

`--update-ep {sxp,fex}` is an opt-in for upstream-perturbation EP
shifts.  Conflicts with `--fp-mode=track` -- the wf-side SXP
overwrites the EP vpt that track set.  Pair `--fp-mode=sxp` with
`--update-ep=sxp` for consistent behavior.

## Key files

| File | Role |
|------|------|
| `src/cmake/source/pymacos.f90` | `MODULE api` — ~3700 LOC, one Fortran wrapper per public Python entry.  USES smacos modules directly (Kinds, elt_mod, macos_mod, ...). |
| `src/pymacos/macos.py` | typed Python API — input validation, state globals, raises Exception on Fortran-side failure. |
| `src/pymacos/__init__.py` | Win DLL search-path shim; re-exports `macos`. |
| `tests/sensitivities/channels.py` | `ZernikeCoefChannel` + Rx-parse helpers.  Calls `m.modify()` in `apply/restore`. |
| `tests/sensitivities/jacobian.py` | `sensitivity_jacobian(channels, wf_func, delta)` — central/forward FD engine. |
| `tests/sensitivities/dw_dz_zernike.py` | dw/dz driver (Zernike-coef perturbations); writes `.mat` in m2v.m convention. |
| `tests/sensitivities/dw_dx.py` | dw/dx driver (rigid-body perturbations).  `RigidBodyChannel` + DOF-aware `FocalPlaneChannel` with track/sxp/srs/fex modes; `--update-ep` opt-in for upstream EP shifts. |
| `tests/proper_compare/run_broadband_vortex.py` | Cycle 5 vortex coronagraph driver. |
| `tests/Rx/e5hex1.in` | local copy of `~/dev/macos/ZGD_test_files/e5hex1.in` — sensitivity-matrix test Rx (7 hex segments + FreeForm lens).  Contains `ApStop= 0 0 0` (object-space stop) because the segmented primary is the natural stop but pymacos's `stop()` refuses Segment surfaces. |

## .mat output convention (sensitivities)

Modelled on `~/matlab/m2v.m`.  First call extracts non-zero pixels of
the nominal OPD column-major; same mask reused for every column of
the Jacobian, so a MATLAB workflow can take a fresh measurement and
call `w = m2v(opd_meas, indx)` to line up row-for-row with `dwdz`.

Variables in `dwdz_<rx_stem>.mat`:
- `dwdz`           — `(Nw, Nz)` float64 Jacobian
- `w_nom`          — `(Nw, 1)`  float64 nominal wavefront vector
- `indx`           — struct `{i: float64 col, j: float64 col, size: 1×2 float64}`
- `channel_names`  — `(Nz, 1)` cell array of strings
                      ("Elt 4 MonZern15", etc.)
- `nGridPts`       — float64 scalar (= mat_shape(1))
- `mat_shape`, `model_size`, `wf_elt`, `delta`, `n_zcoef`,
  `zmode_start` — float64 (MATLAB prefers loaded scalars as doubles;
   `int64` surprises arithmetic downstream)
- `rx`, `method`, `kinds` — strings / cellstr

If you add fields, cast to `float64` for numeric values and `object`
ndarray for cellstr.

## Tests

- **CodeV cross-validation**:  `pytest test_api_rx_grating.py test_masks.py`
  (geometric paths; 6601 tests).
- **PROPER cross-validation**: `pytest proper_compare/` or
  `./run_proper_tests.sh` (physical-optics; subprocess-per-phase to
  avoid model-size leak).
- **Sensitivities**: not a pytest suite — run
  `tests/sensitivities/run_dw_dz_zernike.sh` and then MATLAB's
  `verify_dwdz.m` (in the same directory) against the loaded .mat.

## Recent activity (May 2026)

- **dr-dev2 branch**: sensitivity-matrix engine (this file, channels,
  Jacobian, .mat output, MATLAB verify).  Two drivers:
  - `dw_dz_zernike.py` -- Zernike-coef perturbations (Zern, MonZern,
    FFZern).  Default Rx `e5hex1.in` at model_size=256, modes 4..45
    → 378-channel Jacobian.
  - `dw_dx.py` -- rigid-body perturbations (Rx,Ry,Rz,Tx,Ty,Tz per
    actual optic).  `FocalPlaneChannel` with track/sxp/srs/fex modes;
    `--update-ep {sxp,fex}` opt-in for upstream EP-shift capture.
  Both drivers write `.mat` files in m2v.m convention and emit a
  per-element panel figure.  Shell wrappers `run_dw_dz_zernike.sh` /
  `run_dw_dx.sh` handle Intel oneAPI + venv setup.
  Companion pymacos wrappers `sxp()`, `srs()`, `ors()` for the EP/FP
  setup workflow (FEX → ORS → SRS).  `sxp()` requires the SXP
  command on the macos side (joint-dev `ca2f82b`).
- **dr-dev branch (merged via main?)**: Cycle 5 vortex coronagraph
  + oversized-rays scheme.  `apodize_complex` wrapper, FreeForm
  surface helpers (`findFreeFormElts`, the Mon/FFZern get/set pairs).
