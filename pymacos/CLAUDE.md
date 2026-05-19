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

## Key files

| File | Role |
|------|------|
| `src/cmake/source/pymacos.f90` | `MODULE api` — ~3700 LOC, one Fortran wrapper per public Python entry.  USES smacos modules directly (Kinds, elt_mod, macos_mod, ...). |
| `src/pymacos/macos.py` | typed Python API — input validation, state globals, raises Exception on Fortran-side failure. |
| `src/pymacos/__init__.py` | Win DLL search-path shim; re-exports `macos`. |
| `tests/sensitivities/channels.py` | `ZernikeCoefChannel` + Rx-parse helpers.  Calls `m.modify()` in `apply/restore`. |
| `tests/sensitivities/jacobian.py` | `sensitivity_jacobian(channels, wf_func, delta)` — central/forward FD engine. |
| `tests/sensitivities/dw_dz_zernike.py` | driver; writes `.mat` in m2v.m convention (column-major non-zero mask of nominal OPD). |
| `tests/proper_compare/run_broadband_vortex.py` | Cycle 5 vortex coronagraph driver. |
| `tests/Rx/e5hex1.in` | local copy of `~/dev/macos/ZGD_test_files/e5hex1.in` — sensitivity-matrix test Rx (7 hex segments + FreeForm lens). |

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
  Jacobian, .mat output, MATLAB verify).  See git log for the commit
  series.  Default test Rx is `e5hex1.in` (model_size=256, modes
  4..45 → 378-channel Jacobian).
- **dr-dev branch (merged via main?)**: Cycle 5 vortex coronagraph
  + oversized-rays scheme.  `apodize_complex` wrapper, FreeForm
  surface helpers (`findFreeFormElts`, the Mon/FFZern get/set pairs).
