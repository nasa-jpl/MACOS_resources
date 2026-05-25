# GMI regression suite

Six lightweight tests that catch the most common GMI-breaking change
classes.  Each test loads a known Rx, calls the mex via the standard
`call_GMI.m` entry point, and compares the result against a committed
`.mat` reference.

> **Current GMI state (2026-05-24):** GMI's work computes correctly --
> all `call_GMI` invocations return the right answers and the
> regression suite passes -- but MATLAB SIGSEGVs at process-exit
> teardown (after `quit` has fired and every `.m` script has
> returned).  Suspected Fortran-module finalizer in libsmacos.a
> tripping on second mex-unload.  Cosmetic for batch + interactive
> use: results are written before the crash; the user just sees
> MATLAB die loudly on its way out.
>
> Because of this `run_regression.sh` can't trust MATLAB's exit
> code, so the success gate is "did the script print its
> completion marker before MATLAB died?"  Drop the marker-based
> gate and trust the exit code once the finalizer bug is fixed.

## Run

```bash
./run_regression.sh           # all tests, exits non-zero on any fail
./run_regression.sh --bootstrap   # (re)generate the .mat references
                                  # after an intentional change
```

Needs MATLAB (any `R20xx` under `/usr/local/MATLAB/`) and a built
`GMI.mexa64` in the parent directory.

## What's covered

| Test | Rx | Channel | Catches |
|---|---|---|---|
| 01 smoke (optiix) | `optiixonaxisz1_v4_pmsm_met` | nominal | mex won't load, missing symbol, signature regression |
| 02 nominal repro (optiix) | optiix | nominal x2 | `SetToNominalSettings` state-drift |
| 03 Zernike response (optiix) | optiix | Z4 on Elt 4 | Zernike-channel apply path, `iNode=4..nzern+3` indexing, ELSE-branch reset |
| 04 smoke (e5hex1) | `Rx_e5hex1.in` (FreeForm chain) | nominal | basic FreeForm loading |
| 05 nominal repro (e5hex1) | e5hex1 | nominal x2 | FreeForm save/restore (`pFF/xFF/yFF/zFF`, `pData/xData/...`) |
| 06 Zernike response (e5hex1) | e5hex1 | Z4 on Elt 8 (Zernike-typed) | Zernike apply on a mixed FreeForm+Zernike chain |

Tolerance: hard absolute `|a - b| <= 1e-12` (in mm OPD units).  Loose
enough to survive harmless re-link-order drift, tight enough that any
real numeric change shows.

## Files

```
regression/
├── README.md
├── run_regression.sh            entry point
├── regression_main.m            top-level runner
├── bootstrap_reference.m        regenerate references
├── tests/
│   ├── test01_smoke_optiix.m
│   ├── test02_nominal_repro_optiix.m
│   ├── test03_zern_response_optiix.m
│   ├── test04_smoke_e5hex1.m
│   ├── test05_nominal_repro_e5hex1.m
│   └── test06_zern_response_e5hex1.m
├── lib/
│   ├── compare_within.m         absolute-tolerance equality check
│   ├── init_optiix.m            param struct for the Optiix Rx
│   └── init_e5hex1.m            param struct for the e5hex1 Rx
├── reference/                   committed `.mat` ground truth
│   ├── nominal_optiix.mat
│   ├── zern_response_optiix.mat
│   ├── nominal_e5hex1.mat
│   └── zern_response_e5hex1.mat
└── Rx/                          prescription files seen by SMACOS OLD
    ├── Rx_e5hex1.in
    └── optiixonaxisz1_v4_pmsm_met.in   (copied from parent on first run)
```

## When a test fails

The summary table tells you which test + the max |diff| vs reference.

- **All tests fail with mex error**: rebuild GMI.  `source ~/dev/macos/makegmi.sh`.
- **Just `nominal_repro_*` fails (round-trip non-zero)**: state-drift bug; check `ObtainNominalSettings` / `SetToNominalSettings` in `GMI.F` for fields not being snapshotted (e.g., the FreeForm `pFF/xFF/yFF/zFF` class of fix).
- **All tests fail with `vs reference`**: an intentional change has shifted the numbers.  Inspect the diff, regenerate via `./run_regression.sh --bootstrap`, commit the new `reference/*.mat`.
- **One specific `*_response_*` fails**: the perturbation-apply path for that channel is broken.

## Adding a test

Drop a `testNN_*.m` in `tests/`, add it to the `tests` cell array in
`regression_main.m`, regenerate references via the bootstrap script,
commit both the new test and the new `reference/*.mat`.
