# pymacos

Python interface to MACOS / SMACOS.

The intent is to expose the SMACOS Fortran library through a typed Python API
so prescriptions can be driven (and their outputs compared against other
optical codes such as CodeV) from Python scripts and notebooks instead of the
interactive MACOS prompt.

> Status: under active development by collaborators outside the main MACOS
> source tree. This README captures the layout as it stands so that downstream
> changes in `macos_f90/` can be assessed against the pymacos surface and the
> build can be reproduced.

## Layout

```
pymacos/
├── src/
│   ├── instructions.txt        Win/Linux build steps (Intel oneAPI)
│   ├── instructions_2.txt      same, more detail + Linux specifics
│   ├── macos/                  EMPTY — placeholder for the SMACOS source tree
│   │                           (see "Build" below; cmake expects a sibling
│   │                            build of libsmacos / libnpsol / libblaslib /
│   │                            liblapacklib at macos/cmake/build/{libs,mod})
│   ├── cmake/
│   │   ├── CMakeLists.txt      builds the f2py wrapper into pymacosf90.*.so
│   │   └── source/
│   │       ├── pymacos.f90     ~3600 LOC; MODULE api wrapping SMACOS()
│   │       └── pymacos.inc
│   └── pymacos/
│       ├── __init__.py         Win DLL search-path shim; re-exports macos.py
│       ├── macos.py            ~3000 LOC; typed Python API on top of pymacosf90
│       ├── toolbox.py
│       ├── version.py
│       ├── macos_param.txt     namelists for model sizes 128..4096
│       └── pymacosf90.*.so     <-- f2py-generated, dropped here by cmake
└── tests/
    ├── context.py              sys.path shim + wavelength decorators
    ├── test_settings.py        global pass/fail tolerances + Qform helper
    ├── rx_data.py              fixtures: RxDataBase + per-Rx data classes
    ├── test_api_rx_grating.py  pytest, parametrized over macos_size
    ├── test_masks.py
    ├── macos_param.txt
    └── Rx/
        ├── Rx_Mask_Parabolas.in        MACOS prescription
        ├── Rx_Mask_Parabola_glb.in
        ├── Rx_Mask_parabolas.seq       CodeV sequence file (cross-validation)
        ├── Rx_Mask_parabolas_glb.seq
        └── Grating_example_001.in
```

## Architecture

Three layers, top-down:

```
┌────────────────────────────────────────────────────┐
│  Python user code / pytest                         │
└────────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────┐
│  src/pymacos/macos.py        (~75 public functions)│
│    typed API, input validation, module globals      │
│    (_SYSINIT, _isRx, _NELT, _MODELSIZE)             │
└────────────────────────────────────────────────────┘
                       │  imports as `lib`
                       ▼
┌────────────────────────────────────────────────────┐
│  pymacosf90.*.so       (built by numpy.f2py)        │
└────────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────┐
│  src/cmake/source/pymacos.f90   MODULE api          │
│    each Python entry maps to a Fortran subroutine,  │
│    which calls SMACOS(command, CARG, DARG, IARG,    │
│                       LARG, RARG, OPDMat, RaySpot,  │
│                       RMSWFE, PixArray)             │
└────────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────┐
│  SMACOS static libs (libsmacos / libnpsol /         │
│    libblaslib / liblapacklib) from the main         │
│    macos_f90 source tree                            │
└────────────────────────────────────────────────────┘
```

`pymacos.f90` `use`s the SMACOS modules directly (`Kinds`, `math_mod`,
`param_mod`, `src_mod`, `elt_mod`, `macos_mod`, `smacos_mod`), so the
build needs the `.mod` files from the SMACOS build in addition to the
static libraries.

## Public Python API (src/pymacos/macos.py)

~75 functions. Grouped by purpose:

| Group | Examples |
|-------|----------|
| Lifecycle | `init(model_size)`, `load(rx)`, `save(rx)`, `has_rx`, `num_elt`, `rx_modified`, `model_size` |
| Source | `src_info`, `src_csys`, `src_wvl`, `src_fov`, `src_size`, `src_finite`, `src_sampling`, `sys_units` |
| Element pose | `elt_vpt`, `elt_rpt`, `elt_psi`, `elt_csys`, `elt_srf_csys` |
| Element surface | `elt_kc`, `elt_kr`, `elt_zrn_*`, `elt_grid_*`, `elt_grating_*` |
| Groups | `elt_grp`, `elt_grp_rm`, `elt_grp_wipe`, `elt_grp_fnd`, `elt_grp_any`, `elt_grp_max_size` |
| Perturbation | `prb_elt`, `prb_grp` |
| Trace / outputs | `trace_rays`, `opd`, `spot`, `fex`, `xp`, `stop`, `modify` |

Each `elt_*`-style getter/setter follows the pattern: pass `None` to get
the current value, pass a value (scalar or array) to set it. Inputs are
validated against the loaded prescription before the Fortran call.

## Build

Two cmake passes. **Intel oneAPI (icx/icpx/ifx) is required** —
the cmake hard-codes Intel compilers; gfortran is not currently
supported. cmake **≥ 3.31**, Python **≥ 3.13**, NumPy **≥ 2.2**.

### 1. SMACOS static libraries

`src/macos/` is empty in the repo — it is meant to hold (or symlink to) a
checkout of the main MACOS source tree. The pymacos cmake expects the
artifacts at:

```
pymacos/src/macos/cmake/build/libs/   libsmacos.a libnpsol.a libblaslib.a liblapacklib.a
pymacos/src/macos/cmake/build/mod/    *.mod
```

How that build is produced is outside the pymacos repo. Two practical
options:

- Drop in (or symlink) the MACOS source tree at `src/macos/` and run the
  cmake build under `src/macos/cmake/build/` per the upstream MACOS
  instructions.
- Build SMACOS libraries from the main `~/dev/macos/` tree using
  `source ./makems.sh` and then copy/symlink the resulting `libsmacos.a`,
  `libnpsol.a`, `libblaslib.a`, `liblapacklib.a` and `.mod` files into
  the directory layout above.

Make sure the SMACOS libs were built with the same `mpts` / model-size
namelist (`macos_param.txt`) that the Python tests expect; pymacos
defers model size to `init(macos_size)` at runtime but the underlying
arrays are sized at compile time from `param_mod`.

### 2. pymacosf90 module

```bash
cd pymacos/src/cmake
mkdir -p build && cd build
rm -rf *
source /opt/intel/oneapi/setvars.sh intel64   # path is site-specific
cmake -DCMAKE_C_COMPILER=icx \
      -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_Fortran_COMPILER=ifx \
      -S ..
make
```

cmake runs `numpy.f2py` against `pymacos.f90` to generate
`pymacosf90module.c` + a Fortran wrapper, then `Python_add_library`
compiles them and links against the four SMACOS libs. The resulting
`pymacosf90.cpython-3XX-*.so` is dropped into `src/pymacos/` so that
`from . import pymacosf90 as lib` resolves at import time.

Quick smoke test in the same shell:

```bash
cd pymacos/src/pymacos
python -c "import pymacosf90 as f; print(f.__doc__)"
```

Should print the list of exported f2py routines.

### Windows notes

`src/instructions.txt` covers the Visual Studio 2022 + NMake path; the
pymacos cmake branches on `WIN32` to use `.lib` suffixes and
`/names:uppercase /assume:nounderscore`. Anaconda Python ships the
Intel/MS shared libs; standalone Python needs Intel oneAPI
redistributables and the `os.add_dll_directory(...)` call already wired
into `src/pymacos/__init__.py` (edit the path there if your Intel install
is not under `C:\Program Files (x86)\Intel\oneAPI\2025.3`).

## Tests

```bash
cd pymacos/tests
pytest                          # all tests
pytest test_api_rx_grating.py   # one file
pytest -k Grating               # one selection
```

Tests are parametrized over `macos_size` via a module-scoped fixture
(`@pytest.fixture(params=(128,))`) so the same test can be run against
multiple compiled array bounds. Tolerances live in `test_settings._Tol`
(positional 1e-10, directional 1e-13, path-length 1e-11, value 1e-15).

`tests/Rx/` ships both `.in` (MACOS) and `.seq` (CodeV) versions of each
prescription so the trace outputs from pymacos can be checked against
CodeV's reference output — this is the principal cross-validation path.

## Gaps / notes for downstream changes

- **No upstream MACOS pin.** `src/macos/` is empty and there is no
  `.gitmodules`. Whoever builds the libs needs to know which MACOS
  revision they came from. If we break the `elt_mod` / `macos_mod` /
  `smacos_mod` interface on the macos side, pymacos's
  `pymacos.f90 module api` will fail to compile here.
- **Toolchain pins** are tight: cmake 3.31, Python 3.13, NumPy 2.2,
  Intel oneAPI. No gfortran path today.
- **API/Fortran symmetry:** every public Python function maps to a
  Fortran subroutine; the count is currently ~75 Python defs vs ~79
  Fortran subroutines (some helpers like `_chk_*`, `getEltSrfZern` are
  Python-only, and some get/set pairs collapse to one Python
  function via a `setter` flag on the Fortran side).
- **No README / CHANGELOG / CI** in the upstream repo. The two
  `instructions*.txt` files are the only build doc.
- **State machine:** `_SYSINIT` and `_isRx` are module-level globals;
  every entry point goes through `_chk_macos_and_rx_loaded()` before
  hitting Fortran. Tests must `init()` before `load()`, and `init()`
  is one-shot per process (changing model size requires a fresh
  interpreter).
