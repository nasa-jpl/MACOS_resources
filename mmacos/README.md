# mmacos — MATLAB mex bridge to MACOS / SMACOS

The MATLAB sibling of [pymacos](../pymacos).  A single `mmacos.mexa64`
shares the language-neutral `MODULE macos_api_mod` (compiled into
`libsmacos.a`) that pymacos's f2py wrappers also bind into.  Same
SMACOS-call layer, two languages, one source of truth.

## Three user surfaces

All three share the same libsmacos.a state — there's only one Fortran
session per MATLAB process.  Pick whichever notation suits your code:

### 1. Raw mex (power-user / debug)

```matlab
mmacos('init', 128)
n = mmacos('load_rx', '/path/Rx_Cass')
[nRays, rmsWFE] = mmacos('trace_rays', n)
opd = mmacos('opd');
```

91 commands wired (13 hand-written + 78 auto-generated from
`macos_api_mod.F90` signatures).  No validation, no unit conversion —
exact pass-through to the Fortran API.

### 2. `+macos/` function package (primary user surface)

```matlab
addpath('/home/dcr/dev/MACOS_resources/mmacos')

macos.init(128)
n = macos.load_rx('/path/Rx_Cass.in')       % accepts .in extension
s = macos.trace();                           % returns struct(.nRays, .rmsWFE)
W = macos.opd();                             % N×N real
I = macos.intensity(n);                      % N×N real
cf = macos.complex_field(n);                 % N×N complex

% Perturbations: SI metres in / out.  Library converts to BaseUnits internally.
macos.perturb(2, 'rotation', [1e-6;0;0], 'translation', [0;0;1e-9], ...
              'frame', 'global');

% Unit conversion in MATLAB.
dx_mm = macos.dx_at(n, 'mm');                % 'm' default; 'mm', 'cm', 'um', 'native'
```

23 functions in `+macos/`.  Validation via `arguments` blocks, struct
returns for multi-field results, SI ↔ BaseUnits conversion via CBM,
split `get_X` / `set_X` so MATLAB autocomplete surfaces both halves
of each contract.

### 3. `macos.Session` class (OO veneer)

```matlab
m = macos.Session(128);
m.load_rx('/path/Rx_Cass.in');
s = m.trace();
W = m.opd();
m.perturb(2, 'rotation', [1e-6;0;0], 'frame', 'global');
```

Thin handle class wrapping the function package.  Same methods, same
state — useful when MATLAB code reads more naturally with dot
notation.  No per-instance Fortran state (libsmacos.a owns it).

## Build

Requires:
- MATLAB R2024a or newer (auto-detected under `/usr/local/MATLAB`)
- A built macos tree at `~/dev/macos/build_release_gfortran/` (default)
  — `cd ~/dev/macos && source ./makegfortran.sh release` gets you there
- gfortran (default) OR Intel oneAPI ifx; both produce numerically
  identical mex artefacts

```bash
cd /home/dcr/dev/MACOS_resources/mmacos
make                                                  # gfortran default
make FC=ifx MACOS_BUILD_DIR=~/dev/macos/build_release # ifx alternative
make MATLAB_DIR=/usr/local/MATLAB/R2025b              # specific MATLAB
make test                                             # quick smoke
make unittest                                         # full matlab.unittest suite
```

`mmacos.mexa64` lands next to the Makefile.  `make clean` removes
build artefacts.

### gfortran vs ifx

gfortran is the default: it produces a mex that **exits MATLAB cleanly**
(same reason GMI defaults to gfortran).  ifx works but historically
SIGSEGV-d at MATLAB process exit because of how `libifcoremt`'s parked
threads outlive the mex DSO.  Mitigated for mmacos by `-reentrancy=none`
on the ifx link (single-threaded `libifcore.so.5` — same workaround
GMI uses).  Both compilers produce bit-identical numeric results.

## Tests

Two layers:

| Layer | Run | Purpose |
|---|---|---|
| Quick smoke | `make test` | `test_mmacos.m` (raw mex) and `test_macos_pkg.m` (+macos + Session).  `fprintf`-style — easy to read while debugging.  ~10 s. |
| Full unittest | `make unittest` | 50 `matlab.unittest` tests across 5 classes in `tests/`.  Assertion-based regression layer.  ~6 s cold + 5 s suite. |

Unittest classes:

| Class | Coverage |
|---|---|
| `tMmacosCmd` | Raw `mmacos('cmd', ...)` mex surface |
| `tMacosPkg` | `+macos/` function package |
| `tMacosSession` | `macos.Session` class delegation |
| `tCrossSurface` | State coherence across all three surfaces |
| `tPerturbRoundtrip` | CPERTURB_PROG round-trip invariants (incl. the 1-ULP psi residual; pinned so a prospective renormalize fix lands cleanly) |
| `tCodeVGrating` | CodeV grating-API regression (Phase 4 slice 1) |
| `tCodeVApeMasks*`, `tCodeVObsMasks*` | CodeV mask regressions, 8 classes (Phase 4 slice 2) |
| `tProperCompareCassFF` | PROPER vs mmacos Cass-FarField comparison (Phase 5 slice 1) |

The Phase 5 PROPER suite requires MATLAB PROPER v3.3.1 at
`~/dev/proper_matlab/` (download from
[sourceforge.net/projects/proper-library](https://sourceforge.net/projects/proper-library/files/)).
`run_mmacos_tests.sh` auto-adds it to the path; tests skip with a
helpful error if `prop_begin` isn't resolvable.

Filter options:
```bash
./run_mmacos_tests.sh tMacosPkg        # one class
./run_mmacos_tests.sh -k roundtrip     # method names matching substring
```

## Codegen

`gen_mex_wrappers.py` parses `~/dev/macos/macos_f90/macos_api_mod.F90`
and emits `mmacos_gen.F` (78 `do_<name>` mex helpers plus a
`gen_dispatch` fallback that the main `mexFunction` falls through to
after its 13 hand-written cases).  Re-run after any api signature
change:

```bash
python3 gen_mex_wrappers.py
```

The codegen handles scalar / ≤2D-array intent(in/out/inout), local
`integer, parameter` aliases for elt_mod dim symbols (e.g.
`mZernCoef`), multi-line subroutine arg lists, and routines that
declare `OK` / `setter` as `integer` rather than `logical`.

Two signatures still need hand-written helpers if exercised:
`elt_csys_get` (rank-3 array) and the single-element-form
`perturb_elt` (collides with the array-form `prb_elt`; the array form
wins under the cmd name `prb_elt`, codegen emits the single form under
`perturb_elt`).

## Files

| File | Role |
|---|---|
| `mmacos_mex.F` | Hand-written mex helpers + dispatcher (13 cmds) |
| `mmacos_gen.F` | Auto-generated mex helpers + `gen_dispatch` (78 cmds) |
| `gen_mex_wrappers.py` | Codegen script |
| `mmacos_gen_cmds.txt` | Machine-readable full cmd inventory |
| `+macos/` | Function-package user surface (23 funcs + `Session.m`) |
| `tests/` | matlab.unittest suite (5 classes, 50 tests) |
| `tests/private/rx_fixture_path.m` | Shared Rx-corpus locator |
| `run_mmacos_tests.sh` | Bash entrypoint for the unittest suite |
| `test_mmacos.m` | Raw-mex quick smoke (used by `make test`) |
| `test_macos_pkg.m` | +macos + Session quick smoke |
| `test_state_after_roundtrip.m` | Diagnostic probe for the ULP residual |
| `Makefile` | GMI-style build; `make` / `make test` / `make unittest` |
| `mmacos.mexa64` | Built artefact (gitignored) |
| `~/dev/macos/macos_f90/macos_api_mod.F90` | Shared backbone (in libsmacos.a) |

Links against:
- `libsmacos.a` (carries `macos_api_mod`, the SMACOS engine, etc.)
- `libfitslib.a` (FITS I/O — pulled in by smacos's plot stack)
- Optionally `libnpsol.a` + `liblapacklib.a` + `libblaslib.a` when the
  macos tree was built with `-DUSE_NPSOL=ON`
- MATLAB `libmx.so` + `libmex.so` + `libmat`
