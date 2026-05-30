# MACOS_resources/mmacos

MATLAB mex bridge to MACOS / SMACOS — the sibling of `../pymacos`,
sharing the same `MODULE macos_api_mod` backbone in `libsmacos.a`.

For end-user docs (build, usage, command surface) see `README.md`.
This file is the working-memory cheatsheet of gotchas not derivable
from the code.

## Layering (since §5.4 Phase 2)

```
user MATLAB
  → macos.Session(model_size)      % OO veneer; m.load_rx(...), m.opd(), ...
  → macos.opd(), macos.trace(), ...% function-style package (+macos/)
  → mmacos('cmd', args...)         % single mex with command dispatch
  → mmacos_mex.F + mmacos_gen.F    % SELECT CASE on cmd string ->
                                   %   hand-written do_<name> in mex.F,
                                   %   codegen do_<name> in gen.F,
                                   %   gen_dispatch fallback
  → MODULE macos_api_mod           % in libsmacos.a — language-neutral
                                   %   SMACOS-call backbone (shared
                                   %   with pymacos)
  → smacos engine                  % libsmacos.a
```

All three top layers (`macos.Session`, `+macos/` functions, raw
`mmacos(...)`) share libsmacos.a state — there's only one Fortran
session per MATLAB process.  Pick whichever surface fits the code:

- `mmacos('cmd', ...)` — power-user / debugging surface.  No
  validation, no unit conversion, exact pass-through.
- `macos.<name>(...)` — primary user-facing surface.  Validates args,
  converts SI ↔ BaseUnits where physical (perturb translations,
  dx_at units), returns MATLAB-idiomatic shapes (structs not tuples).
- `macos.Session` — handle class that wraps the package functions for
  dot-notation flow (`m.trace().nRays`).  No per-instance state.

`macos_api_mod` lives at `~/dev/macos/macos_f90/macos_api_mod.F90` and
is compiled INTO `libsmacos.a` — mmacos doesn't compile it locally,
just `use macos_api_mod` + link the lib.  Likewise pymacos's
`pymacos_f2py.f90` `use`s the same module.

## Gotchas

### Default compiler: gfortran (not ifx)
GMI's lesson learned: an ifx-linked mex SIGSEGVs at MATLAB process
exit because `libifcoremt.so.5` parks worker threads in the host
process and they outlive the mex DSO.  Workaround on the link line is
`-reentrancy=none` (switches to the single-threaded `libifcore.so.5`).
mmacos's Makefile already applies it under the ifx arm, but gfortran
sidesteps the question entirely, so it's the default.  Both produce
bit-identical numeric results.

To force ifx: `make FC=ifx MACOS_BUILD_DIR=~/dev/macos/build_release`.
Note that ifx and gfortran put their object files in separate
`build_release[_gfortran]` trees, so the `MACOS_BUILD_DIR` choice
needs to match the `FC` choice.

### Load_rx strips the `.in` extension
macos's `OLD` command always appends `.in`, so passing `'foo.in'`
makes it try to open `foo.in.in`.  Same `.in`-stripping workaround
pymacos applies in `macos.py:_load_rx`.  See the test_mmacos.m
fileparts-based stripping.

### `macos_init_all()` corrupts heap on model_size transitions
mmacos surfaces this bug when matlab.unittest's full suite runs the
Phase 5 PROPER-comparison tests (model_size=512) after the Phase 3/4
tests (model_size=128) in the same MATLAB session: the next
FFT-bearing trace aborts in `malloc()`/`free()` with `invalid size`
or `unaligned tcache chunk` or `munmap_chunk: invalid pointer`.

Same bug pymacos has (its `run_proper_tests.sh` invokes a separate
pytest process per phase to dodge it).  Logged as a real
engine-level fix in macos/PLAN.md §0.

**Workaround in `run_mmacos_tests.sh`:** the full-suite run splits
into per-model_size matlab -batch invocations.  When you pass a
filter arg (`./run_mmacos_tests.sh tFoo`) the script runs a single
invocation — assumes the user has narrowed to one model_size group.

If you add a new test class that uses a different model_size, update
the `SUITE_SIZE*` definitions at the bottom of
`run_mmacos_tests.sh` so the split-suite path includes it.

### `clear mmacos` hangs `matlab -batch` on R2026a — and so does implicit exit
Don't put `clear mmacos` (or `clear mex`) inside a batch-mode script.
MATLAB's mex teardown stalls when the mex was loaded from the
session's own classpath in `-batch` mode, presumably waiting on a
worker handshake that the headless session never completes.  This is
why the smoke test omits the `clear mmacos` it had during early
development.

**Stronger version of the same bug surfaced 2026-05-30:** matlab
-batch ALSO hangs at IMPLICIT process exit (after the batch script
returns normally) when a mex has been loaded.  Discovery: three
zombie `test_mmacos()` batches from 1.8 days prior were found
sleeping at 0% CPU, holding ~1 GB RAM each (3 GB total).

**Fix:** always end batch scripts with an explicit `exit(0)`.  Both
`run_mmacos_tests.sh` (full unittest) and the Makefile's `test`
target (quick smoke) now do this.  The matlab.unittest framework
already gets it right somehow — only the bare `addpath; func(); ` form
hangs — but the safe rule is to terminate every batch invocation
with `exit(0)`.

If you find another hung MATLAB process holding a mex, kill it with
`kill <pid>`; the mex unloads cleanly when the process is signalled
(it's only the orderly-exit path that stalls).

### Trace-dependent commands need `trace_rays` first
`opd`, `intensity`, `complex_field`, `dx_at`, `apodize` all read state
populated during a prior trace.  Calling them after just `load_rx` +
`modified_rx` returns either zero buffers or an `mmacos: <cmd> failed`
exception.  Call `mmacos('trace_rays', nElt)` first.

`modified_rx` between commands wipes the trace state — useful for a
clean restart but a foot-gun if you didn't mean to.

### cpp eats `//` at end-of-line in fixed-form `.F` sources
`mmacos_mex.F` is fixed-form Fortran compiled with `-cpp` (or `-fpp`).
cpp in default mode treats `//` at end of line as a C++ line comment
and elides it AND its newline — which in Fortran means the string
concat operator vanishes.  Symptoms: `'foo: ' //\n   & 'bar'` gets
preprocessed to `'foo: '     & 'bar'` and the compiler then reports
"Operands of binary numeric operator `/`" or "Missing `)`".

Fixes (use any):
1. Single-line the literal (often easiest — `-ffixed-line-length-132`
   gives plenty of room).
2. Assemble the message in a temporary CHARACTER buffer, then pass it
   to `mexErrMsgTxt`.  See the `do_init` mexFunction's `CASE DEFAULT`
   block for the pattern.
3. Avoid the trailing `//` form — split before the operator instead:
   `'foo: '`, newline `  &  // 'bar'`.

## +macos/ package conventions

User-facing surface lives in `MACOS_resources/mmacos/+macos/` (one
`.m` per public function) plus `MACOS_resources/mmacos/+macos/Session.m`
(the classdef).  When extending it:

- **Naming.**  Split getters and setters into `get_<name>` /
  `set_<name>`.  Don't mirror pymacos's overloaded form
  (`elt_vpt(srf)` vs `elt_vpt(srf, vpt)`) — MATLAB autocomplete
  surfaces both half of the contract separately and is easier to grep.
- **Validation.**  Use the `arguments` block (R2019b+).  For element
  ids: `(1,1) double {mustBeInteger, mustBePositive}`.  For vectors:
  `(3,1) double`.  For optional opts: `opts.<name>`.
- **Unit conventions.**  All user-facing translations are in **SI
  metres**.  Convert to BaseUnits via `1/CBM` inside the package
  function (not in the mex layer).  Same for `dx_at(srf, unit)` — the
  mex returns metres, the package function converts.
- **Returns.**  Prefer structs over multi-output for related fields
  (e.g. `trace` returns `s.nRays`, `s.rmsWFE`).  Vector outputs as
  column vectors (`vpt(:)`).
- **Validation defaults.**  If a default arg can't be supplied at
  declaration time (e.g. `srf = num_elt()`), use a positional
  `nargin < N` check + `validateattributes` instead of an `arguments`
  block default — `mustBePositive` etc. fire on the unset sentinel
  otherwise.

When the package function is mostly a thin pass-through, mirror it as
a one-line method in `Session.m`.  When it has non-trivial logic
(unit conversion, struct packing), keep the logic in the package
function and have `Session.m` delegate via `macos.<name>(...)`.

### Cmd-name vs api-routine-name convention

The hand-written mex cmd `'prb_elt'` calls api `prb_elt` (array form,
6×N).  The codegen-emitted cmd `'perturb_elt'` calls api `perturb_elt`
(single-element form, 3-vec th + 3-vec del + useLocalCoord).  The
package wrappers expose:

- `macos.perturb(srf, 'rotation', th, 'translation', del_SI, 'frame', f)`
  → mmacos cmd `'perturb_elt'` (single-element, SI→BaseUnits inside
  the `.m`).
- `macos.perturb_many(srf_vec, prb_6xN, ifGlobal)` → mmacos cmd
  `'prb_elt'` (array form; translations already in BaseUnits, no
  conversion).
- `macos.perturb_grp(...)` (not yet written) → mmacos cmd `'prb_elt_grp'`
  (group form via GPERTURB).

### Adding a new command
Two paths depending on whether the mapping from `macos_api_mod`
signature → mex helper is mechanical:

**Path A — codegen handles it (most cases).**  Just add the routine to
`macos_api_mod.F90`, then re-run `python3 gen_mex_wrappers.py` from
`MACOS_resources/mmacos/`.  The script regenerates `mmacos_gen.F`
with a new `do_<name>` helper and a new `CASE` in `gen_dispatch`.
The main `mexFunction`'s `CASE DEFAULT` falls through to
`gen_dispatch`, so the command becomes callable from MATLAB with no
edits to `mmacos_mex.F`.  Re-run `make` and the new command is wired.

**Path B — hand-write the helper.**  Required when:
- The arg shape exceeds rank 2 (e.g. `elt_csys_get`'s 3×3×N csys).
- The cmd needs argument repacking, e.g. complex-array split/interleave
  (`do_complex_field` / `write_imag`).
- The mmacos cmd name differs from the api routine name (e.g. `apodize`
  → `cfield_apodize`, `opd` → `opd_val`, `intensity` → `int_cmd+int_get`).
- A name collision between two api routines that map to the same mex
  cmd (e.g. the array-form `prb_elt` is hand-wired as cmd `perturb_elt`,
  so the single-element-form api `perturb_elt` is excluded from codegen
  via the `HAND_WRITTEN` set).

For Path B:
1. Add `subroutine do_<name>(nlhs, plhs, nrhs, prhs)` at the bottom of
   `mmacos_mex.F`.  Inside: `use macos_api_mod, only: <api routines>`;
   validate nrhs; copy in via `mxCopyPtrToReal8`; call into
   `macos_api_mod`; copy out via `mxCopyReal8ToPtr` (allocating via
   `mxCreateDoubleMatrix` / `mxCreateDoubleScalar`).
2. Wire the dispatch: add `CASE ('cmd_name') CALL do_<name>(...)` to
   the `mexFunction` `SELECT CASE` block (BEFORE the `CASE DEFAULT`
   fall-through so it beats `gen_dispatch`).
3. Add the api routine name to `HAND_WRITTEN` in
   `gen_mex_wrappers.py` so codegen doesn't double-emit it.
4. Re-run `python3 gen_mex_wrappers.py` to refresh `mmacos_gen.F`.
5. Add a row to the README's "MVP command surface" table.
6. Extend `test_mmacos.m` with a `check('cmd_name returns ...', ...)`.

### Codegen prhs/plhs convention
Generated helpers in `mmacos_gen.F` follow a uniform layout:
- `prhs(1)` is the command name (consumed by `mexFunction` before
  dispatch).
- `prhs(2..)` are the api routine's `intent(in)` and `intent(inout)`
  args in declaration order.
- `plhs(1..)` are the api routine's `intent(out)` and `intent(inout)`
  args in declaration order, with `ok` SKIPPED (replaced by
  `mexErrMsgTxt` on failure).
- Array dim args (e.g. `n` in `prb_elt_grp(ok, iElt(n), prb(6,n), ifGlobal(n), n)`)
  are passed explicitly by the MATLAB caller — codegen does NOT auto-
  derive them from input array shape.  The `MacosSession` class veneer
  (Phase 2) will hide this; for now, callers compute `n = length(iElt)`
  themselves.
- `intent(inout)` args are read from `prhs` AND written back to `plhs`,
  even in getter mode — the api routine zeros the buffer then fills it.
  Callers in getter mode can pass `zeros(...)` as a placeholder.

### Codegen idiosyncrasies worth remembering
- Routines declaring `OK` and `setter` as `integer` instead of `logical`
  (e.g. `src_wvl`) — the parser picks up the declared type and emits the
  matching Fortran local + comparison (`ok == 0` for integer-ok,
  `.not. ok` for logical-ok).  PASS / FAIL convention is shared (1/0).
- Local `integer, parameter :: mZernCoef = mZernModes` aliases inside
  some api routines (where the dim symbol differs from the elt_mod
  symbol).  Codegen replicates the alias in the helper and pulls in the
  rhs symbol via `use elt_mod, only: mZernModes`.
- Continuation-line subroutine arg lists (e.g.
  `elt_srf_mon_zrn_coef(ok, iElt, ZernMode, MonZernCoef_, &\n setter, reset, N)`)
  are parsed correctly — the SUB_RE regex tolerates `&\n` inside the
  paren group.

For scalar outputs use `mxCreateDoubleScalar(dble(value))`.  For 2-D
output use `mxCreateDoubleMatrix(int(M, kind=8), int(N, kind=8), 0)`
(the trailing `0` is `mxComplexity` = real).  For complex output use
`mxComplexity = 1` and the `write_imag` helper at the bottom of
`mmacos_mex.F`.

### `int8(...)` for mwSize copy arguments
`mxCopyPtrToReal8(ptr, target, N)` and `mxCopyReal8ToPtr(...)` want
their N argument as an `INTEGER(KIND=8)` (mwSize on 64-bit Linux).
Pass `int8(N)` or `int(N, kind=8)`.  Passing a plain integer compiles
under ifx but mismatches the prototype under gfortran's stricter
checks.

## Tests: two layers

| Layer | Run | Purpose |
|---|---|---|
| Quick smoke | `make test` (or `matlab -batch "test_mmacos"`) | One pass per surface: `test_mmacos.m` for raw mex, `test_macos_pkg.m` for +macos / Session.  `fprintf`-style output — easier to read while debugging.  ~10-15 s. |
| Full unittest | `make unittest` (or `./run_mmacos_tests.sh`) | 50 `matlab.unittest` tests across 5 classes in `tests/`.  Regression layer with assertion-based expectations.  ~6 s for the cold session + ~5 s for the suite. |

The two layers are intentional redundancy.  The smoke scripts pre-date
the unittest suite and serve as readable diagnostics — they print
state values so a developer can eyeball what changed.  The unittest
suite is the safety net: assertion-based, CI-friendly, encoding
specific invariants (e.g. `tPerturbRoundtrip` pins the ULP residual
finding so a future psi-renormalize fix doesn't regress).

`run_mmacos_tests.sh` shortcuts:

| Form | Scope | Wall time |
|---|---|---|
| `./run_mmacos_tests.sh` | full suite (split by size) | ~11 min |
| `./run_mmacos_tests.sh fast` | size=128 EXCEPT masks | ~10 s |
| `./run_mmacos_tests.sh masks` | CodeV mask suite | ~10 min |
| `./run_mmacos_tests.sh proper` | Phase 5 PROPER cmp | ~15 s |
| `./run_mmacos_tests.sh tFooClass` | one class | varies |
| `./run_mmacos_tests.sh -k substr` | method-name substring filter | varies |

Dev loop guidance: when iterating on a Phase 6+ slice that doesn't
touch masks or PROPER, use `./run_mmacos_tests.sh fast` between
edits — it runs the small classes in ~10 s.  Save the full
`./run_mmacos_tests.sh` for pre-commit checks.

### Standing rule: grow the regression suite alongside every phase

Per macos/PLAN.md §5.4: when a new +macos wrapper, helper, or mex
command lands in any phase, add a `tCodeV*` (or topical) test that
exercises it — even if the immediate motivating task didn't require
it.  The longer-term goal is a continuously expanding `make
unittest` covering the realistic mmacos surface; by Phase 8
(cross-language verification) there should already be substantial
mmacos-side coverage to compare bit-for-bit against pymacos.

Shared test conventions:
- `tests/proper_compare/` — Phase 5 PROPER-comparison suite (requires
  MATLAB PROPER at `~/dev/proper_matlab/`; auto-added to path by
  `run_mmacos_tests.sh`).  Pattern: one geometry struct in
  `+geometries/`, one `proper_run_<geom>.m` + `macos_run_<geom>.m`
  pair driving each engine, one `tProperCompare<Geom>.m` test class
  asserting `compare_and_record(...).max_abs_aligned < tol`.  PNG
  artefacts written to `results/phase<N>/` (gitignored).
- `tests/private/rx_fixture_path.m` — resolve named Rx fixtures from
  the pymacos corpus.
- `tests/private/rx_mask_params.m` — RX_PARAMS dict (Rx_Mask_Parabolas
  line numbers and dx_fact convention).
- `tests/private/rx_grating_001_data.m` — reference values for the
  grating slice (transcribed from pymacos rx_data.py).
- `tests/private/tolerances.m` — shared (abs, rel) tolerance constants
  mirroring pymacos `_Tol`.  Prefer these over hardcoded scalars in
  new tests so the precision contract stays consistent.
- `tests/private/{hexagon, rectangular_polygon, poly_lines,
  chk_polygon_pts, ray_pos_at_srf_in_tangent_plane}.m` — mask-
  geometry helpers ported from pymacos test_masks.py.

## Key files

| File | Role |
|---|---|
| `mmacos_mex.F` | Hand-written mex helpers + dispatcher (13 cmds), ~600 LOC |
| `mmacos_gen.F` | Auto-generated mex helpers + `gen_dispatch` (78 cmds) |
| `gen_mex_wrappers.py` | Codegen script — re-run on api signature change |
| `+macos/` | Function-package user surface (23 funcs + Session class) |
| `tests/` | matlab.unittest suite (5 classes, 50 tests) |
| `tests/private/rx_fixture_path.m` | Shared Rx-corpus locator |
| `run_mmacos_tests.sh` | Bash entrypoint for the unittest suite |
| `Makefile` | GMI-style build; `make test` (smoke) / `make unittest` (full) |
| `test_mmacos.m` | Raw-mex quick smoke |
| `test_macos_pkg.m` | +macos + Session quick smoke |
| `test_state_after_roundtrip.m` | Diagnostic probe for the ULP residual |
| `~/dev/macos/macos_f90/macos_api_mod.F90` | Shared backbone (in libsmacos.a) |
