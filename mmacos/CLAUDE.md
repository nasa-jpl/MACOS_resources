# MACOS_resources/mmacos

MATLAB mex bridge to MACOS / SMACOS — the sibling of `../pymacos`,
sharing the same `MODULE macos_api_mod` backbone in `libsmacos.a`.

For end-user docs (build, usage, command surface) see `README.md`.
This file is the working-memory cheatsheet of gotchas not derivable
from the code.

## Layering (since §5.2)

```
user MATLAB
  → mmacos('cmd', args...)         % single mex with command dispatch
  → mmacos_mex.F                   % SELECT CASE on cmd string ->
                                   %   per-command do_<name>(plhs/prhs)
  → MODULE macos_api_mod           % in libsmacos.a — language-neutral
                                   %   SMACOS-call backbone (shared
                                   %   with pymacos)
  → smacos engine                  % libsmacos.a
```

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

### `clear mmacos` hangs `matlab -batch` on R2026a
Don't put `clear mmacos` (or `clear mex`) inside a batch-mode script.
MATLAB's mex teardown stalls when the mex was loaded from the
session's own classpath in `-batch` mode, presumably waiting on a
worker handshake that the headless session never completes.  The mex
naturally unloads at process exit anyway — let it.  This is why the
smoke test omits the `clear mmacos` it had during early development.

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

## Key files

| File | Role |
|---|---|
| `mmacos_mex.F` | Single mex, `SELECT CASE` dispatch, ~600 LOC |
| `Makefile` | GMI-style build; ifx + gfortran arms; MATLAB auto-detect |
| `test_mmacos.m` | `matlab -batch` smoke test (init + load_rx + modify + perturb + trace + opd/intensity/cfield/dx_at) |
| `~/dev/macos/macos_f90/macos_api_mod.F90` | The shared backbone (compiled into libsmacos.a) |
