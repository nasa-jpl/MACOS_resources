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
1. Add the matching `subroutine do_<name>(nlhs, plhs, nrhs, prhs)` at
   the bottom of `mmacos_mex.F`.  Inside: `use macos_api_mod, only:
   <api routines>`; validate nrhs; copy in via `mxCopyPtrToReal8`;
   call into `macos_api_mod`; copy out via `mxCopyReal8ToPtr`
   (allocating via `mxCreateDoubleMatrix` / `mxCreateDoubleScalar`).
2. Wire the dispatch: add `CASE ('cmd_name') CALL do_<name>(...)` to
   the `mexFunction` `SELECT CASE` block.
3. Add a row to the README's "MVP command surface" table.
4. Extend `test_mmacos.m` with a `check('cmd_name returns ...', ...)`.

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
