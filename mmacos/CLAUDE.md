# MACOS_resources/mmacos

> **Post-compaction / post-upgrade — re-read the docs first.** After a
> context compaction or a tooling upgrade, before resuming build/engine
> work, re-read the doc set across ALL working dirs (the conversation
> summary drops mechanical how-to): `macos/CLAUDE.md` + `macos/PLAN.md`;
> this file + `MACOS_resources/{pymacos,GMI}/CLAUDE.md` + `README.md`s;
> and the agent `MEMORY.md` (esp. the build/test workflow entries).

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

**Source layout (since the 2026-06-17 reorg):** the mex sources
(`mmacos_mex.F`, `mmacos_gen.F`), the codegen (`gen_mex_wrappers.py`,
`mmacos_gen_cmds.txt`), the **built `mmacos.mexa64`**, and the `+macos/`
package all live under **`src/`** (mirrors pymacos's `src/pymacos`).  The
MATLAB path needs `mmacos/src` on it — the `Makefile` (`SRCDIR`), the test
runner, and the examples all point there.  `Makefile`,
`run_mmacos_tests.sh`, `tests/`, `examples/`, and the smoke `test_*.m`
stay at the top level.

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

User-facing surface lives in `MACOS_resources/mmacos/src/+macos/` (one
`.m` per public function) plus `MACOS_resources/mmacos/src/+macos/Session.m`
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
`MACOS_resources/mmacos/src/`.  The script regenerates `mmacos_gen.F`
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

## Design layer (`src/+macos/+design/`, Sprint 2A-i, 2026-06-12)

`macos.design.System` is the high-level **import / analysis / optimize**
front-end (plan: `~/dev/macos/PLAN_DESIGN_LAYER.md`).  It wraps the
existing `+macos` surface; it does NOT touch Fortran.

- `System.from_rx(path, 'model_size', N)` — load via SMACOS, read
  element/source params back through the getters into a plain `spec`
  struct (state-as-data; **no MATLAB text parser**).  `from_rx` leaves
  the Rx loaded.
- `vary(elt, param, ...)` — declare a design var (pure spec edit).
- `sensitivities(...)` — **harvests** `dw_dx` + `dw_dz_zernike`,
  returns them as **separate** `out.rigid` / `out.zern` structs (user
  joins `[rigid.dwdx, zern.dwdz]` if wanted).  Bitwise-equal to the
  standalone drivers — never re-derives FD.
- `evaluate(x)` / `optimize(...)` — rigid-body MVP; fmincon over
  [0,1]-normalized bounds with a ray-loss penalty; restores nominal by
  `load_rx` (Q5 says that's bit-stable).

**Conventions baked into the surface (don't regress):**
- **DOFs are name-based** (`'Tz'`, `'despace'`, `'tilt'`), NEVER the
  0-based `dw_dx` index — that index lives only in the private
  `dofs_to_idx_` translator.  `vary` stores `dof_name`, not a number.
- **Rigid perturbations are LOCAL/EltCoord frame** (`RigidBodyChannel`
  hardwires `'frame','local'`): `Ty` is the element's own y, not global
  Y.  The `{'global','local'}` knob in `dw_dx` is `group_coords` (groups
  only).
- Examples (REORGANIZED 2026-06-19, `d9978d8`): `examples/` now has three
  category dirs — `sensitivities/`, `design/` (telescope builder examples
  + `tma_widefield/`), `coronagraph/` (coro_planet_demo, coro_walkthrough,
  `coro/` = the old design/coro E1–E4).  `tCoroContrast`'s path fixture
  points at `templates/30_instruments/coro_experiments`.

### Layout viewer — Telescope.diagram / view_layout (Sprint 4, 2026-06-19)
For the off-axis-fold work (the coaxial TMA self-obscures — M1 + FP sit
on the M2→M3 beam).  `Telescope.diagram()` = cheap marginal-beam side
view (exterior clipping).  `Telescope.view_layout(plane,opts)` = the
revealing view: the engine's REAL ray bundle (`macos.draw_rays`, a wrapper
over the engine `draw_rays_cmd`/`draw_rays_get` data-only-DRAW getter —
see macos/CLAUDE.md) plus conic-sag surfaces drawn to each optic's ACTUAL
beam FOOTPRINT (not aperture); opts `plane` (YZ/XZ/XY), `istart`/`iend`
(slice), `hide` (drop surfaces), `nrays`.  Slicing + hiding kill the
FALSE conflicts a 2-D projection paints (a fold sends light behind the
PM).  `check_clipping()` (3-D, phantom-free) + the off-axis fold builder
are the remaining pieces.  These touch the engine, so the mex needs a
relink after pulling: rebuild `build_release_gfortran`, re-run
`gen_mex_wrappers.py`, `rm src/mmacos.mexa64`, `make FC=gfortran`.
(macos `f3e98e5` + mmacos `caf3b86` are LOCAL/UNPUSHED.)

### General Rx viewer — macos.view_rx / met_geom / design.met_view (2026-07-16)
`macos.view_rx()` = prescription-agnostic 3-D scene from the LOADED Rx
only (Dave: "work with any prescription — beam, optics, MET paths if
present").  **v2 (solid look, Dave-planned):** beam = sparse-but-FILLED
rings-and-spokes bundle cut from the engine per-trace ray-position
history (`macos.ray_hist` → api `ray_hist_set`/`ray_pos_hist_get`,
exposing traceutil_mod `RayPosHist`/`LRayOKHist` — Lou's Vis3D
substrate; slot 1 = source plane); optics = SOLID sag-following SHELLS
with lighting + meridian profile curves on BOTH faces (the flat-back
plate read as a cylinder — Dave) — aperture truth via
`macos.get_elt_info` (api `elt_info_get`: EltID/ApType/ApVec/xObs/
lMon/PolyApVtx), conic sag from Kc/Kr with the SIGN calibrated per
element against the actual crossings (no KrElt convention baked in),
thickness aperture/12; **consecutive Refractor pairs JOIN into one
glass solid**; Reference/Return/FocalPlane/Obscuring = outline frames;
ApType=None → smoothed ray-footprint hull; **Segment elements on a
hex-segmented source draw as EXACT hex tiles** (api `src_seg_get`:
GridType/nSeg/width/gap + consensus clocking from segment centers —
no overlap, gaps read).  Options: bundle 'rings'|'rim'|'fans', bodies
'solid'|'outline'|'patch', `show` 'beam'|'beam+met'|'met', ray_color
(per-channel colors, LightTools-deck style — Dave's reference:
MACOS_sandbox/181202-Layouts PDF); a RING circles the beam at the
source plane (collimated-source location cue).  `macos.view_std` =
standard beam-aligned 4-panel figure (front-from-behind-source / back
/ iso / side; SOURCE AT LEFT, light travels right; per-panel [az el]
fine-tune; manual camera + `camva('auto')` — a hand-computed
CameraViewAngle badly misframes, and titles collide with axis labels
unless the panels are `axis off`).  GOTCHAS:
(1) `ray_hist('on')` must DIRTY the trace (`macos.modify()`) or a
previously-traced session returns an empty history (grid-setter
retrace class — the veneer does it); (2) engine `PolyApVtx` is
IN-PLANE (SetCvxPolyApVtx projects out psi; frame xa=xObs⊥psi,
ya=psi×xa, origin VptElt, minus ApVec(1:2) centroid) — reconstruct
in-plane, don't expect the emitted 3-D vertices back; (3) the DRAW fan
resamples its OWN rays — validate ray_hist against it to the
source-grid pitch, not exact identity.  `macos.design.met_view(seg,am)`
= the segmented-primary annotated wrapper (tiles, face-on panel, M2-M3
inset).  Tests: tViewRx (SUITE_FREEFORM, model 256), tMet, tMetView.
`findall(fig)` CANNOT reach the sgtitle layout Text (R2026a) — titles
mirror onto `fig.Name`.  Engine leg in macos_api_mod (same rebuild
chain as any api_mod change; pymacos export of the new routines =
deferred, mmacos-only for now).

### Segment physical apertures + rxpoly (2026-07-16, e5_pie example)
`segment_rx(..., 'emit_apertures', true)` → `design.seg_apertures`:
every segment's PHYSICAL boundary lands in the merged .in as
`ApType=Polygonal` + explicit `xObs` (**from psiElt, NOT zMon** — face
triads clock; frames now carry `.psi`) + 3-D `PolyApVec`.  Hex = exact
hex_tile corners; **pie center cell = a HEXAGON** (the (X,L,R)
hex-coord tiling's central cell, footprint-verified — NOT a disc),
apothem `(width−gap)/2`, flats facing ring-1 wedge centers; pie wedges
= convex chorded sector + convex `PolyObsVec` (non-convex = convex ap
minus convex obs); **ring-1 wedges abut the hexagon along straight
CHORDS** (flat (w−g)/2 + gap g → chord at (w+g)/2; obscuration = apex
TRIANGLE, NOT an inner arc — an arc draws a spurious circle around
Seg 1, Dave), deeper rings obscure with the inner-sector arc.  Gap
sits at INTERNAL shared edges only (g/2 each side; tiling rim carries
none).  rxpoly GOTCHA: the polyshape subtract of %.10E-rounded Rx
vertices can leave numeric slivers as extra regions — seg_boundary
takes the LARGEST region, never `boundary(shp,1)` blindly (a sliver as
boundary #1 made one tile vanish from met_view).  `ap_pad` 0 = physical (gap
rays clip, correct — Dave); `gap/2` = trace-neutral midline.
`seg_boundary` gained `source="auto"|"tiling"|"rxpoly"`: **auto uses
'rxpoly'** when every segment block declares PolyApVec — boundary =
declared polygon MINUS its obscuration (polyshape subtract), so
add_met/met_view place launchers on Rx-DECLARED edges (works for
imported segmented Rx).  Worked example: `templates/20_segmentation/e5_pie/`
(manual-grade, figure per step).  GOTCHA there: the engine OPD grid
maps x along the ROW index and e5's `xGrid=(−1,0,0)` — overlaying
tiling geometry on OPD maps needs the affine centroid calibration in
`e5_pie.m:pupil_axes_` (transpose + mirror aware), not a hard-coded
±R/N pixel map.  Tests: tSegmentRx test_emit_apertures_and_rxpoly +
test_seg_apertures_hex.

### Optical-design reference & fixtures (Sprint 2A-ii builder)
The de-novo `macos.design.Telescope` builder (closed-form Cass / RC /
Gregorian / DK layout + conics) is backed by a self-checked reference
set in **`MACOS_resources/optical_design/`** — a shared root dir (beside
`rx_converter`/`segmirmaker`), NOT under mmacos, because it is
language-agnostic and the JSON fixtures are the cross-language parity
golden specs (`macos/PLAN_DESIGN_LAYER.md` §3).  From mmacos tests,
resolve it with a `tests/private/` path helper (mirror
`rx_fixture_path.m`).

- **Equations & families:** `optical_design/TELESCOPE_DESIGN_REFERENCE.md`
  (Schroeder closed forms — first-order layout, two-mirror master
  conditions, conic constants per family, Korsch TMA, freeform).
- **Agent guide (READ before editing any optical math):**
  `optical_design/OPTICAL_DESIGN_AGENT_GUIDE.md`.
- **Regression fixtures:** `optical_design/fixtures/*.json` —
  `telescope_design_fixtures.json` (5 two-mirror `(f,D,m,β)→R1,R2,K1,K2`
  rows), `tma_fixture.json` (Korsch TMA, conics solved to null
  S_I/II/III), `jwst_anchor.json` (real JWST M1/M2; M3+spacings TODO,
  do NOT fabricate).  `optical_design/seidel.py` is the paraxial+Seidel
  **oracle** (print-only; never writes fixtures).  `make_fixtures.py` /
  `make_tma_fixture.py` regenerate the JSON and self-validate.
- **Convention (FIXED — do not reinterpret):** `R>0` = concave /
  converging; two-mirror conics in the Schroeder `(m,β)` convention
  (`β` = back-focal-dist / f1).  If MACOS's internal radius/mag sign
  differs, write the translation layer explicitly and test it against
  the fixtures.
- **Hard gate:** after ANY change to conic-solving / layout /
  aberration code, run the fixtures.  A conic mismatch `>1e-6` or a
  nonzero S_I/II/III on the TMA row is **stop-and-fix** — never widen
  the tolerance or hand-edit a fixture; regenerate with the matching
  `make_*.py` and explain.
- **Coronagraph back-end (Sprint 3)** lives in the same shared dir:
  `optical_design/CORONAGRAPH_DESIGN_{RULES,AGENT_GUIDE}.md` +
  `coronagraph_layout.py`.  First-order LAYOUT only (NOT contrast/EFC —
  those are FALCO).  Two HARD rules: `quarter_talbot_sep_mm` is a
  reference scale, NEVER the DM separation; and units there are
  **mm + nm** (this telescope layer is **metres**) — chain only via the
  dimensionless F/#, convert lengths explicitly.

## Veneer parity with pymacos (audit 2026-06-12)

> **Superseded in part by the full 3-layer audit 2026-07-02:**
> `MACOS_resources/doc/mmacos_pymacos_parity.md` — mmacos is now a
> strict SUPERSET at the raw layer (5 api_mod routines have no pymacos
> wrapper: draw_rays_cmd/get, xps_cmd, ray_status_get, elt_z); both
> directions of user-surface gaps enumerated there with a re-run recipe.

**Engine-level parity is complete** — every implemented pymacos function
maps to a shared `macos_api_mod` routine mmacos also exposes as a raw
`mmacos('cmd',...)`.  Remaining gaps are convenience `+macos` veneers,
not capability.  First batch added 2026-06-12: `spot`, `fex` +
`get_xp`/`set_xp`, `get_elt_kc`/`set_elt_kc`/`get_elt_kr`/`set_elt_kr`.
Still raw-only (no veneer): grid-surface family (`elt_srf_grid_*`),
`src_size`/`src_csys`/`src_info`, surface-csys, `elt_zrn_type`/
`norm_rad`, `set_ray_info`, `elt_grp` query helpers.  Tracker:
`~/dev/macos/PLAN.md` §11.6.

Other engine wrappers added this cycle: `get_ray_status(N)` (per-category
RayStat counts; complements binary `get_ray_info`) backed by
`ray_status_get` in `macos_api_mod`.

## slsqplib link (Makefile)

`libsmacos.a` references `slsqp_` (via `design_slsqp_optim_mod`) but the
SLSQP objects live in their own archive `libslsqplib.a` (CMake target
`slsqplib`).  The Makefile links `$(SLSQP_LIB)` alongside `$(SMACOS_LIBS)`
(wildcard-guarded for older trees).  Without it any sls-dev/opt-dev
mmacos build fails with `undefined reference to slsqp_`.

## Sensitivities: dw/d(grid) segment example — RESOLVED 2026-06-29

The per-segment dW/d(grid-data) example now images correctly — each grid-figure
poke is localized to its segment (matches FreeForm).  The "spurious per-segment
piston" flagged 2026-06-28 was NOT a conjugate/reset artifact: it was the engine
**`GridSrf` (SrfType-9) bug** — GridSrf forwarded an all-zero grid frame to
FreeFormSrf, collapsing the grid index to the center pixel so the grid acted as a
pure piston (the figure was discarded).  Fixed in the macos engine (sls-dev
`03db580` / opt-dev `1b535a5`, threads the real grid frame; see
`macos/macos_f90/CLAUDE.md`).  GMI regression 6/6 bit-identical.

- **Rx** (`SegDemo3conic.in`, self-contained): PM-conic Reference=elt1, S1=elt2,
  6 GridData segments S2..S7=elts3..8, SecMir=9, FclPlaneReturn=10, ExitPupil=11
  (wf_elt), FclPlane=12, ApStop=0 0 0.  Exit pupil set via the FP-Return-before-
  EP-Return 2-pass pattern (`add_pupil`); `reset_xp_method='sxp'` (FEX collapses on
  SegDemo3 — a SEPARATE, real issue, unrelated to the piston).
- **GridSrfdx / figure-scale gotcha:** the bundled grid file `zern41em5z155em3.txt`
  (a steep z41, ~5 mm) is built for `GridSrfdx=0.01`.  At a tighter scale
  (0.0071/0.0063) its full extent lands on the segment and the trace DIVERGES —
  for BOTH FreeForm AND GridData (so it's figure-scale, not a GridData bug).  Use
  `GridSrfdx=0.01` (segment samples the figure's gentle center) or a gentler file.
  Flat segments are NOT acceptable (Dave: unusable in a real telescope) — the demo
  must carry a real figure.
- **GridFile resolves from CWD** (engine `GridInit`): run with cwd = the dir
  holding `zern41em5z155em3.txt` (`templates/20_segmentation/grid_surface/`) or co-locate it
  (deferred).  Else "File ... does not exist -sub GridInit" → flat nominal.

Plotting helpers (`sensitivities/`, GENERIC across all dw_d*_multi):
- **`plot_dw_channels`** — all-channels overview.
- **`plot_dw_per_element(out, 'center'|'multi', here, prefix)`** (NEW) — one page
  PER ELEMENT (= per segment for grid; per optic for dwdx/dwdz/dwdsurf),
  center-field AND multi-field, parula + zero-mask, no thresholding/caxis band-aids
  (retired — the dW is localized now).  Auto-detects the per-field cell
  (`per_field_dwd{x,z,s,g}`); wired into all the `run_dwd*_multi` examples.
- **`macos.gs_zernike_segment_basis`** — GS-orthonormalized Zernike basis over a
  segment's true (irregular) aperture, piston/tip/tilt projected out; one basis
  covers all clocked segments.  Trace to the PM Reference (can't trace a Segment).
- **`macos.segment_grid_basis`** (NEW 2026-06-30, sls-dev `245af94`) — the
  PER-SEGMENT generalization: steps EVERY grid segment (find_grid_elts), builds a
  bespoke Voronoi+hull mask + Z-mode stack `out.seg(s).B [N×N×K]` in each
  segment's OWN clocked (xData,yData) frame.  `zern_type` 'ansi' (engine
  ZerntoMon1/NormANSI, default) | 'noll'; `orthogonalize` (GS vs plain circular).
  Voronoi WEDGES are correct (`ApType=None` → nearest-centre footprint); for the
  congruent SegDemo3conic flower the per-segment bases are ~98% congruent (it
  matters for clipped EDGE segments).  See memory [[project-gridmat-generator]].
- **Per-segment influence path**: `grid_channels` / `dw_dgrid` / `dw_dgrid_multi`
  `'influence'` now accepts `[N×N×K]` (all elts), a `segment_grid_basis` struct
  (per-segment, keyed by iElt), OR a cell per grid elt — so `run_dwdgrid*` can use
  per-segment bases.  `macos.write_grid_file` writes a grid to the engine GridFile
  format (GridInit reads `DO j: READ(GridMat(i,j),i=1,N)` → file line j = column j;
  fprintf column-major → GridMat==M, no transpose; validated by load+trace).
  Example `examples/gen_segment_gridmat/` (driver + SegDemo3conic.in with
  GridFile=none + README).  **Do NOT collapse modes into the Rx** — that
  (modes×coefs in the engine) is a deferred bigger task.
- **`mmacos_setup.m`** (repo root, committed 245af94) — run once per MATLAB session
  to put `src` + the sensitivities helpers on the path; anchored to its own
  location (no hardcoded user paths).  Example drivers carry no addpath.

Two segmented dwdgrid worked examples shipped 2026-07-01 (sls-dev `c07ed65`):
`examples/run_dwdgrid_multi_multisegbasis` (PER-segment -- feeds the
`segment_grid_basis` struct as `dw_dgrid_multi` `'influence'`; verified 72 chan
/ 0.0% inter-seg support overlap / 92px centroid spread) and
`examples/run_dwdgrid_multi_singlesegbasis` (SINGLE shared basis via
`gs_zernike_segment_basis`).  Naming (Dave): `multi`=one-basis-per-segment,
`single`=one-shared-basis (both `*segbasis`).  Both segmented drivers also live
as generic library templates in `sensitivities/` (RX='' -> bundled default).
`plot_dw_per_element.m`, `gs_zernike_segment_basis.m` (doc-ref fixed off the
deleted `check_pmref.in`), the generic `run_dwdgrid_multi`, and the
`run_dwd{x,z,surf}_multi` examples are committed too (`c07ed65` + `8beb774`).
The experimental `run_dwd*_SegDemo` scratch dirs were DELETED; their unique
FreeForm/Zernike fixtures (`SegDemo3ff.in`, `SD3ff.in`, `SegDemo3zern.in` + the
gs bespoke driver) are preserved untracked in
`~/dev/MACOS_sandbox/segdemo_fixtures/`.  `gen_segment_gridmat` + `mmacos_setup.m`
shipped earlier (245af94).

## Polarization (PLAN_POLARIZATION)

Engine-side conventions, coating models, the Phase-2 Jones layer and the
Phase-3a vector chain all live in **`macos/macos_f90/CLAUDE.md`** — read
that before touching anything polarization-related; do not duplicate it
here.  mmacos-side facts only:

- Surface: `+macos/{polarization,vector_diffraction,coating,ray_field,
  jones_pupil,pol_maps,pol_zernike}.m`.  All ride codegen Path A except
  `ray_field`; `pol_maps`/`pol_zernike` are pure MATLAB (no engine call).
- **`pol_zernike` (Phase 2b, 2026-07-26)** — Zernike expansion of the
  Dvec/retvec maps into standard polarization-aberration terms.  Mode
  indices are the MACOS ANSI 1-based numbers (`MonZernModes=`); the
  evaluator is shared with `zernike_grid_basis` via
  **`+macos/private/ansi_zernike_eval.m`** so an influence-basis mode
  index and an aberration-report mode index cannot drift.  LEAST-SQUARES
  fit, not a projection — circular Zernikes are non-orthogonal on an
  obscured pupil.  Expected two-mirror answer: astig0 in s1, astig45 in
  s2, equal, everything else round-off ("polarization astigmatism").
- **UNITS TRAP in `tJonesPupil`:** `macos.coating` takes thickness in
  element **BaseUnits** (documented exception to the SI-metres veneer
  convention), and the class uses TWO fixtures with different BaseUnits —
  `Rx_Cass_FarField` is `m`, the Bench-emitted fold rig is `mm`.  One
  shared constant silently meant 200 um on the Cassegrain; gates still
  passed (any optically thick layer satisfies them) but the mmacos and
  pymacos Jones coefficients disagreed in the 8th digit.  Now split into
  `thkAl` / `thkAlBench`; with that, the two bindings agree to 11 digits.
- Tests: `tPolarization` (Phase 1 state/round-trip), `tJonesPupil`
  (Phase 2a/2b), `tVecChain` (Phase 3a Tranche 1) — all in `SUITE_FAST`.
- **`tests/Rx/` is a new fixture home.**  `rx_fixture_path` searches the
  shared pymacos corpus FIRST and `mmacos/tests/Rx/` second, so an
  mmacos-only prescription (currently `Rx_VecChain.in`) resolves by name
  with no caller change.  `Rx_VecChain.in` is duplicated into
  `pymacos/tests/Rx/` for the mirrored pytest — keep the two in sync.
- **Writing a new Rx fixture by hand: two traps.** (1) The file needs the
  trailing `nOutCord=`/`Tout=` **Output Coordinate System** block or
  `load_rx` fails with no diagnostic (the parser leaves `nElt=0` and the
  api's only check is `nElt>0`). (2) MACOS **merges consecutive elements
  that share a `PropType`** into ONE propagation, so a "two-leg" chain
  written as two `NFPlane` elements in a row is silently one leg — bracket
  each hop `NFPlane` → `Geometric` (the `Rx_Coro.in` idiom).
- **Validation report driver: `tools/pol_validation_report/`** (2026-07-26,
  PLAN worklist item 7).  `./make_polval.sh` re-runs every polarization
  validation case, writes `macos/docs/macos-manual/polval/media/*.png` +
  `generated/numbers.json`, then substitutes into the `polval/*.md.in`
  prose (which carries **no numeric literals**, only `@@TOKEN@@`).  Build
  with `make polval` / `polval-pdf` from `docs/macos-manual`;
  `polval-regen` is the same thing from the docs side.  Three guards:
  the renderer writes nothing unless every token resolves; the driver
  asserts 19 gate thresholds mirroring tPolarization/tJonesPupil/tVecChain
  and ABORTS on regression; `tools/check_polval.py` (prerequisite of
  `make polval`) rejects a stale template, a figure newer than the
  numbers, or a surviving placeholder.  Gates this box can't run
  (pymacos/ifx, GMI, the historical pre-fix engine) live in
  `external.json` with command + capture date.  **One model size (128),
  one MATLAB session** — the `macos_init_all()` heap bug applies here
  like everywhere else.  Adding a phase's evidence section: see the
  tool's README.
- **Pre-existing, unrelated:** `Rx_Coro_FPM.in` SIGSEGVs at model size 256
  (`macos.intensity` at any element).  Verified against the pre-Phase-3a
  engine — not a polarization bug.  Use `Rx_Coro.in` at 1024 (what the
  proper_compare suite does) for coronagraph-chain work.
- **Model size must be ≥ the Rx's `nGridpts` (Dave, 2026-07-27).**
  `Rx_Coro.in` declares `nGridpts=511`, so it needs model **≥ 512**.  Run it
  smaller and the engine prints `Too many grid points. Resetting npts to N`
  and carries on — but that path is not safe: at 128 `macos.intensity(21)`
  SIGSEGVs in roughly a third of runs (same registers every time) and
  `trace`-then-`intensity` crashed 3/3.  At 512 and 1024 repeat runs are
  bit-identical.  There is an `MREset` CLI command (128..8192) for changing
  model size; it is **not** exposed in `macos_api_mod`, so from mmacos the
  lever is `macos.init(N)`.  An earlier PLAN note claiming `Rx_Coro` runs at
  128 is wrong and has been corrected — it appears to work, then doesn't.
- **FIXED 2026-07-27 — the odd-mirror `r_p` sign defect** (macos `cb29ea5`
  Reflector + `25c4386` Refractor; resources `9bc2029` fold-gate
  de-circularization + `fc2e22e` the new gates).  `Reflector`'s
  reflected-p̂ basis disagreed with its Fresnel `r_p` sign, so ONE on-axis
  mirror turned x-polarized light into a 50/50 x/y mixture; a mirror PAIR
  cancelled it EXACTLY (an involution) and the error is unitary, which is
  why `tJonesPupil` (2-mirror Cassegrain), `tVecChain` (no mirrors) and the
  coated 45° fold gate all passed.  Odd-mirror polarized results are
  trustworthy again.  Two lasting lessons: **an "analytic" reference
  transcribed from the engine's own expression is circular in exactly the
  sign it should check** (the fold gate's `RPa`, and the same staleness in
  the polval driver, which its gate guard caught), and **fixture parity
  matters** — every fixture in the suite had an even mirror count or none.
  New gates: `tPolarization/test_odd_mirror_crosspol_{pec_analytic,
  rho2_law}` — the whole PEC single-reflection Jones against a Born&Wolf
  closed form, with AOI and azimuth taken from RAY DIRECTIONS so no
  pupil-grid mapping is assumed (median 2.1e-15).  Reproducer:
  `tools/pol_sp_sign_probe/probe_sp_sign.m`.  Packet (with the closeout
  entry): `macos/REVIEW_POL_SP_SIGN_2026-07-27.md`; engine detail in
  `macos/macos_f90/CLAUDE.md`; report evidence in `polval/50_sp_sign`.
  **Watch out when reading `ray_field`:** `RayE`/`RayDir` are the CURRENT
  trace state, not a per-element history — `trace(e)` then `ray_field(e)`,
  or you get the state at whatever element you last traced to.
- **CLOSED 2026-07-28 (macos `a5e4288`)** — coated and uncoated `Refractor`
  transmission used DIFFERENT amplitude normalizations (the coated branch
  omitted the radiometric `sqrt(n2 cos02/(n1 cos01))` factor): a coated lens
  under-transmitted by ~18% in amplitude vs the same surface uncoated
  (0.8164965809 = 1/sqrt(1.5) exactly with an index-matched layer; 1/1.5 in
  INTENSITY at the detector plane).  Fixed by one factor applied ONCE after
  the Airy recursion; the branch now has `tPolRadiometric` (13 tests,
  SUITE_FAST) against the Abeles characteristic matrix, on new fixtures
  `Rx_Refract.in` / `Rx_Refract45.in`.  Binding-side things worth knowing:
  the 45° fixture tilts the ELEMENT (unlike `Rx_PolElt_Tilt.in`, which
  tilts the BEAM — a refractor's Fresnel physics depends on the ray-normal
  angle, which element tilt does change, whereas a straight-through
  polarizer's does not), and **Macleod's `2*eta0/(eta0*B+C)` is the
  TANGENTIAL amplitude coefficient**, larger than the ordinary Fresnel
  `t_p` by `cos_sub/cos_inc` (1.2472 at 45° into n=1.5) — convert before
  comparing to a measured `E_out/E_in`, or a correct engine looks broken.
  Engine detail + scope: `macos/macos_f90/CLAUDE.md`; report evidence in
  `polval/80_radiometric`.
- **`complex_field(..., 'reset_trace', false)` is bit-identical and ~100×
  faster** for the 2nd/3rd component plane at the same element (0.01 s vs
  0.83 s at model 512) — reading Ex/Ey/Ez costs ONE propagation, not three.
- **Phase 2c (`pol_contrast_floor`, 2026-07-27)** — co/cross/longitudinal
  split at the DETECTOR on the engine's component planes; analyzer = dominant
  eigenvector of the pupil coherency matrix (referenced to the mean OUTPUT
  state, never the input).  Tests `tPolContrast` (model **256**,
  SUITE_FREEFORM) + `tPolContrastCoro` (model **512**, its own
  `SUITE_POL_512` batch); `./run_mmacos_tests.sh polfloor` runs both.
  Three things to know:
  (1) **The coherency matrix's conjugation order is a trap.**  In MATLAB `'`
  conjugates its LEFT operand, so `C_12 = Σ E_1 conj(E_2)` is `Ey'*Ex`, NOT
  `Ex'*Ey`.  Backwards builds `conj(C)`, whose dominant eigenvector is the
  CONJUGATE analyzer — identical for any LINEAR input state and exactly
  ORTHOGONAL for a circular one (cross/co 1.4e-6 → 7.1e+05).  The circular
  input state in `tPolContrast` exists to catch exactly this.
  (2) **Tranche 1 caps what the floor can see.**  The component planes are
  seeded from `RayE` at the FIRST physical-optics leg and thereafter only get
  a common scalar phase, so a polarizing surface after that leg never reaches
  the grid.  `Rx_Cass_FarField` (mirrors, then one far-field hop) carries the
  full train; `Rx_Coro` carries 0.84 of the ray-level cross-pol bare, 0.57
  coated, and reports the coating sensitivity with the WRONG SIGN.  The
  function measures this itself (`.scope`) and warns.
  (3) **A coating can be overwritten but never cleared** (`coat_set` takes
  ≥1 layer), so every set in a `'coatings'` sweep must cover the same
  elements and the sweep leaves the last set applied — `load_rx` to reset.
- **Phase 3 elements (`polarizer` / `waveplate` / `elt_jones`, 2026-07-27)** —
  `TrPolarizer` (EltID 15, finished from a name-table-only stub) and the new
  `WavePlate` (EltID 18).  Engine detail + all four conventions live in
  `macos_f90/CLAUDE.md`; mmacos-side facts:
  (1) **`macos.polarizer`/`macos.waveplate` follow the OVERLOADED
  query-when-no-opts form** (`coating.m`'s shape), not the `get_`/`set_`
  split the naming convention prescribes — consistency inside the
  polarization family beat the general rule, since `polarization`,
  `vector_diffraction` and `coating` are all overloaded.
  (2) **Axis storage is deliberately asymmetric**: the API stores the axis
  as given (a query returns what you wrote), the Rx parser UNITIZES on load
  (matching `psiElt=`).  So a non-unit axis comes back normalized after a
  save/reload round trip — `tPolElement/test_save_roundtrip` pins it.
  Retardance is in WAVES at the current wavelength on both sides but stored
  physically, so a query after `set_src_wvl` legitimately returns a
  different number.
  (3) **Fixtures `tests/Rx/Rx_PolElt.in` + `Rx_PolElt_Ref.in` are a PAIR** —
  the second is the geometric twin (Reference surfaces in place of the
  polarizing elements) that makes "polarization-off is bit-identical" a test
  rather than an inspection.  Any geometric edit to one belongs in the other.
  (4) Element order in the fixture is load-bearing: all four polarizing
  elements precede the single physical-optics leg, or the Tranche-1 seed
  would miss them.
  (5) Tests: `tPolElement` (27, SUITE_FAST).
  (6) **The off-normal axis convention is SETTLED (2026-07-27): project the
  MATERIAL axis.**  For a waveplate that IS the declared (fast) axis; for a
  polarizer it is the ABSORBING direction `psi x PolAxis`, projected into
  the ray's transverse plane, extinguished, with its orthogonal partner
  transmitted.  `PolAxis=` still declares the pass axis.  The alternative
  (project the pass axis) is Fainman & Shamir's model and was what shipped
  first; Korger et al., *Opt. Express* **21**, 27032 (2013) Eq. (5)–(6)
  measured that a real tilted polarizer follows the material-axis one.
  Three mmacos-side facts: normal incidence is **bit-identical** across the
  flip (`test_normal_incidence_unchanged_by_the_material_flip` pins the
  pre-flip values literally); a polarizer's declared axis is now taken
  modulo its component along the element normal, and an axis parallel to
  the normal extinguishes; and the gates live on a THIRD fixture,
  `tests/Rx/Rx_PolElt_Tilt.in`, which tilts the **beam** — tilting the
  element does nothing to a collimated on-axis bundle, which is why the
  original packet could only measure the ambiguity in MATLAB.  Section F
  gates both dispatch chains, the grid side via the crossed-analyzer null,
  which sits 7.11° apart under the two rules (9.1e-33 vs 1.53e-2 of
  relative detector power).  Evidence: polval §6.7.
- **polval driver is now split per model size** (2026-07-27): 128 / 256 / 512,
  one `matlab -batch` each (the `macos_init_all()` heap bug), each writing
  `generated/parts/numbers_<size>.json`, merged by `merge_numbers.py`.  Adding
  a case at a new size = a new branch in `run_pol_validation`'s switch, its own
  block in `gate_limits()`, and a size in `make_polval.sh`'s `MODELS`.

## Commit hazard: `git add -A` under `pymacos/tests`

`pymacos/tests/` carries LARGE untracked artifact trees that are **not**
gitignored: `proper_compare/results_cycle4/` and `results_cycle5/` (PROPER
comparison `.npy` dumps, 56 MiB each), `Rx/IntLog.txt`, and the
`sensitivities/*/`+`sensitivities/results/` PNG outputs.  Only
`results/phase<N>/` is ignored.

`git add -A pymacos/tests` therefore stages ~740 MiB of someone's working
tree.  This happened 2026-07-26 and was caught only because the push to
`nasa-jpl/MACOS_resources` STALLED -- diagnosed as payload, not network
(`ssh -T git@github.com` and `git ls-remote` were both instant).  Fixed by
`git reset --soft` + staging the intended files BY EXPLICIT PATH; payload
741 MiB -> 1.09 MiB, and the commit was still unpushed so no history
damage.

**Rule: stage by explicit path in these repos, never `-A` on a directory
you did not create.**  Before any push, sanity-check the payload:

```sh
git rev-list --objects origin/<branch>..HEAD \
  | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
  | awk '$1=="blob"{s+=$2} END{printf "%.2f MiB\n", s/1048576}'
```

## Key files

| File | Role |
|---|---|
| `src/mmacos_mex.F` | Hand-written mex helpers + dispatcher, ~600 LOC |
| `src/mmacos_gen.F` | Auto-generated mex helpers + `gen_dispatch` (90 cmds) |
| `src/gen_mex_wrappers.py` | Codegen script — re-run on api signature change |
| `src/+macos/` | Function-package user surface + `+design/` + `+channels/` |
| `tests/` | matlab.unittest suite; `tests/README.md` maps class→suite→coverage |
| `tests/private/rx_fixture_path.m` | Shared Rx-corpus locator |
| `examples/design/` | Design-layer runnable examples (sensitivities, align) |
| `run_mmacos_tests.sh` | Bash entrypoint; `fast` / `masks` / `proper` / `<Class>` |
| `Makefile` | GMI-style build; links libsmacos + slsqplib (+ fits) |
| `~/dev/macos/macos_f90/macos_api_mod.F90` | Shared backbone (in libsmacos.a) |
