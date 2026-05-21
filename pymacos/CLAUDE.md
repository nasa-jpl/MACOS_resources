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

### Group sensitivities via macos GPERTURB

`GroupedRigidBodyChannel` dispatches the rigid-body group
perturbation through macos's `GPERTURB` (= `CPERTURB_GRP_DVR` in
funcsub.F) rather than synthesizing it in Python.  The factory
`grouped_rigid_body_channels(...)` builds one channel per
(group, DOF) and the driver consumes them inline with the per-
element channels.

Why GPERTURB and not Python-side linear superposition:
rigid camera motion produces an EP/FP coupling that cancels two
individually-huge column contributions (e.g. perturbing the
Return reference of the EP alone shifts the OPD reference sphere
by a huge amount; in the rigid motion that shift is cancelled by
the FP moving with the camera).  Linear superposition of per-
element columns can't reach that cancellation within numerical
precision, so Reference/Return members had to be silently
dropped from the synthesis -- a workaround that bias the answer
unless the group geometry is well behaved.  GPERTURB perturbs the
coupled members at once, so the cancellation falls out naturally.

How the channel manages EltGrp: macos's `EltGrp(0:N, iElt)` data
structure allows only one group per element.  Each channel
`apply()` snapshots `ref_elt`'s current EltGrp, writes the
desired members via `m.elt_grp(ref_elt, members)`, runs
`m.prb_grp(ref_elt, 6vec, glb_csys=False)` -> `SMACOS('GPERTURB',
...)`, and the original EltGrp is restored on the matching
`restore()`.  Overlapping groups (the same element in multiple
Python-side groups) work because each channel installs and
tears down its own EltGrp around the perturbation.

Optional FP follow-up for groups whose members include a
FocalPlane element (controlled by `fp_mode` -- propagated from
`--fp-mode` with `track` mapped to `auto`, since `track` has no
group analog):
- `none`: GPERTURB only; OPD is referenced to the rigidly-
  dragged EP element.  This is the "ground-based metrology"
  view -- the customer asks "what does this DOF do to the
  wavefront in a fixed coordinate frame?".  Reproduces the prior
  Python-side direct vpt/psi/rpt-setter rigid-body math to ~3
  significant figures on Rx_e5hex1.in's Grp[9-13] Rx.
- `sxp` (default in `auto` when FP is in the group): GPERTURB
  then `SXP` to recompute the EP pose+radius from the post-
  perturbation chief-ray geometry.  This is the "what the
  sensor measures" view -- a rigid camera tilt is correctly
  recovered as a small residual (= optical aberration internal
  to the assembly), not as a large EP-referenced tilt.
- `fex`, `srs`: same as the FocalPlaneChannel modes.

Rx_e5hex1.in Grp[9-13] Rx, delta=1e-8, model_size=128:
- `--fp-mode none` -> 1.207 (rigid camera tilt, EP-fixed view)
- `--fp-mode sxp`  -> 8.86e-4 (rigid camera tilt is internally
  cancelled when EP is re-derived from chief ray)

### Source perturbations

`SourceChannel` wraps `pymacos.macos.perturb_src` -- the iElt=0
branch of macos's `CPERTURB` (funcsub.F:41-92), routed through
`SMACOS('PERTURB', IARG(1)=0, DARG=6vec)`.  macos rotates
`ChfRayDir`, `xGrid`, `yGrid` (and `SegXGrid` if segmented) and
rigidly rotates `ChfRayPos` around `StopPos` -- so the chief ray
is automatically re-aimed through the stop after a source
rotation, no follow-up needed.

`--src-stop-mode {obj,elt,none}` (default `obj`) controls an
optional secondary re-aim through the stop on top of CPERTURB's
built-in one.  For Rx_e5hex1.in (`ApStop= 0 0 0`) it's a no-op
because CPERTURB already aimed through (0,0,0).  Set `none` when
cross-checking against group perturbations that do not include a
stop re-aim, otherwise the boundary conditions differ.

For a collimated source (zSource >= 1e10): Tx/Ty/Tz at the noise
floor (no OPD response, by translational invariance); Rx/Ry give
a pure wavefront tilt; Rz at the noise floor (rotation about the
beam axis is invariant).

### Cross-check: source vs all-optics-group

By Newtonian relativity, rotating the source by +δ is equivalent
to rotating every optical element rigidly by -δ.  The dw/dx
columns must satisfy this.  To get `cos = -1.000000` exactly:

- Group must include EVERY element 1..nElt (incl. FP, Refs,
  Returns).  Dropping bookkeeping surfaces breaks the rigid-body
  identity.
- Both source and group perturbations expressed in the SAME frame.
  `--group-coords global` (default) ensures the group rotation axes
  are global, matching the source's global ChfRayDir frame.  With
  `--group-coords local` the group rotates about `ref_elt`'s TElt
  axes, which for Rx_e5hex1.in is reflected relative to global ->
  cos collapses to -0.87 with a 2x2 reflection in the (Rx, Ry)
  cross-correlation.
- `--src-stop-mode none` so neither side double-re-aims the stop
  (CPERTURB's built-in aim is already consistent with GPERTURB
  pivoting at `RptElt(ref_elt) = StopPos`).
- `--fp-mode none` so the EP element is rigidly dragged with the
  group, not re-derived from chief ray.

Observed (Rx_e5hex1.in, model_size=128, delta=1e-8):
- Rx, Ry: cos = -1.000000, scale = -1.000000
- Rz: -1.0 (both at noise)
- Tx, Ty, Tz: uncorrelated noise (both ~1e-3, dominated by
  numerical precision; the physics predicts exactly 0)

### Stop re-enforcement: who needs it, who doesn't

`SourceChannel.apply` re-enforces the stop via `--src-stop-mode`
(default `obj` -> `m.stop_obj(0,0,0)`).  This is **redundant for
the source side**: macos's CPERTURB iElt=0 branch already rotates
`ChfRayPos` rigidly around `StopPos` (funcsub.F:73-86), so the
chief ray is intrinsically re-aimed.  The flag survives only as
"belt and suspenders" + escape hatch when the prescription has no
ApStop and a downstream tool wants to inject one.

`GroupedRigidBodyChannel.apply` re-enforces the stop via
`--group-stop-mode` (default `obj`).  GPERTURB **does not touch
ChfRayDir or ChfRayPos** -- it only changes element poses.  So for
the standard apply-from-nominal / measure / restore pattern the
re-aim is also a no-op (the source wasn't perturbed; chief ray is
still nominal; stop still trivially hit).  Verified empirically on
Rx_e5hex1.in by comparing `--group-stop-mode=obj` vs `=none` with
a deliberately off-StopPos pivot (group members reordered so
ref_elt=Elt 8, RptElt ~ (0, -5479, -21308)): both modes gave
identical cos=-1.000000 with residual 9.7e-7.

Where the re-aim DOES matter: stacked-channel workflows that
apply more than one channel before measuring (e.g. SourceChannel
followed by GroupedRigidBodyChannel as part of a larger custom
perturbation), or any future channel kind that does modify the
chief ray.  Keeping `obj` as the default makes the channel
robust to those compositions at the cost of one cheap
`stop_obj` call per apply().

### Direct dw/dx-matrix testing via predict_global_rigid_response

`channels.predict_global_rigid_response(macos, dwdx, col_map,
members, prb_global, pivot_global)` takes a per-element dw/dx
block + a global rigid-body 6-vector + a global pivot and
synthesizes the predicted OPD response WITHOUT calling macos
again.  Useful for:

- regression-testing the saved .mat file ("does the matrix know
  about this perturbation?");
- predicting the response of an arbitrary rigid body whose member
  list / pivot / 6-vector differ from what was measured;
- diagnosing per-element column bugs (a mismatch with the
  separately-measured GPERTURB column localizes the bad columns).

Math: for each member m with local-to-global rotation R_m and
RptElt offset p_m from the pivot P,
  theta_m_loc = R_m^T @ theta_global
  T_m_loc     = R_m^T @ (T_global + theta_global × (p_m - P))
  dw_pred    += Σ_j theta_m_loc[j] * dwdx[:, (m, R_j)]
              + Σ_j T_m_loc[j]      * dwdx[:, (m, T_j)]

**Unit gotcha:** the per-element translation columns are measured
with `perturb(translation_m=...)` -- SI metres.  But
`elt_rpt(m)` returns RptElt in the prescription's BaseUnits
(typically mm).  `predict_global_rigid_response` calls
`base_unit_to_metres()` and multiplies the offset by CBM so the
cross-product `θ × (p_m - P)` produces metres, matching the
column unit.  Without this conversion the prediction overshoots
by 1/CBM (~1000x for an mm-BaseUnits Rx).

**Coverage requirement:** `col_map` must hold columns for every
(member, dof_idx) the group touches.  By default
`rigid_body_channels` excludes Reference/Return surfaces;
`--include-non-optics` in dw_dx.py opts them in so the saved .mat
has full coverage.

Validation (Rx_e5hex1.in, model_size=128, delta=1e-8,
group=1..13 including the 2 Returns + FP, predicted at pivot
RptElt(Elt 1)=(0,0,0)):
- Rx prediction vs measured GPERTURB: relative residual 1.1e-5
- Ry prediction vs measured GPERTURB: relative residual 1.0e-5
- Rz, Tx/Ty/Tz: both sides at noise floor; differences are
  noise-vs-noise.

### Group synthesis weights in the .mat output

`group_synthesis_matrix(macos, members, dofs, pivot)` returns the
weight matrix `W` of shape (n_members * n_dof, 6) such that
    dwdx_group_pred[:, dof_g] = dwdx_perelt[:, member_dofs] @ W[:, dof_g]
Rows are ordered ELEMENT-MAJOR, DOF-MINOR; columns are 0..5 =
global (Rx, Ry, Rz, Tx, Ty, Tz).  Rotation rows are unitless
(rad/rad); translation rows from theta × offset are in metres.

`dw_dx.py` computes W per group (pivot = `RptElt(first member)`,
i.e. the GPERTURB default) and writes three cell arrays to the
.mat for MATLAB-side consistency-checking workflows:

- `group_W{k}`               -- (N_rows, 6) weight matrix
- `group_member_dof_idx{k}`  -- (N_rows, 1) 1-based column indices
                                 into `dwdx` for the corresponding
                                 (member, local_dof) entries
- `group_member_idx{k}`      -- (N_members, 1) member element IDs

MATLAB usage:
    cols = group_member_dof_idx{k};
    dwdx_pred = dwdx(:, cols) * group_W{k};        % (Nw, 6)
    dwdx_meas = dwdx(:, find_group_columns(k));    % from channel_names
    residual  = dwdx_pred - dwdx_meas;             % near machine noise
                                                   % on physically-
                                                   % well-conditioned DOFs

Useful for diagnosing per-element column bugs (mismatch in one
W^T * dwdx_perelt row vs the measured group column localizes the
broken member), and for predicting the response of arbitrary
rigid-body perturbations without re-running pymacos.

### dw/dx noise floor: where it comes from (and what doesn't help)

For a collimated source on Rx_e5hex1.in at delta=1e-8, the
noise-floor "signal" on a DOF that should be physically zero
(Src Tx/Ty/Tz, Grp[all] Tx/Ty/Tz) sits around **~1e-3 OPD/m**
(equivalently ~1 nm OPD response per 1 mm rigid-body translation).
Sufficient for HWO at typical perturbation deltas; this section
documents what governs it so it's not relitigated.

**Tightening Brent solver TOL in surfsub.F won't help.**  The
Brent convergence test uses
    `TOL1 = 2*EPS*ABS(B) + 0.5*TOL`
where `EPS = εmach ≈ 2.22e-16` and B is the ray path length being
solved for.  For typical optics with B ~ 1000 mm, the
`2*EPS*|B| ~ 4e-13 mm` floor dominates.  SFFZPSolve already
passes TOL=2e-15 (line 2089), AZPSolve at one call site uses
1e-14 (line 918) and at the other 1e-16 (line 1184, the latter
Dave-tightened in 2016) -- but the 0.5*TOL contribution is
~1e-15 or less, so the EPS floor pins TOL1 the same way regardless.
Every ray-surface intersection in a typical macos trace is
already solved to ~4 femtometers of L precision.

GSZPSolve and UDSZPSolve (legacy ZrnGridData / User-Defined
surfaces, surfsub.F:1309 + 2949) DO use looser `tol=1d-10`, but
neither shows up in Rx_e5hex1.in's trace -- FreeForm uses
SFFZPSolve, not GSZPSolve.

**The actual noise floor is in the OPD assembly downstream of
the trace** -- the differential between chief-ray and grid-ray
cumulative path lengths (each ~1e5 mm), or the reference-sphere
fit at the EP, or central-difference `(w_+ - w_-)/(2*delta)`
where each `w_±` is itself a sum of these subtractions.  Each
w vector is precise to ~1e-13 mm in absolute terms; dividing
by 2*delta = 2e-8 gives ~5e-6 OPD/m as a back-of-envelope
floor, which is within an order of magnitude of what we see.

Options if a future task needs to push below this:
1. **Richardson extrapolation** in `dw_dx.py` -- measure at
   ±d and ±d/2 to subtract out the leading numerical-error
   term.  ~2× measurement cost.
2. **Analytic differential** instead of finite-difference --
   compute path-length sensitivities once per element via
   adjoint methods.  Major refactor.
3. **Rewrite the macos OPD assembly** with Kahan summation or
   analytic differential at the cancellation hotspot.  Needs
   a profiling pass first to identify the lossy site.

None of these are needed for HWO at the current delta scale.

## Key files

| File | Role |
|------|------|
| `src/cmake/source/pymacos.f90` | `MODULE api` — ~3700 LOC, one Fortran wrapper per public Python entry.  USES smacos modules directly (Kinds, elt_mod, macos_mod, ...). |
| `src/pymacos/macos.py` | typed Python API — input validation, state globals, raises Exception on Fortran-side failure. |
| `src/pymacos/__init__.py` | Win DLL search-path shim; re-exports `macos`. |
| `tests/sensitivities/channels.py` | `ZernikeCoefChannel` + Rx-parse helpers.  Calls `m.modify()` in `apply/restore`. |
| `tests/sensitivities/jacobian.py` | `sensitivity_jacobian(channels, wf_func, delta)` — central/forward FD engine. |
| `tests/sensitivities/dw_dz_zernike.py` | dw/dz driver (Zernike-coef perturbations); writes `.mat` in m2v.m convention. |
| `tests/sensitivities/dw_dx.py` | dw/dx driver (rigid-body perturbations).  `RigidBodyChannel` + DOF-aware `FocalPlaneChannel` (track/sxp/srs/fex/none) + `GroupedRigidBodyChannel` (macos GPERTURB, global or local coords) + `SourceChannel` (macos `PERTURB` iElt=0); `--include-source`, `--include-non-optics`, `--group-coords`, `--update-ep` opt-ins. |
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
