# OPD conventions — sign, reference, and array orientation (engine, CLI, mmacos, pymacos)

_Measured 2026-08-07 on `CassWithExitPupil.in` (65-point grid, defocus +
secondary-tilt probes) and `e5pie` (segmented).  Every claim below carries
its probe.  Engine at macos `dev` 443140a._

## Summary

- **One OPD computation serves every consumer.**  The CLI, mmacos, and
  pymacos return the **identical array** — same values, same indexing,
  same piston — verified to 3.5e-15 (round-off) on identical pokes.
- **Sign: a ray longer than the reference is positive.**  Unchanged
  since the original import.
- **Reference: the chief ray when it survives the trace; otherwise the
  bundle mean.**  On centrally-obscured or segmented pupils the chief
  typically does not survive (hole/gap at the pupil center), so
  **mean-removed piston is the practical norm** — measured on every
  deck tried.
- **The apparent mmacos "90° rotation" is display, not data**: the array
  is `OPD(i,j)` with **first index i = global X, second index j =
  global Y**; MATLAB's `imagesc(W)` draws the first index vertically
  (and downward), which is a transpose+flip — exactly a 90° rotation —
  of the CLI plot's x-right / y-up rendering.
- **The "×(−1)" is a convention choice, not an error**: macos reports
  optical path difference (longer = positive); interferometer-style
  wavefront maps and PROPER's `prop_add_phase` use the opposite sign.

## 1. Definition and sign

The engine computes, per ray ([tracesub.F:190-233](../../..//macos/macos_f90/tracesub.F)):

```
OPD(i,j) = CumRayL(ray) − Ref            [longer path ⇒ positive]
```

with `Ref` chosen in priority order:

1. **Chief-ray path length** — when `LUseChfRayIfOK` is set (the
   `LOAD` command sets it in every consumer) **and the chief ray
   traced OK**;
2. an Rx-declared `OPDRefRayLen`;
3. otherwise the **bundle mean** — `OPD = L − mean(L)` (mode 3 also
   removes the mean explicitly, so the returned map has zero mean).

The textbook definition (Born & Wolf; Welford; Mahajan, *Optical
Imaging and Aberrations* I) is chief-ray-referenced through a reference
sphere; sign conventions differ by community — Mahajan's W > 0 for a
longer ray path matches macos; interferometry practice (surface/
wavefront maps) and PROPER's `prop_add_phase` use the negative.  The
mean-referenced form is the same quantity with best-fit piston removed.

**Measured (defocus probe)**: focal plane moved +0.1 mm downstream —
marginal rays gain more path than the chief, and the map shows the
positive-rim bowl in all three consumers (mean-removed: rim +1.86e-7,
center −1.85e-7; deck units metres).  A ray shorter than the reference
is negative, exactly as expected.

**Practical piston**: on `CassWithExitPupil` (obscured), its
unobscured variant (M1 hole), and `e5pie` (segmented; loses exactly one
ray — the chief), the chief ray is lost and every consumer returns the
**mean-removed** map.  The chief-referenced branch engages only on
decks where the central ray survives (e.g. off-axis/unobstructed
forms).  The CLI's printed `Average OPD` is the pre-removal diagnostic
`DAvgL`, **not** the piston of the returned map.

## 2. Array orientation

Established by two tilts of the secondary (rotation vector is global
(θx,θy,θz); PERTURB order is rotations then translations):

| probe | physics | measured gradient |
|---|---|---|
| rotate about global X | OPD ramp along global **Y** | entirely along **second index j** |
| rotate about global Y | OPD ramp along global **X** | entirely along **first index i** |

So for all three consumers: **`OPD(i,j)`: i (row) = global X, j
(column) = global Y.**  pymacos returns the same layout (no transpose
at the f2py boundary).

Per-display behavior on the *same* array:

- **CLI plot** (`OPD <elt>` → giza): draws i → horizontal (+x right),
  j → vertical (+y up) — the natural physical view.  Its axes are
  labeled only with coordinate values, not "x"/"y".
- **MATLAB, naive `imagesc(W)`**: draws i (rows) vertically, downward
  — the CLI view transposed and flipped, which reads as a 90°
  rotation.  Correct recipe:
  `imagesc(xv, yv, W.'); axis xy;` (transpose, y-up).
- **numpy, naive `plt.imshow(W)`**: same transpose class.  Correct:
  `plt.imshow(W.T, origin='lower')`.

## 3. Veneer options (added 2026-08-07)

`macos.opd` and every `dw_d*` sensitivity driver (`dw_dx`, `dw_dgrid`,
`dw_dsurf`, `dw_dz_zernike`, and their `_multi` variants), plus
pymacos `opd()`, accept two options.  Defaults preserve the historical
behavior exactly:

- `'orient'` — `'raw'` (default): the engine array, (i,j) = (X,Y).
  `'xy'`: transposed so rows run along Y and columns along X; display
  with `imagesc(xv,yv,W); axis xy` (MATLAB) or
  `imshow(W, origin='lower')` (numpy) for the x-right/y-up view that
  matches the CLI plot.  In the `dw_d*` drivers the m2v index
  bookkeeping (`indx`/`indxall`) is remapped to the transposed grid;
  Jacobian row order is unchanged (each row keeps its physical pixel),
  so `macos.v2m(dw(:,k), out.indx)` unflattens correctly in either
  orientation.
- `'sign'` — `'opl'` (default): longer path positive.  `'wavefront'`:
  negated (interferometer-style wavefront error; the sign PROPER's
  `prop_add_phase` expects).  In the `dw_d*` drivers this negates the
  Jacobians and nominal wavefronts; centroid/spot outputs are not
  wavefronts and are untouched.

The applied choice is stamped in `out.opd_orient` / `out.opd_sign`.
Shared implementation: `+macos/private/apply_opd_convention.m`.
Validated: `xy`+`wavefront` equals `-raw.'` exactly (0.0 residual) on
`macos.opd` and on `dw_dz_zernike`/`dw_dgrid` columns.

## 4. Reconciling with wavefront-sign consumers

To use a macos OPD map where a *wavefront error* (interferometer
convention) or PROPER `prop_add_phase` input is expected, negate it:
`W_wavefront = −W_macos`.  This is the established `opd_sign_flip`
precedent from the pymacos↔PROPER validation campaign and the CTB
phase-factor export (`meta.opd_sign`).

## 5. Reproduction

All probes are two commands each; deck copies with `SaveOPDMap= Yes`
in the header dump the raw array to `Opd_macos.txt` (line j holds
`OPDMat(1:n, j)`).

- Defocus (sign/piston): `PERT <detector> GLOBAL 0,0,0 0,0,1e-4` →
  `OPD <detector>`.
- Orientation: `PERT <secondary> GLOBAL 1e-6,0,0 0,0,0` → `OPD <exit
  pupil>` (repeat with `0,1e-6,0`).
- mmacos/pymacos: `macos.perturb(elt,'rotation',[1e-6;0;0],'frame','global')`
  / `m.perturb(elt, rotation_rad=(1e-6,0,0), in_local_coords=False)`,
  then `trace` + `opd`.

## 6. Open items (flagged, not resolved here)

1. **Rigid-body PERT on `Element= Segment` is a silent no-op**
   (sharpened 2026-08-07): verified on `e5pie` AND `e2e_pie`, with and
   without a declared/`macos.stop` stop, global and local frames,
   rotations and translations, CLI and mmacos alike.  PERTURB *does*
   update the element data (VptElt verified moved), yet the trace is
   bit-identical; a raw `set_elt_vpt` of the same amount (which moves
   ONLY the vertex, breaking the element's internal consistency)
   produces a large global response.  Interpretation: the segmented
   source registration co-moves with the segment frames, cancelling
   self-consistent rigid motion exactly.  **Workaround (verified)**:
   perturb segments through the figure channel — FF-Zernike piston/
   tilt (`set_elt_ff_zrn_coef`) or grid maps (`elt_grid_add`,
   `dw_dgrid`) respond correctly; a 0.25 µm FF piston toward the
   source measures −419 nm on the wedge against the −2d(1−f) =
   −427 nm prediction (2% = incidence obliquity).  Root cause of the
   rigid-channel cancellation still needs an engine-level
   investigation.  (The `'elts'` filter of `dw_dz_zernike`/`dw_dgrid`
   also appears not to restrict the channel set — separate nit.)
2. The exact gate that keeps the chief-referenced branch dormant on
   these decks (lost chief vs. flag state) was not isolated; measured
   behavior is mean-mode in every tested configuration.
3. CLI PNG plotting: with a file device, a second plot in the same
   session reuses the open device — plots can land in the first file.
   Use one plot per session (or an explicit new device name) when
   exporting.
