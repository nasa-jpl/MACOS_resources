# ns_griddata — how `ZrnGrData` composes: Zernike + GridData on an NSReflector

Answers the iris double-pass "ZGD" questions (Luis).  Segments **A1/A3/A5** are
`Element=NSReflector` surfaces hit twice (iElt 17/19/21 on pass 1, 35/37/39 on
pass 2).  Each carries **both** a Zernike figure (`ZernCoef`, ANSI modes
14 15 19 38) and a grid figure (`GridFile=zern41em5z155em3.txt`,
`GridSrfdx=1.1e-2`).  Five prescriptions differ **only** in the `Surface=` type
of those six element entries:

| File | Surface= | Figure applied |
|---|---|---|
| `iris_dp_conic.in` | `Conic` | none (baseline) |
| `iris_dp_Zern.in` | `Zernike` | Zernike only (grid lines parsed, ignored) |
| `iris_dp_GD.in` | `GridData` | grid only (`ZernCoef` parsed, ignored) |
| `iris_dp_ZGD_flat.in` | `ZrnGrData` | Zernike + FLAT grid (`flat.txt`) |
| `iris_dp_ZGD.in` | `ZrnGrData` | Zernike + grid (both) |

Each variant's ExitPupil (elt 55) OPD minus the conic baseline isolates that
surface type's figure contribution.  The driver then checks:

1. **Superposition** — `dZGD == dZern + dGD`: a `ZrnGrData` surface applies the
   Zernike leg and the grid leg independently and adds them.
2. **Flat grid is inert** — `ZGD_flat == Zern`: an all-zero grid contributes
   nothing, and `ZrnGrData`'s Zernike leg is identical to `Surface=Zernike`.

Run (after `mmacos_setup`):

```matlab
mmacos_setup
run ns_griddata
```

Outputs `ns_griddata_decomposition.png` (2×3: dZern / dGD / dZGD / dZern+dGD /
superposition residual / flat-grid residual) and `ns_griddata.mat`.

## The reusable utility

`ns_griddata.m` calls **`macos.opd_psf`** (in `src/+macos/`) — a small, general
front-end for any `.in`: it loads the Rx, optionally cleans up the exit pupil
with FEX, traces, and returns/displays/saves the OPD (and, optionally, a PSF via
INTENSITY):

```matlab
macos.opd_psf('iris_dp_ZGD.in', 'wf_elt', 55, 'psf', true, ...
              'save_png', true, 'save_mat', true);
```

## What the original version of this example caught

A `GridFile=` line with an inline comment kept the trailing **tabs** before the
`%` — e.g. `GridFile=  zern41em5z155em3.txt<TAB><TAB>% flat.txt`.  The parser
stored `"zern41em5z155em3.txt<TAB><TAB>"`, `GridInit` reported *"does not
exist"*, zeroed the grid, and the figure was **silently dropped** (identical OPD
with the figure on or off).  Fixed in the engine (strip trailing tabs from
`GridFile=` in `msmacosio.inc` + `GridInit`); a clean/`SAVE`'d Rx works.

> **Note on the aperture stop:** the ExitPupil OPD needs a correct `ApStop=` in
> the Rx header.  A wrong/absent stop can push the source outside the usable
> aperture so every ray is obscured — then `opd` reports a `9.9999D+36` RMS
> (and, unguarded, can segfault).  Find the stop interactively with
> `STOP <iElt>`, but note that **`SAVE` does not currently round-trip `ApStop=`
> (or comments)** — a SAVE'd Rx comes back with no `ApStop=` and no comments —
> so add the `ApStop=` line to the Rx directly.  (Both the ApStop-SAVE gap and
> the opd-segfault-on-zero-rays are tracked engine TODOs.)

> **Engine note (non-sequential grids):** this example caught a second bug —
> `NSReflector` elements trace through the standard `Reflector` routine, but
> the non-sequential call sites (tracesub.F / propsub.F) passed the grid
> coordinate frame (`pData/xData/yData/zData`) indexed by the wrong element
> (`iElt`, the NS-group entry, instead of `imin`, the element actually hit).
> A frameless element's vectors are all zero, so every ray mapped to the
> grid's **center pixel** and the figure degenerated to a per-segment piston
> (measured: 2·bounces·cosθ·GridMat(128,128) exactly, ~0.3% residual from the
> cosθ variation).  Fixed by indexing the frame with `imin`.  The refractive
> NS path (`NSRefractor` routine) still carries a null grid frame — a tracked
> engine TODO (not exercised by iris, which is all-reflective).
>
> **GridSrfdx:** the grid's physical span is `(nGridMat−1) × GridSrfdx` in
> base units, centered on `pData`.  For a 256-grid to span a 280-diameter
> segment: `GridSrfdx = 280/255 ≈ 1.1`.  Too small a value leaves most of the
> footprint outside the grid (fh=0 there); the original 0.2 sampled only the
> central ±25 units.
