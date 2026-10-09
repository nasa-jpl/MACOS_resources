# NOTE — folding an even-radial asphere into a Zernike surface (attempts + the fix)

CCMac, 2026-10-04.  Written for CC-Linux / whoever lands the exact emitter
hook (see the PLAN_DESIGN_LAYER "Sprint 6+ deferred" to-do and the
`BRIEF_to_dyson5.md` note that points here).

## Why this came up

Round-4 step 5 (freeform on the outer field) wanted to add NON-symmetric
Zernike terms **on top of** TO's stage-B aspheres, so the freeform solve starts
from the asphere-corrected centre (1.3 px) and only has to fix the edges.  But
`Telescope.build` emits a mirror as `Surface= Aspheric` **or** `Surface=
Zernike`, never both (`if hasAsph && ~hasFree`): the engine applies the
even-radial `AsphCoef` ONLY on `Surface= Aspheric` (SrfType 3); the Zernike (8)
and FreeForm (14) surfaces carry a SINGLE monomial field (`MonSrf`/`FreeFormSrf`
= conic + Mon).  So "conic + asph + Zernike on one mirror" requires expressing
the even-radial asphere as symmetric (m=0) Zernike content and merging it with
the freeform modes.

The asphere sag (surfsub.F `SAsphere`) is `fh = Σ_i asph(i)·h^(2i+2)`, h the
radial coordinate in metres — a finite polynomial in h², so it maps EXACTLY onto
the symmetric modes piston + defocus + spherical orders (n = 0,2,4,6,8 …).

## What I verified (correct)

- **Index set.**  Under MACOS ANSI/OSA ordering the m=0 1-based indices are
  **1, 5, 13, 25, 41** (piston, defocus, primary/secondary/tertiary spherical);
  Noll is 1, 4, 11, 22, 37.  (ANSI index 4 is astig45, 11 is secondary astig —
  NOT spherical; the emitter's "ANSI 4–11" aberration labels were Noll-style and
  misled step 5's first mode set.)  Confirmed via `macos.zernike_grid_basis`'s
  own doc and `src/+macos/private/ansi_zernike_eval.m`.
- **Piston matters.**  A constant sag term on a REFLECTOR axially displaces the
  vertex and MOVES rays — it is not a harmless OPD offset — so the fold must
  match ρ⁰ (include mode 1), not just ρ²…ρ^(2na+2).
- **`lMon` is honored.**  Emitting the fold with lMon = ap_r, 0.5·ap_r, 2·ap_r
  (projection + emit kept consistent) gives an IDENTICAL traced sag — the engine
  reads the emitted normalization radius for `Surface= Zernike`.
- **The projection is exact in the right basis.**  Both an analytic √(n+1)·R_n⁰
  solve and a least-squares projection onto `macos.zernike_grid_basis` (the
  engine-exact evaluator) give the SAME coefficients — so the coefficients are
  the correct `NormANSI` (RMS-normalized) values.

## The wall (the ~2× I could not close with ZernCoef)

A/B round-trip = same asphere traced two ways (ray positions at the FP), on the
real stage-B 1.5k section, M1 asphere `[0.0016, -1.21]` (ap_r 0.1007, beam
spread 0.57 mm):

| fold emitted as | coef source | max\|A−B\| | note |
|---|---|---|---|
| `ZernType= ANSI` (unnormalized, =1) | √(n+1)-normalized | 0.97 mm | ~0.71× too weak |
| `ZernType= NormANSI` (=4) | √(n+1)-normalized (grid-basis) | 7.4 mm | **~2.07× too strong** |
| `ZernType= NormANSI`, coef × 0.5 | " | 0.34 mm | ~1.03× — a factor ≈2, plus ~10% per-mode residual |

So the `Surface= Zernike` **`ZernCoef`** path applies a ~2× scale (plus a
small per-mode residual) relative to the `MonZern` convention that
`zernike_grid_basis` is gated against.  Why it stayed hidden until now:
`optimize_freeform` / e5mono round-trip because they **solve** `ZernCoef`
self-consistently (relative); this fold is the first **absolute** asphere→
`ZernCoef` conversion, which is what exposes the factor.  I did not pin the
exact factor+residual (it is genuine engine-side `ZerntoMon` convention work).

## The fix (to land later)

Emit the freeform via the **`FreeForm` (SrfType 14) `MonZern` channel**
(`MonZernType= NormANSI`, `nMonZernCoef`/`MonZernModes`/`MonZernCoef`), NOT
`Surface= Zernike` / `ZernCoef`.  `macos.zernike_grid_basis` is documented and
gated (`tRunCompare`) to match `MonZern` EXACTLY ("a grid poke and the matching
MonZern coefficient produce the identical sag"), so the fold becomes exact by
construction.  This also means `optimize_freeform`'s CALIB `OptZern` DOF must
target the `MonZern` channel on a FreeForm surface rather than `ZernCoef` on a
Zernike surface.  No engine change — the FreeForm surface and `MonZern` keywords
already exist; it is a `Telescope.build` emit change plus an `optimize_freeform`
DOF-channel change, both gated by the A/B round-trip below.

## What is in the tree now (local, unpushed)

- `Telescope.asph_to_zern_` — the projection helper (correct, MonZern-convention
  coefficients via `zernike_grid_basis`).  KEEP; reusable by the MonZern emit.
- `ansi_zernike_eval` `norm_rms_ansi_` — extended from a mode-15 table to the
  analytic √(n+1)/√(2(n+1)) RMS formula (bit-identical for 1–15, needed for
  mode 25).  KEEP (clean improvement; existing gates stay green).
- `Telescope.build` combined (hasAsph && hasFree) emit path — **made to ERROR**
  (informative, pointing here) rather than emit the ~2×-wrong `ZernCoef` deck,
  until the MonZern emit lands.  The A/B harness to gate it is
  `/tmp/ab_fold.m` (reproduced in the gate `tTmaAsphZernFold` to be added with
  the fix).

Step 5 proceeds meanwhile via route **B**: a self-consistent `optimize_freeform`
solve over the correct symmetric modes {5, 13, 25} + non-symmetric from the
conic base — no absolute conversion, no `ZernCoef` convention dependence.
