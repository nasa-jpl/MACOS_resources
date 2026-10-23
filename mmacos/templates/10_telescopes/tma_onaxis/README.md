# `tma_onaxis/` — on-axis three-mirror anastigmat designer (Korsch / j18mono form)

A parameterized Korsch TMA — concave M1, **convex** secondary (convex by geometry:
it sits before the M1 focus, `KrElt=-|R|`), concave M3 behind M1 — with the three
j18mono features:

1. a **real intermediate focus BETWEEN M1 and M2** (the field-stop / metrology-
   injection plane);
2. a **slight off-axis bias** that tips the focal plane OUT of the M2→M3 beam
   (the j18 "slightly off-axis detector"); the M2 central obscuration remains —
   this is the *obscured* baseline, whose unobscured eccentric-pupil cousin is
   [`../tma_offaxis`](../tma_offaxis);
3. the **exit pupil after M3 is ASSESSED** (FEX), not constrained.

## `tma_onaxis.m` — the designer (bias sweep)

Pipeline: `macos.design.tma_layout` (closed-form Cassegrain feed for the
intermediate focus + M3 relay for the system f/#; **convex-secondary aware** — the
unfolded paraxial, not the n-flip) → `add_mirror(...,'convex',true)` (`seidel_seed`
returns the correct unfolded focus + a **K=0** sphere seed) → `optimize(conics)` →
diffraction-limited.

**The off-axis bias is a SWEEP.** It must clear the FP, but off-axis aberration
grows ~quadratically with it, so the script sweeps `BIAS_SWEEP_ARCMIN`, prints the
clearance-vs-WFE table, and **recommends + builds the LEAST bias that both clears
the FP and is diffraction-limited.** Example (D=1 m, f/1.5 primary, f/8 system):

| bias (′) | RMS WFE (waves) | FP out of beam |
|---|---|---|
| 1 | 0.0074 | YES |
| 2 | 0.0113 | YES |
| 3 | 0.0172 | YES |
| 4 | 0.0305 | YES |
| 6 | 0.0758 | YES |

→ recommended **1′** → **0.0074 waves** (diffraction-limited), M3 + FP clear (M2
central obscuration), exit pupil after M3 at z = 0.26·D, r = 0.147 m.

Knobs: aperture, primary f/#, system f/#, secondary magnification, intermediate-
focus location, M3-behind distance, the bias-sweep list. A FAST primary may leave
higher-order spherical after the 3 conics — raise the bias floor or set
`ASPHERE_TERMS > 0` (an even-radial M1 asphere polish) for the aggressive regime.

> The earlier "the n-flip M3 relay is hard to place in the aggressive j18 regime"
> caveat is **resolved** — `tma_layout` and `seidel_seed` are now convex-secondary
> aware (the unfolded paraxial). A *constrained, accessible* exit pupil + finite-
> pupil FSM fold is still the freeform driver's job (fix the radii for the EP,
> correct with Zernike departures that don't move the radii).

## `scale_j18.py` — scale the validated j18 design (alternative)

A uniform length-scaling of the **j18mono** prescription (preserves every angle,
conic constant, and the f/# — only lengths scale): `python3 scale_j18.py 1000` →
`j18_scaled.in` at D = 1 m. Useful when you want j18's exact conics rather than a
from-scratch solve. (Verified D=1 m → 0.0018 waves at 2.3 µm.)
