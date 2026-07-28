# pol_external_anchor — protected-metal polarization, anchored to publication

External anchor for the thin-film polarization machinery, closing the
worklist item Fable opened in `REVIEW_POL_2C_2026-07-27.md`:

> **§3 last paragraph — the 151× stays model-relative; correct instinct.**
> The MgF₂/Al number additionally stacks the thin-film recursion through a
> quarter-wave overcoat — that is the part wanting an external anchor
> before it appears in any budget document.

Everything the polarization work had reported for a coated mirror was
gated against *our own* analytics:

| existing gate | what it actually covers |
|---|---|
| `tJonesPupil` Fresnel gate | an optically **thick single layer** — i.e. a bare interface |
| `tPolRadiometric` | the Abeles matrix in **transmission**, uncoated↔coated normalization |

Neither exercises a real **dielectric-on-metal** stack, which is what a
protected mirror is and what the 151× claim rests on.

## Method — reproduce the publication's own configuration

The point is to isolate the *machinery* from *index-table disagreement*.
Aluminum index tables genuinely disagree with each other; van Harten et al.
say so themselves ("the values of k are widely varying throughout the
literature") and **fit** k rather than adopting a table. A mismatch traceable
to index tables is not a machinery error, and conflating the two would make
the anchor meaningless.

So: drive the engine with **their** indices, **their** film thickness,
**their** wavelengths, **their** incidence angles, and compare curve-on-curve
against **their** model. `macos.coating` takes arbitrary n, κ and physical
thickness, so this needs no engine change and no fixture surgery.

Two comparisons, reported separately because they answer different questions:

- **(a) machinery check** — engine vs publication at the publication's own
  inputs. Any disagreement here is *ours*. Tolerance stated from the
  publication's own error bars.
- **(b) context check** — our 632.8 nm / 110 nm-MgF₂ configuration against
  the nearest published configurations. **Never a gate**: the nearest
  published protected-Al work uses a different overcoat (Al₂O₃, ~4 nm) at
  different wavelengths, so a numerical difference is expected and is not
  evidence about the engine.

## Result

**(a) Machinery: anchored.** Over the publication's four wavelengths and six
angles inside their measured range, compared per ray at each ray's own
incidence cosine:

| quantity | worst deviation | their stated accuracy |
|---|---|---|
| diattenuation ([1,2] Mueller element) | 2.828e-14 | ±0.01 |
| retardance, same Mueller units | 4.937e-14 | ±0.01 |

Non-vacuous: omitting the 4.12 nm oxide, or substituting the historical 50 nm
value, both exceed their accuracy — which independently reproduces the
paper's own central claim.

**(b) The finding.** The Phase-2c "151×" is *arithmetically correct* — the
independent analytic gives 6.35, the engine's own Jones pupil 6.42, and
`pol_contrast_floor` 5.42 for MgF₂/Al ÷ bare Al — but the configuration is
**mislabelled**, and the design rule drawn from it has the **wrong sign**.
`Rx_Cass_FarField` runs at `Wavelen = 1.0E-06` m while the coating constants
are commented "Al at 632.8 nm" and "~quarter wave at 632.8 nm"; at 1 µm the
110 nm MgF₂ film is **0.607** quarter-waves. The overcoat trade *reverses*
across the quarter-wave condition — the same film gives 0.18× at 632.8 nm,
6.35× at 1 µm, and a true quarter wave (181.2 nm at 1 µm) gives **0.0157×**,
i.e. a real protected-Al overcoat *suppresses* the floor by ~1.8 decades
rather than costing a decade. No engine change follows.

## The source

G. van Harten, F. Snik & C. U. Keller, *"Polarization properties of real
aluminum mirrors I. Influence of the aluminum oxide layer"*, PASP **121**,
377–383 (2009), doi:10.1086/599043; preprint arXiv:0903.2740v1.

Chosen because it is the rare paper with **both** halves: a full Mueller
matrix measured by ellipsometry over a swept incidence angle at several
wavelengths, **and** every model input stated numerically.

| their input | value | provenance |
|---|---|---|
| Al₂O₃ index n_f | 1.61 / 1.61 / 1.60 / 1.60 | Table 2, ±0.01, Eriksson et al. (1981) |
| Al index ñ (real) | 0.769 / 0.958 / 1.200 / 1.470 | Table 2, ±0.01, Lide (2008), interpolated |
| Al index k (imag) | 5.88 / 6.30 / 6.85 / 7.33 | Table 2 "Fit 1" — fitted, with the oxide in the model |
| oxide thickness | 4.12 ± 0.08 nm | abstract + Sec. 4, long-term value |
| Al film | 220 ± 10 nm on glass | Sec. 2 (they *model* it as semi-infinite) |
| λ | 500, 550, 600, 650 nm | Sec. 2, 10 nm bandpass |
| AOI | 14 angles, 6–70° | Sec. 2 |
| **accuracy** | **±0.01 per normalized Mueller element** | Sec. 2 — the gate tolerance |

Their model (their Eqs 1–6) is the Macleod / Born & Wolf characteristic-matrix
form. `vh_thinfilm.m` implements it **from the textbook/publication form,
never transcribed from `elemsub.F`** — an "analytic" copied out of the engine
is circular in exactly the coefficient it should check, which is how the 2022
r_p sign defect survived every gate for four years.

## Two conventions that had to be settled, not assumed

1. **Index sign.** The paper uses `N = n − ik`, k ≥ 0 = loss — the *same*
   convention MACOS stores (`DCMPLX(n,−κ)`), so no translation is needed.
   This is load-bearing, not cosmetic: the paper reports that the opposite
   sign made their fitted oxide come out at ~50 nm instead of ~4 nm.

2. **p̂ frame.** The paper's Mueller matrix must reduce to `diag(1,1,−1,−1)`
   at normal incidence (a mirror flips handedness), which forces
   ε_p − ε_s → 0 as θ → 0: the **fixed transverse** p̂. MACOS uses the
   **ray-following** p̂ (perfect conductor: RS = −1, RP = +1 at normal
   incidence). So `Δ_engine = Δ_paper + π`. The gate **measures** this
   bridge rather than trusting it.

A third had to be settled by evidence: the paper's Eqs (5)–(6) print in an
order PDF text extraction scrambles, so which of η = N cos θ / η = N / cos θ
belongs to s was ambiguous. Settled by the *sign* of the [1,2] Mueller
element — it must be positive for a metal (R_s > R_p), and their Fig. 1a
plots it on a 0.00–0.15 axis. The Macleod-standard assignment
(η_s = N cos θ) gives 0.087–0.115 at 70° across their wavelengths; the
swapped one gives the wrong sign.

## Frame-free measurement (`vh_measure.m`)

The obvious construction — tJonesPupil's Fresnel gate — divides out the input
state's projection on the s and p axes, which requires knowing the engine's
per-ray **launch frame**, in practice a hard-coded `xGrid`. That is safe at
45° and quietly wrong elsewhere.

Measured cost of getting it wrong: diattenuation came out nearly **flat in
AOI** (−3.1e−3 at 2° vs −4.7e−3 at 45°) where an isotropic surface must give
D ∝ θ² — the same "flat where physics demands a power law" signature that
exposed the r_p sign defect. Only the 45° point agreed, which is exactly the
one angle the existing gate runs at.

So instead: trace **two** orthogonal input states and build the 2×2 map `M`
from the (unknown) input frame to the (s, p_r) output frame. For an isotropic
surface the physics is `diag(r_s, r_p)` and the unknown frame is a rotation
`R(φ)`:

```
M = diag(r_s, r_p) R(φ) = [ r_s cosφ   -r_s sinφ ;
                            r_p sinφ    r_p cosφ ]
  =>   r_s/r_p = M11/M22 = -M12/M21
```

φ cancels identically, and the two independent estimates cross-check each
other (`m.consistency`) — a built-in validity guard rather than an assumption.

## Files

| file | role |
|---|---|
| `vh_data.m` | the published inputs, one copy, each with provenance |
| `vh_thinfilm.m` | the analytic — Macleod characteristic matrix, from the textbook |
| `vh_measure.m` | frame-free per-ray r_s/r_p from the engine |
| `vh_anchor.m` | the harness: (a) machinery sweep + (b) context check |
| `vh_diag.m` | side-by-side engine/analytic table, written to a file |
| `vh_cass_probe.m` | does the Phase-2c "151×" follow from the engine's own coefficients? (§8.3 of polval) |

Gate: `tests/tPolExternal.m`. Findings and the disposition of the 151×
claim: `macos/REVIEW_POL_EXTERNAL_2026-07-28.md`.

## Running

```
matlab -batch "mmacos_setup; addpath('tools/pol_external_anchor'); vh_anchor"
matlab -batch "mmacos_setup; addpath('tools/pol_external_anchor'); vh_diag('/tmp/vh.txt')"
```

MATLAB's batch stdout is captured as a rolling tail in this environment, so
`vh_diag` writes to a file on purpose.
