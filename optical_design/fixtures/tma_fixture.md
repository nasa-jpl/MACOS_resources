# Three-Mirror Anastigmat (Korsch TMA) Regression Fixture

Companion to `TELESCOPE_DESIGN_REFERENCE.md` (Section 6) and `telescope_design_fixtures.md`.
Two artifacts here:

1. **`tma_fixture.json`** — a *self-consistent synthetic* Korsch TMA whose three conics
   were **solved** to null the third-order spherical, coma, and astigmatism sums for a
   fixed first-order layout. Residuals are machine-zero **by construction**, so it is an
   exact regression target and a unit test of the three-mirror aberration machinery.
2. **`jwst_anchor.json`** — **real published JWST** values (M1, M2, EFL, f/#) for an
   independent trace-only check, with M3 + spacings flagged TODO (see below).

Generated and self-checked by `make_tma_fixture.py`, which uses the paraxial+Seidel
engine in `seidel.py`. The engine is **validated on the two-mirror cases you already
trust** before it is used on three mirrors: the classical Cassegrain must return
S_I≈0 with nonzero coma, and the HST-like RC must return S_I≈S_II≈0 with residual
astigmatism. Both assertions pass, which is what certifies the aspheric-vs-spherical
sign and scale.

## Synthetic Korsch TMA (the regression target)

Convention: n-flip unfolded paraxial; `R > 0` = concave/converging; metres; stop at M1.
EFL sign is negative under the odd-reflection unfolded convention — the magnitude is the
physical EFL.

| Quantity | M1 | M2 | M3 |
|---|---|---|---|
| RoC R (m) | 3.000 | 0.280 | 2.800 |
| conic K | **−0.974748** | **−1.034525** | **−4.552729** |
| → next vertex (m) | t₁ = 1.370000 | t₂ = 2.620000 | t₃ = 1.064384 (to focus) |

Derived: EFL = −5.0342 m (|EFL|/D = **f/5.03**), primary f/1.50, D = 1.0 m, field 0.05°.

Expected Seidel sums (the assertions):

| Sum | Expected | Meaning |
|---|---|---|
| S_I | 0 (<1e−12) | spherical corrected |
| S_II | 0 (<1e−12) | coma corrected |
| S_III | 0 (<1e−12) | astigmatism corrected → anastigmat |
| S_IV (Petzval) | −1.097e−06 | field curvature, **not** nulled here |

K1 ≈ −0.975 (near-parabolic primary) and K2 ≈ −1.03 (mild hyperboloid) are realistic and
JWST-like; the tertiary works harder (K3 ≈ −4.55) at this fast f/5. S_IV is reported but
not zeroed: nulling all four sums is the full flat-field Korsch condition and needs a
fourth layout degree of freedom (a power or spacing), which is the next exercise once the
solver is in MACOS.

**How to use as a test:** build the geometry from the table, trace, and assert the
emitted conics match to ~1e−6 and that S_I, S_II, S_III evaluate below 1e−10 (your trace
will compute them from real-ray fans; the paraxial fixture gives the target). Because the
conics were *solved*, any drift in your aberration evaluation shows up as a nonzero sum.

## JWST anchor (`jwst_anchor.json`) — trace-only, complete before use

Verified from public sources: M1 RoC 15.880 m, K = −0.9967; M2 RoC 1778.913 mm,
K = −1.6598; EFL 131.4 m, f/20, 6.6 m pupil; Korsch TMA. **M3 radius/conic and the
M1–M2–M3 vertex spacings are intentionally left as TODO** — they are in Lightsey et al.
2012 (Opt. Eng. 51, 011003) and McElwain et al. 2023 (PASP, arXiv:2301.01779, Table 2).
Fill those in from the source before tracing; I did not reproduce them because I could not
extract them to fixture precision, and a fabricated prescription is worse than none.
(JPL is on the author list of the McElwain paper, so the full table should be readily
accessible to you.)
