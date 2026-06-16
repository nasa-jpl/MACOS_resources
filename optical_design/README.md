# Optical-design reference & regression fixtures

Language-agnostic ground truth for the **design layer** built on MACOS —
shared by the MATLAB binding (`mmacos/+macos/+design/`, the live
consumer) and the eventual Python port (`pymacos`).  Kept at the
`MACOS_resources` root, beside `rx_converter` / `segmirmaker`, so both
bindings (and their users) can find it and so the JSON fixtures serve as
the **shared golden specs** for the cross-language byte-identical-`.in`
parity criterion (`macos/PLAN_DESIGN_LAYER.md` §3).

| Path | What | Use it for |
|---|---|---|
| `TELESCOPE_DESIGN_REFERENCE.md` | Schroeder closed forms: first-order layout, two-mirror master conditions, conic constants per family, Korsch TMA, freeform | **reasoning** |
| `OPTICAL_DESIGN_AGENT_GUIDE.md` | how to use this set; FIXED conventions; the stop-and-fix gate | **read before editing any optical math** |
| `fixtures/telescope_design_fixtures.{json,md}` | 5 two-mirror `(f,D,m,β) → R1,R2,K1,K2` rows (RC / Cass / DK / Gregorian) | **assert these numbers** |
| `fixtures/tma_fixture.{json,md}` | Korsch TMA, conics solved to null S_I/II/III (machine-zero by construction) | **assert these numbers** |
| `fixtures/jwst_anchor.json` | real published JWST M1/M2; M3 + spacings TODO **on purpose** | real-world anchor (trace-disabled until completed) |
| `seidel.py` | paraxial + Seidel **oracle** (print-only; never writes fixtures) | ground truth / regeneration spec |
| `make_fixtures.py`, `make_tma_fixture.py` | self-validating regenerators | reproduce the JSON from scratch |

**Conventions (FIXED — do not reinterpret):** `R > 0` = concave /
converging; two-mirror conics in the Schroeder `(m, β)` convention
(`β` = back-focal-distance / f1).  If MACOS's internal radius/magnitude
sign differs, write the translation layer explicitly and test it against
the two-mirror fixtures before trusting anything downstream.

**Verification gate (hard stop):** after any change to conic-solving,
first-order-layout, or aberration-evaluation code, run the fixtures.
A conic mismatch `> 1e-6` on a two-mirror row, or a nonzero
S_I/II/III on the TMA fixture, is **stop-and-fix** — never widen the
tolerance or hand-edit a fixture; regenerate with the matching
`make_*.py` and explain the discrepancy.

`prepare-public-release` note: this is internal development reference;
exclude from any public-bound snapshot until the design layer ships.

The generators need `numpy`; run them with the pymacos venv
(`pymacos/.venv/bin/python`).
