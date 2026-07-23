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
the two-mirror fixtures before trusting anything downstream.  (The MACOS
emission mapping is now pinned: `KcElt=K`, `KrElt=-|R|`, `psiElt`→CoC —
see the agent reference memory and `mmacos/+macos/+design/Telescope.m`.)

**Verification gate (hard stop):** after any change to conic-solving,
first-order-layout, or aberration-evaluation code, run the fixtures.
A conic mismatch `> 1e-6` on a two-mirror row, or a nonzero
S_I/II/III on the TMA fixture, is **stop-and-fix** — never widen the
tolerance or hand-edit a fixture; regenerate with the matching
`make_*.py` and explain the discrepancy.

## Coronagraph back-end (instrument layer, Sprint 3)

Companion set for the coronagraph treated as a telescope back-end
(Lyot-type, two-DM wavefront control).  **First-order *layout* only — NOT
a diffraction/contrast model, and the sample is NOT a regression
fixture.**

| Path | What | Use it for |
|---|---|---|
| `CORONAGRAPH_DESIGN_RULES.md` | architecture, plane sequence, OAP-relay rules, the two-DM quarter-Talbot derivation, FPM/Lyot/apodizer rules, contrast-limiting effects | **reasoning** |
| `CORONAGRAPH_DESIGN_AGENT_GUIDE.md` | how to use this set; the hard rules | **read before editing coronagraph layout** |
| `coronagraph_layout.py` | first-order layout generator: `(D_beam, N_act, λ, IWA/OWA, F/#s, pixel)` → OAP focal lengths, pupil sizes, FPM scales, Nyquist F/#, DM-sep envelope | **call `design()` for a layout** |
| `coronagraph_layout.md` | usage notes + the quarter-Talbot caveat | **reasoning** |
| `coronagraph_layout.json` | **sample output** (Roman-class default) — **NOT golden truth** | example only |

**Coronagraph-specific rules (do not violate):**
- **`quarter_talbot_sep_mm` is a reference SCALE, never the DM1–DM2
  separation.**  It is frequency-dependent (~metres at OWA, ~hundreds of
  metres at IWA); the real ~1 m operating separation is optimized in the
  diffraction model (FALCO), not here.  Carry it as `dm_sep_upper_scale`.
- **Units are mm + nm here** (the telescope generators are **metres**).
  The only safe chaining point is the dimensionless `input_fnum` (feed
  the TMA's output F/# in).  Convert any physical length m↔mm explicitly.
- **λ/D is referenced to the telescope aperture, not the instrument
  beam** — this module is aperture-agnostic; don't substitute `D_beam`.
- This is layout, not performance: **no contrast, dark-hole depth, mask
  profiles, polarization, or EFC** come from here — those live in the
  diffraction/control layer (FALCO / PROPER).  Its self-checks are
  *constraint* checks (OWA ≤ N/2, Nyquist, `IWA<OWA`), not
  ground-truth comparisons — so `coronagraph_layout.json` is **not** a
  fixture and lives outside `fixtures/` deliberately.

`prepare-public-release` note: this is internal development reference;
exclude from any public-bound snapshot until the design layer ships.

The generators need `numpy`; run them with the pymacos venv
(`pymacos/.venv/bin/python`).
