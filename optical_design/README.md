# optical_design — design-layer reference fixtures & tools

Language-agnostic ground truth for the **MACOS design layer**
(`mmacos/+macos/+design/`, and the Python port in `pymacos`).  The JSON
fixtures here are the shared golden specs that the bindings are validated
against; the Python tools are the oracles that generate and check them.

Kept at the `MACOS_resources` root so both bindings — and their users —
can find it.

## Contents

| Path | What it is |
|---|---|
| `fixtures/telescope_design_fixtures.json` | 5 two-mirror rows `(f, D, m, β) → R1, R2, K1, K2` (Ritchey-Chrétien / classical Cassegrain / Dall-Kirkham / Gregorian). Golden. |
| `fixtures/tma_fixture.json` | Korsch three-mirror anastigmat with conics solved to null spherical/coma/astigmatism (Seidel S_I/S_II/S_III ≈ machine zero). Golden. |
| `fixtures/jwst_anchor.json` | Real published JWST M1/M2 as a real-world anchor. M3 + spacings are intentionally incomplete (trace-disabled) until filled in. |
| `seidel.py` | Paraxial + Seidel-aberration oracle for coaxial mirror trains (print-only; never writes fixtures). The ground-truth engine. |
| `make_fixtures.py` | Regenerates `telescope_design_fixtures.json` from the closed-form two-mirror relations; self-validating. |
| `make_tma_fixture.py` | Regenerates `tma_fixture.json`; solves the TMA conics against `seidel.py` and asserts the aberration residuals. |
| `coronagraph_layout.py` | First-order layout generator for a Lyot-type, two-DM coronagraph back-end. `design(spec)` → OAP focal lengths, pupil sizes, focal-plane scales, Nyquist F/#. |
| `coronagraph_layout.json` | A sample `coronagraph_layout.py` output (Roman-class defaults). Illustrative example — **not** a golden fixture. |

## Who consumes these

- `mmacos/+macos/+design/Telescope.m` and `seidel_seed.m` — the shipped
  MATLAB design layer — are ported and validated against the two-mirror
  and TMA fixtures.
- The mmacos test suite reads the fixtures via
  `mmacos/tests/private/design_fixture_path.m` (see `tDesignTelescope`).

## Conventions (fixed — do not reinterpret)

- **Telescope generators / fixtures use metres.** `R > 0` = concave /
  converging mirror. Two-mirror conics follow Schroeder's `(m, β)`
  convention, where `β` = back-focal-distance / f₁.
- **`coronagraph_layout.py` uses mm + nm** (not metres). The only safe
  chaining point between the metre-based telescope layout and this module
  is the dimensionless `input_fnum` (feed the telescope's output F/#);
  convert any physical length m↔mm explicitly.
- Coronagraph working angles are in **λ/D referenced to the telescope
  aperture**, not the instrument beam — the module is aperture-agnostic.
- `coronagraph_layout.py` produces **first-order layout only** — no
  contrast, dark-hole depth, mask profiles, or wavefront-control
  performance (those belong to a diffraction/control model such as PROPER
  or FALCO). Its `quarter_talbot_sep_mm` is a reference scale, not the
  operating DM1–DM2 separation.

## Regenerating / verifying the fixtures

The golden JSON fixtures are reproducible from scratch — never hand-edit
them. After any change to the conic-solving or aberration code, regenerate
and let the tools self-check:

```sh
# needs numpy (e.g. the pymacos venv):
python3 make_fixtures.py        # -> fixtures/telescope_design_fixtures.json
python3 make_tma_fixture.py     # -> fixtures/tma_fixture.json (asserts S_I/II/III ≈ 0)
python3 seidel.py               # prints the paraxial + Seidel oracle output
```

A conic mismatch `> 1e-6` on a two-mirror row, or a nonzero
S_I/S_II/S_III on the TMA fixture, means the generating code changed
behaviour — fix the code and regenerate, don't widen the tolerance or
edit the fixture by hand.
