# Working With the Optical Design Reference & Fixtures (agent guide)

This file tells a coding agent (Claude Code) how to use the telescope-design reference
material and regression fixtures in this repo. Read it before touching any optical-math
code (conic solving, first-order layout, aberration evaluation, Rx emission).

The single biggest failure mode in this area is **silently guessing a sign or
parameterization convention**. Do not do that. The conventions below are fixed; if a
formula in the code disagrees with them, the code is wrong, not the convention.

---

## 1. What these files are (and what each is for)

| File | Role | Use it to… |
|---|---|---|
| `TELESCOPE_DESIGN_REFERENCE.md` | **why/how** | get the closed-form conics, the master third-order conditions, and the design philosophy for Cassegrain / RC / Dall-Kirkham / Gregorian / Korsch TMA / freeform |
| `telescope_design_fixtures.md` / `.json` | **assert-these-numbers** | two-mirror regression targets `(f,D,m,β) → R1,R2,K1,K2` |
| `tma_fixture.md` / `.json` | **assert-these-numbers** | three-mirror anastigmat regression target (conics solved to null S_I,S_II,S_III) |
| `jwst_anchor.json` | **real-world anchor** | a published JWST trace-only check — **incomplete on purpose** (see §5) |
| `seidel.py` | **runnable oracle** | recompute paraxial + Seidel sums; regenerate/extend fixtures; precise spec of the aberration algorithm |
| `make_fixtures.py`, `make_tma_fixture.py` | **generators** | reproduce every fixture from scratch; they self-validate |

Rule of thumb: **read the `.md` files when reasoning; load the `.json` files when
testing; run the `.py` files when you need ground truth.** Prefer the JSON over the
markdown tables for anything you consume programmatically — no parsing ambiguity.

---

## 2. Conventions (fixed — do not reinterpret)

**Radius sign (shared by fixtures and the engine):** `R > 0` means concave / converging
(a concave primary has `R1 = 2·f1 > 0`). This is consistent across `telescope_design_fixtures.*`,
`tma_fixture.*`, and `seidel.py`.

**Two-mirror conic closed-forms** in `TELESCOPE_DESIGN_REFERENCE.md` are parameterized in
the **Schroeder `(m, β)` convention**: `m` = secondary magnification, `β` =
(primary-vertex→focus distance)/`f1`, with `β > 0` for a focus behind the primary vertex.
Do **not** mix this parameterization with a raw radius-sign formula from elsewhere; the
RC conic is only correct in this convention.

**Seidel engine (`seidel.py`)** uses an **n-flip unfolded paraxial model**: `n` starts
at +1 and flips sign at every mirror. One artifact of this: the reported **EFL is
negative under an odd number of reflections** — the *magnitude* is the physical EFL, the
sign is bookkeeping. Aberration sums (`S_I…S_IV`) are sign-meaningful and must match the
fixtures.

If MACOS internally uses a different radius sign rule or magnification sign, write the
**translation layer explicitly** and test it against the two-mirror fixtures before
trusting anything downstream. Convention mismatches are the first thing to suspect when a
fixture fails.

---

## 3. How to use the fixtures as tests

Wire the JSON fixtures into the test suite (pytest, or MATLAB `matlab.unittest`) so they
are **executable**, not prose. For each design:

1. Build the geometry from the fixture inputs.
2. Trace it through MACOS.
3. Assert the emitted `R1, R2, (R3), K1, K2, (K3)` match the fixture to **~1e-6**.
4. Assert the relevant aberration sums evaluate below **1e-10**:
   - two-mirror RC: `S_I ≈ S_II ≈ 0`, `S_III ≠ 0`
   - classical Cassegrain / Dall-Kirkham: `S_I ≈ 0`, `S_II ≠ 0` (DK shows large coma off-axis — that *is* the test)
   - TMA: `S_I ≈ S_II ≈ S_III ≈ 0` (`S_IV`/Petzval is reported, not nulled — see §6)

Because the TMA conics were **solved** (not hand-entered), any drift in the aberration
evaluation shows up immediately as a nonzero sum. Treat that as signal.

---

## 4. Verification gate (hard stop)

After **any** change to conic-solving, first-order-layout, or aberration-evaluation code:

- Run the design fixtures.
- A nonzero `S_I/S_II/S_III` on the TMA fixture, or a conic mismatch `> 1e-6` on any
  two-mirror row, is a **stop-and-fix**, not a warning. Do not proceed, do not "adjust
  the tolerance," do not edit the fixture to pass. Fix the code or surface the conflict.
- If you believe a fixture itself is wrong, regenerate it with the matching generator
  (`make_fixtures.py` / `make_tma_fixture.py`) and explain the discrepancy — do not
  overwrite a fixture by hand.

The generators self-validate: `make_tma_fixture.py` asserts the engine reproduces the
two-mirror RC (`S_I=S_II=0`) and classical Cassegrain (`S_I=0, S_II≠0`) *before* it
trusts the three-mirror solve. Keep that ordering if you extend them.

---

## 5. Provenance (do not promote TODO values to facts)

`jwst_anchor.json` mixes verified and missing data **on purpose**:

- **Verified (cite-backed):** M1 `RoC 15.880 m, K=-0.9967`; M2 `RoC 1778.913 mm,
  K=-1.6598`; EFL `131.4 m`; `f/20`; 6.6 m pupil; Korsch TMA.
- **TODO:** M3 radius/conic and the M1–M2–M3 vertex spacings. These live in
  Lightsey et al. 2012 (Opt. Eng. 51, 011003) and McElwain et al. 2023
  (PASP, arXiv:2301.01779, Table 2).

**Do not fill the TODO fields by guessing, interpolating, or inferring** from the
verified values. Either a human supplies the table values, or the JWST case stays
trace-disabled. A fabricated prescription is worse than a flagged gap. The synthetic TMA
in `tma_fixture.json` is the fixture to trust for regression; the JWST anchor is a
real-world cross-check only once completed.

---

## 6. Known limits / next steps

- The synthetic TMA nulls `S_I, S_II, S_III` but **not** `S_IV` (Petzval / field
  curvature). Full field flattening is the additional Korsch condition and needs a fourth
  layout degree of freedom (a power or spacing). Do not "fix" the nonzero `S_IV` by
  perturbing conics — it is independent of the conics by construction.
- When implementing the analytic Korsch closed-form solver (Reference §6), use
  `seidel.py` as the validation harness: solve the conics analytically, then assert they
  reproduce the fixture's machine-zero sums.
- For freeform work (Reference §7), the canonical internal surface representation is
  Zernike / φ-polynomial departure (not XY-polynomial) because of the nodal-aberration
  -theory term-to-aberration correspondence. Convert on emit.

---

## 7. Suggested CLAUDE.md snippet (keep CLAUDE.md lean — point, don't paste)

Do **not** paste the reference doc or fixtures into CLAUDE.md; it bloats every session.
Add only a pointer like this:

```
## Optical design layer
- Equations & families: docs/TELESCOPE_DESIGN_REFERENCE.md
- Agent guide (read before editing optical math): docs/OPTICAL_DESIGN_AGENT_GUIDE.md
- Regression fixtures: tests/fixtures/*.json  (run after any optical-math change)
- Convention (FIXED): R>0 = concave/converging; two-mirror conics in Schroeder (m,β);
  verify against the reference doc before editing. Fixture failure = stop-and-fix.
```

Then open the referenced files on demand rather than carrying them in context.
