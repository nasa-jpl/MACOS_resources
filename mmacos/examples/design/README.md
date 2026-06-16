# Design-layer examples (`+macos/+design`)

Runnable worked examples for the MATLAB design layer.  Two front-ends,
one analysis core (PLAN_DESIGN_LAYER §1.0): **import** an existing
prescription (`macos.design.System`, Sprint 2A-i) or **build** one from
design intent (`macos.design.Telescope`, Sprint 2A-ii).  Both feed the
same `vary → evaluate → optimize` surface and the bitwise-verified
Phase 7 sensitivity machinery.  Each script ends in `exit(0)` (batch-mode
rule, `../../CLAUDE.md`).

| Example | Front-end | Flow | Shows |
|---|---|---|---|
| **`example_telescope_design.m`** | builder | `Telescope → build → describe → from_rx → evaluate → vary → optimize → save` | **The design template / manual example.** Design a telescope from a handful of numbers; audit the derived layout + conics; see the family idea (RC vs Cassegrain conics on one layout); align a perturbed secondary; export. **Start here to design your own** — change the family + four first-order numbers in Stage 1. |
| `example_telescope_align.m` | builder | `Telescope → build → from_rx → vary → evaluate → optimize` | Focused: the builder→analysis loop — a built Cassegrain recovers an M2 despace error. |
| `example_sensitivities_from_rx.m` | import | `from_rx → describe → sensitivities → table` | Import any Rx, get a labeled rigid-body + Zernike sensitivity table (the 2A-i headline). |
| `example_align_from_rx.m` | import | `from_rx → sensitivities → vary → evaluate → optimize` | Import + align: declare a design variable, misalign an element, optimize it back. |

The builder examples derive geometry from closed-form optics
(`../../../optical_design/` reference + fixtures); the import examples run
against the committed `e5hex1.in` in `../sensitivities/e5hex1/` and run
unchanged on a CodeV/Zemax-converted prescription (the expected dominant
entry point, §1.0).

Run one:

```bash
matlab -batch "run('$(pwd)/example_telescope_design.m')"
```

These are distinct from `../sensitivities/` (lower-level `macos.dw_dx` /
`macos.dw_dz_zernike` driver demos); the design-layer examples wrap that
machinery behind the `+macos/+design` API.
