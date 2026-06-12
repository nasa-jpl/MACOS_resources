# Design-layer examples (`+macos/+design`)

Runnable worked examples for the MATLAB design layer (Sprint 2A-i and
on): import a prescription, analyze it, and optimize it — all over the
`macos.design.System` surface and the bitwise-verified Phase 7
sensitivity machinery.  Each script ends in `exit(0)` (batch-mode rule,
`../../CLAUDE.md`) and runs against the committed `e5hex1.in` fixture in
`../sensitivities/e5hex1/`.

| Example | Flow | Shows |
|---|---|---|
| `example_sensitivities_from_rx.m` | `from_rx → describe → sensitivities → table` | The 2A-i headline: import any Rx, get a labeled rigid-body + Zernike sensitivity table (the Thrust A §1.3 recipe at package level). |
| `example_align_from_rx.m` | `from_rx → sensitivities → vary → evaluate → optimize` | The full sprint: declare a design variable, misalign an element, and let `optimize()` drive it back to minimum WFE. |

Run one:

```bash
matlab -batch "run('$(pwd)/example_align_from_rx.m')"
```

The same flow runs unchanged on a CodeV/Zemax-converted prescription —
that is the expected dominant entry point (PLAN_DESIGN_LAYER §1.0).

These are distinct from `../sensitivities/` (lower-level `macos.dw_dx` /
`macos.dw_dz_zernike` driver demos); the design-layer examples wrap that
machinery behind the `+macos/+design` API.
