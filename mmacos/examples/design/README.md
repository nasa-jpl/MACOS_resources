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
| `bench_layout/example_bench_layout.m` | builder | `Bench add_* → emit → staged optimize → conjugate trim → render` | **Lay out and optimize an optical BENCH** (interferometer-test-arm topology): build source → baffle → L1 → BS reflect → DM → BS transmit → folds → L2 → focal mask → detector sequentially with `macos.design.Bench` add-optic utilities (analytic chief tracking incl. Snell walk-off through the tilted BS plate), then collimate L1 at the DM, focus L2 at the mask, and place the detector at the DM-pupil conjugate via the DM-tilt test. `Bench` also provides `add_oap` (off-axis parabola sections) and `add_relay` (Offner-type concentric 3-mirror 1:1 relay). |
| `bench_ifo/example_bench_ifo.m` | builder | `twyman_green → emit both arms → PZT frames → deGroot PSI → recover figure` | **A working INTERFEROMETER with data processing**: compensated Twyman-Green (1.5 mm plate BS + double-passed compensator, glass paths balance exactly), PZT phase stepping (λ/8 → exactly π/2 double-pass), de Groot windowed 7-frame PSI + Hariharan-5 cross-check; recovers a weak-sphere test-optic figure to 0.00% radius error / 0.0014 nm residual. |
| `bench_ifo_dm/example_bench_ifo_dm.m` | builder | `PROPER prop_dm → GridData DM → PSI twice (differential) → per-ray pupil map → calibrated DM-plane surface` | **Phase 1 DM metrology on the IFO**: PROPER influence-function DM (checkerboard or random pokes) as a GridData test optic; DIFFERENTIAL PSI (poked − baseline, complex-subtracted — statics cancel, no unwrapping); `PSI_MODE` 'detector' (physical superposition of independently traced arms — includes the detector-leg retrace term) vs 'recomb' (tail-immune coherent-add injection); EMPIRICAL per-ray DM↔detector pupil map (magnification/anamorphism/rotation/nonlinear distortion report + mag-calibrated DM-plane surface product). Algorithm chain verified at 3e-5 pm; instrument-vs-truth residual dominated by the singlet-L2 pupil imaging (L2 redesign = the improvement path). |

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
