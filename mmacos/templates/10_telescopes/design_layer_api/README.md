# Design-layer API examples (`+macos/+design`)

Runnable worked examples for the MATLAB design layer -- the four
scripts that introduce the two front-ends.  The bench / coronagraph /
interferometer rows below moved to their own thread directories in the
2026-08 reorganization (see [`../../00_INDEX.md`](../../00_INDEX.md));
their descriptions are kept here until the Week-2 rewalk absorbs this
file into that index.  Two front-ends,
one analysis core (PLAN_DESIGN_LAYER §1.0): **import** an existing
prescription (`macos.design.System`, Sprint 2A-i) or **build** one from
design intent (`macos.design.Telescope`, Sprint 2A-ii).  Both feed the
same `vary → evaluate → optimize` surface and the bitwise-verified
Phase 7 sensitivity machinery.  Each script ends in `exit(0)` (batch-mode
rule, `../../../CLAUDE.md`).

| Example | Front-end | Flow | Shows |
|---|---|---|---|
| **`example_telescope_design.m`** | builder | `Telescope → build → describe → from_rx → evaluate → vary → optimize → save` | **The design template / manual example.** Design a telescope from a handful of numbers; audit the derived layout + conics; see the family idea (RC vs Cassegrain conics on one layout); align a perturbed secondary; export. **Start here to design your own** — change the family + four first-order numbers in Stage 1. |
| `example_telescope_align.m` | builder | `Telescope → build → from_rx → vary → evaluate → optimize` | Focused: the builder→analysis loop — a built Cassegrain recovers an M2 despace error. |
| `example_sensitivities_from_rx.m` | import | `from_rx → describe → sensitivities → table` | Import any Rx, get a labeled rigid-body + Zernike sensitivity table (the 2A-i headline). |
| `example_align_from_rx.m` | import | `from_rx → sensitivities → vary → evaluate → optimize` | Import + align: declare a design variable, misalign an element, optimize it back. |
| [`../../40_benches/bench_layout/example_bench_layout.m`](../../40_benches/bench_layout) | builder | `Bench add_* → emit → staged optimize → conjugate trim → render` | **Lay out and optimize an optical BENCH** (interferometer-test-arm topology): build source → baffle → L1 → BS reflect → DM → BS transmit → folds → L2 → focal mask → detector sequentially with `macos.design.Bench` add-optic utilities (analytic chief tracking incl. Snell walk-off through the tilted BS plate), then collimate L1 at the DM, focus L2 at the mask, and place the detector at the DM-pupil conjugate via the DM-tilt test. `Bench` also provides `add_oap` (off-axis parabola sections) and `add_relay` (Offner-type concentric 3-mirror 1:1 relay). |
| [`../../40_benches/bench_ifo/example_bench_ifo.m`](../../40_benches/bench_ifo) | builder | `twyman_green → emit both arms → PZT frames → deGroot PSI → recover figure` | **A working INTERFEROMETER with data processing**: compensated Twyman-Green (1.5 mm plate BS + double-passed compensator, glass paths balance exactly), PZT phase stepping (λ/8 → exactly π/2 double-pass), de Groot windowed 7-frame PSI + Hariharan-5 cross-check; recovers a weak-sphere test-optic figure to 0.00% radius error / 0.0014 nm residual. |
| [`../../40_benches/bench_ifo_dm/example_bench_ifo_dm.m`](../../40_benches/bench_ifo_dm) | builder | `PROPER prop_dm → GridData DM → PSI twice (differential) → per-ray pupil map → calibrated DM-plane surface` | **Phase 1 DM metrology on the IFO**: PROPER influence-function DM (checkerboard or random pokes) as a GridData test optic; DIFFERENTIAL PSI (poked − baseline, complex-subtracted — statics cancel, no unwrapping); `PSI_MODE` 'detector' (physical superposition of independently traced arms — includes the detector-leg retrace term) vs 'recomb' (tail-immune coherent-add injection); EMPIRICAL per-ray DM↔detector pupil map (magnification/anamorphism/rotation/nonlinear distortion report + mag-calibrated DM-plane surface product). Algorithm chain verified at 3e-5 pm. The `l2_trade/` subdir holds the detector-leg redesign: retrace mechanism analysis (linear slope-coupling of the singlet tail) + architecture trade → a field lens behind the FocalMask (`TAIL_ARCH`, the wired default) cuts the instrument-vs-truth residual 6.76 → 0.97 nm rms at 50 nm pokes (doublet L2 ties at 0.98; see `l2_trade/TRADE_NOTE.md`). |
| [`../../30_instruments/bench_ctb/example_ctb.m`](../../30_instruments/bench_ctb) | builder | `Bench add_oap ×8 → emit → staged optimize → pupil_zone_map + polarization diagnostics → PIL step (extra Rx)` | **A CORONAGRAPH TESTBED relay** (DST2R-like, all-reflective): source → OAP1 → 2 DMs → OAP2/3 → apodizer → OAP4 → FPM → OAP5 → Lyot → OAP6 → field stop → OAP7 → backend → OAP8 → FPA, an 8-OAP 2-DM relay alternating pupil/focus planes, built with near-normal folds (`add_oap` `'f'` + `AOI`; conjugate `r=f/cos²(AOI)`). Staged per-conjugate geometric optimization (Stage A solves Kr-only — a cheaper sphere), diffraction-limited (0.0014 λ WFE, 0 vignetting). Diagnostics: `macos.pupil_zone_map` (DM1-pupil-zone → FPA spot, the imaging-quality standard) + a polarization report (Al-coated mirrors → `jones_pupil`/`pol_maps`, mean vs contrast-relevant variation; documents why fold compensation is non-trivial). Adds a **pupil-imaging-lens (PIL)** design step emitting `ctb_pil{150,75}.in` — reproduces D. Marx's DST2R lens/camera trade (150 mm vs 75 mm: ~2.7× pupil-image size, camera moves ~100 mm). |

The builder examples derive geometry from closed-form optics
(`../../../../optical_design/` reference + fixtures); the import examples run
against the committed `e5hex1.in` in `../../50_sensitivities/e5hex1/` and run
unchanged on a CodeV/Zemax-converted prescription (the expected dominant
entry point, §1.0).

Run one:

```bash
matlab -batch "run('$(pwd)/example_telescope_design.m')"
```

These are distinct from [`../../50_sensitivities/`](../../50_sensitivities)
(lower-level `macos.dw_dx` / `macos.dw_dz_zernike` driver demos); the
design-layer examples wrap that machinery behind the `+macos/+design`
API.  The telescope-design *progression* (rc_* -> tma_* -> freeform) is
the rest of [`../`](..).
