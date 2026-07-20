# mmacos stage runners — the design-to-simulation pipeline

These functions ARE the product (Dave 2026-07-19): a small set of
reusable, tested stage runners that hand a prescription file from
stage to stage.  Each runner is a MATLAB *function* — explicit input
paths + an options struct in, an artifact struct out, no hidden state —
and each emits a THOROUGH text report plus figures beside its
artifacts.  `mmacos_setup.m` puts this directory on the path.

```
 design            ->  monolithic-PM .in
 segmentation      ->  segmented .in            (+ Hx edge-sensor sidecar)
 sensitivities     ->  dwdx / dwdz / dwdgrid    (.mat, center or multi-field)
 met               ->  met .in + metopt .in     (+ dedx, dldx, dxde/dxdl/dwde/dwdl)
 compare           ->  linear model vs mmacos on stepped x/z/grid states
 simulate          ->  time-history PSFs, uncontrolled + controlled
```

**Handoff contract:** the interface between stages is the `.in` file
plus *declared sidecars* — segmentation also hands its SegMirMaker
`Hx.m` to the MET stage; sensitivities hands its Jacobian `.mat` to
MET/compare/simulate.  Runners rehydrate everything else from the
prescription itself (`macos.design.seg_from_rx`), so a stage can be
re-run in a fresh session from files alone.

| runner | status | in | out |
|---|---|---|---|
| `run_design` | PLANNED (today: `design/examples/e2e/s1_telescope.m` + `s2_instrument.m` over `tma_layout`/`offner_layout`) | design params | mono `.in` + report/views |
| `run_segmentation` | **SHIPPED** | mono `.in` | segmented `.in` (physical apertures) + `Hx.m` + footprint/views figures + parity report |
| `run_sensitivities` | **SHIPPED** | (segmented) `.in` | `*_sens.mat` (dwdx/dwdz/dwdgrid, + dwdsurf opt-in) + conditioning report + per-element pages in `*_pages/` |
| `run_met` | **SHIPPED** | segmented `.in` + `Hx.m` + jac `.mat` | `*_met.in`, `*_metopt.in`, `*_met.mat` (dedx/dldx/gains), report + views |
| `run_compare` | **SHIPPED** (Dave's spec 2026-07-19: poke every model DOF in turn — rigid x at 100 nrad/100 nm, then z figure modes, then grid influence DOFs at 100 nm; per poke TWO graphics — mmacos ENGINE vs LINEAR MODEL — each an OPD map above stacked `l` / `e_piston` / `e_gap` / `e_shear` bars; dwell 1.6 s.  Figure pokes' l/e bars show the `macos.design.dmet_dfig` **dmdz/dmdgrid** figure-sensing blocks — mode shape at each sensor/launcher mount point — vs the engine's rigid METcalc/Hx zeros) | segmented `.in` + `Hx.m` + jac `.mat` + met `.mat` | per-poke frames + `*_compare.gif` + agreement report + `*_compare.mat` (`dwdu` control columns + `dmdz`/`dmdgrid` for the simulator's estimator H) |
| `run_simulator` | PLANNED (Dave 2026-07-19: SINGLE-STEP STATIC estimator — the OSE form x̂ = x̄ + K·(m − m̄) with converged steady-state gains, m = [l; e]; controller = regularized wavefront-nulling on dwdu; background: `MACOS_sandbox/Documents/OSE_Eqns_2019.pdf` + the 2025 JATIS HWO paper) | all of the above | time-history PSFs + controlled/uncontrolled stats |

The `sensitivities/run_dwd*_multi.m` scripts (and their self-contained
copies under `sensitivities/examples/`) are thin CONFIG wrappers over
`run_sensitivities` — same user interface, one algorithm source.
Edge sensors (2026-07-19 model): SegMirMaker emits 2 sensor locations
x 3 axes (normal + in-plane pair) per shared segment edge, purely
differential (no absolute-piston anchor row); `macos.design.
edge_sensors` ingests the axis/location tags for the MET stage.

Worked examples driving the runners: `design/examples/e2e/` (the full
six-stage sequence — the canonical template), `design/examples/e5_seg/`
+ `e5_pie/` (single-stage segmentation/MET narratives on the e5
fixture).

Conventions: every runner call is reproducible from its report header;
geometry options are in the prescription's BaseUnits, noise/prior
sigmas in SI; figures land beside the artifacts; examples never call
`exit(0)` (batch wrappers do).
