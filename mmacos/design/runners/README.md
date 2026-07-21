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
| `run_simulator` | **SHIPPED** (Dave 2026-07-20: ingest a time series in x / z / grid coefficients — opening with µm-to-mm misalignments — and play it through the ENGINE showing BOTH the UNCORRECTED and the CORRECTED performance.  An initial image-based wavefront control `u = -pinv(dwdu)·w(frame 1)` is solved from the engine wavefront — as a Tikhonov ridge (the OSC form; a plain pinv noise-amplifies near-degenerate dwdu combos into huge canceling commands, a hard SVD cutoff over-truncates the correction) — and then the **metrology loop closes** (`met_loop`, default on; Dave 2026-07-20): the post-WFC state is the control TARGET, and each frame the sensed drift `δm = m − m_ref` drives an **RBCS estimator/controller** (Tesch, *RBCS Algorithms* ch 2.3+3.3): a **weighted-LS / BLUE estimator** (§2.3.2 eq 11) `δx̂ = R_meas·δm`, `R_meas = (Hᵀ N⁻¹ H + R_x⁻¹)⁻¹ Hᵀ N⁻¹` (H = dmdx, N = sensor-noise cov, R_x = state/disturbance prior), followed by a **min-pose-error controller** (§3.3.1 eq 16-17) `u_t = u_{t−1} − k_p·δx̂(control DOFs)`.  A **raw pseudo-inverse** (the basic-LS estimator §2.3.1 eq 10) **DIVERGES** — it amplifies un-modelled δm content (figure drift, linearization residual) by 1/σ_min in the weakly-observed rigid directions and the integrating loop runs away (verified: 5.6×10⁶ nm blow-up, 34 mm commands; the BLUE prior R_x pulls weak/pinned DOFs to prior-0 and holds the same case to ~5 nm, 74 nm commands).  This is STATE weighting (noise + disturbance statistics), **not** wavefront-impact weighting (§3.3.2 eq 19, deliberately not used per Dave: "estimate the state without weighting the WF impact").  `met_loop=false` HOLDS the initial u instead and the drift accumulates.  Figure drift aliases into x̂ through the l/e_piston rows (the simple estimator has no figure states) — negligible at realistic few-nm figure drift; figure states via dmdz/dmdgrid are the s7b estimator's job.  Knobs: `meas_noise` ([σ_l σ_e], default [1pm 1nm]), `state_prior` ("auto" from the drift stats, or an ndof vector), `ctrl_gain` (k_p, default 0.5<1 for loop margin), `ctrl_reg` (command-size penalty ρ).  **`wfc_reset_times`** (default []): times (s) at which to RE-RUN the image-based wavefront control mid-history — Tesch's periodic **WF Maintenance Activity** (updates the calibration pose x_cal).  Each reset re-nulls the accumulated wavefront with a fresh ridge solve on the control DOFs and re-references the MET-loop target to the new pose; marked with a dashed line on the rms-WFE / Strehl time plots.  **`wfc_reset_tol`** (default = wfc_tol): a tighter reset ridge engages the weakly-observed LATERAL DOFs that counter segment focus/astig on a parabolic parent (Dave 2026-07-21: an x move changes a segment's local best-fit radius → focus + a bit of astig; y/twist → astig; verified on the s4 dwdx — focus RB-residual 0.017, astig 0.31–0.49, higher order uncorrectable).  **`wfc_on_frame`** (default 1): delays the WFC initialization so the movie opens uninitialised at the as-deployed WFE ("no system starts perfect").  **`loop_senses_figure`** (default true): set false to model a truss that reads RIGID POSE only, so figure accumulates unseen for the periodic WFC to catch.  **`meas_bias`** (nmet×T, default []): a slow metrology calibration-drift trajectory the loop holds while the image-based reset re-references past it (the WF-maintenance core use case).  Each movie frame = OPD uncorrected \| OPD corrected + PIX psf + COMPOSE broadband psf + `m = [l; e]` bar charts, with ACCUMULATING rms-WFE (unc vs corr, log scale) and Strehl-vs-time curves (unc λ₀ / corr λ₀ / corr broadband; peak ratio to the nominal psf).  **TWO-PASS engine schedule**: the incremental perturb path's fixed-order single-axis rotations don't commute, so toggling ±u every frame would accumulate a systematic ~\|u_rot\|² phantom rotation (~µrad over 100 frames at 50 µrad control) — the runner plays the whole uncorrected history first, reloads the Rx, then plays the corrected history.  m bars are the validated linear measurement model `[dldx; dedx]·(x+u) + dmdz·z + dmdgrid·g`; when the history has no grid states the sim runs on the met Rx and the engine `l` is cross-checked.  The initial control is ITERATED (`wfc_iters` Gauss–Newton refinements, default 3 — one linear solve leaves the ~0.5% column nonlinearity of µm-to-mm states as a µm residual), and the per-frame corrected-leg engine-vs-linear `w_rel` is computed on the DRIFT INCREMENT (frame t − frame 1), extending the s6 gate to MIXED drift states) | all of the above + a `ts` struct (`.dt`, `.x` 6nb×T SI, `.z`/`.g` BaseUnits×T in jac channel order) | `t*.png` frames + `*_sim.gif` + per-frame report + `*_sim.mat` (u, m_hist, rms/Strehl curves both legs, dmdz/dmdgrid) |
| estimator/controller loop | PLANNED (Dave 2026-07-19: SINGLE-STEP STATIC estimator — the OSE form x̂ = x̄ + K·(m − m̄) with converged steady-state gains, m = [l; e]; controller = regularized wavefront-nulling on dwdu, recomputed continuously instead of run_simulator's held one-shot u; background: `MACOS_sandbox/Documents/OSE_Eqns_2019.pdf` + the 2025 JATIS HWO paper.  Builds on run_simulator: same ts ingestion, m_hist becomes the estimator input, dwdu the control authority) | run_simulator artifacts | controlled/uncontrolled stats |

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
