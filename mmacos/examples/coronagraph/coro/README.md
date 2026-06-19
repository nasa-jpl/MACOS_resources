# Coronagraph division-of-labor experiments (Sprint 1, E1–E4)

MATLAB-side coronagraph scoring + cost characterization on the Phase-5
`Rx_Coro` corpus, backing `macos/PLAN_DESIGN_LAYER.md` §8 Sprint 1.
These answer "how much of diffraction-based optimization lives in MATLAB
vs Fortran" — empirically, with measured numbers.

## Drivers

| Script | What it measures |
|---|---|
| `E1_darkzone_contrast.m` | MATLAB-side dark-zone contrast on `Rx_Coro_FPM` (FPM+Lyot) vs no-mask reference; reproduces the 3.21e6 suppression baseline; per-eval wall time at nλ=3 |
| `E2_dm_modes_cost.m` | DM Fourier modes as `fmincon` vars (DM = Elt 4 grid surface); confirms the (nModes+1)-trace FD scaling; reports the multiplexed separable-poke Jacobian (~37 traces, actuator-count-independent) |
| `E3_calib_timing.m` | Existing (ray-trace) CALIB inner loop timing on the DM; projects a diffraction-scoring DarkZone inner loop (naïve FD = hours vs multiplexed ≈ 2 min) |
| `E4_lambda_loop_cost.m` | load / set-λ / trace / propagate cost split per wavelength; whether a MACOS-side `spectral_run` is worth building (verdict: no, for reflective) |

Run (each ends with `exit(0)` under `-batch`):
```matlab
addpath(<mmacos>/src); addpath(<mmacos>/examples/design/coro);
out = E1_darkzone_contrast();
```

## Scoring machinery (ported from pymacos `contrast.py`)

`radial_profile`, `first_airy_null`, `lambda_over_D_pixels`,
`radial_contrast`, and `dark_zone_metrics` (per-pixel dark-zone stats:
mean / peak / floor / median / energy — the **selectable optimization
objectives**; supports a one-sided `'side'` or `'sector'` region, since
a 1-DM system digs a deeper one-sided dark zone than a full annulus).
Pinned by `tests/tCoroContrast.m` (pure math, in `SUITE_FAST`).

## Results, logs, and disk hygiene (`results/`)

- `save_coro_workspace(tag, out)` saves the **full** workspace (incl.
  the 1024² intensity arrays + DM states) to
  `results/<tag>_<timestamp>.mat` so analysis resumes without
  re-tracing.  Keeps only the last 2 per tag.
- `coro_log_start(tag)` streams the run to `results/<tag>.log`
  (overwritten each run) — **watch a run live with**
  `tail -f results/<tag>.log` at ~zero load (don't spin a second
  MATLAB — it contends for the single Home-license seat).
- `clean_results.sh` deletes `results/*.mat` older than `RETAIN_DAYS`
  (default 7); installed as a **weekly user cron** entry.
- `results/*.mat` and `*.log` are `.gitignore`d (local state).
