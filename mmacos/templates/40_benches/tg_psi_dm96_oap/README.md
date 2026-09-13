# tg_psi_dm96_oap — the 96×96 Twyman–Green DM gauge, all-reflective (OAP) variant

The reflective sibling of [`../tg_psi_dm96`](../tg_psi_dm96): the two lenses of
the polarization phase-shifting Twyman–Green DM surface gauge (L1 collimator,
L2 focuser) become **off-axis parabolas** (OAPs). The refractive rig in
`../tg_psi_dm96` is the untouched record; this directory is the variant beside
it, driven by the **same** builder (`macos.design.twyman_green`, new option
`'optics'`) and **one** parameterized runner that runs *both* rigs.

## Why reflective

Removes the transmitted-glass-path and homogeneity rows from the cost budget,
buys achromatic legs and no ghost surfaces; the price is OAP alignment
sensitivity (measured here) and same-plane fold aberration (measured here). The
gauge's known roll-off is geometric (the tail conjugate + pupil distortion), so
the reflective train is also the natural place to re-attack it.

## Design (what changed, and what did not)

- **Only L1 and L2 become OAPs** (`twyman_green('optics','oap')`). The field
  lens FL in the tail stays a lens (Dave's ruling); its cost line is reported.
- The fold is confined to the **source→OAP1** and **OAP2→detector** legs: OAP1
  re-emits collimated along the lens rig's post-L1 direction, so the **plate BS,
  both arms, the compensator, the QWPs/analyzer and the recomb plane are
  geometrically unchanged** — the Stage-A clearance, the A2 sampling budget and
  the tail bookkeeping carry over.
- Both OAPs **fold in the plane of the BS** (same-plane folds). The fold AOIs
  are re-solved by Stage A for the new legs (near-normal preferred; must clear
  the bodies inside the leg cap). OAP conjugates are kept, not focal lengths:
  pole→focus `= F1` (collimator) / `= F2` (focuser); the parent focal follows.
- OAPs are perfect conductors first (`IndRef=1, Extinc=1e22`; RS=−1, RP=+1,
  polarization-neutral) so the lens/OAP comparison is geometric. A coated-Al
  (`coat_set`) row is the polarization-cost stretch.
- **The tail is re-tuned for the OAP focuser** (`tg96_tail('bench.optics','oap',
  …)`) — the fieldlens trims were fit to L2's aberrations and do not transfer.

## Run it yourself

```matlab
% interactive (no exit); edit tg96_params.m, then:
tg96_run                                  % lens rig (equivalence gate)
tg96_run('bench.optics','oap','tag','oap')% reflective rig
```
```bash
# headless, memory-capped (model 1024 ≈ 11 GB); one model size per process:
./tg96_batch.sh lens
TG96_MEMMAX=20G ./tg96_batch.sh oap "'bench.optics','oap'"
# tail retune (do first for the OAP rig; writes <tag>_tail.mat):
matlab -batch "tg96_tail('tag','oap','bench.optics','oap')"
```
Outputs land in `runs/<tag>/`: `<tag>_report.txt`, `<tag>.mat`,
`<tag>_{test,ref}.in`, `<tag>_layout.png`, `<tag>_closure.png`,
`<tag>_transfer.png`. The `figs` stage also renders the optics layout two ways:
- **`<tag>_render.png`** — the FULL RAYTRACE render via **`macos.view_rx`**: the
  test arm traced to the detector, optics as solid bodies on their real conic
  sag + apertures, the beam as a filled ray bundle read back from the engine
  (correct for the folded OAP legs), in two panels — **TABLE PLANE** (looking
  down on the bench) and **ISO** — for checking beam-vs-edge clearances. This is
  the deck_zwfs slide-4 recipe (`zwfs_dm96/zwfs_wf_figs.m`).
- **`<tag>_sketch.png`** / `_sketch_ref.png` — the lighter `Bench.sketch`
  schematic (chief-ray polyline, aperture-sized footprint bars, element names +
  leg lengths), test and reference arms.

The OAP rig shows L1/L2 folding the beam off-axis; the lens rig is near-collinear. Memory-bound? drop a trimmed `macos_param.txt` in the run
dir via `P.param_file` (keep `mGridMat ≥` the DM grid, 384 here).

## Files

| file | role |
|---|---|
| `tg96_params.m` | every knob of record + `bench.optics`, OAP fold AOIs, `calib_mode`, `place.*`, `d4`, `loop.*` |
| `tg96_run.m`    | Stage A–E + Stage PLACE (D1) + Stage MATRIX (D2) + Stage D4 + Stage LOOP (D7), one path for lens+OAP |
| `tg96_place.m`  | window placement from the ray affine (`dmg_frame`) + directional-parity + robust affine refit |
| `tg96_apply_parity.m` | detector-mm → field pixel under the resolved field-array parity |
| `tg96_tail.m`   | re-tune FL_F/FL_Kc/D_MASK_FL/DET_TRIM per optics (unaligned null) |
| `tg96_run_batch.m` / `tg96_batch.sh` | `matlab -batch` wrapper (exit only here) + launcher |

## Calibration mode (Dave 2026-09-10, the ZWFS-S10 default)

`battery.calib_mode`:
- **`matrix`** (default) — the MEASURED response matrix dw/da: poke every
  `matrix_step`-th actuator on a sparse grid, step through the offsets so every lit
  actuator is poked once, cut each response from its own detector window (placed by
  the ray affine, `tg96_place`), assemble J and estimate commands by regularized LS.
  Windows are placed by the affine + a robust refit, NOT a parity search — this is
  what lets the folded OAP rig register where `register_two_pokes` cannot. On the lens
  it BEATS the kernel record (flat/single-10nm 0.9916/2.2pm vs 0.9654/21pm;
  flat/random 0.9894 vs 0.9203) with flat modal transfer 0.96–0.99 across the band.
- **`kernel`** — the record: `register_two_pokes` + one interpolated truth map,
  compared in detector-pixel space (Stage C–E as first shipped).

## D1 window-placement gate + D4 alignment sensitivity

- `tg96_run('bench.optics','oap','stages',{'bench','place'})` runs the D1 gate:
  response CoM vs the affine-predicted pixel for every lit actuator, with a
  non-vacuity check (the old axis-aligned mapping fails). **Lens 100% within 2px;
  OAP 75% (window containment 98.9%)** — the OAP shortfall is the fold's astigmatism
  (physical: single-poke == multiplexed; the exact-centre poke reads 0).
- `battery.d4=true` (OAP) runs Stage D4: perturb OAP1/OAP2 by `d4_dec_um` /
  `d4_tilt_urad` and re-read the null shift + the single-actuator differential. The
  null moves ~1.5 nm/µm and ~10 nm/µrad but the differential gain stays 0.99 — the
  null and alignment drift are common-mode and cancel in the differential.

## Results

Lens equivalence gate EXACT (model 1024); matrix calibration beats the kernel record;
the OAP reflective gauge images the DM, reads a localized differential as well as the
lens (single 0.9948/2.2pm), and its differential is robust to the 12.9 nm null and to
OAP misalignment — the fold's cost is astigmatism cross-talk (~0.42) on dense/high-
order patterns (random-10nm 0.75 vs lens 0.99). Full numbers + tables + departures in
**`REPORT_oap.md`**.

## Closed-loop hold metric (D7) — the on-orbit servo mode

`tg96_run('stages',{'bench','loop','figs'})` runs Stage LOOP: the DM held at the
30 nm working surface by a proportional loop (gain 0.5, 60 cycles) closed through
the four-step reading and its measured matrix (calibrated ON that surface). The
loop code is the **shared** `../dm_gauge_lib/dmg_loop.m` — the identical file the
ZWFS runs (`P.loop.*` knobs, drift `seed 77` shared with the ZWFS so both gauges
see the same realizations). Cost: K+1 traced states per (drift, photon level), an
hour-class job at model 1024 (`tg96_batch.sh`). **Lens result** (`runs/loop_lens`):
holds the walk and the photon noise like the ZWFS stepped/vector readings but at
**~2.6× the photons per measurement** (3 pm from 5.5e12 noise-only / 2.0e13 walk;
thermal floor 13.1 pm = the rate/g lag). **Departure:** the IFO four-step has a
**noiseless step floor ~8.6%** (unlike the ideal ZWFS stepped reading's 0.000) —
the gauge's geometric roll-off + modal cross-talk surfacing in closed loop
(L-like, not S-like). **OAP row** (`runs/loop_oap`, bare Al): the residual fold
cross-talk (a benign-looking 0.95 dense gain open-loop) becomes the DOMINANT
closed-loop floor — 3 pm noise-only at 3.4e13, the 2 pm walk floors at 4.1 pm
(never 3), thermal 39 pm, step floor ~27.6%. **The loop distinguishes the lens
(holds) from the OAP (does not) where the open-loop battery did not.** Numbers +
the ZWFS comparison table in **`REPORT_oap.md`**.
