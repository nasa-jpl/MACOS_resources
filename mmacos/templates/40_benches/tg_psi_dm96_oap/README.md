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
`<tag>_transfer.png`. Memory-bound? drop a trimmed `macos_param.txt` in the run
dir via `P.param_file` (keep `mGridMat ≥` the DM grid, 384 here).

## Files

| file | role |
|---|---|
| `tg96_params.m` | every knob of record + `bench.optics` and the OAP fold AOIs |
| `tg96_run.m`    | Stage A–E, parameterized; drives lens and OAP through one path |
| `tg96_tail.m`   | re-tune FL_F/FL_Kc/D_MASK_FL/DET_TRIM per optics (unaligned null) |
| `tg96_run_batch.m` / `tg96_batch.sh` | `matlab -batch` wrapper (exit only here) + launcher |

## Results

_(Filled by the model-1024 runs: lens equivalence gate vs the `../tg_psi_dm96`
S3/S4 record, then the OAP rig side-by-side — null before/after tail retune,
arm-state departure, single-actuator + dense-random differential rows in pm,
the modal transfer, the same-plane-fold effect on the 0.136 mm distortion row,
and the OAP alignment-sensitivity table. See `REPORT_oap.md`.)_
