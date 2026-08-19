# dst2 — faithful DST2R prescription (parallel reference)

The **actual DST2R design** (Brandon's CODE V layout), kept alongside the
generic coronagraph testbed in `../` so we can develop on the clean generic
CTB while tracking the real instrument in parallel.

## Contents

- **`raw/dst2v2_aox_v5_di_2025_05_07_fiximgtilt.seq`** — Brandon's CODE V
  sequence (ascii, CODEV 2024.03), the source of record. 8 OAPs (K=−1),
  focal lengths `f = |RDY|/2 = [2500 1524 1143 1350 675 635 635 762]` mm —
  **identical to the generic CTB `F_OAP` seeds**. Stop at dm1; 500 nm; metres.
- **`raw/DST2V2_AOX_V5_stop_at_dm1_F0{1..5}.IN`** — cv2macos output of that
  design (one Rx per field point), the "stop at dm1" variant matching the seq.
  `cv2macos` (`../../../../rx_converter/codev/cv2macos.seq`) is a CODE V macro;
  it was run on Brandon's lens to produce these — **CODE V is not needed here**.
- **`load_dst2.m`** — imports + traces all 5 fields, renders F01, saves
  `dst2.mat`. Writes the loadable `dst2_F0{1..5}_norm.in` (below).
- **`dst2_F0{1..5}_norm.in`** — engine-loadable normalizations (generated).
- **`dst2_F01_view_rx.png`** — the on-axis layout render.

## Two fix-ups the loader applies (non-destructive; `raw/` untouched)

The 2013-era cv2macos output needs light massaging for the current engine:

1. **Parse:** tighten spaced keywords (`ApType = Circle` → `ApType= Circle`)
   and add `BaseUnits`/`WaveUnits` (mm) — the current `msmacosio` reader wants
   them; the old converter omitted them. Without this, `load_rx` fails.
2. **Apertures:** the OAP `ApVec` circles carry the CODE V off-axis **decenter**
   (e.g. `ApVec= 75 200 0` = radius 75, 200 mm off-axis). The engine applies a
   circular aperture about the element **vertex** (the parent-parabola vertex),
   which for an off-axis section is far from the beam → every ray blocked
   (`ok=0`). Same off-axis-aperture frame issue documented for `add_oap`. The
   loader neutralizes them (`ApType=None`); the beam is well inside the optics,
   and **dm1 is the functional stop**, so the clear apertures aren't load-bearing.

## Status

All 5 fields load and trace clean (3210/3210 rays, 16 elements). **RMS WFE
≈ 0.5 waves** — this is the faithful import as converted, NOT diffraction-
limited: the cv2macos `.IN` places elements in global coordinates but does not
reproduce CODE V's focus/clocking exactly (see the converter's own header
caveats), and the reference `.IN` header carries 575 nm vs the seq's 500 nm
design wavelength. Geometry is right; residual WFE reflects converter limits —
which is exactly why active **development happens on the clean generic CTB**
(`../example_ctb.m`, 0.0014 λ) and DST2 is kept as the parallel reference.

## Relationship to the generic CTB

| | generic CTB (`../`) | DST2 (here) |
|---|---|---|
| OAP focal lengths | `[2500…762]` (DST2R seeds) | `[2500…762]` (same) |
| Topology | 8-OAP 2-DM relay | 8-OAP 2-DM relay (same) |
| Layout | planar, round-number legs | Brandon's exact spacings + decenter clocking (3-D) |
| Apertures/stop | DM stop, clean | dm1 stop, off-axis circles neutralized |
| WFE | 0.0014 λ (optimized) | ~0.5 λ (as-converted) |
| Role | **development** | **faithful reference** |
