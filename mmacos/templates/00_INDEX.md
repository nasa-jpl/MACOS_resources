# Template index

**STUB (2026-08-19).**  The mapping table below is authoritative; the
guided tour that goes with it is written during the Week-2 rewalk.

Every directory here is a **template**: a runnable, parameterized
starting point you copy and adapt.  Universal helpers live in the
library (`../src/+macos/`, `../design/src/`, `../design/runners/`) —
templates point into it, never house it.  Design *challenges* (a fixed
target plus our worked answer) live in [`../challenges/`](../challenges).

| Thread | Directory | Entry point |
|---|---|---|
| **T1** telescopes of increasing capability | [`10_telescopes/`](10_telescopes) | `design_layer_api/example_telescope_design.m`, then the `rc_*` → `tma_*` → freeform ladder |
| **T2** segmentation + interface coordinates | [`20_segmentation/`](20_segmentation) | `e5_pie/e5_pie.m` |
| **T3** instruments (imager, coronagraph) | [`30_instruments/`](30_instruments) | `coro_walkthrough/coro_walkthrough.m`; imager rung = `../templates/80_end_to_end/e2e/s2_instrument.m` until a standalone template is extracted |
| **T4** linear-model sensitivities (dw/dx, dw/dz, dw/dgrid) | [`50_sensitivities/`](50_sensitivities) | `../../sensitivities/run_dwdx_multi.m` (asset decks live here) |
| **T5** visualization of x, z, grid errors | [`60_visualization/`](60_visualization) | `view_rx_demo/view_rx_demo.m` |
| **T6** time-sequence simulation | [`70_simulation/`](70_simulation) | `../templates/80_end_to_end/e2e/s7_simulate.m` (pointer — see that directory's README) |
| — benches | [`40_benches/`](40_benches) | `bench_layout/example_bench_layout.m` |
| — end-to-end flows | [`80_end_to_end/`](80_end_to_end) | `e2e/` (s1–s7), `e2e2/` |
| — polarization | [`90_polarization/`](90_polarization) | `bench_ifo_pol/example_bench_ifo_pol.m` |

**Before you transpose or negate an OPD map by hand, read
[`../doc/opd_conventions.md`](../doc/opd_conventions.md).**  Orientation
(`macos.opd('orient','xy')`), sign (`'sign','wavefront'`) and the map's
reference (`macos.opd_ref('chief')`, which matters on segmented pupils)
are all options on the API — every template uses them rather than
post-processing the array.

Run any template with the package on the path:

```matlab
run('<path-to>/mmacos/mmacos_setup.m');
run('<path-to>/mmacos/templates/10_telescopes/tma_offaxis/tma_offaxis.m');
```
