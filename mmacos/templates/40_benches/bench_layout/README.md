# bench_layout — Laying out and optimizing an optical bench

A generic folded-bench worked example: build a complete test-bench train SEQUENTIALLY with the macos.design.Bench add-optic utilities, then optimize it in three physically-staged steps. The bench is the classic surface-gauge / interferometer test-arm topology: point source -> baffle (pupil stop) -> L1 (collimating singlet) -> beam-splitter REFLECT -> DM (test mirror, retro) -> beam-splitter TRANSMIT (through the tilted plate, real walk-off) -> Fold1 -> Fold2 -> L2 (imaging singlet) -> focal-mask Reference (internal focus) -> Detector at the DM-PUPIL IMAGE.

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    example_bench_layout

## What a run gives

The optimized bench is 13 elements, and the builder's chief ray and the engine's agree to < 1e-09.

## Source

Generated 2026-10-26 from the header of `example_bench_layout.m` and the mmacos manual's run of this template (Part II, macos 1b2ff44e); this README is the template's own now -- edit it with the code.
