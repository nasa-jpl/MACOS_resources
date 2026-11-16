# bench_ifo — Twyman-Green interferometer on a bench, with phase-shifting (de Groot) data processing

A generic Twyman-Green IFO built with the macos.design.Bench add-optic utilities, then RUN as an instrument: simulate 7 phase-shifted interferograms and recover the test-optic surface with windowed phase-shifting-interferometry (PSI) processing -- the full simulate -> detect -> process -> compare-to-truth loop.

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    example_bench_ifo

## What a run gives

With the test optic a flat (the injected sphere radius is effectively infinite), the phase-shifted interferometer recovers it to a post-fit residual of 0.0014 nm rms surface, at a median fringe visibility of 0.833; the two phase-recovery algorithms (de Groot 7-frame and Hariharan 5-frame) agree to the floor of the arithmetic, printed as a bound.

## Source

Generated 2026-10-26 from the header of `example_bench_ifo.m` and the mmacos manual's run of this template (Part II, macos 1b2ff44e); this README is the template's own now -- edit it with the code.
