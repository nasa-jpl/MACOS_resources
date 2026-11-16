# bench_ifo_dm — Twyman-Green IFO testbed, phase 1: measure a DM (PROPER model)

Follow-on to templates/40_benches/bench_ifo (read that first): the same compensated Twyman-Green rig -- now built in ONE call with macos.design.twyman_green -- with the test optic replaced by a DEFORMABLE MIRROR carrying random actuator pokes. The DM surface comes from the PROPER DM model (proper.prop_dm influence functions, |command| < POKE_NM), is loaded into MACOS as a GridData surface in the optic's own local frame, and the 7-frame de Groot PSI pipeline recovers the surface map, which is then compared point-by-point against the PROPER truth map.

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    example_bench_ifo_dm

## What a run gives

A DM surface of 16.92 nm rms, made with PROPER, is recovered through the traced pupil map with correlation 0.998426 and a residual of 0.969 nm rms; the phase-shifting itself is exact (the PSI against the direct field phase reads in the 1e-5 pm range), so the residual is the pupil imaging: its magnification, its distortion, and the registration the run refines.

## Source

Generated 2026-10-26 from the header of `example_bench_ifo_dm.m` and the mmacos manual's run of this template (Part II, macos 1b2ff44e); this README is the template's own now -- edit it with the code.
