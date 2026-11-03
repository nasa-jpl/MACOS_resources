# bench_ifo_pol — Bench IFO polarization: ray-level Jones scoring (slice 1 of 3)

Polarization-honest Twyman-Green: a real Al coating on the beam-splitter face, ifPol on, scored with the GATED Phase-2 layer (jones_pupil + pol_maps). The coated BS face is hit as an EXTERNAL reflection in the test arm (air->Al, element 'BSrefl') and as an INTERNAL reflection in the reference arm (glass->Al, two glass transits, element 'BScrefr') -- genuinely different Jones -- so the arm-differential diattenuation / retardance at recombination is a MEASURED result, and its pupil variation is the polarization-induced PSI phase-error contribution.

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    example_bench_ifo_pol

## What a run gives

The coated beamsplitter's reflection matches the Fresnel equations to the floor of the arithmetic at 45 degrees (gate 1), and the coating is inert with polarization off (gate 2). Between the arms at recombination the plate leaves a diattenuation of 7.2133e-02 and a retardance of 8.3521e-02 rad, uniform across the pupil, so it costs the co-polarized fringe almost nothing.

## Source

Generated 2026-10-26 from the header of `example_bench_ifo_pol.m` and the mmacos manual's run of this template (Part II, macos 1b2ff44e); this README is the template's own now -- edit it with the code.
