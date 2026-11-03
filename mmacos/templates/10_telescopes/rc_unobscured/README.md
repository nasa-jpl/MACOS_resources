# rc_unobscured — Parameterized unobscured off-axis Ritchey-Chretien designer

Pick the four design knobs below; the script builds the on-axis RC parent, takes an UNOBSCURED off-axis section (eccentric pupil), refigures + field-optimizes it to diffraction-limited, verifies it is unobscured, and saves the deliverable (.in + .mat + figures).

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    rc_unobscured

## What a run gives

With an f/1.5 primary at f/20 the eccentric pupil clears the secondary's shadow at a decenter of 0.642 m, and after the axial refigure and the field optimization the worst field over ±0.5′ is 0.0281 waves, diffraction-limited, with all three optics clear.

## Source

Generated 2026-10-26 from the header of `rc_unobscured.m` and the mmacos manual's run of this template (Part II, macos 1b2ff44e); this README is the template's own now -- edit it with the code.
