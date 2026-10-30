# tma_freeform — Off-axis unobscured three-mirror freeform designer

The companion to wf2_freeform, where freeform genuinely earns its keep. Pick the knobs below; the script seeds a coaxial Korsch TMA (3 conics null 3rd-order S_I/II/III), takes an UNOBSCURED off-axis section (eccentric pupil), refigures the 3 conics + rigid bodies to the best a 3-mirror can do over the field, then layers freeform Zernike departures on M2 + M3 -- holding every radius and conic -- and re-optimizes. Verifies it is unobscured and saves the deliverable (.in + .mat + figures).

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    tma_freeform

## What a run gives

Taken off axis by 0.89 m to clear its beams, the three-conic design reaches 0.0213 waves at its worst field over ±3′; freeform departures on M2 and M3, every radius and conic held, take it to 0.0176 waves. On this field the conic limit is already diffraction-limited, so the freeform's gain is small; `wf2_freeform` shows why a third mirror is what makes the freeform useful at all.

## Source

Generated 2026-10-26 from the header of `tma_freeform.m` and the mmacos manual's run of this template (Part II, macos 1b2ff44e); this README is the template's own now -- edit it with the code.
