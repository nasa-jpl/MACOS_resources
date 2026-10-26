# tma_widefield — Does each mirror earn its keep? Wide-field 2-mirror (Ritchey-Chretien) vs 3-mirror (Korsch TMA)

Multi-mirror telescopes exist to widen the corrected field of view. This example makes that quantitative: it builds a 2-mirror RC and a 3-mirror TMA at the same aperture and f/#, optimises BOTH over the same wide field, and compares the residual wavefront error vs field. The physics: an RC nulls 3rd-order spherical AND coma with its two conics, so it is excellent on-axis -- but it has NO degree of freedom left for field astigmatism, which grows ~field^2. A TMA's third conic supplies exactly that DOF (a Korsch anastigmat nulls spherical + coma + astigmatism). So as the field opens up the 2-mirror hits a "wall" and the 3-mirror sails past it.

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    example_tma_widefield

## What a run gives

Optimized over the same 20′ field at f/8, the 3-mirror design holds 0.0613 waves at 15′ (0.1200 at 20′), while the 2-mirror, its two conics spent across the whole field, sits at 0.4519 waves even on axis: the third conic is what buys the field.

## Source

Generated 2026-10-26 from the header of `example_tma_widefield.m` and the mmacos manual's run of this template (Part II, macos 1b2ff44e); this README is the template's own now -- edit it with the code.
