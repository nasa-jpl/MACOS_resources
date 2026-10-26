# tma_offaxis — Unobscured off-axis convex-secondary TMA

j18mono-geometry 3-mirror anastigmat (M1 concave, M2 CONVEX, M3 concave, f/20): exercise the convex-secondary seidel_seed fix (K=0 seed + correct unfolded focus), conic-optimize to a diffraction-limited on-axis anastigmat, then take it OFF-AXIS as an eccentric-pupil section, clear the beam, and re-balance over a 2-D field.

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    tma_offaxis

## What a run gives

The convex-secondary seed conic-optimizes on axis to 0.0016 waves; taken off axis by 6.399 m and re-balanced over a 3 x 3 grid on the ±2′ field, its worst field is 0.077 waves at 2.3 µm, just outside λ/14, which the run prints as a residual. Its all-sphere baseline line reads 0 waves, which an uncorrected sphere cannot be; not traced.

## Source

Generated 2026-10-26 from the header of `tma_offaxis.m` and the mmacos manual's run of this template (Part II, macos 1b2ff44e); this README is the template's own now -- edit it with the code.
