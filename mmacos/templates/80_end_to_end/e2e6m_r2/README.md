# e2e6m_r2 — e2e6m round 2, R1: the DM-bearing coronagraph back end

Round 1's back end carried apodizer / FPM / Lyot only; its close-out named the missing capability and called the extension parameter work. This runner does that work: the CTB topology (8 OAPs + two flat-fold DMs, DST2R-class near-normal folds) instanced at observatory scale behind the 6 m telescope, built in METRES with `macos.design.Bench` and spliced with `macos.design.append_rx` so ONE train carries a telescope perturbation to a contrast number.

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    r1_backend()

## What a run gives

Round 2 hangs the coronagraph testbed's topology (8 OAPs, two DMs) at observatory scale off round 1's segmented telescope: the spliced train is 37 elements, all of the telescope's 983 passing rays reach the science focus, and the whole train fits a 7.451 m union inside the 8.0 m shroud. It reads round 1's committed telescope, the f/25.39 design, not a live rerun of round 1, which today departs from it.

## Source

Generated 2026-10-26 from the header of `r1_backend.m` and the mmacos manual's run of this template (Part II, macos 1b2ff44e); this README is the template's own now -- edit it with the code.
