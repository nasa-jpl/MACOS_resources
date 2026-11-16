# tma_3plus1 — 3+1 coronagraph front end -- unobscured conic TMA + M4 pupil relay

Three conic mirrors (j18mono-geometry Korsch, 6.6 m f/20, unobscured eccentric-pupil section) form the SCIENCE FOCUS -- the shared focal plane where instrument pickoffs live (the 3-mirror holds a 5'-diameter circular field at ~0.095 waves @ 1 um; see ../tma_centered). A 4th FIELD MIRROR just past that focus relays the CORONAGRAPH CHANNEL (a ~30 arcsec patch) to FP2 and -- the point -- relays the exit pupil to an accessible plane and FLATTENS it (the bare TMA pupil carries +0.7 mm defocus / 1.1 mm astig, macos.pupil_quality; the relayed pupil comes out ~10x flatter with M4 varying in the patch balance). A relay cannot carry the wide field: at EFL=132 m a 2.5' half-field walks +-96 mm across M4 -- wide field stays at the TMA focus, the relay serves the small-field channel.

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    tma_3plus1

## What a run gives

The 30″ coronagraph patch is balanced to 0.0337 waves worst, and the relayed pupil carries 0.199 mm of astigmatism against the bare TMA's 1.1 mm. The template reports its own limit: the worst AOI spread is 24.1 degrees, outside the 15-degree preference on M1 and M2, which its sibling `tma_3plus1_aoi_search.m` addresses.

## Source

Generated 2026-10-26 from the header of `tma_3plus1.m` and the mmacos manual's run of this template (Part II, macos 1b2ff44e); this README is the template's own now -- edit it with the code.
