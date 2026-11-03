# coro_planet_demo — Illustrated coronagraph + off-axis planet injection

`coro_planet_demo()` runs the canonical CoroExample recipe through the mmacos framework and writes two 4-panel figures (matching the CoroExample.jou "pgp 4" plot pages) into the example folder: coro_surfaces.png the coronagraph surfaces (jou: int 1/6/11/16) [entrance pupil | star occulted at FPM | pupil at Lyot | star at det] coro_planet.png the planet injection (jou: ffp + add) [suppressed star | planet alone | star+planet | difference]

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    out = coro_planet_demo()

## What a run gives

The template prints no number: its product is two pictures. The surfaces page is sound (the Lyot pupil's bright rim is the mask working). The planet page is not: asked for a planet at sky-angle cosines [+2e-5 +2e-5], it composes a shifted, clipped speckle field rather than a planet image; a known defect.

## Source

Generated 2026-10-26 from the header of `coro_planet_demo.m` and the mmacos manual's run of this template (Part II, macos 1b2ff44e); this README is the template's own now -- edit it with the code.
