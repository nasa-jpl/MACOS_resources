# sz_tma — Sphere + Zernike unobscured three-mirror telescope (e5mono-derived)

A wide-field design strategy where GEOMETRY and CORRECTION fully decouple: 0th-order layout = three BASE SPHERES (Kc=0). The sphere radii set first-order power (f/#, EFL); the fold tilts unobscure the beam and keep it compact. No conics -> none of the conic/ telecentric coupling that bogs down a Korsch search. CORRECTION = Zernike departures on each mirror, optimized by MACOS's native multi-field design optimizer (CALIB OptZern). Because the departures are tiny sag perturbations they don't move the chief-ray geometry, so once the layout meets packaging the optimizer cannot break it.

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    sz_tma

## What a run gives

The Zernike departures take the all-sphere layout to 0.0054 waves at the field center, diffraction-limited, but over the full ±2′ field the worst point is 4.1915 waves and the area-weighted mean 2.3925: on this 8 m layout the 15 modes per mirror do not hold the field.

## Source

Generated 2026-10-26 from the header of `sz_tma.m` and the mmacos manual's run of this template (Part II, macos 1b2ff44e); this README is the template's own now -- edit it with the code.
