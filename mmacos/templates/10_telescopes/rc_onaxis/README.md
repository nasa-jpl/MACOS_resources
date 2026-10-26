# rc_onaxis — Parameterized on-axis Ritchey-Chretien designer

The classical Cassegrain-form RC: a two-mirror APLANAT (zero 3rd-order spherical AND coma by construction), so it is diffraction-limited on-axis. Its field is limited by ASTIGMATISM, which two conics cannot null -- that limit is precisely why wide-field telescopes add a third mirror (a TMA). This driver builds the RC from your aperture + f-numbers, reports its first-order properties (including the central obscuration the secondary casts), shows WFE vs field, then field-balances the conics for a WIDER corrected field and saves that as the deliverable (better-across-the-field usually beats perfect-on-axis for a real instrument). This is the OBSCURED baseline. For the unobscured (coronagraph-feed) version that decenters to an eccentric pupil, see ../rc_unobscured.

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    rc_onaxis

## What a run gives

The aplanat is diffraction-limited on axis and out to 2.7 arcmin, where the astigmatism two conics cannot null crosses λ/14; balancing the radius and conic over ±4′ trades a little on-axis quality for a field diffraction-limited out to 3.3 arcmin. The secondary's central obscuration is 0.19 of the aperture.

## Source

Generated 2026-10-26 from the header of `rc_onaxis.m` and the mmacos manual's run of this template (Part II, macos 1b2ff44e); this README is the template's own now -- edit it with the code.
