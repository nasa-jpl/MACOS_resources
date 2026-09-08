# ep_dome_probe -- where must a sensitivity OPD be read?

Companion to `macos/REPORT_ep_dome_review.md` (2026-09-08).  On the e2e6m
imaging deck (`templates/80_end_to_end/e2e6m/s3_imager_full.in`, no exit-pupil
element) a segment-tilt sensitivity column was read at nElt-1 -- the powered
OAPim -- and came out as a one-signed dome.  The proposed fix read it at the
FocalPlane instead.  This probe reads the same perturbation at THREE surfaces
and adds a global-field-tilt test:

| DOF (1e-6 rad) | OAPim (nElt-1) | FocalPlane | exit-pupil sphere |
|---|---|---|---|
| Seg1 Rx | 8.0e-6 rms, one-signed | 4.1e-8, FLAT piston | 5.9e-7, bipolar ramp +/-1e-6 |
| Seg8 Rx | 7.7e-6, one-signed | 2.3e-6, FLAT piston | 6.0e-7, bipolar ramp |
| global field tilt | 1.9e-6 | 4.4e-10 (blind) | 1.4e-6 |

The focal-plane OPD is the path to each ray's landing point; a displaced
perfect image has equal paths, so that read is blind to tilt and turns a
segment tilt into a segment piston.  The exit-pupil sphere (the add_pupil pair
+ FEX) gives the textbook 2*alpha*half-segment ramp.  Read sensitivities at an
exit-pupil reference sphere; refuse (or place one) when the deck has none.

Files: `dome_probe.m` (writes `out/dome_probe.txt` + `.mat`),
`make_pupil_deck.py` (inserts the add_pupil pair into a bare-focal deck).
Needs the 2026-09-08 engine: Segment stops accepted and the element-STOP
source-frame handedness fix (without it `macos.stop(1)` on this left-handed
segmented deck mirrors the source grid and obscures 732 of 985 rays).
