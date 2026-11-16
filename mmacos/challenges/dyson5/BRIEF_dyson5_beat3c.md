# dyson5 beat 3c -- the collisions brief on R4: apertures, clearance, the Offner re-posed, the package, sizes, renders, the twin on R4 (2026-10-01, TO lane on Fable)

Brief: `macos/BRIEF_to_dyson5.md` addenda 5, 6, 7.  Gates: tSpectrometerRx
4/4 (new case 'dyson_apertures': the declared apertures vignette NOT ONE
ray, which pins the aperture frame), tGratingOpl 2/2.

## Apertures are declared (addendum 6.1)

`spectrometer_geom` now measures every surface's FOOTPRINT over the
multi-field, multi-lambda bundle (slit centre + ends x band centre + edges x
chief + 2 rings) in the surface's APERTURE frame -- xObs written as global x
(projected into the vertex tangent plane), yObs = psi x xObs, which is what
`tracesub.F` builds (`zObs = psi; yObs = zObs x xObs`) -- and
`spectrometer_rx(..., 'apertures', true)` writes `ApType Circular, ApVec =
(radius + 5 mm, xc, yc)` on every optical surface with `xObs=` explicit.
Measured: 0 of 1258 rays vignetted on every deck (the stages assert it); the
engine's view_rx now draws real bodies -- the block is a cylinder of its
footprint, the grating a 270 mm cap, the meniscus a thin plate.

## Clearance is a number (addendum 6.2): `design/src/spectrometer_clearance.m`

For every LEG (slit -> first surface, ..., last surface -> FPA) against every
BODY the leg does not start or end on: the smallest distance from any ray
segment of the leg to the body's sampled surface (its aperture disc + 5 mm
mount, lifted onto the real sphere/plane; ~2 mm samples), minus the mount.
Bodies that are one physical part are grouped and never scored against their
own legs: the block's four surface records, the meniscus's two faces x two
passes, the Offner concave's two zones.  Two MECHANICAL bodies at the Dyson
face: the slit mask (64 x 4 x 1 mm plate on the slit) and the FPA package
(the 54 x 9 mm active area + 5 mm carrier, 10 mm deep behind the face, cold
shield height 0 by default).  Negative = blocked; every stage that emits a
deck prints the table (worst first) and FAILS on a negative entry.

| deck | worst pair | clearance |
|---|---|---|
| Dyson seed (s1) | returning beam (block sphere -> face out) vs FPA package | **+1.11 mm** |
| Dyson seed | incoming cone (face in -> sphere out) vs slit mask | +2.10 mm |
| Dyson R3 | same pair | +1.16 mm |
| Dyson R4 | returning beam vs FPA package (meniscus faces grouped as one part) | **+1.19 mm** |
| Offner, re-posed at 0.22 R | M3 -> FPA vs grating | **+10.24 mm** |

The Dyson's slit/FPA mechanics (addendum 6.4) are therefore MARGINAL as
drawn: +1.1 mm between the returning beam and a 5 mm carrier, with NO cold
shield -- any shield height fails the gate.  That is the review's fold-prism
answer, and it is now a parameter (`pkg_shield_m`) whose non-zero value
turns the gate red: the fold prism at the slit is R5's first item.

## The Offner, re-posed and solved (addendum 6.3)

At the seed's ring (6 mm) the Offner's slit->M1 and M3->FPA beams crossed the
grating body (CC's probe).  Moving the slit to 0.20 R still failed (the
beams clip the grating's mount by 4 mm); at 0.22 R = 110 mm the beams pass
beside it (+10.2 mm).  But the concentric Offner at that ring is astigmatic
(CRF 15 px), so `offner_solve.m` solves the classical corrections under the
ladder's operands: convex grating radius x 1.00340 (250.85 mm), second
concave zone radius x 0.95097 (475.5 mm) with its centre 0.29 / 0.15 mm
off.  Result (chain): keystone 0.0275 px, smile 0.0066 px, CRF 1.20 px, SRF
3.69 px, EE 0.225 -- a real, cleared layout; the spectral astigmatism (SRF)
is what an Offner-specific ladder would chase next.  The Offner runs at
F/2.8 (source cone 0.359 rad full) against the Dyson's F/1.8, stated in the
s1 record and the trade table.

## Element sizes (addendum 6.5) -- trade table columns

`dyson5_s3_trade.txt` gains block diameter and thickness, grating diameter
(and radius), meniscus diameter; length = slit plane to grating vertex.

## Renders of record (addendum 5)

`dyson5_view_figs` (CC) runs from s1 and s3 for every emitted deck:
`<deck>_view3d.png`, `<deck>_viewyz.png`; the mm-scaled chain sections
(`spectrometer_layout_fig`) stay as the 2-D complement beside them.

## The twin on R4 (addendum 7)

`P.twin_rung = 'R4'`: s2w re-emits the R4 deck and runs the propagation twin
on it.  **R4 ensquared energy WITH diffraction 0.747, geometric 0.759** --
the pair for the deck; SRF_wave 2.05 px, CRF_wave 1.22 px (ray 1.33).

**A finding to settle in beat 4.**  On the seeds the wave and ray centroids
agree to 0.001 px; on R3 they part by up to 0.030 px and on R4 by 0.122 px,
growing with wavelength (`dyson5_s2w_R3control.txt`, `dyson5_s2w.txt`).
`tGratingOpl` passes on this engine, so it is not the OPL defect.  The PSF
intensity centroid is the AMPLITUDE-weighted mean of the ray aberration and
the ray centroid is unweighted; R3 adds an off-centre block and R4 four more
oblique refractions, so the Fresnel transmission across the pupil is less
uniform where the offset is larger.  That is the likely mechanism, NOT a
proven one: beat 4 computes the amplitude-weighted ray centroid from the
engine's pupil amplitude and compares.  It matters: if real, keystone on
the detector (a centroid) is the wave number, up to 0.12 px against Joe's
0.1 px, while the ray-side keystone is 0.0026 px -- and the wave centroid
becomes the operand the optimizer should hold.

## Engine facts pinned

- Aperture frame: `xObs` as written (default = a cyclic permutation of psi:
  for psi = (0,0,-1) that is -x!), `zObs = psi`, `yObs = psi x xObs`; a
  circular aperture is `ApVec = (radius, xc, yc)` in that frame, a
  rectangular one `(x1, x2, y1, y2)`.  Always write `xObs=` when declaring a
  decentred aperture.
