# dyson5 beat 2c -- the propagation twin (2026-10-01, TO lane on Fable)

Brief: `macos/BRIEF_to_dyson5.md` addendum ("back every ray metric with a
PROPAGATION run"), on the engine with CC's chord-ruled grating directions
(799498b) and medium-aware kernels (fcd85d7).  Gates on it: tSpectrometerRx
2/2 (chain default 'planes', CC 264711f), tGratingImmersed, tGlassDispersion.

## Delivered

| item | state |
|---|---|
| `spectrometer_rx(..., 'terminal','farfield','L_ref',L)` | FP_return / ExitPupil (reference sphere, radius L, centred on the chief focus) / FPA -- the Rx_Cass_FarField idiom, posed on the chief |
| `design/src/spectrometer_wave.m` | per (slit x, lambda): stop first, exact chief, `set_xp` re-pose + FPA vertices on the chief pierce, `complex_field`, PSF centroid / widths / ensquared, SRF_wave & CRF_wave, energy; grid index 1 = X (slit), 2 = Y (dispersion), centre N/2+1 |
| runner stage `s2w` (opt-in, model 512) | order-0 Offner VALIDATION + both forms at 3 x 3; record `dyson5_s2w.txt` |
| `tests/tGratingOpl` (SUITE_FAST) | eikonal-consistency gate: order 0 PASS (8e-11 m), order -1 FAIL today (4.06 waves) |

## Why not prop_layout's exit pupil

The Offner is telecentric: the convex grating sits at M3's focal distance, so
its image (the exit pupil) is at infinity -- FEX returns a crossing 484 m PAST
the focus, and the far-field terminal's reversed rays never reach it (every
ray lost at the ExitPupil element).  A reference sphere of chosen radius L
centred on the chief's focus, vertex upstream, is all the far-field leg needs;
the twin poses it itself.  Convention pinned from Rx_Cass_FarField: psi along
the beam toward the focus, KrElt = -L, zElt = L, centre = vertex + L psi.

## Validation: the order-0 relay

Offner with the grating at order 0 (a pure concentric relay), slit centre,
1 um: pupil OPD on the sphere 7.8e-11 m; PSF centroid offset 0.019 px before
and 0 after the centre-pixel fix (N/2+1 -- the half-pitch told which); rms
widths 0.52 px (the Airy rings), **ensquared energy in one 18 um pixel
0.944**, CRF 1.0x.  The runner's s2w validation row repeats it at the band
centre (1.44 um): offset (-0.0005, -0.0005) px, EE 0.853 (a bigger Airy
disc, lambda F = 4 um), CRF 1.04 px.  The terminal reproduces the Airy spot.

## ENGINE FINDING #3 (CC's lane): the grating's optical-path jump

At order -1 the same deck's rays converge to 0.05 um at the FPA (and match the
exact chain to 1e-9 m per ray, tSpectrometerRx), yet the pupil OPD the engine
reports on the reference sphere is **4.06 waves rms** -- a tilt plus a
coma-like residual (1.4 waves after the tilt).  The PSF is displaced from
the chief by an offset exactly proportional to lambda (Offner 1.26 / 4.76 /
8.26 px at 380 / 1440 / 2500 nm = a fixed OPL tilt; Dyson -0.49 / -1.75 /
-2.94 px) and blurred (SRF 13 px, EE 0.5 %).  Rays and path lengths disagree.

Mechanism, read in `Snells_Law_Grating`: the OPL jump is
`dL = (nb r - na i) . rho_prj` with `rho_prj` the hit vector projected into
the LOCAL tangent plane, i.e. `(m lambda/d)(s0 . rho_prj)`.  The phase of
equidistant groove PLANES is the groove count, `(m lambda/d)(s0 . rho)`, with
`rho` the hit vector from the vertex along the fixed ruling direction `s0`.
The two differ by `(m lambda/d)(rho.N)(s0.N)` ~ `(m lambda/d) rho^3/(2R^2)`:
cubic in the footprint, zero on a FLAT grating (why every air fixture was
blind), 12 waves at the Offner grating's 45 mm footprint and R = 250 mm.

**Engine-free confirmation** (`scratchpad/opl_chain.m`, the chain's own OPL
on the same reference sphere, 441 rays): chord-ruled groove phase
`(s0.rho)` -> **0.0004 waves rms**; local-projection phase `(s0.rho_prj)` ->
**4.9 waves rms** (the engine's 4.06, same class, different ray weighting).

**Fix candidate:** `dL = Order*lambda/RuleWidth * dot(s0, rho)` with `s0` the
unit rule direction in the vertex tangent plane (what `DPERPUVEC` already
produces), `rho` from the vertex, plus the non-grating eikonal part as is.
**Gate:** `tGratingOpl` goes green with no test change; `spectrometer_wave`'s
order -1 PSF then centres on the chief.

## Status of the wave numbers

`dyson5_s2w.txt` carries the order -1 twin for both forms as measured today:
those rows are the OPL defect, not the design, and are labelled so in the
record.  The ray-side scorer (s2) is unaffected -- it never uses OPL.

## Next

When #3 lands: re-run `s2w` (no code change), compare wave vs ray centroid
maps (expect < 0.01 px where the PSF is symmetric), SRF/CRF from the PSF, then
the slit-width diffraction loss (a far-field leg from a rectangular slit
aperture to the grating plane, against the sinc^2 closed form).  Beat 3: the
Dyson departure ladder under the scorer.
