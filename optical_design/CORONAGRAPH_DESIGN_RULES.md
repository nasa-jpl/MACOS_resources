# Coronagraph Instrument Design Rules (Lyot-type, two-DM wavefront control)

> **Purpose.** Design rules for a coronagraph treated as a telescope back-end instrument,
> for the MACOS design layer (coronagraph-driven Sprint 1). The architecture below is the
> one you specified; it is the canonical Lyot-type layout with sequential-DM wavefront
> control, and it matches the Roman Coronagraph Instrument (CGI), the JPL High-Contrast
> Imaging Testbed (HCIT), and university testbeds (SCoOB, THD, SEAL). Rules are stated as
> layout constraints the model must satisfy, with the quantitative core (the two-DM
> Talbot spacing) derived in §4. References in §10.

---

## 1. Plane sequence (your layout, mapped to named planes)

The instrument alternates **pupil planes** and **focal planes**, each pupil↔focus hop
performed by one off-axis parabola (OAP) acting as a Fourier-transforming relay.

| # | Element | Plane type | Conjugate to | Function |
|---|---|---|---|---|
| 0 | Telescope focus (instrument input) | focal | star image | hand-off from telescope |
| 1 | OAP1 | — | — | collimate FP → pupil |
| 2 | **Apodizer** | **pupil** | telescope exit pupil | shape pupil amplitude/phase (diffraction control) |
| 3 | OAP2 + OAP3 (pair) | — | — | pupil → focus → pupil relay (4-f) |
| 4 | **DM1** | **pupil** | telescope exit pupil | phase control (and often fine-steering) |
| 5 | **DM2** | **near-pupil**, at ≈ Z_T/4 | — (deliberately out of pupil) | phase+amplitude control (Talbot mixing) |
| 6 | OAP4 | — | — | pupil → focus |
| 7 | **Focal-plane mask (FPM)** | **focal** | star image | reject on-axis starlight (occult / phase) |
| 8 | OAP5 | — | — | focus → pupil |
| 9 | **Lyot stop** | **pupil** | telescope exit pupil | block FPM-diffracted light at pupil edge |
| 10 | OAP6 | — | — | pupil → focus |
| 11 | **Detector** | **focal** | star image | science image / dark hole |

Active elements live at conjugates: **apodizer, DM1, Lyot stop at pupil conjugates; FPM
and detector at focal conjugates; DM2 deliberately off-pupil** (§4). This is the single
most important structural rule — every powered/active element must sit at (or, for DM2,
at a controlled distance from) a plane conjugate to the telescope exit pupil or focus.

**Ordering note.** You place the apodizer *before* the DM relay. Roman CGI places the
apodizer (shaped pupil) *after* the DMs, in a filter wheel just downstream of DM2. Both
work; the choice affects how wavefront control interacts with the apodization (DMs
upstream of the apodizer see the unapodized beam). Keep it deliberate and consistent in
the model.

---

## 2. The pupil↔focus relay principle (4-f backbone)

Each pupil-to-pupil hop is a **4-f relay**: pupil at the front focus of OAP_a → common
focus → pupil at the back focus of OAP_b. Properties the layout must honor:

- **Pupil magnification** between two relay OAPs is `M = f_b / f_a` (ratio of focal
  lengths). Use this to match the beam to each device's physical size (DM actuator array,
  mask diameter, Lyot stop). Size every device in the model from this chain.
- **Beam f-number** in each focal plane is set by the OAP focal length and the collimated
  beam diameter: `F/# = f_OAP / D_beam`. Slower (larger F/#) beams ease fabrication and
  reduce OAP aberration (§3) at the cost of length.
- A focal plane only exists where the beam is converging (between the paired OAPs and at
  FPM/detector); a pupil only exists in collimated space (apodizer, DM1, Lyot). Do not
  place a pupil-plane device in a converging beam or vice versa.

---

## 3. OAP relay design rules

OAPs are used because they are **all-reflective** (no chromatic aberration) and give an
unobstructed beam. Their cost is **off-axis aberration**:

- A parabola images parallel-to-axis beams perfectly, but an off-axis (field) beam
  suffers **coma that grows with off-axis angle**; astigmatism and field-dependent
  distortion follow. Keep off-axis angles small.
- **Anti-symmetric pairing cancels relay aberration.** Pair the two OAPs of a relay with
  **equal focal lengths and equal-but-opposite off-axis distances**, symmetric about the
  intermediate focus. The second OAP's aberration then cancels the first's, leaving only
  surface-figure error. (This is the GRAVITY M5/M7 rule; use it for the OAP2+OAP3 pupil
  relay and ideally throughout.)
- **Beam walk.** As line-of-sight (pointing/jitter) or wavelength changes, the beam
  footprint *walks* across each optic. Beam walk on a non-pupil optic imprints a
  low-order phase mode and, chromatically, an uncorrectable amplitude error. Mitigate by
  (a) keeping active optics at pupil conjugates, (b) oversizing optics for clearance, and
  (c) keeping the beam stable to the micron level across critical surfaces (Roman holds
  the FSM-relay beam to micron-level walk).
- **Polarization aberration.** Every oblique reflection adds retardance/diattenuation
  that scales with **angle of incidence (AOI)** and depends on coating. This is a hard
  floor for 1e-10 contrast. Rules: minimize AOI (testbeds keep max AOI ~12° on the
  worst OAPs), minimize the number of oblique reflections, keep AOIs uniform across the
  beam, and consider splitting polarization into two channels. Budget polarization
  aberration explicitly; it does not respond to scalar DM control.
- **Small science field helps.** Coronagraph FoV is tiny (<~1 arcsec), so simple OAPs
  suffice — the design is driven by *pupil* quality and stability, not field correction.

---

## 4. Two-DM wavefront control and the quarter-Talbot rule (quantitative core)

**Why two DMs.** A single DM at a pupil corrects **phase only**. Real contrast is limited
by **amplitude** errors (reflectivity non-uniformity, out-of-pupil surface errors that
Fresnel-propagate into amplitude). One DM can null a *one-sided* (half) dark hole; a
**symmetric (full) dark hole requires simultaneous phase + amplitude control**, hence two
DMs in series.

**The Talbot mechanism.** Consider a weak sinusoidal phase ripple of spatial period `Λ`
(amplitude ε) at a pupil: `φ(x) = ε cos(2πx/Λ)`. After Fresnel propagation a distance `z`,
each spatial-frequency component acquires the Fresnel phase `exp(−iπλz/Λ²)`, and the pure
phase ripple develops an **amplitude** modulation:

```
amplitude modulation ∝ ε · sin(π λ z / Λ²)
```

Define the **Talbot length** `Z_T = 2Λ²/λ`. Then:

```
full phase→amplitude conversion when  π λ z / Λ² = π/2
  ⇒  z = Λ²/(2λ) = Z_T / 4        (the quarter-Talbot distance)
```

So a DM placed at **z = Z_T/4 from the pupil** can inject, via its (phase-only) surface,
a correction that arrives at the next pupil as **amplitude** — which is exactly what is
needed to cancel amplitude errors. DM1 (at the pupil) handles phase; DM2 (at ≈ Z_T/4)
provides the phase↔amplitude leverage. (Physical-optics models confirm complete
phase→amplitude conversion at 0.25 Z_T.)

**Frequency dependence — why it's an optimization, not a single number.** `Z_T` depends
on `Λ`, so a *fixed* DM separation is the exact quarter-Talbot distance for only **one**
spatial frequency. The dark hole spans a band of spatial frequencies, so in practice the
separation is **optimized numerically** to best mix phase↔amplitude across the controlled
band. Map angular position to pupil period via `Λ = D_beam / α` for a speckle at
`α·(λ/D)`; substitute into `z = Λ²/(2λ)` to get the quarter-Talbot distance for that
frequency. Choose a representative `α` (often near the high end of the dark hole) and
refine by simulation.

**Actuator count sets the dark-hole outer working angle.** With `N` actuators across the
beam (pitch `d`, `D_beam = N·d`), the highest controllable spatial frequency is
`1/(2d) = N/(2 D_beam)`, giving:

```
OWA ≈ (N/2) · (λ/D)
```

IWA is set by the coronagraph (FPM/apodizer), typically `~2–4 λ/D`. So the controllable
dark hole runs roughly `IWA … (N/2)λ/D`.

**Real-world anchor numbers** (HCIT / Roman-class kilo-DMs): **~46–48 actuators across
the beam, ~1 mm inter-actuator pitch, ~1 m DM1–DM2 separation** (semi-analytic optima of
~1–2 m appear in the literature for these beam sizes). Use these to sanity-check a layout:
if your model's Z_T/4 lands far from ~1 m for a ~mm-pitch DM, recheck `D_beam` and the
target spatial frequency.

---

## 5. Apodizer plane

Shapes the pupil so the on-axis PSF has a region of deep diffraction suppression. Families:

- **APLC** (apodized-pupil Lyot): grayscale pupil apodizer + occulting FPM + Lyot stop;
  apodizer and Lyot stop **jointly optimized**.
- **SPC / shaped pupil** (binary mask): band-limited in spatial frequency; deeper local
  cancellation at a given IWA; shares the path/WFC with HLC in Roman.
- **PIAA**: achieves apodization by **aspheric remapping mirrors** instead of an
  absorbing mask (high throughput), at the cost of strong off-axis beam-walk aberration
  that a following corrector/2nd PIAA must undo.

Rule: the apodizer sits at a pupil conjugate and must be **registered to the pupil** (and
to any obstruction/spider features) tightly; misregistration leaks starlight.

---

## 6. Focal-plane mask (FPM)

At a focal conjugate; rejects on-axis starlight so the Lyot stop can block it. Types:
hard occulting spot (e.g. radius ~2 λ/D), **band-limited mask**, **vector vortex** (VVC,
charge 4/6 — intrinsically achromatic), hybrid Lyot (HLC), or the APLC occulter. Design
rules:

- IWA scales with FPM size/charge; smaller IWA ⇒ more starlight leakage ⇒ tighter
  tolerances.
- Make the FPM **reflective** so the rejected stellar core can feed a **low-order
  wavefront sensor (LOWFS)** — standard for maintaining the dark hole during observation.
- For larger field stops, include the **focal-plane stop explicitly** in the model;
  field-stop diffraction is significant well outside the science FoV.

---

## 7. Lyot stop

At the pupil conjugate downstream of the FPM; blocks the starlight the FPM diffracted to
the pupil edge / around the central obstruction. Sizing rules:

- For a **monochromatic APLC**, the optimal Lyot stop **matches the telescope aperture**
  (180°-rotated for apertures lacking circular symmetry), padded slightly only for
  alignment tolerance.
- For **broadband / image-constrained** designs, **oversize the central-obstruction
  replica** significantly and undersize the outer edge; jointly optimize with the apodizer.
- Throughput vs. contrast is the core trade: a tighter Lyot stop suppresses more diffracted
  starlight but costs planet throughput. Carry both Lyot geometry and apodizer as coupled
  free variables.

---

## 8. Detector plane

Final OAP focuses to the science detector. Sample at **Nyquist or finer**: pixel pitch
`p` and final beam `F/#` satisfy `λ·F/# ≥ 2p` (so the PSF core spans ≥2 pixels). Final
beams are slow (testbeds run ~F/22–F/48 at the science focus) to give comfortable
sampling and relax detector pitch.

---

## 9. Contrast-limiting effects to budget explicitly

- **Polarization aberration** (§3) — scalar DMs cannot correct it; budget separately.
- **Chromatic / out-of-pupil surface errors** — surface error on a non-pupil optic
  becomes a **wavelength-dependent amplitude** error (Talbot) that two DMs correct only
  partially; this drives per-optic surface-figure requirements, weighted toward spatial
  frequencies *inside* the control band (IWA…OWA).
- **Beam walk / stability** — pointing, jitter, and thermal drift; mitigated by pupil
  conjugation, FSM, and LOWFS.
- **Co-design principle** — the coronagraph (apodizer/FPM/Lyot) and the wavefront-control
  system are **inseparable**: design the masks for the target contrast/IWA/throughput
  assuming perfect optics, then size the DMs to recover the contrast that real
  amplitude/phase errors remove. Do not optimize masks and DMs independently.

---

## 10. Control layer (brief — deferred to FALCO in this plan)

The optical layout above is the *plant*; the controller is separate. Standard methods:
**Electric Field Conjugation (EFC)**, **stroke minimization**, and **speckle nulling**
for high-order control (HOWFSC, science-camera-based), plus **LOWFSC** (FPM-reflected
starlight) for fast low-order drift. Modal DM control belongs in the inner loop;
actuator-level EFC integrates via FALCO. This document fixes the *layout rules* the
controller assumes (pupil conjugation, DM count/spacing, dark-hole reach); it does not
specify the control algorithm.

---

## 11. References

- Pueyo et al. 2009, "Optimal dark hole generation via two sequential deformable mirrors,"
  ApJ 692, 1834 — two-DM phase+amplitude formalism.
- Pueyo et al. 2011, "Optimal Dark Hole Generation via Two Deformable Mirrors with Stroke
  Minimization," arXiv:1111.5111.
- Give'on, Kern, Shaklan et al. 2007 — EFC; Bordé & Traub 2006 — single-DM half-dark-hole
  limit (Talbot).
- Shaklan & Green 2006, "Reflectivity and optical surface-height requirements in a
  broadband coronagraph," Appl. Opt. 45, 5143 — out-of-pupil/Talbot amplitude errors,
  surface requirements.
- Mazoyer et al. 2017, MNRAS 469, 218 — two-DM optical configuration vs. dark-hole depth
  (optimum DM separations).
- Zimmerman et al. (SPLC), JATIS — shaped-pupil Lyot, apodizer+Lyot joint optimization.
- N'Diaye, Soummer et al. — APLC for arbitrary apertures (Lyot sizing rules).
- WFIRST/Roman CGI reports (arXiv:1305.5422, 1503.03757) — flight architecture: FSM at
  pupil, OAP relay to DM1, two-DM WFCS, apodizer wheel, reflective FPM → LOWFS, Lyot.
- Ashcraft et al. (SCoOB, arXiv:2406.18886, 2509.0287x) — testbed: 8 OAPs, FSM, DM, VVC,
  Lyot; polarization-aberration end-to-end model (AOI rules).
- GRAVITY beam-stabilization (arXiv:1212.5133) — anti-symmetric OAP-pair aberration
  cancellation.
- Anand/Douglas et al. POPPY (arXiv:1806.06467) — Talbot phase→amplitude at 0.25 Z_T.
