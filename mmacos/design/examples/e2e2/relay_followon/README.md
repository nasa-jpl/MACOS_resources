# relay — parked follow-on work

**Not part of the e2e2 flow.** The telescope closes out at stage 2
(Dave, 2026-08-02: "close out the telescope flow at S2, relay
separately"). This directory holds the relay driver and, more usefully,
the record of two failed attempts, so the next pass starts from what is
already known rather than rediscovering it.

## Why it is parked rather than finished

The relay was originally in the flow to rescue the wavefront. It is not
needed for that any more: at the adopted 0.4° box the folded telescope
reaches **20.4 nm** (+LS tip/tilt) / **30.0 nm** (centroid), inside the
35.7 nm diffraction bar, with no relay at all. So the relay's job is now
to **feed instruments**, and it should be designed to *its own*
requirements instead of to a wavefront deficit — a different problem,
with different acceptance criteria, and no reason to solve it inside the
telescope flow.

## What was tried, and how each failed

Both attempts failed the same way — **a relay was scaled from another
design instead of being laid out for this one** — and the second did it
inside the branch adopted to avoid the first.

### (1) Offner, scaled by aperture — killed the trace

e2e's concentric 1:1 Offner scaled ×3/5 along with the aperture:
R 2.0 → 1.2 m, ring radius h 0.25 → 0.15 m. Result: 318 rays lost to
**surface miss** at the convex stop mirror, then CALIB singular.

A relay is sized by the **image** it accepts, not by the aperture:

| | image half-height = EFL·tan(half-field) |
|---|---|
| e2e (72 m, ±2′) | 0.042 m — h = 0.25 m, ample |
| e2e2 (60 m, ±0.3°) | **0.314 m** — h = 0.15 m, **0.48×**. Dead. |

The field went up 9× in angle and the image with it, while the aperture
went *down*. For h ≳ 0.35 m with the Offner's `h << R/2` concentricity
margin, **R ≳ 2.1 m and realistically 2.5–3 m** — a concave mirror of
1.25–1.5 m radius, comparable to the 3 m primary.

### (2) Bench (zigzag), scaled by legs — 1577 nm

Adopted because a bench relay has no ring-radius constraint, so it does
not have to grow with the image. It didn't fail geometrically — it
failed on **first order**. e2e's legs (1.5, 2.0 m) scaled ×3/5 to
(0.9, 1.2) were laid out for e2e's conjugates and its ±2′ field, not for
a 60 m EFL at ±12′. Result: **1577.6 nm** with the corrector flat,
against 20.4 nm with no relay at all.

## The field-corrector hypothesis is UNTESTED

Stage 2 established, by measurement, that its residual is field-varying —
astigmatism reversing sign across the field, z5 spread/mean **4.48** —
and that figure on the Korsch's three **pupil-conjugate** mirrors cannot
touch it, since a fixed figure subtracts the same map at every field.
The proposed lever is a mirror near a **focus**, where each field point
lands on its own patch of glass so a fixed figure acts field-dependently
(e2e's rule 11, "M4 near the focus is the reflective field-corrector").

That remains **unverified**. On the broken bench relay the corrector
scored 7899.8 nm against 1577.6 without it, and the per-branch `min` rule
correctly kept the uncorrected design — but a corrector cannot be
evaluated on a relay that is itself 77× off the bar, and the astigmatism
spread/mean reading 5.06 against stage 2's 4.48 says nothing either way.
**Nothing here confirms or refutes the hypothesis.** Do not cite these
numbers as evidence against it.

## What a proper pass needs

1. **First-order layout for THIS system.** Collimator and camera focal
   lengths from the telescope's actual conjugates; legs and tilts sized
   to the ±12′ beam; not scaled from anything.
2. **A stated requirement.** What does the relay serve — field, pupil
   access, number of instrument ports, packaging envelope? The telescope
   no longer constrains it, so the requirement has to come from
   somewhere else.
3. **Then, and only then, the corrector experiment.** With a relay that
   works, put freeform on the focus-conjugate mirror and see whether the
   field-varying term moves. That is a clean test of the stage-2
   diagnosis and worth doing on its own merits.

`relay_followon.m` runs as-is against `s2_fold.{in,mat}` and carries the
full branch-point analysis in its `[0]` block, including option (1) with
the sizes it would need. It is kept runnable so the next pass can change
one thing at a time.
