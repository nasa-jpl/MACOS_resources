# Afocal Three-Mirror Telescope — First-Order Regression Fixture

Companion to `tma_fixture.md`, which covers the *focal* three-mirror case.  This one
exists because an afocal telescope is specified by two first-order conditions that a
focal fixture cannot exercise at all:

1. **the afocal condition** — the marginal ray leaves parallel (`u_out = 0`), so there
   is no back focus to check and no EFL to quote;
2. **the interface-pupil condition** — the chief ray images the stop onto a plane a
   specified distance past the last mirror, at pupil magnification 1/M.

`afocal_first_order` (mmacos `design/src`) traces the two paraxial rays that carry both.
This fixture pins it.

## The system

J.M. Rodgers' coaxial 30× afocal TMA, on-axis variant (`CentAfo_Coaxial_OnAxisFOV`,
2026-08-02).  EPD 1000 mm, λ = 1 µm, stop 50 mm ahead of M1, parabolic primary at
f/1.25, convex secondary, tertiary recollimating from the intermediate image.  The
transcription and its audit are in `mmacos/challenges/rodgers2/PACKET.md` §1.

| | M1 | M2 | M3 |
|---|---|---|---|
| R (m), magnitude | 2.500000 | 0.468780 | 0.580811 |
| convex | no | **yes** | no |
| conic K | −1.000000 | −1.782496 | −1.001754 |
| → next vertex (m) | 1.049239 | 1.689655 | — |

Radii are **magnitudes** with a separate convex flag, matching the MACOS emission
convention: every mirror is `KrElt = −|R|`, and convex versus concave is *geometry*,
never a radius sign.

## The assertions

| quantity | expected |
|---|---|
| `u_out` (marginal) | −3.1032482538e−06 rad — the afocal condition |
| magnification, chief | 30.000095942 |
| magnification, Lagrange | 30.002014049 |
| exit beam diameter | 0.033331095652 m |
| **exit pupil past M3** | **0.343362746564 m** |
| exit pupil diameter | 0.033333226732 m |
| intermediate image past M2 | 1.399265598708 m |

Values are carried at full double precision in the JSON, not rounded — a fixture asserted
to 1e−6 against a 7-digit expectation tests the rounding, not the model.

Reproduce to **1e−6 relative**.  The two magnifications are returned separately on
purpose: the Lagrange invariant makes them identical *only* while the train is afocal,
so their disagreement is an independent check that `u_out` really is zero.

## The witness

His coldstop sits **344.173 mm** past M3 — a number he placed by hand in CODE V, and an
input to nothing above.  The paraxial exit pupil lands **0.81 mm** from it on a 33 mm
beam.  That agreement is the check that the transcription, the sign conventions and the
paraxial model are all consistent; nothing in the fixture is fitted to it.

It also carries the single most important fact about this benchmark: **the three-mirror
already closes both first-order conditions.**  Any four-mirror form study run against it
has to justify the fourth mirror on aberration, not on first order — a study that closes
only the first-order conditions returns a flat.  See `mmacos/challenges/afocal4/
FORM_STUDY.md`.

## Gate policy

**Stop and fix, never widen.**  A drift here means the paraxial model, the convex-flag
convention or the stop handling has changed, and every afocal layout closed against it
is wrong by the same amount.

## Which variant, and why it matters

Use the **on-axis** variant, as here.  His later re-solves (`_NewConics`,
`TiltDecM2M3_`) moved R3 by 3.8% to recollimate the **real** marginal ray at f/1.25;
paraxially they leave 41 µrad of convergence and a 4.8% small exit beam.  Both are
correct answers to different questions — but seeding a first-order study from the
re-solved variant builds a 31.4× telescope.
