# afocal4 S4b — one-page status

**The S4 trade is retracted on packaging and re-derived. Every number below is engine-truth
from a committed prescription.** Detail: `RESULTS.md` §S4b. Reproduce: `afocal4_s4b` and
`afocal4_ladder('prefix','b_')`.

---

## What was wrong

The S4 designs put the collimator, the field mirror, the interface pupil and the whole
instrument behind the pupil **in front of the primary**, inside the incoming beam. Sky is
at −z, so behind M1 is +z: his three-mirror parent reads **z_M3 = +640 mm**, the delivered
S4 rung reads **−442 mm**. One extra mirror flips the parity of the back end, and neither
the S3 packaging check (train length, incidence angles, self-obscuration) nor the S4 gates
looked at instrument-volume placement at all.

## The buildable trade

Two basins, swept with the same machinery, merit and gates; every point solved from two
seeds; the constraint enforced as a **wall** in the closure, never as a merit term.

| iface mm | basin 1 — his front end |  |  | basin 2 — image behind M1 |  |  | max instr Ø |
|---|---|---|---|---|---|---|---|
| | WFE nm | blur µm | breath % | WFE nm | blur µm | breath % | (basin 2) |
| 50 | 11 776 | 759 | 0.247 ✓ | 14 481 | 238 | 1.371 | 0 mm |
| 90 | 5040 | 872 | 0.113 ✓ | 17 066 | 191 | 0.561 | 0 mm |
| 140 | 4591 | 1074 | 0.082 ✓ | 16 367 | 153 | 0.086 ✓ | 71 mm |
| 220 | 8928 | 800 | 0.098 ✓ | 12 524 | 152 | 0.118 ✓ | 210 mm |
| **343** | **266** | 761 | 3.875 | 10 521 | **160** | **0.081 ✓** | **464 mm** |
| target | 71 | 47 | 0.4 | 71 | 47 | 0.4 | — |

## What the constraint cost, against the free curve

| at 140 mm, +0.6° | WFE nm | blur µm | breathing % | wander µm | M |
|---|---|---|---|---|---|
| *S4 rung 3 — NOT BUILDABLE* | *9600* | *167* | *0.113* | *171* | *30.0156* |
| S4b anchor (his front end held) | **3451** | 1141 | 0.046 ✓ | 1159 | 29.9805 ✓ |
| S4b variant (image behind M1) | 16 760 | **149** | 0.149 ✓ | **154** | 30.0047 ✓ |

**The constraint does not cost image quality — it splits the S4 design's performance and
forces a choice.** Compliance moves the fourth mirror off the intermediate image, where it
gains a footprint (its conic does wavefront work) and loses the field conjugate (it stops
doing pupil work). Putting it back on the image requires a 4.5% slower secondary to push
the image 900 mm behind M1, which costs length: **2.0 m envelope at 140 mm, 2.9 m at
343 mm, against his 1.7 m.**

**The sharpest form of the answer** is the 343 mm row-pair, the only standoff where the
package closes around a real instrument: a fourth mirror that has gone to a **flat** gives
266 nm of wavefront and his three-mirror's 3.9% breathing; a real fourth mirror gives
160 µm blur and 0.081% breathing for 10.5 µm of wavefront. **Pupil control costs a factor
of 40 in wavefront; declining it returns you to the telescope Rodgers already has.**
Unconstrained, the same exchange cost a factor of 6. No target is met anywhere on either
buildable curve.

## The folded demonstration

`afocal4_b_final_folded.in`, from `afocal4_b_final.in`, both committed.

* **Null, measured**: WFE agrees to **4.8e-12**, blur to 2e-14, wander to 3e-14,
  magnification to 2e-16 — and unchanged when the fold's aperture is quadrupled.
* **Package**: interface pupil at [+0.304, −0.004, +0.614] m, the stated 1000 mm envelope
  running to [+1.304, −0.017, +0.614] m, z-slab **+0.464 … +0.764 m, entirely behind the
  primary**, closest approach of any other traced bundle **+137 mm**.
> **DATED CORRECTION (2026-08-31).**  The interface-pupil coordinates and the
> instrument z-slab quoted above for the FOLDED deck do not reproduce from the
> committed file.  Measured by `packaging/check_record` on
> `afocal4_b_final_folded.in`: interface pupil **[+0.2483, −0.0051,
> +1.3782] m**, envelope ending **[+1.1990, −0.3151, +1.3782] m**, z-slab
> **+1.2282 … +1.5282 m**; the fold and the interface plane are both at
> z = +1.3782 m.  The statement the numbers support is unaffected — every z is
> positive, so the instrument is entirely behind the primary — and the
> fold-is-null result is a measurement of a different quantity.  The recipe
> itself is superseded (see the head of this section), so this corrects the
> historical record, not a live design.
>
> *(Signed off by Dave 2026-08-31 via CC; `macos/BRIEF_afocal4_wall.md` §7.)*

* Renders: `afocal4_b_layout_{unfolded,folded}.png` (yz + xz with the envelope drawn);
  deck-grade `rodgers2_final_{layout,folded,trade}.png`.

## The feasible window

The three clauses do not bind in the same place.

* **`z_M3 ≥ 500 mm`** binds nowhere once the seeder is compliant — every delivered point
  clears it by 0 to 1900 mm.
* **Fold daylight** binds at the short end: at 50 mm there is **−2.7 mm** of gap at every
  station, so no fold fits and the pupil cannot leave the beam.
* **Instrument volume** sets the window everywhere else. The fold needs ~20 mm of lever
  arm; what is left of the standoff becomes the pupil's lateral offset once folded, and the
  largest instrument that clears the near-axis bundles is `2 × (offset − beam radius)`.

**For a 300 mm instrument the window is `iface ≳ 250 mm`** — his own three-mirror, at
344 mm, admits 344 mm. Below ~90 mm nothing fits at all. **The pupil metrics want a short
standoff and the package wants a long one**, which is a second exchange rate crossing the
first, and it is the question the operating point and the instrument envelope now pose
jointly rather than separately.

## Also closed

* **The Mersenne fails structurally**: its second confocal pair lives inside the M1–M2
  space, M3 at −0.542 m and M4 at **−0.942 m**. Already closed on wavefront in §4; now
  closed again on packaging, independently.
* **Rung 4's rigid bodies bought nothing** — bit-identical to rung 3, against 0.4% in S4
  and 25% in his three-mirror.
* **The buildable four-mirror wants his coldstop tilt**: refit tilts 0.0° / 4.04° / 4.52°
  against his hand-tuned 0° / 4.29° / 3.58°. The unbuildable one wanted twice that.

## Open for Dave and Mike

The operating point and the instrument envelope, which the packaging clause makes one
question rather than two. Both trade curves are the deliverable; neither is a
recommendation.

## Caveats stated rather than smoothed over

* The anchor and variant solves ended on `exitflag 3` at 115 and 77 evaluations —
  converged on the merit's own tolerance, in local basins, reported as such.
* At the φ₄ → 0 trade point the pupil metric is **ill-conditioned** (its exit pupil sits at
  the interface plane); a fold that is an exact isometry moves its breathing 6×. Read that
  row's pupil column as the three-mirror's, which is what it is.
* Basin 2 was found by hand-seeding, not by the solver. There may be others.

> **S4c CLOSES THE FIRST TWO CAVEATS (2026-08-04; `STATUS_S4C.md`).** The `exitflag 3`
> stops were a 17%-low finite difference, not convergence: re-solved from three seeds at
> ~1900 evaluations per point, basin 2's wavefront column improves by 0.02–10.8%
> (12 915 / 17 063 / 15 928 / 12 286 / 10 407 nm) and every conclusion above stands. The
> 90 mm design turns out to sit on the packaging wall to the millimetre rather than at a
> minimum. The φ₄ → 0 conditioning caveat is unchanged.
