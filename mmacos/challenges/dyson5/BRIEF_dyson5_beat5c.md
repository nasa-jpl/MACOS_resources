# dyson5 beat 5c -- the telescope at Jim's numbers: the three-mirror family exhausted, the two-mirror Schwarzschild images but does not package

Written for: CCL (review while Dave is away) and Dave.  Addenda 26-31 of
`BRIEF_to_dyson5.md`.  Runner stages `t3s` / `t3w` / `t3o` / `t3e` (the
three-mirror) and `t4` (the two-mirror modified Schwarzschild, TMS), knobs in
`dyson5_params.m`.  Records `dyson5_t3s_*`, `dyson5_t3w_*`, `dyson5_t3o_*`,
`dyson5_t4*`.  Every number below is the ENGINE's unless it says "chain".

## 1. The target (addendum 26, Jim)

Two or four spectrometer modules, each with its own telescope: 550 km, 30 m
GSD -> IFOV 54.5 urad, f 330 mm, D 183 mm at F/1.8.  The 3k module (54 mm
slit, CaF2 240 mm Dyson `size:F:240`, apparent pupil 41.5 m behind the slit)
sees 9.38 deg cross-track; the 1.5k module (27 mm slit, silica 130 mm Dyson
`size:D:130`, 24.8 m) sees 4.69 deg.  Both strips 0.3 deg along track.

## 2. The three-mirror (the template's coaxial-parent family) -- exhausted

* First-order screen (`tma_screen`, 5187 rows per module): rows package at
  9-10 deg offset (y2 0.30, t1 200 mm, +17 mm) with M3 at 0.44-0.50 of its
  radius -- unlike the EMIT 24.6 deg case.
* S1 y2 walk (R1 held): every step counts; at the packaging corner the
  on-axis parent is 255.4 nm (3k) / 78.1 nm (1.5k), no edge loss.
* The offset solves (S3 -> S4 -> S5 at 9 and 10 deg, addenda 27-28) leave a
  FIELD-CONSTANT residual -- the parent's own aberration off its axis: best
  3k S4 at 9 deg 2 944 nm with the clearance gate failing (+3.4 mm); 1.5k S5
  at 9 deg 10 882 nm.  CC's engine trace of the 9 deg decks: 6-65 px rms
  spots (addendum 29).  The family is exhausted; the three-mirror passes to
  CCMac by the design layer's Telescope class solved at the biased field.

## 3. The two-mirror modified Schwarzschild (Mouroulis & Green 2018 sec. 5.1)

Convex primary, concave secondary, the stop VIRTUAL at M2's front focal
point (telecentric output), an inverted telephoto.  Closed-form first order
`design/src/tms_paraxial.m` (flat field by equal radii) and
`tms_firstorder.m` (general: R1 eliminated by the EFL); the exact chain
`tms_geom.m` (telescope_geom's schema; the virtual stop = its object-space
image, a pass plane ahead of M1).  Paraxial checks: EFL 330.0000 mm, exit
chief slope <= 3e-17 at every spacing.

| rung (coaxial: the M1 obscures the image cone) | 3k: max / on axis px | 1.5k: max px | notes |
|---|---|---|---|
| seed (aplanat conics, d 165) | 11.98 / 0.51 | 3.21 | |
| R1a (+ M2 h^4) | 7.59 / 4.33 | 4.33 | converged; trades centre for edge |
| R1b (+ M2 h^6) | 7.55 / 4.36 | 4.36 | function-eval limit |
| **R2c** (+ d, R2, M1 h^4 h^6, Petzval row; EACH MODULE ITS OWN STRIP) | **4.26 / 2.28** | **1.10** (0.59 on axis) | converged; d on its 320 mm bound |

Engine = chain to <= 7.5e-13 m on EVERY ray (stepwise trace, below).  R2c
sizes: 3k M2 399 x 346 mm, 6.1 kg at 40 kg/m^2 (assumed lightweighted; 23.2
kg solid Zerodur), envelope 622 mm; 1.5k M2 372 x 345 mm, 5.4 kg (19.1),
620 mm.  Scaling check: the review's own TMS (f 24 mm, 3-6 um) at f 330 mm is
2.3-4.6 px -- R2c sits there.

## 4. Packaging -- the TMS does not package with an image at this scale

`design/src/tms_clear.m`: exact-ray clearances of the four leg/body pairs
(in x M2, in x IMG, M1->M2 x IMG, M2->img x M1), OI_CLEAR's disc model,
signed.  Coaxial R2c: in x M2 -214 / -230 mm, M2->img x M1 -106 / -107 mm.
* R2c's own shape, bias 0-30 deg x pupil decentre 0-200 mm: best -40 mm.
* Family-wide first-order screen (spheres, flat), d 50-320 mm x bias 0-40
  deg x decentre +-300 mm: ONLY d 320 mm at bias 40 deg, decentre +100 mm
  clears, by +9.0 / +9.2 mm -- the edge of the scan.
* The bias walk with the clearance WALL (R3w: bias 10 -> 40 deg, each step
  from the previous, the pupil decentre free, the wall weighted to dominate;
  every solve ran its 1000-iteration cap, exitflag 0):

  | bias | 1.5k: image px / clearance mm | 3k: image px / clearance mm |
  |---|---|---|
  | 10 deg | 254 / -148 (did not move from R2c) | 365 / -158 (did not move) |
  | 20 deg | 220 / -42 (did not move) | 342 / -53 (did not move) |
  | 30 deg | **232 / +5.1 PASS** | **221 / +4.9** (0.1 mm short) |
  | 40 deg | **249 / +5.1 PASS** | 209 / +5.2 -- engine vs chain 3.4e-2 m on some ray: NOT TRUSTED |

  Engine = chain to <= 6.5e-12 m on every other row.  At 30 deg the wall
  bought clearance by swinging M1's conic (-37.9 for 1.5k, -675 for 3k);
  the image got no weight while the wall dominated.
* The 30 deg POLISH (R3w30 re-solved with the wall at its knee, so the
  image rows drive it): the 1.5k solve did not move at all in 1000
  iterations -- every variable, the 231.6 px and the +5.1 mm identical; the
  3k polish was stopped for the same reason.  The clearance operand is a
  minimum over sampled leg points: not smooth enough for a finite-difference
  LM, so at the knee every step is rejected.

So the TMS entry in Jim's comparison is: on axis (obscured) it images --
**1.10 px** for the 1.5k module, **4.26 px** for the 3k -- and it PACKAGES only
with the strip 30-40 deg off axis, where the wall-dominated designs sit at
**200-250 px**.  That is a stalled solve's value, NOT a minimum: what the TMS
can image once packaged is unanswered until the clearance operand is smooth
(section 7, item 1).  No end-to-end row is possible without a packaged,
imaging design.

## 5. An engine discrepancy, briefed to CC

On the R2c deck (the entrance-pupil Reference 5.3 mm ahead of the convex
M1), `macos.trace(nElt)` in ONE call scatters every ray but the chief (93 mm
rms), while `trace(1)..trace(nElt)` reproduces the chain to 0.4 um.
Reproducer `repro_trace_onecall.m`.  Hypothesis only: ConSrf's |L^2 - mpr|
root choice after a Reference.  Stage t4 scores STEPWISE and checks every
ray against the chain traced from the engine ray's own entrance-pupil state.

## 6. Own errors, corrected on the way

* A ring sampling in the bridge spot (over-weights the edge, x1.24): fixed
  to the engine's uniform grid.
* A silent text-replacement miss left the R3 / first R3w solves without the
  wall, the Petzval row and the raised budget; those records were deleted
  and the walk re-run (section 4).
* The t4 size line reported the straight EP -> image distance (495 mm);
  now the M2-to-image envelope (563 mm for R1a).

## 7. Open

1. A smooth clearance operand (a closed-form or softmin distance) so a wall
   can actually steer the TMS solve -- the one thing between "packages only
   at 40 deg" and a solved packaged design.
2. The engine discrepancy (CC's lane).
3. The three-mirror by CCMac's route.
