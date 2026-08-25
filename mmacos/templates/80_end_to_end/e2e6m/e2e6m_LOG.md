# e2e6m — campaign log

The narrative record of the Keysight end-to-end use case: design a 6 m
unobscured telescope, segment it, add an imager + coronagraph back end,
harvest sensitivities, run a time series.  Appended as the work goes;
every substantive question, decision, and gate outcome lands here.
Not a command transcript — the story a reader replays.

Brief: `macos/BRIEF_to_e2e6m.md` (2026-08-24).  Branch `dev-candidate`
in `~/dev/MACOS_res_dev`.

---

## 2026-08-24 — S0: orientation, and the first hard question

**Read.** Brief, `templates/00_INDEX.md`, the `offset_imager` README
(the designer decision is made — offset_imager driven through
`oi_story`/`oi_walk`), `bench_ctb` README (geometry + diffraction
layers, the mask-quartet contract), `design/runners/run_sensitivities`
(the `groups` option the brief's S4 exhibit needs), the `e2e` README
(s1–s7 heritage the campaign follows).

**Toolchain checked, not assumed.**  Engine `libsmacos.a` (both ifx and
gfortran trees) and `mmacos.mexa64` both built 2026-08-22; no engine
source is newer than the libraries, so no rebuild is owed.
`MACOS_HOME=/home/dcr/dev/macos/macos_f90`.

### Q1 — does a 6 m offset-field imager fit an 8 m shroud?

Raised BEFORE spending a solve, because it shapes the S1 envelope.

The `offset_imager` README's feasibility screen is
`tan(offset) x |M1->stop| >= 1.5 x EPD`.  At EPD = 6 m that demands a
field-walk separation of **9 m**, and that separation is a LATERAL
displacement: M2/M3 sit ~9 m off the M1 axis.  The bounding diameter of
such a train is roughly `EPD/2 + sep + r_M2 ~ 12-13 m` — comfortably
outside the 8 m shroud gate the brief imposes at S3.

The brief's other envelope instruction, "EFL-ratio form-true rescale of
the rodgers3 W-fold", makes it worse arithmetically: rodgers3 is EPD
75 mm at F/4, EFL 0.3 m, and its envelope is `z_m1 = 0.665 m`,
`spacings = [-0.723 0 0.741] m`.  At EPD 6 m and F/15 the EFL is 90 m,
a x300 rescale — a **200 m long** observatory, with a 90 m field walk.
The README itself warns that the rescale is only form-true at the same
F#, and we are moving F/4 -> F/12..20.  So the rescale is a seed
heuristic here, not a prescription; the binding facts are the
unobscuration screen and the shroud.

The screen's 1.5x factor is CONSERVATIVE, not physical: it was compiled
from the t4 retraction, whose failing case sat 14x below it, so the
factor has never been probed near 1.  The physical requirement is only
that M2's glass clear the incoming beam's edge:
`sep >= EPD/2 + r_M2 + margin`.  With a fast primary (r_M2 well under
1 m at the stop) that is ~3.5-4.5 m, and the bounding diameter comes
back to ~7.5 m — inside 8 m, but tight.

**Answer: measure it, do not argue it.**  Next step is a cheap
paraxial+seed-trace ENVELOPE SCREEN over (offset, M1->stop leg, M1
radius, F#) that reports, per candidate: the exact first-order solve,
the traced footprint geometry, `oi_clear`'s signed clearance, and the
bounding cylinder about the shroud axis.  The S1 envelope is then a
measured pick with the shroud gate already in hand, and S3's shroud
figure is scoring a design that was chosen to be scoreable.

---

## 2026-08-24 — S1, first attempt: two defects and a form problem

### Finding 1 (fixed) — the source standoff does not scale with the aperture

Every candidate returned "candidate would not trace": the engine reported
all 277 rays a **surface miss at element 1**, while `oi_close`'s own
internal traces succeeded.  Cause: six sites in the offset_imager
toolchain (`oi_score`, `oi_clear`, `oi_close` x2, `oi_solve`,
`oi_layout_fig`) place the source plane a hard-coded **0.75 m** ahead of
the entrance-pupil construction point.  The launch plane is normal to
the CHIEF, so at a field offset it is tilted: its rim sits
`+-(EPD/2)*sin(offset)` in z.  At EPD 6 m and 22.5 deg that is
**1.15 m** — the rim of the ray grid starts BEHIND M1's vertex, and the
engine is right to call it a miss.  Correct for the 75-200 mm instances
the template was built on; wrong on the first large one.

Fixed with a shared `design/src/oi_standoff.m` = `max(0.75, 1.5*EPD)`,
threaded through all six sites (`G` now carries `EPD_m`; the two
hand-built geometry structs fall back to 0.75).  **Every committed
instance has EPD <= 0.5 m, so `max()` returns 0.75 and their results are
bit-identical by construction** — no re-run owed.

### Finding 2 — the 1.5x EPD field-walk advisory is the wrong measure

The advisory compares `tan(offset) x |M1->stop|` against `1.5 x EPD`.
But the entry beam and the post-M1 beam walk in OPPOSITE directions: at
the M2 station the entry footprint sits at `-L1*tan(offset)` and the
reflected bundle at `+L1*tan(offset)`, so the centre-to-centre
separation is **`2*L1*tan(offset)`**, and unobscuration needs only
`2*L1*tan(offset) > EPD/2 + r_M2`.  On a 6 m at L1 = 9 m that is
satisfied from about 8 deg, not the 34 deg the 1.5x rule implies.  The
screen measures the real thing (`oi_clear`, signed) and reports the
advisory as advisory.

### Finding 3 (open, structural) — the rodgers3 W-fold form does not scale

`oi_paraxial` normalizes the marginal ray to `y_in = 1` at M1.  For the
rodgers3 instance the solved first order gives **`y3 = 2.837`** — the
beam at M3 is 2.8x the entrance RADIUS.  That is fine at EPD 75 mm
(M3 = 213 mm) and fatal at EPD 6 m: **M3 would be 17 m across**, at any
form-true scaling, because `y3` is dimensionless.  So the brief's
"EFL-ratio form-true rescale of the rodgers3 W-fold" cannot be taken
literally here; the rescale is a seed heuristic for the same F#, and we
are moving F/4 -> F/12..20.

Enumerating the first-order roots directly (bisection on
`EFL(c2) = EPD*F#` under Petzval = 0, rather than the template's Newton,
so ALL roots are seen) over `L1` 6-14 m, `|R1|/L1` 1.6-8, `L3/L1`
-1.4..+1.4 and F/12..20, and keeping only roots whose image is REAL
(the back focus is a signed CODE V thickness; after M3 the beam
direction reverses, so a real image needs `BFD*t_2 < 0`), the compact
survivors all put **M3 essentially at the intermediate focus**
(beam radius 20-60 mm) with a 0.5-1.5 m back focus.  The alternative
family has M3 at 1.0-1.2 m radius and a 23-28 m back focus — a 50 m
track.  A finer scan is running to see whether a middle family exists.

### The enumeration, and the reversal (Dave, 2026-08-24)

Finding 3 closed with a full root enumeration rather than an argument.
For each `(L1, L3, F#)` the offset_imager first order is a ONE-parameter
family in `R1` (EFL exact + Petzval = 0 eliminate R2 and R3), so every
solution is reachable: reduce to `c3 = c2 - c1`, scan `c2` on a fine
grid, bisect every sign change of `EFL(c2) - EPD*F#`.  That sees roots
the template's Newton (from `c2 = c3 = -1`) misses -- and it does miss
them: at `L1 = 12, |R1|/L1 = 2.20, L3/L1 = 0.90, F/12` the Newton
returns `R2 = -12.6, R3 = -24.1, BFD = +41 m` while a compact root
`R2 = -2.67, R3 = -2.97, BFD = -0.68 m` exists at the same inputs.

Sign rule used, because it decides real vs virtual and is easy to get
backwards: `BFD` is a SIGNED CODE V thickness and the beam REVERSES at
M3, so a real image requires `BFD * t_2 < 0`.  Checked against the
rodgers3 deck (`t_2 = +0.7408`, `BFD = -0.851`) before trusting it.

Result over `L1` 6-18 m, `|R1|/L1` 1.90-3.20 (step 0.05), `L3/L1`
0.2-2.0 (step 0.05), F/12-20: **no root is simultaneously compact
(track < 30 m), real-focus, and larger than 0.08 m of beam radius at
BOTH M2 and M3.**  The compact real-focus roots all put M3 at the
intermediate focus with `|R2| = 2.6-7 m`; the alternative family has
M3 at 1.0-1.2 m radius and a 23-28 m back focus (a ~50 m track).

Then the engine confirmed the consequence directly.  At `L1 = 12 m`,
offset 12 deg, `|R1|/L1 = 2.20`, `L3/L1 = 1.20`, F/12 -- a compact
real-focus root, `R2 = -2.587 m` -- the rays reach M1 and the stop and
then **all 277 miss element 3**: the field walk at M2 is
`L1*tan(12 deg) = 2.55 m`, and the M2 sphere only exists out to
`|y| = |R2| = 2.587 m`.  The beam falls off the edge of the mirror.
Decenters, which would move M2 under its own beam, do not open until S4.

**The field-offset form has an aperture ceiling**: the offsets that
unobscure are the offsets that walk the beam off a mirror whose radius
the compactness requirement has already made small.  Independently, the
rodgers3 form puts the marginal ray at **2.837 entrance RADII** on M3 --
a dimensionless number, so a 17 m M3 at ANY form-true 6 m scaling.

**Dave ratified the reversal (2026-08-24): S1 moves to
`freeform_unobscured`** -- the sphere+Zernike tilted-fold front end, his
own 2026-07-06 direction, already a 500 nm coronagraph front end with an
`add_pupil` exit pupil and a clearance gate.  Rejected with reasons on
the record: `tma_unobscured` fails the shroud gate by arithmetic (its
AOI-safe eccentric decenter costs 3.6xD = 21.6 m against 8 m); template
surgery on offset_imager re-solves a form with M3 at the intermediate
focus; the far-back-focus family means a 40-55 m track.

**Shroud gate ruled: deployed, DIAMETER-ONLY** -- `packaging_report`'s
radial extent about the incoming-beam axis, `<= 8 m`; length free and
stated; the entry corridor reported separately as the sunshade keep-out,
not counted against the diameter.

### The four aperture rules (Dave, attached to the ratification)

`realize_apertures` has an OPEN frame defect (`Telescope.m` ~1930):
footprint centres are measured in GLOBAL XY and emitted as LOCAL
`ApVec`, so a saved tilted-fold `.in` loses every ray on reload
(`sz_tma.in` carries the latent; `clear_realized_apertures` is the
documented stopgap; `freeform_unobscured` itself runs apertures-off).
Since S2 onward lives on SAVED decks, this arc runs under four rules:

1. **Design apertures-off.**
2. **Apertures enter only** via the S2 segmentation machinery (the PM)
   and `aperture_full_field` (the rest of the train).
3. **Every `save()` is gated by a reload-ray-count check** -- the
   cheapest gate in the arc, and it catches the latent instantly.
4. If `aperture_full_field` turns out to share the defect, the proper
   `ray_bundle`-based frame fix is IN SCOPE, not a carried stopgap.

Noted on rule 4, to be settled when apertures first matter:
`aperture_full_field` computes its bounding box from
`macos.draw_rays('XY', ...)`, i.e. GLOBAL x,y (it does assert the global
grid frame), and then documents the result as "in the element's local
aperture plane".  On a tilted fold those are not the same plane, so the
defect looks shared.  Measured, not assumed, before S2 emits anything.

### S1 layout search

`s1_layout_search.m` (new, thin): the `freeform_unobscured` topology at
D = 6 m against the three hard gates -- f/# at the FP in [12, 20],
`packaging_report` shroud diameter <= 8 m, `check_clipping` UNOBSCURED
-- with `aoi_report`'s per-mirror spread (the < 15 deg coronagraph
polarization preference) as the tiebreak.  First reference points, from
a direct build at D = 6 m:

| layout | EFL | f/# | shroud | train | clear |
|---|---|---|---|---|---|
| heritage x0.75 (R 38.65/6.65/2.25, T 16.5/21) | 135.1 m | 22.5 | 11.1 m | 21.0 m | 4/4 |
| shorter M2->M3 (T 16.5/15) | 28.0 m | 4.67 | 10.1 m | 16.3 m | 4/4 |

Both clear; neither is in the f/# band, and both are over the 8 m
shroud.  The search sweeps `R1, T1, T2, R2, R3` at the heritage tilts,
then refines the tilts locally on the winners.

---

## 2026-08-24 — S1 on freeform_unobscured: the layout, and two tool defects

### The layout search

`s1_layout_search.m` sweeps the `freeform_unobscured` topology (three
base spheres, fold TILTS doing the unobscuration) at D = 6 m against the
three hard gates.  Diagnostics per row: the per-element RADIAL EXTENT
about the incoming-beam axis and the names of the bodies that obstruct,
because a barren grid has to say WHY.

Two things the sweep taught, both geometric:

**The fold optics live in a 1 m annulus.**  The entry beam is 6 m wide,
so anything beside it sits at |y| > 3 + its own beam radius; the 8 m
gate caps it at 4 - that radius.  M2, M3 and the FP all have to fit in
that annulus, which is why the feasible band in `tilt1` and `tilt2` is a
few tenths of a degree wide.  Below `tilt2 ~ 5.3` deg M3 and the FP sit
at |y| ~ 3.1 and clip the entry beam (`OBSC(M3+FP)`); above ~6.0 deg the
shroud passes 8 m.

**M2 must be small, which means M2 near the primary focus.**  Its
clearance needs `|y_M2| - r_M2 > 3` and the shroud needs
`|y_M2| + r_M2 <= 4`, so `r_M2 < 0.5 m` before any candidate exists at
all.  `T1/f1 = 0.911` gives `r_M2 = 0.27 m`.

WINNER (72-point final sweep, one survivor):
`R = [38.400, 3.800, 3.500] m`, `T = [17.500, 14.000] m`,
`tilt = [-5.60, 5.90, 8.00] deg` -> EFL 81.33 m = **f/13.55** (band
12-20), shroud **7.904 m** (gate 8.0), train 17.37 m, **UNOBSCURED**
4/4, AOI spread max 10.5 deg (preference < 15).

### Defect 2 (closed) — aperture_full_field measured in the wrong frame

Dave's rule 4 said to measure this rather than assume it.  Measured:
on the tilted-fold design `aperture_full_field` reported M2's aperture
centre at **yc = -3.405 m** where the true in-plane offset from VptElt
is **-0.031 m** -- it was reading the GLOBAL beam centre off
`draw_rays` and handing it back as a LOCAL `ApVec` offset.  It shares
the `realize_apertures` defect, and worse than that: two errors at once,
no shift to `VptElt` AND no rotation into the element frame.

The engine's convention, read off the source rather than guessed:
`ChkRayTrans` (elemsub.F) forms `rho = intersection - VptElt` and
projects `px = xObs.rho`, `py = yObs.rho`; `tracesub.F`/`propsub.F`
build the triad as `zObs = psiElt`, `yObs = unit(zObs x xObs)`,
`xObs = yObs x zObs`, seeded by the parsed `xObs=` or -- for every deck
the design layer emits, which declares none -- by the ChkDf2 default
`xObs = (psi_z, psi_x, psi_y)` (iosub.inc), which is NOT orthogonal to
psi and is made so by those two cross products.

Fixed (macos `6703a38`): a new `obs_frame_` helper reproducing that
construction, the footprint taken from `ray_bundle` (the FULL grid at
every element and field, not the two DRAW meridian fans), and the radius
taken as the farthest sample about the box centre rather than the box
half-DIAGONAL -- which oversized a round beam by sqrt(2), and an
oversized clear aperture is the element that has to be built.  Added
`apply_full_field_apertures`, because `aperture_full_field` only
MEASURES and `spec` is read-only outside the class, so until now no
caller could apply its report at all.  `realize_apertures` is
deliberately left alone: the rodgers1 corpus is scored through it.

Gate: `tests/tApertureFrame.m`, on a self-built tilted-fold fixture --
centres are element-local, the triad matches the engine construction,
applied apertures survive a standalone reload.  NON-VACUOUS: the last
test patches the emitted `ApVec` centres back to the pre-fix global-XY
values and requires the reload to lose the beam.

### The first S1 run, and what it exposed

Ran with the heritage mode set and default normalization.  Clearance
PASS 4/4 and the reload gate PASS (1185/1185 rays), but three failures:

| gate | got | want |
|---|---|---|
| f/# at the FP | 7.19 | 12-20 |
| shroud diameter | 8.206 m | <= 8.0 m |
| dense +-1' map, -tilt max | 0.381 waves | <= 0.071 |

The f/# and the shroud moved TOGETHER, from 13.55 / 7.904 on the base
spheres -- the tell that the CORRECTION changed the first order, not
that the layout was wrong.  `freeform_unobscured`'s mode set is
`[3 4 5 9 ...]`, which carries y-TILT (3) and POWER (5); power on a
mirror is optical power, so the solve spent EFL to buy wavefront.  The
project's own Zernike-solve doctrine (power pinned to the radii, tilt to
the pointing) says hold them out.

And the wavefront wall has its own cause: `set_freeform` defaults `lmon`
(the Zernike normalization radius) to the element BODY `ap_r`, which on
this train is the design-phase 3 m stub while the beam is **0.277 m at
M2 and 0.259 m at M3** -- every mode normalized to ~11x the lit patch,
over which they are nearly degenerate.  Its own docstring warns about
exactly this.  Measured now with the frame-correct
`aperture_full_field` used purely as a RULER (no apertures applied --
rule 1) and handed to `optimize_freeform` per mirror.

A/B in flight: correct `lmon` with the heritage modes, versus correct
`lmon` with power and tilt held out.

### The correction: what actually moved the number

Five variants of the freeform stage on the SAME layout, run in parallel,
each an 8-minute CALIB solve.  Centre-field result (waves RMS @ 500 nm,
from an uncorrected 6454):

| variant | modes | lmon | S0 centre |
|---|---|---|---|
| baseline | heritage 15, to 8th order | body ap_r | 1.251 |
| vB | heritage 15 | measured footprint x1.15 | 2.924 |
| vA | + 10th-order terms (24 modes) | body ap_r | **0.364** |
| vD | 24 modes, piston in, power+tilt out | body ap_r | CALIB abort |

Two findings, both worth keeping:

**The centre-field number is a bad predictor -- wait for the field
stages.**  Read at S0 the ranking said "extend the mode set, leave the
normalization alone".  Read at the end it says the opposite: the
measured-footprint normalization, which had the WORST S0 (2.924),
produced the BEST field result, and the extended mode set, which had the
best S0 (0.364), produced the worst.  Full table below.

**Piston is a correlated DOF here.**  With power and tilt held out but
PISTON (mode 1) kept -- offset_imager's set, where piston is a
frozen-thickness despace surrogate -- CALIB aborts with "DOFs for
optimization are correlated" and the process core-dumps.  Piston on a
mirror is degenerate with the OPD reference in this configuration.  The
power-pinned variants are re-run without it.

**Why power pinning matters at all**: the first S1 run's f/# and shroud
moved TOGETHER (13.55 -> 7.19, 7.904 m -> 8.206 m) because mode 5 is
optical POWER -- the solve spent EFL to buy wavefront.  Either the mode
set holds power out (and the base-layout gate means what it says), or
the f/# gate has to be applied to the CORRECTED system with an outer
closure.  The A/B decides which.

### The five-variant table, read at the end

Dense 7x7 map over +-1', -tilt max, waves RMS @ 500 nm (bar 0.071):

| variant | modes | lmon | S0 centre | S2 worst | dense -tilt | EFL corrected | shroud |
|---|---|---|---|---|---|---|---|
| vB | 15 (8th order) | measured x1.15 | 2.924 | 0.0889 | **0.0865** | 174.6 m (f/29.1) | 7.333 m |
| vC | 15, 500 iters | body ap_r | 1.251 | 0.1306 | 0.182 | 47.8 m (f/7.96) | 7.990 m |
| vA | 24 (10th order) | body ap_r | 0.364 | 0.4407 | 0.675 | 57.9 m (f/9.65) | 7.331 m |
| vD/vE/vF | power pinned out | body ap_r | -- | CALIB abort + SIGSEGV | | | |

**Power cannot be pinned out through CALIB on this train.**  All three
variants that dropped mode 5 (with and without piston, with and without
y-tilt) abort with "DOFs for optimization are correlated" and then
**core-dump** -- in a mex that is the host process.  Worth a separate
note: a detected optimizer degeneracy should return an error, not
SIGSEGV; this is the same class as the AVAR `stop` that used to kill the
host (closed by the IACCEPT_S sweep).

So power stays in the basis, and it is spent: vB's corrected EFL is
**2.1x** its base-sphere value.  The tell is in the coefficients -- the
first run's M1 departure came out at **2.4 mm**, which is not the
"micron-scale departure that does not move the chief ray" the
sphere+Zernike doctrine assumes.  The base spheres are simply too far
from the right shape, and the Zernike stage was being asked to supply
conic-level sag.

**The fix is to give the layout its conics.**  A conic is first-order
NEUTRAL -- it changes the shape, not the paraxial power -- so it buys
aberration without spending EFL, and it leaves the freeform departures
small enough that the doctrine's premise actually holds.  `[2a]` now
runs `Telescope.optimize` conic-only over the full field before the
Zernike stage, and reports the EFL across it as a check that it really
was neutral.  This is the `tma_unobscured` recipe's ordering, applied
inside the sphere+Zernike topology.

### The conic pre-stage: neutral, and not helpful here

Measured, since the reasoning above predicted it would help:

- **It IS first-order neutral**, as claimed: EFL across the conic solve
  is **81.311 m -> f/13.55**, exactly the base-sphere value, to the digit
  printed.  That much of the argument stands and the check stays in the
  runner.
- **It takes the field worst from 8793 to 1313 waves** on its own.
- **And it makes the freeform stage WORSE**: the inner-field result goes
  to **16.05 waves** against **0.066** without it.  The conic solve
  drives K to `[-11.28, -123.45, +214.60]`, and the Zernike basis --
  normalized on the measured footprint -- cannot work from there.

So the pre-stage is off by default (`P.tel.conic_stage`), kept as a knob
with the neutrality result recorded next to it.  The knob earns its
place: "conics are free first-order-wise" is the kind of claim that gets
re-derived every six months, and now it has a number.

### The f/# is closed on the base layout, not in the basis

With power in the basis and unpinnable, the freeform stage moves the
first order by a measured **2.15x** -- so the f/# gate cannot be read
where `s1_layout_search` reads it (the base spheres).  It has to be read
on the CORRECTED system, with the base layout as the knob.

`s1_close_fno.m` secants on the **M3 base radius**, running a full
`s1_telescope` per iterate and reading back the corrected f/#.  R3 is
the right knob: strongest EFL lever in this topology, weakest packaging
one.  The secant is taken on `log(f/#)` because the map is strongly
nonlinear and one-signed, and the packaging gates are re-read every
iterate -- an f/# bought by breaking the shroud is not a solution.  Each
iterate keeps its own full artifact set, so the winner is just the
directory whose numbers the report quotes.

### Enabling code for S3, built now because S3 cannot start without it

Two gaps, both real, both committed with gates:

**`macos.design.append_rx` (macos `03bd70f`).**  A `Bench` emits its own
prescription and a `Telescope` emits another; nothing joined them.  Only
ONE train can carry a telescope perturbation through to a coronagraph
contrast number, so the back end has to become MORE ELEMENTS of the
telescope's deck.  `append_rx` does that -- base header + base elements +
the add deck's elements renumbered, then the `nOutCord` terminator --
and checks the two things that fail silently: **BaseUnits** (element
coordinates are raw numbers, so a millimetre bench spliced onto a metre
telescope shrinks by 1000 and still "traces"; it refuses to convert
rather than blanket-scaling lengths, angles, indices and mode numbers
alike) and the **terminator block** (a deck without it loads as
nElt = 0).  Geometry is deliberately NOT checked -- this is text, not a
trace; the caller traces the result and counts rays, which the gate does.

**`Bench` gains `'baseunits'`** (default `'mm'`, so every existing bench
is bit-identical), so the back end can be built in the telescope's
metres in the first place.

Gates: `tests/tAppendRx.m` (4/4) and `tests/tApertureFrame.m` (4/4),
both added to `SUITE_FAST`.

---

## 2026-08-24 — S3 geometric layer: the splice works first try

`s3_backend.m` builds a 4-OAP coronagraph relay off the telescope focus
-- collimate to an accessible pupil (apodizer site), focus to the FPM,
re-collimate to the Lyot pupil, focus to the science detector -- in
METRES, and appends it to the telescope deck with `append_rx`, dropping
the telescope's terminal quartet (a `FocalPlane` mid-train terminates
it).  Smoke-run against the interim S1 deck:

```
[3] spliced: 3 telescope + 8 bench = 11 elements
[4] loads at 11 elements                                [PASS]
    1185/1185 rays pass (telescope alone 1185)          [PASS]
    Apodizer elt  5: beam radius 0.080558 m  (pupil)
    FPM      elt  7: beam radius 0.0015431 m (focus)
    Lyot     elt  9: beam radius 0.078351 m  (pupil)
    Science  elt 11: beam radius 0.0032561 m (focus)
[5] shroud on the full train: 8.204 m against 8.0 m
    train length 17.09 m
```

Both conjugate classes land where the builder says: the two pupils come
out at 80 mm radius and the two foci at 1.5-3.3 mm, measured from the
traced ray positions, not from the builder's own bookkeeping.

**The back end costs essentially NOTHING in shroud diameter.**  8.204 m
on the full train against 8.206 m for that telescope ALONE -- the relay
lives at |y| ~ 3.5 m with 80 mm beams, so its radial extent (~3.6 m) is
inside the telescope's own 4.1 m.  The annulus that the telescope's fold
optics already occupy has room for the instrument.  (This run used the
interim S1 deck, which was itself over the gate; the number to quote is
the one measured on the closed S1.)

Masks are NOT declared in the deck -- the three mask sites are passive
`Reference` markers and the masks are applied in MATLAB, per the ctb
contract.  An obscuration declared on a `Reference` clips rays only and
the diffraction wavefront sails through it untouched: that is the ctb
README's silent failure mode, and it is how a coronagraph deck can look
right and suppress nothing.

---

## 2026-08-24 — S2: the segmented primary, and a gate that measured itself

`s2_segmentation.m` is a thin driver over `run_segmentation`: a 2-ring
HEX tiling of the 6 m primary.  Result on the interim S1 deck:

```
[1] 19 segments, width 1.2 m flat-to-flat, 24 elements
    bare segmented: 985 rays, 985 pass, rmsWFE 5.175e-08 m
                    (parent 6.196e-08, -16.5%)
[3] physical apertures (pad 0): 983 pass -- 2 gap/rim rays clip
[5] standalone reload: 24 elts, 983/985 rays -> VERIFIED
```

**1.2 m flat-to-flat on a 6 m aperture**, the e2e-s3 / JWST class the
brief asked for, and only 2 of 985 rays lost to the gaps.  The parent's
solved M1 figure rides onto every segment as a FreeForm channel; the
footprint figure shows each segment's TRACED footprint inside its
EMITTED aperture polygon, which is the graphics gate.

### The poke gate found the OPD reference, not a frame error

The one-segment poke-localization check first read
**ratio out/in = 0.0559 -- FAIL**, with the message "the segment grid
frame is not its clocked Mon frame".  It was not.  The arithmetic gives
it away: 52 rays inside out of 982 valid, inside rms 1.886e-8 m, and
`52/982 x 1.886e-8 = 1.0e-9` against a measured outside rms of
**1.055e-9**.  The whole of the "leak" was the engine's default
whole-aperture MEAN OPD reference -- one global scalar, so poking ONE
segment moves it by `(N_seg/N_total) x (that segment's response)` and
that shift is subtracted from every ray.  My gate then removed piston
globally and read the result as leakage.

With `macos.opd_ref('chief')` set AFTER `load_rx` (every load resets it)
and no global piston removal:

```
    |dW| inside  the poked segment: rms 1.992e-08 m over  52 rays
    |dW| outside the poked segment: rms 0        m over 930 rays
    ratio out/in 0  [PASS]
```

**Exactly zero** outside, over 930 rays.  The segmentation frames are
right, the poke is perfectly local, and the inside response is 1.992e-8
against a 1e-8 piston -- the factor 2 a reflection gives.

Worth keeping as a rule: **on a segmented pupil, any per-segment
measurement needs the chief reference before it means anything.**  A
mean-referenced map will show every unperturbed segment pistoning, and
the number looks exactly like a frame bug.

---

## 2026-08-24 — S1 closes: the f/# is not a knob, and the design point is measured

### The closure REFUSED, and the refusal is the result

`s1_close_fno` ran five iterates and met no gate set:

```
iterate 1: R3 3.5000 -> f/29.10 | 0.0865 waves | shroud 7.333 | clear
iterate 2: R3 2.9750 -> f/25.14 | 1.4480 waves | shroud 7.327 | clear
iterate 3: R3 1.4000 -> f/15.54 | 0.4268 waves | shroud 7.290 | clear
iterate 4: R3 1.4955 -> f/27.20 | 0.0986 waves | shroud 7.502 | clear
iterate 5: R3 1.4050 -> f/25.74 | 0.3036 waves | shroud 7.271 | clear
```

Read iterates 3 and 5: **R3 moves 0.36% and the corrected f/# jumps
15.54 -> 25.74.**  Widen the view with the +-0.35' probes and it is
worse than non-monotone -- it is not a function of R3 in any useful
sense:

| R3 (m) | corrected f/# | dense -tilt max (waves) |
|---|---|---|
| 1.70 | 20.09 | 0.1611 |
| 1.78 | **8.56** | 0.3600 |
| 1.85 | 26.90 | 0.1052 |
| 2.00 | **25.39** | **0.0473** |

**The corrected f/# is an OUTCOME of which basin the freeform solve
lands in, not a controllable function of the base layout.**  Each basin
spends a different amount of optical power, and the basin is chosen by
the starting geometry in a way no continuous search can steer.  A secant
was never going to work here; neither would anything else that assumes
smoothness.  (The log-secant in `s1_close_fno` is kept -- it is the right
tool for the question it was asked, and its FAILURE is what established
this.)

### Field width is not a lever either

Tested from a third direction: at the SAME R3 = 1.70, narrowing the
design field from +-0.35' to +-0.20' made the residual WORSE
(0.0816 -> 0.2408 waves).  Narrower fields do not monotonically help;
the basin dominates.  So three independent knobs -- R3, field width, and
the Zernike mode set -- all show the same signature, which is what makes
this a property of the solve rather than a coincidence of one sweep.

### The design point, chosen on the CORRECTED system

`R3 = 2.000 m`, field `+-0.35'`, everything else as the layout search
picked it:

| gate | measured | verdict |
|---|---|---|
| dense 7x7 map over +-0.35', -tilt max | **0.0473 waves** (bar 0.071) | **PASS** |
| shroud diameter | **7.450 m** (gate 8.0) | **PASS** |
| clearance | 4/4 bodies clear | **PASS** |
| reload ray count | 1185/1185 | **PASS** |
| f/# at the FP | **25.39** (band 12-20) | **FAIL** |

Four of five.  The f/# is 27% above the band's top, and the honest
framing is that it is not reachable at diffraction-limited wavefront on
this layout family -- the whole trade table above is the evidence.
Worth saying plainly: **f/25 is not obviously the wrong answer for this
instrument.**  A slower OTA puts more lambda/D at the coronagraph's
focal-plane mask, which eases the hardest fabrication in the back end.
The band was set for a relay that has not been designed yet; this is a
number for Dave to rule on, with the cost of forcing it now measured.

### The field: +-0.35' is a choice, not a retreat

A 0.7' (42 arcsec) box, well inside the brief's "<= 0.1 deg".  The
coronagraph science field is a few lambda/D, which at f/25 and 500 nm is
ARCSECONDS -- the design field is already two orders of magnitude larger
than the science field.  +-1' was carried through the whole closure and
costs a factor of a few in residual for field nobody uses.

### A shroud number that disagreed with itself

Caught reading vN2's report: `packaging_report` at stage [4] said
**7.450 m PASS** and the shroud FIGURE at stage [6] said **8.306 m
FAIL** -- same design, both "the union of everything".  Stage [4] runs
BEFORE `add_pupil`; the figure ran after, and was counting the terminal
quartet's `Element=Return` surfaces.  An exit-pupil REFERENCE SPHERE is
a mathematical surface the propagator uses, at a radius with nothing to
do with the hardware envelope.

Both now gate on HARDWARE only (`Element=Return` excluded; `Reference`
mask and pupil markers KEPT -- those are real mounts), with the
propagation surfaces drawn dotted and the panel labelled so.  The same
rule is applied in `s3_backend`, where the kinds are read from the deck
text because a spliced deck has no spec object.  This was the difference
between PASS and FAIL on an S1 gate.

### Engine robustness note

CALIB detects "DOFs for optimization are correlated" and then **SIGSEGVs
rather than returning** -- in a mex that is the host process.  Hit four
times: every variant that dropped mode 5 from the Zernike set (vD, vE,
vF), and one plain layout (R3 = 2.5, vN1) that triggered it with the
stock mode set.  Same class as the AVAR `stop` that used to kill the
host, closed by the IACCEPT_S sweep.

---

## 2026-08-24 — S2 on the final S1 deck

Re-run against the closed S1 telescope (R3 = 2.0, +-0.35'):

```
[0] parent: 1185 src rays, 1185 pass, rmsWFE 2.772e-09
[1] 19 segments, width 1.2 m flat-to-flat, 24 elements
    bare segmented: 985 rays, 985 pass, rmsWFE 2.485e-09 (-10.4%)
[3] physical apertures: 983 pass -- 2 gap/rim rays clip
[5] standalone reload: 24 elts, 983/985 -> VERIFIED
[gate] poke: inside rms 1.991e-08 over 52 rays, outside rms 0 over 930
       ratio out/in 0  PASS
```

The -10.4% rmsWFE change from parent to segmented is the SOURCE GRID
changing, not the optics: `segment_rx` switches the source to a Hex grid
(1185 circular rays -> 985 hex rays), so the two numbers are averages
over different ray sets.  Worth stating because it reads like a
segmentation gain otherwise.

One reporting bug fixed here, worth remembering: **MATLAB's `.` matches
NEWLINE by default.**  `regexp(txt, '(?m)^.*key.*$', 'match','once')`
swallowed the whole file, so the S2 report came out as six copies of the
runner's log.  `'dotexceptnewline'` is the option; PCRE habits do not
transfer.

---

## 2026-08-24 — S3: the back end, and three bugs the gates caught

### The geometric train

`s3_backend.m` builds a 4-OAP coronagraph relay off the telescope focus
in METRES and appends it with `append_rx`.  It splices onto the
**SEGMENTED** deck: the coronagraph has to see the segmented pupil, and
S4's sensitivities need the segments and the back end in ONE train.

```
[3] spliced: 21 telescope + 8 bench = 29 elements
[4] loads at 29 elements                              PASS
    983/985 rays pass (telescope alone 983)           PASS
    Apodizer elt 23: beam radius 0.023541 m  (pupil)
    FPM      elt 25: beam radius 0.00044964 m (focus)
    Lyot     elt 27: beam radius 0.022895 m  (pupil)
    Science  elt 29: beam radius 0.00094973 m (focus)
[5] shroud on the full train: 7.451 m vs the 8.0 m gate  PASS
```

**The back end costs 1 mm of shroud diameter** -- 7.451 m against the
telescope's own 7.450 m.  The relay lives at |y| ~ 3.5 m with 24 mm
beams, so its radial extent sits inside the telescope's 4.1 m.  The 1 m
annulus that the fold optics occupy has room for the instrument.

### Bug 1 -- the conjugate is f/cos^2(AOI), not f

The first spliced train read **5.8921 waves rms** at the exit pupil
against 0.0055 for the telescope alone.  Decomposed: 5.66 of it pure
FOCUS, 0.229 reading as astigmatism, 0.0021 left after that.  An
off-axis parabola's pole-to-focus conjugate is `r = f/cos^2(AOI)` --
`add_oap`'s own docstring says so and I placed every marker at `f`.
1.011x at 6 deg is ~10 mm at f/20, and that is the whole 5.89 waves.

With the conjugates right the full geometric train is **0.0045 waves
raw -- BETTER than the telescope alone** (0.0055), and the 0.229
"astigmatism" is gone too, so it was a cross-term of the defocus, not
intrinsic.  At 6 deg AOI these OAPs are effectively aberration-free at
this beam speed.

The same error threw the science PSF **154 px off-centre**, which I
first read as a mis-posed exit-pupil sphere.  It was not: FEX's pose
agreed with the chief-ray construction to **6.7e-16 m and 2.1e-8 rad**.
A longitudinal error in a converging OFF-AXIS beam displaces the focus
laterally too -- worth remembering, because "PSF off-centre" reads as a
pointing or pupil problem and here it was defocus.

### Bug 2 -- `macos.trace(k)` is not a station on a segmented deck

`prop_layout`'s first version harvested chief pierces with a
per-station `macos.trace(k)` loop, as `ctb_prop_layout` does.  On a
SEGMENTED deck the first 19 elements are parallel segments of the same
mirror, so "trace past element 5" is not a station and the engine
rejects it.  Replaced with ONE traced pass read through `ray_hist`,
whose ray 1 is the chief.

### Bug 3 -- the prolate apodizer had not converged

`ctb_aplc` reported `Lambda0 = 1.0017 (conv=0, 200 it)`.  The eigenvalue
of the APLC operator is the fraction of energy the occulter passes and
is bounded by 1, so **1.0017 is not a number, it is a non-convergence
flag.**  Measured: this pupil converges at **2387 iterations**
(Lambda0 = 0.999994); the ctb default cap of 200 is simply too few for a
6 m pupil on a 1024 grid.

It mattered, and not only to the third digit:

| quantity | 200 it (unconverged) | converged |
|---|---|---|
| apodizer throughput | 0.337 | **0.123** |
| segmented DZ mean | 2.595e-06 | **8.700e-07** |
| APLC vs throughput-matched BLC, segmented | 0.45x (BLC WINS) | **1.05x (tie)** |

The unconverged apodizer would have had me report that a band-limited
mask beats an APLC on a segmented pupil.  It does not; that was an
artifact of stopping the power iteration early.  `prolate_iter` is now a
pass-through on `ctb_aplc` (default 200, so the committed ctb numbers do
not move) and e2e6m asks for 5000.

### Gate: `tests/tPropLayout.m` (3/3)

Chief ray preserved to 1e-9 at every original station, PSF on the FFT DC
pixel, and -- the one that matters -- **an EMPTY focal quartet is the
IDENTITY**: the pupil after the sandwich reproduces the pupil before it
to 1e-6 relative.  That is what makes it safe to insert a quartet
wherever a mask might go.  NON-VACUOUS: flipping `EPreturn2`'s zElt from
+R to -R (the one sign the recipe warns about) must break it, and does.

### The result: what the segment gaps cost a clear-pupil APLC

Same apodizer, same 2.8 lambda/D occulter, same 0.90 Lyot, same
telescope prescription -- only the primary differs:

| primary | DZ mean (3-15 l/D) | DZ median | on-axis suppression |
|---|---|---|---|
| 19-segment | 8.700e-07 | 4.567e-08 | 6.17e+03 |
| monolithic | **3.640e-10** | 5.327e-12 | 2.82e+07 |
| **ratio** | **2390x** | **8573x** | 4576x |

The monolithic arm lands at **3.64e-10**, beside `bench_ctb`'s committed
clear-pupil **2.1e-10** -- an independent check that the whole chain
(telescope -> `append_rx` -> `prop_layout` -> APLC) is sound, since
nothing in this arc was tuned to reproduce that number.

The curve shape says where the loss lives: the monolithic contrast falls
monotonically to ~1e-12 by 15 lambda/D while the segmented one plateaus
near 1e-6 with lobed structure across the whole dark zone.  That is
gap-scattered light, not a wavefront residual -- the two trains share a
0.0045-wave wavefront and differ only in the pupil's high-spatial-
frequency structure, which the clear-pupil prolate never saw.

APLC vs a throughput-matched band-limited mask, at 0.10 net throughput:
**109x in the APLC's favour on the monolithic pupil, 1.05x -- a tie --
on the segmented one.**  The APLC's advantage IS a clear-pupil
advantage, and the gaps erase it.

**Stated limitation, per the brief:** re-optimizing the apodizer for the
segmented aperture (a segmented-APLC / SCDA design) is out of scope.
8.700e-07 is what a CLEAR-PUPIL APLC does on a segmented pupil, and the
2390x is the size of the prize a segmented design would be chasing.

### Deferred (Dave, 2026-08-24): a more capable coronagraph

The `ctb` model carries **DM1, DM2, apodizer, focal mask, Lyot stop and
a FIELD STOP**.  The e2e6m back end currently carries apodizer / mask /
Lyot only -- enough to score the APLC and to give S4/S5 a real
coronagraph exit pupil, and that is where this stage stops for now.

What is missing and why it matters:

- **DM1 + DM2** are the reason a coronagraph has a controllable dark
  zone at all.  Without them the contrast here is an OPEN-LOOP number:
  it is what the optics deliver, not what a wavefront-control loop would
  hold.  Two DMs also make the dark zone ANNULAR rather than one-sided
  (see the dark-zone-geometry note in the agent memory), which changes
  what the 3-15 lambda/D annulus even means.
- **A field stop** at the post-mask focus removes the light scattered
  outside the science field before it reaches the detector.

Both are `Bench` primitives already (`add_mirror` on the collimated
pupil for the DMs, `add_reference` for the field stop) and both are
`prop_layout` station kinds already ('optic' and 'focus'), so the
topology extension is parameter work, not new machinery.  The APLC
numbers above stand as the open-loop baseline that a DM-controlled
version will be measured against.

---

## 2026-08-24 — S4: the linear model, and what a rigid PM actually does

`run_sensitivities` on the FULL train (`s3_seg_prop.in` -- segmented
primary plus coronagraph back end, one deck), 25 perturbed optics
(19 segments + M2, M3 and the four OAPs), field set = centre + 4 corners
of the +-0.35' box, OPD at the CORONAGRAPH exit pupil with a per-field
FEX reset.  31.2 min.

```
dwdxall 201504 x 156  (150 per-element + 6 GROUP columns)
dwdzall 201504 x 152  (19 segments x 8 MonZernike modes)
dwdgall 201504 x 114  (19 segments x 6 influence modes)
dwdx segment-only cond+ 2.162e+06
```

Channel counts scale from the e2e s3 heritage (7 segments -> 90/56/42)
exactly as the segment count does.

**Why the deck is the DIFFRACTION deck.**  Two reasons, both load-
bearing: it is the deck S5 propagates, so the linear model and the
contrast series describe the same object; and the OPD needs an
exit-pupil anchor, which the bare geometric train does not have (its
nElt-1 is a powered mirror and FEX refuses).  The Return/NF surfaces are
transparent to rays, so the geometric harvest does not notice them.

### The group exhibit -- three different physics in one table

Column RMS of dW/d(DOF), OPD-metres per rad and per metre:

| DOF | PM group (19 as one body) | one segment | ratio |
|---|---|---|---|
| Rx | 2.7555 | 0.14263 | **19.32** |
| Ry | 2.7397 | 0.14292 | **19.17** |
| Rz | 0.27056 | 2.4374e-06 | **111004** |
| Tx | 0.071678 | 0.0037254 | 19.24 |
| Ty | 0.070540 | 0.0037135 | 19.00 |
| Tz | 0.014366 | 0.44623 | **0.0322** |

- **Tip/tilt and lateral: ratio ~ 19, the member count.**  A rigid tilt
  of the assembly is 19 alike columns adding up, so a per-segment budget
  is neither conservative nor optimistic -- it is right.
- **Piston: 0.0322, a factor 31 BELOW 1.**  Moving all 19 segments
  together in piston is very nearly a GLOBAL piston, which the wavefront
  reference removes.  A per-segment piston budget therefore
  **overstates** the assembly's sensitivity by ~31x.  This is the
  intra-group compensation the exhibit exists to find.
- **Clocking: 111004x.**  An individual segment's Rz response is
  essentially null (2.4e-06) -- spinning a segment about its own normal
  on a near-flat parent does nothing -- while clocking the WHOLE PM is a
  real rotation of the aperture.  The group channel finds a live DOF
  that per-segment analysis calls dead.

That is the case for groups in one table: for one DOF the per-segment
budget is right, for another it is 31x pessimistic, and for a third it
misses the DOF entirely.

### Artifact hygiene

`s4_sens.mat` is **226 MB** -- gitignored, with a committed
`s4_sens.fp.json` fingerprint standing in (the e2e policy: derived
binaries >= ~20 MB are rebuilt, not stored, and the fingerprint keeps a
generation change auditable without the blob).  The first run also wrote
a **second** 226 MB copy into `s4_run.mat`, because the driver saved the
runner's `art` struct verbatim; `s4_run.mat` now keeps pointers and
metadata only.

---

## 2026-08-24 — S5: the time series, and an OPEN discrepancy the check found

MET is **not in scope** here, stated per the brief.  `run_compare` and
`run_simulator` both REQUIRE the MET products (`run_met`: Stewart truss,
dedx/dldx, estimator blocks), and standing that up means reconciling
run_met's body list with a Jacobian harvested on the FULL train --
integration work this stage does not do.  Consequences, both stated in
the runner header and the report:

- there is no metrology loop.  The control is **image-based**: it sees
  the wavefront, not a truss, so the corrected leg is an **optimistic
  bound**.  A real loop estimates the state from noisy metrology and
  does worse.
- `run_compare`'s SUBSTANCE is kept: the driver pokes sample DOFs and
  checks the ENGINE against the linear model.  What is dropped is the
  l/e measurement bars, which are metrology.

### The check earned its place immediately

```
elt  1 dof 0 (Rx): |engine| 9.604e-14  |model| 1.426e-10  rel 1.5e+03
elt  4 dof 5 (Tz): |engine| 4.438e-10  |model| 4.432e-10  rel 0.0017
elt  8 dof 3 (Tx): |engine| 2.343e-11  |model| 3.578e-12  rel 1.01
elt 12 dof 2 (Rz): |engine| 6.952e-14  |model| 7.150e-15  rel 1.03
elt 16 dof 0 (Rx): |engine| 5.356e-10  |model| 1.401e-10  rel 1.03
elt 19 dof 5 (Tz): |engine| 4.466e-10  |model| 4.440e-10  rel 0.0061
```

A linearity ladder over three decades separates two behaviours:

| poke | seg 1 Rx | seg 1 Tz | seg 16 Rx | seg 16 Tz |
|---|---|---|---|---|
| 1 nrad/nm | 0.0007 | 1.0003 | 3.8228 | 0.9988 |
| 10 | 0.0015 | 1.0003 | 3.8242 | 0.9988 |
| 100 | 0.0141 | 1.0003 | 3.8382 | 0.9988 |
| 1000 | 0.1409 | 1.0003 | 3.9790 | 0.9988 |

(engine / model, same alignment and units throughout)

- **Tz is exact** -- 1.0003 and 0.9988 at EVERY amplitude, on two
  different segments.  So the units, the row alignment, the column
  selection and the perturbation machinery are all right.  Nothing
  generic is wrong.
- **The CENTRE segment's Rx is second-order**: the ratio grows ~10x per
  decade of poke, i.e. the engine response goes as amp^2 and its
  FIRST-order term is zero.
- **An outer segment's Rx is linear but 3.82x the column**, constant
  across all four amplitudes.

Ruled out by measurement, not by argument: the **OPD reference** is not
the cause -- repeating the whole ladder with the harvest's default
(mean) reference instead of 'chief' reproduces every number to five
digits.

**Not resolved.**  It is not a noise floor (the ladder is smooth and
scale-free), not a units error (Tz would fail too), and not the
reference.  Two candidates left, both testable and neither tested: the
harvest's focal-plane TRACKING (`fp_mode='track'`, which removes the
pointing component of each response, and a segment tilt is largely
pointing) and the real-ray STOP AIMING (perturbing the element the chief
bounces off re-aims the whole source bundle -- which would explain why
the CENTRE segment, the one the chief lands on, is the anomalous one).

### What that costs, and the scope call

**It is not academic.**  With all six DOFs in the control basis the
corrected leg came out WORSE than the uncorrected one (WFE
0.0079 -> 0.0223 waves against 0.0046 -> 0.0243) -- the ridge solved on
columns the engine does not reproduce and pushed the wrong way.

So CONTROL is restricted to the DOF the check validates -- segment
**piston** -- while the DRIFT still moves all six, because the physics
does not care what we can control.  Piston is also the physical control
DOF for a segmented primary (phasing), so this is a real demonstration
rather than a fallback.  `P.ts.control_dofs` carries the restriction and
the reason.

**This is the argument FOR the full compare stage.**  A six-DOF control
basis that makes things worse is exactly what `run_compare` exists to
catch, and a cut-down check caught it here only because it was pointed
at the same columns the controller would use.

### Check broadly, control narrowly

First cut of the restriction made the gate sample only the CONTROLLED
columns -- so it passed at 0.00966 and stopped showing the problem at
all.  That is the wrong shape for a gate.  The driver now builds TWO
bases: `Ball` (all six rigid-body DOFs) for the CHECK, and the
restricted one for CONTROL, and the sampler walks every DOF class rather
than a linspace over column index (a linear sweep can miss a whole DOF,
which is exactly how a broken rotation channel hides).  The report now
reads:

```
elt 19 dof 0 (Rx): rel 56.9      dof 3 (Tx): rel 1.01
elt 19 dof 1 (Ry): rel 1.01      dof 4 (Ty): rel 54.8
elt 19 dof 2 (Rz): rel 1.04      dof 5 (Tz): rel 0.00614
worst over ALL six DOFs      56.9     [FAIL]
worst over CONTROLLED DOFs   0.00614  [PASS]
```

Only piston reproduces.  The stage reports its own failure and states
what it did about it.

### The series

41 frames at 10 s (a ~400 s soak), random walk 0.3 nm / 0.3 nrad per
step plus a correlated drift of 6 nm / 6 nrad per 100 s on all six DOFs
of all 19 segments; contrast scored every 5th frame (9 of 41), each
point a full 1024-grid propagation through the APLC chain.

| leg | WFE start -> end (waves) | contrast start -> end |
|---|---|---|
| uncorrected | 0.0046 -> 0.0243 | 8.601e-07 -> 2.078e-06 |
| corrected (piston, held from frame 2) | 0.0035 -> 0.0231 | 9.213e-07 -> 1.990e-06 |

Control effort: segment piston rms 88.8 nm, max 90.6 nm.

The correction helps and then fades, which is the honest shape of a
ONE-SHOT correction held against a growing drift: it is solved at frame
2 and never updated, so the benefit decays as the state walks away from
where it was solved.  Piston-only control also cannot touch the tilt
part of the drift.  Both are consequences of decisions recorded above,
not surprises.
---

## 2026-08-24 — S6: the deck, and what the figure inventory turned out to be

`deck_e2e6m.py` builds `deck_e2e6m.md` and hands it to the committed
`make_brief_slides.py`; every number arrives through
`e2e6m_records.py`, which `sys.exit`s on a parse miss rather than
printing a plausible default.  Thirteen slides: title + conventions,
six stage slides, a closer, one plain `Backup Slides` divider, four
backup slides.  DRAFT.

### The parse-miss rule earned its keep immediately

The first build died on `worst rel err` — because the final S5 run
had *changed the report*, splitting one worst-case line into two
(`worst over ALL six DOFs` / `worst over CONTROLLED DOFs`).  A parser
that fell back to a default would have quietly put the FAIL number on
a slide labelled PASS.  Both are now parsed and both appear.

### What I found when I went looking for figures to pair

DECK_STYLE wants every result slide to pair a layout figure with its
map.  Two of the six stage figures were not deck-grade, and the reason
is worth recording because it is a property of the models, not of
matplotlib:

- **`s2_segmented_views.png` / any 4-view render.**  Four panels at
  slide size are unreadable, and the panel labels collide.  Fixed by
  the DECK_STYLE remedy — `crop_panel()` lifts ONE panel (iso), keeps
  its content, drops its title (the slide caption carries it).  Nothing
  is redrawn.
- **`s4_dwdx_channels.png` is blank at slide size.**  156 channel maps
  on one axis grid renders as 156 labels over invisible dots.  It is a
  fine *diagnostic* at full resolution and useless as evidence.  Not
  used.  (Worth a look at the channel-montage figure's default layout
  the next time someone touches `run_sensitivities` — it stops being
  legible somewhere well below 156 channels.)

What replaced it is better anyway: **`s4_svspec.png`**, the singular-value
spectra of the three Jacobians.  Rigid-body motion collapses seven
decades across 156 channels (`cond+ 2.162e+06` on the 114 segment-only
columns) while the figure and influence bases stay inside one decade.
That is the same fact S5 runs into from the other side — a wavefront
measurement constrains far fewer directions than the assembly has
freedoms — so the two slides now argue the same point in sequence
instead of the error budget slide sitting half empty.

### A figure I built, measured, and did not use

The brief asks for a layout/`print_chain` gate at S3, so `s3_train_fig.m`
now renders `s3_seg_full.in` (iso + side) and `s3_seg_back.in` (the back
end alone).  Both are committed as gate artifacts.  **Only the full-train
iso is in the deck**, and the back-end render is not, for a measured
reason: the relay mirrors are centimetre-class against a 6 m primary, so
in the full-train view they are labels with no visible body, and in their
own view the train is so much longer than it is wide that the render is
ray lines with specks on them.  The packaging *map* (`s3_seg_shroud.png`)
carries the instrument slide instead — it is the evidence for the claim
the slide actually makes.

### One shared-tool change

`make_brief_slides.py` **stripped** `**bold**` instead of rendering it,
so DECK_STYLE's "compact bullets with bold lead-in labels" could not be
expressed.  `clean()` now leaves the markers, `para()` splits them into
alternating runs, `est_text_h()` and the table cells strip them
themselves.  Backwards-compatible: both heritage rodgers3 decks rebuild
to scratch paths at unchanged slide counts (19 and 9) and render
correctly — checked, not assumed.  **Neither committed rodgers3 .pptx was
regenerated**; `deck_rodgers3_final.pptx` is untouched per the brief.

### Style gate

`STYLE_REPORTS.md` §5 and `DECK_STYLE.md`, run against the built deck —
reported in-window with the build.

---

## 2026-08-25 — S3b: an apodizer for the segmented pupil, and the model that could not design one

S3 measured what the segment gaps cost a clear-pupil APLC (2390×).  This
slice tried to buy it back with the Carlotti/Vanderbei/Kasdin linear
program.  **The LP machinery works and is committed; it does not beat the
incumbent on this train, and the reason is measured rather than
guessed.**  Both halves are the deliverable.

### What the pupil turned out to be

The brief insisted the mask come from the ENGINE rather than a redrawn
hexagon, and that was load-bearing three separate times:

- **The pupil is NOT x-symmetric.**  y-flip residual 6.5e-08 (exact);
  x-flip 2.8e-02.  That asymmetry is the tilted-fold relay's own
  anamorphism — visible in the apodizer figure as a vertically elongated
  prolate.  A redrawn hexagon would have been symmetric in both axes and
  would have permitted a 4× fold that the real pupil does not.
- **The pupil carries phase**: 0.108 rad rms at the apodizer plane.
  Small, and not zero — see the cycle it cost below.
- **The gaps are 1.06 px wide at model 1024 and vanish below it.**  So
  the brief's escalation step "coarse optimization grid upsampled for
  application" could NOT be applied to the pupil: coarsening the pupil
  optimizes against an aperture with no gaps, i.e. against the wrong
  problem.  It is applied to the VARIABLES instead — block-constant tiles
  with the operator at full pupil resolution.  2364 variables at block
  3 px, y-folded.

### The generator

`design/src/apodizer_lp.m`.  Maximize throughput subject to ± bounds on
Re and Im of the dark-zone field, through the EXISTING occulter and Lyot
(N'Diaye's APLC operator, still linear, so still an LP).  Babinet puts
the only fine focal sampling inside the occulter.  The Lyot kernel is
closed-form by the **projection-slice identity** — the stop is radially
symmetric, so its focal kernel is the transform of its COLUMN SUMS,
exact for the array as built.  Solves in 50–94 s at 2364 variables /
5524 rows.  On a clear circular pupil it returns a shaped-pupil solution
that meets its target exactly.

Three self-tests, all of which earned their place:
- **MFT round-trip 7.1e-08.**  Run on a Gaussian, not the pupil: a hard
  disc's spectrum never fits a finite focal window, so a pupil round-trip
  measures window truncation (~1e-2) and would hide a real normalisation
  error underneath it.
- **Lyot kernel 4.1e-03** against a direct 2-D sum at oblique separations.
  The first version of the kernel was a Hankel quadrature over a binned
  radial profile and this test caught it at 2.7e-2.
- **Origin measured, not assumed**: centroid at [512.45 512.50], i.e.
  0.746 px from `floor(N/2)+1` and 0.054 px from `(N+1)/2`.  On an even
  grid those two conventions differ by half a pixel, and half a pixel is
  a linear phase ramp, i.e. a shifted dark zone.

### Two bugs the gates caught, both worth remembering

**1. The pupil is complex, and I modelled it as real.**  Cost: a full
cycle.  The amplitude-only operator tracked the engine to ~4× on a
smooth (prolate) mask — which looked survivable — and the LP then
optimized against that 4×, producing a mask the model scored at 3.1e-10
and the engine at 1.7e-6.  **5536× apart, and the mask was WORSE than
the incumbent it was meant to beat.**  Switching the operator to the
complex field closed 5536× to 4.6× in one edit.  The lesson is not
"phase matters" — it is that an optimizer will always find the
modelling error, so the model has to be right BEFORE it is optimized
against, and a 4× error on a smooth test case is not reassurance.

**2. The traced field's global phase made the LP return A = 0.**  The
on-axis field sum came out at 135.2°, so `Re(E00)/|E00| = −0.709` —
NEGATIVE.  The objective (Por's real-part convexification) was then
maximizing a negative quantity and the contrast bound `b·Re(E00)` was
itself negative, so the only feasible apodizer was the zero one.  It
reads exactly like an infeasible design problem.  A global phase is
unobservable; `apodizer_lp` now rotates it out internally.

### Localising the disagreement: five experiments

With the complex pupil the model still sat ~4.5–4.9× from the engine on
smooth masks, so before trusting any LP result I isolated it one
variable at a time:

| experiment | result | verdict |
|---|---|---|
| bare PSF, no masks, model vs engine | median ratio **1.0115** over 1–20 λ/D | MFT, normalisation, λ/D scale all correct |
| engine's mask application | `‖E1 − E0·Φ‖/‖E0·Φ‖ = **0.0**` | the engine multiplies the field exactly as assumed |
| radial profile, prolate | every ring lines up in radius; ratio flat at **4.9×** across 4–18 λ/D | a constant factor, NOT a scale error |
| Babinet term scaled 0 → 4 | answer moves **1%** | the FPM is irrelevant at 3–15 λ/D here |
| Lyot radius, both threshold rules, both planes | **127.12 px** everywhere | the stop is right |
| ENGINE's own post-apodizer field through the model | **4.16e-06** vs engine 7.9e-07 | the fault is the PROPAGATION model |

So the model has a **floor near 4e-06 whatever it is handed** — five
times ABOVE the 8.7e-07 the incumbent prolate already achieves.  A
single Fourier transform is a ~10–20% approximation to this back end
(FPM quartet, reference-sphere returns, near-field legs), and an
apodized dark zone is a ~1000× cancellation, so below a few 1e-6 the
residual IS the model error.

**The signature to remember:** the model-vs-engine divergence GROWS with
how hard the optimizer is pushed — 4.8× at target 1e-5, 13.6× at 3e-6,
40.0× at 1e-6.  That monotone growth is the tell that an optimizer is
mining a model, and it is visible without knowing the true answer.

### What was delivered instead

The fallback, upgraded.  Rather than importing an arbitrary published
hex apodizer, apply the **published METHOD to our aperture**: Soummer
(2005) Eq. 3 defines the APLC apodizer as the dominant eigenfunction of
the APLC operator over ANY support, so `ctb_apod_prolate` now takes a
`'support'` argument (N'Diaye et al. 2016 Paper V).  An eigenfunction is
a robust object rather than a fine-tuned cancellation, so model error
perturbs it instead of dominating it — and the ENGINE scores it either
way.  Λ0 = 0.5376 over the segmented aperture, converged in 1301
iterations.  (Worth noting: the CIRCULAR prolate on this pupil reports
Λ0 = 1.0000, i.e. saturated at the eigenvalue's physical ceiling.)

| configuration | DZ mean | DZ median | throughput |
|---|---|---|---|
| bare segmented APLC (S3 record) | 8.700e-07 | 4.567e-08 | 0.1000 |
| aperture-specific prolate | 7.909e-07 | 8.081e-08 | 0.0726 |
| best LP mask (target 1e-6) | 1.505e-06 | 2.052e-07 | 0.1747 |
| clear-pupil reference (S3 mono) | 3.640e-10 | 5.327e-12 | 0.1000 |

**Recovery: 1.10× in mean, 0.57× in median, at 0.73× the throughput —
i.e. none.**  No LP rung beat the incumbent either.  Redesigning the
apodizer ALONE, against a fixed 2.8 λ/D occulter and a 0.90 Lyot, does
not buy back what the gaps cost.  That is consistent with the
literature: the segmented-aperture APLC result is a CO-optimization of
apodizer × FPM × Lyot, which this brief explicitly deferred.

### What would unblock it

Either (a) an engine-faithful design operator — the LP needs columns
that match the propagation it will be scored against, which for this
train means the real multi-leg chain, not one Fourier transform; or
(b) co-optimization of occulter radius and Lyot geometry alongside the
apodizer, which is where the published segmented-APLC results come from.
(a) is the larger and more reusable piece: with it, `apodizer_lp` becomes
immediately usable, since the LP itself is validated.

### Gates

1. **Model vs engine: FAIL at 39.96× against a 3× bar** — recorded, not
   relaxed.  The disagreement is the finding.
2. **Recovery: 1.10× mean / 0.57× median at 0.73× throughput**, plane and
   λ/D range as the S3 record.
3. **tCtbProp 7 passed, 0 failed**, 1 incomplete (`test_proper_arbiter_fpm_leg`,
   filtered by assumption — PROPER absent in this environment, a
   pre-existing gate).  `ctb_aplc` gained `'apodizer'` and `'skip_blc'`;
   `ctb_apod_prolate` gained `'support'`.  All three default to the
   previous behaviour exactly.

---

## 2026-08-25 — close-out 1: the phantom apertures Dave caught

`s2_segmented_views.png` drew M2 and M3 as primary-sized domes, and the
figure was right: the DECK said so.

### What was actually wrong

`Telescope`'s emitter has a fallback that stamps a vertex-centred
`ApType=Circular / ApVec=e.ap_r` on any powered on-axis mirror.  `ap_r`
is the design-phase BODY radius, which on a 6 m telescope is ~3 m on
EVERY mirror.  Measured footprints on the real deck:

| element | traced footprint radius | old declaration | fiction |
|---|---|---|---|
| M1 | 3.0108 m | 3.3 m | 1.1x (honest margin) |
| M2 | 0.2740 m | 3.0 m | **11x** |
| M3 | 0.0138 m | 3.0 m | **218x** |

One correction to the brief's description, from the file rather than
from memory: only **M2 and M3** carried the fiction.  `FP_return`,
`ExitPupil` and `FP` already emitted `ApType=None` — the Return and
FocalPlane branches of the same chain were already right.  So the fix
is two elements wide, not six.

Nothing clipped (a 3 m stop around a 0.27 m beam is not a stop), which
is exactly why it survived: no ray moved, no number moved, and the only
thing that ever complained was a picture.  **The graphics gate is the
one that fired.**

### The fix

A new public `Telescope.declare_apertures(names)`: only the named
elements emit a hard aperture, everything else emits `ApType=None`.  It
wins over a realized aperture too — a caller who says "this element
carries no declaration" should not be quietly overridden.  Not calling
it leaves the default policy untouched, so the rodgers1 corpus and every
other design are unaffected.  `s1_telescope` now calls
`t.declare_apertures({'M1'})`, which is aperture rule 1 said out loud:
this design is solved apertures-off, apertures enter at S2 (segmentation,
for the PM) and S3 (`aperture_full_field`, for the rest).

### Gates

- **The regenerated views render M2/M3 at footprint scale.**  The domes
  are gone; M2 is a small disc near the focus and M3 a sliver, against a
  19-segment primary that now dominates the frame as it should.
- **Reload ray count UNCHANGED**: 1185/1185 on the telescope,
  983/985 on the segmented deck.
- **Downstream unchanged, and not by assertion**: S1, S2 and S3 reports
  regenerated and diffed against the pre-fix copies.  All three are
  **byte-identical apart from their wall-clock line** — including the
  S3 contrast table and the 2389.80x gap ratio.  S4 and S5 were
  therefore NOT re-run: every measured quantity on the path into them is
  bit-identical, so 35 minutes of regen would have reproduced their
  inputs exactly.

## 2026-08-25 — close-out 2: the imager leg

The demo's second instrument.  From the shared collimated pupil a
pick-off sends the beam to its own camera:

    ... telescope ... -> OAP1 (collimate) -> shared pupil
      -> PICK-OFF fold -> OAP_IM (f 0.90 m) -> imager detector

### A deployable pick-off, not a beamsplitter, and why

A permanent beamsplitter sits in the beam always, so the CORONAGRAPH
deck would have to carry its two transmitting surfaces — which changes
`s3_seg_full.in` and invalidates the S4 sensitivities and the S5 series
already built on it.  The brief allows "a post-pupil fold if a
beamsplitter fights the packaging", and here it fights.  So the two
instruments are two CONFIGURATIONS of one observatory (pick-off in /
pick-off out) — an ordinary instrument idiom — and both legs are counted
in the shroud gate regardless of which is deployed.  The pick-off folds
in the chief-Y plane while the coronagraph's OAPs all fold in chief-X,
so the legs separate instead of competing for the same annulus.

### Numbers

- shared collimated pupil **47.3 mm** (f/25.39 into a 1.20 m collimator);
  measured at the marker as 0.023747 m radius against 0.023633 asked,
  0.5% — PASS.
- camera f 0.90 m -> **f/19.0**, lambda/D **9.52 um** at the imager
  focus, Nyquist on a 5 um pixel.
- geometric spot radius at the detector **0.87 um**, i.e. lambda/D is
  11x larger — the image is diffraction-dominated, not aberration-
  dominated.
- **rms WFE 0.0042 waves @ 500 nm, Strehl 0.9993**, at the IMAGER leg's
  own exit pupil, piston+tip/tilt removed, Strehl exact from the OPD
  (never a pixel-peak ratio).  Diffraction limit 0.071 waves — PASS by
  17x.  (S1's 0.0473 waves is the telescope-only record at the
  TELESCOPE best-focus XP: a different anchor, quoted for reference.)
- ray count **983/985 — identical to the telescope alone**: the leg
  loses nothing.

### The shroud, and a measurement bug worth recording

First pass reported 7.448 m for both legs where S3's committed number is
7.451 m.  Three millimetres, and entirely a METHOD difference: I had
written a second radial-extent rule that also excluded `Element=Reference`
and measured max ray radius instead of `hypot(centre)+footprint`.  Two
shroud numbers measured two ways is how a demo ends up quoting 7.451 on
one slide and 7.448 on the next.

Promoted S3's rule to **`design/src/shroud_deck.m`** and switched BOTH
runners to it; S3 re-ran and reproduced **7.451 m** exactly.  The helper
also takes `'extra'` decks and returns the union.

**Result: imager leg 7.451 m, coronagraph leg 7.451 m, union 7.451 m
against the 8.0 m gate — PASS.**  The union EQUALS each leg because the
6 m primary sets the envelope and both instrument legs are
centimetre-class: **the second instrument costs nothing in shroud
diameter.**  That is the demo point, and it is measured rather than
asserted.

A figure bug caught on the way: plotting both decks in full painted the
second leg exactly over the first (they share ~all their elements), so
the figure showed one leg while the legend claimed two.  `shroud_deck`
now draws later decks' UNIQUE elements only, plus a zoom inset on the
instrument cluster — which is a few pixels at 6 m scale otherwise.

### Suites

`tApertureFrame` + `tAppendRx` + `tPropLayout`: **11 passed, 0 failed,
0 incomplete.**

---

## 2026-08-25 — close-out 3: the deck brought current

15 slides: title + 7 main + one plain `Backup Slides` divider + 6
backup.  Still DRAFT, still generator-built, still every number parsed
from a stage report.

**What changed on the main path.**  The old "The instrument" slide
became **"Two instruments | A second camera costs nothing in shroud
diameter"** — the imager's numbers beside the coronagraph's, and the
figure swapped for the two-leg shroud view.  That kicker is the
measured result, not a flourish: both legs and their union all come to
7.451 m because the 6 m primary sets the envelope.

**The S3b negative went to Backup, as two slides not one.**
DECK_STYLE is explicit that a slide needing a second table should split
rather than shrink, and the first draft had two.  So:
- *"What the segment gaps cost, and what does not buy it back"* — the
  four-row contrast table and the 1.10× recovery.
- *"How to tell an optimizer is mining its model"* — the ladder, and
  the transferable point: a FIXED offset is a calibration error, an
  offset that GROWS with the demand is the optimizer spending the
  model's error, and you can see it without knowing the true answer.

**Three stale figures caught by rendering, not by reasoning.**  The
aperture fix changed what `view_rx` draws, which moved the panels inside
the multi-view renders — so the committed crop boxes silently clipped.
Worse, `s3_train_iso.png` had not been regenerated at all, so the
closing slide was still showing the phantom domes the campaign had just
removed.  Re-ran `s3_train_fig`, re-MEASURED both panel boxes from the
current renders (ink-row profiling, not nudging), and noted in the
generator that a stale box must be re-measured rather than adjusted by
eye.  **A figure pipeline that crops by fixed fractions needs its
fractions re-derived whenever the source figure is regenerated** — that
is now a comment at both call sites.

Also fixed: the closer's caption claimed "three telescope mirrors and
the instrument relay" when, post-fix, the secondaries and instrument
optics no longer render as bodies at 6 m scale.  The caption now says
what the picture shows.  And `*italic*` reached a slide as literal
asterisks — the builder renders `**bold**` only.

### Style gate (STYLE_REPORTS §5, against both style files)

| item | result |
|---|---|
| 1. titles carry the argument | every title is plain descriptive + a kicker carrying a number |
| 2. sentence without number/mechanism/decision | none |
| 3. prose restating a table | none — tables are quoted only for interpretation |
| 4. units + convention + provenance | conventions once on the title slide; both new tables carry their own normalisation line |
| 5. caveats not buried | the S3b negative is two backup slides, and "What is not in this model" remains |
| 6. figures deck-grade, captions say what to see | 9 figures, 9 captions, all re-rendered and checked |
| 7. length in budget | 7 main + 6 backup |
| 8. register scan | no first person, no superlatives, no ALL-CAPS emphasis |
| mechanics | all 15 pages rendered; bottom-edge (7.32 in) clip check clean on every one |

### Suites

`tApertureFrame` + `tAppendRx` + `tPropLayout` 11/11; `tCtbProp` 7 passed
0 failed 1 filtered-by-assumption (PROPER absent, pre-existing).
