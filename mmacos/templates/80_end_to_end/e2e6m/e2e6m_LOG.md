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
