# Demo beat 22b — the adjacent problem, driven live

_Rehearsal script for the Keysight/CodeV demo (~2026-09-01).
CHOREOGRAPHY REVISED 2026-08-30 (deck restructure, Dave's rulings):
the ask + kickoff happen at the DEMO-INTRO SLIDE (9), right after the
challenge-3 frontier; **DAVE switches to MATLAB and launches
`oi_demo_step(<width>)` himself**; the reveal is slide 22, just before
the closing discussion (~26–32 min of cover).  The framing on the
slide: **"no AI in this loop"** — the design knowledge is compiled
into the driver, the without-AI half of slide 5's objective, running
live.  The Twyman-Green DM gauge (slides 13–15) is Dave solo on the
desktop, in a SEPARATE MATLAB from the one running this solve._

---

## The claim, in one sentence

The committed walk table is **compiled design knowledge** — five solved
boxes at one offset in one envelope — and the driver **extends it to a
box nobody solved in advance**, on a spec taken from the room, with the
answer predicted before it is computed.

---

## What is actually pinned (say this, briefly, so the knob is credible)

One knob: **field-box full width, 5–15°**.  Everything else is fixed —
offset **+22.5°**, EPD **150 mm**, **F/3.3**, λ 1 µm, the ×1.65 Rodgers
W-fold envelope, exit chief pinned horizontal, clearances 40/25 mm.

The width is the only knob because it is the only **validated
continuation axis**: box width shrinks the aberration span and the
clearance demand together and monotonically, so a solved design carries
into a wider box.  Walking the *offset* re-enters the documented t4
field-walk infeasibility.  That is a recorded finding, not a limitation
of the demo.

---

## What was actually measured (2026-08-28, the pinned knobs)

Three widths were run end to end at exactly the configuration the demo
uses.  **These are the fallback bundle**, and they are also the honest
answer to "how close does the prediction land?"

| ask | warm start | predicted | measured | floor (pred → meas) | exit err | verdict |
|---|---|---|---|---|---|---|
| 7×7° | step 1 (5°) | 18.0 nm | **20.0 nm** (1.11×) | 77.6 → 93.8 mm | 0.002° | PASS |
| 12×12° | step 3 (11°) | 33.7 nm | **33.6 nm** (1.00×) | 24.8 → 24.9 mm | 0.012° | PASS |
| 14×14° | step 4 (13°) | 54.9 nm | **51.2 nm** (0.93×) | 21.2 → 24.9 mm | 0.001° | PASS |

Every one of them PASSES both gates.  Wall time ~15 min each — but see
"Timing" below, because that number decides when the solve is launched.

**If the ask lands exactly on a committed width (5, 8, 11, 13, 15),
expect the live number to come in ABOVE the prediction — say so first.**
At a committed width the "prediction" is that row itself, and that row
was solved with 30 Gauss-Newton iterations on a 5×5 grid, where the live
run takes ONE polish step off the width below.  Measured at 11°: 39.1 nm
against the committed 27.3 (1.43×) — both gates still PASS, floor 57.9 mm.
Between committed widths there is no such asymmetry (12° lands 1.00×),
which is exactly why the default spec is 12.  If someone calls a
committed width, get out in front of it:

> "That one's already in the table, so you're asking me to reproduce a
> result — and the table's version had thirty solver iterations behind
> it where this gets one.  It'll land above the record; watch that it
> lands in the right neighbourhood and still clears every constraint."

## Pre-flight (before the room)

```bash
# 1.  The fallback bundle is present -- THREE widths, not just the default
ls  <mmacos>/templates/10_telescopes/offset_imager/demo_adjacent/
#    oi_demo_{7,12,14}deg_{map,layout,fields}.png  _verdict.txt  .in  _run.mat

# 2.  A MATLAB is warm and the engine loads (one model size per process)
MACOS_HOME=/home/dcr/dev/macos/macos_f90 matlab -batch \
  "run('<mmacos>/mmacos_setup.m'); macos.init(256); disp ok; exit(0)"
```

Have a terminal open on the offset_imager directory and the walk report
(`t5_walk/t5_walk_REPORT.md`) reachable — its per-step table is the
frontier being extended, and it is worth showing for ten seconds.

## Timing — READ THIS BEFORE SCHEDULING THE BEAT

A step at the pinned knobs takes **~15 min** (measured three-abreast;
~13.5 min of it is the solve).  It does **not** fit inside the demo's
8-minute box, and there is no knob left that buys the time back: the
solve is deck write/parse bound, so the ray grid is free; iterations are
already at 1; and dropping the solve grid from 5×5 to 3×3 — the obvious
saving — puts the 12° answer at 41.0 nm, i.e. WORSE than the committed
13° row, which is the one result that would undercut the whole beat.

So the solve must be launched **early** — the deck takes the spec at
slide 9 (the demo-intro slide, with the frontier fresh on screen) and
reveals at slide 22: ~26–32 min of cover for the ~15-min solve.  If the
spec somehow can only be taken late, run the
beat off the pre-generated bundle and say so plainly ("I ran this one
this morning") — the claim survives; a stalled console does not.

---

## The beat, in four moves

### 1 — Spec relay (top of the demo, ~20 s)

> "Give me a field box.  Anything between 5 and 15 degrees full width —
> it sits at 22½ degrees off axis either way, in a fixed envelope."

Take the number.  Repeat it back.  Then launch, **in a second MATLAB, in
the background**, so the rest of the demo runs while it solves:

```bash
OI_DEMO_WIDTH=12  MACOS_HOME=/home/dcr/dev/macos/macos_f90 \
matlab -batch "run('<...>/offset_imager/run_oi_demo.m')" > /tmp/oi_demo.log 2>&1 &
```

Nothing to watch here — the engine's own trace chatter fills the log.
That is why it goes to a file and the reveal comes from the verdict
block.

### 2 — The prediction, stated BEFORE the answer exists (~40 s)

Within a few seconds of launch — before any solving — the driver prints
the frontier bracket:

```bash
grep -A8 'frontier prediction' /tmp/oi_demo.log
```

Read it out.  **This is the beat.**  The claim is not "the tool can
solve it"; it is "the table already told us roughly what the answer must
be, and the driver goes and confirms it."

> "You asked for 12°.  The frontier brackets that: 11×11 at 27.3 nm with
> a 25.1 mm clearance floor, 13×13 at 40.0 nm with 24.6 mm.  So expect
> low-30s of nanometres, and a floor sitting right on the spec knee.
> Let's find out."

_(Rehearsed value, if 12° is the ask: it lands at **33.6 nm** against a
33.7 nm prediction and a **24.9 mm** floor — both gates PASS.  Do not
quote that on stage as a prediction; it is here so you know what is
coming and are not surprised by your own demo.)_

Then, one sentence on why it is not a cold start:

> "It doesn't start from scratch — it warm-starts from the 11-degree
> design the walk already solved.  Cold at this offset is the failure of
> record: 595 micrometres, a hundred and four of a hundred and
> twenty-one fields losing every ray."

### 3 — The other beat runs (the Twyman-Green gauge)

Hand over.  The solve keeps going.

### 4 — The reveal (~90 s)

```bash
grep -n 'OI-DEMO DONE' /tmp/oi_demo.log            # it finished
cat  <the path printed as 'OI-DEMO verdict'>       # the block
```

Then open the two figures the same line names: `_map.png` (the dense
strict-WFE map over the box) and `_layout.png` (the four-panel solid
hardware view).

Read the verdict block top to bottom — it is written to be read aloud.
Land on three things, in this order:

1. **the headline** — dense-map maximum, with its metric stated;
2. **the frontier line** — predicted vs measured, as a ratio;
3. **the gates** — exit direction and clearance, PASS or FAIL, named.

> "Predicted low-30s; measured `<X>` nanometres.  Exit beam still
> horizontal to `<Y>` of a degree.  Clearance floor `<Z>` millimetres
> against a 25 requirement — [PASS: it clears / FAIL: it is short by
> `<d>`, and that is the honest answer]."

**If a gate fails, that is the good version of this beat, not the bad
one.**  The clearance knee is the physics of the problem: the committed
floors cross the 25 mm spec between the 11° and 13° rows, so a wide ask
*should* be tight.  The deficit is priced — the endgame re-solve in this
same envelope lands 47.1 nm at a 30.89 mm floor.  Say that; it is the
difference between a tool that reports and a tool that markets.

**But do NOT pre-announce a failure at a wide ask — measurement says it
passes.**  The brief's original seed scripted ≥13° as an honest deficit.
That is now wrong, and in the interesting direction: at 14° the frontier
LINE predicts a 21.2 mm floor (below spec), and the re-solve delivers
**24.9 mm and both gates PASS**, at 51.2 nm against a predicted 54.9.
The reason is the same one the endgame found at 15° — the walk's widest
rows stopped early on the WFE-only plateau break while the clearance
hinge was still pulling, so they *understate* what the envelope holds.
The right line at a wide ask is:

> "The frontier says this one should be tight — interpolating its rows
> puts the clearance under spec.  But those wide rows were stopped early,
> so the line understates them.  Watch what a proper re-solve does."

…and then read the floor off the block.  That is a stronger beat than a
deficit, and it is what the numbers actually do.

---

## Scripted refusals — HELD IN RESERVE

Per Dave's ruling: **do not stage a plant and do not spend live time
demonstrating a refusal.**  These lines exist so that an off-script ask
is answered accurately in one breath, not so a refusal gets performed.

### Width outside 5–15°

The driver refuses before solving anything and prints its own sentence.
Read it, or say:

> "That's outside the range this driver has actually walked.  There's no
> solved neighbour to continue from, and a cold solve at that offset is
> the 595-micrometre failure I mentioned.  The honest answer is a full
> re-instance — that's an overnight run, and I'd send you the result."

### The carried design will not trace the asked box (the F8 path)

**You will almost certainly never see this one.**  Measured: widening the
box alone never trips it — the carried design still traces at 60°, at
90°, even at 150° (it just gets catastrophically bad, 5.7 mm of
wavefront).  The range screen catches a wide ask first.  This path is
defensive code inherited from `oi_walk`, where it guards the *cold-start*
failure mode.  Keep the line in your pocket, do not plan around it:

> "The screen says the design I'd start from doesn't even trace at that
> box — every ray misses.  The rule this driver follows is: never solve
> forward from a design already scored untraceable.  So it stops and
> says so rather than handing you a partial map with a number on it."

### "Can you move the offset instead of the width?"

> "Not on this driver, deliberately.  Box width is the only validated
> continuation axis — it shrinks aberration and clearance demand
> together.  Walking the offset runs into a documented infeasibility we
> already retracted a result over.  It's a legitimate problem, it's just
> a fresh instance rather than a continuation."

### "Change the aperture / the F-number / the envelope"

> "That's a new instrument, not an adjacent problem — the whole envelope
> gets re-seeded and re-walked.  It runs, it just runs overnight rather
> than over coffee.  Happy to set one going and send you the report."

---

## Fallback ladder (top to bottom, take the first that works)

1. **Pre-generated runs at 7, 12 and 14°** — `demo_adjacent/oi_demo_*`.
   Same wrapper, same pinned knobs, run in advance.  Show the verdict
   text and the two figures; the narration is identical, in the past
   tense.  Three widths means a spec near any of them can still be
   answered with a real artifact rather than a table.
2. **Committed walk artifacts** — any step of `t5_walk/`
   (`t5_walk_k0*_map.png`, `_layout.png`) plus `t5_walk_REPORT.md`.
   These are the frontier itself; the beat becomes "here is the compiled
   knowledge" without the live extension.
3. **The frontier table + the endgame slide** — talk over the numbers.
   5/8/11/13/15° → 10.9/21.5/27.3/40.0/69.8 nm at floors
   98.0/67.4/25.1/24.6/17.8 mm; the endgame closes the 15° deficit at
   47.1 nm / 30.89 mm.

---

## Metric, if asked (it is printed with every number)

Strict RMS wavefront error; reference sphere centred on the spot
centroid on the frozen focal plane; anchored at the exit pupil;
piston-only removal.  The headline is the **maximum** over an 11×11
dense field map across the box — and the solve set (3×3) is deliberately
**not** the scoring set.

---

## Timing and knobs of record

See `README.md` (section "Live demo: the adjacent-problem beat") and the
delivery log at the foot of `macos/BRIEF_r3_adjacent_demo.md` for the
measured wall times at the pinned knobs.
