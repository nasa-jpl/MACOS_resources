# REPORT — the coronagraph's input-field servo (C+)

TO for Dave, 2026-09-15. Branch `dev-candidate`. `BRIEF_to_gauge_close` item 7,
from `macos/NOTES_gauge_in_coronagraph.md` section **C+**: a dichroic at the
apodizer pupil pulls out-of-band light to a vector-Zernike or pinhole reading of
the coronagraph's **input field** — amplitude and phase — and the two DMs servo
that field to the one recorded when the dark hole was dug. That holds the hole
against everything upstream of the pickoff, not only the DMs.

The live status table for the whole close-out is at the top of
`../../40_benches/tg_psi_dm96_oap/REPORT_reflective.md`.

## Status

| step | what | state |
|---|---|---|
| 0 | the beam-size precondition | **DONE — and it RETRACTED my conclusion.** Measured: DM beam 21.244 mm diameter, so `ctb_dm.m` is right and the note's 33-cycle crossover is wrong (it is **16.5**). §0 |
| 1 | the reading at the apodizer conjugate | not started |
| 2 | separability of the two DMs vs spatial frequency | **unblocked** — crossover 16.5 cycles (period 1.285 mm); not started |
| 3 | the servo: drift injected upstream, hold, contrast | not started |

## 0. RETRACTED AND SETTLED: the beam is 21.24 mm, and the DM model is right

**The engine has overruled my reading of the generator, and the retraction is
the point of this section.** Everything below the horizontal rule was written
before `ctb_beam_probe` ran; it argued from `example_ctb.m` that the beam is
42.75 mm across and that `ctb_dm.m` uses a radius as a diameter. **That
conclusion is wrong.** It is kept, marked, because the probe exists precisely
because a document can be read wrongly, and deleting the wrong reading would
hide why the measurement was worth taking.

**Measured, `ctb_dcr.in` at model 512, 50618 traced rays:**

| plane | elt | **fp radius** | **fp diameter** | declared clear r | README's chain |
|---|---|---|---|---|---|
| DM1 | 2 | 10.6222 | **21.2444** | 22.5 | "DM 21.4" |
| DM2 | 5 | 10.6225 | **21.2451** | 22.5 | — |
| Apodizer | 13 | 7.9347 | **15.8695** | — | "apod 16.0" |
| Lyot | 20 | 3.9696 | **7.9392** | — | "Lyot 8.0" |

The README's chain *"DM 21.4 → apod 16.0 → Lyot 8.0"* matches the measured
**DIAMETERS** to three figures at every station. So the chain is diameters, its
`fprintf` label *"pupil beam radii"* is simply wrong, and:

- **`ctb_dm.m` is CORRECT.** `beam_d_mm = 21.3` IS the beam diameter (measured
  21.244), the pitch `21.3/32 = 0.666 mm` is right, and the 32×32 lattice spans
  the whole beam. No EFC result is affected. My three commits claiming
  otherwise (`2e83452`, `2bcbb72`, `afb31f5`) are **retracted on this point**.
- **The note's separability prediction is what is wrong**, and by the factor I
  identified even though I mis-assigned it. Section C+ quotes *"a 50 % amplitude
  conversion at 33 cycles across the beam (0.67 mm pitch, 500 mm, 550 nm)"*.
  The **pitch is right**; the **cycle count is not**. Measured on the engine's
  own beam: 50 % conversion at a period of **1.285 mm = 16.5 cycles across the
  beam**, 100 % at 0.742 mm = 28.6 cycles. **The crossover is 16.5, not 33.**
- The `e2e6m` comparison (`2bcbb72`) still stands as written — that model
  doubles a measured radius and its lattice spans its beam — but it is now an
  example of the CORRECT convention that the CTB *also* follows, not a contrast.

**What survives from the original argument** is the thing that prompted it: the
two halves of the note's prediction are mutually inconsistent, since
`32 × 0.67 = 21.4`, not the 42.8 the 33-cycle figure needs. That was right. The
half I picked as the error was the wrong half.

**A genuine oddity the probe exposed, which is NOT a DM-model bug.** The
generator computes `w_DM = R_DM·FILL = 22.5 × 0.95 = 21.375` and the README
describes that as filling 95 % of the DM. The measured beam RADIUS is 10.62 mm
— **47 % of the DM's 22.5 mm clear radius**, not 95 %. So either MACOS's source
`Aperture` is a full-cone angle where the generator treats it as a half-angle,
or the intended fill is twice what the deck achieves. The DM lattice correctly
spans the beam that exists either way, so nothing downstream is wrong; but the
CTB's DMs are being illuminated over half the radius the sheet intends, and
that is worth a look in the CTB lane.

**Step 2 is unblocked**, with its crossover at **16.5 cycles** (a period of
1.285 mm — the form that needs no beam-size convention, as promised).

---

## 0b. The original argument, SUPERSEDED — kept for why the probe was written

*(Everything from here to the end of this section is the pre-measurement
reading. Its conclusion is retracted above.)*

### Before anything is built on it: how wide is the CTB beam?

Step 2 asks where the two DMs separate, and the note predicts it: *"a 50 %
amplitude conversion at 33 cycles across the beam (0.67 mm pitch, 500 mm,
550 nm)"*. That prediction cannot be taken forward as written, because **its two
halves are inconsistent with each other**, and the committed documents disagree
about the fact they rest on.

The Fresnel conversion of DM2's phase into amplitude at DM1's plane is
`sin(pi*lambda*z/L^2)` for a ripple of period `L`; it reaches 50 % at
`L^2 = 6*lambda*z`. With `lambda = 550 nm` and `z = 500 mm` that is
`L = 1.284 mm`, so the cycle count depends entirely on the beam's width:

| beam across the DM | 50 % conversion | 100 % conversion |
|---|---|---|
| 21.4 mm | **16.6 cycles** | 28.9 cycles |
| 42.8 mm | **33.3 cycles** | 57.7 cycles |

The note's 33 is the second row, i.e. a **42.8 mm** beam. But the same sentence
quotes a **0.67 mm** actuator pitch, and the DM is 32 actuators across:
`32 x 0.67 = 21.4 mm`, the FIRST row. A 32-actuator DM spanning 42.8 mm would
have a 1.34 mm pitch. **Both halves of the prediction cannot be right.**

The documents do not settle it, and they contradict each other:

- `README.md` (source model): *"the source NA is set to put a `R_DM·FILL` beam
  on the DM"*. `R_DM` is the DM **radius**, 22.5 mm, so `R_DM·FILL` is a
  **radius** — and the chain it introduces, *"DM 21.4 → apod 16.0 → Lyot 8.0"*,
  is a list of radii. That makes the beam **42.8 mm across**.
- `ctb_dm.m` declares `'beam_d_mm'  controlled beam **diameter** on the DM
  (default 21.3, the measured CTB footprint at DM1/DM2 — gate1b probe)` and
  uses it as a diameter throughout: `pitch = beam_d_mm/nact` = 0.666 mm, and
  `active = hypot <= beam_d_mm/2 + pitch`. That makes the beam **21.3 mm
  across**, and it is where the 0.67 mm pitch comes from.

If the README is right, the DM lattice spans half the pupil and every EFC result
on this bench controls only its inner quarter by area. If `ctb_dm.m` is right,
the note's separability crossover is **16.6 cycles, not 33** — a factor of two
in the answer Dave asked for.

### The deck GENERATOR settles the intent, and it is not the DM model's reading

`example_ctb.m`, which generates the deck, is unambiguous in code rather than
prose:

```matlab
P.R_DM  = 22.5;                  % DM aperture radius, mm (stop)
P.FILL  = 0.95;                  % source NA fills FILL of DM
P.AP    = P.R_DM*P.FILL / P.r(1);            % source numerical aperture
...
w_DM = P.AP*P.r(1);
fprintf('pupil beam radii (mm): DM %.2f  apod %.2f  Lyot %.2f  backend %.2f\n', ...
```

`w_DM = AP*r(1) = R_DM*FILL = 22.5 x 0.95 = **21.375 mm**`, and the line that
prints it is labelled **"pupil beam radii"**. So the README's chain
*"DM 21.4 -> apod 16.0 -> Lyot 8.0"* is a list of RADII, and **the beam at the
DM is 42.75 mm across.** The committed deck agrees: its `Aperture=` is
8.4850531440e-3, and `22.5*0.95/8.4850531440e-3 = 2519.1 mm`, which is the
source-to-OAP1 conjugate the generator computes.

**So `beam_d_mm = 21.3` is the beam RADIUS being used as a DIAMETER**, and it is
used that way everywhere it matters:

| in `ctb_dm.m` | value with 21.3 | value on the real 42.75 mm beam |
|---|---|---|
| `pitch_mm = beam_d_mm/nact` | 0.666 mm | 1.336 mm |
| `active = hypot <= beam_d_mm/2 + pitch` | radius **11.32 mm** | radius 22.7 mm |
| fraction of the pupil's AREA the lattice covers | **28 %** | 100 % |

No caller overrides it: `ctb_dm_jacobian` carries the same 21.3 default and
`ctb_efc` reads `d.beam_d_mm` back out of the Jacobian, so the whole EFC chain
inherits it. The README's own summary line repeats the slip in words — *"880
active actuators per DM (centers within beam radius + 1 pitch)"* — describing
`beam_d_mm/2` as the beam radius.

**What I am and am not claiming.** I have not re-run any dark hole and I am not
saying the committed contrasts are wrong: a DM that controls the inner 28 % of
the pupil area still digs a hole, and every EFC result on this bench was
self-consistent with the model it used. What needs re-reading is what those
results MEAN — a 32x32 DM spanning half the pupil's diameter has half the
spatial-frequency reach, so the controllable dark-zone extent in lambda/D is
half what a full-pupil reading of the same actuator count would suggest. That is
a question for the CTB lane, not for this report; it is raised here because item
7 step 2 cannot be answered without settling it, and because it would be worse
to answer step 2 on top of it quietly.

**The engine still gets the last word,** because the generator describes intent
and the committed `ctb_dcr.in` is what runs.

**This is not settleable from the documents, so it goes to the engine.**
`ctb_beam_probe.m` traces the committed deck and reports the measured ray
footprint at DM1, DM2, the apodizer and the Lyot stop, as a radius *and* a
diameter, beside each element's declared clear aperture — then prints the
crossover its own measurement implies. No deck is modified. It is queued behind
item 4 in `../../40_benches/tg_psi_dm96_oap/runs/closechain5.sh`.

Until it has run, **step 2's number is open and step 2 is not started** — and
on the generator's beam the note's own formula gives **33.3 cycles at 50 %
conversion, which is the number the note quotes**. So section C+'s prediction
looks right and its parenthetical "0.67 mm pitch" is the part that is wrong:
the pitch on a 42.75 mm beam is 1.336 mm. The
rest of section C+ — what the pickoff cannot see, the chromatic transfer, and
the stellar photon table — does not depend on this and stands as the note has
it.

### Corroboration: the e2e6m model does NOT have this slip

**Checked in the SOURCE, not in the note** -- the same document-trust that
produced this problem would be a poor way to close it. `e2e6m_r2/r1_dm.m:57`:

```matlab
beam_d = 2 * 0.023771;              % measured pupil at the DMs (r1 gate)
...
dm = <dm model>('nact', 32, 'beam_d_mm', beam_d, 'pitch_mm', beam_d/32);
```

It takes the MEASURED pupil radius and **doubles it** before filling
`beam_d_mm`, then divides by 32 for the pitch: 47.54 / 32 = 1.486 mm, which is
the note's 1.48. (That model works in metres throughout -- a 20 nm poke is
written `20e-9` -- so the `_mm` suffix is carrying metres there; internally
consistent, and beside the point, which is the doubling.) The CTB call fills
the same field with the radius and does not double.

So the two benches, the same field name, the same helper: one doubles, one does
not. That is what localizes the error to one bench and shows what the correct
relationship is:

| | CTB | e2e6m space relay |
|---|---|---|
| measured pupil RADIUS | 21.375 mm (generator) | 23.771 mm (`r1` gate) |
| what reaches `beam_d_mm` | **21.3 — the radius** | **47.54 — the radius DOUBLED** |
| actuators x pitch | 32 x 0.67 = **21.4 mm** | 32 x 1.48 = **47.4 mm** |
| does the lattice span the beam? | **no — it spans the RADIUS** | **yes — it spans the DIAMETER** |

So the convention
the CTB model should follow is already in use next door: `nact x pitch` equals
the beam DIAMETER. On the CTB that would make the pitch 1.336 mm, and the
note's 0.67 mm is the radius divided by the actuator count.

This also means item 7 step 2's prediction transfers unchanged for the e2e6m
package (the note's 900 / 144 / 36 / 14 / 4 stroke table) and needs re-deriving
only for the CTB.

### What the slip does to the DM's reach — the two readings, stated precisely

It is not simply "the DM is too small". The modelled lattice and a correctly
scaled one trade area against spatial frequency in opposite directions, and
step 2's question — where do the two DMs separate — sits exactly on that axis:

| | as modelled (21.3 mm lattice) | scaled to the beam (42.75 mm lattice) |
|---|---|---|
| pitch | 0.666 mm | 1.336 mm |
| Nyquist period (2 x pitch) | 1.332 mm | 2.672 mm |
| **cycles across the 42.75 mm beam at Nyquist** | **32** | **16** |
| pupil AREA the lattice covers | **28 %** | 100 % |

So the model as it stands claims **twice the spatial-frequency reach over a
quarter of the pupil area**; a correctly scaled 32-across DM has half the reach
over all of it. For a dark hole those are different instruments: the outer
working angle scales with cycles across the beam, so the current model implies
an outer edge near 32 λ/D where the corrected one implies 16 — while the
corrected one has the whole pupil contributing to it.

**Which of the two the CTB hardware actually is, this report does not decide.**
A real 32x32 DM has a physical pitch, and whether the bench demagnifies the
beam onto it or fills it is a fact about the bench, not about the model. What
is certain is that the deck and the DM model currently disagree: the deck puts
a 42.75 mm beam on the element and the DM model varies the surface over the
inner 21.3 mm of it. One of the two has to move, and which one is a question
for the CTB lane with the hardware in hand.

Step 2's measurement is well posed either way if it is reported as a PERIOD in
mm rather than in cycles across the beam — a period needs no beam-size
convention. That is how it will be reported, with the cycle conversion given
once the probe has fixed the beam.

## 1. The reading at the apodizer conjugate — the prescription, not yet built

Written out so the build is a transcription rather than a design, and so the
sizing can be checked before anything is traced. **Nothing here has been run.**

### The pupil the gauge sees is settled independently of the DM question

Re-deriving the generator's own chain from `P.F_OAP = [2500 1524 1143 1350 675
635 635 762]` and `w_DM = R_DM·FILL`:

| plane | radius | |
|---|---|---|
| DM1 / DM2 | 21.38 mm | `w_DM` |
| **apodizer** | **16.03 mm** | `w_DM·F₃/F₂` |
| Lyot | 8.02 mm | `·F₅/F₄` |
| backend | 8.02 mm | `·F₇/F₆` |

That reproduces the README's printed *"DM 21.4 → apod 16.0 → Lyot 8.0 → backend
8.0"* to three figures, which is a **third** independent confirmation that the
chain is radii: the prose, the code, and now the arithmetic agreeing. So the
gauge's input pupil is **32.06 mm across**, and — importantly — this number does
not depend on the `beam_d_mm` question at all. That question is about the DM
LATTICE, not about the beam, so step 1 is not blocked by it. Step 2 is.

### The branch

A dichroic at `Apodizer_Pst` (already a `Reference` in the committed deck, one
element ahead of `Apodizer`, so no new station has to be invented) sends the
out-of-band light to:

| element | what | sized by |
|---|---|---|
| focusing lens, f = 300 mm | brings the 32.06 mm collimated pupil to a focus | F/9.4 |
| the ZWFS mask at that focus | λ/D = **5.92 µm** at 632.8 nm | dimple at the record's 2.0 λ/D = **11.8 µm** across |
| a pupil reimage behind it | back to a camera | the `zwfs_dm96` NF1/NF2 sandwich idiom, `dmg_zwfs_gauge` |

`f = 200` and `400 mm` give F/6.2 and F/12.5 with λ/D of 3.95 and 7.89 µm; 300
is the middle and puts the dimple at a size the VSG2 mask already has.

**Detector sampling is not the constraint here, and that is worth saying,**
because it is on the tg96 rig. The field being read is the coronagraph's input
field and the actuators that correct it are the two 32-across DMs, so the
reimaged pupil needs ~2 px per actuator — 64+ px across — where `zwfs_dm96`
needs 193 for a 96×96 DM. The gauge camera is not the hard part of this build.

### What it must be gated against

The brief's gate: the reading reproduces the engine's complex field at the
apodizer to the V5 record (0.016 / 0.128 pm through 5 / 20 % amplitude dips).
`mask.v_clear` is the reading that returns amplitude AND phase, which is what a
field servo needs and what the phase-only pair of record does not give.

Out of band is **632.8 nm against the CTB's 550 nm science band**, and the
chromatic transfer the note's bound 3 describes applies: surface phase carries
as OPD, amplitude made by out-of-pupil phase scales as λ, so the servo holds a
MODEL-propagated target rather than the sensed map. That is the deck's
second-color slide's physics and it is the same engine two-plane propagation.
