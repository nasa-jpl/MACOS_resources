# t5 — the unguided re-instance experiment (offset_imager at new parameters)

Run 2026-08-22 by a doc-only proxy user.  Sources allowed and used:
`oi_story` / `offset_imager_params` help text, the template
`README.md`, `challenges/rodgers3/{PACKET.md,README.md}` (incl. the
2026-08-21 addendum), and the two things those documents explicitly
point at (`run_t3.m`, `t3/r3t_REPORT.md`).  No CLAUDE.md, no PLAN/BRIEF
files, no git history, no template implementation source until after a
failure was recorded.

Requested instance: **EPD 150 mm, F/3.3, λ 1 µm, 15°×15° box offset
22.5°, clearances > 40 / > 25 mm, exit beam horizontal, tag `t5`.**
Packaging envelope and seed were mine to choose from the docs.

---

## 1. VERDICT

**FAIL.**  One documented call did not produce a five-stage ladder.
Three runs, three crashes, none past stage 2 of 5; no counter-designs,
no `t5_STORY.md`, no deck-asset manifest.  Every crash is the same
underlying condition — **a field point at the offset that loses all its
rays** — surfacing as an unhandled empty/NaN inside a figure helper,
never as a diagnosed message.

| attempt | envelope choice | got through | died in | cause |
|---|---|---|---|---|
| 1 | 2× rodgers3 (EPD ratio) | S1 (20.9 nm) | `oi_map_fig:43` | S2 dense map **121/121 fields NaN** |
| 2 | 1.65× rodgers3 (EFL ratio, the rescale the docs name) | S1 (30.6 nm) | `oi_layout_fig:140` | S2 field-envelope bundle empty (`sc.rays{1}` is a double) |
| 3 | as attempt 2, `stages=[1 3 4 5]` (documented S2 skip) | S1 (30.6 nm) | `oi_map_fig:43` | S3 dense map all-NaN; solve started from a design scoring the **1e9 no-rays sentinel** |

Time to first successful stage: **19 min 44 s** (attempt 1, incl. MATLAB
start + seed solve); **5 min 17 s** once the seed radius was better
posed (attempt 2).  Total to the FAIL verdict: ~55 min of run time over
three attempts, after ~25 min of reading.

Artifacts: `templates/10_telescopes/offset_imager/t5_unguided/attempt{1,2,3}/`
(per-attempt `t5_REPORT.md` stub, S1 figures + deck, the map/layout PNGs
that were written before each crash, the driver, and the tail of each
console log).

## 2. The ladder I got

Only S1 completed, and only its report section was written.  The metric
tag the template prints, verbatim:

> Metric: strict RMS WFE, centroid reference on the frozen stage FPA,
> exit-pupil anchor, piston-only removal; dense 11x11 map over the
> 15x15° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| stage | attempt 1 | attempt 2 / 3 |
|---|---|---|
| S1 map max | **20.9 nm** at XAN +0.0 YAN +3.0 | **30.6 nm** at XAN −1.5 YAN +1.5 |
| S1 solve | 21661.4 → 16.6 nm qmean, 23 iters | 19889.7 → 23.1 nm qmean, 6 iters |
| radii R1..R3 | 16.967 / −2.073 / −1.847 m | 14.378 / −1.896 / −1.675 m |
| conics K1..K3 | **−401.14** / 5.295 / 0.0691 | **−37.19** / 4.301 / 0.0827 |
| exit chief | 180.000°, err 0.000° vs pin → **PASS** | 180.000°, err 0.000° → **PASS** |
| clearance floor | −99.1 mm → **FAIL** (gate ≥ 25 mm; WARN < 40 mm) | −97.2 mm → **FAIL** |
| S2 | crash | 1 finite field of 121 → crash |
| S3 | not reached | (attempt 3) all-NaN → crash |
| S4, S5, counters (a)/(b), STORY.md | never reached | never reached |

The exit-beam gate works and reads PASS at the pin.  The clearance gate
works, reports signed penetration, and reads FAIL — expected at S1,
which the PACKET says is unconstrained (its own T3 S1 floor is 3.4 mm).

## 3. FRICTION LIST

Numbered, verbatim, in the order encountered.

**F1 — the README's own "Run it" recipe does not run.**  README.md says:

```matlab
run('<path-to>/mmacos/mmacos_setup.m');
OUT = offset_imager();
```

Actual result:

```
mmacos: path set from /home/dcr/dev/MACOS_res_dev/mmacos
Unrecognized function or variable 'offset_imager_params'.
Error in dump_defaults (line 2)
P = offset_imager_params();
```

`mmacos_setup.m` does not put `templates/10_telescopes/offset_imager`
on the path.  **Guessed:** add `addpath('<template dir>')` — because
`run_t3.m`, which the PACKET points at, does exactly that in its body.
A user with only the README has nothing to copy.  Cost: one failed run.

**F2 — `oi_story` is undocumented outside its own help text.**  The
README's "Run it" and "The stages" sections describe `offset_imager`
only; `oi_story` appears in neither the README, the PACKET, nor the
challenge README.  `oi_story` is the function that produces the
counter-designs, the story summary and the deck-asset manifest — i.e.
the whole claimed deliverable.  A new user reading the docs
front-to-back never learns it exists, runs `offset_imager`, and gets no
counters and no `<tag>_STORY.md`.

**F3 — no documented way to choose a packaging envelope.**
`z_m1_m`, `spacings_m` and `seed_R1_m` are declared "the designer's
envelope — inputs, not solved" and are the only parameters the brief
left open.  The docs supply: the rodgers3 values, the (retired) t4
values, and one geometric rule buried in the PACKET addendum §C
("field-walk separation tan(offset)×leg … against a beam-plus-patch
need").  There is no formula, no scaling rule stated as a rule, and no
worked "how to pick one for a new instrument".
**Guessed twice.**  (a) Form-true 2× scale of the rodgers3 envelope,
matched to my 2× EPD, keeping the M1 f/# and the beam-to-leg ratios of
the validated instance: `z_m1_m 1.3299136`,
`spacings_m [-1.4457936 0 1.481656]`, `seed_R1_m 17.6`.  (b) After that
failed, the one rescale the docs actually *name* — addendum §C's
"rodgers3 W-fold × EFL ratio", ratio 0.495/0.300 = 1.65:
`z_m1_m 1.09717872`, `spacings_m [-1.19277972 0 1.2223662]`,
`seed_R1_m 14.52`.  Both pass the addendum's own field-walk arithmetic
by 2×; both fail in the same place.  Neither guess is discoverable as
"the" answer from the docs.

**F4 — `seed_R1_m` silently sets how pathological S1 becomes, and the
docs say nothing about it.**  Its help entry is "M1 radius seed for the
first-order seed solver (the third first-order condition …)".  Between
the two guesses in F3 — R1 seed 17.6 m vs 14.52 m, i.e. R1/EFL 35.6 vs
his 29.3 — the *converged* S1 conic moved from **K1 = −401.14** to
**K1 = −37.19**, and the solve from 23 iterations to 6.  Nothing tells a
user that this knob is load-bearing, what a sane value looks like for a
new EPD/F#, or that a large |K1| is a red flag.  For reference, the T3
reference run's S1 conics are −2.42 / 3.77 / 0.083; both of mine are
one to two orders of magnitude out and the docs give no way to notice.

**F5 — the documented failure mode has no documented guard.**  The
PACKET states plainly: "our S1 solves 4× deeper on-axis than his r1,
and harder-tuned on-axis aspheres cost proportionally more at the
offset — the S2 rung measures the S1 design as much as the offset."
My S1 lands at 20.9 / 30.6 nm — 5–8× deeper than his 159 nm r1 — and the
S2 rung does not merely cost more, it **cannot be traced at all**.  The
docs name the mechanism and offer no knob, no cap, no warning threshold
and no "if your S1 is much better than 150 nm, do X".

**F6 — total ray loss crashes the template instead of reporting.**
Three distinct unhandled-empty crashes, all downstream of the same
condition.  Verbatim:

```
 One or more rays become undefined at element   1
 A total of     1185 rays were lost
   Bracket/iter:       1185
...
{Error using assert
The condition input argument must be convertible to a scalar logical.
Error in oi_map_fig (line 43)
    assert(abs(mp.max_nm - max(W(fin))) < 1e-9, 'oi_map_fig:stats');
```

```
 One or more rays become undefined at element   3
 A total of     1185 rays were lost
   Surface miss:       1173
   Bracket/iter:         12
...
{Brace indexing is not supported for variables of type double.
Error in oi_layout_fig (line 140)
    E = sc.rays{1};  ex_p = E{4}.pos(:,1);  ex_d = E{4}.dir(:,1);
```

(The MATLAB traceback prints the offending source line itself, which is
enough to read the mechanism: `max(W(fin))` returns `[]` when no field
is finite, so `assert` gets an empty condition; and an empty ray bundle
comes back as a double instead of a cell.  No further source reading was
needed.)  Neither message names the
optical condition; a new user sees a MATLAB type error, not "every
field in the box lost all its rays at element 3."

**F7 — a dense-map headline can be computed from ONE surviving field
point, with no warning.**  Attempt 2's S2 map figure is captioned, by
the template:

```
[map max 768194.5 nm, avg 768194.5, std 0.0]
```

`avg == max`, `std == 0`: exactly **one of 121 field points** returned a
finite value; the other 120 are NaN.  Had the layout figure not crashed
one call later, that single point would have been written into the
report as the "dense 11×11 map maximum" — the packet metric — with
nothing in the table, the caption or the console distinguishing it from
a fully-sampled map.  Attempt 1's version of the same figure is
captioned `[map max NaN nm, avg NaN, std NaN]` and still renders.
The failed fraction is never reported anywhere.

**F8 — the flow proceeds from a design it has already scored as
unusable.**  Attempt 3 printed:

```
  S3 start candidates: carry 1000000000.0 nm, fresh seed 1000000000.0 nm -> carry
```

Both candidates score the 1e9 nm no-rays sentinel — the template
*knows*, at that line, that neither start traces — and it selects one,
solves against a flat objective, and dies two minutes later in the map
assert.  A single check at that line would have turned three opaque
crashes into one accurate sentence.

**F9 — `stages` looks like an escape hatch and is not.**  The params
help says "stages: which of S1..S5 to run (default 1:5)", and the
README's stage table invites reading them as separable.  Skipping the
failing S2 (`stages=[1 3 4 5]`, attempt 3) does not help: S3 seeds from
the S1 design, which is what actually cannot trace at the offset.  The
docs do not say which stages seed from which.

**F10 — `clear_m` ordering is undocumented; I guessed and was right.**
The help says only "list of clearance requirements, m".  The brief's
"> 40 / > 25 mm" had to be entered in some order.  **Guessed**
`[0.040 0.025]` by pattern-matching Mike's `[0.050 0.035]`.  Confirmed
correct only after the run, from the report line the template emits:
`clearance floor -99.1 mm (FAIL; gate >= 25 mm; WARN < 40 mm)` — the
*second*, smaller value is the hard gate and the first is a warn level.
That is stated in PACKET §5 prose ("min-over-all-pairs ≥ 35 mm with
WARN < 50 mm") but not in the parameter's own help, which is where a
user setting it is looking.

**F11 — `exit_dir` needs a fact the template docs don't state.**
Default is `[]` = report-only, so "exit beam horizontal" must be
entered as a vector.  The help says "a 1×3 unit vector pins the exit
chief direction (Mike r2+: exit beam horizontal = [0 0 −1] after an odd
mirror count … stated per instance)".  Nothing in the *template's*
README or help says the template builds a **three**-mirror train
(the mirror count appears only in the rodgers3 challenge README, which
is describing Mike's design, not the template).  **Guessed** `[0 0 −1]`
on the odd-count rule; the S1 gate then read `180.000°, err 0.000°`,
which confirms it.

**F12 — the defaults are not the values the only reference run uses.**
`gn_iters` defaults to 12; `run_t3.m` — the sole worked example, which
the PACKET's "Reproduction instructions" point at — passes 30, and
`exit_tol_deg` 0.1 against a default of 1.  Neither departure is
mentioned in any help text or README.  I copied run_t3's 30.  (This
turned out not to matter: attempt 2's S1 converged in 6 iterations, so
the pure-default call would have produced the identical S1 and the
identical S2 failure.)

**F13 — the addendum's headline lesson contradicts the shipped
default, and the template never says so.**  PACKET addendum §B
concludes that the S5 rung was **solve-field-limited**: "9 fields
under-determine 82 variables", and 25 solve fields take the template
from 118.2 nm to 45.4 nm.  The `nsolve` default is still **3** (= 9
fields), the only worked example uses 3, and neither the parameter's
help entry nor the README carries the finding.  I left it at 3 to stay
faithful to "one documented call" and recorded the tension; a user who
read only the help text would never learn to raise it, and a user who
read only the addendum would not know the default had not been changed.

**F14 — no progress output during a multi-hour solve.**  Between the
stage banner and the stage's one-line result, the console carries
nothing but engine trace spam (attempt 1: ~159 000 lines of
`Optical Train Summary` / `RMS OPD error` between "S1" and
"s1: map max 20.9 nm").  No iteration counter, no elapsed time, no ETA,
and no wall-clock figure anywhere in the docs for how long a five-stage
run takes — so there is no way to tell a slow stage from a hung one.
(I twice concluded the job was hung when it had in fact already died.)

**F15 — the run does not clean up after itself and writes a partial
report.**  Each crash left a `t5_REPORT.md` containing the header plus
the S1 section only, with no marker that the run aborted.  Re-running
into the same `outdir` would silently interleave old and new sections.

## 4. What I did NOT need

Never opened, and never missed:

- `oi_solve.m`, `oi_close.m`, `oi_deck.m`, `oi_paraxial.m`, `oi_seed.m`,
  `oi_gates.m`, `oi_score.m`, `oi_clear.m`, `oi_zern_seed.m`,
  `oi_apply_fpa.m`, `oi_fieldset.m`, `offset_imager.m`.  (The two helper
  lines quoted in F6 came from the MATLAB traceback, not from opening
  the files.)
- `rodgers3.m`, `build_r3.m`, `parse_seq.py`, `rodgers3_seq.m`, the five
  `TML_*.seq` decks, `r3_*.in`, `r3_s0_report.txt`, `probe_native_stop.m`,
  `r5_negctl.m`, `run_s5_budget.m`, `run_s5_signed.m`, `s5_budget/`.
- `run_t4.m` and `t4_wide/` — the PACKET's retraction (§C) told me the
  instance was retired and why, which was enough; I never needed the
  artifacts.  This is the docs working.
- `DECK2_PLAN.md`, `deck_rodgers3.*`, `tests/tOffsetImager.m`,
  `tests/tRodgers3.m`, anything under `design/src`.

Two documents earned their keep and should stay on any new user's
reading path: **PACKET.md §2 + the addendum** (the metric statement, the
band rule, the t4 retraction and its geometry lesson) and the
**`offset_imager_params` help text** (the only complete knob list —
every parameter I set, I set from it).  The template README was the
weakest link: it is the entry point, its run recipe does not work (F1),
and it omits the function that does the advertised job (F2).

## 5. The one-line answer

The claim "a re-do with different parameters is this one call" holds as
far as **calling** goes — the call is genuinely one line and the
parameter surface is well documented.  It does not yet hold as far as
**completing** goes: at a parameter set that is not rodgers3, the S1
solve ran away to conics one to two orders of magnitude out, the offset
box became untraceable, and the template had no guard, no diagnostic and
no documented knob to notice or prevent it.
