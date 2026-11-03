# PLAN — deck 2: the re-walk deck (comprehensive)

**Status: PLAN, for Dave's review (2026-08-22).  Nothing built.**
Deck 1 (`deck_rodgers3.pptx`) argues the RESULT in 8 slides.  Deck 2
teaches the SOLUTION: a CodeV-literate reader re-walks the whole thing
from this deck + the repo, with nothing left in anyone's head.  Same
concise register (STYLE_REPORTS §1-§3: titles are claims, numbers carry
conventions, no fluff) — but a different GENRE, declared up front:

> **Genre + budget deviation (for the §5 gate).**  This is a WORKBOOK
> deck: ~16 content slides in six lettered sections, each section
> inside the ≤7-slide spirit.  The §3 "second deck, not denser slides"
> rule is exactly what this deck is; Dave waives or trims at review.

Generator: `deck_rodgers3_walk.py`, same discipline as deck 1 — every
number parsed from committed records (r3t_REPORT, r3_s0_report, PACKET
+ addendum, s5_budget mat/pngs), figures = committed PNGs, built via
the committed `make_brief_slides.py`.  Never hand-edit the .pptx.

## Sections and slides

**Cover** — title + the metric contract stated ONCE, large (strict RMS
WFE, centroid ref, exit-pupil anchor, piston-only, dense 11×11 map MAX
— every number in the deck inherits this).  DRAFT tag until sign-off.

**A. The problem and its truth files (2)**
1. The challenge as received: system, constraints, the five reported
   rungs.  What is committed vs referenced (the .seq decks ARE here;
   the source pptx is not).
2. Reading the truth: `rodgers3_seq` (machine-generated), packaging
   READ AT RUNTIME (stations≠spacings trap, the retracted F/4.95),
   the metric decode from the slides' own EMF metadata + the 2989×
   negative control.  Claim: decode conventions, then gate them —
   never assume.

**B. The machinery (4)**
3. The five-stage flow + first-order identities (EFL exact, Petzval,
   stop pose, FP pose — re-derived every iterate, never penalized).
4. The solver: per-ray GN residuals (per-field RMS stalls — show the
   16 µm→nm one-iteration recovery), vertex-radial scales (r^8
   under-scaling kills FD), chord steps, plateau rule.
5. Constraints as ROWS: the exit-direction equality (why a wall
   freezes a non-compliant start); the clearance hinge — and the
   three-defect story as the WHY behind the signed piercing measure
   (sampled-past-piercing / no-gradient-at-zero / unpaired rays),
   with the t4 blocking figure as the exhibit.  Claim: hardware
   drawings are gates.
6. Scoring and gates: solve set ≠ scoring set (and the S5 lesson:
   solve fields must match the variable count), oi_gates' knee, the
   hull-vs-disk glass models.

**C. The walk (5)**
7-11. S1 → S2 → S3 (two-candidate start) → S4 (buildability: 58.3
   unconstrained through glass → 113.6 @ 34.11 confirmed) → S5
   (of-record 118.2, probes A/B/C table, the honest 45.4 @ 33.4).
   Each slide: what is varied, what is held, the command, the figure,
   the number with its gate.

**D. Counters and validation (2)**
12. Counter-designs (sz-start 73.1; released-terms 1373 — pinning
    doctrine) — what they prove and what they bound.
13. The gate map: which claim lives in which test (tRodgers3 bands +
    negctl, tOffsetImager smoke, the oi_clear reference floors), and
    the suite counts.

**E. Re-walk instructions (1)**
14. The exact sequence: `rodgers3()`, `run_t3()` / `oi_story(...)`,
    `run_s5_budget()`, `run_s5_signed()` — with runtimes, expected
    artifacts, and the drift tolerances (what "reproduced" means
    quantitatively).  One slide, one column, monospace.

**F. Discussion (2, questions stated as questions)**
15. The unguided-runner experiment: `oi_story` at a NEW parameter set,
    driven by an agent restricted to the committed docs (the
    new-user proxy; see BRIEF_oi_unguided).  Result slide — either
    "the template re-instances hands-off" with the new ladder, or the
    friction list verbatim ("or not!").  BLOCKED on that run.
16. Should mmacos embed an agent helper?  The evidence from this arc,
    both ways:
    - FOR: every defect found here was a CONCEPTUAL coupling (frames,
      metrics, constraint models) that documentation had not
      prevented; agent-in-the-loop found and fixed them in hours; the
      gotcha corpus (CLAUDE.md lineage) already exists and could ship
      as a runtime advisor (`macos.assist`).
    - AGAINST: the same knowledge can be COMPILED — validators,
      preflights, gates (the prescription validator, the pose-bound,
      the piercing measure ARE compiled lore); slide 15's experiment
      measures how far docs+gates alone go; a runtime AI adds
      maintenance, trust, and opacity questions.
    - The question for the audience: where is the line between
      knowledge compiled into checks and knowledge served by an
      assistant?  What would YOU want it to catch at call time?

## Order of work

1. Slide-15 input: the unguided run (TO, BRIEF_oi_unguided) — start
   it FIRST, it is the long pole and its outcome shapes §F.
2. `deck_rodgers3_walk.py` sections A-E (records already exist).
3. §F once the run lands; §5 style gate run and reported; STOP for
   Dave's sign-off (outward-facing, same rule as deck 1).
