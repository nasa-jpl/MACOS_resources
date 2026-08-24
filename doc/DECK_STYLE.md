# Technical deck style guide (D. Redding)

Distilled from the rodgers3 deck: a CC-generated workbook draft transformed into the
final presentation. Apply when generating or revising any technical slide deck.

## Structure

- **Lean main path, success story only.** The main slides tell the arc that worked:
  challenge → validation → method → results → discussion. Diagnostics, failed attempts,
  methodology defenses, and reproduction mechanics go to backup — even if hard-won.
- **One plain "Backup Slides" divider** separates main from backup. Do not prefix
  individual backup titles with "Backup ·" or letter/number codes.
- **One idea per slide.** If a slide needs a second table or a fourth dense bullet,
  split it into two slides rather than shrink the text.
- **State metrics and conventions once**, up front (title slide or slide 2). Never
  repeat them as per-slide footers.
- **Every result slide pairs its evidence**: for optical design stages, the layout
  (ray-trace) figure and the performance map together on the stage's slide.

## Titles and text

- **Plain descriptive titles**: "Step 2: the move", "The solver", "Counter-designs
  bound the space". Never coded headers ("C8 — S2, ...") or aphorisms ("the disaster
  is the pedagogy"). Modest and concrete beats dramatic ("First try at a second
  instrument" not "Re-running the family found a broken check").
- **Compact bullets with bold lead-in labels**: `**The fix:** ...`. State the
  mechanism once; do not narrate (no what-we-saw / why / how chains when one tight
  paragraph covers it).
- **No ALL-CAPS emphasis.** Use the bold lead-in instead.
- **Kickers** (one italic line under the title) carry the slide's headline number or
  claim.
- Numbers carry the argument: quote the measured value, the reference value, and the
  ratio.

## Vocabulary

- **Translate project-internal coinages to standard optics/optimization terms** for
  any audience outside the project. Observed mappings:
  - rungs → steps (of the reported design ladder)
  - gates / falsifiers → checks / regression tests / negative controls
  - hinge rows → weighted penalty terms (one-sided soft constraints)
  - walls → hard barrier constraints
  - pinning doctrine → held parameters
  - the metric contract → the WFE metric definition (stated once)
  - arbiter → "defines what X means"
  - honest rows/maps → corrected / signed measures
- Standard technical vocabulary the audience owns is fine (basin of convergence,
  warm start, Gauss–Newton, Zernike) — gloss briefly on first use if the audience
  may vary.

## Framing

- **Compared work is the respected reference standard, not a rival.** e.g. CODE V
  results are "reproduced" and "followed", the source author is credited on the
  title slide, and matching the reference is presented as the license to compare.
- Negative results are stated plainly and priced ("a negative result is a valid
  result"), then resolved — the arc ends on what now works.

## Figures

- **Never regenerate analysis figures**; they are evidence. Extract from the source
  package and place.
- **Minimize whitespace aggressively.** Autocrop margins; for multi-panel (4-view)
  layout figures, recompose at the panel level: blank the centered figure title
  (the slide caption carries it), tight-crop each panel keeping its label,
  reassemble with narrow gutters. Typical gain: panels render 1.5–2× larger at the
  same slide footprint.
- **Captions carry the figure's title/claim** in one italic line; figures with
  self-describing internal labels (map max in the axis label) need no redundant
  caption.
- **Reuse orphaned figures**: when a slide is cut or merged, a strong figure from it
  should fill an empty area on the surviving slide.
- Size maps ≥ ~3 in tall so axis labels read from the back of the room.

## Mechanics (pptxgenjs or equivalent)

- 16:9, white background, restrained two-color palette (deep accent for headings,
  charcoal body), Arial, titles ~26 pt, body 11–13 pt, small gray page numbers.
- Tables: filled header row, alternating row tint, right-aligned numerics.
- Validate and render every revision; visually QA for caption collisions, clipped
  text at the 7.5 in bottom edge, and figure overlap before delivering.
