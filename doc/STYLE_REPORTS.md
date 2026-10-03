# Report & deck style — standing rules (reviewed before every write)

> Governs every generated .pptx / .docx / report .md (deck sources,
> PACKET/RESULTS/STATUS files, .doc reports).  LIVING DOCUMENT, jointly
> developed: Dave's rulings are appended dated in §6; superseded rules
> are struck, not deleted.  THE GATE: before building or sending any
> artifact, the author (human or agent) runs the §5 checklist against
> the source and reports the result — violations fixed or waived by
> Dave, never silently shipped.  Outward-facing artifacts (to Mike,
> meetings, external readers) additionally get Dave's sign-off on the
> source before send.

## 1. Register

- Academic, depersonalized, active voice.  "The comparison exposed…",
  never "we were surprised…".  No first-person narrative in
  deliverables.
- Facts with provenance.  Every number carries units, its reference/
  convention, and its source (run, commit, or file) — a tilt without
  its frame, or a WFE without its rung, is not a result.
- No fluff: no rhetorical openers, superlatives, "exciting" framing,
  or filler adverbs.  Delete any sentence that survives deletion.
- Avoid jargon: be literal and use short words.  Name things by what
  they are ("the last mirror moves in front of M1", not "the back-end
  parity flips"); coined shorthand ("the fork", "basin 2", "girth")
  stays in working notes — a deliverable defines a term once or does
  without it.
- Depersonalized except when essential to name: "the delivered
  design", "the reported values", "the CODE V decks" — not "his".
  A person is named ("Rodgers", "Mike") only when the sentence is
  about the person: an action item, a question to them, an
  attribution that matters.
- Significant figures match the measurement: do not quote six digits
  of a 2% effect.

## 2. Concision mechanics

- Every sentence carries a number, a mechanism, or a decision.  A
  sentence with none of the three is cut.
- Never restate in prose what an adjacent table or figure already
  shows; prose adds interpretation or consequence only.
- One idea per bullet; bullets ≤ 2 rendered lines wherever possible;
  ≤ 5 bullets per slide column.
- Findings lead with the verdict, then the evidence ("X is not
  buildable: every design puts…", not a chronology of how it was
  discovered).  Chronology belongs in PACKETs, not decks.

## 3. Decks (.pptx via make_brief_slides.py)

- Slide title = the claim; subhead = the evidence or consequence in
  one line.  A reader of titles alone gets the argument.
- ≤ 7 content slides for a study deck; status decks one slide per
  major thread.  More content → a second deck, not denser slides.
- Tables: short enumerable facts; targets in the final row/column
  when scoring; no explanation inside cells.
- Footnote line (~) carries caveats, in-work flags, and conventions —
  never new results.
- Questions for the audience are stated as questions, on one slide,
  not scattered.
- Figures: deck-grade only (white ground, stated scale bar, labeled
  elements, insets where the subject is pixels at architecture scale,
  labels never on ray lines).  Caption = one sentence saying what to
  SEE, not what the figure is.  Plot conventions per the demo-plot
  rules (exact Strehl from OPD, autoscaled panels, non-obscuring
  legends).

## 4. Documents (.docx via pandoc; PACKET/RESULTS/STATUS .md)

- One-page summary first ("verdict up front"), then detail.
- Retract in place: corrections are appended and dated; history is
  never rewritten.
- Every section that reports numbers ends with an artifacts list
  (files + how to reproduce, one command each).
- Rules-earned sections state the rule AND the failed alternative
  that earned it.

## 5. The pre-write checklist (run and report before every build)

1. Titles alone carry the argument?           [decks]
2. Any sentence without number/mechanism/decision?  Cut it.
3. Any prose restating a table/figure?        Cut it.
4. Every number: units + convention/reference + provenance?
5. Caveats and in-work items in footnotes, not buried or omitted?
6. Figures deck-grade; captions say what to see?
7. Length inside budget (§3)?  If not, split, don't compress.
8. Register scan: first person, superlatives, fluff → rewrite.
Report format, in-window before building: "Style gate: clean" or the
violation list with fixes.  Outward-facing: Dave approves the source
before send.

## 6. Rulings (append dated; strike, don't delete)

- 2026-08-01 (Dave): concise, active voice, facts only, no fluff,
  academic, depersonalized; match an existing deck's register when
  editing it.  (Origin of §1.)
- 2026-08-04 (Dave): this file created; the §5 gate becomes standing —
  reviewed before every write, jointly amended here.
- 2026-08-04 (Dave): avoid jargon — be literal, use short words.
  (Folded into §1.  Earned by the Rodgers decks: "parity flip",
  "basin", "girth", "the fork".)
- 2026-08-04 (Dave): depersonalized except when essential to name —
  "his" becomes "the delivered/reported/CODE V …" or names the person
  outright when the person is the point.  (Folded into §1.)
- 2026-08-04 (Dave): claims of feasibility stay modest — meeting a
  packaging constraint is a screen, not a layout; say what still
  needs work (the Rodgers2 "buildable" de-emphasis: the long back
  focal distance needs serious repackaging).
- 2026-08-24 (Dave, via the rodgers3 final-deck edit with Claude AI):
  **doc/DECK_STYLE.md adopted as the governing guide for DECK
  preparation** — distilled from transforming the CC-generated
  workbook draft into deck_rodgers3_final.pptx.  Where it supersedes
  prior deck practice: lean success-story main path with diagnostics
  and reproduction mechanics in a Backup section (one plain divider);
  plain descriptive titles (no coded headers like "C8 — S2", no
  aphorisms); metric/conventions stated ONCE up front, never as
  per-slide footers; no ALL-CAPS emphasis (bold lead-in labels
  instead); kickers carry the headline number; project-internal
  coinages translated to standard optics/optimization vocabulary
  (rungs→steps, gates→checks/negative controls, hinge rows→weighted
  penalty terms, walls→hard constraints, pinning→held parameters);
  every result slide pairs layout + performance map; figures
  autocropped and panel-recomposed, never regenerated.  The §5 gate
  stands and is run against BOTH files for decks; the report-side
  rules of THIS file are unchanged.  Slide tooling (kickers, backup
  divider, panel recompose) may require extending the committed
  builder — pptxgenjs or equivalent per DECK_STYLE Mechanics.
