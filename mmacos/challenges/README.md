# Design challenges

A **challenge** is a stated design target — aperture, field, f/#,
packaging, and the metric it will be scored on — paired with our worked
answer in MACOS, so the two can be compared surface by surface.  Where a
template is something you copy and re-knob, a challenge is a fixed
problem with a reference solution: the deck, the script that produced
it, and the numbers it hits.

Scoring rule: every claim in a challenge is a number a re-run reproduces.
Each directory states the metric it is scored on (which sphere the
wavefront is referenced to, which field points, which wavelength, how the
field is sampled and weighted) *before* it reports a value, because the
same design scores differently under different conventions — reproducing
the convention is part of reproducing the answer.

| Challenge | Target |
|---|---|
| [`rodgers1/`](rodgers1) | The wide-field three-mirror anastigmat set from a published CODE V design study — reproduce all three of his designs, then beat them. |
| [`rodgers2/`](rodgers2) | The afocal follow-on: on-axis and off-axis afocal three-mirror forms. |
| [`afocal4/`](afocal4) | The four-mirror afocal trade — pupil control versus wavefront, and what buildability costs. |
| [`rodgers3/`](rodgers3) | The offset-field imager: his 5-rung ladder (159/8810/168/117/53 nm) reproduced from the .seq decks, and the `offset_imager` template run head-to-head. |

Templates live one level over, in [`../templates/`](../templates).
