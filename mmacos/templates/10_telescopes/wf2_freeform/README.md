# wf2_freeform — Wide-field 2-mirror freeform designer

Pick the knobs below; the script builds an aplanatic Ritchey-Chretien and shows what a freeform Zernike departure on the secondary buys -- in two views, holding every radius and conic (the layout) FIXED: A. PER-FIELD POWER. At a single off-axis field, freeform NULLS the astigmatism the 2-mirror conics leave behind -- ~25x, to diffraction-limited. B. WIDE-FIELD REALITY. Over a symmetric +-FOV field, freeform extends the diffraction-limited field only modestly: a 2-mirror has just ONE surface away from the pupil, so its departure gives field-LINEAR astigmatism control, which cannot cancel the field- SQUARED astigmatism across a wide field. That is exactly what a THIRD mirror buys (-> the off-axis 3-mirror freeform driver).

## Running it

From this directory, after `run('<path-to>/mmacos/mmacos_setup.m')`:

    wf2_freeform

## What a run gives

At a single off-axis field (7′) a freeform departure on the secondary nulls the conics' residual, 0.0801 to 0.0023 waves; over the whole ±7′ field it buys almost nothing (worst 0.143 waves, the diffraction-limited field unchanged at 5′), because a 2-mirror has one surface away from the pupil. The header's "~25x" per-field figure reads 34x on this run.

## Source

Generated 2026-10-26 from the header of `wf2_freeform.m` and the mmacos manual's run of this template (Part II, macos 1b2ff44e); this README is the template's own now -- edit it with the code.
