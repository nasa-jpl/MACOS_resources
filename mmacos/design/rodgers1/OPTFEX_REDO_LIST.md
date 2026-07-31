# OPTFEX redo list — what may improve under the fixed CALIB merit

**Date:** 2026-07-31 · **Engine change:** macos branch `optfex-fix`
(affirmative `OptFEX= Yes` parse + unified `LOptIfFEX` default).
**Evidence:** `PACKET.md` Addendum 5; gates `mmacos/tests/tOptFex.m`.

---

## What changed, and who it can touch

Before the fix, `OptFEX= Yes` was silently a no-op — the parser
(`msmacosio.inc:327-329`) carried only the `'No'` branch, so a prescription
could turn CALIB's per-field FEX **off** and never **on**. Every native
optimizer run to date therefore minimised

> `std(OPL)` to each ray's **own intercept on the terminal FocalPlane**,

i.e. the OPD on the detector *plane*. On a tilted image surface that quantity
carries `(transverse ray aberration) × tan(tilt)` — an artifact that on the
rodgers1 offset TMA is ~22× the wavefront error itself (Addendum 3 §A.1), and
which the optimizer can reduce by moving the image rather than by improving the
wavefront.

**The rule for this list.** A solve is **affected** when its detector plane is
appreciably tilted or displaced relative to the arriving chief ray at the fields
being optimised — because only then does the plane-sampling term diverge from
the wavefront. A solve is **exempt** when the beam meets the focal plane at
(near) normal incidence across the optimised field set, since there
plane-sampling ≈ wavefront and the old merit was already the right quantity to
within `O(tan²)`.

**Nothing on this list is wrong today.** These solves converged against a
well-defined objective and their reported numbers are what that objective gives.
The question is whether a *better* design exists that the old merit could not
see. Redo is Dave's schedule, not a defect backlog.

**Default change — corpus impact: none.** Every prescription in either repo
that carries an optimization block sets `OptFEX` explicitly, and all four say
`No`:

| deck | OptFEX | affected by fix? | affected by default change? |
|---|---|---|---|
| `macos/ZGD_test_files/opt_example.in` | `No` | no — `'N'` branch unchanged | no — explicit |
| `macos/ZGD_test_files/opt_example_asph.in` | `No` | no | no |
| `macos/ZGD_test_files/opt_example_constrained.in` | `No` | no | no |
| `MACOS_resources/pymacos/tests/Rx/opt_example.in` | `No` | no | no |

Documentation-only hits: `macos_f90/Lou-UpdateNotes.txt` (line 652 documents
`OptFEX= YES  % whether do FEX during optimization` — the affirmative behaviour
was always the *intent*), `pymacos/src/pymacos/macos.py:4258` (docstring).
The single real behaviour change is the **interactive `macos` CLI** default on a
hand-written deck that has an Opt block and no `OptFEX=` line (was `.TRUE.`
there, `.FALSE.` on the SMACOS path); **no such deck exists in either repo.**

---

## A. Affected — off-axis / tilted focal plane

| # | item | why affected | expected direction |
|---|---|---|---|
| A1 | `design/rodgers1` stages 3 & 4 (`run_epd4060.m`) | the case that found this: +0.5° offset, FPA tilted 14.3°; merit artifact ~22× the WFE | **improves.** His own conics score 115.3 nm max where ours score 181.2 (Addendum 4 §B); his rigid body is 1.83× deeper than ours. Target: S3 ≤ ~115 nm, S4 ≤ ~1.5× his 39.8 |
| A2 | `design/examples/e2e/s1_telescope.m` | VIS telescope solved at a field bias with an extraction tilt; FP is not normal to the chief | improves; magnitude unknown — s1's reported DL margin is the thing to re-check |
| A3 | `design/examples/e2e/s2_instrument.m` (Offner relay + extraction tilt, field centre −0.7′) | ring-field relay, deliberately off-axis, tilted image; the ±2′ DL and Strehl ≥0.965 claims rest on the solve | improves or unchanged; the **Strehl/DL headline numbers are the ones to re-derive** |
| A4 | `design/examples/tma_offaxis/tma_offaxis.m` | off-axis by construction | improves |
| A5 | `design/examples/tma_unobscured/{tma_unobscured,tma_unobscured_search}.m` | unobscured ⇒ off-axis field bias | improves |
| A6 | `design/examples/rc_unobscured/rc_unobscured.m` | as A5 | improves |
| A7 | `design/examples/tma_centered/{tma_centered_foldfp,tma_centered_fold_search}.m` | fold + FP extraction ⇒ tilted detector | improves |
| A8 | `design/examples/wf2_freeform/wf2_freeform.m` | wide-field freeform at a field bias | improves |
| A9 | `design/examples/tma_freeform/tma_freeform.m`, `freeform_unobscured` | freeform solves over a biased box | improves; freeform DOFs have more freedom to chase the artifact, so possibly the largest change on the list |
| A10 | `design/examples/sz_tma` (sphere+Zernike TMA) | biased box, Zernike DOFs; same reasoning as A9 | improves |
| A11 | `design/examples/tma_3plus1/{tma_3plus1,tma_3plus1_optimize,tma_3plus1_aoi_search}.m` | 3+1 with a fold; detector not normal | improves |
| A12 | `design/src/zern_jacobian_solve.m` + the zern-solve doctrine study | its SVD spectrum and "pathological solve" verdict were computed against the old merit | the **doctrine conclusions should be re-checked**, not just the numbers |
| A13 | `design/src/tma_conic_recipe.m`, `field_zone_lmon.m` | helper solves used by the above | follow A1–A11 |
| A14 | `examples/design/tma_widefield/example_tma_widefield.m`, `rc_offaxis/example_rc_offaxis.m` | off-axis / wide-field | improves |

## B. Exempt — normal-incidence focal plane

| # | item | why exempt |
|---|---|---|
| B1 | `design/examples/tma_onaxis/tma_onaxis.m` | on-axis, FP normal to the chief ⇒ plane-sampling ≈ wavefront |
| B2 | `design/examples/rc_onaxis/rc_onaxis.m` | as B1 |
| B3 | `examples/design/example_telescope_design.m`, `example_telescope_align.m`, `example_align_from_rx.m` | on-axis two-mirror demos |
| B4 | `examples/coronagraph/coro/E3_calib_timing.m` | a **timing** benchmark; the solved values are not a deliverable |
| B5 | `pymacos/tests/test_calib.py`, `mmacos/tests/tCalib.m` | deck sets `OptFEX= No` explicitly and asserts wiring, not optical quality |
| B6 | `mmacos/tests/tDesignOptimize.m`, `tDesignTelescope.m` | assert optimizer plumbing and convergence, not absolute WFE; on-axis fixtures |
| B7 | `macos/ZGD_test_files/opt_example*.in` | explicit `OptFEX= No`; regression fixtures whose value is bit-stability |

## C. Not optimizer output, but downstream of A

| # | item | why |
|---|---|---|
| C1 | e2e s3–s7 (segmentation, MET layout, linear model, simulator) | all built on the s1/s2 solves (A2/A3). If those move, the MET budgets and the 3.4 nm-class numbers move with them |
| C2 | `design/examples/*/**_report.txt`, `*.mat`, `*.png` baselines for every A-row | regenerate with the solve |
| C3 | the design-layer sales-pitch numbers quoted in `PLAN_DESIGN_LAYER.md` | sourced from A-row examples |

---

## Redo protocol (when Dave schedules it)

1. Insert an exit pupil (`add_pupil`) so `nElt-1` is the ExitPupil `Return`; the
   design layer then emits `OptWFElt = nElt-1` and `OptFEX= Yes` and sets the
   stop automatically.
2. **Alternate** solve ↔ `align_focal_plane(...,'allow_pupil',true)`: the merit's
   reference sphere is centred on the chief intercept on the *detector*, so the
   detector must be re-fitted from the converged design and the solve repeated
   until the plane is stationary.
3. Score with `strict_wfe`/`strict_wfe_deck` — an **independent** path from the
   in-loop merit, so a solve that games its own objective shows up.
4. Keep the old result as a suffixed-parallel artifact; do not overwrite.
