# e2e2 — the improved TMA design flow

Input parameters → Korsch axial starting point → off-axis → fold → relay
and focal plane → scoring, each stage a gated step that emits a committed
`.in`, a parameter-provenance table, a thorough report, and standard
views.  **The product is the FLOW** — reusable stage drivers a user can
re-parameterize — not one telescope.

Sibling of `../e2e/`, which this supersedes as the design-flow example.
It folds in everything the Rodgers offset-field reproduction taught us
(`../../rodgers1/PACKET.md`, Addenda 8–10): joint solve, stated
references, solve set ≠ scoring set, pupil gate, parameter provenance per
stage.  `e2e` remains the reference for the stages e2e2 does *not*
duplicate — segmentation onward.

All user knobs live in **`e2e2_params.m`** — one file, commented.  Nothing
is computed there: a number in that file was **chosen**, a number in a
report was **derived**.

## The design point

The Rodgers offset-field coaxial TMA, **scaled to D = 3 m** (every length
× 3/5 from his 5 m `.seq` geometry), **f/20** (EFL 60 m) off an f/1.2358
primary, **λ = 500 nm**, a **0.2° × 0.2° used box**, and the M1
perforation scaled with it (0.2056 linear obscuration).  Target:
**diffraction-limited at 500 nm across the used box** — RMS ≤ λ/14 ≈ 36 nm
and Strehl ≥ 0.8, both reported.

Scaling a validated design rather than inventing one buys a **first-order
gate no de-novo layout has**.  f/# is scale-invariant, so the constraint
that pins M3's radius — his `CUY UMY -0.025` paraxial marginal-angle
solve — carries over unchanged, and the builder derives R3 from the f/20
constraint alone.  The two must agree.  They do, to **2.4 ppm**.

Note the wavelength is *not* scaled.  His study closed at ≤ 40 nm at
1 µm; geometric wavefront error scales with the design, so that is ~24 nm
here — the target is reachable in principle, but at 500 nm it is a much
tighter ask in waves, and only at the reference convention his numbers
were reported under.

## Stages

| runner | consumes | produces |
|---|---|---|
| `s1_axial.m` | `e2e2_params.m` | the axial anchor: `s1_axial.in/.mat`, `s1_views.png`, `s1_wfe_field.png`, `s1_report.txt` |

*(stages 2–5 land as they are built; this table grows with them)*

### S1 — the Korsch axial starting point

A coaxial on-axis three-mirror anastigmat, solved jointly with its
detector and scored on a uniform grid.  **Result: 0.638 nm max RMS over
the used box** at the best-focus + LS-tip/tilt rung (0.890 nm at the
centroid reference), Strehl ≥ 0.9999 — a negligible anchor, which is the
whole job of this stage: whatever stage 2 measures as the off-axis
collapse has to be attributable to the field bias and not to a sloppy
starting point.

Three gates run **before any wavefront number is believed**, ordered by
what a failure would invalidate:

1. **The conic solver**, against the shared `tma_fixture.json` — engine-free
   arithmetic.  `max|ΔK| = 4.6e-08` against a 1e-06 bar.  A miss here is
   stop-and-fix; never widen the bar, never hand-edit the fixture.
2. **The first-order layout**, against the scaled reference geometry.
   Derived R3 = 1.612788 m vs 1.612784 m — **2.38e-06 relative**.
3. **The pupil**, against the declared `Aperture=`: greatest chord and
   outermost radius, both 0.997696 ×, zero rays outside.

Free cross-check, reported but deliberately **not** gated: conic
constants are dimensionless, so they do not scale.  A correct solve must
land on the reference stage-1 conics without ever having been given them.
It does — `|ΔK| = [2.9e-06, 4.4e-05, 5.7e-05]` — which validates the whole
chain (layout → builder → CALIB) at once.

## The design procedure — why the solves are staged this way

Each rule below was established by a **failed run**, and the failure is
recorded with it.  The runners implement all of them, so a parameter
change should just work; if you restructure the chain, keep the rules.

1. **Never fit a detector on an unsolved design.**
   `align_focal_plane` fits each field's best focus *from rays*.  Called
   on the K = 0 spherical seed — where this design carries **15 mm RMS**
   of spherical aberration — it locks onto the spherical caustic
   **1.796 m** from the paraxial focus, reports an **11 mm** best-focus
   blur, and moves the detector there.
   *Failed alternative:* the first version of `s1_axial.m` did exactly
   that.  The design still converged — the FPA piston in the joint DOF
   set spent itself undoing the excursion — so nothing looked wrong in
   the WFE.  The damage was silent and downstream: `add_pupil` had
   derived `FP_return` and the ExitPupil sphere from the abandoned
   station, so the **saved deck declared an exit pupil belonging to a
   design that no longer existed**.

2. **And do not fit one on a rotationally symmetric design at all.**
   Solving the conics first does not rescue the fit here.  On a coaxial
   on-axis telescope the per-field foci form a rotationally symmetric
   **disc**: the plane fit's two in-plane singular values are *equal*, its
   basis is arbitrary, and it returned a normal **90° from the arriving
   chief** — a detector edge-on to its own beam.  There is nothing for it
   to find.  Symmetry fixes the normal along the axis; the builder's
   paraxial `derive` places the station (correct to **22 µm** here,
   measured by how far the joint solve then moved it); the FPA piston
   refines what is left.  The fit earns its keep in stage 2, where the
   field bias breaks the symmetry and the focal surface genuinely tilts.

3. **Joint solve, never alternate.**  Conics and the detector's tip and
   focus go into ONE CALIB DOF set.  The merit's reference sphere is
   centred on each field's chief-ray intercept *on the detector*, so
   solve-then-refit-FPA chases two objectives and need not contract —
   measured drifting 0.6–13 mm per round on the rodgers1 TMA.  What S1
   runs is a **seeding sequence**, not an alternation: one on-axis conic
   solve to leave the spherical regime, then one joint solve.

4. **Quote the traced first order, never the paraxial seeder.**  The
   seeder only *places* the focal plane before the solve refines it, and
   it is 96 % wrong on the 100 m Rodgers deck.  Here paraxial and traced
   EFL agree (60.000 vs 59.999 m) — which is a *result*, not a licence to
   quote the seeder elsewhere.

5. **Name the rung every number came from.**  All four reference rungs
   (chief / centroid / +best focus / +LS tip/tilt) are tabled every time;
   the verdict rung only picks which one the pass/fail reads.  On comatic
   fields the rungs spread by 1.3–1.7×, and per-field focus is worth < 3 %
   of that — the tilt treatment is the whole game.

6. **Score on a uniform grid, never on the solve set.**  Statistics come
   from a uniform 9 × 9 over the used box; the solve set is reported
   alongside only to show what the optimizer saw.  An edge-weighted solve
   sampling biases the average ~8 % at an identical max.

7. **Gate the pupil after every source or aperture edit**, on the
   *greatest chord* — never a span.  The ColSource pupil was 5 % oversize
   for decades and read *exactly right* along both axes; only the diagonal
   exposed it (macos PR #70, PACKET Addendum 10).  The engine is correct
   now; the gate stays because the failure mode is silent and the cost is
   one trace.

8. **Frame before angle.**  Every reported tilt names its reference.  The
   focal plane gets both angles it has: the **beam AOI** (arriving chief
   vs the detector normal — the FPA acceptance angle, the assembly driver)
   and the **mechanical tilt** (detector normal vs the optical axis — the
   mount requirement).  A "14.3° tilted detector" that turned out to be
   the same surface as CODE V's −0.07° is what earned this one.

## Found while building this example

- **The best-focus rung was losing to the rung below it.**
  `strict_rungs` slid the reference sphere with `fminbnd` at its default
  `TolX` — 0.1 mm of focus slide on a metre deck, ≈ 31 nm of wavefront at
  f/20.  On a design corrected to the nm level the search cannot resolve
  its own minimum: rung 3 returned **2.386 nm where rung 2 read 0.890 nm**.
  On the rodgers1 deck, ~100× more aberrated, the same defect is 1.7e-4
  relative — invisible exactly where the metric was developed, ruinous
  exactly where it is now used.  Fixed (relative tolerance + an explicit
  `ff(0)` evaluation, so the ordering is true by construction); the
  rodgers1 rung-3/rung-4 artifacts are stale by 2–3e-4 and flagged for
  regeneration as a separate reviewed step.

## Handoff

The `.in` that stage 5 emits feeds the **existing** pipeline unchanged —
`run_segmentation` → `run_sensitivities` → `run_met` → `run_compare` →
`run_simulator` (`design/runners/`, worked end-to-end in `../e2e/`).
Segmentation and everything after it are deliberately **not** duplicated
here.

## House rules

Figures and reports land in this directory.  No `exit(0)` inside these
scripts — batch wrappers supply it.  Run the stages in order with
`run('.../sN_*.m')` after `mmacos_setup`.
