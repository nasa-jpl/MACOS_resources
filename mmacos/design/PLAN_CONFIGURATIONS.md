# PLAN — a CONFIGURATIONS axis for the sensitivity supervisors

**Status: EXECUTED 2026-08-20 (branch `dev-candidate`, unpushed —
Dave reviews before any push).**  §7's five deliverables are in, in
order, one commit each.  Where the implementation departs from this
plan, the plan text is left as written and the departure is recorded
here.

* **§7.1 the option** — `'configs'` on all four `macos.dw_d*_multi`,
  with the §2 whitelist, the §2.1 ordering, and the snapshot / restore /
  assert cycle in `src/+macos/private/config_axis.m`.
* **§7.2 the gate** — `tRunSensitivities`, **10/10 green** at execution
  (11/11 where a SegMirMaker build is present; **14/14** after the three
  departure-#6 promotion gates were added 2026-08-21).  The
  preserved-surface rule is verified against a **pre-change git
  worktree**, not merely internally: with `'configs'` absent, dwdx /
  dwdz / dwdsurf come back `isequal` on `dwdxall`, `w0_stacked`,
  `OPDall`, `indxall` and the per-field cells.
* **§7.3 the driver** — `run_dwdx_5zoom_5fov.m`, RUN over the full 25
  blocks: 54585 × 132, rank **127 stacked vs 124 for the nominal
  configuration alone**.  Resumable, checkpoints pruned on success.
* **§7.4 the family** — four drivers shipped, **all four now RUN on the
  §6 fixture** (2026-08-21: the two figure rungs' fixture gap, departure
  #6, is closed — see below).
* **§7.5 `configs_from_table`** — landed early, because both the gate
  and the drivers use it.

### Departures from the plan, and why

1. **Restore is hybrid, not snapshot-only (§2).**  Snapshot write-back
   is exact for the three absolute setters and is used for them.  It is
   NOT sufficient for `perturb`: `CPERTURB_PROG` also moves `xObs`, the
   figure frames `pMon`/`pData`/`pFF`, `CRIncidPosNom`, and the
   metrology and HOE points, and the veneer can read only some of those
   and write fewer — so a write-back would leave a rotated aperture
   frame behind, and the assertion **as originally scoped would not have
   caught it**, because the quantities it re-reads restore fine.
   `perturb` is undone by the engine's own inverse (the negated call),
   which is exact — `Qform` is Rodrigues about the combined axis, the
   pivot algebra composes to the identity, and it is the mechanism
   `RigidBodyChannel.restore` already relies on for every Jacobian
   column.  The snapshot is kept and **widened to `xObs` and the figure
   frame** as the VERIFIER.  Restore writes; snapshot checks.

2. **The assertion tolerance is not ULP-tight.**  The configuration
   element is typically also a Jacobian channel, so between apply and
   assert the channel loop has run ~60 poke/restore cycles on it and
   left 1.7e-12 in the vertex (measured, e5hex1).  A ULP-tight bound
   reports that as a configuration failure — it did, first time.  RTOL
   1e-9 keeps three-plus decades over the observed floor and still fires
   on anything the size of a configuration; the worst drift is printed
   every run.

3. **Row count is sum-over-configurations, not `Nc*Nw` (§3.1).**  A
   configuration that vignettes a field contributes fewer rows
   (measured on e5hex1: 104/104/101/104/89).  Slice a block with
   `indxall.config == c`; the blocks are contiguous.

4. **The canvas is TILED, and the row order is built, not derived**
   (revised 2026-08-21 on Dave's steer).  The configurations sit on
   their own outer grid at the positions their schedule implies — a
   five-state zoom lands on the corners and centre of a 3×3, each cell
   holding that state's whole field canvas, so the zoom grid reads
   exactly as the field grid does inside one cell.  The stacked ROW
   order is built configuration-major instead of being read off that
   canvas: `w` for one configuration stacks its FIELDS, `w` for the run
   stacks the CONFIGURATIONS.  Deriving it would break that — `m2v`
   walks column-major, so any outer layout that varies down a column
   interleaves the blocks.  Both live in `macos.config_canvas`, which
   the four supervisors AND `run_sensitivities`' resume stitch share, so
   the two paths cannot drift apart.  `v2m` scatters by `sub2ind` and is
   indifferent to row order, so the round trip is exact.

5. **`run_sensitivities` gained `'resume_dir'`, `'stop_elt'` and
   `'elts'`.**  The first because §7.3's resumable-workspace requirement
   cannot be met from a driver that makes one call; the second because
   the §6 fixture's stop is an ELEMENT and the deck carries no
   `ApStop=`; the third to scope a harvest (and keep the gate cheap).

6. **CLOSED 2026-08-21 — §7.4's figure rungs now have a fixture.**  Was:
   this deck's 19 segments were `Surface= Conic`, so `find_freeform_elts`
   was empty (no MonZernike channels) and `find_grid_elts` was empty
   **before and after** `grid_augment_rx` (which writes the grid channel
   but does not promote `Surface=`); both figure drivers shipped with a
   preflight that said so instead of returning an empty harvest.  Dave
   approved the fixture promotion; it is done.

   * **The promotion.**  A new reusable helper
     `macos.design.promote_segments_freeform` swaps every segment's
     `Surface= Conic → FreeForm`, injects a `MonZernType= BornWolf`
     channel with zero coefficients, drops the fixture's inert Conic-era
     grid/`ZernCoef` lines (which would go LIVE under FreeForm), and
     synthesizes a clocked Mon frame + `lMon` for the one frame-less
     segment (the on-axis centre).  Applied in place to
     `jwst_ote_designc.in`; the header documents the zero-amplitude
     fixture channels.  Engine facts it turns on (verified against the
     Fortran): a FreeForm's Mon channel is live only when `lMon > 0`
     (`SetFreeFormFlags`: `ifMon = lMon>0` — the sleeper gate);
     `MonZernType= BornWolf → MonZernTypeL = 2` dispatches `ZerntoMon2`;
     an all-zero `MonZernCoef` is inert (`MonomialEval` returns zero sag,
     `ZerntoMon` is a pure linear map).

   * **Inert, and gated (not claimed).**  `test_zoom_fixture_promotion_is_inert`
     traces the committed FreeForm deck against the Conic "before"
     (reconstructed by swapping `Surface=` back — the leftover Mon lines
     are inert on Conic) and pins `max|dOPD| ≤ 1e-8 mm`.  Measured
     **7.3e-11 mm** (sub-picometer, ~3e-8 waves): the Conic-vs-FreeForm
     intersection-solver floor, not any added wavefront.  Not ULP-tight
     for the same reason `config_axis`'s restore RTOL is not — different
     solvers, not different physics.  The conic base and every pose are
     unchanged, so `dw/dx` and `dw/dsurf` and their committed artifacts
     are untouched (not regenerated).

   * **Both rungs RUN.**  `run_dwdz_5zoom_5fov.m` (MonZernike): 54585×57,
     25 blocks, rank 55.  `run_dwdgrid_5zoom_5fov.m` (segment grid):
     54585×57, 25 blocks, rank 55 — full modal rank IS the localization
     gate (a wrong-frame poke collapses it, the e5pie 15/42 failure).
     The grid rung anchors `segment_grid_basis` at a SEGMENT
     (`pm_ref_elt = 4`), because this deck's primary is the segmented
     set with no dedicated near-pupil Reference; the default anchor gave
     degenerate footprints.  Two gates added:
     `test_promoted_fixture_feeds_both_figure_rungs` (non-empty, non-zero
     Jacobian per rung) and `test_promoted_segment_poke_localizes` (a
     MonZern poke on the synthesized centre-segment frame and an outer
     segment have near-disjoint support).  `tRunSensitivities` **14/14**.

   A robustness fix landed earlier (before the promotion): `grid_augment_rx`
   emitted the grid channel at the block's `zMon=` line and read `lMon`
   from a map filled in file order, so a deck that declares `lMon` AFTER
   `zMon` — this one — aborted.  It now pre-scans, and parses Fortran `D`
   exponents (`sscanf('%g')` stopped at the `D` and would have returned
   the mantissa).

**Original plan follows, unchanged.**

**Status: APPROVED FOR EXECUTION (Dave, 2026-08-20).  Nothing
implemented yet.**  Written 2026-08-19 (Luis round 2, item 5); reviewed
and amended 2026-08-20 — the §5 questions are SETTLED (answers recorded
in place), and four review fixes are folded in: the v1 setter
whitelist (§2), the reload hatch's session-state obligation (§2), the
exit-pupil policy per block (§2.1), and the §7 corrections.

The fixture is committed: Luis's zoom decks carry proprietary data, so
§6 builds one from `j18sc.in` instead (staged as
`jwst_ote_designc.in`), and its feasibility is already measured.

## 1. The requirement, as understood

A prescription can carry **configuration states** — "zoom positions" in
the classical sense, but in our systems more often a *compensation*
state: the j18-family steering mirror at a pupil fold that is re-pointed
to cancel spacecraft pointing drift.  A configuration is a named set of
element setting overrides.

Sensitivities must be evaluated per **(configuration, field)** — e.g.
5 zooms × 5 fields = 25 blocks — from ONE call, without a user weaving
per-configuration driver files around our supervisors.

Today the supervisors have a field axis and no configuration axis.  The
only way to get a second configuration is a second `.in` and a second
call, and then the caller owns stitching the blocks together — which is
exactly what Luis is doing by hand.

## 2. What a configuration is

```matlab
cfg = struct( ...
  'name', 'zoom3', ...
  'set',  { { ...
      {'perturb', 4, 'rotation', [0; 1.2e-4; 0], 'frame','local'}, ...
      {'set_elt_vpt', 7, [0; 0; 812.5]} ...
  } } );
```

A **list of setter invocations**, each `{fname, args...}` dispatched
against the Session (`m.(fname)(args{:})`).

**v1 WHITELIST (review fix, 2026-08-20).**  The snapshot below records
POSE state only, so v1 accepts only setters whose effects the snapshot
can restore: `perturb`, `set_elt_vpt`, `set_elt_psi`, `set_elt_rpt`,
`set_elt_csys`.  Any other name in a configuration list is a LOUD error
at validation time, before anything is applied.  The Session surface
also carries `set_elt_kr/kc`, `set_elt_zrn_coef`, `set_elt_grid`,
grating and `set_src_*` setters — a configuration invoking those would
apply cleanly and then RESTORE SILENTLY WRONG, which is exactly the
contamination failure the assertion exists to prevent.  Extending the
axis to those setters means extending the snapshot per state category
first; that is a later, separate step.

Three properties make the setter-list the right shape rather than, say,
a struct of DOF values:

1. **It is the surface we already have.**  Anything a user can do to an
   element between two traces is expressible; nothing new to design.
2. **It is inspectable and printable** — the report can name what each
   configuration did, and the `.mat` can carry it verbatim.
3. **It composes with the standing gotchas.**  The runner applies the
   list, calls `modify()` once, and restores afterwards — the
   modify()-after-set rule is enforced in ONE place instead of in every
   user's driver.

### Apply / restore

Restore is the part that must not be hand-rolled.  Two candidate
mechanisms, and the sketch takes the second:

- *Inverse setters* (apply `-delta` for perturb, re-write the saved
  value for absolute setters).  Cheap, but every new setter needs an
  inverse and a perturb that changes vertex AND frame is easy to get
  subtly wrong.
- **Snapshot / restore the touched elements.**  Before a configuration,
  record `get_elt_vpt/psi/rpt` (+ `get_elt_csys`) for every element the
  list mentions; after the field loop, write them back.  Element-scoped,
  setter-agnostic (within the §2 whitelist), and verifiable: the runner
  re-reads the snapshot after restore and asserts it matches, so a
  configuration that fails to restore is a loud error rather than silent
  contamination of the NEXT configuration's block.  **This assertion is
  the load-bearing part of the design** — a per-configuration Jacobian
  silently computed from the previous configuration's geometry is the
  failure mode that would be hardest to notice.

Snapshot scope addition (review fix): also record element `nElt-1` (the
`Return` the exit-pupil machinery writes into — see §2.1) at run START
and restore it at run END, so the session is left clean; EXCLUDE
`nElt-1` from the per-configuration assertion, because FEX legitimately
rewrites it every field.

`reload_rx` per configuration is the brute-force alternative.  It is
simpler-looking but NOT obviously correct: a reload discards SESSION
state, not just parse effort — the stop set by `stop_info_set` (the §6
fixture's stop-at-25 exists ONLY as session state; the deck carries no
`ApStop=`), the sampling override, and the OPD reference
(`opd_ref_set` is reset by every load).  An escape hatch
(`'config_restore','reload'`) that fails to re-apply those produces
wrong-stop Jacobians that look fine.  If the hatch is kept, it MUST
re-apply stop/sampling/opd_ref after every reload and assert the stop
element matches; otherwise drop it.

### 2.1 Exit-pupil policy per configuration block (review fix)

`dw_d*_multi` already re-finds the exit pupil PER FIELD
(`'reset_xp'` default true; FEX writes the pupil reference into element
`nElt-1`).  The configuration axis composes with that as follows, and
the implementation must keep this order:

1. apply the configuration's setter list;
2. `modify()` once;
3. call the supervisor for the block — its per-field `reset_xp` then
   derives every field's exit pupil FROM THE CONFIGURED GEOMETRY (a
   pupil-fold FSM tilt moves the EP; that is physics, not drift);
4. restore the snapshot; assert (excluding `nElt-1`).

Consequence of (3): `nElt-1` carries per-(configuration, field) pupil
state during the run — which is why it is in the run-level snapshot and
out of the per-configuration assertion.

## 3. Proposed API

### 3.1 Supervisors (`macos.dw_d*_multi`)

One new option, defaulting to the current behaviour:

```matlab
out = macos.dw_dx_multi(m, rx, 'field_x_rad', fx, 'field_y_rad', fy, ...
                        'configs', cfgs);      % cfgs: 1xNc struct array
```

* `'configs'` **absent or empty ⇒ byte-identical to today's call.**
  This is the run_dwd* preserved-surface rule; every existing runner,
  example, and committed baseline must be unaffected, and that is a
  gate, not an aspiration.
* Output gains a leading configuration axis:

  | field | today | with configs |
  |---|---|---|
  | `per_field_dwdx` | `Nf x 1` cell | `Nc x Nf` cell |
  | `field_table` | `Nf x 4` | unchanged (fields are shared) |
  | `config_table` | — | `Nc x 1` struct: name + the setter list |
  | `dwdxall` | `Nw x Nz` | `(Nc*Nw) x Nz` — configurations stack as extra ROWS |
  | `w0_stacked` | `Nw x 1` | `(Nc*Nw) x 1` |
  | `indxall` | i/j/size | gains `config` index |

  **Rows, not a third dimension.**  The canonical form is
  `wall = J*x + w0`; a configuration adds observations of the SAME state
  vector `x`, exactly as a field point does.  Stacking rows keeps every
  downstream consumer (`run_compare`, the MET optimiser, the simulator)
  working unchanged — they already treat the stacked wavefront as opaque.
  A third array dimension would break all of them.

* Column identity is asserted, not assumed: the channel list is built
  ONCE, before the configuration loop, and each configuration's block is
  checked to have produced the same `channel_names`.  A configuration
  that changes the element count (it must not) is an error.

### 3.2 Runner (`design/runners/run_sensitivities.m`)

```matlab
art = run_sensitivities(rx, 'fov_rad', F, 'configs', cfgs, ...)
```

Pass-through only; the report gains a per-configuration section and the
SV spectrum is plotted per configuration and for the stacked system (the
stacked one is the design-relevant number: it is the conditioning of the
estimation problem you actually have).

### 3.3 Constructing configurations

A small helper so the common case is one line rather than nested cells:

```matlab
cfgs = macos.design.configs_from_table(T)
```
with `T` a table whose first column is the configuration name and whose
remaining columns are `elt.dof` (`4.Ry`, `7.Tz`, ...) — the shape a
zoom/compensation schedule naturally arrives in (a spreadsheet).

## 4. What this does NOT do

- **No engine change.**  Configurations are a binding-layer loop.
- **No multi-configuration OPTIMISATION.**  Evaluating sensitivities per
  configuration is this item; optimising a design across configurations
  (a genuine multi-config merit function, CALIB-side) is a separate,
  larger question.
- **No zoom interpolation.**  Configurations are discrete named states.

## 5. Questions — SETTLED (Dave, 2026-08-20, via review)

1. **Configuration, not DOF.**  The requirement is "evaluate the
   Jacobian at N points along the compensator's range", so the
   configuration axis is right — AND the same element may
   simultaneously be a channel column (§6's modelling point); the two
   roles compose, they do not compete.  If a later use case is "solve
   for the compensator", that is a channel-list entry, not a
   configuration.
2. **One field set shared across configurations.**  Per-configuration
   field tables are deferred until a real zoom needs them (and then
   `run_compare`'s assumptions get re-checked first).
3. **Restore-by-snapshot is the default**; the reload hatch only
   survives with the session-state re-application obligation of §2.
4. **Byte-identical when `configs` is absent IS the acceptance
   criterion** (the preserved-surface rule), gated by the §7 test.

## 6. The fixture — built here, not sourced

Luis's zoom decks carry proprietary data and cannot be shared, so the
worked example uses a deck already in hand.  It is optically meaningless
by construction; the deliverable is the DRIVER, and the numbers only have
to be finite, reproducible, and responsive.

**Deck:** `templates/50_sensitivities/zoom_5x5/jwst_ote_designc.in` —
staged 2026-08-19 from the sandbox `j18sc.in`, body byte-identical, header
rewritten.  Element 25 is `FSM`, a flat `Reflector` at a pupil — the
steering mirror the configuration axis moves.  Element 27 (`nElt-1`) is
`ExitPupil`, a `Return`, so `reset_xp`'s FEX write has somewhere to land.

Provenance, since the deck is now in a repo whose `dev` and `main` are
public: the three powered mirrors match the published JWST OTE
prescription to the published precision (McElwain et al. 2023, PASP 135,
058001, Table 2 — open access, CC-BY 3.0; design source TRW).  It is NOT
the flight prescription (15 mm segment gaps and 1313.25 mm flat-to-flat
segments against the flight 7 mm and 1.32 m — a pre-freeze design study),
and its fold mirrors are ours, added to unfold the train, carrying no
design authority.  The original header's "From unfolded LM Rx" was
incorrect and is removed.

**Configurations (5).**  Centred, then the FSM tilted 0.5 arcmin
(1.45444e-4 rad) to each corner of a square, applied as a LOCAL-frame
rotation `[±0.5', ±0.5', 0]`:

    z0   [ 0      0     ]
    zUL  [-0.5'  +0.5'  ]      zUR  [+0.5'  +0.5' ]
    zLL  [-0.5'  -0.5'  ]      zLR  [+0.5'  -0.5' ]

**Fields (5).**  The stock `make_5field_set` at
`field_x_rad = field_y_rad = 1 arcmin` (2.90888e-4 rad): centre plus four
corners.  25 blocks.

**Settings.** `model_size 512`, `ngridpts 63` (the deck declares
`nGridpts=1024`), stop at element 25 — the deck carries no `ApStop=` and
`reset_xp` requires one.

### Feasibility, measured before planning

Probe at model 512 / ngridpts 63 / stop 25, OPD at element 27:

| state | rays | valid | lost | RMS WFE (mm) |
|---|---|---|---|---|
| nominal | 2301 | 2184 | 0 | 6.846e-06 |
| FSM 0.5' corners (4) | 2301 | 2184 | 0 | 1.457e-02 … 1.460e-02 |
| field ±1' (4 corners) | 2301 | 2182–2184 | 0 | 6.381e-01 … 6.393e-01 |

No ray loss in any of the 8 perturbed states, and both axes move the
wavefront by orders of magnitude — the finite-difference machinery has
something to differentiate.  117 rays are obscured throughout (the
central obscuration), and the chief ray runs `LRayOK=1, LRayPass=0,
RayStatus=Obscured`, so the fixture also exercises the obscured-chief OPD
reference (§1.2 of `doc/opd_conventions.md`).

The ±1 arcmin field WFE is 0.64 mm, about 278 waves at the deck's 2.3 µm.
That is not a design; it is a load case.  Say so in the driver header so
no reader mistakes the numbers for a result.

### The modelling point this fixture forces

The configuration element (25) is ALSO one of the elements whose
rigid-body DOFs are Jacobian channels.  So a configuration moves an
optic that is itself a variable, and element 25's columns are then
evaluated about a SHIFTED operating point.  That is not a conflict to
design around — it is what a zoom-dependent sensitivity *is*.  Two
consequences the implementation must honour:

- snapshot/restore (§2) must cover the configuration element even though
  the channel machinery also perturbs it, and the restore assertion must
  run AFTER the channel loop has finished restoring its own poke, not
  interleaved with it;
- the channel list is built once, before the configuration loop, so
  element 25's channels are the same columns in all 25 blocks.  Assert
  `channel_names` equality per block; a configuration that changed the
  element count would silently misalign the stack.

## 7. Deliverables, in BUILD order (review fix: option before driver)

1. The `'configs'` option itself — `macos.dw_dx_multi` first, then the
   `run_sensitivities` pass-through (§3), with the §2 whitelist,
   snapshot/restore/assertion, and the §2.1 ordering.
2. The `tRunSensitivities` case, WITH deliverable 1: `'configs'` absent
   reproduces the current output BYTE-FOR-BYTE (the preserved-surface
   rule), and a 2-config run stacks to `2*Nw` rows with identical
   `channel_names` per block and a passing restore assertion.
3. `run_dwdx_5zoom_5fov.m` — the driver, in
   `templates/50_sensitivities/zoom_5x5/` beside the committed
   `jwst_ote_designc.in` (self-contained, per the templates rule).
   Thin, over `run_sensitivities` with `'configs'`, matching the
   existing `run_dwdx_multi.m` register.  The 25-block run gets the
   resumable-workspace treatment: save per-block progress into the
   artifact dir so a killed run resumes, and prune on success.
4. `run_dwdz_5zoom_5fov.m`, `run_dwdsurf_5zoom_5fov.m`,
   `run_dwdgrid_5zoom_5fov.m` once the axis is proven on dwdx.  The grid
   rung needs `grid_augment_rx` on a 19-segment deck; treat it as its own
   step, not a copy-paste.  dwdgrid also touches grid STATE — check the
   §2 whitelist interaction before starting (a configuration must still
   be pose-only there).

## 8. Still open before writing code

Nothing.  §5 is settled; execute §7 in order.
