# PLAN — a CONFIGURATIONS axis for the sensitivity supervisors

**Status: SKETCH, for Dave's review.  Nothing implemented.**
Written 2026-08-19 (Luis round 2, item 5).  Gated on Dave's sign-off on
the API below, and on the §5 question of whether a compensation state is
a configuration or a DOF.

The fixture is NOT gated: Luis's zoom decks carry proprietary data, so
§6 builds one from `j18sc.in` instead, and its feasibility is already
measured.

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
against the Session (`m.(fname)(args{:})`).  Three properties make this
the right shape rather than, say, a struct of DOF values:

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
  setter-agnostic, and verifiable: the runner re-reads the snapshot after
  restore and asserts it matches, so a configuration that fails to
  restore is a loud error rather than silent contamination of the NEXT
  configuration's block.  **This assertion is the load-bearing part of
  the design** — a per-configuration Jacobian silently computed from the
  previous configuration's geometry is the failure mode that would be
  hardest to notice.

`reload_rx` per configuration is the brute-force alternative.  It is
simpler and obviously correct, but it discards the per-field exit-pupil
state and costs a full parse per configuration; keep it as an escape
hatch (`'config_restore','reload'`) rather than the default.

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

## 5. Open questions for Dave

1. **Is the compensation state a CONFIGURATION or a DOF?**  A steering
   mirror that compensates pointing is, in a controls sense, part of the
   plant, not a discrete zoom.  If the intended use is "evaluate the
   Jacobian at 5 points along a compensator's range", the configuration
   axis is right.  If it is "solve for the compensator", it is a DOF and
   belongs in the channel list.  The sketch assumes the former, because
   that is what "5 zooms × 5 fields" describes.
2. **Should a configuration be allowed to change the FIELD SET?**  A real
   zoom changes the field of view.  The sketch shares one field set
   across configurations (which is what makes row-stacking clean).  If
   per-configuration fields are needed, `config_table` carries its own
   field table and the stacked form still works — but `run_compare`'s
   assumptions need re-checking.
3. **Restore-by-snapshot vs reload** — the sketch's default (§2).
4. **Baseline impact:** none, by construction (`configs` absent ⇒
   identical). Confirm that is the acceptance criterion.

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

## 7. Deliverables and order

1. `run_dwdx_5zoom_5fov.m` — the driver, in its own template directory
   with `j18sc.in` copied beside it (self-contained, per the templates
   rule).  Thin, over `run_sensitivities` with `'configs'`, matching the
   existing `run_dwdx_multi.m` register.
2. The `'configs'` option itself — `macos.dw_dx_multi` first, then the
   `run_sensitivities` pass-through (§3).
3. `run_dwdz_5zoom_5fov.m`, `run_dwdsurf_5zoom_5fov.m`,
   `run_dwdgrid_5zoom_5fov.m` once the axis is proven on dwdx.  The grid
   rung needs `grid_augment_rx` on a 19-segment deck; treat it as its own
   step, not a copy-paste.
4. A `tRunSensitivities` case: `'configs'` absent reproduces the current
   output BYTE-FOR-BYTE (the preserved-surface rule), and a 2-config run
   stacks to `2*Nw` rows with identical `channel_names` per block.

## 8. Still open before writing code

The four questions in §5 — chiefly whether a compensation state is a
configuration or a DOF (§5.1), which decides whether this fixture is the
right shape at all.
