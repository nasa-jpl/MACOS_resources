# PLAN — a CONFIGURATIONS axis for the sensitivity supervisors

**Status: SKETCH, for Dave's review.  Nothing implemented.**
Written 2026-08-19 (Luis round 2, item 5).  Gated on two inputs that are
not in this tree: (a) Dave's sign-off on the API below, and (b) Luis's
workaround files, which show the call shapes he is hand-weaving today.

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

## 6. Before implementing

- Get Luis's workaround files (item 5d) and check the sketch's call shape
  against what he actually wrote.
- Get a j18-family deck with real steering-mirror compensation states —
  **there is no j18 prescription anywhere in either repo**, so the worked
  example (item 5b) cannot be built until one arrives.
