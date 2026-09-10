# The sensitivity (Jacobian) tools — call graph and debugging map

One page for the "many levels of bridge functions" problem (Luis,
2026-09-05).  Until the shared-core rearchitecture lands
(BRIEF_luis_round3 S3), this is the map of what calls what, where
element eligibility is decided, and where to look first.

## The stack, top to bottom

```
dw_dx_multi ─┐
dw_dsurf_multi ├─ multi-field SUPERVISORS (one per family, ~700 lines
dw_dz_zernike_multi │  each; field loop + 'configs' axis + reset_xp /
dw_dgrid_multi ─┘   pupil_find machinery — currently DUPLICATED 4x)
      │  loops fields/configs, forwards 'elts', 'params', 'delta', …
      ▼
dw_dx / dw_dsurf / dw_dz_zernike / dw_dgrid
      │  single-field DRIVERS: load Rx (reload_rx), build channels,
      │  hand them to the shared FD engine
      ▼
+channels/  CHANNEL BUILDERS (one per DOF family)
      │  rigid_body_channels    ← parse_rx_actual_optic_elts_ (Rx text)
      │  surf_channels          ← find_powered_elts   (ENGINE query)
      │  zernike_channels       ← find_zern_elts      (Rx text)
      │  grid_channels          ← find_grid_elts      (ENGINE query)
      │  freeform_{monzern,ffzern}_channels ← find_freeform_elts (engine)
      │
      │  ALL SIX apply the explicit-'elts' contract via ONE validator:
      │  +channels/private/require_elts.m — an id you explicitly
      │  request that cannot be served ERRORS
      │  (macos:channels:eltNotEligible) with a named reason.
      │  Auto-discovery (elts=[]) filters silently.
      ▼
Channel objects (SurfChannel, RigidBodyChannel, ZernChannel, …)
      │  one poke DOF each: apply(+d) / apply(-d) / undo via Session
      ▼
dwdz_for_current_source        ← the ONE finite-difference engine
      │  loops channels, calls wf_func() per poke, assembles columns
      ▼
Session / mmacos mex → SMACOS engine (trace + opd)
```

## OPD conventions carried by every driver (orient / sign / opd_ref)

All eight `dw_d*` drivers (four singles, four `_multi` fronts), the
supervisor core and `run_sensitivities` take the same three options and
record them in the output (`opd_orient`, `opd_sign`; the reference is
re-applied after EVERY Rx reload, because a load resets it):

| option | values | what it does |
|---|---|---|
| `orient` | `raw` (default) / `xy` | array layout; `xy` = index 1 along global X, `imagesc`-ready (doc/opd_conventions.md) |
| `sign` | `opl` (default) / `wavefront` | negate every wavefront-valued output |
| `opd_ref` | `mean` (default) / `chief` | the OPD reference: whole-aperture mean OPL, or the chief ray's own OPL (`macos.opd_ref`, PLAN 0.x) |

**Why `opd_ref` matters on a SEGMENTED deck (Luis 2026-09-09, the
"residual on the other segments" report).**  Neither reference is wrong:
the two columns are the SAME correct data differing by one constant
(Dave 2026-09-10) -- what changes is what the unpoked segments READ.  A
Jacobian column is an OPD DIFFERENCE, and each OPD map is referenced.  Under `mean`, poking ONE
segment shifts the aperture mean, so every OTHER segment reads the
constant `-(N_k/N) * mean(poked response)` -- measured through
`macos.dw_dsurf` on e5hex1 (segment 2, Kr / Kc, orient xy, no PTT
removal): 4.32e-4 / 2.04e-2 per unit parameter, i.e. 14.7% / 12.0% of
the poked segment's rms, identical on all six other segments to 3e-16.
Under `chief` the other segments read EXACTLY 0.  The one case `chief`
does not localise is the chief ray's OWN segment: its reference moves
with the poke and the others read `-m(chief)` (measured: -2.18e-5 per unit
Kr on e5hex1, 5.0% of the centre segment's own rms).  A nominal-anchored
FIXED-LENGTH reference (the engine's `OPDRefRayLen` branch, one api
wrapper away) would localise every column; until then use
`'opd_ref','chief'` and read the centre-segment column knowing that.
`surf_remove_ptt` / `remove_ptt` is a further convention on the same
data: it fits global piston/tip/tilt to the whole column (which the
poked segment biases), so the unpoked segments then show a tilt instead
of a flat offset, identically under either reference.
Gate: `tOpdRef/test_driver_single_segment_poke_is_local_under_chief`.
**Also fixed the same day:** `run_sensitivities` did not forward `elts`
to the dwdsurf channel (the other three channels had it), so a runner
call asking for one segment's Kr/Kc harvested every powered element.
**And the one Luis was actually looking at (2026-09-10):** the per-element
CENTRE-FIELD page (`plot_dw_per_element(..., 'center', ...)`, the
`<name>_pages/*_center.png` files) rebuilt its pixel index with `m2v` on
`per_field_w_nom_2d`, which `orient xy` has transposed, while the
per-field Jacobian rows stay in the raw m2v order -- so under `xy` a
single-segment poke smeared into diagonal streaks (the raw page was
clean).  `sensitivities/per_field_indx.m` builds the index on the
raw-orientation map and remaps it with the same rule as
`apply_opd_convention`; the page is now the raw page transposed, exactly
(gate `tRunSensitivities/test_per_element_page_index_follows_orient_xy`).
Rule: never rebuild an index from a map that an orientation option may
have transposed -- use `indxall` (multi) or `per_field_indx` (per field).

## The figures: size first, count second (2026-09-10)

Dave: *"change the way dwd\* data is plotted to make the plots large
enough to be interpretable, even with large numbers of segments.  This
will mean many more plots and pages."*

The panel size is fixed FIRST and the panels per page follow from it --
never the other way round.  `sensitivities/dw_page_layout.m` owns the
rule and every dW plotter goes through it:

| floor | default | what it measures |
|---|---|---|
| `panel_in` | 3.5 in | the smaller side of ONE OPD map, at print resolution |
| `tile_in` | 1.2 in | ONE FIELD TILE inside a multi-field canvas panel |

A canvas panel is its tile count times `tile_in`, grown if that would
leave it under `panel_in` -- so a single-field map is a 1x1 canvas and
the two rules are one rule.  The floor is a MINIMUM: a page with room to
spare grows its panels to fill the 16:9 envelope (`page_in`), so a
two-panel page is not drawn small.  A page may grow to `page_max_in`
(32 x 20 in) to keep ONE element's channels together -- an element's Kr
and Kc, or its six DOFs, are read against each other -- and only a block
too big for even that is split onto `_p02`.

Why it matters, measured on `templates/50_sensitivities/zoom_5x5`
(jwst_ote_designc, 5 configurations x 5 fields = a **9 x 9 tile**
canvas, 63-ray maps):

* the dwdsurf all-channels sheet was 42 panels on one 700 x 1678 px
  page -- each field tile about **3 px**.  It is now 21 pages, one per
  optic, Kr and Kc side by side, each tile about **165 px**.
* the per-element `multi` page drew its canvas at 4.3 in inside a
  2042 x 1386 px page (the rest was SUBPLOT margin); it is now the page.
* `dwdx` is 138 channels: hundreds of pages, which is the point.

Three consequences worth knowing:

1. **Pages are sized in INCHES with a manual `PaperPosition`**
   (`dw_page_fig`).  With MATLAB's default `PaperPositionMode 'auto'` the
   printed size is the figure's PIXEL size over `ScreenPixelsPerInch`, so
   the same script printed a different page on a different screen, and a
   tall figure was clamped to the screen height first.
2. **Axes are placed explicitly** (`dw_page_axes` + `dw_draw_map`), not
   by `subplot`, whose default margins give away about 30% of every cell;
   the colorbar goes in a reserved gutter because `colorbar` otherwise
   SHRINKS its own axes.
3. **`plot_dw_channels` returns a manifest, not a figure handle.**
   `<name>_<ch>_channels.png` is always written -- the single page when
   one suffices, else the INDEX contact sheet (the dense sheet it used to
   be, each thumbnail labelled with its page), with the full-size pages
   in `<name>_pages/`.  The harvest-wide `<name>_pages_index.txt` lists
   every file.

`per_element` gained the **`field`** mode: one page per element per
(configuration, field), single-field maps at full size -- the mode for a
many-segment deck.  It is opt-in and warns with the page count.

Gates: `tRunSensitivities/test_page_layout_sizes_before_it_counts`,
`..._pagination_keeps_element_blocks_whole`,
`..._page_grows_to_the_envelope_when_panels_are_few`,
`..._small_deck_keeps_one_page_and_its_filename`,
`..._many_channels_paginate_with_an_index`,
`..._field_mode_pages_every_configuration_and_field`,
`..._per_field_indx_is_configuration_aware`,
`..._runner_writes_the_page_folder_and_index`.

## Where eligibility is decided (the class of bug you are chasing)

* An element missing from a Jacobian, with `'elts'` passed explicitly →
  since 2026-09-05 this ERRORS with the reason.  If you see a silent
  drop you are on a pre-`cdf7bc7` tree.
* An element missing from AUTO-discovery → the family's `find_*` /
  parser above.  `surf` and `grid` are engine-truth; `zernike` and
  `rigid_body` still parse the Rx text (S3 unifies them; a deck whose
  declared nElt disagrees with its Element-block count WILL mis-index a
  text parse).
* Powered-capable Element kinds: Reflector, Refractor, NSReflector,
  NSRefractor, Segment — plus finite Kr (`|Kr| < 1e21`).  Gratings/HOE
  excluded by ruling (2026-09-05).  Sentinel-variant flats (Kr=-1e18
  class) currently count as powered — recorded question.

## Debugging order that works

1. `macos.get_elt_info(k)` — the ENGINE's element type, not the .in
   text.  `macos.get_elt_kr(k)` for the powered filter.
2. The family's `find_*` on the loaded deck — is the element in the
   auto set?
3. Pass `'elts'` explicitly — the error message now names the reason.
4. Only then descend into the driver/supervisor.

## Verification discipline (Luis ask 3)

* Every fix's commit names its regression gate and states that the gate
  FAILS at the parent SHA (grep the log for "parent" / "pre-fix").
* Refactors state their recorded-baseline A/B max|diff|.
* Migrate from a SHA, not a branch tip, and note the SHA in reports;
  run the mmacos suite on the clean checkout before migrating.

## Gates that cover this stack

`tDwDsurf`, `tDwDx`, `tDwDzZernike`, `tDwDgridElts`, `tDwDxGroups`,
`tJacobianCheck`, `tRunSensitivities`, `tEltTypeCoverage` (the
element-type matrix), `tNsFlowOfLight` (engine NS root selection).
