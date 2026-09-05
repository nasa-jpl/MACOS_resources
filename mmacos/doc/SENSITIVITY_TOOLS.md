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
