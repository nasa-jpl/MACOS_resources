# Session state — metric pinning / strict rung / lane (2026-07-30) — PAUSED for Dave

Paused mid-investigation at Dave's request. This records exactly where things
stand. **Nothing committed or pushed this session.** Working in a throwaway
`origin/dev` worktree at `/tmp/rodgers_dev` (tip c6e071a). MEX = pol MEX copy.

## The crux result (the reason to pause)

Stage 2 (Rodgers' frozen on-axis design, evaluated at the +0.5° offset box,
EPD 4060), engine detector RMS WFE across the 9×9 box:

| detector placement | box max / avg | Rodgers box |
|---|---|---|
| **NO `align_focal_plane`** (detector at builder FP, z=+0.627 m) | **13.40 / 10.92 nm** | 374.6 / 199.9 |
| **WITH `align_focal_plane`** (per-stage best-fit tilted plane) | 0.01 / 0.01 nm | — |

### EXACTLY how each number was computed (load-bearing discriminator evidence for TO)

Both rows: **whose OPD** = the engine's own `macos.trace(nElt).rmsWFE` (equivalently
`std` of `macos.opd()`), i.e. `SUBROUTINE OPD`'s piston-removed OPL RMS — NOT a
design-layer refsphere/global metric. **Which detector** = element `nElt`, the
terminal `FocalPlane`. **Per field:** `t.trace_at_field(offset)` (re-emits the Rx
with the chief ray pointed at bias+offset, re-traces), looped over the 9×9 box
`macos.design.field_grid(6', 9)`. Fresh Telescope object, **no `realize_apertures`
called before** (avoids the stale-clip-aperture all-rays-lost trap). Build recipe =
committed rodgers stage-2: build → (optionally) `align_focal_plane('grid',5,
'span_arcmin',6)` → measure. NO optimizer (stage 2 is frozen conics).

- **WITH-align row (0.01 nm):** `align_focal_plane` was allowed to REFIT the
  detector position/tilt to the box's best-fit surface before measuring. The
  engine was thus allowed to move the detector per the box → it removes the
  field-curvature/defocus/tilt walk → ~0. This is a detector that was refit to
  null the box.
- **NO-align row (13.40 nm):** detector left where `build()` put it — the builder's
  **Seidel-seed** FP at **z = +0.627 m** (the known ~4× mis-placed fast-relay seed;
  NOT Rodgers' paraxial −5095 mm-equiv, and NOT the −3.256 m that `align` finds).
  So the engine was allowed NO refit here, but the fixed plane is the WRONG one.

**⚠ CAVEAT FOR TO — the 13.4 nm number is physically inconsistent with a
detector-tied metric and is therefore evidence of SILENT RE-REFERENCING, not a
clean measurement.** A frozen on-axis anastigmat at 0.5° off-axis on a FIXED
detector should show hundreds of nm (Rodgers: 375). 13.4 nm at a fixed (if wrong)
plane means something in the `trace_at_field` → `OPD` path is still re-referencing
each field (chief-tie / per-field OPL origin / piston+implicit-tilt) so the box
spread never accumulates against a common surface. **Do not treat 13.4 nm as "the
box WFE at a fixed detector."** It is a symptom that the engine's reported RMS is
not a common-surface field-map RMS. Pinning down that re-referencing (what OPL
origin/reference each field uses at the FP, and whether the chief tie is implicit)
is the core of TO's forensic pass, alongside the Fermat/second-order point above.

- `align_focal_plane` **flattens the box to ~0** — it moves the detector to the
  box's own best-fit tilted/defocused surface, removing exactly the
  field-curvature/astigmatic focus walk Rodgers measures against a FIXED
  detector. Every strict-rung construction I tried had `align_focal_plane` in
  the build recipe (inherited from the committed rodgers1 stages), which is why
  they all read ~0.
- **BUT** even the NO-align number (13.4 nm) is ~28× BELOW Rodgers' 374.6. So
  align is one factor, not the whole story. Two possibilities remain open for
  Dave's look:
  1. detector z: NO-align leaves the detector at the builder's **seed** FP
     (z=+0.627 m, the known ~4× mis-placed Seidel seed), NOT Rodgers' paraxial
     −5095 mm / the −3.256 m align finds. So "NO-align" is ALSO not Rodgers'
     fixed detector — it's the wrong fixed plane. The correct test is the box
     WFE at Rodgers' OWN paraxial detector, held fixed, FOV varied.
  2. `trace_at_field` re-emits the chief per field; need to confirm it does not
     also re-reference/re-focus in a way that shrinks the box spread.

## Dave's guidance received this session (verbatim intent)

1. EP OPD FEX-generated has FOCUS but little tilt; chief tie removes tilt except
   coma-induced tilt-like residual.
2. FEX does NOT optimize WFE — it FINDS the pupil per field, tying it to the ray
   pierce point on the FIXED FP surface.
3. FEX mechanism: (1) paraxial-ray-trace finds pupil; (2) sphere placed at pupil;
   (3) sphere radius = chief-ray distance to the FIXED detector, so sphere is
   centered at the chief pierce point on the detector. Changing FOV, FP stays.
4. Validation loop (once solution is in): change field → fex → opd, per NxN
   field, NO re-optimization.
5. "Take a breath, document, pause." ← we are here.

## What is SOLID vs OPEN (engine research, cited, `/Users/dcr/dev/macos/macos_f90`, pol-core)

> SOLID this arc (travel with the branch): the **lane verdict** (Q3, config-only,
> with citations), the **emit-path diagnosis** (Return→ApType=None, c6e071a), and
> the **XP-row tilt attribution** (Q2). The **0.002 row meaning is OPEN/disputed**
> (Fermat second-order — see below). The **13.4 nm number is a re-referencing
> symptom, not a measurement** (see the crux section). Strict-rung numbers all read
> ~0 because `align_focal_plane` was in the recipe — provisional, superseded by
> TO's fixed-detector pass.

- **Q1 / the 0.002 row — RESOLUTION OPEN, SEE DISPUTE (2026-07-30, Dave).**
  Citation kept as the right starting point for TO's forensic pass, but my
  "genuine, not a fit-tautology" conclusion is **DISPUTED and must not be relied
  on**: if the engine OPD is path length referenced to each ray's OWN intercept,
  **Fermat's principle protects OPL equality to second order, so a near-zero RMS
  is STRUCTURAL** — it need not indicate a well-corrected wavefront, and is not
  evidence either way. TO to resolve.
  Citation (starting point, not a verdict): `SUBROUTINE OPD` tracesub.F:21-250.
  RMSWFE = piston-only (mean `DAvgl`, :179/:222-232) std of chief-referenced
  (`RefCumRayL=CumRayL(1)`, :138-144) OPL; sample std :235-239. What I did NOT
  establish: what surface/point each ray's OPL is referenced to at the FP, and
  whether Fermat makes the near-zero automatic. **Treat the 0.002 row's meaning
  as an open forensic question.**
- **Q2 / XP hidden removal:** ExitPupil sphere CoC+axis set by FEX (tracesub.F:
  255-566; axis=−chief dir :563; vertex=chief crossing), applied at
  macos_cmd_loop.inc:2662-2671. With a FIXED on-axis CoC the off-axis raw OPD
  carries image-displacement TILT (the 2.64 nm XP row is tilt-dominated; removing
  a linear pupil term → 0.001 nm, confirmed).
- **Q4 / Return OPL:** `.opl` at a single Return is ONE-WAY source→sphere-surface,
  NOT negated (ifReturn negation only fires on the 2nd of a Return pair;
  tracesub.F:3793-3797). RayPos at EP = landing point ON the sphere
  (elemsub.F:2387-2390). Points lie on the C0 sphere to 2.3e-14 m.
- **Q5 / recenter recipe:** macos.opd() at EP is ALREADY referenced to the engine
  CoC C0. To move to Cf: `OPD_Cf = OPD_engine + (|pos−Cf| − |pos−C0|)`, mean
  removed. (My earlier bug: `opl + |pos−Cf|` alone re-injects the sphere sag →
  spurious 151 nm. But since |pos−C0|=const=R0, the differential ≈ const here, so
  the real off-axis content is what macos.opd() shows: tilt-dominated.)
- **Q3 / LANE VERDICT — CONFIGURATION-ONLY (no new engine mode).** CALIB
  (`funcs_app`, design_optim.F:643-881) forces `LUseChfRayIfOK=.false.`
  (design_optim.F:659-664) → piston-only mean-subtract branch of OPD; and re-runs
  FEX per field (smacos_compute.inc:391-397) → per-field chief-ray-tied sphere.
  So Dave's strict metric (per-field chief-tied sphere, piston-only) is EXACTLY
  what CALIB already does, given an ExitPupil element + system STOP + FEX on. Step
  3 = deck/config, MY lane, no TO/Fortran. Caveat: CALIB removes piston only (not
  extra tilt/focus); "piston-only" matches — config-only holds.

## The contamination trap (recurring, cost several runs)

`realize_apertures` leaves clip apertures INSTALLED on exit → any later raw
`trace+opd` on the SAME object loses all rays (9.999e39 sentinel, n=0). ALL raw
per-field measurements MUST be on a FRESH object with no prior realize call.
`spec` is read-only outside the class (use `trace_at_field`, `macos.set_xp`, not
`t.spec.elt(..)=`).

## Open question for Dave (the pause point)

The strict metric as I built it reads ~0 because `align_focal_plane` sits in the
stage recipe and flattens the box. Removing align gives 13.4 nm (still 28× under
Rodgers) but at the WRONG fixed plane (builder seed z=+0.627 m). **The right next
step is Dave's recipe exactly: hold the detector at Rodgers' paraxial focus
(−5095 mm equiv / the −3.256 m align target as a FIXED plane, NOT re-aligned per
build), FOV-vary, FEX per field, OPD — and see if the box regrows to ~375.** I
paused rather than guess which fixed plane is "his."

## Files in the worktree (scratch, uncommitted)

- `strict_rung.m` — FEX-per-field loop (Dave's sanctioned metric)
- `strict_rung_stages.m` — stages 2-4 driver (⚠ uses align → reads ~0)
- `ep_decompose.m` — raw/tilt/focus ladder at fixed EP sphere (tilt-dominated finding)
- `strict_geom.m`, `strict_diag.m`, `sgdiag*.m`, `diff_test.m`, `opd_test.m`,
  `recon.m`, `s2check.m`, `common.m`, `fixedfp.m` — diagnostics (DELETE before any commit)
- `rodgers1_epd4060_strict_rung.mat` — last ep_decompose save
- PACKET.md — Addendum 3 skeleton written (per-row defs + rider DONE; strict rung,
  stage-2, lane placeholders `<!-- ... -->` NOT yet filled)

## NOT done (resume after Dave's look)

- Finalize the strict-rung number against the CORRECT fixed detector
- Fill PACKET Addendum 3 §C/§D/§E (strict rung, stage-2 validation, lane verdict)
- Delete diagnostics; keep strict_rung.m + one clean driver
- tDesignTelescope green; commit + push (dev + pol-ifo)
