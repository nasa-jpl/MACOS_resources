# offset_imager — wide-field imager with an offset field

A parameterized, illustrated five-stage design flow for the recurring
problem: a wide field box that must sit **far off the optical axis**
(packaging, stray light, or scan geometry demand it), and what each
class of design freedom buys back.  The flow is the product; the
`challenges/rodgers3` instance (Mike Rodgers' 22°-offset imager ladder)
is the validation.

## Run it

```matlab
run('<path-to>/mmacos/mmacos_setup.m');
OUT = offset_imager();                                   % the rodgers3 instance
OUT = offset_imager(struct('EPD_m',0.200,'Fno',2.5, ...  % your instrument
        'box_deg',[10 10],'offset_deg',12));
```

All parameters live in `offset_imager_params.m` (single source of
truth, the e2e2 pattern).  Artifacts: per-stage decks `oi_s*.in`,
figures `oi_s*_{layout,map}.png`, the assembled `oi_REPORT.md`, and
`oi_run.mat`.

## The stages (each = a solve + a layout figure + a dense WFE map + a report section)

| stage | freedom opened | the lesson |
|---|---|---|
| S1 | symmetric conics + aspheres, solved at the ON-AXIS box | the classical coaxial wide-field imager |
| S2 | FPA tilt/focus refit ONLY, field box moved to the offset | the **disaster map** — what the offset costs when nothing else follows |
| S3 | re-solve the symmetric surfaces AT the offset field | the bias doctrine: solve at the used field (expect oblate-class conic flips) |
| S4 | + mirror tilts/decenters (+ radii stay open) | constraint set becomes live: exit-beam direction, clearances |
| S5 | + Zernike surface departures (aspheres replaced) | what true freeform buys at fixed packaging |

## Architecture

- `offset_imager_params.m` — every knob (EPD, F#, box, offset, λ,
  packaging spacings, constraint set, Zernike term set, densities).
- `oi_paraxial.m` — signed-convention paraxial chain: EFL/BFD/Petzval +
  the first-order seed solve (EFL exact, Petzval = 0).
- `oi_seed.m` — parameter set → starting design struct (spheres).
- `oi_close.m` — the first-order closure run at EVERY solve iterate
  (afocal4 doctrine: identities re-derived, never penalized):
  EFL = EPD·F# exactly (R3 eliminated), stop posed by the
  entrance-pupil construction, FP posed on the traced exit chief.
- `oi_deck.m` — design struct → MACOS prescription.  The stop is a
  Reference element carrying the **native element-bound stop**
  (`macos.stop` → engine `ChiefRayAiming` real-ray aiming, A/B'd against
  the Stage-0 Newton aiming in `challenges/rodgers3/probe_native_stop.m`
  to ≤0.04 nm).  The header `ApStop=` (StopPos) form is deliberately NOT
  used — it aims with no optics traversal, wrong for a stop behind M1.
- `oi_score.m` — the metric: strict RMS WFE (design/src kernel),
  centroid reference on the stage's frozen FPA, exit-pupil anchor,
  piston-only removal.  Stated next to every quoted number.
- `oi_solve.m` — damped Gauss–Newton over per-field WFE residuals with
  natural per-variable scales; walls (not penalties) for constraints;
  solve set ≠ scoring set.
- `oi_gates.m` — exit-beam direction + beam/mirror clearance gates.
- `oi_map_fig.m` / `oi_layout_fig.m` — the per-stage illustrations.
  Each stage emits THREE figures: `*_layout.png` (the `macos.view_std`
  four-panel solid-body hardware render), `*_fields.png` (a Y-Z
  elevation with per-field beam ENVELOPES — filled patches, not ray
  spaghetti — plus stations and the exit-chief annotation), and
  `*_map.png` (the dense strict-WFE-vs-field map).

## Conventions inherited from the rodgers3 challenge

Global frame, metres, beam enters +z; `KrElt` = signed CODE V radius;
fields tangent-composed; Zernikes `ZernType= BornWolf` with lMon frozen
at the traced footprint (power pinned to radii, tilt to pointing —
the Zernike solve doctrine).  The paraxial sign convention in
`oi_paraxial.m` was validated against the rodgers3 r1 deck by real rays
(engine plate scale vs paraxial EFL).

## Suite coverage

`tests/tOffsetImager.m` (freeform group, size 256) runs a reduced-knob
smoke of S1–S3 at a second parameter set — proving the template is
parameterized, not a rodgers3 replay.  The full five-stage runs live in
`challenges/rodgers3/PACKET.md` (T3) and this directory's committed
report (T4).
