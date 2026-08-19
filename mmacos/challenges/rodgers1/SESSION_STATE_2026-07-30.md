# Session state — strict metric: CLOSED 2026-07-30

This file was CCMac's mid-investigation handoff. **The investigation is now
closed; see `PACKET.md` ADDENDUM 3 for the result.** The handoff's two flagged
numbers and both of its open questions are resolved, and none of them meant what
they appeared to mean. This page is kept so the record of what was suspected —
and why it was wrong — is not lost.

## Result

Dave's strict metric — per-field reference sphere anchored at the exit pupil,
centred on that field's chief-ray incidence point on the FROZEN detector,
piston-only removal — is implemented in `strict_wfe.m` (pure MATLAB, exact, no
engine change) and validated by three gates in `strict_rung_gates.m`:

| gate | result |
|---|---|
| 1. displaced-detector discriminator | PASS — growth matches the closed-form sphere difference to 0.02% at +10 mm; per-ray residual 1.78 nm on a 661 nm growth |
| 2. on-axis anchor (stage 1) | 0.277 / 3.570 / 0.968 nm vs Rodgers 0.446 / 1.463 / 0.606 — same ~1 nm scale |
| 3. stage 2, EPD 4060, frozen detector, 9x9 box | **429.6 / 246.8 nm vs his 374.6 / 199.9 — 1.15x max, 1.23x avg. PASS.** |

Open ask 1 (the RMS reference-sphere convention) **closes**.  Open ask 2 (his FPA
surface) is **not load-bearing** — sliding the sphere to per-field best focus buys
0.1%.  Open ask 3 (EPD) is **supported** at D = 4060 mm.

Scored across all four committed stage solves (PACKET §D.2): S1 1.60x, **S2
1.15x**, S3 1.98x, **S4 2.98x** — the un-optimised stages agree and the
optimised ones do not, so the residual is a **merit-function** question, not a
metric one.  Rodgers' stage-4 rigid body scores ~3x better than ours under the
strict metric, so the `K_M3`/`Ydec` degenerate-valley explanation is NOT
confirmed and **step 3 (re-optimise against the strict metric) is indicated** —
on its own brief.

## What the handoff's flagged numbers actually were

- **"13.40 nm at the builder-seed FP (z = +0.627 m)"** was **13.4 mm**.
  `macos.trace(k).rmsWFE` is returned in BaseUnits (metres for these decks), not
  waves; `epd4060_pupil_check.m:73` multiplied it by `lam_nm`, so every number
  from that path is low by 1e6.  13.4 mm is exactly what a detector 3.9 m from
  focus and tilted 14.3 deg must read — `(beam width) x tan(tilt)`.  It was
  **never evidence of silent re-referencing**; the design layer re-emits the same
  detector at every field (PACKET Addendum 3 §A.1, §A.2).
- **"0.002 nm, each field to its own chief focus"** was **~2 um** — same units
  error — and it is dominated by the detector-tilt artifact, not by wavefront
  error.  The Fermat / quasi-tautology suspicion was **not** the mechanism:
  Fermat protects OPL to a stigmatic *point*, not to a *plane*; displace the
  plane and the piston-removed OPL grows as `dz*(sec(theta) - <sec(theta)>)`,
  coefficient measured at `6.43e-05` here.
- **"the strict rung reads ~0 nm because `align_focal_plane` refits the
  detector"** — the recipe was right.  Align once, then hold: that IS Rodgers'
  FPA DOF, and `align_focal_plane` writes its fit into `spec`
  (`Telescope.m:454-455`), so `trace_at_field` re-emits the SAME detector
  afterwards.  The ~0 was, again, the units error.

## What was SOLID and travelled

- The **lane verdict** (§E: configuration-only — CALIB already implements the
  strict metric given an ExitPupil element + system STOP + FEX), re-checked
  against the forensics and unchanged.
- The **emit-path fix** (`add_pupil` Return surfaces -> `ApType=None`), already
  on `dev` at `c6e071a`.
- The **XP-row attribution** (a fixed on-axis CoC retains image-displacement
  tilt) — right in kind, wrong by 1e6 in magnitude.

## One trap worth remembering

`trace_at_field(F)` **adds** `spec.field_bias` to `F`, so it takes a
box-RELATIVE box.  `realize_apertures('fields',F)` is the one branch that does
NOT add the bias and so takes the ABSOLUTE box.  Handing the absolute box to
`strict_wfe` doubles the bias, silently evaluates a `+1.0 deg` box, and produces
a plausible-looking 12x "miss".
