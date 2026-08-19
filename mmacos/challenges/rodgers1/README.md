# rodgers1 — offset-field coaxial-TMA study

Reproduces J.M. Rodgers' four-stage coaxial-TMA offset-field study
(`macos_sandbox/Design/Rodgers/260728-TMA_Offsetfield-jmr.pptx`, ORA/CODE V,
λ = 1000 nm) in the MACOS design layer, and compares — stage by stage — the
achieved RMS wavefront-error field map and the solved optical parameters
against Rodgers' CODE V results.

This is a **design example** in the `mmacos/design/` sense: a user-facing,
parameter-driven, documented driver you open, set a few knobs at the top (or
pass as name/value), and run. Adapt it for related offset-field / DOF-ladder
studies by changing the prescription in `rodgers_common.m` and the knobs below.

## THE RESIDUAL IS CLOSED — read this first (2026-08-01)

**MACOS was not tracing the pupil the prescription declares.**  `sourcsub.F:220`
(`ColSource`) sets the circular-grid acceptance radius to
`Aperture/2 + Aperture/npts`, not `Aperture/2` — an oversize of
**1 + 2/(nGridpts−1)**, i.e. **+5 %** on this deck's `nGridpts=41`, putting
**118 of 1252 rays (9.4 %) outside the declared 5000 mm pupil**.  It hid because
the traced pupil is a disc clipped by the lattice square: measured along either
axis it reads **exactly 5000.0000 mm**, and only the **diagonal** is oversize
(5233.06 mm).  `PtSource` is not affected, so a collimated and a point source
trace different pupils for the same `Aperture=`.

Mask the rays to the declared pupil and grant the reference convention CODE V's
field-map RMS uses (per-field best focus + least-squares tip/tilt), and Rodgers'
three designs reproduce on the max to **1.000× / 1.008× / 1.001×** — three
designs whose reported WFE spans a factor of 9.4.

* Measurement + masking harness: **`pupil_audit.m`** (five-way pupil measurement,
  then the ladder on both pupils).
* Write-up, element-by-element `.seq` ↔ `.in` conversion audit, and the fix
  scoping: **`PACKET.md` Addendum 10**.
* **The engine is NOT fixed** — the masking is post-processing in MATLAB.  The
  fix moves every collimated deck in both repos and wants its own slice.

## The configuration (2026-07-31)

Mike supplied the four CODE V `.seq` files on 2026-07-31.  They pin the inputs
the slides never stated: **EPD = 5000 mm** (not the 2000/4060 we inferred), a
**15-point half-box** field set, and a **central hole in M1**.  EPD 5000 is not
optional — M3's radius is held by a `CUY UMY -0.025` solve that only closes
there.

* Truth, transcribed verbatim: **`rodgers_seq.m`**; reachable as
  `rodgers_common('seq')`.
* Re-run at truth: **`run_seq`** (sections 0–5).
* Write-up, reconciliation table and verdict: **`PACKET.md` Addendum 8**.

Headline: the residual band at the true configuration is **2.04× / 2.18× /
2.90×** of his reported numbers under the metric as ruled — the earlier
1.15×/1.26×/1.63× is **retracted** as an artifact of EPD 4060.  Granting the one
remaining convention (per-field tip/tilt removal) collapses it to **1.09×**, and
the last 9 % is the pupil defect above (Addendum 10).  The "14.3° tilted image
surface" of §4b was a frame artifact; our detector and his agree to 0.022°.

Also retracted by Addendum 10: §8.4's "real `|u′|` 0.026165 vs paraxial 0.025 —
a few-percent real-vs-paraxial difference is expected."  It was the pupil
(2618.6/2500 = 1.047 against 0.026165/0.025 = 1.047).

`rodgers_common()` with no argument is unchanged, so everything below still
reproduces the committed EPD-2000 / EPD-4060 artifacts bit-for-bit.

## Run

```matlab
% from anywhere (the driver sets its own paths):
run('~/dev/MACOS_resources/mmacos/challenges/rodgers1/rodgers1.m')   % all 4 stages
```
```matlab
rodgers1('stages',[1 2])            % pure-evaluation stages only
rodgers1('EPD_mm',2000,'map_n',9)   % change aperture / field-map density
rodgers1('save',false,'plots',false)% numbers only, no files
out = rodgers1();                    % struct: per-stage stats, conics, rigid, ladders
```
Batch: `matlab -batch "run('.../rodgers1.m'); exit(0)"` (needs `MACOS_HOME` set;
on Apple Silicon `unset FC` first — see the mmacos build notes).

## The four stages

| | what | DOF set | our merit |
|---|---|---|---|
| 1 | on-axis 0.2°×0.2°, verbatim conics | — (evaluate) | field-map RMS WFE |
| 2 | +0.5° offset, verbatim conics, FPA re-fit | — (evaluate) | " |
| 3 | offset, re-optimize conics + FPA | conic ×3 | native CALIB |
| 4 | offset, + M2/M3 Ydec+α-tilt, re-opt | per-elt: M1 conic; M2/M3 conic+DY+TIP | native CALIB |

Stages 1–2 are a **pure evaluation** of the verbatim prescription; stages 3–4
run our native optimizer with Rodgers' exact DOF sets and compare the solved
parameter values.

## Headline result

- **Layout** reproduces exactly: `align_focal_plane` recovers Rodgers' paraxial
  focus (−3256 mm) to 0.05 mm; on-axis WFE ≈ 0.
- **Stage-3 conics match Rodgers to 4 decimal places** — our optimizer
  independently lands on his CODE V solution.
- **Absolute field-map WFE reproduces once the RMS reference convention matches**
  (Addendum 3). Under Dave's ruled **strict metric** — a per-field reference
  sphere anchored at the exit pupil, centred on that field's chief-ray incidence
  point on the *frozen* detector, piston-only removal — stage 2 at his aperture
  reads **429.6 / 246.8 nm against his 374.6 / 199.9 (1.15× / 1.23×)**. The
  earlier "4–6× off" (and the 11–20× of Addendum 1) came from taking the RMS on
  the **detector plane**, which on this 14.3°-tilted image surface carries
  (transverse ray aberration) × tan(tilt) — an artifact ~22× the wavefront error
  itself, and not a low-order pupil term, so no `refsphere` fit removes it.
- **Scored across all four stages** (Addendum 3 §D.2), the pattern separates
  cleanly: the stages our optimizer never touched agree (S1 1.60×, **S2
  1.15×**) and the ones it solved do not (S3 1.98×, **S4 2.98×**). The
  evaluation metric and the optics are therefore both fine; what differs is
  **what our optimizer minimised** — CALIB was driven by the detector-plane
  WFE (Addendum 4 §A).
- **His own designs, built verbatim** (Addendum 4 §B): his stage 3 reads
  **115.3 / 53.7 nm vs his 91.6 / 46.4 — 1.26× / 1.16×**, so the strict metric
  now reproduces *two* of his designs. His stage-4 parameters read **64.6 µm**;
  an injection round-trip of our own stage-4 reproduces §D.2 to 5e-6, so the
  failure is that his stated `Ydec`/`α` are **not in our frame**. The stage-4
  rigid-body comparison is therefore **retracted as uninterpretable** and the
  `K_M3`/`Ydec` degenerate-valley question is reopened. His *conics* are better
  than ours by 1.57× under the strict metric — that finding stands.
- **His stage 4, once the convention is decoded** (Addendum 5): his `ADE` sign
  is opposite to ours (`YDE` matches) — measured over all 16 sign combinations
  with a 30× margin. His S4 then reads **64.9 / 35.4 nm vs his 39.8 / 22.5 —
  1.63× / 1.57×**, so the strict metric reproduces **all three** of his
  designs (1.15×, 1.26×, 1.63×). In one frame his rigid body is −2.1..−2.8×
  ours — the same compensation pattern on the **opposite branch** of a
  degenerate valley — and his branch is better by **1.83×**, not the ~3× first
  claimed.
- **Step 3 is BLOCKED on a one-line engine change** (Addendum 5 §D).
  `OptFEX= Yes` is a **no-op**: `msmacosio.inc:327-329` parses the keyword but
  carries only the `'N'` branch, so a deck can turn the per-field FEX off and
  never on. Without it CALIB minimises the tilt of a sphere stuck at the
  on-axis image (1.8e-3..2.6e-3 m off-axis vs 1.1e-7..4.3e-7 m with FEX) and
  the solve runs away. Gate 0 (the merit *is* the strict metric when FEX runs)
  passes at **2.7e-9**. The design-layer plumbing is in place but **guarded by
  a hard error** until the engine lands.

See **`PACKET.md`** for the full comparison tables, the engine forensics
(Addendum 3 §A), the metric ladder, and what is left open.

## The strict metric

```matlab
strict_rung_gates(9)     % gates 1/2/3 end-to-end -- reproduces Addendum 3 §C/§D
out = strict_wfe(t, F)   % the metric itself; F is a BOX-RELATIVE field list (rad)
strict_ladder(5)         % what the strict residual is made of + the best-focus floor
strict_stage_table(9)    % score all four COMMITTED stage decks + emit the 4 maps
his_designs(9)           % build + score RODGERS' OWN S3/S4 verbatim (Addendum 4/5)
convention_decode()      % decode his YDE/ADE frame (Addendum 5 §A)
gate0_merit_identity()   % in-loop merit == strict metric?  (Addendum 5 §C)
fex_in_loop_check()      % is CALIB actually running FEX?   (Addendum 5 §D)
```

`strict_wfe` is pure MATLAB and computes the ray-to-sphere OPL exactly
(`strict_sphere_opl`); no paraxial expansion and no engine reference-surface
default is involved. **Re-run gate 1 (the displaced-detector discriminator)
before trusting any change to it** — a metric that does not grow by the analytic
sphere-difference amount when the detector moves is self-referencing.

⚠ Three harness traps, all of the "plausible wrong answer" kind: (1) a driver
that saves a deck after a probe loop bakes the **last probe field** into it and
silently offsets every later scan (Addendum 4 §E — it turned a 1.26× result into
a 4.17× one); (2) `macos.perturb`'s local `+Rx` reads back through `rigid_of` as
**−α**; (3) the committed `*_epd4060_stage*.in` decks were saved after
`realize_apertures` and carry its clip apertures. Each is now guarded by an
assert in the code that found it.

⚠ `macos.trace(k).rmsWFE` is in **BaseUnits (metres)** for these decks, not
waves — unlike `realize_apertures(...).wfe`, which does divide by λ. Multiplying
`rmsWFE` by λ in nm understates it by 1e6; that error produced the "0.002 nm" and
"13.4 nm" rows this study had to un-pick (Addendum 3 §A.2).

## Files

- `rodgers1.m` — the driver.
- `rodgers_common.m` — verbatim prescription + Rodgers' ground-truth stats (nm).
- `rodgers1_stage{1..4}.in` / `_*.png` / `rodgers1_results.mat` — artifacts.
- `run_epd4060.m` + `rodgers1_epd4060_*` — the same sequence at Dave's measured
  4060 mm aperture (Addendum 1).
- `epd4060_pupil_check.m` + `rodgers1_epd4060_stage4_pupil.in` — the explicit
  exit-pupil cross-check (Addendum 2; read its units caveat above).
- `strict_wfe.m`, `strict_sphere_opl.m`, `strict_rung_gates.m`,
  `strict_ladder.m`, `strict_wfe_deck.m`, `strict_stage_table.m` +
  `rodgers1_epd4060_strict_*` — the strict metric and its gates (Addendum 3).
- `his_designs.m` + `rodgers1_epd4060_{rodgersS3,rodgersS4,oursS4roundtrip}.*` —
  Rodgers' own solves built verbatim and scored (Addendum 4).
- `diag_*.m` — the diagnostics (focus, grid convergence, metric ladder, aperture
  sweep, FPA strategy) supporting the packet's findings.
- `pupil_audit.m` + `rodgers1_pupil_audit.mat` — the five-way measurement of the
  traced pupil and the masking harness that closes the residual (Addendum 10).
