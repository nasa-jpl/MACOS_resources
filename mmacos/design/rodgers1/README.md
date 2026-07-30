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

## Run

```matlab
% from anywhere (the driver sets its own paths):
run('~/dev/MACOS_resources/mmacos/design/rodgers1/rodgers1.m')   % all 4 stages
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
- **Absolute field-map WFE runs 4–6× off**, traced to a **field-map RMS metric
  definition** (CODE V per-field best-focus reference sphere vs our global-plane
  `std(OPD)`; the per-field-defocus-removed metric matches on-axis to 3%). This
  and the Stage-4 rigid-body values are **flagged for Fable**, not tuned past.

See **`PACKET.md`** for the full comparison tables, the metric ladder, and the
open questions.

## Files

- `rodgers1.m` — the driver.
- `rodgers_common.m` — verbatim prescription + Rodgers' ground-truth stats (nm).
- `rodgers1_stage{1..4}.in` / `_*.png` / `rodgers1_results.mat` — artifacts.
- `diag_*.m` — the diagnostics (focus, grid convergence, metric ladder, aperture
  sweep, FPA strategy) supporting the packet's findings.
