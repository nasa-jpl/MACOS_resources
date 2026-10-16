**Open: `MACOS_resources/mmacos/templates/50_sensitivities/e5hex1_budget/e5hex1_wfe_budget.xlsx`** — the live-formula WFE calculator (Excel or LibreOffice; regenerate with `run_budget_e5hex1`).

# e5hex1 error budget — worked example

An optical **error budget** built from a MACOS `dw/dx` sensitivity harvest,
following Redding 2014 (SPIE), *"Optical error budgeting using linearized
ray-trace models"*: a properly-formed RSS budget **is** a covariance
analysis, and Monte-Carlo gives the identical answer.

Run it:

```matlab
run_budget_e5hex1          % harvest dw/dx on e5hex1.in, then run_budget
```

Pipeline: `macos.dw_dx_multi` (5 fields, 6 DOF/optic, model 128) →
`run_budget` → covariance + Monte-Carlo WFE budget. Artifacts land here:

| file | contents |
|---|---|
| `e5hex1_wfe_budget.xlsx` | the **live-formula calculator** (below) |
| `e5hex1_budget.xlsx` | plain-value export: **WFE** (the roll-up table) and **Coefficients** (one row per line) |
| `e5hex1_budget.csv`  | the roll-up table |
| `e5hex1_budget.mat`  | the `coef.*` struct (drops into a HabEx/ATLAST-style sheet) |
| `e5hex1_budget_report.txt` | text report |
| `e5hex1_budget.png`  | per-group WFE bar chart |

## The group table is the interface

`budget_spec_e5hex1.m` returns the **group table** — one row per budget
line, grouped by rigid-body DOF type. The same schema can be authored as a
CSV or an Excel sheet (see `macos.budget.read_spec`); `macos.budget.suggest_groups`
auto-proposes a skeleton from any harvest. Columns:

```
Name  Elements  DOFs  Modes  Param  Channel  Dist  Sigma  Units  Min  Max  Cov  Control
```

- `Dist` = `normal` (uses `Sigma`) | `uniform` / `worstcase` (uses `Min`/`Max`).
- `Units` = natural engineering units (`nrad`, `nm`, `nm/Zi`, `mm`, …);
  reconciled against the deck's base units internally.
- A group must be **unit-homogeneous** (rotations and translations cannot
  share one σ).

## The WFC stage (controls phase, iSeq = 1)

The example runs **post-WFC**: the `Control` column tags the actuated
groups — **6 DOF per PM segment (7 segments) + the SM's 6 DOF**, 48
actuators (Dave's ruling, 2026-10-15). **This deck has no DM and no FCA**,
so figure-type errors would pass the controller untouched; here every
error is rigid-body. The pseudo-inverse controller `G = pinv(dw/du)`
(Eq. 20) leaves, per error group, the fitting residual
`alpha_post = sqrt(trace(Pu D S D' Pu')/n)` (Eq. 30), and adds two kinds of
row:

- **WFS Sensing Error** (Eq. 32), 3 nm rms of WHITE per-sample noise (iid
  over the Jacobian's rays; Dave's ruling 2026-10-16, the default): only
  its controllable part leaks, `alpha^2 = rank(dw/du)/n` = 48/51225 here.
  The HabEx sheet's reading (alpha = 1, the 3 nm passes whole) is one edit
  in the workbook — type 1 in the WFS line's `Coefficients!M` cell; the
  workbook's **Log** sheet names the cell.
- **one actuator row per control group** (Eq. 31, unprojected), 1 nm per
  DOF; a rotation takes 1 nm of EDGE motion, `theta = 1 nm / r_edge`
  (segment vertex radius 1510.73 mm → 0.662 nrad; SM radius 615 mm →
  1.626 nrad) — the conversion is in `budget_spec_e5hex1.m`.

Measured (`tRunBudget/test_wfc_stage_cov_equals_mc`): pre-WFC 317.4 nm,
post-WFC **3.156 nm (cov) vs 3.179 nm (MC, 1000 draws, controller + sensing
+ actuator draws in the loop)**; the actuator rows dominate (PM piston
2.60 nm, PM tilt 1.28 nm, SM tilt 1.25 nm), the sensing row is 0.092 nm.

## The workbook (live formulas)

`run_budget` also writes **`e5hex1_wfe_budget.xlsx`**, the HabEx-form
calculator the system engineer edits (`macos.budget.write_workbook`):

- **White cells are the inputs you edit; gray cells are formulas** (the
  legend sits in WFE!M1).  Typing over a gray cell replaces its formula.
- **WFE** — one row per budget line grouped by the table's `Section`
  column (Primary Mirror, Secondary Mirror, Lens, Detector, then WFC for
  the sensing / actuator rows). σ inputs sit in white cells (C initial
  alignment, G drift); E/F = C × Coefficients!D/E (pre/post WFC), I/J =
  G × Coefficients!G/H (pre/post metrology), K = SQRT(SUMSQ(F, J)).  Then
  the WFE Summary: one row per section, Everything Else, TOTAL, Design WFE
  (input), Payload, Requirement (input), Margin.
- **Coefficients** — the same rows; D..H point at the raw `<key>0` /
  `<key>f` blocks the tool writes (nm per unit input), plus the
  Monte-Carlo check.
- **WFE BD** (roll-up tree), **PSD** / **StrTherm** (stubs for the
  engineer's non-ray-trace terms), **Log** (what it was generated from).

Change a σ and the budget re-rolls — no MATLAB. Measured: LibreOffice's
recalculated Payload is MATLAB's `coef.total` to 5.6e-16 (3.15580786565109 nm).
This deck has no metrology truss, so its drift column is empty (σ = 0);
`run_budget(..., 'met', ...)` fills it (see `macos.budget.project_met`).

See `macos/CLAUDE.md`'s "Error budgets" section and `macos.budget` for the
kernel. The SPOT metric is a later phase.
