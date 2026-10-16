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
| `e5hex1_budget.xlsx` | **WFE** sheet (σ inputs + per-group RMS WFE) and **Coefficients** sheet (the sigma-free α per group) |
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

- **WFS Sensing Error** (Eq. 32), 3 nm rms per wavefront sample (iid over
  the Jacobian's rays): only its controllable part leaks,
  `alpha^2 = rank(dw/du)/n` = 48/51225 here.
- **one actuator row per control group** (Eq. 31, unprojected), 1 nm per
  DOF; a rotation takes 1 nm of EDGE motion, `theta = 1 nm / r_edge`
  (segment vertex radius 1510.73 mm → 0.662 nrad; SM radius 615 mm →
  1.626 nrad) — the conversion is in `budget_spec_e5hex1.m`.

Measured (`tRunBudget/test_wfc_stage_cov_equals_mc`): pre-WFC 317.4 nm,
post-WFC **3.156 nm (cov) vs 3.179 nm (MC, 1000 draws, controller + sensing
+ actuator draws in the loop)**; the actuator rows dominate (PM piston
2.60 nm, PM tilt 1.28 nm, SM tilt 1.25 nm), the sensing row is 0.092 nm.

See `macos/CLAUDE.md`'s "Error budgets" section and `macos.budget` for the
kernel. The metrology stage (iSeq = 2) and the SPOT metric are later phases.
