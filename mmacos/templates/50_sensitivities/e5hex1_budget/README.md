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

See `macos/CLAUDE.md`'s "Error budgets" section and `macos.budget` for the
kernel. Controls (`Pu`) and the SPOT metric are later phases.
