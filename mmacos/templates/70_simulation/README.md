# `70_simulation/` — time-sequence simulation (T6)

**Pointer directory.**  The time-sequence simulator is not yet a
standalone template; it is stages s6 and s7 of the end-to-end flow:

| Stage | Script | What it produces |
|---|---|---|
| s6 compare | [`../80_end_to_end/e2e/s6_compare.m`](../80_end_to_end/e2e/s6_compare.m) | linear model vs engine on stepped x / z / grid states |
| s7 simulate | [`../80_end_to_end/e2e/s7_simulate.m`](../80_end_to_end/e2e/s7_simulate.m) | uncontrolled + controlled time-history PSFs (Tikhonov-ridge WFC, RBCS MET loop) |

The library behind them is `design/runners/run_compare.m` and
`design/runners/run_simulator.m` — a standalone template gets extracted
here if the Week-2 rewalk pulls one out.
