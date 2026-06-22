# `mmacos/design/` — parameterized telescope design drivers

This directory holds **design drivers**: runnable scripts you open, set a
handful of design knobs at the top, and run to produce a *complete,
optimized* MACOS telescope design — a `.in` prescription + a `.mat` spec +
report figures.

They are distinct from [`../examples/`](../examples), which are fixed demos
that each illustrate one feature of the design layer. A driver here is meant
to be **tuned**: change the aperture / f-numbers / off-axis distance / field
at the top and re-run to explore the design trade.

## The pattern

Every driver follows the same shape:

1. **User knobs** at the top (aperture, primary f/#, system f/#, off-axis
   distance, field, …).
2. **Build** the on-axis parent via `macos.design.Telescope`.
3. **Restructure** as the design needs (off-axis eccentric section,
   decenter, fold, …).
4. **Optimize** to diffraction-limited over the field (`optimize`).
5. **Verify** (e.g. `check_clipping` for an unobscured design).
6. **Save** the deliverable (`save` → `.in`, `save_spec` → `.mat`) and write
   report figures (`view_field_map`, `view_orthoviews`).

Generated artifacts (`.in` / `.mat` / `.png`) are git-ignored — re-run the
driver to regenerate them. Run from MATLAB with the package on the path:

```matlab
addpath('~/dev/MACOS_resources/mmacos/src');
run('~/dev/MACOS_resources/mmacos/design/rc_unobscured/rc_unobscured.m');
```

## Drivers

| Folder | Design |
|---|---|
| [`rc_unobscured/`](rc_unobscured) | Parameterized **unobscured off-axis Ritchey-Chrétien** (eccentric-pupil section). The off-axis distance trades against secondary size — a faster primary and/or slower system f/# shrinks the secondary and lets you go less off-axis (≈0.89·D → 0.64·D, floor ≈0.5·D). Focus behind M1. |
