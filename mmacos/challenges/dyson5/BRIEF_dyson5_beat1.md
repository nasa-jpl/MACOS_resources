# dyson5 beat 1 -- report (2026-09-30, TO lane; run on Fable at Dave's "read and go")

Brief: `macos/BRIEF_to_dyson5.md`.  Deliverable of the first beat: gates
1a/1b green **or a STOP report**, `dyson_layout` with the condition verified
and the F/1.8 / 54 mm scaling on record, and the runner skeleton.

## STOP report -- gate 1b (and therefore 1a) fails on an ENGINE defect

**`GlassElt=` is inert in the current tree, CLI and bindings alike.**
Not this lane's fix (brief: "engine fixes are not this lane's").

- **Measured (mmacos, `tGlassDispersion`):** a `GlassElt= Silica` face
  tilted 30 deg refracts with n = **1.000000000000000** at 0.5, 1.0 and
  2.0 um (expected 1.4623 / 1.4504 / 1.4381); the index does not move.
  The fixed-index control (`IndRef= 1.5`, no GlassElt) reads 1.5 to
  1e-12 at every wavelength -- the measurement is sound.
- **Reproduced in the CLI** (`build_release_gfortran/bin/macos`, pty):
  the catalog loads at start-up ("Glass table ... 201 glasses"), then
  `old Rx_GlassPlate` prints `Default used for GlassCoef(1)` and `(2)`
  and `show 1` reports `IndRef=1.0E+00`.  The same `Default used for
  GlassCoef` line appears 8 times in the mmacos gate log.
- **Mechanism (read, not guessed):** the parser resolves the name by
  scanning `GlassName(1:mGlass)` (msmacosio.inc ~:2796).  Every load
  runs `reinitialise_variables()` (smacosio.F:155 / macosio.F:176) ->
  `elt_mod_init_vars()` -> `GlassName(:) = ''` (elt_mod.F ~:940) -- the
  catalog is blanked BEFORE the lookup.  The catalog is (re)loaded only
  at SMACOS first entry (`smacos.F:177`, inside `IF (.NOT.ifInit)`) and
  at the CLI's model-size reset (`macos_cmd_loop.inc:173`,
  `rl_macos_glass.inc`), never per load.  So `LGlass(iElt)` stays
  false, `GlassCoef_FLG` false ("Default used"), CTRACE never calls
  `getIndex`, and the element keeps its written `IndRef` -- the silica
  block traces as air.
- **Fix candidates (CC's call):** (a) stop blanking `GlassName`/
  `GlassTable` in `elt_mod_init_vars` (they are catalog state, not Rx
  state; check the very first allocation still starts them blank before
  the catalog load), or (b) reload the catalog at MBFile6 entry after
  `reinitialise_variables` (the CLI's `rl_macos_glass.inc` is the
  ready-made include).  Rebuild only after
  `tg_psi_dm96_oap/runs/fix2x2.done` exists (the descent 2x2 owns the
  build).  Both gates below go green on the fix with no test change.
- **Gate 1a consequence:** with the block traced as air, the immersed
  grating residual is exactly `(n-1) m lambda0/d` (0.02312 at 0.5 um,
  0.04504 at 1.0 um -- (1.4623-1)*0.05, (1.4504-1)*0.10) and the AIR form
  holds to 3.5e-17, so the grating equation is consistent and the
  immersed form is simply untested until the glass works.  The air
  control PASSES.  The `IndRef= 1` trap leg is written and will pin the
  engine's `nb = IndRef(iElt)` assignment once the block is glass.

Engine facts pinned on the way (in the reference doc §2): the Grating
branch passes `nb = IndRef(iElt)` as written and never overwrites it with
the incident medium (a Reflector does), so an immersed grating must carry
`GlassElt` on its own element; `ray_info_get` returns the OUTGOING
direction at the traced element (its "direction before Srf" comment is
stale -- CTRACE overwrites RayDir in place), which is how the gates read
directions.

## Delivered

| item | state |
|---|---|
| `tests/Rx/Rx_GratingImmersed.in`, `Rx_GlassPlate.in` | written, headers carry the engine facts |
| `tests/tGratingImmersed.m` (4 tests) | 1 pass (air control) / 3 fail -- all on the glass defect |
| `tests/tGlassDispersion.m` (3 tests) | 1 pass (fixed-index null) / 1 fail (Silica) / 1 Incomplete (CaF2, by assumption) |
| both registered in `SUITE_FAST` | yes (they will show red in CCL's suite runs until the fix; deliberate -- gates fail closed) |
| CaF2 in the engine glass table | macos dev-candidate **0ca61c1** (`macos_glass_list.txt` + regenerated `glass_builtin.f90`), NOT rebuilt |
| `design/src/dyson_layout.m` | condition verified by exact 3-D trace: blur 0.67 um at R_g = n r/(n-1) vs 4.2/1.2/2.5/5.1 um at 0.95/0.98/1.02/1.05; image at -h; h-exponent 4.17 |
| `design/src/dyson_scaling.m` | r-exponent -3.19; **r_required = 213 mm (R_g 687 mm)** at the spec's slit corner (28.16 mm) for 0.25 px |
| runner `dyson5_params.m` + `dyson5_run.m`, stage s0 | record `dyson5_s0_scaling.{txt,mat,png}` |
| `optical_design/SPECTROMETER_DESIGN_REFERENCE.md` | written; on `DEV_FILES.md` + `release-exclude.txt` |
| `challenges/README.md` row, `challenges/dyson5/README.md` | written |

Mouroulis & Green 2018 was not fetchable (JS wall); the Dyson condition
was verified numerically instead and matches Dyson 1959 / Mertz 1977
("the block fills (n-1)/n of the slit-grating space").

## Next beat (unchanged build order, after CC's glass fix lands)

s1 deck emission (Dyson in the JPL form: block + air gap + concave grating
in air, the Offner from `offner_layout`), s2 `spectrometer_score`, the
Offner wave-twin harness first (all air).  Open design question for the
room, to carry into the record: at 54 mm the classical seed is 2x too
large -- which departure (asphere on the block face, meniscus, slit air
gap) buys the most per element?
