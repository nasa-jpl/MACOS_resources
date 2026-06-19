# mmacos tests

`matlab.unittest` regression suite for the mmacos mex bridge and the
`+macos` package.  Driven by `../run_mmacos_tests.sh`, which groups the
classes into named suites (the file listing alone doesn't tell you which
class runs when — this table does).

For the **why** behind the testing approach (smoke vs. unittest layers,
the `exit(0)` batch-mode rule, the `clear mmacos` hang, the model_size
heap-corruption history) see `../CLAUDE.md` → "Tests: two layers" and
"Gotchas".  This file is just the map.

## How to run

| Command | Runs | ~Time |
|---|---|---|
| `./run_mmacos_tests.sh` | full suite (every group below except on-demand) | minutes |
| `./run_mmacos_tests.sh fast` | `SUITE_FAST` — dev iteration loop | ~10 s |
| `./run_mmacos_tests.sh masks` | `SUITE_MASKS` — heavyweight CodeV mask suite | ~10 min |
| `./run_mmacos_tests.sh proper` | PROPER-comparison (512 then 1024) | minutes |
| `./run_mmacos_tests.sh -k <substr>` | every method whose name matches `*<substr>*` | varies |
| `./run_mmacos_tests.sh <ClassName>` | one class (this is how on-demand classes run) | varies |

Always invoked through the script so each batch ends in `exit(0)`
(a bare `matlab -batch "addpath; func()"` hangs at process exit once a
mex is loaded — see CLAUDE.md).

## Suite groups → classes

| Suite | Model size | Classes |
|---|---|---|
| `SUITE_FAST` | 128 | tMmacosCmd, tMacosPkg, tMacosSession, tCrossSurface, tPerturbRoundtrip, tCodeVGrating, tSrsBugFlatZ, tDwDzZernike, tDwDx, tDwDxGroups, tDesignSystem, tDesignVary, tDesignSensitivities, tDesignOptimize, tVeneerXP, tCoroContrast², tBandLimitedMask¹ |
| `SUITE_FREEFORM` | 256 | tFreeFormComposite, tCalib |
| `SUITE_MASKS` | 128 | tCodeVApeMasks{Circ,Ellipse,Polygon,Rect}, tCodeVObsMasks{Circ,Ellipse,Polygon,Rect} |
| `SUITE_PROPER_512` | 512 | tProperCompareCassFF, tProperCompareCassFFAberrations |
| `SUITE_PROPER_1024` | 1024 | tProperCompareCoroNFprop, tProperCompareCoroPhase3, tProperCompareCoroApodizer, tProperCompareCoroDMPhase |
| **on-demand** (no group) | 128 | **tEndurance** |

¹ `tBandLimitedMask` is pure math (no macos calls); it lives under
`proper_compare/` but is glob-pulled into `SUITE_FAST` because it's
quick and model-size-agnostic.

² `tCoroContrast` is also pure math (no macos calls) — it pins the
ported `contrast.py` λ/D machinery in `examples/coronagraph/coro/` against
an analytic Airy pattern; model-size-agnostic, lives in `SUITE_FAST`
because it's quick.

The no-arg full run is `[FAST, MASKS, FREEFORM, PROPER_512,
PROPER_1024]` — i.e. **everything except `tEndurance`**, which is
excluded for being slow (~31 s) and using a Linux `/proc` RSS probe.

## What each class covers

| Class | Covers |
|---|---|
| tMmacosCmd | raw `mmacos('cmd', …)` dispatch surface (no veneer) |
| tMacosPkg | `+macos/` package functions (trace, opd, intensity, ray-status, src/elt getters+setters) |
| tMacosSession | `macos.Session` OO veneer / dot-notation flow |
| tCrossSurface | cross-surface ray geometry |
| tPerturbRoundtrip | perturb→modify→trace round-trip; pins the psiElt-renormalize ULP residual |
| tCodeVGrating | grating ray trace vs CodeV reference |
| tSrsBugFlatZ | SRS regression (flat-zElt clobber) |
| tDwDzZernike | dw/dz Zernike sensitivity channels |
| tDwDx | dw/dx single + multi-field channels (per-elt / source / FP) |
| tDwDxGroups | dw/dx group (GPERTURB) channels |
| tBandLimitedMask | band-limited mask math (pure) |
| tFreeFormComposite | SrfType 14 FreeForm (conic+Mon+FF+grid) composite |
| tCalib | CALIB optimizer wrappers (Phase 1) |
| tCodeV*Masks* | aperture / obscuration mask shapes vs CodeV (circ/ellipse/polygon/rect) |
| tProperCompareCassFF[Aberrations] | far-field PSF vs PROPER, nominal + SM perturbations |
| tProperCompareCoro* | near-field / coronagraph propagation vs PROPER (NF, phase3, apodizer, DM phase) |
| tDesignSystem | `macos.design.System.from_rx` import core — engine-readback spec matches direct getters (Sprint 2A-i) |
| tDesignVary | `macos.design.System.vary` declaration layer — name-based DOF addressing, alias expansion, Zernike modes, roles, validation (Sprint 2A-i) |
| tDesignSensitivities | `macos.design.System.sensitivities` — rigid + Zernike blocks bitwise-match standalone dw_dx / dw_dz_zernike (Sprint 2A-i) |
| tDesignOptimize | `macos.design.System.evaluate` / `optimize` — fmincon recovers a despace misalignment to min WFE; ray-loss guard; family guard (Sprint 2A-i) |
| tVeneerXP | spot / fex / get_xp / set_xp veneers on a STOP-bearing Rx (e5hex1); + regression that spot-on-stopless fails fast (engine infinite-loop fix) |
| tCoroContrast | ported `contrast.py` λ/D machinery (radial_profile / first_airy_null / lambda_over_D_pixels / radial_contrast) vs an analytic Airy disk — Sprint-1 E1 dark-zone merit (pure math) |
| tEndurance | load/trace endurance — bit-identical rmsWFE + flat memory over many iters (Q5) |

## Adding a class

Drop `t<Name>.m` in `tests/` (or `tests/proper_compare/` for PROPER-
comparison classes) and **add it to the relevant `SUITE_*` in
`run_mmacos_tests.sh`** — classes are listed explicitly, so an unlisted
class won't run in `fast` or the full suite (only via
`./run_mmacos_tests.sh <ClassName>`).  If it runs at a model_size not
already in its group, put it in the matching-size group instead.
Per the standing regression rule, every new wrapper / helper lands with
a test here.  Shared fixtures and helpers live in `private/`
(`rx_fixture_path`, tolerances, polygon helpers, …).
