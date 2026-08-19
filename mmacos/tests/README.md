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
| `SUITE_FAST` | 128 | tMmacosCmd, tMacosPkg, tMacosSession, tCrossSurface, tPerturbRoundtrip, tCodeVGrating, tSrsBugFlatZ, tDwDzZernike, tDwDx, tDwDxGroups, tDesignSystem, tDesignVary, tDesignSensitivities, tDesignOptimize, tVeneerXP, tCoroContrast², tBandLimitedMask¹, tPolarization, tJonesPupil, tVecChain, tPolElement |
| `SUITE_FREEFORM` | 256 | tFreeFormComposite, tCalib, tReadGridFile, tViewRx, tSurfInspect, tPolContrast |
| `SUITE_MASKS` | 128 | tCodeVApeMasks{Circ,Ellipse,Polygon,Rect}, tCodeVObsMasks{Circ,Ellipse,Polygon,Rect} |
| `SUITE_PROPER_512` | 512 | tProperCompareCassFF, tProperCompareCassFFAberrations |
| `SUITE_POL_512` | 512 | tPolContrastCoro |
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
| tPolarization | PLAN_POLARIZATION Phase 1 — `polarization` / `vector_diffraction` / `coating` (Model A round-trip) / `ray_field` state + geometry gates |
| tJonesPupil | Phase 2a/2b — two-trace Jones pupil (double-pole / local-sp / global bases) + `pol_maps` polar decomposition + `pol_zernike` low-order expansion; unitarity, Fresnel-analytic fold, 2θ symmetry, and the two-mirror "polarization astigmatism" literature form.  NOTE: uses two fixtures with DIFFERENT BaseUnits (Rx_Cass_FarField = m, Bench fold rig = mm), so the Al thickness is two constants — see `thkAl`/`thkAlBench` |
| tPolContrast | Phase 2c — `pol_contrast_floor` exactness gates at model 256.  Rx_VecChain: a polarization-neutral train must report NO floor (cross exactly 0, co-polarized channel == the scalar run and its contrast curve to 1e-12).  Rx_Cass_FarField: Parseval on the co/cross split, energy closure against the engine intensity, floor by component, the coating sensitivity sweep, NaN-masking of small denominators, and the full-carry scope check.  The CIRCULAR input state is load-bearing — it is the only case that can see a conjugated coherency matrix |
| tPolContrastCoro | Phase 2c on the coronagraph chain at model **512** (Rx_Coro declares nGridpts=511).  Floor by component, Parseval/closure at scale, and — pinned deliberately — the Phase 3a Tranche-1 shortfall: the grid carries 0.84 of the ray-level cross-polarized fraction bare, 0.57 coated, and the coating sensitivity comes out with the WRONG SIGN.  Tranche 2 must change these two tests |
| tPolElement | Phase 3 polarizing elements — `TrPolarizer` (EltID 15, finished from a name-table-only stub) and `WavePlate` (EltID 18, new), on `tests/Rx/Rx_PolElt.in` + its geometric twin `Rx_PolElt_Ref.in`.  Every physics gate is a closed-form Jones identity written from the textbook, never transcribed from the engine: Malus, exact crossed extinction, QWP linear→circular with the SIGNED S₃ (which pins the retardance convention — a `|S₃|` gate would accept either), HWP 2θ, two QWPs ≡ one HWP, unitarity for linear AND circular input.  Each mechanism carries an in-suite non-vacuity A/B, because the pre-existing behaviour of these EltIDs was "silently do nothing".  Also: pol-off bit-identical to the Reference-surface twin, and the grid-carries-the-train tripwire for the two-dispatch-chain trap.  Section F gates the SETTLED off-normal convention — project the MATERIAL (absorbing) axis, per Korger et al., *Opt. Express* **21**, 27032 (2013) — on a third fixture, `Rx_PolElt_Tilt.in`, which tilts the BEAM to 20° (tilting the element does nothing to a collimated on-axis bundle): the transmitted axis lands on the material-rule closed form and misses the pass-axis one by the predicted 3.5616°, the crossed-analyzer null discriminates the two rules on the DETECTOR plane (9.1e-33 vs 1.53e-2), the degenerate azimuths are pinned as a vacuity guard, and normal incidence is asserted bit-identical against pre-flip captures |
| tVecChain | Phase 3a Tranche 1 — vector propagation across a multi-leg chain on `tests/Rx/Rx_VecChain.in`: polarized-scalar ≡ scalar bit-identically, vector ≡ scalar at round-off for x/45°/circular input, per-leg energy, mask throughput, far-field normalization A/B |

## Adding a class

Drop `t<Name>.m` in `tests/` (or `tests/proper_compare/` for PROPER-
comparison classes) and **add it to the relevant `SUITE_*` in
`run_mmacos_tests.sh`** — classes are listed explicitly, so an unlisted
class won't run in `fast` or the full suite (only via
`./run_mmacos_tests.sh <ClassName>`).  If it runs at a model_size not
already in its group, put it in the matching-size group instead.
Per the standing regression rule, every new wrapper / helper lands with
a test here.  Shared fixtures and helpers live in `private/`
(`rx_fixture_path`, tolerances, polygon helpers, …).  Rx fixtures come
from the shared pymacos corpus; an mmacos-ONLY prescription goes in
`tests/Rx/` and `rx_fixture_path` finds it there as a fallback.
