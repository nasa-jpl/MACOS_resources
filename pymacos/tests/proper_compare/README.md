# macos vs PROPER comparison harness

Validation reference for macos physical-optics paths (INT, PIX, OPD with
diffraction, far-field DFT) that CodeV cannot reach. Uses
[PyPROPER3](https://proper-library.sourceforge.net/) (John Krist's
Python port of PROPER) as the comparator.

## Installation

PROPER is not on PyPI as a working modern wheel — the source archive's
build scripts use the legacy `ez_setup` bootstrap. Install procedure
that works as of 2026-05:

```bash
cd /tmp
wget https://files.pythonhosted.org/packages/65/63/3d569f23800bd829e47f0c3be0d7774291bf3f8d8653f8ca3ad176f76cf8/PyPROPER3-3.2.1.tar.gz
tar xzf PyPROPER3-3.2.1.tar.gz
source ~/dev/MACOS_resources/pymacos/.venv/bin/activate
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
cp -r PyPROPER3-3.2.1/proper "$SITE/"
uv pip install astropy
python -c "import proper; print('OK')"
```

## Layout

```
proper_compare/
├── conftest.py                       # per-phase results-dir fixtures,
│                                       3-panel plot + .mat report,
│                                       comparison metrics (peak/sum
│                                       norm, centroid alignment)
├── geometries/
│   ├── cass_farfield.py              # Phase 1: Rx_Cass_FarField.in
│   ├── coro_nfprop.py                # Phase 2: Rx_Coro.in Elt 2 → 3
│   └── circular_pupil_focus.py       # toy reference (PROPER side only)
├── test_cass_ff.py                   # Phase 1: nominal + with-OPD
├── test_cass_ff_aberrations.py       # Phase 1: SM Tx/Ty/Tz perturb
├── test_coro_nfprop.py               # Phase 2: NF prop comparison
├── test_psf.py                       # toy circular-pupil sanity test
├── results_phase1/                   # Phase 1 artefacts (gitignored)
│   ├── report.md
│   ├── cass_ff_*.png
│   └── cass_ff_*.mat
└── results_phase2/                   # Phase 2 artefacts (gitignored)
    ├── report.md
    ├── coro_nfprop_*.png
    └── coro_nfprop_*.mat
```

## Run

From the pymacos root:

```bash
./run_proper_tests.sh              # straight run
./run_proper_tests.sh --build      # rebuild pymacosf90 first (after a
                                   #   macos_f90/ change + makems.sh)
./run_proper_tests.sh -v           # forwarded to pytest
```

Or by hand:

```bash
source .venv/bin/activate && source /opt/intel/oneapi/setvars.sh intel64
cd tests && pytest proper_compare/ -v
```

## Status

- **Phase 1 — `test_cass_ff.py` + `test_cass_ff_aberrations.py`**:
  both engines fully wired against `Rx_Cass_FarField.in`.  macos's
  INT array via `pymacos.intensity(6)`; OPD at the exit pupil via
  `pymacos.opd()` after `trace_rays(5)`.  PROPER takes amplitude
  directly from macos's mask and adds macos OPD as phase.  Image-
  plane PSF comparison uses **peak-normalised** metric.  **Agreement
  at numerical precision (max |a-b| ~ 1e-11)** for nominal + six SM
  Tx/Ty/Tz perturbation cases.

- **Phase 2 — `test_coro_nfprop.py`**: near-field plane-to-plane
  propagation between Elt 2 and Elt 3 of `Rx_Coro.in`.  Both engines
  start from macos's diffraction-grid complex field at Elt 2 (via
  the new `pymacos.complex_field()` wrapper exposing `WFElt`),
  propagate 774 mm, compare at Elt 3.  Pupil-plane intensity uses
  **sum-normalised** (flux-conservation) metric.  **Agreement at
  5e-12 RMS, 2.5e-10 max** in fraction-of-flux-per-pixel -- at
  double-precision FFT round-off floor.  Peak-normalised metric
  would have read 9e-6 here, but that's a normalisation artefact
  (peak inside a flat-top is noise-dominated) -- the engines
  agree on the physics to roughly seven significant figures inside
  the bright pillar and to round-off in the dark region.

- **`test_psf.py` + `circular_pupil_focus.py`**: leftover from the
  initial scaffolding. PROPER side works
  (`test_proper_circular_psf_runs`), macos side is `pytest.mark.skip`
  because `CircularPupilFocus.macos_rx_text()` is a stub that raises
  `NotImplementedError`. The cass_ff and coro_nfprop suites supersede
  this; the toy test is kept only as a minimal PROPER-only sanity
  check.

## Critical learnings (don't lose these)

Reaching the observed agreement required three reconciliations to
the straightforward macos→PROPER hand-off (plus two metric choices
that turn out to be necessary for the residual to be readable).
Future regressions can usually be diagnosed by checking these in
order.

1. **Mask-matched amplitude.** Take PROPER's amplitude DIRECTLY from
   macos's mask (`opd != 0`) via `prop_multiply`, instead of building
   PROPER's amplitude from analytical `prop_circular_aperture +
   prop_circular_obscuration + prop_rectangular_obscuration`. The
   analytical aperture treats pixels as illuminated that macos zeroed
   out (spider, edge ray losses, ...); zero-padding the OPD at those
   pixels then breaks the smooth wavefront tilt across the aperture
   and halves the apparent PSF shift under a tilt-class perturbation.

2. **OPD sign flip.** macos's OPD-positive convention is opposite to
   PROPER's `prop_add_phase` input.  Without the flip, PROPER's PSF
   shifts in the *opposite* direction to macos's INT result.
   `opd_sign_flip=True` by default.  Worth checking the macos source
   for which sign convention is actually documented.

3. **Normalisation choice (`norm_kind` in `compare_and_record`).**
   `'peak'` (Strehl-norm, default; right for image-plane Airy-style
   PSFs) or `'sum'` (flux-norm; right for pupil-plane / near-field
   intensities).  Using peak-norm on a flat-top NF PSF inflates the
   reported residual by ~4 orders of magnitude because of a tiny
   mismatch in how the two engines normalise their peaks.  Phase 2
   passes `norm_kind='sum'`.

4. **Centroid (not peak) alignment.**  Post-comparison roll uses
   intensity-weighted center of mass instead of `argmax`.  Peak
   position inside a flat-top is noise-dominated; centroid is robust
   for both NF flat-top pillars and FF Airy peaks.  The roll is
   Python-side only and never touches macos's `WFElt` -- continuing
   propagation past the current element is unaffected.

5. **Diffraction-grid wavefront pass-through (Phase 2+).**  For
   perturbation-aware comparisons in the coronagraph chain, pass
   macos's complex field at the propagation start plane via
   `pymacos.complex_field()` (exposes `WFElt(:,:, iEltToiWF(iElt))`).
   This beats the legacy `opd()` for NF tests because `opd()`
   returns the SOURCE-ray-grid OPD (e.g. 512×512 over the source
   aperture), while `complex_field()` returns the diffraction-grid
   wavefront (mdttl × mdttl) that matches PROPER's expected
   sampling.

## Design notes

- **Single source of truth for geometry.** Each problem is a
  `geometries/*.py` dataclass with constants and a renderer for each
  engine. Tests pull from the dataclass; the two engines can never
  disagree on input parameters.
- **No resampling in the OPD path.** PROPER's entrance-pupil pixel
  pitch is chosen so it matches macos's source-grid pitch exactly
  (`proper_grid_n * beam_ratio == macos nGridpts`); macos's 256×256
  OPD zero-pads cleanly into PROPER's 512×512 grid. The
  `allow_resample=True` flag falls back to `scipy.ndimage.zoom` but is
  off by default — resampling can mask real engine disagreement with
  interpolation error.
- **Sampling match at the focal plane.** macos at `model_size=512`
  emits 512×512 at dx=2.78e-6 m; PROPER at N=512, beam_ratio=0.5
  emits 512×512 at the same dx. Direct pixel-by-pixel comparison.
- **Tolerances** live in `conftest._TolPO` (formal table) and in each
  test's assertion (currently `< 1e-6` on `max_abs` after the two
  corrections above).
- **Artefacts.** Every test writes a PNG (log-stretch intensity panels,
  diverging colormap diff panel, 64-pixel central crop), a `.mat` with
  raw + sum-normalised + peak-normalised arrays plus geometry metadata
  and (for perturbation tests) the macos OPD, plus ASCII central crops
  for either engine.
