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
├── conftest.py                       # pymacos session (init at 512),
│                                       3-panel plot + .mat/.txt report
│                                       fixture, comparison metrics
├── geometries/
│   ├── cass_farfield.py              # Rx_Cass_FarField.in + matching
│   │                                   PROPER thin-lens model; macos_run
│   │                                   and proper_run on one dataclass
│   └── circular_pupil_focus.py       # toy reference (PROPER side only)
├── test_cass_ff.py                   # nominal + nominal-with-OPD
├── test_cass_ff_aberrations.py       # SM Tx/Ty/Tz perturbations
├── test_psf.py                       # toy circular-pupil sanity test
└── results/                          # gitignored, regenerated each run
    ├── report.md                     # cumulative quantitative table
    ├── <test>.png                    # 3-panel macos / PROPER / diff
    ├── <test>.mat                    # full arrays + metadata
    ├── <test>.macos.txt              # ASCII 64x64 central crop
    └── <test>.proper.txt             # ditto, PROPER side
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

- **`test_cass_ff.py` and `test_cass_ff_aberrations.py`**: both engines
  fully wired against `Rx_Cass_FarField.in`. macos's INT array is
  retrieved via `pymacos.intensity(6)`; macos's OPD at the exit pupil
  is retrieved via `pymacos.opd()` after `trace_rays(5)`. PROPER takes
  its amplitude pattern directly from macos's mask and adds macos OPD
  as phase. **Agreement is at numerical precision (max |a-b| ~ 1e-11
  on Strehl-normalised PSFs)** for nominal + all six SM Tx/Ty/Tz
  perturbation cases.

- **`test_psf.py` + `circular_pupil_focus.py`**: leftover from the
  initial scaffolding. PROPER side works
  (`test_proper_circular_psf_runs`), macos side is `pytest.mark.skip`
  because `CircularPupilFocus.macos_rx_text()` is a stub that raises
  `NotImplementedError`. The cass_ff suite supersedes this; the toy
  test is kept only as a minimal PROPER-only sanity check.

## Critical learnings (don't lose these)

Reaching the 1e-11 agreement required two corrections to the
straightforward macos-OPD-to-PROPER path. Both live in
`geometries/cass_farfield.py`'s `proper_run`. Disabling either
(via the `opd_sign_flip=False` arg, or by switching back to the
analytical aperture model) reproduces the symptoms below — useful
when debugging a future regression.

1. **Mask-matched amplitude.** Take PROPER's amplitude DIRECTLY from
   macos's mask (`opd != 0`) via `prop_multiply`, instead of building
   PROPER's amplitude from analytical `prop_circular_aperture +
   prop_circular_obscuration + prop_rectangular_obscuration`. The
   analytical aperture treats pixels as illuminated that macos zeroed
   out (spider, edge ray losses, ...); zero-padding the OPD at those
   pixels then breaks the smooth wavefront tilt across the aperture
   and halves the apparent PSF shift under a tilt-class perturbation.

2. **OPD sign flip.** macos's OPD-positive convention is opposite to
   PROPER's `prop_add_phase` input. Without the flip, PROPER's PSF
   shifts in the *opposite* direction to macos's INT result.
   `opd_sign_flip=True` by default. Worth checking the macos source
   for which sign convention is actually documented.

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
