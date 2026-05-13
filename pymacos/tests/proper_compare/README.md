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
├── conftest.py                  # fixtures: pymacos session, tolerances, grid helpers
├── geometries/                  # one .py per problem; both engines render from same source
│   └── circular_pupil_focus.py  # circular unobstructed pupil → on-axis focus
├── test_psf.py                  # PSF / intensity comparisons
├── (later) test_propagation.py  # wavefront at intermediate planes
├── (later) test_apodization.py  # complex masks through propagation
└── (later) test_coronagraph.py  # Lyot / classical / apodized coronagraphs
```

Run:

```bash
cd ~/dev/MACOS_resources/pymacos/tests
pytest proper_compare/ -v
```

## Status

- **PROPER side wired and verified**: `test_proper_circular_psf_runs`
  passes — PROPER produces an Airy PSF at the focal plane with
  expected sampling.
- **macos side not yet wired**: `test_compare_circular_psf` is
  `pytest.mark.skip`. The blocker is generating a valid macos `.in`
  prescription for the geometry (`macos_rx_text()` in
  `geometries/circular_pupil_focus.py` raises `NotImplementedError`).
  Filling this in is the next implementation step.

## Design notes

- **Single source of truth for geometry.** Each problem lives in a
  `geometries/*.py` dataclass with constants and a renderer for each
  engine. Tests pull from the dataclass; the two engines can never
  disagree on input parameters.
- **Grid alignment.** PROPER and macos may emit PSFs on slightly
  different physical grids. `conftest.resample_to_common_grid` does
  centered cropping under the assumption both arrays are centered on
  the PSF peak. Diffraction-limited cases satisfy this; off-axis or
  asymmetric cases will need a proper resampling step.
- **Tolerances** live in `conftest._TolPO` — `intensity_abs=1e-3`,
  `intensity_rel=1e-3`, `phase_waves=1e-3`. Loose by design;
  tighten as confidence grows.
- **Unit/sign reconciliation.** A one-time calibration table will
  likely be needed for Fourier sign, OPD sign, and pixel-pitch
  conventions. Document discrepancies here as they're discovered.
