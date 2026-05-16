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
│   ├── coro_nfprop.py                # Phase 2/3/4a: Rx_Coro.in NFPlane,
│   │                                   sphere-to-plane, pupil-to-pupil
│   └── circular_pupil_focus.py       # toy reference (PROPER side only)
├── test_cass_ff.py                   # Phase 1: nominal + with-OPD
├── test_cass_ff_aberrations.py       # Phase 1: SM Tx/Ty/Tz perturb
├── test_coro_nfprop.py               # Phase 2: NFPlane Elt 2 → 3
├── test_coro_nfprop_phase3.py        # Phase 3a/b + 4a:
│                                       NFPlane 5→6, sphere-to-plane 8→9,
│                                       pupil-to-pupil 8→10 (through focus)
├── test_psf.py                       # toy circular-pupil sanity test
├── results_phase1/                   # Phase 1 artefacts (gitignored)
├── results_phase2/                   # Phase 2 artefacts (gitignored)
└── results_phase3/                   # Phase 3/4a artefacts (gitignored)
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
  `pymacos.complex_field()` exposing `WFElt`), propagate 774 mm,
  compare at Elt 3.  Pupil-plane intensity uses **sum-normalised**
  (flux-conservation) metric.  **Agreement at 2.4e-14 RMS, 4.8e-13
  max** with the runtime `pymacos.dx_at()` query (see learning 6
  below) and `nGridpts=511` (odd, true grid-center pixel).  That's
  double-precision FFT round-off.

- **Phase 3 — `test_coro_nfprop_phase3.py`**: three additional Coro
  steps further down the chain.
  - **3a (NFPlane 5→6):** mirror of Phase 2 but at the second
    NFPlane.  Intervening DM (Elt 4) + reference apertures clip the
    beam to ~1/4 its Elt-2 footprint (R~63 px vs ~128 px), dropping
    the Fresnel number from ~2770 to ~684.  **Agreement at 3.7e-8
    max, 1.2e-9 RMS** — sampling-limited at the smaller beam.
    Investigation (2026-05-14) ruled out dx mismatch (matches to
    1e-16), sub-pixel grid origin (centroids match at 14+ digits),
    and kernel-boundary handling (apodization moved PROPER *away*
    from macos, confirming both engines correctly diffract the hard
    aperture edges).  Floor is sampling-limited for this geometry
    at N=1024 — expected to drop as ~1/N² with denser grid.
  - **3b (sphere-to-plane 8→9):** Siegman–Sziklas style; macos's
    `NF1/NF2` matches PROPER's `prop_lens(f) + prop_propagate(f)`
    bit-for-bit.  **4.2e-11 max, 4.8e-13 RMS** (peak-normalised,
    image-plane metric).
  - **4a (pupil-to-pupil 8→10 through focus):** chains the
    sphere-to-plane plus a second `prop_propagate(f)` past focus to
    Elt 10.  **5.9e-13 max, 1.3e-14 RMS** (sum-normalised).
    Pupil-side sampling auto-rebins back to the coarse grid by
    construction.
  - **4b (NFPlane 13→14):** mirror of 3a, downstream of the focus.
    **7.4e-5 max, 1.4e-6 RMS** -- the only step that doesn't hit
    numerical precision.  Diagnostic confirmed the residual is a
    localized diffraction ring at r=17-34 mm from earlier hard
    apertures (M1/DM/OAPs), not a handoff leak; agreement is
    machine-precision (1e-13) across the rest of the 80%-filled
    wavefront.  Re-enabling the 20 mm Lyot doesn't help (the ring
    sits INSIDE the Lyot's clear aperture); the Lyot adds its own
    edge residual at r=20 mm.

- **Phase 5 — `test_coro_nfprop_phase3.py` (continued)**: full chain
  to the science focal plane (Elt 21), with and without a basic
  coronagraph mask + Lyot.
  - **5 step 1 (no mask):** `Rx_Coro_noLyot.in`, ExitPupil (Elt 20,
    spherical, Kr=-951.4 mm) -> FocalPlane (Elt 21).  Macos's
    `PropType=FarField` is the same sphere-to-plane physics as
    Phase 3b/4a at a different f; reuses `CoroSphereToPlane`
    directly with `focal_length_m=0.9514`.  **2.3e-9 max, 2.7e-11
    RMS** (Strehl-normalised).
  - **5 step 2 (FPM + Lyot):** `Rx_Coro_FPM.in` adds a 400 um
    radius circular obscuration at Elt 9 (Element=Obscuring) and
    a 14 mm radius circular Lyot stop at Elt 14.  Sizes tuned by
    /tmp/fpm_lyot_sweep.py + /tmp/lyot_sweep.py (FPM diminishing
    returns above ~400 um at this F#; Lyot ~14 mm crops the
    leakage halo at the pupil edge cleanly).  Macos applies both
    masks during its trace; PROPER ingests the post-mask cfield at
    Elt 20 and propagates the last sphere-to-plane step.
    **2.6e-6 max, 1.0e-8 RMS** (Strehl-normalised).  Looser than
    step 1's 2.3e-9 because the suppressed wavefront has hard-edge
    diffraction structure that the two engines' finite-grid kernels
    sample slightly differently -- same sampling-limited regime as
    Phase 3a.  In absolute units the residual is ~2.5e-13 against
    a peak of 9.5e-8.  On-axis suppression: **factor 3.2 million**
    versus the no-mask baseline (0.304 -> 9.5e-8).

  **Critical gotcha (debugged 2026-05-14):** Elt 9 (the FPM
  element) MUST be declared `Element= Obscuring`, not
  `Element= Reference`, for macos to apply the circular obscuration
  to the diffraction-grid wavefront.  A `Reference`-typed element
  carries `nObs / ObsType / ObsVec` metadata but only applies it
  to geometric rays during ray tracing; the diffraction `WFElt`
  array sails through untouched.  This silently broke the original
  Phase 5.2 setup (FPM=132 um + Lyot=20 mm) where suppression was
  only ~17% -- the apparent "weak coronagraph" was actually no
  coronagraph at all, just flux trimming at the 20 mm Lyot.

  Prescriptions:
  - `Rx_Coro_noLyot.in` : no FPM, no Lyot (Phase 4b / 5.1 baseline)
  - `Rx_Coro_FPM.in`    : FPM=400 um + Lyot=14 mm (Phase 5.2)
  - `Rx_Coro_FPM_noLyot.in` : FPM=400 um, no Lyot (sweep helper)
  - `Rx_Coro.in`        : pristine, do not edit -- original
                          prescription used as the source for all
                          variants

  Designing a properly-matched Lyot coronagraph (and later
  apodizers) will use a different setup; this prescription is
  retained as a "good enough" test case for macos<->PROPER
  validation across the full chain.

- **Phase 6 — `test_coro_apodizer.py` + `apodizer.py`**:
  pupil apodisation via a NEW pymacos wrapper.
  - **6a (done):** `pymacos.apodize(srf, mask)` multiplies macos's
    `WFElt(:,:, iEltToiWF(srf))` in place by a user-supplied real
    NxN mask -- companion to PROPER's `prop_multiply`.  The same
    numpy array goes to both engines, so apodisation is
    bit-identical with no parametric drift.  `apodizer.py` provides
    `build_apodised_mask(N, dx, aperture_fn, taper_fn, K)` using
    KxK super-sampling for sub-pixel area-weighted aperture edges
    -- low-N results converge to high-N as the same physical shape
    rather than re-quantising the boundary.  First Phase 6 test
    applies a Gaussian-edge taper (r0=18 mm, sigma=2 mm, truncated
    at 26 mm) at Elt 5 and compares NFPlane Elt 5 -> 6 with macos
    and PROPER.  **4.0e-8 max, 1.1e-9 RMS** (sum-norm) -- same
    sampling-limited regime as un-apodised Phase 3a (3.7e-8),
    confirming the wrapper integrates correctly with downstream
    propagation.

- **Contrast scoring — `test_coro_contrast_curve.py` +
  `contrast.py`**: radial dark-zone contrast vs lambda/D at the
  science focal plane.  Decouples scoring from the engine-specific
  peak/sum normalisations (macos peak = 0.30 vs PROPER peak = 0.012
  for the same physical PSF -- different internal conventions, but
  the Strehl-normalised contrast `mean_intensity(r) / peak_unaberrated`
  is engine-independent).
  - lambda/D derived empirically from the un-coronagraphed PSF's
    first Airy null (1.22 lambda/D), with a fractional-depth guard
    (the detected null must be < 5% of peak) so spurious sub-pixel
    local minima on the central Airy peak's falling slope don't get
    picked up.  No need to know the prescription's effective pupil
    diameter at the science focal plane (depends on the full
    magnification chain through Elts 15-20).
  - First scoring pass on Phase 5.1 + 5.2: lambda/D = 8.6 px on this
    1024 grid; **dark-zone contrast (3-10 lambda/D) ~ 1e-9 to 3e-10**;
    bright outer-ring artefact at ~15 lambda/D (Lyot edge diffraction).
    Plot in `results_phase3/contrast_curves.png`.
  - This curve becomes the baseline that Phase 6b/c (band-limited
    apodisers, HWO designs) will score against.  When new apodised
    setups land they just register additional curves on the same
    overlay plot via `plot_contrast_curves({...})`.

## Phase 6 roadmap (next steps)

- **6b: gold-standard apodiser construction.**  Add a second mask-
  builder in `apodizer.py` using analytic Fourier-domain construction
  of the binary aperture (Airy-disk-like ring in k-space, low-pass
  filter to grid Nyquist, inverse FFT) so the resulting real-space
  mask is perfectly band-limited with no super-sampling artefacts.
  Slots into the same API as `build_apodised_mask`.
  - **6b-1 (done):** `BandLimitedCircle` + `build_band_limited_mask`
    in `apodizer.py`.  Math-level characterisation
    (`test_band_limited_mask.py`): integral preservation to 6+ sig
    figs, shape invariance < 2% across N in {128..1024}, visualisation
    in `results_phase3/band_limited_vs_super_sampled.png`.  Surprise
    null result: at K=16 super-sampling, SS aliasing magnitude
    (~1e-4) is dwarfed by the legitimate Airy F(k) tail extending
    to Nyquist (~1e-3 at N=256, ~1e-4 at N=1024).  Both methods
    produce essentially equivalent Fourier content in the mask;
    Phase 6b's advantage lives in PROPAGATION RESULT, not in mask
    spectra.

  - **6b-2 (deferred):** the natural propagation-based comparison
    (BL vs SS mask as an externally-applied Lyot at Elt 14, scored
    via radial contrast at the science focal plane across N) ran
    into a real-physics limit of `pymacos.apodize`.  See "apodize
    limitation" note below.  We'll come back to 6b-2 either after
    upgrading the test prescription to a fully-diffractive chain
    (see option 4 below), or after extending apodize to also clip
    rays.  For now, BL is validated mathematically; its propagation
    payoff over SS hasn't been demonstrated yet, and may not exist
    until contrast targets cross the K=16 SS aliasing floor
    (~1e-8 ish).

- **6c: HWO-style coronagraph designs.**  Once the gold-standard
  builder is in place, apply it to specific HWO candidate masks:
  CGI-style shaped pupils, PIAA apodisers, vortex masks (the last
  requires extending pymacos.apodize to accept a complex mask;
  trivial Fortran extension -- amplitude+phase array splitting,
  same template as cfield_get).  Each new mask is a config-only
  change at this point -- the harness is built.

- **6d (provisional): wavefront-control loop demonstrations.**
  Apply the apodise + DM-pattern + Lyot chain repeatedly to drive
  the dark-zone contrast via an EFC-like update rule.  This is
  where macos's DM model (Elt 4) starts to matter directly.
  Out of scope until 6a/b/c land and HWO-relevant prescriptions
  are available.

## Cross-cutting concerns (apply across all phases)

Three infrastructure pieces that intersect every phase past 6a and
should be built once for reuse:

- **Broadband contrast scoring.**  Coronagraph performance is only
  meaningful in a wavelength band -- monochromatic contrast can hide
  pathologies that bite at the band edges.  Implementation: sample
  5-7 wavelengths uniformly across a 10-20% band centered on the
  prescription wavelength, run macos and PROPER at each wavelength,
  sum focal-plane intensities incoherently to produce
  `I_broadband = sum_i I(lambda_i)`, score that with the same
  `radial_contrast()` machinery the monochromatic case uses.
  Wavelength is set per-run via `pymacos.src_wvl()` on the macos
  side and passed to `proper.prop_begin(beam_d, lambda, n, ratio)`
  on the PROPER side.  Contrast axis uses the centre wavelength's
  lambda/D.

- **Aberration sweeps.**  Re-run every phase against
  representative perturbations:
  - DM-commanded shape (Zernike coefficients via
    `pymacos.elt_srf_zrn_set(...)` on Elt 4 in Rx_Coro.in);
  - segment phase errors (where applicable);
  - rigid-body OAP perturbations (already exercised by the Phase 1
    Cass FF perturbation tests; extend to the Coro chain).
  Most perturbations live BEFORE the PROPER handoff at Elt 20, so
  PROPER just ingests the perturbed cfield and propagates the last
  step -- same pattern as Phase 5.2's FPM+Lyot handoff.  The
  macos<->PROPER residual confirms PROPER's propagation works on
  aberrated wavefronts; the contrast metric shows how much each
  aberration costs dark-zone performance.

- **Sampling-density (N) sweep.**  Run every phase at
  N in {256, 512, 1024, 2048} (with the matching odd `nGridpts` in
  the prescription) and record:
  - macos<->PROPER agreement (max + RMS, sum- or peak-norm per
    phase's existing metric);
  - radial dark-zone contrast at a few key separations (3, 5, 10
    lambda/D);
  - wall-clock runtime per test;
  - peak memory (N=2048 needs ~2 GB at double precision per
    NxN array -- the system has crashed VS Code at N=2048 once
    already; tread carefully).
  Establishes the "knee" of accuracy vs N so we know which N is
  appropriate for what kind of test (cheap development iteration
  vs production scoring vs convergence checks).  This sweep is
  also where Phase 6b's "super-sampling vs band-limited mask"
  comparison shows its teeth: gold-standard masks should give
  N-independent dark-zone contrast where super-sampled masks
  degrade as N drops.

**Sequencing suggestion (informal):** N sweep first -- it's cheap to
set up and decides what N to use for everything else.  Then
aberrations (still monochromatic) to validate the perturbation
plumbing.  Then broadband.  A fully realistic HWO-style test
eventually combines all three: broadband + aberrated + at the chosen
N.

## pymacos.apodize limitation (debugged 2026-05-16)

`pymacos.apodize(srf, mask)` modifies `WFElt(:,:, iEltToiWF(srf))` --
the diffraction-grid wavefront -- in place.  Macos's downstream
propagation **does** pick this up: a zero mask at Elt 14 takes the
science focal plane (Elt 21) to exact zero.

But macos's prescription-driven aperture stops (`Element=Reference`
with `ApType=Circular`, etc.) do TWO things during the trace, not
one:

1. Multiply WFElt by the aperture mask (the half pymacos.apodize
   replicates).
2. **Clip the rays** -- set `LRayPass = False` for rays that miss
   the aperture's geometric footprint.  Those rays then carry no
   flux through the subsequent geometric props.

Geometric props (PropType=Geometric) propagate rays + per-ray OPD;
the next diffractive plane reconstructs WFElt from the ray grid.
So the "ray clip" half is necessary for hard-edge apertures whose
effect reaches the next pupil/focal plane via the ray channel.

Concrete bite: replacing the 14 mm Lyot at Elt 14 (macos
`ApType=Circular`) with `pymacos.apodize(14, BL_circle(14mm))` at
the same physical location only achieves ~factor 17 of Phase 5.2's
3.2-million suppression at the science focal plane.  The "missing"
factor 200 is in the rays that should have been clipped at Elt 14
but instead propagated.

**Honest use of pymacos.apodize:**
- **Smooth apodisers** (Gaussian taper, polynomial roll-off,
  super-Gaussian, etc.) where the WFElt is the meaningful
  representation -- the ray-channel contribution from "outside the
  apodiser" is fuzzy by design.  Phase 6a's Gaussian-edge test is
  this regime.
- **Downstream-only effects**, where the apodise lives immediately
  before a diffractive prop (NFPlane, NF1/NF2, FarField) and the
  result is read out before any subsequent geometric props.
  Examples in `Rx_Coro.in`: apodise at Elts 2, 5, 8, 13, 20 (each
  the start of a diffractive prop).

**For hard-edge apertures in a real chain**, use macos's
prescription-driven `ApType=Circular`/etc.  That's what Phase 5.2
relies on and it works correctly.

**Long-term fix options** (in increasing order of effort):

- *Extend `pymacos.apodize` to also clip rays.* Sibling
  `apodize_with_rays(srf, mask)` that sets `LRayPass = False` for
  rays outside the mask support (with a threshold for partial
  transmission).  Roughly 30 more lines of Fortran.
- *Wire macos's prescription parser for external mask arrays.*
  `ApType=External` reads a user-loaded NxN array via a new
  parameter.  Touches macos source -- more invasive.
- ***Replace geometric props with NF-props between coextant reference
  surfaces.*** The physically-most-correct model: every element gets
  a NFPlane (or appropriate diffractive) prop to the next, with the
  optic represented by a reference surface coincident with the
  physical optic.  Then WFElt propagates diffractively at every step,
  apertures are WFElt multiplications, and `pymacos.apodize` works
  end-to-end.  Requires prescription rework -- worth it for the
  HWO-realistic case.  (Dave's suggestion, 2026-05-16.)

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

6. **Runtime dx query via `pymacos.dx_at()` (Phase 2+).**  macos
   stores per-element diffraction-grid pitch `dxElt(iElt)` in the
   prescription's BaseUnits (mm for `Rx_Coro.in`, m for
   `Rx_Cass_FarField.in`).  For plane-to-plane and sphere-to-sphere
   propagations, `dxElt` is **set by the ray-grid geometry** -- a
   strictly geometric calculation, NOT an analytical
   `extent / (N-1)` shortcut.  Reach into `dxElt` directly with
   `pymacos.dx_at(srf, unit='m'|'mm'|'native')` (default `'m'`,
   returns SI metres via `dxElt * CBM` in the Fortran wrapper).
   Hardcoded 5-sig-fig dx values from macos's diagnostic output cap
   agreement at ~1e-7; runtime queries unlock 1e-13 in Phase 2 and
   Phase 4a.

7. **Odd `nGridpts` for true grid-center pixel.**  Even-N grids
   place the FFT-center between four pixels, which manifests as a
   sub-pixel Δcom = (0,0) but a stubborn ~1e-5 max residual that
   doesn't go away no matter how careful the handoff.  Setting
   `nGridpts` to an odd value (`511`, `1023`, ...) puts a true
   pixel at the optical axis and drops Phase 4a from 1e-5 to 1e-13
   on its own.

8. **`prop_end` return type guard.**  PROPER's `proper.prop_end(wfo)`
   returns either complex amplitude (`field.dtype.kind == 'c'`) or
   already-squared real intensity, depending on internal state.
   Always guard:
   ```python
   intensity = np.abs(field) ** 2 if field.dtype.kind == 'c' else field
   ```
   Squaring an already-real intensity is a silent 5-order-of-
   magnitude bug.

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
