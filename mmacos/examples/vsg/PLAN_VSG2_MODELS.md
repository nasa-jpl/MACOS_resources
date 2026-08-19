# PLAN — VSG2 models: IFO (Twyman-Green) + ZWFS

> **Executor:** Claude Code (Opus 4.8 or later), working in
> `~/dev/MACOS_resources/mmacos`.  Read this WHOLE file before writing
> code.  Also read first: `mmacos/CLAUDE.md` (all of it),
> `MACOS_resources/pymacos/CLAUDE.md` §"DM phase-imprint validation
> (Phase 6)" (the apodize-and-propagate pattern this plan reuses), the
> worked example `templates/30_instruments/coro_walkthrough/coro_walkthrough.m`
> (the propagation/figure idiom to copy), and the review doc in this
> directory.  Source hardware docs: `VSG2_Info.pptx` (this dir) and the
> authoritative ZWFS-upgrade deck `~/Documents/DCR'26/mmacos/VSG2
> Zernike Wavefront Sensor Update -v2.pptx` — the latter fixes the DM
> pupil diameter (D = N_act·pitch·√2 ≈ 68 mm, NOT 48 mm), F/# ≈ 3.6,
> and the 9-spot etched-depth transmissive mask array (see §6/§8).

## 1. Context (what VSG2 is — condensed)

Twyman-Green interferometer measuring a deformable mirror (DM), in a
vacuum chamber.  HeNe (632.8 nm, frequency-stabilized) → SM fiber →
baffle → L1 collimator (Newport PAC097AR.14 achromat, D=76.2 mm,
EFL 750 mm — the layout figure labels it "700mm EFL", a typo per the
review doc) → beamsplitter → {test arm: DM} / {reference arm: flat on
PZT phase-shift stage} → recombine at BS → L2 (Newport PAC095AR.14,
D=76.2 mm, EFL 250 mm) → Fold1 → Fold2 → ND → Andor NEO sCMOS
(2560×2160 px, 6.5 µm).  Camera is at the DM's image conjugate
(DM→L2 ≈ 1175 mm, L2→camera ≈ 317 mm; thin-lens-exact for f=250 mm;
magnification ≈ 0.27).  The DM is the system pupil.  There is an
internal point-source focus 250 mm past L2 (~67 mm before the camera)
— that is where the ZWFS phase dimple goes.

**Illuminated pupil diameter (authoritative — VSG2 ZWFS deck, slides
5/16):** the DM is illuminated over the circle CIRCUMSCRIBING its
square actuator grid, so the collimated beam diameter is
**D = N_act·pitch·√2**, NOT the square edge.  For the 48×48 AOX DM
(1.0 mm pitch): D = 48·√2 ≈ **68 mm** → imager **F/# = 250/68 ≈
3.57–3.7** (the deck quotes both 3.57 for D=70 and 3.7 for D=68).
The bench also supports a **96×96 BMC DM at 0.4 mm pitch** (same
optics, different pupil sampling); model the DM format as a
selectable param, not a constant.  DM image at the camera ≈
17.9–18.9 mm across, sampled at **13.4 binned (3×3) px/mm** in the
DM plane (>10 px per AOX actuator — adequate for both modes).

**Dual-mode hardware (deck):** mode switching is by two remotely
actuated stages in-chamber — a **Xeryon XYZ stage** carrying the ZWFS
mask (positions the selected dimple onto the internal focus) and an
**Aerotech ATS-100 linear stage** blocking the reference arm in ZWFS
mode.  The ZWFS "mask" is a **1 mm fused-silica substrate (Thorlabs
W4101FT1, one-side AR)** carrying a 3×3 ARRAY of 9 selectable dimples
(see §6); its clear 4×4 mm patch is used for the interferometer mode,
so BOTH modes share the substrate and its focus shift.

**Two models to build (Dave, 2026-07-22):**

1. **IFO** — two *linked* Rx (test arm, reference arm) propagated to
   the BS; complex amplitudes coherently mixed at the BS exit (per
   wavelength if finite band); the COMBINED beam then propagates
   through the common back-end (L2 → folds → detector).
2. **ZWFS** — single Rx (reference arm blocked) propagated to the
   internal focus, the Zernike λ/4 dimple applied there, then
   propagated to the detector pupil image.

Both need mmacos drivers that (a) ingest a perturbation spec and apply
it CONSISTENTLY across the Rx set, (b) do the mixing (IFO) at the BS /
the mask (ZWFS) at the focus, (c) produce detector frames.

## 2. Architecture decisions (already made — do not relitigate)

- **Coherent mixing is user-space MATLAB.**  `macos.compose` is
  incoherent-only (engine CADD unimplemented).  Mixing = complex array
  arithmetic on `macos.complex_field` outputs.
- **Field injection mechanism ("mix at BS, propagate combined"):**
  the common-path host is the **test-arm Rx itself**.  Sequence:
  1. Load ref Rx, apply perturbs (incl. PZT step), trace,
     `E_r = macos.complex_field(MIX)` where MIX = the BS-exit plane
     element.
  2. Load test Rx, apply perturbs, trace,
     `E_t = macos.complex_field(MIX)`.
  3. `E_mix = a_t*E_t + a_r*E_r` (a_t, a_r = scalar BS amplitude
     factors from params; 50/50 → both |a|² = 0.25 in intensity).
  4. `mask = E_mix ./ E_t`, with `mask(abs(E_t) < tol) = 0` (guard the
     outside-aperture zeros).  `macos.apodize_complex(MIX, mask)` —
     the field at MIX is now exactly E_mix.
  5. `E_det = macos.complex_field(DET, 'reset_trace', false)` — the
     engine propagates the combined field through the common back-end.
     (`reset_trace=false` is MANDATORY here — the default MODIFY wipes
     the apodization; see mmacos/CLAUDE.md + apodize_complex.m help.)
  This is exact because apodize multiplies the live field:
  E_t·(E_mix/E_t) = E_mix.  It is the same apodize-then-NF-propagate
  pattern validated against PROPER in pymacos Phase 6b.
- **Both Rx share an identical common back-end** (Source leg, L1, BS,
  and everything after the BS exit: MIX plane, FOCUS plane, L2, folds,
  detector are the SAME elements with the same geometry in both files).
  Only the arm differs (DM vs RefMirr+PZT).
- **The internal-focus plane element (FOCUS) lives in BOTH Rx** from
  day one.  The IFO ignores it (no mask); the ZWFS apodizes there.
  So the ZWFS model is the test Rx + a mask — no third prescription.
- **PZT phase shifting is a real perturbation**: `macos.perturb(RefMirr,
  'translation', [0 0 Tz])` (SI metres) along the mirror normal.
  Normal-incidence double pass: phase = 4π·Tz/λ, so Tz = λ/8 per π/2
  step.  Keep an `analytic_pzt` option (multiply E_r by exp(iδ)) as a
  cross-check knob — the two must agree to numerical precision for a
  flat, aligned ref mirror.
- **Finite band** = outer wavelength loop: `macos.set_src_wvl(lam)`,
  redo steps 1–5 per λ, sum detector INTENSITIES incoherently.  HeNe
  default is a single λ; the loop is the generalization hook (temporal
  coherence envelope falls out of the λ sum automatically).
- **Perturbation consistency across linked Rx**: a canonical
  element-name table (Source, L1a, L1b, BS, DM, RefMirr, MIX, FOCUS,
  L2a, L2b, Fold1, Fold2, DET) with a per-Rx name→iElt map.  The apply
  function takes (spec, map-for-currently-loaded-Rx) and is called
  after EVERY `load_rx` — there is only one Fortran session, and
  `load_rx` resets all state, so each frame is
  load → apply → trace → query.  A shared element perturbed in one Rx
  MUST be perturbed identically in the other (same SI values, same
  frame); the elt-map indirection is what guarantees it.

## 3. File layout to create

All under `~/dev/MACOS_resources/mmacos/examples/vsg/`:

```
examples/vsg/
  README.md              what/why/how-to-run
  vsg2_params.m          SINGLE source of truth: every hardware number.
                         Placeholders carry a TODO(Dave) comment.
  Rx/vsg2_test.in        Source→L1→BS→DM→BS→[MIX]→[FOCUS]→L2→folds→DET
  Rx/vsg2_ref.in         Source→L1→BS→RefMirr→BS→[MIX]→[FOCUS]→L2→folds→DET
  vsg2_elt_map.m         canonical name → iElt for each Rx (+ self-check)
  vsg2_apply_perturbs.m  ingest spec struct, apply to loaded Rx
  vsg2_arm_field.m       load+apply+trace+complex_field(MIX) for one arm
  vsg2_mix_detect.m      steps 3–5 of §2 (mix, inject, propagate, camera)
  vsg2_ifo_frame.m       one full IFO detector frame (λ loop inside)
  vsg2_psi.m             PSI acquisition loop: drive PZT, collect frames
  vsg2_psi_extract.m     pure-math N-frame phase extraction (no mex);
                         algorithms: 4step | hariharan5 | degroot7 |
                         zygo13 (default) — see §5.2
  vsg2_unwrap2d.m        Itoh row/column 2-D unwrap (MVP quality)
  vsg2_ifo_demo.m        end-to-end demo driver + figures
  vsg2_zwfs_mask.m       build dimple mask on the FOCUS grid
  vsg2_zwfs_frame.m      trace→apodize(FOCUS)→detector intensity
  vsg2_zwfs_calibrate.m  poke-built interaction matrix G
  vsg2_zwfs_reconstruct.m  pinv(G) + analytic small-phase estimator
  vsg2_zwfs_demo.m       end-to-end demo driver + figures
  figures/               (gitignored) PNG output
tests/tVsg2Psi.m         gate V0 (pure MATLAB, no mex — any suite)
tests/tVsg2Ifo.m         validation gates V1–V4 as asserts
tests/tVsg2Zwfs.m        validation gates Z1, Z3 as asserts
```

Conventions: `arguments` blocks per mmacos/CLAUDE.md; SI in/out at the
driver surface; tolerances from `tests/private/tolerances.m` where
applicable; drivers runnable headless (`matlab -batch`, figures
invisible→PNG, end with `exit(0)` — batch-hang gotcha).  Use
`mmacos_setup.m` for paths.  Work on `sls-dev`.  This is DRIVER + Rx
work: no mex/engine edits expected.  If an api gap appears, use the
Path A codegen route from mmacos/CLAUDE.md, and stop and ask before
any engine (Fortran) change.

## 4. Part 0 — shared foundation (do first)

### Stage 0.1 — params + Rx authoring (idealized Stage A)
- `vsg2_params.m`: wavelength(s), BaseUnits (mm), model_size (256
  default; 512 for fidelity runs), DM: **selectable format** —
  default **48×48 AOX, 1.0 mm pitch** (Xinetics-style square actuator
  grid; Dave 2026-07-23) with a **96×96 BMC, 0.4 mm pitch** option
  (both on the same optics per the deck).  Stroke ±200 nm surface
  (AOX).  **Illuminated pupil D = N_act·pitch·√2** (beam fills the
  circle circumscribing the square grid — deck slides 5/16): AOX →
  D ≈ 68 mm, BMC (96·0.4·√2) → D ≈ 54 mm.  BS amplitude factors (see
  BS design below), PZT step table, distances (1175, 317, 250
  nominal; see arm-distance procedure below), camera (6.5 µm,
  2560×2160; **binning 3×3**, DM-plane scale ≈ 13.4 binned px/mm),
  **F/# at focus = 250/D ≈ 3.57–3.7 for AOX**, ε leakage default 0.
  ZWFS dimple/mask params live in their own block (see §6.1 + the
  mask-array table in §6 — a 9-spot selectable array, etched depth
  346.2 nm = π/2, NOT a single idealized 1.06·λ·F# plane).
- **BS design (selected — Dave delegated the choice):** wedged
  plate beamsplitter, fused silica, ~10 mm thick, 30 arcmin wedge,
  dielectric 50/50 coating at 632.8 nm / 45°, AR-coated second
  surface.  Rationale: in a Twyman-Green BOTH detected arms carry
  exactly one reflection and one transmission at the BS (test:
  T then R; ref: R then T), so arm amplitudes are balanced at the
  camera for ANY split ratio — 50/50 just maximizes throughput.
  The wedge steers second-surface ghosts off the detector; the AR
  coat suppresses them.  Polarization: SM-fiber output is polarized —
  orient it s-polarized at the BS (dielectric R/T at 45° differ s vs
  p; fixing s keeps the split ratio deterministic and the fringe
  contrast maximal).  Stage A models the BS as a plane with scalar
  a_t = a_r = 0.5 amplitude factors; Stage B adds the real substrate
  (two refracting surfaces + wedge) per arm and the ghost paths.
- **Stage A geometry: UNFOLDED, on-axis.**  No fold mirrors, BS as a
  zero-thickness plane (its split enters only through a_t/a_r), L1/L2
  as single ideal surfaces or simple doublets — whatever gets the
  conjugates right fastest.  The arm path lengths (BS→DM, BS→RefMirr)
  must be settable independently in params (arm OPD matters for finite
  band).  Elements:
  - DM: flat Reflector carrying `Surface=FreeForm` (or GridData) with
    an nGridMat grid, zero-initialized; `GridSrfdx` = aperture/N.
    Runtime figure via `macos.set_elt_grid`.  **Transpose gotcha:**
    pymacos's elt_grid transposes, apodize does not — VERIFY the
    mmacos `set_elt_grid` convention empirically with an asymmetric
    test figure (e.g. a single off-center poke) before trusting signs
    and orientation; record the finding in the README.
  - RefMirr: flat Reflector (perturbable: PZT Tz + tip/tilt).
  - MIX: a Reference plane at the BS exit.  FOCUS: a Reference plane
    at 250 mm past L2.  DET: at the DM conjugate (317 mm past L2).
- **Arm distances (Dave 2026-07-23): approximate from the drawing,
  refine for crisp pupil imaging.**  Procedure: (a) pixel-scale the
  layout figure (`VSG2_layout.png` in this directory, extracted from
  the pptx) using the annotated ~1175 mm DM→L2
  path as the ruler to estimate BS→DM and BS→RefMirr; the 1175 mm
  total includes the fold legs — the unfolded Stage A model uses
  path-length totals, so only the split between BS→arm and BS→L2
  needs estimating.  (b) Then REFINE the model's DET position by the
  imaging criterion, not the thin-lens number, using
  **`macos.pupil_quality`** (Dave's pointer: engine XPS = FEX
  extended from chief-ray-only to the whole ray grid; fits the
  per-ray pupil-crossing cloud as a low-order Zernike surface).
  Recipe: set the stop at the DM (it IS the system pupil), give the
  test Rx a Return/Reference element near the expected detector
  location, then `pq = macos.pupil_quality(DET_ret)`:
    - `pq.vertex` = where the DM image geometrically forms → place
      DET there (this is the "crisp pupil imaging" refinement of the
      ~317 mm nominal; with the real doublets the principal-plane
      shift lands here automatically).
    - `pq.defocus` / `pq.astig` / `pq.sag_rms,sag_pv` = pupil-image
      surface curvature and astigmatism = the irreducible geometric
      blur across a FLAT detector — record these as the first
      entries of the pupil-imaging error budget.
    - `pq.uv` (flat entrance coords) vs the crossing cloud = the
      DM→camera geometric DISTORTION map — feed this to the Phase 1
      error budget (mapping error between DM actuator coordinates
      and camera pixels) instead of deriving it separately.
  Prereqs per the veneer help: a stop must be set; EP_ELT is a
  Return/Reference element; the routine restores the nominal trace
  before returning.  Record solved-vs-nominal distances in the
  README.  Arm-length mismatch is not critical for monochromatic
  stabilized HeNe (long coherence) but matters for the finite-band
  generalization — keep both arm lengths as independent params.
- **Final optimization — lens design for pupil image quality (Dave
  2026-07-23):** if, after the DET-position refinement, the
  `pupil_quality` metrics at the camera (defocus/astig/sag residual,
  distortion) still exceed the gauge error budget with the catalog
  doublets, run a LENS-DESIGN pass to assure excellent pupil imaging:
  vary L2 (spacing/orientation first — cheapest; then substitute a
  custom doublet or add a field-flattening element near DET) with
  the pupil_quality metrics as the merit function.  Use the
  `macos.design` vary/optimize machinery where it fits, or a direct
  fminsearch over the few lens DOFs with pq.sag_rms + distortion as
  the cost.  Consult `optical_design/TELESCOPE_DESIGN_REFERENCE.md` /
  `OPTICAL_DESIGN_AGENT_GUIDE.md` conventions before touching optical
  math (project standing rule).  This is the LAST step of Stage B —
  only invoked if the catalog lenses fall short; record the
  before/after pupil_quality numbers either way.
  - Reflector OPD = 2× surface displacement — keep the factor-of-2
    bookkeeping in ONE place (the PSI scale back to surface).
- **Stage B (later, after IFO V-gates pass on Stage A):** real Newport
  doublet prescriptions (WebFetch the PAC097AR.14 / PAC095AR.14 radii/
  glasses/thicknesses from newport.com — do NOT invent values; if
  unavailable, design an equivalent achromat and mark it as such),
  BS substrate (thickness/wedge/orientation per arm — **TODO(Dave)**),
  folds, camera pixelation + noise.
- Done when: both Rx load; `macos.trace()` runs; a spot/first-order
  check confirms DET is at the DM conjugate (perturb DM with a known
  tilt → image displacement matches m≈0.27), and FOCUS is at the
  source-image focus (spot size minimal there).

### Stage 0.2 — elt map + consistency assertions
- `vsg2_elt_map('test')` / `('ref')` return struct name→iElt.
  Self-check function: load each Rx, and for every SHARED name assert
  `get_elt_vpt/psi/rpt` identical across the two Rx (tolerance 1e-12),
  and `macos.dx_at(MIX)` identical after tracing both.  The mixing is
  meaningless if the two grids differ — this assertion is a hard gate.
- Done when: self-check passes and is called at the top of every demo
  driver.

### Stage 0.3 — perturbation ingestion
- Spec = struct array, e.g.
  `p(k) = struct('elt','DM', 'kind','rigid', 'dof','Rx', 'value',1e-6)`
  (SI rad/m; via `macos.perturb`, local frame per the design-layer
  convention), or `('elt','DM','kind','figure','grid',NxN_metres)`
  (via set_elt_grid), or `('elt','DM','kind','zern','modes',...,
  'coefs',...)`.  `vsg2_apply_perturbs(spec, map)` applies to the
  loaded Rx, skipping entries whose elt isn't in this Rx's map (the
  DM entry is a no-op in the ref Rx) — but SHARED elts always apply.
- Remember the trace-state rules: apply BEFORE `trace`; `modify()` is
  implied by load_rx; coefficient writes need `macos.modify()` before
  trace (mirror the pymacos gotcha — verify whether the mmacos
  veneers already do it; the channels layer does).
- Done when: a DM Z4 poke applied through the spec changes
  `trace().rmsWFE` in the test Rx and not in the ref Rx.

## 5. Part 1 — IFO model plan

### 5.1 Frame pipeline (implements §2 exactly)
- `vsg2_arm_field(arm, spec, lam)` → E at MIX (load, apply, set λ,
  trace, complex_field).  ~2 load_rx per frame is acceptable cost.
- `vsg2_mix_detect(E_t, E_r, spec, lam)` → detector intensity:
  reload test Rx (it is the common-path host), apply spec, trace,
  E_t_check = complex_field(MIX) (assert equal to the passed E_t —
  cheap determinism check), build mask, apodize at MIX,
  `complex_field(DET,'reset_trace',false)`, return |E|².
- `vsg2_ifo_frame(spec, pzt_step)` wraps the λ loop and the PZT entry
  in the spec; returns the (native-grid) detector frame.  Camera-pixel
  resampling (native grid → 6.5 µm pixels at 0.27×) is a Stage B
  option flag, off by default.

### 5.2 PSI measurement + reconstruction
- `vsg2_psi(spec, 'algorithm', name)`: implement a GENERIC N-frame
  engine — frames I_j at phase steps j·α, phase =
  atan2(Σ s_j·I_j, Σ c_j·I_j) with per-algorithm coefficient tables:
  - `'4step'`: α=π/2, φ = atan2(I4−I2, I1−I3) (bring-up/debug).
  - `'hariharan5'`: α=π/2, φ = atan2(2(I2−I4), 2I3−I1−I5).
  - `'degroot7'`: α=π/2, φ = atan2(−I1+7I3−7I5+I7,
    −4I2+8I4−4I6) (de Groot 1995, Appl. Opt. 34, 4723; transcribed
    from Griesmann's NIST PSA-toolbox paper Eq. 5, 0-indexed there).
  - **`'zygo13'` (DEFAULT — the real bench algorithm, Dave):**
    α=π/4 (45°/frame), 13 frames g0..g12, from CLAIM 11 of de Groot's
    Zygo patent US 6,359,692 B1 (MetroPro implements this as the
    "High Res" phase mode per col. 12; the 13-frame algorithm
    originates in US 5,473,434):
      tan θ = [−3(g0−g12) − 4(g1−g11) + 12(g3−g9) + 21(g4−g8)
               + 16(g5−g7)]
            / [−4(g1+g11) − 12(g2+g3+g9+g10) + 16(g5+g7) + 24·g6]
    Implementation notes: numerator coefficients are ANTIsymmetric
    and denominator coefficients SYMMETRIC about g6, and both sum to
    zero (DC-insensitive) — assert these three properties on the
    coefficient tables at init as a transcription guard.  Patent-
    documented properties to reproduce in V5: insensitive to 2nd/3rd/
    4th-harmonic fringe content and highly resistant to intensity
    noise.
- **PZT step size follows the algorithm**: phase per frame δ =
  4π·Tz/λ (normal-incidence double pass), so α=π/4 → Tz = λ/16
  ≈ 39.6 nm/frame (total 13-frame travel ≈ 475 nm); α=π/2 → λ/8.
  The step table lives in params and is derived from α — never
  hardcode nm values in the drivers.
- **The PSI engine is a driver deliverable in its own right**
  (Dave): `vsg2_psi.m` (frame acquisition loop over the optical
  model) + `vsg2_psi_extract.m` (the pure-math N-frame phase
  extraction — coefficient tables, atan2, no mex dependency).  Keep
  the extraction separable so it can be applied to REAL bench frames
  later, not only simulated ones: it takes a 3-D stack of frames +
  an algorithm name, returns wrapped phase.  Unit-test it standalone
  (gate V0 below) before any optical-model use.
- `vsg2_unwrap2d`: Itoh sequential row/column unwrap — adequate for
  smooth DM figures; note the upgrade path (quality-guided) in the
  header comment, don't build it now.
- Surface = φ·λ/(4π) (double pass), minus a stored NOMINAL
  reconstruction (calibration frame with spec = nominal) so static
  model artifacts subtract out the way a real gauge subtracts its
  reference measurement.
- Compare to the injected DM grid mapped through the DM→camera
  magnification (and the empirically-determined grid orientation from
  Stage 0.1).

### 5.3 Validation gates (each becomes an assert in tVsg2Ifo)
- **V0 PSI-engine unit gate (pure MATLAB, no mex — its own test
  class `tVsg2Psi`, runs in any suite):** synthesize I_j = A +
  B·cos(φ_true + j·α) pixel fields with known φ_true; every
  algorithm must recover φ_true to numerical precision.  Then the
  error-compensation properties: (a) phase-step miscalibration
  (α → 1.05·α): zygo13/hariharan5/degroot7 error ≪ 4step error;
  (b) fringe harmonics (add 2nd/3rd/4th-harmonic terms to I_j):
  zygo13 error stays near floor while 4step degrades — this
  numerically reproduces the US 6,359,692 / US 5,473,434 claims and
  guards the coefficient transcription end-to-end.
- **V1 null:** nominal spec, δ=0 → fringe-free frame (intensity
  spatial rms / mean below tol); PSI output ≈ 0.
- **V2 tilt fringes:** ref-mirror tilt θ (e.g. 50 µrad) → straight
  fringes with spacing λ/(2θ) in DM-plane units; assert measured
  spatial frequency to ~1%.
- **V3 linearity (validates the injection trick):** mix-at-BS-then-
  propagate (the pipeline) vs propagate-each-arm-to-DET-then-add
  (query complex_field(DET) per arm, sum in MATLAB).  Because
  everything after MIX is linear, these must agree; assert relative
  RMS < 1e-10.  If this fails, the apodize/reset_trace plumbing is
  wrong — stop and fix before proceeding.
- **V4 known figure:** DM Z4 poke at λ/20 rms → PSI-reconstructed
  surface matches injected within a stated budget (set the number
  from the first passing run, then pin it — regression style, like
  tPerturbRoundtrip).  Also run a spatial-frequency sweep (sinusoids
  2..N_act/2 cycles/aperture) → gauge transfer-function curve figure.
- **V5 (analysis, not a test):** ±5% PZT step error → error map vs
  the analytic 4-step PSI error formula; first entry of the error
  budget.
- Register the new test class in `run_mmacos_tests.sh` SUITE_SIZE
  groups (one model_size per MATLAB invocation — heap-bug rule).

### 5.4 Deliverables
- `vsg2_ifo_demo.m` producing: layout sanity figure (view_rx),
  null + tilt-fringe frames, PSI reconstruction vs truth panel,
  transfer-function curve.  README section explaining the mixing
  architecture (§2) for the manual.

## 6. Part 2 — ZWFS model plan

### 6.1 Frame pipeline
- Test Rx only (ref arm blocked = absent).  `vsg2_zwfs_frame(spec)`:
  load test Rx, apply spec, trace, build mask on the FOCUS grid,
  `apodize_complex(FOCUS, mask)`,
  `I = |complex_field(DET,'reset_trace',false)|²`.
- `vsg2_zwfs_mask(lam, spot_id)`: trace nominal, `dxf =
  macos.dx_at(FOCUS)`; dimple DIAMETER = (λ/D factor)·λ·F# per the
  selected spot from the deck's mask-array table (below), so
  r0 = 0.5·factor·λ·F#; mask = ones except exp(i·phase) inside
  r ≤ r0 (grid centered per the engine's focal-grid convention —
  determine it empirically from the nominal PSF centroid, don't
  assume).  **Phase from etched depth, not assumed π/2:** the real
  mask is a physical etch of depth t in fused silica, phase =
  2π(n−1)t/λ.  Deck value t = 346.2 nm, n(632.8 nm) = 1.45702 →
  phase = π/2 by construction; keep `phase` derived from
  `(t, n, λ)` params so a finite-band λ sweep sees the correct
  CHROMATIC phase error (the etch is π/2 only at 632.8 nm).
  **Sampling gate:** require ≥ ~5 samples across the dimple
  (2·r0/dxf ≥ 5).  The smallest deck dimples are TINY (2.26–2.8 µm
  at F/3.6) — expect the native focal sampling at model 256 to be
  too coarse; follow the oversized-rays / windowing recipe from the
  Cycle-5 vortex work (see `macos.window`/`window_off` veneers and
  pymacos `tests/proper_compare/run_broadband_vortex.py`) before
  resorting to model 512.  (Slide 7 sanity check: dimple Fresnel
  distance Z_F = a²/λ = 0.007–0.073 mm ≪ Z = 67 mm to the pupil
  image → far-field propagation; the smallest dimple's diffraction
  covers the whole 17.9 mm pupil.)
- **Mask-array table (deck slide 6 — the REAL selectable spots).**
  A 3×3 array on one 25.4 mm FS substrate, spots on a 5 mm pitch,
  all etched to the SAME depth (346.2 nm → π/2 at HeNe):

  | Spot ID | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|---|---|---|
  | Dia (λ/D) | 3.0 | 2.0 | — | 2.0 | — | 1.0 | — | 1.22 | 1.06 |
  | Dia (µm) | 6.78 | 4.52 | — | 4.52 | — | 2.26 | — | 2.76 | 2.40 |

  "—" = no dimple (clear reference windows).  There is also a clear
  4×4 mm patch used for the interferometer mode.  `spot_id` selects
  the row; default to the classic ZWFS **1.06 λ/D (spot 9)** unless a
  driver overrides.  The substrate itself contributes **~1.7 nm rms
  spherical aberration + a 0.31 mm focus shift** (deck slide 5) —
  Stage A may ignore these; Stage B adds them as a bias term on the
  common back-end (they are common to BOTH modes, so they largely
  subtract in the IFO reference measurement but bias the ZWFS).
- Leakage case: reuse the IFO mixer with a_r → ε·a_r to model
  imperfect reference-arm blocking; this is how the blocker-extinction
  spec gets derived later.

### 6.2 Calibration + reconstruction
- `vsg2_zwfs_calibrate`: interaction matrix G by poking basis modes
  (Zernikes 4..15 first; per-actuator pokes once DM params are real)
  at small amplitude (λ/200) through the model: columns ΔI/Δa on the
  detector pupil support.  `vsg2_zwfs_reconstruct`: â = pinv(G)·ΔI.
- Also implement the analytic small-phase estimator (N'Diaye-style:
  I ≈ P² + 2b² + 2Pb(sin φ − cos φ), with the reference wave b
  computed in MATLAB as the pupil field spatially filtered by the
  dimple) — cross-check, and the fast on-orbit candidate.

### 6.3 Validation gates
- **Z1 identity:** mask = ones → detector frame equals the no-mask
  frame exactly (bit-level within numerical tol).  Validates the
  apodize plumbing at FOCUS.
- **Z2 flat-DM reference intensity:** nominal frame vs the analytic
  formula using the MATLAB-computed b — agree to a few % over the
  pupil interior.
- **Z3 linearity:** inject Z4/Z5/Z6 at λ/100 rms → reconstructed
  mode/amplitude correct to a few % (assert in tVsg2Zwfs).
- **Z4 dynamic range (analysis):** amplitude sweep → reconstruction
  error vs input rms; report the breakdown point (expect λ/8–λ/4
  scale).  This quantifies the IFO-vs-ZWFS operating envelope.
- **Z5 photon noise (analysis):** MC with Poisson noise at
  parameterized flux → σ_φ vs 1/√N_ph theory.

### 6.4 Deliverables
- `vsg2_zwfs_demo.m`: mask/PSF/pupil-frame montage (coro_walkthrough
  idiom), linearity curve, dynamic-range plot, noise curve.
- Follow-on (separate slice): ε-leakage sweep → reference-blocker
  extinction spec; IFO-vs-ZWFS crossover summary figure.

## 7. Known gotchas checklist (bind these — all from CLAUDE.mds)
- `reset_trace=false` on every post-apodize query, both models.
- Trace before any opd/intensity/complex_field/dx_at.
- One model_size per MATLAB process; split test invocations by size;
  update SUITE_SIZE* in run_mmacos_tests.sh for the new classes.
- `load_rx` strips `.in`; pass paths accordingly.
- End every `matlab -batch` script with `exit(0)`.
- GridFile paths resolve from CWD (if a GridFile is ever used in the
  Rx; runtime set_elt_grid avoids this).
- set_elt_grid orientation/transpose: verify empirically (Stage 0.1)
  before any sign-sensitive comparison.
- Do not hand-edit fixtures or widen tolerances to make a gate pass;
  diagnose instead (project standing rule).

## 8. Hardware data status (updated 2026-07-23, reconciled with the
VSG2 ZWFS deck "…Update -v2.pptx")
RESOLVED (Dave): DM = Xinetics/AOX format, square grid, 1.0 mm actuator
pitch, stroke ±200 nm.  **DM count RESOLVED by the deck: default
48×48 AOX (1.0 mm pitch), with a 96×96 BMC (0.4 mm pitch) alternate —
model as a selectable format.**  BS = our selection (wedged 50/50
plate, §4 Stage 0.1).  Arm distances = scale from drawing, refine via
`macos.pupil_quality` (§4).  PSI = Zygo 13-frame (§5.2, coefficients
verified from US 6,359,692 B1 claim 11).
**RESOLVED by the deck (were TODO(Dave)):**
- **Illuminated pupil D = N_act·pitch·√2** (circle circumscribes the
  square grid) → AOX ≈ 68 mm, F/# ≈ 3.57–3.7.  Do NOT use the square
  edge (48 mm) as the beam diameter.
- **ZWFS mask = a 9-spot selectable array** on a 1 mm FS substrate
  (Thorlabs W4101FT1, one-side AR), spots 1.0–3.0 λ/D, all etched
  346.2 nm (π/2 at HeNe).  See the §6 mask-array table.  Substrate
  bias ≈ 1.7 nm rms SA + 0.31 mm focus shift (common to both modes).
- **Camera binning 3×3**, DM-plane scale ≈ 13.4 binned px/mm; DM
  image ≈ 17.9–18.9 mm.
- **Ref blocker = Aerotech ATS-100 linear stage; mask on a Xeryon
  XYZ stage** (mode switching is remote/in-chamber — the ZWFS
  ref-blocking discussion in the review doc is realized this way).
STILL OPEN (TODO(Dave)): ref-flat figure quality; fiber NA (sets
beam-overfill vs the DM aperture — deck implies the beam fills the
circumscribing circle); camera flux/QE for photon-noise runs; BS
substrate thickness/wedge/compensator for Stage B.
Note the stroke number in context: ±200 nm surface = ±400 nm OPD
≈ ±0.63λ — full DM stroke EXCEEDS the ZWFS linear range (λ/8–λ/4
scale), which quantitatively motivates the IFO-for-absolute /
ZWFS-for-small-residual pairing measured in gate Z4.
