# tg_psi_dm96 — the optimized-plate gauge for a Xinetics 96×96 DM

The shallow-plate (option-3) polarization-PSI Twyman–Green, laid out
for a 96×96-actuator, 1.0 mm-pitch DM (96 mm beam — the aperture class
where a PBS cube goes custom).  `tg96.m` is the whole campaign: Stage A
clearance solve (margins printed as numbers), Stage A2 sampling budget
(asserted — Dave 2026-09-03: sampling is part of the design), Stage B
scaled build, Stage C battery, Stage D actuator-lattice transfer curve.

## Ultimate target (Dave 2026-09-04)

Measurement error ~ 1 pm.  The run-10 map-space numbers (single-poke
0.049 nm = 49 pm; differential 0.021 nm = 21 pm) are 20-50x above it;
the pm road runs through the differential protocol + actuator-space
scoring (Stage E').

## Findings so far (reports are the record)

- `tg96_report_ng63.txt` — 63 px across the pupil: detector-blind at
  actuator scale (single 1 mm actuator reads 73/150 nm).  Detector
  Nyquist 31.5 cyc/pupil vs the DM's 48.  MODEL 1024 is load-bearing:
  mGridMat is NOT the model size (512 caps at 256) and the `nGridMat=`
  parse is unguarded — see `macos/BRIEF_gridmat_guard.md`.
- `tg96_report_ng385_chkreg.txt` — 385 px (1 Mpix frame): single
  actuator recovers 144.2/150 ✓, but the modal curve is INVARIANT to
  the 6× detector change → not aliasing.  Magnitudes follow
  cos(πp/96) exactly = a ONE-ACTUATOR registration offset: the
  checkerboard registration target is blind to ±1-actuator shifts by
  symmetry (shifted one actuator it equals its own negative — which
  also supplies the curve's sign flip).  Register on a single
  actuator, never a symmetric pattern (the recorded parity trap).
- `tg96_report.txt` — current: registration on the single-actuator
  map (rerun in flight at this writing).
- Standing systematic: ~9.1 nm rms null, invariant to sampling and
  model size — the uniformly-scaled detector tail is not
  diffraction-tuned at the new scale (common-mode; the differential
  protocol removes it).  Fix = re-run the l2_trade tail optimization
  at this scale.

## The instrument, measured (run 9, 2026-09-04 — tg96_report.txt)

- **Transfer curve at 1 Mpix (385 px across the pupil): the gauge
  RESOLVES 1 mm actuators.**  Gain 1.008 at 0.7 cyc/pupil rolling
  mildly and monotonically to 0.956 at 33.9, ~0.92 at the DM Nyquist
  (48), 0.835 at the 96×96 checkerboard (67.9).  Cross-talk ≤ 1.4%.
  All positive, no folds — a calibratable MTF.
- Single 1 mm actuator at 150 nm recovers 144.2 nm; unaligned piston
  scale +0.115%; departure from orthogonality 0.143° at BS 7°.
- Gauge floor at command scale: held-out random command reads 8.10 nm
  rms residual on a 30.6 nm input, decorrelated from the command
  (corr −0.09) — same family as the 9.1 nm static null.  OWNER: the
  uniformly-scaled (never re-tuned) detector tail; re-running the
  l2_trade tail optimization at this scale is the known knob before
  closure numbers are quotable at v1 class (0.183 nm).
  Checkerboard closure 0.79 corr / 4.31 nm on 6.41 reflects that floor
  plus the 0.835 gain at the pattern's own frequency.

## Run 10 (tuned tail + Stage E, 2026-09-04) — the trade surfaces

- **Tail retune VALIDATED at full scale: null 9.11 → 0.1345 nm rms**
  (tg96_tail.m optimum, found at reduced res, transfers to 4 digits).
- **Stage E, the differential metric (Dave): deviations are measured
  BASE-INDEPENDENTLY.**  A 10 nm single-actuator deviation reads to
  0.021 nm rms (corr 0.991) about the flat and 0.024 nm about a 30 nm
  working state — the common systematic cancels as designed.  A 10 nm
  rms random deviation reads 3.67 nm rms (37%) — identically about
  both bases — so the differential error is INSTRUMENT TRANSFER, not
  systematic coupling.
- **The trade:** the null-only tail objective bought its 68× by moving
  the imaging conjugate — pupil MTF fell (checkerboard gain 0.835 →
  0.504) and mapping nonlinearity rose 0.0022 → 0.136 mm (~1.3
  actuators of warp).  Closure corr 0.54 and held-out 11.9 nm are that
  MTF/distortion, not gauge phase error.  NEXT tail iteration needs a
  JOINT objective (null + transfer + distortion); and Stage E scoring
  should apply the measured transfer before declaring residuals.
- ZWFS framing: the differential floor is now instrument-MTF-limited —
  the quantity a ZWFS arm competes against on the same truth.
- **Scoring ruling (Dave 2026-09-04): score DM-state recovery in
  ACTUATOR space** — fit the DM actuation model to the measurement
  (or to the recovered WF), score actuator CHANGES.  The run-10
  battery/differential numbers above are MAP-space and stand as the
  map-space record; **Stage E′ (actuator-space rescore) is queued**
  and supersedes the "apply measured transfer first" plan — the
  influence-function fit absorbs instrument roll-off inside the DM
  band.  The ZWFS campaign (40_benches/zwfs_dm96) scores this way
  from the start.

## Registration doctrine (earned across runs 3–9; the failure series
   is preserved as tg96_report_*_fail.txt)

The instrument-to-DM mapping has FOUR distinct degree-of-freedom
classes, each needing its own calibrated observable — none recoverable
from a symmetric target, none from a search:
1. **Scale + rotation**: from the ray-measured affine (0% anamorphism,
   2.2 µm nonlinearity here).
2. **Translation**: from ONE CENTER poke, exactly, by blob-centroid
   match (the center is parity-invariant — right for translation,
   useless for parity).
3. **Array parity** (8-fold flip/transpose group, deck-dependent):
   from ONE OFF-CENTER poke by direct overlap — |corr| 0.967 vs 0.003
   runner-up here (parity 5 = transpose).
4. **Measurement sign** (the four-step h = ±ψλ/4π convention,
   deck-dependent): the SIGN of that same overlap.
Anti-patterns, each measured here: a checkerboard registration is
degenerate under the whole parity group AND ripples at every integer
actuator (any refiner parks arbitrary offsets — 2.25 actuators in run
6); a delta target gives a correlation refiner no gradient (run 4
diverged to ±1e5 gains); selection must be on |corr| or a
deck-negative sign hides the right parity (run 7).

## WF-estimate figures + the illuminated-fill catch (2026-09-04)

`tg96_wf_figs.m` (deck figures, cases matched to zwfs_dm96): traced-rig
render + applied/sensed/error triptychs.  Center actuator at 20 nm:
gain 0.984, 0.049 nm rms.  Defocus at 8 nm amplitude: gain 1.024,
0.086 nm rms.  Display doctrine earned there: apply the deck-measured
meas-sign (−1 here); define radial patterns on the ILLUMINATED radius
(~38 mm — the source cone fills 74% of the 51.4 mm aperture), not the
aperture.  **Sampling-budget amendment: A2's "px across pupil" counts
GRID px; the illuminated pupil is 74% of that**, so real margins are
~0.74× the printed ones (385 px → 143-illuminated-px classes at
NGRID 193 configs fall UNDER the 2× rule).  Count illuminated px.

## Stage E' (tg96_eprime.m, 2026-09-04): the differential benchmark in
   actuator currency

Scoring-ruling option (b): fit the TRUE influence kernel to the
measured maps (Tikhonov lattice deconvolution); the residual is the
instrument error in actuator commands.  Registration re-derived
(parity 1, sign -1 in this script's basis; 0.934 vs 0.0002).  Rows
(differenced measurements, scored over 3228 lit actuators):

  base           deviation            gain    resid     raw
  flat           single act 10 nm    0.860    40 pm    47 pm
  random 30 nm   single act 10 nm    0.869    38 pm    45 pm
  flat           random 10 nm rms    0.769   3567 pm  4234 pm
  random 30 nm   random 10 nm rms    0.769   3570 pm  4235 pm

Base-independence holds in the new currency.  The random-row deficit
(gain 0.77) is the instrument's high-frequency roll-off carried into
actuator space — by design under option (b).  THE HEAD-TO-HEAD
CURRENCY: IFO differential single-actuator = ~40 pm; the 1 pm target
is 40x below it.

## Sensitivity (tg96_sens.m, 2026-09-04): linear to 0.1 pm

Differential single/grid pokes on flat and 30 nm base, 10 nm → 0.1 pm:
gain constant (0.853 single / 0.783 grid, base-independent), floor
proportional to amplitude, SNR flat (207 single / 19.6 grid).  No
additive floor in the noiseless model; numerics whisper ~0.3% gain
jitter at 0.1 pm.  With the constant gain calibrated, absolute
accuracy ~0.1–0.3% of the change — pm-class on nm-class changes.

## Next configurations (Dave, 2026-09-03)

1. **All-reflective: replace the lenses with OAPs.**  Bench builder
   OAP machinery exists (`Bench` OAP + Offner work).  Removes the
   cube/plate transmitted-glass-path and homogeneity rows from the
   cost budget entirely; buys achromatic legs and no ghost surfaces;
   the price is OAP alignment sensitivity — measurable with this
   battery unchanged.
2. **Zernike wavefront sensor (ZWFS).**  On the real VSG2's own
   roadmap (VSG2_Info.pptx; `vsg_wip/vsg2_params.m` sketches the
   mask: transmissive etched substrate, 9-spot array).  A phase
   dimple at focus — the CTB focal-plane phase-mask machinery is the
   starting point.  Different sensing modality: compare against PSI
   on the same DM truth, same battery metrics.
