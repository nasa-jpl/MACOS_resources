# tg_psi_dm96 — the optimized-plate gauge for a Xinetics 96×96 DM

The shallow-plate (option-3) polarization-PSI Twyman–Green, laid out
for a 96×96-actuator, 1.0 mm-pitch DM (96 mm beam — the aperture class
where a PBS cube goes custom).  `tg96.m` is the whole campaign: Stage A
clearance solve (margins printed as numbers), Stage A2 sampling budget
(asserted — Dave 2026-09-03: sampling is part of the design), Stage B
scaled build, Stage C battery, Stage D actuator-lattice transfer curve.

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
