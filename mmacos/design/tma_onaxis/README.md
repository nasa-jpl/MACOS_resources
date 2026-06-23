# `tma_onaxis/` — on-axis three-mirror anastigmat (Korsch / JWST form)

Two ways to get an on-axis obscured TMA (M1 + convex SM + tertiary behind M1,
real intermediate focus, FSM fold, slightly off-axis detector):

## 1. `scale_j18.py` — scale the validated j18 design (robust, recommended)

A uniform length-scaling of the **j18mono** prescription (an early-JWST obscured
TMA, diffraction-limited with all the packaging features). Scaling preserves
every angle, conic constant and the f/# — only lengths scale — so the result is
a real, working j18-like TMA at any aperture.

```bash
cp <macos>/j18mono.in ./j18mono.in     # the reference seed (not committed)
python3 scale_j18.py 1000              # -> j18_scaled.in at D = 1 m
```

Verified: D=1 m → **0.0018 waves** at 2.3 µm (= j18's 0.012 × k), all 5 features,
proper central obscuration. Useful if not fully versatile — re-optimize the
conics for a different f/# if needed.

## 2. `tma_onaxis.m` — parametric from-scratch Korsch designer

`tma_layout` (scaled Korsch fixture) → field-optimize(radius+conic) →
`optimize_aspheres` → diffraction-limited. Knobs: aperture, primary f/#, system
f/#, M3-behind, intermediate-focus location.

**Status / scope.** This nails the *gentle* Korsch regime (diffraction-limited,
sane mirrors) and the new `tma_layout` Cassegrain solve places the intermediate
focus exactly where you ask (`int_focus_m`, before/after M1). But in the
aggressive j18 regime (fast primary, focus well before M1) the first-order M3
relay is hard to place from the n-flip Seidel convention, and the exit pupil is
near-telecentric — **EP control and a finite-pupil FSM fold are a FREEFORM
problem** (fix the radii for the layout, correct the wavefront with Zernike
departures that don't move the radii). That's the next driver. For a guaranteed
working design today, use `scale_j18.py`.
