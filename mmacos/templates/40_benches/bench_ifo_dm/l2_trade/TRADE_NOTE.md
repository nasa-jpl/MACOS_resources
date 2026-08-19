# L2 pupil-relay trade: detector-leg redesign for DM metrology

**Result: a field lens ~6 mm behind the FocalMask takes the physical-instrument
vs-truth residual from 6.76 nm to 0.97 nm rms at 50 nm checker pokes (corr
0.9984), meeting the ≤ 1 nm gate with all guards green.  Linear in poke:
0.097 nm at 5 nm pokes — 0.019 nm of error per nm of poke command (5.7 % of
the truth rms, down from 40 %).**

Executed per `PLAN_IFO_PUPIL_RELAY.md` on branch `ifo-l2`.  Metric harness:
`ifo_l2_metric.m` (gate 0 reproduced the phase-1 6.758 nm exactly).  All
numbers: `POKE_NM=50`, checker, `SEED=7`, model 256, detector-mode direct
differential fields (≡ PSI to 3e-5 pm).

## 1. What drives the baseline retrace (mechanism, `run_mechanism.m`)

The 6.76 nm is **linear slope-coupling of the detector leg** — see
`mech_figure.png`:

- Running the identical measurement at the **Recomb plane** (before L2) gives
  0.308 nm at magnification exactly 1.0000: the whole error is the tail.
  The det−rc difference map is 7.07 nm rms, concentrated at the pupil rim
  (outer/inner rms factor 2.54), mostly decorrelated from the truth map
  (gain component only κ = 0.13).
- Tilting the zero-poke DM rigidly by α and plane-fitting the differential
  phase isolates the tail's response to a uniform beam deflection: the
  non-tilt residual is **exactly linear in α** (exponent 1.00), at **0.146 nm
  rms per µrad**.  The checker's rms *beam* deflection is 42.7 µrad (2× the
  21.4 µrad rms surface slope); 0.146 × 42.7 = **6.2 nm ≈ the observed 6.8** —
  mechanism closed.  The recovered-tilt gain is 1.051 at DC vs ~1.13 at the
  checker frequency: the coupling is frequency-dependent, so it cannot be
  calibrated out with a global gain (and Dave's doctrine forbids fitting it).
- The chief-ray lever at the detector is 9.18 mm/rad of DM tilt — the
  thin-lens detector placement sits ~4.6 mm off the true DM conjugate.
  The angular mapping is otherwise perfectly linear.

Physically: a poked-DM slope deflects each beamlet; the deflected beamlet
sees the singlet relay's imaging aberration gradient (field curvature /
zonal error of a lens used at conjugates it was never corrected for) and
picks up a phase error proportional to the deflection — a classic retrace
error, largest at the rim where the aberration gradient is largest.

## 2. The trade (`run_l2_trade.m`)

Architectures were optimized on the cheap physics coefficient (`k_tilt`,
the tilt probe above, ~15 s/eval) with the detector held at the DM conjugate
by a lever-null trim, then scored with the full M1 metric:

| arch | M1 (nm rms) | k_tilt (nm @2 µrad) | corr | mag det→DM | nl distortion | guards |
|---|---|---|---|---|---|---|
| singlet (baseline) | 6.758 | 0.262 | 0.9351 | 0.810 | 0.09 % | PASS |
| singlet + conjugate trim | 4.471 | 0.257 | 0.9675 | 0.825 | 0.09 % | PASS |
| **field lens (winner)** | **0.969** | **0.051** | **0.9984** | 10.53 | 0.03 % | PASS |
| doublet L2 | 0.979 | 0.058 | 0.9984 | 0.818 | 0.04 % | PASS |

- **Conjugate trim alone** (`DET_TRIM` = −5.54 mm) removes the tilt-parallel
  (gain) part — M1 6.76 → 4.47 — but leaves the slope-coupling coefficient
  untouched, exactly as the mechanism predicts.  Free (no new part); worth
  taking regardless.
- **Field lens** (`'tail_arch','fieldlens'`): a small (Ø12 mm) plano-conic
  lens behind the mask — `FL_F` 25.02, `FL_Kc` −2.113, `D_MASK_FL` 6.28,
  `DET_TRIM` +1.085.  It re-images the pupil through a tiny, nearly
  aberration-free zone (the beam footprint near the focus is sub-mm),
  collapsing the slope-coupling coefficient 5× and M1 7×.  The optimizer
  rides the short-focal-length bound (f = 25, a buildable catalog cap);
  shorter still helps — there is headroom on this curve.
- **Doublet** (`'tail_arch','doublet'`): L2 split into two f = 500 plano
  singlets 25 mm apart, conics `L2A_Kc` −3.575 / `L2B_Kc` +2.329,
  `MASK_TRIM` +1.615, `DET_TRIM` +2.971.  **Statistically tied with the
  field lens** (0.979 vs 0.969 on one poke realization) while keeping the
  baseline pupil-image scale and format.  Mask spot 0.89 µm — inside the
  1 µm guard but with little margin.  Stage-1 lesson baked into the runner:
  the mask must be re-seated at the *true* (thick-lens) focus via
  `MASK_TRIM`, or the spot floor is pure defocus that conics cannot touch
  (and a zero-seeded `fminsearch` parameter is frozen by its 0.00025
  default simplex — pre-scan for a nonzero seed).

## 3. Costs and the choice

- **Field lens** (wired as the example default — the numeric winner): one
  added Ø12 mm element, f ≈ 25 mm (short but catalog-class); mask–lens–
  detector all within ~30 mm — a compact, rigid subassembly.  The pupil
  image demagnifies ~10.5× det→DM (≈ 4.3 mm across vs 55 mm): a real camera
  needs its pixels on a smaller format, though the checker period still
  maps to ~0.66 mm there — easily resolved.  M2 nonlinear distortion
  improves (0.09 % → 0.03 % of beam).
- **Doublet**: same performance, two elements instead of one added, keeps
  the baseline detector format and layout envelope — the better choice if
  the camera or packaging argues against the field lens.  Both are wired
  in `twyman_green` and the example; switching is one parameter.
- Alignment (both): the detector-conjugate condition is set by `DET_TRIM`
  via the DM-tilt lever null — the same test works on the real bench (tilt
  the DM, null the image walk).

## 4. Polarization footnote (for pol-core)

The all-reflective OAP-pair relay (C3) was not evaluated: C1 met the gate
first.  If it is ever revisited, note that two OAPs at working AOIs buy
polarization aberration (s/p splitting per reflection) that the refractive
tails avoid at near-normal incidence — evaluate under the pol-core Jones
machinery, not here.

## 5. Reproduce

```bash
matlab -batch "run('run_gate0.m'); exit(0)"      # harness vs phase-1 numbers
matlab -batch "run('run_mechanism.m'); exit(0)"  # mechanism figure
matlab -batch "run('run_l2_trade.m'); exit(0)"   # full trade (~40 min)
```
