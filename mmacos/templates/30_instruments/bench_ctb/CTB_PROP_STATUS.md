# CTB diffraction layer — work-in-progress status (hand-off)

_Updated 2026-08-25. Latest work at "SESSION 11" (binned masks) below; older kept._

## SESSION 11 (2026-08-25) — every mask generated at 8× and binned; the vortex core was the floor

Dave's directive: gray-scale the occulting/apodizing surfaces — generate
at sub-pixel resolution, bin to the model grid.  Amplitude masks already
were (ctb_mask_disk/softcircle, K=8 area-weighted edges — verbatim
"generate high and bin" for a binary shape).  The change is the PHASE
masks: new shared `ctb_mask_vortex` (8×-binned exp(i·m·θ), the 64
sub-pixel phasor shifts accumulated at model resolution so no 8N grid is
built; K=1 = legacy for A/B) now feeds all four former inline copies
(ctb_vortex, ctb_vortex_matched, ctb_mask_compare, ctb_phase_masks), and
`ctb_mask_phase` composes gray zone edges from the supersampled disks
(V = 1 + (e^{iφ}−1)·D(r) per zone).

**Why it matters — the vortex-floor diagnosis (ideal-pupil probe, no
bench):** the directly-sampled vortex floors at 3.0e-7 REGARDLESS of the
bench — the mis-phased pixels of the undersampled singularity sit on the
stellar Airy peak and scatter ~0.2% of the starlight inside the Lyot.
Complex-binning cancels those phasors (|V|=0 at the core pixel, smooth
~1-px taper): ideal-pupil floor 3.0e-9 (8×; 4× gives 3.4e-9,
converged).  An explicit 1 λ/D opaque core dot is WORSE (9.8e-9 — its
own edge diffracts).  Charge 2 is the counterexample (excision/binning
medicine is for even charge ≥ 4).

**Bench head-to-head regenerated (ctb_mask_compare, N=1024, annulus
3–15 λ/D):** APLC 2.11e-10 @27% (unchanged) · **vortex charge-6/Lyot-0.90
2.9e-7 → 1.41e-8 (21×) @81%** — second-deepest · band-limited 2.72e-8
@36% (unchanged) · hard 2.47e-7 @25% (unchanged) · Roddier 3.2e-6 →
2.40e-6 · dual-zone 6.8e-6.  The amplitude rows moving only in the last
digit is the control: they were already binned.

**The Lyot trade is now real (ctb_vortex_matched sweep):** with the core
artifact gone, closing the stop buys genuine depth — charge 4: 6.6e-11 @
frac 0.50 (25% T, APLC-class), 1.9e-9 @ 0.80 (64% T — Pareto-dominates
band-limited), 8.0e-9 @ 0.90 (81% T); charge 6: 3.0e-10 / 6.4e-9 /
1.6e-8 at the same fracs.  Flux inside the stop 0.0–0.1% to frac 0.90 —
the analytic "all light outside" property, visible at last.  The old
"the Lyot can stay open for free" reading was an artifact of the core
floor; the driver header records the inversion.  Vortex floor is now set
by the bench residual (ideal-pupil binned floor 3.0e-9 vs bench 1.4e-8).

Figures regenerated: ctb_mask_compare.{png,mat}, ctb_vortex.png
(hard-vs-vortex now 1670× at Lyot 0.50), ctb_vortex_matched.png,
ctb_phase_masks*.png.  Deck slide 5 updated (table re-sorted, vortex
bullet = the sampling finding).  README "mask sampling rule" paragraph
added to the mask-block section.

**Lyot sweep vs the fixed designs (ctb_vortex_lyot_sweep, Dave's ask):**
dense fraction grid 0.50–0.99 × charges 4/6 against the committed
APLC/BLC/hard points (same grid/annulus/normalization).  Charge 4
under-runs every fixed design at every throughput: **0.60 stop =
8.8e-11 @ 36% T — deeper than the APLC (2.1e-10 @ 27%) at more
throughput, with no apodizer to fabricate**; 0.70 = 4.1e-10 @ 49%;
0.90 = 8.0e-9 @ 81%.  Past 0.90 the leak ring arrives inside the stop
(flux inside 0.1% → 1%) — useful dial range 0.50–0.90.  Charge 4 ~4×
deeper than charge 6 throughout (smaller pixel-averaged core).  Figure
ctb_vortex_lyot_sweep.{png,mat}; deck gains main slide 6 (19 slides,
downstream renumbered).

## SESSION 10 (2026-08-25) — DM layer: actuators, engine-measured Jacobian, EFC dark hole

Roadmap deepening (Dave: "more specific coronagraph work" on the CTB
substrate, ahead of the e2e6m redo).  The two flat DMs become
controllable: `ctb_dm_rx` emits `ctb_dm.in` (DM blocks → `Surface=
GridData`, 256-grid channel in the ELEMENT's own frame — pData=VptElt /
xData=xObs / zData=psiElt, the localization rule), `ctb_dm` is a 32×32
influence-function actuator model (pitch 0.666 mm = beam/32, Gaussian,
12% coupling, 880 active per DM), `ctb_chain` is the reusable masked-chain
runner (0.4 s/run at N=512), `ctb_dm_jacobian` measures G =
dE(dark-zone)/d(act) with 1760 forward pokes (h = 2 nm, 11.3 min; 37 MB
.mat gitignored + .fp.json), and `ctb_efc` closes the EFC loop on the
engine with a per-iteration α line search against MEASURED contrast.

**Result: dark zone 3–15 λ/D mean 2.934e-7 → 8.055e-9 (36×) in 19
iterations, strokes 9.9/8.6 nm rms** (3–8 λ/D: 40×; 8–15: 25×; fixed G,
never relinearized; linear-achievable floor 4.5e-9 at 11 nm rms → the
measured floor is within 2× of linear-optimal).  Figure `ctb_efc.png`.

Two traps recorded (README "DM layer" section): (1) **the real-stacked
solve** — complex-SVD EFC commands pass `double` validation and the mex
silently drops Im(da); achieved field decorrelates (corr 0.13, ratio
0.17) and the loop crawls at 3%/iter.  Solve `[Re G; Im G] da = −[Re e;
Im e]`; `ctb_dm.apply` now rejects complex.  (2) **pupil OPD reads at the
exit pupil** (`macos.trace(30)`), not the FPA default — at the FPA a DM
bump reads as a global low-order term ~10× its sag (looks like an engine
grid-amplitude bug; is not).  Engine-side validation along the way: grid
readback bit-exact, sag→OPD = 2·cos(AOI) within 2%, poke localization
mirror-exact, superposition 1e-13, chain bit-repeatable.

**Both DMs are load-bearing (measured 2026-08-25):** the same loop on
the same measured G restricted to DM1-only reaches 1.30e-7 (2.3×, stalls
after 2 iterations — phase-only control at the stop cannot touch the
symmetric speckle half); DM2-only reaches 2.55e-7 (1.1×); the pair
reaches 8.06e-9.  The annulus is a two-DM product, as the dark-zone
geometry rule predicts.

Gates: `tests/tCtbDm.m` added to SUITE_CTB_512 (emitter frame audit, grid
readback, sag→OPD scale/sign/location, speckle pair, chain contrast pin
2.934e-7, Jacobian column linearity, EFC-digs-2×-in-3-iters smoke).

Next (this arc): relinearization schedule (re-measure G around the dug
state) for the next depth decade; pairwise-probing estimator (lab-facing
sensing); time-series drift + dark-zone maintenance on the run_simulator
pattern; the engine-faithful mask/apodizer design operator (S3b unblock).

## SESSION 9 (2026-08-06) — proper_ctb_run: end-to-end PROPER hand-off (option 1)

Roadmap follow-on to SESSION 8: close the hand-off gap.  `proper_ctb_check`
verifies the export PER LEG (each re-seeded from our field); a PROPER user
doing their OWN analysis needs a SINGLE-PASS pure-PROPER model that runs
end-to-end from our data alone.  Deliverable = the `.mat` + one driver a
PROPER user runs with no macos: reproduce our bare PSF and our coronagraph
contrast.  Pure MATLAB; no engine edit.

**STOP-AND-FLAG -> Dave chose OPTION 1.**  A single CONTINUOUS PROPER beam
DM1->FPA does NOT reproduce macos (FPA pitch ratio **0.71**, corr **0.005**).
Root cause: macos samples every intermediate focus on the **system
exit-pupil Fraunhofer pitch** (EP-sphere radii R = 7017/1000/416/360 mm),
NOT the local geometric focus set by each OAP focal length (f=|Kr|/2 =
2.47...0.68 m).  Per-station diagnostics: PUPIL planes track macos to
~1.0-1.15 and PROPER beam diameters match the design pupil sizes
(42.8->32.5->17->7.1 mm) — imaging is right — but FOCAL pitch diverges 4-10x
at each focus and compounds.  One grid can't carry both the pupil pitch and
the focal pitch across an f-f relay; per-leg replay escapes it by re-seeding
each leg on its own macos-matched grid.  Flagged; Dave picked **option 1**:
one pure-PROPER *script* seeded from OUR exported fields (single-*script*,
not single-*beam*).

**SHIPPED:**
- **`ctb_phase_export.m` -> v2** (`meta.format_version=2`).  Added
  `stations.EFL_m` (=|Kr|/2 SI for powered OAPs; NaN elsewhere — ExitPupil's
  focusing radius is its FarField sphere R in `legs`/`spheres`, not |Kr|/2)
  and a **`masks`** block (4 stand-alone arrays + metre params: Apodizer soft
  circle r0=15mm sigma=2mm; FPM hard occulter 2.70 lambda/D, array on the
  macos focal grid FOR REFERENCE, rebuild at consumer focal dx from
  `radius_m`; Lyot 0.50 of bare pupil; FieldStop OPEN placeholder).  Preview
  downsampler + `out` struct carry `masks`.  Regen'd `.mat` (312 MB,
  gitignored) + preview (2.9 MB) + `.fp.json` (0.08 MB).
- **`proper_ctb_run.m`** (NEW) — reads ONLY the `.mat` + PROPER, asserts the
  orientation probe at startup, two runs:
  - **bare** = terminal replay from the exported ExitPupil field
    (`prop_lens(R)+prop_propagate(R)`, R=0.35995 m);
  - **coronagraph** = pure-PROPER Fourier cascade seeded at the exported
    Apodizer pupil field (apodize -> lens(f4)+prop(f4) -> FPM occulter ->
    prop(f5)+lens(f5) -> Lyot -> lens(f6)+prop(f6) -> FPA), f4/f5/f6 =
    1.3494/0.6756/0.6355 m from `EFL_m`.
- **Rider** — `proper_ctb_check.m plot_check_` now colours OAP-fed pupils
  (DM1, ExitPupil in s2s) **grey/info**, not green/gated; gate logic
  unchanged; both check PNGs regenerated, both modes still PASS.

**GATES (MATLAB PROPER, N=1024, 500 nm, v2 export; PASS):**
- **bare**: pitch ratio **1.0000**, intensity corr_I **1.000000** — the
  exported and PROPER bare PSFs are visually identical.
- **coronagraph**: dark-zone mean contrast **1.4e-8** over 3-15 lambda/D
  (lambda/D~3.75 px).  Gate is **ONE-SIDED-DEEP**: upper bound <=2x shipped
  (5.8e-7) is the real gate; lower bound >=shipped/50 is only a pathology
  floor.  The cascade is legitimately ~20x DEEPER than the shipped macos
  value (2.9e-7) because the idealised Fourier relay seeded at the Apodizer
  carries upstream aberration but OMITS the downstream OAP4->FPA real-optic
  figure that scatters light in macos (`meta.screen_method`).  Do NOT gate
  two-sided.
- **mid-chain Lyot**: REPORTED, NOT GATED — same-grid corr_I 0.93, beam-dia
  ratio 4.3x (the cascade forms the Lyot on PROPER's own sampling; a raw
  correlation across that scale gap is not a valid gate — README rule 2).

**Consumes** the as-committed `ctb_s2s_dcr.in` (untouched).  README
"Hand-off package" subsection + `proper_ctb_run.png` (bare macos-vs-PROPER,
coro FPA, radial-contrast panel).  All on `pol-ifo`, commit-only.

---

## SESSION 8 (2026-08-06) — CTB phase-factor export for external PROPER models

Roadmap item 5 (deck_ctb "Next"): export the CTB **full model**
(`ctb_s2s_dcr.in`, 44 elts, N=1024, 500 nm) per-plane phase factors so an
EXTERNAL PROPER user can consume this model's surfaces/fields and check theirs
plane by plane.  Pure MATLAB; no engine edit.

**SHIPPED:**
- `ctb_phase_export.m` → `ctb_phase_export_N1024.mat` (`-v7.3`, ~320 MB,
  gitignored) with `meta / stations(18) / legs(17) / spheres(4) / screens(18)`.
  Stations = 8 OAPs + 2 DMs + 4 mask/focus planes + ExitPupil + FPA; metres
  throughout (`dx_at` runtime SI, not dxElt); center px floor(N/2)+1; OPD sign
  flip stamped (`OPD_m=-angle(E)λ/2π`, opposite `prop_add_phase`); orientation
  probe baked in (+X pupil ramp → FPA peak dcol=-32).  `legs` classifies every
  hop (NFPlane p2p / through-focus quartet / FarField / geometric) with sphere
  radii.  `screens` = per-optic added OPD via the diff-of-consecutive-plane-OPD
  construction (documented; clean per-optic split not engine-readable).
- `proper_ctb_check.m` — reads ONLY the `.mat` (no mmacos), PROPER-optional
  (skip-with-message).  Two modes: `'s2s'` (replay legs) / `'collapsed'`
  (consume our E as hand-off).
- Large-file policy (dd8f11b pattern): `.mat` gitignored by explicit path;
  committed `ctb_phase_export_N1024.fp.json` (87 KB, `jac_fingerprint.m` copied
  in) + `ctb_phase_export_preview.mat` (96×-downsampled, ~3 MB).  Regen line in
  README.
- README "Phase-factor export" section (external-user interface doc: format
  field-by-field, sign/orientation/units, both modes with run lines, pinned
  gate numbers) + `proper_ctb_check_{s2s,collapsed}.png`.

**GATE NUMBERS (MATLAB PROPER, N=1024, 500 nm; both modes PASS):**
- THROUGH-FOCUS + FarField (Focus23/FPM/FieldStop/FPA), replayed from the
  FEEDING SPHERE: intensity peak-norm **corr_I = 1.000000** — the arbiter class.
- COLLAPSED-mode pupils (all 6): corr_I ≥ **0.961**.
- S2S-mode direct-NFPlane pupils (Apodizer/Lyot/CheckPoint): corr_I ≥ **0.9998**.

**TWO INTERFACE FINDINGS (flagged to Dave, documented for users — NOT engine
bugs):**
1. **Through-focus legs replay ONLY from the feeding reference sphere.**
   Seeding PROPER at the optic/mask plane fails (dx mismatch ~4.5×, corr~0);
   seeding at the EPreturn sphere with its dx + R and `prop_lens(R)+
   prop_propagate(R)` reproduces macos at corr 1.000000 (the arbiter).  Hence
   the `spheres` block was ADDED to the export (v1 → v2) as the replay enabler.
2. **Collimated NFPlane pupil→pupil legs: raw complex fields do NOT match
   after PROPER `prop_propagate`.**  macos NFPlane reads on a PLANAR reference
   (curvature re-zeroed); PROPER accumulates the full Fresnel quadratic
   reference-sphere phase.  So INTENSITY agrees (corr ~0.95) but the raw
   complex fields differ by a large quadratic term (DM1→DM2 raw corr ≈ −0.8).
   Convention difference, not error (NF p2p validated 2.4e-14 macos-vs-macos).
   Consequence: judge collimated legs by intensity, or consume our E directly
   (the always-valid `collapsed` hand-off).  Powered OAPs are the external
   user's own `prop_lens` — `optic`-kind stations are mid-beam, not hand-off
   planes, reported-not-gated.

**Consumes**: the as-committed `ctb_s2s_dcr.in` (untouched).  All artifacts on
`pol-ifo`, commit-only.

---

## SESSION 7 (2026-08-06) — compact-deck DM1→DM2 near-field prop fix + comparison re-run

Dave: "the compact model does not have the right setup for the DM1-DM2
propagation, while the full model does — replace the DM1-DM2 reference
surfaces / zElts from the full model."

**THE BUG.**  `ctb_dcr.in`'s DM1→DM2 NFPlane leg (`P1_start`/`P1_end`) was
mis-set: `P1_start` zElt = **−399.94** (too short) and `P1_end` Vpt at
**x=1528** — one radius BEHIND DM1 (wrong side) — vs the true DM1-DM2
spacing ~500 mm (DM1 x=1928, DM2 x=2428).

**THE FIX.**  Copied the numerically-correct values from the full model's
`Prop1_start`/`Prop1_end` (`ctb_s2s_dcr.in`), keeping the compact deck's own
element names/indices (drivers index by station, not name):
- `P1_start` zElt  −3.9993878644E+02 → **−4.99923483044516D+02**
- `P1_end`  Vpt/Rpt (1528.31,−104.90) → **(2428.174,−103.305)** (now AT DM2)
- `Extinc` 1.0E+22 → **0e0** (both planes)
The `psiElt`/`KrElt` were already identical between the decks.

**VERIFIED.**  Fixed compact deck now matches the full model bit-for-bit at
DM1 (sum 1.0000, r 1.062e-2 m) and DM2 (sum 0.9894, r 1.062e-2 m); FPA still
centred [513,513].  All six mask drivers re-run green; ranking unchanged
(APLC deepest, vortex best throughput; mask contrasts shifted only in the 3rd
digit — the masks act at downstream conjugates, so the DM1→DM2 leg only sets
the pupil field feeding them, e.g. APLC 1.6e-10 → 2.1e-10).

**COMPACT-vs-FULL COMPARISON re-run (N=512, the SESSION-4 metric):**
- BARE agreement UNCHANGED: peak-norm corr **0.9989**, EE(±10px) diff 0.21%
  (compact 0.9653 / full 0.9632).  The fix corrects the compact deck's
  INTERNAL DM planes without moving the terminal-PSF agreement.
- CORO (apod + 2.70 λ/D hard occulter + Lyot 0.50) suppression: compact
  **5.09e4** / full **2.90e4** → gap **1.76×**, DOWN from the SESSION-4 2.6×
  (that pair was at a 3 λ/D occulter).  Correcting the DM1→DM2 near-field
  prop brings the compact model CLOSER to the full model's diffraction, as
  expected.  A residual gap remains because the compact deck still omits the
  OAP→OAP inter-optic p2p legs the full deck carries — the SESSION-4 point;
  PROPER (FPM-leg corr 1.000000) stays the arbiter, not model-vs-model.
- Regenerated `ctb_coro_compare_bare.png` / `_coro.png` +
  `ctb_mask_compare.png`/`.mat` on the fixed deck.

---

## SESSION 6 (2026-08-05) — literature focal-plane-mask families (Dave's go)

Dave's work order: build the standard coronagraph mask/apodizer pairings
from the literature on the existing apodize/apodize_complex + per-lambda-grid
machinery (no engine work).  All formulae were pulled VERBATIM from the ar5iv
LaTeX of the source papers (a WebFetch summarizer hallucinated every equation,
so the raw HTML was the arbiter).  Four families shipped + a comparison table.

**Verified-formula flags baked into the builders (the ones that bite):**
- **BLC is `1 - sinc` in AMPLITUDE**, not `1 - sinc^2` (that is the intensity
  `|M-hat|^2`).  `sinc` here = UN-normalised `sin(z)/z`, NOT MATLAB's `sinc`.
- **Lyot trim = `(1 - epsilon)`** retained diameter (KCG Eq. 2), NOT `1 - 2*eps`
  (a source showing `1-2eps` uses a half-bandwidth epsilon).
- **Roddier & Roddier 1997 uses NO apodizer** — extinction is by flux balance
  (pi spot at 0.53 lambda/D encircling 50% Airy energy).  The prolate-apodized
  RRPM/ARPM is the LATER Aime/Soummer work.
- **Dual-zone (N'Diaye 2012) zones are PURE PHASE, non-pi, wavelength-sliding**
  (phi = 2 pi OPD lambda0/lambda); the amplitude apodization lives in the
  entrance PUPIL, not the mask.  d1=0.874, d2=1.445 lambda0/D; OPD1=0.309,
  OPD2=0.678 lambda0 (phi1=1.94, phi2=4.26 rad at lambda0).
- **Ideal even-charge vortex on a clear pupil** sends on-axis starlight
  identically outside the geometric pupil (Jenkins 2008 Eq. 1); the Lyot need
  NOT be undersized as for a hard occulter.

**NEW builders (mask primitives, reusable):**
- `ctb_mask_bandlimited.m` — K&T 2002 order-4 (`1-sinc` amplitude, Eq. 7) +
  KCG 2005 order-8 (Eq. 13, m=1/l=3); separable | radial | linear 2-D forms.
  Unit-tested: order-8 IWA 8.94 lambda/D matches published; near-origin
  intensity slopes 4.00 / 8.00 confirm the null orders.
- `ctb_apod_prolate.m` — Soummer 2005 Eq. 3 prolate apodizer by POWER
  ITERATION of `[pupil] o FT^-1 o [occulter] o FT`; returns the dominant
  node-less prolate + Lambda0.  Converges to the classic smooth bell.
- `ctb_mask_phase.m` — R&R 1997 pi-disk + N'Diaye dual-zone complex focal
  masks (via apodize_complex).

**NEW drivers (each = one mask kind + gates/figure):**
- `ctb_bandlimited.m` (the PRIORITY).  Gates: (a) ideal null suppression
  **4.4e5** (floor 2.9e-12), trim rule `(1-eps)R` VERIFIED (99.9% of star
  flux outside it); (b) contrast+throughput vs epsilon (eps 0.10->0.55:
  1.2e-8@81% -> 3.3e-8@20%); (c) bandpass — fixed-metres mask **1.1x**
  chromatic over 10% (BL masks are chromatic BY DESIGN), lambda/D-rescaled
  mask 1.0x (machinery adds no spurious chromatic error).  Clear pupil, no
  apodizer.
- `ctb_vortex_matched.m` (the CHEAPEST WIN).  Ideal charge-6 vortex on the
  CLEAR pupil sends ~99% of on-axis flux OUTSIDE the geometric pupil (Lyot
  panel shows the textbook bright ring, dark centre).  The C/T knee opens the
  Lyot to frac 0.90 -> **3.2x throughput** (charge 6) / 3.6x (charge 4) vs the
  unmatched frac-0.50 baseline, at 2-2.6x contrast cost.  NOTE: the original
  `ctb_vortex.m` left the apodizer ON and used the hard-occulter Lyot 0.50,
  so it saw only a Lyot GRADIENT and 1.7x; the clear-pupil matched-Lyot
  config here is the correct ideal-vortex demonstration.
- `ctb_aplc.m`.  Prolate apodizer + 2.8 lambda/D occulter (Soummer 2011 GPI:
  occulter DIAMETER 5.6 lambda/D, cross-checked arXiv:1103.6085) + near-full
  Lyot.  DZ mean **1.6e-10** (suppression 3.3e6) — **174x deeper than the
  throughput-matched BLC** at equal 27% throughput.  The post-occulter Lyot
  pupil shows the APLC edge-ring signature.
- `ctb_phase_masks.m`.  R&R pi-mask + dual-zone reported beside the vortex.
  On the UNMATCHED clear pupil these are shallow (R&R 3.3e2, dual-zone 4.6e2,
  vortex 6.5e2 suppression) — correct: deep R&R/dual-zone need the matched
  entrance-pupil apodizer (the ARPM/DZPM), noted in the builders.  Honest
  sampling caveat: the 0.53 lambda/D R&R spot is only ~2px at N=1024.
- `ctb_mask_compare.m`.  One ROW per family on the same deck/grid/annulus/
  normalisation; contrast-vs-throughput scatter + FPA thumbnails + a .mat
  table.  Ranking (annulus 3-15 lambda/D, Strehl-norm):

  | mask kind        | DZ mean  | suppression | throughput |
  |------------------|----------|-------------|------------|
  | APLC             | 1.6e-10  | 3.3e6       | 27%        |
  | band-limited     | 2.6e-8   | 4.4e5       | 36%        |
  | hard occulter    | 2.4e-7   | 6.3e4       | 25%        |
  | vortex (matched) | 2.9e-7   | 6.5e2       | **81%**    |
  | Roddier pi-mask  | 3.1e-6   | 3.3e2       | 81%        |
  | dual-zone        | 6.3e-6   | 4.6e2       | 81%        |

  APLC deepest; BLC second; vortex best throughput.  (Throughput = off-axis
  proxy: Lyot area, times (1-eps)^2 or apodizer Phi^2-fill where applicable.)

**HLC — DEFERRED to the FALCO integration (explicit).**  The hybrid Lyot
coronagraph's FPM profile (a metal+dielectric complex-amplitude occulter) is a
CO-OPTIMIZATION product of the FALCO/EFC design loop, NOT a closed-form profile
you can build from a disk + a phase bump.  Recorded here so nobody hand-waves
an "HLC" from a formula; it enters when FALCO is wired in.

**All on `pol-ifo`.  Pure MATLAB; no engine change** (the E.1 finding stands:
`cfield_apodize_complex` / `macos.apodize_complex` already exists).  Every
driver defaults N=1024, MACOS_HOME-gated, headless-safe (PNGs verified
numerically colored, not by eye).

---

## SESSION 6b (2026-08-05) — mechanical queue closed (merged from pol-ifo-mech)

The four items SESSION 4 deferred, plus the engine guard it flagged. The
science sections below are untouched.

| Item | Verdict | Commit |
|---|---|---|
| E.3 `xp_fnd` EltID guard | closed | macos `dev` e1d3c2b + MACOS_res_dev `dev` bdfe0e5 |
| add_pupil FarField emission | no fix needed | evidence below |
| s2s generation rules | closed | pol-ifo-mech 4ca9fc2 |
| `tCtbProp` | closed | pol-ifo-mech e2e2511 |
| `README.md` diffraction section | closed | this commit |

**E.3 `xp_fnd` guard.** `xp_fnd` ran FEX on `nElt-1` unconditionally and
returned PASS even when the FEXIT handler had declined to write (it writes
only a Return(8) or Reference(3) surface), so a deck whose penultimate
element is a powered optic silently kept a stale exit pupil. A new
`XpEltOrWarn()` precheck returns FAIL naming the element and its EltID; it
runs before the stop check so the engine's stdout and the binding's error
identifier always name the same cause. mmacos raises `macos:fex:noPupilElt`,
distinct from `macos:fex:noStop`. The four `dw_d*_multi` supervisors were
built on the old silent decline — their per-field FEX did nothing and
`finalize` issued one `macos:dw_dx_multi:noPupil` warning — so the call moved
into `reset_xp_guard('fex', ...)`, which absorbs the new FAIL into that same
verdict. Gates: bare focal deck fails cleanly, e5hex1 radius unchanged from
the pre-change mex to all digits, stopless deck still reports noStop,
`dw_dx_multi` with `reset_xp` on a no-pupil deck still completes with stamp
`no-effect`; mmacos fast 253/0, tPupilMap 12/0. `sxp_fnd` dispatches SXP, a
FEXIT clone with the same element-type check, and carries the same gap —
left for its own go.

**add_pupil FarField — the record was right, no fix needed.** A three-mirror
telescope built through `macos.design.Telescope`, `add_pupil`, `save`: the
emitted deck carries `PropType= FarField` on the ExitPupil and Geometric
everywhere else (`Telescope.m` emits FarField only at that element), with
`KrElt = -R` and `zElt = +R`, R = 3.5731681773874477. The engine runs
"Far-field sphere-to-plane diffraction" on it and the PSF forms on the DC
pixel — peak 1.02e4 at [129,129] of 256. Two values differ from the CTB
terminal convention and neither is read by the transform: `FP_return` carries
zElt = +R rather than 1e22, and the FocalPlane 1e20 rather than 1e22
(`FFPROP` takes only the sphere's zElt).

**s2s generation rules.** `ctb_prop_layout.m` now emits BOTH models from
`ctb_planar_stageF.in` on the validated conventions — quartet per focus,
near-field pair per propagated leg, far-field terminal — writing
`ctb_dcr_gen.in` (31) and `ctb_s2s_dcr_gen.in` (44) beside the committed
decks. The sphere radius is measured by FEX on a truncated deck ending in
`<upstream optics>/FPreturn/EPreturn/focus-as-FocalPlane`, reproducing the
hand decks' radii from arbitrary seeds to ≤ 4.4e-8 mm at all three foci and
the terminal. Acceptance at model 512: chief ray at all ten real optics
1.64e-11 mm (committed decks 1.36e-11 mm under the same measure), PSF peak on
[257,257] and dx 2.4039e-5 m on all four decks, bare correlation 0.999999 per
model, peak ratio 1.000000 for the full model, quartet audit identical for
generated and committed (3 quartets, 3 zElt-equal, spheres centred to
9.1e-13 mm, 14 Returns, even).

**The committed compact deck's DM1→DM2 leg was wrong, and is fixed.** It
propagated that leg over 399.94 mm where the chief distance between the two
stations is 499.92 mm — 0.8× it, the old builder's 10%/90% plane placement,
inherited by the hand deck, with `P1_end` landing 399.94 mm *behind* DM1 and
reached by a negative ray length. The full deck's `Prop1` pair was already
correct. Raised as a flag, then fixed on Dave's instruction (2026-08-06) by
lifting `Prop1_start`'s zElt and `Prop1_end`'s Vpt/Rpt verbatim into
`P1_start`/`P1_end` — three lines, so the two committed decks now agree on
that leg to all digits and the generator reproduces both.

Effect: the leg is collimated pupil-to-pupil, so only edge diffraction
changes. Bare FPA peak 7.006e-2 → **6.990e-2** (0.23%); compact-vs-full
correlation 0.998863 → **0.998895**; generated-vs-committed compact peak
ratio 0.9977 → **0.999999**. `tCtbProp` pins updated, 8/8. **Analysis
outputs committed before 2026-08-06 — the contrast / planet / bandpass /
vortex figures and their recorded numbers — were produced on the short leg
and carry that 0.23%.** Not regenerated here (masks work order).

**`tCtbProp`** (8 checks, asset-gated, model 512, own runner batch
`SUITE_CTB_512` and a `./run_mmacos_tests.sh ctb` shortcut): quartet audit on
both decks, NF1/NF2 round-trip identity, centred-PSF pins, the 0.998895
compact-vs-full correlation, the PROPER arbiter (skipped with a message when
PROPER is absent), and generator-reproduces-committed. 8 pass, 0 fail, none
skipped; the PROPER leg reproduces the recorded arbiter values exactly. The
two quartet pins were checked against the bug they exist for: negating every
`EPreturn2` zElt in a scratch copy fails both, the round trip reporting
42%–125% relative field error at the post-mask sphere.

**Superseded artefacts.** `ctb_planar_prop.in` (29 elements, the old
three-surface Sin/Sout/Sret construction) is no longer produced by
`ctb_prop_layout.m` and nothing references it. The "What exists" and
"Augmented deck element map" sections below describe that builder and are
historical.

---

## SESSION 5 (2026-08-05) — performance pass (centering, sampling, mask/Lyot optimize)

Dave: "find better performance -- the hard occulter is not centered, the
resolution is poor, and I'm not sure of the best mask/Lyot radii for
3-7 lambda/D; check the literature."  Three fixes + a sweep:

1. **CENTERING BUG (half a pixel) -- FIXED.**  The mask builders centred
   disks/apodizers on `c=(N-1)/2`, but MACOS's FarField/NF2 focus lands on
   the FFT DC pixel `floor(N/2)` (0-based) = 1-based N/2+1 (verified: focus
   at [257,257]/512, [513,513]/1024).  So every occulter sat half a pixel
   off the beam -> asymmetric residual leak.  Consolidated the duplicated
   builders into shared `ctb_mask_disk.m` / `ctb_mask_softcircle.m` centred
   on `floor(N/2)`; the vortex singular pixel and all beam_radius_ centres
   moved to `floor(N/2)+1` too.  The vortex benefited most (an off-centre
   singularity leaks badly): its advantage over the hard occulter went
   1.7x -> 2.9x.

2. **SAMPLING -- default bumped 512 -> 1024.**  lambda/D at the FPA = N/pupil
   px (pupil fixed at the deck's nGridpts=255): N=512 gave 2.0 px/lambda/D
   (Nyquist edge, poor), N=1024 gives 4.0 px, N=2048 -> 8.0.  Bumping
   model_size zero-pads the pupil -> finer PSF, no Rx change.  All analysis
   drivers (contrast/planet/bandpass/vortex) now default N=1024; the FPA
   panels resolve clean symmetric Airy rings.

3. **MASK/LYOT OPTIMIZE -- `ctb_optimize_masks.m` (NEW).**  Literature box
   (Wikipedia Coronagraph; Sivaramakrishnan+ 2001; HCIT class): occulter
   ~1-3 lambda/D, Lyot ~75-90% pupil.  Pure mean-contrast in 3-7 lambda/D
   pushes monotonically to bigger occulter + smaller Lyot (both reject more
   light -> throughput + IWA cost), so the driver also records throughput
   (Lyot area fraction r_lyot^2) and a contrast/throughput knee.  A WIDENED
   sweep (occulter 2.0-3.5, Lyot 0.45-0.75) found a ROBUST, ISOLATED
   contrast null at **r_fpm = 2.70 lambda/D** (a diffraction resonance of
   this bench's occulter/Lyot structure): C ~ 3-4e-7 there vs ~1e-5 at
   2.9-3.2 lambda/D.  Confirmed at N=1024 (persists, localises to 2.70,
   broad in Lyot 0.40-0.50 -> NOT a grid artifact).  **Shipped defaults:
   r_fpm=2.70 lambda/D, r_lyot=0.50** (deep contrast at 25% throughput;
   Lyot is the user's throughput dial).  Dave chose "widen the search
   first"; the sweep box is now wide enough that the optimum is interior,
   not on an edge.

**NET PERFORMANCE GAIN (old 3.0/0.85 @ N=512 -> new 2.70/0.50 @ N=1024):**
dark-zone mean contrast 4.6e-6 -> **2.9e-7 (~16x deeper)**, coro FPA peak
1.9e-5 -> 1.4e-6.  PROPER arbiter still 1.0000 / corr 1.000000 at the finer
sampling (diffraction layer stays validated).  Bandpass: the deeper, sharper
2.70 null is MORE chromatic (mono 2.4e-7 -> broadband 4.9e-7, 2.1x over 10%,
vs the old shallow occulter's 1.02x) -- an honest deep-vs-broadband trade.

4. **TRAIN RENDER ray-gap -- FIXED.**  The both-endpoints-in-window segment
   test dropped the physical beam segment connecting two real optics that
   bracket an off-bench reference-sphere detour (beam appeared to vanish
   between elements).  Now keeps all in-window crossings and draws ONE
   polyline through them -> continuous fold path DM1->FPA.

5. **TRAIN RENDER rewritten (full-only, dots on beam, no backtrack).**  Three
   fixes rolled together:
   - **Dots ON the beam.**  The old dots were at VptElt = the OAP PARENT-axis
     vertices at z=0, far from where the off-axis beam strikes (z=-113..-643)
     -> dots sat on a flat z=0 line, not on the rays.  Now each optic dot is
     placed at its MEAN RAY CROSSING (from draw_rays), on the beam.
   - **No "extra bit past DM1".**  OAP2 genuinely sits at x=1774 (left of DM1
     at 1928) -- that leftward fold is real.  The spurious bit was the
     Focus23 reference-return PLANE at x=1740 (diffraction bookkeeping) that
     an all-crossings polyline routed through.  The beam is now drawn through
     REAL optics only (Reflector + FocalPlane), so it never backtracks to a
     reference surface.  source->OAP1->DM1->DM2->OAP2->... is clean.
   - **Full deck only** (Dave: "no need to plot both") + source star + the
     three mask stations (Apodizer/FPM/Lyot) as magenta rings on the beam.
     Near-coincident labels (OAP1~DM2, FPM~OAP8, FPA~OAP7 on this folded
     bench) are staggered onto a lower row so they don't overprint.

---


## SESSION 4 (2026-08-05) — Dave's hand-built decks + comparison driver

Dave hand-built two correct decks (supersede my generated ctb_planar_prop.in):
- **`ctb_dcr.in`** (31 elts) — "compact" model: NFPlane DM1->DM2, then NF1+NF2
  through Focus23 / FPM / FieldStop (the 4-surface FPreturn/EPreturn quartet),
  FarField ExitPupil->FPA.  Missing the inter-optic plane-to-plane props.
- **`ctb_s2s_dcr.in`** (44 elts) — "full" surface-to-surface model (adds the
  p2p props).  Full s2s generation rules deferred (Dave: "save for later").
Both are PRE-ALIGNED (ORS/SRS/FEX already applied + saved); load and produce a
CENTRED PSF at [257,257]/512 with dx=2.404e-5 m, no runtime setup needed.

**Both models agree** (bare, no coro): peak-norm corr 0.9989, EE within <0.3%
by +/-10px; compact peak slightly higher / core slightly tighter (it omits the
p2p diffraction).  Station indices:
  compact: DM1=2 DM2=5 Apodizer=13 FPM=17 Lyot=20 ExitPupil=30 FPA=31
  full:    DM1=2 DM2=5 Apodizer=16 FPM=22 Lyot=27 ExitPupil=43 FPA=44

**`ctb_coro_compare.m`** — parameterized compact-vs-full comparison driver
(SHIPPED this session).  Side-by-side INT at DM1/DM2/Apodizer/FPM/Lyot/
ExitPupil/FPA (rows) x models (cols).  Coronagraph masks (apodizer/FPM/Lyot)
applied IN MATLAB (macos.apodize multiply, reset_trace=false, continue prop).
Shared lambda/D across models (analytic first_order_properties.lamD_px) for the
DISPLAY scale only.  Verified:
  - bare:  compact peak 7.01e-2, full 6.03e-2 (matches direct compare).
  - coro (apod+fpm+lyot, 3 lam/D hard occulter): compact FPA peak 1.89e-5,
    full 4.28e-5; suppression 3.71e3 (compact) / 1.41e3 (full).

**SUPPRESSION CORRECTION (2026-08-05, finding 2).**  The earlier "~1e6
suppression both" number was an ARTIFACT of an FPM-sizing bug: the occulter
radius was computed as `r_fpm_lamD * lamD_px_FPA(16.8) * dx_FPM`, mixing the
**FPA** plate scale into the **intermediate FPM** plane -> occulter ~8x too
large in radius -> far more starlight blocked -> inflated apparent
suppression.  Fixed: the occulter is now sized in FPM-LOCAL lambda/D
(`lam*R/D_beam`, R = the NF1-sphere zElt at FPM-1, D_beam the geometric beam
diameter measured on that sphere) and painted on a DETERMINISTIC focal grid
`dx_f = lam*R/(N*dx_sphere)` -- verified equal to the engine's `dx_at(FPM)`
to ratio 1.0000 in the validated quartet (so the old "dx_at garbage at NF2"
gotcha was itself an artifact of the WRONG through-focus construction; with
Dave's quartet dx_at at NF2 is now trustworthy, but we compute dx_f
deterministically regardless).  A ~1e3 suppression is physically correct for
a HARD-EDGE occulter with no apodization; 1e6-class needs band-limited /
apodized masks (deferred).  The Lyot radius is now keyed to the BARE
geometric pupil radius (finding 4), not a radius measured from the post-FPM
intensity (which the Babinet ring inflates).  The compact-vs-full coro gap
(factor ~2.6) is the finding-3 point -- compact omits inter-optic diffraction;
PROPER is the arbiter, not model-vs-model agreement.
  Usage: out=ctb_coro_compare('coro',true|false, 'apodizer'/'fpm'/'lyot',
  't/f', 'models',<struct w/ .name/.rx/.elt>, ...).  Pass your own decks via
  'models'.  Params: r_apod_m, r_apod_taper_m, r_fpm_lamD, r_lyot_frac.

**PROPER ARBITER (2026-08-05, finding 3) -- `ctb_proper_compare.m`.**
Model-vs-model agreement is NOT validation.  The arbiter is a per-leg
cross-compare against MATLAB PROPER on the CTB deck's own sampling
(recipe validation-ladder item 1).  Ran it for the NOVEL kernel the whole
chain rests on -- the FPM through-focus leg (feed sphere FPM-1 -> FPM
focus), macos NF1 sphere->plane vs PROPER prop_lens(R)+prop_propagate(R),
beam_ratio=1 so PROPER pitch == macos pitch bit-for-bit.  RESULT:
  - dx_focal ratio macos/PROPER = **1.0000** (both 2.1267e-5 m)
  - peak-norm correlation = **1.000000**
  - centroid offset = **0.000 px**
So the CTB geometry reproduces PROPER's diffraction focus EXACTLY.  The
diffraction layer is sound; the compact-vs-full suppression gap (factor
~2.6) is a modeling-fidelity difference (compact omits inter-optic
diffraction), not a bug.  (Needs `~/dev/proper_matlab` on the path; the
macos leg runs standalone and skips the PROPER column if absent.)

**NEW WORK A/B/C (2026-08-05).**
- **A -- `ctb_train_render.m`:** deck-grade layout figure, XZ fold plane
  (YZ/XY degenerate on this planar bench), compact stacked over full,
  shared axis scale + 500mm scale bar, mask planes marked (dashed
  verticals + top labels off the ray lines), EP reference spheres as
  magenta rings.  Rays clipped to the physical bench window (the
  off-bench EP-return spheres at x~-3700 are hidden).  Compact and full
  render near-identically -- correct, they share the physical optics.
- **B -- `ctb_contrast.m`:** radial dark-zone contrast vs lambda/D at the
  FPA, Strehl-normalised to the bare (no-mask) peak, reusing the coro/
  helpers (radial_contrast / dark_zone_metrics).  lambda/D is
  DETERMINISTIC (lam*R_fpa/D_ep / dx_FPA = 2.006 px) -- SYSPROP returns 0
  on this finite-conjugate deck and the Airy-null finder gave a spurious
  16.8.  2-DM -> full annular dark zone; 3-15 lam/D: mean 4.6e-6,
  median 4.7e-7, floor 2.0e-11 (n=2732 px).  Also repointed
  ctb_coro_compare's shared_lamD_ to the same deterministic geometry
  (was 4.0/16.8 spurious).
- **C -- `ctb_planet.m`:** off-axis companion in the dark zone, incoherent
  star+planet sum.  KEY: the CoroExample WINDOW/FFP recipe does NOT
  displace a companion here -- Dave's "vertices ON the chief ray" rule
  makes the FPM AND FPA grids re-centre on the tilted chief after
  ffp+ors/fex, so the planet lands back at pixel centre and the occulter
  clips it (verified).  Instead inject the companion as a PUPIL PHASE
  RAMP on the complex field at DM1 (macos.apodize_complex): k cycles
  across the pupil -> planet at exactly k lambda/D, at the FIXED centred
  FPM (clears the occulter) and FPA.  Same "modify cfield in MATLAB"
  contract as the masks.  Verified: 6 lam/D target -> planet peak at
  5.98 lam/D; at flux ratio 1e-3 the planet (5.5e-4) stands above the
  star residual (2.7e-4) -- the difference panel recovers it cleanly.

- **D -- `ctb_bandpass.m`:** finite-bandpass CTB, nwf wavelengths summed
  INCOHERENTLY, mono vs broadband dark-zone contrast.  Focal masks rebuilt
  per lambda (hard rule).  TWO findings baked in:
  1. `set_src_wvl` DOES propagate to the grid (dx_at(FPA) = 2.28/2.40/2.52
     e-5 m at 475/500/525 nm) -- verified.
  2. MACOS's FarField FPA RE-GRIDS per lambda, so each mono PSF is the same
     pixel size on the NxN array; naively summing there CANCELS the
     chromatic effect (first cut gave mono==broadband exactly).  Correct
     broadband resamples each lambda's PSF onto ONE COMMON PHYSICAL detector
     grid (nominal-lambda pitch), flux-conserving -- what macos.compose does
     internally.  Then the broadband PSF visibly smooths (rings wash out)
     and contrast degrades a physically-sensible 1.018x over a 10% band with
     a FIXED-METRES occulter (fpm_size='meters', default -- subtends varying
     lambda/D -> chromatic).  fpm_size='lamD' gives the achromatic reference
     (mask auto-scales -> mono==broadband to precision; isolates "does the
     machinery add chromatic error" -> no).  compose() sums only engine
     traces (no MATLAB masks), so the sum is MATLAB-side, mask-aware.

- **E -- `ctb_vortex.m` + engine-scope findings (STOP-and-FLAG).**  The
  work order's E was framed engine-first (add cfield_apodize_c, fix dx_at
  at NF planes, guard xp_fnd, THEN build the vortex).  Scoping the engine
  first (as directed) changed the plan -- **the vortex needs NO engine
  change**:
  - **E.1 cfield_apodize_c ALREADY EXISTS.**  `macos_api_mod.F90:5239`
    `cfield_apodize_complex(OK, MASK_RE, MASK_IM, N, iElt)` multiplies
    WFElt by a complex NxN mask in place -- exactly the stage-2 API.
    Surfaced as `macos.apodize_complex`; used already in work C and here.
    So the charge-m scalar vortex is a pure-MATLAB phase mask on that API.
    `ctb_vortex.m`: exp(i*m*theta) at the FPM (singular pixel -> phase 0,
    O(1/N^2) defect, documented), Lyot rejects the star.  Charge-6 vs hard
    occulter, 2-15 lam/D: hard mean 6.7e-6, vortex 3.9e-6 (1.7x deeper) AND
    smaller inner working angle.  Idealized exp(i*m*theta) is achromatic by
    construction (composes with the bandpass driver).
    HONESTY CAVEAT: the Lyot panel shows a gradient, not the textbook
    "all starlight in a ring outside the pupil" -- the CTB exit pupil is
    not matched/circularized for an idealized vortex, so 1.7x is real but
    modest, not a perfect-vortex total rejection.  A bench designed for a
    vortex (clean circular matched Lyot) would do far better.
  - **E.2 dx_at at NF planes: NO LIVE BUG on the validated quartet.**  The
    old "2.6e26 garbage at NF2" gotcha was an artifact of the WRONG
    through-focus construction; on Dave's committed quartet dx_at(FPM)
    matches the deterministic dx_f to ratio 1.0000 (finding 2).  No fix
    needed; the drivers compute dx_f deterministically regardless.
  - **E.3 xp_fnd EltID guard: GENUINELY OPEN (the one real engine item).**
    `xp_fnd` (macos_api_mod.F90:5396) runs FEX on nElt-1 UNCONDITIONALLY
    (IARG(1)=nElt-1, line 5422) with no check that nElt-1 is Return/
    Reference.  On a deck whose penultimate element is a powered optic this
    silently mis-places the exit pupil.  A guarded version would return
    FAIL with a reason when EltID(nElt-1) is not Return/Reference.  THIS
    needs Dave's go (engine edit + relink chain).  ==> FLAGGED to Dave.
    **CLOSED 2026-08-05 (SESSION 6): macos dev e1d3c2b + MACOS_res_dev dev
    bdfe0e5.**

NEXT (deferred, engine-first items need Dave's go): E.3 xp_fnd guard;
full s2s generation rules; add_pupil FarField fix; tCtbProp test + README.  The generated ctb_planar_prop.in / ctb_prop_layout.m
are SUPERSEDED by Dave's hand decks for the terminal/mask structure -- keep for
the builder logic (NFPlane zElt rule, _Sret flip-back) but realign to the
4-surface EP quartet when regenerating.
**ALL FOUR CLOSED 2026-08-05 (SESSION 6): e1d3c2b/bdfe0e5 (guard), 4ca9fc2
(generation rules -- ctb_prop_layout.m rewritten to the quartet, and it now
emits BOTH models), add_pupil verified correct as emitted (no fix), e2e2511
(tCtbProp) and this commit (README).**

---

_Older: paused 2026-08-04, mid-diagnosis; see "OPEN ITEM"/"RESOLVED" below._

## What exists (branch `pol-ifo`, in `bench_ctb/`, all UNTRACKED so far)

> HISTORICAL (SESSION 6): `ctb_prop_layout.m` was rewritten in 4ca9fc2 and no
> longer emits `ctb_planar_prop.in`; nothing references that deck. `tCtbProp.m`
> exists (e2e2511). The current file inventory is in `README.md`.

- `Coro_propagation_summary.md` — recipe of record (copied in per work order).
- `ctb_prop_layout.m` — builder: loads `ctb_planar_stageF.in`, traces to get the
  chief-ray geometry, inserts the diffraction reference/return/sphere surfaces,
  bumps `nGridpts` to 255, writes `ctb_planar_prop.in`. Regenerable.
- `ctb_planar_prop.in` — the augmented deck (currently **29 elements**).

Not yet written: `ctb_prop_legs.m`, `ctb_propagate.m`, `+ctbmask/*`, `tCtbProp.m`.

## Augmented deck element map (current, 29 elts) — HISTORICAL, that builder is gone

`OAP1(1) DM1(2) P1_start(3,NFPlane) P1_end(4) DM2(5) OAP2(6)
Focus23_Sin(7,NF1) Focus23(8,NF2) Focus23_Sout(9) OAP3(10)
Apodizer_Pst(11,NFPlane) Apodizer(12) OAP4(13) FPM_Sin(14,NF1) FPM(15,NF2)
FPM_Sout(16) OAP5(17) Lyot_Pst(18,NFPlane) Lyot(19) OAP6(20)
FieldStop_Sin(21,NF1) FieldStop(22,NF2) FieldStop_Sout(23) OAP7(24)
Backend(25) OAP8(26) FP_return(27) ExitPupil(28,FarField) FPA(29)`

## Runtime setup (what the driver must do after load)

```matlab
macos.stop(2);                                   % stop at DM1 (as example_ctb.m)
for e = [7 9 14 16 21 23], mmacos('ors_run', double(e)); end   % refine 6 near-focus spheres
macos.fex(1);                                    % place terminal exit-pupil sphere (nElt-1)
```

## VERIFIED GOOD

- Deck loads and traces. `ok_pass = 333` (full diffraction grid) is preserved
  **end-to-end, source through OAP8**.
- **Chief-ray intersections match stageF to ~1e-13 at EVERY real optic
  OAP1..OAP8** (the decisive stageF-vs-prop comparison — `/tmp/ctb_cmp.m`).
  The negative-length reference legs (OAP3, Apodizer, OAP7 show chief L<0) are
  INTENDED and correct — not a bug.
- Field reads clean at every intermediate pupil and focus: Apodizer, FPM, Lyot,
  FieldStop, Backend, OAP8 all have sensible sum (~0.99) and pupil dx (~378 µm).
- All 6 near-focus sphere pairs (NF1 → NF2 focus → Return/Geometric) work.

## RESOLVED 2026-08-04 (session 2): the chief-direction-reversal bug

Dave spotted "ray 1 after OAP7 goes the wrong direction."  Confirmed by
comparing chief-ray DIRECTION (not just position) stageF-vs-prop: dot=-1
(reversed) across the OAP3/OAP4 and OAP7/OAP8 blocks; the ray BUNDLE blew up
at OAP8 (spread 1.3e8).  Root cause: each through-focus block was missing the
SECOND return surface.  Rx_Coro's block is FOUR surfaces:
  Sin(NF1) -> focus(NF2) -> 1stPropEnd(Return sphere, one radius PAST focus,
  reverses chief) -> **CorMaskReturn (Return FLAT, back AT the focus, flips
  chief forward again)** -> next OAP.
I had only three (no CorMaskReturn), so the reversed bundle hit the next
powered mirror and diverged.  FIX: added a `_Sret` flat Return (Element=Return,
Surface=Flat, psi=+uin, at the focus vertex, Kr=zElt=-1e22) after each `_Sout`.
Also made Sin/Sout a symmetric mirror pair about the focus (same |Kr|=r, Sin
z=+r, focus z=-r, Sout z=-r), matching Rx_Coro exactly.

RESULT: all real optics OAP1..FPA now match stageF dot=+1.0, posdiff~1e-13.
OAP8 bundle spread 3.84 (was 1.3e8).  **PSF FORMS at the FPA: peak 0.19,
sum 6870, dx=-1.99um** (cf Rx_Coro peak 0.25).  Deck is now 32 elements.

New element map (32 elts): OAP1(1) DM1(2) P1_start(3) P1_end(4) DM2(5) OAP2(6)
Focus23_Sin(7) Focus23(8) Focus23_Sout(9) Focus23_Sret(10) OAP3(11)
Apodizer_Pst(12) Apodizer(13) OAP4(14) FPM_Sin(15) FPM(16) FPM_Sout(17)
FPM_Sret(18) OAP5(19) Lyot_Pst(20) Lyot(21) OAP6(22) FieldStop_Sin(23)
FieldStop(24) FieldStop_Sout(25) FieldStop_Sret(26) OAP7(27) Backend(28)
OAP8(29) FP_return(30) ExitPupil(31,FarField) FPA(32).
Runtime ORS spheres now = [7 9 15 17 23 25]; fex(1) for the EP.

## CORRECTED MODEL (Dave 2026-08-05, table re-synced to the committed decks 2026-08-05)

My through-focus construction (Sin/focus/Sout/Sret) was WRONG.  NF1 and NF2
are the two halves of ONE near-field prop through a focal mask, but:
  - **NF1 = FarField sphere->plane** (EP sphere -> mask plane)
  - **NF2 = plane->sphere** (mask plane -> EP sphere)
Both use the EXIT-PUPIL sphere (Kr = -(EP->mask distance), a LARGE radius),
NOT a modest near-focus sphere.

**THE VALIDATED TEMPLATE IS DAVE'S COMMITTED DECK `ctb_dcr.in`, NOT any
hand-transcribed table.**  The as-committed Focus23 quartet (elts 7-10),
read verbatim, is the convention of record.  Each focal mask is a
**4-surface** quartet (NOT five — there is no trailing FPreturn2):

  FPreturn   (Return, Flat,  Geometric, at mask plane,  zElt = 1e22)
  EPreturn   (Return, Conic, NF1,  EP sphere, Kr=-R,    zElt = +R)
  <mask>     (Return, Flat,  NF2,  at mask plane,        zElt = 1e22)  % MASK HERE
  EPreturn2  (Return, Conic, Geometric, same sphere,    zElt = +R)    % SAME SIGN as EPreturn

  - **All four surfaces are `Element=Return`** (not Reference).
  - **Both sphere zElts are +R, identical to all digits** (e.g. Focus23:
    +7017.8526119080789).  The engine's NF1 chirp uses zStart=zElt(EPreturn)
    and zEnd=zElt(iElt+1)=zElt(EPreturn2); the mask sandwich is transparent
    (no spurious defocus) IFF **zEnd == zStart, SIGN INCLUDED**.  EPreturn2
    at **-R** produces S~2R — the exact defocus failure the round-trip
    investigation diagnosed.  Do NOT write -R.
  - **FPreturn and the mask both carry zElt=1e22** (the "plane" radius),
    NOT 0 and NOT 1e30.
  - R = distance EP->mask (the exit-pupil sphere radius, = -Kr).  The sphere
    vertex sits one R on the incoming (-chief) side of the focus (Focus23:
    focus x=+3274.6, EPreturn vertex x=-3743.3, R=7017.85), psi=+chief
    (pointing toward the focus / centre of curvature).  EPreturn and EPreturn2
    share the SAME pose and Kr (they are the same physical sphere, entered
    then exited).
  - Terminal FF leg mirrors this as a 3-surface triple: FP_return(Return,
    Flat, Geometric, zElt=1e22) -> ExitPupil(Return, Conic, FarField, Kr=-R,
    zElt=+R) -> FPA(FocalPlane, Flat, Geometric, zElt=1e22).  zElt=+R (positive)
    for the FarField sphere, same convention as EPreturn.
  - NFPlane p2p leg (DM1->DM2): P1_start(NFPlane, zElt=-L) -> P1_end(Geometric,
    zElt=0); the DIFFERENCE = -L = chief L (Focus23 example: L~399.94).

  The manual `CoroExample.in` (ret1_1/ret2_1/CoroMask/ret2_2/ret1_2) and
  `Rx_Coro.in` are the upstream lineage, but where a table and the deck
  disagree, **the deck wins** — it is the numerically-validated artefact.

### Alignment procedure (Dave point 1) -- do NOT hand-set psi/vpt
For each near-field prop: **ORS iElt on the STARTING reference element, then
SRS iElt+1 iElt** to slave/align the paired end element to it.  (CoroExample
.jou: `ors 5; ors 7; fex 15`.)  ORS aligns psi to the chief + fits the sphere
radius; SRS solves the partner's zElt/pose from the OPL between them.  Some of
my reference surfaces are currently NOT beam-aligned -> use ORS/SRS, not the
set_psi/set_vpt hand-alignment I added.

### Terminal (unchanged conclusion)
CoroExample terminal = ret1_3(flat,Geom) -> ret2_3(EP sphere, FarField) ->
foc_pln.  Matches Rx_Coro.  add_pupil SHOULD emit FarField for the EP->FP leg
(fix Telescope.m if not).

### zElt audit (session 3) -- these are NOW correct per the manual:
NFPlane p2p: start zElt=-L, end zElt=0 (Delta = -L = chief L).  Fixed for
P1/Apodizer/Lyot legs.  FarField: zElt = EP->image = L.  The through-focus
zElts will be RESET by the quartet rebuild + ORS/SRS.

## (superseded) REMAINING — terminal PSF is OFF-CENTRE + FarField terminal structure

1. PSF forms (peak ~0.19, sum ~6470, dx=-1.99um) but lands OFF-CENTRE at
   ~[284,407] in the 512 grid (centre 256).  APPLIED DAVE'S RULE (2026-08-04):
   every diffraction surface axis || chief, vertex at chief incidence point.
   In ctb_prop_layout.m the reused markers (Focus23/FPM/FieldStop/FPA) now get
   psi=cd(chief dir), vpt=cp(chief pierce) via set_psi/set_vpt.  The terminal
   ExitPupil is verified FULLY axis-aligned at runtime:
     chief . (EP->FP) = 1.000000, FP perp-dist from chief-through-EP = 0.0 mm,
     EP vtx = chief pierce exactly, psi = chief dir (antiparallel EP->FP form
     centres better: parallel gave [370,496], antiparallel [284,407]).
   So the residual decentre is NOT vertex/axis geometry -- it is the
   TRANSVERSE output-grid ROLL (xGrid/yGrid about the chief axis).  The deck
   header xGrid is the GLOBAL transverse basis; on a folded bench the chief
   axis at the FPA is rotated ~0.6 deg in XY vs that xGrid, so the far-field
   output grid is rolled and the on-axis PSF projects off-centre along one
   axis (row ~centred, column ~150px off).
   REFERENCE: Rx_Coro (on-axis) FPA peak is dead-centre [513,513]/1024; its
   EP is trivially axis+roll-aligned (all on the z-axis).  The CTB needs the
   far-field output frame's xGrid to track the LOCAL chief frame at the EP,
   not the global source xGrid.
   OPEN QUESTION FOR DAVE: what sets the far-field output-grid transverse
   roll -- the ExitPupil element's xGrid/TElt frame, the source xGrid, or a
   WINDOW/Tout frame?  Need the convention to roll the FPA grid onto the
   local chief frame so the on-axis PSF centres.
   - Runtime terminal config that produces the PSF: fex(1); EP vtx=chief
     pierce, psi=-chiefdir (antiparallel), Kr=zElt=-|EP-FP|; FPA zElt=1e22.
2. Dave: add_pupil SHOULD emit FarField for the EP->FP leg; if it doesn't,
   FIX add_pupil (Telescope.m ~2680) then align the CTB terminal to Rx_Coro's
   FocalPlane(Return) -> ExitPupil(Return,FarField) -> FocalPlane triple.
   Current CTB terminal = FP_return(30) + ExitPupil(31,FarField) + FPA(32),
   which is that same triple -- verify it matches Rx_Coro once centred.

## (historical) OPEN ITEM — the terminal OAP8 → FPA far-field leg

Symptom: **FPA reads peak=0, sum=0, dx=3.5e-17** (no PSF). Everything upstream
is fine (verified above).

Root cause identified: the terminal triple is built wrong vs the proven
diffraction template. The ONLY terminal that forms a real PSF in the tree is
**Rx_Coro** (`pymacos/tests/Rx/Rx_Coro.in`), which is a **2-surface** terminal:

```
ExitPupil (Return, Conic, PropType=FarField)  ->  FocalPlane
```
with, from a working trace (Rx_Coro FPA: peak 0.25, sum 23.7, dx=-5.8µm):
- ExitPupil: Vpt = FP_vpt + radius along the +z (incoming) side, i.e. ONE FULL
  RADIUS from the FP; `psi = unit(FP - EP)` (points EP->FP, toward the image /
  centre of curvature); `KrElt = zElt = -radius` (both NEGATIVE); NOTHING
  between the last optic and this sphere.
- FocalPlane: at the image; `psi` faces back toward the EP; `zElt = 1e22`.

My current deck instead has a **3-surface** terminal
`FP_return(flat at FPA) -> ExitPupil(sphere before FPA) -> FPA`, which scrambles
the pairing (FP_return sits between OAP8 and the EP; the EP is on the wrong side
at half the radius). `add_pupil`'s FP_return construct (Telescope.m:742) is a
GEOMETRIC layout aid; the DIFFRACTION terminal is the Rx_Coro 2-surface form.

### DECISION PENDING (Dave): terminal structure
- Option A (recommended): rebuild terminal as the Rx_Coro 2-surface
  `ExitPupil(FarField) -> FPA`, EP one radius from FPA on the incoming side,
  drop FP_return.
- Option B: keep the add_pupil 4-surface `OAP8 - FP_return - EP - FPA` (Dave
  flagged this); needs FarField wired correctly — not yet forming a PSF.

## Reference decks
- `pymacos/tests/Rx/Rx_Coro.in` — the proven NF/FF encoding + working FarField
  terminal (elts 20-21). PSF forms on load.
- `mmacos/templates/10_telescopes/tma_onaxis/tma_onaxis.in` — add_pupil FP_return/
  ExitPupil/FP triple, but GEOMETRIC (not the diffraction terminal).

## Gotchas already burned (don't rediscover)
- `.in` regexes MUST be line-anchored: `iElt` is a substring of `psiElt`, so an
  unanchored `iElt=\s+\d+` clobbers the leading digit of psiElt. Fixed in
  `ctb_prop_layout.m` (set_prop/set_z/renumber all use `^\s*` + `'lineanchors'`).
- ~~`dx_at` at an NF2 focal plane returns garbage (e.g. 2.6e26) — cosmetic readout,
  field still propagates. Use `abs()` and don't trust dx at NF2 planes.~~
  RETRACTED (SESSION 4 finding E.2): an artefact of the superseded
  through-focus construction. On the committed quartet `dx_at(FPM)` matches
  the deterministic `lam*R/(N*dx_sphere)` to ratio 1.0000.
- MATLAB on this box: `/Applications/MATLAB_R2024a.app/bin/matlab`; the mmacos MEX
  is `mmacos.mexmaca64`; `MACOS_HOME` set. `startup.m` prints a harmless pyenv
  error. Run sandboxed for license.
- CLI exe (if needed): `/Users/dcr/dev/macos/build_release_gfortran/bin/macos`.
- The two entries above are the Mac box. On the Linux box: `matlab` on PATH,
  the mex is `mmacos.mexa64`, the CLI is
  `~/dev/macos/build_release_gfortran/bin/macos`.
- Some installed mexes predate the `'plane'` argument on
  `macos.complex_field`; the veneer passes three arguments and errors against
  them. Fall back to `mmacos('complex_field', iElt, reset)` — what
  `ctb_proper_compare` and `tCtbProp` do.
