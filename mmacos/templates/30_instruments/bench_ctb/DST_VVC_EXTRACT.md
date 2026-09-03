# DST / VVC extraction — Llop-Sayson et al., SPIE 13092, 130921Y (2024)

Source: `macos/demo_session/responses/Jorge_VVC_SPIE_2024.pdf` (via David
Marx, 2026-09-02).  "Vector Vortex Coronagraph Experiments in Vacuum
Towards 1e-10 Contrast" — Llop-Sayson, Ruane, Serabyn, Mejia Prada,
Walter, Allan (JPL/HCIT).  Purpose here: (1) the DST hardware-flaw
inventory as CANDIDATE CTB MODEL FEATURES, (2) the DST layout + EFC
practice, (3) the honest comparison against our idealized CTB vortex
results.

## 1. The flaw inventory (their Fig. 1 taxonomy + measurements)

Two masks characterized: **Record Holder Mask** (RH) and **Second Best
Mask** (SB) — limited by DIFFERENT phenomena, comparable in broadband.

### 1a. Vortex shape errors (fast-axis orientation errors)
- Two consequences: MODEL error → *chromatic control residual* (the
  controller's vortex ≠ the real one; grows with bandwidth), and
  incoherent ZEROTH-ORDER leakage (non-coronagraphic PSF, same pol
  state as an off-axis source → the analyzer CANNOT strip it, DMs
  cannot control it).
- Their modal test profiles (charge 6): random Gaussian σ=π/40 with
  azimuthal frequency ≤ 20; charge-slope error of amplitude π/40;
  sinusoidal error π/40 at frequency 5.  Mode 0 is the most pernicious
  (zeroth-order leak); odd modes leak past the Lyot; even modes null at
  the Lyot in theory but modes below the design charge amplify
  low-order sensitivity.
- MEASURED (their Fig. 9, digitizable line plot): fast-axis orientation
  error vs separation — RH ≈ 0.133 rad @ 5 px falling to ≈ 0.035 @ 50
  px; SB ≈ 0.153 → 0.020 rad.  Character "between random and uniform",
  > π/40 only near the center; their upper bound: ≲ 1% of modal power
  outside the charge mode.

### 1b. Polarization leakage (bulk retardance error)
- Leak intensity |c_L|² ~ (ε_r/2)²; with a circular-analyzer extinction
  of ~1e4, **retardance better than 1° is needed to reach better than
  1e-9** — their stated rule.
- MEASURED (Fig. 6 maps at 650 nm; Fig. 7 curves, digitizable): RH mean
  retardance peaks 179.4° near 630 nm, < 1° error over a wide band
  around 600–650; SB 177.9–179.1° in-band, with a spatial cross/spiral
  pattern dipping to ~175–177° near center (RH map is uniform speckle).
- RH's mono/narrowband floor is polarization-leakage-limited (concentric
  uncorrected rings in the dark-hole image).

### 1c. Inclusions and imperfections near the center
- Amplitude errors at/near the focal plane = strongly CHROMATIC speckle,
  not fully correctable in broadband → incoherent light in the hole.
- MEASURED: RH has a **central defect ~4 µm** plus an inclusion at a
  separation near the working angles (Nikon LV100, 0.12 µm/px, all
  three mask planes imaged).  At their F/32.7 and 625 nm, λF# = 20.4
  µm, so 4 µm ≈ **0.2 λ/D — exactly the size they call severely
  limiting**.  SB is clean at the vortex plane (small center dot only).
  LCP photoalignment intrinsically fails at the vortex center — a
  central defect always exists at some size.

### 1d. Ghosts from internal reflections in the mask sandwich
- Two substrate reflections: ghost 1 (upstream substrate) does NOT see
  the vortex → off-axis-PSF-like, the dangerous one; ghost 2 (back
  substrate) passes the vortex twice → "ring of fire" at the Lyot,
  weaker.  The FPM is TILTED against the chief ray to dump the
  first-surface reflection.
- MEASURED (Fig. 10 reflectance proxy = WFS-camera/science-camera pupil
  intensity ratio, digitizable): RH ~0.5–2% falling with λ; SB 1–6%
  with strong chromatic structure (AR-coating quality difference).  SB's
  narrowband floor ~1e-9 is a ghost at **~12 λ/D south of center**.

### 1e. DM electronics noise
- The modulated-component floor in mono (~3e-11, speckly, random across
  four repeated runs) is attributed to DM least-significant-bit
  quantization noise (Ruane et al. 2020, JATIS 6, 045002).

## 2. DST layout + EFC practice

**Train** (their Fig. 4, from Ruane 2022; full design: Patterson et al.
2019, SPIE 11117): supercontinuum NKT SuperK EXTREME + VARIA tunable
filter (~1% BW) — mono runs use a Thorlabs LP637-SF70 diode laser —
fiber through vac/air feedthrough; source assembly LP (Thorlabs
LPVIS-100) + achromatic QWP (Tower Optical, 25.4 mm) = **circular
polarizer**; OAP relay → **DM1, DM2 = two Northrop Grumman AOA Xinetics
48×48**; OAPs → **FPM at F/32.7** → OAP → **Lyot stop = 80% of beam
diameter** → field stop → OAPs → **QWP + LP circular analyzer**
(extinction ~1e4 assumed) → Andor Neo 5.5 camera.  Vacuum ~0.1 mTorr
held for weeks; thermal stability ~10 mK over 24 h.

**EFC practice**: FALCO (Riggs 2018; github ajeldorado/falco-matlab);
field estimation by **pairwise probing**; regularization run with a
**β-bumping schedule** (Sidick 2017) — the evenly spaced intensity
jumps in every EFC-vs-iteration curve; **validation by four repeated
EFC runs from a flat-DM start** (2.44 ± 0.16e-10 mono);
modulated/unmodulated decomposition as the standard limit diagnostic;
narrowband ~1% sweeps of center wavelength 610–660 nm; broadband 10% at
625 nm; **half-plane dark hole 3–8 λ/D**; program target 1e-10 @ 10%.

## 3. Their results vs our idealized CTB chain

Their numbers: mono VVC record **2.44 ± 0.16e-10** (RH; pol leakage +
DM LSB ~3e-11 modulated); narrowband 1%: RH ~4.3–11e-10 across 615–660,
SB 1.2–1.7e-9 (ghost-limited); broadband 10% @ 625: RH ~1.5e-9, SB
~1.9–2.8e-9; best DST VVC broadband ever 1.4e-9 (Ruane 2022); the
overall DST record 4e-10 @ 10% is a CLASSICAL LYOT, not a VVC.

Ours (CTB_PROP_STATUS, charge-4 sandwich, N=512, annulus 3–15 λ0/D,
perfect sensing, control ≡ truth): mono pol-only **5.77e-13** (pol
floor 1.07e-15); 10% band + pol @ Lyot 0.60 **1.96e-11** relin; @ Lyot
0.80 (≈ their Lyot geometry, T 64%) **1.21e-10**.

**Verdict: we do not "match" them — we undercut them, by exactly the
amount their non-idealities explain.**
**RESOLVED (CCMac S1, 2026-09-03, commit 9092766): the matched-config
N=512 floor (~4e-10, only 1.6× below DST) was a COARSE-GRID VORTEX-CORE
ARTIFACT — the suspected Session-11/12 sampling effect, confirmed by
probes.  At N=1024 the idealized matched-config baseline (charge 6 /
Lyot 0.80) is DEEP: annulus 3.33e-14, half-plane 3–8 λ/D **2.04e-14 ≈
12,000× below DST's 2.44e-10**.  Rule for this campaign: every
matched-config defect run is N=1024; N=512 charge-6/open-Lyot numbers
are not quotable.**
- Mono: ours 5.8e-13 vs their 2.44e-10 → **~420× deeper**, consistent
  with their own attribution (retardance leak + DM electronics — both
  ABSENT in our ideal model; our pol floor 1e-15 is the coated-train
  residual only).
- 10% band at the matched Lyot: ours 1.21e-10 vs their ~1.5e-9 →
  only **~12×** — because chromaticity limits BOTH systems and is
  physics, not hardware.  Their "chromatic control residual" is the
  hardware-error twin of our measured "one DM setting cannot null
  three wavelengths."
- Regime caveats for any quoted comparison: charge 4 (ours) vs 6
  (theirs); full annulus 3–15 (ours) vs half-plane 3–8 λ/D (theirs);
  perfect sensing (ours) vs pairwise probing; 880-active DMs vs 48×48.

## 4. Feature plan — a defected VVC mask in the CTB model

Rather than pixel-scraping the 2-D map JPEGs, parameterize each class
at their measured magnitudes (the four line plots — Figs. 7, 9, 10,
13/15 — digitize cleanly; Fig. 6/5 fix the spatial character):

1. **Retardance-error mask**: per-λ bulk retardance from the Fig. 7
   curves (RH + SB variants) + spatial pattern (RH uniform speckle
   ±0.3°; SB radial dip to ~175–177° at center) — the circular-sandwich
   Jones machinery already carries (ε/2)² leaks and physical-thickness
   chromaticity.
2. **Fast-axis error**: θ_err(r,φ) added to the vortex axis — their
   three modal classes at π/40 plus the measured Fig.-9 radial decay.
3. **Central defect + inclusion**: amplitude blot of 0.2 λ/D at center
   + a point inclusion at a working separation (FPM-plane amplitude
   multiply — existing mask machinery).
4. **Ghosts**: incoherent shifted PSF at ~12 λ/D scaled by the Fig.-10
   chromatic reflectance (COMPOSE/ADD), + the back-substrate Lyot
   ring variant.
5. **DM LSB quantization** on commands (their ~3e-11 modulated floor).
6. **Validation targets**: reproduce their Fig. 13/15 floor-vs-λ curves
   class by class; their 1°-retardance → 1e-9 rule is a free gate.

This is exactly the "full model based on the measurements presented
here" the paper itself names as needed future work — and the
knowledge-error phase Marx flagged the paper as guidance for.
