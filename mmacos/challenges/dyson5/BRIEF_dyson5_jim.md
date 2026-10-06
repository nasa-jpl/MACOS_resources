# dyson5 round 3 — Jim's comparison, and "do better than Fresnel"

Written for: Dave (round 3 of `macos/BRIEF_ccmac_dyson_size.md`; for Jim).
CCMac, 2026-10-02, `dev-candidate`. New files only: driver `dyson5_jim.m`,
records `dyson5_jim_3a.txt` / `dyson5_jim_3b.{txt,mat}`. Engine scores (7×7 grid)
on the just-rebuilt gfortran engine (`e43b126`); gates re-checked green
(tSpectrometerRx 7, tGratingImmersed 4, tAsphCalib 2). CaF₂ is **not priced** —
the single-crystal carve *volume* is stated for Jim to price.

## 3a. Two bigger-CaF₂ spectrometers vs four smaller-silica ones

Jim's question: is **four** telescopes + spectrometers (1.5k px, 54→27 mm slit,
fused-silica, no meniscus) better than **two** (3k px, 54 mm slit, bigger CaF₂, no
meniscus)? Every row engine-scored, re-cut from the round-2 trade:

| configuration | r = thick | CRF | EE | crossings / thru | glass / module | **system glass** | single-crystal CaF₂ |
|---|---|---|---|---|---|---|---|
| **2 × CaF₂, 3k, 54 mm** (F) | 240 mm | 1.213 | 0.824 | 4 / 0.881 | 4.49 L / 14.3 kg | **9.0 L / 28.5 kg** | **12.4 L** (carve) |
| **4 × SiO₂, 1.5k, 27 mm** (D) | 130 mm | 1.026 | 0.999 | 4 / 0.872 | 0.75 L / 1.6 kg | **3.0 L / 6.6 kg** | n/a (melt) |
| — 4 × SiO₂ small end | 100 mm | 1.247 | 0.797 | 4 / 0.872 | 0.43 L / 0.9 kg | 1.7 L / 3.8 kg | n/a |
| — 2 × CaF₂ for EE ≈ 1 | 300 mm | 1.022 | 1.000 | 4 / 0.881 | 7.43 L / 23.6 kg | 14.9 L / 47.3 kg | 19.6 L |
| *ref* SiO₂ 3k 54 mm, **4 mm meniscus** | 220 mm | 1.327 | 0.759 | **8 / 0.760** | 3.74 L / 8.2 kg | 7.5 L / 16.4 kg | n/a |
| *ref* CaF₂ 1.5k 27 mm, small | 80 mm | 1.024 | 0.995 | 4 / 0.881 | 0.28 L / 0.9 kg | 1.1 L / 3.5 kg | 2.3 L |

Counts: **2× system** = 2 telescopes / 2 gratings / 2×3k detectors; **4× system**
= 4 / 4 / 4×1.5k (same total pixels). (Carve = single-crystal CaF₂ cylinder, clear
diameter + 20 mm by thickness + 20 mm — the blank a lens is ground from. Fused
silica is a melt: no carve, any size. Grating diameter and length are in
`dyson5_jim_3a.txt`. The throughput column counts the BLOCK's air-glass crossings;
a cold detector adds a dewar window — two more crossings — to every row equally, so
the relative comparison is unchanged. See 3b route 1 for the window and its cementing.)

**On the spectrometer side the four-silica architecture wins on glass, decisively.**
Equal throughput (both 4 uncoated crossings, ~0.87–0.88) and equal-or-better image
(silica-130 EE 0.999 vs CaF₂-240 EE 0.824 — matching CaF₂'s EE needs 300 mm blocks),
but **6.6 kg of fused silica against 28.5 kg of CaF₂ requiring 12.4 L of single
crystal**. Fused silica is cheap and makes to any size; the CaF₂ price, Jim says,
grows faster than that 12.4 L. The cost of four is entirely in the **telescope and
detector count** (2 → 4 of each, though the same total pixels) — the trade Jim owns.
Both no-meniscus options beat the silica-with-meniscus reference on throughput
(0.87 vs 0.76) because they drop the meniscus (round 2).

## 3b. "Maybe you and AI can do better" — three measured routes

Baseline: the 130 mm silica, 27 mm-slit, no-meniscus block (D), uncoated **0.872**
over 4 air-glass crossings.

**1. Cement the detector window — the real crossing lever.** (Corrected count.)
Depositing the slit on the face removes *no* crossing: the beam arrives in air and
enters the glass at the slit plane whatever carries the mask (that only removes the
mask's mechanical standoff). The lever is the **detector window** — a cold detector
sits behind a dewar window, so the honest uncoated baseline is block-exit + window-in
+ window-out = three crossings on the detector side, **six in all, 0.81**. **Cementing
the window to the block** index-matches the block-exit/window-in pair → **6 → 4
crossings, 0.81 → 0.87**, with the ray geometry (and so CRF 1.026, EE 0.999, clearance
+0.38 mm) unchanged. AR-coating the convex pair then reaches **~0.91**. A bare "2
crossings" is not reachable with a cold detector behind a window.

**2. A broadband AR coating — modest over a 6:1 band (+4%).** Scored with the
engine's Abeles stack (`macos.design.thinfilm_rt`), normal incidence, 400–2500 nm;
indices (refractiveindex.info, representative, mild dispersion neglected): MgF₂
1.384, Al₂O₃ 1.63, Ta₂O₅ 2.10, substrate silica 1.450 / CaF₂ 1.429.

| coating (4 crossings) | silica | CaF₂ |
|---|---|---|
| uncoated | 0.872 | 0.881 |
| single λ/4 MgF₂ | 0.903 | 0.902 |
| optimised 2-layer (MgF₂/Al₂O₃) | **0.908** | 0.906 |

A simple AR buys **~+4%**, far less than route 1's +7%, because a 6:1 band is
extreme for an anti-reflection stack (a V-coat is narrowband; broadband ARs are
routine over ~2:1, not 6:1). A high/low stack built on **SiO₂** as the low layer is
useless — SiO₂'s index ≈ the silica substrate, so it does nothing and the high
layers only add reflection. The levers combine with route 1: cement the window
(6 → 4 crossings) **and** AR the convex pair → **~0.91**. The honest ceiling for a
few-layer AR here is ~0.92 per four crossings; graded/9-layer BBAR would do better
but is its own program.

**3. The working distance — the small module is forgiving.** Scanning the slit and
detector standoff together (R3 re-solved at each), 130 mm silica, 27 mm slit:

| standoff | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 mm |
|---|---|---|---|---|---|---|
| CRF | 1.025 | 1.027 | 1.037 | 1.065 | 1.163 | 1.355 |
| EE | 1.000 | 1.000 | 1.000 | 0.905 | 0.717 | 0.484 |
| clearance | +0.43 | +0.86 | +1.32 | +1.80 | +2.25 | +2.70 mm |

**Up to ~1.5 mm of working distance is essentially free** (CRF ≤ 1.04, EE 1.0), and
clearance *grows* with it; 2 mm is acceptable (CRF 1.065, EE 0.905); the image goes
past ~2.5 mm. This is far more forgiving than the full 54 mm module (R5's sweep:
~0.5 px per mm). So at the two-module scale the working distance is **not** where
we are stuck — the ~1.5 mm the mask + window + a cold-shield lip would want is in
hand without asking Jim for his tricks.

## 3c. The Offner at F/1.8 and the 54 mm slit (addendum 46, 2026-10-06)

The all-reflective alternative, tried at the Dyson's own speed and slit: 3000 x 500 px
at 18 µm, 380–2500 nm, 2-px slit, F/1.8, chord-ruled grooves (as the engine), equal
field weights, scored by the engine on the s2 scorer (7 slit positions x 7 wavelengths).
Stage `o18` (`dyson5_off18.m`), record `dyson5_off18.txt`.

**Conventions first.** CRF/SRF are FWHM in pixels; the scorer's window is ±8 px, so
">15" means the window clipped and the rms blur is the number (u across the slit, v
along the dispersion, max over the grid). Clear aperture = footprint + 5 mm; blank =
clear + 5 mm mount; mass = blanks at 10 mm Zerodur-class (2530 kg/m³), an assumption,
not a lightweighted design. The Dyson rows carry their edged glass plus their grating
at the same blank rule.

**Three things the run found before the optics did.**
1. *The 0.22 R ring is F/2.8's rule.* At F/1.8 the grating (the stop) is R/(2F) =
   0.28 R across, and at 0.22 R both beams pass THROUGH it (−33 / −47 / −61 / −75 mm at
   R 0.5 / 0.75 / 1.0 / 1.25 m). The smallest ring that clears it by +5 mm is 0.28–0.29 R.
2. *The clearance gate could not see it.* It measured each leg's distance to body
   points sampled every 2 mm, which cannot go negative for a mirror or grating disc,
   so a beam straight through the grating read +0.2 mm. `spectrometer_clearance` now
   tests every segment against the surface inside its aperture + mount. All 54 Dyson
   records (6 ladder rungs, 48 size-trade rows) re-gate bit-identical; gate
   `tSpectrometerRx/test_clearance_sees_a_beam_through_a_body` (must-fail: F/1.8 at
   0.22 R; must-pass: 0.30 R and the F/2.8 sibling). A second latent bug in the same
   tool: since beat 4d it crashed on any form without box bodies (the Offner).
3. *The two concave zones cannot overlap once their figures differ.* At the fixed ring,
   the classical corrections give the second zone its own radius while the two clear
   apertures still share 25–35 mm: not buildable. The free-ring solve (the ring a
   variable, bounded below by the clearing ring; CC's addition) carries a wall on the
   zone gap and ends with the zones touching. The F/2.8 record's own zones overlap by
   9.4 mm of clear aperture (the bare footprints clear by +0.6 mm), which is marginal.

**The table** (engine; the "free" rows are the buildable ones):

| form | R or block r (mm) | ring | smile (px) | keystone (px) | CRF (px) | SRF (px) | EE | length (mm) | largest optic (mm) | mass (kg) | clearance, worst pair |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dyson, CaF₂, no meniscus | 240 | – | 0.005 | 0.006 | 1.21 | 2.02 | 0.82 | 787 | 318 | 16.4 | +0.54 slit mask / block face |
| Dyson, silica + meniscus (R4) | 220 | – | 0.005 | 0.003 | 1.33 | 2.03 | 0.76 | 694 | 278 | 9.9 | +0.79 slit mask / block face |
| Offner seed | 500 | 0.29 R | 0.157 | 0.184 | >15 (rms 34) | >15 (rms 46) | 0.00 | 504 | 641 | 8.9 | +5.5 M3→FPA / grating |
| Offner free | 500 | 0.32 R | 0.165 | 0.177 | 2.24 | >15 (rms 6.9) | 0.01 | 501 | 325 | 4.8 | +6.9 M3→FPA / grating |
| Offner free | 750 | 0.30 R | 1.206 | 0.038 | 9.60 | >15 (rms 8.9) | 0.01 | 755 | 462 | 9.9 | +16.8 slit→M1 / grating |
| Offner free | 1000 | 0.29 R | 0.672 | 0.049 | 2.50 | >15 (rms 7.2) | 0.01 | 1005 | 593 | 16.0 | +9.1 M3→FPA / grating |
| Offner free | 1250 | 0.29 R | 0.627 | 0.031 | 3.76 | >15 (rms 8.3) | 0.01 | 1255 | 728 | 24.1 | +9.0 M3→FPA / grating |

(The seed and fixed-ring rows at every R are in `dyson5_off18.txt`. Corrected rows CRF
2.2–10.3 px, SRF window-clipped everywhere, energy in a pixel ≤ 0.014.)

**Where the spectral blur comes from.** It is 6.6–8.9 px rms at every R and every
correction step, so it is not the mirrors' scale. Per wavelength (free rows) it is a
λ-independent floor, 3.3 / 5.3 / 4.8 / 6.5 px at 380 nm, plus a part that grows with λ
(+3.6 px at R 0.5 down to +1.8 px at R 1.25 by 2500 nm). The first is aberration in the
dispersion direction that the four classical corrections and the ring do not remove
(they buy CRF, not SRF). The second is the grating at F/1.8: the curved-surface groove
law makes it larger (9.6 vs 6.8 px at R 0.5, 2500 nm), so chord ruling is already the
better of the two. Reaching SRF < 3 px needs the spectral rms near half a pixel, 7–15x
below what the solve reaches.

**Stop rule applied.** Step 2 does not bring SRF under 3 px at any R ≤ 1.25 m, so per
the brief the run stops there: the conic-per-zone step (3) was not run, and the e2e
join (optional) was not attempted. The next form, if Jim and Joe want the
all-reflective route at F/1.8, is the Offner–Chrisp (a separate M3 with its own
freedoms), or a grating with designed groove spacing (e-beam), which addresses only
the λ-proportional part.

**For Dave.**
- *Does the F/1.8 Offner reach the Dyson rows?* No. The best buildable point (R 0.5 m,
  ring 0.32 R) reaches CRF 2.24 px, but its spectral blur is 6.9 px rms (SRF beyond the
  scorer's window, against the Dyson's 2.02–2.03) and almost no energy lands in a pixel
  (0.01 against 0.76–0.82). A larger R does not help: the spectral blur stays 7–9 px.
- *What it costs in size:* at R 0.5 m it is the Dyson's length (501 mm vs 694–787) with a
  325 mm concave blank and a 154 mm grating, 4.8 kg of 10 mm blanks, so it is not size
  that rules it out. The negative is at F/1.8; the review's Offner holds its SRF only
  at F/2.8, and the record's own F/2.8 sibling already sits at SRF 3.69 px.

Records: `dyson5_off18.{txt,mat}`, the decks `dyson5_off18_R<mm>_{seed,corr,free}.in`
with their `_maps` / `_layout` figures, the engine renders
`dyson5_off18_R050_free_{view3d,viewyz}.png`, the solves
`dyson5_off18_R<mm>[_free]_s1_offner_solve.txt`.

## Bottom line for Jim

- **Four small silica spectrometers** (1.5k, 27 mm, no meniscus) are the glass win:
  6.6 kg of cheap fused silica vs 28.5 kg of CaF₂ (12.4 L single crystal), same
  throughput, better image. The price is two more telescopes and detector chains.
- On "better than Fresnel": with the cold detector's dewar window counted, the
  uncoated chain is 6 crossings / **0.81**. **Cementing the window** to the block
  takes it to 4 / **0.87** (depositing the slit removes nothing — the beam enters
  glass at the slit in air regardless); a broadband AR on the convex pair adds the
  rest to **~0.91** (a 6:1 band is extreme for an AR). The small module also holds
  ~1.5 mm of working distance, so that is not binding here.

Records: `dyson5_jim_3a.txt` (the table + grating/length geometry), `dyson5_jim_3b.txt`
(the three routes), `dyson5_jim_3b.mat`. Driver `dyson5_jim.m` (`'3a'` re-cut, `'3b'`
the routes). CaF₂ left unpriced.
