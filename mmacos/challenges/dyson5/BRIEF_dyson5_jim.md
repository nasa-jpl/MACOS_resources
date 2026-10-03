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
