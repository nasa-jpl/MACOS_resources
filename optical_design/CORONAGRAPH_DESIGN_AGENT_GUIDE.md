# Working With the Coronagraph Design Rules & Layout Generator (agent guide)

This file tells a coding agent (Claude Code) how to use the coronagraph back-end material
in this repo. Read it before touching coronagraph layout code, before consuming
`coronagraph_layout.json`, and before quoting any number from the generator as a spec.

Companion to `OPTICAL_DESIGN_AGENT_GUIDE.md` (telescope side). Same discipline: the
biggest failure mode here is **treating a reference scale or a first-order scaffold as a
buildable specification.** Do not do that.

---

## 1. What these files are

| File | Role | Use it to… |
|---|---|---|
| `CORONAGRAPH_DESIGN_RULES.md` | **why/how** | get the architecture, plane sequence, OAP-relay rules, two-DM/Talbot theory, FPM/Lyot/apodizer rules, contrast-limiting effects |
| `coronagraph_layout.py` | **first-order layout generator** | turn `(D_beam, N_act, λ, IWA/OWA, F/#s, pixel)` into OAP focal lengths, pupil sizes, DM separation scale, FPM scales, Nyquist F/# |
| `coronagraph_layout.json` | **sample output** | an example layout; **not** a regression fixture (see §3) |
| `coronagraph_layout.md` | **usage notes** | invocation, conventions, the quarter-Talbot caveat |

Read the `.md` files when reasoning; call `design()` in the `.py` when you need a layout;
do **not** treat the `.json` as golden truth.

---

## 2. The quarter-Talbot number is NOT a buildable separation (read first)

The generator emits `quarter_talbot_sep_mm`. **Do not hardcode it as the DM1–DM2
separation.** It is a reference *scale*, and it is frequency-dependent:

- `z_qT(α) = Λ²/(2λ)` with `Λ = D_beam/α` — this is the distance at which a phase ripple
  of period `Λ` fully converts to amplitude (`= Z_T/4`).
- Because `z_qT ∝ (1/α)²`, there is **no single distance** that satisfies the
  quarter-Talbot condition across the dark-hole band: it is ~metres at the OWA and
  ~hundreds of metres at the IWA. The generator reports it at IWA / band-mid / OWA
  precisely to make that spread visible.
- The **real operating separation** (~1 m for HCIT/Roman-class beams) is a **numerical
  optimization** across the band, further constrained by packaging and beam-walk, and
  sits *below* the quarter-Talbot-at-OWA value.

**Rule:** carry this field as `dm_sep_upper_scale`, never as `dm_sep`. The operating
separation is pinned down in the diffraction/control model (FALCO / end-to-end), not here.
If you find yourself writing the printed number into a layout as the DM spacing, stop.

---

## 3. This is first-order layout, not a diffraction model — and not a fixture

Distinguish this clearly from the telescope deliverables:

- The telescope fixtures (`tma_fixture.json`, etc.) carry **machine-zero residual
  self-checks that prove correctness**. Treat them as golden regression targets.
- `coronagraph_layout.py` self-checks are **constraint checks only** — OWA reachability
  (`OWA ≤ N/2`), Nyquist (`λ·F# ≥ 2·pixel`), positivity, `IWA < OWA`. They verify the
  request is *consistent*, not that an output is *correct against ground truth*.
- Therefore **do not** treat `coronagraph_layout.json` as a regression fixture, and **do
  not** expect contrast, dark-hole depth, mask transmission profiles, polarization
  aberration, or EFC results from this module. Those live in the diffraction/control
  layer (FALCO, HCIPy, PROPER). Asserting coronagraph *performance* against this geometric
  scaffold is a category error.

What this module is good for: sizing OAP focal lengths, pupil diameters at each conjugate,
focal-plane physical scales, the Nyquist final F/#, and bounding the DM-separation
envelope. Stop at first-order layout.

---

## 4. Conventions (fixed — do not reinterpret)

- **Units differ from the telescope modules.** This module is **mm + nm**; the telescope
  generators are in **metres**. The only safe chaining point is `input_fnum`
  (dimensionless) — feeding the TMA's output F/# in is fine. If you pass any *physical
  length* between the telescope and coronagraph layers, convert m↔mm explicitly; a silent
  unit mix is the obvious footgun.
- **λ/D means the telescope aperture, not the instrument beam.** Working angles are
  dimensionless λ/D referenced to the telescope D. This module is **aperture-agnostic** —
  it never sees the telescope diameter. The focal-plane scales it emits (`α·λ·F#`) are
  internally self-consistent; if you independently convert λ/D to millimetres you must use
  the **telescope** aperture, which is not in this module. Do not substitute `D_beam`.
- **Relays are unit-magnification by default** → pupil = `D_beam` at apodizer, DM1, Lyot.
  Real instruments magnify between relays to match device sizes; magnification = ratio of
  the paired OAP focal lengths. Do not assume the pupils are *necessarily* equal — that's
  a default, not a constraint.
- **Talbot form.** The code uses `z = Λ²/(2λ)` directly to avoid the `Z_T = 2Λ²/λ` vs
  `Z_T = Λ²/λ` convention ambiguity. If you rewrite it via a looked-up "Talbot length,"
  you can pick up a factor-of-2 error. Keep the conversion-distance form.

---

## 5. Inputs are design knobs, not optima

`input_fnum`, `fpm_fnum`, and `pixel_um` are **design inputs with placeholder defaults**
(F/20, F/30, 13 µm), driven in reality by packaging, mask fabrication limits, and the
LOWFS/detector. Do not treat the defaults as recommended values, and do not "optimize"
them without the requirements that actually set them. `talbot_design_lod` defaults to the
OWA; changing it changes only the reported scale, not a physical optimum.

---

## 6. Hard rules / what not to do

- Do **not** promote `quarter_talbot_sep_mm` to the DM separation (§2).
- Do **not** treat `coronagraph_layout.json` as a golden fixture or assert performance
  against it (§3).
- Do **not** mix mm and metres across the telescope/coronagraph boundary (§4).
- The `assert`s vanish under `python -O`. If you gate CI on the constraint checks, convert
  them to explicit `raise` / `sys.exit(1)` so the gate can't be silently optimized away
  (same note as `seidel.py`).
- Do **not** assume a returned layout is buildable: the module does not check beam-walk
  clearance, mount collisions, total track length, or that OAP off-axis angles are small
  enough for the anti-symmetric-pair aberration cancellation (`CORONAGRAPH_DESIGN_RULES.md`
  §3) to hold. Those are downstream checks.

---

## 7. Boundary with the diffraction/control layer

This module fixes the first-order *layout* the controller assumes: pupil conjugation, DM
count, the DM-separation envelope, dark-hole reach (`OWA = N/2 · λ/D`), and the relay/
focal-plane scales. It does **not** specify: apodizer+Lyot joint optimization, FPM
profiles, polarization aberration budgets, or the wavefront-control algorithm (EFC, stroke
minimization, speckle nulling; LOWFSC + HOWFSC). Per the project plan, actuator-level EFC
integrates via FALCO; modal DM control is the inner loop. Keep that split: layout here,
contrast there.

---

## 8. Suggested CLAUDE.md pointer (keep CLAUDE.md lean)

```
## Coronagraph instrument layer
- Design rules: docs/CORONAGRAPH_DESIGN_RULES.md
- Agent guide (read before editing coronagraph layout): docs/CORONAGRAPH_DESIGN_AGENT_GUIDE.md
- Layout generator: tools/coronagraph_layout.py  (first-order layout only; NOT a fixture)
- HARD RULE: quarter_talbot_sep_mm is a reference scale, never the DM separation.
- Units here are mm+nm (telescope layer is metres); only chain via dimensionless F/#.
- Contrast/EFC/masks live in the diffraction layer (FALCO), not here.
```
