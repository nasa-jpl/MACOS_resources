# BRIEF: ctb_dst_defects — a hardware-faithful VVC in the CTB model

**Goal.**  Implement the DST's measured vector-vortex mask defects
(Llop-Sayson et al. SPIE 2024; extraction in
`bench_ctb/DST_VVC_EXTRACT.md`) as options in the CTB chain, build
composite models of their two real masks (Record Holder, Second Best),
and validate by reproducing their published narrowband and broadband
EFC floors.  This turns the CTB from an idealized testbed model into a
hardware-faithful one, and is the paper's own named future work ("a
full model based on the measurements presented here").  Scope: S0–S3
below; the estimation / knowledge-error phase (S4 sketch) is a
FOLLOW-ON brief gated on S3.

**Posture.**  Control ≡ truth throughout this brief (perfect sensing)
— the defects go into BOTH the plant and the Jacobian, except where a
defect is by nature unknown to the controller (stated per item).  The
Marx truth/control vocabulary labels every run.

---

## S0. Digitize the published curves (ground truth tables)

Pull the figure bitmaps (`pdfimages`) and hand-pick points (~10–15
per curve) into committed CSVs under `bench_ctb/dst_curves/`:

- `retardance_vs_lambda_{rh,sb}.csv`  (Fig. 7)
- `axis_error_vs_sep_{rh,sb}.csv`     (Fig. 9)
- `reflectance_vs_lambda_{rh,sb}.csv` (Fig. 10)
- `floor_vs_lambda_nb_{rh,sb}.csv`, `floor_vs_lambda_bb_{rh,sb}.csv`
  (Figs. 13, 15 — the VALIDATION TARGETS, never inputs)

2-D maps (Figs. 5, 6) are NOT scraped — spatial character is
parameterized (RH: uniform speckle ±0.3°; SB: central radial dip to
~175°).  GATE S0: overlay plot of digitized points on the paper
figures, eyeballed clean.

## S1. Charge-6 baseline on DST-matched scoring

- Add/confirm vortex charge as a first-class option (DST masks are
  charge 6; our record is charge 4 — parity changes pol behavior).
- Add the DST scoring protocol beside ours: HALF-PLANE dark hole
  3–8 λ/D, Lyot 0.8; report both it and our 3–15 annulus.
- Run the IDEALIZED charge-6 floors: mono, 1% narrowband sweep
  610–660, 10% broadband @ 625 — the reference every defect run
  compares against, and the "our idealized system" row of the final
  table.  GATE S1: baseline committed with la diagnostics; expected
  decades below their measured floors.

## S2. Defect classes — one option + one gate each (independent lanes)

**2a. Bulk retardance ε(λ, r)** — per-λ retardance from the Fig.-7
CSV per mask, times the spatial pattern; circular analyzer with
extinction 1e4.  Existing sandwich/Jones machinery (physical-thickness
chromaticity already correct).  *Controller knows the design (180°),
not the error.*  GATE: the paper's rule reproduced as a curve —
retardance error sweep vs floor crossing 1e-9 at ~1°; RH's
concentric-ring signature in the mono hole.

**2b. Fast-axis error θ_err(r,φ)** — radial envelope from the Fig.-9
CSV × azimuthal mix (random f≤20 + slope term), amplitude tuned so
≲1% modal power sits outside the charge mode (their bound).  GATE:
modal decomposition of the generated mask matches their statistics;
mode-0 leak measured and reported un-analyzable (co-polarized with an
off-axis source).

**2c. Central defect + inclusion** — amplitude disk 0.2 λ/D at center
(their 4 µm at F/32.7) + a point inclusion at ~5 λ/D; FPM-plane
amplitude multiply.  GATE: their signature — mono nearly unaffected,
broadband floor degrades (chromatic speckle in the hole).

**2d. Ghosts** — incoherent shifted PSF at 12 λ/D scaled by the
Fig.-10 chromatic reflectance (COMPOSE/ADD path), plus the weaker
back-substrate Lyot-ring variant.  *Never in the Jacobian —
incoherent.*  GATE: SB's ~1e-9 narrowband ghost floor reproduced with
their reflectance curve.

**2e. DM command quantization** — LSB step on DM commands; sweep step
size to bracket their ~3e-11 modulated floor (step value INFERRED,
flagged — we lack Ruane 2020's hardware number).  GATE: repeated EFC
runs give run-to-run random speckle at the quantization floor
(their 4-run ±0.16e-10 behavior).

## S3. Composite mask models + the DST protocol (the headline)

Assemble **RH model** (2a-RH + 2b-RH + 2c full + 2d-RH) and **SB
model** (2a-SB + 2b-SB + 2c dot-only + 2d-SB with 12 λ/D ghost).  Run
the DST protocol on each: mono EFC (4 repeats), 1% sweep 610–660,
10% @ 625, half-plane 3–8 λ/D, modulated/unmodulated split reported
per run (in the model both components are known exactly).

**GATE S3 (success criteria, stated before running):**
1. ORDERING: RH beats SB in mono/narrowband; the two converge in
   broadband (their central result).
2. MECHANISMS: RH narrowband limited by retardance leak, SB by the
   ghost; RH broadband degrades ~5× from mono, SB stays flat.
3. FLOORS: within ~2–3× of the Fig. 13/15 digitized curves (honest
   target — their optics' WFE screens and exact probe/estimator noise
   are not in our model in this brief).
4. Every mechanism attribution is falsifiable by switching that
   defect off (non-vacuity: each defect's removal moves the floor).

Deliverables: LOG sections per stage; `ctb_vvc` option surface
documented; summary figure floor-vs-λ (ours vs digitized theirs, all
four curves); the "idealized vs RH-model vs measured-DST" three-row
table per band.

## S4 (sketch — FOLLOW-ON brief, gated on S3)

Pairwise-probing estimator through the truth model (first
truth ≠ control beyond CF5b); chromatic control residual demo
(narrow-subband control model vs broadband truth); DST-style periodic
β-bump schedule; optionally actual FALCO as the control model (folds
into the queued FALCO integration).  Success = the Marx diagram made
quantitative: floor decomposed into estimation / model error /
incoherent.

---

## Decisions for Dave (recommendations first)

1. **Scope cut at S3** (S4 = follow-on brief)?  Recommend YES.
2. **Charge 6 added** for the validation runs (charge 4 kept as our
   record's baseline)?  Recommend YES — parity matters.
3. **DST half-plane scoring** as the primary metric for these runs,
   annulus secondary?  Recommend YES.
4. **Machine assignment**: S3's sweeps (2 masks × ~11 center-λ × EFC
   runs, N=512 per-λ Jacobians, cached) are the compute bulk —
   candidate for CCMac's Mac (18×, parity-gated) while this box does
   S0–S2.  Recommend YES once S2 gates pass here.

Rough effort: S0 ~1 h; S1 ~half day (Jacobian measures dominate);
S2 lanes ~half day each, parallelizable; S3 = the compute night(s).
