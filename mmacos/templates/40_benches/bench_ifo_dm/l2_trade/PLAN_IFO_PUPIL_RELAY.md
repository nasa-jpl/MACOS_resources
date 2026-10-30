# IFO detector-leg redesign: better pupil imaging for DM metrology

> Execution plan for an Opus-class session. Self-contained: read this + the two
> example drivers + `+macos/+design/{Bench,twyman_green}.m` and you have everything.
> Branch: **`ifo-l2`** (stacked on `bench-builder`) in MACOS_resources. This work is
> MATLAB-only — **no Fortran, no mex relink**. Do NOT do this work in `pol-core`
> (that is the polarization-exposure lane; the only contact point is the C3
> footnote, deferred there). Phase-1 findings this builds on: CC Fable-5 session
> 2026-07-25, `templates/40_benches/bench_ifo_dm/` + agent memory `project_bench_builder`.

## 1. Problem statement

`templates/40_benches/bench_ifo_dm/example_bench_ifo_dm.m` (phase 1, closed 2026-07-25)
measures a PROPER-modeled DM through a compensated Twyman-Green built by
`macos.design.twyman_green`. Measured error budget, checkerboard pokes
`POKE_NM=50`, `SEED=7`, model 256:

| Term | Value | Nature |
|---|---|---|
| PSI algorithm chain (stepping + deGroot + differential) | **3e-5 pm rms** | machine precision — closed, do not touch |
| Tail-immune measurement vs truth (`PSI_MODE='recomb'`) | **0.402 nm rms** | resampling-limited, linear in poke |
| Physical instrument vs truth (`PSI_MODE='detector'`) | **6.76 nm rms** | **the RETRACE / pupil-imaging term of the singlet-L2 detector leg** — rim-concentrated, linear in poke amplitude |
| Pupil map (det→DM affine) | mag 0.8101, anam 0.000%, rot 180.000°, nonlinear 0.0205 mm rms (0.09 % of beam) | measured per-ray |

The retrace term arises because DM-slope-deflected test rays take slightly
different paths through the tail (Recomb → L2 → FocalMask → Detector) than the
reference sees; a real instrument has exactly this error. **Dave's doctrine
(2026-07-25): reduce it with BETTER OPTICS, not with fancier data processing** —
no higher-order fit terms, no cleverer estimators. Target: physical-instrument
residual **≤ 1 nm** at 50 nm checker pokes (stretch: approach the 0.402 nm
recomb-mode floor). Because the term is linear in poke, report the final number
also as **nm-of-error per nm-of-poke** — the transferable figure.

## 2. Non-negotiable context (learned the hard way — do not rediscover)

1. **Engine field-phase sign:** macos complex-field phase ADVANCES as optical path
   SHORTENS (verified to 1e-4 pm via PZT frame ratios; same convention pymacos↔
   PROPER reconcile with `opd_sign_flip`). The conventions in `psi_run`
   (`th = +((1:7)-4)*pi/2`, `h = +phi*LAM/(4*pi)`, `h_opd = +(opd_pk-opd_b)/2`)
   are the empirically consistent set. **Do not "fix" any sign.** The block
   comments in the driver document this.
2. **±π branch artifact:** never take `std()` of wrapped angles — values
   clustered at ±π (the same physical angle) fake ~0.16 rad of rms structure.
   Always subtract in the complex domain: `angle(exp(1i*(a-b)))`.
3. **PSI ≡ direct field phase** in detector mode, to 3e-5 pm. Metric evaluations
   therefore DO NOT need the 14-frame PSI sequence: use direct differential
   fields (§4). ~10× cheaper per candidate.
4. **The registration refine is similarity-only** (offset+rot+scale). The retrace
   term is nonlinear rim structure — the refine cannot absorb it, so the vs-truth
   metric is fair. Do not extend the refine with higher-order terms (doctrine).
5. **A checkerboard is sign-symmetric**: a value-sign flip is absorbed by the
   orientation scan. Never use the checker to arbitrate sign conventions.
6. Mechanics: every `matlab -batch` wrapper ends `exit(0)` (hangs otherwise);
   `cd` to the example dir before running (GridFile resolves from cwd); PNGs
   re-read after overwrite may be STALE — unique filenames, verify numerically;
   MATLAB `corr` needs a toolbox — use `corrcoef`.
7. Bench conventions: **Kr always negative, psi toward the concave side** for
   mirrors (convex = geometry, never a sign flip); lens emission is the
   engine-verified pairing (`collimate`: flat front + powered back Kr=+(n−1)f;
   `focus`: powered front Kr=−(n−1)f + flat back), sag-aware thickness;
   distances chain through `b.E(k).s`; `twyman_green`'s `tail()` computes
   `det_leg` from the thin-lens conjugate of the TEST-OPTIC plane — any new
   tail must recompute the detector conjugate the same way (the invariant:
   detector sits at the DM-pupil image).

## 3. Deliverables

1. `twyman_green(..., 'tail_arch', A)` with `A ∈ {'singlet' (default, unchanged),
   'fieldlens', 'doublet'}` (+ `'oap'` stretch) — new tail architectures behind an
   option; baseline behavior bit-identical when the option is omitted.
2. Trade runner `templates/40_benches/bench_ifo_dm/l2_trade/run_l2_trade.m`:
   evaluates every architecture with the §4 metric, writes a results table
   (`.mat` + printed) + one comparison figure.
3. The winner wired into `example_bench_ifo_dm.m` as a parameter (default =
   winner if it meets the gate, else document why not); committed artifacts
   regenerated.
4. A short trade note (markdown, this directory) — see §7.
5. `tests/tBench.m` addition: each new tail arch builds, traces with zero ray
   loss, analytic chief agrees with engine chief at every element (existing
   pattern — compare against `b.E(k).rpt`).

## 4. The metric harness (build FIRST, gate everything on it)

`l2_trade/ifo_l2_metric.m`: given a `twyman_green` options struct, return:

- **M1 (primary): physical-instrument vs-truth residual.** Build poked + baseline
  test arms exactly as the phase-1 driver does; differential phase from DIRECT
  fields — `phi = angle(Ed_pk .* conj(Ed_b))`, `Ed = macos.complex_field(T.iDET)`
  per arm-state (justified by context item 3); `h = +phi*LAM/(4*pi)`; then the
  EXISTING 5b machinery (per-ray pupil map → orientation scan → similarity
  refine → spline truth resample), lifted into a function. Returns residual
  nm rms + corr.
- **M2: pupil-map report** — mag, anam, rot, nonlinear-distortion rms (per-ray
  affine fit; code exists in the driver's 5b).
- **Guards (all must pass for a candidate to count):**
  - zero ray loss at every traced element (`macos.get_ray_status`);
  - focal-mask spot ≤ 1 µm rms transverse (the mask must still work — reuse
    `spot_cost` from `example_bench_layout.m`);
  - detector at the DM-pupil conjugate: DM-tilt test (tilt at a pupil ⇒ no
    translation at its conjugate; Stage-C pattern in `example_bench_layout.m`);
  - baseline (zero-grid) differential null < 1e-8 rad rms;
  - note the tail is COMMON-PATH (post-Recomb, both arms) so added glass is
    automatically balanced — but verify via the null guard, not by assertion.

**Gate 0 (before any design work):** the harness on the unmodified singlet rig
reproduces M1 = 6.76 ± 0.1 nm and the pupil-map numbers in §1. If not, stop and
reconcile against `run_final.log` / `bench_ifo_dm.mat` in the example dir — do
not proceed on a broken harness.

Also record M1 at `POKE_NM=5` for baseline and winner (linearity + the per-nm
figure). Runtime: direct-field M1 ≈ 6 traces ≈ ~1 min per candidate config.

## 5. Candidates, in order

**Analysis first:** before optimizing anything, decompose WHERE the baseline
retrace comes from — the DM slope range maps to a field-angle range ±α at L2;
plot the per-ray chief-height mapping distortion over that α range. One
mechanism figure buys better decisions than 50 blind optimizer iterations.

### C1 — Field lens at the FocalMask (cheapest, textbook, do first)
A lens AT (or just behind) the intermediate focus re-images the pupil without
disturbing the image. `tail()` variant: FocalMask reference + `add_lens`
immediately after; recompute the detector distance from the two-lens conjugate
of the DM (the DM is the stop, not L2's aperture). Variables: `f_FL`, bending/
conic, detector distance. Pupil conjugate by construction; conic/bending by
minimizing M1 directly (`fminsearch`, ≤ 50 evals) or the M2 nonlinearity as the
cheap inner objective, confirmed with M1.

### C2 — Doublet L2 (two air-spaced singlets)
Variables: power split, separation, two bendings/conics. Stage 1: hold focus at
the mask (spot cost — bench_layout Stage-B pattern). Stage 2: minimize M2
nonlinearity, confirm with M1. Keep total track within ~1.5× the current
envelope — the bench must stay a bench.

### C3 — OAP-pair relay (stretch; only if C1+C2 both miss the gate)
All-reflective tail via `add_oap` ×2. No glass — but it folds the beam (planar
Bench: `sketch()` it, keep the layout non-self-intersecting) and buys
polarization aberration at the OAP AOIs — record that as a note FOR the
`pol-core` work; do not evaluate polarization here.

## 6. Acceptance

- Winner: **M1 ≤ 1 nm** at POKE_NM=50, all guards green (stretch ≤ 0.5 nm).
- Linearity confirmed at POKE_NM=5 (~10× down; report nm/nm).
- `tail_arch` omitted ⇒ existing example output unchanged (regression: rerun
  `example_bench_ifo_dm.m` unmodified, diff the printed numbers).
- `tBench` green; `run_mmacos_tests.sh fast` green.
- Commits on `ifo-l2`, per-deliverable; artifacts (`.in`, `.mat`, PNGs)
  committed per the examples convention; no `exit(0)` inside example code
  (batch wrapper only).

## 7. Reporting (the trade note)

Answer, in order: (1) what drives the baseline retrace (mechanism figure);
(2) the table — M1/M2/guards per architecture; (3) the winner + per-nm-of-poke
figure; (4) costs (elements, track length, alignment sensitivity, qualitative);
(5) the polarization footnote for C3. Lead with the number; write for Dave.

## 8. Out of scope

Glass selection / chromatic design (monochromatic 632.8 nm bench), fabrication
tolerancing, engine/Fortran changes, any data-processing "fix" for retrace, the
DO_ABERR / DO_NEARFIELD phases (separate, staged in the driver), and all
polarization evaluation (pol-core / pol-ifo).
