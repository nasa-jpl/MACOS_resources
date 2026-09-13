# pdi_dm96 — the point-diffraction DM surface gauges (P, PF), same DM truth

The point-diffraction readings of the DM-gauge campaign, split out of
`../zwfs_dm96/` on 2026-09-13 (Dave's ruling: the PDI gets its own
directory).  **The code stays shared** — `zwfs_run` and
`../dm_gauge_lib`, nothing is copied; this directory holds the
parameter sheet, the runner, the P/SRI bench decks and figures, and the
records.  Records taken before the move (`pdi193*`, `ploop193`,
`pcam193*`) are still in `../zwfs_dm96/runs/`, where `deck_pdi` cites
them; everything from 2026-09-13 is in `runs/` here.

Two readings, both point-diffraction interferometers, both scored in the
campaign's one currency (actuator-space rows on a 96×96 DM at a 30 nm
working surface, photons per measurement, the closed-loop hold, the
capture range to 10%):

- **P** — the **stepped pinhole**, common path: a pinhole substrate with
  an attenuating surround in the mask seat at the internal focus, its
  phase stepped.  Same bench as the Zernike sensor; the plate is the
  only difference.
- **PF** — the **P/SRI** of Dube, Nejadriahi, Sidick, Jewell, Redding,
  Lou and Basinger, *Proc. SPIE* **13092**, 130926F (2024): the
  reference is the mode of a single-mode waveguide in its own arm,
  phase-shifted thermo-optically, recombined with the unfiltered beam.
  Two benches: the reference **synthesized** on the ZWFS test arm (the
  record through 2026-09-12), and, since 2026-09-13, the reference
  **traced** through a buildable two-arm bench
  (`macos.design.psri_bench`; `pdi.bench 'psri'`).

## Run it yourself

    P = pdi_params;  out = pdi_run(P);                 % the record sheet
    pdi_run('pdi.DIA_LAMD', 1.0, 'stages', {'bench','battery','figs'})
    ./pdi_batch.sh TAG "pdi_params, 'stages',{'bench','loop','figs'}"

`pdi_batch.sh` runs it headless, memory-capped and logged, and is
serialized with `../zwfs_dm96/zwfs_batch.sh` on the same lock — **one
engine MATLAB at a time on this box** (a model-1024 run needs ~10 GB).
Stages, knobs and outputs are `zwfs_run`'s; `pdi_params.m` documents
what this sheet adds.  Figures: `psri_layout_fig` (the P/SRI bench:
solves the reference lens's conic and the pinhole seat's trim, emits
`psri_test.in` / `psri_ref.in`, prints the balance, draws
`psri_layout.png` and `psri_render.png`), `pdi_layout_fig` (the
common-path form's layout).  Both draw in the deck recipe
(`pdi_vfig_util`).

## Reproducing the record

Every number in `REPORT_gauge_pdi.md` and in the findings below comes
from one of these runs.  The whole chain is `runs/gmaster.sh`; each
sequence stands alone and can be re-run on its own.

| sequence | runs | what they answer |
|---|---|---|
| `runs/gsmoke.sh` | `sm_psri_nl`, `sm_knobs` | dev-resolution (model 512, 65 rays, 48×48 DM) smoke of every path the record uses: the P/SRI bench through noise AND loop, and the three new loop knobs |
| `runs/gseq1.sh` | `pfdeck`, `pfdeck_frz`, `pfdeck_loop`, `cap385p`, `cap385p_b60/90/120/160`, `noise193p_b30/60/120/160` | **the P/SRI with both arms traced** (rows, photons, loop) against the synthesized reference, with the frozen-reference control; **capture range and photons** for P and PF |
| `runs/gseq2.sh` | `pin20_1024`, `pin20_loop`, `pin10_2048`, `pin10_loop` | **the pinhole diameter of record**: 2.0 λ/D at model 1024 / 193 rays against 1.0 λ/D at 2048 / 385 |
| `runs/gseq3.sh` | `descent193`, `descent193s`, `descent193f`, `intra193_0`, `intra193` | **the descent** (capturing the DM's initial figure) and the **within-measurement DM drift** |
| `runs/gseq4.sh` | `rw193_1e3`, `rw193_1e2`, `rw193_1e1` | **the reference arm's own drift** (P/SRI), three sizes, with P as the common-path control |
| `runs/gfigs.sh` | — | the two layout figures |

The comparison baselines are the pre-split records in
`../zwfs_dm96/runs/`: `pdi193fbase` (the rows with the matrix on the 30
nm surface), `pdi193f` (photons), `ploop193` (the loop), `pdi193state`
(P with a shutter frame per state), `pdi193d1` (the 1 λ/D pinhole),
`pdi193se_ls` / `pdi193se_sh5` (the step-scheme trade), `pcam193*` (the
camera drift), `cap385` / `noise193_b*` (the ZWFS readings' capture
range and photons, for the side-by-side).

## The two benches

| | test field | reference | cost per state |
|---|---|---|---|
| `pdi.bench 'zwfs'` (default) | the ZWFS test arm's camera | **synthesized**: the recollimated LP01 mode (P), or the pinhole-diffracted flat field, scaled by the state's coupling κ — a complex SCALAR, so the reference's shape is fixed by construction | one trace |
| `pdi.bench 'psri'` | `psri_test.in`'s camera | **traced**: `psri_ref.in` carries the beam through Lr1, the physical pinhole in its near-field sphere bracket, Lr2, the fold and BS3 to its own camera; whatever the real arm does to the reference — amplitude, shape, piston — is in the frames, while the solver still uses the flat state's R₀ | two traces, two deck loads |

Only **PF** lives on the `psri` bench: the test arm's seat is empty
there (the P/SRI filters in the other arm), so there is no dimple
reading and no common-path P on it.

## Findings

### The gauge-deck record (2026-09-13)

The full tables, with every run tag, are in **`REPORT_gauge_pdi.md`**
beside this file.  So the two cannot drift apart, the NUMBERS live in
the report and this section states the FINDINGS.

- **The P/SRI's reference arm, TRACED** (`pdi.bench 'psri'`;
  `runs/pfdeck`, `runs/pfdeck_frz`).  The reading now has a bench on
  which the reference is not modeled: `psri_ref.in` carries the state
  through Lr1, the physical pinhole in its near-field sphere bracket,
  Lr2 and BS3 to its own camera, while `psri_test.in` carries the test
  beam to the same camera plane.  Two traces per state.  *The control
  that makes it a measurement:* `pdi.ref_frozen` traces the arm ONCE on
  the flat and holds it — same bench, same lenses, same aberrations,
  only still.  It reads **0.000 pm** where the moving arm reads **5.889
  pm** on a 13 nm figure, so that error is ENTIRELY the reference moving
  with the state; and **the synthesized LP01 model sits exactly with the
  frozen one**, because a reference whose state dependence is one
  complex scalar is, to a solver that takes κ = 1, a frozen reference.
  Differentially the motion is a 10%-class effect on the noise floor
  (dense row 129 → 144 pm) and nothing at all on the gain or the range.
  **The P/SRI's argument survives being built:** gain inside 0.7% out to
  a 480 nm rms working surface, flatter than the model of it (+12.8%
  there) and far past where S and P fold at 120 nm.  N(1 pm) 2.00e14
  traced vs 1.9e14 synthesized — the reference model does not set the
  photon cost; the 60/40 pickoff does.  *Trap worth keeping:*
  `macos.dx_at` at a plane returns 0 until the field has been
  PROPAGATED there — call `complex_field` first, or the pinhole disk
  comes out all-ones and the "reference" is the whole beam.

- **Four shared loop knobs** in `../dm_gauge_lib/dmg_loop.m`, gated on
  the synthetic instrument in `mmacos/tests/tDmgLoop.m` (G9–G12): the
  descent (`start_rms`), on-surface re-calibration (`recal_every`, with
  the `ins.recal(cmd)` contract CCMac mirrors), the drift developing
  WITHIN one measurement's scan (`intra`, `aux.dstep`) and a
  non-common-path reference arm's phase walk (`ref_walk`,
  `aux.ref_phase`).  `aux` reaches an instrument only when a knob is on,
  and the drift increments are now drawn once ahead of the loop in the
  same order from the same stream, so **every run taken before this
  reproduces bit-for-bit**.

### The readings, and the record through 2026-09-12

*(Moved verbatim from `../zwfs_dm96/README.md`.)*

- **P / PF (2026-09-12, Dave: "another sensor, using a point-diffraction
  IFO approach -- see papers by Brandon Dube"): the point-diffraction
  readings, on the same bench.**  Plan and literature:
  `macos/BRIEF_pdi_campaign.md`; the paper is Dube, Nejadriahi, Sidick,
  Jewell, Redding, Lou, Basinger, SPIE 13092-178 (2024) -- the phase-
  shifting self-referenced interferometer (P/SRI): a non-common-path
  interferometer whose reference is the mode of a single-mode waveguide
  in a photonic chip, phase-shifted thermo-optically, read by the five-
  frame Schwider-Hariharan scan, the change taken by complex division so
  it never wraps.  Two readings in `dm_gauge_lib/dmg_pdi_gauge`, both
  threaded through every stage of this runner (classes 5 / 6; the deck
  and the sheet: `pdi_params` / `pdi_run`):
  **P**, the stepped pinhole (common path): mask `t + (e^{i theta_k} -
  t) D` at the FocalMask (`D` the pinhole disk, `t` the surround
  amplitude transmission, 'auto' = the reference's rms amplitude / the
  beam's), the classical step fit per pixel, `|b|^2` from the flat's
  pinhole-only frame (iterated with the phase) or a frame per state, the
  reference phase iterated as the exact readings do.  **PF**, the P/SRI
  as their MATLAB model has it: the recollimated LP01 mode (step-index
  J0/K0 field, V 2.3, b 0.5, core radius 0.5 lam/D at the focus -- the
  paper's Thorlabs set) at unit rms over the pupil, scaled by the
  state's overlap coupling into the mode relative to the flat's (kappa,
  a complex SCALAR: the shape is fixed by construction), pickoff 0.6
  (their beamsplitter R), reference amplitude = min(visibility-1 match,
  the pickoff budget f |c0|^2); exact in one pass, no |b|^2 degeneracy.
  Both: the wrapped differential; photons counted at the camera with
  `throughput` printed.  *Gates (bench stage):* G5 the same 100 nm sparse
  pokes as G4 -- P 0.33 pm, PF 0.000 pm of a 12 nm figure; **G7: at
  t = 1 and the dimple's diameter the pinhole reading IS the stepped
  reading S (5.2e-15)**; G6 the reference's motion under the 30 nm
  surface split into a Strehl-class amplitude SCALE (0.86 at every
  diameter) and the SHAPE change: **0.10% at 0.5 lam/D, 0.26% at 1,
  0.58% at 1.5, 1.06% at 2 (the dimple), 2.6% at 3** -- the PDI
  argument in one table; the waveguide's shape change is 0, its
  coupling 0.86 with 0.025 rad.  Sampling: the pinhole gets the
  dimple's rule (>= 6 px at the mask plane: 7.9 px at 2 lam/D here; a
  1 lam/D pinhole is 3.96 px and warns).  *Record 1 (runs/pdi193, flat
  matrix; its PF carried the pinhole-shaped reference, the first
  idealization):* on the 30 nm surface V / P / PF agree -- single
  0.939 / 0.933 / 0.940 (25 pm), grid 0.997 / 0.992 / 0.998, dense
  0.999 / 0.993 / 1.001 -- where S reads 0.752 / 0.829 / 0.818; N(1 pm)
  at the camera S 5.4e13, V 4.7e13, **P 3.3e13**, PF 2.1e14.  *Record 2
  (runs/pdi193f flat, runs/pdi193fbase the matrix ON the surface; PF =
  the LP01 reference, visibility 0.863, throughput 0.752):* flat matrix
  PF 0.940 / 25 pm single, 0.998 grid, 1.001 dense; ladder 0.77 at 120
  nm, 0.55 at 240 where S / V / P fold; N(1 pm) PF 1.9e14 (S 5.4e13, V
  4.7e13, P 3.3e13): per DETECTED photon the waveguide form is the most
  expensive -- with the 60/40 split 60% of the light goes to an arm
  that returns 59% of it as reference and the test beam keeps 40%, so
  the modulation is a smaller fraction of the detected flux than the
  stepped pinhole's, whose reference rides on the same beam (divide by
  0.75 / 0.82 for incident photons).  **Matrix on the surface: single
  10 nm S 0.9885 / 5 pm / SNR 2120, V 0.9935 / 4 / 2835, P 0.9935 / 4 /
  2790, PF 0.9935 / 4 / 2842; grid 1 nm all four 0.999 / 3-4 pm; dense
  10 nm S 0.984 / 681 pm, V 0.9999 / 331, P 0.9985 / 338, PF 1.0002 /
  330** -- the three exact readings are indistinguishable at the
  operating point.  *Ladder (47 sites, the 30 nm matrix):* 60 nm S 0.66,
  V 0.98, P 0.94, PF 1.006; **120 nm S / V / P fold (-0.01), PF 1.02;
  240 nm PF 1.06; 480 nm PF 1.13** (floor 1.6 nm: the 30 nm matrix on a
  16x surface).  A reference that does not depend on the surface has no
  fold: the P/SRI's range is the wrap of the DIFFERENCE, not of the
  surface.  P folds with V because its reference amplitude collapses
  with the Strehl and its flat |b|^2 assumption breaks (the per-state
  shutter frame, `pdi.b2 'state'`, is runs/pdi193state).  *Shutter
  frame per state (runs/pdi193state, 5 frames):* P becomes PF's twin on
  every row AND the ladder -- **1.02 / 1.06 / 1.13 at 120 / 240 / 480
  nm** -- so P's fold was the flat |b|^2 assumption, not the pinhole.
  *A 1 lam/D pinhole (runs/pdi193d1; 3.96 px at the mask plane, the
  budget line warns):* t_auto 0.28, eta_pin 0.24, throughput 0.29,
  visibility 0.94, G5 0.002 pm (the reference iteration converges 100x
  better than at 2 lam/D); rows identical to PF's (0.9935 / 4 pm; 0.9992
  / 3; 1.0002 / 330) and **no fold with the flat |b|^2 (1.02 / 1.06 /
  1.13 at 120 / 240 / 480 nm)**: the classical pinhole regime buys the
  P/SRI's range in the common path, for 29% of the light (N(1 pm) at the
  camera 2.9e14 vs S 1.0e14 with this run's on-surface matrix; 1e15
  incident).  Camera 1/f
  drift in the loop: `dmg_loop` opt.cam (an offset random-walking
  `loop.cam_walk` electrons per pixel per cycle, constant within a scan
  unless `cam_intra`); readings whose step weights sum to zero (S, P,
  PF) subtract it exactly (tDmgLoop G8), the single-frame readings and
  the simultaneous pair imprint it on the DM.  *Camera drift, the
  paper's number (runs/pcam193, all six readings, 0.13 e per pixel per
  cycle = ~1 e over the run, 1e13 and 1e15 photons per cycle):*
  **invisible to every reading** -- hold error, bias and spectrum equal
  to the noise-only rows to the printed digit (L 1.39 / 0.14 pm at 1e13
  / 1e15 with and without it; S 1.52 / 0.15; V 1.17 / 0.12; P 1.45 /
  0.14; PF 2.50 / 0.25).  Why: at 1e13 photons per measurement a lit
  pixel collects ~3e8 photons per frame, so an electron is 1e-4 of its
  shot noise; the immunity argument lives in the photon-starved regime
  of Roman's LOWFS (per-pixel counts of 1e2-1e3 per frame, integrated
  over 12 h).  Hence `loop.cam_unit 'rel'`: the walk as a fraction of
  the mean photons per lit pixel per frame (a bias / gain drift scaled
  to the signal), runs/pcam193r (1e-3 per cycle) and pcam193ri (the
  whole step within each scan); the electron form with the whole step
  within each scan (runs/pcam193i) is equally invisible, every entry
  the noise-only value.  **The relative form (runs/pcam193r, the offset
  walking 1e-3 of the mean photons per lit pixel per frame, per cycle,
  ONE scale per scan -- the first attempt scaled per frame and is kept
  as runs/pcam193r_perframe, superseded): the single-frame reading L
  imprints the walk at 10.8 nm hold error (noise-only 1.4 / 0.14 pm at
  1e13 / 1e15), the simultaneous pair V at 89 pm, and the zero-sum
  readings S / P / PF read their noise-only values to the printed digit
  (1.52 / 1.45 / 2.50 pm at 1e13, 0.15 / 0.14 / 0.25 at 1e15; spectra
  identical).**  The PSI immunity, measured on the bench: a camera bias
  that is constant within a scan cannot reach a reading whose step
  weights sum to zero.  *The whole step developing WITHIN each scan
  (runs/pcam193ri, cam_intra 1):* the zero-sum readings now pay for the
  frame-to-frame part -- S 5.4 pm, P 5.3 pm, PF 10.7 pm hold error at
  1e15 (spectra 0.14 / 0.41 / 5.4 pm in the three bands for S and P) --
  2000x less than the single-frame reading's 10.8 nm and 17x less than
  the pair's 89 pm, both unchanged.  What a temporal PSI cannot remove is
  the drift between its own frames; the cure on hardware is the scan
  rate (the paper's continuous triangle-wave scans).  I+ on this base
  floors at 885 pm
  regardless (its fold-flipped sites, S11).  *Closed loop
  (runs/ploop193, P and PF on the S11 seeds, the matrix on the working
  surface; 82 min):* both contract at 0.509 per cycle (reading gain 0.98
  at loop gain 0.5, as V) and take the 1 and 10 nm steps to 0.000 pm --
  no fixed error; noise-only hold 3 pm from **P 2.3e12** (S 2.6e12, V
  1.5e12, L 2.1e12) and **PF 7.0e12** photons per cycle; under the 2 pm
  walk **P 7.0e12** (S 7.5e12, V 5.3e12) and **PF 2.5e13**; thermal
  floor 9.9 pm for both (the proportional loop's lag, as every exact
  reading); the held residual's spectrum under the walk identical to
  V's (0.25 / 0.72 / 2.2 pm in the three bands).  PF's single-shot noise
  is 1.7x P's at every level (sig_n 13.5 vs 7.8 pm at 1e12), the 60/40
  split again.  Pending in this record:
  pdi193se_sh5 (the five-frame Schwider-Hariharan half of the step-
  error trade).  *A 2% step-size error, four-step least squares
  (runs/pdi193se_ls, matrix on the 30 nm surface):* the ABSOLUTE
  reading carries it -- the flat reads 6.9e-2 rad rms (P) / 5.7e-2 (PF)
  instead of 1e-15, and the 12 nm sparse-poke figure comes back with
  421 / 251 pm rms error (3.5% / 2.1%) -- while the DIFFERENTIAL rows
  barely move: single 10 nm P 0.9918 / 4 pm, PF 0.9908 / 4 (0.9935 / 4
  without the error); grid 0.998 / 3 and 1.002 / 3; dense 0.998 / 354
  and 1.004 / 353 (338 / 330); ladder at 60 nm P 0.94, PF 1.02.  The
  *The five-frame Schwider-Hariharan scan under the same 2% step error
  (runs/pdi193se_sh5):* the flat reads 1.3e-5 rad rms, the 12 nm figure
  comes back with **4.9 / 2.2 pm** rms error (P / PF; 421 / 251 under
  least squares), and the differential rows are the error-free ones to
  the digit (single 0.9935 / 4 pm, grid 0.9992 / 3, dense 0.9985 / 338
  and 1.0002 / 330; ladder 60 nm P 0.94, PF 1.005).  de Groot's
  zero-sum weights buy first-order immunity to the step size for one
  extra frame -- the paper's scheme, confirmed.  pcam193 / pcam193i (the camera drift, all
  six readings, within-scan 0 / 1).  Figure: `<tag>_pdi.png` (the focal
  spot with pinhole, dimple and mode; the reference amplitudes; the
  reference's motion by diameter; the visibility maps).  Deck:
  `macos/demo_session/deck_pdi.md` (DRAFT).  *The P/SRI as a BUILDABLE
  bench (Dave 2026-09-12: "separate but balanced arms, the reference
  through the pinhole / phase shifter"):* `macos.design.psri_bench` +
  `psri_layout_fig.m` (`psri_layout.png`, `psri_render.png`,
  `psri_test.in`, `psri_ref.in`) -- the TG96 front end, then a
  Mach-Zehnder in the returned chief's frame: BS2 splits; the test arm
  transmits, folds (M1), carries a normal-incidence glass compensator and
  reflects off BS3's front face; the reference arm reflects to Lr1 (f
  300, F/2.9), the pinhole seat in its NF sphere bracket at the true
  focus (+1.10 mm), Lr2 (Lr1 mirrored about the pinhole), fold M3, and
  transmits BS3; both exit on ONE chief into the tuned tail.  Solved on
  construction: the compensator (21.9 mm) for equal chief optical paths
  (4453.1 mm each), M3 for coincident exit chiefs (2e-13 mm); both decks
  trace 3210 / 3210 rays to one camera plane (chiefs 2e-12 mm apart,
  footprints 3.60 / 3.29 mm).  Lr1's conic -0.578 solved on the trace
  (0.1 um ray blur; the add_lens seed gives 0.31 mm).  Traps recorded:
  the F/2.9 diffraction focus is ~20 um deep (find the ray focus first);
  the engine's OPD at a tilted fold reads geometry and at the camera the
  tail's convergence dominates (measure collimation on a plane normal to
  the beam); the collimated beam behind the DM carries the front end's
  5.8 um rms residual (L1), which the tuned tail cancels for the test arm
  and the pinhole filters for the reference -- so Lr2 must be Lr1's
  mirror image (reversibility: the recollimated wave reproduces the
  recombination plane's, 5.78 vs 5.80 um), never "solved" against that
  residual.  NEXT: the PF reading on the two decks (reference from
  psri_ref.in through the pinhole) instead of the synthesized reference --
  **DONE 2026-09-13, see below.**

### Layouts and parts (2026-09-13)

Both figures are drawn by the ENGINE from the emitted deck
(`macos.view_rx`) in the deck recipe Dave set on 2026-09-12 and
`../zwfs_dm96/zwfs_vlayout.m` carries: the fold plane seen from above,
passive bookkeeping planes hidden, elements NAMED (not E-numbers) with
leader lines off the beam, the crowded node as a cropped panel at full
width, 15-17 pt type in an 1800 px figure.  The recipe itself is
`pdi_vfig_util.m`, so the two scripts cannot drift apart.

- `pdi_layout.png` (`pdi_layout_fig`) — the common-path form.  The
  hardware IS the ZWFS test arm; the one difference is the plate in the
  seat, so the figure says that in the caption instead of drawing a
  second bench.  *(The pre-2026-09-13 version was a `Bench.sketch` with
  E-number labels at 9-11 pt — it did not meet the recipe and was
  redone.  Its companion `pdi_layout_tail.png` is gone: the recipe puts
  the crowded node in the SAME figure as a cropped panel, so a second
  file would be a second thing to keep in step.)*
- `psri_layout.png` / `psri_render.png` (`psri_layout_fig`) — the
  two-arm P/SRI: panel 1 the whole bench with both arms' own traces
  overlaid (green test, blue reference), panel 2 the reference arm's
  node (Lr1, the pinhole seat, Lr2, M3) cropped at full width.  The
  same script solves the bench (the reference lens's conic and the
  pinhole seat's trim) and prints the balance record, so the figure and
  the numbers cannot disagree.

**Parts — the common-path pinhole (reading P).**  Everything else is
the shared front end (CCL's list).

| part | size | coating | count | purpose |
|---|---|---|---|---|
| pinhole substrate | fused silica plate in the mask seat; pinhole **5.27 µm** diameter (2.0 λ/D at F/4.17, 632.8 nm) | surround attenuated to **t = 0.719 in amplitude** (0.52 in power) — matched to the pinhole-diffracted reference so the fringe visibility is ~1; pinhole clear | 1 | passes the spot's core as the reference wave and attenuates the rest of the beam to match it |
| phase stepping | the pinhole's phase stepped through 0, π/2, π, 3π/2 (4 frames), or the 5-frame Schwider–Hariharan scan −π…π | — | 4 or 5 steps | the exact linear solve, no quarter-wave fold |
| camera | at the reimaged pupil, ~2.5 detector px per actuator at 193 rays (~20 µm px, 1 Mpix class, at 385) | — | 1 | the pupil image the reading is solved on, pixel by pixel |

At 1.0 λ/D the same plate is a **2.64 µm** pinhole with **t = 0.282**
(throughput 0.29 instead of 0.82) — the classical-pinhole regime; see
the pinhole-diameter trade below.

**Parts — the P/SRI (reading PF), the buildable two-arm bench.**  All
Mach–Zehnder incidences are **45°** by construction (the interferometer
is rectangular: BS2 splits, each arm folds once, BS3 recombines).

| part | size / focal length / AOI | coating | count | purpose |
|---|---|---|---|---|
| pickoff beamsplitter BS2 | plate, 2.57 mm thick, n 1.5, clear aperture ≥ 103 mm, **AOI 45°** | **R 0.60 / T 0.40** (the paper's pickoff: 60% of the power to the reference arm) | 1 | splits the returned beam into the reference and test arms |
| reference lens Lr1 | f **300 mm**, 102.9 mm clear aperture → **F/2.92**; plano-convex, n 1.5, conic **−0.5784** on the powered face | AR | 1 | focuses the reference arm onto the pinhole |
| pinhole / waveguide seat | **3.69 µm** diameter (2.0 λ/D at F/2.92), opaque surround; seated at the TRUE focus, **+1.096 mm** beyond the thin-lens one | opaque surround | 1 | the spatial filter that makes the reference; in the photonic form this seat holds the single-mode waveguide facet |
| photonic phase shifter | thermo-optic, on the waveguide chip | — | 1 | the phase steps, in the reference arm only |
| reference lens Lr2 | Lr1 **mirrored about the pinhole** — same f, same conic, symmetric placement | AR | 1 | recollimates the filtered reference.  It must NOT be re-solved against the front end's residual: see the reversibility note below |
| folds M1 (test), M3 (reference) | flat, 150 mm across, **AOI 45°**; M3's position is solved so the two exit chiefs coincide | protected metal | 2 | close the two legs of the rectangle |
| compensator (test arm) | **21.857 mm** of n = 1.5 glass, normal incidence, clear aperture ≥ 103 mm | AR | 1 | equalizes the two chief optical paths (4453.1061 mm each, to 0 mm) against the reference arm's two lenses |
| recombination beamsplitter BS3 | plate, 2.57 mm thick, n 1.5, **AOI 45°** | 50/50 | 1 | recombines; the test arm reflects off its front face, the reference arm transmits |
| camera | as above, one camera at the pupil image | — | 1 | both arms land on it — chiefs 2e−12 mm apart, camera planes 2e−13 mm apart, 3210 of 3210 rays through each arm |

Each arm transmits ONE plate and reflects off the other's front face,
so the plate glass is equal by construction; only the reference arm's
two lenses need compensating.
