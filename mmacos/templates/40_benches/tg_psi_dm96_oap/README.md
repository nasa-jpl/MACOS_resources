# tg_psi_dm96_oap — the 96×96 Twyman–Green DM gauge, all-reflective (OAP) variant

The reflective sibling of [`../tg_psi_dm96`](../tg_psi_dm96): the two lenses of
the polarization phase-shifting Twyman–Green DM surface gauge (L1 collimator,
L2 focuser) become **off-axis parabolas** (OAPs). The refractive rig in
`../tg_psi_dm96` is the untouched record; this directory is the variant beside
it, driven by the **same** builder (`macos.design.twyman_green`, new option
`'optics'`) and **one** parameterized runner that runs *both* rigs.

## Why reflective

Removes the transmitted-glass-path and homogeneity rows from the cost budget,
buys achromatic legs and no ghost surfaces; the price is OAP alignment
sensitivity (measured here) and same-plane fold aberration (measured here). The
gauge's known roll-off is geometric (the tail conjugate + pupil distortion), so
the reflective train is also the natural place to re-attack it.

## Design (what changed, and what did not)

- **Only L1 and L2 become OAPs** (`twyman_green('optics','oap')`). The field
  lens FL in the tail stays a lens (Dave's ruling); its cost line is reported.
- The fold is confined to the **source→OAP1** and **OAP2→detector** legs: OAP1
  re-emits collimated along the lens rig's post-L1 direction, so the **plate BS,
  both arms, the compensator, the QWPs/analyzer and the recomb plane are
  geometrically unchanged** — the Stage-A clearance, the A2 sampling budget and
  the tail bookkeeping carry over.
- Both OAPs **fold in the plane of the BS** (same-plane folds). The fold AOIs
  are re-solved by Stage A for the new legs (near-normal preferred; must clear
  the bodies inside the leg cap). OAP conjugates are kept, not focal lengths:
  pole→focus `= F1` (collimator) / `= F2` (focuser); the parent focal follows.
- OAPs are perfect conductors first (`IndRef=1, Extinc=1e22`; RS=−1, RP=+1,
  polarization-neutral) so the lens/OAP comparison is geometric. A coated-Al
  (`coat_set`) row is the polarization-cost stretch.
- **The tail is re-tuned for the OAP focuser** (`tg96_tail('bench.optics','oap',
  …)`) — the fieldlens trims were fit to L2's aberrations and do not transfer.

## Run it yourself

```matlab
% interactive (no exit); edit tg96_params.m, then:
tg96_run                                  % lens rig (equivalence gate)
tg96_run('bench.optics','oap','tag','oap')% reflective rig
```
```bash
# headless, memory-capped (model 1024 ≈ 11 GB); one model size per process:
./tg96_batch.sh lens
TG96_MEMMAX=20G ./tg96_batch.sh oap "'bench.optics','oap'"
# tail retune (do first for the OAP rig; writes <tag>_tail.mat):
matlab -batch "tg96_tail('tag','oap','bench.optics','oap')"
```
Outputs land in `runs/<tag>/`: `<tag>_report.txt`, `<tag>.mat`,
`<tag>_{test,ref}.in`, `<tag>_layout.png`, `<tag>_closure.png`,
`<tag>_transfer.png`. The `figs` stage also renders the optics layout two ways:
- **`<tag>_render.png`** — the FULL RAYTRACE render via **`macos.view_rx`**: the
  test arm traced to the detector, optics as solid bodies on their real conic
  sag + apertures, the beam as a filled ray bundle read back from the engine
  (correct for the folded OAP legs), in two panels — **TABLE PLANE** (looking
  down on the bench) and **ISO** — for checking beam-vs-edge clearances. This is
  the deck_zwfs slide-4 recipe (`zwfs_dm96/zwfs_wf_figs.m`).
- **`<tag>_sketch.png`** / `_sketch_ref.png` — the lighter `Bench.sketch`
  schematic (chief-ray polyline, aperture-sized footprint bars, element names +
  leg lengths), test and reference arms.

### Run it yourself: the pupil image (Fang Shi's question)

```matlab
% interactive (no exit); the knobs are P.pupil in tg96_params.m:
tg96_pupilq('rig','lens');     tg96_pupilq('rig','oap');      % crossing-cloud quality: distortion, surface, blur, the seat per tilt
tg96_pupilsim('rig','lens');   tg96_pupilsim('rig','oap');    % the detailed simulation: zone PSFs, the DM field through them, the compromise plane, the Fourier check
```
```bash
# headless (model 512, ~3 GB, ~6 min per rig for pupilsim, ~1 min for pupilq); both tools, both rigs:
./tg96_pupil_batch.sh both
./tg96_pupil_batch.sh lens "'tool','sim','fourier',false"      # one rig, one tool, a knob overridden
```
Outputs land in `runs/pupilq_<rig>/` and `runs/pupilsim_<rig>/`: `<tag>_report.txt`
(every number quoted in the deck and the reports), `<tag>.mat`, and the figures
`_psf.png` (the leg's complex PSF and transfer phase at three DM zones), `_surface.png`
(each zone's image vs the detector plane, the astigmatic split, the band-edge
wavefront), `_gain.png` (phase gain and amplitude cross-talk vs radius, as built and
at the compromise plane), `_work.png` (the 30 nm working surface recovered), and
`_fourier.png` (zone-PSF model vs the plane-to-plane propagation).  `tg96_pupilsim`
opens the baffle, widens the source cone and puts the aperture on the DM (P.pupil.dm_ap)
so the DM is the stop in fact; `dm_ap 0, overfill 0` reproduces the deck as emitted.

The OAP rig shows L1/L2 folding the beam off-axis; the lens rig is near-collinear. Memory-bound? drop a trimmed `macos_param.txt` in the run
dir via `P.param_file` (keep `mGridMat ≥` the DM grid, 384 here).

## Parts lists (for the gauge deck; geometry from Stage A, scale s = 96/56)

**Lens rig — shared front end + interferometer:**

| part | size / f / AOI | coating | count | for |
|---|---|---|---|---|
| source (filtered HeNe 632.8 nm) | 51 mm beam radius (collimated) | — | 1 | illumination |
| collimator L1 | f 857 mm, ⌀ 103 mm | AR | 1 | collimate onto the DM |
| beamsplitter (plate) | AOI 7°, 2.6 mm thick | polarizing 50/50 | 1 | split the arms |
| compensator plate | matched, 171 mm from BS | AR | 1 | balance the BS glass path |
| 96×96 DM (test object) | 96 mm, 1 mm pitch, 700 mm leg | protected Al | 1 | surface under test |
| reference flat + PZT | ⌀ ≥ 103 mm, ~564 mm leg | protected Al | 1 | reference return + PZT four-step |
| focuser L2 | f 429 mm, ⌀ 103 mm | AR | 1 | image the DM pupil |
| field lens FL | f 43 mm, ⌀ 21 mm | AR | 1 | pupil-imaging tail |
| camera | 385 px per pupil | — | 1 | pupil-image detector |
| snapshot optics: PolIn, arm QWPs, OutQWP, analyzer | quarter-wave; 45/0/45/0/0° | — | 5 | polarization four-step |
| v2 snapshot: MacNeille cube (replaces plate BS + comp + polarizers) | 12.7 mm, ZnS/cryolite on n_g 1.655, symmetric stack | MacNeille | 1 | diattenuation-free split |

**OAP rig — reflective front end (replaces L1, L2):**

| part | off-axis dist / f / AOI | coating | count | for |
|---|---|---|---|---|
| OAP1 (collimator) | f 857 mm, AOI 5°, off-axis 149 mm (+22 mm margin) | bare / protected Al | 1 | collimate + fold source→DM |
| OAP2 (focuser) | f 429 mm, AOI 9°, off-axis 132 mm (+6 mm margin) | bare / protected Al | 1 | image + fold DM→camera |

The OAPs fold in the BS plane (off-axis dist = f·|sin(180−2·AOI)|); the tail is
re-tuned for the OAP focuser. BS, DM, reference flat, field lens, camera and the
polarization optics are as the lens rig. Full deck report: **`REPORT_gauge_ifo.md`**.

## Files

| file | role |
|---|---|
| `tg96_params.m` | every knob of record + `bench.optics`, OAP fold AOIs, `calib_mode`, `place.*`, `d4`, `loop.*`, `battery.deck`/`noise`, `pzt.step_err`, `loop.cam_*` |
| `tg96_run.m`    | Stage A–E + Stage PLACE (D1) + Stage MATRIX (D2) + Stage DECK (deck items 1+2: rows/capture/photons) + Stage D4 + Stage LOOP (D7, +PZT step error, +camera drift), one path for lens+OAP |
| `REPORT_gauge_ifo.md` | the gauge-deck report (the IFO lanes): rows on the 30 nm surface, capture range, the three phase-shift forms, lenses-vs-OAPs, parts lists |
| `tg96_place.m`  | window placement from the ray affine (`dmg_frame`) + directional-parity + robust affine refit |
| `tg96_apply_parity.m` | detector-mm → field pixel under the resolved field-array parity |
| `tg96_tail.m`   | re-tune FL_F/FL_Kc/D_MASK_FL/DET_TRIM per optics (unaligned null), **gated by a multi-site row read by lattice deconvolution** — see "The tail of record" |
| `tg96_samp.m`   | detector-frame map -> DM frame through `tg96_place`'s OWN affine + parity; the fold rotation the shared `dmg_samp` cannot express |
| `tg96_run_batch.m` / `tg96_batch.sh` | `matlab -batch` wrapper (exit only here) + launcher |
| `tg96_pupilq.m`  | pupil image quality of the detector leg (Fang Shi, 2026-09-16): the DM as the stop, crossing cloud at the camera (distortion vs one affine, pupil surface, blur over the actuator band), the rodgers2 set at the seat per tilt; `runs/pupilq_<rig>` |
| `tg96_pupilsim.m` | the detailed pupil-image SIMULATION (Dave, 2026-09-17): the leg's coherent PSF per DM zone from the rays (the intercept walk over a 2-D tilt set integrates to the zone wavefront), the DM field through those PSFs (sinusoids, pokes, the 30 nm working surface; gain and amplitude cross-talk vs radius), the compromise detector plane, and a plane-to-plane Fourier cross-check of the tail; opens the baffle and puts the aperture ON the DM (the DM is the stop in fact); `runs/pupilsim_<rig>` |

## The tail of record, and the tuner's open problem

**On the reflective (OAP) rig the tail of record is the GEOMETRIC SEED**, not a
tuned set: `bench.tail_from_mat false`. It reads a single actuator at **0.9809**
with a clean break ladder, where the tuner's own winner read **0.0338**
(`runs/tailB` vs `runs/tailA`, REPORT_reflective §4.5). The lens rig keeps its
tuned tail, which does read (0.9968).

**Why the tuner's objective is not trusted here, and what is open.** Every
quantity `tg96_tail` computes about its candidate — the flat-DM null, the
recovered poke peak, the localization `conc`, the wrap fraction — preferred the
tail that does not read: measured cleanly and one run at a time, the old winner
scores `conc` 1.000, wrap 0.66, null 0.0223 nm and cost 0.0015 against the
seed's 0.0089. So the objective is optimizing something **orthogonal to
readability**, and no reweighting of those four terms can fix that. *Why* is
open (tags `objseed3` / `objwin3` for the clean A/B; `tailA` / `tailB` for the
rows), and it is a real piece of work that has not been started.

**What was done instead — the winner gate.** The tuner no longer certifies its
own winner. After the tune it reads a multi-site actuator row through the ray
affine, in **actuator space** — the quantity the battery measures — and refuses
any winner below `gate_gain` (0.95), returning the geometric seed with the
reason printed. The affine comes from `tg96_place`, i.e. from the ray trace of
the very bench under test, so a tail that has walked the detector off the DM's
pupil conjugate still gets its own honest mapping and still reads low: the
actuator's response is no longer imaged onto its own site. Cost: one placement
plus one poked map, once per tune, not per fminsearch evaluation.

**The first version of the gate measured the wrong thing, and this is how.**
It took ONE POINT SAMPLE — the measured map at the detector pixel the affine
sends an actuator's centre to, over the DM surface at that same centre — and
it failed its own two-leg test in the direction that matters: it ACCEPTED
`objwin3` (battery 0.0338) at 0.9804 and REFUSED the lens rig's tuned tail
(battery 0.9968) at −0.8285. A point sample at the peak reads how
CONCENTRATED the response is, and that is set by the MAGNIFICATION: the broken
tail's 6.125 against the seed's 10.44 spreads the response over fewer detector
pixels and so dilutes the peak less, reading HIGHER. Magnification is not
readability, and a gate built on it prefers exactly the tails it exists to
refuse. It was made ADVISORY the same day rather than left enforcing.

**STATUS: the gate ENFORCES, and its criterion is RELATIVE to the seed**
(Dave, 2026-09-16). It refuses a winner that reads worse than the geometric
seed it would fall back to, both measured through the same estimator on the
same bench:

| tail | battery | gate reads | seed reads | ratio | verdict |
|---|---|---|---|---|---|
| `objwin3` (bad) | 0.0338 | −0.1621 | 0.8200 | **0.1977** | REFUSED ✓ |
| `lens_tail` (good) | 0.9968 | 0.8074 | 0.8528 | **0.9467** | ACCEPTED ✓ |
| `thk22_tail` (good) | 0.9885 | 0.9104 | 0.9168 | **0.9930** | ACCEPTED ✓ |

**Why a ratio, after two absolute thresholds failed.** The point-sample measure
INVERTED the verdict (it tracked magnification). The lattice measure orders
tails correctly but does not share the battery's SCALE — it reads 8–19 % low,
and not because of the regularizer (the `act_lam` sweep is flat to ~1 % from
0.05 to 0.002) — so an absolute 0.95 refused two tails the battery certifies at
0.99. The gate's real decision was never "is this tail good in the abstract"; it
is "keep the winner, or hand back the seed". The seed is the alternative and it
is measurable, so comparing the two through one estimator cancels whatever
systematic scale that estimator carries. The seed columns above show it working:
on good tails the seed reads 0.82–0.92, right alongside the winner, and the
common bias divides out.

**The two constants, and what each is allowed to decide.** `gate_rel` (0.90) is
the gate. `gate_seed_floor` (0.30) is NOT — it decides only whether the seed is
a usable REFERENCE, never whether the winner passes; measured tails that read
sit at 0.81–0.92 and one that does not reads −0.16, so 0.30 sits in that gap.
When the seed itself does not read, the gate KEEPS the winner and says so
loudly rather than failing closed, because refusing would hand back a fallback
no better than what it refused.

**Margin, measured.** The tightest case is `lens_tail` at 0.9467, a margin of
0.047 over `gate_rel`. Run-to-run spread is **exactly zero** — three runs of
that leg return 0.8074 / 0.8528 / 0.9467 bit-identically, because the trace,
the placement, the poke sites and the `pcg` solve are all deterministic. So the
margin does not have to absorb measurement noise at all; what it must absorb is
variation ACROSS benches and tails, which is why `gate_rel` is kept loose. A
verdict sweep (`t_gate`) confirms every verdict is unchanged for `gate_rel`
0.30–0.90 and only diverges at 0.95, so 0.90 sits inside the stable band rather
than at its edge.

**If this ever needs changing, move `gate_rel` DOWN, never the measure up to
meet it.** Two gates have already been certified on a number that fit rather
than a number that meant something.

**The measure of record is lattice deconvolution** (`row_gain_`, 2026-09-16).
The bench's OWN measured influence stencil — taken from the anchor poke
`tg96_place` already traces, so it costs nothing extra — is deconvolved off a
multi-site poked map over the illuminated lattice (`dmg_act_fit`), and the
recovered command is regressed on the commanded one with `score_`'s gain
verbatim, `Ad(lit)\a(lit)`, so the gate and the battery report the SAME
quantity. Magnification divides out because the stencil and the map are both
measured through the same tail and both resampled into the DM frame; what
survives is whether a command at a site reappears at that site, with its
amplitude, without leaking to its neighbours. The stencil comes from the
anchor and the row is poked at DIFFERENT, well-separated sites — fitting the
map the stencil was built from would return ~1 by construction and gate
nothing. The sites are spread across the pupil, so a registration that
degrades off-axis is in the measurement rather than at one lucky pixel.

**Resampling uses `tg96_samp`, and the reason first given for it was WRONG.**
The claim was that the shared `dmg_samp` — which expresses the registration as
an axis permutation plus per-axis signs plus one isotropic scale, the 8-parity
family — could not represent this rig's mapping, because two folds at 20° and
25° must put a non-90° rotation into it. **Measured, that is false.**
`tg96_samp` now reports how far `mag·Linv` sits from the nearest signed
permutation, and on every bench tested it is **0.0 % away with anisotropy
1.0000**: the OAP rig at an exact 90° (a permutation), the lens rig at 0°. So
`dmg_samp` would have worked, and the fold angles do not enter. The physics the
first claim missed is that a fold mirror REFLECTS the pupil, it does not rotate
it about the axis; image rotation comes from out-of-plane fold geometry, and
these folds are coplanar. The fold angle drives aberration, not image rotation.

`tg96_samp` is kept anyway, on the weaker and honest grounds: it is
`tg96_place`'s own affine and resolved parity evaluated on the DM grid, so the
bench has one registration convention rather than two, and it stays correct if
a future layout does go out of plane. It is **not** load-bearing for
correctness on any bench measured so far. The diagnostic prints every run, so
the day a bench does leave the permutation family, the number says so.

Gate it yourself, without paying for a tune:

```matlab
tg96_tail('verify_tail','objwin3_tail.mat', <the bench args it was tuned with>)
```

`runs/gateseq3.sh` runs exactly that on both rigs — the OAP rig's known-bad
tail (must be REFUSED) and the lens rig's tuned tail (must be ACCEPTED). The
first leg is the non-vacuity: a gate that accepted everything would pass the
second leg alone.

## Calibration mode (Dave 2026-09-10, the ZWFS-S10 default)

`battery.calib_mode`:
- **`matrix`** (default) — the MEASURED response matrix dw/da: poke every
  `matrix_step`-th actuator on a sparse grid, step through the offsets so every lit
  actuator is poked once, cut each response from its own detector window (placed by
  the ray affine, `tg96_place`), assemble J and estimate commands by regularized LS.
  Windows are placed by the affine + a robust refit, NOT a parity search — this is
  what lets the folded OAP rig register where `register_two_pokes` cannot. On the lens
  it BEATS the kernel record (flat/single-10nm 0.9916/2.2pm vs 0.9654/21pm;
  flat/random 0.9894 vs 0.9203) with flat modal transfer 0.96–0.99 across the band.
- **`kernel`** — the record: `register_two_pokes` + one interpolated truth map,
  compared in detector-pixel space (Stage C–E as first shipped).

## D1 window-placement gate + D4 alignment sensitivity

- `tg96_run('bench.optics','oap','stages',{'bench','place'})` runs the D1 gate:
  response CoM vs the affine-predicted pixel for every lit actuator, with a
  non-vacuity check (the old axis-aligned mapping fails). **Lens 100% within 2px;
  OAP 75% (window containment 98.9%)** — the OAP shortfall is the fold's astigmatism
  (physical: single-poke == multiplexed; the exact-centre poke reads 0).
- `battery.d4=true` (OAP) runs Stage D4: perturb OAP1/OAP2 by `d4_dec_um` /
  `d4_tilt_urad` and re-read the null shift + the single-actuator differential. The
  null moves ~1.5 nm/µm and ~10 nm/µrad but the differential gain stays 0.99 — the
  null and alignment drift are common-mode and cancel in the differential.

## Results

Lens equivalence gate EXACT (model 1024); matrix calibration beats the kernel record;
the OAP reflective gauge images the DM, reads a localized differential as well as the
lens (single 0.9948/2.2pm), and its differential is robust to the 12.9 nm null and to
OAP misalignment — the fold's cost is astigmatism cross-talk (~0.42) on dense/high-
order patterns (random-10nm 0.75 vs lens 0.99). Full numbers + tables + departures in
**`REPORT_oap.md`**.

## Closed-loop hold metric (D7) — the on-orbit servo mode

`tg96_run('stages',{'bench','loop','figs'})` runs Stage LOOP: the DM held at the
30 nm working surface by a proportional loop (gain 0.5, 60 cycles) closed through
the four-step reading and its measured matrix (calibrated ON that surface). The
loop code is the **shared** `../dm_gauge_lib/dmg_loop.m` — the identical file the
ZWFS runs (`P.loop.*` knobs, drift `seed 77` shared with the ZWFS so both gauges
see the same realizations). Cost: K+1 traced states per (drift, photon level), an
hour-class job at model 1024 (`tg96_batch.sh`). **Lens result** (`runs/loop_lens`):
holds the walk and the photon noise like the ZWFS stepped/vector readings but at
**~2.6× the photons per measurement** (3 pm from 5.5e12 noise-only / 2.0e13 walk;
thermal floor 13.1 pm = the rate/g lag). **Departure:** the IFO four-step has a
**noiseless step floor ~8.6%** (unlike the ideal ZWFS stepped reading's 0.000) —
the gauge's geometric roll-off + modal cross-talk surfacing in closed loop
(L-like, not S-like). **OAP row** (`runs/loop_oap`, bare Al): the residual fold
cross-talk (a benign-looking 0.95 dense gain open-loop) becomes the DOMINANT
closed-loop floor — 3 pm noise-only at 3.4e13, the 2 pm walk floors at 4.1 pm
(never 3), thermal 39 pm, step floor ~27.6%. **The loop distinguishes the lens
(holds) from the OAP (does not) where the open-loop battery did not.** Numbers +
the ZWFS comparison table in **`REPORT_oap.md`**.
