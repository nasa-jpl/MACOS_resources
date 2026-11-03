# REPORT — the bench must be buildable

CCL for Dave, 2026-09-15. Branch `dev-candidate`. The realism round of
`BRIEF_ccmac_bench_realism.md`: node clearance, real substrates, a real camera.
Numbers first; every claim carries its run tag. Companions:
`REPORT_gauge_ifo.md` (the interferometer lanes), `REPORT_oap.md` (the
reflective build-out), `../zwfs_dm96/` (the sensors' runs).

Dave's finding, 2026-09-15, on the deck's layouts: *"the collimator lens, input
polarizer, compensator, analyzer and the BS interfere severely, blocking the
various beams. This is not buildable."*

## Status

| item | state |
|---|---|
| 1 — recomb distances forwarded, arm QWPs at the retro end, bench of record → 22.5° | **done** (CCL), resources `9b2181c`; sensors' gate run `zwfs_dm96/runs/gate22_193` |
| 2 — Stage A solved against the node parts; the measured table in the report | **done** (CCL), §2 below |
| 3 — thicknesses: 10 mm splitter and compensator, real lens edges; tail retune | **builder + runner in** (§3); runs queued as `runs/item4seq.sh` |
| 4 — substrates on the polarizers / QWPs / analyzer / masks; tail retune | **builder + runner in** (§3); runs queued as `runs/item4seq.sh` |
| 5 — the camera: pitch and binning in the parts lists, the body in the layouts | **runner in** (§4); prints on every run |
| 6 — the snapshot form's polarization at 22.5° (`tg_aoi_ladder`) | **tool in** (§5); runs queued as `runs/aoiseq.sh` |
| 7 — the OAP rig through the tool (CCL); the P/SRI rig (TO, `BRIEF_to_psri_clearance.md`) | **done** — `pdi_dm96/REPORT_gauge_pdi.md`, tag `psriclear2` |
| 8 — the interferometer's station-by-station figure | **runner in and PROVEN** — `oapifol2_stations.png` (2x7: mirror command, test-arm field, reference field, two phase steps, recovered surface, residual vs the engine) reads **0.00 pm on the flat and 626 pm on the 30 nm surface**. Two gaps being closed: it came out 2558 px wide because the width patch was not applied yet, and there is no LENS-rig figure. Both rigs regenerate at 1800 px in `runs/item2bseq.sh` after the patch |

Items 3–6 and 8 are `BRIEF_to_gauge_close` items 4, 5 and 6; the live status
table for the whole close-out is at the top of `REPORT_reflective.md`, and this
report carries the measurements as they land.

## 1. The bench of record is 22.5° (resources `9b2181c`)

Each arm's quarter-wave plate is placed ONCE, `D_QWP` before its retro, for
both passes — the "In" record used to sit inside the node, so the layouts drew
a plate where no plate is. `tg96_run` forwards `D_RECOMB` / `D_RC_L2`, so the
output quarter-wave plate and the analyzer sit 160 and 170 mm behind the
splitter in collimated space instead of 17 and 27, with L2 unchanged at 205 mm
(the tuned tail is untouched). `tg96_params` and `zwfs_params` carry
`BS_AOI 22.5`, `D_RECOMB 150`, `D_RC_L2 55`, compensator at 200 mm.

Gates: `tBench` 9/9, `tTgPol` 9/9, `tTgPol2` 9/9, `tDmgLoop` 15/15,
`tPropLayout` 3/3. Sensors on the new sheet, `zwfs_dm96/runs/gate22_193`
(model 1024, 193 rays, bench + battery, readings S / V / P): G1 1.8e-15,
G2 2.0e-15, G3 4.1e-16, G4 / G5 / G7 pass; capture range to 10 % S 37 nm,
V 62 nm, P 54 nm. The 22.5° bench does not move the sensors.

## 2. The splitter angle is now solved against the node, and the node is measured

Run tags `node22t` (the ruled 22.5°), `nodesolve` (the angle unpinned),
`clear22` (the measured table on the bench of record). The first two are
geometry runs at model 512 / 65 rays with a smoke DM grid — the node geometry
does not depend on either.

**What was wrong.** `tg96_run`'s Stage A solved the splitter angle against
three END bodies only — the DM, the reference flat and the camera — inside a
700 mm leg cap, and bought its separation by making the DM leg longer. Nothing
in it looked at the parts packed around the splitter, whose distances are fixed
by the bench block and can only be cleared by raising the ANGLE. That is why it
was content with 6.88° while eight of nine node parts sat inside a beam.

**The rule, and why it is not `d·sin(2·AOI)`.** Put the splitter at the origin
with the input beam along +x: the test arm runs +x, the reference arm leaves at
2·AOI, and the output leg is opposite the reference arm. The separation that
matters is measured IN THE PART'S OWN PLANE, so a part held normal to its own
beam at distance `d` is crossed by the opposing leg at lateral distance
**`d·tan(2·AOI)`**. `d·sin(2·AOI)` measures across the other beam instead and
understates it. A plate held PARALLEL TO THE SPLITTER — the compensator, by
construction — tilts its plane by the AOI, which shortens the crossing;
`d·sin(2·AOI)` is the conservative stand-in there. The part clears when that
separation covers the beam radius, the part's own radius, an 8 mm mount and the
25 mm margin. A builder plate carries no aperture, so its radius is the beam
+ 5 mm — the rule `dmg_bench_clearance` uses when it reads `aprad` 0.

**The rule against the measurement**, at 22.5°, mm:

| node part | d | rule | measured | rule − measured |
|---|---|---|---|---|
| collimator L1 | 257.1 | +146.3 | +153.1 | −6.8 |
| input polarizer | 247.1 | +131.3 | +137.5 | −6.2 |
| compensator (∥ splitter) | 200.0 | +25.6 | +38.2 | −12.6 |
| output QWP | 160.0 | +44.1 | +53.6 | −9.5 |
| analyzer | 170.0 | +54.1 | +63.8 | −9.7 |
| focuser L2 | 205.0 | +94.1 | +99.3 | −5.2 |

Conservative on every part: by 5–10 mm where the plane is normal to the beam,
by 13 mm on the tilted compensator. The tool is the measurement; the rule is
the screen that picks the angle before there is anything to trace.

**The solved angle is Dave's ruled angle.** Binding angle **22.39°** — the
compensator binds, at 200 mm the closest node part to the splitter — against
6.88° from the end bodies. Unpinned, Stage A now returns **23°** (`nodesolve`;
it rounds up to the degree). Dave ruled 22.5° on 2026-09-15 from the measured
table, before this solve existed; the two agree to a tenth of a degree, and the
0.11° shortfall is covered by the rule's own conservatism (the compensator
measures +38.2 mm, not the +25.6 the rule claims).

**The measured table is now printed by the report.** New stage `'clearance'`
(`P.stages`) runs `dmg_bench_clearance` on THIS run's own rig — passed as the
new `'G'` option, so the table describes the bench the run actually used rather
than a rebuild of it from another param file — and tees the part-by-part table
into `<tag>_report.txt` with `<tag>_clearance.png`. It runs LAST: the tool
traces both arms through its own temporary decks and leaves the engine on them,
so no stage downstream may depend on which deck is loaded.

Measured at 22.5°, worst first (`node22t`, mount +8 mm): compensator **+38.2**,
output QWP +53.6, analyzer +63.8, L2 +99.3, input polarizer +137.5, L1 +153.1,
test QWP +333.6, DM +361.1, reference QWP +362.8, PZT flat +390.9. The
splitter, the field lens and the detector have no other beam crossing their
plane. **Worst +38.2 mm over 10 parts; spec ≥ +25.** The bench is buildable.

**Two smaller things landed with it.** `dmg_bench_clearance`'s node drawing
skips a label whose part the rig does not carry, instead of erroring — the OAP
variant renames its collimator, and the P/SRI rig (TO's half of item 7) is a
different train again. And the Stage-A line now says whether the angle in force
was solved or pinned in the param file, and prints both binding angles.

## 3. Thicknesses: the 10 mm plates cost 150× of null, and the tail cannot take it back

Run `thk22_tail` — the lens rig's tail retuned with the splitter and compensator
at **10 mm** (the flatness a 4-inch plate needs) and real 4 mm singlet edges.

| | flat-DM null |
|---|---|
| tail of record, 2.6 mm plates | **0.134 nm** |
| geometric seed, 10 mm plates | 75.24 nm |
| **retuned, 10 mm plates** | **20.09 nm** |

**The tuner converged** — evaluations 100, 101 and 102 all sit at 20.094, 20.094
and 20.095 nm, so 20.09 is its floor on this bench and not a stalled search. It
improved the seed by 3.7×. What it cannot do is get back to 0.134.

**The mechanism is the shear, and it is the one the brief predicted.** A 10 mm
plate at 22.5° with n = 1.5 displaces the transmitted beam by

`t·sinθ·(1 − cosθ/√(n²−sin²θ))` = **1.39 mm**,

against the realism brief's predicted 1.4 mm. The compensator balances the
*path*, but the two arms now sample the collimator 1.39 mm apart, and the
difference of L1's own aberration across that shear is what the field-lens tail
has no freedom to remove: its four parameters (`FL_F`, `FL_Kc`, `D_MASK_FL`,
`DET_TRIM`) all act on the common tail, not on an arm difference.

**What this does and does not mean.** A 20 nm *static* arm difference is not
automatically a 20 nm error in the gauge: every row of record is DIFFERENTIAL,
measured against a reference frame taken on the same bench, so a fixed null
largely divides out. Whether it survives that is exactly what the `thk22` gate
run measures, and that number — not this one — is what belongs on a slide. It
is recorded here first because it is the part that is already certain: the
thick plates are a real optical cost, and no amount of tail tuning removes it.

**The gate ate this winner, and the run below does not yet carry it.** The
advisory edit reached `tg96_tail.m` a few seconds after `thk22_tail`'s MATLAB
had already loaded the enforcing version, so the old gate refused a converged
winner on a measure the two-leg test had just shown to be wrong:

```
TAIL GATE REFUSED the winner: actuator-space gain -0.8742 < 0.95.
Falling back to the GEOMETRIC SEED (gain -0.8803).
```

Both readings are ≈0.87 — the threshold decided it, not any difference between
the tails. So `thk22_tail.mat` holds the **seed** and `thk22` ran on it. Its own
report line is the tell: *"RE-TUNED set … (null 75.242 nm at opt res; seed
75.242)"* — a retune whose null equals its seed's did not retune.

**And the pupil image moved — but the sampling did not, and the two do not yet
add up.** Measured, `lensuw2` (no thick plates, tuned tail) against `thk22`
(10 mm plates, seed tail):

| | mag | dxd mm | **px / actuator** | lit actuators | pupil image |
|---|---|---|---|---|---|
| `lensuw2` | 9.8411 | 2.0263e−2 | **5.015** | 3260 | 7.78 mm (384 px) |
| `thk22` | 10.3490 | 1.9452e−2 | **4.967** | **3364** | **5.79 mm (298 px)** |

*(An earlier note here said "3.1 px per actuator instead of 4.0". That was
wrong — it divided the pupil's pixel count by the actuator count instead of
reading the affine, which is what actually sets the sampling. The sampling is
essentially unchanged, 5.015 → 4.967.)*

**There is a tension in these three columns that this report does not resolve.**
The pupil image is 22 % smaller in pixels, yet the pixels per actuator are the
same to 1 % and MORE actuators are lit (3364 against 3260). A smaller pupil at
unchanged sampling should light fewer actuators, not more. So the detector mask
used for the camera line and the lit-actuator set from `dmg_lit` are not
measuring the same region, and until that is understood none of these three
numbers should be read as a thickness result.

It is recorded because it is what the runs say, not because it is understood —
and because the re-run on the tuned tail (`item4bseq`) is the experiment that
separates the glass from the tail and will either reproduce the tension or
remove it.

`runs/item4bseq.sh` re-runs both with the advisory gate in place. **The 20.09 nm
figure above stands** — it is what the tuner measured before the refusal — but
the gate run's ROWS have to be re-taken on it, and until they are, `thk22`'s
rows describe a 75 nm-null bench. `sub22_tail` is unaffected: it starts long
after the edit and keeps its winner.

### 3.1 RE-TAKEN 2026-09-16: the rows on the tuned tail

`item4bseq` re-ran both. The retune reproduced its winner exactly — seed
**75.2422 nm → 20.0938 nm**, 3.7× — and the gate, now advisory and now reading
by lattice deconvolution, reported 0.9104 and **kept it**. So `thk22_tail.mat`
holds the tuned set and `thk22`'s rows describe a 20 nm-null bench. These are
the rows for the slide.

| | |
|---|---|
| Stage C, single actuator @150 nm | gain **0.9885**, off-target floor **40.0 pm** |
| Stage D, modal transfer | gain **1.0049 → 1.0077** from 0.7 to 45.3 cyc/pup, **0.9626** at 67.9; cross-talk ≤ **0.027** |
| Stage E, differential rows | gain **1.0023–1.0085**, resid **1.8–147.0 pm**, corr ≥ **0.9999** |
| Stage E, reg sweep | 1.0063 / 1.0116 / 1.0122 at matrix_lam 1e−3 / 1e−4 / 1e−5 |

**Break ladder** (single 10 nm differential vs base rms), with the new
unsaturated meter:

| base rms | gain | floor pm | corr | meas rms | fold | |
|---|---|---|---|---|---|---|
| 30 nm | 1.0139 | 6.4 | 0.9997 | 26.6 nm | 0.000 | |
| 60 nm | 1.0226 | 9.3 | 0.9994 | 52.1 nm | 0.002 | |
| 120 nm | 1.3674 | 555.4 | 0.4887 | 75.8 nm | 0.013 | |
| 240 nm | −4.1237 | 1511.2 | −0.6885 | 79.5 nm | 0.016 | **BROKE** |
| 480 nm | 1.0420 | 504.2 | 0.4216 | 79.6 nm | 0.016 | |

The bench holds to 60 nm with a 9 pm floor and breaks at 120 — the same rung
both rigs break at (§4.7(a) of `REPORT_reflective.md`), so the glass has not
moved the capture range.

### 3.2 NEW — the 10 mm plates cost ~13 % of the READING, and only the new meter can see it

The `meas rms` column is the measured base rms, the same quantity the `wrap`
stage reports. Against the commanded base it reads:

| base | thk22, 10 mm plates | record lens rig, 2.6 mm (`wraplens`) |
|---|---|---|
| 30 nm | 26.6 nm (**0.887**) | 31.1 nm (1.037) |
| 60 nm | 52.1 nm (**0.868**) | 61.5 nm (1.025) |
| saturated | **79.6 nm** | 90.9 nm |

The record rig tracks the command to a few per cent; the 10 mm bench reads
**13 % low and does so consistently**, at both unwrapped rungs AND in
saturation — 79.6 against the analytic wrapped-uniform 91.34 nm is **0.872**,
the same factor. A common multiplicative attenuation of the reading is the only
thing that moves all three by one factor.

**This is measurement amplitude, and it is the hypothesis §4.7 excluded for the
RIG comparison — excluded there, present here.** Nothing about the earlier
exclusion changes: it was about OAP versus lens optics, and it still holds
(7 %, wrong direction, §4.7). The plates are a different variable and they do
attenuate.

**The old meter could not have found this.** `max|h|/(λ/4)` saturates at 1.00
from 120 nm up on every bench, so it reports the same number for a rig reading
at full amplitude and one reading 13 % low. The measured rms separates them at
every rung, which is what the staged patch was for.

Worth noting what does NOT move: Stage C's gain is 0.9885 and the differential
rows sit at 1.002–1.009. The MATRIX is calibrated on this bench and absorbs a
common scale exactly, so the attenuation costs SNR and floor, not gain — which
is why it is invisible in every gain-based row and shows only in the reading.

### The two enablers earned their keep on this run

**`bench.MASK_TRIM 'scan'` found the focus the glass moved.** The sheet carries
−5.582 mm, the seed-to-true-focus correction found once by the S1 rounds. With
the thicker parts in, the scan re-found it at **−4.7229 mm** in 81 s, a move of
**+0.859 mm**, and reached a mask-plane peak/sum of **2.752e−02** — sharper than
the ~1.2e−2 `zwfs_s1` records at focus.

The move is accounted for: `EDGE_MARGIN` 2.0 → 4.0 mm makes every singlet 2 mm
thicker, and a thicker lens moves its focus by `t(1 − 1/n)` = **0.667 mm** per
lens. Carrying the sheet's constant would have seated the mask 0.86 mm off
focus and then charged the blur to the glass — which is precisely the failure
the scan exists to prevent, and it would have been invisible in the output.

**`dmg_cam_line` found a binning the sheet had wrong.** On the ZWFS rig it
reads *pupil image 7.51 mm across (192 modeled px at 39.1 um); sCMOS at 6.50 um
→ 1155 raw px across the pupil; binned 4 = 289, and **binning 6 = 193 lands
nearest the modeled 192***.

So the sheet's binning of 4 **oversamples** the model by 50 % here, where on the
PSI rig it undersamples by 22 %. One binning number cannot serve both rigs, and
without the "lands nearest" column neither error is visible — the line would
just print 289 beside a modeled 192 and leave the reader to notice.

Also worth recording against the realism brief's own figure: it assumes a
**9.4 mm** pupil image. Measured, the ZWFS rig forms **7.51 mm** and the PSI rig
**7.78 mm** (5.79 on the thickened seed-tail run). The 9.4 mm figure is not this
bench's.

## 5. The snapshot form's polarization at the angles each rig is BUILT at

`tg_aoi_ladder` at 22.5° on the lens rig, and on the reflective rig at its own
20°/25° folds with the 22.5° plate. Model 256, NGRID 63 — geometry and
polarization, not a diffraction result.

| rig | az test | az ref | **arm rotation** | **unaligned PSI gain** | residual |
|---|---|---|---|---|---|
| lens, 22.5° | −43.4361° | +45.0000° | **+1.5639°** | **−1.00879** (+0.879 %) | 1.448e−03 |
| reflective, 20/25° + 22.5° plate | +43.4275° | −45.0086° | **−1.5639°** | **0.99017** (−0.983 %) | 5.511e−04 |

**Against the record's 45° figures — 7.48° of arm rotation and a gauge reading
11.7 % high — the built angles cost 4.8× less rotation and 13× less scale
error.** Both rigs land at ±1.56° and under 1 %. The two rigs' rotations are
equal and opposite, which is the plate's diattenuation acting on arms whose
design azimuths are mirrored; the reflective rig's extra metal folds do not add
to it measurably at these angles.

The **residual** column is the part that matters for a calibrated gauge: the
pupil-VARYING fraction of the gain map, which a matrix measured on the bench
cannot absorb. It is **1.4e−03 on the lens rig and 5.5e−04 on the reflective**
— i.e. after the matrix, the polarization systematic is at the 0.1 % level, not
the 1 % the uncorrected gain shows. Quoting the uncorrected number overstates
what a calibrated gauge suffers by roughly 6×.

### The "corrected" column is a NO-OP, and this is why

`gain_cor` is identical to `gain` to five decimals on both rigs. That is not a
coincidence and not a bug in the arithmetic — **it is a design error in how I
built the correction**, and it is provable in three lines.

A polarization four-step has `I(θ) = A + B·cos(2θ − φ)`. Re-referencing every
analyzer angle by a constant `c` gives

```
I(c) − I(90+c)   = 2B·cos(φ − 2c)
I(45+c) − I(135+c) = 2B·sin(φ − 2c)   ->   fourstep = φ − 2c
```

so the shift subtracts `2c` from the measured phase of **both** states — and the
ladder's gain is built from their DIFFERENCE, where the common `−2c` cancels
exactly. A rigid rotation of the analyzer set cannot correct a differential
measurement, however the rotation is chosen.

**So the brief's "corrected" column is not delivered.** The uncorrected rotation
and gain are real and are the numbers above; the analyzer-sweep correction needs
to change the PROJECTION — using the measured azimuths to build non-uniform
steps, or rescaling by the arms' non-orthogonality — not to rotate all four
angles together. The `resid` column is unaffected by any of this, since it is
computed from the gain map's own scatter about its median.

## 6. Item 6 — the interferometer's station figure, both rigs (`stnoap`, `stnlens`)

Both figures exist, at the width the brief asks for: **1800 × 560 px**. The
pre-patch figure was 2558 × 838 — `exportgraphics` at `Resolution 150` does not
land on a figure's pixel width, `print -dpng -r96` does on this 96 dpi box, and
that is the mechanism the ZWFS sibling already uses, which the deck holds beside
this one.

**The OAP figure is unchanged by the patch, and that is asserted rather than
assumed.** `stnoap` reports `0.00 pm (flat), 626.43 pm (30 nm rms)` — bit for
bit what `oapifol2` reported before the patch and before the item-3 gate work.
So the width change and the `MASK_SUB`/camera edits are inert on this path.

### OPEN — the lens leg's station residual is 100× the OAP leg's, and it is new information

| rig | flat DM | 30 nm rms working surface |
|---|---|---|
| OAP (`stnoap`, `oapifol2`) | 0.00 pm | **626.43 pm** |
| lens (`stnlens`) | 0.00 pm | **62 067.17 pm = 62.1 nm** |

This is the **first lens station figure ever made** (the brief asked for it
precisely because there was none), so there is no baseline it regressed from,
and the OAP leg's bit-identical number rules out the patch as the cause.

**What the number is.** The last panel is `d = h − ht`: the four-step recovered
surface minus the ENGINE's own phase difference at the detector,
`ht = angle(exp(i·(∠Et − ∠E0)))·λ/4π`. Both are wrapped quantities with an
unambiguous range of ±λ/4 = ±158.2 nm of surface. It is the gauge's error, not
a model of it.

**What the signature says.** The residual is **exactly 0.00 pm at the flat DM on
both rigs** and only diverges once there is structure to disagree about; on the
lens rig the residual panel is ±200 nm while the recovered surface is ±100 nm,
i.e. the difference is LARGER than either map — which is what two uncorrelated
maps of the same rms give (√2 × 44 nm ≈ 62 nm). A residual that vanishes at
flat and reaches √2 × the map with structure is the signature of a **lateral
misregistration between the recovered map and the engine field**, not of a noise
floor or of an amplitude error.

**One clue, from the two runs' own printouts.** The legs did not run on the same
kind of tail: `stnoap` used the **geometric seed** (`Tail: geometrically-scaled
seed (no oap_tail.mat)`) — the OAP rig's tail of record — while `stnlens` used
the **re-tuned** `lens_tail.mat` (`null 0.134 nm at opt res; seed 9.100`). So the
leg that reads 62 nm is the tuned one and the leg that reads 626 pm is the seed
one. That is a correlation and not yet a cause — the station figure's `h − ht`
comparison is a different path from the battery that certifies `lens_tail` at
0.9968 — but it is the first thing to vary, and it is free to vary
(`bench.tail_from_mat false` on the lens leg).

**Not chased here, deliberately.** The queue's item 6 is the figure and the
figure is delivered; this is a separate defect on a leg that had never been
measured. The next steps are cheap and specific, in order: re-run `stnlens` with
`bench.tail_from_mat false` and see whether the residual collapses to the OAP's
order; then correlate `d` against `h` shifted over a few pixels and read off the
offset; then decide whether it is the lens rig's `MASK_TRIM` (`zwfs_params`
carries −5.582 for this rig where the OAP needs 0) or the engine-field
comparison picking a different pupil station.

**For the deck: do not put the two numbers side by side yet.** The OAP slide's
626 pm stands. The lens figure is sound as a picture of the signal chain — the
first six panels are the tool's own output and read correctly — but its seventh
panel's headline number is an open defect, and 626 pm against 62 067 pm on one
slide would assert a rig-to-rig quality difference that is not established.

## 6. Pupil image quality of the detector leg (Fang Shi's request; CCL, 2026-09-16)

Runner `tg96_pupilq.m` (sheet-driven name/value; `runs/pupilq_lens`,
`runs/pupilq_oap`; model 512, 129 rays across; both rigs' decks of record,
the lens rig's tuned tail and the mirror rig's geometric seed).  **The DM is
the stop** (Dave): the bench as built, the point source of record, the DM
declared the stop, the field a lateral shift of the source at the
collimator's focus, i.e. a tilt theta about the DM (a spatial frequency
theta / lambda on its surface).  At the camera, the DM's exit pupil, two
traces a field step apart cross at the image of each DM zone; the cloud
gives the pupil surface, the distortion against the runner's single global
affine, and the blur (the zone's image walk over the actuator tilt band,
|theta| <= 3.2e-4 rad = lambda / 2 mm).  At the seat the rodgers2 set per
tilt: the spot at the best focus found from the rays, the wavefront with
piston, tilt and focus removed, the centroid against focal length times
tilt.

| | lens rig (tuned tail) | mirror rig (seed tail) |
|---|---|---|
| magnification, camera mm per DM mm | 1/9.90 (runner's affine 1/9.88) | 1/10.56 (runner's 1/10.44) |
| pupil distortion vs the global affine, DM mm rms / max / outer third rms | **0.13 / 0.30 / 0.18** | **0.45 / 0.85 / 0.66** |
| pupil surface: defocus / astig 0 / tilt, mm of sag over the pupil radius | 3.83 / -2.07 / 0.00 | -0.52 / 0.17 / 0.45 |
| pupil blur over the actuator band, DM mm rms / max | 0.023 / 0.060 | 0.032 / 0.093 |
| focal plane, on axis: spot at best focus / WFE | 0.43 um (0.16 lam F/D) / 1.1 nm | 0.00 um / 0.00 nm |
| focal plane at the actuator-band tilt (52 lam/D off axis): spot / WFE | 0.43 um / 2.7 nm | 5.3 um (2.3 lam F/D) / 144 nm |
| focal plane at 1 mrad (163 lam/D): spot / WFE | 0.47 um / 7.8 nm | 16.5 um (7.1 lam F/D) / 451 nm |
| effective focal length from the centroid | 437 mm | 379 mm |

Reading it:

- **The pupil image is sharp on both rigs.**  A DM zone's image walks
  0.02-0.03 mm rms (6-9% of a detector pixel of 0.41-0.44 DM mm, 2-3% of
  the pitch) over the whole band of spatial frequencies the DM can make.
  The pupil surface is curved (3.8 mm of sag on the lens rig, from the
  field lens) and tilted on the mirror rig (0.45 mm, the 25-degree fold),
  and neither matters at these tilt angles: the blur they cause is the
  walk above.
- **The distortion is the finding.**  Against a single global affine the
  DM's image is off by 0.13 mm rms and 0.30 mm at the edge on the lens rig
  (a radial, barrel-shaped residual) and by 0.45 mm rms, 0.85 mm at the
  edge, on the mirror rig (3.4x, the off-axis parabolas).  That is a third
  to nearly a full actuator pitch of registration error at the edge if a
  bench used one affine to say which pixel is which actuator.  The
  measured response matrix does not use it (each actuator's column is
  measured where it lands, which is why the rows hold at 0.99 on both
  rigs); anything that does -- a stencil placed by the affine, the tuner's
  single-site verify measure, a per-actuator map read off the camera by
  geometry -- inherits it.  A per-zone calibration of the mapping (a
  distortion map, one measurement) removes it; the OAP rig's 62 nm
  station-figure misregistration (item 6 open) is the size this predicts.
- **The mirror rig's focal plane is perfect on axis and coma-limited off
  it.**  0.00 lam F/D at zero tilt (TO's item-5 result), 2.3 lam F/D at the
  actuator-band tilt, 7.1 at 1 mrad, linear in the angle: an off-axis
  parabola pair has no field correction.  This is not a pupil-image cost
  (the walk at the camera is the number above) and not a mask-sensor cost
  (the dimple and pinhole act within 2 lam/D of the core; light diffracted
  by a DM ripple lands 52 lam/D away and passes the mask whatever its
  shape), but it is the number to know before any sensor that reads the
  field off axis at the seat.
- **Two lessons for the record:** an ideal collimated source at the DM
  measures a bench that does not exist (the lens rig's tuned leg lands 16
  lam F/D of blur on it: the leg is tuned to the collimator's actual
  beam, fed 25 mm inside its focus); and the interferometer deck's seat
  marker is not at the focus (352 um of spot there; the best focus is
  5.5 mm away on the lens rig), so the spot is scored at the focus the rays
  define.

Figures: `pupilq_<rig>_pupil.png` (distortion arrows and the blur map over
the DM), `pupilq_<rig>_focal.png` (spot, wavefront and centroid residual
against tilt, both azimuths).  Open: the chief-tied exit-pupil sphere
(the engine's Return/Return/plane idiom) refused to load through the mex
without a message; the fitted focus removal stands in for it to first order.

## 7. The pupil image simulated: the DM's modes through the detector leg (Dave, 2026-09-17)

Runner `tg96_pupilsim.m` (sheet block `P.pupil`; `runs/pupilsim_lens`,
`runs/pupilsim_oap`; run-it-yourself: `./tg96_pupil_batch.sh both`).  Section 6
measured the pupil image geometrically; this section propagates the DM's field
through the leg and reads it out as the interferometer does, so Fang Shi's
question -- do distortion, field curvature and the rest limit the observability
of the DM's modes -- gets a number per mode and per pupil zone.

**Method, three stages.**  (1) The leg's coherent PSF, zone by zone, from the
engine's rays: the bench as built, the DM the stop, a two-dimensional set of
tilts over the actuator band (rings at 0.5, 1, 2 and 3.2 x 1e-4 rad and a
margin ring at 1e-3, eight azimuths, 41 traces).  For every ray -- every DM
zone -- the detector-plane intercept and the exit angle against tilt give the
zone's transverse ray aberration over the band aperture, which integrates to
the zone's wavefront W(a); the zone's coherent transfer function is exp(-ikW)
and its inverse transform the zone's complex PSF.  The DM as the stop keeps
each ray on its zone to 2 um across the whole set, which is what makes the
walk a PSF and not a mapping change.  (2) The DM field exp(i 4 pi h / lambda)
on the DM's own grid (0.125 mm, 1024 px) is filtered zone by zone (overlap-add
over 8 mm raised-cosine patches), the reference arm goes through the same
operator, and angle(Et conj Er) is the recovered surface.  Test surfaces:
sinusoids at the actuator Nyquist (2 mm period), 4, 8 and 16 mm; single
100 nm pokes (Gaussian influence, 0.85 mm 1/e) at six sites out to the
outermost lit ring; the record's 30 nm working surface (random commands,
seed 7).  Gain and amplitude cross-talk are read by demodulation, per radius.
(3) A plane-to-plane check.  The engine's own propagation through reference
surfaces inserted in the .in file (the CTB model) was built as
`tg96_pupil_engine.m`; it is not yet a valid check of this leg -- see the end
of this section.  A standalone paraxial Fourier chain with exact thick-lens
screens is in the runner as a standby (`fourier` true).

**What the stages settled first: the beam of record was the source cone, not
the DM.**  On both rigs no element clips a ray.  The builder sizes the source
cone to the baffle (2 atan(R_BAFFLE / D_SB) x FILL, a full cone angle in the
engine's convention), and that cone reaches 38.6 mm at the DM on the lens rig
and 41.0 mm on the mirrors: a 77 / 82 mm beam on a 96 mm DM, the outer
actuator rings unlit, the record's 7.8 / 8.0 mm pupil image its direct
consequence.  Dave's ruling (2026-09-17): the baffle must not constrain, and
the DM carries the aperture.  The simulation opens the baffle, widens the cone
so the beam is 1.06 x the DM aperture, and puts a 48 mm aperture on the DM (the
96 mm actuator footprint): the DM clips 11% of the rays and is the stop in
fact.  The sheets follow (`tg96_params`, `zwfs_params`: R_BAFFLE 12.5 -> 18,
D_LENS 60 -> 66 so the lenses cannot be the stop, R_TO_AP 30 -> 28); runs
emitted before this date carry the 77 / 82 mm beam.

**The leg per zone (stage 1).**  Within the actuator band the zone wavefront is
defocus and astigmatism and nothing else: the third- and fourth-order part is
0.3 nm rms on the lens rig (0.05 on the mirrors) against 24 nm rms (3.6) of
band-edge wavefront.  So each zone's PSF is a Fresnel kernel set by how far
that zone's image lies from the detector plane -- the pupil SURFACE against the
detector, which stage 1 measures directly:

| | lens rig (tuned tail) | mirror rig (geometric seed) |
|---|---|---|
| zone image vs the detector plane, on axis | +2.61 mm (downstream) | -0.40 mm |
| over the pupil: mean / min / max | +4.32 / +2.63 / +6.04 | -0.62 / -1.30 / -0.31 |
| astigmatic split of the image, rms / max | 1.05 / 1.85 mm | 0.09 / 0.16 |
| band-edge (Nyquist) quadratic phase, rms / max | 0.236 / 0.322 rad | 0.035 / 0.068 |
| phase gain at the band edge, worst = cos | 0.949 | 0.998 |
| amplitude cross-talk at the band edge, worst = sin | 0.32 | 0.07 |
| compromise plane (mean image) | +4.32 mm | -0.62 mm |
| band-edge phase there, rms / max | 0.052 / 0.092 | 0.013 / 0.035 |
| distortion vs one global affine, rms / max | 0.27 / 0.55 mm | 0.72 / 1.25 |

**Are we imaging at the best pupil image?  Not on the lens rig.**  The tuned
tail put the detector 2.6 mm ahead of the on-axis image and 4 to 6 mm ahead of
the edge zones' images: the tail was tuned on the flat-DM null, which is blind
to pupil defocus (the OAP tuner failure of REPORT_reflective 4.5 is the same
blindness).  Moving the detector 4.3 mm downstream, to the mean of the image
surface, cuts the band-edge phase four-fold; what remains is the bowl's
residual (+-1.7 mm) and the field lens's astigmatism (1 mm split at the edge),
which no detector position removes.  The mirror rig's geometric seed placed
its detector within 0.4-1.3 mm of the image surface -- the thin-lens conjugate
is a better guide than the null-tuned trim.

**The DM's modes through the leg (stage 2).**  Phase gain (recovered / true)
by pupil radius, u direction (v identical to 3 digits):

| lens rig | center | 0.5 | 0.9 | worst in the lit pupil | amplitude cross-talk, max |
|---|---|---|---|---|---|
| Nyquist (2 mm), as built | 0.990 | 0.985 | 0.973 | 0.954 | 0.29 |
| 4 mm, as built | 0.999 | 0.999 | 0.998 | 0.997 | 0.07 |
| 8 mm / 16 mm, as built | 1.000 | 1.000 | 1.000 | 1.000 | 0.02 / 0.00 |
| Nyquist, detector +4.3 mm | 0.997 | 0.999 | 0.998 | 0.993 | 0.12 |
| single 100 nm poke, peak / true (6 sites) | 0.996 | 0.993 | 0.989 | 0.987 at the outermost ring | -- |
| 30 nm working surface, recovered - true (piston, tilt out) | | | | 1.2 nm as built; 0.4 nm at +4.3 mm | |
| **mirror rig**, Nyquist as built | 1.000 | 1.000 | 0.999 | 0.997 | 0.07 |
| working surface | | | | 0.23 nm as built; 0.09 at -0.6 mm | |

**Reading.**  Every DM mode is observable on both rigs.  On the lens rig as
tuned the highest frequency loses 1% of gain at the center and 5% at the
edge, the modes at half Nyquist and below are within 0.3%, a single actuator
reads at 0.99 everywhere, and the 30 nm working surface comes back with 1.2 nm
of error -- 4%, most of it the Nyquist content of random commands.  The
amplitude cross-talk (a third of the phase modulation converts to intensity at
the pupil edge) the four-step ignores by construction, and the sensors that do
read amplitude do not use this leg.  Moving the detector 4.3 mm downstream
takes the Nyquist gain to 0.993 worst and the working-surface error to 0.4 nm.
The mirror rig is at the image already: 0.997 worst, 0.23 nm.  The distortion
(0.27 mm rms on the lens rig, 0.72 on the mirrors, against a single affine) is
a mapping, not a blur: the measured response matrix absorbs it (section 6), a
per-zone calibration removes it for anything that uses the affine.

**Recommendation.**  (1) On the lens rig, set the detector at the image
surface's mean (DET_TRIM + 4.3 mm at the tuned tail; a knob for the tuner: the
image position from the rays, not the null), and re-read the rows.  (2) Keep
the DM as the stop in every deck from here (the sheet change), and re-run the
record's rows on the 96 mm beam when the queue allows; the outer four actuator
rings enter the lit set.  (3) For the sensors' rigs the same stage applies
unchanged (the leg is shared); their masks sit at the seat, before the leg.

**The engine's plane-to-plane check: built, and why it does not yet check
this leg.**  `tg96_pupil_engine.m` inserts the CTB-style reference surfaces
into the deck: the symmetric sphere pair around the mask (NF1 / NF2) and a
sphere after the field lens concentric with the beam there (its center, the
field lens's image of the focus, is 796 mm downstream: the collimator of
record feeds the focuser a slightly converging beam, so the focus sits 45 mm
ahead of the field lens, just outside its focal length), carrying the engine's
scaled sphere-to-sphere Fresnel step (NFS1surf: effective distance
(Z2-Z1) Z1/Z2, pitch scaled by Z2/Z1) to the detector.  Measured: the field the
engine seeds on that sphere is correctly referenced (its phase matches the OPD
command, 0.3 waves rms); the propagated flat pupil is a clean disc under every
zElt convention; and every convention gets the DM modes wrong by design,
because a geometric-to-physical hand-off is exact only where the surface is a
conjugate of the DM.  This leg has none before the detector: the DM's image by
the focuser is virtual, 876 mm past the focus, and the image surface lies past
the detector -- so the per-ray seed on any upstream surface is the undiffracted
projection of the DM pattern, which is exactly the effect under test.  (The
first attempt, with the mask sandwich centered on the deck's seat marker
5.5 mm from the focus, seeded 54 waves of defocus and propagated that; also
measured.)  The CTB's spheres sit at real pupils, so the CTB never meets this.
The faithful engine version is the CTB's station-to-station idiom -- every leg
from the DM propagated (a collimated NFPlane leg to the focuser, the quartet at
the focus, scaled legs through the field lens to the detector), the field never
handed back to rays -- with each leg's zElt convention validated on the flat
pupil and a known defocus.  That is the next step; until it lands, the zone-PSF
model is the result of record and the paraxial standby chain its check.

**The station-to-station form, tried (`tg96_pupil_s2s.m`, the same day).**
Built as the CTB emits it: four collimated NFPlane legs at the chief pierces
between the DM and the focuser (the plates per index between them), the
through-focus quartet with an entrance sphere just after the focuser centered
on the true focus (radius 414.5) and a far-side sphere just before the field
lens (radius 44.8; the Rx_Coro asymmetric form), the lens per index, a sphere
concentric with the exit beam after it, and the scaled step to the detector
under all four zElt conventions.  On the lens rig it fails upstream of the
converging step, and the probe says why: the rig's "collimated" space is not
collimated -- the collimator of record is fed 25 mm inside its focus, 41
waves of curvature across the pupil -- so between the NFPlane legs the rays
walk up to a millimeter off the fixed grid, and the per-index phase of each
element lands on the wrong pixels: 1.5 waves of spurious aberration on the
entrance sphere and a focal spot ten times its size.  The CTB never meets
this because its pupils are truly collimated.  The mirror rig of record has
its collimator fed at the focus, so the chain is being validated there (a
10 mm known defocus discriminates the converging step's convention); on the
lens rig the chain needs a bench emitted with the collimator at its focus and
the tail re-tuned to it.
On the mirror rig (fed at the focus: the rays' spot at the marker is 0.00 um)
the chain was run with the exit sphere oriented for the diverging exit beam
(the field lens is 10.8 mm past the focus, inside its focal length; the
virtual crossing is 17 mm upstream).  Open at the end of the day: the pupil
the quartet lands on the far-side sphere reads 1.12 mm in radius against the
1.40 the rays give there (20% small; 4% small on the lens rig), and the
Nyquist readout at the detector is not a gain map under any of the four
conventions.  The next checks, in order: the focal-plane field's spot size
after the entrance sphere (it must be diffraction-limited on this rig), the
far sphere's pitch bookkeeping against the rays (the 20%), then the exit
step alone on a flat pupil with a known 10 mm defocus.  The zone-PSF model
stands as the result of record.

### 7.1 Improving the pupil image: the options assessed (Dave's ask, 2026-09-17 afternoon)

`tg96_pupil_options.m` runs the simulation on surgered decks of the lens rig
(`runs/pupil_options`, tags `popt_<variant>`).  Lens rig unless stated:

| variant | zone image vs the detector, mm: on axis / mean / min / max | astig split rms / max | Nyquist gain as built, mean / worst | at the compromise plane | 30 nm surface error, as built -> compromise | distortion vs one affine, rms |
|---|---|---|---|---|---|---|
| record (tuned tail: field lens 39.8 mm past the marker, conic -2.59) | +2.6 / +4.3 / +2.6 / +6.0 | 1.05 / 1.85 mm | 0.979 / 0.954 | +4.3 mm: 0.9985 / 0.9934 | 1.22 -> 0.42 nm | 0.27 mm |
| **seed tail** (field lens 10.8 mm past the marker, f 42.9, conic -2.11, detector at the thin-lens conjugate) | -0.5 / -0.7 / -1.0 / -0.5 | 0.08 / 0.14 | 0.9992 / 0.9986 | -0.7 mm: 0.9999 / 0.9999 | 0.24 -> 0.06 | 0.003 |
| record with a spherical field lens (conic 0) | +2.6 / +1.5 / +0.3 / +2.6 | 0.59 / 1.01 | 0.996 / 0.992 | +1.5 mm: 0.9991 / 0.9942 | 0.54 -> 0.34 | 0.30 |
| true collimation (hyperbolic collimator, source at its focus), tuned tail | +2.7 / +3.9 / +2.6 / +5.1 | 0.80 / 1.35 | 0.982 / 0.965 | +3.9 mm: 0.9991 / 0.9963 | 1.12 -> 0.32 | 0.13 |
| true collimation + seed tail | -0.5 / -0.8 / -1.1 / -0.5 | 0.11 / 0.21 | 0.9991 / 0.9980 | -0.8 mm: 0.9999 / 0.9999 | 0.26 -> 0.08 | 0.035 |
| mirror rig, record (seed tail) | -0.4 / -0.6 / -1.3 / -0.3 | 0.09 / 0.16 | 0.9993 / 0.9974 | -0.6 mm: 0.9999 / 0.9994 | 0.23 -> 0.09 | 0.72 |

**Reading.**
- **The tail geometry is the whole story.**  The null tuner moved the field
  lens from 10.8 mm past the focus to 39.8 mm (its own focal length, a
  telecentric pupil image) and bent its conic to -2.59.  At that station the
  pupil rides 4.6 mm high on a 12 mm asphere, and the lens's zonal power
  (about half the bowl: the spherical-lens row) and its geometry (the rest)
  put the edge zones' images 3.4 mm past the center's, with 1 mm of
  astigmatic split.  The seed geometry, the pupil 1.2 mm high on the lens,
  images the DM flat to within a millimeter, with 0.08 mm of astigmatism and
  a distortion of 0.003 mm -- a hundred times smaller than the tuned tail's.
  The mirror rig, which never left the seed geometry, has the same pupil
  quality; its 0.72 mm distortion is the off-axis parabola pair's, a mapping
  the matrix absorbs.
- **The detector move is right but second-order once the tail is right:**
  the seed tail's compromise plane is 0.7 mm from the plane as built and
  buys 0.9992 -> 0.9999.  On the tuned tail it is the 4.3 mm move already
  recommended.
- **A field lens or flattener in front of the detector cannot fix the
  tuned tail's bowl.**  In image space that bowl is 3.4 mm deep over a
  4.9 mm radius, a surface of about 5 mm radius of curvature; a Petzval
  flattener for it would need a focal length of about -3.5 mm at n 1.5,
  which is not an optic.  The bowl is a product of the relay's geometry
  and is removed by the geometry, not compensated behind it.
- **True collimation is worth doing for its own sake, not for the pupil
  image.**  The collimator of record is fed 14 mm inside the hyperbola's
  focus (the exact conic for the flat-first plano-convex form is a
  Cartesian oval; the hyperbola leaves 6e-5 rad rms, ten times better
  than the record's 5.8e-4).  It halves the distortion on the tuned tail
  (0.27 -> 0.13 mm) and changes the gain little; with the seed tail it is
  0.999 either way.  Its real value: the physical-optics station-to-station
  chain needs it (section 7 above), and every tail re-tune should start
  from a collimated beam.
- **What the seed tail costs:** the flat-DM null, 9.1 nm against the tuned
  0.134 nm (REPORT_gauge_ifo).  That null is a fixed pattern the reference
  frame removes; the rows and the servo never saw it.  The tuner bought it
  with the pupil image.

**Recommendation.**  (1) Lens rig: return the field lens to the seed
station (10.8 mm past the focus) and hold it there; re-tune only the conic
and the detector trim, with the zone-image surface from the rays in the
objective (a knob for `tg96_tail`: the image position from a differential
trace, as `tg96_pupilsim` measures it), and set the detector at the image
surface's mean.  (2) Mirror rig: the 0.6 mm detector move, nothing else.
(3) Fix the collimator on the lens rig (`SRC_AT_FOCUS`, the hyperbolic
conic) before the next tail tune, as the mirror rig already has.  (4) No
flattener.  (5) Re-run the record's rows on the corrected tail and the
96 mm beam together.

## 8. Package A of the redo: the bench collimated for real (TO, 2026-09-17)

`BRIEF_to_tg_redo.md` package A, on the sheets carrying the decided substrates
(A0, resources `37193e5`: 10 mm splitter and compensator, 2 mm fused-silica
plates under every polarizing element, the 2 mm mask plate, 4 mm singlet edges,
protected aluminum on the parabolas).  New tool `tg96_collimate.m`
(`runs/coll_lens`); builder and runner changes in `twyman_green.m`,
`tg96_run.m`, `tg96_tail.m`; the numbers land in `tg96_params.m` and
`zwfs_params.m`.

### 8.1 What was actually wrong with the lens rig (and it was not the source alone)

The brief's item 1 asked for `SRC_AT_FOCUS` on the lens rig and a re-solve of
the two conics.  The first measurement said that is not enough, and said why:

**`L1_Kr` 236.866 is `(n-1) * 473.7`.**  `l2_trade` matched the collimator's
RADIUS to the conjugate the source really sat at -- `F1 - zsource` = 475 mm --
not to `F1` = 500.  Move the source out to `F1` and that lens has 5 % of surplus
focal length, which comes straight out as residual curvature.  Measured: a solve
free in `(SRC_TRIM, L1_Kc)` with `L1_Kr` held at the record's value runs the
conic to **-4.68**, walks the source back 27 mm (undoing `SRC_AT_FOCUS`
exactly), and still leaves **2.6e-4 rad rms** -- a fit fighting a geometry
error.  So the radius is solved with the conic, and the source stays AT the
conjugate, which is what `F1` in the sheet has to mean.

### 8.2 The solve, and the three gates

`tg96_collimate` runs three stages against the engine's own rays, over the rays
that **reach the DM** (the cone overfills it; the rays it throws away are the
outermost, where a singlet is worst -- tuning the collimator on light the bench
never uses is how you get a good number and a bad bench):

| bench | exit-ray spread after L1 | waves over the beam | focal spot at the seat | marker vs the ray focus |
|---|---|---|---|---|
| as the sheet described it (record figures, source 25 mm inside) | 1.27e-03 rad rms | 46.8 | 0.48 um rms | -10.4 mm |
| source at the conjugate, record figures | 9.7e-05 | 3.7 | -- | +13.7 mm |
| **re-solved** | **6.3e-09** | **0.0** | **0.17 um rms** | **0.000 mm** |

```
P.bench.SRC_AT_FOCUS = true;    P.bench.SRC_TRIM = 0;
P.bench.L1_Kr = 249.246312;     P.bench.L1_Kc = -0.583016;
P.bench.L2_Kr = -124.076;       P.bench.L2_Kc = -0.581843;
P.bench.MASK_TRIM = 1.231759;
```

GATES (the brief's): exit spread < 1e-4 rad rms **PASS** (6.3e-09); focal spot
< 1 um rms **PASS** (0.17); mask marker within 0.5 mm of the ray focus **PASS**
(0.000, by construction -- stage 3 solves it).

**The solved radius checks out in closed form, which is how we know it is not
a fit artifact.**  A plano singlet's front principal plane sits `t/n` behind its
flat face, so a lens that collimates a source `d` in front of its powered vertex
has focal length `f = d - t(1 - 1/n)`.  Here `t` = sag + edge margin = 7.745 mm
at the scale the runner builds (3.745 + 4.0), so `t(1 - 1/n)` = 2.58 mm:

| | focal length the radius implies | the conjugate it collimates | + the principal-plane term |
|---|---|---|---|
| record, `L1_Kr` 236.866 | 473.73 | 473.73 + 1.51 = **475.2** | = `F1 - zsource` = 475 |
| re-solved, `L1_Kr` 249.246312 | 498.49 | 498.49 + 1.51 = **500.0** | = `F1` |

(sheet units; the runner multiplies by `s` = 96/56.)  The two differ by exactly
`zsource`.  The record's lens was the right lens for the bench as the engine
actually built it -- which is why nothing ever looked wrong -- and the new one
is the right lens for the bench the sheet describes.

**And the residual is the right size for the right reason.**  6.3e-09 rad rms
over a 48 mm beam is 2.4e-4 waves -- which is not "zero because the fit ran out
of numbers".  The flat face refracts the diverging cone before the powered face
sees it, and its spherical aberration at this conjugate is a ~6 um longitudinal
shift = 4.6e-07 rad of angular spread; a conic has exactly the freedom to cancel
that fourth-order term, and what is left is the sixth-order one, smaller by
u^2 = (56.5/857)^2 = 4.4e-3 -> 2e-09 rad.  Measured 6.3e-09.  So the conic is
not APPROXIMATING the Cartesian oval here so much as cancelling the plate, and
the 1e-4 gate has three orders of magnitude of headroom.

**Two things worth keeping.**  (1) **The conic barely moved**: -0.583016 against
the record's -0.5829, four figures, while the radius moved 5 %.  A conic belongs
to the shape and the plano orientation, not to the conjugate; only the radius
was ever wrong.  (2) **The seat is diffraction-limited either way** -- 0.17 um
rms against a lambda F/D of 2.8 um -- so the 1 um gate is not what limits the
mask plane; the 10.4 mm seat error was, and the zwfs sheet has carried that same
error as a constant (`MASK_TRIM = -5.582`) since S1.  It is now re-scanned per
run there (`'scan'`), because the collimation fix moved it (-5.6 -> +1.2 on the
tg96 rig's own optics) and a carried constant seats the mask off focus and then
blames the glass.

**The beam is 58.3 mm at the DM** against its 48 mm aperture: the source moved
25 mm farther from L1, so the cone reaches 5 % wider than the 55.5 mm of the
morning's stop change.  **The DM is the stop** and 68 % of the ray grid gets
through -- the sampling price of making it so, and `P.clear.beam_r` is 59 now
(the clearance solve must see the beam that exists).  The DM leg and the
splitter angle do not move at that radius (450 mm, 22.5 deg).

**A builder trap closed on the way.**  `SRC_AT_FOCUS` was forwarded to the
builder from INSIDE `stage_B_`'s `if optics == 'oap'` block, so setting it on
the lens rig was accepted by the sheet and silently dropped -- the same trap the
zwfs sheet records for `MASK_SUB`.  It, `SRC_TRIM` and `MASK_TRIM` are now
forwarded on both rigs, by `tg96_run` and by `tg96_tail`.

### 8.3 The tail tuner now optimizes the READING (Dave's ruling)

`P.tail` in the sheet: `objective 'reading'`, `free {'FL_Kc','DET_TRIM'}`,
`reading_stage`.  The cost is what the pupil stage measures the camera
recovering off the DM -- stage 2's working-surface error, or (default) its
stage-1 band-edge-phase proxy, which is 4x cheaper and tracked it on every case
run on 2026-09-17.  The flat-DM null is computed every evaluation and PRINTED,
never optimized: a common misplacement of the detector cancels in an arm
difference, so the null is blind to exactly the failure that produced the
record's bowl.  `free` holds the field lens at the geometric seed station.
`tg96_pupilsim` gained `'stages'` and `'figs'` so it can be called per
evaluation; a failed evaluation now prints its error instead of returning a
silent 1e6 (an objective that cannot measure looks exactly like one that
measures a terrible tail).

**The seed tail on the collimated bench, before any tuning** (`lens96`
evaluation 1): band-edge phase **0.0122 rad rms / 0.0237 max** (gate < 0.06
max), phase gain **0.9997 worst** (gate >= 0.998), image surface flat to
0.7 mm over the pupil (on axis +0.28, mean -0.07), astigmatic split 0.15 mm
rms, distortion **0.023 mm rms**, flat-DM null 59.3 nm (reported).  Two of the
three package-A gates are met by the geometry alone; the distortion is 2x the
brief's 0.01 mm figure -- that figure came from the seed-tail row measured on
the UNcollimated bench, where the same table already showed true collimation
RAISING the distortion (0.003 -> 0.035 on the tuned tail's row), and it is a
mapping the registration affine absorbs.

### 8.4 The lens rig's tail, tuned on the reading (`lens96`)

109 evaluations, `FL_Kc` and `DET_TRIM` free, `FL_F` and `D_MASK_FL` held at the
geometric seed -- so the field lens stayed 10.8 mm past the focus, which is the
station that images the DM flat.

| | seed | winner |
|---|---|---|
| `FL_F` / `D_MASK_FL` (held) | 42.8932 / 10.7614 | same |
| `FL_Kc` | -2.11278 | **-7.77234** |
| `DET_TRIM` | 1.8606 | **1.9951** |
| band-edge phase (the objective) | 0.0122 rad rms / 0.0237 max | **0.0000 / 0.0001** |
| zone image vs the detector | +0.28 on axis, 0.7 mm of departure over the pupil | **flat to 0.002 mm** |
| astigmatic split of the zone images | 0.146 mm rms / 0.256 max | **0.056 / 0.102** |
| Nyquist phase gain, worst over the pupil | 0.9997 | **1.0000** |
| distortion vs one global affine | 0.023 mm rms | 0.041 mm rms |
| flat-DM null (reported, never optimized) | 59.3168 nm | 59.3168 nm |

**The winner gate KEPT it, and says the two tails read the same.**
`winner/seed = 0.9970 >= 0.90` -- one multi-site actuator row through the ray
affine, in actuator space, winner 1.0913 against the seed's 1.0945 (5 sites,
magnification 10.28 DM-mm per detector-mm, err 691 pm on a floor of 632, SNR
173).  So the honest reading of this tune is: **at the seed station the tail was
already as good as it gets for the READING, and what the tune bought is the
pupil IMAGE.**  0.9997 -> 1.0000 of Nyquist gain is the third decimal, and the
row gate cannot separate the two at all; the image surface going from 0.7 mm of
departure to 0.002 mm is a real change and it is what Fang Shi's question, the
camera's focus tolerance and the 385-px sampling budget are about.

**CORRECTED in 8.8 -- the row gate is not sensitive enough to settle this, and
the full measure says the tune DOES buy reading.**  What stands from the gate is
narrower than it first looked: the winner gate cannot SEPARATE the two tails.
Its estimator runs at SNR ~170 on a 5-site row, so a 0.3 % difference is under
its floor, and "cannot separate" is not "no difference".  Measured on ONE bench
with the full stage-2 measure (8.8): the tuned tail reads the 30 nm working
surface to **42 pm against the seed tail's 97 pm**, a factor of 2.3.  The trade
put to Dave below is therefore a real one in both directions, not a free
choice.

**The field lens's conic is the pupil-image field-curvature knob, and the null
objective had no way to see it.**  The field lens sits 10.8 mm past the focus,
so each DM zone lands at its own radius on it -- up to 1.2 mm, which is what
makes the pupil 1.2 mm high there -- and an `r^4` term at that station is
exactly a field-curvature term for the pupil image.  The tune used it: defocus
AND astigmatism improve together, which is the tell that the proxy is not being
gamed (a proxy that only saw defocus could be satisfied by trading it for
astigmatism, and the astigmatic split would have risen).

**What it costs: distortion, and only that.**  0.023 -> 0.041 mm rms, and this
is the residual AFTER one global affine, i.e. what the registration cannot
absorb.  At the DM scale that is 4 % of an actuator pitch; the mirror rig of
record carries 0.72 mm there -- 72 % of a pitch -- and still reads 0.997.  It
does not meet the brief's `< 0.01 mm` figure, and that figure should be
retired: it was read off the seed-tail row of the table in 7.1, measured on the
UNCOLLIMATED bench, where the same table already showed true collimation RAISING
distortion (0.003 -> 0.035 mm on the tuned tail's row).  Collimating the bench
changes the mapping; it does not degrade the reading.

**Buildability is a question about the used aperture, not about the conic.**
K = -7.77 on a 12 mm blank of R = 21.45 mm is 128 um of departure from the
sphere at its edge -- a real asphere -- but the beam only uses the central
1.2 mm radius, where the departure is **0.20 um** (0.15 um of it new, i.e. 0.24
waves against the seed conic).  Holding the pupil image flat to the measured
0.002 mm needs that `r^4` term to ~10-20 %, i.e. 20-40 nm of figure over
1.2 mm.  So the part to specify is a mild asphere over a 2.4 mm clear aperture;
the sensible build is a smaller blank, and the 12 mm one is 10x oversized for
this beam whatever the figure.

**The null did not move, at all.**  59.3168 nm, four decimals, on every one of
the 109 evaluations.  With the station held, both free knobs are blind to an arm
DIFFERENCE: `DET_TRIM` moves a detector both arms share, and the field lens's
conic acts on a 1.2 mm-high pupil in a leg both arms share.  The null's large
excursions in the record (9.1 -> 0.134 nm) came from `D_MASK_FL` and `FL_F`,
which is precisely how the record's tuner walked the field lens off its station.
This is the ruling's own evidence: on this bench the null is not a weak
objective, it is not an objective at all.

**Open, and measured next:** 59.3168 nm is 6.5x the record's 9.1 nm seed null,
and the four-step's unambiguous range is lambda/4 = 158 nm.  A fixed pattern of
that size is removed by the reference frame and never reaches a differential
reading, but it is close enough to the fold to be worth attributing before
package C spends a day on rows: `runs/nullab.sh` switches the collimation and
the substrates off one at a time against the record's 9.1 nm.

### 8.5 The mirror rig: the seat belongs to the focuser, and the source leg's plate defocuses the parabola

Two things, one of them a defect in the first version of this work and one a
real consequence of the decided substrates.  Both were found the same way: the
mirror rig's `DET_TRIM` tune was moving the detector the WRONG WAY against the
brief's own prediction (-0.6 mm), and a prediction that disagrees with a
measurement is worth a look before it is worth a shrug.

**`MASK_TRIM` is a property of the FOCUSER, not of the bench.**  The first
version of the sheet carried one global field, so the lens rig's +1.2318 mm
solved seat -- which is its plano singlet's principal plane, the mask's own
plate and what is left of its spherical aberration -- was being applied to the
mirror rig, whose parabola has none of that.  0.69 mm of it is somebody else's
optics, and that is OUTSIDE package A's own 0.5 mm seat gate.  Fixed: anything
in `P.oap.*` with the name of a `P.bench.*` knob overrides it when
`optics == 'oap'`, in `tg96_run`'s `stage_B_` and in `tg96_tail` alike.  No deck
of record was emitted with the wrong value -- the chain was stopped at the tail
tune.

**The input polarizer's substrate defocuses the collimator.**  A parabola fed
at its focus is exact, which is why `SRC_TRIM` was 0 for this rig and why
nothing was expected here.  But `POL_IN 'source'` -- the reflective rig's own
arrangement, since an OAP collimator's conjugate leg comes back along the
collimated axis -- puts the polarizer's 2 mm plate in the DIVERGING leg, and a
plane-parallel plate displaces the apparent source by `t(1-1/n)` = 0.63 mm ALONG
the light.  The parabola is then fed that far inside its focus.  Measured:

| mirror rig | exit-ray spread after the collimator | waves over the beam | seat vs the ray focus |
|---|---|---|---|
| substrates in, `SRC_TRIM` 0 (what the sheet said this morning) | 1.710e-05 rad rms | 0.70 | +0.539 mm |
| **`SRC_TRIM` -0.370629 solved** | **1.358e-08** | **0.00** | **-0.0001** |

It passes the 1e-4 gate either way; it is removed because it is free to remove
and it was the largest thing left in that space.  `tg96_collimate` now solves
the source station for the mirror rig (one dimension: a parabola has no conic
to trade against it).

**And the seat then lands exactly where physics says it should.**  With the
source right, the mirror rig's remaining seat error is the mask plate and
nothing else: solved **0.631902 mm** against `t(1-1/n)` = 2(1-1/1.4585) =
**0.6285 mm**, three figures.  Before the source station was solved it read
0.5393 -- the 0.7 waves of defocus were being paid for at the seat, which is
exactly the kind of compensation that makes a bench look fine and read wrong.

**The mirror rig's tail, on the corrected bench (`oap96`, `DET_TRIM` alone).**
24 evaluations: `DET_TRIM` 1.8606 -> **1.9755**, band-edge phase 0.0143 -> 0.0130
rad rms, everything else held.  Winner gate `winner/seed = 0.9971` (winner
0.9935, seed 0.9964) -- KEPT, and again the row gate cannot tell the two apart.

Two things read oddly on this rig and both have the same cause.  **The
prediction that started this section does not transfer:** the brief asked for
`DET_TRIM` -0.6 mm, the image-surface mean measured on the OLD mirror bench
(the 82 mm source cone, no substrates, the seat at 0); the tune finds **+0.11**,
because the new bench moved the mask +0.63 mm and five 2 mm plates shifted the
stations behind it.  And **the proxy's rms improves while its max degrades**
(0.0290 -> 0.0350 rad, both far inside the 0.06 gate).  That is what a
single-knob tune looks like on a TILTED image surface: this rig's pupil surface
carries 0.45 mm of tilt from the OAP pair, and a detector shift can only trade
one edge of the pupil against the other.  The lens rig had a BOWL, which the
field lens's conic can flatten; a tilt is not in the tail's reach at all.

### 8.6 Two things the emitted decks said that were not on the list

**(a) The clearance table's negative rows are bookkeeping, and the fix is in
the grouping.**  `redo_lens`'s physical-parts check reports five rows at -94 to
-102 mm (`QWPref`, `Sub2f`, `Sub2b`, `Sub3f`, `Sub3b`), every one of them a
SUBSTRATE FACE scored against the beam that goes through its own element.
`dmg_bench_clearance` groups a part's records by name stem so a plate is never
tested against its own beam, but the builder names substrate faces neutrally
(`Sub<k>f` / `Sub<k>b`) ON PURPOSE -- every arm descriptor in this lane picks
the wave plates out with `contains(name,'QWP')`, and a face called
`QWPtestInf` would be handed to `macos.waveplate` as a plate -- so the stem
rule cannot see whose substrate they are.  The two PASSES of one plate get
different numbers besides (`Sub2*` outbound, `Sub3*` back through the same
glass).  The deck ORDER knows: an `f` face is followed by its element, a `b`
face preceded by it, and inheriting the element's stem also inherits its
`In`/`Out` pass grouping.  Fixed there, with the segment endpoints read
through the same map; decks with no such faces are untouched by construction.

**The compensator's +10.4 mm is NOT bookkeeping, and it is not the plates
either -- it is the stop ruling.**  A builder plate carries no aperture, so its
scored radius is the beam plus 5 mm, and widening the beam costs margin twice:

| | beam radius | part r | separation | margin (spec >= 25) |
|---|---|---|---|---|
| record (`lensuw2`, `oapifo2`, `lens22h`) | 51.4 | 56.4 | 141.4 | +25.6 |
| redo (the DM as the stop) | 59.0 | 64.0 | 141.4 | **+10.4** |

The compensator was clearing by 0.6 mm of margin before the beam was opened.
Making the DM the stop took 15.2 mm and put it under.  One number fixes it --
`D_BS_CMP` 200 -> ~225 mm physical gives +28.1 -- but that is a layout decision
with a parts-list consequence and it is Dave's, not this package's.  Both rigs
report it (1 of 6 node parts on the lens rig, 1 of 5 on the mirror rig).

**(b) The lens rig's station residual is a WRAP, and the brief's hypothesis for
it is dead.**  `redo_lens` reads **51443 pm** on the 30 nm working surface
against `redo_oap`'s **463 pm** (the record: 62 nm and 626 pm).  The brief
expected the pupil-image bowl to explain it and the seed tail to close it.  The
bowl is GONE -- the zone image surface is flat to 0.002 mm and the distortion
is 0.041 mm against the mirror rig's 0.72 -- and the residual barely moved, so
that hypothesis is refused: the rig with the better pupil image has the worse
residual by 100x.

What the figure shows instead (`redo_lens_stations.png`, bottom right panel):
dense salt-and-pepper at +-2.5e5 pm on a panel whose own scale is +-250 nm,
i.e. **isolated pixels thrown by about lambda/2 = 316 nm**.  The mirror rig's
same panel is 0-3000 pm and sparse.  Both rigs' RECOVERED surfaces look like
the mirror command they are reading, so the gauge is not broken; what differs
is where the recovered map and the engine's own field FOLD, and they do not
fold at the same pixels.  Both quantities in that comparison are wrapped at
lambda/4 = 158 nm by construction (`ht = angle(exp(i*(angle(Et)-angle(E0))))`),
and this rig carries a 59 nm fixed pattern plus a 30 nm surface underneath it.
Package C item: count the pixels whose residual exceeds lambda/4 and see
whether removing them collapses the rms -- that separates "a few folded pixels"
from a broad error, and `battery.unwrap` already exists for the first.  A
candidate cause specific to this rig is in section 7's own measurement:
amplitude cross-talk up to a third of the phase modulation at the lens rig's
edge, which the four-step ignores.

### 8.7 Package A's gate record: the pupil stage on the EMITTED decks

`tg96_pupil_batch` on `redo_lens_test.in` and `redo_oap_test.in`, at the pupil
stage's own 129 rays across -- independent of the 65 the tunes ran at, so the
winner is certified at a sampling it was not fitted at.  Tags
`pupilsim_redo_lens`, `pupilsim_redo_oap`.

| | lens rig | mirror rig | record (lens / mirror, REPORT 7 + 7.1) |
|---|---|---|---|
| Nyquist phase gain, worst in the lit pupil | **1.0000** | **0.9994** | 0.954 / 0.997 |
| gain by radius, 10 bins | 1.000 throughout | 1.000 throughout | fell to 0.95 at the edge |
| band-edge quadratic phase | **0.000 rad rms / 0.000 max** | **0.013 / 0.035** | -- |
| amplitude cross-talk per unit phase, max | **0.005** | 0.036 | up to 0.33 at the lens edge |
| zone image surface vs the detector | **flat to 0.001 mm** | +0.22 on axis, -0.67..+0.32 (tilt 0.43, defocus -0.45) | +2.6..+6.0 (lens) |
| astigmatic split of the zone images | 0.055 mm rms | 0.092 | 1.05 (lens) / 0.09 (mirror) |
| single pokes, peak / width / centroid shift | 0.9998 / 0.999 / <= 0.4 um | -- | 0.99 (lens) |
| **30 nm working surface: recovered - true** | **0.042 nm rms** (0.13 % of it) | **0.092 nm** (0.29 %) | **1.22 nm** / 0.23 nm |
| pupil distortion vs one global affine | 0.041 mm rms | 0.722 | 0.27 / 0.72 |

**GATES (the brief's package A item 2, both rigs): Nyquist gain >= 0.998 worst
PASS; band-edge phase < 0.06 rad max PASS; distortion < 0.01 mm rms FAIL on
both, and that figure is withdrawn** -- see 8.4: it was read off a row measured
on the uncollimated bench, the mirror rig has carried 0.72 mm through every
record it ever set while reading 0.997, and the number is a residual AFTER the
registration affine.  The flat-DM null is reported, not gated: 59.3 nm (lens),
73.9 nm (mirror).

**The headline is the working surface.**  The bench reads its own 30 nm surface
to **42 pm** on the lens rig, against 1.22 nm for the record's tuned tail on the
record's bench -- a factor of 29 -- and to 92 pm on the mirror rig against
0.23 nm, a factor of 2.5.  The error is flat across the spatial bands until the
very top of the actuator band (0.0003 of the surface below 0.25 cyc/mm, 0.0046
at the Nyquist itself).

**And the lens rig's amplitude cross-talk is gone.**  0.005 per unit phase,
where section 7 measured up to a third of the phase modulation at that rig's
edge and noted that the four-step ignores it.  That was the leading candidate
for the station figure's folded pixels (8.6b); it is no longer present on this
tail, so that explanation weakens and the fold stands on its own.

**What the mirror rig cannot do, and why it is fine.**  Its zone image surface
keeps 0.43 mm of TILT and 0.45 mm of defocus -- the off-axis parabola pair's,
not the tail's -- so a detector shift cannot flatten it, only pivot about it
(0.013 rad rms of band-edge phase against the lens rig's 0.000).  It still
reads 0.9994 worst, because 0.035 rad of quadratic phase costs
`1 - cos(0.035)` = 6e-4 of gain.  The distortion is the same 0.72 mm mapping it
has always had, and the registration affine has always absorbed it.


### 8.8 What the tune bought, on ONE bench -- and a check that the gate does not depend on the tool's own cone

Two follow-on runs (`runs/redoseq2.sh`), both at the pupil stage's 129 rays:

| lens rig, collimated bench, substrates in | seed tail | tuned tail (`lens96`) |
|---|---|---|
| `FL_Kc` | -2.11278 | -7.77234 |
| band-edge phase | 0.012 rad rms / 0.024 max | **0.000 / 0.000** |
| **30 nm working surface: recovered - true** | 0.097 nm rms | **0.042 nm** |
| ... at the lowest band (0-0.062 cyc/mm) | 0.0075 of the surface | **0.0003** |
| pupil distortion vs one affine | 0.023 mm rms | 0.041 mm |
| compromise detector plane | -0.07 mm | -0.00 mm |

**So the tune is worth 2.3x in the reading**, and most of that is at the LOW
spatial frequencies (25x at the bottom band) -- which is what a residual field
curvature does to a pupil relay, and exactly what the field lens's conic was
free to remove.  This is the comparison the winner gate could not make: its
5-site row estimator runs at SNR ~170 and reported the two tails at a ratio of
0.9970, i.e. under its own floor.  **A gate that cannot separate two candidates
has not said they are the same**, and it is worth saying that plainly because
the first reading of this section said it had.

The trade is therefore: a field lens conic of -7.77 instead of -2.11 -- 0.20 um
of departure from the sphere over the 1.2 mm the beam actually uses, 0.15 um of
it new -- buys a factor of 2.3 in how well the bench reads its own DM, and costs
1.8x in a distortion residual that is 4 % of an actuator pitch.  Dave's call;
`bench.tail_from_mat false` selects the seed if the simpler part wins.

**And the gate does not depend on the pupil stage's own cone.**  The stage of
record re-aims the source cone so the beam is 1.06x the DM aperture, while the
emitted bench's cone is 1.21x; run instead on the deck's OWN cone (`overfill 0,
dm_ap 0`) the numbers are 0.041 nm against 0.042, distortion 0.041 against
0.041, band-edge phase 0.000 against 0.000.  The DM is the stop either way, so
the illuminated pupil is the same 48 mm and only the marginal vignetting
differs -- which is what the tool's cone knob was always assumed to mean, now
measured (`pupilsim_redo_lens_owncone`).

### 8.9 Where the 59 nm flat-DM null comes from: the BEAM, not the substrates and not the tail

`runs/nullab.sh`, five benches, each number the tuner's own SEED evaluation at
its reduced resolution (model 512, 193 rays), the field lens at the geometric
seed station throughout:

| bench | beam at the DM | substrates | collimated | flat-DM null |
|---|---|---|---|---|
| the record's, as it was | 77 mm | no | no | **9.78 nm** |
| + the DM made the stop | 96 | no | no | **53.60** |
| + the decided substrates | 96 | yes | no | **92.40** |
| **+ collimated for real (the redo bench)** | 96 | yes | yes | **59.32** |
| collimated, ideal elements | 96 | no | yes | 53.74 |

(9.78 reproduces the 9.1 nm of record to within the knobs that have moved since.)

**The dominant term is the beam, and it is Dave's stop ruling, not this
package.**  Lighting the whole 96 mm DM instead of 77 mm of it takes the null
from 9.8 to 53.6 nm.  The flat-DM null IS the two arms' aberration difference
over the illuminated aperture, and that difference grows fast with aperture:
the outer 10 mm of the DM's radius is where a same-plane fold pair differs
most.  Nothing in the tail, the substrates or the collimation is responsible
for 44 of the 59 nm.

**The substrates cost 39 nm on the misfed bench and 5.6 nm on the collimated
one -- seven times less.**  That is the cleanest statement of what collimating
the rig actually bought: a plane-parallel plate in a COLLIMATED beam is pure
optical path and aberrates nothing; in a converging or diverging one it adds
spherical aberration and astigmatism in proportion to the convergence.  The
record's "collimated" space carried 47 waves of curvature, so its plates were
never in a collimated beam.  The 5.6 nm that remains is the plates that are
genuinely not in collimated space -- the mask's own plate in the converging
beam, and the output plates after L2.

**And collimation ALONE does nothing to the null** (53.60 -> 53.74 with ideal
elements): with no glass to aberrate, the arm difference does not care whether
the space is collimated.  The two changes only interact.

**What this means for package C.**  59 nm rms is a fixed pattern, removed by
the reference frame, and no differential reading sees it -- the gate record of
8.7 was measured with it in place and reads the 30 nm surface to 42 pm.  But
the raw four-step is unambiguous only over +-lambda/4 = +-158 nm, so it is
worth knowing before a day of rows: the question is not the null's rms but its
PV against that fold, and 8.6b already shows the lens rig folding pixels in the
station figure.  If a row table comes back with salt-and-pepper, `battery.unwrap`
is the first thing to turn on, not the last.

## 9. Package B: the physical-optics chain, leg by leg on the improved bench (TO, 2026-09-17)

`tg96_pupil_s2s` gained a `checks` block (default on) that measures each leg in
the order light meets it, BEFORE the convention sweep -- the sweep reports the
end of the chain, where every leg's error is already mixed into one gain map.
Run on the emitted `redo_lens` deck through its own pupil-stage run
(`s2s_redo_lens`).  **The chain now runs on the lens rig at all**, which it
could not before (section 7: 41 waves of curvature walked the rays off the grid
between the near-field legs).

| check | on the improved bench | on the bench of record | gate |
|---|---|---|---|
| 1a the flat pupil's phase at the entrance sphere | **0.2824 wave rms** | 1.5 waves | < 0.02 |
| 1b a Nyquist sinusoid's height there | **0.8232**, at 0.5096 cyc/mm vs the DM's 0.5000 | -- | 1.00 +- 0.01 |
| 2a the focal field's first zero | 1.5 um vs the Airy 3.4 | a 34 um blob (mirror rig) | +-10 % |
| 2b the pupil radius at S2, field vs RAYS | 1.000 vs 1.089 mm (**-8.1 %**) | -20 % | 2 % |
| 2c the PITCH at S2, engine's label vs the scaled frame | 0.006016 vs 0.006398 (**-6.0 %**) | -- | 2 % |

**2c is the brief's suspect #1 confirmed and sized.**  The engine sets `dxElt`
from the propagator only for PropType 3/10/15 and the regridded 5/14 pair;
PropType 11 -- `PL2SPH`, the F -> S2 leg -- is in neither list, so S2's pitch
falls through to the ray-spacing branch (`utilsub.F` ~:590), while `PL2SPH`'s
plain inverse FFT hands the array back in the ENTRANCE sphere's pitch, i.e. in
the Sziklas-Siegman scaled frame.  The two agree only if the ray spacing at S2
really is `dx(S1) * R2/R1`.  It is 6 % short.  **And 2b follows from 2c**:
rescale the field's disc by the pitch ratio and 1.000 mm becomes 1.063 against
the rays' 1.089, so most of the radius error IS the label.

**2a is a sampling artifact, not a blob, and the check needs rewriting.**  The
focal plane's pitch is 1.98 um (`lam*R1/(N*dx1)`, N the model size) against an
Airy radius of 3.36 um, so the first zero lands inside the first ring and a
zero-crossing test cannot resolve it.  The old bench's 34 um blob is gone; this
is the opposite problem.  The gate should be encircled energy against the Airy
prediction, not a zero crossing.

**1b's frequency shift is unexplained and is the sharpest clue left.**  The mode
returns 1.9 % higher in frequency -- the pupil SCALE at the entrance sphere is
not the DM's -- and at 0.8232 of its height.  A weak phase grating propagating
`z` loses `cos(pi lam z f^2)`; at the Nyquist 0.5 cyc/mm and the legs' 585 mm
that is 0.958, and 0.8232 implies an effective 1216 mm, about twice the legs.
Both numbers point at the same thing and neither is explained yet.

### 9.1 The quartet was built around a bench that had no glass before the focus

The five checks above were all measured on the chain AS IT WAS.  Running them
one leg at a time turned three of the five failures into one construction bug
and one engine constraint, and the fix needed no fudge factor.  The progression,
each row a run in `runs/`:

| deck construction | 1a phase at S1 | 2b field vs rays at S2 | 2c pitch | 2d the RAY BUNDLE vs R2/R1 |
|---|---|---|---|---|
| as built (`s2s_redo_lens`) | 0.2824 **wrapped** | -8.1 % | -6.0 % | -6.2 % |
| + entrance sphere centred on the PRE-plate focus (`s2s_r1trim`) | **0.0085** | **-1.8 %** | -6.1 % | -6.2 % |
| + F and S2 moved to the mask's station, plate INSIDE the sandwich (`s2s_order`) | 0.0085 | field 0.000 -- **empty** | **-0.0 %** | **-0.0 %** |
| **plate relocated into the gap after the focuser** (`s2s_glass2`) | **0.0070** | **-1.9 %** | **-0.0 %** | **-0.0 %** |
| (control: no mask plate at all, `s2s_nomask`) | 0.0089 | -1.8 % | -0.0 % | -- |

**The bug: the builder appended all three quartet elements straight after the
focuser, so anything the deck puts between the focuser and the focus lands
AFTER S2 in deck order -- and MACOS traces in deck order.**  The rays ran S1 ->
F -> S2 and then turned back 7 mm upstream into the mask plate's glass.  At S2
they were therefore still heading for the PRE-plate focus while F had been
placed at the POST-plate one: 6.2 % on the bundle, measured ray to ray with no
estimator in the comparison (CHECK 2d), and the same 6.1 % on the pitch.  **The
record's decks had no mask plate** -- nothing at all between the focuser and the
focus -- so this could not have shown up before the substrates were decided.

**The engine constraint, learned by violating it:** the NF1 sphere must be
IMMEDIATELY followed by the NF2 plane.  `SPH2PL` fires at the element after the
NF1 element and reads its `zEnd` from the one after THAT (`propsub.F` ~:1958,
`zEnd = zElt(iElt+1)`); with substrate faces in between, `zEnd` is a face's
`zElt`, which the builder sets to 0, and the chirp `S ~ (Z2-Z1) Z1/Z2` divides
by it.  The field comes back empty -- measured, CHECK 2b radius 0.000, pitch
0.00 um -- which is a clean failure but not an obvious one from the outside.

**So the plate is relocated into the gap right after the focuser, keeping its
own 2.000 mm of thickness, and the entrance sphere goes just after it.**  That
is exact to first order, for a reason worth keeping: a plane-parallel plate's
only effect on a converging beam is a focus shift of `t(1-1/n)`, and that does
not depend on WHERE in the cone the plate sits.  The plate is then in the
geometric leg that seeds the sphere, so the wavefront at S1 already heads for
the true focus and **no `r1_trim` is needed** -- that knob was the right
compensation for the wrong placement, and it is left in the tool only as a
diagnostic.  (Two failed placements are recorded above because each one taught
something: 2-4 mm ahead of an S1 that sits 0.5 mm past the focuser puts the
glass BEHIND the focuser's exit face, and the engine answers that with an empty
grid, `dxElt` 1e10.)

**Where package B stands.**  Four of the five legs are now right on the bench as
it will be built: the collimated legs deliver a flat pupil to the entrance
sphere (0.0070 wave rms), the through-focus quartet scales the bundle exactly as
R2/R1 (-0.0 %), and the engine's pitch label agrees with the scaled frame to
four figures.  Two remain:

- **1b, the only substantive one:** a Nyquist sinusoid reaches S1 at 0.8324 of
  its height.  A weak phase grating over the legs' 585 mm should lose
  `cos(pi lam z f^2)` = 0.958.  Before reading anything physical into the
  difference, the estimator needs its own control -- it reads the amplitude off
  a single Fourier bin of a hard-masked pupil, and leakage from that mask alone
  could account for it.  The control is to run the identical estimator on a
  SYNTHETIC mode of the same frequency over the same mask and divide.
- **2a is a limitation of the check, not of the chain:** the focal plane's pitch
  is 1.98 um against an Airy radius of 3.3, so a first-zero test cannot resolve
  what it is asked to find.  It should be encircled energy against the Airy
  prediction.

### 9.2 CHECK 1b closed: the Nyquist loss is INTERPOLATION, not propagation

The estimator's own control reads **1.0000** on a synthetic mode of the same
frequency over the same mask, so the 17 % is real and not spectral leakage.  It
is also not Fresnel.  Sweeping three frequencies and asking what `z` would
explain each loss as Talbot (`cos(pi lam z f^2)`):

| f (cyc/mm) | height at S1, DM grid 0.40 mm | ... 0.28 mm | z if Talbot (0.40) |
|---|---|---|---|
| 0.125 | 0.9926 | 0.9968 | 3731 mm |
| 0.250 | 0.9634 | 0.9798 | 2083 |
| 0.500 | 0.8324 | 0.8915 | 1128 |

**The fitted z is not constant -- it runs as 1/f -- so the loss is not a
propagation at all**, and the "about twice the legs" arithmetic that 585 mm
invited was reading meaning into a number that had none.  The loss goes as
`f^2`, which is a BLUR, and it has a name: **linear interpolation, whose
transfer is `sinc^2(f dx)`.**  There are two of them in series, and their
product accounts for the measurement:

| | f dx | sinc^2 | |
|---|---|---|---|
| the DM's GridData surface onto the rays (`dx_g` 0.40) | 0.205 | 0.8751 | |
| the rays onto the diffraction grid (`dx` 0.2595) | 0.133 | 0.9455 | |
| **product** | | **0.8274** | measured **0.8324** |
| the same with `dx_g` 0.28 | | **0.9111** | measured **0.8915** |

Within 0.6 % and 2.2 %.  **So the Nyquist loss is the DM SURFACE MODEL's own
sampling, not the near-field legs** -- it is paid before any propagation
happens, the ray path pays exactly the same toll, and `tg96_pupilsim` never saw
it because its stage 2 works on the DM's own grid.  Two consequences:

- **The gate is wrong as written.**  It compares the field at S1 against the
  COMMANDED amplitude; the honest reference is what the rays actually see at
  the DM.  Rewritten that way it would pass, and it would still catch a real
  leg error.
- **The number itself is worth carrying into package C**: at the sheet's
  `dx_g` 0.28 the model loses ~11 % of a Nyquist actuator mode to
  interpolation before optics are involved at all.  That is a floor under
  every gain the gauge reports at the top of the actuator band, and it is
  the reason to prefer a finer DM grid over a finer ray grid when either
  would fit.

**CHECK 2a does NOT improve with sampling and is therefore a real excess.**
Halving the focal pitch (model 1024, 0.99 um) leaves the 83.8 % encircled-energy
radius at **3.96 um against the Airy 3.34** -- the same number, so it is not
quantisation.  With the pupil's measured radius (46.81 mm, not 48) the Airy is
3.42 and the excess is 16 %.  The ray spot at that plane is 0.17 um rms, so the
WAVEFRONT is not the cause; the likely cause is the seeded pupil's AMPLITUDE not
being uniform (it carries the ray-density and interpolation structure above).
Open, and it wants the pupil's amplitude profile plotted against a top hat
before anything else is tried.

### 9.3 CHECK 2a closed: the focal field IS the pupil's Airy pattern; the excess was the statistic

`tg96_pupil_amp` (new) measures the seeded pupil's AMPLITUDE at the entrance
sphere and asks what it alone does to the focal spot: the measured amplitude
with a FLAT phase, a top hat of the same 50 % radius through the identical
transform as the control, and the engine's own focal field beside both.

**The pupil is a near-perfect hat**: ripple **0.0003 rms** of the mean inside
0.9 of the radius, edge 90->10 % over **0.983 mm** (3.8 pixels) on a 94 mm
pupil, and **0.23 %** of the energy outside the 50 % radius.

**And all three focal profiles overlay across five decades** -- same rings, same
nulls, first null at ~3.4 um against the Airy prediction of 3.41
(`pupilamp_lens_amp.png`).  So the quartet delivers the pupil's Airy pattern and
there is no excess to explain.

**The "16 % excess" was the encircled-energy statistic.**  An 83.8 % EE radius
is normalised by the energy on the WHOLE grid, so it moves with whatever sits in
the far wings: the same three fields, indistinguishable in profile, give
**2.97 / 3.57 / 3.96 um**.  That also explains why halving the focal pitch left
3.96 alone -- quantisation was never the issue.  CHECK 2a now reports the first
NULL POSITION, found in a bracket around the prediction, with the EE radius
still printed and labelled.

**One more measurement artefact caught on the way**, and it is the same
species: the radial profile binned at `dx/2` on a grid of pitch `dx` leaves
rings near the centre with no pixels in them, and a bin left at ZERO reads as a
perfect null -- the bracketed search duly took one at 1.73 um on a field whose
real null is at 3.4.  Empty bins are NaN now.  **A hole is not a zero.**

### 9.4 Package B, where it stands

| leg | verdict |
|---|---|
| 1a the collimated legs deliver a flat pupil to the entrance sphere | **PASS**, 0.0070 wave rms |
| 1b a Nyquist mode's height there | **closed** -- the 11-17 % is the DM grid's own interpolation (9.2), not the legs; the gate wants re-basing on what the RAYS see at the DM |
| 2a the focal field against the Airy | **PASS** -- first null **3.46 um** against the Airy 3.34 (`s2s_final3`), and the profiles overlay across five decades (9.3) |
| 2b the pupil radius at S2, field vs rays | **PASS**, -1.9 % |
| 2c the engine's pitch label vs the scaled frame | **PASS**, -0.0 % |
| 2d the ray bundle itself vs R2/R1 | **PASS**, -0.0 % |

**What it cost to get there, and the pattern worth carrying:** of the five
checks, THREE were measuring their own limits rather than the chain -- 1a's
statistic saturating at the wrapped-phase ceiling, 1b's suspected leakage (which
the control cleared, turning it into a real finding about the DM grid), and 2a's
EE normalisation.  Only 2c and 2d measured the chain, and those two found the
real defect: the quartet built ahead of the mask plate.  **On a chain this new,
a failing gate says nothing until it is shown it could have passed.**

**The check suite as it now stands** (`s2s_final3`, model 1024, the sheet's DM
grid): 1a PASS, 1b FAIL-by-design (it compares against the COMMANDED amplitude;
re-based on what the rays see at the DM it passes, and the number it is really
reporting belongs in 9.2), 2a PASS, 2b PASS, 2c PASS, 2d PASS.

**Still open, and both are named rather than guessed:** the convention sweep at
the exit step (the brief's item 3) has not been re-run since the deck-order fix,
so the four `zElt` candidates have not been retested on a correct quartet; and
the readout comparison against `tg96_pupilsim` (item 4) waits on that.  The
mirror rig (item 5) has not been run through the chain at all.
