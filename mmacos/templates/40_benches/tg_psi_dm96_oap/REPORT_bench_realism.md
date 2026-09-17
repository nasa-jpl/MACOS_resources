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
