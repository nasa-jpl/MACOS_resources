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
| 8 — the interferometer's station-by-station figure | **runner in** (§6); produced by the `figs` stage of item 2's runs |

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
