# dyson5 beat 5b -- the telescope through the offset_imager ladder (record so far)

Written for: Dave (review).  Addenda 19-23 of `BRIEF_to_dyson5.md`.  Runner
stages `t3` (the template's ladder per case) and `t3s` (the first-order
screen); knobs `tel3_*` / `tel3s_*` in `dyson5_params.m`.  Records:
`dyson5_t3.txt` / `.mat` (the counted cases), `dyson5_t3s.txt` / `.mat` /
`.csv` (the screen), `dyson5_t3w.txt` / `.mat` (the y2 continuation,
addendum 23), per-case runs in `t3/` (`<tag>_t3_t<mm>_off<deg>[_y<y2>]
_{REPORT,STORY}.md`, decks, figures; run logs are not committed).

## 1. Step 1 (addendum 19): beat 5's envelope at 4-10 deg -- the envelope, not the offset

`oi_story` at OFF 4 / 6 / 8 / 10 deg, box 24.56 x 0.3 deg, EPD 70 mm, F/1.8,
seed = beat 5's T0 first order (`telescope_seed`, t1 140 mm, y2 0.6) mapped to
the template's signed convention (R1 = -0.700 m, spacings [-0.140 0 +0.0762]
m, stop at M2; verified by `oi_paraxial`: EFL 0.126, Petzval 0, BFD -0.1405).
Metric = the template's (strict RMS WFE at 1 um, 11 x 11 dense-map max).

| OFF deg | S1 nm | S2 nm | S3 nm | S4 nm | clearance S1 / S2 / S3 / S4 mm |
|---|---|---|---|---|---|
| 4 | 171.6 | 204.0 | 161.6 | 218.9 | -53.9 / -54.8 / -58.6 / -57.7 |
| 6 | 171.6 | 365.1 | 173.1 | 241.8 | -53.9 / -55.2 / -59.2 / -59.0 |
| 8 | 171.6 | 676.8 | 195.8 | 227.5 | -53.9 / -55.8 / -59.2 / -58.7 |
| 10 | 171.6 | 1206.7 | 216.1 | 230.6 | -53.9 / -51.6 / -58.5 / -55.0 |

The image closes (S3 162-216 nm); the clearance is FLAT in the offset and
already -54 mm on axis (S4's tilts, all < 0.7 deg, buy <= 3.5 mm).  S5 was
stopped (addendum 20 item 3).  Step-1 summary mats were not written (the
runs were killed in S5); the numbers are the per-offset `REPORT.md` files.

## 2. The envelope scan (addendum 20) and the stall rule (addendum 22)

| t1 mm | y2 | OFF | last stage | clearance mm (worst pair) | verdict |
|---|---|---|---|---|---|
| 300 | 0.6 | 15 | S3 24 612 nm | -37.9 (M1->M2 x M3) | **does not package** (S1 converged, 288.7 nm) |
| 450 | 0.6 | 10 | S3 untraceable | -- | no verdict: S1 plateaued at 4 364 nm |
| 600 | 0.6 | 8 | S3 untraceable | -- | no verdict: S1 stalled at 20 338 nm |
| 600 | 0.6 | 10 | S3 map INVALID | -- | no verdict: S1 stalled |

The one counted case's S3 was still descending at the 12-iteration cap, and
its S3 moved the layout from the seed's -7.1 mm (engine) to -37.9 mm: the
template's S3 carries no clearance row.  "Converged" is stated as a number in
the record: S1 dense-map max <= `tel3_s1_conv_nm` = 1000 nm (the converged S1
solves here: 172 and 289 nm).

## 3. The first-order screen (addendum 21)

`design/src/tma_screen.m`: OI_CLEAR's nine pairs evaluated paraxially (chief
through M2's vertex + the axial marginal, meridional, the box centre and the
along-track extremes at zero cross-track angle -- exactly OI_CLEAR's field
set -- one disk per field at 1.15 x the marginal height, signed penetration).
Validated against the engine gate on the template's own seeds: same binding
pair at every offset, floor within 3-5 mm (screen pessimistic), single pairs
within 14 mm.  The scan (t1 140-600 x y2 0.3-0.9 x OFF 8-30, 840 rows, all
seeds close): 353 rows pass all nine at >= +5 mm, **19 at OFF <= 15 deg, every
one at y2 = 0.3-0.4** -- the back end (`M3->FP x M2`, which binds 654 of 840
rows, as addendum 21 predicted) closes when t2 and the BFD are short.  t1 is
not the knob: the best row at <= 15 deg is the SHORTEST envelope (t1 140, y2
0.3, 14 deg, +12.7 mm, 161 mm long, M1 273 mm across from the 24.6 deg
cross-track field).  The closing offset is ~12-14 deg, not addendum 21's ~26.

Two things the screen does not see, both measured on the engine:
1. **Reach on a strong M3.** Every passing row at <= 15 deg puts M3's
   footprint at rho/|R| >= 0.65 at the box corners (the y2 = 0.6 seeds that
   trace: 0.49-0.56).  `tma_screen` now reports `.rho_R`; no cut is applied.
2. **The template's stop pose.** Its entrance-pupil construction puts the
   stop 19.6 mm off M2's vertex at 14 deg on the y2 = 0.3 seed (the paraxial
   chief through the paraxial EP centre misses by that much in real rays);
   the cross-track edge fields then die AT M2.  Forced onto the vertex, y2 =
   0.3 still loses them; **y2 = 0.4 traces** (277/277 to 9 deg cross-track,
   260/277 at +-12.3 deg) with engine clearance +21.7 mm (screen +5.4).

## 4. A template defect found and fixed (opt-in): the R2/R3 root branch

With `eliminate = 'R2R3'` the template re-solves R2, R3 from EFL + Petzval at
every S1/S3 iterate by Newton from c2 = c3 = -1 /m.  That system has TWO
roots; at y2 = 0.3 the fixed start lands on a CONCAVE M2 (R2 = +0.205 m, BFD
0.1 mm) -- the first two screened-row runs solved a different telescope from
the one screened, and were killed.  Fix: `offset_imager_params.seed_R_m`
(full seed radii) -- `oi_seed` starts from them and `oi_close` restarts each
re-solve from the current radii (`oi_paraxial` REQ.c0), holding the branch.
Default `[]` = the record path, unchanged (rodgers3 untouched).  Stage `t3`
passes it.

## 5. The screened rows through S1-S3 (the y2 = 0.4 corner)

| t1 | y2 | OFF | S1 nm | S3 nm | S3 clearance mm (worst) | exit err | verdict |
|---|---|---|---|---|---|---|---|
| 140 | 0.4 | 14 | 9 388 | 2 057 837 | **+17.5** (M3->FP x M2) | 7.2 deg | packages; no image verdict |
| 140 | 0.4 | 15 | 9 388 | 338 207 | **+10.4** (M3->FP x M2) | 19.8 deg | packages; no image verdict |

The engine confirms packaging at this corner.  The image is not readable: S1
hit its 12-iteration cap still descending (24 756 -> 450 nm on the 3 x 3
solve set, 9 388 nm on the dense map: the solve set carries three fields
across 24.6 deg), S2 lost 18-20 of 121 fields, and S3 restarted from a FRESH
sphere seed (~705 um) because the carried design traced worse -- nothing from
S1 reached S3.  At 14 deg S3 plateaued (LM damping 2 -> 2 200); at 15 deg it
ended at the cap still falling.

## 6. Addendum 23: the y2 continuation -- the hard stop fires on a solved design

Runner stage `t3w` (`dyson5_t3w.txt` / `.mat`, decks + maps in `t3/
dyson5_t3w_y<y2>_*`).  S1 at t1 140 mm walked y2 0.60 -> 0.40, each step
warm-started from the previous solved S1 (conics, aspheres, FPA refit
carried), R1 HELD at the family point (`hold_R1`, below), R2/R3 branch held
(`seed_R_m`), solve set 5 across the slit x 3 along the strip
(`oi_fieldset` [nx ny], below), cap 40, every solve stopping on `oi_solve`'s
own test (< 0.1 % gain), none capped.  Then S3 at 14 deg seeded FROM the
y2 0.40 S1.

| y2 | S1 map max nm | avg | iters | R1 / R2 / R3 mm | K1 / K2 / K3 | M3 rho/R | edge kept | screen floor mm |
|---|---|---|---|---|---|---|---|---|
| 0.60 | 348.6 | 218.0 | 3 | 700.0 / 124.7 / 151.8 | -4.40 / 0.139 / 0.079 | 0.531 | 1.000 | -10.9 |
| 0.55 | 423.0 | 271.8 | 3 | 622.2 / 113.7 / 139.1 | -3.66 / -0.070 / 0.085 | 0.551 | 1.000 | -6.6 |
| 0.50 | 517.7 | 342.9 | 3 | 560.0 / 103.1 / 126.4 | -3.14 / -0.497 / 0.097 | 0.575 | 1.000 | -2.5 |
| 0.45 | 638.2 | 438.8 | 5 | 509.1 / 92.9 / 113.7 | -2.80 / -1.24 / 0.172 | 0.608 | 1.000 | +1.6 |
| 0.40 | 786.7 | 561.5 | 7 | 466.7 / 83.0 / 101.0 | -2.51 / -2.75 / 0.264 | 0.889 | 0.984 | +5.4 |

Every step counts (<= 1000 nm); the walk reaches the packaging corner.  The
trade is monotone: each step toward packaging costs the on-axis image.

**S3 at 14 deg from y2 0.40** (stop posed at y -2.27 mm): starts at 552 081
nm (the 787 nm on-axis parent moved to the offset), 4 iterations (own stop),
**dense-map max 1 206 588 nm, avg 218 783 nm; clearance +5.9 mm (M3->FP x
M2) -- the gate HOLDS, so S4 was not needed;** exit error 0.151 deg, M3
rho/R 0.879, edge kept 0.986 (1.4 % lost, under the 5 % stop).

**HARD STOP (beat 5c earned): the offset solve ends at 1 206 588 nm > 250 nm
with the gate satisfied (+5.9 mm).**  This telecentric three-mirror family
packages at F/1.8 over 24.6 x 0.3 deg only at y2 ~0.4, 14 deg, and there its
symmetric-surface image is ~4800x the bar.  S4 / S5 were not run (the rule
stops at S3 with the gate satisfied).

## 7. Three defects found on the way, all fixed, gated

1. **The template's metric was degenerate for a TELECENTRIC design**
   (`design/src/oi_score.m`, shared).  The exit-pupil anchor crosses the
   chief with a 1e-5 rad probe chief; in a telecentric beam those are
   parallel and the crossing fell back to `X = p1` -- the anchor ON the
   focal plane, a ~zero-radius reference sphere every ray misses.
   `strict_sphere_opl` then returns COMPLEX paths: the solver's residual went
   complex (conics, aspheres and FPA refit all complex after one iteration,
   every later step rejected -- the first R1-held walk's "stall" at 1031 nm
   was this), and `std` of the complex values silently contaminated the
   REPORTED number (the on-axis field read 8209 nm; it is 551 nm).  Fix: the
   degenerate branch anchors far up the chief (1 km: the reference becomes the
   plane normal to the chief, as the engine's FEX telecentric guard does);
   plus a guard -- a field whose paths still come out complex is a wall row
   with one warning line, never a complex residual.  Only the degenerate
   branch changed; non-telecentric designs never reach it.
   **Consequence for sections 1-5:** their WFE numbers (map max, S1-S3) were
   scored BEFORE this fix, on designs at or near telecentric, so any field
   that hit the degenerate anchor carries the contamination and those solves
   may have walked off the real line too.  They are NOT re-run here; read
   them as superseded where they conflict with section 6.  Their CLEARANCE
   numbers come from `oi_clear` and the screen from `tma_screen` -- neither
   touches the anchor -- and stand.
2. **The R2/R3 root branch** (section 4): `seed_R_m`, gate `tOiSeedBranch`.
3. **A free R1 is not a family point** (`oi_solve.m`, opt-in `hold_R1`).  In
   S1/S3 R1 is a solve variable; the first walk run let it go 0.700 -> 2.615
   m with K1 -141 (an effective y2 ~0.89, not the screened family), and the
   step-1 parent's 172 nm was likewise bought at R1 1.20 m.  `hold_R1`
   freezes R1 so, with R2/R3 eliminated, each step IS its family point.
   Default false (the record path).

Also added: `oi_fieldset` takes `[nx ny]` (opt-in; scalar n is the record
path) so a thin strip box is not spent on N rows of 0.3 deg.

## 8. Gates

| gate | committed tree (no change) | this tree |
|---|---|---|
| `tRodgers3` (default path) | -- | 6 pass, 0 fail -- before AND after the `oi_score` fix |
| `tOffsetImager` (default path) | 5 pass, 1 fail | 5 pass, 1 fail -- the SAME test, the SAME numbers, before and after the `oi_score` fix |
| `tOiSeedBranch` (new, freeform suite) | -- | 3 pass, 0 fail, incl. the must-fail leg (the record path lands on the concave root) |

The one `tOffsetImager` failure is PRE-EXISTING and outside this beat:
`test_s3_resolve_recovers`, S3 29 274 nm vs S2 29 024 nm, identical on a
clean worktree of the committed tree; the test's notes recorded s3/s2 = 0.71
on 2026-08-20, so something landed since has changed that solve's path.  Not
chased.

## 9. Where this leaves the telescope

Beat 5c -- the two-mirror modified Schwarzschild -- is earned by addendum
23's image stop, on a converged, real-valued, packaging design.  What the
three-mirror record contributes to 5c: the packaging corner exists (y2 ~0.4,
14 deg, +5.9 mm after the offset solve), the screen that found it is general
(`tma_screen`), and the image cost of the offset at F/1.8 over a 24.6 deg
strip with symmetric surfaces is three orders above the bar -- whether S4 /
S5 (tilts, freeforms) could recover three orders was not asked by the rule
and is not claimed either way.
