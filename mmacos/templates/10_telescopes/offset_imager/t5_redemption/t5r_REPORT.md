# t5-redemption -- offset_imager run

2026-08-22 12:39:28.  EPD 150 mm, F/3.3 (EFL 0.495 m held as an identity), lambda 1.00 um, box 15x15° offset +22.5°, spacings [-1.19278 0 1.22237] m, model 256, nGridpts 41.

Every WFE number below: strict RMS WFE, sphere centred on the spot centroid on the stage's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense-map MAXIMUM over the box.

## S1 coaxial, on-axis box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 15x15° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.495000 m = EPD 150 mm x F/3.3 |
| paraxial BFD | -1.4043 m |
| petzval c1-c2+c3 | 0.000e+00 1/m |
| plate scale | 143.99 um/arcmin |
| stop semi-diameter (traced) | 92.7 mm |
| radii R1..R3 | 14.50216 / -1.89308 / -1.67449 m |
| conics K1..K3 | -23.694 / 3.1871 / 0.081691 |
| solve | s1: 19889.7 -> 67.3 nm (qmean over solve set), 1 iters |
| **map max** | **74.9 nm** at XAN -7.5 YAN -7.5 |
| map avg / std / min | 66.4 / 2.3 / 63.8 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -97.0 mm (FAIL; gate >= 25 mm; WARN < 40 mm) |

Figures: `t5r_s1_layout.png`, `t5r_s1_map.png`.  Deck: `t5r_s1.in`.

## S2 offset box, FPA tilt/focus refit only

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 15x15° box at YAN +22.5°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.495000 m = EPD 150 mm x F/3.3 |
| paraxial BFD | -1.4043 m |
| petzval c1-c2+c3 | 0.000e+00 1/m |
| plate scale | 143.99 um/arcmin |
| stop semi-diameter (traced) | 267.5 mm |
| radii R1..R3 | 14.50216 / -1.89308 / -1.67449 m |
| conics K1..K3 | -23.694 / 3.1871 / 0.081691 |
| solve | s2: 1000000000.0 -> 1000000000.0 nm (qmean over solve set), 1 iters |
| **map max** | **INVALID -- 104/121 fields lost every ray** (finite-only max 1994488.4 nm) |
| map avg / std / min | 296751.3 / 470227.5 / 12531.0 nm |
| exit chief | NaN° in Y-Z; err NaN° vs pin -> FAIL |
| clearance floor | 0.0 mm (FAIL; gate >= 25 mm; WARN < 40 mm) |

Figures: `t5r_s2_layout.png`, `t5r_s2_map.png`.  Deck: `t5r_s2.in`.

**The cost of the offset:** map max grows 26616x (75 -> 1994488 nm) when the box moves 22.5° off axis with nothing but the FPA allowed to follow.

## S3 symmetric surfaces re-solved at the offset box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 15x15° box at YAN +22.5°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.495000 m = EPD 150 mm x F/3.3 |
| paraxial BFD | -1.4048 m |
| petzval c1-c2+c3 | 0.000e+00 1/m |
| plate scale | 143.99 um/arcmin |
| stop semi-diameter (traced) | 90.2 mm |
| radii R1..R3 | 14.44142 / -1.89471 / -1.67496 m |
| conics K1..K3 | -0.0076478 / -0.00021831 / -0.00011905 |
| solve | s3: 520353.4 -> 519861.8 nm (qmean over solve set), 3 iters |
| **map max** | **597822.9 nm** at XAN -7.5 YAN +30.0 |
| map avg / std / min | 517647.5 / 34493.0 / 475903.5 nm |
| exit chief | 96.267° in Y-Z; err 83.733° vs pin -> FAIL |
| clearance floor | -211.7 mm (FAIL; gate >= 25 mm; WARN < 40 mm) |

Figures: `t5r_s3_layout.png`, `t5r_s3_map.png`.  Deck: `t5r_s3.in`.

Conic migration under the bias doctrine (solve at the used field): K = [-23.69 3.187 0.08169] -> [-0.007648 -0.0002183 -0.0001191].

## S4 + mirror tilt/decenter + radii

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 15x15° box at YAN +22.5°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.495000 m = EPD 150 mm x F/3.3 |
| paraxial BFD | -1.4046 m |
| petzval c1-c2+c3 | -2.757e-05 1/m |
| plate scale | 143.99 um/arcmin |
| stop semi-diameter (traced) | 90.1 mm |
| radii R1..R3 | 14.45365 / -1.89447 / -1.67485 m |
| conics K1..K3 | -0.0020454 / 0.00038235 / -0.00011901 |
| YDE (mm) | +7.424 / -2.748 / -1.409 |
| ADE (deg) | -0.1057 / -0.0850 / -0.0809 |
| solve | s4: 519861.8 -> 517529.9 nm (qmean over solve set), 3 iters |
| **map max** | **595565.2 nm** at XAN +7.5 YAN +30.0 |
| map avg / std / min | 515043.2 / 34673.0 / 473004.7 nm |
| exit chief | 96.719° in Y-Z; err 83.281° vs pin -> FAIL |
| clearance floor | -205.3 mm (FAIL; gate >= 25 mm; WARN < 40 mm) |

Figures: `t5r_s4_layout.png`, `t5r_s4_map.png`.  Deck: `t5r_s4.in`.

## S5 + Zernike departures (aspheres replaced)

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 15x15° box at YAN +22.5°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.495000 m = EPD 150 mm x F/3.3 |
| paraxial BFD | -1.4046 m |
| petzval c1-c2+c3 | -3.002e-05 1/m |
| plate scale | 143.99 um/arcmin |
| stop semi-diameter (traced) | 90.1 mm |
| radii R1..R3 | 14.45859 / -1.89434 / -1.67481 m |
| conics K1..K3 | 0.048046 / 0.00058707 / -0.00010928 |
| YDE (mm) | +8.376 / -3.751 / -2.650 |
| ADE (deg) | -0.1247 / -0.1157 / -0.1064 |
| solve | s5: 509294.7 -> 508238.6 nm (qmean over solve set), 3 iters |
| **map max** | **594473.2 nm** at XAN -7.5 YAN +30.0 |
| map avg / std / min | 513890.2 / 34711.7 / 471801.7 nm |
| exit chief | 96.875° in Y-Z; err 83.125° vs pin -> FAIL |
| clearance floor | -203.6 mm (FAIL; gate >= 25 mm; WARN < 40 mm) |

Figures: `t5r_s5_layout.png`, `t5r_s5_map.png`.  Deck: `t5r_s5.in`.

## The ladder

| stage | map max (nm) | map avg | map std |
|---|---|---|---|
| s1 | 74.9 | 66.4 | 2.3 |
| s2 | 1994488.4 | 296751.3 | 470227.5 |
| s3 | 597822.9 | 517647.5 | 34493.0 |
| s4 | 595565.2 | 515043.2 | 34673.0 |
| s5 | 594473.2 | 513890.2 | 34711.7 |
