# dyson5_t3_t300_off15 -- offset_imager run

2026-10-02 05:41:09.  EPD 70 mm, F/1.8 (EFL 0.126 m held as an identity), lambda 1.00 um, box 24.5553x0.3° offset +15°, spacings [-0.3 0 0.0762573] m, model 256, nGridpts 41.

Every WFE number below: strict RMS WFE, sphere centred on the spot centroid on the stage's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense-map MAXIMUM over the box.

## S1 coaxial, on-axis box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1691 m |
| petzval c1-c2+c3 | -8.882e-16 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 30.2 mm |
| radii R1..R3 | -2.05773 / -0.15308 / -0.16536 m |
| conics K1..K3 | -19.809 / 0.51544 / 0.095217 |
| solve | s1: 26744.8 -> 61.2 nm (qmean over solve set), 12 iters |
| **map max** | **288.7 nm** at XAN -9.8 YAN +0.0 |
| map avg / std / min | 163.8 / 94.1 / 60.9 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -51.2 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_t300_off15_s1_layout.png`, `dyson5_t3_t300_off15_s1_map.png`.  Deck: `dyson5_t3_t300_off15_s1.in`.

## S2 offset box, FPA tilt/focus refit only

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +15°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1691 m |
| petzval c1-c2+c3 | -8.882e-16 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 29.9 mm |
| radii R1..R3 | -2.05773 / -0.15308 / -0.16536 m |
| conics K1..K3 | -19.809 / 0.51544 / 0.095217 |
| solve | s2: 47268.2 -> 7793.0 nm (qmean over solve set), 4 iters |
| **map max** | **9969.2 nm** at XAN -12.3 YAN +15.2 |
| map avg / std / min | 4630.8 / 2726.1 / 1706.0 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -36.9 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_t300_off15_s2_layout.png`, `dyson5_t3_t300_off15_s2_map.png`.  Deck: `dyson5_t3_t300_off15_s2.in`.

**The cost of the offset:** map max grows 35x (289 -> 9969 nm) when the box moves 15° off axis with nothing but the FPA allowed to follow.

## S3 symmetric surfaces re-solved at the offset box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +15°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1706 m |
| petzval c1-c2+c3 | 8.882e-16 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 30.0 mm |
| radii R1..R3 | -2.11035 / -0.15408 / -0.16619 m |
| conics K1..K3 | -28.058 / 0.79942 / 0.09242 |
| solve | s3: 47268.2 -> 23279.0 nm (qmean over solve set), 12 iters |
| **map max** | **24611.5 nm** at XAN -12.3 YAN +14.8 |
| map avg / std / min | 23032.7 / 986.5 / 21777.7 nm |
| exit chief | 179.614° in Y-Z; err 0.386° vs pin -> PASS |
| clearance floor | -37.9 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_t300_off15_s3_layout.png`, `dyson5_t3_t300_off15_s3_map.png`.  Deck: `dyson5_t3_t300_off15_s3.in`.

Conic migration under the bias doctrine (solve at the used field): K = [-19.81 0.5154 0.09522] -> [-28.06 0.7994 0.09242].

## The ladder

| stage | map max (nm) | map avg | map std |
|---|---|---|---|
| s1 | 288.7 | 163.8 | 94.1 |
| s2 | 9969.2 | 4630.8 | 2726.1 |
| s3 | 24611.5 | 23032.7 | 986.5 |
