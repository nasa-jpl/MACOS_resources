# dyson5_t3_t140_off15_y40 -- offset_imager run

2026-10-02 06:01:16.  EPD 70 mm, F/1.8 (EFL 0.126 m held as an identity), lambda 1.00 um, box 24.5553x0.3° offset +15°, spacings [-0.14 0 0.050653] m, model 256, nGridpts 41.

Every WFE number below: strict RMS WFE, sphere centred on the spot centroid on the stage's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense-map MAXIMUM over the box.

## S1 coaxial, on-axis box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.0601 m |
| petzval c1-c2+c3 | 0.000e+00 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 40.5 mm |
| radii R1..R3 | -0.39739 / -0.06853 / -0.08281 m |
| conics K1..K3 | -0.81159 / 0.33944 / -0.009698 |
| solve | s1: 32150.2 -> 449.7 nm (qmean over solve set), 12 iters |
| **map max** | **9388.3 nm** at XAN +7.4 YAN +0.1 |
| map avg / std / min | 4779.6 / 3326.6 / 161.1 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -17.9 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_t140_off15_y40_s1_layout.png`, `dyson5_t3_t140_off15_y40_s1_map.png`.  Deck: `dyson5_t3_t140_off15_y40_s1.in`.

## S2 offset box, FPA tilt/focus refit only

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +15°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.0601 m |
| petzval c1-c2+c3 | 0.000e+00 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 97.1 mm |
| radii R1..R3 | -0.39739 / -0.06853 / -0.08281 m |
| conics K1..K3 | -0.81159 / 0.33944 / -0.009698 |
| solve | s2: 15807796.7 -> 19718303.4 nm (qmean over solve set), 2 iters |
| **map max** | **INVALID -- 18/121 fields lost every ray** (finite-only max 43316869.4 nm) |
| map avg / std / min | 29479257.5 / 7880195.8 / 19392618.2 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -75.4 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_t140_off15_y40_s2_layout.png`, `dyson5_t3_t140_off15_y40_s2_map.png`.  Deck: `dyson5_t3_t140_off15_y40_s2.in`.

**The cost of the offset:** map max grows 4614x (9388 -> 43316869 nm) when the box moves 15° off axis with nothing but the FPA allowed to follow.

## S3 symmetric surfaces re-solved at the offset box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +15°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.0855 m |
| petzval c1-c2+c3 | -1.776e-15 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 13.1 mm |
| radii R1..R3 | -0.46978 / -0.08353 / -0.10160 m |
| conics K1..K3 | 0.05061 / -0.098675 / -0.018075 |
| solve | s3: 378253.1 -> 95845.9 nm (qmean over solve set), 12 iters |
| **map max** | **338207.0 nm** at XAN +12.3 YAN +15.1 |
| map avg / std / min | 106758.0 / 119545.4 / 27482.8 nm |
| exit chief | -160.187° in Y-Z; err 19.813° vs pin -> FAIL |
| clearance floor | 10.4 mm (PASS; gate >= 5 mm) |

Figures: `dyson5_t3_t140_off15_y40_s3_layout.png`, `dyson5_t3_t140_off15_y40_s3_map.png`.  Deck: `dyson5_t3_t140_off15_y40_s3.in`.

Conic migration under the bias doctrine (solve at the used field): K = [-0.8116 0.3394 -0.009698] -> [0.05061 -0.09867 -0.01808].

## The ladder

| stage | map max (nm) | map avg | map std |
|---|---|---|---|
| s1 | 9388.3 | 4779.6 | 3326.6 |
| s2 | 43316869.4 | 29479257.5 | 7880195.8 |
| s3 | 338207.0 | 106758.0 | 119545.4 |
