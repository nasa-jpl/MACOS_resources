# dyson5_t3_t600_off10 -- offset_imager run

2026-10-02 05:41:09.  EPD 70 mm, F/1.8 (EFL 0.126 m held as an identity), lambda 1.00 um, box 24.5553x0.3° offset +10°, spacings [-0.6 0 0.0762688] m, model 256, nGridpts 41.

Every WFE number below: strict RMS WFE, sphere centred on the spot centroid on the stage's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense-map MAXIMUM over the box.

## S1 coaxial, on-axis box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1466 m |
| petzval c1-c2+c3 | 8.882e-16 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 19.8 mm |
| radii R1..R3 | -2.91300 / -0.14295 / -0.15032 m |
| conics K1..K3 | 17.599 / -0.017717 / 0.0018678 |
| solve | s1: 29215.6 -> 20337.6 nm (qmean over solve set), 3 iters |
| **map max** | **23754.3 nm** at XAN -12.3 YAN +0.1 |
| map avg / std / min | 18341.1 / 3782.2 / 12772.4 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -43.9 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_t600_off10_s1_layout.png`, `dyson5_t3_t600_off10_s1_map.png`.  Deck: `dyson5_t3_t600_off10_s1.in`.

## S2 offset box, FPA tilt/focus refit only

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +10°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1466 m |
| petzval c1-c2+c3 | 8.882e-16 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 62.0 mm |
| radii R1..R3 | -2.91300 / -0.14295 / -0.15032 m |
| conics K1..K3 | 17.599 / -0.017717 / 0.0018678 |
| solve | s2: 339452.0 -> 318162.2 nm (qmean over solve set), 5 iters |
| **map max** | **1648346.5 nm** at XAN -7.4 YAN +10.1 |
| map avg / std / min | 937498.3 / 356393.1 / 295834.6 nm |
| exit chief | 179.999° in Y-Z; err 0.001° vs pin -> PASS |
| clearance floor | -33.1 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_t600_off10_s2_layout.png`, `dyson5_t3_t600_off10_s2_map.png`.  Deck: `dyson5_t3_t600_off10_s2.in`.

**The cost of the offset:** map max grows 69x (23754 -> 1648347 nm) when the box moves 10° off axis with nothing but the FPA allowed to follow.

## S3 symmetric surfaces re-solved at the offset box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +10°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1490 m |
| petzval c1-c2+c3 | -1.776e-15 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 21.3 mm |
| radii R1..R3 | -3.00000 / -0.14454 / -0.15186 m |
| conics K1..K3 | 0 / 0 / 0 |
| solve | s3: 1000000000.0 -> 1000000000.0 nm (qmean over solve set), 1 iters |
| **map max** | **INVALID -- 121/121 fields lost every ray** (finite-only max NaN nm) |
| map avg / std / min | NaN / NaN / NaN nm |
| exit chief | NaN° in Y-Z; err NaN° vs pin -> FAIL |
| clearance floor | 0.0 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_t600_off10_s3_layout.png`, `dyson5_t3_t600_off10_s3_map.png`.  Deck: `dyson5_t3_t600_off10_s3.in`.

Conic migration under the bias doctrine (solve at the used field): K = [17.6 -0.01772 0.001868] -> [0 0 0].

## The ladder

| stage | map max (nm) | map avg | map std |
|---|---|---|---|
| s1 | 23754.3 | 18341.1 | 3782.2 |
| s2 | 1648346.5 | 937498.3 | 356393.1 |
| s3 | NaN | NaN | NaN |
