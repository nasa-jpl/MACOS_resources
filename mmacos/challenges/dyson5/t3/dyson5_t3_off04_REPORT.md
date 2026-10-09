# dyson5_t3_off04 -- offset_imager run

2026-10-02 04:28:22.  EPD 70 mm, F/1.8 (EFL 0.126 m held as an identity), lambda 1.00 um, box 24.5553x0.3° offset +4°, spacings [-0.14 0 0.076231] m, model 256, nGridpts 41.

Every WFE number below: strict RMS WFE, sphere centred on the spot centroid on the stage's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense-map MAXIMUM over the box.

## S1 coaxial, on-axis box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1786 m |
| petzval c1-c2+c3 | -1.776e-15 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 27.3 mm |
| radii R1..R3 | -1.20212 / -0.15062 / -0.17218 m |
| conics K1..K3 | -23.005 / 0.13893 / 0.070689 |
| solve | s1: 24181.9 -> 118.1 nm (qmean over solve set), 12 iters |
| **map max** | **171.6 nm** at XAN -9.8 YAN +0.0 |
| map avg / std / min | 134.6 / 22.7 / 113.2 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -53.9 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_off04_s1_layout.png`, `dyson5_t3_off04_s1_map.png`.  Deck: `dyson5_t3_off04_s1.in`.

## S2 offset box, FPA tilt/focus refit only

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +4°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1786 m |
| petzval c1-c2+c3 | -1.776e-15 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 27.3 mm |
| radii R1..R3 | -1.20212 / -0.15062 / -0.17218 m |
| conics K1..K3 | -23.005 / 0.13893 / 0.070689 |
| solve | s2: 3668.1 -> 164.5 nm (qmean over solve set), 3 iters |
| **map max** | **204.0 nm** at XAN +12.3 YAN +4.2 |
| map avg / std / min | 148.5 / 29.2 / 104.1 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -54.8 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_off04_s2_layout.png`, `dyson5_t3_off04_s2_map.png`.  Deck: `dyson5_t3_off04_s2.in`.

**The cost of the offset:** map max grows 1x (172 -> 204 nm) when the box moves 4° off axis with nothing but the FPA allowed to follow.

## S3 symmetric surfaces re-solved at the offset box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +4°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1915 m |
| petzval c1-c2+c3 | 0.000e+00 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 29.5 mm |
| radii R1..R3 | -1.63653 / -0.16121 / -0.17881 m |
| conics K1..K3 | -47.888 / 0.088459 / 0.057366 |
| solve | s3: 3668.1 -> 87.4 nm (qmean over solve set), 12 iters |
| **map max** | **161.6 nm** at XAN +9.8 YAN +4.2 |
| map avg / std / min | 110.8 / 26.4 / 69.9 nm |
| exit chief | 179.984° in Y-Z; err 0.016° vs pin -> PASS |
| clearance floor | -58.6 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_off04_s3_layout.png`, `dyson5_t3_off04_s3_map.png`.  Deck: `dyson5_t3_off04_s3.in`.

Conic migration under the bias doctrine (solve at the used field): K = [-23 0.1389 0.07069] -> [-47.89 0.08846 0.05737].

## S4 + mirror tilt/decenter + radii

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +4°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1883 m |
| petzval c1-c2+c3 | -3.132e-02 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 29.0 mm |
| radii R1..R3 | -1.51985 / -0.15975 / -0.17751 m |
| conics K1..K3 | -38.641 / 0.12766 / 0.059824 |
| YDE (mm) | -0.066 / +0.028 / -0.029 |
| ADE (deg) | +0.0925 / -0.0259 / -0.0004 |
| solve | s4: 87.4 -> 72.0 nm (qmean over solve set), 4 iters |
| **map max** | **218.9 nm** at XAN +7.4 YAN +4.2 |
| map avg / std / min | 146.1 / 59.9 / 72.4 nm |
| exit chief | -179.989° in Y-Z; err 0.011° vs pin -> PASS |
| clearance floor | -57.7 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_off04_s4_layout.png`, `dyson5_t3_off04_s4_map.png`.  Deck: `dyson5_t3_off04_s4.in`.
