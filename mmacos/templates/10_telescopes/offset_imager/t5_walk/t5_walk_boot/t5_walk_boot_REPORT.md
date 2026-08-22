# t5-walk -- offset_imager run

2026-08-22 14:40:39.  EPD 150 mm, F/3.3 (EFL 0.495 m held as an identity), lambda 1.00 um, box 5x5° offset +22.5°, spacings [-1.19278 0 1.22237] m, model 256, nGridpts 41.

Every WFE number below: strict RMS WFE, sphere centred on the spot centroid on the stage's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense-map MAXIMUM over the box.

## S1 coaxial, on-axis box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 5x5° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.495000 m = EPD 150 mm x F/3.3 |
| paraxial BFD | -1.5604 m |
| petzval c1-c2+c3 | 0.000e+00 1/m |
| plate scale | 143.99 um/arcmin |
| stop semi-diameter (traced) | 682.0 mm |
| radii R1..R3 | 5.41468 / -2.79439 / -1.84317 m |
| conics K1..K3 | 26.034 / 1.5874 / 0.04611 |
| solve | s1: 3891.8 -> 632.8 nm (qmean over solve set), 8 iters |
| **map max** | **756.0 nm** at XAN +2.5 YAN -2.5 |
| map avg / std / min | 534.5 / 58.7 / 484.7 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -119.2 mm (FAIL; gate >= 25 mm; WARN < 40 mm) |

Figures: `t5_walk_boot_s1_layout.png`, `t5_walk_boot_s1_map.png`.  Deck: `t5_walk_boot_s1.in`.

## S2 offset box, FPA tilt/focus refit only

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 5x5° box at YAN +22.5°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.495000 m = EPD 150 mm x F/3.3 |
| paraxial BFD | -1.5604 m |
| petzval c1-c2+c3 | 0.000e+00 1/m |
| plate scale | 143.99 um/arcmin |
| stop semi-diameter (traced) | 735.1 mm |
| radii R1..R3 | 5.41468 / -2.79439 / -1.84317 m |
| conics K1..K3 | 26.034 / 1.5874 / 0.04611 |
| solve | s2: 401169216.0 -> 107710131.8 nm (qmean over solve set), 9 iters |
| **map max** | **154364316.4 nm** at XAN -1.0 YAN +25.0 |
| map avg / std / min | 108876297.1 / 19472445.8 / 81362467.9 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -84839.8 mm (FAIL; gate >= 25 mm; WARN < 40 mm) |

Figures: `t5_walk_boot_s2_layout.png`, `t5_walk_boot_s2_map.png`.  Deck: `t5_walk_boot_s2.in`.

**The cost of the offset:** map max grows 204189x (756 -> 154364316 nm) when the box moves 22.5° off axis with nothing but the FPA allowed to follow.

## S3 symmetric surfaces re-solved at the offset box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 5x5° box at YAN +22.5°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.495000 m = EPD 150 mm x F/3.3 |
| paraxial BFD | -1.4419 m |
| petzval c1-c2+c3 | 0.000e+00 1/m |
| plate scale | 143.99 um/arcmin |
| stop semi-diameter (traced) | 82.0 mm |
| radii R1..R3 | 10.86884 / -2.03021 / -1.71067 m |
| conics K1..K3 | -12.619 / 4.4649 / 0.092906 |
| solve | s3: 50311.9 -> 35.8 nm (qmean over solve set), 30 iters |
| **map max** | **57.4 nm** at XAN +0.0 YAN +22.5 |
| map avg / std / min | 39.3 / 10.2 / 20.3 nm |
| exit chief | -179.926° in Y-Z; err 0.074° vs pin -> PASS |
| clearance floor | 108.8 mm (PASS; gate >= 25 mm) |

Figures: `t5_walk_boot_s3_layout.png`, `t5_walk_boot_s3_map.png`.  Deck: `t5_walk_boot_s3.in`.

Conic migration under the bias doctrine (solve at the used field): K = [26.03 1.587 0.04611] -> [-12.62 4.465 0.09291].

## S4 + mirror tilt/decenter + radii

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 5x5° box at YAN +22.5°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.495000 m = EPD 150 mm x F/3.3 |
| paraxial BFD | -1.4050 m |
| petzval c1-c2+c3 | -2.250e-02 1/m |
| plate scale | 143.99 um/arcmin |
| stop semi-diameter (traced) | 94.7 mm |
| radii R1..R3 | 11.60264 / -2.07259 / -1.69155 m |
| conics K1..K3 | -18.416 / 4.7789 / 0.094794 |
| YDE (mm) | +62.798 / +0.422 / -0.732 |
| ADE (deg) | -0.0007 / +0.0664 / +0.0385 |
| solve | s4: 35.8 -> 15.8 nm (qmean over solve set), 30 iters |
| **map max** | **18.9 nm** at XAN +2.5 YAN +25.0 |
| map avg / std / min | 13.4 / 2.0 / 9.7 nm |
| exit chief | -179.998° in Y-Z; err 0.002° vs pin -> PASS |
| clearance floor | 111.3 mm (PASS; gate >= 25 mm) |

Figures: `t5_walk_boot_s4_layout.png`, `t5_walk_boot_s4_map.png`.  Deck: `t5_walk_boot_s4.in`.

## S5 + Zernike departures (aspheres replaced)

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 5x5° box at YAN +22.5°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.495000 m = EPD 150 mm x F/3.3 |
| paraxial BFD | -1.4161 m |
| petzval c1-c2+c3 | -1.775e-02 1/m |
| plate scale | 143.99 um/arcmin |
| stop semi-diameter (traced) | 94.8 mm |
| radii R1..R3 | 11.15580 / -2.07825 / -1.69905 m |
| conics K1..K3 | -18.403 / 4.8624 / 0.089742 |
| YDE (mm) | +105.925 / +0.324 / -3.487 |
| ADE (deg) | -0.0736 / +0.0716 / -0.0327 |
| solve | s5: 4613.8 -> 7.8 nm (qmean over solve set), 30 iters |
| **map max** | **10.9 nm** at XAN +0.0 YAN +20.0 |
| map avg / std / min | 6.8 / 1.7 / 3.9 nm |
| exit chief | 179.999° in Y-Z; err 0.001° vs pin -> PASS |
| clearance floor | 98.0 mm (PASS; gate >= 25 mm) |

Figures: `t5_walk_boot_s5_layout.png`, `t5_walk_boot_s5_map.png`.  Deck: `t5_walk_boot_s5.in`.

## The ladder

| stage | map max (nm) | map avg | map std |
|---|---|---|---|
| s1 | 756.0 | 534.5 | 58.7 |
| s2 | 154364316.4 | 108876297.1 | 19472445.8 |
| s3 | 57.4 | 39.3 | 10.2 |
| s4 | 18.9 | 13.4 | 2.0 |
| s5 | 10.9 | 6.8 | 1.7 |
