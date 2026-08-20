# rodgers3-T3 -- offset_imager run

2026-08-20 00:23:04.  EPD 75 mm, F/4 (EFL 0.300 m held as an identity), lambda 1.00 um, box 20x20° offset +22°, spacings [-0.722897 0 0.740828] m, model 256, nGridpts 41.

Every WFE number below: strict RMS WFE, sphere centred on the spot centroid on the stage's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense-map MAXIMUM over the box.

## S1 coaxial, on-axis box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 20x20° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.300003 m = EPD 75 mm x F/4 |
| paraxial BFD | -0.8510 m |
| petzval c1-c2+c3 | 0.000e+00 1/m |
| plate scale | 87.27 um/arcmin |
| stop semi-diameter (traced) | 46.5 mm |
| radii R1..R3 | 8.79875 / -1.14707 / -1.01478 m |
| conics K1..K3 | -2.4185 / 3.7734 / 0.082795 |
| solve | s1: 13461.7 -> 23.7 nm (qmean over solve set), 7 iters |
| **map max** | **37.8 nm** at XAN -6.0 YAN +2.0 |
| map avg / std / min | 25.8 / 9.2 / 11.2 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | 0.0 mm (FAIL; gate >= 35 mm; WARN < 50 mm) |

Figures: `r3t_s1_layout.png`, `r3t_s1_map.png`.  Deck: `r3t_s1.in`.

## S2 offset box, FPA tilt/focus refit only

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 20x20° box at YAN +22°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.300003 m = EPD 75 mm x F/4 |
| paraxial BFD | -0.8510 m |
| petzval c1-c2+c3 | 0.000e+00 1/m |
| plate scale | 87.27 um/arcmin |
| stop semi-diameter (traced) | 47.8 mm |
| radii R1..R3 | 8.79875 / -1.14707 / -1.01478 m |
| conics K1..K3 | -2.4185 / 3.7734 / 0.082795 |
| solve | s2: 142864.5 -> 142018.1 nm (qmean over solve set), 3 iters |
| **map max** | **303585.5 nm** at XAN +10.0 YAN +32.0 |
| map avg / std / min | 67609.0 / 64369.8 / 32517.0 nm |
| exit chief | -180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | 0.0 mm (FAIL; gate >= 35 mm; WARN < 50 mm) |

Figures: `r3t_s2_layout.png`, `r3t_s2_map.png`.  Deck: `r3t_s2.in`.

**The cost of the offset:** map max grows 8030x (38 -> 303586 nm) when the box moves 22° off axis with nothing but the FPA allowed to follow.

## S3 symmetric surfaces re-solved at the offset box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 20x20° box at YAN +22°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.300003 m = EPD 75 mm x F/4 |
| paraxial BFD | -0.9172 m |
| petzval c1-c2+c3 | -1.110e-16 1/m |
| plate scale | 87.27 um/arcmin |
| stop semi-diameter (traced) | 53.2 mm |
| radii R1..R3 | 4.24640 / -1.45210 / -1.08207 m |
| conics K1..K3 | 1.4357 / 4.417 / 0.10998 |
| solve | s3: 25618.9 -> 140.4 nm (qmean over solve set), 30 iters |
| **map max** | **252.0 nm** at XAN +0.0 YAN +22.0 |
| map avg / std / min | 132.9 / 63.2 / 42.3 nm |
| exit chief | 179.971° in Y-Z; err 0.029° vs pin -> PASS |
| clearance floor | 0.0 mm (FAIL; gate >= 35 mm; WARN < 50 mm) |

Figures: `r3t_s3_layout.png`, `r3t_s3_map.png`.  Deck: `r3t_s3.in`.

Conic migration under the bias doctrine (solve at the used field): K = [-2.419 3.773 0.08279] -> [1.436 4.417 0.11].

## S4 + mirror tilt/decenter + radii

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 20x20° box at YAN +22°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.300003 m = EPD 75 mm x F/4 |
| paraxial BFD | -0.8630 m |
| petzval c1-c2+c3 | -2.323e-02 1/m |
| plate scale | 87.27 um/arcmin |
| stop semi-diameter (traced) | 46.7 mm |
| radii R1..R3 | 6.60089 / -1.26016 / -1.03277 m |
| conics K1..K3 | -11.784 / 4.9842 / 0.10904 |
| YDE (mm) | +143.116 / -6.077 / -5.721 |
| ADE (deg) | -2.5329 / -0.3044 / -0.0647 |
| solve | s4: 140.4 -> 37.4 nm (qmean over solve set), 30 iters |
| **map max** | **58.3 nm** at XAN +0.0 YAN +14.0 |
| map avg / std / min | 37.5 / 9.2 / 22.2 nm |
| exit chief | -179.979° in Y-Z; err 0.021° vs pin -> PASS |
| clearance floor | 0.0 mm (FAIL; gate >= 35 mm; WARN < 50 mm) |

Figures: `r3t_s4_layout.png`, `r3t_s4_map.png`.  Deck: `r3t_s4.in`.

## S5 + Zernike departures (aspheres replaced)

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 20x20° box at YAN +22°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.300003 m = EPD 75 mm x F/4 |
| paraxial BFD | -0.8679 m |
| petzval c1-c2+c3 | -1.068e-02 1/m |
| plate scale | 87.27 um/arcmin |
| stop semi-diameter (traced) | 46.1 mm |
| radii R1..R3 | 6.66657 / -1.24003 / -1.03400 m |
| conics K1..K3 | -11.902 / 5.2901 / 0.10742 |
| YDE (mm) | +214.979 / -6.477 / -8.471 |
| ADE (deg) | -2.8594 / -0.3801 / -0.1843 |
| solve | s5: 1387.7 -> 18.8 nm (qmean over solve set), 30 iters |
| **map max** | **78.4 nm** at XAN -10.0 YAN +28.0 |
| map avg / std / min | 42.6 / 21.5 / 8.4 nm |
| exit chief | 179.999° in Y-Z; err 0.001° vs pin -> PASS |
| clearance floor | 0.0 mm (FAIL; gate >= 35 mm; WARN < 50 mm) |

Figures: `r3t_s5_layout.png`, `r3t_s5_map.png`.  Deck: `r3t_s5.in`.

## The ladder

| stage | map max (nm) | map avg | map std |
|---|---|---|---|
| s1 | 37.8 | 25.8 | 9.2 |
| s2 | 303585.5 | 67609.0 | 64369.8 |
| s3 | 252.0 | 132.9 | 63.2 |
| s4 | 58.3 | 37.5 | 9.2 |
| s5 | 78.4 | 42.6 | 21.5 |
