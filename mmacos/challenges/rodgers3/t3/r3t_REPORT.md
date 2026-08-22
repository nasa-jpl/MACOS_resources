# rodgers3-T3 -- offset_imager run

2026-08-20 10:43:40.  EPD 75 mm, F/4 (EFL 0.300 m held as an identity), lambda 1.00 um, box 20x20° offset +22°, spacings [-0.722897 0 0.740828] m, model 256, nGridpts 41.

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
| clearance floor | 3.4 mm (FAIL; gate >= 35 mm) |

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
| clearance floor | 27.0 mm (FAIL; gate >= 35 mm) |

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
| clearance floor | 13.2 mm (FAIL; gate >= 35 mm) |

Figures: `r3t_s3_layout.png`, `r3t_s3_map.png`.  Deck: `r3t_s3.in`.

Conic migration under the bias doctrine (solve at the used field): K = [-2.419 3.773 0.08279] -> [1.436 4.417 0.11].

## S4 + mirror tilt/decenter + radii

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 20x20° box at YAN +22°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.300003 m = EPD 75 mm x F/4 |
| paraxial BFD | -0.8574 m |
| petzval c1-c2+c3 | -4.513e-02 1/m |
| plate scale | 87.27 um/arcmin |
| stop semi-diameter (traced) | 48.4 mm |
| radii R1..R3 | 6.26952 / -1.31089 / -1.03361 m |
| conics K1..K3 | -10.237 / 5.3456 / 0.11055 |
| YDE (mm) | +112.240 / +3.975 / +2.631 |
| ADE (deg) | +1.5591 / +0.2636 / +0.0355 |
| solve | s4: 140.4 -> 70.7 nm (qmean over solve set), 30 iters |
| **map max** | **113.6 nm** at XAN +0.0 YAN +12.0 |
| map avg / std / min | 68.5 / 20.2 / 34.6 nm |
| exit chief | -179.995° in Y-Z; err 0.005° vs pin -> PASS |
| clearance floor | 34.1 mm (PASS; gate >= 35 mm; WARN < 50 mm) |

Figures: `r3t_s4_layout.png`, `r3t_s4_map.png`.  Deck: `r3t_s4.in`.

## S5 + Zernike departures (aspheres replaced)

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 20x20° box at YAN +22°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.300003 m = EPD 75 mm x F/4 |
| paraxial BFD | -0.8651 m |
| petzval c1-c2+c3 | -3.362e-02 1/m |
| plate scale | 87.27 um/arcmin |
| stop semi-diameter (traced) | 47.5 mm |
| radii R1..R3 | 6.09627 / -1.30605 / -1.03807 m |
| conics K1..K3 | -11.493 / 6.288 / 0.1051 |
| YDE (mm) | +106.278 / +6.116 / +0.277 |
| ADE (deg) | +2.0120 / +0.8457 / -0.0950 |
| solve | s5: 2150.1 -> 35.0 nm (qmean over solve set), 30 iters |
| **map max** | **118.2 nm** at XAN -10.0 YAN +16.0 |
| map avg / std / min | 52.0 / 20.3 / 14.0 nm |
| exit chief | -180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | 34.6 mm (PASS; gate >= 35 mm; WARN < 50 mm) |

Figures: `r3t_s5_layout.png`, `r3t_s5_map.png`.  Deck: `r3t_s5.in`.

## The ladder

| stage | map max (nm) | map avg | map std |
|---|---|---|---|
| s1 | 37.8 | 25.8 | 9.2 |
| s2 | 303585.5 | 67609.0 | 64369.8 |
| s3 | 252.0 | 132.9 | 63.2 |
| s4 | 113.6 | 68.5 | 20.2 |
| s5 | 118.2 | 52.0 | 20.3 |
