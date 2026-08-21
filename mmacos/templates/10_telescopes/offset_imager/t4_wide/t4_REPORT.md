# t4-wide -- offset_imager run

2026-08-20 12:54:04.  EPD 200 mm, F/2.5 (EFL 0.500 m held as an identity), lambda 1.00 um, box 10x10° offset +12°, spacings [-0.1 0 1.1] m, model 256, nGridpts 41.

Every WFE number below: strict RMS WFE, sphere centred on the spot centroid on the stage's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense-map MAXIMUM over the box.

## S1 coaxial, on-axis box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 10x10° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.500000 m = EPD 200 mm x F/2.5 |
| paraxial BFD | -1.2812 m |
| petzval c1-c2+c3 | -1.110e-16 1/m |
| plate scale | 145.44 um/arcmin |
| stop semi-diameter (traced) | 112.4 mm |
| radii R1..R3 | 2.02856 / -6.37108 / -1.53865 m |
| conics K1..K3 | 4.595 / 6.42 / 0.081059 |
| solve | s1: 22345.3 -> 18.1 nm (qmean over solve set), 11 iters |
| **map max** | **30.0 nm** at XAN +0.0 YAN +0.0 |
| map avg / std / min | 16.2 / 7.5 / 6.1 nm |
| exit chief | 180.000° in Y-Z (report-only; no pin) |
| clearance floor | 9.5 mm (PASS; gate >= 5 mm; WARN < 10 mm) |

Figures: `t4_s1_layout.png`, `t4_s1_map.png`.  Deck: `t4_s1.in`.

## S2 offset box, FPA tilt/focus refit only

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 10x10° box at YAN +12°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.500000 m = EPD 200 mm x F/2.5 |
| paraxial BFD | -1.2812 m |
| petzval c1-c2+c3 | -1.110e-16 1/m |
| plate scale | 145.44 um/arcmin |
| stop semi-diameter (traced) | 112.4 mm |
| radii R1..R3 | 2.02856 / -6.37108 / -1.53865 m |
| conics K1..K3 | 4.595 / 6.42 / 0.081059 |
| solve | s2: 60652.0 -> 502.1 nm (qmean over solve set), 3 iters |
| **map max** | **942.1 nm** at XAN -5.0 YAN +17.0 |
| map avg / std / min | 282.1 / 259.2 / 33.0 nm |
| exit chief | -175.254° in Y-Z (report-only; no pin) |
| clearance floor | 9.7 mm (PASS; gate >= 5 mm; WARN < 10 mm) |

Figures: `t4_s2_layout.png`, `t4_s2_map.png`.  Deck: `t4_s2.in`.

**The cost of the offset:** map max grows 31x (30 -> 942 nm) when the box moves 12° off axis with nothing but the FPA allowed to follow.

## S3 symmetric surfaces re-solved at the offset box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 10x10° box at YAN +12°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.500000 m = EPD 200 mm x F/2.5 |
| paraxial BFD | -1.2815 m |
| petzval c1-c2+c3 | 0.000e+00 1/m |
| plate scale | 145.44 um/arcmin |
| stop semi-diameter (traced) | 111.1 mm |
| radii R1..R3 | 2.21861 / -4.98389 / -1.53520 m |
| conics K1..K3 | 5.9542 / 27.969 / 0.082686 |
| solve | s3: 60652.0 -> 125.0 nm (qmean over solve set), 15 iters |
| **map max** | **173.6 nm** at XAN +0.0 YAN +12.0 |
| map avg / std / min | 117.9 / 30.6 / 48.8 nm |
| exit chief | -177.081° in Y-Z (report-only; no pin) |
| clearance floor | 9.8 mm (PASS; gate >= 5 mm; WARN < 10 mm) |

Figures: `t4_s3_layout.png`, `t4_s3_map.png`.  Deck: `t4_s3.in`.

Conic migration under the bias doctrine (solve at the used field): K = [4.595 6.42 0.08106] -> [5.954 27.97 0.08269].

## S4 + mirror tilt/decenter + radii

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 10x10° box at YAN +12°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.500000 m = EPD 200 mm x F/2.5 |
| paraxial BFD | -1.2703 m |
| petzval c1-c2+c3 | -1.102e-02 1/m |
| plate scale | 145.44 um/arcmin |
| stop semi-diameter (traced) | 110.4 mm |
| radii R1..R3 | 2.20748 / -5.29095 / -1.53134 m |
| conics K1..K3 | 5.9016 / 28.096 / 0.084085 |
| YDE (mm) | +11.148 / -10.651 / -16.207 |
| ADE (deg) | -1.1317 / -0.5297 / +0.2017 |
| solve | s4: 125.0 -> 63.0 nm (qmean over solve set), 15 iters |
| **map max** | **81.9 nm** at XAN -5.0 YAN +17.0 |
| map avg / std / min | 60.3 / 8.3 / 42.9 nm |
| exit chief | -175.544° in Y-Z (report-only; no pin) |
| clearance floor | 6.8 mm (PASS; gate >= 5 mm; WARN < 10 mm) |

Figures: `t4_s4_layout.png`, `t4_s4_map.png`.  Deck: `t4_s4.in`.

## S5 + Zernike departures (aspheres replaced)

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 10x10° box at YAN +12°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.500000 m = EPD 200 mm x F/2.5 |
| paraxial BFD | -1.2773 m |
| petzval c1-c2+c3 | -4.175e-03 1/m |
| plate scale | 145.44 um/arcmin |
| stop semi-diameter (traced) | 110.2 mm |
| radii R1..R3 | 2.19983 / -5.17766 / -1.53399 m |
| conics K1..K3 | 2.7549 / -43.803 / 0.086696 |
| YDE (mm) | +27.524 / -11.923 / -22.969 |
| ADE (deg) | -1.2482 / -0.6083 / +0.2012 |
| solve | s5: 17925.6 -> 28.8 nm (qmean over solve set), 15 iters |
| **map max** | **40.5 nm** at XAN -5.0 YAN +13.0 |
| map avg / std / min | 26.2 / 7.5 / 12.6 nm |
| exit chief | -175.436° in Y-Z (report-only; no pin) |
| clearance floor | 7.9 mm (PASS; gate >= 5 mm; WARN < 10 mm) |

Figures: `t4_s5_layout.png`, `t4_s5_map.png`.  Deck: `t4_s5.in`.

## The ladder

| stage | map max (nm) | map avg | map std |
|---|---|---|---|
| s1 | 30.0 | 16.2 | 7.5 |
| s2 | 942.1 | 282.1 | 259.2 |
| s3 | 173.6 | 117.9 | 30.6 |
| s4 | 81.9 | 60.3 | 8.3 |
| s5 | 40.5 | 26.2 | 7.5 |
