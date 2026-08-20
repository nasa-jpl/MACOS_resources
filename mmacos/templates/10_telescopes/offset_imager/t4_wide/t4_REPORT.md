# t4-wide -- offset_imager run

2026-08-19 21:44:52.  EPD 200 mm, F/2.5 (EFL 0.500 m held as an identity), lambda 1.00 um, box 10x10° offset +12°, spacings [-0.1 0 1.1] m, model 256, nGridpts 41.

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
| clearance floor | 4.4 mm (FAIL; gate >= 35 mm; WARN < 50 mm) |

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
| clearance floor | 2.0 mm (FAIL; gate >= 35 mm; WARN < 50 mm) |

Figures: `t4_s2_layout.png`, `t4_s2_map.png`.  Deck: `t4_s2.in`.

**The cost of the offset:** map max grows 31x (30 -> 942 nm) when the box moves 12° off axis with nothing but the FPA allowed to follow.

## S3 symmetric surfaces re-solved at the offset box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 10x10° box at YAN +12°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.500000 m = EPD 200 mm x F/2.5 |
| paraxial BFD | -1.2814 m |
| petzval c1-c2+c3 | 0.000e+00 1/m |
| plate scale | 145.44 um/arcmin |
| stop semi-diameter (traced) | 111.9 mm |
| radii R1..R3 | 2.09939 / -5.74209 / -1.53732 m |
| conics K1..K3 | 4.0963 / 22.728 / 0.081424 |
| solve | s3: 502.1 -> 123.6 nm (qmean over solve set), 14 iters |
| **map max** | **176.8 nm** at XAN +0.0 YAN +12.0 |
| map avg / std / min | 116.9 / 31.6 / 49.2 nm |
| exit chief | -175.989° in Y-Z (report-only; no pin) |
| clearance floor | 2.4 mm (FAIL; gate >= 35 mm; WARN < 50 mm) |

Figures: `t4_s3_layout.png`, `t4_s3_map.png`.  Deck: `t4_s3.in`.

Conic migration under the bias doctrine (solve at the used field): K = [4.595 6.42 0.08106] -> [4.096 22.73 0.08142].

## S4 + mirror tilt/decenter + radii

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 10x10° box at YAN +12°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.500000 m = EPD 200 mm x F/2.5 |
| paraxial BFD | -1.2696 m |
| petzval c1-c2+c3 | -1.161e-02 1/m |
| plate scale | 145.44 um/arcmin |
| stop semi-diameter (traced) | 111.1 mm |
| radii R1..R3 | 2.10071 / -6.07266 / -1.53301 m |
| conics K1..K3 | 4.1834 / 16.824 / 0.08294 |
| YDE (mm) | +12.588 / -9.771 / -17.484 |
| ADE (deg) | -1.0362 / -0.4645 / +0.3160 |
| solve | s4: 123.6 -> 61.9 nm (qmean over solve set), 15 iters |
| **map max** | **80.8 nm** at XAN -5.0 YAN +17.0 |
| map avg / std / min | 59.3 / 8.4 / 41.5 nm |
| exit chief | -174.949° in Y-Z (report-only; no pin) |
| clearance floor | 2.8 mm (FAIL; gate >= 35 mm; WARN < 50 mm) |

Figures: `t4_s4_layout.png`, `t4_s4_map.png`.  Deck: `t4_s4.in`.

## S5 + Zernike departures (aspheres replaced)

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 10x10° box at YAN +12°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.500000 m = EPD 200 mm x F/2.5 |
| paraxial BFD | -1.2764 m |
| petzval c1-c2+c3 | -4.909e-03 1/m |
| plate scale | 145.44 um/arcmin |
| stop semi-diameter (traced) | 111.0 mm |
| radii R1..R3 | 2.08922 / -5.96615 / -1.53570 m |
| conics K1..K3 | 1.6017 / -65.266 / 0.084402 |
| YDE (mm) | +20.577 / -10.910 / -22.616 |
| ADE (deg) | -1.1111 / -0.5017 / +0.3143 |
| solve | s5: 14321.7 -> 31.1 nm (qmean over solve set), 15 iters |
| **map max** | **43.0 nm** at XAN +5.0 YAN +13.0 |
| map avg / std / min | 28.6 / 7.7 / 14.3 nm |
| exit chief | -174.818° in Y-Z (report-only; no pin) |
| clearance floor | 2.5 mm (FAIL; gate >= 35 mm; WARN < 50 mm) |

Figures: `t4_s5_layout.png`, `t4_s5_map.png`.  Deck: `t4_s5.in`.

## The ladder

| stage | map max (nm) | map avg | map std |
|---|---|---|---|
| s1 | 30.0 | 16.2 | 7.5 |
| s2 | 942.1 | 282.1 | 259.2 |
| s3 | 176.8 | 116.9 | 31.6 |
| s4 | 80.8 | 59.3 | 8.4 |
| s5 | 43.0 | 28.6 | 7.7 |
