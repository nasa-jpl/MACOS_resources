# dyson5_t3_off06 -- offset_imager run

2026-10-02 04:28:22.  EPD 70 mm, F/1.8 (EFL 0.126 m held as an identity), lambda 1.00 um, box 24.5553x0.3° offset +6°, spacings [-0.14 0 0.076231] m, model 256, nGridpts 41.

Every WFE number below: strict RMS WFE, sphere centred on the spot centroid on the stage's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense-map MAXIMUM over the box.

## S1 coaxial, on-axis box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1786 m |
| petzval c1-c2+c3 | -1.776e-15 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 27.6 mm |
| radii R1..R3 | -1.20212 / -0.15062 / -0.17218 m |
| conics K1..K3 | -23.005 / 0.13893 / 0.070689 |
| solve | s1: 24181.9 -> 118.1 nm (qmean over solve set), 12 iters |
| **map max** | **171.6 nm** at XAN -9.8 YAN +0.0 |
| map avg / std / min | 134.6 / 22.7 / 113.2 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -53.9 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_off06_s1_layout.png`, `dyson5_t3_off06_s1_map.png`.  Deck: `dyson5_t3_off06_s1.in`.

## S2 offset box, FPA tilt/focus refit only

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +6°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1786 m |
| petzval c1-c2+c3 | -1.776e-15 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 27.5 mm |
| radii R1..R3 | -1.20212 / -0.15062 / -0.17218 m |
| conics K1..K3 | -23.005 / 0.13893 / 0.070689 |
| solve | s2: 6992.9 -> 282.6 nm (qmean over solve set), 3 iters |
| **map max** | **365.1 nm** at XAN +12.3 YAN +6.2 |
| map avg / std / min | 180.6 / 81.0 / 122.6 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -55.2 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_off06_s2_layout.png`, `dyson5_t3_off06_s2_map.png`.  Deck: `dyson5_t3_off06_s2.in`.

**The cost of the offset:** map max grows 2x (172 -> 365 nm) when the box moves 6° off axis with nothing but the FPA allowed to follow.

## S3 symmetric surfaces re-solved at the offset box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +6°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1921 m |
| petzval c1-c2+c3 | -8.882e-16 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 29.8 mm |
| radii R1..R3 | -1.66665 / -0.16176 / -0.17914 m |
| conics K1..K3 | -50.457 / 0.098839 / 0.056638 |
| solve | s3: 6992.9 -> 98.4 nm (qmean over solve set), 12 iters |
| **map max** | **173.1 nm** at XAN -9.8 YAN +6.2 |
| map avg / std / min | 123.6 / 25.6 / 86.1 nm |
| exit chief | 179.984° in Y-Z; err 0.016° vs pin -> PASS |
| clearance floor | -59.2 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_off06_s3_layout.png`, `dyson5_t3_off06_s3_map.png`.  Deck: `dyson5_t3_off06_s3.in`.

Conic migration under the bias doctrine (solve at the used field): K = [-23 0.1389 0.07069] -> [-50.46 0.09884 0.05664].

## S4 + mirror tilt/decenter + radii

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +6°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1902 m |
| petzval c1-c2+c3 | -3.539e-02 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 29.8 mm |
| radii R1..R3 | -1.60362 / -0.16158 / -0.17854 m |
| conics K1..K3 | -44.535 / 0.12978 / 0.058058 |
| YDE (mm) | -0.098 / -0.024 / -0.087 |
| ADE (deg) | +0.6229 / -0.0467 / +0.0154 |
| solve | s4: 98.4 -> 115.7 nm (qmean over solve set), 3 iters |
| **map max** | **241.8 nm** at XAN -9.8 YAN +5.8 |
| map avg / std / min | 175.0 / 53.5 / 97.3 nm |
| exit chief | 179.979° in Y-Z; err 0.021° vs pin -> PASS |
| clearance floor | -59.0 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_off06_s4_layout.png`, `dyson5_t3_off06_s4_map.png`.  Deck: `dyson5_t3_off06_s4.in`.
