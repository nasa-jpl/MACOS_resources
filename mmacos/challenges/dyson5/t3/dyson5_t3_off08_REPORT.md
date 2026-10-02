# dyson5_t3_off08 -- offset_imager run

2026-10-02 04:28:22.  EPD 70 mm, F/1.8 (EFL 0.126 m held as an identity), lambda 1.00 um, box 24.5553x0.3° offset +8°, spacings [-0.14 0 0.076231] m, model 256, nGridpts 41.

Every WFE number below: strict RMS WFE, sphere centred on the spot centroid on the stage's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense-map MAXIMUM over the box.

## S1 coaxial, on-axis box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1786 m |
| petzval c1-c2+c3 | -1.776e-15 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 27.9 mm |
| radii R1..R3 | -1.20212 / -0.15062 / -0.17218 m |
| conics K1..K3 | -23.005 / 0.13893 / 0.070689 |
| solve | s1: 24181.9 -> 118.1 nm (qmean over solve set), 12 iters |
| **map max** | **171.6 nm** at XAN -9.8 YAN +0.0 |
| map avg / std / min | 134.6 / 22.7 / 113.2 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -53.9 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_off08_s1_layout.png`, `dyson5_t3_off08_s1_map.png`.  Deck: `dyson5_t3_off08_s1.in`.

## S2 offset box, FPA tilt/focus refit only

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +8°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1786 m |
| petzval c1-c2+c3 | -1.776e-15 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 27.8 mm |
| radii R1..R3 | -1.20212 / -0.15062 / -0.17218 m |
| conics K1..K3 | -23.005 / 0.13893 / 0.070689 |
| solve | s2: 11710.3 -> 514.8 nm (qmean over solve set), 3 iters |
| **map max** | **676.8 nm** at XAN +12.3 YAN +8.2 |
| map avg / std / min | 253.8 / 189.8 / 123.4 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -55.8 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_off08_s2_layout.png`, `dyson5_t3_off08_s2_map.png`.  Deck: `dyson5_t3_off08_s2.in`.

**The cost of the offset:** map max grows 4x (172 -> 677 nm) when the box moves 8° off axis with nothing but the FPA allowed to follow.

## S3 symmetric surfaces re-solved at the offset box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +8°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1902 m |
| petzval c1-c2+c3 | -8.882e-16 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 29.8 mm |
| radii R1..R3 | -1.57733 / -0.16007 / -0.17814 m |
| conics K1..K3 | -44.695 / 0.12889 / 0.058016 |
| solve | s3: 11710.3 -> 125.9 nm (qmean over solve set), 12 iters |
| **map max** | **195.8 nm** at XAN +7.4 YAN +8.2 |
| map avg / std / min | 153.2 / 22.6 / 122.7 nm |
| exit chief | 179.983° in Y-Z; err 0.017° vs pin -> PASS |
| clearance floor | -59.2 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_off08_s3_layout.png`, `dyson5_t3_off08_s3_map.png`.  Deck: `dyson5_t3_off08_s3.in`.

Conic migration under the bias doctrine (solve at the used field): K = [-23 0.1389 0.07069] -> [-44.69 0.1289 0.05802].

## S4 + mirror tilt/decenter + radii

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +8°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1874 m |
| petzval c1-c2+c3 | -3.399e-02 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 29.5 mm |
| radii R1..R3 | -1.48560 / -0.15908 / -0.17707 m |
| conics K1..K3 | -37.567 / 0.16907 / 0.060181 |
| YDE (mm) | -0.500 / +0.161 / -0.198 |
| ADE (deg) | +0.4175 / -0.0500 / +0.0614 |
| solve | s4: 125.9 -> 129.5 nm (qmean over solve set), 3 iters |
| **map max** | **227.5 nm** at XAN -7.4 YAN +7.8 |
| map avg / std / min | 177.0 / 39.5 / 122.9 nm |
| exit chief | -179.984° in Y-Z; err 0.016° vs pin -> PASS |
| clearance floor | -58.7 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_off08_s4_layout.png`, `dyson5_t3_off08_s4_map.png`.  Deck: `dyson5_t3_off08_s4.in`.
