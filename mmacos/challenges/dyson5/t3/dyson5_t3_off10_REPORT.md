# dyson5_t3_off10 -- offset_imager run

2026-10-02 04:28:22.  EPD 70 mm, F/1.8 (EFL 0.126 m held as an identity), lambda 1.00 um, box 24.5553x0.3° offset +10°, spacings [-0.14 0 0.076231] m, model 256, nGridpts 41.

Every WFE number below: strict RMS WFE, sphere centred on the spot centroid on the stage's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense-map MAXIMUM over the box.

## S1 coaxial, on-axis box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1786 m |
| petzval c1-c2+c3 | -1.776e-15 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 28.3 mm |
| radii R1..R3 | -1.20212 / -0.15062 / -0.17218 m |
| conics K1..K3 | -23.005 / 0.13893 / 0.070689 |
| solve | s1: 24181.9 -> 118.1 nm (qmean over solve set), 12 iters |
| **map max** | **171.6 nm** at XAN -9.8 YAN +0.0 |
| map avg / std / min | 134.6 / 22.7 / 113.2 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -53.9 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_off10_s1_layout.png`, `dyson5_t3_off10_s1_map.png`.  Deck: `dyson5_t3_off10_s1.in`.

## S2 offset box, FPA tilt/focus refit only

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +10°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1786 m |
| petzval c1-c2+c3 | -1.776e-15 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 28.2 mm |
| radii R1..R3 | -1.20212 / -0.15062 / -0.17218 m |
| conics K1..K3 | -23.005 / 0.13893 / 0.070689 |
| solve | s2: 17891.8 -> 915.1 nm (qmean over solve set), 3 iters |
| **map max** | **1206.7 nm** at XAN -12.3 YAN +10.2 |
| map avg / std / min | 427.0 / 371.3 / 86.8 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -51.6 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_off10_s2_layout.png`, `dyson5_t3_off10_s2_map.png`.  Deck: `dyson5_t3_off10_s2.in`.

**The cost of the offset:** map max grows 7x (172 -> 1207 nm) when the box moves 10° off axis with nothing but the FPA allowed to follow.

## S3 symmetric surfaces re-solved at the offset box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +10°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1917 m |
| petzval c1-c2+c3 | 8.882e-16 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 30.4 mm |
| radii R1..R3 | -1.64389 / -0.16134 / -0.17889 m |
| conics K1..K3 | -49.417 / 0.1295 / 0.056402 |
| solve | s3: 17891.8 -> 157.9 nm (qmean over solve set), 12 iters |
| **map max** | **216.1 nm** at XAN -7.4 YAN +10.2 |
| map avg / std / min | 184.2 / 16.4 / 154.3 nm |
| exit chief | 179.950° in Y-Z; err 0.050° vs pin -> PASS |
| clearance floor | -58.5 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_off10_s3_layout.png`, `dyson5_t3_off10_s3_map.png`.  Deck: `dyson5_t3_off10_s3.in`.

Conic migration under the bias doctrine (solve at the used field): K = [-23 0.1389 0.07069] -> [-49.42 0.1295 0.0564].

## S4 + mirror tilt/decenter + radii

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +10°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1891 m |
| petzval c1-c2+c3 | -3.271e-02 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 30.2 mm |
| radii R1..R3 | -1.55513 / -0.16053 / -0.17796 m |
| conics K1..K3 | -41.357 / 0.16482 / 0.05817 |
| YDE (mm) | -0.570 / +0.210 / -0.269 |
| ADE (deg) | +0.4467 / -0.0513 / +0.0867 |
| solve | s4: 157.9 -> 163.5 nm (qmean over solve set), 3 iters |
| **map max** | **230.6 nm** at XAN -7.4 YAN +9.8 |
| map avg / std / min | 195.1 / 23.4 / 161.7 nm |
| exit chief | -179.945° in Y-Z; err 0.055° vs pin -> PASS |
| clearance floor | -55.0 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_off10_s4_layout.png`, `dyson5_t3_off10_s4_map.png`.  Deck: `dyson5_t3_off10_s4.in`.
