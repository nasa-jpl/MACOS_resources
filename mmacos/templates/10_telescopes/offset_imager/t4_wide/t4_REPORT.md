# t4-wide -- offset_imager run

2026-08-19 18:16:34.  EPD 200 mm, F/2.5 (EFL 0.500 m held as an identity), lambda 1.00 um, box 10x10° offset +12°, spacings [-0.1 0 1.1] m, model 256, nGridpts 41.

Every WFE number below: strict RMS WFE, sphere centred on the spot centroid on the stage's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense-map MAXIMUM over the box.

## S1 coaxial, on-axis box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 10x10° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.500000 m = EPD 200 mm x F/2.5 |
| paraxial BFD | -1.2812 m |
| petzval c1-c2+c3 | -1.110e-16 1/m |
| plate scale | 145.44 um/arcmin |
| stop semi-diameter (traced) | 112.5 mm |
| radii R1..R3 | 2.01889 / -6.47170 / -1.53884 m |
| conics K1..K3 | 6.6709 / 10.199 / 0.080329 |
| solve | s1: 22345.3 -> 18.1 nm (qmean over solve set), 14 iters |
| **map max** | **30.1 nm** at XAN +0.0 YAN +0.0 |
| map avg / std / min | 16.2 / 7.5 / 6.2 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
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
| stop semi-diameter (traced) | 112.5 mm |
| radii R1..R3 | 2.01889 / -6.47170 / -1.53884 m |
| conics K1..K3 | 6.6709 / 10.199 / 0.080329 |
| solve | s2: 60609.6 -> 501.6 nm (qmean over solve set), 3 iters |
| **map max** | **941.1 nm** at XAN -5.0 YAN +17.0 |
| map avg / std / min | 281.8 / 258.9 / 32.9 nm |
| exit chief | -175.258° in Y-Z; err 4.742° vs pin -> FAIL |
| clearance floor | 2.0 mm (FAIL; gate >= 35 mm; WARN < 50 mm) |

Figures: `t4_s2_layout.png`, `t4_s2_map.png`.  Deck: `t4_s2.in`.

**The cost of the offset:** map max grows 31x (30 -> 941 nm) when the box moves 12° off axis with nothing but the FPA allowed to follow.

## S3 symmetric surfaces re-solved at the offset box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 10x10° box at YAN +12°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.500000 m = EPD 200 mm x F/2.5 |
| paraxial BFD | -1.2813 m |
| petzval c1-c2+c3 | 0.000e+00 1/m |
| plate scale | 145.44 um/arcmin |
| stop semi-diameter (traced) | 112.4 mm |
| radii R1..R3 | 2.03809 / -6.27582 / -1.53847 m |
| conics K1..K3 | 5.924 / 21.606 / 0.0808 |
| solve | s3: 501.6 -> 127.5 nm (qmean over solve set), 3 iters |
| **map max** | **180.3 nm** at XAN +0.0 YAN +12.0 |
| map avg / std / min | 121.0 / 30.4 / 58.0 nm |
| exit chief | -175.250° in Y-Z; err 4.750° vs pin -> FAIL |
| clearance floor | 2.0 mm (FAIL; gate >= 35 mm; WARN < 50 mm) |

Figures: `t4_s3_layout.png`, `t4_s3_map.png`.  Deck: `t4_s3.in`.

Conic migration under the bias doctrine (solve at the used field): K = [6.671 10.2 0.08033] -> [5.924 21.61 0.0808].

## S4 + mirror tilt/decenter + radii

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 10x10° box at YAN +12°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.500000 m = EPD 200 mm x F/2.5 |
| paraxial BFD | -1.2813 m |
| petzval c1-c2+c3 | 1.110e-16 1/m |
| plate scale | 145.44 um/arcmin |
| stop semi-diameter (traced) | 112.4 mm |
| radii R1..R3 | 2.03809 / -6.27582 / -1.53847 m |
| conics K1..K3 | 5.924 / 21.606 / 0.0808 |
| solve | s4: 1000000000.0 -> 1000000000.0 nm (qmean over solve set), 1 iters |
| **map max** | **180.3 nm** at XAN +0.0 YAN +12.0 |
| map avg / std / min | 121.0 / 30.4 / 58.0 nm |
| exit chief | -175.250° in Y-Z; err 4.750° vs pin -> FAIL |
| clearance floor | 2.0 mm (FAIL; gate >= 35 mm; WARN < 50 mm) |

Figures: `t4_s4_layout.png`, `t4_s4_map.png`.  Deck: `t4_s4.in`.
