# t5-unguided -- offset_imager run

2026-08-22 09:33:57.  EPD 150 mm, F/3.3 (EFL 0.495 m held as an identity), lambda 1.00 um, box 15x15° offset +22.5°, spacings [-1.44579 0 1.48166] m, model 256, nGridpts 41.

Every WFE number below: strict RMS WFE, sphere centred on the spot centroid on the stage's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense-map MAXIMUM over the box.

## S1 coaxial, on-axis box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 15x15° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.495000 m = EPD 150 mm x F/3.3 |
| paraxial BFD | -1.4939 m |
| petzval c1-c2+c3 | -1.110e-16 1/m |
| plate scale | 143.99 um/arcmin |
| stop semi-diameter (traced) | 90.3 mm |
| radii R1..R3 | 16.96744 / -2.07320 / -1.84747 m |
| conics K1..K3 | -401.14 / 5.2949 / 0.069068 |
| solve | s1: 21661.4 -> 16.6 nm (qmean over solve set), 23 iters |
| **map max** | **20.9 nm** at XAN +0.0 YAN +3.0 |
| map avg / std / min | 15.2 / 5.0 / 6.7 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -99.1 mm (FAIL; gate >= 25 mm; WARN < 40 mm) |

Figures: `t5_s1_layout.png`, `t5_s1_map.png`.  Deck: `t5_s1.in`.
