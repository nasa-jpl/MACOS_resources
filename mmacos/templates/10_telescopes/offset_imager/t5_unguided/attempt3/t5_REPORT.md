# t5-unguided -- offset_imager run

2026-08-22 10:20:42.  EPD 150 mm, F/3.3 (EFL 0.495 m held as an identity), lambda 1.00 um, box 15x15° offset +22.5°, spacings [-1.19278 0 1.22237] m, model 256, nGridpts 41.

Every WFE number below: strict RMS WFE, sphere centred on the spot centroid on the stage's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense-map MAXIMUM over the box.

## S1 coaxial, on-axis box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 15x15° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.495000 m = EPD 150 mm x F/3.3 |
| paraxial BFD | -1.4053 m |
| petzval c1-c2+c3 | 0.000e+00 1/m |
| plate scale | 143.99 um/arcmin |
| stop semi-diameter (traced) | 92.6 mm |
| radii R1..R3 | 14.37842 / -1.89643 / -1.67545 m |
| conics K1..K3 | -37.191 / 4.3012 / 0.082681 |
| solve | s1: 19889.7 -> 23.1 nm (qmean over solve set), 6 iters |
| **map max** | **30.6 nm** at XAN -1.5 YAN +1.5 |
| map avg / std / min | 21.2 / 7.7 / 8.4 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -97.2 mm (FAIL; gate >= 25 mm; WARN < 40 mm) |

Figures: `t5_s1_layout.png`, `t5_s1_map.png`.  Deck: `t5_s1.in`.
