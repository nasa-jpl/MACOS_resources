# rodgers3-T3 -- offset_imager run

2026-08-19 17:49:00.  EPD 75 mm, F/4 (EFL 0.300 m held as an identity), lambda 1.00 um, box 20x20° offset +22°, spacings [-0.05794 0 0.682888] m, model 256, nGridpts 41.

Every WFE number below: strict RMS WFE, sphere centred on the spot centroid on the stage's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense-map MAXIMUM over the box.

## S1 coaxial, on-axis box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 20x20° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.300000 m = EPD 75 mm x F/4 |
| paraxial BFD | -0.8377 m |
| petzval c1-c2+c3 | 2.237e-01 1/m |
| plate scale | 87.27 um/arcmin |
| stop semi-diameter (traced) | 40.5 mm |
| radii R1..R3 | 8.79489 / -0.85220 / -0.94038 m |
| conics K1..K3 | -1.9811e-06 / -0.00022646 / 0.056901 |
| solve | s1: 16203.8 -> 1802.2 nm (qmean over solve set), 5 iters |
| **map max** | **2425.9 nm** at XAN +10.0 YAN +10.0 |
| map avg / std / min | 1099.4 / 452.4 / 622.6 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | 3.2 mm (FAIL; gate >= 35 mm; WARN < 50 mm) |

Figures: `r3t_s1_layout.png`, `r3t_s1_map.png`.  Deck: `r3t_s1.in`.
