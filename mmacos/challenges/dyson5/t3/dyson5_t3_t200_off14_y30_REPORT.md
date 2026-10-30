# dyson5_t3_t200_off14_y30 -- offset_imager run

2026-10-02 05:55:20.  EPD 70 mm, F/1.8 (EFL 0.126 m held as an identity), lambda 1.00 um, box 24.5553x0.3° offset +14°, spacings [-0.2 0 0.0379436] m, model 256, nGridpts 41.

Every WFE number below: strict RMS WFE, sphere centred on the spot centroid on the stage's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense-map MAXIMUM over the box.

## S1 coaxial, on-axis box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | 0.0001 m |
| petzval c1-c2+c3 | 0.000e+00 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 8.0 mm |
| radii R1..R3 | -0.57143 / 0.13562 / 0.10960 m |
| conics K1..K3 | 0 / 0 / 0 |
| solve | s1: 574393.6 -> 574393.6 nm (qmean over solve set), 1 iters |
| **map max** | **28957761.6 nm** at XAN -12.3 YAN +0.1 |
| map avg / std / min | 611171.7 / 3683832.2 / 4694.8 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -10.0 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_t200_off14_y30_s1_layout.png`, `dyson5_t3_t200_off14_y30_s1_map.png`.  Deck: `dyson5_t3_t200_off14_y30_s1.in`.
