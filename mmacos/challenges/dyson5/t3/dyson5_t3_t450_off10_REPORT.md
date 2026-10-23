# dyson5_t3_t450_off10 -- offset_imager run

2026-10-02 05:41:09.  EPD 70 mm, F/1.8 (EFL 0.126 m held as an identity), lambda 1.00 um, box 24.5553x0.3° offset +10°, spacings [-0.45 0 0.076265] m, model 256, nGridpts 41.

Every WFE number below: strict RMS WFE, sphere centred on the spot centroid on the stage's frozen FPA, anchored at the exit pupil, piston-only removal (design/src strict kernel); headline = dense-map MAXIMUM over the box.

## S1 coaxial, on-axis box

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +0°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1165 m |
| petzval c1-c2+c3 | 8.882e-16 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 15.1 mm |
| radii R1..R3 | -1.61510 / -0.12110 / -0.13090 m |
| conics K1..K3 | -0.49442 / -0.09472 / 0.012395 |
| solve | s1: 28310.0 -> 4364.4 nm (qmean over solve set), 9 iters |
| **map max** | **10594.6 nm** at XAN +7.4 YAN -0.1 |
| map avg / std / min | 6911.3 / 3046.8 / 334.5 nm |
| exit chief | 180.000° in Y-Z; err 0.000° vs pin -> PASS |
| clearance floor | -34.3 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_t450_off10_s1_layout.png`, `dyson5_t3_t450_off10_s1_map.png`.  Deck: `dyson5_t3_t450_off10_s1.in`.

## S2 offset box, FPA tilt/focus refit only

Metric: strict RMS WFE, centroid reference on the frozen stage FPA, exit-pupil anchor, piston-only removal; dense 11x11 map over the 24.5553x0.3° box at YAN +10°; solve set 3x3 (solve set != scoring set).

| quantity | value |
|---|---|
| EFL (identity) | 0.126000 m = EPD 70 mm x F/1.8 |
| paraxial BFD | -0.1165 m |
| petzval c1-c2+c3 | 8.882e-16 1/m |
| plate scale | 36.65 um/arcmin |
| stop semi-diameter (traced) | 233.0 mm |
| radii R1..R3 | -1.61510 / -0.12110 / -0.13090 m |
| conics K1..K3 | -0.49442 / -0.09472 / 0.012395 |
| solve | s2: 1000000000.0 -> 1000000000.0 nm (qmean over solve set), 1 iters |
| **map max** | **INVALID -- 121/121 fields lost every ray** (finite-only max NaN nm) |
| map avg / std / min | NaN / NaN / NaN nm |
| exit chief | NaN° in Y-Z; err NaN° vs pin -> FAIL |
| clearance floor | 0.0 mm (FAIL; gate >= 5 mm; WARN < 5 mm) |

Figures: `dyson5_t3_t450_off10_s2_layout.png`, `dyson5_t3_t450_off10_s2_map.png`.  Deck: `dyson5_t3_t450_off10_s2.in`.

**The cost of the offset:** map max grows NaNx (10595 -> NaN nm) when the box moves 10° off axis with nothing but the FPA allowed to follow.
