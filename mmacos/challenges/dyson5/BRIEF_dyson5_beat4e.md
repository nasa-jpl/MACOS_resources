# dyson5 beat 4e -- the closure envelope (addendum 11): for which parameters does R4 close?

Written for: Dave (review; the sentence for the run-it-yourself slide).
Record: `dyson5_s4env.txt`, `dyson5_s4env.mat`, `dyson5_s4env.png`, one deck
per point `dyson5_s4env_<axis>_<value>.in`.  Runner:
`dyson5_run(struct('stages', {{'s4env'}}))`; tool `dyson5_envelope.m`.

## 1. Method

From the R4 of record, one axis at a time (addendum 11's table), every
point a FULL R4 solve -- all eleven variables (R_g factor, face offset,
block conic and h^4/h^6, block centre dy/dz, meniscus vertex, thickness and
two curvatures), lsqnonlin on the exact chain, 30 iterations, warm-started
from the record -- then the deck emitted with apertures and scored in the
ENGINE on the 7 x 7 grid.  A point CLOSES when smile and keystone < 0.1 px,
CRF < 1.5 px, SRF < 2.0 px AND no variable sits on a bound (a solve on its
bounds is not a closed design; the bound-hitting variables are named).
The FPA stays 54 x 9 mm: the pixel count follows the slit length and the
pixel pitch.  The meniscus vertex's lower bound follows the block radius
(20 mm beyond it; 0.24 m at the record's 220 mm, so the record is
unchanged).  Then the two-axis corner: the first failing value of the first
two failing axes, together.

| axis | points |
|---|---|
| F-number | 1.6, 1.8, 2.0, 2.2, 2.8 |
| block radius | 150, 180, 220, 260, 300 mm |
| slit length | 30, 40, 54, 60 mm (pixel count follows) |
| pixel | 18, 30 um (the paper's; pixel count follows) |
| glass | Silica, CaF2 |

## 2. Result (dyson5_s4env.txt; engine scores on the 7 x 7 grid, R4 re-solved per point)

| axis | value | smile | keystone | CRF | SRF | EE | closes | why not |
|---|---|---|---|---|---|---|---|---|
| F-number | 1.6 | 0.0049 | 0.0039 | 1.370 | 2.033 | 0.681 | no | on a bound (meniscus thickness) |
| F-number | 1.8 (record) | 0.0051 | 0.0026 | 1.327 | 2.032 | 0.759 | yes | |
| F-number | 2.0 | 0.0047 | 0.0018 | 1.210 | 2.031 | 0.916 | yes | |
| F-number | 2.2 | 0.0046 | 0.0009 | 1.148 | 2.033 | 0.954 | yes | |
| F-number | 2.8 | 0.0032 | 0.0017 | 1.069 | 2.038 | 0.952 | no | on bounds (meniscus vertex, thickness) |
| block radius | 150 mm | 0.0190 | 0.0112 | 2.371 | 2.032 | 0.408 | no | CRF |
| block radius | 180 mm | 0.0125 | 0.0103 | 1.607 | 2.032 | 0.606 | no | CRF |
| block radius | 220 mm (record) | 0.0051 | 0.0026 | 1.327 | 2.032 | 0.759 | yes | |
| block radius | 260 mm | 0.0043 | 0.0038 | 1.211 | 2.033 | 0.685 | yes | |
| block radius | 300 mm | 0.0073 | 0.0177 | 1.249 | 2.032 | 0.622 | yes | |
| slit length | 30 mm | 0.0017 | 0.0054 | 1.047 | 2.025 | 0.955 | yes | |
| slit length | 40 mm | 0.0028 | 0.0009 | 1.089 | 2.027 | 0.901 | yes | |
| slit length | 54 mm (record) | 0.0051 | 0.0026 | 1.327 | 2.032 | 0.759 | yes | |
| slit length | 60 mm | 0.0060 | 0.0052 | 1.548 | 2.034 | 0.608 | no | CRF (by 0.05 px) |
| pixel | 18 um (record) | 0.0051 | 0.0026 | 1.327 | 2.032 | 0.759 | yes | |
| pixel | 30 um (the paper's) | 0.0030 | 0.0016 | 1.030 | 2.016 | 1.000 | yes | |
| glass | Silica (record) | 0.0051 | 0.0026 | 1.327 | 2.032 | 0.759 | yes | |
| glass | CaF2 | 0.0040 | 0.0050 | 1.032 | 2.024 | 1.000 | yes | |

**The sentence for the slide:** designs close for F/1.8-F/2.2, block
radii 220-300 mm, slits to 54 mm, 18 and 30 um pixels, silica and CaF2;
the first metric to fail outside is the CRF -- at 150 / 180 mm radii
(2.37 / 1.61 px: the h^4/r^3 law of s0) and at the 60 mm slit (1.548 px,
by 0.05).  F/1.6 and F/2.8 are not called closed because their solves
end on the meniscus bounds (thickness 4 mm at F/1.6; vertex and thickness
at F/2.8) with the CRF itself at 1.37 / 1.07 px -- the meniscus wants to
be thinner / elsewhere than the ladder's bounds allow, a bound to widen,
not a form that fails.  Smile and keystone never exceed 0.02 px on any
point (the ladder's operands hold them 5-50x under spec everywhere).

The corner (the first failing value of the first two failing axes: F/1.6
with r = 150 mm) is in the record's text; it does not close (CRF).
