# tma_longslit — the long-slit TMA front end for a slit-fed spectrometer

A parameterized design flow for a three-mirror telescope that feeds a
**long slit**:
- a wide strip of sky imaged onto a straight slit;
- **telescentric** at the slit, so the spectrometer behind it sees every field the same way;
- with a bounded **cone**: no marginal ray faster than the spectrometer accepts, the F/# the same in both directions;
- and a long working distance behind the last mirror, so the spectrometer fits in it.

The form is the **SBG VSWIR zig-zag TMA** (Bradley et al., *Finalized optical
design of the SBG VSWIR Wide Swath Imaging Spectrometer*, ICSO 2024, Proc. SPIE
13699, 1369945, Fig. 4b):
- a weak concave M1;
- a small convex M2 that is the **stop**;
- M3 carrying nearly all of the power;
- the slit 300 mm behind M3.

The beam turns the same way at M1 and M3 and back at M2. It is a zig-zag, not a
Korsch ring, and it has **no intermediate focus**.

The default run is the dyson5 3k front end at Joe's spec: f 330 mm, D 183 mm,
F/1.8, a ±4.7° strip onto a 54 mm slit.

## Run it yourself

```matlab
run('<path-to>/mmacos/mmacos_setup.m');
addpath('<path-to>/mmacos/templates/10_telescopes/tma_longslit');
OUT = tma_longslit_run({'first_order','section'});       % first order + the engine section (seconds)
OUT = tma_longslit_run('figure');                        % the figure ladder (15-80 min a rung)
OUT = tma_longslit_run('e2e');                           % the best clear rung joined to the dyson5 3k Dyson
OUT = tma_longslit();                                    % all four, one call
OUT = tma_longslit_run({'first_order','section'}, ...    % YOUR instrument: any field of tma_longslit_params
        struct('f_m',0.250,'D_m',0.125,'strip_half_deg',3.0,'slit_m',0.026,'tag','mine'));
```

All parameters live in `tma_longslit_params.m`, the single source of truth.
Every stage reads only that struct, prints its table, writes
`<tag>_<stage>.txt` and saves `<tag>_<stage>.mat`.  Decks:
`<tag>_section.in`, `<tag>_R<n>_<rung>.in`.

## The two facts the first order forces

1. **A tilted sphere cannot do it.**  The legs are the same in the fold plane
   and across it, so both sections need the same power per mirror.  A mirror
   tilted by i supplies 2/(R_t cos i) in the fold plane and 2 cos i/R_s across
   it.  So each mirror must carry local radii at the chief with
   **R_t/R_s = 1/cos² i** (1.33 / 1.57 / 1.07 at the default 30 / 37 / 15°).
   An **off-axis conic section** does this natively.  With the parent axis at
   the AOI to the local normal, all three are off-axis paraboloids.  That is the
   seed (`tls_section`); the figure stage decides the conics.
2. **The stop at M2 makes telecentricity closed-form**: M2 sits at M3's front
   focus.  With the default legs the marginal ray never crosses the axis
   between mirrors, so there is no real intermediate image.  A Korsch with a
   real M2–M3 focus cannot be telecentric with the stop at M2 (dyson5
   addendum 37); this family does not have that focus.

## Stages (`tma_longslit_run`)

| stage | what | engine |
|---|---|---|
| `first_order` | the paraxial solve for the stop at `P.stop`, with M1 and halfway-to-M2 stops alongside: powers, the R_t/R_s each section must carry, intermediate focus, pupils, footprints | no |
| `section` | the 3-D section: three off-axis conics on the chief, the stop an ELEMENT stop on M2 (`ApStop= dx dy` in M2's block), emitted and measured per strip field | yes |
| `figure` | the ladder `P.ladder`, each rung an engine-traced Levenberg–Marquardt solve warm from the rung its `.from` names; `P.resume_upto` reuses recorded rungs | yes |
| `e2e` | the best rung that clears with every ray inside the cone bound, joined to the spectrometer of `P.e2e` (default the dyson5 3k Dyson of record) through `challenges/dyson5/dyson5_t5f`, both rolls, plus the joined-deck clearance | yes |

**The first order is re-derived, never penalized.**  R_t and R_s are not
DOFs: every iterate re-solves them from the first order of the current legs
and AOIs, so EFL, back focus and telecentricity hold exactly.  Two solves with
them free traded the first order for spots: F/2.37 across the slit, then the
slit 117 mm in and the chiefs 2.2° off.

The figure solve (`tls_figure`) varies the **generator**, not the deck text:
- per mirror, the local radii and the off-axis angle (the conic follows);
- h⁴/h⁶ as sag at the lit radius;
- the focus;
- in the geom rung, the AOIs and legs.

The section is rebuilt from those every evaluation.  Conic and asphere are
solved together so that the chief always runs through the three poles and the
stop always sits exactly at the M2 pole.  Rows per solve field:
- SPOT: every ray, about the as-placed centroid;
- PLATE: the chief vs f·tan θ;
- BOW: the chief's across-slit position, the strict intercept;
- TELE: the chief angle to the slit normal;
- CONE: a hinge outside `cone_fnum`;
- WD: the working distance.

**The cone rows are a per-axis wall on the extreme ray angles at the slit.**
The engine's CALIB `OptBeamSize=` row cannot serve: it is twice the largest
ray distance from the bundle centroid at one element (`utilsub.F`
`GetBeamSizeCmd`), a single isotropic footprint extent.  At the slit it
measures the spot, not the cone.

## What the ladder taught (all measured; `macos/REPORT_dyson5_cprime.md`)

- **Even aspheres are the wrong basis on these sections.**  The poles sit
  0.9–1.5 m from their parent axes, so h⁴ about the parent axis is almost all
  pole tilt and curvature, which the closure absorbs.  The asphere rung moves
  the cost 3 %.
- **The freeform is a pole-frame polynomial** (`Surface= Monomial`, degree
  3–6, even powers of x).  Degree ≥ 3 adds nothing to value, slope or
  curvature at the pole, so the first order is untouched; a Zernike departure
  would leak tilt (coma) and curvature (spherical).  This is the rung that
  images.
- **The layout stays the form.**  Unbounded AOIs collapse to ~2°, a coaxial
  train that self-obscures by 70 mm.  Bounded ±5°, they run to the bounds and
  cost the clearance for little.  The freeform rungs keep the seed layout.
- **Smile reads the CENTROID bow.**  Coma moves a field's centroid off its
  chief: 2.7 µm of chief bow came with 9.1 µm of centroid bow, and the e2e
  smile read 0.73 px.  The merit carries both rows.

## The join's own diagnostic: `tls_dyson_chief_tilt`

`T = tls_dyson_chief_tilt(P, M)` scores the spectrometer ALONE (the record's
own path: `spectrometer_geom` → `spectrometer_rx` → `spectrometer_score`).
Every slit point's input chief is tilted by a telescope's measured chief
angle at the slit (`M.chief_x_mrad` / `chief_y_mrad` from `tls_measure`),
imposed by wrapping the chain's `G.aim`; nothing in the library changes.
The legs are base / along / cross / along×2 / both.  It attributes an e2e
spectral error to the telescope's telecentric error, or rules that out,
without a solve.

On R7 it ruled it out.  With the base leg equal to the record (smile
0.0050, CRF 1.213, SRF 2.024), the Dyson's smile stays at 0.005 px even at
twice R7's along-track pattern.

**The e2e smile is the telescope's chief-minus-centroid offset across the
slit.**  t5f lands each field's CHIEF on the slit line and scores the
CENTROID, so the field variation of (chief − centroid) across the slit (the
coma's across-slit part) reads as smile.  Predicted vs e2e:

| rung | predicted | e2e |
|---|---|---|
| R4 | 0.655 px | 0.732 |
| R5 | 0.086 | 0.090 |
| R6 | 0.122 | 0.149 |
| R7 | 0.135 | 0.140 |
| R8 | 0.133 | 0.138 |

## Two smile conventions (CC 2026-10-07; Dave has the question)

The e2e join (`dyson5_t5f`) launches one sky direction per field and scores
each field's spot centroid at the FPA.  Which point of the field it puts on
the slit line decides what "smile" measures:

- **`'centroid'` launch — smile (slit-filled), the spec's convention.**  Each
  field's bundle centroid lands on the slit line.  An extended scene fills
  the slit whatever the telescope's chief does, so this is the smile a
  straight, uniformly lit slit shows.  It is the paper's convention too: its
  telescope-fed smile equals its DSI-alone smile (1.3 % both, Tables 2–3).
- **`'chief'` launch — the point-source across-slit shift.**  Each field's
  chief lands on the slit line, and the centroid is scored.  The telescope's
  chief-minus-centroid offset across the slit (the across-slit part of its
  coma) then reads as smile.  It is the record's convention until
  2026-10-07, and it is stated beside the smile, not scored against it.

`tma_longslit_run('e2e')` runs both and tables the pair.

Two more conventions, both from the paper and both easy to get wrong:
- **The paper's SRF is in co-added pixels.**  Its 1.8 is 64.8 µm, since a
  co-added pixel is 2 × 18 µm.  Its smile bound, "5 % of a co-added pixel",
  is 1.8 µm = 0.10 of OUR pixel.  Every table here therefore carries the
  paper column in micrometres.
- **SRF has a floor set by the slit.**  rect(2 px) ⊗ rect(1 px) ⊗ Airy is
  2.01–2.02 px for a perfect spectrometer, so an SRF of 2.0x px is the slit,
  not the optics.
`P.tel5f_launch` (dyson5) selects one; the default stays `'chief'`, so
every existing record reproduces.

## Gotchas

- **Element stop and `macos.stop`.**  Until 2026-10-07 the api's
  `stop_info_set` refused a stop element ≥ nElt−2, i.e. M2 of any
  four-element telescope.  CC fixed that the same day, but `tls_measure` and
  `tls_figure` do not depend on it: each field is a copy of the deck with its
  `ChfRayDir`, and the deck's element stop is applied at load.
- **MATLAB `regexprep` and `.`**: `.` matches newlines unless you pass
  `'dotexceptnewline'`.  Without it, the `ChfRayDir` substitution ate the
  whole deck.
- **Asphere terms on a far off-axis section tilt the local normal.**  M1's
  pole is ~880 mm from its parent axis.  A naive "move the vertex so the pole
  stays on the surface" fix lets an h⁴ term bend the chief off the layout.
  `tls_section` solves the parent (R, K, h) so that conic plus asphere give the
  designed normal and local radii at the pole.  This is gated in
  `tTmaLongslit`.

## Results (the dyson5 3k default)

**Deck of record: R9 `ffo`** (`tls_R9_ffo.in`, recorded as
`challenges/dyson5/dyson5_cprime_3k.in`).  R9 = R7 + the OFF rows ×1000:
the chief − centroid offset across the slit, the point-source shift.  It
holds R7's pixel floor (FWHM 1.02 px, EiP 1.00 at every field, cone and
clearance unchanged) and cuts the edge offset 2.43 → 0.03 µm.

End to end, roll 0 / 180:

Pixels are our 18 µm detector pixels.  The paper's bounds (Bradley 2024
Table 1) are in micrometres, in its own units: its SRF is in **co-added**
pixels of 36 µm.

| | R9, px | R9, µm | R7, px | Joe (px) | paper (Table 1, µm) |
|---|---|---|---|---|---|
| smile (slit-filled) | **0.022 / 0.024** | 0.40 / 0.43 | 0.010 | < 0.1 | < 1.8 (5 % of a co-added px) |
| point-source across-slit shift | **0.016 / 0.018** | 0.29 / 0.32 | 0.140 | — | — |
| keystone | 0.006 / 0.009 | 0.11 / 0.16 | 0.006 | < 0.1 | < 1.8 (10 % of a px) |
| CRF | 1.174 / 1.262 | 21.1 / 22.7 | 1.176 | < 1.5 | < 50.4 (2.8 px) |
| SRF | 2.025 / 2.025 | 36.5 | 2.025 | 1.5–2.0 | < 64.8 (1.8 co-added px) |
| ARF (telescope, across slit) | 1.02 | 18.4 | 1.02 | — | < 50.4 (2.8 px) |
| EiP | 0.849 / 0.805 | | 0.853 | > 0.75 | |
| joined clearance | +0.6 mm | | +0.6 | > 0 | |

**The 3k module meets Joe's spec end to end, and the paper's five with
margin, under both smile conventions.**

**SRF 2.025 px is the slit FLOOR, not the spectrometer.**  The scorer's SRF
is rect(2-px slit) ⊗ LSF ⊗ rect(1 px) ⊗ Airy.  For a perfect spectrometer
that is 2.010 px at 0.38 µm and 2.023 px at 2.5 µm (the scorer's 0.01-px
LSF bins plus diffraction).  R9's SRF sits on that floor at every
wavelength, at most 0.0015 px (0.03 µm) above it.  So Joe's "1.5–2.0" is
met at its floor: 1.5 would take a 1.5-px slit, not a better spectrometer.
And the paper's 1.8 is 1.8 CO-ADDED pixels, 64.8 µm, which our 36.5 µm
passes with margin.

The R7 tables below are the telescope that R9 refined; per-field, R9
matches them to the digits shown except the bow columns.

**R7 `ffw`** (`tls_R7_ffw.in`).  It runs R4's warm start through the degree 3–6 pole-frame
freeform, with the along-slit spot rows ×3 and the outer solve fields
×1/1/2/3/3.  Engine numbers, model 256.

**Telescope alone, per field** (mirror-symmetric; ±):

| field | rms µm | FWHM along / across slit px | EiP | chief to slit normal | F/# along / across | chief bow / centroid bow µm | x vs f tan θ µm |
|---|---|---|---|---|---|---|---|
| 0 | 1.77 | 1.02 / 1.02 | 1.00 | 0.000° | 1.894 / 1.806 | 0 / 0 | 0 |
| 1.175° | 2.15 | 1.02 / 1.02 | 1.00 | 0.002° | 1.894 / 1.803 | 0.1 / 0.06 | 1.2 |
| 2.35° | 2.51 | 1.02 / 1.02 | 1.00 | 0.008° | 1.893 / 1.801 | 0.5 / 0.16 | 9.9 |
| 3.525° | 1.96 | 1.02 / 1.02 | 1.00 | 0.018° | 1.892 / 1.798 | 1.4 / 0.04 | 33.5 |
| 4.7° | 2.25 | 1.02 / 1.02 | 1.00 | 0.033° | 1.889 / 1.797 | 3.2 / 0.75 | 79.4 |

- Plate scale 329.94 mm local, along-track 330.07 mm.
- Footprints M1 234.6 × 201.7, M2 131.7 × 165.9, M3 215.3 × 172.9 mm.
- Clearance +12.2 mm; working distance 299.4 mm.
- Freeform departure P-V 0.46 / 0.55 / 1.11 mm.

**End to end**, joined to the 3k Dyson of record (CaF2 240,
`size:F:240`), roll 0 / 180:

| | R7, px (µm) | Dyson alone, px | 3k (c) record, px | Joe (px) | paper (Table 1, µm) |
|---|---|---|---|---|---|
| smile, point-source (chief launch) | 0.140 / 0.141 (2.5) | 0.005 | 1.64 | < 0.1 | < 1.8 |
| keystone | 0.006 / 0.009 (0.11) | 0.006 | 0.05 | < 0.1 | < 1.8 |
| CRF | 1.175 / 1.255 (21.2) | 1.21 | 4.02 | < 1.5 | < 50.4 |
| SRF | 2.025 / 2.025 (36.5) | 2.024 | 4.99 | 1.5–2.0 | < 64.8 |
| ARF (telescope, across slit) | 1.02 (18.4) | — | — | — | < 50.4 |
| energy in a pixel | 0.853 / 0.810 | 0.82 | — | > 0.75 | — |
| grating admits | 0.988 / 0.989 | | | | |
| joined clearance | +0.60 / +0.61 mm | +0.54 | +0.06 | > 0 | |

- **CRF and SRF are the Dyson's own floors:** on this spectrometer the
  telescope is no longer the limit (SRF: the slit floor, above).
- **R7's point-source smile, 0.14 px, is closed in R9** (0.016 px).  It was
  not the telescope's slit-plane centroid bow (≤ 0.75 µm = 0.04 px).
- **The cone along the slit, F/1.89, is Joe's spec:** D 183 mm at f 330 mm
  is itself F/1.803.  "A little faster than the spectrometer" (Jim) needs
  the paper's 192 mm entrance beam, which is a spec question, not a solve
  question.

**The two constraint row sets** that make it a long-slit front end, both in
every rung:
1. **TELE:** the chief angle to the slit normal per field.
2. **CONE:** a hinge on the working F/# of the marginal rays at the slit,
   along and across the slit, outside [1.7, 1.8].

Telecentricity is also exact at first order by construction (the stop at
M2, M3's front focus).  The bound F/# anamorphicity is the paper's
constraint; PLATE_Y holds its first-order half.

## Results (the 1.5k default, `tma_longslit_1k5`)

The 1.5k module is resolved by the same method (CC 2026-10-08):
- the strip is ±2.35°, 1500 px and the slit 27 mm;
- f, D and the speed are the 3k's (330 mm, 183 mm, F/1.8);
- the seed and the row sets are the same (tag `tls1k5`);
- it is joined to the 1.5k Dyson of record (silica 130, `size:D:130`).

**Deck of record: R5 `ffc`** (`tls1k5_R5_ffc.in`, recorded as
`challenges/dyson5/dyson5_cprime_1k5.in`).

**The ladder** (engine, model 256, 41-pt grid, 9 strip fields; `tls1k5_figure.txt`):

| rung | DOFs | rms µm | worst FWHM px | min EiP | clearance mm |
|---|---|---|---|---|---|
| R0 seed | first order | 1876–1968 | 15.6 | 0.00 | +10.1 |
| R1 `conic` | θ, slit dz | 151–187 | 10.8 | 0.00 | +17.1 |
| R2 `asph` | + h⁴, h⁶ | 57–152 | 14.7 | 0.00 | +13.9 |
| R3 `ff34` | freeform, degree 3–4 | 17–21 | 1.92 | 0.22 | +1.2 |
| R4 `ff` | freeform, degree 3–6 | 0.73–0.98 | **1.02** | 1.00 | **−0.3** |
| R5 `ffc` | R4 + the CLEAR wall | 0.76–0.96 | **1.02** | 1.00 | **+2.4** |

- **Aspheres are not enough at 1.5k either**: R2 is still 57–152 µm.
- **R4 is the first rung at 1.02 px at every field**, but its M3 → slit leg
  enters M2's body by 0.3 mm.
- **R5 adds the CLEAR wall**: a row in `tls_figure` that scores the record's
  clearance rule every iteration, at the centre and edge fields with coarse
  sampling, as a hinge to a 2 mm target, weighted at merit scale with
  `w_clear` 3000.  It buys +2.4 mm and keeps the floor.

**R5 telescope alone:**
- plate 329.98 / 329.75 mm (local / edge);
- worst chief 0.008° to the slit normal;
- F/# along 1.893–1.894, across 1.801–1.805 (no ray below F/1.7);
- footprints M1 203.8 × 201.5, M2 130.8 × 167.3, M3 189.2 × 172.1 mm;
- working distance 299.4 mm.

**End to end**, joined to the 1.5k Dyson, roll 0 / 180 (`tls1k5_e2e.txt`):

| | R5, px | R5, µm | 1.5k record (GM `bAs`), px | Joe (px) | paper (µm) |
|---|---|---|---|---|---|
| smile (slit-filled, centroid launch) | **0.009 / 0.011** | 0.17 / 0.20 | 0.082 / 0.088 | < 0.1 | < 1.8 |
| point-source shift (chief launch) | 0.009 / 0.011 | 0.16 / 0.19 | 0.504 / 0.503 | — | — |
| keystone | 0.009 / 0.012 | 0.17 / 0.22 | 0.014 / 0.009 | < 0.1 | < 1.8 |
| CRF | 1.029 / 1.028 | 18.5 | 1.255 / 1.211 | < 1.5 | < 50.4 |
| SRF | 2.024 / 2.024 | 36.4 | 2.214 / 2.205 | 1.5–2.0 | < 64.8 |
| ARF (telescope, across slit) | 1.018 | 18.3 | | — | < 50.4 |
| energy in a pixel | 0.974 / 0.990 | | 0.385 / 0.295 | > 0.75 | |
| grating admits | 0.987 | | 0.989 / 0.988 | | |
| joined clearance | +0.6 mm (M3 → slit vs the block face) | | | > 0 | |

**The 1.5k module meets Joe's spec end to end under both launches, except
SRF, which is the slit floor** (2.010–2.023 px; see the 3k section).  It
improves on the 1.5k record of three aspheres in every metric except
keystone at roll 180: 0.012 against 0.009 px, both about a tenth of the bound.  Unlike the
3k, its two launches agree to 0.001 px: R5 carries no chief − centroid
offset at the slit to separate them.

## Files

`tma_longslit.m` (demo), `tma_longslit_params.m`, `tma_longslit_run.m`,
`tls_first_order.m`, `tls_design.m`, `tls_section.m`, `tls_measure.m`,
`tls_clearance.m`, `tls_figure.m`, `tls_e2e.m`, `tls_clearance_joined.m`, `tls_dyson_chief_tilt.m`.
`tma_longslit_1k5.m` (the 1.5k run).  Test: `mmacos/tests/tTmaLongslit.m` (SUITE_FAST).  Records: `tls_*.txt` /
`.mat` / `.in` (the default run); `runs_figure_try*` are the superseded tries
the report cites.
