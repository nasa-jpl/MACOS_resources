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

| | R9 | R7 | Joe | paper |
|---|---|---|---|---|
| smile (slit-filled) px | **0.022 / 0.024** | 0.010 | < 0.1 | < 0.05 |
| point-source across-slit shift px | **0.016 / 0.018** | 0.140 | — | — |
| keystone px | 0.006 / 0.009 | 0.006 | < 0.1 | < 0.1 |
| CRF px | 1.174 / 1.262 | 1.176 | < 1.5 | < 2.8 |
| SRF px | 2.025 / 2.025 | 2.025 | < 1.5–2.0 | < 1.8 |
| EiP | 0.849 / 0.805 | 0.853 | > 0.75 | |
| joined clearance | +0.6 mm | +0.6 | > 0 | |

**The 3k module meets Joe's spec end to end on smile, keystone, CRF and
energy in a pixel, under both smile conventions.**  SRF, 2.025 px, is the
Dyson alone's 2.024: 0.025 over the 2.0 upper bound, the spectrometer's own
floor.

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

| | R7 | Dyson alone | 3k (c) record | Joe | paper (Table 1) |
|---|---|---|---|---|---|
| smile px | 0.140 / 0.141 | 0.005 | 1.64 | < 0.1 | < 0.05 (5 %) |
| keystone px | 0.006 / 0.009 | 0.006 | 0.05 | < 0.1 | < 0.1 (10 %) |
| CRF px | 1.175 / 1.255 | 1.21 | 4.02 | < 1.5 | < 2.8 |
| SRF px | 2.025 / 2.025 | 2.024 | 4.99 | < 1.5–2.0 | < 1.8 |
| ARF px (telescope, across slit) | 1.02 | — | — | — | < 2.8 |
| energy in a pixel | 0.853 / 0.810 | 0.82 | — | > 0.75 | — |
| grating admits | 0.988 / 0.989 | | | | |
| joined clearance | +0.60 / +0.61 mm | +0.54 | +0.06 | > 0 | |

- **CRF and SRF are the Dyson's own floors:** on this spectrometer the
  telescope is no longer the limit.
- **Smile, 0.14 px, is the one requirement still open.**  It is not the
  telescope's slit-plane centroid bow (≤ 0.75 µm = 0.04 px).
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

## Files

`tma_longslit.m` (demo), `tma_longslit_params.m`, `tma_longslit_run.m`,
`tls_first_order.m`, `tls_design.m`, `tls_section.m`, `tls_measure.m`,
`tls_clearance.m`, `tls_figure.m`, `tls_e2e.m`, `tls_clearance_joined.m`, `tls_dyson_chief_tilt.m`.
Test: `mmacos/tests/tTmaLongslit.m` (SUITE_FAST).  Records: `tls_*.txt` /
`.mat` / `.in` (the default run); `runs_figure_try*` are the superseded tries
the report cites.
