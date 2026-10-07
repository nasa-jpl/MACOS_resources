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

**Provisional deck of record: R4** (`tls_R4_ff34.in`, recorded as
`challenges/dyson5/dyson5_cprime_3k.in`, CC 2026-10-07).  Engine numbers,
the telescope joined to the 3k Dyson of record (CaF2 240):

| | R4 (c′) | 3k (c) record | Joe | paper |
|---|---|---|---|---|
| smile | 0.73 px | 1.64 | < 0.1 | < 0.05 |
| keystone | 0.009 px | 0.05 | < 0.1 | < 0.1 |
| CRF | 2.51 px | 4.02 | < 1.5 | < 2.8 |
| SRF | 3.90 px | 4.99 | < 1.5–2.0 | < 1.8 |
| telescope alone: chief to slit normal | ≤ 0.034° | | < 0.5° | |
| cone at the slit (F/#, slit axis / across) | 1.89 / 1.82, no ray below F/1.7 | F/1.19 rays | [1.7, 1.8] | |
| M2 footprint | 132 × 165 mm | | | |
| clearance (telescope / joined) | +13.1 / +0.56 mm | +0.06 | > 0 | |

R5/R6 trade CRF for smile and SRF (0.09 / 7.0 / 2.5 px).  A reweighted
rung (along-slit spot rows, outer fields) is running.  The full ladder,
with every rung's per-field table, is in `tls_figure.txt` and in
`macos/REPORT_dyson5_cprime.md`.

## Files

`tma_longslit.m` (demo), `tma_longslit_params.m`, `tma_longslit_run.m`,
`tls_first_order.m`, `tls_design.m`, `tls_section.m`, `tls_measure.m`,
`tls_clearance.m`, `tls_figure.m`, `tls_e2e.m`, `tls_clearance_joined.m`.
Test: `mmacos/tests/tTmaLongslit.m` (SUITE_FAST).  Records: `tls_*.txt` /
`.mat` / `.in` (the default run); `runs_figure_try*` are the superseded tries
the report cites.
