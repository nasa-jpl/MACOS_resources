# rodgers2 — the 30× afocal TMA benchmark

Reproduces J.M. Rodgers' four-variant coaxial **afocal** TMA study
(`~/dev/MACOS_sandbox/Design/Rodgers2/`, ORA/CODE V, λ = 1000 nm, EPD
1000 mm, 30×, 0.5°×0.5° field box offset +0.6° in Y) in the MACOS design
layer, and supplies the **interface-pupil metric his deck does not have**.

Plan: `../PLAN_AFOCAL4.md`.  This directory covers **S1 + S2** — the
measurement infrastructure and the scored 3-mirror baseline.  The 4-mirror
form study is S3 and is a separate gate.

## Headline

* **His CODE V afocal field-map RMS is our rung 2** (piston + per-field
  tip/tilt removed).  At that rung, on a uniform grid over his box, our
  numbers reproduce his **15 / 430 / 160 / 119 nm** to
  **0.952 – 1.015×** on the max and **0.994 – 1.042×** on the average, on
  all four variants.
* **His in-box averages are area averages**, not means of his nine solve
  points — his 3×3 set is one-third corners, and scored on it the S1
  average ratio is 2.11× where a uniform grid reads 1.04×.
* **The transcription decode is a 10⁹ margin.**  His "recenter" coordinate
  break lands the coldstop vertex on the *traced* exit chief to 2e−7 mm;
  the opposite ADE sense misses by 211–247 mm on a 33 mm beam.
* **The pupil claim is now a table.**  His 30× measures **28.686×** at the
  box centre on the unoptimised offset variant (his slides say 28.7×) and
  is restored to 30.0015× by the conic re-solve — but the magnification
  still **breathes ±3.6% across the field** measured normal to each field's
  own exit chief (±3.8% as read on the fixed tilted coldstop, which adds
  that plane's own obliquity — PACKET §4 refinement, 2026-08-03), the pupil
  image is blurred by 0.5–2.7% of the pupil diameter, and 0.9 mm rms of
  footprint wander remains at the placed coldstop.
* **CALIB runs on an afocal deck, converges, and wrecks the design** — its
  merit there is 10³–10⁴× the wavefront error.  MATLAB outer solve is the
  path (Dave, 2026-08-02).

Full tables, the element-by-element `.seq` ↔ `.in` audit, the traps and
what is still open: **`PACKET.md`**.

## Run

```matlab
run('~/dev/MACOS_resources/mmacos/challenges/rodgers2/rodgers2.m')  % everything
```
```matlab
rodgers2('sections',0:1)     % transcription + the WFE ladder only
rodgers2('map_n',15)         % denser uniform scoring grid
rodgers2('save',false)       % numbers only, no artifacts
R = rodgers2();              % struct: first-order gate, ladders, pupil table
calib_afocal_probe           % the S1d CALIB probe
```
Batch: `matlab -batch "run('.../rodgers2.m'); exit(0)"` with
`MACOS_HOME` set.  Model size 256; one MATLAB process per model size.

## The four variants

| | `.seq` | what changed | his max / avg (nm) |
|---|---|---|---|
| S1 | `CentAfo_Coaxial_OnAxisFOV` | on-axis box | 15 / 4.0 |
| S2 | `CentAfo_Coaxial_06degOffsetFOV` | box offset +0.6°, optics unchanged | 430 / 154 |
| S3 | `..._06degOffsetFOV_NewConics` | M2/M3 radii + conics re-solved | 160 / 93 |
| S4 | `CentAfo_TiltDecM2M3_...` | + M2/M3 tilt and decentre | 119 / 48 |

## Files

* `rodgers2_seq.m` — the verbatim `.seq` transcription, data only.  Every
  number is quoted from his files; the CODE V syntax decode is in the
  header.  Transcription, not parsing: no `.seq` reader exists and none is
  to be written.
* `rodgers2_deck.m` — renders one variant to a MACOS `.in`.  Every
  conversion convention is named in its header, including the recenter
  coordinate break and why the interface flat is `Element= Reference`.
* `rodgers2.m` — the study driver (four sections; see `PACKET.md`).
* `calib_afocal_probe.m` — the S1d probe: does CALIB work on a deck ending
  in a flat, and what is it minimising there?
* `rodgers2_S{1..4}_*.in` — the four committed decks.
* `rodgers2_S{1..4}_*_ladder.png` — afocal-ladder field maps, three rungs.
* `rodgers2_S{1..4}_*_pupil.png` — the four-part pupil ladder.
* `rodgers2_results.mat`, `rodgers2_calib_probe.mat` — every number.

## The metrics this study introduced

All in `design/src`, shared, and gated — nothing here is example-local.

| kernel | what |
|---|---|
| `afocal_plane_opl` | OPL to a flat reference; `strict_sphere_opl`'s R→∞ limit |
| `afocal_refs` / `afocal_rungs` | the 3-rung afocal ladder: piston / +tip-tilt (boresight) / +power (residual divergence, nm **and** µrad) |
| `afocal_wfe_deck` / `afocal_ladder_deck` | deck-path scorers, one rung and all three |
| `afocal_score_psf` | appends an ideal lens behind the interface pupil in a **separate** scoring deck, so PSF / Strehl / `design_report` work; the delivered `.in` stays clean |
| `pupil_map` | the cone-convergence interface-pupil metric — the four-part ladder |

Gates: `tests/tAfocalKernel.m` (11/11), `tests/tPupilMap.m` (7/7), both in
`SUITE_FREEFORM` (model size 256).

⚠ **Name the rung.**  Rungs 1 and 3 bracket rung 2 by 1.4–2.2× and
0.73–0.99× on these decks.  A quoted afocal WFE without its rung is not a
number.

⚠ **`macos.trace(k).rmsWFE` and `macos.opd()` are BaseUnits (metres)**, not
waves, for these decks.  Multiplying by λ in nm understates by 1e6.

⚠ **The interface flat must be `Element= Reference`.**  `Return` reverses
the ray directions; the OPL is unchanged, so it hides from any
piston-only check, but any metric built from the exit chief is then built
backwards.  `PACKET.md` §5.0.
