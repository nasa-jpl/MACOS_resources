# zemax_cassegrain — a Zemax telescope into the segmentation → sensitivities → MET chain

The bridge from a converted prescription to the design pipeline
(`design/runners/README.md`).  One runner, `zemax_cassegrain.m`, takes the
Zemax converter's output through four steps and leaves the stage artifacts
beside it:

| step | call | in → out |
|---|---|---|
| 0 | `rx_converter/zemax/zmx2macos.py zemax_cassegrain.zmx -o zemax_cassegrain.in` (Python, numpy only) | the `.zmx` → a 3-element deck |
| 1 | `macos.design.add_exit_pupil` | `zemax_cassegrain.in` → `zc_xp.in`: FP_return + ExitPupil before the focal plane, 5 elements |
| 2 | `run_segmentation` (elt 1, Hex, 1 ring, gap 5 mm) | `zc_hex1.in` + `zc_hex1Hx.m`: the primary → 7 hexagonal segments, 11 elements |
| 3 | `run_sensitivities` (dwdx + dwdz, centre + 4 corners at 1 arcmin) | `zc_hex1_sens.mat` + report + pages |
| 4 | `run_met` (fiducials on the secondary, launchers on the segments) | `zc_hex1_met.in` + `zc_hex1_met.mat` + report + figures |

**Why step 1.**  Zemax (and CODE V) have no exit-pupil element, and every
wavefront read in the chain takes the OPD at element nElt-1 and refuses a deck
whose nElt-1 is an optic (`+macos/private/wf_elt_auto.m`; the focal-plane OPD
is blind to tilt).  `add_exit_pupil` writes the pair `Telescope.add_pupil`
emits for a designed telescope, located by the engine's FEX on the deck's own
chief ray, and verifies the result by reload and trace.  The converter can
write the same pair itself (`zmx2macos.py --exit-pupil`); on this deck the two
agree to 1.5e-9 mm.

## The telescope

The converter's synthetic f/6 Cassegrain, 500 mm: primary R −1000 mm K −1 (the
stop), secondary R −240 mm K −1.96 400 mm in front, image 200 mm behind the
primary, plus a 25 mm hole in the primary (Zemax `OBSC 0 25`, written as a
`Circle` obscuration and carried onto the centre segment).  `zemax_cassegrain.zmx`
is OpticStudio-style text; `zemax_cassegrain.in` is the converter's output and
is committed so the MATLAB runner needs no Python.

## Measured (2026-11-04, model 256, nGridpts 65)

- **Exit pupil.**  The stop's image in the secondary, a textbook number:
  1/s' = 2/R2 − 1/s with s = 400 → s' = −92.3077 mm; vertex z −492.3077 mm,
  sphere radius 692.3077 mm.  The engine's FEX lands on it to 1.5e-9 mm.  The
  pupil pair costs no ray (3097 → 3097 at the focal plane); on-axis OPD at the
  pupil 6.1e-13 mm.
- **Segmentation.**  7 segments, 166.7 mm flat to flat; 2521 source rays, 2488
  pass the physical apertures (the gap rays clip), rms 6.2e-13 mm vs the parent's
  6.4e-13; the artifact reloads standalone with the same counts (VERIFIED).
- **Sensitivities.**  dwdx 12440 × 54 (9 bodies: 7 segments, the secondary, the
  focal plane) and dwdz 12440 × 56 (8 MonZernike modes per segment) over 5
  fields; piston removed.  The whole-system dwdx conditioning is 2.6e9 and the
  segment-only block 1.2e17: on a PERFECT, rotationally symmetric telescope the
  centre segment's clocking (rz) does nothing to the wavefront (its column norm
  is 0), so the segment block is rank-deficient by construction.  A real
  prescription with figure error does not have this.
- **Metrology.**  42 launcher beams (6 per segment, on the segment edges with
  5 mm clearance) to 6 fiducials at r = 45 mm on the secondary; 72 edge-sensor
  rows.  dldx engine-FD vs analytic 1.3e-6 max relative.  Post-control
  wavefront residual as a fraction of the prior (1 µrad / 1 µm deploy, 1 nm
  sensors): edge sensors only 0.176, MET only 0.154, edge + MET 0.0306;
  Monte-Carlo 200 draws within 0.9 % of the analytic trace.
- **Layout search** (`OPTIMIZE = true`): 4 sweeps × 68880 layouts, about 850 s
  per sweep, and no gain over the as-built truss (edge + MET 0.0306 either way),
  so the knob defaults to off and the committed report is the as-built run.
  `zc_hex1_met_report_optimize.txt` is the search's report.

## Three things the chain needed from an imported deck

1. **The stop.**  The converter puts the stop ON the primary (element-form
   `ApStop=`); segmenting that element drops it (`segment_rx` warns).  The
   sensitivities stage is told the stop POSITION, `'stop', [0 0 0]`, the primary's
   vertex in object space.
2. **Bodies.**  A two-mirror telescope has no third optic for an aft launcher
   ring; the image reference is a Return, not a body the harvest perturbs, so
   `run_met` runs without an aft leg (bodies = segments + secondary).  The e2e
   template's `s5_met.m` shows the aft leg on a fold.
3. **Apertures.**  Zemax `DIAM` is a drawing size and is not written, so the
   secondary has no aperture and the layout optimizer cannot size its rim: the
   fiducial radius grid is given explicitly (`RFID_GRID`).  The launcher
   separation gate is 15 mm (the runner's 50 mm default is for metre-class
   segments).

Two library fixes came out of this run, each with a must-fail gate:
`per_field_indx` built its pixel index from the nominal map's non-zero pixels,
which drops valid pixels that read exactly 0 on a perfect telescope
(`tests/tPerFieldIndx.m`; the core now exports `per_field_mask_2d`), and
`run_met` died drawing an optimized truss it had not built when no layout was
feasible (`tests/tRunMet.m`, the infeasible-gate test).

## Run

```matlab
run('<path-to>/mmacos/mmacos_setup.m');
run('<path-to>/mmacos/templates/20_segmentation/zemax_cassegrain/zemax_cassegrain.m');
```

About three minutes at model 256 with the layout search off (measured 2026-11-04: pupil 10 s, segmentation 10 s, sensitivities 2.5 min, MET 10 s); the search adds about an hour.  Needs the
SegMirMaker binary (`segmirmaker/README.md`) and a built mmacos mex.  The
committed products are the two decks, the pupil deck, the segmented deck and
its Hx, and the three reports; the `.mat`, pages and figures regenerate.
