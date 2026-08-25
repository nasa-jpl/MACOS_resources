# e2e6m — the 6 m unobscured coronagraph end-to-end example

Design an unobscured 6 m visible telescope, segment its primary, hang an
imager and an APLC coronagraph off it with `macos.design.Bench`, harvest
the linear-model sensitivities, and play a random time series through the
engine — the Keysight end-to-end use case, built entirely from the
committed stage runners.

Every knob lives in **`e2e6m_params.m`**.  Each stage consumes the
previous stage's saved `.in` + `.mat`, so a knob change re-runs from the
first stage it affects.  The narrative — every question raised, every
decision and its result, every gate outcome — is in
**`e2e6m_LOG.md`**; read that first if you are picking this up.

## The three hard numbers (Dave, 2026-08-24)

| | |
|---|---|
| wavelength | **500 nm** — the diffraction-limit claim is at 500 |
| OTA f/# | **12–20** |
| launch shroud | **8 m DIAMETER**, deployed, length free and stated |

The shroud gate is `packaging_report`'s radial extent of every body and
beam about the incoming-beam axis.  The ENTRY CORRIDOR — the incoming
beam upstream of the primary — is drawn in the fit figure but reported
separately as the sunshade keep-out: it is sky, not hardware.

## The four aperture rules (Dave, 2026-08-24)

`realize_apertures` measures footprint centres in global XY and emits
them as element-local `ApVec`, so a saved tilted-fold `.in` loses every
ray on reload.  Since S2 onward lives on saved decks:

1. **Design apertures-off.**
2. **Apertures enter only** via the S2 segmentation machinery (the PM)
   and `aperture_full_field` (the rest of the train).
3. **Every `save()` is gated by a reload ray count.**
4. If `aperture_full_field` shares the defect, fix the frame properly.

Rule 4 fired: it did share it, and worse (no vertex shift AND no
rotation).  Fixed in `macos.design.Telescope` — see the LOG and
`tests/tApertureFrame.m`.

## Stages

| runner | consumes | produces |
|---|---|---|
| `s1_layout_search.m` | `e2e6m_params` | the 0th-order layout pick: `s1_layout_search.{txt,mat}` |
| `s1_telescope.m` | the picked layout | the telescope: `s1_telescope.{in,mat}`, `s1_{layout,wfe_field,fpmap,shroud}.png`, `s1_{report,design_report}.txt` |
| `s1_close_fno.m` | `s1_telescope.m` | drives the CORRECTED f/# into band by secanting the M3 base radius (the freeform stage spends power, so the f/# gate cannot be read on the base spheres): `s1_fno/itNN/` |
| `s2_segmentation.m` | `s1_telescope.in` | 19-segment primary with physical apertures: `s2_segmented.{in,Hx.m}`, footprint/aperture + view figures, `s2_report.txt` |
| `s3_backend.m` | `s1_telescope.in` | the 4-OAP coronagraph relay, built in METRES and SPLICED onto the telescope (`macos.design.append_rx`) so one train carries a telescope perturbation to a contrast number: `s3_{back,full}.in`, `s3_shroud.png`, `s3_report.txt` |

| `s3_coro.m` | `s3_backend` on both primaries | the APLC scored on the segmented AND monolithic trains, so the gap cost is a measured difference: `s3_{seg,mono}_prop.in`, `s3_{seg,mono}_aplc.png`, `s3_contrast.png`, `s3_coro_report.txt` |

| `s3_train_fig.m` | `s3_seg_{full,back}.in` | the S3 layout gate: `s3_train_iso.png` (full train), `s3_back_iso.png` (relay alone) |
| `s4_sensitivities.m` | `s3_seg_full.in` | `dwdx` (with the 19 segments ALSO perturbed as one rigid body), `dwdz`, `dwdgrid` over 5 fields: `s4_sens.mat` (gitignored, `.fp.json` committed), `s4_run.mat`, `s4_*.png`, `s4_{report,sens_report}.txt` |
| `s5_timeseries.m` | `s4_run.mat` + `s3_seg_prop.in` | the drift series with a held image-based correction, contrast scored every 5th frame through the APLC chain: `s5_series.png`, `s5_run.mat`, `s5_report.txt` |
| `s3_imager.m` | `s2_segmented.in` + `s3_seg_full.in` | the IMAGER leg (deployable pick-off -> its own camera) and the two-leg shroud gate: `s3_imager_{leg,full,ep}.in`, `s3_imager_{report.txt,run.mat,shroud.png}` |
| `s3b_pupil.m` | `s3_seg_prop.in` | the pupil AS TRACED at the apodizer plane + its symmetry: `s3b_pupil.{mat,png}` |
| `s3b_apodizer.m` | `s3b_pupil.mat` | the LP ladder (every rung engine-scored) + the aperture-specific APLC apodizer: `s3b_{report.txt,run.mat,apodizer.png}` |
| `deck_e2e6m.py` | every stage REPORT | `deck_e2e6m.{md,pptx}` — DRAFT, generator-built, never hand-edited |

**Reading order for the deck:** `e2e6m_records.py` parses the reports and
`sys.exit`s on any miss, so a stale report is a build failure rather than
a wrong slide.  `deck_e2e6m.py` writes the slide markdown and calls the
committed `challenges/rodgers3/make_brief_slides.py`.  Figures are the
committed stage PNGs, autocropped (and, for the 4-view renders, one panel
lifted) into a gitignored `deckfig/` — never redrawn.

**Two instruments, one observatory.**  The coronagraph leg and the imager
leg share the telescope and the OAP1 collimator and diverge at the shared
collimated pupil, where a DEPLOYABLE PICK-OFF feeds the imager's own
f/19 camera.  Not a beamsplitter: a permanent one would put two
transmitting surfaces in the coronagraph deck and invalidate the S4
sensitivities and S5 series built on it, so the two instruments are two
CONFIGURATIONS of one observatory -- and both are counted in the shroud
gate regardless of which is deployed.  Imager: 0.0042 waves RMS, Strehl
0.9993 at its own exit pupil.  Shroud: 7.451 m for either leg and for
the union, against the 8 m gate -- the 6 m primary sets the envelope, so
**the second instrument costs nothing in shroud diameter**.

**Apertures are declared where they are real.**  `s1_telescope` calls
`Telescope.declare_apertures({'M1'})`, so only the primary emits a hard
`ApType/ApVec`.  Without it the emitter's fallback stamps each mirror's
design-phase BODY radius -- ~3 m on M2 and M3, whose beam footprints are
0.274 m and 0.014 m.  Nothing clipped, so no ray and no number ever
moved; the layout FIGURE was the only thing that complained, drawing
primary-sized domes where the secondaries are.  Graphics are gates.

**S3b: the apodizer redesign did not recover the gap cost, and the
report says so.**  `design/src/apodizer_lp.m` implements the
Carlotti/Vanderbei/Kasdin linear program through the existing occulter
and Lyot, and it is validated (MFT round-trip 7e-8, closed-form Lyot
kernel, correct solutions on a clear circular pupil).  On THIS train it
does not beat the incumbent circular prolate, for a measured reason: a
single-Fourier design model floors near 4e-06 against the engine — five
times above the 8.7e-07 the incumbent already reaches — and the
model-vs-engine divergence GROWS with the target (4.8x -> 13.6x -> 40x),
which is the signature of an optimizer mining its model.  Six isolating
experiments are in the LOG.  The delivered design is the
aperture-specific APLC apodizer (Soummer 2005 Eq. 3 over the traced
aperture): 1.10x in dark-zone mean, 0.57x in median, at 0.73x the
throughput — essentially no recovery.  Apodizer-only redesign against a
fixed occulter and Lyot is not enough; the published segmented-APLC
results come from co-optimizing apodizer x FPM x Lyot, which was
deferred.

**Where the S5 result is honest and where it is bounded.**  The linear
model reproduces the engine for segment PISTON (0.6% relative) and for
nothing else; control is therefore restricted to piston while the drift
still moves all six freedoms, and the deck's backup section says so with
the table.  The correction is image-based and solved once — an optimistic
bound, not a metrology-loop result.  MET was out of scope.

**Deferred coronagraph capability (Dave, 2026-08-24).**  The `ctb` model
carries DM1, DM2, apodizer, mask, Lyot stop and a field stop; this back
end carries apodizer / mask / Lyot only.  The contrast here is therefore
an OPEN-LOOP number -- what the optics deliver, not what a
wavefront-control loop would hold -- and the dark zone is not the
annulus two DMs would give.  Both are existing `Bench` primitives and
existing `prop_layout` station kinds, so the extension is parameter
work.

## Why this designer

The brief named `offset_imager`.  It was retired for this instance on
measurement: the field-offset three-mirror form has an **aperture
ceiling** at 6 m — the offsets that unobscure are the offsets that walk
the beam off a mirror whose radius the compactness requirement has
already made small.  The full root enumeration is in the LOG and in
`s1_layout_search.m`'s header.  `freeform_unobscured`'s sphere+Zernike
tilted-fold topology is the designer of record here, ratified 2026-08-24.

## Run

```matlab
run('<path-to>/mmacos/mmacos_setup.m');
addpath('<path-to>/mmacos/templates/80_end_to_end/e2e6m');
s1_layout_search();      % the layout pick (a sweep; minutes)
s1_telescope();          % the telescope
s2_segmentation();       % the segmented primary
```

House rules: figures and reports land in this directory; no `exit(0)`
inside the example scripts (batch wrappers supply it).
