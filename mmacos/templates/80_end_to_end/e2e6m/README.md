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

(S4–S6 land as the campaign proceeds.)

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
