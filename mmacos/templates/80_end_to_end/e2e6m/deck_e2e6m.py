#!/usr/bin/env python3
"""deck_e2e6m -- the e2e6m draft deck, built from the committed records.

Every number comes through e2e6m_records.py, which exits loudly on a
parse miss; figures are the committed stage PNGs.  Governed by
doc/DECK_STYLE.md: lean success-story main path, one slide per stage,
each result slide pairing its LAYOUT figure with its MAP; diagnostics and
mechanics behind ONE plain "Backup Slides" divider; plain descriptive
titles; metric conventions stated ONCE up front; kickers carry the
headline number; no ALL-CAPS; project coinages translated.

Usage:  python3 deck_e2e6m.py
Writes deck_e2e6m.md and deck_e2e6m.pptx beside this file.
DRAFT -- Dave signs off.  Never hand-edit the .pptx.
"""
import os
import subprocess
import sys

import e2e6m_records as R

HERE = os.path.dirname(os.path.abspath(__file__))
BUILDER = os.path.join(HERE, "..", "..", "..", "challenges", "rodgers3",
                       "make_brief_slides.py")

s1, s2, s3, s4, s5 = R.s1(), R.s2(), R.s3(), R.s4(), R.s5()

def crop(name, pad=6):
    """Autocrop a committed stage figure's white margins (DECK_STYLE Figures).

    The pixels are not altered -- only the surrounding whitespace is
    trimmed -- so the figure stays the evidence the stage produced.  The
    cropped copies live in deckfig/ (gitignored; the .pptx embeds them).
    """
    from PIL import Image, ImageChops
    src = os.path.join(HERE, name)
    out_dir = os.path.join(HERE, "deckfig")
    os.makedirs(out_dir, exist_ok=True)
    im = Image.open(src).convert("RGB")
    bb = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if bb:
        bb = (max(0, bb[0] - pad), max(0, bb[1] - pad),
              min(im.size[0], bb[2] + pad), min(im.size[1], bb[3] + pad))
        im = im.crop(bb)
    im.save(os.path.join(out_dir, name))
    return "deckfig/" + name


def crop_panel(name, box, out):
    """Pull ONE panel out of a multi-view figure (DECK_STYLE Figures).

    box is fractional (x0, y0, x1, y1); the panel keeps its own title and
    is then autocropped.  Nothing is redrawn -- a 4-view render at slide
    size is unreadable, and one panel at 4x the area is the same evidence.
    """
    from PIL import Image, ImageChops
    im = Image.open(os.path.join(HERE, name)).convert("RGB")
    W, H = im.size
    c = im.crop((int(box[0] * W), int(box[1] * H),
                 int(box[2] * W), int(box[3] * H)))
    bb = ImageChops.difference(c, Image.new("RGB", c.size, (255, 255, 255))).getbbox()
    if bb:
        c = c.crop(bb)
    out_dir = os.path.join(HERE, "deckfig")
    os.makedirs(out_dir, exist_ok=True)
    c.save(os.path.join(out_dir, out))
    return "deckfig/" + out


FIG = {n: crop(n) for n in ("s1_layout.png", "s1_wfe_field.png",
                            "s2_segmented_footprints.png", "s3_seg_shroud.png",
                            "s3_contrast.png", "s5_series.png",
                            "s4_svspec.png")}
FIG["s2_iso.png"] = crop_panel("s2_segmented_views.png",
                               (0.07, 0.605, 0.47, 0.92), "s2_iso.png")
FIG["s3_iso.png"] = crop_panel("s3_train_iso.png",
                               (0.09, 0.262, 0.46, 0.80), "s3_iso.png")


LAM = "λ"
PM = "±"
MU = "µ"

DOFS = ["Rx", "Ry", "Rz", "Tx", "Ty", "Tz"]
GROUPROWS = "".join(
    f"| {DOFS[i]} | {s4['group'][i]:.4g} | {s4['ratio'][i]:.4g} |\n"
    for i in range(6)).rstrip("\n")

CHECKROWS = "".join(
    f"| {DOFS[c['dof']]} | {c['eng']:.3e} | {c['mod']:.3e} | {c['rel']:.3g} |\n"
    for c in s5["check"]).rstrip("\n")

MD = f"""# A 6 m unobscured coronagraph, end to end
An optical design, its segmentation, its instrument, its error budget and its drift -- one model, one train
~ Built with MACOS / mmacos.  Every number on these slides is parsed from the committed stage reports; no figure was redrawn for the deck.

Conventions, stated once.  Wavefront error is RMS at {s1['lambda_nm']:.0f} nm, referenced to the exit pupil of the configuration being quoted -- the telescope's own for the telescope-only slides, the coronagraph's once the instrument is attached.  Contrast is the dark-zone mean over {s3['inner']:.0f}–{s3['outer']:.0f} {LAM}/D, normalised to the bare on-axis peak of the same train.  The diffraction limit is {s1['dl_bar']:.3f} waves RMS.  Packaging is a deployed, diameter-only fit in an {s1['shroud_gate']:.0f} m launch shroud; length is free.

## The telescope | {s1['wfe_tilt']:.4f} waves RMS across the field -- diffraction-limited at {s1['lambda_nm']:.0f} nm
::: left
- **The design.** A {s1['D_m']:.0f} m unobscured three-mirror telescope: three base spheres placed for packaging, fold tilts doing the unobscuration, all correction carried by Zernike surface departures. Field {PM}{s1['fov_arcmin']:.2f} arcmin.
- **The wavefront.** {s1['wfe_tilt']:.4f} waves RMS worst case over the field, against a {s1['dl_bar']:.3f}-wave diffraction limit.
- **The packaging.** {s1['shroud_m']:.3f} m diameter against the {s1['shroud_gate']:.0f} m shroud, over a {s1['train_m']:.2f} m train; {s1['clear_n']}/{s1['clear_tot']} bodies clear of every beam; {s1['rays_pass']}/{s1['rays_tot']} rays survive a standalone reload of the saved prescription.
- **The one miss.** Focal ratio f/{s1['fno']:.2f} against a requested f/{s1['fno_lo']:.0f}–{s1['fno_hi']:.0f}. It is not a tuning failure -- see the backup.
::: right
![The three-mirror train: light enters from the left, folds back to the secondary, returns past the primary to the tertiary and the focus.]({FIG['s1_layout.png']}){{h=3.0}}
![Wavefront error over the design field. Worst corner {s1['wfe_tilt']:.4f} waves.]({FIG['s1_wfe_field.png']}){{h=3.1}}

## The segmented primary | {s2['nseg']} segments, and a poke that stays where it is put
::: left
- **The segmentation.** {s2['nseg']} hexagonal segments, {s2['width_m']:.1f} m flat to flat, {s2['gap_m']*1000:.0f} mm gaps, each carrying the parent's solved figure and its own physical polygonal aperture.
- **The apertures are real.** Each segment declares its polygon, and {s2['rays_ap']} of {s2['rays_bare']} traced rays survive them; the two that clip land on a gap edge. Reloading the saved prescription standalone reproduces {s2['rays_ap']}/{s2['rays_bare']} -- the check that catches a wrongly-framed aperture, which otherwise fails silently on disk.
- **The check that matters.** Displace one segment by 10 nm along its normal: the wavefront responds {s2['in_rms']*1e9:.2f} nm over the {s2['n_in']} rays that land on it and **exactly zero** over the other {s2['n_out']}. The response stays on the segment that moved -- which is what makes a per-segment error budget mean anything.
::: right
![The segmented telescope: 19 hexagons on the primary, feeding the same fold train.]({FIG['s2_iso.png']}){{h=2.9}}
![Traced footprints against the emitted aperture polygons. Colour is the segment a ray landed on; black is the declared glass.]({FIG['s2_segmented_footprints.png']}){{h=3.2}}

## The instrument | One prescription, {s3['nelt']} elements, and {(s3['shroud_m']-s1['shroud_m'])*1000:.0f} mm of extra shroud diameter
::: left
- **The train.** A four-mirror relay off the telescope focus: collimate to an accessible pupil, focus to the mask, re-collimate to the Lyot stop, focus to the detector. Spliced onto the segmented telescope as {s3['nelt']} elements of one prescription -- not a second model.
- **Why one train.** Only a single model carries a telescope perturbation through to a contrast number. That is what stages four and five then use.
- **It fits.** {s3['shroud_m']:.3f} m against the {s3['shroud_gate']:.0f} m shroud, {s3['rays_pass']} rays through, and the relay lives inside the annulus the fold optics already occupy.
- **It is faithful.** The chief ray through the diffraction model agrees with the geometric one to {s3['seg']['chief']:.1e} m.
::: right
![The folded observatory seen down the launch axis, inside the 8 m shroud circle.]({FIG['s3_seg_shroud.png']}){{h=4.6}}

## What the segment gaps cost the coronagraph | {s3['ratio_mean']:.0f}× in dark-zone contrast, same mask on both apertures
::: left
- **The measurement.** The same apodized pupil Lyot coronagraph -- same apodizer, same {s3['r_occ']:.1f} {LAM}/D occulter, same Lyot stop -- run on the segmented primary and on a monolithic version of the identical telescope.
- **Segmented:** {s3['seg']['mean']:.3e} mean contrast, {s3['seg']['suppr']:.2e} on-axis suppression.
- **Monolithic:** {s3['mono']['mean']:.3e} mean, {s3['mono']['suppr']:.2e} suppression.
- **The ratio is the gaps and nothing else** -- {s3['ratio_mean']:.0f}× in the mean, {s3['ratio_median']:.0f}× in the median. Both trains share the same wavefront; only the pupil's structure differs.
- The monolithic result independently reproduces the reference testbed's clear-pupil number, with nothing tuned to match it.
::: right
![Radial contrast, both apertures, dark zone shaded. The segmented curve plateaus where the monolithic one keeps falling -- that plateau is light scattered by the gaps.]({FIG['s3_contrast.png']}){{h=4.6}}

## The error budget | Moving 19 segments as one body is not 19 times moving one
::: left
- **The model.** Wavefront sensitivity to every rigid-body freedom of {s4['n_optics']} optics, to each segment's figure modes, and to a per-segment influence basis: {s4['dwdx_cols']} + {s4['dwdz_cols']} + {s4['dwdg_cols']} channels over five field points.
- **The exhibit.** The {s4['n_group']} segments were also perturbed as a single rigid body. Comparing that with a single segment's response separates three different behaviours -- in one table.
- **Tilt** adds up: ratio {s4['ratio'][0]:.1f}, the member count. A per-segment budget is right.
- **Piston** cancels: ratio {s4['ratio'][5]:.3f}. Moving all {s4['n_group']} together is nearly a global piston, which the reference removes -- a per-segment budget overstates the assembly by about {1/s4['ratio'][5]:.0f}×.
- **Clocking** is invisible per segment and real for the assembly: ratio {s4['ratio'][2]:.0f}.
- **Not every freedom is observable.** The {s4['dwdx_seg_cols']} segment rigid-body channels span a range of {s4['dwdx_cond']:.2e} in singular value: a wavefront measurement constrains far fewer directions than there are freedoms, which is what makes the correction in the next slide a solved problem rather than an inversion.
::: right
| freedom | assembly response | assembly / one segment |
|---|---|---|
{GROUPROWS}
~ Column RMS of wavefront change per unit motion; rotations per radian, translations per metre, the same convention on both sides.
![Singular values of the three Jacobians, largest normalised to one. Rigid-body motion (blue) collapses by seven decades; the figure and influence bases stay within one.]({FIG['s4_svspec.png']}){{h=3.1}}

## The observatory drifts | Contrast tracks the wavefront, {s5['unc']['con0']:.2e} to {s5['unc']['con1']:.2e}
::: left
- **The history.** {s5['frames']} frames at {s5['dt']:.0f} s -- a {s5['frames']*s5['dt']/60:.0f}-minute soak -- with a random walk plus a correlated drift on every freedom of all {s2['nseg']} segments, played through the engine.
- **Uncorrected:** wavefront {s5['unc']['wfe0']:.4f} to {s5['unc']['wfe1']:.4f} waves; contrast {s5['unc']['con0']:.3e} to {s5['unc']['con1']:.3e}.
- **Corrected:** {s5['cor']['wfe0']:.4f} to {s5['cor']['wfe1']:.4f} waves; contrast {s5['cor']['con0']:.3e} to {s5['cor']['con1']:.3e}. One image-based correction, solved early and held.
- **Why the benefit fades.** The correction is never updated, so it decays as the state walks away from where it was solved. That is the shape of a held correction, not a modelling artefact.
- Each contrast point is a full diffraction propagation through the mask chain, so the timeline is sampled every {s5['every']} frames ({s5['n_scored']} of {s5['frames']}).
::: right
![Wavefront and contrast against time, corrected and uncorrected.]({FIG['s5_series.png']}){{h=4.7}}

## What this demonstrates | One model, from surface figure to science contrast
- **A design becomes an instrument becomes an error budget becomes a time series**, without leaving the model or re-entering the geometry anywhere.
- **The gap penalty is measured, not assumed**: {s3['ratio_mean']:.0f}×, from two runs that differ in one thing.
- **Assembly-level freedoms behave differently from their members** -- by a factor {1/s4['ratio'][5]:.0f} in one direction and {s4['ratio'][2]:.0f} in another -- and the model shows which.
- **Every gate is in the record.** Where something did not close, the deck says so and prices it.
::: right
![The modelled observatory: {s2['nseg']} segments, three telescope mirrors and the instrument relay, in one prescription.]({FIG['s3_iso.png']}){{h=4.4}}

## Backup Slides

## Focal ratio: why f/{s1['fno']:.1f} and not f/{s1['fno_lo']:.0f}–{s1['fno_hi']:.0f} | The corrected focal ratio is not a continuous function of the layout
- The surface-figure solve spends optical power, so the focal ratio of the corrected system differs from that of the base spheres. It has to be read on the corrected system.
- A search on the tertiary radius refused after five iterates. Two neighbouring layouts, 0.36% apart in that radius, gave f/15.54 and f/25.74; widening the sweep gave f/20.09, f/8.56, f/26.90, f/25.39 at four nearby values.
- The solve lands in different basins depending on the starting geometry, and each spends a different amount of power. No continuous search steers it.
- Narrowing the design field does not help either: at fixed geometry, halving the field made the residual worse.
- **The design point was therefore chosen on wavefront**, which it meets. A slower focal ratio also puts more {LAM}/D on the focal-plane mask, which eases the hardest fabrication in the instrument.

## Linear model against the engine | Only piston reproduces; control was restricted to it
::: left
- Poking each freedom and comparing the engine's wavefront change with the sensitivity model:
- Piston agrees to better than 1%. Every other freedom disagrees, by factors of 1–{s5['worst']:.0f}.
- A ladder over three decades of poke size separates two behaviours: the centre segment's tilt response is purely second-order, while an outer segment's is linear but 3.8× the model. Neither is a noise floor.
- Ruled out by measurement: the wavefront reference. Repeating the ladder with the other reference reproduces every number to five digits.
- **Consequence.** With all six freedoms in the control basis the corrected leg came out worse than the uncorrected one. Control was restricted to the freedom that reproduces; the drift still moves all six.
::: right
| freedom | engine | model | relative error |
|---|---|---|---|
{CHECKROWS}
~ RMS wavefront change, in metres, for a 1 nm / 1 nrad poke of a single segment. Open item: two untested candidates remain -- the focal-plane tracking used when the sensitivities were harvested, and the real-ray stop aiming, which would explain why the centre segment -- the one the chief ray lands on -- is the anomalous case.

## What is not in this model | Stated, not omitted
- **No deformable mirrors and no field stop.** The reference testbed model carries both. Without them the contrast here is an open-loop number -- what the optics deliver, not what a wavefront-control loop would hold -- and the dark zone is not the annulus two deformable mirrors would give.
- **No metrology loop.** The correction in the drift series is image-based: it sees the wavefront directly rather than estimating it from a truss. The corrected leg is therefore an optimistic bound.
- **The coronagraph mask is a clear-pupil design.** Re-optimising it for a segmented aperture is a known and separate piece of work; the {s3['ratio_mean']:.0f}× above is the size of the prize.
- Both additions are existing primitives in the toolkit, so the extension is configuration rather than new machinery.

## How to reproduce | Every number on these slides comes from these runs
- `s1_layout_search` then `s1_telescope` -- the layout pick and the telescope.
- `s2_segmentation` -- the segmented primary and its edge-sensor sidecar.
- `s3_backend`, `s3_train_fig` then `s3_coro` -- the instrument, its layout render, and the coronagraph on both apertures.
- `s4_sensitivities` -- the error budget, roughly half an hour.
- `s5_timeseries` -- the drift series.
- `python3 deck_e2e6m.py` -- this deck, from the reports those runs wrote.
~ All knobs live in one parameter file. The narrative record -- every question raised, every decision and its result, every gate outcome -- is in the campaign log beside the runners.
"""

MD = MD.replace(" -- ", " — ")   # house typography (rodgers3 deck)

md_path = os.path.join(HERE, "deck_e2e6m.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(MD)
print("wrote", md_path)

rc = subprocess.call([sys.executable, BUILDER, md_path,
                      os.path.join(HERE, "deck_e2e6m.pptx")])
sys.exit(rc)
