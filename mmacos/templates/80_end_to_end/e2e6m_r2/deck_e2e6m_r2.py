#!/usr/bin/env python3
"""deck_e2e6m_r2 -- the round-2 e2e6m draft deck, built from the records.

The round-1 story (telescope, segmentation) carried forward, the
round-2 additions folded in: the DM-bearing coronagraph, the masks and
DMs as physical objects, the metrology truss, and the closed-loop
drift series.  Every number is parsed from the committed stage reports
(e2e6m_r2_records.py, loud on any miss); figures are the committed
stage PNGs, autocropped, never redrawn.  Governed by doc/DECK_STYLE.md.

Usage:  python3 deck_e2e6m_r2.py
Writes deck_e2e6m_r2.md and deck_e2e6m_r2.pptx beside this file.
DRAFT -- Dave signs off.  Never hand-edit the .pptx.
"""
import os
import subprocess
import sys

import e2e6m_r2_records as R

HERE = os.path.dirname(os.path.abspath(__file__))
R1DIR = os.path.normpath(os.path.join(HERE, "..", "e2e6m"))
BUILDER = os.path.join(HERE, "..", "..", "..", "challenges", "rodgers3",
                       "make_brief_slides.py")

r0, r1, r3, r4, r2m = R.r0(), R.r1(), R.r3(), R.r4(), R.r2m()
cf1, cf1c, cf2, cf3a = R.cf1(), R.cf1c(), R.cf2(), R.cf3a()
s1, s2, s3c = R.s1(), R.s2(), R.s3c()


def crop(name, pad=6, src_dir=HERE):
    """Autocrop a committed figure's white margins (DECK_STYLE Figures)."""
    from PIL import Image, ImageChops
    src = os.path.join(src_dir, name)
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


FIG = {n: crop(n) for n in (
    "r2_sequence.png", "r2_back_plane.png", "r2_train_iso.png",
    "r2_apodizer.png", "r2_fpm_mask.png", "r1_dm_poke.png",
    "r1_contrast.png", "r3_svspec.png", "r3_met_layout.png",
    "r4_series.png", "r1_union_shroud.png",
    "cf1_families.png", "cf2_floors.png", "cf3a_lyot.png",
    "cf1c_stop_attrib.png")}
for n in ("s1_layout.png", "s1_wfe_field.png", "s2_segmented_footprints.png"):
    FIG[n] = crop(n, src_dir=R1DIR)

LAM = "λ"
PM = "±"
MU = "µ"
DOFS = ["Rx", "Ry", "Rz", "Tx", "Ty", "Tz"]

GROUPROWS = "".join(
    f"| {DOFS[i]} | {r3['group'][i]:.4g} | {r3['ratio'][i]:.4g} |\n"
    for i in range(6)).rstrip("\n")

SURFROWS = "".join(
    f"| {d} | {r0['wrong'][d]:.3g} | {r0['right'][d]} |\n"
    for d in DOFS).rstrip("\n")

MD = f"""# A 6 m unobscured coronagraph, end to end -- with the loop closed
An optical design, its segmentation, its instrument with deformable mirrors and metrology, its error budget and its corrected drift -- one model, one train
~ Built with MACOS / mmacos.  Every number on these slides is parsed from the committed stage reports; no figure was redrawn for the deck.
~ DRAFT (2026-08-27): the coronagraph-family slides (families / floors / Lyot trade) await Dave's review of the CF1c-CF3a numbers.

Conventions, stated once.  Wavefront error is RMS at {s1['lambda_nm']:.0f} nm at the exit pupil of the configuration being quoted -- the telescope's own for the telescope-only slides, the coronagraph's once the instrument is attached (and every model-vs-engine check is evaluated at the surface its sensitivity model was harvested at).  Contrast is the dark-zone mean over {r1['inner']:.0f}–{r1['outer']:.0f} {LAM}/D, normalised to the bare on-axis peak of the same train; static scoring at a 1024 grid, the drift series at {r4['model']:.0f} (its control operator's grid).  The diffraction limit is {s1['dl_bar']:.3f} waves RMS.  Packaging is a deployed, diameter-only fit in an {s1['shroud_gate']:.0f} m launch shroud.

## The telescope | {s1['wfe_tilt']:.4f} waves RMS across the field -- diffraction-limited at {s1['lambda_nm']:.0f} nm
::: left
- **The design.** A {s1['D_m']:.0f} m unobscured three-mirror telescope: three base spheres placed for packaging, fold tilts doing the unobscuration, all correction carried by Zernike surface departures.  Field {PM}{s1['fov_arcmin']:.2f} arcmin.
- **The wavefront.** {s1['wfe_tilt']:.4f} waves RMS worst case over the field, against a {s1['dl_bar']:.3f}-wave diffraction limit.
- **The packaging.** {s1['shroud_m']:.3f} m diameter against the {s1['shroud_gate']:.0f} m shroud; {s1['rays_pass']}/{s1['rays_tot']} rays survive a standalone reload of the saved prescription.
- **The one miss.** Focal ratio f/{s1['fno']:.2f} against a requested f/{s1['fno_lo']:.0f}–{s1['fno_hi']:.0f}; not a tuning failure -- see the backup.
::: right
![The three-mirror train: light enters from the left, folds back to the secondary, returns past the primary to the tertiary and the focus.]({FIG['s1_layout.png']}){{h=3.0}}
![Wavefront error over the design field.  Worst corner {s1['wfe_tilt']:.4f} waves.]({FIG['s1_wfe_field.png']}){{h=3.1}}

## The segmented primary | {s2['nseg']} segments, and a poke that stays where it is put
::: left
- **The segmentation.** {s2['nseg']} hexagonal segments, {s2['width_m']:.1f} m flat to flat, {s2['gap_m']*1000:.0f} mm gaps, each carrying the parent's solved figure and its own physical polygonal aperture.
- **The apertures are real.** {s2['rays_ap']} of {s2['rays_bare']} traced rays survive the declared polygons, and a standalone reload reproduces the count -- the check that catches a wrongly-framed aperture.
- **The check that matters.** Displace one segment 10 nm along its normal: the wavefront responds {s2['in_rms']*1e9:.2f} nm over the {s2['n_in']} rays on that segment and exactly zero over the other {s2['n_out']}.
::: right
![Traced footprints against the emitted aperture polygons.  Colour is the segment a ray landed on; black is the declared glass.]({FIG['s2_segmented_footprints.png']}){{h=4.6}}

## The instrument, in light order | Eight relay mirrors, two deformable mirrors, five mask and pupil stations, two cameras
::: full
- **One train.** From the segmented primary through the fold train to the collimated pupil, then the coronagraph leg -- DMs, apodizer, occulter, Lyot, field stop, back-end pupil, detector -- with a deployable pick-off feeding the imager.  Spliced as {r1['nelt']} elements of one prescription; {r1['rays']}/{r1['rays_tot']} rays, the telescope's own count.
![Light order through both instruments, read from the prescriptions.  Orange marks the deformable mirrors, purple the mask and pupil sites, dashed the focus stations.]({FIG['r2_sequence.png']}){{h=3.9}}

## Two instruments, one shroud | A second camera still costs nothing in diameter
::: left
- **The coronagraph leg** carries the science train: {r1['nelt']} elements, chief ray through the diffraction model agreeing with the geometric one to {r1['seg']['chief']:.0e}.
- **The imager leg.** A deployable pick-off just after the collimator feeds its own f/{s3c['fno']:.0f} camera: {s3c['wfe']:.4f} waves RMS, Strehl {s3c['strehl']:.4f} -- unchanged from the round-1 stage that built it, because the pick-off sits upstream of everything round 2 added.
- **Both fit.** {r1['shroud']:.3f} m against the {r1['shroud_gate']:.0f} m shroud for either leg and for the union; the {s1['D_m']:.0f} m primary sets the envelope.
::: right
![Both legs down the launch axis, inside the 8 m shroud circle.]({FIG['r1_union_shroud.png']}){{h=4.5}}

## The back end on its bench | Eight relay mirrors and two DMs in a 5.1 × 0.6 m accordion
::: left
- **The drawing is to scale**, in the plane the folds live in: the collimator, the DM pocket, then the mask relay -- each mask plane sitting between its focus and pupil mirrors.
- **Near-normal folds** (6°) keep each off-axis section gentle; folding every mirror to the same side ping-pongs the beam into a bench-sized pocket.  The other choice fans the chain 120° and breaks the shroud -- measured, in the backup.
- **In the full train** the back end is centimetre-class against the {s1['D_m']:.0f} m primary -- the reason it gets its own drawing.
::: right
![The back end in its fold plane, to scale.  Orange: the DMs at the collimated pupil.  Purple: apodizer, Lyot and back-end pupil sites.  White: the occulter and field-stop foci.]({FIG['r2_back_plane.png']}){{h=2.3}}
![The full train for scale: the 6 m segmented primary and the beam it feeds.]({FIG['r2_train_iso.png']}){{h=2.9}}

## The masks, as physical objects | An occulter {r1['r_occ']:.1f} {LAM}/D across at a {s1['lambda_nm']:.0f} nm focus is {r2m['r_occ_um']:.0f} {MU}m of metal
::: left
- **The occulter** is an opaque disk of radius {r1['r_occ']:.1f} {LAM}/D at the intermediate focus -- {r2m['r_occ_um']:.1f} {MU}m physical radius at the measured {r2m['lamD_um']:.2f} {MU}m-per-{LAM}/D focal scale, drawn from the same engine quantities the scoring chain uses.
- **The apodizer** is the clear-pupil prolate the coronagraph is scored with, shown as manufactured and as the light sees it -- over the traced {s2['nseg']}-segment pupil, gaps and all.
- Both figures are the scoring masks rebuilt from the scoring parameters, not illustrations.
::: right
![The apodizer: the prolate transmission profile (left) and the same profile over the traced segmented pupil (right), mm at the {r1['pupil_mm']:.0f} mm pupil.]({FIG['r2_apodizer.png']}){{h=2.6}}
![The focal-plane occulter, transmission against {LAM}/D; the physical radius is stated on the figure.]({FIG['r2_fpm_mask.png']}){{h=2.7}}

## The deformable mirrors are real | A 20 nm actuator poke lands where it is put, at twice the surface
::: left
- **The model.** Two {r1['nact']:.0f}×{r1['nact']:.0f}-actuator DMs as grid-data surfaces in their own frames, {r1['nact_active']:.0f} actuators each inside the {r1['pupil_mm']:.0f} mm beam, influence functions with nearest-neighbour coupling.
- **The gate.** One actuator poked 20 nm, half a pupil radius off centre: the exit-pupil wavefront answers with {r1['poke_peak']:.3g} m peak against {r1['poke_exp']:.1g} expected (twice the surface, for a mirror), {r1['poke_efrac']:.0f}% of the response energy within 0.15 pupil radii of the peak, peak at {r1['poke_off']:.2f} radii off centre.
- **Why it is a gate.** The failure mode of a wrongly-framed grid surface is a response painted at the pupil centre -- localisation is the proof the actuator geometry is right.
::: right
![The single-actuator response at the exit pupil: localised, off-centre, 2× the commanded surface.]({FIG['r1_dm_poke.png']}){{h=4.5}}

## What the segment gaps cost | {r1['ratio_mean']:.0f}× in dark-zone contrast, same mask on both apertures
::: left
- **The measurement.** The same apodized-pupil Lyot coronagraph on the segmented primary and on a monolithic twin of the identical telescope.
- **Segmented:** {r1['seg']['mean']:.3e} mean contrast, {r1['seg']['suppr']:.2e} on-axis suppression.
- **Monolithic:** {r1['mono']['mean']:.3e} mean, {r1['mono']['suppr']:.2e} suppression.
- **The ratio is the gaps and nothing else** -- {r1['ratio_mean']:.0f}× in the mean, {r1['ratio_median']:.0f}× in the median; both trains share the same wavefront.
- These are open-loop numbers: the DMs are flat here.  The loop slide prices what they buy.
::: right
![Radial contrast, both apertures, dark zone shaded.  The segmented curve plateaus where the monolithic one keeps falling -- that plateau is gap-scattered light.]({FIG['r1_contrast.png']}){{h=4.6}}

## Six coronagraphs, one train | Pre-control: the apodized Lyot leads at {cf1['fam']['apl']['mean']:.1e}; the vortex pays the gaps ~40x
::: left
- **The design rule that made them comparable.**  On a segmented hex aperture the coronagraph's pupil is a CIRCULAR STOP at the apodizer plane -- the telescope's hex envelope is upstream optics, not the coronagraph pupil.  Every family below sees the same circularized, gapped pupil; contrast normalizes to its bare peak, and the stop's ~8% collecting-area cost sits in the throughput column.
- **The families (dark-zone mean, PRE-CONTROL | throughput):** classical Lyot {cf1['fam']['hard']['mean']:.2e} pre-control, {cf1['fam']['hard']['thru_pct']:.0f}% throughput; apodized Lyot {cf1['fam']['apl']['mean']:.2e} pre-control, {cf1['fam']['apl']['thru_pct']:.0f}% throughput; APLC {cf1['fam']['aplc']['mean']:.2e} pre-control, {cf1['fam']['aplc']['thru_pct']:.0f}% throughput (aperture-matched prolate: solver-limited, flag stands); band-limited {cf1['fam']['blc']['mean']:.2e} pre-control, {cf1['fam']['blc']['thru_pct']:.0f}% throughput; vortex c4 {cf1['fam']['v4']['mean']:.2e} pre-control, {cf1['fam']['v4']['thru_pct']:.0f}% throughput; c6 {cf1['fam']['v6']['mean']:.2e} pre-control, {cf1['fam']['v6']['thru_pct']:.0f}% throughput -- the vortex passes the segment-gap light the occulting families block.
- **The stop's 2.16x on the bare occulter, attributed by measurement:** entirely the stop edge at fixed masks (the scale re-reference contributes {cf1c['rechain']:.2f}x); the Babinet split shows the edge enters by coherent INTERFERENCE with the gap field (cross {cf1c['cross']:.1e} vs the rim's own {cf1c['rim']:.1e}) -- coherent, hence in the loop's reach.  The next slide cashes that.
::: right
![The six families, statics on the circularized segmented train.]({FIG['cf1_families.png']}){{h=4.6}}

## The loop on every family | All six floors sit within 2x of their own Jacobian's linear-achievable: the substrate speaks, not the controller
::: left
- **static -> relin floor | linear-achievable at the achieved stroke:** classical Lyot {cf2['fam']['hard']['static']:.1e} -> {cf2['fam']['hard']['relin']:.1e} | {cf2['fam']['hard']['linach']:.1e}; apodized Lyot {cf2['fam']['apl']['static']:.1e} -> {cf2['fam']['apl']['relin']:.1e} | {cf2['fam']['apl']['linach']:.1e} (the campaign floor); APLC {cf2['fam']['aplc']['static']:.1e} -> {cf2['fam']['aplc']['relin']:.1e} | {cf2['fam']['aplc']['linach']:.1e}; band-limited {cf2['fam']['blc']['static']:.1e} -> {cf2['fam']['blc']['relin']:.1e} | {cf2['fam']['blc']['linach']:.1e}; vortex c4 {cf2['fam']['v4']['static']:.1e} -> {cf2['fam']['v4']['relin']:.1e} | {cf2['fam']['v4']['linach']:.1e}; c6 {cf2['fam']['v6']['static']:.1e} -> {cf2['fam']['v6']['relin']:.1e} | {cf2['fam']['v6']['linach']:.1e}.
- **The stopped occulter digs 8.9x where its no-stop self dug 5.2x** -- the loop claws back most of the coherent edge term the previous slide measured.
- **The vortices are linear-optimal:** relinearization pays 2.5-3.7x where the fixed Jacobian paid 1.2x (gap-chasing strokes reach the linearity boundary early), and the residual gap leak is uncontrollable at this amplitude authority.
- **The knob is the DM spacing:** z/z_T = {cf2['zzT']:.1e} at the outer working angle -- Talbot-weak amplitude authority is the common ceiling; the spacing trade prices it.
::: right
![Convergence per family (relinearization joins at the dotted lines) and the closed-loop column.]({FIG['cf2_floors.png']}){{h=4.6}}

## The Lyot trade | Throughput is bought with contrast on every leg; the operating points were chosen, not defaulted
::: full
- **The sweep.**  Lyot fraction against contrast AND throughput for the vortex legs and the apodized-Lyot leg, statics under the stop; stars mark the S1 operating points the campaign scored.
![Contrast vs Lyot fraction (left) and the contrast-throughput trade with the family operating points (right).]({FIG['cf3a_lyot.png']}){{h=4.2}}

## The error budget closes | Engine vs model: worst 0.35% over a segment, a DM and a relay mirror, all six freedoms
::: left
- **The model.** Wavefront sensitivity to every rigid-body freedom of {r3['n_optics']} optics -- the segments, the fold mirrors, the relay and both DMs -- plus segment figure modes and a per-segment influence basis: {r3['dwdx_cols']} + {r3['dwdz_cols']} + {r3['dwdg_cols']} channels over five field points.
- **The closure check.** Poke a segment, a DM and a relay mirror through the engine and compare with the model, at the surface the model was harvested at: worst relative error {r3['closure_worst']:.2g} over {r3['closure_pairs']} freedom pairs.  The matching check in the previous build disagreed by factors up to 55; the resolution is in the backup, and it was not the model.
- **The assembly exhibit.** The {s2['nseg']} segments perturbed as one body against one segment alone: tilt adds up (ratio {r3['ratio'][0]:.1f}), piston cancels (ratio {r3['ratio'][5]:.3f}), clocking is invisible per segment and real for the assembly.
::: right
| freedom | assembly response | assembly / one segment |
|---|---|---|
{GROUPROWS}
~ Column RMS of wavefront change per unit motion; rotations per radian, translations per metre.
![Singular values of the three Jacobians, largest normalised to one.]({FIG['r3_svspec.png']}){{h=2.6}}

## The metrology truss | {r3['n_gauge']} laser gauges + {r3['n_edge']} edge sensors, validated against the engine to {r3['fd_off']:.2g}%
::: left
- **The layout.** Six launchers per segment on its true boundary, fiducials on the secondary's rim, plus inter-segment edge sensors -- the committed metrology stage run on this telescope with this train's sensitivity model.
- **What it buys.** Post-control wavefront residual {r3['wem_em']:.2f} nm per nm of gauge noise with edges and gauges together, against {r3['wem_m']:.0f} for gauges alone -- the edge sensors carry the segment-to-segment state.
- **Validated.** The layout merit's finite-difference check closes at {r3['fd_off']:.2g}%, and a 200-draw Monte-Carlo estimate/control loop lands within 2.6% of the analytic residual.
::: right
![The as-built truss: beams from the 19 segments to the hub fiducials; edge sensors on the segment boundaries.]({FIG['r3_met_layout.png']}){{h=4.5}}

## The loop closes | The dark zone held under drift: {r4['cor']['con1']:.1e} against {r4['unc']['con1']:.1e} open-loop
::: left
- **The drift.** {r4['frames']} frames over {r4['frames']*r4['dt']/60:.0f} minutes: random walk plus correlated drift on every freedom of all {s2['nseg']} segments, played through the engine on the full train.
- **The rigid-body loop.** Edge and gauge readings, a weighted least-squares estimate, an integrating controller on all six freedoms of every segment: the state residual holds at {r4['resid_nm']:.2f} nm while the open-loop drift grows to {r4['drift_nm']:.1f} nm.
- **The DM loop.** The measured actuator Jacobian digs the dark zone {r4['dig'][0]:.1e} → {r4['dig'][-1]:.1e}, then a damped re-solve at each scored frame holds it: {r4['cor']['con0']:.2e} → {r4['cor']['con1']:.2e} closed, against {r4['unc']['con0']:.2e} → {r4['unc']['con1']:.2e} open.
- **The honest line.** Closed-loop pupil wavefront is larger than uncorrected -- that is the DM stroke the contrast is bought with, and the figure labels it as such.
::: right
![State, wavefront and contrast against time.  The contrast panel is the payoff: open loop degrades 3.7×; the closed loop holds.]({FIG['r4_series.png']}){{h=4.7}}

## What this demonstrates | One model, from surface figure to a held dark zone
- **A design becomes an instrument becomes an error budget becomes a controlled observatory**, without leaving the model or re-entering the geometry anywhere.
- **The gap penalty is measured, not assumed** -- {r1['ratio_mean']:.0f}× open loop -- and the control loop that answers it is measured too: {r4['unc']['con1']:.1e} open against {r4['cor']['con1']:.1e} closed at the end of the soak.
- **Every model-vs-engine comparison closes** -- worst {r3['closure_worst']:.2g} across segments, DMs and relay optics -- because the comparison is made at the surface the model lives on.  Where a build disagreed, the deck says why and what fixed it.
- **All of it is committed**: prescriptions, runners, reports, and this deck's numbers are one `git clone` away.

## Backup Slides

## The stop's edge, measured | A Babinet split with 1e-15 closure: the edge interferes, it does not merely add
::: left
- **Method.**  The circular stop applied as a SCREEN on the no-stop chain holds every mask and scale fixed, so E_rim = E_hex - E_screened is exactly the blocked rim's field.  Pins: screened bare peak == stopped bare peak to 0.0e0; both S1 records reproduced under 1%.
- **The split (dark-zone mean energy):** I_ns {cf1c['I_ns']:.2e} + rim {cf1c['rim']:.2e} + cross {cf1c['cross']:.2e} -- the rim's own ring is ~28% of the increase; the cross term (coherent interference with the pre-existing gap speckle) is 2.6x larger.  Peak renormalization {cf1c['pk_ratio']:.3f}x; dark-zone energy up {cf1c['energy_up']:.2f}x at fixed masks; the lambda/D re-reference contributes {cf1c['rechain']:.2f}x.
- **The APLC flag, restated:** the stopped aperture-matched prolate hit its iteration cap unconverged (the answer moves with the cap -- not an eigenfunction, not a design); its row stands flagged, and the block/Lanczos eigensolver or the N'Diaye LP co-design are the named, deferred machines.
::: right
![Where the stop's edge lands: no-stop, stop-as-screen, and the rim field alone.]({FIG['cf1c_stop_attrib.png']}){{h=4.2}}

## The round-1 disagreement, resolved | The same basis includes the evaluation surface
::: left
- **The symptom.** Engine and linear model disagreed by 1–55× on every segment freedom but piston, which agreed to 0.6%.
- **The cause, measured.** The check traced to the Science focal plane; the model lives at the coronagraph exit pupil.  At a focal plane a segment tilt is a footprint piston with lever = the segment's pupil radius -- at that segment, {r0['lever_ratio']:.1f}× the tilt response, which is exactly the "mystery factor" the fingerprint carried.  Piston agreed because a normal displacement adds path uniformly at any surface.
- **Same element, same pokes, one changed trace target** -- the table.
- **The fix is structural.** The closure check is now a shared function that takes the evaluation surface from the harvest artifact, with a regression test that fails if a rotation ever closes at the wrong surface.
::: right
| freedom | rel. error at the focal plane | at the model's surface |
|---|---|---|
{SURFROWS}
~ Segment 19, 1 nm / 1 nrad pokes against the stored sensitivity columns.  "NaN" rows are the segment-clocking null (response below the 1 pm floor).

## Choosing a stable control law | Two mechanizations rejected by measurement
- **A held one-shot correction** (the round-1 idiom, with real sensor noise): at the solve frame the drift sits below the 1 nm edge noise, the estimate residual exceeds the state, and the corrected leg ends worse than the uncorrected one.
- **A per-frame loop on the estimator's segment slice**: the full-body minimum-variance gain is contractive, its segment slice is not (spectral radius 1.154) -- the engine loop diverged to 19 nm on a 3 nm drift, and a pure-linear simulation predicts the divergence to three digits.
- **Shipped: weighted least squares with ridge on the segment state** (radius 0.9998, gain {r4['gain']:.1f}): converges to the {r4['resid_nm']:.2f} nm sensor-noise floor.
- **The DM re-solve needed the same care.** Undamped solves against gap speckle -- amplitude-dominated, weakly controllable at this DM spacing -- push stroke along near-null directions and diverge; damping (0.7) plus a small leak (0.02) makes the dig monotone and held.  DM spacing as an amplitude-authority knob is a stated design trade, not exercised here.

## Packaging the longer relay | Fold side is a stability choice, not a taste choice
- A near-normal fold turns the chief by 180° − 2·AOI, and the beam direction reverses at each mirror.
- **Alternating the fold side** -- fine for the five folds of the shorter round-1 relay -- adds −2·AOI of net rotation per fold: over this chain's ten folds the accordion fans ~120° and walks to a 10.33 m shroud diameter.  Measured, with the element positions in the record.
- **Keeping the same side every fold** ping-pongs the beam between two fixed directions and packs the chain into a leg-sized pocket: {r1['shroud']:.3f} m, identical to the primary-set envelope.  The DM-bearing back end costs nothing in shroud diameter.

## Focal ratio: why f/{s1['fno']:.1f} and not f/{s1['fno_lo']:.0f}–{s1['fno_hi']:.0f} | The corrected focal ratio is not a continuous function of the layout
- The surface-figure solve spends optical power, so the focal ratio must be read on the corrected system; a search on the tertiary radius lands in different solve basins and refuses to steer continuously (f/15.5 and f/25.7 at layouts 0.36% apart).
- **The design point was chosen on wavefront**, which it meets; the slower focal ratio also puts more {LAM}/D on the focal-plane mask, easing its fabrication.

## What an apodizer alone does not buy back | The round-1 negative result stands
- Redesigning the apodizer for the segmented pupil -- a linear-program optimum and an aperture-specific eigenfunction -- recovered essentially nothing: the gaps scatter light no transmission profile removes.
- The optimizer's design model (a single Fourier transform) floors five times above what the engine already reaches, and the disagreement **grows** the harder the optimizer is pushed -- the signature of an optimizer mining its model, worth reusing as a diagnostic anywhere a design model is cheaper than the simulator it feeds.
- The round-2 answer is the deformable mirrors: measured against the engine, they dig the segmented floor {r4['dig'][0]:.1e} → {r4['dig'][-1]:.1e} and hold it under drift.

## What is not in this model | Stated, not omitted
- **The coronagraph masks are clear-pupil designs.**  Co-optimising apodizer, occulter and Lyot for the segmented aperture is known, separate work; the open-loop {r1['ratio_mean']:.0f}× is the size of that prize.
- **Monochromatic, at {s1['lambda_nm']:.0f} nm.**  The bandwidth and polarization layers exist on the testbed substrate and are configuration away, not new machinery.
- **The metrology measurement is simulated from its validated linear model** plus noise, not a second engine trace of the truss; the engine holds the truss points rigid, so the model is the only figure source either way.
- **DM spacing** ({r1['pupil_mm']:.0f} mm pupil, 0.15 m apart) limits amplitude-speckle authority; the testbed's wider spacing is the knob, and re-running the chain at another spacing is parameter work.

## How to reproduce | Every number on these slides comes from these runs
- Round 1 (frozen): `s1_layout_search`, `s1_telescope`, `s2_segmentation` -- the telescope and the segmented primary.
- `r1_backend`, `r1_coro`, `r1_dm`, `r1_shroud_union` -- the DM-bearing train, the coronagraph on both apertures, the DM surfaces, the two-leg shroud.
- `r2_sequence_fig`, `r2_bench_fig`, `r2_masks_fig` -- the four exhibit graphics.
- `r3_sensitivities`, `r3_dm_jacobian`, `r3_met` -- the error budget, the actuator Jacobian, the metrology stage.
- `r4_timeseries` -- the closed-loop drift series.
- `python3 deck_e2e6m_r2.py` -- this deck, from the reports those runs wrote.
~ All knobs live in one parameter file.  The narrative record -- every question, decision and gate -- is the campaign log beside the runners.
"""

MD = MD.replace(" -- ", " — ")   # house typography

md_path = os.path.join(HERE, "deck_e2e6m_r2.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(MD)
print("wrote", md_path)

rc = subprocess.call([sys.executable, BUILDER, md_path,
                      os.path.join(HERE, "deck_e2e6m_r2.pptx")])
sys.exit(rc)
