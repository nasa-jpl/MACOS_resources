# A 6 m unobscured coronagraph, end to end
An optical design, its segmentation, its instrument, its error budget and its drift — one model, one train
~ Built with MACOS / mmacos.  Every number on these slides is parsed from the committed stage reports; no figure was redrawn for the deck.

Conventions, stated once.  Wavefront error is RMS at 500 nm, referenced to the exit pupil of the configuration being quoted — the telescope's own for the telescope-only slides, the coronagraph's once the instrument is attached.  Contrast is the dark-zone mean over 3–15 λ/D, normalised to the bare on-axis peak of the same train.  The diffraction limit is 0.071 waves RMS.  Packaging is a deployed, diameter-only fit in an 8 m launch shroud; length is free.

## The telescope | 0.0473 waves RMS across the field — diffraction-limited at 500 nm
::: left
- **The design.** A 6 m unobscured three-mirror telescope: three base spheres placed for packaging, fold tilts doing the unobscuration, all correction carried by Zernike surface departures. Field ±0.35 arcmin.
- **The wavefront.** 0.0473 waves RMS worst case over the field, against a 0.071-wave diffraction limit.
- **The packaging.** 7.450 m diameter against the 8 m shroud, over a 17.37 m train; 4/4 bodies clear of every beam; 1185/1185 rays survive a standalone reload of the saved prescription.
- **The one miss.** Focal ratio f/25.39 against a requested f/12–20. It is not a tuning failure — see the backup.
::: right
![The three-mirror train: light enters from the left, folds back to the secondary, returns past the primary to the tertiary and the focus.](deckfig/s1_layout.png){h=3.0}
![Wavefront error over the design field. Worst corner 0.0473 waves.](deckfig/s1_wfe_field.png){h=3.1}

## The segmented primary | 19 segments, and a poke that stays where it is put
::: left
- **The segmentation.** 19 hexagonal segments, 1.2 m flat to flat, 25 mm gaps, each carrying the parent's solved figure and its own physical polygonal aperture.
- **The apertures are real.** Each segment declares its polygon, and 983 of 985 traced rays survive them; the two that clip land on a gap edge. Reloading the saved prescription standalone reproduces 983/985 — the check that catches a wrongly-framed aperture, which otherwise fails silently on disk.
- **The check that matters.** Displace one segment by 10 nm along its normal: the wavefront responds 19.91 nm over the 52 rays that land on it and **exactly zero** over the other 930. The response stays on the segment that moved — which is what makes a per-segment error budget mean anything.
::: right
![The segmented telescope: 19 hexagons on the primary, feeding the same fold train.](deckfig/s2_iso.png){h=2.9}
![Traced footprints against the emitted aperture polygons. Colour is the segment a ray landed on; black is the declared glass.](deckfig/s2_segmented_footprints.png){h=3.2}

## Two instruments | A second camera costs nothing in shroud diameter
::: left
- **The coronagraph.** A four-mirror relay off the telescope focus: collimate to an accessible pupil, focus to the mask, re-collimate to the Lyot stop, focus to the detector. Spliced onto the segmented telescope as 29 elements of one prescription — not a second model. Only a single model carries a telescope perturbation through to a contrast number, which is what stages four and five need.
- **The imager.** A deployable pick-off at the shared pupil feeds its own f/19 camera: 0.0042 waves RMS, Strehl 0.9993, and a 0.87 µm geometric spot against a 9.5 µm diffraction core — the image is limited by diffraction, not by the optics.
- **Both fit.** 7.451 m against the 8 m shroud, for either leg and for the two together. The 6 m primary sets the envelope; the instruments are centimetre-class.
- **The models are faithful.** The chief ray through the diffraction model agrees with the geometric one to 1.2e-15 m, and the imager leg loses no rays the telescope did not already lose (983/985).
::: right
![Both legs down the launch axis, inside the 8 m shroud circle. The instrument optics are the inset — at this scale they are a few pixels.](deckfig/s3_imager_shroud.png){h=4.6}

## What the segment gaps cost the coronagraph | 2390× in dark-zone contrast, same mask on both apertures
::: left
- **The measurement.** The same apodized pupil Lyot coronagraph — same apodizer, same 2.8 λ/D occulter, same Lyot stop — run on the segmented primary and on a monolithic version of the identical telescope.
- **Segmented:** 8.700e-07 mean contrast, 6.17e+03 on-axis suppression.
- **Monolithic:** 3.640e-10 mean, 2.82e+07 suppression.
- **The ratio is the gaps and nothing else** — 2390× in the mean, 8573× in the median. Both trains share the same wavefront; only the pupil's structure differs.
- The monolithic result independently reproduces the reference testbed's clear-pupil number, with nothing tuned to match it.
::: right
![Radial contrast, both apertures, dark zone shaded. The segmented curve plateaus where the monolithic one keeps falling — that plateau is light scattered by the gaps.](deckfig/s3_contrast.png){h=4.6}

## The error budget | Moving 19 segments as one body is not 19 times moving one
::: left
- **The model.** Wavefront sensitivity to every rigid-body freedom of 25 optics, to each segment's figure modes, and to a per-segment influence basis: 156 + 152 + 114 channels over five field points.
- **The exhibit.** The 19 segments were also perturbed as a single rigid body. Comparing that with a single segment's response separates three different behaviours — in one table.
- **Tilt** adds up: ratio 19.3, the member count. A per-segment budget is right.
- **Piston** cancels: ratio 0.032. Moving all 19 together is nearly a global piston, which the reference removes — a per-segment budget overstates the assembly by about 31×.
- **Clocking** is invisible per segment and real for the assembly: ratio 111004.
- **Not every freedom is observable.** The 114 segment rigid-body channels span a range of 2.16e+06 in singular value: a wavefront measurement constrains far fewer directions than there are freedoms, which is what makes the correction in the next slide a solved problem rather than an inversion.
::: right
| freedom | assembly response | assembly / one segment |
|---|---|---|
| Rx | 2.756 | 19.32 |
| Ry | 2.74 | 19.17 |
| Rz | 0.2706 | 1.11e+05 |
| Tx | 0.07168 | 19.24 |
| Ty | 0.07054 | 19 |
| Tz | 0.01437 | 0.0322 |
~ Column RMS of wavefront change per unit motion; rotations per radian, translations per metre, the same convention on both sides.
![Singular values of the three Jacobians, largest normalised to one. Rigid-body motion (blue) collapses by seven decades; the figure and influence bases stay within one.](deckfig/s4_svspec.png){h=3.1}

## The observatory drifts | Contrast tracks the wavefront, 8.60e-07 to 2.08e-06
::: left
- **The history.** 41 frames at 10 s — a 7-minute soak — with a random walk plus a correlated drift on every freedom of all 19 segments, played through the engine.
- **Uncorrected:** wavefront 0.0046 to 0.0243 waves; contrast 8.601e-07 to 2.078e-06.
- **Corrected:** 0.0035 to 0.0231 waves; contrast 9.213e-07 to 1.990e-06. One image-based correction, solved early and held.
- **Why the benefit fades.** The correction is never updated, so it decays as the state walks away from where it was solved. That is the shape of a held correction, not a modelling artefact.
- Each contrast point is a full diffraction propagation through the mask chain, so the timeline is sampled every 5 frames (9 of 41).
::: right
![Wavefront and contrast against time, corrected and uncorrected.](deckfig/s5_series.png){h=4.7}

## What this demonstrates | One model, from surface figure to science contrast
- **A design becomes an instrument becomes an error budget becomes a time series**, without leaving the model or re-entering the geometry anywhere.
- **The gap penalty is measured, not assumed**: 2390×, from two runs that differ in one thing.
- **Assembly-level freedoms behave differently from their members** — by a factor 31 in one direction and 111004 in another — and the model shows which.
- **Every gate is in the record.** Where something did not close, the deck says so and prices it.
::: right
![The modelled observatory: the 19-segment primary and the beam it feeds, in one prescription. The secondaries and the instrument optics are decimetre- and centimetre-class and do not resolve at 6 m scale.](deckfig/s3_iso.png){h=4.4}

## Backup Slides

## Focal ratio: why f/25.4 and not f/12–20 | The corrected focal ratio is not a continuous function of the layout
- The surface-figure solve spends optical power, so the focal ratio of the corrected system differs from that of the base spheres. It has to be read on the corrected system.
- A search on the tertiary radius refused after five iterates. Two neighbouring layouts, 0.36% apart in that radius, gave f/15.54 and f/25.74; widening the sweep gave f/20.09, f/8.56, f/26.90, f/25.39 at four nearby values.
- The solve lands in different basins depending on the starting geometry, and each spends a different amount of power. No continuous search steers it.
- Narrowing the design field does not help either: at fixed geometry, halving the field made the residual worse.
- **The design point was therefore chosen on wavefront**, which it meets. A slower focal ratio also puts more λ/D on the focal-plane mask, which eases the hardest fabrication in the instrument.

## Linear model against the engine | Only piston reproduces; control was restricted to it
::: left
- Poking each freedom and comparing the engine's wavefront change with the sensitivity model:
- Piston agrees to better than 1%. Every other freedom disagrees, by factors of 1–57.
- A ladder over three decades of poke size separates two behaviours: the centre segment's tilt response is purely second-order, while an outer segment's is linear but 3.8× the model. Neither is a noise floor.
- Ruled out by measurement: the wavefront reference. Repeating the ladder with the other reference reproduces every number to five digits.
- **Consequence.** With all six freedoms in the control basis the corrected leg came out worse than the uncorrected one. Control was restricted to the freedom that reproduces; the drift still moves all six.
::: right
| freedom | engine | model | relative error |
|---|---|---|---|
| Rx | 2.482e-12 | 1.419e-10 | 56.9 |
| Ry | 9.257e-10 | 1.426e-10 | 1.01 |
| Rz | 7.950e-14 | 9.395e-15 | 1.04 |
| Tx | 2.389e-11 | 3.682e-12 | 1.01 |
| Ty | 6.640e-14 | 3.679e-12 | 54.8 |
| Tz | 4.466e-10 | 4.440e-10 | 0.00614 |
~ RMS wavefront change, in metres, for a 1 nm / 1 nrad poke of a single segment. Open item: two untested candidates remain — the focal-plane tracking used when the sensitivities were harvested, and the real-ray stop aiming, which would explain why the centre segment — the one the chief ray lands on — is the anomalous case.

## What the segment gaps cost, and what does not buy it back | The redesign recovers 1.10×
::: left
- **The attempt.** Redesign the apodizer for the segmented pupil two ways: the globally optimal transmission mask from a linear program, and the aperture-specific version of the same eigenfunction the incumbent uses.
- **Neither beats the incumbent.** The aperture-specific mask recovers 1.10× in dark-zone mean and 0.57× in median, at 0.73× the throughput. No rung of the linear program beat it either.
- **Why, measured.** The design model is a single Fourier transform; the real back end is a multi-leg chain. It reproduces the bare image to 1.2% but floors near 4e-06 against the engine — five times above the contrast the incumbent already reaches.
- **A negative result, priced.** The gap penalty is real and an apodizer alone does not remove it. What would fix it is a design operator matching the propagation it is scored against — queued, not attempted here.
::: right
| configuration | dark-zone mean | throughput |
|---|---|---|
| bare segmented (baseline) | 8.700e-07 | 0.100 |
| aperture-specific mask | 7.909e-07 | 0.073 |
| linear-program mask | 1.505e-06 | 0.175 |
| clear-pupil reference | 3.640e-10 | 0.100 |
~ Same plane, same mask geometry, same 3–15 λ/D annulus as the main-path number.

## How to tell an optimizer is mining its model | The disagreement grows 4.8× → 40.0× as you push it
::: left
- **The design model and the engine are not the same code.** One is a single Fourier transform, the other propagates the real multi-leg chain. Ask the optimizer for a deeper dark zone and it spends the difference.
- **The pattern is the diagnosis.** A fixed offset would be a calibration error. An offset that grows monotonically with the demand is the optimizer finding more of the model's error the harder it is pushed — and it shows up without knowing the true answer.
- **Isolated in six experiments.** The bare image agrees to 1.2%; the engine applies the mask with residual exactly zero; every diffraction ring lands at the right radius; and feeding the engine's own post-mask field back through the model still lands on the model's floor.
- **Worth reusing.** Any time a design model is cheaper than the simulator it feeds, run the ladder and watch the trend rather than a single agreement number.
::: right
| LP target | model says | engine says | apart by |
|---|---|---|---|
| 1e-05 | 1.13e-06 | 5.37e-06 | 4.8× |
| 3e-06 | 1.92e-07 | 2.61e-06 | 13.6× |
| 1e-06 | 3.77e-08 | 1.50e-06 | 40.0× |
~ Dark-zone mean, bare-peak normalised. The model's floor near 4e-06 sits five times above the 8.70e-07 the incumbent already reaches.

## What is not in this model | Stated, not omitted
- **No deformable mirrors and no field stop.** The reference testbed model carries both. Without them the contrast here is an open-loop number — what the optics deliver, not what a wavefront-control loop would hold — and the dark zone is not the annulus two deformable mirrors would give.
- **No metrology loop.** The correction in the drift series is image-based: it sees the wavefront directly rather than estimating it from a truss. The corrected leg is therefore an optimistic bound.
- **The coronagraph mask is a clear-pupil design.** Re-optimising it for a segmented aperture is a known and separate piece of work; the 2390× above is the size of the prize.
- Both additions are existing primitives in the toolkit, so the extension is configuration rather than new machinery.

## How to reproduce | Every number on these slides comes from these runs
- `s1_layout_search` then `s1_telescope` — the layout pick and the telescope.
- `s2_segmentation` — the segmented primary and its edge-sensor sidecar.
- `s3_backend`, `s3_train_fig`, `s3_coro`, `s3_imager` then `s3b_pupil` + `s3b_apodizer` — the two instruments, the coronagraph on both apertures, and the apodizer redesign.
- `s4_sensitivities` — the error budget, roughly half an hour.
- `s5_timeseries` — the drift series.
- `python3 deck_e2e6m.py` — this deck, from the reports those runs wrote.
~ All knobs live in one parameter file. The narrative record — every question raised, every decision and its result, every gate outcome — is in the campaign log beside the runners.
