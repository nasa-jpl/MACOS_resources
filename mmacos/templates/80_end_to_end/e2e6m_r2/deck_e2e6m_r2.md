# A 6 m unobscured coronagraph, end to end — with the loop closed
An optical design, its segmentation, its instrument with deformable mirrors and metrology, its error budget and its corrected drift — one model, one train
~ Built with MACOS / mmacos.  Every number on these slides is parsed from the committed stage reports; no figure was redrawn for the deck.

Conventions, stated once.  Wavefront error is RMS at 500 nm at the exit pupil of the configuration being quoted — the telescope's own for the telescope-only slides, the coronagraph's once the instrument is attached (and every model-vs-engine check is evaluated at the surface its sensitivity model was harvested at).  Contrast is the dark-zone mean over 3–15 λ/D, normalised to the bare on-axis peak of the same train; static scoring at a 1024 grid, the drift series at 512 (its control operator's grid).  The diffraction limit is 0.071 waves RMS.  Packaging is a deployed, diameter-only fit in an 8 m launch shroud.

## The telescope | 0.0473 waves RMS across the field — diffraction-limited at 500 nm
::: left
- **The design.** A 6 m unobscured three-mirror telescope: three base spheres placed for packaging, fold tilts doing the unobscuration, all correction carried by Zernike surface departures.  Field ±0.35 arcmin.
- **The wavefront.** 0.0473 waves RMS worst case over the field, against a 0.071-wave diffraction limit.
- **The packaging.** 7.450 m diameter against the 8 m shroud; 1185/1185 rays survive a standalone reload of the saved prescription.
- **The one miss.** Focal ratio f/25.39 against a requested f/12–20; not a tuning failure — see the backup.
::: right
![The three-mirror train: light enters from the left, folds back to the secondary, returns past the primary to the tertiary and the focus.](deckfig/s1_layout.png){h=3.0}
![Wavefront error over the design field.  Worst corner 0.0473 waves.](deckfig/s1_wfe_field.png){h=3.1}

## The segmented primary | 19 segments, and a poke that stays where it is put
::: left
- **The segmentation.** 19 hexagonal segments, 1.2 m flat to flat, 25 mm gaps, each carrying the parent's solved figure and its own physical polygonal aperture.
- **The apertures are real.** 983 of 985 traced rays survive the declared polygons, and a standalone reload reproduces the count — the check that catches a wrongly-framed aperture.
- **The check that matters.** Displace one segment 10 nm along its normal: the wavefront responds 19.91 nm over the 52 rays on that segment and exactly zero over the other 930.
::: right
![Traced footprints against the emitted aperture polygons.  Colour is the segment a ray landed on; black is the declared glass.](deckfig/s2_segmented_footprints.png){h=4.6}

## The instrument, in light order | Eight relay mirrors, two deformable mirrors, five mask and pupil stations, two cameras
::: full
- **One train.** From the segmented primary through the fold train to the collimated pupil, then the coronagraph leg — DMs, apodizer, occulter, Lyot, field stop, back-end pupil, detector — with a deployable pick-off feeding the imager.  Spliced as 37 elements of one prescription; 983/985 rays, the telescope's own count.
![Light order through both instruments, read from the prescriptions.  Orange marks the deformable mirrors, purple the mask and pupil sites, dashed the focus stations.](deckfig/r2_sequence.png){h=3.9}

## Two instruments, one shroud | A second camera still costs nothing in diameter
::: left
- **The coronagraph leg** carries the science train: 37 elements, chief ray through the diffraction model agreeing with the geometric one to 5e-16.
- **The imager leg.** A deployable pick-off just after the collimator feeds its own f/19 camera: 0.0042 waves RMS, Strehl 0.9993 — unchanged from the round-1 stage that built it, because the pick-off sits upstream of everything round 2 added.
- **Both fit.** 7.451 m against the 8 m shroud for either leg and for the union; the 6 m primary sets the envelope.
::: right
![Both legs down the launch axis, inside the 8 m shroud circle.](deckfig/r1_union_shroud.png){h=4.5}

## The back end on its bench | Eight relay mirrors and two DMs in a 5.1 × 0.6 m accordion
::: left
- **The drawing is to scale**, in the plane the folds live in: the collimator, the DM pocket, then the mask relay — each mask plane sitting between its focus and pupil mirrors.
- **Near-normal folds** (6°) keep each off-axis section gentle; folding every mirror to the same side ping-pongs the beam into a bench-sized pocket.  The other choice fans the chain 120° and breaks the shroud — measured, in the backup.
- **In the full train** the back end is centimetre-class against the 6 m primary — the reason it gets its own drawing.
::: right
![The back end in its fold plane, to scale.  Orange: the DMs at the collimated pupil.  Purple: apodizer, Lyot and back-end pupil sites.  White: the occulter and field-stop foci.](deckfig/r2_back_plane.png){h=2.3}
![The full train for scale: the 6 m segmented primary and the beam it feeds.](deckfig/r2_train_iso.png){h=2.9}

## The masks, as physical objects | An occulter 2.8 λ/D across at a 500 nm focus is 27 µm of metal
::: left
- **The occulter** is an opaque disk of radius 2.8 λ/D at the intermediate focus — 27.3 µm physical radius at the measured 9.76 µm-per-λ/D focal scale, drawn from the same engine quantities the scoring chain uses.
- **The apodizer** is the clear-pupil prolate the coronagraph is scored with, shown as manufactured and as the light sees it — over the traced 19-segment pupil, gaps and all.
- Both figures are the scoring masks rebuilt from the scoring parameters, not illustrations.
::: right
![The apodizer: the prolate transmission profile (left) and the same profile over the traced segmented pupil (right), mm at the 47 mm pupil.](deckfig/r2_apodizer.png){h=2.6}
![The focal-plane occulter, transmission against λ/D; the physical radius is stated on the figure.](deckfig/r2_fpm_mask.png){h=2.7}

## The deformable mirrors are real | A 20 nm actuator poke lands where it is put, at twice the surface
::: left
- **The model.** Two 32×32-actuator DMs as grid-data surfaces in their own frames, 880 actuators each inside the 47 mm beam, influence functions with nearest-neighbour coupling.
- **The gate.** One actuator poked 20 nm, half a pupil radius off centre: the exit-pupil wavefront answers with 3.92e-08 m peak against 4e-08 expected (twice the surface, for a mirror), 100% of the response energy within 0.15 pupil radii of the peak, peak at 0.53 radii off centre.
- **Why it is a gate.** The failure mode of a wrongly-framed grid surface is a response painted at the pupil centre — localisation is the proof the actuator geometry is right.
::: right
![The single-actuator response at the exit pupil: localised, off-centre, 2× the commanded surface.](deckfig/r1_dm_poke.png){h=4.5}

## What the segment gaps cost | 1298× in dark-zone contrast, same mask on both apertures
::: left
- **The measurement.** The same apodized-pupil Lyot coronagraph on the segmented primary and on a monolithic twin of the identical telescope.
- **Segmented:** 4.705e-07 mean contrast, 6.18e+03 on-axis suppression.
- **Monolithic:** 3.624e-10 mean, 2.82e+07 suppression.
- **The ratio is the gaps and nothing else** — 1298× in the mean, 3809× in the median; both trains share the same wavefront.
- These are open-loop numbers: the DMs are flat here.  The loop slide prices what they buy.
::: right
![Radial contrast, both apertures, dark zone shaded.  The segmented curve plateaus where the monolithic one keeps falling — that plateau is gap-scattered light.](deckfig/r1_contrast.png){h=4.6}

## The error budget closes | Engine vs model: worst 0.35% over a segment, a DM and a relay mirror, all six freedoms
::: left
- **The model.** Wavefront sensitivity to every rigid-body freedom of 31 optics — the segments, the fold mirrors, the relay and both DMs — plus segment figure modes and a per-segment influence basis: 192 + 152 + 114 channels over five field points.
- **The closure check.** Poke a segment, a DM and a relay mirror through the engine and compare with the model, at the surface the model was harvested at: worst relative error 0.0035 over 18 freedom pairs.  The matching check in the previous build disagreed by factors up to 55; the resolution is in the backup, and it was not the model.
- **The assembly exhibit.** The 19 segments perturbed as one body against one segment alone: tilt adds up (ratio 16.9), piston cancels (ratio 0.028), clocking is invisible per segment and real for the assembly.
::: right
| freedom | assembly response | assembly / one segment |
|---|---|---|
| Rx | 2.619 | 16.88 |
| Ry | 2.506 | 16.12 |
| Rz | 0.2475 | 9.324e+04 |
| Tx | 0.06566 | 16.2 |
| Ty | 0.06706 | 16.6 |
| Tz | 0.0136 | 0.0282 |
~ Column RMS of wavefront change per unit motion; rotations per radian, translations per metre.
![Singular values of the three Jacobians, largest normalised to one.](deckfig/r3_svspec.png){h=2.6}

## The metrology truss | 114 laser gauges + 252 edge sensors, validated against the engine to 0%
::: left
- **The layout.** Six launchers per segment on its true boundary, fiducials on the secondary's rim, plus inter-segment edge sensors — the committed metrology stage run on this telescope with this train's sensitivity model.
- **What it buys.** Post-control wavefront residual 0.86 nm per nm of gauge noise with edges and gauges together, against 0 for gauges alone — the edge sensors carry the segment-to-segment state.
- **Validated.** The layout merit's finite-difference check closes at 0%, and a 200-draw Monte-Carlo estimate/control loop lands within 2.6% of the analytic residual.
::: right
![The as-built truss: beams from the 19 segments to the hub fiducials; edge sensors on the segment boundaries.](deckfig/r3_met_layout.png){h=4.5}

## The loop closes | The dark zone held under drift: 2.5e-07 against 1.7e-06 open-loop
::: left
- **The drift.** 41 frames over 7 minutes: random walk plus correlated drift on every freedom of all 19 segments, played through the engine on the full train.
- **The rigid-body loop.** Edge and gauge readings, a weighted least-squares estimate, an integrating controller on all six freedoms of every segment: the state residual holds at 0.36 nm while the open-loop drift grows to 2.9 nm.
- **The DM loop.** The measured actuator Jacobian digs the dark zone 4.6e-07 → 1.9e-07, then a damped re-solve at each scored frame holds it: 1.91e-07 → 2.46e-07 closed, against 4.65e-07 → 1.72e-06 open.
- **The honest line.** Closed-loop pupil wavefront is larger than uncorrected — that is the DM stroke the contrast is bought with, and the figure labels it as such.
::: right
![State, wavefront and contrast against time.  The contrast panel is the payoff: open loop degrades 3.7×; the closed loop holds.](deckfig/r4_series.png){h=4.7}

## What this demonstrates | One model, from surface figure to a held dark zone
- **A design becomes an instrument becomes an error budget becomes a controlled observatory**, without leaving the model or re-entering the geometry anywhere.
- **The gap penalty is measured, not assumed** — 1298× open loop — and the control loop that answers it is measured too: 1.7e-06 open against 2.5e-07 closed at the end of the soak.
- **Every model-vs-engine comparison closes** — worst 0.0035 across segments, DMs and relay optics — because the comparison is made at the surface the model lives on.  Where a build disagreed, the deck says why and what fixed it.
- **All of it is committed**: prescriptions, runners, reports, and this deck's numbers are one `git clone` away.

## Backup Slides

## The round-1 disagreement, resolved | The same basis includes the evaluation surface
::: left
- **The symptom.** Engine and linear model disagreed by 1–55× on every segment freedom but piston, which agreed to 0.6%.
- **The cause, measured.** The check traced to the Science focal plane; the model lives at the coronagraph exit pupil.  At a focal plane a segment tilt is a footprint piston with lever = the segment's pupil radius — at that segment, 6.5× the tilt response, which is exactly the "mystery factor" the fingerprint carried.  Piston agreed because a normal displacement adds path uniformly at any surface.
- **Same element, same pokes, one changed trace target** — the table.
- **The fix is structural.** The closure check is now a shared function that takes the evaluation surface from the harvest artifact, with a regression test that fails if a rotation ever closes at the wrong surface.
::: right
| freedom | rel. error at the focal plane | at the model's surface |
|---|---|---|
| Rx | 56.9 | 3.34e-05 |
| Ry | 1.01 | 3.48e-05 |
| Rz | 1.04 | NaN |
| Tx | 1.01 | 0.00131 |
| Ty | 54.8 | 0.00131 |
| Tz | 0.00614 | 1.06e-05 |
~ Segment 19, 1 nm / 1 nrad pokes against the stored sensitivity columns.  "NaN" rows are the segment-clocking null (response below the 1 pm floor).

## Choosing a stable control law | Two mechanizations rejected by measurement
- **A held one-shot correction** (the round-1 idiom, with real sensor noise): at the solve frame the drift sits below the 1 nm edge noise, the estimate residual exceeds the state, and the corrected leg ends worse than the uncorrected one.
- **A per-frame loop on the estimator's segment slice**: the full-body minimum-variance gain is contractive, its segment slice is not (spectral radius 1.154) — the engine loop diverged to 19 nm on a 3 nm drift, and a pure-linear simulation predicts the divergence to three digits.
- **Shipped: weighted least squares with ridge on the segment state** (radius 0.9998, gain 0.5): converges to the 0.36 nm sensor-noise floor.
- **The DM re-solve needed the same care.** Undamped solves against gap speckle — amplitude-dominated, weakly controllable at this DM spacing — push stroke along near-null directions and diverge; damping (0.7) plus a small leak (0.02) makes the dig monotone and held.  DM spacing as an amplitude-authority knob is a stated design trade, not exercised here.

## Packaging the longer relay | Fold side is a stability choice, not a taste choice
- A near-normal fold turns the chief by 180° − 2·AOI, and the beam direction reverses at each mirror.
- **Alternating the fold side** — fine for the five folds of the shorter round-1 relay — adds −2·AOI of net rotation per fold: over this chain's ten folds the accordion fans ~120° and walks to a 10.33 m shroud diameter.  Measured, with the element positions in the record.
- **Keeping the same side every fold** ping-pongs the beam between two fixed directions and packs the chain into a leg-sized pocket: 7.451 m, identical to the primary-set envelope.  The DM-bearing back end costs nothing in shroud diameter.

## Focal ratio: why f/25.4 and not f/12–20 | The corrected focal ratio is not a continuous function of the layout
- The surface-figure solve spends optical power, so the focal ratio must be read on the corrected system; a search on the tertiary radius lands in different solve basins and refuses to steer continuously (f/15.5 and f/25.7 at layouts 0.36% apart).
- **The design point was chosen on wavefront**, which it meets; the slower focal ratio also puts more λ/D on the focal-plane mask, easing its fabrication.

## What an apodizer alone does not buy back | The round-1 negative result stands
- Redesigning the apodizer for the segmented pupil — a linear-program optimum and an aperture-specific eigenfunction — recovered essentially nothing: the gaps scatter light no transmission profile removes.
- The optimizer's design model (a single Fourier transform) floors five times above what the engine already reaches, and the disagreement **grows** the harder the optimizer is pushed — the signature of an optimizer mining its model, worth reusing as a diagnostic anywhere a design model is cheaper than the simulator it feeds.
- The round-2 answer is the deformable mirrors: measured against the engine, they dig the segmented floor 4.6e-07 → 1.9e-07 and hold it under drift.

## What is not in this model | Stated, not omitted
- **The coronagraph masks are clear-pupil designs.**  Co-optimising apodizer, occulter and Lyot for the segmented aperture is known, separate work; the open-loop 1298× is the size of that prize.
- **Monochromatic, at 500 nm.**  The bandwidth and polarization layers exist on the testbed substrate and are configuration away, not new machinery.
- **The metrology measurement is simulated from its validated linear model** plus noise, not a second engine trace of the truss; the engine holds the truss points rigid, so the model is the only figure source either way.
- **DM spacing** (47 mm pupil, 0.15 m apart) limits amplitude-speckle authority; the testbed's wider spacing is the knob, and re-running the chain at another spacing is parameter work.

## How to reproduce | Every number on these slides comes from these runs
- Round 1 (frozen): `s1_layout_search`, `s1_telescope`, `s2_segmentation` — the telescope and the segmented primary.
- `r1_backend`, `r1_coro`, `r1_dm`, `r1_shroud_union` — the DM-bearing train, the coronagraph on both apertures, the DM surfaces, the two-leg shroud.
- `r2_sequence_fig`, `r2_bench_fig`, `r2_masks_fig` — the four exhibit graphics.
- `r3_sensitivities`, `r3_dm_jacobian`, `r3_met` — the error budget, the actuator Jacobian, the metrology stage.
- `r4_timeseries` — the closed-loop drift series.
- `python3 deck_e2e6m_r2.py` — this deck, from the reports those runs wrote.
~ All knobs live in one parameter file.  The narrative record — every question, decision and gate — is the campaign log beside the runners.
