# A 6 m unobscured coronagraph, end to end — with the loop closed
An optical design, its segmentation, its instrument with deformable mirrors and metrology, its error budget and its corrected drift — one model, one train
~ Built with MACOS / mmacos.  Every number on these slides is parsed from the committed stage reports; no figure was redrawn for the deck.
~ DRAFT (2026-08-27): the coronagraph-family slides (families / floors / Lyot trade) await Dave's review of the CF1c-CF3a numbers.

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

## Six coronagraphs, one train | Pre-control: the apodized Lyot leads at 4.4e-07; the vortex pays the gaps ~40x
::: left
- **The design rule that made them comparable.**  On a segmented hex aperture the coronagraph's pupil is a CIRCULAR STOP at the apodizer plane — the telescope's hex envelope is upstream optics, not the coronagraph pupil.  Every family below sees the same circularized, gapped pupil; contrast normalizes to its bare peak, and the stop's ~8% collecting-area cost sits in the throughput column.
- **The families (dark-zone mean, PRE-CONTROL | throughput):** classical Lyot 3.50e-06 pre-control, 23% throughput; apodized Lyot 4.37e-07 pre-control, 9% throughput; APLC-as-implemented 2.18e-06 pre-control, 8% throughput (cap-limited prolate, not design-grade on this pupil; family DEFERRED); band-limited 1.64e-06 pre-control, 12% throughput; vortex c4 1.64e-05 pre-control, 33% throughput; c6 1.87e-05 pre-control, 33% throughput — the vortex passes the segment-gap light the occulting families block.
- **The stop's 2.16x on the bare occulter, attributed by measurement:** entirely the stop edge at fixed masks (the scale re-reference contributes 1.00x); the Babinet split shows the edge enters by coherent INTERFERENCE with the gap field (cross 6.5e-06 vs the rim's own 2.5e-06) — coherent, hence in the loop's reach.  The next slide cashes that.
::: right
![The six families, statics on the circularized segmented train.](deckfig/cf1_families.png){h=4.6}

## The loop on every family | All six floors sit within 2x of their own Jacobian's linear-achievable: the substrate speaks, not the controller
::: left
- **static -> relin floor | linear-achievable at the achieved stroke:** classical Lyot 3.5e-06 -> 3.9e-07 | 2.1e-07; apodized Lyot 4.5e-07 -> 1.1e-07 | 1.1e-07 (the campaign floor); APLC-as-implemented 1.1e-06 -> 7.5e-07 | 7.5e-07 (implementation verdict, not family); band-limited 1.6e-06 -> 1.3e-06 | 7.2e-07; vortex c4 1.6e-05 -> 5.4e-06 | 5.5e-06; c6 1.8e-05 -> 3.9e-06 | 4.7e-06.
- **The stopped occulter digs 8.9x where its no-stop self dug 5.2x** — the loop claws back most of the coherent edge term the previous slide measured.
- **The vortices are linear-optimal:** relinearization pays 2.5-3.7x where the fixed Jacobian paid 1.2x (gap-chasing strokes reach the linearity boundary early), and the residual gap leak is uncontrollable at this amplitude authority.
- **The knob is the DM spacing:** z/z_T = 3.7e-03 at the outer working angle — Talbot-weak amplitude authority is the common ceiling; the spacing trade prices it.
::: right
![Convergence per family (relinearization joins at the dotted lines) and the closed-loop column.](deckfig/cf2_floors.png){h=4.6}

## The Lyot trade | Throughput is bought with contrast on every leg; the operating points were chosen, not defaulted
::: full
- **The sweep.**  Lyot fraction against contrast AND throughput for the vortex legs and the apodized-Lyot leg, statics under the stop; stars mark the S1 operating points the campaign scored.
![Contrast vs Lyot fraction (left) and the contrast-throughput trade with the family operating points (right).](deckfig/cf3a_lyot.png){h=4.2}

## Both dials lose to the operating point | Lyot 0.90 and 0.15 m spacing were confirmed by measurement, not defaulted
::: left
- **The gate.**  The campaign's operating point — apodized Lyot, 1.08e-07 floor at 9.5% throughput, spacing 0.15 m — was put to both of its dials before the physics layers ran.
- **The Lyot dial stalls closed-loop.**  The static dial is flat (previous slide), but at L=0.98 the loop stops in two iterations at 1.58e-07 — 1.46x the contrast for 1.19x the throughput, a net loss in a throughput/contrast merit — and the wide-ladder retry does not dig.  Static-free is not loop-free.
- **The spacing dial's knee IS the baseline.**  Floors 1.08e-07 / 4.98e-07 / 9.67e-07 / 9.00e-07 at 0.15/0.40/0.70/1.10 m: the Talbot expectation INVERTS — wider spacing raises the linear-achievable bound and the loop cannot take it, because the STATIC degrades faster than the authority grows.  Attributed by measurement: the added light is the same amplitude-type gap-Fresnel family (symmetric fraction 0.986 at every spacing), evolved over the longer DM1-to-DM2 leg.  Packaging never discriminates (7.451 m shroud at every point, measured).
::: right
![The Talbot knob measured: closed-loop floor and linear-achievable vs DM spacing — the bound deepens as the floor worsens.](deckfig/cf3b_spacing.png){h=4.4}

## The physics layers at the operating point | Polarization does not set the floor; the chromatic penalty is 1.3x over a 20% band; the layers do not interact
::: left
- **Polarization.**  All 31 reflectors coated (protected aluminum), Jones-pupil screens at the exit pupil, complex-mean normalized.  The co-polarized imprint is 9.5% — twenty times the CTB's gentle train — yet almost entirely common-mode: the screened floor equals the unscreened one, and the UNCONTROLLABLE polarization floor is 4.4e-13.
- **Bandwidth.**  One 9-color superset Jacobian, per-band block subsets: floors 1.62e-07 / 1.64e-07 / 1.72e-07 / 2.07e-07 at 0/5/10/20% — the floor is gap-speckle-owned, not chromatic (the inverse of the CTB vortex, whose floor was chromaticity).
- **Together.**  10% band + screens: 1.72e-07 — the band-only floor to three digits.  The mask memoization behind the sweep is gated bit-exact (backup).
::: right
![The physics ladder at the gate point: polarization, the band ladder, and band+pol together.](deckfig/cf4_physics.png){h=4.4}

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
- **Toned down 10x (0.3 nm-class drift): the hold is the mechanization's.**  The open loop no longer degrades (4.65e-07 -> 4.63e-07) while the closed loop holds 2.46e-07 — identical to the 3 nm hold: the ~2.5e-07 level is the loop's own noise-injection floor, and control cadence/gain becomes the knob at low drift (flagged, not retuned; the estimator still tracks at 7.2e-11 against 3.0e-10 of drift).
::: right
![State, wavefront and contrast against time.  The contrast panel is the payoff: open loop degrades 3.7×; the closed loop holds.](deckfig/r4_series.png){h=4.7}

## The restart ladder | The controller was the bottleneck: 1.2e-06 → 1.13e-09 at d = 1.10 m, with the substrate still pricing 3.5e-11
::: left
- **The recipe (rinse and repeat):** EFC to a floor, relinearize about the dug state, restart at that floor.  10 digging rounds over 4.6 h unattended; 3 stall rounds then prove the plateau — no accepted step at any Tikhonov α down to 1e-10, and two independent G measurements at the unchanged state agree to four digits.
- **What it removed:** the single-linearization stall at 9.0e-07 (the spacing sweep's d = 1.10 m entry) — four decades above the linear bound at this spacing.
- **What remains, priced:** la(G) holds 2.0e-11–3.8e-11 at every dug state; the plateau at 1.13e-09 is the monotone full-Tikhonov step rule — the measured case for FALCO-grade step machinery.  Strokes end at 33 nm rms of the 50 nm bound.
::: right
![The ladder: achieved floor by restart round against the linear-achievable substrate re-measured at every dug state.  The early linear claims (rounds 1–2) are flat-state optimism; they settle once the linearization is honest.](deckfig/cf3d_dig.png){h=4.5}

## Inside the dig, station by station | The controller does the gap work: the DM commands imprint the hex-gap lattice
::: full
![Seven planes, DMs flat against the dug state: pupil, apodizer, FPM before/after the occulter, Lyot before/after the stop, science-plane contrast.  The in-walk dark-zone means reproduce the scored record exactly (1.22e-06 / 1.13e-09).](deckfig/cf3d_stations_wide.png){h=2.8}
- **Reading the strip:** the flat row's segment-lattice speckle becomes the dug row's sculpted field; the science panel shows the full 3–15 λ/D annular dark hole carved to the 1e-9 floor.
- **Two measured readables:** DM1's command map carries the hex-gap lattice — the DMs are doing the gap work themselves (the no-apodizer A/B is the scoped follow-on); and the 0.90 Lyot stop removes only ~0.1% of the post-FPM energy — this train's rejected light is interior gap structure, not edge rings.
![The dug command state: DM1 imprints the gap lattice; strokes ~33 nm rms, peaks ~160 nm.](deckfig/cf3d_dm_state.png){h=1.7}

## What this demonstrates | One model, from surface figure to a held dark zone
- **A design becomes an instrument becomes an error budget becomes a controlled observatory**, without leaving the model or re-entering the geometry anywhere.
- **The gap penalty is measured, not assumed** — 1298× open loop — and the control loop that answers it is measured too: 1.7e-06 open against 2.5e-07 closed at the end of the soak.
- **Every model-vs-engine comparison closes** — worst 0.0035 across segments, DMs and relay optics — because the comparison is made at the surface the model lives on.  Where a build disagreed, the deck says why and what fixed it.
- **All of it is committed**: prescriptions, runners, reports, and this deck's numbers are one `git clone` away.

## Backup Slides

## The stop's edge, measured | A Babinet split with 1e-15 closure: the edge interferes, it does not merely add
::: left
- **Method.**  The circular stop applied as a SCREEN on the no-stop chain holds every mask and scale fixed, so E_rim = E_hex - E_screened is exactly the blocked rim's field.  Pins: screened bare peak == stopped bare peak to 0.0e0; both S1 records reproduced under 1%.
- **The split (dark-zone mean energy):** I_ns 1.09e-05 + rim 2.53e-06 + cross 6.54e-06 — the rim's own ring is ~28% of the increase; the cross term (coherent interference with the pre-existing gap speckle) is 2.6x larger.  Peak renormalization 1.178x; dark-zone energy up 1.83x at fixed masks; the lambda/D re-reference contributes 1.00x.
- **The APLC flag, restated:** the stopped aperture-matched prolate hit its iteration cap unconverged (the answer moves with the cap — not an eigenfunction, not a design); its apodizer also moves 2x with grid size (1.07e-6 at N=512 vs 2.18e-6 at 1024) — an implementation verdict, not a family verdict; the family is DEFERRED and the block/Lanczos eigensolver or the N'Diaye LP co-design are the named machines.  Bound reading (all rows): lin-ach is the relin G's rank-curve at the achieved stroke — a scale, not a strict inequality; Tikhonov solutions can undercut it by O(20%) (vortex c6 does).
::: right
![Where the stop's edge lands: no-stop, stop-as-screen, and the rim field alone.](deckfig/cf1c_stop_attrib.png){h=4.2}

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
- `cf_chain` + `cf1_families`..`cf4_physics` — the coronagraph-family campaign; `cf3d_deepdig` + `cf3d_stations` — the restart ladder and its station graphics.
- `python3 deck_e2e6m_r2.py` — this deck, from the reports those runs wrote.
~ All knobs live in one parameter file.  The narrative record — every question, decision and gate — is the campaign log beside the runners.
