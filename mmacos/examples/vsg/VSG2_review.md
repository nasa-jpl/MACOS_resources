● Review: what VSG2 is

  A Twyman-Green interferometer for measuring DM surface figure, in a vacuum chamber: frequency-stabilized HeNe → SM fiber → baffle → L1 collimator (Newport PAC097AR.14 achromat, D=76.2 mm, EFL 750 mm) →
  BS → test arm (DM, which is the system pupil) / reference arm (flat on a PZT stage for temporal phase shifting) → recombine → L2 (PAC095AR.14, EFL 250 mm) → two folds → ND → Andor NEO sCMOS (2560×2160,
  6.5 µm pixels), focused on the DM surface.

  Three things worth noting from the numbers:

  - The conjugates check out exactly. DM→L2 ≈ 1175 mm and L2→camera ≈ 317 mm satisfy the thin-lens equation for f=250 mm to within a millimeter (1/(1/250 − 1/1175) = 317.6). Magnification ≈ 0.27, so the
  camera's 14.0×16.6 mm active area sees a ~52×62 mm field at the DM — generous sampling for any plausible DM.
  - There's a free focal plane waiting for the ZWFS. The source is collimated, so L2 forms a point-source focus at 250 mm — about 67 mm before the camera, which sits at the pupil conjugate. A λ/4 phase
  dimple dropped at that internal focus (with the reference arm blocked) turns the exact same camera into a Zernike wavefront sensor with essentially no re-layout. That's almost certainly why the slide
  says the ZWFS upgrade is planned, and it makes your task 2 a clean incremental design.
  - Two small issues in the slide: the layout figure labels L1 "700mm EFL" while the text and the PAC097 part number say 750 mm (typo, presumably); and the "Beam splitter:" bullet is empty — the BS spec
  (substrate thickness, wedge, coating, whether there's a compensator plate in the reference arm) is exactly the thing that drives the dominant arm-asymmetry bias in a Twyman-Green with a converging
  output, so we'll need it.

  How the toolchain maps onto this

  The pieces we need already exist and are validated:

  - DM as a grid surface — elt_grid/GridData with the dw/dgrid influence-function machinery (segment_grid_basis, write_grid_file). The pymacos Phase 6b/6c work already validated MACOS DM phase imprints
  against PROPER, including the finding that post-trace imprinting differs from real ray-trace through the deformed mirror at the ~1e-3–1e-2 level — directly relevant to gauge accuracy.
  - Two-arm interference — MACOS traces sequentially, so the interferometer becomes two Rx (test path, reference path) sharing common elements; grab complex_field at the camera from each and sum in user
  space: I = |E_test + E_ref·e^{iδ}|². The PZT shift is a real perturb Tz on the reference flat (λ/8 steps), so beam-walk during the shift is captured, not idealized.
  - ZWFS mask — apodize_complex at the internal focus, reusing the Cycle-5 vortex coronagraph oversized-rays machinery, which solved exactly this focal-plane-mask problem.
  - Tolerancing — the dw_dx rigid-body Jacobian drivers give alignment/drift sensitivities of the gauge itself for free.
  - Servo sim — the s7 RBCS closed-loop drift/recontrol simulator is the template for task 3's active DM loop.

  Plan

  Phase 0 — data + as-built model. Collect the missing hardware data (list below). Fetch the real PAC097/PAC095 doublet prescriptions from Newport (radii/glasses — published) or start paraxial-ideal and
  refine. Build the two-arm Rx pair with the DM as a grid surface, verify conjugates and pupil mapping with view_rx/view_layout, and validate the nominal fringe pattern (null fringe, then tilt fringes).

  Phase 1 — performance analysis (task 1). Synthesize the full measurement: inject a known DM figure, run the phase-shift sequence through the coherent two-arm model, apply the actual PSI algorithm
  (4/5-bucket), unwrap, divide by 2 — and difference against the injected truth. That end-to-end transfer function is the gauge performance. Break out the error budget: DM→camera pupil distortion (mapping
  error at high spatial frequencies), retrace error at large DM slopes, BS substrate asymmetry between arms, camera sampling vs actuator print-through frequencies, phase-shifter miscalibration, and
  alignment-drift sensitivities from dw_dx. Deliverable: a VSG2 error-budget report plus a reusable "simulated gauge" driver.

  Phase 2 — ZWFS variant (task 2). Model the λ/4 dimple (~1.06 λ/D diameter) at the 250 mm internal focus; build the reconstructor and the calibration curve. The interesting trade is that the ZWFS is
  common-path (no reference arm → vibration-insensitive, photon-efficient) but small-dynamic-range (~±λ/4), while the IFO is absolute and large-range — so "in addition" likely wins: IFO for
  absolute/bootstrap, ZWFS for fast low-noise figure sensing. Quantify the crossover (noise vs stroke) with matched simulations on the same DM figures.

  Phase 3 — face-on on-orbit gauge (task 3). Concept study first: the flight-proven pattern is a ZWFS fed by light the coronagraph already rejects (Roman LOWFS-style), vs a dedicated injected metrology
  beam at normal incidence to the DM. First-order layouts through optical_design/ conventions (mm+nm units, coronagraph rules), then a closed-loop sim: G-matrix from dw/dgrid influence functions, ZWFS
  sensor model from Phase 2, drift model, closed-loop residual → contrast impact through the existing e2e coronagraph model. This phase deliberately builds on Phases 1–2 so the sensor model is already
  validated when the servo work starts.

  What I need from you when we start Phase 0: DM make/actuator count/pitch/aperture/stroke; the BS spec (and whether there's a compensator); reference-flat figure quality; fiber NA or beam diameter at the
  DM; the actual PSI algorithm in use; and any exact distances beyond the two on the slide. I've saved all of this — the system summary, the plan, and the missing-data list — to project memory so it
  carries across sessions.

✻ Cogitated for 3m 9s

❯ Won't the return from the Ref Mirror -- which generates fringes -- need to be blocked in ZWFS mode?                                                                                                        

● Yes — that's exactly why I'd put "reference arm blocked" in the ZWFS concept, and it's worth being precise about why and how, because it drives a small hardware change.

  Why it must be blocked, not just ignored. The ZWFS reads pupil intensity, and any coherent reference return adds at the field level: I = |E_test + E_ref|². The reference beam focuses onto the same phase
  dimple (it's also a collimated beam feeding L2), so it produces its own ZWFS-transformed pupil pattern (a static bias you could in principle calibrate) plus the full-contrast cross term — which is just
  the Twyman-Green fringe pattern again. That cross term moves with every picometer of arm-length drift, so vibration and thermal OPD between the arms would read as DM figure change. Leaving the reference
  on would throw away the single biggest advantage of the ZWFS mode: common-path immunity.

  Options, roughly in order of preference:

  1. Shutter/flag in the reference arm — standard, clean, total extinction. The one wrinkle is that the bench is in a vacuum chamber, so it needs a remotely actuated flag (or you accept a chamber cycle to
  reconfigure, which kills the "switch modes on the fly" use case).
  2. Tilt the reference mirror off and block at the internal focus. If the "Ref Mirr & PZT Stage" has tip/tilt range, a tilt θ walks the reference focus 2θ·250 mm sideways at the ZWFS mask plane — 2 mrad
  buys 1 mm. A field stop there (or just a small ZWFS mask substrate/aperture) passes the test-arm core and dumps the reference spot. No new actuator if the stage already tilts, but edge scatter off the
  stop needs a look.
  3. PZT phase-washing (ramp the PZT through ≥2π during each exposure): averages the cross term to zero but leaves the incoherent reference background — extra shot noise, reduced signal contrast, and the
  reference's own ZWFS bias. Only attractive as a zero-hardware-change quick experiment, not as the operating mode.

  What this adds to the Phase 2 plan: the simulation should include a leakage case, not just the ideal test-arm-only Rx. With amplitude leakage ε past the blocker, the drifting cross term enters at
  ~2ε·|E_test||E_ref| — running that through the ZWFS reconstructor against the sensor noise floor gives a required extinction spec (shutter quality, or tilt+stop geometry) as a concrete deliverable. The
  two-Rx model makes this trivial: it's the same coherent sum we build for IFO mode, with E_ref scaled by ε.

  Note the flip side too: in IFO mode the ZWFS dimple would sit in the common path after recombination, so if it's left permanently installed it phase-marks both arms near their shared focus — mostly
  common-mode but not exactly (the two arms' foci land on it with slightly different geometry after any DM figure change). Cleanest is to make the ZWFS mask insertable/removable along with the reference
  blocker — one two-position mechanism can swap {mask out + ref open} ↔ {mask in + ref blocked}, which also keeps the mode switch to a single moving part in the chamber.
