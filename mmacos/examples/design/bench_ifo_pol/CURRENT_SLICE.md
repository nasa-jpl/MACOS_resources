# IFO Polarization Slice 1 — CURRENT STATE (2026-07-27, post-rewrite)

**Branch:** `pol-ifo` (MACOS_resources), off `bench-builder`, merged `ifo-l2` + `pol-core`
**Engine:** macos `pol-core`, built gfortran @ `build_release_gfortran` (DONE this session)
**mmacos mex:** rebuilt this session (`unset FC && make`)

---

## IMPORTANT: the pre-compaction harness was WRONG and has been REWRITTEN

The first-pass `example_bench_ifo_pol.m` (before this session) was structurally
broken — do NOT restore it:
- fabricated a 9-output `ray_field` signature (real one returns a STRUCT
  `.Ex .Ey .Ez .kx .ky .kz .nx .ny .nz .status`)
- hand-rolled a `local-sp` s/p decomposition (pol_maps documents it as
  artifact-prone) instead of using the gated `jones_pupil`/`pol_maps` layer
- used a scalar per-component ratio `E_test./E_ref` instead of the 2×2
  matrix arm-differential `M = J_test · inv(J_ref)`
- Gate 1 assumed NORMAL incidence; the BS is at 45° AOI

The Bench coating-passthrough (`coating_idx/ext/thk` on add_mirror/add_bs_reflect
and `coating_bs` on twyman_green) was also REMOVED — it applied `macos.coating`
during in-memory bench construction, before the Rx exists in the engine
(`push` only appends to `b.E`; the engine sees nothing until `.emit()`+`load_rx`).
Coatings are now applied AFTER `load_rx` in the example.  Bench.m /
twyman_green.m are back to their pre-slice state (verify with `git diff`).

## Current harness design (CORRECT approach)

`example_bench_ifo_pol.m`:
1. Build uncoated TG, emit both arm .in files.
2. Coated-face element indices from the in-memory bench:
   test arm `BSrefl` (external air→Al reflect, elt 4),
   ref arm `BScrefr` (internal glass→Al reflect, elt 8) — DIFFERENT Jones.
3. **Gate 1** — single-surface 45° bare-Fresnel analytic on the test-arm BS
   reflection, mirroring `tests/tJonesPupil.m:test_fold_fresnel_analytic`
   (textbook Born&Wolf r_s/r_p, non-circular in the r_p sign).
4. Arm Jones pupils at recombination via `jones_pupil` (double-pole basis);
   ref arm forced into the test arm's exit basis (`'axis'`,`'xref'`).
   Differential `M = J_test·inv(J_ref)` → `pol_maps` → D / retardance.
5. PSI phase-error contribution = pupil variation of co-pol fringe phase.
6. **Gate 2** — pol-off bit-identity: coating inert with pol OFF (<1e-12 mm OPD).

## STATUS: BOTH GATES GREEN, PACKET WRITTEN, HOLDING FOR REVIEW

- **Gate 1 PASS** — RS/RP magnitude resid 1.13e-14, phase resid 2.97e-14
  (< 1e-12).  The L1-diattenuation diagnosis was correct: fixed by using
  the measured field INCIDENT on the BS (`ray_field(iBS-1)` from a
  separate trace) as the reference input.
- **Gate 2 PASS** — pol-off OPD bit-identity, 0.000e+00 mm both arms.
- **Result:** arm-differential D mean 7.21e-2 / retardance 8.35e-2 rad;
  pupil VARIATION 2–4e-8 (round-off, collimated common-path); PSI phase
  error 2.3e-6 nm RMS @ 632.8 nm.
- **Packet:** `~/dev/macos/REVIEW_POL_IFO_SLICE1_2026-07-27.md` written.
- **Two findings in the packet:** (1) the Bench builder stamps `Extinc=1e22`
  on transmitting Refractors = opaque glass under ifPol (invisible on the
  scalar path; fixed in-example, Bench.m policy = Dave's call); (2) Gate 1
  must use the field incident on the BS, not the source launch state.

## NEXT STEP (resume here)

- **HOLD FOR DAVE'S REVIEW before push** (he asked; first pass was wrong).
  When cleared: `cd ~/dev/MACOS_resources`, add the example dir + the macos
  packet, commit, `git push -u origin pol-ifo`.  Slices 2 (BS AOI trade) +
  3 (polarizing PSI) after review.

## Files touched (this branch)

- `mmacos/examples/design/bench_ifo_pol/example_bench_ifo_pol.m` — REWRITTEN harness
- `mmacos/examples/design/bench_ifo_pol/CURRENT_SLICE.md` — this file
- `mmacos/src/+macos/+design/Bench.m` — coating passthrough REMOVED (back to baseline)
- `mmacos/src/+macos/+design/twyman_green.m` — coating_bs REMOVED (back to baseline)

## Known traps

- **Mac FC env:** shell has `FC=gfortran-16`; mmacos Makefile rejects. `unset FC && make`.
- **Coatings need a loaded Rx:** apply `macos.coating` AFTER `load_rx`, never during
  Bench construction. `load_rx` CLEARS coating state (re-coat after every load).
- **RayE is per-element overwritten:** to read incident + reflected fields you need
  TWO traces (to iBS-1 and to iBS).
- **No `macos.elt_name` / `macos.n_elt`:** use `{G.bt.E.name}` and `macos.num_elt()`.
- **Reference for Gate 1:** `tests/tJonesPupil.m:test_fold_fresnel_analytic` is the
  proven single-surface 45° Fresnel gate — copy its analytic, don't reinvent.
