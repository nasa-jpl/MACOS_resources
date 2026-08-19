# PLAN — mmacos templates/challenges reorganization (2026-08-18, Dave + CC, rev 2)

Driver: the Keysight (CodeV team) demo ~2026-09-01.  **Keysight will
access the repo**, so the reorganization happens BEFORE the demo
(Dave's call, rev 2 — supersedes rev 1's post-demo phasing).

Framing (Dave): most of what we ship are **design templates** —
parameterized starting points a user copies and adapts; the rodgers
cases are **design challenges** — a target spec with our worked
answer.  Whether template or challenge, **these folders must not house
universal helpers or drivers**: those belong in a rationally organized
library, and the folders point to them.

The demo threads:
  T1 telescopes of increasing capability/complexity
  T2 segmentation + interface coords
  T3 instruments (imager, coronagraph)
  T4 linear-model sensitivities (dw/dx, dw/dz, dw/dgrid)
  T5 visualization of x, z, grid errors
  T6 time-sequence simulation
Polarization: kept, not a Keysight driver.

## Survey findings (2026-08-18; details in rev-1 record)

1. Two mirrored trees, both live and test-referenced:
   `design/examples/` vs `examples/design/`, and
   `sensitivities/examples/` vs `examples/sensitivities/`.
2. Library code hiding in example dirs — exactly the violation Dave's
   principle targets:
   - `examples/sensitivities/` carries `mimg/m2v/v2m/pad` and is on
     the MATLAB path via `mmacos_setup.m`;
   - `examples/coronagraph/coro/` carries the ported contrast-scoring
     machinery, consumed by `tCoroContrast` via PathFixture.
3. `sensitivities/run_dwd*.m` drivers resolve their Rx decks by
   directory-NAME matching — moving an asset dir silently breaks its
   driver.
4. Two already-stale path refs (`gs_zernike_segment_basis.m:15`;
   tRunSensitivities/tRunMet absolute cross-repo anchors) — the
   failure mode careless moves multiply.
5. `examples/vsg` is planning material, not runnable;
   `tma_unobscured` is a superseded trade study (README says so).
6. Coupling checklist (every referencing file) — §Move checklist.

## Target tree

```
mmacos/
  src/+macos/            ALL universal helpers (display, contrast
                         scoring, ...) — the library proper
  design/src/            design kernels (strict WFE, layouts, ...)
  design/runners/        the stage-driver library (run_sensitivities,
                         run_met, run_compare, run_simulator, presets)
  templates/
    00_INDEX.md          the guided tour, ordered T1->T6 (doubles as
                         the demo script skeleton)
    10_telescopes/       rc_onaxis, rc_unobscured, rc_offaxis,
                         tma_onaxis, tma_offaxis, tma_centered,
                         tma_widefield, wf2_freeform, tma_freeform,
                         sz_tma, freeform_unobscured, tma_3plus1
                         (tma_unobscured kept, README marked
                         "superseded by freeform_unobscured")
    20_segmentation/     e5_pie, e5_seg, grid_surface
    30_instruments/      coro_walkthrough, coro_planet_demo, bench_ctb
                         (imager rung = pointer into 80_end_to_end/e2e
                         s2 until a standalone template is extracted)
    40_benches/          bench_layout, bench_ifo, bench_ifo_dm
                         (vsg parked here as vsg_wip/ — planning only,
                         README says so)
    50_sensitivities/    e5hex1, the run_dwd* asset dirs,
                         gen_segment_gridmat, e5hex2_refzern
    60_visualization/    view_rx_demo
    70_simulation/       pointer into e2e s6/s7; standalone driver if
                         the rewalk extracts one
    80_end_to_end/       e2e, e2e2 — the flagship flows, kept whole
    90_polarization/     bench_ifo_pol (+ future pol demos)
  challenges/
    rodgers1/  rodgers2/  afocal4/
    README.md            what a challenge is: the spec, our answer,
                         the scoring rules
```

`examples/` and `design/examples/` cease to exist; no compatibility
symlinks (users re-cloned recently; Keysight arrives fresh — one tree,
no ghosts).

## Sequencing — two weeks

**Week 1 = the reorganization**, on `dev-candidate` (AFTER the
in-flight gates pass and the Luis pushes land — Luis re-pulls when
this lands; do not block his GridData fix on the reorg).

Step 0 — library rationalization (each its own commit, suite-gated):
  a. Hoist `mimg/m2v/v2m/pad` -> `src/+macos/`; drop the
     `mmacos_setup.m` addpath.
  b. Hoist the contrast-scoring helpers out of
     `examples/coronagraph/coro/` -> `src/+macos/`; re-point
     `tCoroContrast`.
  c. Make every `run_dwd*` driver's deck path explicit CONFIG (kill
     directory-name matching).
     **PRESERVED SURFACE (Dave 2026-08-18): the `run_dwd*` framing
     stays.**  The `sensitivities/run_dwd*.m` runner family
     (run_dwdx_multi, run_dwdz_multi, run_dwdgrid_multi +
     single/multisegbasis, run_dwdsurf_multi) is IN USE — the e2e s4
     harvests were built by copying these tested configurations, and
     user workflows call them by name.  The reorg may move their asset
     dirs (that is what the explicit CONFIG enables) and may relocate
     the runners into the driver library, but their NAMES and call
     forms do not change; if their directory moves, thin same-name
     forwarders remain at `sensitivities/` for one release.
  d. Fix the two stale refs + the absolute cross-repo anchors
     (repo-relative).
Step 1 — `git mv` per the target tree, ONE thread-directory per
  commit, updating that commit's slice of the move checklist.
Step 2 — tree-wide grep for every old path: zero live hits.
Step 3 — full `run_mmacos_tests.sh`: 0 fail, with asset-gated classes
  (tCtbProp, the rodgers1-gated kernel tests) EXECUTING, not skipping
  — a skip here means the gate silently lost its assets (the
  segmirmaker_bin lesson).  GMI regression untouched.

**Week 2 = the rewalk on the NEW tree** (near-release-quality bar):
  e2e s1–s7 rerun on the merged engine (expect the documented
  post-pupil-fix count shifts), e2e2, rodgers1 PACKET Addendum-10
  reconciliation, coro_walkthrough, view_rx_demo, bench_layout, T1
  ladder spot-runs; write `templates/00_INDEX.md` and
  `challenges/README.md`; add the missing one-paragraph READMEs
  (rc_onaxis, tma_freeform, sz_tma, wf2_freeform, coro_planet_demo,
  e5hex1); GMI/README.md stale ff_hex line.  Regenerated artifacts
  land at their FINAL paths — the rewalk validates the reorg.

## Move checklist (every known referencing file, from the coupling sweep)

- tests: tCtbProp (asset gate path), tCoroContrast (PathFixture),
  tE2E2Axial, tFingerprint, tAfocal4, tDwDx, tAfocalKernel,
  tStrictKernel, tPupilMap, tOptFex, tPupilAperture,
  tRunSensitivities, tRunMet, tests/README.md
- build/config: mmacos_setup.m, mmacos/.gitignore (4 explicit paths:
  examples/sensitivities glob, 2× e2e .mat, bench_ctb .mat),
  run_mmacos_tests.sh (bench_ctb gate comment)
- src: gs_zernike_segment_basis.m, twyman_green.m, Bench.m (See also)
- docs: design/runners/README.md, examples/design/README.md (absorbed
  into 00_INDEX.md), doc/ (verified clean)
- sensitivities: the 6 run_dwd*.m deck paths, save_dw_multi.m
- release-exclude.txt / DEV_FILES.md: verify no example paths in the
  strip list move out from under it (challenges/ and templates/ stay
  public; PLAN_* files are auto-stripped as before)

## Decisions taken (Dave, 2026-08-18)

- Reorg BEFORE the demo (Keysight accesses the repo).
- Template/challenge framing; folders point into the library, never
  house it.
- Numbered prefixes: yes.
- Benches: a numbered rung in the ladder (40_benches), not a side bin.

## Still open

- Name check: `templates/` + `challenges/` at mmacos root (proposed
  here) vs everything under one `templates/` root — current proposal
  keeps challenges separate because their contract (spec + worked
  answer) differs from copy-and-adapt.
- Where the imager-instrument rung's standalone template comes from
  (extract e2e s2's Offner relay into 30_instruments/ during the
  rewalk, if time allows).
- vsg: parked as 40_benches/vsg_wip — confirm or move to doc/plans.
