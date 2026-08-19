# Developer-only files — strip from public `main`, keep on `dev`

Manifest of developer/agent process files in the **MACOS_resources** repo that
should be **removed from the public `main` branch** but **preserved on `dev`**
for the 2026 public-release split.  These hold agent instructions and internal
audits — *not* external user documentation (READMEs, interface docs, example
guides) and *not* functional code/fixtures.

This file is itself dev-only — it appears in its own strip list below.
`release-exclude.txt` is the machine-readable mirror of the strip list
(one path per line), consumed by the strip step:
`git rm -r -q --pathspec-from-file=release-exclude.txt`.
Keep the two in sync; the .txt is also dev-only.

Re-audited 2026-08-06 against `dev` + `pol-ifo` (union — covers the
pol-ifo→dev merge).  Additions marked (2026-08-06) below.

## Strip list (paths from repo root)

```
DEV_FILES.md
release-exclude.txt
GMI/CLAUDE.md
mmacos/CLAUDE.md
pymacos/CLAUDE.md
segmirmaker/CLAUDE.md
segmirmaker/.vscode/launch.json
segmirmaker/.vscode/tasks.json
doc/mmacos_pymacos_parity.md
doc/STYLE_REPORTS.md
mmacos/design/PLAN_AFOCAL4.md
mmacos/design/PLAN_TMA_E2E2.md
mmacos/challenges/afocal4/STATUS_S4B.md
mmacos/challenges/afocal4/STATUS_S4C.md
mmacos/challenges/rodgers1/OPTFEX_REDO_LIST.md
mmacos/challenges/rodgers1/SESSION_STATE_2026-07-30.md
mmacos/templates/40_benches/vsg_wip/PLAN_VSG2_MODELS.md
mmacos/templates/40_benches/vsg_wip/VSG2_review.md
mmacos/templates/30_instruments/bench_ctb/CTB_PROP_STATUS.md
mmacos/templates/40_benches/bench_ifo_dm/l2_trade/PLAN_IFO_PUPIL_RELAY.md
mmacos/templates/90_polarization/bench_ifo_pol/CURRENT_SLICE.md
optical_design/CORONAGRAPH_DESIGN_AGENT_GUIDE.md
optical_design/OPTICAL_DESIGN_AGENT_GUIDE.md
optical_design/CORONAGRAPH_DESIGN_RULES.md
optical_design/TELESCOPE_DESIGN_REFERENCE.md
optical_design/AGENT_NOTES.md
optical_design/coronagraph_layout.md
optical_design/fixtures/telescope_design_fixtures.md
optical_design/fixtures/tma_fixture.md
optical_design/fixtures/afocal_tma_fixture.md
```

2026-08-06 additions and why:
- `doc/STYLE_REPORTS.md` — internal report/deck style gate (agent process).
- `mmacos/design/PLAN_*.md` — sprint plans (working state).
- `afocal4/STATUS_S4B.md`, `STATUS_S4C.md` — session status blocks.
- `rodgers1/OPTFEX_REDO_LIST.md`, `SESSION_STATE_2026-07-30.md` — working
  lists / session state.
- `vsg/PLAN_VSG2_MODELS.md`, `VSG2_review.md` — plan + agent-voiced review
  transcript.
- `bench_ctb/CTB_PROP_STATUS.md` — work-in-progress hand-off record
  (SESSION blocks; the user-facing content lives in the bench_ctb README
  and Coro_propagation_summary, both KEPT).
- `bench_ifo_dm/l2_trade/PLAN_IFO_PUPIL_RELAY.md`,
  `bench_ifo_pol/CURRENT_SLICE.md` — plan / in-flight state.
- `optical_design/fixtures/afocal_tma_fixture.md` — matches its two
  stripped sibling fixture notes.

## `optical_design/` — KEEP the code + fixtures (FUNCTIONAL, do NOT strip)

The shipped MATLAB design layer (`mmacos/+macos/+design/Telescope.m`,
`seidel_seed.m`) and the test suite (`tDesignTelescope` via
`mmacos/tests/private/design_fixture_path.m`) are ported/validated against
these — stripping them breaks tests:

- `optical_design/fixtures/{telescope_design_fixtures,tma_fixture,jwst_anchor}.json`
- `optical_design/{seidel,coronagraph_layout,make_fixtures,make_tma_fixture}.py`  (provenance tooling — KEPT per Dave)
- `optical_design/coronagraph_layout.json`

Only the agent-voiced `.md` docs in `optical_design/` are stripped (above).
`optical_design/README.md` is a user-facing README documenting the kept
fixtures/tools and **stays on `main`**; the previous agent-voiced README was
preserved as `optical_design/AGENT_NOTES.md` (dev-only, in the strip list).

## Explicitly KEPT on `main` (external documentation — do NOT strip)

- `README.md`, `GMI/README.md`, `mmacos/README.md`, `pymacos/README.md`
- `mmacos/doc/README.md`, `mmacos/doc/elt_grid_add.md` (interface doc)
- `mmacos/design/README.md` + `mmacos/design/runners/README.md`
- `mmacos/templates/00_INDEX.md` + `mmacos/templates/*/*/README.md`
- `mmacos/challenges/README.md`
- `mmacos/sensitivities/README.md`
- `mmacos/tools/fex_sweep/README.md`
- `GMI/regression/README.md`, `mmacos/tests/README.md`, `pymacos/tests/proper_compare/README.md` (test-run guides — user-usable)

2026-08-06 additions (KEPT — study records and technical documentation):
- `mmacos/challenges/rodgers1/{PACKET,README}.md`,
  `mmacos/challenges/rodgers2/{PACKET,README}.md` — benchmark-study records,
  retractions in place (the study precedent).
- `mmacos/challenges/afocal4/{FORM_STUDY,RESULTS,README}.md` — study
  records, same family as the packets.
- `mmacos/templates/80_end_to_end/e2e2/E2E2_REPORT.md` + e2e2/relay_followon READMEs
  — worked-example report.
- `mmacos/templates/30_instruments/bench_ctb/{README,Coro_propagation_summary}.md`
  + `bench_ctb/dst2/README.md` — house rules + the diffraction recipe of
  record (technical documentation, not process).
- `mmacos/templates/40_benches/bench_ifo_dm/l2_trade/TRADE_NOTE.md` — trade-study
  record (result-first).
- `mmacos/tools/pol_{external_anchor,sp_sign_probe,validation_report}/README.md`
  — tool run guides.
- `*.fp.json` fingerprint sidecars anywhere — functional (large-file
  policy), never stripped.

## Not tracked here (won't be on `main` *or* `dev` unless committed to `dev`)

- Agent memory lives OUTSIDE both repos at
  `~/.claude/projects/-home-dcr-dev-macos/memory/` (`MEMORY.md` + `memory/*.md`).
