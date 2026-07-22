# Developer-only files — strip from public `main`, keep on `dev`

Manifest of developer/agent process files in the **MACOS_resources** repo that
should be **removed from the public `main` branch** but **preserved on `dev`**
for the 2026 public-release split.  These hold agent instructions and internal
audits — *not* external user documentation (READMEs, interface docs, example
guides) and *not* functional code/fixtures.

This file is itself dev-only — it appears in its own strip list below.

## Strip list (paths from repo root)

```
DEV_FILES.md
GMI/CLAUDE.md
mmacos/CLAUDE.md
pymacos/CLAUDE.md
segmirmaker/CLAUDE.md
segmirmaker/.vscode/launch.json
segmirmaker/.vscode/tasks.json
doc/mmacos_pymacos_parity.md
optical_design/CORONAGRAPH_DESIGN_AGENT_GUIDE.md
optical_design/OPTICAL_DESIGN_AGENT_GUIDE.md
optical_design/CORONAGRAPH_DESIGN_RULES.md
optical_design/TELESCOPE_DESIGN_REFERENCE.md
optical_design/README.md
optical_design/coronagraph_layout.md
optical_design/fixtures/telescope_design_fixtures.md
optical_design/fixtures/tma_fixture.md
```

## `optical_design/` — KEEP the code + fixtures (FUNCTIONAL, do NOT strip)

The shipped MATLAB design layer (`mmacos/+macos/+design/Telescope.m`,
`seidel_seed.m`) and the test suite (`tDesignTelescope` via
`mmacos/tests/private/design_fixture_path.m`) are ported/validated against
these — stripping them breaks tests:

- `optical_design/fixtures/{telescope_design_fixtures,tma_fixture,jwst_anchor}.json`
- `optical_design/{seidel,coronagraph_layout,make_fixtures,make_tma_fixture}.py`  (provenance tooling — KEPT per Dave)
- `optical_design/coronagraph_layout.json`

Only the agent-voiced `.md` docs in `optical_design/` are stripped (above).
Note: after the strip the kept `.py`/`.json` lose their (agent-voiced) README;
write a short user-facing one on `main` if desired.

## Explicitly KEPT on `main` (external documentation — do NOT strip)

- `README.md`, `GMI/README.md`, `mmacos/README.md`, `pymacos/README.md`
- `mmacos/doc/README.md`, `mmacos/doc/elt_grid_add.md` (interface doc)
- `mmacos/design/README.md` + `mmacos/design/examples/*/README.md` + `mmacos/design/runners/README.md`
- `mmacos/examples/*/README.md` (coro, coro_walkthrough, design, view_rx_demo)
- `mmacos/sensitivities/README.md` + `mmacos/sensitivities/examples/*/README.md`
- `mmacos/tools/fex_sweep/README.md`
- `GMI/regression/README.md`, `mmacos/tests/README.md`, `pymacos/tests/proper_compare/README.md` (test-run guides — user-usable)

## Not tracked here (won't be on `main` *or* `dev` unless committed to `dev`)

- Agent memory lives OUTSIDE both repos at
  `~/.claude/projects/-home-dcr-dev-macos/memory/` (`MEMORY.md` + `memory/*.md`).
