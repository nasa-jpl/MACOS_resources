# Polarization validation report — figure & number driver

Regenerates the evidence for
`macos/docs/macos-manual/polval/POLARIZATION_VALIDATION` (the multi-section
report built by `make polval`). Specified in `macos/PLAN_POLARIZATION.md`,
"Validation document" deliverable.

```
./make_polval.sh [polvalDir]        # measure + render, one command
```

`polvalDir` defaults to `<macos repo>/docs/macos-manual/polval`, resolved
from this script's own location (`macos` and `MACOS_resources` are siblings
under `~/dev`). Equivalent entry point from the docs side:

```
cd ~/dev/macos/docs/macos-manual
make polval-regen     # this driver
make polval           # docx + HTML
make polval-pdf       # PDF
```

## The contract

**No number in the report is typed by hand.** The prose lives in
`polval/*.md.in` and contains substitution placeholders only; the rendered
`polval/*.md` are generated. Three mechanisms keep that honest:

| Mechanism | Catches |
|---|---|
| `render_polval.py` resolves every token before writing anything | a token with no measurement behind it — and never leaves a half-rendered `.md` on disk |
| `check_polval.py` (runs as a prerequisite of `make polval`) | a `.md.in` edited without re-rendering; a figure newer than `numbers.json`; a leftover placeholder |
| provenance block stamped into the report | which engine/binding commit, model size, MATLAB and host the numbers came from |

## Files

| File | Role |
|---|---|
| `run_pol_validation.m` | MATLAB driver — runs one model size's cases, writes `media/*.png` + `generated/parts/numbers_<size>.json` |
| `merge_numbers.py` | merges the per-size parts into `generated/numbers.json` |
| `render_polval.py` | substitutes measured numbers into `polval/*.md.in` → `polval/*.md` |
| `external.json` | gates this box cannot run, with their producing command and capture date |
| `make_polval.sh` | the one command (driver per size, merge, render) |

## Per-model-size batches

`macos_init_all()` corrupts the heap across `model_size` transitions
(`mmacos/CLAUDE.md`), so the driver runs **one size per MATLAB process**:

| Size | Cases | Why that size |
|---|---|---|
| 128 | Phase 1, Phase 2a/2b, Phase 3a Tranche 1, the r_p sign fix | historical; `Rx_Cass_FarField` is exercised here as the landed tests do |
| 256 | Phase 2c exactness gates | `Rx_VecChain` and `Rx_Cass_FarField` both declare `nGridpts=256` |
| 512 | Phase 2c coronagraph chain | `Rx_Coro` declares `nGridpts=511` — it only *appears* to run at 128 |

Each run writes `generated/parts/numbers_<size>.json` (gitignored — an
intermediate); `merge_numbers.py` combines them into the committed
`generated/numbers.json`. The merge refuses to proceed when two parts define
the same token (which part wins would depend on file ordering) or when their
provenance disagrees (parts from different sessions must not be stitched into
one report).

**Adding a size:** a new branch in `run_pol_validation`'s `switch`, its own
block in `gate_limits()`, and the size added to `MODELS` in `make_polval.sh`.
Gate thresholds are per size, so a token listed in one size's table but never
measured by that size's group fails the run — the same way a regressed value
does.

## What the driver measures, and what it cannot

The driver covers Phase 1 exposure gates, Phase 2a/2b Jones-pupil physics,
Phase 3a Tranche 1 vector-chain gates, the `Reflector` s/p sign fix, and the
Phase 2c contrast floor — everything reachable from mmacos on this box.

Four classes of number are **not** reachable from here and live in
`external.json` instead, each with the command that produces it and the date
it was captured. The report labels every one of them *(external, captured
DATE)* so a reader can tell which numbers `polval-regen` refreshed:

* the pymacos/ifx suite and the PROPER comparisons (other binding, other
  compiler);
* the GMI regression (other repository, long runtime);
* the pre-fix engine A/B numbers, which are **historical** — reproducing them
  means rebuilding the engine with `propsub.F` from before `a0f6dba` and is
  not possible from the current tree.

The pre-fix numbers are worth keeping precisely because they are what makes
the gates demonstrably non-vacuous. If you re-capture them, rebuild the
pre-fix engine by copying `propsub.F` aside and restoring it afterwards
rather than using `git stash` — a stash is easy to drop by accident, and the
copy is also what auto permission mode allows without prompting.

## Constraints

* **One model size, one MATLAB session.** See *Per-model-size batches* above
  — this is why the driver is invoked once per size rather than once.
* **Every batch invocation ends `exit(0)`.** `matlab -batch` hangs at
  implicit process exit once a mex has been loaded.
* **Needs a current mex.** After any change to `macos_api_mod.F90` or the
  engine, rebuild and relink before regenerating, or the report will stamp
  the new commit while measuring the old binary. The provenance block records
  the commit, not the binary — it cannot detect this for you.

## Adding a gate

1. Measure it in `run_pol_validation.m` and record it with `addval(V, NAME,
   value, fmt, units, gate, test)`. The `gate` and `test` strings are what the
   report's gate index cites, so name the test that actually pins it in CI.
2. Reference `@@NAME@@` from the relevant `polval/*.md.in` section: the
   claim, the figure, the measured number, the truth it is compared against.
3. If it produces a figure, write it with `savefig_` so it lands in `media/`
   and participates in the freshness check.
4. Re-run `make_polval.sh`. An unresolved token fails the render; a figure
   newer than the numbers fails the build.

Per the standing rule in `PLAN_POLARIZATION.md`, every phase's
definition-of-done includes its evidence section here, alongside its cmdref
and manual entries.
