# sens_core_ab — deck-agnostic A/B harness for the sens-core merge gate

Extends the sens-core acceptance evidence (10 run-sets on e5hex1, all
byte-identical) to arbitrary prescriptions — the JPL-private pass CCMac
runs before the merge to dev-candidate.  Tasking brief:
`macos/BRIEF_ccmac_sens_core.md`.

## Usage (one MATLAB process per invocation — the state-leak doctrine)

```bash
# PRE tree = a worktree at the last pre-sens-core commit:
git worktree add /tmp/prewt eda6c5a
cp mmacos/src/mmacos.mexa64 /tmp/prewt/mmacos/src/   # reuse the built mex

# per deck (shell loop; ONE matlab -batch per side):
matlab -batch "addpath('<this dir>'); sens_core_ab( \
   '/tmp/prewt/mmacos/mmacos_setup.m', '<deck.in>', 'out_pre/<deck>', 1e-5, 1e-5)"
matlab -batch "addpath('<this dir>'); sens_core_ab( \
   '<sens-core>/mmacos/mmacos_setup.m', '<deck.in>', 'out_post/<deck>', 1e-5, 1e-5)"
matlab -batch "addpath('<this dir>'); sens_core_ab_compare('out_pre/<deck>','out_post/<deck>')"
```

## Rules that keep the comparison honest

- **Same engine build on both sides** (the harness varies only the
  mmacos tree), so the engine axis cancels by construction.  The
  engine old-vs-new question on NS decks is a SEPARATE check (plain
  trace + ray-status comparison; see the tasking brief).
- **Field half-widths inside the deck's vignetting margin on the PRE
  tree** — its dw_dx_multi still hard-errors on a fully-vignetted
  field, and a deck's nominal chief can be off-axis (the e5hex1
  lesson: nominal 3.5e-4 rad off in y).  Start at 1e-5 rad.
- **Error parity is a pass**: a deck that legitimately refuses an
  option set (no stop for reset_xp, no Zernike surfaces, no grids)
  records the error string; identical strings pre/post compare clean.
  Only a pre/post DIFFERENCE is a finding.
- **Expectation on segment-class NS decks (our corpus): byte-identical
  everywhere.**  Any delta = stop and report (deck, set, field,
  max|diff|), not adjudication — Dave 2026-09-07.
