# mmacos documentation

Human-facing **reference entries** for the `+macos` API — one Markdown
file per function (or per closely-related family). These are the source
material for the eventual user manual and a model for each function's
in-`.m` help block. Keep them short, runnable, and honest about
constraints.

> Status: seeded 2026-06-20 with [`elt_grid_add.md`](elt_grid_add.md) as
> the **worked exemplar** of the entry template below. Add entries as the
> API surface is documented; there is no obligation to backfill all at
> once.

## Entry template

Every entry follows the same skeleton so the manual reads consistently
and readers know where to look. Copy this and fill it in:

```markdown
# `function_name` — one-line tagline

One short paragraph: what it does and *why you'd reach for it*.

## Call forms
    macos.function_name(args)      % package function
    m.function_name(args)          % Session method (if applicable)

## Arguments
A table: name | type | meaning + constraints.  State units and any
index/orientation convention explicitly.

## Semantics
Behaviour, return value, and side-effects.  Call out in-place mutation,
accumulation, or anything that persists across calls.

## Typical use
A short, runnable block (load → act → evaluate).  Show how to obtain any
value the call needs (e.g. a live size) rather than hard-coding it.

## Errors
A table: error identifier | the condition that raises it.  Reassure the
reader that bad input fails loudly, not silently.

## Notes
Gotchas, units, differences from the pymacos sibling, backing wrappers.

## See also
Related entries.
```

## House conventions

- **Units at the surface.** State the unit of every physical quantity
  (SI metres unless the value is in prescription/Rx units — say which).
- **Element ids are positive** in mmacos (no negative "from-the-end"
  indexing — that's the package convention; note it where it differs
  from pymacos).
- **Error identifiers** are `macos:<function>:<reason>` and each entry
  lists them. Functions validate input and raise rather than corrupt.
- **Examples are runnable.** Prefer a `macos.Session` object in
  examples. Batch scripts that run under `matlab -batch` must end in
  `exit(0)` (see the top-level `CLAUDE.md`); a doc snippet that's only a
  fragment need not.
- **Match the `.m` help.** An entry should stay consistent with the
  function's own help block; when they drift, fix both.
