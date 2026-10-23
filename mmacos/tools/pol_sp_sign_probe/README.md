# `pol_sp_sign_probe` — reproducer for the reflected-p̂ / Fresnel-r_p sign conflict

One script, no arguments, ~15 s.  It exists to make an open engine defect
reproducible in one command rather than described in prose.

```sh
cd ~/dev/MACOS_resources/mmacos
matlab -batch "run('mmacos_setup.m'); addpath('tools/pol_sp_sign_probe'); probe_sp_sign; exit(0)"
```

## What it measures

`Rx_Cass_FarField.in` (the Phase-2 gate fixture), x-polarized source, perfect-
conductor mirrors, model 256.  Element 2 is after the Primary (**one** mirror),
element 3 after the Secondary (**two**).  The observable is `RayE` read through
`macos.ray_field`, so the diffraction layer is not involved.

Expected output against the shipped engine (`pol-core`):

```
  mirrors    Py/Px        max|Ey/Ex|
        1    1.0163e+00   5.7348e+03
        2    7.0612e-07   1.9743e-03

  after ONE mirror, |Ey/Ex| vs radius (flat => not AOI-driven):
    rho =  31.9 px:  median |Ey/Ex| = 1.0142e+00
    rho =  63.8 px:  median |Ey/Ex| = 1.0160e+00
    rho =  95.7 px:  median |Ey/Ex| = 1.0098e+00
    rho = 127.6 px:  median |Ey/Ex| = 1.0051e+00
```

A single mirror at under 2° AOI cannot mix a uniformly x-polarized beam into
50/50 x/y: an isotropic surface generates `O(sin²β)` cross-polarization, β the
local surface slope, ≈1e-3 here.  And the second block is the fixture-free
tell — the effect is **flat in pupil radius**, where any real AOI-driven
effect must vanish on axis and grow as ρ².

## Why the regression suite is green anyway

The defect acts as a reflection of the transverse field about the local p̂.
That is an **involution** (so a mirror pair cancels it exactly) and it is
**unitary** (so a unitarity gate cannot see it).  `Rx_Cass_FarField` has
exactly two mirrors, `Rx_VecChain` has none, and the Fresnel-analytic fold
gate compares a *ratio* against an analytic form transcribed from the engine's
own expression — circular in this sign.

## Confirming the diagnosis

Restore the standard Fresnel `r_p` — the line already sitting commented out as
`! dcr's original` at `macos_f90/elemsub.F:454` — rebuild, and re-run.  The
one-mirror number drops to `2.0724e-04` (4900×) while the two-mirror number is
**bit-identical**.  Revert afterwards; this is not a landed fix.

Full analysis, suite impact (285/287 with the patch; the two that move are
both basis-artifact assertions) and the open decision:
`macos/REVIEW_POL_SP_SIGN_2026-07-27.md`.
