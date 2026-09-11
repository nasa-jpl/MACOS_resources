# dm_gauge_lib — the ONE copy of the DM-gauge scoring machinery

Shared by `tg_psi_dm96/` (the polarization-PSI interferometer) and
`zwfs_dm96/` (the Zernike sensor): registration, DM-frame sampling,
actuator-space fitting, modal correction, and the two instruments'
measurement factories.  Extracted 2026-09-05 from the S3-era copies
(`zwfs_s3.m` / `tg96_s3.m` / `zwfs_s2.m` / `tg96_eprime.m` at
`10cf593`), which carried three near-identical private copies each —
factored BEFORE S4 so the head-to-head runs off a single scoring
implementation.  The committed S1–S3 scripts keep their private copies
as the historical record; `*_s3.m` were retrofitted onto this lib and
re-run as the equivalence gate; S4 consumes the lib only.

Unifications (deliberate, recorded):
- `dmg_act_fit`: pcg tolerance 1e-12 / 400 iterations (the S3 flavor;
  S2/E′ used 1e-10/300 — differences land far below the pm reporting
  precision).
- Everything else is verbatim.

Conventions that ride with the lib (from the campaign memory):
PARb=[1 2 1 1] with sgn=+1 (zwfs) / −1 (ifo, in the shared candidate
enumeration); S_CONV=−1; MASK_TRIM −5.582; spot 2.0; b2cal from the
flat disk-frame; BETA 0.1; seeds rng(7) base / rng(23) random dev.

`dmg_zwfs_gauge` calls `zwfs_mask` — run ZWFS scripts with
`zwfs_dm96/` as cwd (all campaign scripts cd to their own dir).

**Third ZWFS reading (2026-09-09, `zwfs_s7iter`): `measI` / `reconI` /
`solveI` — the ITERATED-REFERENCE exact solve** (literature import #1,
`macos/REPORT_zwfs_lit_scan.md`).  Per pixel (Ruane 2020 eq 36-37 /
N'Diaye 2013 eq 6-7, general complex `E0`, `b`, `c`):
`cos(phi - Theta) = (I - A^2 - |c|^2|b|^2) / (2 A |c| |b|)`,
`Theta = arg c + arg b - arg E0`, principal branch `phi = Theta -
acos(.)` (the quarter-wave sensor's -pi/4 .. 3pi/4); the reference
wave `b` is re-propagated from the estimate through the FFT surrogate
of the mask model `b(E) = T(D .* Ti(E))`, `T = fftshift(fft2(fftshift
(.)))/N` (= the engine's PL2SPH; the geometric tail is the identity on
the grid), `NITER` times (opt.NITER, default 5; 0 = exact solve with a
FROZEN b).  ONE masked frame; `A = |E0|` (the flat's amplitude — exact
only on a DM-CONJUGATE pupil, see `gate.roundtrip`; pass the state's
own unmasked frame as `I0` otherwise).  `reconI(Ia, I0, plus, b0,
niter)`: `plus` = per-pixel logical selecting the OTHER branch (from
`plusFromX(X)` of a stepped retrieval of the base — the 'I+' protocol:
the base costs 4 frames once, every differential frame is one);
`b0`/`niter` = seed b / iteration override (an ORACLE solve seeds the
engine's Eb of the state itself).  `solveI` returns `[phi, info]`
(info.dphi = rms update per iteration, nclamp = clamped px, b).
`frameI == frameL` (the same frame).  Gates the factory now carries:
`gate.bsur` (surrogate vs the engine's Eb on msk, 1e-15 class) and
`gate.roundtrip` (unmasked entrance->exit sphere identity; WARNS once
if > 1e-9: an asymmetric sandwich Fresnel-defocuses the pupil —
`twyman_green` 'nf_legacy', the S1-S6 record — and A = |E0| no longer
holds under a DM state).

**Closed-loop hold metric (2026-09-11, `dmg_loop`): the on-orbit
servo mode as ONE shared loop.**  `dmg_loop(ins, opt)` holds a DM at a
set point against a drift ('walk' | 'thermal' | 'step') by a
proportional loop closed through an instrument given as four handles
(`measure` -> noiseless frames, `noisy` -> photon noise at N per measurement,
`diff` -> the reading's differential map, `est` -> actuator changes) and
a lit mask; it scores the steady-state hold error (rms over lit), the
bias with the noise averaged out, the in-run single-shot noise, the
transient (rho = 1 - gG, tau) on a step, and the residual's spectrum in
cycles per aperture, with the theory lines (noise-only sigma_n
sqrt(gG/(2-gG)); walk (sigma_d^2 + g^2G^2 sigma_n^2)/(gG(2-gG)); ramp
lag rate/gG).  Drift realizations are seeded on the full actuator grid
so two instruments see the same pattern.  Gated on a synthetic linear
instrument by `tests/tDmgLoop.m` (8 gates, SUITE_FAST).  Consumers:
`zwfs_run` stage 'loop'; the IFO via `tg96_run` (CCMac, brief oap2
addendum).  Spec: `macos/BRIEF_loop_metric.md`.

Files: dmg_frame (ray-affine mag), dmg_anchor (poke-A translation),
dmg_register (8-parity + sign search), dmg_samp (parity-aware DM-frame
sampling), dmg_stencil (kernel stencil), dmg_lit (illuminated-actuator
mask), dmg_act_fit (Tikhonov lattice deconvolution), dmg_modal_corr
(Wiener, radial | separable), dmg_color_comb (multi-COLOR multi-channel
Wiener on the lattice: a_hat = sum_k G_k A_k / (sum_k G_k^2 + beta^2),
S6 color stage, 2026-09-08), dmg_loop (the closed-loop hold metric,
2026-09-11), dmg_ifo_gauge (four-step PSI measurement
factory), dmg_zwfs_gauge (dimple mask + frozen-linear + iterated-reference exact +
phase-stepped factory), dmg_say (report tee).
