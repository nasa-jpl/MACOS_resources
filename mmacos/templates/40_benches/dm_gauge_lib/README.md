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

Files: dmg_frame (ray-affine mag), dmg_anchor (poke-A translation),
dmg_register (8-parity + sign search), dmg_samp (parity-aware DM-frame
sampling), dmg_stencil (kernel stencil), dmg_lit (illuminated-actuator
mask), dmg_act_fit (Tikhonov lattice deconvolution), dmg_modal_corr
(Wiener, radial | separable), dmg_ifo_gauge (four-step PSI
measurement factory), dmg_zwfs_gauge (dimple mask + frozen-linear +
phase-stepped factory), dmg_say (report tee).
