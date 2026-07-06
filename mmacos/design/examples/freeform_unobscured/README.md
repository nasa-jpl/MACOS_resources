# `freeform_unobscured/` — sphere+Zernike unobscured telescope: the visible 3+n front end

The **sphere+Zernike (freeform) strategy** applied to the visible-band
(500 nm) front end for a coronagraph, imager, and spectrometer — the
direction chosen after the conic eccentric-pupil section
([`../tma_unobscured`](../tma_unobscured), kept as the trade study)
proved shroud-expensive and coupling-bound (Dave, 2026-07-06).

Why sphere+Zernike wins here:

- **Geometry and correction decouple.** The 0th-order layout is three
  base SPHERES placed for packaging; the fold TILTS unobscure the beam
  with **M2 staying close to the source→M1 beam** — the tilted-fold
  topology is the shroud-cheap alternative the eccentric section cannot
  reach (its AOI-safe decenter cost 3.6×D of shroud).
- **All correction lives in the Zernike departures** (CALIB OptZern,
  staged center → field). The departures are micron-scale sags that do
  not move the chief ray: once the layout packages, the optimizer
  cannot break the packaging — none of the conic/telecentric coupling
  that bogged the Korsch searches.
- **It holds at 500 nm.** The e5mono-derived geometry corrects from
  ~71,000 waves (all-sphere, uncorrected) to **0.03 waves RMS at the
  field center — diffraction-limited at the visible bar** (35 nm RMS).
  The FIELD is where the visible fight is: a small corrected field is
  fine for the coronagraph; the wide imaging field is what the **+n**
  mirrors are for (the standing rule: a wide field needs a 4th powered
  mirror, not more freeform on three).

## The script

`freeform_unobscured.m` — [1] all-sphere layout (f/3.2 M1, convex M2,
real intermediate focus between M2 and M3 = the metrology-injection
point) → [2] staged Zernike correction S0/S1/S2 (center →
inner field → full ±1′, 2-D area-weighted) → [3] `align_focal_plane`
(5×5 field-foci grid: FP tilt + defocus + field-curvature map) →
[4] realized apertures + clearance (UNOBSCURED) → [5] `add_pupil`
(exit pupil emits `PropType=FarField` — INT at the FP is the PSF) +
save + figures → [6] `design_report` (live EFL, f/#s, WFE ladder with
Strehl, FP tilt, packaging/AOI) → [7] the +n roadmap.

Artifacts (`.in`/`.mat`/`.png`/`.txt`) regenerate on each run and are
git-ignored.

## Knobs (top of the script)

`D`, `LAM` (500 nm), `FOV_ARCMIN` (design half-field; the visible DL
bar is tight — widen it and stage [2] shows the honest cost), the
all-sphere layout `R`/`TBET`/`TILT` (e5mono heritage), and the Zernike
`MODES`/`ZTYPE` set. See `../tma_centered/README.md` for the general
adaptation guide and `../sz_tma/` for the 1 µm baseline of the same
geometry.

## The +n roadmap (next stages)

1. **Repackage** — M2 closer to the input beam; intermediate focus
   closer to M2 (Dave's standing steps).
2. **+1: the 4th powered mirror** — widen the shared imaging field
   (the relay serves each instrument a small patch).
3. **Coronagraph integration** at the FarField exit pupil; imager and
   spectrometer pickoffs at the aligned (tilted) focal plane.
4. Segment M1 (SegMirMaker) when the front end freezes.
