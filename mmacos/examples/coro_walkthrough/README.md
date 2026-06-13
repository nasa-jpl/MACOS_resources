# Coronagraph walkthrough (illustrated manual example)

A fully-illustrated end-to-end propagation through the HCIT-style Lyot
coronagraph (`Rx_Coro_FPM.in`), with MATLAB figures at the **key
surfaces** of the relay.  Intended as the manual's worked coronagraph
example; reproducible from one script.

## Run

```matlab
addpath('<mmacos>');
addpath('<mmacos>/examples/coro_walkthrough');
coro_walkthrough();                       % -> figures/  (headless OK)
% or: coro_walkthrough(rx_path, outdir)
```

Runs headless under `matlab -batch` (figures are created invisible and
exported to PNG).  Model size 1024; ~1-2 min.

## The relay and the figures

The Lyot coronagraph reshapes starlight across a pupil -> focal ->
pupil -> focal chain:

| Surface | Elt | Role |
|---|---|---|
| Entrance pupil | 2  | input aperture |
| DM pupil       | 4  | deformable-mirror pupil |
| FPM / CorMask  | 9  | focal-plane mask blocks the stellar core |
| Lyot pupil     | 14 | **starlight diffracted to the pupil edge is rejected here** |
| Exit pupil     | 20 | post-Lyot pupil |
| Science focal plane | 21 | the dark hole |

Figures written to `figures/`:

- **`coro_surfaces.png`** — 2x3 montage at the six key surfaces
  (pupil amplitude; focal/mask-plane log-intensity).  Shows the
  coronagraph principle: the FPM blocks the core, the bright ring it
  diffracts to the pupil edge is caught by the Lyot stop, and the
  science focal plane is left dark.
- **`coro_beforeafter.png`** — 2x2 before/after with **shared
  normalisation per row** (so the effect is visible, not hidden by
  per-panel auto-scaling): top row = focal-mask plane without vs with
  the FPM (stellar core blocked); bottom row = science focal plane
  without vs with the coronagraph, both normalised to the
  no-coronagraph peak, FOV extending past the 7-10 lambda/D dark hole
  (dashed rings mark it).
- **`coro_darkzone.png`** — radial contrast vs lambda/D, no-mask
  baseline vs coronagraph, with the 7-10 lambda/D dark-zone mean +
  floor annotated (reuses the Sprint-1 `../design/coro` scoring
  helpers).
- **`coro_broadband.png`** — monochromatic vs **COMPOSE** broadband PSF
  (850 nm +/- 10%, 7 wavelengths).  `macos.compose` assembles the
  per-wavelength PSFs on a fixed pixel grid — one step in the
  walkthrough.

## Notes

- The generated `figures/*.png` are reproducible from the script and
  are `.gitignore`d; copy the ones the manual embeds into the manual's
  asset dir as needed.
- Uses `macos.complex_field` (pupil amplitude), `macos.intensity`
  (focal/mask planes), `macos.compose` (broadband), and the
  `../design/coro` dark-zone scoring helpers.
