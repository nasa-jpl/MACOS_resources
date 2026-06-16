# Coronagraph Layout Generator

Companion to `CORONAGRAPH_DESIGN_RULES.md`. `coronagraph_layout.py` turns a beam/DM/
wavelength/working-angle spec into a concrete plane-by-plane layout, self-checking the
constraints as it goes. Sits alongside the telescope generators.

## Use

```bash
python3 coronagraph_layout.py [out_dir]      # writes coronagraph_layout.json
```
or import:
```python
from coronagraph_layout import design
r = design(D_beam_mm=46.3, N_act=48, lam_nm=550, IWA_lod=3, OWA_lod=24,
           input_fnum=20, fpm_fnum=30, pixel_um=13)
```

## What it emits (Roman-class default shown)

| Quantity | Value | Rule |
|---|---|---|
| actuator pitch | 0.965 mm | `D_beam / N` |
| OWA supported | 24 λ/D | `N/2` — must be ≥ target OWA |
| quarter-Talbot sep (at OWA) | 3383 mm | `Λ²/(2λ)`, `Λ = D_beam/α` — see note |
| final F/# (Nyquist) | F/48 (→2.03 px/λD) | `λ·F# ≥ 2·pixel` |
| FPM IWA / OWA radius | 0.0495 / 0.396 mm | `α·λ·F#_fpm` |
| OAP focal lengths | 926 / 1389 / 1389 / 2222 mm | `f = D_beam·F#` |

## Conventions & checks

- Lengths mm, wavelength nm in. Collimated-space relays are **unit-magnification** by
  default → pupil = `D_beam` at apodizer, DM1, Lyot. For a different device pupil size,
  set relay magnification = ratio of OAP focal lengths.
- Working angles are in **λ/D (telescope aperture)** and are independent of physical beam
  size: `N` actuators sample `N/2` cycles across the pupil = `N/2` λ/D.
- Hard self-checks (assert): `OWA_target ≤ N/2` (else prints the required `N`); Nyquist
  `λ·F#_final ≥ 2·pixel`; positive separations; `IWA < OWA`.

## The quarter-Talbot number — read this

`z = Λ²/(2λ)` is the distance at which a pure phase ripple of period `Λ` **fully converts
to amplitude** (= Z_T/4 with Z_T = 2Λ²/λ). Because `Λ = D_beam/α`, the value is
frequency-dependent: the generator reports it at the IWA, the band geometric mean, and
the OWA. The OWA value (shortest) is the **reference scale**, not a hard target.

In practice the operating DM1–DM2 separation is **numerically optimized** across the
dark-hole band and constrained by packaging and beam-walk; for HCIT/Roman-class beams it
lands near **~1 m**, i.e. *below* the quarter-Talbot-at-OWA scale. Use `z_qT` to bound the
layout envelope, then optimize the actual separation in the diffraction model (FALCO/
end-to-end). Do not treat the printed `quarter_talbot_sep_mm` as the final separation.

## Notes / extension points

- `input_fnum`, `fpm_fnum`, `pixel_um` are the main knobs; `talbot_design_lod` defaults to
  the OWA. Feeding the TMA's output F/# into `input_fnum` chains this onto the telescope
  generators.
- Not modeled here (belongs in the diffraction/control layer): apodizer+Lyot joint
  optimization, polarization aberration, exact mask profiles, EFC. This fixes the
  first-order *layout* the controller assumes.
