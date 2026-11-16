# Overcoat chromatic ladder — the 2c coating trade on both sides of λ/4

Companion evidence run for
`macos/REVIEW_POL_OVERCOAT_CHROMATIC_2026-07-28.md` and polval §8.3.

```
cd MACOS_resources/mmacos
matlab -batch "mmacos_setup; addpath('tools/pol_overcoat_chromatic'); oc_ladder; exit(0)"
```

## What it settles

`REVIEW_POL_EXTERNAL_2026-07-28.md` found that the 110 nm MgF₂ overcoat the
Phase-2c coating ladder applies is **0.607 quarter-waves** at the fixture's
own 1 µm — not the 0.96 its `% ~quarter wave at 632.8 nm` comment describes —
and that the overcoat polarization trade **reverses** across the
quarter-wave condition. That correction was carried by an independent
analytic. This tool puts **engine** numbers on both sides, so the design rule
is a measurement rather than an assertion.

Fable's ruling was that **gated fixtures do not move**: `Rx_Cass_FarField`
stays at `Wavelen = 1e-6`, `tPolContrast`'s 27.898 / 151.31 stay asserted.
The companion wavelength is applied at **runtime** with `macos.set_src_wvl`
after `load_rx`. Nothing on disk changes.

## Why it is a real measurement and not a unit conversion

`macos.coating` takes **physical** thickness; the engine divides by the
**current** `Wavelen` when it applies the layer phase. A film is therefore a
fixed piece of glass under a wavelength change and its optical thickness
moves — which is the whole mechanism. The `achromatic` control below is what
proves that is the mechanism and not something else about changing λ.

## The runs

Four ladder points per wavelength, on `Rx_Cass_FarField` at model 256,
x-polarized, both mirrors coated:

| point | stack |
|---|---|
| baseline | uncoated (as loaded) |
| bare Al | 200 nm Al |
| MgF₂/Al | 110 nm MgF₂ over Al — the film the 2c ladder applies |
| trueQW/Al | λ/(4·1.38) of MgF₂ over Al — a genuine quarter wave *there* |

Plus one **control**, at 632.8 nm only: `achromatic`, 110 nm × (632.8/1000) =
69.6 nm of MgF₂. That is the film with the same optical thickness *in waves*
at 632.8 nm that the real 110 nm film has at 1 µm — i.e. what a wrongly
achromatic treatment (thickness pinned in waves rather than in metres) would
have produced. It must **not** reverse.

## Two ratios, two questions

* `.ratio_mgf2` — total cross-polarized **power**, MgF₂/Al over bare Al. What
  a designer sees. It includes the irreducible **geometric** cross term that
  no coating removes.
* `.ratio_excess` — the same ratio formed from the coating **excess** over the
  uncoated baseline (identically the ratio of the two `d_cross_rel`). The
  coating-only quantity, and the one directly comparable with the
  pure-Fresnel analytic in `tools/pol_external_anchor`.

They diverge most at the true quarter-wave point, because that is where the
coating term is nearly extinguished and the geometric floor dominates the
total — `.qw_over_uncoated` reports how far above that floor it sits.

**Total power, never an annulus mean.** A fixed pixel annulus subtends a
different λ/D range at the two wavelengths, so an annulus statistic would mix
the coating effect with the diffraction scale. Total power has no such
dependence and is the quantity the reversal is a statement about.

## Gated by

`tPolContrast/test_overcoat_trade_reverses_across_the_quarter_wave_condition`
(mmacos, the `polfloor` group at model 256). The analytic counterpart —
same physics, no engine — is
`tPolExternal/test_overcoat_trade_reverses_with_optical_thickness`.
