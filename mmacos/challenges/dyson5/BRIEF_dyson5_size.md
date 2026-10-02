# dyson5 — how small can the Dyson block be at R4 performance?

Written for: Dave (the size question in `macos/BRIEF_ccmac_dyson_size.md`).
CCMac (Claude Code on the Mac), 2026-10-02, `dev-candidate`. Records:
`dyson5_size.{txt,mat,png}`, decks `dyson5_size_<family>_r<mm>.in`, driver
`dyson5_size_trade.m`, figure `dyson5_size_fig.m`. New files only — nothing in
`dyson5_run.m` / `dyson5_params.m` / `dyson5_envelope.m` / `dyson_ladder.m` was
touched. All scores are the ENGINE's (the ladder's `r.engine`), on the 7×7 grid,
each point a full R4 solve warm-started by continuation from the next-larger radius.

## The answer, first

**Yes — but the lever is the SLIT, not the glass.** The 220 mm silica monolith
(221 mm thick, 3.7 L edged, 8.2 kg) is a floor only because it carries the whole
54 mm slit on one block. Keep the full slit and it cannot shrink: silica already
fails by 200 mm, and switching to CaF₂ buys only 180 mm — at the *same* 8 kg,
because CaF₂ is 45 % denser.

**Split the focal plane into two 27 mm modules** (1500 px each, sharing the
3000-px swath) and the block collapses:

| configuration | block r | thickness | edged | mass | CRF / EE | vs record |
|---|---|---|---|---|---|---|
| **record (R4)** silica, 54 mm | 220 mm | 221 mm | 3.74 L | **8.2 kg** | 1.327 / 0.759 | — |
| CaF₂, 54 mm (best 1-module) | 180 mm | 181 mm | 2.48 L | 7.9 kg | 1.196 / 0.819 | thinner, same mass |
| **silica, 27 mm (2 modules)** | **100 mm** | **101 mm** | 0.44 L | **1.0 kg** | 1.050 / 0.958 | **8× lighter, 2.2× thinner** |
| CaF₂, 27 mm (2 modules) | 80 mm | 81 mm | 0.28 L | 0.9 kg | 1.037 / 0.972 | 9× lighter |

A **100 mm silica block, 101 mm thick, 1.0 kg**, matches R4 with margin (CRF 1.05
vs 1.33, EE 0.96 vs 0.76). Two of them total ≈ 2 kg of glass against the 8.2 kg
monolith — still ~4× lighter, each piece less than half as thick, and each half
as non-uniform (below). The "200 mm of heavy, awkward, non-uniform glass" is a
consequence of the one-block/54-mm-slit choice, not of the Dyson form.

## The solver check (step 1) — the deck does not change

The closure envelope (beat 4e) warm-started every point from the 220 mm record,
and you flagged that a stalled cold solve can masquerade as a bad design. By
**continuation** (each radius seeded from the next-larger solved design, never a
cold start) the two suspect rows come back essentially unchanged:

| radius | beat 4e (warm from record) | this walk (continuation) |
|---|---|---|
| 180 mm | CRF 1.607, EE 0.606 | CRF **1.611**, EE 0.605 |
| 150 mm | CRF 2.371, EE 0.408 | CRF **2.351**, EE 0.409 |

Differences are ~0.02 px — solver noise, not a different basin. **The envelope's
radius axis was the design's limit, not the solver's**: the fifth-order h⁴/r³ blur
of the concentric form (stage s0's law). Nothing in the deck needs retargeting.
The identity point (silica, 220 mm, seeded from itself) reproduces the Linux R4
record to four decimals (CRF 1.3267, EE 0.7591, SRF 2.0315, smile 0.00507,
keystone 0.00264) — the Mac gfortran engine is bit-equivalent to Linux here.

## What binds at the small end

It is **always the image**, never the mechanics:

- **Silica, 54 mm:** CRF/EE at once — 1.342 / 0.739 already by 200 mm. 220 mm is
  the floor.
- **CaF₂, 54 mm:** CaF₂'s higher index narrows the in-glass marginal cone, so the
  blur law bites later — closes to 180 mm (CRF 1.196), fails by 160 (1.44, and
  the 160 mm point sits on the 4 mm meniscus-thickness bound; the bound-scaled
  re-run still fails at 1.42, so 160 is a real image failure, not a bound).
- **Silica, 27 mm:** closes 220→100 mm with EE ≈ 1.0 the whole way; at 80 mm smile
  and keystone break 0.1 px (0.10 / 0.12) and CRF jumps to 1.69; at 60 mm the
  block is too small to form the concentric relay — the engine loses the chief
  ray geometrically. The 190 mm point only misses on the 4 mm meniscus-thickness
  bound; scaled to the radius it closes (CRF 1.024, EE 1.0), so the 100–190 mm
  band is solid.
- **CaF₂, 27 mm:** closes 190→80 mm; fails 60 (SRF 2.28, EE 0.40). The 220 mm
  point is flagged only because its meniscus vertex sits on the record's own
  lower bound (block r + 20 mm) — its image is perfect; the family's clean
  representative is 80 mm.

**Clearance never binds.** The slit-to-detector gap is set by the package, not
the block, so it does not shrink — `spectrometer_clearance` (run exactly as stage
s3 runs it) stays positive on every point, +0.4 to +1.0 mm throughout. It is a
persistently *tight* gap (~0.5 mm, no cold shield), but the image fails first at
every small radius. If anything the small blocks relieve it slightly.

## Uniformity and thermal (order-of-magnitude estimates)

The double-pass on-axis glass path falls with the block, so index inhomogeneity
and thermal-gradient wavefront error fall with it. Over the double pass, a 1×10⁻⁶
index inhomogeneity and a 0.1 K gradient (|dn/dT| ≈ 1×10⁻⁵ /K for both glasses)
each give, at 1 µm:

| block | double-pass path | ≈ wavefront error (each term) |
|---|---|---|
| 221 mm (record) | 0.44 m | **0.44 waves** |
| 100 mm silica | 0.20 m | **0.20 waves** |
| 80 mm CaF₂ | 0.16 m | **0.16 waves** |

The 100 mm block is ~2× more uniform and thermally stable than the monolith, for
free. (CaF₂'s dn/dT is *negative* — opposite sign to silica — so a two-glass or
athermalized build is a further lever if a gradient is the worry; not scored here.)

## Recommendation

1. **If a two-module focal plane is acceptable** — and for a 3000-px VSWIR push-
   broom the swath is commonly split anyway — go to **100 mm silica blocks**:
   1.0 kg each, 101 mm thick, R4 performance with comfortable margin (CRF 1.05,
   EE 0.96), ~2× better uniformity, in cheap, available, well-behaved fused
   silica. Two blocks ≈ 2 kg total vs 8.2 kg. This is the real answer to "much
   smaller at the same performance."
2. **If one module is mandatory**, the only shrink available is **CaF₂ at 180 mm**
   — thinner (181 vs 221 mm) and 2.48 vs 3.74 L, but ~the same 8 kg (denser), and
   CaF₂ at 180 mm is expensive and harder to make than a silica monolith. Likely
   not worth it; keep the 220 mm silica block.
3. The size wall is the concentric fifth-order blur, which scales with field
   height on the face — halving the slit is worth far more than any glass or
   meniscus move. If you want to push a single module smaller, that needs the
   paper's separate-mirror compact variant or a freeform, not a bigger search.

## Reproduce

```matlab
run('<...>/mmacos/mmacos_setup.m');  addpath('<...>/mmacos/challenges/dyson5');
OUT = dyson5_size_trade();            % families A, B, C by continuation (~2 h)
dyson5_size_fig(OUT);                 % dyson5_size.png
OUT = dyson5_size_trade(struct('which', {{'A'}}));   % one family
```

Baseline gates green on the Mac before the run: tSpectrometerRx 7, tGratingImmersed
4, tGlassDispersion 3, tGratingOpl 2, tAsphCalib 2. Engine rebuilt (gfortran) and
the mex relinked carrying the beat-4c LM-failure fix.
