# A REAL polarizing beamsplitter — the MacNeille cube (TG v2)

The [v1 rig](../tg_psi_dm) carries the polarization **conceptually**: the
splitter is a perfect-conductor plate plus a compensator, and each arm opens
with an *ideal* `TrPolarizer` at normal incidence.  That model earned its
keep — it found a real defect, the beamsplitter rotating the test arm 7.479°
and the gauge reading **11.7% high**.

v2 replaces the concept with the **component**: a cemented **MacNeille
polarizing cube**, one coated interface at 45° between two prisms of the same
glass, built out of ordinary coated engine surfaces.  Then it asks what is
left of the v1 error, and what has taken its place.

**Nothing here is new engine capability.**  MACOS has carried trustworthy
coated s/p physics at arbitrary AOI since 2026-07-27/28 (the `r_p` sign fix,
the incident-medium fix, the transmission radiometric factor, anchored to
published Mueller data at ~1e-14).  So a PBS is coating physics: no ray
splitting, no `RfPolarizer` element, still two decks — the engine does not
split rays.

| file | what it is |
|---|---|
| `example_tg_psi_dm_v2.m` | the gated measurement — six gates, the closure, all the numbers |
| `demo_tg_psi_v2.m` | the same, as seven narrated self-timing beats, one PNG each |
| `v2_*.in`, `v1_*.in`, `demo2_*.in` | the emitted prescriptions (both arms, both rigs) |
| `dm2_*.txt`, `demo2_flat.txt` | DM grid maps — **gitignored**, ~1 MB each; run either script once to recreate what the decks reference |
| `tg_psi_dm_v2.mat` | every result struct |
| `tg_psi_dm_v2_*.png`, `demo2_beat*.png` | figures and pre-rendered demo backups |

New builder pieces, all in `mmacos/src/+macos/+design/`:

| | |
|---|---|
| `pbs_macneille.m` | the MacNeille design: prism index, layer stack, quarter-wave-**at-angle** thicknesses |
| `thinfilm_rt.m` | the textbook reference — Macleod's characteristic matrix, **general incident medium**, r **and** t |
| `Bench.pbs_cube` / `Bench.add_pbs_pass` | the cube as a token + one traversal per call |
| `twyman_green('pbs','cube')` | the v2 rig; `'plate'` (default) still emits v1 bit-identically |

Gates: `mmacos/tests/tTgPol2.m` (9 tests, model 128, `SUITE_FAST`, ~6 s).
v1's `tTgPol` (9 tests) is **unchanged and still green** — v2 is a sibling,
not a replacement, and v1 remains the rehearsed demo default.

```
cd <this dir>
matlab -batch "run('example_tg_psi_dm_v2.m')"    # gated, ~4 min
matlab -batch "run('demo_tg_psi_v2.m')"          # the beats + PNGs, ~40 s
demo_tg_psi_v2                                   # interactive: pauses per beat
```

---

## Headline

| | v1 plate + ideal polarizers | v2 MacNeille cube |
|---|---|---|
| arm rotation from orthogonal | **7.4786°** | **5.3e-06°** |
| PSI scale gain, as designed | **1.11661** (11.7% high) | **0.999999** |
| alignment step | solve a +3.768° waveplate clock | **none — every azimuth is a design constant** |
| compensator plate | required | **none**; arms balance to 2.3e-13 mm |
| fringe visibility | 0.996612 | 1.000000 |
| delivered power (same source) | 0.1691 | **0.3836 — 2.27×** |
| arm separation after a 10° waveplate error | 13.15° | 0.369° (transmitting arm) / **3.1e-05°** (reflecting arm) |
| DM map recovered | corr 0.998434, 0.304 nm rms residual | corr **0.999573**, **0.183 nm** rms residual |
| detector registration between arms | sub-pixel | **0.00e+00 px, pitch to 5.5e-14** |

The v2 rig is better on every line, and the reasons are structural rather
than lucky.  They are worth stating separately, because each is a design
rule you can carry to another instrument.

---

## 1. The coating is the textbook stack, and the engine reproduces it

MacNeille (US Patent 2,403,731, 1946), in the form Macleod gives in
*Thin-Film Optical Filters*: choose the prism index so the internal angle at
every H/L interface is Brewster's,

```
n_g · sin(45°) = n_H·n_L / sqrt(n_H² + n_L²)
```

With the classic visible pair — **ZnS n_H = 2.35, cryolite n_L = 1.35** —
that fixes **n_g = 1.655468**, a dense flint (≈ SF2).  This is why real
MacNeille cubes are not made of BK7: the *glass* is part of the coating
design.  Internal angles come out **θ_H = 29.876°, θ_L = 60.124°**, summing
to 90.000000° — Brewster, to 1.4e-14.

Layers are quarter waves **at angle** (`n·d·cosθ = λ/4`), so the physical
thickness carries the internal angle, not just the index: **77.64 nm** for
ZnS, **235.25 nm** for cryolite, with the two outer H layers at half that.

Probing the engine's diagonal with a pure s and a pure p input, against
`macos.design.thinfilm_rt`:

| | engine | Macleod | rel. err |
|---|---|---|---|
| `T_s` | 4.198875192923e-04 | 4.198884783816e-04 | 2.3e-06 |
| `T_p` | 0.999999999996 | 1.000000000000 | 4.0e-12 |
| `R_s` | 0.999580112481 | 0.999580111522 | 9.6e-10 |
| `R_p` | 3.98e-12 | 2.60e-30 | (both null) |

`R_s + T_s = 1.000000000000` and `R_p + T_p = 1.000000000000` — and that is
a **cross-deck** check, not an internal one: the reference arm's deck
measures R and the test arm's measures T, of the same coating.  Extinction
**T_p/T_s = 2382 : 1**, which is realistic (real MacNeille cubes run ~10³).

Two conventions collapse here and it is worth knowing why the comparison is
so clean.  The cube is **cemented**, so `n_inc == n_sub`, and therefore both
the engine's radiometric factor `sqrt(n_sub·cos_sub/(n_inc·cos_inc))` and
Macleod's tangential-vs-Fresnel factor `cos_sub/cos_inc` are identically 1.
On a non-cemented interface they are not, and the 45° p case is exactly
where a missed factor looks like a plausible physics error
(`REVIEW_POL_RADIOMETRIC_2026-07-28`).

The analytic is written from the textbook, never transcribed from
`elemsub.F`.  That is not fussiness: an "analytic" copied out of the engine
is circular in exactly the coefficient it is supposed to check, which is how
the 2022 `r_p` sign defect survived every gate for four years
(`REVIEW_POL_SP_SIGN_2026-07-27`).

## 2. Brewster is not enough — the finding this build produced

The first stack built here satisfied the MacNeille condition at every H/L
interface and was still a **2.1% R_p** polarizer.  Working out why is the
most transferable thing in this template.

The Brewster condition equalizes the **tilted p admittances** `η_p = n/cosθ`
— both come to **2.7101** — so for the p component the entire stack is *one
homogeneous slab*.  That kills every internal p reflection, which is the
whole point.  But the slab still has two boundaries **with the prism**
(`η_p = 2.3414`), and those are not Brewster.  What happens there depends on
one number: the slab's total p phase thickness.

| stack | quarter waves | p slab is | R_p | T_p |
|---|---|---|---|---|
| `H(LH)^4` | 9 (**odd**) | a quarter-wave layer | **2.11e-02** | 0.9789 |
| `(½H L ½H)^4` | 8 (**even**) | a half-wave **absentee** | **0** | 1.0000 |

Both are textbook MacNeille designs.  One is a polarizer.  The symmetric
period is the default; `'design','qw'` builds the counterexample, and it is
what makes every p-null assertion in `tTgPol2` non-vacuous.

Symmetry earns its place a second time, independently: `r` of a stack
depends on which side you approach from *unless the stack is symmetric*, and
this cube is used from **both** sides (the test arm transmits then reflects;
the reference arm reflects then transmits).  A symmetric stack makes the two
arms interchangeable by construction, so the interferogram carries no
coating-induced differential piston.

## 3. Why the v1 arm rotation is *structurally* absent

v1's diagnosis was right and is unchanged: every non-normal element between
polarizer and recombination is a diattenuator (`t_s ≠ t_p`), and a
diattenuator **rotates a linear state toward its high-transmission axis**.
v1 put each arm's state at 45° to the splitter's s/p axes, where that
rotation is **first order** in the diattenuation.

The cube puts each arm's state **on** a coating eigenaxis — the test arm on
p, the reference arm on s — and a diattenuator cannot rotate a state that is
already one of its eigenstates.  Measured: the arms leave at
**−89.999997°** and **−0.000002°**, i.e. **5.3e-06°** from orthogonal, with
no alignment step at all.  The output QWP sits at a fixed 45°, a design
constant rather than a solved one.

That also disposes of the compensator.  Every cube traversal is
`a/2 → diagonal → a/2` from whichever port you enter by, so the arms' glass
paths are identical **by construction** (measured 2.3e-13 mm), and the two
beams land on the same detector pixels to **0.00e+00 px**.

## 4. The error budget inverts — and that is the real instrument result

v1's memorable lesson was that a waveplate azimuth error costs **scale** and
hides from the fringe monitor: 11.7% of scale for 0.17% of contrast, a
factor of ~70.  On the cube it is the other way round.

Turn an arm's waveplate off design by ε and watch what the return pass does
with the resulting cross-polarized component:

| ε (deg) | v1 plate | v2 transmitting arm | v2 reflecting arm | v2 gain | v2 visibility |
|---|---|---|---|---|---|
| 0 | −7.4786 | −5.3e-06 | −5.3e-06 | 0.999999 | 1.000000 |
| 2 | −3.5520 | −7.09e-02 | −2.3e-06 | 0.999999 | 1.000000 |
| 3.768 | **−0.0001** | −1.34e-01 | −8.5e-06 | 0.999999 | 0.999980 |
| 5 | −2.5215 | −1.79e-01 | −1.3e-05 | 0.999999 | 0.999926 |
| 10 | −13.1490 | −3.69e-01 | −3.1e-05 | 1.000000 | 0.998447 |

(columns are `|arm separation| − 90`, degrees)

- The arm that **reflects** on its return is re-projected onto the coating's
  own eigenaxis, so the error is cleaned to `r_p` — which is zero.  **3.1e-05°
  at 10° of error.**
- The arm that **transmits** is cleaned only to the extinction ratio
  `t_s/t_p = 0.0205`.  **0.369°** — still 36× better than v1, and the
  asymmetry between the two arms is a real, measurable prediction.
- **The scale does not move at all.**  The error emerges as *contrast*
  (1.000000 → 0.998447), i.e. on the fringe monitor, which is exactly where
  an operator is already looking.

**The PBS converts an invisible systematic into a visible one.**  That is a
better reason to buy one than throughput.

> The v1 column is non-monotonic on purpose: v1 *starts* 7.479° off, so its
> curve dips through a minimum at ε = 3.768 — **−0.0001°**, i.e. precisely
> the waveplate clock v1 has to solve for, recovered here as the zero of a
> sensitivity sweep rather than by a secant solve.  v2 is flat at the 1e-05
> level across the whole ladder: there is no minimum to find.

## 5. What it costs when the cube is not perfect

The engine reproduces the analytic on detuned designs too, so the tolerance
questions are answerable:

| configuration | R_p | T_p | PSI gain | arms from orthogonal |
|---|---|---|---|---|
| MacNeille symmetric (design) | 2.6e-30 | 1.000000 | 0.999999 | 5.3e-06° |
| odd-QW termination `H(LH)^4` | 2.11e-02 | 0.978883 | 1.003904 | 0.229° |
| prism SF2, n = 1.6477 (catalogue) | 2.53e-03 | 0.997471 | 1.002106 | 0.124° |
| prism n = design + 0.02 | 1.89e-02 | 0.981136 | 0.995145 | 0.286° |

So even a *badly* specified cube — catalogue glass instead of the design
index — reads only **0.21%** high, against v1's 11.7%.  The prism index is a
real design variable, not a detail: `pbs_macneille('n_glass', …)` detunes it
deliberately and re-solves the quarter-wave-at-angle thicknesses so the
comparison stays self-consistent.

## 6. Efficiency

Both returns leave by the same port — that is what a PBS is for — so the
cube delivers **2.27×** the plate rig's power from the same source.  The
budget from the declared stack alone: `T_p · R_s = 0.999580` through the
diagonal, times four AR'd faces at `T = 0.995110` each (single-layer MgF₂
quarter wave, 114.64 nm, on n = 1.6555 glass), giving **0.980** per arm.
Bare prism faces would give 0.777 — the AR is worth 26% of the light and is
the reason the faces are coated at all.

## 7. The gauge still closes, and closes better

Model 256, ray grid 63, HeNe, 16×16 actuators at 3.5 mm, checkerboard at
50 nm of surface:

```
truth 6.35 nm rms | recovered 6.26 nm rms | residual 0.183 nm rms interior
correlation 0.999573 | magnification 10.5258, 0.0000% anamorphic,
0.0068 mm rms nonlinear distortion (all measured from the trace)
```

against v1's 0.304 nm and 0.998434.  Everything v1 pinned still holds and is
re-checked here: a `GridData` value **is** the surface height (a 20 nm grid
piston and a 20 nm rigid optic shift agree to 2.3e-10 rad), three traces per
arm span every analyzer angle (7.6e-11), the sweep contains only DC/2θ/4θ
(6θ/2θ = 3.0e-14 from **traced** frames), and the 4θ term is 8.895e-04 —
*identical to v1*, which correctly identifies it as the detector leg's, not
the splitter's.

---

## Demo beat timings (measured, headless batch)

`demo_tg_psi_v2.m` times itself; these are from a `matlab -batch` run on a
box with **no graphics acceleration**, which is the worst case.

| beat | measured |
|---|---|
| 1 build + emit both arms + print the stack | **0.20 s** |
| 2 `view_std` layout | 8.06 s ← figure render |
| 3 coating: engine vs textbook, + the odd-QW trap | 3.75 s |
| 4 null (the analyzer basis for both arms) | **2.22 s**, 6 traces |
| 5 live single-actuator poke (`set_elt_grid`) | 3.06 s |
| 6 analyzer sweep, 36 frames | **2.62 s** — 0.034 s of it is the sweep, **zero traces** |
| 7 full DM + four-step PSI + closure | 18.13 s ← registration + 4-panel export |

**Two beats are not seconds-fast and it would be dishonest to imply
otherwise** — but they are not v2's fault, and the right comparison is
against the rig that is already rehearsed.  Same box, same batch mode,
MATLAB startup included:

| | total |
|---|---|
| v1 `demo_tg_psi.m` | 49.8 s |
| v2 `demo_tg_psi_v2.m` | **47.0 s** |

v2 is marginally *faster*: the coating beat it adds costs about what the
alignment beat it removes used to.  Beat 2 at 8 s and beat 7 at 18 s are
dominated by figure rendering and export — beat 7 spends most of its time in
`scatteredInterpolant` + `fminsearch` registration and a four-panel `print`
— and both are the same code paths v1 uses.  Every beat writes its PNG, so
**the mitigation is the pre-rendered backup**: if a beat stalls in front of
a room, show `demo2_beat2_layout.png` / `demo2_beat7_recovery.png` and keep
talking.  On an accelerated desktop both are materially faster; if beat 7
must be live-fast, cut the registration and show the raw recovered map.

---

## Notes for anyone extending this

- **`mCoat = 10` is a hard engine ceiling on Model-A coating layers, and the
  Rx parser does not check it.**  `Coating= 11` loads *without complaint*
  and writes past the end of `IndRefArr`/`EltCoatThk`; the only visible
  symptom is `coat_get` failing afterwards.  Found building this cube (the
  first stack was 11 layers).  `pbs_macneille` asserts on it, and the engine
  fix is written up as a separate item — **not** patched as a side effect of
  this work.  N = 4 (9 layers) is the largest symmetric design that fits,
  and at 2382:1 it is if anything more representative of a real cube.
- **An empty matrix cannot mean both "default" and "none".**  `pbs_coat`
  defaults to `NaN(1,3)`; `zeros(0,3)` is an explicitly **bare** cemented
  interface.  That distinction is load-bearing for the structural gate
  below, and the first version of it silently ran the coated stack.
- **A bare cemented interface carries no light at all**, and the first
  reading — "the arms come out parallel" — is wrong.  Glass against the same
  glass gives `R = 0` exactly, and *each arm reflects off the diagonal once*
  (the test arm on its return, the reference arm on the way out), so both
  arms are extinguished.  Do not report an azimuth for a zero field;
  `arm_state` divides by it.  `tTgPol2` measures the delivered power
  instead (bare/coated < 1e-9).
- **Report `R_p` absolutely, never as a relative error.**  It is a designed
  null; `|engine − analytic| / analytic` on a quantity that is 2.6e-30 is
  noise dressed as a number.
- **The cube subtracts its own half-side from `D_RC_L2`** so the
  Recomb → L2 → mask → detector conjugate is the same geometry as the plate
  rig and the `l2_trade` tail trims transfer verbatim.  Change the cube side
  and that follows automatically; override `D_RC_L2` and it does not.
- The helper block in `example_tg_psi_dm_v2.m` is **copied** from v1 rather
  than shared.  v1 is frozen as the demo default, and a shared helper file
  would make every v2 edit a v1 risk.

## Deliberately not done

- **No engine work of any kind** — everything is builder / example /
  prescription level.  The `mCoat` gap above is written up, not fixed here.
- **The stretch items**, both recorded rather than attempted: the full
  curve-on-curve external anchor against a published MacNeille cube's own
  measured curves (the design *rule* and the material indices are cited
  above and driven verbatim, but no published R/T curve is reproduced), and
  merging this, v1 and `../bench_ifo_pol` onto one PSI/Jones source.
- **The waveplates are still ideal retarders** — no o/e walk-off, no face
  Fresnel loss, retardance independent of incidence angle.  Unchanged from
  v1, and unchanged by choice: a real crystal plate's field-of-view effect
  needs a birefringent-plate model, which would be engine work.
- **No angular-spectrum study of the cube.**  A MacNeille cube's extinction
  degrades away from its design angle, and the beam here is collimated to
  tens of microradians, so the effect is below everything reported.  A rig
  with a converging beam at the cube would need it.
