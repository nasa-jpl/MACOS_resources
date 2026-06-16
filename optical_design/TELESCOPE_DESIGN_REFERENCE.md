# Telescope Design Families — Closed-Form Reference

> **Purpose.** Background + governing equations for the design layer built on MACOS.
> Written for direct consumption by Claude Code. Equations are the *implementation
> targets*: first-order layout, conic constants, and the third-order conditions that
> define each family. Named designs are presented as **special cases of a single set
> of master conditions** — implement the master conditions once, and the families fall
> out as parameter choices. Cross-check every formula against Schroeder *Astronomical
> Optics* (2nd ed.) before committing; multiple equivalent parameterizations exist in
> the literature and sign conventions differ between texts.

---

## 0. How to use this file

- Section 2 fixes notation and sign conventions. **Do not mix conventions** — the RC
  conic formula below is validated only in the convention stated.
- Section 3 is the surface (sag) representation MACOS should emit.
- Section 4 is the master third-order machinery for two-mirror systems.
- Sections 5–7 are the families: two-mirror (Cassegrain/RC/Dall-Kirkham/Gregorian),
  three-mirror anastigmats (Korsch TMA), and free-form.
- Section 8 gives reference prescriptions to use as regression fixtures.
- Section 9 lists primary sources to transcribe from / point at.

---

## 1. Family overview (when to reach for each)

| Family | Mirrors | Corrects (3rd order) | Field | Typical use |
|---|---|---|---|---|
| Classical Cassegrain | 2 | spherical | narrow (coma-limited) | compact, simple |
| Ritchey–Chrétien (RC) | 2 | spherical + coma (aplanat) | wider (astigmatism-limited) | most pro research scopes (HST, VLT, Keck) |
| Dall–Kirkham (DK) | 2 | spherical (spherical secondary) | narrow (strong coma) | easy-to-test secondary; amateur/cost |
| Gregorian | 2 | spherical | narrow | real intermediate image (field stop / Lyot-friendly) |
| Korsch TMA | 3 | spherical + coma + astigmatism (+ flat field) | wide | JWST, Roman, survey/space imagers |
| Free-form | ≥3 | all of the above, **off-axis / unobscured** | wide, no obstruction | coronagraph-friendly, compact space optics |

**Coronagraph note:** the unobscured branch (off-axis TMA, free-form) is the relevant
one when central obstruction would wreck contrast. The RC/Cassegrain conics below are
still needed as on-axis building blocks and as starting points before unobscuring.

---

## 2. Conventions and notation

Optical (Schroeder) convention. Light travels left→right; concave-toward-incoming
radius is negative by the usual sign rule, but the *formulas below use magnitudes for
m and treat β as defined here* — keep it consistent.

| Symbol | Meaning |
|---|---|
| `f`   | system effective focal length (EFL) |
| `f1`  | primary mirror focal length (`f1 = R1/2`) |
| `m`   | secondary magnification, `m = f / f1` (magnitude; `m > 1` for Cassegrain) |
| `R1`, `R2` | vertex radii of curvature of primary, secondary |
| `K1`, `K2` | conic constants (Schwarzschild constant). `K = -e^2`. K=0 sphere, K=-1 paraboloid, K<-1 hyperboloid, -1<K<0 prolate ellipsoid, K>0 oblate ellipsoid |
| `β`   | back-focal-distance parameter = (primary-vertex → final-focus distance) / `f1`. **β > 0 ⇒ focus behind the primary vertex** (standard Cassegrain). |
| `k`   | marginal-ray-height ratio `y2 / y1` (height at secondary ÷ height at primary); ≈ `d/D` for the illuminated apertures |
| `p`   | radius ratio `R2 / R1` |
| `D`, `d` | clear diameters of primary, secondary |

`m` and `β` are the two free first-order parameters that, with `f` and `D`, pin the
whole two-mirror layout.

---

## 3. Conic surface representation (what MACOS emits)

Rotationally symmetric conic, sag `z` as a function of radial coordinate `r`:

```
z(r) = (c * r^2) / ( 1 + sqrt( 1 - (1 + K) * c^2 * r^2 ) )      where c = 1/R
```

Equivalent implicit form (Schroeder Eq. 4):

```
r^2 = 2*R*z - (1 + K) * z^2
```

Free-form / general surface = base conic **plus** a departure polynomial (Section 7).

---

## 4. Two-mirror master conditions (third order)

These are the workhorses. Implement these three; everything in Section 5 is a root of
them.

### 4.1 Spherical aberration = 0

```
1 + K1 = (k^4 / p^3) * [ K2 + ((m+1)/(m-1))^2 ]
```

equivalently, solving for the secondary:

```
K2 = (p^3 / k^4) * (1 + K1)  -  ((m+1)/(m-1))^2
```

(Schroeder Eq. 4.5.3 / standard two-mirror result; `k = y2/y1`, `p = R2/R1`.)

### 4.2 Coma = 0 (aplanatic ⇒ RC)

Imposing zero third-order coma on top of 4.1 fixes the primary conic directly:

```
K1 = -1  -  2*(1 + β) / ( m^2 * (m - β) )
```

### 4.3 Astigmatism = 0 (anastigmat condition — needs the right geometry)

```
K1 + 1 = 4 * (m^2 + β) * (1 + β) / ( m * (m - β) )^2
```

A two-mirror system **cannot** generally null spherical + coma + astigmatism + field
curvature simultaneously — that is what the third mirror buys you (Section 6).

### 4.4 Residual aberrations (for verification / merit functions)

Angular sagittal coma and angular astigmatism for a two-mirror system already corrected
for spherical aberration, field angle `Θ`:

```
ASC = [ 1 + (K1 + 1) * m^2 * (m - β) / (2*(1 + β)) ] * Θ / (4 * (m * f/D)^2)

AAS = [ (m^2 + β) / (m*(1 + β))
        - (K1 + 1) * m * (m - β)^2 / (4*(1 + β)^2) ] * Θ^2 / (2 * m * f/D)
```

Use these as cheap closed-form checks before handing geometry to the full MACOS trace.

---

## 5. Two-mirror named families (conic constants)

All share the same `R1`, `R2` for given `f, D, m, β`; they differ **only** in `K1, K2`.

### 5.1 Classical Cassegrain — paraboloid + hyperboloid (spherical-free)
```
K1 = -1
K2 = -((m + 1)/(m - 1))^2
```

### 5.2 Ritchey–Chrétien — both hyperboloid (aplanat: spherical + coma free)
```
K1 = -1  -  2*(1 + β) / ( m^2 * (m - β) )
K2 = -((m + 1)/(m - 1))^2  -  2*m*(m + 1) / ( (m - β) * (m - 1)^3 )
```
Both `K1, K2 < -1` ⇒ both hyperbolic; primary only slightly past parabolic.
(Validated: HST-like `m≈10.4, β≈0.27` ⇒ `K1≈-1.0023`, `K2≈-1.50`.)

### 5.3 Dall–Kirkham — ellipsoid primary + **spherical** secondary
Set `K2 = 0` in 4.1 and solve:
```
K2 = 0
K1 = -1  +  (k^4 / p^3) * ((m + 1)/(m - 1))^2
```
Spherical-free on axis; large uncorrected coma (narrow field). Secondary is a sphere ⇒
easy to fabricate/test — that's the whole point of DK.

### 5.4 Classical Gregorian — paraboloid primary + concave ellipsoid secondary
Secondary placed **beyond** prime focus (real intermediate image):
```
K1 = -1
K2 = -((m - 1)/(m + 1))^2
```
The real intermediate image is the feature to exploit: a field stop / Lyot stop sits
naturally there, and the pupil is accessible — relevant for coronagraph front ends.

### 5.5 Generalized / Modified two-mirror (free K1)
Pick any `K1`, get the spherical-free `K2` from 4.1:
```
K2 = (p^3 / k^4) * (1 + K1)  -  ((m + 1)/(m - 1))^2
```
This is the knob for "modified RC / Dall-Kirkham" trades and for seeding optimization.

---

## 6. Three-mirror anastigmats (Korsch TMA)

### 6.1 Why three mirrors
Three powered surfaces give enough degrees of freedom to simultaneously null the four
primary (Seidel) sums:
```
ΣS_I   = 0   (spherical)
ΣS_II  = 0   (coma)
ΣS_III = 0   (astigmatism)
ΣS_IV  = 0   (Petzval / field curvature)   <-- the one 2-mirror can't reach
```
A TMA that nulls I–III is an **anastigmat**; nulling IV as well gives a flat field.

### 6.2 Petzval (field curvature) sum for mirrors
For mirrors in air the Petzval contribution of surface `i` is `±2/R_i` (sign alternates
with each reflection). Flat field requires:
```
Σ_i  (s_i * 2 / R_i) = 0          s_i = +1, -1, +1, ... per reflection
```
This is the condition that two strong same-sign mirrors fight and a third relieves —
hence the appeal of the Korsch geometry where M1/M3 vs M2 curvatures balance.

### 6.3 Practical design path for the layer
Korsch's papers give **closed-form** starting solutions; transcribe them as the seed,
then refine numerically. Recommended implementation:

1. Choose first-order layout: powers `φ1, φ2, φ3`, separations, stop location, and
   target `f`, `f/#`, FOV, back focus. (Korsch 1972/1977 give analytic relations.)
2. Express the four Seidel sums as functions of `{φ_i, K_i, separations, stop}`.
3. Solve `ΣS_I = ΣS_II = ΣS_III = ΣS_IV = 0` for the conics + one geometry DOF
   (4 equations; use the Korsch closed form as the seed so the solver lands in the
   right basin).
4. Emit Rx, hand to MACOS for the real-ray check.

> The full explicit Korsch conic/spacing expressions run several pages and are easy to
> transcribe with sign errors — pull them directly from Korsch (1977) or Schroeder's
> three-mirror chapter rather than reconstructing. Treat them as seed + validation, not
> as the production solver.

### 6.4 Off-axis / unobscured TMA
For the coronagraph branch: take the on-axis Korsch solution, then **bias the field**
(use an off-axis field point as the new axis) or decenter/tilt to clear the obstruction.
This is exactly the regime where the residual aberrations become non-rotationally-
symmetric and free-form surfaces (Section 7) earn their keep.

---

## 7. Free-form optics (wide field / unobscured)

### 7.1 Surface representation
Base conic + departure. Two common departure bases:

**Zernike (φ-polynomial) departure** — preferred for aberration-theory reasoning:
```
z(r,φ) = (c*r^2) / (1 + sqrt(1 - (1+K)*c^2*r^2))  +  Σ_j  a_j * Z_j(ρ, φ)
```
where `ρ = r / r_norm` and `Z_j` are Fringe/Standard Zernikes. Each Zernike term maps to
a specific field-dependent aberration via Nodal Aberration Theory (NAT) — that mapping
is what makes the design tractable rather than blind optimization.

**XY-polynomial departure** — common in fabrication/Code V/Zemax interchange:
```
z(x,y) = (c*r^2)/(1 + sqrt(1 - (1+K)*c^2*r^2))  +  Σ_{m,n} C_{mn} * x^m * y^n
```

Decide one canonical internal representation for the layer and convert on emit; Zernike
is the better internal choice because of the NAT correspondence.

### 7.2 Design method (the part worth following)
Use Bauer/Schiesser/Rolland's procedure rather than brute-force optimization:
1. Pick a **starting geometry** and rank candidates by their freeform-correction
   potential (their tiering method) — geometry choice dominates final performance.
2. Leave the rotationally-invariant 3rd-order aberrations **uncorrected** until after
   unobscuring; correct them as the system is folded.
3. Add freeform terms tied to the *specific* field-dependent aberration each one
   controls (per NAT), minimizing total surface departure (manufacturability).

This is the single most useful free-form reference to point CC at, and it's open
access (see Section 9).

### 7.3 NAT in one paragraph
When you break rotational symmetry, aberrations no longer have a single field center —
they develop *nodes* distributed across the field (binodal astigmatism, field-asymmetric
coma, etc.). NAT predicts where those nodes land as a function of each surface's
contribution, so freeform terms can be placed to move/cancel nodes intentionally. This
is the analytic alternative to wavefront-fitting-and-hope.

---

## 8. Reference prescriptions (regression fixtures)

Use these to validate that MACOS-emitted Rx reproduces published behavior. Build the
geometry from `f, D, m, β`, compute conics from Section 5/6, trace, and compare.

- **RC anchor — HST-like:** 2.4 m aperture, f/24 system, f/2.3 primary
  (`m ≈ 10.4`). Expected `K1 ≈ -1.0023`, `K2 ≈ -1.50`. (Famous numbers; a sign or
  β error shows up immediately here.)
- **Classical Cassegrain:** any `m`; `K1 = -1`, `K2 = -((m+1)/(m-1))^2`. For `m=4`:
  `K2 = -(5/3)^2 = -2.778`.
- **Dall–Kirkham:** same `f,D,m` as a Cassegrain, set `K2=0`, check on-axis spherical
  is nulled and off-axis coma is large.
- **TMA:** use a published Korsch design (JWST and Roman/WFIRST prescriptions are in the
  open literature) as a full three-mirror fixture once Section 6 is implemented.

Drop each as a seed prescription in the design layer **and** as a test case — they do
double duty (Sprint-1 coronagraph reuse of existing prescriptions + regression).

---

## 9. Primary sources

**Two-mirror + general telescope optics (closed-form conics, Seidel sums)**
- Schroeder, *Astronomical Optics*, 2nd ed. (Academic Press, 2000). The single best
  source to transcribe Sections 4–6 from; two- and three-mirror chapters give conic
  constants and third-order coefficients in closed form. (Equation forms above follow
  Schroeder's conventions.)
- Wilson, *Reflecting Telescope Optics I* (Springer). Encyclopedic complement; more
  on aberration balancing and real-world constraints. (RC conics ≈ his Eq. 3.109.)
- Sasian, *Introduction to Aberrations in Optical Imaging Systems* (Cambridge). Most
  code-friendly modern treatment of the aberration math underlying all of the above.

**Three-mirror anastigmats (Korsch TMA)**
- Korsch, "Anastigmatic three-mirror telescope," *Appl. Opt.* 11, 2986 (1972).
- Korsch, *JOSA* (1977) — closed-form solutions for systems corrected to third order.
- Korsch, *Reflective Optics* (Academic Press, 1991) — book-length treatment.

**Free-form**
- Bauer, Schiesser & Rolland, "Starting geometry creation and design method for
  freeform optics," *Nat. Commun.* 9, 1756 (2018). **Open access (PMC5931519)** —
  CC can fetch this directly; it's the procedural design method.
- Rolland, Davies, Suleski et al., "Freeform optics for imaging," *Optica* 8,
  161–176 (2021). Canonical review / taxonomy.
- Thompson, *JOSA A* 22, 1389 (2005) — NAT foundations.
- Fuerschbach, Rolland & Thompson, *Opt. Express* 22, 26585 (2014) — NAT extended to
  freeform surfaces.

---

*Convention reminder: every closed form above is stated in the `(m, β, k, p)`
convention of Section 2. Verify symbol-for-symbol against Schroeder before relying on
any single equation in production code.*
