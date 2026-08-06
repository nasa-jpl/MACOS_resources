# CTB diffraction layer — work-in-progress status (hand-off)

_Updated 2026-08-05. Latest work at "SESSION 5" (performance) below; older kept._

## SESSION 5 (2026-08-05) — performance pass (centering, sampling, mask/Lyot optimize)

Dave: "find better performance -- the hard occulter is not centered, the
resolution is poor, and I'm not sure of the best mask/Lyot radii for
3-7 lambda/D; check the literature."  Three fixes + a sweep:

1. **CENTERING BUG (half a pixel) -- FIXED.**  The mask builders centred
   disks/apodizers on `c=(N-1)/2`, but MACOS's FarField/NF2 focus lands on
   the FFT DC pixel `floor(N/2)` (0-based) = 1-based N/2+1 (verified: focus
   at [257,257]/512, [513,513]/1024).  So every occulter sat half a pixel
   off the beam -> asymmetric residual leak.  Consolidated the duplicated
   builders into shared `ctb_mask_disk.m` / `ctb_mask_softcircle.m` centred
   on `floor(N/2)`; the vortex singular pixel and all beam_radius_ centres
   moved to `floor(N/2)+1` too.  The vortex benefited most (an off-centre
   singularity leaks badly): its advantage over the hard occulter went
   1.7x -> 2.9x.

2. **SAMPLING -- default bumped 512 -> 1024.**  lambda/D at the FPA = N/pupil
   px (pupil fixed at the deck's nGridpts=255): N=512 gave 2.0 px/lambda/D
   (Nyquist edge, poor), N=1024 gives 4.0 px, N=2048 -> 8.0.  Bumping
   model_size zero-pads the pupil -> finer PSF, no Rx change.  All analysis
   drivers (contrast/planet/bandpass/vortex) now default N=1024; the FPA
   panels resolve clean symmetric Airy rings.

3. **MASK/LYOT OPTIMIZE -- `ctb_optimize_masks.m` (NEW).**  Literature box
   (Wikipedia Coronagraph; Sivaramakrishnan+ 2001; HCIT class): occulter
   ~1-3 lambda/D, Lyot ~75-90% pupil.  Pure mean-contrast in 3-7 lambda/D
   pushes monotonically to bigger occulter + smaller Lyot (both reject more
   light -> throughput + IWA cost), so the driver also records throughput
   (Lyot area fraction r_lyot^2) and a contrast/throughput knee.  A WIDENED
   sweep (occulter 2.0-3.5, Lyot 0.45-0.75) found a ROBUST, ISOLATED
   contrast null at **r_fpm = 2.70 lambda/D** (a diffraction resonance of
   this bench's occulter/Lyot structure): C ~ 3-4e-7 there vs ~1e-5 at
   2.9-3.2 lambda/D.  Confirmed at N=1024 (persists, localises to 2.70,
   broad in Lyot 0.40-0.50 -> NOT a grid artifact).  **Shipped defaults:
   r_fpm=2.70 lambda/D, r_lyot=0.50** (deep contrast at 25% throughput;
   Lyot is the user's throughput dial).  Dave chose "widen the search
   first"; the sweep box is now wide enough that the optimum is interior,
   not on an edge.

**NET PERFORMANCE GAIN (old 3.0/0.85 @ N=512 -> new 2.70/0.50 @ N=1024):**
dark-zone mean contrast 4.6e-6 -> **2.9e-7 (~16x deeper)**, coro FPA peak
1.9e-5 -> 1.4e-6.  PROPER arbiter still 1.0000 / corr 1.000000 at the finer
sampling (diffraction layer stays validated).  Bandpass: the deeper, sharper
2.70 null is MORE chromatic (mono 2.4e-7 -> broadband 4.9e-7, 2.1x over 10%,
vs the old shallow occulter's 1.02x) -- an honest deep-vs-broadband trade.

4. **TRAIN RENDER ray-gap -- FIXED.**  The both-endpoints-in-window segment
   test dropped the physical beam segment connecting two real optics that
   bracket an off-bench reference-sphere detour (beam appeared to vanish
   between elements).  Now keeps all in-window crossings and draws ONE
   polyline through them -> continuous fold path DM1->FPA.

5. **SOURCE->OAP1 leg -- ADDED.**  The window was built only from the
   DM1..FPA station vertices, so OAP1 (element 1) and the source fell off
   the left edge.  Now the window also spans OAP1's vertex + the source
   ChfRayPos (get_src_fov), the source is marked with an orange star, and
   OAP1 is labelled (dropped lower to clear DM2, which sits at nearly the
   same X on this folded bench).  Full path source->OAP1->DM1->...->FPA
   is shown.

---


## SESSION 4 (2026-08-05) — Dave's hand-built decks + comparison driver

Dave hand-built two correct decks (supersede my generated ctb_planar_prop.in):
- **`ctb_dcr.in`** (31 elts) — "compact" model: NFPlane DM1->DM2, then NF1+NF2
  through Focus23 / FPM / FieldStop (the 4-surface FPreturn/EPreturn quartet),
  FarField ExitPupil->FPA.  Missing the inter-optic plane-to-plane props.
- **`ctb_s2s_dcr.in`** (44 elts) — "full" surface-to-surface model (adds the
  p2p props).  Full s2s generation rules deferred (Dave: "save for later").
Both are PRE-ALIGNED (ORS/SRS/FEX already applied + saved); load and produce a
CENTRED PSF at [257,257]/512 with dx=2.404e-5 m, no runtime setup needed.

**Both models agree** (bare, no coro): peak-norm corr 0.9989, EE within <0.3%
by +/-10px; compact peak slightly higher / core slightly tighter (it omits the
p2p diffraction).  Station indices:
  compact: DM1=2 DM2=5 Apodizer=13 FPM=17 Lyot=20 ExitPupil=30 FPA=31
  full:    DM1=2 DM2=5 Apodizer=16 FPM=22 Lyot=27 ExitPupil=43 FPA=44

**`ctb_coro_compare.m`** — parameterized compact-vs-full comparison driver
(SHIPPED this session).  Side-by-side INT at DM1/DM2/Apodizer/FPM/Lyot/
ExitPupil/FPA (rows) x models (cols).  Coronagraph masks (apodizer/FPM/Lyot)
applied IN MATLAB (macos.apodize multiply, reset_trace=false, continue prop).
Shared lambda/D across models (analytic first_order_properties.lamD_px) for the
DISPLAY scale only.  Verified:
  - bare:  compact peak 7.01e-2, full 6.03e-2 (matches direct compare).
  - coro (apod+fpm+lyot, 3 lam/D hard occulter): compact FPA peak 1.89e-5,
    full 4.28e-5; suppression 3.71e3 (compact) / 1.41e3 (full).

**SUPPRESSION CORRECTION (2026-08-05, finding 2).**  The earlier "~1e6
suppression both" number was an ARTIFACT of an FPM-sizing bug: the occulter
radius was computed as `r_fpm_lamD * lamD_px_FPA(16.8) * dx_FPM`, mixing the
**FPA** plate scale into the **intermediate FPM** plane -> occulter ~8x too
large in radius -> far more starlight blocked -> inflated apparent
suppression.  Fixed: the occulter is now sized in FPM-LOCAL lambda/D
(`lam*R/D_beam`, R = the NF1-sphere zElt at FPM-1, D_beam the geometric beam
diameter measured on that sphere) and painted on a DETERMINISTIC focal grid
`dx_f = lam*R/(N*dx_sphere)` -- verified equal to the engine's `dx_at(FPM)`
to ratio 1.0000 in the validated quartet (so the old "dx_at garbage at NF2"
gotcha was itself an artifact of the WRONG through-focus construction; with
Dave's quartet dx_at at NF2 is now trustworthy, but we compute dx_f
deterministically regardless).  A ~1e3 suppression is physically correct for
a HARD-EDGE occulter with no apodization; 1e6-class needs band-limited /
apodized masks (deferred).  The Lyot radius is now keyed to the BARE
geometric pupil radius (finding 4), not a radius measured from the post-FPM
intensity (which the Babinet ring inflates).  The compact-vs-full coro gap
(factor ~2.6) is the finding-3 point -- compact omits inter-optic diffraction;
PROPER is the arbiter, not model-vs-model agreement.
  Usage: out=ctb_coro_compare('coro',true|false, 'apodizer'/'fpm'/'lyot',
  't/f', 'models',<struct w/ .name/.rx/.elt>, ...).  Pass your own decks via
  'models'.  Params: r_apod_m, r_apod_taper_m, r_fpm_lamD, r_lyot_frac.

**PROPER ARBITER (2026-08-05, finding 3) -- `ctb_proper_compare.m`.**
Model-vs-model agreement is NOT validation.  The arbiter is a per-leg
cross-compare against MATLAB PROPER on the CTB deck's own sampling
(recipe validation-ladder item 1).  Ran it for the NOVEL kernel the whole
chain rests on -- the FPM through-focus leg (feed sphere FPM-1 -> FPM
focus), macos NF1 sphere->plane vs PROPER prop_lens(R)+prop_propagate(R),
beam_ratio=1 so PROPER pitch == macos pitch bit-for-bit.  RESULT:
  - dx_focal ratio macos/PROPER = **1.0000** (both 2.1267e-5 m)
  - peak-norm correlation = **1.000000**
  - centroid offset = **0.000 px**
So the CTB geometry reproduces PROPER's diffraction focus EXACTLY.  The
diffraction layer is sound; the compact-vs-full suppression gap (factor
~2.6) is a modeling-fidelity difference (compact omits inter-optic
diffraction), not a bug.  (Needs `~/dev/proper_matlab` on the path; the
macos leg runs standalone and skips the PROPER column if absent.)

**NEW WORK A/B/C (2026-08-05).**
- **A -- `ctb_train_render.m`:** deck-grade layout figure, XZ fold plane
  (YZ/XY degenerate on this planar bench), compact stacked over full,
  shared axis scale + 500mm scale bar, mask planes marked (dashed
  verticals + top labels off the ray lines), EP reference spheres as
  magenta rings.  Rays clipped to the physical bench window (the
  off-bench EP-return spheres at x~-3700 are hidden).  Compact and full
  render near-identically -- correct, they share the physical optics.
- **B -- `ctb_contrast.m`:** radial dark-zone contrast vs lambda/D at the
  FPA, Strehl-normalised to the bare (no-mask) peak, reusing the coro/
  helpers (radial_contrast / dark_zone_metrics).  lambda/D is
  DETERMINISTIC (lam*R_fpa/D_ep / dx_FPA = 2.006 px) -- SYSPROP returns 0
  on this finite-conjugate deck and the Airy-null finder gave a spurious
  16.8.  2-DM -> full annular dark zone; 3-15 lam/D: mean 4.6e-6,
  median 4.7e-7, floor 2.0e-11 (n=2732 px).  Also repointed
  ctb_coro_compare's shared_lamD_ to the same deterministic geometry
  (was 4.0/16.8 spurious).
- **C -- `ctb_planet.m`:** off-axis companion in the dark zone, incoherent
  star+planet sum.  KEY: the CoroExample WINDOW/FFP recipe does NOT
  displace a companion here -- Dave's "vertices ON the chief ray" rule
  makes the FPM AND FPA grids re-centre on the tilted chief after
  ffp+ors/fex, so the planet lands back at pixel centre and the occulter
  clips it (verified).  Instead inject the companion as a PUPIL PHASE
  RAMP on the complex field at DM1 (macos.apodize_complex): k cycles
  across the pupil -> planet at exactly k lambda/D, at the FIXED centred
  FPM (clears the occulter) and FPA.  Same "modify cfield in MATLAB"
  contract as the masks.  Verified: 6 lam/D target -> planet peak at
  5.98 lam/D; at flux ratio 1e-3 the planet (5.5e-4) stands above the
  star residual (2.7e-4) -- the difference panel recovers it cleanly.

- **D -- `ctb_bandpass.m`:** finite-bandpass CTB, nwf wavelengths summed
  INCOHERENTLY, mono vs broadband dark-zone contrast.  Focal masks rebuilt
  per lambda (hard rule).  TWO findings baked in:
  1. `set_src_wvl` DOES propagate to the grid (dx_at(FPA) = 2.28/2.40/2.52
     e-5 m at 475/500/525 nm) -- verified.
  2. MACOS's FarField FPA RE-GRIDS per lambda, so each mono PSF is the same
     pixel size on the NxN array; naively summing there CANCELS the
     chromatic effect (first cut gave mono==broadband exactly).  Correct
     broadband resamples each lambda's PSF onto ONE COMMON PHYSICAL detector
     grid (nominal-lambda pitch), flux-conserving -- what macos.compose does
     internally.  Then the broadband PSF visibly smooths (rings wash out)
     and contrast degrades a physically-sensible 1.018x over a 10% band with
     a FIXED-METRES occulter (fpm_size='meters', default -- subtends varying
     lambda/D -> chromatic).  fpm_size='lamD' gives the achromatic reference
     (mask auto-scales -> mono==broadband to precision; isolates "does the
     machinery add chromatic error" -> no).  compose() sums only engine
     traces (no MATLAB masks), so the sum is MATLAB-side, mask-aware.

- **E -- `ctb_vortex.m` + engine-scope findings (STOP-and-FLAG).**  The
  work order's E was framed engine-first (add cfield_apodize_c, fix dx_at
  at NF planes, guard xp_fnd, THEN build the vortex).  Scoping the engine
  first (as directed) changed the plan -- **the vortex needs NO engine
  change**:
  - **E.1 cfield_apodize_c ALREADY EXISTS.**  `macos_api_mod.F90:5239`
    `cfield_apodize_complex(OK, MASK_RE, MASK_IM, N, iElt)` multiplies
    WFElt by a complex NxN mask in place -- exactly the stage-2 API.
    Surfaced as `macos.apodize_complex`; used already in work C and here.
    So the charge-m scalar vortex is a pure-MATLAB phase mask on that API.
    `ctb_vortex.m`: exp(i*m*theta) at the FPM (singular pixel -> phase 0,
    O(1/N^2) defect, documented), Lyot rejects the star.  Charge-6 vs hard
    occulter, 2-15 lam/D: hard mean 6.7e-6, vortex 3.9e-6 (1.7x deeper) AND
    smaller inner working angle.  Idealized exp(i*m*theta) is achromatic by
    construction (composes with the bandpass driver).
    HONESTY CAVEAT: the Lyot panel shows a gradient, not the textbook
    "all starlight in a ring outside the pupil" -- the CTB exit pupil is
    not matched/circularized for an idealized vortex, so 1.7x is real but
    modest, not a perfect-vortex total rejection.  A bench designed for a
    vortex (clean circular matched Lyot) would do far better.
  - **E.2 dx_at at NF planes: NO LIVE BUG on the validated quartet.**  The
    old "2.6e26 garbage at NF2" gotcha was an artifact of the WRONG
    through-focus construction; on Dave's committed quartet dx_at(FPM)
    matches the deterministic dx_f to ratio 1.0000 (finding 2).  No fix
    needed; the drivers compute dx_f deterministically regardless.
  - **E.3 xp_fnd EltID guard: GENUINELY OPEN (the one real engine item).**
    `xp_fnd` (macos_api_mod.F90:5396) runs FEX on nElt-1 UNCONDITIONALLY
    (IARG(1)=nElt-1, line 5422) with no check that nElt-1 is Return/
    Reference.  On a deck whose penultimate element is a powered optic this
    silently mis-places the exit pupil.  A guarded version would return
    FAIL with a reason when EltID(nElt-1) is not Return/Reference.  THIS
    needs Dave's go (engine edit + relink chain).  ==> FLAGGED to Dave.

NEXT (deferred, engine-first items need Dave's go): E.3 xp_fnd guard;
full s2s generation rules; add_pupil FarField fix; tCtbProp test + README.  The generated ctb_planar_prop.in / ctb_prop_layout.m
are SUPERSEDED by Dave's hand decks for the terminal/mask structure -- keep for
the builder logic (NFPlane zElt rule, _Sret flip-back) but realign to the
4-surface EP quartet when regenerating.

---

_Older: paused 2026-08-04, mid-diagnosis; see "OPEN ITEM"/"RESOLVED" below._

## What exists (branch `pol-ifo`, in `bench_ctb/`, all UNTRACKED so far)

- `Coro_propagation_summary.md` — recipe of record (copied in per work order).
- `ctb_prop_layout.m` — builder: loads `ctb_planar_stageF.in`, traces to get the
  chief-ray geometry, inserts the diffraction reference/return/sphere surfaces,
  bumps `nGridpts` to 255, writes `ctb_planar_prop.in`. Regenerable.
- `ctb_planar_prop.in` — the augmented deck (currently **29 elements**).

Not yet written: `ctb_prop_legs.m`, `ctb_propagate.m`, `+ctbmask/*`, `tCtbProp.m`.

## Augmented deck element map (current, 29 elts)

`OAP1(1) DM1(2) P1_start(3,NFPlane) P1_end(4) DM2(5) OAP2(6)
Focus23_Sin(7,NF1) Focus23(8,NF2) Focus23_Sout(9) OAP3(10)
Apodizer_Pst(11,NFPlane) Apodizer(12) OAP4(13) FPM_Sin(14,NF1) FPM(15,NF2)
FPM_Sout(16) OAP5(17) Lyot_Pst(18,NFPlane) Lyot(19) OAP6(20)
FieldStop_Sin(21,NF1) FieldStop(22,NF2) FieldStop_Sout(23) OAP7(24)
Backend(25) OAP8(26) FP_return(27) ExitPupil(28,FarField) FPA(29)`

## Runtime setup (what the driver must do after load)

```matlab
macos.stop(2);                                   % stop at DM1 (as example_ctb.m)
for e = [7 9 14 16 21 23], mmacos('ors_run', double(e)); end   % refine 6 near-focus spheres
macos.fex(1);                                    % place terminal exit-pupil sphere (nElt-1)
```

## VERIFIED GOOD

- Deck loads and traces. `ok_pass = 333` (full diffraction grid) is preserved
  **end-to-end, source through OAP8**.
- **Chief-ray intersections match stageF to ~1e-13 at EVERY real optic
  OAP1..OAP8** (the decisive stageF-vs-prop comparison — `/tmp/ctb_cmp.m`).
  The negative-length reference legs (OAP3, Apodizer, OAP7 show chief L<0) are
  INTENDED and correct — not a bug.
- Field reads clean at every intermediate pupil and focus: Apodizer, FPM, Lyot,
  FieldStop, Backend, OAP8 all have sensible sum (~0.99) and pupil dx (~378 µm).
- All 6 near-focus sphere pairs (NF1 → NF2 focus → Return/Geometric) work.

## RESOLVED 2026-08-04 (session 2): the chief-direction-reversal bug

Dave spotted "ray 1 after OAP7 goes the wrong direction."  Confirmed by
comparing chief-ray DIRECTION (not just position) stageF-vs-prop: dot=-1
(reversed) across the OAP3/OAP4 and OAP7/OAP8 blocks; the ray BUNDLE blew up
at OAP8 (spread 1.3e8).  Root cause: each through-focus block was missing the
SECOND return surface.  Rx_Coro's block is FOUR surfaces:
  Sin(NF1) -> focus(NF2) -> 1stPropEnd(Return sphere, one radius PAST focus,
  reverses chief) -> **CorMaskReturn (Return FLAT, back AT the focus, flips
  chief forward again)** -> next OAP.
I had only three (no CorMaskReturn), so the reversed bundle hit the next
powered mirror and diverged.  FIX: added a `_Sret` flat Return (Element=Return,
Surface=Flat, psi=+uin, at the focus vertex, Kr=zElt=-1e22) after each `_Sout`.
Also made Sin/Sout a symmetric mirror pair about the focus (same |Kr|=r, Sin
z=+r, focus z=-r, Sout z=-r), matching Rx_Coro exactly.

RESULT: all real optics OAP1..FPA now match stageF dot=+1.0, posdiff~1e-13.
OAP8 bundle spread 3.84 (was 1.3e8).  **PSF FORMS at the FPA: peak 0.19,
sum 6870, dx=-1.99um** (cf Rx_Coro peak 0.25).  Deck is now 32 elements.

New element map (32 elts): OAP1(1) DM1(2) P1_start(3) P1_end(4) DM2(5) OAP2(6)
Focus23_Sin(7) Focus23(8) Focus23_Sout(9) Focus23_Sret(10) OAP3(11)
Apodizer_Pst(12) Apodizer(13) OAP4(14) FPM_Sin(15) FPM(16) FPM_Sout(17)
FPM_Sret(18) OAP5(19) Lyot_Pst(20) Lyot(21) OAP6(22) FieldStop_Sin(23)
FieldStop(24) FieldStop_Sout(25) FieldStop_Sret(26) OAP7(27) Backend(28)
OAP8(29) FP_return(30) ExitPupil(31,FarField) FPA(32).
Runtime ORS spheres now = [7 9 15 17 23 25]; fex(1) for the EP.

## CORRECTED MODEL (Dave 2026-08-05, table re-synced to the committed decks 2026-08-05)

My through-focus construction (Sin/focus/Sout/Sret) was WRONG.  NF1 and NF2
are the two halves of ONE near-field prop through a focal mask, but:
  - **NF1 = FarField sphere->plane** (EP sphere -> mask plane)
  - **NF2 = plane->sphere** (mask plane -> EP sphere)
Both use the EXIT-PUPIL sphere (Kr = -(EP->mask distance), a LARGE radius),
NOT a modest near-focus sphere.

**THE VALIDATED TEMPLATE IS DAVE'S COMMITTED DECK `ctb_dcr.in`, NOT any
hand-transcribed table.**  The as-committed Focus23 quartet (elts 7-10),
read verbatim, is the convention of record.  Each focal mask is a
**4-surface** quartet (NOT five — there is no trailing FPreturn2):

  FPreturn   (Return, Flat,  Geometric, at mask plane,  zElt = 1e22)
  EPreturn   (Return, Conic, NF1,  EP sphere, Kr=-R,    zElt = +R)
  <mask>     (Return, Flat,  NF2,  at mask plane,        zElt = 1e22)  % MASK HERE
  EPreturn2  (Return, Conic, Geometric, same sphere,    zElt = +R)    % SAME SIGN as EPreturn

  - **All four surfaces are `Element=Return`** (not Reference).
  - **Both sphere zElts are +R, identical to all digits** (e.g. Focus23:
    +7017.8526119080789).  The engine's NF1 chirp uses zStart=zElt(EPreturn)
    and zEnd=zElt(iElt+1)=zElt(EPreturn2); the mask sandwich is transparent
    (no spurious defocus) IFF **zEnd == zStart, SIGN INCLUDED**.  EPreturn2
    at **-R** produces S~2R — the exact defocus failure the round-trip
    investigation diagnosed.  Do NOT write -R.
  - **FPreturn and the mask both carry zElt=1e22** (the "plane" radius),
    NOT 0 and NOT 1e30.
  - R = distance EP->mask (the exit-pupil sphere radius, = -Kr).  The sphere
    vertex sits one R on the incoming (-chief) side of the focus (Focus23:
    focus x=+3274.6, EPreturn vertex x=-3743.3, R=7017.85), psi=+chief
    (pointing toward the focus / centre of curvature).  EPreturn and EPreturn2
    share the SAME pose and Kr (they are the same physical sphere, entered
    then exited).
  - Terminal FF leg mirrors this as a 3-surface triple: FP_return(Return,
    Flat, Geometric, zElt=1e22) -> ExitPupil(Return, Conic, FarField, Kr=-R,
    zElt=+R) -> FPA(FocalPlane, Flat, Geometric, zElt=1e22).  zElt=+R (positive)
    for the FarField sphere, same convention as EPreturn.
  - NFPlane p2p leg (DM1->DM2): P1_start(NFPlane, zElt=-L) -> P1_end(Geometric,
    zElt=0); the DIFFERENCE = -L = chief L (Focus23 example: L~399.94).

  The manual `CoroExample.in` (ret1_1/ret2_1/CoroMask/ret2_2/ret1_2) and
  `Rx_Coro.in` are the upstream lineage, but where a table and the deck
  disagree, **the deck wins** — it is the numerically-validated artefact.

### Alignment procedure (Dave point 1) -- do NOT hand-set psi/vpt
For each near-field prop: **ORS iElt on the STARTING reference element, then
SRS iElt+1 iElt** to slave/align the paired end element to it.  (CoroExample
.jou: `ors 5; ors 7; fex 15`.)  ORS aligns psi to the chief + fits the sphere
radius; SRS solves the partner's zElt/pose from the OPL between them.  Some of
my reference surfaces are currently NOT beam-aligned -> use ORS/SRS, not the
set_psi/set_vpt hand-alignment I added.

### Terminal (unchanged conclusion)
CoroExample terminal = ret1_3(flat,Geom) -> ret2_3(EP sphere, FarField) ->
foc_pln.  Matches Rx_Coro.  add_pupil SHOULD emit FarField for the EP->FP leg
(fix Telescope.m if not).

### zElt audit (session 3) -- these are NOW correct per the manual:
NFPlane p2p: start zElt=-L, end zElt=0 (Delta = -L = chief L).  Fixed for
P1/Apodizer/Lyot legs.  FarField: zElt = EP->image = L.  The through-focus
zElts will be RESET by the quartet rebuild + ORS/SRS.

## (superseded) REMAINING — terminal PSF is OFF-CENTRE + FarField terminal structure

1. PSF forms (peak ~0.19, sum ~6470, dx=-1.99um) but lands OFF-CENTRE at
   ~[284,407] in the 512 grid (centre 256).  APPLIED DAVE'S RULE (2026-08-04):
   every diffraction surface axis || chief, vertex at chief incidence point.
   In ctb_prop_layout.m the reused markers (Focus23/FPM/FieldStop/FPA) now get
   psi=cd(chief dir), vpt=cp(chief pierce) via set_psi/set_vpt.  The terminal
   ExitPupil is verified FULLY axis-aligned at runtime:
     chief . (EP->FP) = 1.000000, FP perp-dist from chief-through-EP = 0.0 mm,
     EP vtx = chief pierce exactly, psi = chief dir (antiparallel EP->FP form
     centres better: parallel gave [370,496], antiparallel [284,407]).
   So the residual decentre is NOT vertex/axis geometry -- it is the
   TRANSVERSE output-grid ROLL (xGrid/yGrid about the chief axis).  The deck
   header xGrid is the GLOBAL transverse basis; on a folded bench the chief
   axis at the FPA is rotated ~0.6 deg in XY vs that xGrid, so the far-field
   output grid is rolled and the on-axis PSF projects off-centre along one
   axis (row ~centred, column ~150px off).
   REFERENCE: Rx_Coro (on-axis) FPA peak is dead-centre [513,513]/1024; its
   EP is trivially axis+roll-aligned (all on the z-axis).  The CTB needs the
   far-field output frame's xGrid to track the LOCAL chief frame at the EP,
   not the global source xGrid.
   OPEN QUESTION FOR DAVE: what sets the far-field output-grid transverse
   roll -- the ExitPupil element's xGrid/TElt frame, the source xGrid, or a
   WINDOW/Tout frame?  Need the convention to roll the FPA grid onto the
   local chief frame so the on-axis PSF centres.
   - Runtime terminal config that produces the PSF: fex(1); EP vtx=chief
     pierce, psi=-chiefdir (antiparallel), Kr=zElt=-|EP-FP|; FPA zElt=1e22.
2. Dave: add_pupil SHOULD emit FarField for the EP->FP leg; if it doesn't,
   FIX add_pupil (Telescope.m ~2680) then align the CTB terminal to Rx_Coro's
   FocalPlane(Return) -> ExitPupil(Return,FarField) -> FocalPlane triple.
   Current CTB terminal = FP_return(30) + ExitPupil(31,FarField) + FPA(32),
   which is that same triple -- verify it matches Rx_Coro once centred.

## (historical) OPEN ITEM — the terminal OAP8 → FPA far-field leg

Symptom: **FPA reads peak=0, sum=0, dx=3.5e-17** (no PSF). Everything upstream
is fine (verified above).

Root cause identified: the terminal triple is built wrong vs the proven
diffraction template. The ONLY terminal that forms a real PSF in the tree is
**Rx_Coro** (`pymacos/tests/Rx/Rx_Coro.in`), which is a **2-surface** terminal:

```
ExitPupil (Return, Conic, PropType=FarField)  ->  FocalPlane
```
with, from a working trace (Rx_Coro FPA: peak 0.25, sum 23.7, dx=-5.8µm):
- ExitPupil: Vpt = FP_vpt + radius along the +z (incoming) side, i.e. ONE FULL
  RADIUS from the FP; `psi = unit(FP - EP)` (points EP->FP, toward the image /
  centre of curvature); `KrElt = zElt = -radius` (both NEGATIVE); NOTHING
  between the last optic and this sphere.
- FocalPlane: at the image; `psi` faces back toward the EP; `zElt = 1e22`.

My current deck instead has a **3-surface** terminal
`FP_return(flat at FPA) -> ExitPupil(sphere before FPA) -> FPA`, which scrambles
the pairing (FP_return sits between OAP8 and the EP; the EP is on the wrong side
at half the radius). `add_pupil`'s FP_return construct (Telescope.m:742) is a
GEOMETRIC layout aid; the DIFFRACTION terminal is the Rx_Coro 2-surface form.

### DECISION PENDING (Dave): terminal structure
- Option A (recommended): rebuild terminal as the Rx_Coro 2-surface
  `ExitPupil(FarField) -> FPA`, EP one radius from FPA on the incoming side,
  drop FP_return.
- Option B: keep the add_pupil 4-surface `OAP8 - FP_return - EP - FPA` (Dave
  flagged this); needs FarField wired correctly — not yet forming a PSF.

## Reference decks
- `pymacos/tests/Rx/Rx_Coro.in` — the proven NF/FF encoding + working FarField
  terminal (elts 20-21). PSF forms on load.
- `mmacos/design/examples/tma_onaxis/tma_onaxis.in` — add_pupil FP_return/
  ExitPupil/FP triple, but GEOMETRIC (not the diffraction terminal).

## Gotchas already burned (don't rediscover)
- `.in` regexes MUST be line-anchored: `iElt` is a substring of `psiElt`, so an
  unanchored `iElt=\s+\d+` clobbers the leading digit of psiElt. Fixed in
  `ctb_prop_layout.m` (set_prop/set_z/renumber all use `^\s*` + `'lineanchors'`).
- `dx_at` at an NF2 focal plane returns garbage (e.g. 2.6e26) — cosmetic readout,
  field still propagates. Use `abs()` and don't trust dx at NF2 planes.
- MATLAB on this box: `/Applications/MATLAB_R2024a.app/bin/matlab`; the mmacos MEX
  is `mmacos.mexmaca64`; `MACOS_HOME` set. `startup.m` prints a harmless pyenv
  error. Run sandboxed for license.
- CLI exe (if needed): `/Users/dcr/dev/macos/build_release_gfortran/bin/macos`.
