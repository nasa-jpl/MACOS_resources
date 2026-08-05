# CTB diffraction layer — work-in-progress status (hand-off)

_Updated 2026-08-05. Latest work at "SESSION 4" below; older diagnosis kept._

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
Shared lambda/D across models (analytic first_order_properties.lamD_px) so the
FPM is identical -> apples-to-apples.  Verified:
  - bare:  compact peak 7.01e-2, full 6.03e-2 (matches direct compare).
  - coro (apod+fpm+lyot, lam/D=16.8): compact FPA peak 7.99e-8, full 5.29e-8
    (~1e6 suppression both); figure ctb_coro_compare_coro.png renders the
    occulted FPM core, Lyot pupil rejection, suppressed FPA -- numerically
    verified (Read of PNG).
  Usage: out=ctb_coro_compare('coro',true|false, 'apodizer'/'fpm'/'lyot',
  't/f', 'models',<struct w/ .name/.rx/.elt>, ...).  Pass your own decks via
  'models'.  Params: r_apod_m, r_apod_taper_m, r_fpm_lamD, r_lyot_frac.

NEXT (deferred): full s2s generation rules; add_pupil FarField fix; then the
tCtbProp test + README.  The generated ctb_planar_prop.in / ctb_prop_layout.m
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

## CORRECTED MODEL (Dave 2026-08-05) — mask NF1/NF2 = 4-surface EP quartet

My through-focus construction (Sin/focus/Sout/Sret) was WRONG.  NF1 and NF2
are the two halves of ONE near-field prop through a focal mask, but:
  - **NF1 = FarField sphere->plane** (EP sphere -> mask plane)
  - **NF2 = plane->sphere** (mask plane -> EP sphere)
Both use the EXIT-PUPIL sphere (Kr = -(EP->mask distance), a LARGE radius),
NOT a modest near-focus sphere.  Canonical model = manual CoroExample.in
(ret1_1/ret2_1/CoroMask/ret2_2/ret1_2) and Dave's e-mail model.  Each mask is
a QUARTET bracketed by flat FP-returns:

  FPreturn  (Flat,  Geometric, at mask plane,      zElt=0)      % image ref in
  EPreturn  (Conic, NF1,       EP sphere, Kr=-R,   zElt=+R)     % FF sphere->plane
  <mask>    (Flat,  NF2,       at mask plane,      zElt=1e30)   % plane->sphere; MASK HERE
  EPreturn2 (Conic, Geometric, EP sphere, Kr=-R,   zElt=-R)     % EP sphere back
  FPreturn2 (Flat,  Geometric, at mask plane,      zElt=0)      % image ref out

  - R = distance EP->mask (the exit-pupil sphere radius); EPreturn sits one R
    from the mask, psi=+beam; EPreturn2 psi=-beam (symmetric about the mask).
  - The two flat FP-returns (FPreturn/FPreturn2) sit AT the mask plane and
    bracket the quartet (Rx_Coro/CoroExample outer returns).

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
