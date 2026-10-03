# dyson5 beat 4c -- the native optimize (CALIB) on R4, walls on the chain

Written for: Dave (review) and CC (three engine findings, section 3).
Record: `dyson5_s4.txt`, `dyson5_s4.mat`, `dyson5_s4_r4n.in` (the deck of
record for the rung), `dyson5_s4_r4n_seed.in` (the CALIB deck: links + the
optimization block), `dyson5_s4_layout_r4n.png`, `dyson5_s4_maps_r4n.png`,
`dyson5_r4n_view3d.png` / `_viewyz.png`, the trade table's `native` row.
Runner: `dyson5_run(struct('stages', {{'s4'}}))`; tool `dyson_native.m`.

## 1. What was asked (addendum 10) and what the engine allows

"The native optimize with the smile/keystone operands."  CALIB's merit has
no distortion operand: its SPOT target is ONE number per (field, lambda) --
the maximum ray distance to the chief ray at the target element
(`smacos_compute.inc` ~:454, `design_optim.F` ~:213 sizes the objective
`obj_size=1`).  A centroid-position target per (field, lambda), which is
what smile and keystone are, needs an engine change (section 3.1).

So the beat does what the engine supports and holds the walls where they
can be held: CALIB minimises the blur over the deck's radius, conic and
position freedoms, in CHUNKS of 5 iterations; after every chunk the
engine's element state is read back, mapped into the chain's parameters,
proven by an identity check (every vertex, normal, radius and conic agrees
with the engine to 1e-9 m), scored in the ENGINE on the 7 x 7 record grid
and gated: smile and keystone inside 0.05 px (half the spec) and the
clearance gate PASS.  A breach restores the last accepted state.  That is
Dave's rule in its coarse form -- walls on ITERATES, never on reports --
and the finest form the engine allows today.

## 2. Method, in the order the runner does it

1. The chain's solved quantities are FROZEN into parameters so the chain
   can carry an engine-moved design: `P.grating_d` + `P.grating_m` (the
   groove period and signed order), `P.fpa_z` (the focus plane),
   `P.slit_dz` (the slit plane off the grating's centre -- a grating moved
   along the axis in the engine is, in the chain's grating-centred frame,
   the slit plane moved the other way).  Freezing reproduces the R4 chain to
   1e-12 m (asserted).
2. The emitter writes the CALIB block (`spectrometer_rx` 'opt'): 5 slit
   positions x 6 wavelengths (CALIB caps at 12 x 6), SPOT target at the FPA,
   equal weights, `OptChfRayPos` = the ray START (slit + 0.2 mm along the
   chief -- the header's convention; the header IS field 1, the engine's
   parse counts it), `ArrWaveLen` for wavelengths 2-6.  And the double-pass
   LINKS ('links'): `Link= i` on the return-pass copy of every surface the
   beam crosses twice and on the pre-FPA Reference (follows the FPA), so
   the engine's PERTURB / ROC / CONIC / ASPH perturbs move both passes as
   one physical surface (`macos_ops.F` LnkElt loops).  The block's flat face
   is written with opposite normals on its two passes and is NOT linked (a
   piston along psi would move the passes apart).  Gates:
   `tSpectrometerRx/test_links_make_the_return_pass_follow_the_first_pass`
   (with the no-link negative control) and
   `test_opt_block_configures_calib_fields_and_wavelengths` (CALIB reports
   3 fields, 2 wavelengths from the written block).
3. CALIB variables (9): grating DY + PIST (its position against the
   block+slit+FPA assembly -- a real alignment knob), block convex face ROC +
   CONIC (return pass linked), meniscus faces A and B PIST + ROC (linked),
   FPA PIST (Reference linked).  HELD: the groove period (RuleWidth has no
   DOF -- the band span on the FPA is reported after the solve as a check),
   the block's h^4/h^6 asphere (section 3.2), the grating radius (it sets the
   dispersion scale the period was solved for).
4. The mapping back (`dyson_native/map_`): engine `KrElt = -R` with psi
   toward the centre of curvature (the emitter's rule) gives centre = Vpt +
   |Kr| psi; the grating's centre is the frame shift s; the block's centre,
   radius and conic, the meniscus vertices (z, t) and curvatures (c =
   psi_z/|Kr|: the chain's centre sits at the vertex + 1/c along +z), the
   slit plane and the FPA plane all follow, translated by -s.  The identity
   check is what makes the clean re-emitted deck THE engine's design.

## 3. For CC -- three engine findings (none fixed here; engine fixes are CC's lane)

### 3.1 CALIB has no centroid-position operand (the ask)
A SPOT-class target whose objective per (field, lambda) is the chief or
centroid position on the target element against a supplied target position
(two numbers), or, equivalently, a per-field position offset to the SPOT
radius.  With it, smile and keystone are native operands (targets from the
chain's ideal map: v(x_i, lambda_j) = v(x_mid, lambda_j), u(x_i, lambda_j)
= u(x_i, lambda_mid)) and the walls move from between chunks into the
iterations.  `design_optim.F` sizes the objective at ~:213 and fills the
target at ~:408; `smacos_compute.inc` computes the per-field quantity at
~:443-461 (it already has the chief-referenced SPOT array).

### 3.2 OptAsph perturbation slice uses the Zernike count (`smacos_compute.inc:382`)
```
	    CALL MACOS_OPS(cmd,CARG,DARG,IARG,LARG,RARG,
     &                     OPDMat,SpotMat,WFErms,PixMat,
     &                     varZernArr(1:1), ! place holder
     &                     ptbArr(1:1), ! place holder
     &                     varAsphArr(ias+1:ias+n_optAsphArr(ie)),
     &                     ptbArr(ia+1:ia+n_optZernArr(ie)))      <-- n_optAsphArr(ie)
```
With no Zernike DOF on the element the asphere perturbation array passed
to ASPH_PERTURB is EMPTY (zero-length slice) while `IARG(2)` says there are
n coefficients: the op reads past the array.  With both kinds of DOF the
slice has the wrong LENGTH.  Fix: `ptbArr(ia+1:ia+n_optAsphArr(ie))`.
The dyson5 native stage keeps `P.native_asph = false` until this lands.

### 3.3 `OptRayGrid=` corrupts the heap (measured 2026-10-01, model 128, 41-pt deck)
Default (no keyword): `opt_npts = nGridpts/2 - 1 = 19`, 361 rays, runs and
exits clean.  `OptRayGrid= 21` (opt_npts 20): the solve completes, MATLAB
segfaults at exit inside its own interpreter (heap stomp; crash dump
`~/matlab_crash_dump.1377380-1`).  `OptRayGrid= 31`: hangs (300 s timeout).
`OptRayGrid= 41`: segfault during the first LM iteration.  The parse
(`msmacosio.inc` ~:226) checks only `opt_npts > mpts`; something in the
CALIB path is sized for the half-grid.  Reproducer: `dyson5_s4_r4n_seed.in`
with the keyword added (the scratch variants g21/g31/g41 of this session).
The emitter writes the keyword only on request (`opt.raygrid`).

### 3.4 THE BLOCKER -- CALIB's derivative loop steps the SPOT objective at the wavefront stride (`design_optim.F` ~:792)
Pinned in the bounds-checked CLI (`makems.sh debug`, built 2026-10-01; the
seed deck + `calib`):
```
forrtl: severe (408): Subscript #1 of the array YFIT has value 16385 which is greater than the upper bound of 30
  funcs_app  design_optim.F  line 785
```
`funcs_app` fills the objective value with `off=off+obj_size` (~:688,
correct) but the DERIVATIVE columns with `off=off+opd_size` (~:792 and
~:794, inside the `.not. OptBeam` / `OptBeam` branches).  `opd_size =
mpts*mpts = 16384`; `obj_size = 1` for SPOT (and `n_wf_zern` for
WFE_ZMODE).  For the WFE target the two are equal, which is why every
Telescope optimize has been fine; for SPOT the second (field, wavelength)
writes `dyda(16385:16385, i)` into a 30-row array -- a heap stomp that
surfaces as a MATLAB segfault during the solve, at exit, or as a hang,
depending on what sat past the array (measured all three; the gate's 3 x 2
run survived by luck).  Fix: `off=off+obj_size` at both sites.  Until it
lands `P.native_enabled = false` makes stage s4 refuse to run (a clear
error naming this section), and the emitter gate's multi-field CALIB leg
is marked INCOMPLETE (`tSpectrometerRx`, `assumeFail` with the reason),
its 1 x 1 leg still running.

### 3.5 The asphere differential step is round-off (`design_optim.F:198`)
With 3.2 fixed (macos 0d257ff) the OptAsph run no longer overruns, but the
LM fails at once with `gaussj: singular matrix (2)`: `das = 1d-10` (times
`das_scale_factor = 1d-05` per higher order) is the finite-difference step
for aspheric coefficients in base units.  The block's h^4 coefficient is
0.03 m^-3 at a 50 mm half-aperture, so the probe moves the sag by 1e-10 x
0.05^4 = 6e-16 m -- round-off -- and the derivative column is zero.  The
step has to scale with the coefficient's magnitude (or with the sag it
produces at the aperture), as `drc`/`dcc` effectively do for radius and
conic.  Measured in the bounds-checked CLI (no overrun, the LM message) on
`dyson5_s4_r4n_seed.in` with `OptAsph= 2 1 2` on the block face.

### 3.6 The mex dies where the CLI survives: LM failure path
The same deck in the CLI prints the lmlsq failure and returns to the
prompt; in the mex the process segfaults (crash dump 15:52:15, pid
1617419) on the same failure.  `calib_run`'s failure return (or what
`nls_optim_dvr` leaves allocated/deallocated when `lmlsq_success` is
false) is not safe for the binding.  Reproducer: the deck above through
`macos.calib()` with the asphere DOF on.

## 4. Result (after macos 0d257ff, three runs on 2026-10-01)

The engine fix lands: the emitter gate's 3 x 2 CALIB leg runs and the stage
runs end to end -- seed scored, CALIB in chunks of 5, read-back, identity
to 2e-16 m, engine score, walls, clearance, the clean re-emit.

| run | variable set | what CALIB did | wall / gate | result |
|---|---|---|---|---|
| 1 | 'all' (grating DY+PIST, block ROC+CONIC, meniscus PIST+ROC, FPA PIST) | 5 iterations moved the grating along the dispersion direction | keystone 0.003 -> **15.5 px**, chunk REJECTED, seed restored | R4n = R4 |
| 2 | 'blur' + the asphere terms | LM singular at once (3.5), mex crash (3.6) | -- | no record |
| 3 | 'blur' (block ROC+CONIC, meniscus PIST+ROC, FPA PIST) | 10 iterations, every LM step rejected: no variable moved (KrElt, KcElt, VptElt identical; spot sizes equal to 1e-13) | accepted, +0.79 mm | **R4n = R4** |

Non-vacuity: from a deliberate 0.3 mm defocus of the FPA, five CALIB
iterations on the same deck take the spot size from 97 to 21 um (probe
`probe_calib_moves.m`, scratch) -- CALIB moves when there is something to
gain; on R4 of record it finds nothing in its max-radius merit.

**Reading.**  (1) The walls earn their keep in the first five iterations
the engine ever ran on this deck: without a distortion operand CALIB buys
blur with 15 px of keystone.  (2) Under the blur-only freedoms R4 of record
is a local optimum of CALIB's max-radius spot merit as well as of the
chain's rms merit -- the native optimize confirms the ladder rather than
improving it.  (3) The asphere, the one freedom that could still buy blur,
waits on 3.5.  R4 of record stands; the deck's R4 numbers stand.

Decks: the native stage is on the record as a confirmation, not a result;
the keystone-15-px row is the slide's reason for operands.
