# IFO Polarization — CURRENT STATE (2026-07-27)

**Branch:** `pol-ifo` (MACOS_resources), off `bench-builder`, merged `ifo-l2` + `pol-core`
**Engine:** macos `pol-core`, built gfortran @ `build_release_gfortran`
**mmacos mex:** `src/mmacos.mexmaca64` (rebuild with `unset FC && make` if the engine changes)

---

## SLICE 1 — DONE (Twyman-Green, coated BS, ray-level Jones)

Committed + pushed (`5a65c43` + `cafc53c`). Polarization-honest TG at the
canonical 45° fold: arm-differential D=0.0721 / retardance=0.0835 rad,
verified against a full-train textbook Fresnel closed form to all digits;
PSI pupil variation at round-off; two gates green. Packet:
`~/dev/macos/REVIEW_POL_IFO_SLICE1_2026-07-27.md`. Harness:
`example_bench_ifo_pol.m`.

## SLICE 2 — DONE (BS-AOI vs mechanical-clearance trade)

**Landed this session; HOLDING FOR REVIEW, then push.** Packet:
`~/dev/macos/REVIEW_POL_IFO_SLICE2_2026-07-27.md`.

- **Builder:** `twyman_green.m` gains `BS_AOI` (deg, default 45). Fold turn
  = 180−2·AOI; 45° pinned to the exact `[0;-1;0]` literal → default rig
  BIT-IDENTICAL (verified both arms). Compensator + substrate + return
  faces track the BS normal automatically (shared `bs` token).
- **Harness:** `example_bench_ifo_pol_slice2.m`. Sweeps AOI 45→15° (13
  points), re-emits per angle, trace-clean check (100% rays kept
  throughout), per-angle coated Fresnel Gate 1 (≤1.1e-14).
- **Curve gate:** engine mean D/ret vs full-train closed form
  (external-reflect × net-transit-pair vs transit-pair × internal-reflect,
  Δtp=1). All 13 points pass at relD ≤ 3e-13. **Non-vacuity:** drop the
  transit pair (Δtp=0) → misses D by 358% → fails the gate. Confirmed.
- **Three scores:** (1) fringe visibility V=sqrt((1+sqrt(1−D²)cos ret)/2),
  cost 1.5e-3 (45°) → 1e-5 (15°); (2) PSI pupil-variation stays round-off
  (6e-7…2e-6 nm) — a result, collimated common-path; (3) beam-envelope
  clearance (ray_hist, MIN_SEP style) 43.6 mm (45°) → 3.5 mm (15°),
  closest node = compensator return face.
- **Knee: AOI 17.5°** at the 10 mm clearance floor, visibility cost
  1.9e-5 (~78× better than 45°). Figure
  `bench_ifo_pol_slice2_trade.png`; data `bench_ifo_pol_slice2_results.mat`.
- **tBench 5/5** green (only builder consumer touched).

### PUSH when Dave clears (per session rule 5 / slice-1 protocol)
```
cd ~/dev/MACOS_resources && git push   # pol-ifo -> origin
cd ~/dev/macos          && git push    # pol-core -> origin (slice-2 packet)
```
Both repos committed this session; NO PUSH until reviewed.

## SLICE 3 — NEXT (polarizing-PSI variant)

Ideal polarizer/waveplate (the shipped `PolElt`, elt types 15/18) in the
collimated normal-incidence legs + comparison against the slice-1/2
baseline. This is where the PSI pupil-variation (score 2) is designed to
grow above round-off.

## Known traps

- **Mac FC env:** shell has `FC=gfortran-16`; mmacos Makefile rejects.
  `unset FC && make`.
- **Coatings need a loaded Rx:** apply `macos.coating` AFTER `load_rx`.
  `load_rx` clears coating state (re-coat after every load).
- **RayE is per-element overwritten:** incident + reflected fields need
  TWO traces (to iBS-1 and iBS).
- **AOI→out:** turn = 180−2·AOI; pin 45° to `[0;-1;0]` (cosd(90)=6.1e-17
  would perturb every emitted coordinate and break bit-identity).
- **Clearance metric:** score the test-arm EXCURSION vs the incoming
  source→BS beam; the output port (recomb/L2/detector) crosses near the
  BS BY DESIGN and must be excluded (a naive all-legs metric reports a
  false −60 mm collision at every AOI).
