# rodgers3 — the offset-field imager challenge

Mike Rodgers' third design challenge (2026-08-19): a 3-mirror imager,
F/4, EPD 75 mm, λ = 1 µm, 20°×20° FOV — with the whole field box pushed
**22° off axis**.  His 5-slide ladder
(`260802-WFOVimager_Offsetfield-jmr.pptx`, referenced here but not
committed — ask Dave for the deck) walks the recovery:

| rung | design state                                   | his "RMS WFE ≤" |
|------|------------------------------------------------|-----------------|
| r1   | coaxial, on-axis FOV, symmetric aspheres       | 159 nm  |
| r2   | same mirrors, offset FOV, FPA refit only       | 8810 nm |
| r3   | coaxial, aspheres re-optimized at offset       | 168 nm  |
| r4   | + mirror tilts/decenters + radii               | 117 nm  |
| r5   | + 8th-order Zernike surfaces (SPS ZRN) + radii | 53 nm   |

Constraints: exit beam horizontal (r2+); clearances > 50 mm and
> 35 mm (r4+).  Optimization fields: 9-point grid, XAN ±10°.

## The metric — state it before quoting any number

**His "RMS WFE ≤ x" is the MAXIMUM of CODE V's dense RMS-WFE-vs-field
map over the full 20°×20° box** (extracted from the EMF metadata inside
his own slides), *not* the max over the 9 optimization field points —
r2 proves the distinction (9-pt max ≈ 4022 nm vs the 8810 nm map
ceiling).  Our reproduction quotes the same statistic: strict RMS WFE
(design/src kernel), reference sphere centred on the **spot centroid**
on the deck-verbatim .seq FPA, anchored at the exit pupil, piston-only
removal, fields tangent-composed, every field's chief aimed through the
stop centre by real-ray iteration.  Chief-referenced values are stored
alongside in `r3_s0.mat`.

## Stage-0 result (frozen record: `r3_s0_report.txt`)

All five rungs gate — dense-map max / his number = 0.992 / 1.097 /
0.990 / 1.062 / **1.031** (r5 is the ZRN-convention gate; a wrong
Zernike ordering or C-offset scatters it by an order of magnitude).
Band rule: PASS = within [0.8, 1.25]×; nothing is tuned toward his
numbers — the band is the measurement.

## Files

- `TML_*.seq` — the five CODE V decks, verbatim (the truth source).
- `parse_seq.py` → `rodgers3_seq.m` — MACHINE-generated .seq truth.
  **Never hand-edit `rodgers3_seq.m`**; rerun the parser.
- `build_r3.m` — emits `r3_r1.in .. r3_r5.in` from the truth.  Every
  convention decision (asphere sign/units, ZRN frame + C-offset,
  YDE-verbatim/ADE-flipped, tangent-composed fields) is documented in
  its header and screenable via name-value hooks.
- `rodgers3.m` — the gate runner (layout gate → traced convention
  gates → strict WFE on his 9 fields → dense field map → verdict).
  Writes `r3_s0.mat`.
- `write_report.m` — formats `r3_report.txt` from the saved run
  (`r3_s0_report.txt` is the frozen Stage-0 record; re-runs never
  overwrite it).
- `probe_native_stop.m` — the native `macos.stop()` aiming A/B vs the
  runner's Newton aiming (see below).
- Suite coverage: `tests/tRodgers3.m` (freeform group, size 256).

Stage-0 probe scripts and refuted-hypothesis FAIL logs remain in the
sandbox record (`~/dev/MACOS_sandbox/Design/Rodgers3/s0/`, pinned
worktree `~/dev/MACOS_res_r3`).

## Conventions established here (inherited by the template)

1. CODE V ASP A,B,C → `AsphCoef = −ASP·(1e9,1e15,1e21)` (sag along
   +psiElt, h^(2i+2), mm→m).
2. ZRN: `ZernType= BornWolf`; macos mode = SCO C-index − 1 (C1 =
   NRADIUS slot); coefficients VERBATIM ×1e-3 in the element's LOCAL
   frame (Zernike sag rides +zMon, not psi); C2 = piston is real
   content (their frozen-thickness surrogate); lMon = NRADIUS×1e-3.
3. Tilt/dec: ALL YDE verbatim, ALL ADE sense-flipped
   (α_macos = −ADE_CODE V) — measured by bounded screen, refutes the
   odd/even reflection-parity theory.
4. Fields: XAN/YAN tangent-composed, `d = [tan XAN, tan YAN, 1]/‖·‖`.
5. Stop aiming: the header `ApStop= x y z` (StopPos) form aims
   geometrically with NO optics traversal (srcaim.inc) — wrong for a
   stop behind powered optics.  The element-bound `STOP <elt>` path
   (`ChiefRayAiming`, veneer `macos.stop(k)`) is the native real-ray
   aiming — A/B'd against the Newton aiming here.

The reusable, parameterized flow this challenge validates lives in
`../../templates/10_telescopes/offset_imager/`; the head-to-head
comparison with Mike's ladder is `PACKET.md`.
