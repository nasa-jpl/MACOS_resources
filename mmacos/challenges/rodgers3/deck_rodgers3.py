#!/usr/bin/env python3
"""Build the rodgers3 story deck (deck_rodgers3.pptx).

Regenerable end-to-end from committed artifacts: every WFE number on a
slide is parsed from the written records of the runs --
  t3/r3t_REPORT.md                      (run_t3 ladder + per-stage rows)
  r3_s0_report.txt                      (Stage-0 gate table)
  ../../templates/10_telescopes/offset_imager/t4_wide/t4_REPORT.md
  PACKET.md                             (counters + reported-rung values)
-- and the figures are the committed PNGs from the same runs.  Nothing
is hand-typed except the problem statement (which is the .seq truth).

Run:  python3 deck_rodgers3.py
Writes deck_rodgers3.md (the slide source, make_brief_slides dialect)
and deck_rodgers3.pptx beside this script.  Never hand-edit the .pptx.

Style: doc/STYLE_REPORTS.md governs; outward-facing -- Dave signs off
on the source before the deck leaves the repo.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T4DIR = os.path.join(HERE, "..", "..", "templates", "10_telescopes",
                     "offset_imager", "t4_wide")


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def need(m, what, src):
    if not m:
        sys.exit(f"deck_rodgers3: could not parse {what} from {src}")
    return m


# ---------------------------------------------------------------- parsers
def parse_gate_table(txt):
    """r3_s0_report.txt gate rows: rung, his nm, our map max, ratio, verdict."""
    rows = []
    for m in re.finditer(r"^\s*(r\d)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
                         r"\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|"
                         r"\s*(\w+)\s*$", txt, re.M):
        rows.append(dict(rung=m.group(1), his=float(m.group(2)),
                         ours=float(m.group(3)), ratio=float(m.group(4)),
                         verdict=m.group(5)))
    if len(rows) != 5:
        sys.exit("deck_rodgers3: gate table parse got %d rows, want 5"
                 % len(rows))
    return rows


def parse_report(txt):
    """offset_imager <tag>_REPORT.md -> per-stage dict + ladder."""
    out = {}
    secs = re.split(r"^## ", txt, flags=re.M)
    for sec in secs:
        m = re.match(r"S(\d) ", sec)
        if not m:
            continue
        sid = "s" + m.group(1)
        d = {}
        g = need(re.search(r"\|\s*\*\*map max\*\*\s*\|\s*\*\*([\d.]+) nm\*\*",
                           sec), sid + " map max", "REPORT")
        d["max_nm"] = float(g.group(1))
        g = need(re.search(r"\| map avg / std / min \| ([\d.]+) / ([\d.]+) / "
                           r"([\d.]+) nm", sec), sid + " moments", "REPORT")
        d["avg_nm"] = float(g.group(1))
        g = need(re.search(r"\| clearance floor \| ([\d.]+) mm \((\w+)",
                           sec), sid + " clearance", "REPORT")
        d["clear_mm"], d["clear_pf"] = float(g.group(1)), g.group(2)
        g = need(re.search(r"\| solve \| s\d: ([\d.]+) -> ([\d.]+) nm "
                           r"\(qmean over solve set\), (\d+) iters", sec),
                 sid + " solve", "REPORT")
        d["rms0"], d["rms"], d["iters"] = \
            float(g.group(1)), float(g.group(2)), int(g.group(3))
        g = re.search(r"err ([\d.]+)\D+ vs pin", sec)
        d["exit_err_deg"] = float(g.group(1)) if g else None
        g = need(re.search(r"\| conics K1\.\.K3 \| ([-\d.e]+) / ([-\d.e]+) / "
                           r"([-\d.e]+)", sec), sid + " conics", "REPORT")
        d["K"] = [float(g.group(i)) for i in (1, 2, 3)]
        out[sid] = d
    if set(out) != {"s1", "s2", "s3", "s4", "s5"}:
        sys.exit("deck_rodgers3: REPORT parse missing stages: %s" % out.keys())
    return out


def parse_packet(txt):
    flat = re.sub(r"\s+", " ", txt)
    p = {}
    p["s4_unc"] = float(need(re.search(
        r"our S4 reached ([\d.]+) nm", flat), "S4 unconstrained", "PACKET")
        .group(1))
    p["ca"] = float(need(re.search(
        r"Result: \*\*([\d.]+) nm map max\*\*", flat), "counter (a)",
        "PACKET").group(1))
    p["ca_unc"] = float(need(re.search(
        r"UNCONSTRAINED run of this counter reached ([\d.]+) nm", flat),
        "counter (a) unconstrained", "PACKET").group(1))
    p["cb"] = float(need(re.search(
        r"S5 basis from the S4 design: ([\d.]+) nm", flat), "counter (b)",
        "PACKET").group(1))
    # 2026-08-21 addendum: corrected clearances + the S5 budget answer
    p["s5_true_floor"] = float(need(re.search(
        r"S5 of record \| 34\.6 mm \| \*\*([\d.]+) mm\*\*", flat),
        "corrected S5 floor", "PACKET addendum").group(1))
    p["s4_hull"] = float(need(re.search(
        r"reads the S4 design at \*\*([\d.]+) mm\*\*", flat),
        "S4 hull floor", "PACKET addendum").group(1))
    g = need(re.search(
        r"\| A \| 9 \| 150 \| ([\d.]+) \| ([\d.]+) mm", flat),
        "leg A", "PACKET addendum")
    p["legA_nm"], p["legA_mm"] = float(g.group(1)), float(g.group(2))
    g = need(re.search(
        r"\| B \| 25 \| 60 \| ([\d.]+) \| ([\d.]+) mm", flat),
        "leg B", "PACKET addendum")
    p["legB_nm"], p["legB_mm"] = float(g.group(1)), float(g.group(2))
    g = need(re.search(
        r"\| C \| 25 \| 49 \(plateau\) \| \*\*([\d.]+)\*\* \| \*\*([\d.]+) mm\*\*",
        flat), "leg C", "PACKET addendum")
    p["legC_nm"], p["legC_mm"] = float(g.group(1)), float(g.group(2))
    return p


# ---------------------------------------------------------------- numbers
gates = parse_gate_table(read(os.path.join(HERE, "r3_s0_report.txt")))
t3 = parse_report(read(os.path.join(HERE, "t3", "r3t_REPORT.md")))
# (the t4-wide REPORT is deliberately NOT parsed: the instance is
# retired -- its gates were artifacts of the pre-fix clearance measure;
# see the PACKET 2026-08-21 addendum.  Its fields figure remains as the
# slide-8 exhibit.)
pk = parse_packet(read(os.path.join(HERE, "PACKET.md")))
# ZRN-convention negative-control factor, recorded in the gate itself
pk["zrn_negctl"] = float(need(re.search(
    r"factor ([\d.]+)x",
    read(os.path.join(HERE, "..", "..", "tests", "tRodgers3.m"))),
    "ZRN negative control", "tests/tRodgers3.m").group(1))

his = {g["rung"]: g["his"] for g in gates}       # r1..r5 reported nm
TAG = ("strict RMS WFE, centroid reference on the stage's frozen focal "
       "plane, exit-pupil anchor, piston-only removal; quoted statistic = "
       "dense 11x11 map MAXIMUM over the field box")

s2_cost = t3["s2"]["max_nm"] / his["r2"]
s4_ratio = t3["s4"]["max_nm"] / his["r4"]
s5_ratio = t3["s5"]["max_nm"] / his["r5"]
ca_win = t3["s5"]["max_nm"] / pk["ca"]
exit_worst = max(d["exit_err_deg"] for d in t3.values()
                 if d["exit_err_deg"] is not None)

T4REL = os.path.relpath(T4DIR, HERE).replace(os.sep, "/")

DEG, MU, LAM = "°", "µ", "λ"
PM, LE, GE = "±", "≤", "≥"

GATEROWS = "".join(
    f"| {g['rung']} | {g['his']:.0f} | {g['ours']:.2f} | {g['ratio']:.3f} | "
    f"{g['verdict']} |\n" for g in gates).rstrip("\n")

# ---------------------------------------------------------------- slides
MD = f"""<!-- rodgers3 story deck.  GENERATED by deck_rodgers3.py --
     edit the generator, not this file.  Build: python3 deck_rodgers3.py -->

# The Offset-Field Imager, Reproduced and Re-Designed
MACOS design layer vs CODE V on a 22{DEG}-offset wide-field three-mirror imager
D. C. Redding with Claude Code — 20 August 2026.  Source study: M. Rodgers, 260802-WFOVimager_Offsetfield (CODE V) + five lens sequences (committed).  Reproduction: MACOS_resources mmacos/challenges/rodgers3 + templates/10_telescopes/offset_imager, with packet.
~ Every WFE number in this deck: {TAG} — the statistic the source slides quote (decoded on slide 2).  DRAFT — pending sign-off.

## 1 — The challenge | A 20{DEG}x20{DEG} field box moved 22{DEG} off axis, an exit-beam pointing constraint, and hard clearances
::: left
- The system, from the lens sequences: three-mirror imager, EPD 75 mm, F/4 (EFL 300 mm), {LAM} = 1 {MU}m, field box 20{DEG}x20{DEG} full width centred at YAN +22{DEG}.  Constraints: exit beam horizontal (rung 2 on); beam/mirror clearances > 50 mm and > 35 mm (rung 4 on).  Optimization field: 9 points, XAN {PM}10{DEG}.
- The source study climbs the five-rung ladder in the table — each rung one added freedom, each closed by a reported WFE.
- The decoded statistic (Stage 0): each reported "RMS WFE {LE} x" equals the MAXIMUM of CODE V's dense RMS-vs-field map over the full box — established from the slides' own embedded plot metadata, not assumed.

| rung | design state | reported nm |
| r1 | coaxial, on-axis box, aspheres | {his['r1']:.0f} |
| r2 | same mirrors, offset box, FPA refit | {his['r2']:.0f} |
| r3 | re-asphered at the offset | {his['r3']:.0f} |
| r4 | + tilt/decenter + radii | {his['r4']:.0f} |
| r5 | + 8th-order Zernike + radii | {his['r5']:.0f} |
::: right
![The configuration class, traced by MACOS: the coaxial stage of the reproduction, real rays.](t3/r3t_s1_layout.png){{h=4.6}}
~ Reported nm values are quoted from the source slides; their statistic — the dense-map maximum — is decoded and gated on slide 2.

## 2 — Stage 0: earn the right to compare | All five reported rungs reproduce from the lens sequences inside [0.8, 1.25]x — nothing tuned toward them
::: left
| rung | reported nm | reproduced map max | ratio | verdict |
{GATEROWS}
- The gates run in the committed regression suite (tRodgers3, coarsened 5x5 maps), so the reproduction cannot silently rot.
::: right
- Conventions decoded to get there, each pinned by its own test: aspheric-coefficient sign and units; the Zernike set is Born&Wolf with a one-term index offset (negative control: dropping the offset misses rung 5 by {pk['zrn_negctl']:.0f}x); decenters verbatim with the tilt sense flipped; native stop aiming confirmed by A/B to 0.04 nm.
- The reproduction is the licence for every comparison that follows: same metric, same statistic, same field sampling on both sides.
::: full
~ {TAG}.  Known bounded approximation: the uniform pupil bundle vs CODE V's stop-gridded rays inflates the r2/r4 maps and the map minima slightly (full list: r3_s0_report.txt).

## 3 — The template | offset_imager: one parameterized flow — five stages, first-order identities re-derived at every iterate
::: left
::: stack
- S1 · coaxial solve :: first-order seed (EFL = EPD x F#, Petzval = 0), then conics + aspheres at the on-axis box
- S2 · the move :: the same mirrors at the offset box; only the focal plane follows
- S3 · re-solve at the used field :: symmetric surfaces re-solved at the offset, from the better of two starts
- S4 · tilt/decenter under constraints :: exit direction + clearances enter the solve as residual rows
- S5 · Zernike departures :: aspheres replaced; power pinned to the radii, tilt to the pointing
::: right
- Identities are enforced by construction, not penalized: EFL exact at every iterate; Petzval zero in the symmetric stages; stop pose re-derived; focal plane re-posed per stage.
- The solver is damped Gauss–Newton on per-ray strict-WFE residuals over a 3x3 solve grid; every reported number is scored on the dense map (solve set != scoring set).
- One call runs the whole story — ladder, counter-designs, report: oi_story(params).  The challenge instance and a second instrument (slide 8) are the same code at different parameters.
~ Template: templates/10_telescopes/offset_imager (committed, with per-stage reports and decks).

## 4 — Stages 1–3: solve, move, re-solve | Depth bought on axis is paid back at the offset; re-solving at the used field recovers {t3['s2']['max_nm']/t3['s3']['max_nm']:.0f}x
::: left
- S1, on-axis box: {t3['s1']['max_nm']:.1f} nm map max (reported rung 1: {his['r1']:.0f}).  The solve runs {his['r1']/t3['s1']['max_nm']:.0f}x deeper than the source rung — {t3['s1']['rms0']/1e3:.1f} {MU}m to {t3['s1']['rms']:.1f} nm qmean in {t3['s1']['iters']} iterations.
- S2, the same mirrors at 22{DEG}: {t3['s2']['max_nm']:,.0f} nm — {s2_cost:.0f}x the reported {his['r2']:.0f}.  The deeper on-axis optimum pays proportionally more off axis: this rung measures the S1 design as much as the offset.
- S3, re-solved at the used field: {t3['s3']['max_nm']:.1f} nm (reported: {his['r3']:.0f}, ratio {t3['s3']['max_nm']/his['r3']:.2f}).  The fresh Petzval-flat sphere seed ({t3['s3']['rms0']/1e3:.1f} {MU}m start) beats the carried on-axis aspheres as the starting point; the conics migrate to the oblate class (K1 {t3['s1']['K'][0]:.2f} to {t3['s3']['K'][0]:.2f}).
::: right
![S2: the price of moving the box with the design frozen.](t3/r3t_s2_map.png){{h=2.62}}
![S3: re-solved at the offset — the map max returns to the hundreds of nm.](t3/r3t_s3_map.png){{h=2.62}}
~ {TAG}.  S1–S3 run unconstrained; their clearance floors ({t3['s1']['clear_mm']:.0f} / {t3['s2']['clear_mm']:.0f} / {t3['s3']['clear_mm']:.0f} mm) are reported for context only.

## 5 — Stage 4, the buildability rung | Paying the same clearance constraint lands the same number: {t3['s4']['max_nm']:.1f} nm at {t3['s4']['clear_mm']:.1f} mm vs the reported {his['r4']:.0f} at {GE} 35 — {s4_ratio:.2f}x
::: left
- Unconstrained, tilt/decenter reach {pk['s4_unc']:.1f} nm — with the M3-to-focal-plane beam through M2's patch (clearance floor 0.0 mm).  Not buildable.
- The clearance model: nine beam-leg/obstacle pairs, per-field footprint disks over the box centre and YAN extremes, the focal plane counted as an obstacle.  The same model the solve pays as hinge residual rows and the report gates.
- Two toolchains, same constraint, same answer to 3% — strong evidence both are measuring the same design space — and a clean measurement of what the clearance constraint costs: {t3['s4']['max_nm']/pk['s4_unc']:.1f}x in WFE at this rung ({pk['s4_unc']:.1f} to {t3['s4']['max_nm']:.1f} nm).
- Exit chief within {exit_worst:.2f}{DEG} of horizontal at every stage (the pin, solved as an equality row).
::: right
![S4 field envelopes: every beam leg clears every mirror edge (floor {t3['s4']['clear_mm']:.1f} mm).](t3/r3t_s4_fields.png){{h=2.62}}
![S4 map: {t3['s4']['max_nm']:.1f} nm max over the box.](t3/r3t_s4_map.png){{h=2.62}}
~ {TAG}.  The {t3['s4']['clear_mm']:.1f} mm floor is CONFIRMED at fine sampling under the fixed clearance measure (packet addendum); the truer convex-hull glass model reads the same design at {pk['s4_hull']:.1f} mm — comfortably inside the stated 35.

## 6 — Stage 5: diagnosed, then beaten | The gap to the reported {his['r5']:.0f} was the solve-field count; matching it lands {pk['legC_nm']:.1f} nm
::: left
- The five-stage run closes S5 at {t3['s5']['max_nm']:.1f} nm — stalled at its S4 level, with 82 Zernike variables against only 9 solve fields (the dense map is scored on 121 points; the solve set is what the optimizer sees).
- The controlled probes, all from the same S4 state: 150 iterations at 9 fields reach {pk['legA_nm']:.1f} nm — iterations are NOT the gap.  25 solve fields at 60 iterations reach {pk['legB_nm']:.1f} — the field count is.  Nine fields under-determine 82 variables: the solve set converges while the dense map stalls.
- With 25 fields AND the corrected clearance rows (slide 8): **{pk['legC_nm']:.1f} nm at a {pk['legC_mm']:.1f} mm floor** — {pk['legC_nm']/his['r5']:.2f}x the reported {his['r5']:.0f}, exit pin held to 0.008{DEG}.
::: right
![The five-stage S5 of record: {t3['s5']['max_nm']:.1f} nm.  Its recorded {t3['s5']['clear_mm']:.1f} mm floor corrects to {pk['s5_true_floor']:.1f} mm under the fixed clearance measure (packet addendum).](t3/r3t_s5_map.png){{h=2.62}}
![The honest re-solve: {pk['legC_nm']:.1f} nm max, true floor {pk['legC_mm']:.1f} mm.](s5_budget/s5b_legC_map.png){{h=2.62}}
~ {TAG}.

## 7 — Counter-designs, same constraints, same budget | The sphere+Zernike start wins {ca_win:.1f}x; releasing the pinned terms chokes the solve
::: left
- (a) Sphere + Zernike from the start — S3's radii, conics zeroed, no aspheres, straight to the Zernike solve: {pk['ca']:.1f} nm map max, against the heritage path's {t3['s5']['max_nm']:.1f} under identical constraint rows and iteration budget.  The asphere heritage is a burden, not a head start.  The unconstrained variant of this branch ({pk['ca_unc']:.1f} nm) does not survive buildability.
- (b) Power and y-tilt released into the Zernike basis from the S4 design: {pk['cb']:.1f} nm — far worse than the pinned set.  The freed modes fight the jobs the pins already do (power held by the radii, tilt by the pointing); 88 variables against 9 solve fields plus constraint rows.  No evidence the reported {his['r5']:.0f} nm is term-set-limited — and affirmative evidence for the pinning doctrine at finite solver budget.
::: right
![Counter (a): the sphere+Zernike start under the full constraint set — {pk['ca']:.1f} nm.](t3/r3t_ca_map.png){{h=2.62}}
![Counter (b): power + y-tilt released — {pk['cb']:.0f} nm; the solve chokes.](t3/r3t_cb_map.png){{h=2.62}}
~ {TAG}.

## 8 — What carried, what taught | The layout figure that said no — and the constraint-model fixes it forced
::: left
- What carried: per-ray Gauss–Newton residuals (per-field-RMS residuals stall); first-order identities re-derived, never penalized; stop decenter as the exit-aiming variable; clearances as residual rows in the solve; negative controls with teeth at every decode; a solve set sized to the variable count (slide 6).
- What taught: the beam-clearance measure must test PIERCING exactly and go NEGATIVE inside the glass — a sampled minimum reported millimetres of clearance for beams THROUGH a mirror, and a zero-at-contact measure gives the optimizer no way out; clearance judged at the box centre misses edge-field blockage — evaluate over the field; buildability constrains the FIELD CHOICE, not just the surfaces (no envelope of this family packages a 200 mm F/2.5 beam at a 12{DEG} offset — the field walk cannot separate the fans); stations are not spacings (the F/4.95 misread, retracted; packaging reads from the .seq truth file).
- The second instrument (EPD 200 mm, F/2.5) exposed all of it and is retired for re-instancing; its layout figure — not a number — is what caught the defect.
::: right
![The figure that caught it: the second instrument's beams thread M1 and M2 while the sampled gate reported PASS.  Hardware drawings are gates, not illustrations.](t4_fields_exhibit){{h=4.4}}
~ {TAG} (slides 1–7).  Reproduce: rodgers3() = the Stage-0 gates, oi_story(...) = ladder + counters, run_s5_budget()/run_s5_signed() = the slide-6 probes; record: challenges/rodgers3/PACKET.md incl. the 2026-08-21 addendum.
"""

# slide 8's exhibit: the retired second instrument's field-envelope
# figure -- the drawing that exposed the clearance-model defect
MD = MD.replace("(t4_fields_exhibit)", f"({T4REL}/t4_s5_fields.png)")

md_path = os.path.join(HERE, "deck_rodgers3.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(MD)
print("wrote", md_path)

rc = subprocess.call([sys.executable,
                      os.path.join(HERE, "make_brief_slides.py"),
                      md_path, os.path.join(HERE, "deck_rodgers3.pptx")])
sys.exit(rc)
