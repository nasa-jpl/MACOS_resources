"""rodgers3_records -- the committed-record parsers shared by the deck
generators (deck_rodgers3.py, deck_rodgers3_walk.py).  Every number on
any slide comes through here: r3t_REPORT.md, r3_s0_report.txt,
PACKET.md (+ the 2026-08-21 addendum), tests/tRodgers3.m.  Loud
sys.exit on any parse miss -- a deck must not build from guesses."""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


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
    p["s4_true_floor"] = float(need(re.search(
        r"S4 of record \| 34\.1 mm \| \*\*([\d.]+) mm", flat),
        "confirmed S4 floor", "PACKET addendum").group(1))
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


def load():
    """Parse everything once: (gates, t3, pk) with the negctl folded in."""
    gates = parse_gate_table(read(os.path.join(HERE, "r3_s0_report.txt")))
    t3 = parse_report(read(os.path.join(HERE, "t3", "r3t_REPORT.md")))
    pk = parse_packet(read(os.path.join(HERE, "PACKET.md")))
    pk["zrn_negctl"] = float(need(re.search(
        r"factor ([\d.]+)x",
        read(os.path.join(HERE, "..", "..", "tests", "tRodgers3.m"))),
        "ZRN negative control", "tests/tRodgers3.m").group(1))
    return gates, t3, pk
