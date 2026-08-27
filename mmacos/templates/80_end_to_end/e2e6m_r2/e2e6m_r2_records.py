"""e2e6m_r2_records -- committed-record parsers behind deck_e2e6m_r2.py.

Every round-2 number comes through here, out of the stage reports the
runs wrote (r0_*, r1_*, r3_*, r4_*).  Round-1 numbers (the telescope,
the segmentation, the imager) come through round 1's own
e2e6m_records, imported from the frozen ../e2e6m directory.  Loud
sys.exit on any parse miss -- a deck must not build from guesses.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R1DIR = os.path.normpath(os.path.join(HERE, "..", "e2e6m"))
sys.path.insert(0, R1DIR)
import e2e6m_records as R1   # noqa: E402  (round 1, frozen, read-only)


def read(name):
    p = os.path.join(HERE, name)
    if not os.path.isfile(p):
        sys.exit("e2e6m_r2_records: missing %s -- run that stage first" % name)
    with open(p, encoding="utf-8") as f:
        return f.read()


def grab(txt, pat, what, src, cast=float, group=1):
    m = re.search(pat, txt, re.M)
    if not m:
        sys.exit("e2e6m_r2_records: could not parse %s from %s" % (what, src))
    try:
        return cast(m.group(group))
    except ValueError:
        sys.exit("e2e6m_r2_records: %s in %s is not a %s: %r"
                 % (what, src, cast.__name__, m.group(group)))


DOFS = ["Rx", "Ry", "Rz", "Tx", "Ty", "Tz"]


# ----------------------------------------------------------------- R0
def r0():
    t = read("r0_closure_report.txt")
    p = read("r0_probe_report.txt")
    r = {}
    r["wf_elt"] = grab(t, r"wf_elt (\d+)", "wf_elt", "r0_closure_report", int)
    r["worst"] = grab(t, r"worst over \d+ \(elt, DOF\) pairs: ([\d.e+-]+)",
                      "closure worst", "r0_closure_report")
    r["npairs"] = grab(t, r"worst over (\d+) \(elt, DOF\) pairs",
                       "closure pairs", "r0_closure_report", int)
    r["tol"] = grab(t, r"tol ([\d.]+)", "closure tol", "r0_closure_report")
    # Seg19 rows: rel at the WRONG surface (probe [C], engine at nElt)
    # and at the RIGHT one (closure, engine at wf_elt) -- same element,
    # same pokes, only the trace target differs.
    wrong, right = {}, {}
    for d in DOFS:
        m = re.search(r"^\s+%s : \|model\| [\d.e+-]+\s+rel\.err vs engine ([\d.e+-]+)"
                      % d, p, re.M)
        if not m:
            sys.exit("e2e6m_r2_records: no probe [C] row for " + d)
        wrong[d] = float(m.group(1))
        m = re.search(r"^\s+elt 19 %s: .* rel\.err ([\dNa.e+-]+)" % d, t, re.M)
        if not m:
            sys.exit("e2e6m_r2_records: no closure row for elt 19 " + d)
        right[d] = m.group(1)
    r["wrong"], r["right"] = wrong, right
    b = read("r0_bisect2_report.txt")
    tilt = grab(b, r"tilt-about-center ~ ([\d.]+) m/rad", "tilt lever",
                "r0_bisect2_report")
    pist = grab(b, r"vertex-piston ~ ([\d.]+) m/rad", "piston lever",
                "r0_bisect2_report")
    r["lever_ratio"] = pist / tilt
    return r


# ----------------------------------------------------------------- R1
def r1():
    tc = read("r1_coro_report.txt")
    ts = read("r1_seg_report.txt")
    td = read("r1_dm_report.txt")
    tu = read("r1_union_report.txt")
    r = {}
    for tag in ("seg", "mono"):
        blk = tc[tc.index("---- %s " % tag):]
        r[tag] = {
            "mean":   grab(blk, r"dark zone [\d.-]+ lambda/D: mean ([\d.e+-]+)",
                           tag + " mean", "r1_coro_report"),
            "median": grab(blk, r"median ([\d.e+-]+)", tag + " median",
                           "r1_coro_report"),
            "suppr":  grab(blk, r"on-axis suppression ([\d.e+-]+)",
                           tag + " suppression", "r1_coro_report"),
            "chief":  grab(blk, r"chief vs geometric ([\d.e+-]+)",
                           tag + " chief", "r1_coro_report"),
        }
    r["ratio_mean"] = grab(tc, r"ratio ([\d.]+)x\n", "gap ratio mean",
                           "r1_coro_report")
    r["ratio_median"] = grab(tc, r"median segmented .* ratio ([\d.]+)x",
                             "gap ratio median", "r1_coro_report")
    r["topo"] = grab(tc, r"topology cost ([\d.]+)x", "topology factor",
                     "r1_coro_report")
    r["r_occ"] = grab(tc, r"occulter ([\d.]+) lambda/D", "occulter",
                      "r1_coro_report")
    r["inner"] = grab(tc, r"annulus ([\d.]+)-[\d.]+ lambda/D", "inner",
                      "r1_coro_report")
    r["outer"] = grab(tc, r"annulus [\d.]+-([\d.]+) lambda/D", "outer",
                      "r1_coro_report")
    r["nelt"] = grab(ts, r"loads at (\d+) elements", "element count",
                     "r1_seg_report", int)
    r["rays"] = grab(ts, r"(\d+)/\d+ rays pass", "rays", "r1_seg_report", int)
    r["rays_tot"] = grab(ts, r"\d+/(\d+) rays pass", "ray total",
                         "r1_seg_report", int)
    r["pupil_mm"] = grab(ts, r"collimated pupil ([\d.]+) m", "pupil",
                         "r1_seg_report") * 1e3
    r["dm_mm"] = grab(ts, r"on (\d+) mm DMs", "DM aperture", "r1_seg_report")
    r["shroud"] = grab(tu, r"union: ([\d.]+) m", "union shroud",
                       "r1_union_report")
    r["shroud_gate"] = grab(tu, r"against the ([\d.]+) m gate", "shroud gate",
                            "r1_union_report")
    r["poke_peak"] = grab(td, r"peak \|dOPD\| ([\d.e+-]+) m", "poke peak",
                          "r1_dm_report")
    r["poke_exp"] = grab(td, r"2x surface = ([\d.e+-]+)", "poke expected",
                         "r1_dm_report")
    r["poke_off"] = grab(td, r"offset from pupil center ([\d.]+) Rpup",
                         "poke offset", "r1_dm_report")
    r["poke_efrac"] = grab(td, r"energy within 0.15 Rpup of the peak: (\d+)%",
                           "poke energy", "r1_dm_report")
    r["nact"] = grab(td, r"lattice: (\d+) x \d+", "lattice", "r1_dm_report", int)
    r["nact_active"] = grab(td, r", (\d+) active", "active acts",
                            "r1_dm_report", int)
    r["ng"] = grab(td, r"ng (\d+)", "grid n", "r1_dm_report", int)
    return r


# ----------------------------------------------------------------- R3
def r3():
    t = read("r3_report.txt")
    ts = read("r3_sens_report.txt")
    tj = read("r3_dmjac_report.txt")
    tm = read("r3_met_report.txt")
    r = {}
    r["dwdx_rows"] = grab(t, r"dwdxall (\d+) x \d+", "dwdx rows", "r3_report", int)
    r["dwdx_cols"] = grab(t, r"dwdxall \d+ x (\d+)", "dwdx cols", "r3_report", int)
    r["dwdz_cols"] = grab(t, r"dwdzall \d+ x (\d+)", "dwdz cols", "r3_report", int)
    r["dwdg_cols"] = grab(t, r"dwdgall \d+ x (\d+)", "dwdg cols", "r3_report", int)
    r["n_optics"] = grab(t, r"optics (\d+) perturbed", "optics", "r3_report", int)
    r["closure_worst"] = grab(t, r"worst rel ([\d.e+-]+) over \d+ pairs",
                              "closure worst", "r3_report")
    r["closure_pairs"] = grab(t, r"worst rel [\d.e+-]+ over (\d+) pairs",
                              "closure pairs", "r3_report", int)
    r["dwdx_cond"] = grab(ts, r"segment-only: \d+x\d+  cond\+ ([\d.e+-]+)",
                          "segment cond", "r3_sens_report")
    g = re.search(r"Grp\[PM\] \(group\)\s+" + r"([\d.e+-]+)\s+" * 6, ts)
    q = re.search(r"group / Elt 1\s+" + r"([\d.e+-]+)\s+" * 6, ts)
    if not (g and q):
        sys.exit("e2e6m_r2_records: group exhibit rows not found")
    r["group"] = [float(g.group(i + 1)) for i in range(6)]
    r["ratio"] = [float(q.group(i + 1)) for i in range(6)]
    r["jac_cols"] = grab(tj, r"-> (\d+) columns", "G columns", "r3_dmjac_report", int)
    r["jac_spoke"] = grab(tj, r"([\d.]+) s per poke\)", "s/poke", "r3_dmjac_report")
    r["dz_px"] = grab(tj, r"lambda/D: (\d+) pixels", "dz pixels",
                      "r3_dmjac_report", int)
    r["wem_em"] = grab(tm, r"edge\+MET\s+([\d.]+)", "edge+MET WEM",
                       "r3_met_report")
    r["wem_m"] = grab(tm, r"MET only\s+([\d.]+)", "MET-only WEM",
                      "r3_met_report")
    r["mc_nm"] = grab(tm, r"\(200 draws\): ([\d.]+) nm rms", "MC", "r3_met_report")
    r["mc_ana"] = grab(tm, r"\(analytic ([\d.]+)\)", "MC analytic",
                       "r3_met_report")
    r["n_edge"] = grab(tm, r"dxde 120x(\d+)", "edge count", "r3_met_report", int)
    r["n_gauge"] = grab(tm, r"dxdl 120x(\d+)", "gauge count", "r3_met_report", int)
    r["fd_off"] = grab(tm, r"\(analytic [\d.]+, ([\d.]+)% off\)",
                       "FD validation", "r3_met_report")
    return r


# ----------------------------------------------------------------- R4
def r4():
    t = read("r4_report.txt")
    r = {}
    r["frames"] = grab(t, r"history: (\d+) frames", "frames", "r4_report", int)
    r["dt"] = grab(t, r"frames, dt (\d+) s", "dt", "r4_report")
    r["every"] = grab(t, r"every (\d+) frames", "cadence", "r4_report", int)
    r["model"] = grab(t, r"model (\d+)", "model", "r4_report", int)
    u = t[t.index("pass 1 of 2"):]
    r["unc"] = {"wfe0": grab(u, r"WFE ([\d.]+) ->", "unc wfe0", "r4_report"),
                "wfe1": grab(u, r"WFE [\d.]+ -> ([\d.]+)", "unc wfe1", "r4_report"),
                "con0": grab(u, r"contrast ([\d.e+-]+) ->", "unc con0", "r4_report"),
                "con1": grab(u, r"contrast [\d.e+-]+ -> ([\d.e+-]+)",
                             "unc con1", "r4_report")}
    c = t[t.index("pass 2 of 2"):]
    r["cor"] = {"wfe0": grab(c, r"WFE ([\d.]+) ->", "cor wfe0", "r4_report"),
                "wfe1": grab(c, r"WFE [\d.]+ -> ([\d.]+)", "cor wfe1", "r4_report"),
                "con0": grab(c, r"contrast ([\d.e+-]+) ->", "cor con0", "r4_report"),
                "con1": grab(c, r"contrast [\d.e+-]+ -> ([\d.e+-]+)",
                             "cor con1", "r4_report")}
    r["resid_nm"] = grab(c, r"\|x\+u\| rms ([\d.e+-]+)", "residual",
                         "r4_report") * 1e9
    r["drift_nm"] = grab(c, r"drift \|x\| rms ([\d.e+-]+)", "drift",
                         "r4_report") * 1e9
    dig = re.search(r"EFC first dig \(frame 1\): (.+)$", c, re.M)
    if not dig:
        sys.exit("e2e6m_r2_records: no EFC dig line")
    r["dig"] = [float(x) for x in dig.group(1).split(" -> ")]
    r["gain"] = grab(c, r"gain ([\d.]+)", "loop gain", "r4_report")
    return r


# ----------------------------------------------------------------- R2
def r2m():
    t = read("r2_masks_report.txt")
    r = {}
    r["r_occ_um"] = grab(t, r"= ([\d.]+) um at the FPM focus", "occulter um",
                         "r2_masks_report")
    r["lamD_um"] = grab(t, r"focal scale ([\d.]+) um per lambda/D",
                        "focal scale", "r2_masks_report")
    r["thru"] = grab(t, r"throughput over the traced aperture ([\d.]+)",
                     "apod throughput", "r2_masks_report")
    return r


# ------------------------------------------------- round 1, re-exported
def s1():
    return R1.s1()


def s2():
    return R1.s2()


def s3c():
    return R1.s3c()


# ------------------------------------------------- CF (coronagraph families)
CF_KEYS = ["hard", "apl", "aplc", "blc", "v4", "v6"]
CF_NAMES = ["classical Lyot", "apodized Lyot (R1)", "APLC (ap.-matched)",
            "band-limited 4th", "vortex chg 4", "vortex chg 6"]
_NUM = r"([0-9.eE+-]+)"


def cf1():
    """S1 stopped statics: per-family mean/median/suppression/thru/no-stop/ratio."""
    t = read("cf1_report.txt")
    r = {"fam": {}}
    for key, name in zip(CF_KEYS, CF_NAMES):
        pat = (re.escape(name) + r"\s*\|\s*" + _NUM + r"\s*\|\s*" + _NUM +
               r"\s*\|\s*" + _NUM + r"\s*\|\s*" + _NUM + r"%\s*\|\s*" +
               _NUM + r"\s*\|\s*" + _NUM + r"x\s*\|\s*(.+)$")
        m = re.search(pat, t, re.M)
        if not m:
            sys.exit("e2e6m_r2_records: cf1 row for %s not found" % name)
        r["fam"][key] = {
            "name": name, "mean": float(m.group(1)), "median": float(m.group(2)),
            "suppr": float(m.group(3)), "thru_pct": float(m.group(4)),
            "nostop_mean": float(m.group(5)), "stop_ratio": float(m.group(6)),
            "note": m.group(7).strip()}
    return r


def cf1c():
    """The stop-penalty attribution (fixed-mask Babinet decomposition)."""
    t = read("cf1c_report.txt")
    r = {}
    r["total"] = grab(t, r"c_st/c_ns = " + _NUM, "cf1c total factor", "cf1c_report.txt")
    r["edge"] = grab(t, r"fixed-mask stop edge " + _NUM, "cf1c edge factor", "cf1c_report.txt")
    r["rechain"] = grab(t, r"scale-rechain " + _NUM + r"\]", "cf1c rechain factor", "cf1c_report.txt")
    r["pk_ratio"] = grab(t, r"pk_ns/pk_st = " + _NUM, "cf1c pk ratio", "cf1c_report.txt")
    r["energy_up"] = grab(t, r"ENERGY at fixed masks up " + _NUM + r"x", "cf1c energy", "cf1c_report.txt")
    r["rim"] = grab(t, r"rim " + _NUM + r" \+ cross", "cf1c rim term", "cf1c_report.txt")
    r["cross"] = grab(t, r"cross " + _NUM + r" \(closure", "cf1c cross term", "cf1c_report.txt")
    r["I_ns"] = grab(t, r"= I_ns " + _NUM, "cf1c I_ns", "cf1c_report.txt")
    return r


def cf2():
    """S2 per-family floors: static/fixed/relin/lin-ach/strokes/attribution."""
    t = read("cf2_report.txt")
    r = {"fam": {}}
    for key, name in zip(CF_KEYS, CF_NAMES):
        pat = (re.escape(name) + r"\s*\|\s*" + _NUM + r"\s*\|\s*" + _NUM +
               r"\s*\|\s*" + _NUM + r"\s*\|\s*" + _NUM +
               r"\s*\|\s*" + _NUM + r"/\s*" + _NUM + r"\s*\|\s*(.+)$")
        m = re.search(pat, t, re.M)
        if not m:
            sys.exit("e2e6m_r2_records: cf2 row for %s not found" % name)
        r["fam"][key] = {
            "name": name, "static": float(m.group(1)), "fixed": float(m.group(2)),
            "relin": float(m.group(3)), "linach": float(m.group(4)),
            "s1_nm": float(m.group(5)), "s2_nm": float(m.group(6)),
            "attrib": m.group(7).strip()}
    r["zzT"] = grab(t, r"z/z_T at [0-9]+ lambda/D = " + _NUM, "cf2 z/zT", "cf2_report.txt")
    return r


def cf3a():
    """S3a Lyot sweep: per-leg (lyot, contrast, thru) arrays."""
    t = read("cf3a_report.txt")
    r = {"leg": {}}
    for blk in re.split(r"^-- ", t, flags=re.M)[1:]:
        key = blk.split()[0]
        pts = re.findall(r"L=" + _NUM + r": contrast " + _NUM +
                         r" \| thru " + _NUM, blk)
        if not pts:
            sys.exit("e2e6m_r2_records: cf3a leg %s has no points" % key)
        r["leg"][key] = {
            "lyot": [float(p[0]) for p in pts],
            "con":  [float(p[1]) for p in pts],
            "thru": [float(p[2]) for p in pts]}
    return r
