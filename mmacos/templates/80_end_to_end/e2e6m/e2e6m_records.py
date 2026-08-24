"""e2e6m_records -- the committed-record parsers behind deck_e2e6m.py.

Every number on any slide comes through here, out of the stage reports
that the runs themselves wrote: s1_report.txt, s2_report.txt,
s3_seg_report.txt, s3_coro_report.txt, s4_report.txt, s4_sens_report.txt,
s5_report.txt.  Loud sys.exit on any parse miss -- a deck must not build
from guesses, and a silently-defaulted number is worse than no slide.

Pattern lifted from challenges/rodgers3/rodgers3_records.py.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def read(name):
    p = os.path.join(HERE, name)
    if not os.path.isfile(p):
        sys.exit("e2e6m_records: missing %s -- run that stage first" % name)
    with open(p, encoding="utf-8") as f:
        return f.read()


def grab(txt, pat, what, src, cast=float, group=1):
    m = re.search(pat, txt, re.M)
    if not m:
        sys.exit("e2e6m_records: could not parse %s from %s" % (what, src))
    try:
        return cast(m.group(group))
    except ValueError:
        sys.exit("e2e6m_records: %s in %s is not a %s: %r"
                 % (what, src, cast.__name__, m.group(group)))


# ----------------------------------------------------------------- S1
def s1():
    t = read("s1_report.txt")
    d = read("s1_design_report.txt")
    r = {}
    r["D_m"] = grab(t, r"^D = ([\d.]+) m", "aperture", "s1_report.txt")
    r["lambda_nm"] = grab(t, r"lambda = (\d+) nm", "wavelength", "s1_report.txt")
    r["fov_arcmin"] = grab(t, r"map over \+-([\d.]+)'", "design half-field",
                           "s1_report.txt")
    r["wfe_tilt"] = grab(t, r"-tilt max ([\d.]+) waves", "dense-map -tilt max",
                         "s1_report.txt")
    r["dl_bar"] = grab(t, r"DL <= ([\d.]+) waves", "diffraction-limit bar",
                       "s1_report.txt")
    r["efl_m"] = grab(t, r"^\s+EFL ([\d.]+) m ->", "EFL", "s1_report.txt")
    r["fno"] = grab(t, r"EFL [\d.]+ m -> f/([\d.]+)", "f/#", "s1_report.txt")
    r["fno_lo"] = grab(t, r"f/# in \[(\d+) \d+\]", "f/# band low", "s1_report.txt")
    r["fno_hi"] = grab(t, r"f/# in \[\d+ (\d+)\]", "f/# band high", "s1_report.txt")
    r["shroud_m"] = grab(t, r"^\s+shroud ([\d.]+) m diameter", "shroud diameter",
                         "s1_report.txt")
    r["shroud_gate"] = grab(t, r"shroud <= ([\d.]+) m", "shroud gate",
                            "s1_report.txt")
    r["train_m"] = grab(t, r"train ([\d.]+) m", "train length", "s1_report.txt")
    r["clear_n"] = grab(t, r"clearance (\d+)/\d+ bodies clear", "clear count",
                        "s1_report.txt", int)
    r["clear_tot"] = grab(t, r"clearance \d+/(\d+) bodies clear", "body count",
                          "s1_report.txt", int)
    r["rays_pass"] = grab(t, r"RELOAD GATE .*?: \d+ elements, (\d+)/\d+ rays",
                          "reload rays", "s1_report.txt", int)
    r["rays_tot"] = grab(t, r"RELOAD GATE .*?: \d+ elements, \d+/(\d+) rays",
                         "reload ray total", "s1_report.txt", int)
    r["aoi_max"] = max(float(x) for x in re.findall(
        r"AOI\s+[\d.]+ deg \(spread ([\d.]+) deg", d) or sys.exit(
            "e2e6m_records: no AOI spreads in s1_design_report.txt"))
    return r


# ----------------------------------------------------------------- S2
def s2():
    t = read("s2_report.txt")
    r = {}
    r["nseg"] = grab(t, r"-> (\d+) segments", "segment count", "s2_report.txt", int)
    r["width_m"] = grab(t, r"width ([\d.]+) flat-to-flat", "segment width",
                        "s2_report.txt")
    r["gap_m"] = grab(t, r"gap ([\d.]+) m", "gap", "s2_report.txt")
    r["rays_bare"] = grab(t, r"bare segmented: \d+ src rays, (\d+) pass",
                          "bare pass count", "s2_report.txt", int)
    r["rays_ap"] = grab(t, r"physical apertures \(pad \d+\): (\d+) pass",
                        "aperture pass count", "s2_report.txt", int)
    r["ratio"] = grab(t, r"ratio out/in ([\d.eE+-]+)\s+\[", "poke ratio",
                      "s2_report.txt")
    r["in_rms"] = grab(t, r"inside  the poked segment: rms ([\d.eE+-]+) m",
                       "inside rms", "s2_report.txt")
    r["out_rms"] = grab(t, r"outside the poked segment: rms ([\d.eE+-]+) m",
                        "outside rms", "s2_report.txt")
    r["n_in"] = grab(t, r"inside  the poked segment: rms [\d.eE+-]+ m over (\d+)",
                     "inside ray count", "s2_report.txt", int)
    r["n_out"] = grab(t, r"outside the poked segment: rms [\d.eE+-]+ m over (\d+)",
                      "outside ray count", "s2_report.txt", int)
    return r


# ----------------------------------------------------------------- S3
def s3():
    g = read("s3_seg_report.txt")
    c = read("s3_coro_report.txt")
    r = {}
    r["nelt"] = grab(g, r"= (\d+) elements ->", "spliced element count",
                     "s3_seg_report.txt", int)
    r["shroud_m"] = grab(g, r"shroud on the full train: ([\d.]+) m",
                         "full-train shroud", "s3_seg_report.txt")
    r["shroud_gate"] = grab(g, r"against the ([\d.]+) m gate", "shroud gate",
                            "s3_seg_report.txt")
    r["rays_pass"] = grab(g, r"(\d+)/\d+ rays pass \(telescope alone",
                          "spliced ray count", "s3_seg_report.txt", int)

    def arm(tag):
        m = re.search(r"---- %s .*?----(.*?)(?=----|\Z)" % tag, c, re.S)
        if not m:
            sys.exit("e2e6m_records: no '%s' arm in s3_coro_report.txt" % tag)
        b = m.group(1)
        return dict(
            mean=grab(b, r"mean ([\d.eE+-]+),", "%s DZ mean" % tag, "s3_coro"),
            median=grab(b, r"median ([\d.eE+-]+),", "%s DZ median" % tag, "s3_coro"),
            suppr=grab(b, r"on-axis suppression ([\d.eE+-]+)", "%s suppression" % tag,
                       "s3_coro"),
            thru=grab(b, r"net throughput ([\d.]+)", "%s throughput" % tag, "s3_coro"),
            chief=grab(b, r"chief vs geometric ([\d.eE+-]+);", "%s chief" % tag,
                       "s3_coro"),
        )
    r["seg"] = arm("seg")
    r["mono"] = arm("mono")
    r["ratio_mean"] = grab(c, r"dark-zone mean .*?ratio ([\d.]+)x", "mean ratio",
                           "s3_coro_report.txt")
    r["ratio_median"] = grab(c, r"dark-zone median .*?ratio ([\d.]+)x",
                             "median ratio", "s3_coro_report.txt")
    r["inner"] = grab(c, r"annulus ([\d.]+)-[\d.]+ lambda/D", "DZ inner",
                      "s3_coro_report.txt")
    r["outer"] = grab(c, r"annulus [\d.]+-([\d.]+) lambda/D", "DZ outer",
                      "s3_coro_report.txt")
    r["r_occ"] = grab(c, r"occulter ([\d.]+) lambda/D", "occulter radius",
                      "s3_coro_report.txt")
    return r


# ----------------------------------------------------------------- S4
def s4():
    t = read("s4_report.txt")
    s = read("s4_sens_report.txt")
    r = {}
    r["n_optics"] = grab(t, r"optics (\d+) perturbed", "optic count", "s4_report.txt", int)
    r["n_group"] = grab(t, r"PM group = (\d+) members", "group size", "s4_report.txt", int)
    r["dwdx_rows"] = grab(t, r"dwdxall (\d+) x \d+", "dwdx rows", "s4_report.txt", int)
    r["dwdx_cols"] = grab(t, r"dwdxall \d+ x (\d+)", "dwdx cols", "s4_report.txt", int)
    r["dwdz_cols"] = grab(t, r"dwdzall \d+ x (\d+)", "dwdz cols", "s4_report.txt", int)
    r["dwdg_cols"] = grab(t, r"dwdgall \d+ x (\d+)", "dwdgrid cols", "s4_report.txt", int)
    r["dwdx_cond"] = grab(t, r"segment-only: \d+x\d+\s+cond\+ ([\d.eE+-]+)",
                          "dwdx condition number", "s4_report.txt")
    r["dwdx_seg_cols"] = grab(t, r"segment-only: \d+x(\d+)", "dwdx segment cols",
                              "s4_report.txt", int)
    m = re.search(r"Grp\[PM\] \(group\)\s+((?:[\d.eE+-]+\s+){5}[\d.eE+-]+)", s)
    if not m:
        sys.exit("e2e6m_records: no PM group row in s4_sens_report.txt")
    r["group"] = [float(x) for x in m.group(1).split()]
    m = re.search(r"group / Elt \d+\s+((?:[\d.eE+-]+\s+){5}[\d.eE+-]+)", s)
    if not m:
        sys.exit("e2e6m_records: no group/member ratio row in s4_sens_report.txt")
    r["ratio"] = [float(x) for x in m.group(1).split()]
    return r


# ----------------------------------------------------------------- S5
def s5():
    t = read("s5_report.txt")
    r = {}
    r["frames"] = grab(t, r"history: (\d+) frames", "frame count", "s5_report.txt", int)
    r["dt"] = grab(t, r"dt (\d+) s", "frame period", "s5_report.txt")
    r["every"] = grab(t, r"scored every (\d+) frames", "contrast subsample",
                      "s5_report.txt", int)
    r["n_scored"] = grab(t, r"scored every \d+ frames \((\d+) of \d+\)",
                         "scored frames", "s5_report.txt", int)
    unc = re.search(r"UNCORRECTED\s+WFE ([\d.]+) -> ([\d.]+) waves; "
                    r"contrast ([\d.eE+-]+) -> ([\d.eE+-]+)", t, re.S)
    cor = re.search(r"CORRECTED \(control held.*?WFE ([\d.]+) -> ([\d.]+) waves; "
                    r"contrast ([\d.eE+-]+) -> ([\d.eE+-]+)", t, re.S)
    if not unc or not cor:
        sys.exit("e2e6m_records: could not parse the two legs from s5_report.txt")
    r["unc"] = dict(wfe0=float(unc.group(1)), wfe1=float(unc.group(2)),
                    con0=float(unc.group(3)), con1=float(unc.group(4)))
    r["cor"] = dict(wfe0=float(cor.group(1)), wfe1=float(cor.group(2)),
                    con0=float(cor.group(3)), con1=float(cor.group(4)))
    rows = re.findall(r"elt\s+(\d+) dof (\d+): \|engine\| ([\d.eE+-]+)\s+"
                      r"\|model\| ([\d.eE+-]+)\s+rel\.err ([\d.eE+-]+)", t)
    if not rows:
        sys.exit("e2e6m_records: no engine-vs-model rows in s5_report.txt")
    r["check"] = [dict(elt=int(a), dof=int(b), eng=float(c), mod=float(d),
                       rel=float(e)) for a, b, c, d, e in rows]
    r["worst"] = grab(t, r"worst over ALL six DOFs\s+([\d.eE+-]+)",
                      "worst rel err over all DOFs", "s5_report.txt")
    r["worst_ctl"] = grab(t, r"worst over CONTROLLED DOFs\s+([\d.eE+-]+)",
                          "worst rel err over controlled DOFs", "s5_report.txt")
    return r


def all_records():
    return dict(s1=s1(), s2=s2(), s3=s3(), s4=s4(), s5=s5())


if __name__ == "__main__":
    import json
    print(json.dumps(all_records(), indent=2, default=str))
