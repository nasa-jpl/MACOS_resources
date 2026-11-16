import json, math

# ---------------------------------------------------------------------------
# First-order two-mirror layout + conic constants, Schroeder (m, beta) convention.
# Verified: Cassegrain-family relations reproduce HST (R2, K1, K2).
#   f1 = f/m ; R1 = 2 f1 = 2 f/m
#   k = y2/y1 (marginal ray height ratio) ; p = R2/R1
#   Cassegrain branch (convex secondary, virtual object):
#       |s2| = f1 (1+beta)/(m+1)          (secondary -> prime focus)
#       d_sep (M1->M2) = f1 (m-beta)/(m+1)
#       R2 = 2 m |s2|/(m-1) = 2 f (1+beta)/(m^2-1)   (sign: convex)
#       k = (1+beta)/(m+1) ;  p = m(1+beta)/(m^2-1)
#   Gregorian branch (concave secondary, real intermediate image):
#       |s2| = f1 (1+beta)/(m-1)
#       d_sep = f1 (m+beta)/(m-1)
#       |R2| = 2 f (1+beta)/(m^2-1)        (sign: concave -> opposite of Cassegrain)
#       k = (1+beta)/(m-1)
# ---------------------------------------------------------------------------

def cassegrain_layout(f, D, m, beta):
    f1   = f/m
    R1   = 2*f1                       # primary radius (concave)
    s2   = f1*(1+beta)/(m+1)          # secondary -> prime focus
    dsep = f1*(m-beta)/(m+1)          # M1 -> M2 separation
    R2   = 2*f*(1+beta)/(m**2-1)      # secondary radius, convex (Cassegrain)
    b    = beta*f1                    # back focal distance, primary vertex -> focus
    k    = (1+beta)/(m+1)
    p    = R2/R1
    return dict(f1=f1,R1=R1,R2=R2,sep=dsep,bfd=b,k=k,p=p,
                primary_fno=f1/D, system_fno=f/D)

def gregorian_layout(f, D, m, beta):
    f1   = f/m
    R1   = 2*f1
    s2   = f1*(1+beta)/(m-1)
    dsep = f1*(m+beta)/(m-1)          # separation > f1 (secondary past prime focus)
    R2   = -2*f*(1+beta)/(m**2-1)     # concave secondary -> opposite sign
    b    = beta*f1
    k    = (1+beta)/(m-1)
    p    = R2/R1
    return dict(f1=f1,R1=R1,R2=R2,sep=dsep,bfd=b,k=k,p=p,
                primary_fno=f1/D, system_fno=f/D)

def conics(kind, m, beta):
    casseg_sec = ((m+1)/(m-1))**2
    if kind == "classical_cassegrain":
        return -1.0, -casseg_sec
    if kind == "ritchey_chretien":
        K1 = -1.0 - 2*(1+beta)/(m**2*(m-beta))
        K2 = -casseg_sec - 2*m*(m+1)/((m-beta)*(m-1)**3)
        return K1, K2
    if kind == "dall_kirkham":
        return None, 0.0      # K1 solved from spherical condition below
    if kind == "classical_gregorian":
        return -1.0, -((m-1)/(m+1))**2
    raise ValueError(kind)

# spherical-aberration master condition:  1+K1 = (k^4/p^3)[K2 + ((m+1)/(m-1))^2]
def sph_residual(K1, K2, m, k, p):
    return (1+K1) - (k**4/p**3)*(K2 + ((m+1)/(m-1))**2)

def dk_primary_K1(m, k, p):
    # set K2 = 0, solve master condition for K1
    return -1.0 + (k**4/p**3)*((m+1)/(m-1))**2

designs = [
    ("HST-like RC",            "ritchey_chretien",      2.4, 57.6, 10.4, 0.271),
    ("1 m f/8 Classical Cass", "classical_cassegrain",  1.0,  8.0,  4.0, 0.125),
    ("1 m f/8 Dall-Kirkham",   "dall_kirkham",          1.0,  8.0,  4.0, 0.125),
    ("0.5 m f/15 Cassegrain",  "classical_cassegrain",  0.5,  7.5,  5.0, 0.20 ),
    ("1 m f/12 Gregorian",     "classical_gregorian",   1.0, 12.0,  4.0, 0.15 ),
]

rows = []
for name, kind, D, f, m, beta in designs:
    lay = (gregorian_layout if kind=="classical_gregorian"
           else cassegrain_layout)(f, D, m, beta)
    K1, K2 = conics(kind, m, beta)
    if kind == "dall_kirkham":
        K1 = dk_primary_K1(m, lay["k"], lay["p"])
    res = sph_residual(K1, K2, m, lay["k"], lay["p"]) if kind!="classical_gregorian" else float("nan")
    rows.append(dict(
        name=name, family=kind,
        inputs=dict(D_m=D, f_m=f, m=m, beta=beta),
        first_order=dict(
            f1_m=round(lay["f1"],6), R1_m=round(lay["R1"],6), R2_m=round(lay["R2"],6),
            M1_M2_sep_m=round(lay["sep"],6), back_focal_dist_m=round(lay["bfd"],6),
            primary_fno=round(lay["primary_fno"],4), system_fno=round(lay["system_fno"],4),
            k_ratio=round(lay["k"],6), p_ratio=round(lay["p"],6)),
        conics=dict(K1=round(K1,7), K2=round(K2,7)),
        check_spherical_residual=("n/a" if math.isnan(res) else f"{res:.2e}"),
    ))

with open("telescope_design_fixtures.json","w") as fp:
    json.dump(dict(convention="Schroeder (m, beta); beta = back-focal-dist / f1; "
                              "radii magnitudes, R1>0 concave primary, "
                              "Cassegrain secondary convex, Gregorian secondary concave (R2<0)",
                   designs=rows), fp, indent=2)

# pretty print
for r in rows:
    print(f"\n=== {r['name']}  [{r['family']}] ===")
    i=r['inputs']; fo=r['first_order']; c=r['conics']
    print(f"  inputs: D={i['D_m']} m  f={i['f_m']} m (f/{fo['system_fno']})  m={i['m']}  beta={i['beta']}")
    print(f"  primary f/{fo['primary_fno']}   f1={fo['f1_m']} m")
    print(f"  R1={fo['R1_m']} m   R2={fo['R2_m']} m   M1-M2 sep={fo['M1_M2_sep_m']} m   BFD={fo['back_focal_dist_m']} m")
    print(f"  k={fo['k_ratio']}  p={fo['p_ratio']}")
    print(f"  K1={c['K1']}   K2={c['K2']}")
    print(f"  spherical-condition residual: {r['check_spherical_residual']}")
