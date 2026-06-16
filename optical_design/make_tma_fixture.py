"""Self-checking Korsch TMA regression fixture. Run: python3 make_tma_fixture.py
Engine validated against the trusted 2-mirror RC/Cassegrain fixtures, then a 3-mirror
layout whose conics are SOLVED to null S_I,S_II,S_III (residuals machine-zero)."""
import numpy as np, json
from seidel import trace

def aspheric_g(surf, D, th):           # dS_I/dK_j per surface (for the linear solve)
    n=1.0;y=D/2;u=0.0;yb=0.0;ub=th;gs=[]
    for s in surf:
        c=1/s['R'];np_=-n;gI=(np_-n)*c**3*y**4;rho=yb/y if y else 0
        gs.append((gI,gI*rho,gI*rho*rho))
        ph=(np_-n)*c;up=(n*u-y*ph)/np_;ubp=(n*ub-yb*ph)/np_
        y+=s['t']*up;yb+=s['t']*ubp;u,ub=up,ubp;n=np_
    return gs

# --- validate engine on trusted 2-mirror cases (assertions) ---
def two_mirror(f,D,m,beta,K1,K2):
    f1=f/m;R1=2*f1;s2=f1*(1+beta)/(m+1);sep=f1*(m-beta)/(m+1);R2=2*f*(1+beta)/(m**2-1)
    return trace([dict(R=R1,K=K1,t=sep),dict(R=R2,K=K2,t=m*s2)],D/2,0,0,np.deg2rad(0.1))
r=two_mirror(8,1,4,0.125,-1,-(5/3)**2); assert abs(r['SI'])<1e-12 and abs(r['SII'])>1e-9
K1rc=-1-2*1.271/(10.4**2*10.129);K2rc=-(11.4/9.4)**2-2*10.4*11.4/(10.129*9.4**3)
r=two_mirror(57.6,2.4,10.4,0.271,K1rc,K2rc); assert abs(r['SI'])<1e-10 and abs(r['SII'])<1e-10
print("engine validated on 2-mirror RC & Cassegrain (S_I, S_II behave as known).")

# --- chosen realistic Korsch layout (f/2 fast primary, strong convex M2, relay M3) ---
D=1.0; th=np.deg2rad(0.05)
R1,t1,R2,t2,R3 = 3.0, 1.370, 0.280, 2.620, 2.80
# t3 = marginal focus after M3
n=1.0;y=D/2;u=0.0
for R,t in [(R1,t1),(R2,t2)]:
    c=1/R;np_=-n;u=(n*u-y*((np_-n)*c))/np_;y+=t*u;n=np_
c=1/R3;np_=-n;u=(n*u-y*((np_-n)*c))/np_;t3=-y/u
surf=[dict(R=R1,t=t1),dict(R=R2,t=t2),dict(R=R3,t=t3)]

base=trace([dict(R=s['R'],K=0,t=s['t']) for s in surf],D/2,0,0,th)
gs=aspheric_g(surf,D,th)
Mx=np.array([[gs[0][0],gs[1][0],gs[2][0]],[gs[0][1],gs[1][1],gs[2][1]],[gs[0][2],gs[1][2],gs[2][2]]])
K=np.linalg.solve(Mx,-np.array([base['SI'],base['SII'],base['SIII']]))
sv=[dict(R=surf[i]['R'],K=K[i],t=surf[i]['t']) for i in range(3)]
r=trace(sv,D/2,0,0,th)
assert max(abs(r['SI']),abs(r['SII']),abs(r['SIII']))<1e-12, "TMA residuals not null!"
print(f"TMA solved: K1={K[0]:.6f} K2={K[1]:.6f} K3={K[2]:.6f}")
print(f"  EFL={r['EFL']:.4f} (|EFL|/D=f/{abs(r['EFL'])/D:.2f})  t3={t3:.6f}")
print(f"  residuals S_I={r['SI']:.2e} S_II={r['SII']:.2e} S_III={r['SIII']:.2e}  S_IV(Petzval)={r['SIV']:.3e}")

fixture=dict(
  description=("Self-consistent Korsch-type three-mirror anastigmat. Conics solved to "
    "null third-order spherical (S_I), coma (S_II), and astigmatism (S_III) for the "
    "given first-order layout, stop at M1. Residuals are machine-zero BY CONSTRUCTION "
    "-> use as the TMA regression target. S_IV (Petzval) is reported, not nulled: full "
    "field flattening is the extra Korsch condition needing a 4th layout DOF."),
  convention=("n-flip unfolded paraxial; R>0 = concave/converging; metres; stop at M1; "
    "small field 0.05 deg for coma/astig scaling. EFL sign is negative under the "
    "odd-reflection unfolded convention; magnitude is physical. Engine VALIDATED against "
    "the 2-mirror RC (S_I=S_II=0) and classical Cassegrain (S_I=0, S_II=/=0)."),
  layout_m=dict(D=D, R1=R1, t_M1_M2=round(t1,6), R2=R2, t_M2_M3=round(t2,6),
                R3=R3, t_M3_image=round(t3,6)),
  conics=dict(K1=round(float(K[0]),7), K2=round(float(K[1]),7), K3=round(float(K[2]),7)),
  derived=dict(EFL_m=round(r['EFL'],4), f_number=round(abs(r['EFL'])/D,3),
               primary_f_number=round((R1/2)/D,3)),
  expected_seidel=dict(S_I="0 (<1e-12)", S_II="0 (<1e-12)", S_III="0 (<1e-12)",
                       S_IV_petzval=f"{r['SIV']:.4e}"))
json.dump(fixture, open("tma_fixture.json","w"), indent=2)

jwst=dict(
  note=("REAL published JWST OTE values for an independent trace-only check. M1 & M2 "
    "verified from public sources (cite below). M3 radius/conic and the M1-M2-M3 vertex "
    "spacings are in Lightsey et al. 2012 (Opt.Eng. 51, 011003) Table / McElwain et al. "
    "2023 (PASP, arXiv:2301.01779) Table 2 -- fill from the source; NOT reproduced here "
    "because I could not extract them to fixture precision. Do not trace until completed."),
  system=dict(EFL_m=131.4, f_number=20.0, entrance_pupil_diameter_m=6.6,
              field_arcmin="~18 x 9", architecture="Korsch TMA (Korsch 1972)"),
  M1_primary=dict(RoC_m=15.880, conic_K=-0.9967, note="near-parabolic, 18 segments"),
  M2_secondary=dict(RoC_mm=1778.913, conic_K=-1.6598, note="convex"),
  M3_tertiary=dict(RoC="TODO from Lightsey/McElwain Table", conic_K="TODO",
                   note="concave aspheric, in aft-optics; near intermediate image"),
  spacings=dict(t_M1_M2="TODO", t_M2_M3="TODO", t_M3_FSM_focus="TODO"),
  sources=["Lightsey et al. 2012, Opt. Eng. 51(1), 011003",
           "McElwain et al. 2023, PASP (arXiv:2301.01779), Table 2",
           "Stahl, Rules for Optical Metrology (NASA NTRS 20110015786): M1 K=-0.9967, RoC 15.88 m",
           "JWST SMA spec (Ball): RoC 1778.913 mm, K=-1.6598"])
json.dump(jwst, open("jwst_anchor.json","w"), indent=2)
print("\nwrote tma_fixture.json + jwst_anchor.json")
