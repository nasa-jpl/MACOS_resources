import numpy as np

# Paraxial + Seidel engine for coaxial mirror systems (n-flip / unfolded model).
# Convention (validated below against the trusted 2-mirror RC & Cassegrain fixtures):
#   light starts in +z with n=+1; each mirror flips n -> -n.
#   R > 0  => concave / converging mirror (focuses parallel light at +f).
#   surface power phi = (n'-n)*c ;  refraction  n'u' = n u - y*phi
#   transfer  y_next = y + t*u'   (t = positive vertex separation, unfolded)
#   conic aspheric 4th-order departure from base sphere: A4 = K*c^3/8
#   aspheric Seidel:  S_I_a = 8*(n'-n)*A4*y^4 = (n'-n)*K*c^3*y^4
#                     S_II_a = S_I_a*(ybar/y) ; S_III_a = S_I_a*(ybar/y)^2

def trace(surfaces, y1, u1, ybar1, ubar1):
    """surfaces: list of dicts {R, K, t}. t = distance to NEXT surface.
    Returns per-surface paraxial + the Seidel sums S_I..S_IV (and EFL)."""
    n = 1.0
    y, u, yb, ub = y1, u1, ybar1, ubar1
    SI=SII=SIII=SIV=0.0
    H = n*(ub*y - u*yb)            # Lagrange invariant (constant)
    rows=[]
    for s in surfaces:
        c = 0.0 if s['R'] in (None, np.inf) else 1.0/s['R']
        K = s.get('K',0.0)
        nprime = -n                # mirror
        phi = (nprime - n)*c
        # refraction invariants (A = n*i)
        A  = n*(y*c + u)
        Ab = n*(yb*c + ub)
        up  = (n*u  - y*phi)/nprime
        ubp = (n*ub - yb*phi)/nprime
        dun = up/nprime - u/n
        # spherical (paraxial) surface contributions
        sI  = -A*A   * y * dun
        sII = -A*Ab  * y * dun
        sIII= -Ab*Ab * y * dun
        sIV = -H*H * c * (1.0/nprime - 1.0/n)
        # aspheric (conic) contributions
        A4 = K*c**3/8.0
        sIa = 8.0*(nprime-n)*A4*y**4
        rho = (yb/y) if y!=0 else 0.0
        sIIa  = sIa*rho
        sIIIa = sIa*rho*rho
        SI+=sI+sIa; SII+=sII+sIIa; SIII+=sIII+sIIIa; SIV+=sIV
        rows.append(dict(y=y,u=u,yb=yb,ub=ub,A=A,Ab=Ab,phi=phi))
        # transfer
        t = s['t']
        y  = y  + t*up
        yb = yb + t*ubp
        u, ub = up, ubp
        n = nprime
    EFL = -y1/u if u!=0 else np.inf
    return dict(SI=SI,SII=SII,SIII=SIII,SIV=SIV,EFL=EFL,
                y_final=y,u_final=u,H=H,rows=rows)

def two_mirror(f,D,m,beta,K1,K2,gregorian=False):
    f1=f/m; R1=2*f1
    if not gregorian:
        s2=f1*(1+beta)/(m+1); sep=f1*(m-beta)/(m+1); R2=2*f*(1+beta)/(m**2-1)
    else:
        s2=f1*(1+beta)/(m-1); sep=f1*(m+beta)/(m-1); R2=-2*f*(1+beta)/(m**2-1)
    t2=m*s2
    surf=[dict(R=R1,K=K1,t=sep), dict(R=R2,K=K2,t=t2)]
    theta=np.deg2rad(0.1)         # small field for coma/astig scaling
    return trace(surf, y1=D/2, u1=0.0, ybar1=0.0, ubar1=theta), (R1,R2,sep,t2)

if __name__=="__main__":
    print("=== VALIDATION against trusted 2-mirror fixtures ===")

    # Classical Cassegrain: expect SI~0, SII != 0
    r,_=two_mirror(8.0,1.0,4.0,0.125,-1.0,-(5/3)**2)
    print(f"Classical Cassegrain: EFL={r['EFL']:.4f} (want 8.0)  SI={r['SI']:.3e} (want ~0)  SII={r['SII']:.3e} (nonzero)")
    # RC: expect SI~0 AND SII~0
    K1rc=-1-2*(1+0.271)/(10.4**2*(10.4-0.271))
    K2rc=-((10.4+1)/(10.4-1))**2-2*10.4*(10.4+1)/((10.4-0.271)*(10.4-1)**3)
    r,_=two_mirror(57.6,2.4,10.4,0.271,K1rc,K2rc)
    print(f"HST-like RC:          EFL={r['EFL']:.3f} (want 57.6) SI={r['SI']:.3e} (~0)  SII={r['SII']:.3e} (~0)  SIII={r['SIII']:.3e} (nonzero)")

    print("\n=== Synthetic Korsch TMA: solve conics to null S_I, S_II, S_III ===")
    def g_coeffs(surfaces, y1,u1,yb1,ub1):
        """trace with K=0 to get spherical sums and aspheric sensitivity g_j per surface."""
        base=trace([dict(R=s['R'],K=0.0,t=s['t']) for s in surfaces], y1,u1,yb1,ub1)
        # recompute per-surface y, ybar, c, (n'-n) to build g_j
        n=1.0; y,u,yb,ub=y1,u1,yb1,ub1; gs=[]
        for s in surfaces:
            c=1.0/s['R']; nprime=-n
            gI=8.0*(nprime-n)*(c**3/8.0)*y**4         # dS_I/dK_j
            rho=yb/y if y!=0 else 0.0
            gs.append((gI, gI*rho, gI*rho*rho))
            phi=(nprime-n)*c
            up=(n*u-y*phi)/nprime; ubp=(n*ub-yb*phi)/nprime
            y=y+s['t']*up; yb=yb+s['t']*ubp; u,ub=up,ubp; n=nprime
        return base, gs

    # first-order layout (chosen for a clean, real final focus; stop at M1)
    D=1.0; theta=np.deg2rad(0.1)
    lay=[dict(R=4.0,t=1.5), dict(R=1.30,t=2.6), dict(R=1.7,t=None)]
    # set t3 so the marginal ray focuses at the image plane
    tmp=[dict(R=s['R'],K=0.0,t=(s['t'] if s['t'] else 0.0)) for s in lay]
    # trace marginal up to after M3 to find focus distance
    n=1.0;y=D/2;u=0.0
    for i,s in enumerate(tmp[:-1]+[tmp[-1]]):
        c=1/s['R'];nprime=-n;phi=(nprime-n)*c;up=(n*u-y*phi)/nprime
        if i<2: y=y+s['t']*up
        u=up;n=nprime
    t3=-y/u
    lay[2]['t']=t3
    surfaces=lay

    base,gs=g_coeffs(surfaces, D/2,0.0, 0.0,theta)
    import numpy as np
    M=np.array([[gs[0][0],gs[1][0],gs[2][0]],
                [gs[0][1],gs[1][1],gs[2][1]],
                [gs[0][2],gs[1][2],gs[2][2]]])
    b=-np.array([base['SI'],base['SII'],base['SIII']])
    K=np.linalg.solve(M,b)
    K1,K2,K3=K
    solved=[dict(R=surfaces[0]['R'],K=K1,t=surfaces[0]['t']),
            dict(R=surfaces[1]['R'],K=K2,t=surfaces[1]['t']),
            dict(R=surfaces[2]['R'],K=K3,t=surfaces[2]['t'])]
    r=trace(solved, D/2,0.0, 0.0,theta)
    print(f"layout: R1={surfaces[0]['R']} t1={surfaces[0]['t']:.4f} | R2={surfaces[1]['R']} t2={surfaces[1]['t']:.4f} | R3={surfaces[2]['R']} t3={t3:.4f}")
    print(f"solved conics: K1={K1:.6f}  K2={K2:.6f}  K3={K3:.6f}")
    print(f"EFL={r['EFL']:.4f}  |EFL|/D=f/{abs(r['EFL'])/D:.2f}")
    print(f"residuals:  S_I={r['SI']:.3e}  S_II={r['SII']:.3e}  S_III={r['SIII']:.3e}   <-- all ~0 = anastigmat")
    print(f"Petzval     S_IV={r['SIV']:.3e}  (field curvature; independent of conics)")

    # NOTE: this __main__ is the validation ORACLE (print-only).  The committed
    # TMA regression fixture (tma_fixture.json, a realistic f/5.03 Korsch design)
    # is generated by make_tma_fixture.py, NOT here -- the inline toy layout above
    # is a different, illustrative solve.  Do not write tma_fixture.json from here:
    # that would clobber the authoritative fixture with the toy values.
