"""Coronagraph back-end layout generator (Lyot-type, two-DM wavefront control).

Given the beam/DM/wavelength/working-angle spec, emit:
  - actuator pitch and the dark-hole outer working angle it supports
  - the quarter-Talbot DM1->DM2 separation (full phase->amplitude conversion distance)
  - OAP focal lengths and F/# for every relay, plus pupil diameters at each conjugate
  - focal-plane physical scales (IWA/OWA radii at the FPM) and the Nyquist final F/#

Conventions: lengths in mm, wavelength in nm on input. Collimated-space relays are
unit-magnification by default (pupil diameter = D_beam at apodizer, DM1, Lyot); set a
relay magnification = ratio of OAP focal lengths if a device needs a different pupil size.

Working angles are in lambda/D (D = telescope aperture); they are independent of the
physical beam size because N actuators sample N/2 cycles across the pupil = N/2 lambda/D.
"""
import json, os, sys, math

def design(D_beam_mm=46.3, N_act=48, lam_nm=550.0,
           IWA_lod=3.0, OWA_lod=24.0,
           input_fnum=20.0, fpm_fnum=30.0, pixel_um=13.0,
           talbot_design_lod=None):
    lam = lam_nm * 1e-6                       # nm -> mm
    d_act = D_beam_mm / N_act                 # actuator pitch (mm)
    OWA_max = N_act / 2.0                      # lambda/D the DM can reach
    if talbot_design_lod is None:
        talbot_design_lod = OWA_lod           # default: size quarter-Talbot at the OWA

    # --- quarter-Talbot DM separation: full phase->amplitude at z = Lambda^2 / (2 lambda)
    # speckle at alpha (lambda/D) <-> pupil ripple of alpha cycles across D_beam:
    #   Lambda = D_beam / alpha  ;  z_qT(alpha) = Lambda^2 / (2 lambda)
    def z_qT(alpha):                           # mm
        Lam = D_beam_mm / alpha
        return Lam * Lam / (2.0 * lam)
    z_dm = z_qT(talbot_design_lod)
    z_band = {f"{a:g} l/D": round(z_qT(a), 1)
              for a in (IWA_lod, math.sqrt(IWA_lod*OWA_lod), OWA_lod)}

    # --- Nyquist final F/# at the detector: lambda*F# >= 2*pixel
    F_final_min = 2.0 * (pixel_um * 1e-3) / lam
    F_final = math.ceil(F_final_min)           # round up to a slightly slow beam

    # --- OAP focal lengths (unit-mag relays); F/# = f / D_beam in collimated space
    f_OAP1 = D_beam_mm * input_fnum            # collimator: telescope focus -> apodizer pupil
    f_OAP4 = D_beam_mm * fpm_fnum              # pupil(DM) -> FPM focus
    f_OAP5 = D_beam_mm * fpm_fnum              # FPM focus -> Lyot pupil  (unit mag)
    f_OAP6 = D_beam_mm * F_final               # Lyot pupil -> detector focus

    # --- focal-plane physical scales (linear size of 1 lambda/D = lambda * F/#)
    lod_at_fpm = lam * fpm_fnum                # mm per (lambda/D) at the FPM
    IWA_r = IWA_lod * lod_at_fpm               # occulting-spot inner radius (mm)
    OWA_r = OWA_lod * lod_at_fpm
    lod_at_det = lam * F_final                 # mm per (lambda/D) at detector
    psf_core_px = lod_at_det / (pixel_um*1e-3) # pixels per lambda/D (>=2 = Nyquist)

    # ---- self-checks ----
    assert OWA_lod <= OWA_max + 1e-9, (
        f"OWA target {OWA_lod} l/D exceeds DM reach N/2={OWA_max}; need N>={int(math.ceil(2*OWA_lod))}")
    assert lam * F_final >= 2*(pixel_um*1e-3) - 1e-12, "Nyquist not satisfied"
    assert z_dm > 0 and f_OAP1 > 0 and f_OAP6 > 0
    assert IWA_lod < OWA_lod

    planes = [
        ("0  telescope focus", "focal", "-", "-", f"feed F/{input_fnum:g}"),
        ("1  OAP1 collimator",  "OAP",   f"f={f_OAP1:.1f}", "-", "FP -> pupil"),
        ("2  apodizer",         "pupil", f"D={D_beam_mm:.2f}", "-", "diffraction control"),
        ("3  OAP2+OAP3 relay",  "OAP x2","unit mag", "-", "pupil -> focus -> pupil (anti-symm pair)"),
        ("4  DM1",              "pupil", f"D={D_beam_mm:.2f}", f"{N_act} act, pitch {d_act:.3f}", "phase"),
        ("5  DM2",              "near-pupil", f"D={D_beam_mm:.2f}", f"z={z_dm:.1f} mm from DM1", "phase+amplitude"),
        ("6  OAP4",             "OAP",   f"f={f_OAP4:.1f}", f"F/{fpm_fnum:g}", "pupil -> FPM focus"),
        ("7  FPM",              "focal", "-", f"IWA r={IWA_r:.4f} mm", "reject starlight (reflective->LOWFS)"),
        ("8  OAP5",             "OAP",   f"f={f_OAP5:.1f}", "-", "FPM focus -> Lyot pupil"),
        ("9  Lyot stop",        "pupil", f"D={D_beam_mm:.2f}", "match aperture, undersized", "block diffracted starlight"),
        ("10 OAP6",             "OAP",   f"f={f_OAP6:.1f}", f"F/{F_final:g}", "Lyot pupil -> detector"),
        ("11 detector",         "focal", "-", f"{psf_core_px:.2f} px per l/D", "science / dark hole"),
    ]
    return dict(
        inputs=dict(D_beam_mm=D_beam_mm, N_act=N_act, lam_nm=lam_nm,
                    IWA_lod=IWA_lod, OWA_lod=OWA_lod, input_fnum=input_fnum,
                    fpm_fnum=fpm_fnum, pixel_um=pixel_um,
                    talbot_design_lod=talbot_design_lod),
        derived=dict(
            actuator_pitch_mm=round(d_act,4),
            OWA_supported_lod=OWA_max,
            quarter_talbot_sep_mm=round(z_dm,1),
            quarter_talbot_by_freq=z_band,
            final_fnum=F_final, final_fnum_min=round(F_final_min,2),
            psf_core_pixels=round(psf_core_px,2),
            fpm_IWA_radius_mm=round(IWA_r,4), fpm_OWA_radius_mm=round(OWA_r,4),
            f_OAP1_mm=round(f_OAP1,1), f_OAP4_mm=round(f_OAP4,1),
            f_OAP5_mm=round(f_OAP5,1), f_OAP6_mm=round(f_OAP6,1)),
        planes=planes)

if __name__ == "__main__":
    OUT = (sys.argv[1] if len(sys.argv) > 1
           else os.environ.get("CORO_OUT_DIR")
           or os.path.dirname(os.path.abspath(__file__))) or "."
    os.makedirs(OUT, exist_ok=True)

    r = design()   # Roman-class default
    i, d = r["inputs"], r["derived"]
    print(f"=== coronagraph layout (Roman-class default) ===")
    print(f"beam {i['D_beam_mm']} mm, {i['N_act']} act @ {i['lam_nm']} nm; target {i['IWA_lod']}-{i['OWA_lod']} l/D")
    print(f"actuator pitch         {d['actuator_pitch_mm']} mm")
    print(f"OWA supported          {d['OWA_supported_lod']} l/D (target {i['OWA_lod']})")
    print(f"quarter-Talbot sep     {d['quarter_talbot_sep_mm']} mm  (at {i['talbot_design_lod']} l/D)")
    print(f"  z_qT across band     {d['quarter_talbot_by_freq']}")
    print(f"final F/# (Nyquist)    F/{d['final_fnum']} (min {d['final_fnum_min']}) -> {d['psf_core_pixels']} px per l/D")
    print(f"FPM IWA / OWA radius   {d['fpm_IWA_radius_mm']} / {d['fpm_OWA_radius_mm']} mm")
    print(f"OAP f: OAP1={d['f_OAP1_mm']}  OAP4={d['f_OAP4_mm']}  OAP5={d['f_OAP5_mm']}  OAP6={d['f_OAP6_mm']} mm")
    print("\nplane table:")
    for p in r["planes"]:
        print(f"  {p[0]:<22}{p[1]:<11}{p[2]:<16}{p[3]:<22}{p[4]}")

    # anchor sanity vs published Roman/HCIT (~1 m operating separation)
    print(f"\nNOTE: quarter-Talbot at OWA = {d['quarter_talbot_sep_mm']} mm is the reference SCALE.")
    print("Published HCIT/Roman OPERATING separation ~1 m (sub-quarter-Talbot, numerically")
    print("optimized across the band + packaging/beam-walk limits). Treat z_qT as upper-scale.")

    json.dump(r, open(os.path.join(OUT,"coronagraph_layout.json"),"w"), indent=2)
    print(f"\nwrote coronagraph_layout.json to {OUT}")
