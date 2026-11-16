"""zmx2macos gates (BRIEF_to_zmx2macos.md).

G1  the DST deck against the CODE V import dst2_F01_norm.in: relative geometry to 1e-9 m
    and 1e-9 rad, Kr / Kc to 1e-9.
G2  the AFTA testbed deck (X tilts, Y decenters, order flag 1) converts, loads and traces;
    its fingerprint is pinned.
G3  must-fail: UTF-16 read raw, an unsupported TYPE refused by name, a truncated file
    refused (never an empty deck).
Plus: the frame algebra against the textbook rotations, IRREGULA, EVENASPH, glass, and the
converter's own tracer against the engine on every deck.
"""
import math
import subprocess
import sys

import numpy as np
import pytest

from conftest import AFTA, DST, DST_CODEV, ZDIR, convert, cross_trace, need, zm

# ======================================================================================
# small hand-written decks (current OpticStudio layout, ASCII)
# ======================================================================================
HEAD = """VERS 230310 1 1 1
MODE SEQ
NAME {name}
UNIT {unit} X W X CM MR CPMM
{aper}
XFLN 0 0
YFLN 0 0
WAVM 1 0.58756180000000004 1
PWAV 1
"""


def surf(n, typ='STANDARD', curv=0.0, disz=0.0, extra=''):
    return (f'SURF {n}\n  TYPE {typ}\n  FIMP \n\n  CURV {curv!r} 0 0 0 0 ""\n'
            f'  DISZ {disz}\n  DIAM 10 1 0 0 1 ""\n{extra}')


def deck(surfs, name='test', unit='MM', aper='ENPD 20', tail='BLNK \n'):
    return HEAD.format(name=name, unit=unit, aper=aper) + ''.join(surfs) + tail


def singlet(c1=0.01, k1=0.0, asph1=None, typ1='STANDARD'):
    extra = f'  GLAS N-BK7 0 0 1.5168 64.17 0 0 0 0 0 0\n  CONI {k1!r}\n'
    if asph1:
        extra += ''.join(f'  PARM {i} {a!r}\n' for i, a in enumerate(asph1, 1))
    return deck([surf(0, disz='INFINITY'),
                 surf(1, typ1, c1, 6.0, '  STOP\n' + extra),
                 surf(2, 'STANDARD', 0.0, 185.0),
                 surf(3)])


def mirror(c, k=0.0, asph=None, typ='STANDARD'):
    extra = f'  GLAS MIRROR 0 0 1.5 40 0 0 0 0 0 0\n  CONI {k!r}\n'
    if asph:
        extra += ''.join(f'  PARM {i} {a!r}\n' for i, a in enumerate(asph, 1))
    return deck([surf(0, disz='INFINITY'),
                 surf(1, 'STANDARD', 0.0, 100.0, '  STOP\n'),
                 surf(2, typ, c, 0.5 / c, extra),           # focus at -R/2
                 surf(3)])


def write(tmp_path, text, name='t.zmx', enc='ascii', bom=b''):
    p = tmp_path / name
    p.write_bytes(bom + text.replace('\n', '\r\n').encode(enc))
    return p


def conic_minus_sphere(c, k, nmax=8):
    """Even-power coefficients alpha_n (r^2n, n = 1..nmax) of conic(c, k) - sphere(c):
    z = sum_n (2n-2)! / (2^(2n-1) n! (n-1)!) (1+k)^(n-1) c^(2n-1) r^(2n)."""
    a = []
    for n in range(1, nmax + 1):
        b = math.factorial(2 * n - 2) / (2 ** (2 * n - 1) * math.factorial(n) * math.factorial(n - 1))
        a.append(b * ((1 + k) ** (n - 1) - 1.0) * c ** (2 * n - 1))
    return a


# ======================================================================================
# G1: the DST deck against the CODE V import
# ======================================================================================
G1_SEQ = ['oap1', 'dm1', 'dm2', 'oap2', 'oap3', 'apod', 'oap4', 'fpm_front', 'fpm_rear',
          'oap5', 'oap6', 'oap7', 'oap8', 'fpa']


def _ang(a, b):
    return math.atan2(np.linalg.norm(np.cross(a, b)), abs(np.dot(a, b)))


def _g1_pairs(load, engine):
    need(DST)
    text, cv = convert(DST)
    Z = load(text, 'dst.in').by_name()
    C = load(open(DST_CODEV).read(), 'codev.in').by_name()

    def find(E, k):
        return next(e for nm, e in E.items() if nm.endswith(', ' + k) or nm == k)
    z = [find(Z, k) for k in G1_SEQ[:-1]] + [Z['CodeV S33']]   # the image surface
    c = [find(C, k) for k in G1_SEQ]
    return z, c, cv


def test_g1_dst_elements():
    need(DST)
    assert open(DST, 'rb').read(2) == b'\xff\xfe', 'the DST file is UTF-16 LE as saved'
    text, cv = convert(DST)
    assert cv.encoding == 'utf-16-le'
    assert [r['etype'] for r in cv.elts].count('Reflector') == 11
    assert [r['etype'] for r in cv.elts].count('Refractor') == 2
    assert cv.elts[-1]['etype'] == 'FocalPlane' and len(cv.elts) == 14
    assert len(cv.dropped) == 24 and cv.warn == []
    assert 'GlassElt=  Silica' in text and 'ApStop=' in text
    assert text.count('COORDBRK') == 0, 'coordinate breaks emit nothing'


def test_g1_relative_geometry(load, engine, capsys):
    z, c, _ = _g1_pairs(load, engine)
    rows = []
    for i in range(len(G1_SEQ) - 1):
        dz = np.linalg.norm(z[i + 1]['v'] - z[i]['v'])
        dc = np.linalg.norm(c[i + 1]['v'] - c[i]['v'])
        az, ac = _ang(z[i]['p'], z[i + 1]['p']), _ang(c[i]['p'], c[i + 1]['p'])
        rows.append((G1_SEQ[i], G1_SEQ[i + 1], dz, dc, az, ac))
        assert abs(dz - dc) < 1e-6, f'{G1_SEQ[i]}-{G1_SEQ[i+1]}: {dz} vs {dc} mm'
        # The FPA tilt is INCLUDED: the assessment's factor 2 compared Zemax surface 64
        # ("fpa"), which sits between two -7.0877 deg breaks; the image surface 66 carries
        # the full 14.1755 deg, the CODE V fpa's (measured 2026-11-03).
        assert abs(az - ac) < 1e-9, f'{G1_SEQ[i]}-{G1_SEQ[i+1]}: {az} vs {ac} rad'
    with capsys.disabled():
        print('\n  G1 pair                 Zemax mm        CODE V mm     diff    fold deg (both)')
        for a, b, dz, dc, az, ac in rows:
            print(f'  {a:>9}-{b:<10} {dz:14.6f} {dc:14.6f} {dz-dc:9.1e}  {math.degrees(az):9.5f}'
                  f' {math.degrees(ac):9.5f} {az-ac:8.1e}')


def test_g1_rigid_body(load, engine):
    """All 14 vertices and normals at once: the best rotation and translation that maps
    the Zemax deck onto the CODE V deck leaves under 1e-9 m and 1e-9 rad."""
    z, c, _ = _g1_pairs(load, engine)
    P = np.array([e['v'] for e in z]); Q = np.array([e['v'] for e in c])
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    R = (U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt).T
    res = np.linalg.norm((R @ Pc.T).T - Qc, axis=1)
    assert res.max() < 1e-6, res
    for a, b, k in zip(z, c, G1_SEQ):
        assert _ang(R @ a['p'], b['p']) < 1e-9, k
        if abs(b['kr']) < 1e17:                              # powered: psi sign too
            assert np.dot(R @ a['p'], b['p']) > 0, k
            assert abs(a['kr'] / b['kr'] - 1) < 1e-9, k
        assert abs(a['kc'] - b['kc']) < 1e-9, k


def test_g1_traces_and_aims(load, engine):
    need(DST)
    text, cv = convert(DST)
    L = load(text, 'dst.in')
    assert L.n == 14 and L.ok.sum() == L.nrays == 3210
    # diffraction-limited as converted (the CODE V import traces 0.517 here)
    assert L.rms / 5.75e-4 < 2e-3, L.rms
    assert L.rms == pytest.approx(5.5438e-07, rel=0.02)
    # the engine's ApStop re-aim keeps the converter's chief ray
    assert np.linalg.norm(L.chf_dir - cv.chief_d) < 1e-9
    # the chief ray meets every OAP at its OBDC aperture centre (Matt's sign question)
    hits = {h[0]: h[1] for h in cv.chief_hits}
    n = 0
    for r in cv.elts:
        if 'obdc' in r['ap']:
            assert np.allclose(hits[r['num']][:2], r['ap']['obdc'], atol=1e-5), r['num']
            n += 1
    assert n == 8
    worst, nr = cross_trace(cv, L)
    assert nr > 300 and worst < 1e-7, worst


def test_g1_options(load, engine):
    """--keep-dummies (Matt's 38 elements), --local-frames and --base-units m trace alike."""
    need(DST)
    ref = load(convert(DST)[0], 'a.in')
    kd = load(convert(DST, '--keep-dummies')[0], 'b.in')
    lf = load(convert(DST, '--local-frames')[0], 'c.in')
    me = load(convert(DST, '--base-units', 'm')[0], 'd.in')
    assert kd.n == 38 and lf.n == 14 and me.n == 14
    for L, sc in ((lf, 1.0), (me, 1e-3)):
        assert L.ok.sum() == 3210
        assert L.rms == pytest.approx(ref.rms * sc, rel=1e-6)
        assert np.allclose(L.pos_fp / sc, ref.pos_fp, atol=1e-8)
    # --keep-dummies: surface 64 (fpa) shares the image's vertex at half its tilt, and the
    # engine steps a ray backwards only onto a Reference or the surface after one: the
    # focal plane misses the rays that must step back (1459 of 3210, 2026-11-03).  Every
    # ray it keeps lands where the default deck puts it.
    both = kd.ok & ref.ok
    assert both.sum() > 1700
    assert np.allclose(kd.pos_fp[:, both], ref.pos_fp[:, both], atol=1e-8)


# ======================================================================================
# G2: the AFTA testbed deck
# ======================================================================================
def test_g2_afta(load, engine, capsys):
    need(AFTA)
    raw = open(AFTA, 'rb').read()
    lines = raw[2:].decode('utf-16-le').splitlines()
    p3 = sum(1 for i, l in enumerate(lines) if l.startswith('  PARM 3 ') and float(l.split()[2]) != 0)
    p2 = sum(1 for i, l in enumerate(lines) if l.startswith('  PARM 2 ') and float(l.split()[2]) != 0)
    f1 = sum(1 for l in lines if l.startswith('  PARM 6 1'))
    assert p3 >= 30 and p2 >= 25 and f1 >= 2, (p3, p2, f1)    # X tilts, Y decenters, flag 1
    text, cv = convert(AFTA, '--glass', 'STRUCTURE=1')
    assert cv.warn == []
    L = load(text, 'afta.in')
    # fingerprint (2026-11-03): 48 elements; 3210 rays, all traced, one vignetted
    assert L.n == 48
    assert (L.nrays, int(L.ok.sum()), int(L.passed.sum())) == (3210, 3210, 3209)
    assert L.rms == pytest.approx(2.5396e-06, rel=0.02)      # mm, ~lambda/220 at 570 nm
    assert np.linalg.norm(L.chf_dir - cv.chief_d) < 1e-9
    hits = {h[0]: h[1] for h in cv.chief_hits}
    off = [math.hypot(*(hits[r['num']][:2] - np.array(r['ap']['obdc'])))
           for r in cv.elts if 'obdc' in r['ap'] and r['curv'] != 0.0]
    # every OAP's aperture centre (the designer's round numbers): within 0.18 mm measured
    assert len(off) == 13 and max(off) < 0.2, off
    worst, nr = cross_trace(cv, L)
    assert nr > 300 and worst < 1e-7, worst
    with capsys.disabled():
        print(f'\n  G2 AFTA: {L.n} elements, {L.nrays} rays, ok {int(L.ok.sum())}, '
              f'pass {int(L.passed.sum())}, rms {L.rms:.4e} mm; chief to OBDC max '
              f'{max(off):.3f} mm over 13 OAPs; cross-trace {worst:.1e} mm')


def test_g2_unknown_glass_refused(tmp_path):
    """Ruling 2026-11-03: a catalog glass the engine lacks is refused by name (its GLAS nd
    is filler); --glass NAME=1 declares a drawing-only material air."""
    need(AFTA)
    with pytest.raises(zm.ZmxError) as e:
        convert(AFTA)
    msg = str(e.value)
    assert 'STRUCTURE at surfaces 82, 83, 91, 92, 108, 134' in msg
    assert '--glass STRUCTURE=INDEX' in msg
    out = tmp_path / 'never.in'
    r = subprocess.run([sys.executable, str(ZDIR / 'zmx2macos.py'), str(AFTA), '-o', str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 2 and 'REFUSED' in r.stderr and 'STRUCTURE' in r.stderr
    assert not out.exists()
    _, cv = convert(AFTA, '--glass', 'STRUCTURE=1')
    assert len(cv.elts) == 48


def test_unknown_glass_refused_small(tmp_path):
    t = singlet().replace('GLAS N-BK7', 'GLAS NOSUCHGLASS')
    with pytest.raises(zm.ZmxError, match='NOSUCHGLASS at surface 1\\b'):
        convert(write(tmp_path, t))
    text, _ = convert(write(tmp_path, t, 'g.zmx'), '--glass', 'NOSUCHGLASS=1.6')
    assert 'IndRef=   1.6000000000000001E+00' in text and 'GlassElt' not in text


# ======================================================================================
# G3: must-fail
# ======================================================================================
@pytest.mark.parametrize('enc,bom', [('ascii', b''), ('utf-16-le', b'\xff\xfe'),
                                     ('utf-16-le', b''), ('utf-16-be', b'\xfe\xff'),
                                     ('utf-8', b'\xef\xbb\xbf')])
def test_g3_encodings(tmp_path, enc, bom):
    ref = convert(write(tmp_path, singlet(), 'ref.zmx'))[0]
    out = convert(write(tmp_path, singlet(), 'x.zmx', enc, bom))[0]
    assert out.split('\n', 2)[2] == ref.split('\n', 2)[2]


@pytest.mark.parametrize('typ', ['TOROIDAL', 'ZERNSAG', 'ZERNSTAN', 'GRID_SAG', 'DGRATING',
                                 'TILTSURF', 'PARAXIAL', 'BINARY_2'])
def test_g3_unsupported_type_refused(tmp_path, typ):
    p = write(tmp_path, singlet(typ1=typ))
    with pytest.raises(zm.ZmxError, match=f'surface 1 {typ}'):
        convert(p)


def test_g3_cli_refusal_writes_nothing(tmp_path):
    p = write(tmp_path, singlet(typ1='TOROIDAL'))
    out = tmp_path / 'never.in'
    r = subprocess.run([sys.executable, str(ZDIR / 'zmx2macos.py'), str(p), '-o', str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 2 and 'REFUSED' in r.stderr and 'TOROIDAL' in r.stderr
    assert not out.exists()


def test_g3_truncated_refused(tmp_path):
    text = singlet()
    cut = text[:text.index('SURF 2') + 40]                   # ends inside SURF 2
    with pytest.raises(zm.ZmxError, match='truncated'):
        convert(write(tmp_path, cut))
    (tmp_path / 'empty.zmx').write_bytes(b'')
    with pytest.raises(zm.ZmxError, match='empty'):
        convert(tmp_path / 'empty.zmx')
    with pytest.raises(zm.ZmxError, match='no SURF'):
        convert(write(tmp_path, HEAD.format(name='x', unit='MM', aper='ENPD 1'), 'h.zmx'))


def test_g3_truncated_program_deck(tmp_path):
    """Cut inside the surface list: refused.  An odd byte count in UTF-16: refused.  Cut
    after the surface list (in the analysis data): converted, and the missing MNUM said."""
    need(DST)
    raw = open(DST, 'rb').read()
    p = tmp_path / 'cut.zmx'
    at = raw.index('SURF 40'.encode('utf-16-le'))
    p.write_bytes(raw[:at + 60])
    with pytest.raises(zm.ZmxError, match='truncated'):
        convert(p)
    p.write_bytes(raw[:at + 61])
    with pytest.raises(zm.ZmxError, match='not decodable'):
        convert(p)
    at = raw.index('OPDX'.encode('utf-16-le'))
    p.write_bytes(raw[:at])
    text, cv = convert(p)
    assert len(cv.elts) == 14 and any('without its MNUM' in w for w in cv.warn)


def test_g3_other_refusals(tmp_path):
    t = singlet().replace('MODE SEQ', 'MODE NSC')
    with pytest.raises(zm.ZmxError, match='MODE NSC'):
        convert(write(tmp_path, t))
    t = singlet(typ1='EVENASPH', asph1=[1e-3, 0, 0, 0, 0, 0, 0, 0])
    with pytest.raises(zm.ZmxError, match='r\\^2 term'):
        convert(write(tmp_path, t))
    t = singlet().replace('  STOP\n', '  STOP\n  SCBD 2 1 2 0.5 0 0 0 0\n')
    with pytest.raises(zm.ZmxError, match='SCBD'):
        convert(write(tmp_path, t))
    t = singlet().replace('ENPD 20\n', '')
    with pytest.raises(zm.ZmxError, match='aperture definition'):
        convert(write(tmp_path, t))


# ======================================================================================
# the frame algebra against the textbook
# ======================================================================================
def _R(axis, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@pytest.mark.parametrize('flag', [0, 1])
def test_frame_coordbrk_textbook(flag):
    """Zemax: flag 0 decenters then tilts about x, y, z (each about the new axes); any
    other flag tilts z, y, x then decenters.  Right-handed rotations of the frame."""
    dx, dy, rx, ry, rz = 3.0, -7.0, 11.0, -23.0, 37.0
    A0 = _R('y', 5.0) @ _R('x', -2.0)
    b0 = np.array([1.0, 2.0, 3.0])
    f = zm.Frame(); f.A = A0.copy(); f.b = b0.copy()
    f.cbrk(dx, dy, rx, ry, rz, flag)
    if flag == 0:
        A = A0 @ _R('x', rx) @ _R('y', ry) @ _R('z', rz)
        b = b0 + A0 @ [dx, dy, 0]
    else:
        A = A0 @ _R('z', rz) @ _R('y', ry) @ _R('x', rx)
        b = b0 + A @ [dx, dy, 0]
    assert np.allclose(f.A, A, atol=1e-14) and np.allclose(f.b, b, atol=1e-13)
    f.disz(5.0)
    assert np.allclose(f.b, b + 5.0 * A[:, 2], atol=1e-13)


def test_irregula_decenter_is_a_coordbrk_pair(tmp_path):
    """IRREGULA's PARM 1-2 decenter the surface alone: the same as a decenter break, the
    surface, and the break that returns."""
    a = deck([surf(0, disz='INFINITY'), surf(1, disz=10.0, extra='  STOP\n'),
              surf(2, 'IRREGULA', -0.002, 50.0, '  GLAS MIRROR 0 0 1.5 40 0 0 0 0 0 0\n'
                   '  PARM 1 1.5\n  PARM 2 -2.5\n'),
              surf(3)])
    b = deck([surf(0, disz='INFINITY'), surf(1, disz=10.0, extra='  STOP\n'),
              surf(2, 'COORDBRK', extra='  PARM 1 1.5\n  PARM 2 -2.5\n'),
              surf(3, 'STANDARD', -0.002, 0.0, '  GLAS MIRROR 0 0 1.5 40 0 0 0 0 0 0\n'),
              surf(4, 'COORDBRK', disz=50.0, extra='  PARM 1 -1.5\n  PARM 2 2.5\n  PARM 6 1\n'),
              surf(5)])
    ca, cb = convert(write(tmp_path, a, 'a.zmx'))[1], convert(write(tmp_path, b, 'b.zmx'))[1]
    for ra, rb in zip(ca.elts, cb.elts):
        assert np.allclose(ra['vtx'], rb['vtx'], atol=1e-12)
        assert np.allclose(ra['A'], rb['A'], atol=1e-15)


# ======================================================================================
# surfaces and glass, in the engine
# ======================================================================================
@pytest.mark.parametrize('base', ['mm', 'm'])
def test_evenasph_equals_conic_lens(tmp_path, load, engine, base):
    """A K = -1 front face written as a sphere plus its even terms (r^4..r^16) traces as
    the STANDARD paraboloid (curvature > 0: psi along +z, the coefficients' sign kept)."""
    c = 1.0 / 60.0
    a = conic_minus_sphere(c, -1.0)
    std = convert(write(tmp_path, singlet(c, -1.0), 's.zmx'), '--base-units', base)
    asp = convert(write(tmp_path, singlet(c, 0.0, [0.0] + a[1:], 'EVENASPH'), 'a.zmx'),
                  '--base-units', base)
    assert 'Surface=  Aspheric' in asp[0] and 'nAsphCoefs=  7' in asp[0]
    Ls, La = load(std[0], 's.in'), load(asp[0], 'a.in')
    sc = 1.0 if base == 'mm' else 1e-3
    assert La.ok.sum() == Ls.ok.sum() == Ls.nrays
    # 6.5e-8 mm measured: the engine's asphere root solve (its conic path agrees to 1e-14)
    assert np.abs(La.pos_fp - Ls.pos_fp).max() < 2e-7 * sc
    worst, nr = cross_trace(asp[1], La)                      # La is the loaded deck
    assert nr > 300 and worst < 2e-7 * sc, worst
    neg = convert(write(tmp_path, singlet(c, 0.0, [0.0] + [-x for x in a[1:]], 'EVENASPH'),
                        'n.zmx'), '--base-units', base)
    Ln = load(neg[0], 'n.in')
    assert np.abs(Ln.pos_fp - Ls.pos_fp).max() > 1e-2 * sc, 'the sign is visible'


def test_evenasph_equals_conic_mirror(tmp_path, load, engine):
    """A concave mirror (curvature < 0: psi flipped, so the coefficients flip): a sphere
    plus the paraboloid's even terms focuses like the paraboloid."""
    c = -1.0 / 1000.0
    a = conic_minus_sphere(c, -1.0)
    std = convert(write(tmp_path, mirror(c, -1.0), 's.zmx'))
    asp = convert(write(tmp_path, mirror(c, 0.0, [0.0] + a[1:], 'EVENASPH'), 'a.zmx'))
    Ls, La = load(std[0], 's.in'), load(asp[0], 'a.in')
    assert Ls.rms < 1e-12 and La.rms < 1e-9, (Ls.rms, La.rms)   # a paraboloid on axis
    assert np.abs(La.pos_fp - Ls.pos_fp).max() < 2e-7
    sph = load(convert(write(tmp_path, mirror(c, 0.0), 'p.zmx'))[0], 'p.in')
    assert sph.rms > 1e3 * La.rms, 'the sphere alone is visibly worse'


def test_singlet_glass_and_exit_face(tmp_path, load, engine):
    text, cv = convert(write(tmp_path, singlet()))
    assert [r['etype'] for r in cv.elts] == ['Refractor', 'Refractor', 'FocalPlane']
    e1 = text.split('iElt=  1')[1].split('iElt=  2')[0]
    e2 = text.split('iElt=  2')[1].split('iElt=  3')[0]
    assert 'GlassElt=  N-BK7' in e1 and 'GlassElt' not in e2
    n = zm.sellmeier(cv.catalog['n-bk7'][1], 0.5875618)
    assert n == pytest.approx(1.5168, abs=2e-5)                 # nd of N-BK7
    assert f'IndRef=  {n: .16E}' in e1 and 'IndRef=   1.0000000000000000E+00' in e2
    L = load(text)
    assert L.ok.sum() == L.nrays
    worst, nr = cross_trace(cv, L)
    assert nr > 100 and worst < 1e-9, worst


def test_aperture_from_enpd_and_floa(tmp_path):
    _, cv = convert(write(tmp_path, singlet()))
    assert cv.aperture == pytest.approx(20.0)
    t = singlet().replace('ENPD 20', 'FLOA')
    _, cv = convert(write(tmp_path, t, 'f.zmx'))
    assert cv.aperture == pytest.approx(20.0, rel=1e-12)       # the stop is surface 1, r 10
    _, cv = convert(write(tmp_path, singlet(), 'g.zmx'), '--aperture', '12.5')
    assert cv.aperture == 12.5
