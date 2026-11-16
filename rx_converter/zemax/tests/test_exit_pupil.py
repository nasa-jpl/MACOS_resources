"""--exit-pupil gates (BRIEF_to_zmx_exit_pupil.md).

EP1  the analytic Cassegrain: the EP is the stop's image by the secondary.
EP2  the engine's FEX on the emitted deck finds the same EP vertex, radius and psi
     (the Cassegrain, and the DST deck when present).
EP3  chain-ready: element nElt-1 is a Return with a finite Kr, the deck traces the same rays
     at the FP as without the option, and the on-axis Cassegrain's OPD at the EP is tiny.
EP4  without --exit-pupil the output is byte-identical to the converter before the option.
Each gate carries a must-fail case.
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import DST, ZDIR, convert, need, zm

FIX = Path(__file__).resolve().parent / 'fixtures'
CASS = FIX / 'cassegrain_example.zmx'
CASS_REF = FIX / 'cassegrain_example_b025646.in'     # written by res b025646, before the option

# the analytic Cassegrain (Zemax mm): primary R -1000 at z 0 (the stop), secondary R -240
# at z -400, image at z +200.  The EP is the stop's image by the secondary:
# 1/s' = 2/R2 - 1/s with s = 400 and R2 = -240 (the convex secondary) -> s' = -92.3077
EP_Z = -400.0 + 1.0 / (2.0 / -240.0 - 1.0 / 400.0)   # -492.3077 (virtual, behind the secondary)
EP_R = 200.0 - EP_Z                                  # 692.3077


def _ep_line(text, key):
    m = re.search(rf'%\s+{key}\s+(.*)', text)
    return np.array([float(x) for x in m.group(1).split()[:3]])


def _elements(text):
    """The element blocks: list of dicts keyword -> value text."""
    out = []
    for blk in re.split(r'\n\s+iElt=', text)[1:]:
        d = {}
        for line in blk.splitlines():
            if '=' in line and not line.lstrip().startswith('%'):
                k, v = line.split('=', 1)
                d[k.strip()] = v.strip()
        out.append(d)
    return out


def _fex(engine):
    """The engine's FEX (chief-ray axis): (vertex, psi, radius)."""
    r = np.ravel(engine.fex(1))
    return r[4:7], r[1:4], abs(float(r[0]))


def _emitted(engine, n):
    v = np.ravel(engine.elt_vpt(n - 1))
    p = np.ravel(engine.elt_psi(n - 1))
    return v, p, abs(float(np.ravel(engine.elt_kr(n - 1))[0]))


# --------------------------------------------------------------------------------------
# EP1  the analytic Cassegrain
# --------------------------------------------------------------------------------------
def test_ep1_cassegrain_analytic(capsys):
    _, cv = convert(CASS, '--exit-pupil')
    e = cv.ep
    with capsys.disabled():
        print(f"\n  Cassegrain EP z {e['vtx'][2]:.10f} (analytic {EP_Z:.10f}, "
              f"diff {e['vtx'][2] - EP_Z:.2e}); R {e['R']:.10f} (analytic {EP_R:.10f}); "
              f"skew {e['skew']:.1e}")
    np.testing.assert_allclose(e['vtx'][:2], 0.0, atol=1e-12)
    # measured 1.5e-9 mm at the engine's 5e-6 rad probes: the pupil's own quadratic term,
    # 59.17 mm/rad^2 x dth^2 (below)
    assert abs(e['vtx'][2] - EP_Z) < 1e-8
    assert abs(e['R'] - EP_R) < 1e-8
    assert abs(e['R_vertex'] - EP_R) < 1e-8          # the chief lands on the FP vertex
    assert e['skew'] < 1e-9 and e['spread'] < 1e-9   # rotationally symmetric
    # the brief's 1-arcmin probe is not paraxial: the real ray's crossing moves by
    # 59.17 dth^2 (5.0e-6 mm at 1 arcmin, 5.9e-5 at 1 mrad) -- the reason the option
    # probes at the engine's 5e-6 rad
    d1 = zm.exit_pupil(cv, 2.908882e-4)['vtx'][2] - EP_Z
    d2 = zm.exit_pupil(cv, 1.0e-3)['vtx'][2] - EP_Z
    assert 4e-6 < d1 < 6e-6
    assert abs(d2 / d1 - (1.0e-3 / 2.908882e-4) ** 2) < 0.05


def test_ep1_must_fail_moved_secondary(tmp_path):
    """The gate sees a pupil that is not the stop's image: a secondary 1 mm closer."""
    bad = tmp_path / 'cass_bad.zmx'
    bad.write_text(CASS.read_text().replace('DISZ -400.0', 'DISZ -399.0'))
    _, cv = convert(bad, '--exit-pupil')
    assert abs(cv.ep['vtx'][2] - EP_Z) > 1e-3


# --------------------------------------------------------------------------------------
# EP2  the engine's FEX on the emitted deck
# --------------------------------------------------------------------------------------
def _engine_agrees(load, engine, path, tol, capsys, label, *args):
    text, cv = convert(path, '--exit-pupil', *args)
    L = load(text)
    v0, p0, r0 = _emitted(engine, L.n)
    v1, p1, r1 = _fex(engine)
    dv, dr, dp = np.linalg.norm(v1 - v0), abs(r1 - r0), np.linalg.norm(p1 - p0)
    with capsys.disabled():
        print(f'\n  {label}: EP vertex {np.array2string(v0, precision=9)}, R {r0:.9f}; '
              f'engine FEX - converter: vertex {dv:.2e}, R {dr:.2e}, psi {dp:.1e}')
    assert dv < tol and dr < tol and dp < 1e-12
    return text, cv


def test_ep2_engine_fex_cassegrain(load, engine, capsys):
    _engine_agrees(load, engine, CASS, 1e-9, capsys, 'Cassegrain')     # measured 5.7e-14


def test_ep2_engine_fex_dst(load, engine, capsys):
    need(DST)
    _, cv = _engine_agrees(load, engine, DST, 1e-6, capsys, 'DST')     # measured 1.15e-7
    e = cv.ep
    assert abs(e['tan_sag'][0] - e['tan_sag'][1]) < 1e-4               # T/S 5.2e-5, as FEX
    assert e['skew'] < 1e-5


def test_ep2_must_fail_moved_pupil(load, engine):
    """The cross-check catches an EP the engine would not find: the vertex 1 um off."""
    text, cv = convert(CASS, '--exit-pupil')
    z = f"{cv.ep['vtx'][2]: .16E}"
    zbad = f"{cv.ep['vtx'][2] + 1e-3: .16E}"
    head, sep, tail = text.partition('EltName=  ExitPupil')
    assert z in tail
    L = load(head + sep + tail.replace(z, zbad, 2))
    v0, _, _ = _emitted(engine, L.n)
    v1, _, _ = _fex(engine)
    assert np.linalg.norm(v1 - v0) > 9e-4


# --------------------------------------------------------------------------------------
# EP3  chain-ready
# --------------------------------------------------------------------------------------
def _chain_ready(L, engine):
    """+macos/private/wf_elt_auto.m: nElt-1 is a Return/Reference (3/8) with |Kr| < 1e22."""
    names = L.names
    kr = abs(float(np.ravel(engine.elt_kr(L.n - 1))[0]))
    return names[L.n - 2] == 'ExitPupil' and kr < 1e22


def test_ep3_chain_ready_cassegrain(load, engine, capsys):
    text0, _ = convert(CASS)
    L0 = load(text0, 'plain.in')
    n0, ok0 = L0.nrays, int(L0.passed.sum())
    text, _ = convert(CASS, '--exit-pupil')
    els = _elements(text)
    assert [e['Element'] for e in els[-3:]] == ['Return', 'Return', 'FocalPlane']
    assert els[-2]['Surface'] == 'Conic' and els[-3]['Surface'] == 'Flat'
    L = load(text, 'ep.in')
    assert L.n == L0.n + 2 and _chain_ready(L, engine)
    assert (L.nrays, int(L.passed.sum())) == (n0, ok0)
    rms_ep, _, _ = engine.trace_rays(L.n - 1)
    with capsys.disabled():
        print(f'\n  Cassegrain OPD rms at the EP {float(rms_ep):.2e} mm; rays {n0} / pass {ok0}')
    assert float(rms_ep) < 1e-9                       # measured 6.1e-13 mm


def test_ep3_chain_ready_dst(load, engine, capsys):
    need(DST)
    L0 = load(convert(DST)[0], 'plain.in')
    L = load(convert(DST, '--exit-pupil')[0], 'ep.in')
    assert L.n == L0.n + 2 and _chain_ready(L, engine)
    assert (L.nrays, int(L.passed.sum())) == (L0.nrays, int(L0.passed.sum()))
    rms_ep, _, _ = engine.trace_rays(L.n - 1)
    with capsys.disabled():
        print(f'\n  DST OPD rms at the EP {float(rms_ep):.4e} mm (at the FP {L0.rms:.4e})')
    assert abs(float(rms_ep) - 5.0994e-7) < 0.02 * 5.0994e-7


def test_ep3_must_fail(load, engine):
    """Without the option nElt-1 is no pupil; with the sphere's psi reversed (its centre on
    the wrong side of the pupil) the EP read is no longer tiny."""
    L0 = load(convert(CASS)[0], 'plain.in')
    assert not _chain_ready(L0, engine)
    text, cv = convert(CASS, '--exit-pupil')
    head, sep, tail = text.partition('EltName=  ExitPupil')
    psi = re.search(r'psiElt=\s+(.*)', tail).group(1)
    flipped = '  '.join(f'{-float(x): .16E}' for x in psi.split())
    L = load(head + sep + tail.replace(psi, flipped, 1), 'flip.in')
    rms, _, _ = engine.trace_rays(L.n - 1)
    assert not (float(rms) < 1e-9)


# --------------------------------------------------------------------------------------
# EP4  byte-identical without the option
# --------------------------------------------------------------------------------------
def test_ep4_byte_identical_without_option(tmp_path):
    out = tmp_path / 'cass.in'
    subprocess.run([sys.executable, str(ZDIR / 'zmx2macos.py'), str(CASS), '-o', str(out)],
                   check=True, capture_output=True)
    assert out.read_bytes() == CASS_REF.read_bytes()


def test_ep4_the_option_only_adds(capsys):
    """On every deck at hand the option's output is the plain output plus the head lines,
    the pair and the nElt count: nothing else moves."""
    decks = [CASS] + ([DST] if Path(DST).is_file() else [])
    for d in decks:
        a = convert(d)[0].splitlines()
        b = convert(d, '--exit-pupil')[0].splitlines()
        i = b.index('% the exit-pupil pair (--exit-pupil): a flat Return at the focal plane')
        j = next(k for k in range(i + 1, len(b)) if b[k].startswith('% Zemax surface'))
        pair = b[i - 1:j - 1]                       # the blank line before each block
        b2 = b[:i - 1] + b[j - 1:]
        b2 = [l for l in b2 if not (l.startswith('% Exit pupil') or l.startswith('%   '))]
        nb = [k for k, l in enumerate(b2) if 'nElt=' in l]
        na = [k for k, l in enumerate(a) if 'nElt=' in l]
        assert int(b2[nb[0]].split('=')[1]) == int(a[na[0]].split('=')[1]) + 2
        b2[nb[0]] = a[na[0]]
        # the FocalPlane's iElt moves by 2
        fa = [k for k, l in enumerate(a) if l.lstrip().startswith('iElt=')][-1]
        b2[fa] = a[fa]
        assert b2 == a
        assert sum('Element=  Return' in l for l in pair) == 2


def test_ep4_must_fail():
    """The byte comparison is sensitive: another --grid writes a different deck."""
    text, _ = convert(CASS, '--grid', '33')
    assert text.encode() != CASS_REF.read_bytes()
    assert convert(CASS)[0].encode() == CASS_REF.read_bytes()
