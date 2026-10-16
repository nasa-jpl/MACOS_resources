"""The api routines pymacos gained 2026-10-16 (BRIEF_to_bindings A.3), each on a real
deck, each the twin of its mmacos gate (named per test).  Expectations come from the
deck text or a textbook identity, never from the engine query under test.
Fixtures resolve as mmacos's rx_fixture_path does: pymacos tests/Rx first, then
mmacos/tests/Rx (one corpus, no copies).
"""
from pathlib import Path

import numpy as np
import pytest
from context import pymacos as m

import pymacos.api_gen as api_gen

MODEL = 128
RES = Path(__file__).resolve().parents[2]          # MACOS_resources


def rx(name):
    for root in (Path(__file__).resolve().parent / 'Rx', RES / 'mmacos' / 'tests' / 'Rx'):
        if (root / name).is_file():
            return str(root / name)
    raise FileNotFoundError(name)


@pytest.fixture
def cass():
    m.init(MODEL)
    m.load(rx('Rx_Cass_FarField.in'))


# ---- ray_status (mmacos tRayFailElt) -------------------------------------------------
def test_ray_status_names_the_clipping_element(cass):
    for e in (2, 4, 6):
        _, n, _ = m.trace_rays(e)
        st = m.ray_status(n)
        o = st['fail_elt'][st['status'] == 1]
        assert o.size > 0, 'the central obscuration must clip rays (else vacuous)'
        assert set(o.tolist()) == {1}, f'trace({e}): obscured rays name element 1'
        assert sum(st[k] for k in ('n_ok', 'n_obscured', 'n_miss', 'n_bracket',
                                   'n_max_iter', 'n_undef')) == n


def test_ray_status_agrees_with_ray_info_and_propagation(cass):
    _, n, _ = m.trace_rays(6)
    a = m.ray_status(n)
    _, _, _, ok, ps = m.getRayInfo(n)
    # measured 2026-10-16 on this deck (no misses): OK 11484 = LRayOK & LRayPass,
    # Obscured 970 = LRayOK & ~LRayPass -- a clipped ray keeps its geometric path
    assert np.array_equal(a['status'] == 0, ok & ps)
    assert np.array_equal(a['status'] == 1, ok & ~ps) and a['n_obscured'] > 0
    m.load(rx('Rx_Cass_FarField.in'))
    m.intensity(6)
    b = m.ray_status(n)
    assert np.array_equal(a['status'], b['status']) and np.array_equal(a['fail_elt'], b['fail_elt'])


def test_ray_status_refuses_a_wrong_ray_count(cass):
    _, n, _ = m.trace_rays(6)
    with pytest.raises(api_gen.ApiError):
        m.ray_status(n + 1)


# ---- elt_info / elt_obs / elt_z (mmacos tEltTypeCoverage, tDesignTelescope, tSurfInspect)
def test_elt_info_reads_the_deck(cass):
    i1, i2, i5 = m.elt_info(1), m.elt_info(2), m.elt_info(5)
    assert (i1['type'], i2['type'], i5['type']) == ('Obscuring', 'Reflector', 'Return')
    assert i1['ap_vec'][0] == 2.0 and i2['ap_vec'][0] == 2.1
    assert np.array_equal(i1['x_obs'], [1.0, 0.0, 0.0])
    assert np.array_equal(i2['x_obs'], [-1.0, 0.0, 0.0])
    assert i1['poly'].shape == (2, 0)


def test_elt_obs_reads_the_deck(cass):
    o = m.elt_obs(1)
    assert o['n'] == 3
    assert o['type'].tolist() == [1, 2, 2]         # Circle, Rectangle, Rectangle
    assert np.allclose(o['vec'][:3, 0], [0.5, 0.0, 0.0])
    assert np.allclose(o['vec'][:4, 1], [-3.125e-3, 3.125e-3, -3.0, 3.0])
    assert np.allclose(o['vec'][:4, 2], [-3.0, 3.0, -3.125e-3, 3.125e-3])
    assert m.elt_obs(2)['n'] == 0


def test_elt_z_get_and_set(cass):
    assert m.elt_z(2) == 5.4 and m.elt_z(3) == 1.338854098
    m.elt_z(3, 1.25)
    assert m.elt_z(3) == 1.25


# ---- ray_hist / ray_pos_hist (mmacos ray_hist users) ---------------------------------
def test_ray_hist_matches_the_per_element_trace(cass):
    m.ray_hist('on')
    try:
        _, n, _ = m.trace_rays(6)
        h = m.ray_pos_hist(n)
        assert h['P'].shape == (3, n, m.num_elt() + 1)
        for k in (1, 2, 3):
            m.trace_rays(k)
            pos, _, _, ok, _ = m.getRayInfo(n)
            sel = ok & h['ok'][:, k]
            assert sel.sum() > n // 2
            assert np.allclose(h['P'][:, sel, k], pos[:, sel], rtol=0, atol=1e-12)
        # element 1 is the flat obscuring plane at z = -4.2 (deck VptElt)
        assert np.allclose(h['P'][2, h['ok'][:, 1], 1], -4.2, atol=1e-12)
    finally:
        m.ray_hist('off')


# ---- draw_rays / draw_rays3d (mmacos tDrawFrameGuard) ----------------------------------
def test_draw_rays_vertices_sit_on_the_deck_surfaces(cass):
    m.trace_rays(6)
    b2 = m.draw_rays('YZ')
    b3 = m.draw_rays3d('YZ')
    nde, nray = b2['U'].shape
    assert nray == b2['nray'] == b3['nray'] > 0 and b3['P'].shape == (3, nde, nray)
    assert np.array_equal(b2['elt'], b3['elt']) and np.array_equal(b2['nper'], b3['nper'])
    assert (b2['nper'] >= 1).all() and (b2['nper'] <= nde).all()
    valid = np.arange(nde)[:, None] < b2['nper'][None, :]
    on1 = valid & (b3['elt'] == 1)
    on2 = valid & (b3['elt'] == 2)
    assert on1.any() and on2.any()
    assert np.allclose(b3['P'][2][on1], -4.2, atol=1e-9)          # the obscuring plane
    assert (np.hypot(b3['P'][0][on2], b3['P'][1][on2]) <= 2.1 + 1e-9).all()   # primary ApVec
    with pytest.raises(ValueError):
        m.draw_rays('QQ')


# ---- xps (mmacos pupil_quality / tPupilMap) --------------------------------------------
def test_xps_chief_is_the_fex_vertex(cass):
    m.stop(2)
    _, n, _ = m.trace_rays(6)
    xp = np.asarray(m.fex(1)).ravel()
    vpt = xp[-3:]
    m.xps(5)
    pos, _, _, ok, _ = m.getRayInfo(n)
    assert np.allclose(pos[:, 0], vpt, rtol=0, atol=1e-9), 'ray 1 (chief) = the FEX vertex'
    spread = np.ptp(pos[:2, ok], axis=1)
    assert (spread > 0.1).all(), 'the crossing cloud spans the pupil'
    m.trace_rays(6)


# ---- polarizer / waveplate / elt_jones (mmacos tPolElement) ----------------------------
def test_polelt_round_trip_and_jones_identities():
    m.init(MODEL)
    m.load(rx('Rx_PolElt.in'))
    p = m.polarizer(2)
    w = m.waveplate(3)
    assert np.allclose(p['axis'], [1, 0, 0]) and p['elt_type'] == 15
    assert np.allclose(w['axis'], [1, 0, 0]) and w['retardance'] == 0.25 and w['elt_type'] == 18
    m.waveplate(4, axis=[0, 1, 0], retardance=0.125)
    w4 = m.waveplate(4)
    assert np.allclose(w4['axis'], [0, 1, 0]) and np.isclose(w4['retardance'], 0.125)
    m.polarization('on', Ex=(1.0, 0.0), Ey=(0.0, 0.0))
    try:
        m.trace_rays(m.num_elt())
        # textbook: polarizer diag(1, 0); waveplate diag(1, exp(-i 2 pi r))
        assert np.allclose(m.elt_jones(2), np.diag([1, 0]), atol=1e-12)
        for e, r in ((3, 0.25), (4, 0.125)):
            assert np.allclose(m.elt_jones(e), np.diag([1, np.exp(-2j * np.pi * r)]), atol=1e-12)
    finally:
        m.polarization('off')
    with pytest.raises(ValueError):
        m.polarizer(2, axis=[0, 0, 0])


# ---- met / met_geom (mmacos tMet) --------------------------------------------------------
@pytest.fixture
def mono_met(tmp_path):
    tin = RES / 'segmirmaker' / 'test_in'
    (tmp_path / 'flat.txt').write_text((tin / 'flat.txt').read_text())
    a = np.array([0.0, -5471.177517626807, -21308.82954482988])   # m2 Vpt (mm, global)
    b = a + [100.0, 0.0, 0.0]
    c = np.array([0.0, -6571.126153057798, 3678.032705099662])    # fpa Vpt
    d = c + [0.0, 100.0, 0.0]
    v = lambda p: '  ' + '  '.join(f'{x:.15E}' for x in p)        # noqa: E731
    lines = (tin / 'e5mono.in').read_text().split('\n')
    im2 = next(i for i, l in enumerate(lines) if l.strip() == 'EltName=  m2')
    lines[im2 + 1:im2 + 1] = ['          nMetPos=  2', v(a), v(b),
                              '          tMetElt=  5  2', '  1  0', '  0  1']
    ifp = next(i for i, l in enumerate(lines) if l.strip() == 'EltName=  fpa')
    lines[ifp + 1:ifp + 1] = ['          nMetPos=  2', v(c), v(d)]
    f = tmp_path / 'e5mono_met.in'
    f.write_text('\n'.join(lines))
    m.init(MODEL)
    import os
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        m.load(str(f))
    finally:
        os.chdir(old)
    return a, b, c, d


def test_met_lengths_and_geometry(mono_met):
    a, b, c, d = mono_met
    expect = np.array([np.linalg.norm(a - c), np.linalg.norm(b - d)])
    mn = m.met('native')
    assert mn['n'] == 2 and np.allclose(mn['l'], expect, rtol=1e-12, atol=0)
    assert np.allclose(m.met()['l'], expect * 1e-3, rtol=1e-12, atol=0)    # mm deck -> metres
    g = m.met_geom()
    assert g['n'] == 2
    assert np.allclose(g['src_pts'], np.c_[a, b], atol=1e-9)
    assert np.allclose(g['tgt_pts'], np.c_[c, d], atol=1e-9)
    assert g['src_elt'].tolist() == [2, 2] and g['tgt_elt'].tolist() == [5, 5]


def test_met_los_projection(mono_met):
    a, b, c, d = mono_met
    l0 = m.met('native')['l']
    u1 = (a - c) / np.linalg.norm(a - c)
    dmm = np.array([0.05, -0.02, 0.04])
    m.perturb(5, translation_m=tuple(dmm * 1e-3), in_local_coords=False)
    l1 = m.met('native')['l']
    assert np.isclose(l1[0] - l0[0], -u1 @ dmm, rtol=5e-6, atol=0)


# ---- src_seg (mmacos view_rx tiles) ------------------------------------------------------
def test_src_seg_reads_the_deck():
    m.init(MODEL)
    m.load(rx('e5hex1.in'))
    s = m.src_seg()
    assert s == {'grid_type': 3, 'n_seg': 7, 'width': 2.666666666666667e3, 'gap': 50.0}


# ---- fwd_root_flips (mmacos tFwdRoot) / deadband_notes (tRetraceIdempotent) ------------
def test_fwd_root_flips_counts_the_far_sheet_rays():
    m.init(MODEL)
    m.load(rx('Rx_TwoSheetTMA.in'))
    _, n, _ = m.trace_rays(3)
    k = m.fwd_root_flips()
    assert 0 < k < n / 10
    m.trace_rays(2)
    assert m.fwd_root_flips() == 0, 'no flip through M2'


def _set_dir(d):
    dist, pos, _, _ = m.src_fov()
    m.src_fov(src_pos=pos, src_dir=d, src_dist=dist)


def test_deadband_notes_count_sub_band_changes_only():
    # the three legs of mmacos tRetraceIdempotent's band test: a 4500-ulp source
    # direction change is applied and not noted; 4 ulp is kept out and COUNTED;
    # 1 ulp (round-off) is kept out and silent
    ulp = np.finfo(float).eps
    m.init(MODEL)
    m.load(rx('e5hex1.in'))
    ne = m.num_elt()
    m.trace_rays(ne)
    x0, _, z0 = m.src_csys()
    n0 = m.deadband_notes()
    d1 = z0 + 1e-12 * x0
    _set_dir(d1)
    m.trace_rays(ne)
    x1, _, z1 = m.src_csys()
    assert np.linalg.norm(z1 - d1 / np.linalg.norm(d1)) < 1e-15
    assert m.deadband_notes() == n0, 'a change above the band is not a note'
    _set_dir(z1 + 4 * ulp * x1)
    m.trace_rays(ne)
    x2, _, z2 = m.src_csys()
    assert np.array_equal(z2, z1), 'a 4-ulp change is kept out of the frame'
    n2 = m.deadband_notes()
    assert n2 > n0, 'a suppressed 4-ulp change must be counted'
    _set_dir(z2 + 1 * ulp * x2)
    m.trace_rays(ne)
    assert np.array_equal(m.src_csys()[2], z2)
    assert m.deadband_notes() == n2, 'round-off must not be counted'
