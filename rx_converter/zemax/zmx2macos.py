#!/usr/bin/env python3
"""zmx2macos -- convert a Zemax OpticStudio sequential file (.zmx) to a MACOS prescription (.in).

Usage:
    python3 zmx2macos.py design.zmx [-o design.in] [--base-units m|mm] [--keep-dummies]
                         [--local-frames] [--grid N] [--aperture A] [--glass NAME=INDEX]
                         [--glass-list FILE]

The geometry is Matt Wette's (A, b) frame propagation from zmx2macos.c (rev 2.2,
2003-11-04, JPL), ported unchanged: A is the rotation from the current Zemax local frame
to the global frame and b its origin.  b moves on every DISZ and on a coordinate break's
decenters; A rotates on a coordinate break's tilts, in the order its flag gives.  The
global origin is surface 1.  psi follows the MACOS convention the C used: KrElt < 0
always, psi toward the centre of curvature.

The file is read as OpticStudio writes it (UTF-16 LE with a BOM, CRLF) or as plain
ASCII / UTF-8.  A surface the MACOS deck cannot represent is refused by name; the deck is
never written when the conversion is refused.  See README.md.
"""

import argparse
import math
import os
import sys

try:
    import numpy as np
except ImportError:
    sys.exit('zmx2macos needs numpy.  Set up once, no root needed (see QUICKSTART.md):\n'
             '    python3 -m venv --without-pip .venv\n'
             '    python3 -m pip --python .venv/bin/python install numpy\n'
             'then run with .venv/bin/python zmx2macos.py ...')

VERSION = '1.0'
INF = 1.0e22
ZINF = 1.0e10            # a Zemax distance at or above this is infinite

UNIT_TO_M = {'MM': 1.0e-3, 'CM': 1.0e-2, 'IN': 2.54e-2, 'METER': 1.0, 'M': 1.0}
BASE_TO_M = {'mm': 1.0e-3, 'm': 1.0}

# surface types the converter writes; every other TYPE is refused by name
TYPES_OK = ('STANDARD', 'COORDBRK', 'IRREGULA', 'EVENASPH')
# surface keys the converter reads; every other key is ignored and listed once
SURF_KEYS = ('TYPE', 'CURV', 'CONI', 'DISZ', 'DIAM', 'GLAS', 'STOP', 'COMM', 'PARM',
             'CLAP', 'OBDC', 'OBSC', 'SQAP', 'FLAP', 'SCBD')
# Zemax glass names that differ from the engine's catalog (macos_glass_list.txt)
GLASS_ALIAS = {'F_SILICA': 'Silica', 'SILICA': 'Silica', 'FUSED_SILICA': 'Silica',
               'FUSEDSILICA': 'Silica', 'CAF2': 'CaF2'}


class ZmxError(Exception):
    """A named refusal: the file cannot be converted as it stands."""


# ======================================================================================
# reading
# ======================================================================================
def read_lines(path):
    """Return (lines, encoding).  OpticStudio writes UTF-16 LE with a BOM; older files and
    hand edits are ASCII.  Detect by BOM, then by NUL bytes."""
    with open(path, 'rb') as f:
        raw = f.read()
    if len(raw) == 0:
        raise ZmxError(f'{path}: the file is empty')
    if raw[:2] == b'\xff\xfe':
        enc = 'utf-16-le'; raw = raw[2:]
    elif raw[:2] == b'\xfe\xff':
        enc = 'utf-16-be'; raw = raw[2:]
    elif raw[:3] == b'\xef\xbb\xbf':
        enc = 'utf-8'; raw = raw[3:]
    elif b'\x00' in raw[:400]:
        head = raw[:400]
        enc = 'utf-16-le' if head[1::2].count(0) > head[0::2].count(0) else 'utf-16-be'
    else:
        enc = 'utf-8'
    try:
        text = raw.decode(enc)
    except UnicodeDecodeError:
        if enc.startswith('utf-16'):
            raise ZmxError(f'{path}: not decodable as {enc} (a truncated UTF-16 file?)')
        text = raw.decode('latin-1'); enc = 'latin-1'
    return text.splitlines(), enc


# ======================================================================================
# parsing
# ======================================================================================
class Surface:
    def __init__(self, num, lineno):
        self.num = num
        self.lineno = lineno
        self.keys = {}          # key -> list of (tokens, raw line, line number)

    def add(self, key, toks, raw, lineno):
        self.keys.setdefault(key, []).append((toks, raw, lineno))

    def has(self, key):
        return key in self.keys

    def toks(self, key):
        v = self.keys.get(key)
        return v[0][0] if v else None

    def raw(self, key):
        v = self.keys.get(key)
        return v[0][1].strip() if v else ''

    def num_(self, key, i=0, default=None):
        t = self.toks(key)
        if t is None or len(t) <= i:
            return default
        return zfloat(t[i])

    @property
    def type(self):
        t = self.toks('TYPE')
        return t[0] if t else None

    def parms(self):
        p = {}
        for toks, _, _ in self.keys.get('PARM', []):
            if len(toks) >= 2:
                p[int(toks[0])] = zfloat(toks[1])
        return p

    @property
    def comm(self):
        v = self.keys.get('COMM')
        return v[0][1].strip()[4:].strip() if v else ''


class Zmx:
    def __init__(self):
        self.header = {}        # key -> list of tokens lists
        self.surfaces = []
        self.terminator = None
        self.ignored = {}       # key -> count


def zfloat(s):
    s = s.strip().strip('"')
    if s.upper() in ('INFINITY', 'INF'):
        return INF
    return float(s)


def parse(lines, name='file'):
    z = Zmx()
    cur = None
    in_surfs = False
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue                     # blank lines (the FIMP blank) never end a block
        if line[0] not in ' \t':
            toks = line.split()
            key = toks[0]
            if key == 'SURF':
                try:
                    num = int(toks[1])
                except (IndexError, ValueError):
                    raise ZmxError(f'{name} line {i}: cannot read the surface number: {line!r}')
                cur = Surface(num, i)
                z.surfaces.append(cur)
                in_surfs = True
                continue
            if in_surfs:
                if z.terminator is None:
                    z.terminator = key   # BLNK (current) or CONF (2003); anything ends it
                cur = None
                z.header.setdefault('_after_' + key, []).append(toks[1:])
                continue
            z.header.setdefault(key, []).append(toks[1:])
            continue
        if cur is None:
            continue                     # indented line outside a surface (header detail)
        toks = line.split()
        key = toks[0]
        if key in SURF_KEYS:
            cur.add(key, toks[1:], line, i)
        else:
            z.ignored[key] = z.ignored.get(key, 0) + 1
    if not z.surfaces:
        raise ZmxError(f'{name}: no SURF block found (not a sequential .zmx file?)')
    if z.terminator is None:
        raise ZmxError(f'{name}: the surface list does not end (no BLNK/CONF/TOL after '
                       f'SURF {z.surfaces[-1].num}): the file is truncated')
    for k, s in enumerate(z.surfaces):
        if s.num != k:
            raise ZmxError(f'{name} line {s.lineno}: surfaces out of order (SURF {s.num} '
                           f'where SURF {k} was expected)')
        if s.type is None:
            raise ZmxError(f'{name}: SURF {s.num} has no TYPE line (a truncated file?)')
    if len(z.surfaces) < 3:
        raise ZmxError(f'{name}: {len(z.surfaces)} surfaces; an object, an optic and an '
                       'image are the least a deck needs')
    return z


# ======================================================================================
# Matt Wette's frame propagation (zmx2macos.c xf_*), ported unchanged
# ======================================================================================
class Frame:
    """A: columns are the local x, y, z axes in the global frame; b: the local origin."""

    def __init__(self):
        self.A = np.eye(3)
        self.b = np.zeros(3)

    def copy(self):
        f = Frame(); f.A = self.A.copy(); f.b = self.b.copy()
        return f

    def rotx(self, rx):
        if rx == 0.0:
            return
        c, s = math.cos(math.radians(rx)), math.sin(math.radians(rx))
        A = self.A
        for r in range(3):
            a2 = c * A[r, 1] + s * A[r, 2]
            a3 = -s * A[r, 1] + c * A[r, 2]
            A[r, 1], A[r, 2] = a2, a3

    def roty(self, ry):
        if ry == 0.0:
            return
        c, s = math.cos(math.radians(ry)), math.sin(math.radians(ry))
        A = self.A
        for r in range(3):
            a1 = c * A[r, 0] - s * A[r, 2]
            a3 = s * A[r, 0] + c * A[r, 2]
            A[r, 0], A[r, 2] = a1, a3

    def rotz(self, rz):
        if rz == 0.0:
            return
        c, s = math.cos(math.radians(rz)), math.sin(math.radians(rz))
        A = self.A
        for r in range(3):
            a1 = c * A[r, 0] + s * A[r, 1]
            a2 = -s * A[r, 0] + c * A[r, 1]
            A[r, 0], A[r, 1] = a1, a2

    def movx(self, dx):
        if dx != 0.0:
            self.b += self.A[:, 0] * dx

    def movy(self, dy):
        if dy != 0.0:
            self.b += self.A[:, 1] * dy

    def disz(self, dz):
        if dz != 0.0:
            self.b += self.A[:, 2] * dz

    def cbrk(self, dx, dy, rx, ry, rz, flag):
        if flag == 0:                    # decenter, then tilt X, Y, Z
            self.movx(dx); self.movy(dy)
            self.rotx(rx); self.roty(ry); self.rotz(rz)
        else:                            # tilt Z, Y, X, then decenter
            self.rotz(rz); self.roty(ry); self.rotx(rx)
            self.movx(dx); self.movy(dy)


# ======================================================================================
# glass
# ======================================================================================
def default_glass_list():
    env = os.environ.get('MACOS_GLASS_LIST')
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    beside = os.path.join(here, 'macos_glass_list.txt')     # a stand-alone package
    if os.path.isfile(beside):
        return beside
    return os.path.normpath(os.path.join(here, '..', '..', '..', 'macos', 'macos_f90',
                                         'macos_glass_list.txt'))


def load_glass_list(path):
    """{lower-case name: (catalog name, [B1 B2 B3 C1 C2 C3])}, Sellmeier, lambda in um."""
    cat = {}
    if not path or not os.path.isfile(path):
        return cat
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) >= 7:
                try:
                    cat[t[0].lower()] = (t[0], [float(x) for x in t[1:7]])
                except ValueError:
                    pass
    return cat


def sellmeier(coef, lam_um):
    B1, B2, B3, C1, C2, C3 = coef
    L2 = lam_um * lam_um
    return math.sqrt(1 + B1 * L2 / (L2 - C1) + B2 * L2 / (L2 - C2) + B3 * L2 / (L2 - C3))


# ======================================================================================
# conversion
# ======================================================================================
class Conv:
    """The converted system: one record per Zemax surface, in Zemax order."""

    def __init__(self):
        self.recs = []
        self.warn = []
        self.notes = []

    def w(self, msg):
        self.warn.append(msg)


def _apertures(s):
    """The surface's own aperture keys, in lens units and the surface frame."""
    ap = {}
    if s.has('CLAP'):
        ap['clap'] = (s.num_('CLAP', 0, 0.0), s.num_('CLAP', 1, 0.0))
    if s.has('SQAP'):
        ap['sqap'] = (s.num_('SQAP', 0, 0.0), s.num_('SQAP', 1, 0.0))
    if s.has('OBSC'):
        ap['obsc'] = (s.num_('OBSC', 0, 0.0), s.num_('OBSC', 1, 0.0))
    if s.has('FLAP'):
        ap['flap'] = s.num_('DIAM', 0, 0.0)
    if s.has('OBDC'):
        ap['obdc'] = (s.num_('OBDC', 0, 0.0), s.num_('OBDC', 1, 0.0))
    return ap


def build(z, opts):
    """Walk the surfaces with the (A, b) frame; return Conv with one record per surface."""
    cv = Conv()
    cat = load_glass_list(opts.glass_list)
    cv.catalog = cat
    cv.unknown_glass = {}            # name -> [surface numbers]: refused after the walk
    cv.user_glass = {}
    for g in opts.glass or []:
        try:
            k, v = g.split('=')
            cv.user_glass[k.strip().upper()] = float(v)
        except ValueError:
            raise ZmxError(f'--glass {g!r}: write NAME=INDEX')

    # ---- header
    h = z.header
    if h.get('MODE') and h['MODE'][0] and h['MODE'][0][0] != 'SEQ':
        raise ZmxError(f"MODE {h['MODE'][0][0]}: only sequential (MODE SEQ) files convert")
    unit = h.get('UNIT', [['MM']])[0][0].upper() if h.get('UNIT') else 'MM'
    if unit not in UNIT_TO_M:
        raise ZmxError(f'UNIT {unit}: lens unit not recognised (MM, CM, IN, METER)')
    base = opts.base_units or ('mm' if unit == 'MM' else 'm')
    s = UNIT_TO_M[unit] / BASE_TO_M[base]           # lens unit -> base unit
    cv.unit, cv.base, cv.scale = unit, base, s
    waves = {}
    for t in h.get('WAVM', []):
        try:
            waves[int(t[0])] = float(t[1])
        except (IndexError, ValueError):
            pass
    pw = int(h['PWAV'][0][0]) if h.get('PWAV') else 1
    if pw not in waves:
        raise ZmxError(f'PWAV {pw}: no WAVM {pw} line, the primary wavelength is unknown')
    cv.lam_um = waves[pw]
    cv.name = ' '.join(h['NAME'][0]) if h.get('NAME') else ''
    cv.vers = ' '.join(h['VERS'][0]) if h.get('VERS') else ''
    fx = h.get('XFLN', [['0']])[0][:1] or ['0']
    fy = h.get('YFLN', [['0']])[0][:1] or ['0']
    if zfloat(fx[0]) != 0.0 or zfloat(fy[0]) != 0.0:
        cv.w(f'field 1 is ({fx[0]}, {fy[0]}), not on axis: the source is the on-axis field')
    nconf = h.get('_after_MNUM')
    if nconf and nconf[0] and int(nconf[0][0]) > 1:
        cv.w(f'MNUM {nconf[0][0]}: a multi-configuration file; the surfaces as stored '
             '(the active configuration) are converted')

    # ---- refusals that need the whole list
    bad = [(q.num, q.type) for q in z.surfaces if q.type not in TYPES_OK]
    if bad:
        raise ZmxError('surface types the MACOS deck cannot represent: ' +
                       ', '.join(f'surface {n} {t}' for n, t in bad))
    for q in z.surfaces:
        p = q.parms()
        if q.type == 'EVENASPH' and p.get(1, 0.0) != 0.0:
            raise ZmxError(f'surface {q.num} EVENASPH: the r^2 term (PARM 1 = {p[1]}) has '
                           'no MACOS equivalent (AsphCoef starts at h^4)')
        if q.type == 'EVENASPH' and any(p.get(k, 0.0) != 0.0 for k in range(9, 20)):
            raise ZmxError(f'surface {q.num} EVENASPH: terms above r^16 are not read')
        if q.type == 'IRREGULA' and any(p.get(k, 0.0) != 0.0 for k in range(3, 9)):
            raise ZmxError(f'surface {q.num} IRREGULA: tilt or aberration terms (PARM 3-8) '
                           'are not representable; only its decenter (PARM 1-2) is')
        if q.has('SCBD'):
            vals = [zfloat(v) for v in q.toks('SCBD')[3:]]
            if any(v != 0.0 for v in vals):
                raise ZmxError(f'surface {q.num}: SCBD with non-zero values is not '
                               f'interpreted: {q.raw("SCBD")!r}')
            cv.notes.append(f'surface {q.num}: SCBD with all values zero, ignored: '
                            f'{q.raw("SCBD")!r}')

    s0 = z.surfaces[0]
    d0 = s0.num_('DISZ', 0, 0.0)
    cv.obj_inf = abs(d0) >= ZINF
    if s0.has('GLAS') and s0.toks('GLAS')[0] != 'MIRROR':
        raise ZmxError(f"surface 0: an object in glass ({s0.toks('GLAS')[0]}) is not "
                       'supported (the source IndRef is a number)')

    # ---- walk (Matt's parse_surface loop): frame at each surface, then its DISZ
    fr = Frame()
    zsign = 1
    medium = None                    # None = air; else (name-or-None, index at lam, coef)
    stop_num = None
    last = len(z.surfaces) - 1
    for q in z.surfaces[1:]:
        r = dict(num=q.num, type=q.type, comm=q.comm, s=q)
        if q.has('STOP'):
            stop_num = q.num
        if q.type == 'COORDBRK':
            p = q.parms()
            fr.cbrk(p.get(1, 0.0), p.get(2, 0.0), p.get(3, 0.0), p.get(4, 0.0),
                    p.get(5, 0.0), int(p.get(6, 0.0)))
            r.update(kind='cb', A=fr.A.copy(), b=fr.b.copy(), n_before=medium, n_after=medium)
            cv.recs.append(r)
            fr.disz(q.num_('DISZ', 0, 0.0))
            continue
        A = fr.A.copy()
        vtx = fr.b.copy()
        p = q.parms()
        if q.type == 'IRREGULA':     # the decenter moves the surface, not the frame
            vtx = vtx + A[:, 0] * p.get(1, 0.0) + A[:, 1] * p.get(2, 0.0)
        curv = q.num_('CURV', 0, 0.0)
        kc = q.num_('CONI', 0, 0.0) if q.has('CONI') else 0.0
        asph = [p.get(k, 0.0) for k in range(1, 9)] if q.type == 'EVENASPH' else [0.0] * 8
        glas = q.toks('GLAS')
        mirror = bool(glas) and glas[0] == 'MIRROR'
        n_before = medium
        if mirror:
            n_after = medium
            zsign = -zsign
        elif glas:
            n_after = _glass(cv, q, glas)
        else:
            n_after = None
        r.update(kind='surf', A=A, b=fr.b.copy(), vtx=vtx, curv=curv, kc=kc, asph=asph,
                 mirror=mirror, n_before=n_before, n_after=n_after, zsign=zsign,
                 stop=q.has('STOP'), semi=q.num_('DIAM', 0, 0.0), ap=_apertures(q),
                 image=(q.num == last))
        cv.recs.append(r)
        medium = n_after
        fr.disz(q.num_('DISZ', 0, 0.0))
    if cv.unknown_glass:
        where = '; '.join(f"{g} at surface{'s' if len(n) > 1 else ''} "
                          f"{', '.join(str(k) for k in n)}"
                          for g, n in cv.unknown_glass.items())
        hint = ' '.join(f'--glass {g}=INDEX' for g in cv.unknown_glass)
        src = (f'not in the engine catalog {opts.glass_list}' if cat else
               f'unchecked: no glass catalog at {opts.glass_list!r} (give --glass-list)')
        raise ZmxError(f'glass {src}: {where}.  The GLAS line\'s nd is filler, not data; '
                       f'give the index with {hint} (INDEX 1 declares a drawing-only '
                       'material air)')
    if stop_num is None:
        stop_num = 1
        cv.w('no STOP line: surface 1 is taken as the stop (the Zemax default)')
    cv.stop_num = stop_num
    cv.obj_pos = np.array([0.0, 0.0, -d0]) if not cv.obj_inf else None
    return cv


def _glass(cv, q, glas):
    """(catalog name or None, index at the primary wavelength, Sellmeier coef or None)."""
    name = glas[0]
    code = int(glas[1]) if len(glas) > 1 and glas[1].lstrip('-').isdigit() else 0
    nd = zfloat(glas[3]) if len(glas) > 3 else 1.0
    if code == 1:
        cv.w(f'surface {q.num}: model glass ({name}, nd {nd}): written as a fixed '
             f'index {nd} (its dispersion is not converted)')
        return (None, nd, None)
    if name.upper() in cv.user_glass:
        n = cv.user_glass[name.upper()]
        cv.notes.append(f'surface {q.num}: glass {name} written as IndRef {n} (--glass)')
        return (None, n, None) if n != 1.0 else None
    key = GLASS_ALIAS.get(name.upper(), name).lower()
    if key in cv.catalog:
        cname, coef = cv.catalog[key]
        if cname.upper() != name.upper():
            cv.notes.append(f'surface {q.num}: glass {name} written as the catalog\'s {cname}')
        return (cname, sellmeier(coef, cv.lam_um), coef)
    # the GLAS line's nd for a catalog glass is filler (1.5 40, as on MIRROR lines): never
    # written; the glass is refused unless --glass gives its index (Dave, 2026-11-03)
    cv.unknown_glass.setdefault(name, []).append(q.num)
    return (None, 1.0, None)


def nval(m):
    return 1.0 if m is None else m[1]


# ======================================================================================
# a real-ray tracer in the Zemax surface frames (aim the chief ray, size the beam)
# ======================================================================================
def _sag(c, k, asph, x, y):
    r2 = x * x + y * y
    a = 1.0 - (1.0 + k) * c * c * r2
    if a < 0.0:
        return None, None, None
    sq = math.sqrt(a)
    z = c * r2 / (1.0 + sq)
    g = c / sq                       # dz/dr divided by r
    for i, al in enumerate(asph, 1):  # al * r^(2i)
        if al != 0.0:
            z += al * r2 ** i
            g += al * 2 * i * r2 ** (i - 1)
    return z, g * x, g * y


def trace(cv, p, d, upto=None):
    """Trace one ray from point p, direction d (global, lens units).  Returns the list of
    (surface number, local hit xyz) for every surface, or None when the ray fails."""
    p = np.array(p, float); d = np.array(d, float); d /= np.linalg.norm(d)
    hits = []
    for r in cv.recs:
        if r['kind'] != 'surf':
            continue
        A = r['A']
        P = A.T @ (p - r['vtx']); D = A.T @ d
        if abs(D[2]) < 1e-14:
            return None
        t = -P[2] / D[2]
        c, k, asph = r['curv'], r['kc'], r['asph']
        ok = False
        for _ in range(60):
            x, y = P[0] + t * D[0], P[1] + t * D[1]
            zz, gx, gy = _sag(c, k, asph, x, y)
            if zz is None:
                return None
            F = P[2] + t * D[2] - zz
            dF = D[2] - (gx * D[0] + gy * D[1])
            if dF == 0.0:
                return None
            dt = F / dF
            t -= dt
            if abs(dt) <= 1e-13 * (1.0 + abs(t)):
                ok = True
                break
        if not ok:
            return None
        H = P + t * D
        _, gx, gy = _sag(c, k, asph, H[0], H[1])
        N = np.array([-gx, -gy, 1.0]); N /= np.linalg.norm(N)
        Dn = D
        if r['mirror']:
            Dn = D - 2.0 * np.dot(D, N) * N
        else:
            n1, n2 = nval(r['n_before']), nval(r['n_after'])
            if n1 != n2:
                if np.dot(D, N) > 0:
                    N = -N
                mu = n1 / n2
                ci = -np.dot(D, N)
                kk = 1.0 - mu * mu * (1.0 - ci * ci)
                if kk < 0.0:
                    return None
                Dn = mu * D + (mu * ci - math.sqrt(kk)) * N
        p = r['vtx'] + A @ H
        d = A @ Dn
        d /= np.linalg.norm(d)
        hits.append((r['num'], H.copy(), p.copy(), d.copy()))
        if upto is not None and r['num'] == upto:
            break
    return hits


def _hit(hits, num):
    for h in hits:
        if h[0] == num:
            return h
    return None


def _src_ray(cv, frame0, u):
    """The ray for aim parameters u = (a, b): a direction (point source) or a start offset
    (collimated) about the object-space z axis."""
    x0, y0, z0 = frame0
    if not cv.obj_inf:
        d = z0 + u[0] * x0 + u[1] * y0
        return cv.obj_pos.copy(), d / np.linalg.norm(d)
    return cv.start + u[0] * x0 + u[1] * y0, z0.copy()


def aim(cv):
    """Aim the chief ray at the stop surface's vertex (2-D Newton on the source)."""
    frame0 = (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))
    if cv.obj_inf:
        r1 = next(r for r in cv.recs if r['kind'] == 'surf')
        L = 2.0 * max(r1['semi'], 1.0)
        cv.start = r1['vtx'] - L * frame0[2]
    stop = cv.stop_num
    u = np.zeros(2)
    hu = 1e-7 if not cv.obj_inf else 1e-6

    def miss(u):
        p, d = _src_ray(cv, frame0, u)
        hs = trace(cv, p, d, upto=stop)
        h = _hit(hs, stop) if hs else None
        return None if h is None else h[1][:2]

    for it in range(30):
        m = miss(u)
        if m is None:
            cv.w('the chief ray could not be aimed at the stop (a trace failed); it leaves '
                 'along the object-space axis and the engine\'s ApStop aims it at load')
            u = np.zeros(2)
            break
        if np.hypot(*m) < 1e-11 * max(1.0, abs(cv.recs[-1]['b']).max()):
            break
        J = np.zeros((2, 2))
        for j in range(2):
            du = np.zeros(2); du[j] = hu
            mj = miss(u + du)
            if mj is None:
                break
            J[:, j] = (mj - m) / hu
        try:
            u = u - np.linalg.solve(J, m)
        except np.linalg.LinAlgError:
            break
    p, d = _src_ray(cv, frame0, u)
    cv.chief_p, cv.chief_d = p, d
    cv.chief_hits = trace(cv, p, d) or []
    cv.aim_iter = it


def beam(cv, opts):
    """Aperture= (base units): a diameter for a collimated beam, a full angle (rad) for a
    point source.  Returns (value, how)."""
    s = cv.scale
    h = cv.header
    if opts.aperture is not None:
        return opts.aperture, 'given on the command line'
    kinds = [k for k in ('FLOA', 'OBNA', 'FNUM', 'ENPD') if h.get(k) is not None]
    if not kinds:
        raise ZmxError('no aperture definition (ENPD, FNUM, OBNA or FLOA) in the header; '
                       'give --aperture')
    if len(kinds) > 1:
        cv.w(f'several aperture keys ({", ".join(kinds)}): {kinds[0]} is used')
    k = kinds[0]
    if k == 'OBNA':
        na = float(h['OBNA'][0][0])
        if cv.obj_inf:
            raise ZmxError('OBNA with an object at infinity: give --aperture')
        return 2.0 * math.asin(na), f'OBNA {na}: full angle 2 asin(NA)'
    if k == 'ENPD':
        epd = float(h['ENPD'][0][0])
        if not cv.obj_inf:
            raise ZmxError(f'ENPD {epd} with a finite object: give --aperture (a full angle)')
        return epd * s, f'ENPD {epd}'
    if k == 'FNUM':
        fno = float(h['FNUM'][0][0])
        if not cv.obj_inf:
            raise ZmxError(f'FNUM {fno} with a finite object: give --aperture (a full angle)')
        efl = _efl(cv)
        return efl / fno * s, f'FNUM {fno}: EPD = EFL {efl:.9g} / F#'
    # FLOA: the stop's semi-diameter fills the beam (a real-ray solve, four azimuths)
    st = next(r for r in cv.recs if r['kind'] == 'surf' and r['num'] == cv.stop_num)
    Rs = st['semi']
    if Rs <= 0.0:
        raise ZmxError(f'FLOA: the stop (surface {cv.stop_num}) has no semi-diameter')
    vals = [_marginal(cv, az, Rs) for az in (0.0, 90.0, 180.0, 270.0)]
    if any(v is None for v in vals):
        raise ZmxError('FLOA: a marginal ray failed to trace to the stop; give --aperture')
    v = float(np.mean(vals))
    spread = (max(vals) - min(vals)) / v
    if cv.obj_inf:
        return 2.0 * v * s, (f'FLOA: stop semi-diameter {Rs:.9g} at surface {cv.stop_num}; '
                              f'real-ray beam radius {v:.12g} (azimuth spread {spread:.1e})')
    return 2.0 * v, (f'FLOA: stop semi-diameter {Rs:.9g} at surface {cv.stop_num}; real-ray '
                     f'marginal half-angle {v:.12g} rad (azimuth spread {spread:.1e})')


def _perp(d):
    x = np.array([1.0, 0, 0]) - d[0] * d
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0, 1.0, 0]) - d[1] * d
    x /= np.linalg.norm(x)
    return x, np.cross(d, x)


def _marginal(cv, az, Rs):
    """Half-angle (point source) or height (collimated) whose ray meets the stop at radius
    Rs from the stop vertex, in azimuth az about the chief ray."""
    d = cv.chief_d
    ex, ey = _perp(d)
    e = math.cos(math.radians(az)) * ex + math.sin(math.radians(az)) * ey

    def rad(v):
        if cv.obj_inf:
            hs = trace(cv, cv.chief_p + v * e, d, upto=cv.stop_num)
        else:
            hs = trace(cv, cv.chief_p, math.cos(v) * d + math.sin(v) * e, upto=cv.stop_num)
        h = _hit(hs, cv.stop_num) if hs else None
        return None if h is None else math.hypot(h[1][0], h[1][1])

    v1 = 1e-5 if not cv.obj_inf else 1e-3 * Rs
    r1 = rad(v1)
    if not r1:
        return None
    v = v1 * Rs / r1
    va, ra = v1, r1
    for _ in range(40):
        rv = rad(v)
        if rv is None:
            return None
        if abs(rv - Rs) < 1e-13 * Rs:
            break
        vn = v + (Rs - rv) * (v - va) / (rv - ra) if rv != ra else v
        va, ra, v = v, rv, vn
    return v


def _efl(cv):
    d = cv.chief_d
    ex, _ = _perp(d)
    hh = 1e-6 * max(1.0, abs(cv.recs[-1]['b']).max())
    a = trace(cv, cv.chief_p, d)
    b = trace(cv, cv.chief_p + hh * ex, d)
    if not a or not b:
        raise ZmxError('FNUM: the focal-length rays failed to trace; give --aperture')
    da, db = a[-1][3], b[-1][3]
    ang = math.atan2(np.linalg.norm(np.cross(da, db)), np.dot(da, db))
    return hh / math.tan(ang)


# ======================================================================================
# emission
# ======================================================================================
def elements(cv, opts):
    """Choose the elements: one per optical surface; coordinate breaks emit nothing."""
    out = []
    dropped = []
    for r in cv.recs:
        if r['kind'] != 'surf':
            continue
        has_ap = any(k in r['ap'] for k in ('clap', 'sqap', 'obsc', 'flap'))
        if r['image']:
            et = 'FocalPlane'
        elif r['mirror']:
            et = 'Reflector'
        elif r['n_after'] is not None or r['n_before'] is not None:
            et = 'Refractor'            # into glass, or the exit face back to air
        elif has_ap or r['stop']:
            et = 'Reference'
        elif opts.keep_dummies:
            et = 'Reference'
        else:
            dropped.append(r['num'])
            continue
        r['etype'] = et
        out.append(r)
    cv.dropped = dropped
    return out


def _fmt(x):
    return f'{x: .16E}'


def _vec(v):
    return '  '.join(_fmt(x) for x in v)


def emit(cv, elts, opts, aperture, how):
    s = cv.scale
    L = []
    w = L.append
    w('% ---------------------------------------------------------------------------------------')
    w(f'% Converted by zmx2macos {VERSION} from {os.path.basename(opts.input)}')
    if cv.name:
        w(f'% Zemax NAME: {cv.name}')
    if cv.vers:
        w(f'% Zemax VERS: {cv.vers}   lens unit {cv.unit} -> BaseUnits {cv.base}')
    w(f'% {len(cv.recs) + 1} Zemax surfaces -> {len(elts)} elements; coordinate breaks emit '
      'nothing; dummies dropped: ' + (' '.join(str(n) for n in cv.dropped) or 'none'))
    w('% Global frame: origin at Zemax surface 1, axes of the object-space frame.')
    w(f'% Aperture: {how}')
    w(f'% Chief ray aimed at the stop (Zemax surface {cv.stop_num}) by the converter; the '
      'stop element carries ApStop= so the engine re-aims at load.')
    for m in cv.notes:
        w(f'% note: {m}')
    for m in cv.warn:
        w(f'% WARNING: {m}')
    w('% ---------------------------------------------------------------------------------------')
    w('')
    w('% ---------------------------------------------------------------------------------------')
    w('% Source Definition')
    w('% ---------------------------------------------------------------------------------------')
    w('')
    d = cv.chief_d
    if cv.obj_inf:
        pos = cv.chief_p * s
        zs = INF
    else:
        eps = 1.0e-6 * (1.0e-3 / BASE_TO_M[cv.base])     # 1 nm in base units
        pos = cv.chief_p * s + eps * d
        zs = -eps
    gx, gy = _perp(d)
    n0 = 1.0
    w(f'        ChfRayDir=  {_vec(d)}')
    w(f'        ChfRayPos=  {_vec(pos)}')
    w(f'          zSource=  {_fmt(zs)}')
    w('')
    w(f'        BaseUnits=  {cv.base}')
    w(f'        WaveUnits=  {cv.base}')
    w('')
    w(f'           IndRef=  {_fmt(n0)}')
    w(f'           Extinc=  {_fmt(0.0)}')
    w(f'          Wavelen=  {_fmt(cv.lam_um * 1e-6 / BASE_TO_M[cv.base])}')
    w(f'             Flux=  {_fmt(1.0)}')
    w('')
    w(f'         Aperture=  {_fmt(aperture)}')
    w(f'         Obscratn=  {_fmt(0.0)}')
    w('')
    w('         GridType=  Circular')
    w(f'         nGridpts=  {opts.grid}')
    w(f'            xGrid=  {_vec(gx)}')
    w(f'            yGrid=  {_vec(gy)}')
    w('')
    w('% ---------------------------------------------------------------------------------------')
    w('% Element Definitions')
    w('% ---------------------------------------------------------------------------------------')
    w('')
    w(f'             nElt=  {len(elts)}')
    hits = {h[0]: h for h in cv.chief_hits}
    for i, r in enumerate(elts, 1):
        _element(cv, r, i, hits.get(r['num']), opts, w)
    w('% ---------------------------------------------------------------------------------------')
    w('% Output Coordinate System Definition')
    w('% ---------------------------------------------------------------------------------------')
    w('')
    w('         nOutCord=  5')
    for i in range(5):
        row = ['0.0E+00'] * 7
        row[i if i < 2 else i + 1] = '1.0E+00'
        w(('             Tout=  ' if i == 0 else '                    ') + '  '.join(row))
    w('')
    return '\n'.join(L) + '\n'


def _element(cv, r, i, hit, opts, w):
    s = cv.scale
    A = r['A']
    c = r['curv']
    flat = (c == 0.0)
    psi = A[:, 2].copy()
    if c < 0.0:
        psi = -psi                       # psi toward the centre of curvature (Kr < 0)
    kr = -INF if flat else -abs(1.0 / c) * s
    asph = r['asph'][1:]                 # r^4 .. r^16 -> AsphCoef h^4 .. h^16
    nas = max([j + 1 for j, a in enumerate(asph) if a != 0.0] or [0])
    if nas and flat:
        raise ZmxError(f"surface {r['num']} EVENASPH on a flat base: write a non-zero "
                       'curvature (MACOS Aspheric needs a conic base)')
    sgn = -1.0 if c < 0.0 else 1.0       # MACOS coefficients are positive along psi
    coefs = [sgn * a * s ** (1 - 2 * (j + 2)) for j, a in enumerate(asph[:nas])]
    vpt = r['vtx'] * s
    name = r['comm'] or f"S{r['num']}"
    if len(name) > 32:
        cv.notes.append(f"surface {r['num']}: name {name!r} cut to 32 characters")
        name = name[:32]
    # IndRef / GlassElt: the medium after a Refractor, the medium a mirror or the image
    # sits in (a Reference is always in air: a glass boundary is a Refractor)
    med = r['n_after'] if r['etype'] == 'Refractor' else r['n_before']
    w('')
    line = f"% Zemax surface {r['num']} ({r['type']})"
    if hit is not None:
        line += f"; chief ray at x {hit[1][0]:.6g}, y {hit[1][1]:.6g} in the surface frame"
    w(line)
    w(f'             iElt=  {i}')
    w(f'          EltName=  {name}')
    w(f"          Element=  {r['etype']}")
    if r['image'] and not flat:
        cv.w(f"surface {r['num']}: the image surface is curved (CURV {c}); written flat")
        flat, kr, nas = True, -INF, 0
    w(f"          Surface=  {'Flat' if flat else ('Aspheric' if nas else 'Conic')}")
    w(f'            KrElt=  {_fmt(kr)}')
    w(f"            KcElt=  {_fmt(0.0 if flat else r['kc'])}")
    if nas:
        w(f'       nAsphCoefs=  {nas}')
        for g in range(0, nas, 4):      # the reader takes four per line (msmacosio.inc)
            w(('        AsphCoefs=  ' if g == 0 else '                    ') +
              '  '.join(_fmt(a) for a in coefs[g:g + 4]))
    w(f'           psiElt=  {_vec(psi)}')
    w(f'           VptElt=  {_vec(vpt)}')
    w(f'           RptElt=  {_vec(vpt)}')
    w(f'           IndRef=  {_fmt(nval(med))}')
    if med is not None and med[0]:
        w(f'         GlassElt=  {med[0]}')
    w(f"           Extinc=  {_fmt(INF if r['etype'] == 'Reflector' else 0.0)}")
    xo = A[:, 0]
    yo = np.cross(psi, xo)
    w(f'             xObs=  {_vec(xo)}')
    _apert(cv, r, hit, xo, yo, w)
    if r['stop']:
        w('           ApStop=  0.0E+00  0.0E+00')
    w('         PropType=  Geometric')
    w(f'             zElt=  {_fmt(INF)}')
    if opts.local_frames:
        yv = np.cross(A[:, 2], A[:, 0])  # Matt's util_yvec: y = cross(A.3, A.1)
        rows = [[A[0, 0], yv[0], A[0, 2]], [A[1, 0], yv[1], A[1, 2]], [A[2, 0], yv[2], A[2, 2]]]
        w('          nECoord=  6')
        z3 = [0.0, 0.0, 0.0]
        for k in range(6):
            vals = rows[k] + z3 if k < 3 else z3 + rows[k - 3]
            w(('             TElt=  ' if k == 0 else '                    ') +
              '  '.join(_fmt(v) for v in vals))
    else:
        w('          nECoord=  -6')


def _apert(cv, r, hit, xo, yo, w):
    """Apertures and obscurations from CLAP / SQAP / FLAP / OBSC (+ OBDC)."""
    s = cv.scale
    ap = r['ap']
    A = r['A']
    obdc = ap.get('obdc', (0.0, 0.0))
    off = obdc != (0.0, 0.0)
    hx, hy = (hit[1][0], hit[1][1]) if hit is not None else (0.0, 0.0)

    def to_obs(x, y):                    # surface-frame (x, y) -> the obscuration frame
        P = A[:, 0] * x + A[:, 1] * y
        return float(P @ xo), float(P @ yo)

    obs = []
    if 'obsc' in ap:
        mn, mx = ap['obsc']
        cx, cy = to_obs(*obdc)
        if mn > 0.0:
            obs.append(('Annulus', [mx * s, cx * s, cy * s, mn * s]))
        else:
            obs.append(('Circle', [mx * s, cx * s, cy * s]))
    aptype, apvec, cmt = 'None', None, None
    if 'clap' in ap or 'flap' in ap:
        mn, mx = ap['clap'] if 'clap' in ap else (0.0, ap['flap'])
        if off:
            cmt = (f'% CLAP {mx:.9g} centred at ({obdc[0]:.9g}, {obdc[1]:.9g}) in the surface '
                   'frame (OBDC): off the vertex, written ApType= None')
        elif math.hypot(hx, hy) > mx:
            cv.w(f"surface {r['num']}: the chief ray meets it at r = {math.hypot(hx, hy):.6g}, "
                 f'outside its CLAP {mx:.6g}: written ApType= None')
            cmt = f'% CLAP {mx:.9g} about the vertex misses the chief ray: written ApType= None'
        else:
            aptype, apvec = 'Circular', [mx * s, 0.0, 0.0]
            if mn > 0.0:
                obs.append(('Circle', [mn * s, 0.0, 0.0]))
    elif 'sqap' in ap:
        xw, yw = ap['sqap']
        if off:
            cmt = (f'% SQAP {xw:.9g} x {yw:.9g} centred at ({obdc[0]:.9g}, {obdc[1]:.9g}) in '
                   'the surface frame (OBDC): off the vertex, written ApType= None')
        elif abs(hx) > xw or abs(hy) > yw:
            cv.w(f"surface {r['num']}: the chief ray falls outside its SQAP: written ApType= None")
            cmt = '% SQAP misses the chief ray: written ApType= None'
        else:
            ex, ey = to_obs(1.0, 0.0)    # the box is symmetric: only its axes matter
            if abs(abs(ex) - 1.0) > 1e-12:
                cv.w(f"surface {r['num']}: SQAP axes do not align with xObs")
            aptype, apvec = 'Rectangular', [-xw * s, xw * s, -yw * s, yw * s]
    if cmt:
        w(cmt)
    w(f'             nObs=  {len(obs)}')
    for t, v in obs:
        w(f'          ObsType=  {t}')
        w(f"           ObsVec=  {'  '.join(_fmt(x) for x in v)}")
    w(f'           ApType=  {aptype}')
    if apvec is not None:
        w(f"            ApVec=  {'  '.join(_fmt(x) for x in apvec)}")


# ======================================================================================
# driver
# ======================================================================================
def convert(path, opts):
    """Return (deck text, Conv).  Raises ZmxError on a named refusal."""
    lines, enc = read_lines(path)
    z = parse(lines, os.path.basename(path))
    opts.input = path
    cv = build(z, opts)
    cv.header = z.header
    cv.encoding = enc
    cv.ignored = z.ignored
    cv.terminator = z.terminator
    if z.ignored:
        cv.notes.append('ignored keys: ' + ', '.join(f'{k} x{n}' for k, n in sorted(z.ignored.items())))
    if cv.vers and '_after_MNUM' not in z.header:
        cv.w('the file ends without its MNUM line: truncated after the surface list? '
             '(the surfaces themselves are complete)')
    if z.terminator not in ('BLNK', 'CONF'):
        cv.notes.append(f'the surface list ends at {z.terminator}')
    aim(cv)
    aperture, how = beam(cv, opts)
    elts = elements(cv, opts)
    if not any(r['etype'] == 'FocalPlane' for r in elts):
        raise ZmxError('no image surface')
    cv.aperture, cv.how, cv.elts = aperture, how, elts
    text = emit(cv, elts, opts, aperture, how)
    return text, cv


def options(argv=None):
    ap = argparse.ArgumentParser(prog='zmx2macos', description=__doc__.split('\n')[0])
    ap.add_argument('input')
    ap.add_argument('-o', '--output')
    ap.add_argument('--base-units', choices=('m', 'mm'))
    ap.add_argument('--keep-dummies', action='store_true',
                    help='write surfaces with no glass, aperture or stop as Reference elements')
    ap.add_argument('--local-frames', action='store_true',
                    help="write each element's 6x6 TElt from its Zemax frame (Matt's -e l)")
    ap.add_argument('--grid', type=int, default=65, help='nGridpts (default 65)')
    ap.add_argument('--aperture', type=float,
                    help='Aperture= in base units (diameter, or full angle in rad for a point source)')
    ap.add_argument('--glass', action='append', metavar='NAME=INDEX',
                    help='a fixed index for a glass the catalog lacks (repeatable); '
                         'INDEX 1 makes it air')
    ap.add_argument('--glass-list', default=default_glass_list(),
                    help='the engine glass catalog (default: ../macos/macos_f90/macos_glass_list.txt)')
    return ap.parse_args(argv)


def main(argv=None):
    opts = options(argv)
    out = opts.output or os.path.splitext(opts.input)[0] + '.in'
    try:
        text, cv = convert(opts.input, opts)
    except ZmxError as e:
        print(f'zmx2macos: REFUSED: {e}', file=sys.stderr)
        return 2
    with open(out, 'w') as f:
        f.write(text)
    for m in cv.notes:
        print(f'note: {m}', file=sys.stderr)
    for m in cv.warn:
        print(f'WARNING: {m}', file=sys.stderr)
    print(f'zmx2macos: {opts.input} ({cv.encoding}) -> {out}: {len(cv.elts)} elements, '
          f'{len(cv.warn)} warnings', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
