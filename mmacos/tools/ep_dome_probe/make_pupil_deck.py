#!/usr/bin/env python3
"""make_pupil_deck.py SRC.in DST.in -- insert the add_pupil pair (flat Return at
the image, spherical Return seeded at the last optic) before the terminal
FocalPlane of SRC, renumbering iElt/nElt.  FEX then places the sphere (see
dome_probe.m).  Text-level twin of macos.design.Telescope.add_pupil for a deck
that has no Telescope object."""
import re, math, sys
src, dst = sys.argv[1], sys.argv[2]
t = open(src).read()
def vec(block, key):
    m = re.search(r'^[ \t]*' + key + r'=[ \t]*(.*)$', block, re.M)
    return [float(x) for x in m.group(1).split()[:3]]
blocks = re.split(r'(?=^[ \t]*iElt=[ \t]*\d+[ \t]*$)', t, flags=re.M)
head, el = blocks[0], blocks[1:]
n = len(el); prev, img = el[n-2], el[n-1]
m = re.search(r'^% Output Coordinate', img, re.M)
tail = img[m.start():] if m else ''; img = img[:m.start()] if m else img
Vfp, Vprev = vec(img, 'VptElt'), vec(prev, 'VptElt')
d = [Vfp[i] - Vprev[i] for i in range(3)]; r = math.sqrt(sum(x*x for x in d)); u = [x/r for x in d]
fmt = lambda v: '  '.join(f'{x:.16e}' for x in v)
def mk(i, name, srf, kr, psi, vpt, z):
    return (f"             iElt=   {i}\n          EltName=  {name}\n          Element=  Return\n          Surface=  {srf}\n"
            f"            KrElt=  {kr:.16e}\n            KcElt=  0.0000000000E+00\n           psiElt=  {fmt(psi)}\n"
            f"           VptElt=  {fmt(vpt)}\n           RptElt=  {fmt(vpt)}\n           IndRef=  1.000000E+00\n           Extinc=  0.000000E+00\n"
            f"            nCoat=  0\n             nObs=  0\n           ApType=  None\n         PropType=  Geometric\n             zElt=  {z:.16e}\n          nECoord=  -6\n\n")
b1 = mk(n, 'FP_return', 'Flat', -1e22, [-x for x in u], Vfp, r)
b2 = mk(n+1, 'ExitPupil', 'Conic', -abs(r), u, Vprev, r)
img2 = re.sub(r'iElt=[ \t]*%d' % n, 'iElt=   %d' % (n+2), img, count=1)
head2 = re.sub(r'(nElt=[ \t]*)%d' % n, r'\g<1>%d' % (n+2), head, count=1)
open(dst, 'w').write(head2 + ''.join(el[:n-1]) + b1 + b2 + img2 + tail)
print(f"{dst}: inserted FP_return/ExitPupil before element {n} (seed radius {r:.4f}), nElt {n} -> {n+2}")
