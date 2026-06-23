#!/usr/bin/env python3
"""scale_j18.py -- scale the validated j18mono TMA prescription to a target aperture.

A uniform scaling of an optical design preserves all angles, conic constants and
the f/# -- only LENGTHS scale (radii, vertex/reference positions, spacings,
aperture, obscuration, chief-ray position, obscuration vertices).  j18mono.in is
in mm with a 6605 mm aperture; this writes a scaled copy.

Usage:  python3 scale_j18.py <target_D_mm> [src.in] [dst.in]
        (default target 1000 mm = 1 m; default src/dst alongside this file)
"""
import re, sys, os

LENGTH_KEYS = {'KrElt', 'VptElt', 'RptElt', 'zElt', 'Aperture', 'Obscratn',
               'ChfRayPos', 'ObsVec', 'ApVec'}
NUM = re.compile(r'[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?')
J18_D_MM = 6605.0

def scale_rx(src, dst, k):
    def scale_num(m):
        raw = m.group(0)
        v = float(raw.replace('D', 'E').replace('d', 'e'))
        if abs(v) > 1e15:            # sentinels (1e22 flat radius / infinity) -- leave
            return raw
        return f'{v*k:.10E}'
    out = []
    for line in open(src):
        line = line.rstrip('\n')
        m = re.match(r'(\s*)(\w+)=(\s*)(.*?)(\s*)$', line)
        if m and m.group(2) in LENGTH_KEYS:
            pre, key, sp, vals, tail = m.groups()
            out.append(f'{pre}{key}={sp}{NUM.sub(scale_num, vals)}')
        else:
            out.append(line)
    with open(dst, 'w') as f:
        f.write('\n'.join(out) + '\n')

if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    D_mm = float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0
    src = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, 'j18mono.in')
    dst = sys.argv[3] if len(sys.argv) > 3 else os.path.join(here, 'j18_scaled.in')
    k = D_mm / J18_D_MM
    scale_rx(src, dst, k)
    print(f'scaled j18mono ({J18_D_MM:.0f} mm) -> {D_mm:.0f} mm  (k={k:.6f})')
    print(f'wrote {dst}')
