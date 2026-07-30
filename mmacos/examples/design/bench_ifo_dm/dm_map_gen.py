#!/usr/bin/env python
"""Generate a DM surface map with the PROPER DM model -> MACOS GridFile.

Usage: dm_map_gen.py OUT N DX_MM NACT SPACING_MM POKE_NM SEED

Random actuator commands, |cmd| < POKE_NM nanometres, are pushed through
proper.prop_dm's influence-function model; the resulting surface (metres,
N x N at DX_MM sampling) is written in mm in the MACOS GridFile layout
(fixed-width %16.8E, N values per line -- the same format
macos.write_grid_file emits, so macos.read_grid_file returns it in the
engine's GridMat convention).  Run it with the pymacos venv python
(which has PROPER installed).
"""
import sys
import numpy as np
# PROPER 3.2.1 still uses the np.int/np.float aliases removed in numpy 1.24
for _a, _t in (('int', int), ('float', float), ('bool', bool)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)
import proper

out, n, dx_mm, nact, spacing_mm, poke_nm, seed = sys.argv[1:8]
pattern = sys.argv[8] if len(sys.argv) > 8 else 'random'
n, nact = int(n), int(nact)
dx = float(dx_mm) * 1e-3                       # m
wf = proper.prop_begin(n * dx, 632.8e-9, n, 1.0)
if pattern == 'checker':
    # alternating up/down actuators, staggered by rows: cmd = (-1)^(i+j)
    i, j = np.indices((nact, nact))
    cmd = ((-1.0) ** (i + j)) * float(poke_nm) * 1e-9
else:
    rng = np.random.default_rng(int(seed))
    cmd = rng.uniform(-1.0, 1.0, (nact, nact)) * float(poke_nm) * 1e-9
dmap = proper.prop_dm(wf, cmd, nact / 2 - 0.5, nact / 2 - 0.5,
                      float(spacing_mm) * 1e-3)
np.savetxt(out, dmap * 1e3, fmt='%16.8E', delimiter='')
print('dm_map_gen: %dx%d map, surface p-v %.1f nm, rms %.1f nm'
      % (dmap.shape[0], dmap.shape[1],
         1e9 * (dmap.max() - dmap.min()), 1e9 * dmap.std()))
