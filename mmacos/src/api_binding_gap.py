#!/usr/bin/env python3
"""
api_binding_gap.py -- measure which macos_api_mod routines each front end binds.

Reads (all from source, no build needed):
  - macos_api_mod.F90         : the api surface (every SUBROUTINE, parsed by
                                gen_mex_wrappers.parse_subroutines -- the
                                same parse the mex codegen uses), minus the
                                internal helpers in PRIVATE_HELPERS
  - mmacos_gen.F / mmacos_mex.F : the mex layer (a routine is bound when a
                                do_<name> helper or a hand-written helper
                                CALLs it)
  - pymacos_f2py.f90          : the f2py layer (a routine is bound when the
                                `NAME_impl => NAME` rename list carries it)
  - macos.py                  : the Python veneer (a routine is reached when
                                macos.py references `api.<f2py name>`)

Prints a Markdown table (stdout); `--names` prints only the missing lists.

  python3 api_binding_gap.py [--api PATH] [--resources PATH]
"""
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_mex_wrappers import parse_subroutines, PRIVATE_HELPERS  # noqa: E402


def api_routines(api_path):
    src = Path(api_path).read_text()
    names = [s[0] for s in parse_subroutines(src)]
    return [n for n in names if n not in PRIVATE_HELPERS]


def f2py_bound(f2py_path):
    src = Path(f2py_path).read_text()
    return {b for _, b in re.findall(r'(\w+)_impl\s*=>\s*(\w+)', src)}


def f2py_defined(f2py_path):
    src = Path(f2py_path).read_text()
    return {m.lower() for m in re.findall(
        r'^\s*subroutine\s+(\w+)\s*\(', src, re.MULTILINE | re.IGNORECASE)}


def veneer_used(macos_py):
    src = Path(macos_py).read_text()
    return {m.lower() for m in re.findall(r'\bapi\.(\w+)', src)}


def mex_bound(gen_f, mex_f, api_names):
    """api routines CALLed from the mex sources."""
    calls = set()
    for p in (gen_f, mex_f):
        src = Path(p).read_text()
        calls |= {m.lower() for m in re.findall(
            r'\bCALL\s+(\w+)\s*\(', src, re.IGNORECASE)}
    return {n for n in api_names if n.lower() in calls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', default=str(
        HERE / '..' / '..' / '..' / 'macos' / 'macos_f90' / 'macos_api_mod.F90'))
    ap.add_argument('--resources', default=str(HERE / '..' / '..'))
    ap.add_argument('--names', action='store_true')
    a = ap.parse_args()
    res = Path(a.resources).resolve()
    f2py = res / 'pymacos' / 'src' / 'cmake' / 'source' / 'pymacos_f2py.f90'
    mpy = res / 'pymacos' / 'src' / 'pymacos' / 'macos.py'
    genf = res / 'mmacos' / 'src' / 'mmacos_gen.F'
    mexf = res / 'mmacos' / 'src' / 'mmacos_mex.F'

    api = api_routines(a.api)
    fb = f2py_bound(f2py)
    fdef = f2py_defined(f2py)
    ven = veneer_used(mpy)
    mex = mex_bound(genf, mexf, api)

    rows = []
    for n in api:
        in_f2py = n in fb and n.lower() in fdef
        rows.append((n, n in mex, in_f2py, in_f2py and n.lower() in ven))
    nf = sum(r[2] for r in rows)
    nv = sum(r[3] for r in rows)
    nm = sum(r[1] for r in rows)
    miss_f2py = [r[0] for r in rows if not r[2]]
    miss_ven = [r[0] for r in rows if r[2] and not r[3]]
    miss_mex = [r[0] for r in rows if not r[1]]

    if a.names:
        print('f2py missing:', ' '.join(miss_f2py))
        print('veneer missing (f2py-bound, not reached from macos.py):',
              ' '.join(miss_ven))
        print('mex missing:', ' '.join(miss_mex))
        return

    print(f'api public routines (from source): {len(api)}  '
          f'(internal helpers excluded: {len(PRIVATE_HELPERS)})\n')
    print(f'| layer | bound | missing |')
    print(f'|---|---|---|')
    print(f'| mmacos mex (do_* helper CALLs it) | {nm} | {len(api) - nm} |')
    print(f'| pymacos f2py (`_impl =>` + wrapper) | {nf} | {len(api) - nf} |')
    print(f'| pymacos veneer (`api.<name>` in macos.py) | {nv} | {len(api) - nv} |')
    print()
    print('| # | api routine | mex | f2py | veneer |')
    print('|---|---|---|---|---|')
    yn = {True: 'yes', False: '**no**'}
    for i, (n, m, f, v) in enumerate(rows, 1):
        print(f'| {i} | `{n}` | {yn[m]} | {yn[f]} | {yn[v]} |')


if __name__ == '__main__':
    main()
