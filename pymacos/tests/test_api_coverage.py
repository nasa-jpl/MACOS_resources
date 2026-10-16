"""The api coverage gate (BRIEF_to_bindings A.4): every macos_api_mod routine has a
pymacos binding, and every generated stub (pymacos/api_gen.py) agrees with the
signature f2py actually built.

The api routine list is read from the Fortran SOURCE with its own regex -- not
from the generator -- so a routine added to macos_api_mod.F90 without re-running
gen_mex_wrappers.py turns this red.  Must-fail leg: test_gate_names_a_removed_binding
(in-suite) and, recorded once, a side-tree .so with one wrapper removed
(pymacos/CLAUDE.md, "Side-tree build").
"""
import inspect
import keyword
import os
import re
from pathlib import Path

import pytest
from context import pymacosf90

import pymacos.api_gen as api_gen

# internal helpers of macos_api_mod (subroutines) -- not entry points
HELPERS = {'translatesurfaceid', 'translateeltid'}


def api_source():
    home = os.environ.get('MACOS_HOME',
                          str(Path.home() / 'dev' / 'macos' / 'macos_f90'))
    p = Path(home) / 'macos_api_mod.F90'
    # fail closed: no source, no gate (a skip here would read as green)
    assert p.is_file(), f'macos_api_mod.F90 not found at {p} (set MACOS_HOME)'
    return p


def api_routines():
    src = api_source().read_text()
    names = re.findall(r'^\s+subroutine\s+(\w+)\s*\(', src,
                       re.MULTILINE | re.IGNORECASE)
    return [n.lower() for n in names if n.lower() not in HELPERS]


def unbound(routines, f2py_names, stub_names):
    """api routines missing from the f2py layer or from the generated stubs."""
    return [n for n in routines if n not in f2py_names or n not in stub_names]


def f2py_names():
    return {n for n in dir(pymacosf90.api) if not n.startswith('_')}


def test_api_source_lists_routines():
    r = api_routines()
    assert len(r) >= 155 and 'ray_status_get' in r and 'load_rx' in r


def test_every_api_routine_is_bound():
    missing = unbound(api_routines(), f2py_names(), set(api_gen.API_ROUTINES))
    assert missing == [], f'api routines with no pymacos binding: {missing}'


def test_gate_names_a_removed_binding():
    r = api_routines()
    f2 = f2py_names() - {'ray_status_get'}
    assert unbound(r, f2, set(api_gen.API_ROUTINES)) == ['ray_status_get']
    st = set(api_gen.API_ROUTINES) - {'xps_cmd'}
    assert unbound(r, f2py_names(), st) == ['xps_cmd']


def _f2py_signature(name):
    """(required, optional, outputs) from the f2py docstring's first line,
    f2py's own `overwrite_*` copy flags dropped."""
    first = getattr(pymacosf90.api, name).__doc__.strip().split('\n')[0]
    m = re.match(r'^(?:(.*?)\s*=\s*)?(\w+)\((.*)\)$', first)
    assert m and m.group(2) == name, first
    outs = tuple(x for x in (m.group(1) or '').split(',') if x)
    inner = m.group(3)
    req_s, _, opt_s = inner.partition('[')
    req = tuple(x for x in req_s.split(',') if x.strip())
    opt = tuple(x for x in opt_s.rstrip(']').split(',')
                if x.strip() and not x.startswith('overwrite_'))
    return req, opt, outs


@pytest.mark.parametrize('name', sorted(api_gen.F2PY_SIGNATURES))
def test_stub_matches_f2py(name):
    assert _f2py_signature(name) == api_gen.F2PY_SIGNATURES[name]
    req, opt, _ = api_gen.F2PY_SIGNATURES[name]
    params = inspect.signature(getattr(api_gen, name)).parameters
    assert tuple(params) == tuple(k + '_' if keyword.iskeyword(k) else k
                                  for k in req + opt)
    assert all(params[k].default is None for k in opt)
