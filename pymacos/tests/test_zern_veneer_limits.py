"""pymacos Zernike veneer limits = the engine's (PLAN_CONSOLIDATION 1.2 (ww), BRIEF_to_zernike_repairs.md).

setEltSrfZern / setEltSrfZernMode / ZernikeData carried literal limits (ZernType <= 9, modes <= 45) that
predated Noll (10), ExtFringe (11) and the 66-mode set, so the veneer refused what the api takes.  They now
check against api_gen.MZERNTYPE / MZERNMODES, which the generator reads from elt_mod.F.  This gate reads
elt_mod.F itself (not api_gen) for the expected values, sets types 10 and 11 and mode 66 through the veneer
and reads them back from the engine; must-fail legs: type mZernType + 1 and mode mZernModes + 1 are refused.
"""
import os
import re
from pathlib import Path

import numpy as np
import pytest

from context import pymacos as m
from pymacos import api_gen
from test_zernike_orderings import DECK

ELT_MOD = Path(os.environ.get('MACOS_HOME', Path.home() / 'dev/macos/macos_f90')) / 'elt_mod.F'


def engine_param(name):
    hits = set(re.findall(r'\b' + name + r'\s*=\s*(\d+)\b', ELT_MOD.read_text(), re.IGNORECASE))
    assert len(hits) == 1, (name, hits)
    return int(hits.pop())


@pytest.fixture(scope='module')
def zdeck(tmp_path_factory):
    p = tmp_path_factory.mktemp('zlim') / 'z.in'
    p.write_text(DECK.format(ztype='ANSI', n=66))
    m.init(128)
    m.load(str(p))
    return p


def test_constants_are_the_engine_parameters():
    assert api_gen.MZERNTYPE == engine_param('mZernType')
    assert api_gen.MZERNMODES == engine_param('mZernModes')


@pytest.mark.parametrize('ztype, mode', [(10, 66), (11, 46)])
def test_set_type_and_top_mode_through_the_veneer(zdeck, ztype, mode):
    m.load(str(zdeck))
    modes = np.arange(1, api_gen.MZERNMODES + 1)
    coefs = np.zeros(modes.size)
    coefs[mode - 1] = 1e-4
    m.setEltSrfZern(1, 1.0, ztype, modes, coefs)
    assert m.elt_zrn_type(1) == ztype
    assert m.getEltSrfZernMode(1, [mode])[0] == pytest.approx(1e-4, rel=0, abs=1e-18)
    assert m.getEltSrfZernMode(1, [1])[0] == 0.0


def test_must_fail_beyond_the_engine_limits(zdeck):
    m.load(str(zdeck))
    with pytest.raises(ValueError, match='ZernType'):
        m.setEltSrfZern(1, 1.0, api_gen.MZERNTYPE + 1, [1], [0.0])
    with pytest.raises(ValueError, match='ZernMode'):
        m.setEltSrfZern(1, 1.0, 1, [api_gen.MZERNMODES + 1], [0.0])
    with pytest.raises(ValueError, match='ZernMode'):
        m.setEltSrfZernMode(1, [api_gen.MZERNMODES + 1], [0.0])
    assert m.elt_zrn_type(1) == 1                       # the refused calls left the surface alone
