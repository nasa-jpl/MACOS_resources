"""The veneer defaults the 2026-10-23 audit found drifting (PLAN_CONSOLIDATION 1.4a, the defaults audit
beyond the OPTIONS table of test_defaults); twin of mmacos tVeneerDefaults.

  * srs(slave, master) APPLIES the slave (link=True = the CLI's [YES]); link=False declines.  CoroExample's
    near-field pair as in cli_tests/cmd/ORS_SRS.jou: element 7 at z = -7.0 slaved to 5 comes back to
    z = -6.3293 (the journal's 'New z').  pymacos already defaulted True: this leg pins it (mmacos's
    false default made its srs a no-op since 1.2 (gg)).
  * src_size(ape) keeps the obscuration (pymacos already did; mmacos set 0).
  * setEltSrfZern(iElt, lMon, zernType) with the documented defaults (modes 1:45, zero coefficients)
    succeeds: the literal zernMode=45 was ONE mode and its 45 default coefficients failed the size check
    -> red pre-fix.
"""
from pathlib import Path
import os

import numpy as np
import pytest
from context import pymacos as m
from context import rx_path


def _example(name):
    home = Path(os.environ.get('MACOS_HOME', str(Path.home() / 'dev' / 'macos' / 'macos_f90')))
    return str(home.parent / 'docs' / 'macos-manual' / 'examples' / name)


@pytest.fixture
def coro():
    m.init(128)
    m.load(_example('CoroExample.in'))
    m.elt_z(7, -7.0)
    m.modify()
    assert m.elt_z(7) == -7.0


def test_srs_default_applies_the_slave(coro):
    m.srs(7, 5)
    assert m.elt_z(7) == pytest.approx(-6.3293, abs=5e-5)


def test_srs_link_false_declines(coro):
    m.srs(7, 5, link=False)
    assert m.elt_z(7) == -7.0


def test_src_size_keeps_the_obscuration():
    m.init(128)
    m.load(_example('Cassegrain.in'))
    m.src_size(4.0, 1.0)
    m.src_size(3.5)
    ape, obs = m.src_size()
    assert (float(ape), float(obs)) == (3.5, 1.0)


def test_set_elt_srf_zern_documented_defaults():
    m.init(128)
    m.load(str(rx_path('Rx_CalibZern.in')))
    m.setEltSrfZern(1, 1.0, 1)                       # modes 1:45, zero coefficients
    m.setEltSrfZern(1, 1.0, 1, [4, 11], [1e-7, 2e-8])
