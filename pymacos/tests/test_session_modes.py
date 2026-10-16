"""Session modes survive load, and load says so (Dave 2026-10-11, PLAN_CONSOLIDATION 4 item 2);
twin of mmacos tSessionModes.  POLARIZATION / VECTOR / CENTROID / LNEG / CIR are SESSION modes
(set at the engine's allocation, no longer reset by a load); load() warns (RuntimeWarning) with
the non-default ones, from the same engine source as the CLI's LOAD notice (modes()).
Pre-change .so: no modes_get binding, and a load reset polarization.
"""
import warnings

import pytest
from context import pymacos as m
from context import rx_path

RX = 'Rx_AsphCalib.in'


def test_a_default_load_is_silent():
    m.init(128)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        m.load(str(rx_path(RX)))
    md = m.modes()
    assert md['in_force'] == [] and md['pol'] is False and md['lneg'] is True


def test_polarization_survives_load_and_is_named():
    m.init(128)
    m.load(str(rx_path(RX)))
    m.polarization('on')
    try:
        with pytest.warns(RuntimeWarning, match='session modes in force: POLARIZATION'):
            m.load(str(rx_path(RX)))
        assert m.modes()['pol'] is True
    finally:
        m.polarization('off')
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        m.load(str(rx_path(RX)))
