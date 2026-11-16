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


# --- modes_set (Dave 2026-10-28): the two modes that had no setter.
# Pre-change .so: no modes_set binding -- red.

def test_centroid_set_before_load_survives_and_is_named():
    m.init(128)
    m.messages()                       # empty the engine's buffer
    md = m.modes_set(centroid=True)
    try:
        assert md['centroid'] is True and md['in_force'] == ['CENTROID']
        # the command-stack loader matched CENTROID against CENTER (LoadStack,
        # fixed 2026-10-28) and the loop read CENTER's arguments as commands
        msgs = m.messages()
        assert not any('Unknown command' in ln for ln in msgs), msgs
        with pytest.warns(RuntimeWarning, match='CENTROID'):
            m.load(str(rx_path(RX)))
        md = m.modes()
        assert md['centroid'] is True, 'CENTROID survives the load'
        assert md['lneg'] is True, 'an argument left out is left alone'
    finally:
        md = m.modes_set(centroid=False)
    assert md['centroid'] is False and md['in_force'] == []


def test_lneg_set_and_read_back():
    m.init(128)
    m.load(str(rx_path(RX)))
    try:
        md = m.modes_set(lneg=False)
        assert md['lneg'] is False and md['in_force'] == ['NOLNEG']
        assert md['centroid'] is False
        md = m.modes_set(centroid=True, lneg=True)
        assert md['centroid'] is True and md['lneg'] is True
        assert md['in_force'] == ['CENTROID']
    finally:
        md = m.modes_set(centroid=False, lneg=True)
    assert md['in_force'] == []


def test_obs_is_reported_and_reset_by_load():
    # OBS is NOT a session mode (Dave 2026-10-28, PLAN 1.2 (cccc)(2)): modes()
    # reports it in 'obs' and never in 'in_force', and a load resets it to
    # POSITIVE with the engine saying so through the message buffer.  Red on
    # the 7e9ed0a4 engine (OBS kept across the load) and before it (silent).
    m.init(128)
    m.load(str(rx_path(RX)))
    m.obs_set('all')
    try:
        md = m.modes()
        assert md['obs'] == 'all' and md['in_force'] == []
        m.messages()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            m.load(str(rx_path(RX)))
        assert m.obs_set() == 'positive', 'a load resets OBS to POSITIVE'
        assert any('OBS ALL reset to POSITIVE' in s for s in m.messages())
        m.obs_set('negative')
        assert m.modes()['obs'] == 'negative'
        m.messages()
        m.load(str(rx_path(RX)))
        assert m.obs_set() == 'positive'
        assert any('OBS NEGATIVE reset to POSITIVE' in s for s in m.messages())
        m.messages()
        m.load(str(rx_path(RX)))
        assert not any('reset to POSITIVE' in s for s in m.messages())
    finally:
        m.obs_set('positive')
