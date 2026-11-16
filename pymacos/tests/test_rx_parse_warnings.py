"""Two silent drops in the Opt* parse now SAY so (macos PLAN_CONSOLIDATION 1.1b, 2026-10-12) -- the
binding surface of the shared parser (msmacosio.inc; the CLI twins are cli_tests/cmd/
Rx_header_unknown.jou and OptAsph_order.jou).  The engine writes these lines to the process
stdout from Fortran, so they are read here at the file-descriptor level (capfd).

(a) every header match is the FULL key name: OptFooKey= was captured by the 5-character OptFOVWt
    match and read as FOV weights; now it is an unknown header key and named.
(b) an OptAsph= above its element's VarElt= was dropped silently; now one line names the element.
Both decks still LOAD (warn, never refuse).  Pre-fix .so: neither line -- red.

The lines are read from the engine's message buffer (pymacos messages(), PLAN_CONSOLIDATION 1.2 (ee)):
the Fortran stdout this test read through capfd reaches no Python-visible stream on macOS, so it was
skipped there (CCMac's Mac gates 2026-10-19); the buffer carries the same lines on every platform.
"""
from context import pymacos as m
from context import rx_path


def _load_out(name):
    m.init(128)
    m.messages()
    m.load(str(rx_path(name)))
    return '\n'.join(m.messages())


def test_a_look_alike_header_key_is_named():
    out = _load_out('Rx_HdrUnknown.in')     # = macos ZGD tst_hdr_unknown.in
    assert 'Unidentified header string = OptFooKey' in out


def test_optasph_above_varelt_is_named():
    out = _load_out('Rx_OptAsphOrder.in')   # = macos ZGD tst_optasph_order.in
    assert 'OptAsph= at element   1 IGNORED: OptAsph= must follow VarElt=' in out
