function d = rx_grating_001_data()
%RX_GRATING_001_DATA  Reference data for Grating_example_001.in.
%   Mirrors pymacos/tests/rx_data.py::Rx_Grating_001().
%
%   For a single Rx fixture this transcription is the path of least
%   resistance.  When the test_masks.py port lands (6584 tests, much
%   larger reference set), switch to a Python-side .mat export so the
%   pymacos rx_data.py stays the single source of truth.
d.Rx   = 'Grating_example_001.in';
d.nElt = 4;
% Per-surface gratings: cell array { {srf, refl(bool), order, rule_width, dir(3)}, ... }
d.gratings = { 1, true,  1, 0.001, [1; 0; 0] };
end
