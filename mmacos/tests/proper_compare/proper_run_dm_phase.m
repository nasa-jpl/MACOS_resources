function [intensity, sampling] = proper_run_dm_phase(geom, wf)
%PROPER_RUN_DM_PHASE  PROPER counterpart: macos's pre-imprint cfield +
%   the same OPD via prop_add_phase, then NF propagate.
%   Mirror of pymacos test_coro_dm_phase.py:_proper_phase.
arguments
    geom (1,1) struct
    wf   (1,1) struct
end

cfield = complex(wf.complex_field_pre);
opd_m  = double(wf.opd_m);
dx_src = wf.dx_at_src_m;
N      = geom.macos_size;
grid_extent = N * dx_src;

bm = prop_begin(grid_extent, geom.wavelength_m, N, 1.0);
bm = prop_multiply(bm, abs(cfield));
% macos cfield convention: positive phase = positive OPD * 2π/λ.
% PROPER's prop_add_phase opposes macos sign — same Phase 1 sign flip
% applies (the nominal-OPD-extracted-from-cfield needs the flip).
opd_nom = -angle(cfield) * geom.wavelength_m / (2 * pi);
bm = prop_add_phase(bm, opd_nom);

% Add the DM phase imprint — the test's whole point.
bm = prop_add_phase(bm, opd_m);

bm = prop_define_entrance(bm);
bm = prop_propagate(bm, geom.propagation_m);
[wfa, sampling] = prop_end(bm);
intensity = wfa;
end
