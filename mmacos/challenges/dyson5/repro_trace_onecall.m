%REPRO_TRACE_ONECALL  Engine discrepancy: macos.trace(nElt) in ONE call vs element-by-element on the same loaded deck.
%   dyson5 beat 5c (TMS rung R2c, 1.5k module).  The deck is a two-mirror
%   modified Schwarzschild whose entrance pupil (a Reference element, the
%   stop, elt 1) sits only 5.3 mm ahead of the CONVEX primary (elt 2).
%   Collimated source, the chain's aimed chief written as the source,
%   macos.stop(1) first (the Dyson-gate pattern).  Measured 2026-10-03 (TO):
%       one call  trace(5):           rms spot 9.33e-2 m (93 mm), every repeat
%       stepwise  trace(1)..trace(5): rms spot 1.06e-5 m, every ray within
%                                     0.4 um of the exact chain (chain_trace)
%   The chief agrees either way (engine vs chain 1e-13 m).  Hypothesis (NOT
%   proven): the surface after a Reference runs with ifLNsrf (negative L
%   allowed) and ConSrf picks its root by |L^2 - mpr| proximity -- the
%   one-call and stepwise paths reach M1 with different mpr state.  R1a (EP
%   97 mm ahead of M1) shows no difference.
%   Run:  run('<path>/mmacos/mmacos_setup.m'); repro_trace_onecall
here = fileparts(mfilename('fullpath'));
deck = fullfile(here, 'dyson5_t4_r2c_1k5_r2_1500.in');
% the chain's aimed centre-field chief for this deck (collimated, through the EP centre)
p0 = [0; 0; -0.711826];  d0 = [0; 0; 1];
for mode = {'one call', 'stepwise'}
    macos.init(256);  macos.load_rx(deck);  nE = macos.num_elt();
    macos.stop(1);  macos.set_src_fov('src_pos', p0, 'src_dir', d0, 'zSrc', 1e22);  macos.modify();
    if strcmp(mode{1}, 'one call')
        tr = macos.trace(nE);
    else
        for ie = 1:nE, tr = macos.trace(ie); end
    end
    ri = macos.get_ray_info(tr.nRays);  ok = ri.ok_trace(:) & ri.ok_pass(:);  Q = ri.pos(:, ok);
    fprintf('%-9s: %d rays ok of %d, rms spot about the centroid at elt %d = %.4e m\n', mode{1}, nnz(ok), tr.nRays, nE, ...
            sqrt(mean(sum((Q - mean(Q, 2)).^2, 1))));
end
