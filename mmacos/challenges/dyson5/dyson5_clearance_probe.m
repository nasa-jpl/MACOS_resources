%DYSON5_CLEARANCE_PROBE  Where the beams cross the other elements' bodies.
%   One-field-point probe (chief at the slit's +6 mm, 1 um) from the engine's
%   ray history: per-element footprints, and for the Offner the leg-1
%   (slit -> M1) and leg-3 (M3 -> FPA) beam positions at the grating's plane.
%   2026-10-01 result (Dave: "serious blockage"): Offner leg 1 crosses
%   z = -250 mm over y [-39.4, +51.4] mm while the grating's footprint spans
%   [-44.6, +44.6] mm -- the grating sits INSIDE the incoming beam; leg 3
%   crosses over [-57.5, +33.3].  Dyson: nothing crosses a body; grating
%   footprint radius 134 mm at z = 691 (0.70 m radius), block face 42 mm at
%   z = 217; slit +6.0 mm vs this wavelength at -11.95 mm on the face.
%   Every element has ApType None, so the sequential trace cannot see any of
%   it: obstruction is a clearance NUMBER (BRIEF_to_dyson5 addendum 6), and
%   spectrometer_clearance is the tool to replace this probe.
here = fileparts(mfilename('fullpath')); run(fullfile(here, '..', '..', 'mmacos_setup.m'));
D = [here filesep];
for c = {'dyson5_s1_offner','dyson5_s3_r3'}
    macos.init(128); macos.load_rx([D c{1} '.in']); n = sscanf(char(regexp(fileread([D c{1} '.in']), 'nElt=\s*(\d+)', 'tokens', 'once')), '%d');
    macos.ray_hist('on'); s = macos.trace(n); h = macos.ray_hist(s.nRays); P = h.P;
    fprintf('\n== %s: %d elts, %d rays, P %s\n', c{1}, n, s.nRays, mat2str(size(P)));
    for k = 1:n
        ok = logical(h.ok(:,k+1)); Q = squeeze(P(:,ok,k+1)); c0 = mean(Q,2); r = max(vecnorm(Q - c0));
        fprintf('  elt %d: centre y %+8.2f z %+8.2f mm, radius %6.1f mm, y span [%+7.1f %+7.1f]\n', k, 1e3*c0(2), 1e3*c0(3), 1e3*r, 1e3*min(Q(2,:)), 1e3*max(Q(2,:)));
    end
    if strcmp(c{1},'dyson5_s1_offner')
        ok = all(logical(h.ok(:,1:6)),2); S0 = squeeze(P(:,ok,1)); A = squeeze(P(:,ok,2));
        t = (-0.25 - S0(3,:))./(A(3,:)-S0(3,:)); X = S0 + (A-S0).*t; G = squeeze(P(:,ok,3));
        fprintf('  LEG 1 slit->M1 at z=-0.25: y [%+6.1f %+6.1f] mm; GRATING footprint y [%+6.1f %+6.1f] mm\n', 1e3*min(X(2,:)), 1e3*max(X(2,:)), 1e3*min(G(2,:)), 1e3*max(G(2,:)));
        B = squeeze(P(:,ok,4)); F = squeeze(P(:,ok,6)); t2 = (-0.25 - B(3,:))./(F(3,:)-B(3,:)); X2 = B + (F-B).*t2;
        fprintf('  LEG 3 M3->FPA at z=-0.25: y [%+6.1f %+6.1f] mm\n', 1e3*min(X2(2,:)), 1e3*max(X2(2,:)));
    else
        ok = all(logical(h.ok(:,1:8)),2); S0 = squeeze(P(:,ok,1)); F = squeeze(P(:,ok,8));
        fprintf('  slit y %+6.2f mm; FPA footprint y [%+6.2f %+6.2f] mm; gap slit->FPA edge %5.2f mm\n', 1e3*mean(S0(2,:)), 1e3*min(F(2,:)), 1e3*max(F(2,:)), 1e3*(mean(S0(2,:)) - max(F(2,:))));
    end
end

