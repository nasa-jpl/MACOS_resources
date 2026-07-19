%E5_SEG_METOPT  Tier-3 MET-layout optimization on the e5_seg model (v4).
%
% Run AFTER e5_seg.m (loads its saved workspace; GRID='Hex'; needs its
% scratch workdir intact -- same session or a fresh e5_seg run).
% Minimizes the post-control wavefront residual trace(dwdx*P_dx*dwdx')
% over the launcher/fiducial layout, then realizes the winner with
% add_met('launch_pts',...,'pair_map',...) and re-validates it with the
% engine-FD dmet_dx.
%
% v4 (2026-07-19): the search core is now the PRODUCT function
% macos.design.met_layout_opt -- shape-class launcher patterns (one
% pattern per boundary-congruence class, replicated in the segment
% frames; a hex tiling is ONE class, so this reproduces the v3 search
% space).  This script is the thin e5 consumer and the numeric
% regression for the hoist (v3 landed 3.421 nm rms, engine-FD 0.00%).
% 'pattern_frame','radial' preserves the v3 pattern reference (the
% array radial centerline); the shape-class default 'segment' is
% exercised by the e2e pipeline (run_met / s5_met).
%
% Search space and merit are documented in met_layout_opt; the v3
% history (54-DOF merit, MIN_SEP corner gate, spread-vs-cluster
% families) lives in the git log of this file.

EDGE_OFF = 5;                      % launcher clearance off the optical edge, mm
MIN_SEP  = 50;                     % min launcher-launcher separation, mm
R_EXTRA  = 100;                    % aft ("M3") launcher ring radius, mm

here = fileparts(mfilename('fullpath'));
S = load(fullfile(here, 'e5_seg.mat'));
seg = S.seg; nseg = seg.nseg;
hub = nseg + 1;  aft = seg.n_elt - 2;
sige = sqrt(S.Re(1,1)); sigl = sqrt(S.Rl(1,1));

old0 = cd(seg.run.workdir); restore0 = onCleanup(@() cd(old0));
macos.init(512); macos.load_rx(seg.in);

out = macos.design.met_layout_opt(seg, S.dwdx, S.dedx, S.X, ...
    'hub', hub, 'aft', aft, 'r_extra', R_EXTRA, ...
    'sig_edge', sige, 'sig_met', sigl, ...
    'edge_off', EDGE_OFF, 'min_sep', MIN_SEP, ...
    'pattern_frame', 'radial');
best = out.best; r0 = out.r0; w0m = out.w0m; rb = out.rb; wb = out.wb;
fprintf('optimized: rms %.3f nm (baseline %.3f), worst-mode %.3f nm (was %.3f)\n', ...
    rb*1e9, r0*1e9, wb*1e9, w0m*1e9);

%% ---- engine validation of the winner ----------------------------------
am2 = macos.design.add_met(seg.in, seg, 'hub', hub, ...
    'r_fid', best.rfid, 'nf', best.nf, 'fid_clock', best.fclock, ...
    'launch_pts', out.launch_pts, 'pair_map', out.pmap_per_seg, ...
    'extra_sources', aft, 'r_extra', R_EXTRA, ...  % aft Return: no ApVec
    'extra_clock', best.aft_clock, ...             % solved aft block (v4.1)
    'extra_pair_map', mod(best.aft_pmap - 1, best.nf) + 1, ...
    'out_in', fullfile(seg.run.workdir, 'e5_seg_metopt.in'));
macos.load_rx(am2.in); macos.trace();
dm2 = macos.design.dmet_dx([seg.seg_elts, hub, aft]);   % full 54 cols
H = [S.dedx; dm2.dldx];
R = blkdiag(sige^2*eye(size(S.dedx,1)), sigl^2*eye(size(dm2.dldx,1)));
P = S.X - S.X*H'*((H*S.X*H' + R) \ (H*S.X));
G = S.dwdx'*S.dwdx; nw = size(S.dwdx, 1);
rfd = sqrt(trace(P*G)/nw);
fprintf('engine-FD validation of winner: rms %.3f nm (analytic %.3f, %.2f%%)\n', ...
    rfd*1e9, rb*1e9, 100*abs(rfd-rb)/rb);
copyfile(am2.in, fullfile(here, 'e5_seg_metopt.in'));
base = out.base; RFID_GRID = out.rfid_grid;
save(fullfile(here, 'e5_seg_metopt.mat'), 'base', 'best', 'r0', 'w0m', ...
     'rb', 'wb', 'rfd', 'out', 'EDGE_OFF', 'MIN_SEP', 'RFID_GRID');

% MET setup view of the WINNER: optimized launchers filled, baseline
% edge ring as open circles.
fv = macos.design.met_view(seg, am2, 'visible', false, ...
    'overlay_pts', [out.base_launch_pts{:}], 'edge_off', EDGE_OFF, ...
    'title', sprintf(['e5_seg optimized MET layout (%s): %.3f -> %.3f nm rms ' ...
                      '(pmap [%s], rfid=%g, fclock=%.0f deg; open circles = baseline)'], ...
                     best.family(1), r0*1e9, rb*1e9, ...
                     join(string(best.pmap), ' '), best.rfid, ...
                     rad2deg(best.fclock)), ...
    'save', fullfile(here, 'e5_seg_metopt_layout.png'));
close(fv);
fprintf('artifacts: e5_seg_metopt.in / .mat / _layout.png beside the script\n');
