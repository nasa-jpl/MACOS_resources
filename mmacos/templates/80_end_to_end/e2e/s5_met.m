%S5_MET  Stage 5: edge sensors + laser MET on the segmented system.
%
% Stage 5 of the end-to-end worked example -- a THIN DRIVER over the
% general MET stage runner (design/runners/run_met.m; Dave 2026-07-19:
% the product is the reusable stage runners, examples drive them).
% run_met takes the stage-3 segmented .in + its SegMirMaker Hx sidecar
% and the stage-4 Jacobian .mat, and emits:
%
%   e2e_<v>_met.in       as-built Stewart truss: 6 launchers per
%                        segment on its TRUE boundary (wedge/hexagon
%                        polygons, 5 mm clearance), 6 fiducials on the
%                        M2 rim (25 mm inset), aft ring on the FOLD
%   e2e_<v>_metopt.in    the met_layout_opt winner -- ONE launcher
%                        pattern per segment SHAPE CLASS (pie: center
%                        hexagon + wedge), solved in the segment
%                        frames, replicated to same-shape segments
%   e2e_<v>_met.mat      dedx, dldx (engine-FD == analytic gated),
%                        merit table, and the estimator products
%                        dxde / dxdl / dwde / dwdl for stages 5.5/6
%   e2e_<v>_met_report.txt + layout / metric / view figures
%
% GEOMETRY NOTE (the e2e twist vs the e5 fixture): the aft MET leg
% must SEE the M2 fiducials.  M3 sits 2.1 m off-axis on the bench --
% a ring there crosses the M1 plane at ~2 m radius, nowhere near the
% hole.  The on-axis bench-side body is the FOLD (elt 9, VptElt
% (0,0,+0.3)); per Dave (2026-07-19) the stable aft structure extends
% to the SM RADIUS (0.232 m), so the ring sits at that radius and the
% clearance check uses the SM-shadow radius as the effective hole (the
% real hole will be enlarged to match -- queued P.tel change).  Sensed
% bodies = [7 segments, M2 (hub), FM (aft)] = 54 DOFs.  Headline
% metrics are dimensionless WEMs, full / tilt / non-tilt (tilt control
% absorbs tilt); the truss layout is optimized WITHOUT the edge
% sensors and Dave's manual corner-pairs layout is scored alongside.
%
% Run AFTER s3_segmentation.m and s4_jacobians.m.

addpath(fullfile(getenv('HOME'), 'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'), 'dev/MACOS_resources/mmacos/design/src'));
addpath(fullfile(getenv('HOME'), 'dev/MACOS_resources/mmacos/design/runners'));
P = e2e_params();
here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
v = char(P.seg.variant);
rx  = fullfile(here, sprintf('e2e_%s.in', v));
hx  = fullfile(here, sprintf('e2e_%sHx.m', v));
jac = fullfile(here, 's4_jacobians.mat');
assert(isfile(rx) && isfile(hx), 's5 needs e2e_%s.in + Hx -- run s3 first', v);
assert(isfile(jac), 's5 needs s4_jacobians.mat -- run s4 first');

% body bookkeeping (read order): segments lead, then M2, then the fold
txt = fileread(rx);
nseg = numel(regexp(txt, 'Element=\s*Segment', 'match'));
hub = nseg + 1;                       % M2 -- carries the fiducials
aft = nseg + 2;                       % FM (fold) -- aft ring, hole LOS

art = run_met(rx, 'hx', hx, 'jac', jac, 'hub', hub, 'aft', aft, ...
    'r_extra',   P.met.aft_ring_r_mm  * 1e-3, ...   % BaseUnits = metres
    'hole_r',    P.met.hole_r_mm      * 1e-3, ...   % SM shadow radius
    'fid_inset', P.met.fid_rim_inset_mm * 1e-3, ...
    'edge_off',  P.met.edge_off_mm    * 1e-3, ...
    'min_sep',   P.met.min_sep_mm     * 1e-3, ...
    'nf', P.met.nf, ...
    'sig_rot', P.met.sig_rot, 'sig_trans', P.met.sig_trans, ...
    'sig_edge', P.met.sig_edge, 'sig_met', P.met.sig_met, ...
    'mc', P.met.mc_draws, 'model_size', P.seg.model_size, ...
    'opt', struct('nf_grid', 6));   % Dave 2026-07-19: 6 fiducials ->
                                    % segments ~interchangeable (with the
                                    % rotational assignment, every wedge's
                                    % truss is congruent); nf=3 scored
                                    % marginally better but breaks the
                                    % 60-deg symmetry

% stage-report convention: s5_report.txt beside the other stages
copyfile(art.report, fullfile(here, 's5_report.txt'));
% e2e_pie_met.mat is >20 MB and gitignored -- write the committed
% fingerprint sidecar (jac_fingerprint) alongside it: the estimator
% products + generation stamps, auditable without the blob.
jac_fingerprint('write', fullfile(here, sprintf('e2e_%s_met.fp.json', v)), ...
    struct('dedx', art.dedx, 'dldx', art.dldx, 'dldx_opt', art.dldx_opt, ...
           'dxde', art.dxde, 'dxdl', art.dxdl), ...
    struct('product', sprintf('e2e_%s_met', v), 'layout', 'held-or-optimized', ...
           'rx', sprintf('e2e_%s.in', v)));
% promote the optimized layout to the shared preset library: the new
% AS-BUILT for future same-tiling builds (Dave 2026-07-19); consume it
% via run_met(..., 'preset', <file>)
if isfield(art, 'preset_file') && isfile(art.preset_file)
    pdir = fullfile(fileparts(which('run_met')), 'presets');
    if ~exist(pdir, 'dir'), mkdir(pdir); end
    copyfile(art.preset_file, fullfile(pdir, sprintf('%s_met.mat', v)));
    fprintf('preset promoted: design/runners/presets/%s_met.mat\n', v);
end
fprintf('\nStage 5 complete: %s (+ report copied to s5_report.txt)\n', art.mat);
fprintf('Next: run_compare (linear model vs mmacos), then s6 simulator.\n');
