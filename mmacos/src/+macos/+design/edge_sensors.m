function out = edge_sensors(hx_path)
%EDGE_SENSORS  Ingest a SegMirMaker Hx.m edge-sensor model (dedx).
%
%   out = macos.design.edge_sensors(hx_path) loads the MATLAB-loadable
%   edge-sensor measurement matrix SegMirMaker writes alongside the
%   .presc (see macos.design.segment_rx: out.hx) and returns it as the
%   design layer's dedx block of the segmented forward model
%   e = dedx*x + e0   (PLAN_DESIGN_LAYER §6.6 tier-1 backend #1).
%
%   Measurement model (as generated): row 1 = the master segment's
%   absolute piston (trans·zhat of Seg1); every other row = the
%   relative surface-normal displacement of two adjacent segments at
%   their shared-edge sensor point (differential piston + dihedral via
%   moment arms).  Per segment s, columns (s-1)*dof+(1:dof) are
%   [rot·xhat rot·yhat rot·zhat trans·xhat trans·yhat trans·zhat]
%   (6-DOF) or [rot·xhat rot·yhat trans·zhat] (3-DOF), expressed in
%   THAT SEGMENT'S face triad — the same frame segment_rx returns in
%   out.frames and the engine's TElt carries.
%
%   out fields:
%     .dedx        nmeas × nstate sensitivity matrix
%     .nmeas, .nstate, .dof, .nseg
%     .meas_to_seg 2 × nmeas segment pair per row (row 1: [1;1])
%     .dof_names   1 × dof cellstr of the per-segment column meaning
%
%   See also: macos.design.segment_rx, tEdgeSensors.

arguments
    hx_path (1,1) string
end
if ~isfile(hx_path)
    error('macos:design:edge_sensors:file', 'Hx file not found: %s', hx_path);
end

s = load_hx_(hx_path);
if isempty(s.Hx) || isnan(s.nMeas) || isnan(s.nState)
    error('macos:design:edge_sensors:parse', ...
        '%s did not define Hx/nMeas/nState', hx_path);
end

% Row-sparse assignments may leave trailing rows/cols unallocated.
dedx = zeros(s.nMeas, s.nState);
dedx(1:size(s.Hx,1), 1:size(s.Hx,2)) = s.Hx;

m2s = zeros(2, s.nMeas);
m2s(:, 1:size(s.MeasToSeg,2)) = s.MeasToSeg(1:2, :);

nseg = max(m2s, [], 'all');
if nseg <= 0 || mod(s.nState, nseg) ~= 0
    error('macos:design:edge_sensors:shape', ...
        'nState=%d not divisible by the %d segments MeasToSeg references', ...
        s.nState, nseg);
end
dof = s.nState / nseg;
if dof == 6
    dof_names = {'rot_x','rot_y','rot_z','trans_x','trans_y','trans_z'};
elseif dof == 3
    dof_names = {'rot_x','rot_y','trans_z'};
else
    error('macos:design:edge_sensors:dof', 'unexpected DOF/segment: %d', dof);
end

out = struct('dedx', dedx, 'nmeas', s.nMeas, 'nstate', s.nState, ...
             'dof', dof, 'nseg', nseg, 'meas_to_seg', m2s, ...
             'dof_names', {dof_names});
end

function s = load_hx_(hx_path)
% Evaluate the Hx.m assignment script in an isolated workspace.
Hx = []; nMeas = NaN; nState = NaN; MeasToSeg = zeros(2,0); %#ok<NASGU>
run(hx_path);
s = struct('Hx', Hx, 'nMeas', nMeas, 'nState', nState, ...
           'MeasToSeg', MeasToSeg);
end
