function out = edge_sensors(hx_path)
%EDGE_SENSORS  Ingest a SegMirMaker Hx.m edge-sensor model (dedx).
%
%   out = macos.design.edge_sensors(hx_path) loads the MATLAB-loadable
%   edge-sensor measurement matrix SegMirMaker writes alongside the
%   .presc (see macos.design.segment_rx: out.hx) and returns it as the
%   design layer's dedx block of the segmented forward model
%   e = dedx*x + e0   (PLAN_DESIGN_LAYER §6.6 tier-1 backend #1).
%
%   Measurement model (SegMirMaker 2026-07-19, Dave's spec): per
%   SHARED EDGE, TWO sensor locations offset +/-SensorOff from the
%   edge midpoint along the edge direction, THREE axes per location --
%   axis 1 = surface-normal (differential piston + dihedral), axes
%   2/3 = the in-plane pair (radhat/tanhat: gap and shear by edge
%   orientation).  All rows are DIFFERENTIAL (relative motion of the
%   two adjacent segments at the sensor point); there is NO
%   absolute-piston anchor row -- edge sensors have no absolute
%   reference (global piston is unobservable from edges alone).
%   Legacy single-axis Hx files (with the master-piston row 1,
%   MeasToSeg(:,1) = [1;1]) are still ingested; out.axis/loc are
%   zeros for them.
%
%   Per segment s, columns (s-1)*dof+(1:dof) are
%   [rot·xhat rot·yhat rot·zhat trans·xhat trans·yhat trans·zhat]
%   (6-DOF) or [rot·xhat rot·yhat trans·zhat] (3-DOF), expressed in
%   THAT SEGMENT'S face triad — the same frame segment_rx returns in
%   out.frames and the engine's TElt carries.
%
%   out fields:
%     .dedx        nmeas × nstate sensitivity matrix
%     .nmeas, .nstate, .dof, .nseg
%     .meas_to_seg 2 × nmeas segment pair per row
%     .axis        1 × nmeas: 1=piston(normal) 2=gap(in-plane, perp
%                  to the edge) 3=shear(in-plane, along the edge);
%                  0 for legacy files
%     .loc         1 × nmeas: sensor location 1|2 on the edge (0 legacy)
%     .sensor_pos  3 × nmeas world sensor positions (NaN legacy)
%     .has_anchor  true when a legacy absolute-piston row is present
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

axis_ = zeros(1, s.nMeas);
loc_  = zeros(1, s.nMeas);
spos  = nan(3, s.nMeas);
if ~isempty(s.MeasAxis)
    axis_(1:numel(s.MeasAxis)) = s.MeasAxis;
    loc_(1:numel(s.MeasLoc))   = s.MeasLoc;
end
if ~isempty(s.SensorPos)
    spos(:, 1:size(s.SensorPos,2)) = s.SensorPos;
end
has_anchor = (m2s(1,1) == m2s(2,1)) && m2s(1,1) > 0 && all(axis_ == 0);

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
             'axis', axis_, 'loc', loc_, 'sensor_pos', spos, ...
             'has_anchor', has_anchor, 'dof_names', {dof_names});
end

function s = load_hx_(hx_path)
% Evaluate the Hx.m assignment script in an isolated workspace.
Hx = []; nMeas = NaN; nState = NaN; MeasToSeg = zeros(2,0); %#ok<NASGU>
MeasAxis = []; MeasLoc = []; SensorPos = []; %#ok<NASGU>
run(hx_path);
s = struct('Hx', Hx, 'nMeas', nMeas, 'nState', nState, ...
           'MeasToSeg', MeasToSeg, 'MeasAxis', MeasAxis, ...
           'MeasLoc', MeasLoc, 'SensorPos', SensorPos);
end
