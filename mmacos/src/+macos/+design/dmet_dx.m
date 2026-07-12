function out = dmet_dx(elts, opts)
%DMET_DX  FD metrology Jacobian dl/dx over per-element rigid DOFs.
%
%   out = macos.design.dmet_dx(ELTS) pokes each element in ELTS through
%   its six rigid-body DOFs and finite-differences the engine metrology
%   readings (macos.met): the dldx block of the segmented forward model
%   l = dldx*x + l0.  Requires a loaded Rx carrying met blocks (see
%   macos.design.add_met) — met points ride their element under the
%   programmatic perturb path, so pokes register directly.
%
%   DOF convention per element (COLUMN ORDER, matching the SegMirMaker
%   Hx/dedx convention): [rot_x rot_y rot_z trans_x trans_y trans_z] in
%   the element's LOCAL (TElt) frame — for segments from segment_rx
%   that is the segment face triad.  x units are SI (rad, metres);
%   l in SI metres.  Forward difference off the unperturbed baseline
%   (the model is piecewise-smooth straight-line lengths).
%
%   opts: dstep_rot (1e-6 rad), dstep_trans (1e-6 m), frame ('local') --
%   steps sized so eps*l/h stays ~1e-9 on 10-m-class gauges.
%   out: .dldx (nmeas x 6*numel(elts)), .l0 (SI), .nmeas, .elts,
%        .dof_names.
%
%   See also: macos.design.add_met, macos.design.edge_sensors, macos.met.

arguments
    elts (1,:) double {mustBeInteger, mustBePositive}
    opts.dstep_rot (1,1) double {mustBePositive} = 1e-6
    opts.dstep_trans (1,1) double {mustBePositive} = 1e-6
    opts.frame (1,:) char {mustBeMember(opts.frame, {'local','global'})} = 'local'
end

m0 = macos.met();
if m0.n == 0
    error('macos:design:dmet_dx:nomet', ...
        'loaded Rx declares no metrology (run add_met first)');
end
l0 = m0.l;
nE = numel(elts);
dldx = zeros(m0.n, 6*nE);

for k = 1:nE
    for d = 1:6
        th = zeros(3,1); del = zeros(3,1);
        if d <= 3
            th(d) = opts.dstep_rot;    h = opts.dstep_rot;
        else
            del(d-3) = opts.dstep_trans; h = opts.dstep_trans;
        end
        macos.perturb(elts(k), 'rotation', th, 'translation', del, ...
                      'frame', opts.frame);
        lp = macos.met().l;
        macos.perturb(elts(k), 'rotation', -th, 'translation', -del, ...
                      'frame', opts.frame);
        dldx(:, (k-1)*6 + d) = (lp - l0) / h;
    end
end

out = struct('dldx', dldx, 'l0', l0, 'nmeas', m0.n, 'elts', elts, ...
    'dof_names', {{'rot_x','rot_y','rot_z','trans_x','trans_y','trans_z'}});
end
