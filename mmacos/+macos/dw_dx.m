function out = dw_dx(session, rx_path, opts)
%MACOS.DW_DX  Single-field dw/dx rigid-body Jacobian driver.
%   out = macos.dw_dx(SESSION, RX_PATH, ...) loads RX_PATH on SESSION
%   (optional -- pass reload_rx=false to skip), builds rigid-body
%   channels for every actual-optic element (and optionally source
%   channels), runs the FD sweep at the current source state, and
%   returns a struct of the result.
%
%   Channel order: source channels first (if included), then per-element
%   rigid-body channels grouped by element (Elt 1 6-DOF, Elt 2 6-DOF,
%   ..., Elt N 6-DOF).  Stack layout the user works with en bloc.
%
%   Name-value pairs:
%     'dofs'             vector of DOF indices (0..5).  Default all 6.
%     'fp_mode'          'track' (default) | 'srs' | 'sxp' | 'none'
%     'ep_elt'           EP element id (default -1 = nElt-1).
%     'include_source'   logical -- prepend a SourceChannel block.
%                        Default false.
%     'src_stop_mode'    'obj' (default) | 'elt' | 'none'.
%     'src_stop_pos'     1x3 object-space stop coords.  Default [0 0 0].
%     'src_stop_elt'     element id for src_stop_mode='elt'.
%     'include_non_optics'  logical -- include Reference / Return in
%                            the per-element block.  Default false.
%     'stop_elt'         set this element as the system Stop before
%                        the sweep.  Default [] (no STOP changed).
%     'stop_obj_pos'     set object-space Stop here (mutex w/ stop_elt).
%                        Default [] (no STOP changed).
%     'rot_output'       'natural' (default) | 'base-per-rad'.
%                        natural: every column is OPD-in-metres per SI
%                        perturbation (translations are dimensionless
%                        ratios; rotations are m/rad).
%                        base-per-rad: rotations are OPD-in-BaseUnits
%                        per rad (not multiplied by CBM).  Translations
%                        unchanged.
%     'delta'            finite-difference step.  Default 1e-8.
%     'method'           'central' (default) | 'forward'.
%     'exit_pupil_elt'   element id at which to evaluate the OPD.
%                        Default -1 = nElt-1 (XP convention).
%     'verbose'          logical.  Default false.
%     'reload_rx'        logical.  Default true.  Pass false from a
%                        multi-field supervisor so per-field set_src_fov
%                        survives.
%
%   Output struct fields:
%     dwdx           Nw × Nz Jacobian (after rot_output rescaling).
%     w_nom_2d       N × N nominal OPD canvas.
%     w_nom_vec      Nw × 1 nominal OPD values at non-zero mask.
%     indx           m2v.m bookkeeping.
%     channel_names  Nz × 1 cell of channel names.
%     iElt, dof_idx  Nz × 1 vectors (iElt = 0 for source channels).
%     kind           Nz × 1 cell of kind labels (Source / RigidBody /
%                    FocalPlane).
%     rx_path / delta / method / wf_elt / rot_output / base_units / cbm
%
%   See also: macos.dwdx_for_current_source, macos.channels.

arguments
    session
    rx_path                  (1,:) char {mustBeNonempty}
    opts.dofs                (:,1) double = (0:5).'
    opts.fp_mode             (1,:) char {mustBeMember( ...
        opts.fp_mode, {'track','srs','sxp','none'})} = 'track'
    opts.ep_elt              (1,1) double {mustBeInteger} = -1
    opts.include_source      (1,1) logical = false
    opts.src_stop_mode       (1,:) char {mustBeMember( ...
        opts.src_stop_mode, {'obj','elt','none'})} = 'obj'
    opts.src_stop_pos        (1,3) double = [0 0 0]
    opts.src_stop_elt        (1,1) double {mustBeInteger} = 0
    opts.include_non_optics  (1,1) logical = false
    opts.stop_elt            double = []
    opts.stop_obj_pos        double = []
    opts.rot_output          (1,:) char {mustBeMember( ...
        opts.rot_output, {'natural','base-per-rad'})} = 'natural'
    opts.delta               (1,1) double = 1e-8
    opts.method              (1,:) char {mustBeMember(opts.method, ...
                                {'central','forward'})} = 'central'
    opts.exit_pupil_elt      (1,1) double {mustBeInteger} = -1
    opts.verbose             (1,1) logical = false
    opts.reload_rx           (1,1) logical = true
end

if ~isempty(opts.stop_elt) && ~isempty(opts.stop_obj_pos)
    error('macos:dw_dx:stop', ...
        'stop_elt and stop_obj_pos are mutually exclusive');
end

if opts.reload_rx
    session.load_rx(rx_path);
end
n_elt = session.num_elt();
if opts.exit_pupil_elt < 0
    wf_elt = n_elt - 1;
else
    wf_elt = opts.exit_pupil_elt;
end

% BaseUnits + CBM lookup for unit rescaling.
cbm = session.cbm();
if cbm == 0
    error('macos:dw_dx:cbm', ...
        'CBM unavailable (Rx not loaded or BaseUnits not declared)');
end

% Apply system Stop if requested.
if ~isempty(opts.stop_elt)
    session.stop(int32(opts.stop_elt));
elseif ~isempty(opts.stop_obj_pos)
    session.stop_obj(opts.stop_obj_pos(1), opts.stop_obj_pos(2), ...
                      opts.stop_obj_pos(3));
end

% Build channels.  Order: source first (if any), then per-element.
channels = {};
if opts.include_source
    channels = [channels; macos.channels.source_channels(session, ...
        'dofs', opts.dofs, ...
        'stop_mode', opts.src_stop_mode, ...
        'stop_obj_pos', opts.src_stop_pos, ...
        'stop_elt', opts.src_stop_elt)];
    % Re-enforce the baseline stop state so the nominal OPD matches
    % the per-channel measurements.
    if strcmp(opts.src_stop_mode, 'obj')
        session.stop_obj(opts.src_stop_pos(1), opts.src_stop_pos(2), ...
                          opts.src_stop_pos(3));
    elseif strcmp(opts.src_stop_mode, 'elt') && opts.src_stop_elt > 0
        session.stop(int32(opts.src_stop_elt));
    end
end
channels = [channels; macos.channels.rigid_body_channels(session, ...
    rx_path, ...
    'dofs', opts.dofs, ...
    'fp_mode', opts.fp_mode, ...
    'ep_elt', opts.ep_elt, ...
    'include_non_optics', opts.include_non_optics)];
if isempty(channels)
    error('macos:dw_dx:nochan', 'no channels found');
end

% Output-scale closure: rotations under 'base-per-rad' keep
% OPD-in-BaseUnits per rad (scale=1); everything else multiplies
% by CBM to convert OPD to metres.
function s = output_scale_fn(ch)
    if strcmp(opts.rot_output, 'base-per-rad') ...
            && isprop(ch, 'dof_idx') && ch.dof_idx <= 2 && ch.dof_idx >= 0
        s = 1;
    else
        s = cbm;
    end
end

wf_func = @() local_wf(session, wf_elt);

[dwdx, w_nom_2d, w_nom_vec, indx, names] = ...
    macos.dwdx_for_current_source(channels, wf_func, opts.delta, ...
        'method', opts.method, ...
        'output_scale_fn', @output_scale_fn, ...
        'verbose', opts.verbose);

iElt_out   = zeros(numel(channels), 1);
dof_out    = zeros(numel(channels), 1);
kind_out   = cell(numel(channels), 1);
for k = 1:numel(channels)
    if isprop(channels{k}, 'iElt')
        iElt_out(k) = channels{k}.iElt;
    else
        iElt_out(k) = 0;   % SourceChannel
    end
    if isprop(channels{k}, 'dof_idx')
        dof_out(k) = channels{k}.dof_idx;
    end
    kind_out{k} = channels{k}.kind();
end

base_units = session.sys_units();
if isfield(base_units, 'base_unit_id')
    bu_id = base_units.base_unit_id;
else
    bu_id = NaN;
end

out = struct();
out.dwdx          = dwdx;
out.w_nom_2d      = w_nom_2d;
out.w_nom_vec     = w_nom_vec;
out.indx          = indx;
out.channel_names = names;
out.iElt          = iElt_out;
out.dof_idx       = dof_out;
out.kind          = kind_out;
out.rx_path       = rx_path;
out.delta         = opts.delta;
out.method        = opts.method;
out.wf_elt        = wf_elt;
out.rot_output    = opts.rot_output;
out.cbm           = cbm;
out.base_units    = base_units;
end


function W = local_wf(session, wf_elt)
session.trace(wf_elt);
W = session.opd();
end
