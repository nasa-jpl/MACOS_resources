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
%     'elts'             vector of element IDs to include.  Default []
%                        (auto-detect all actual optics from Rx).
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
%                        HISTORICAL NO-OP since 2026-08-25 (Dave): the
%                        Jacobian's OPD numerator emits in the deck's
%                        BaseUnits under BOTH settings -- the same units
%                        as w_nom/opd() and as the dwdz/dwdsurf/dwdgrid
%                        rungs (dwdx was the odd rung out, scaled to
%                        OPD-metres; that made `wall = dwdx*x + w0` mix
%                        units by 1/CBM on non-metre decks).  Columns are
%                        OPD-BaseUnits per rad (rotations) and
%                        OPD-BaseUnits per SI METRE (translations --
%                        the poke denominator is unchanged).  The option
%                        is retained so existing callers keep running.
%     'delta'            finite-difference step. Either:
%                        - (1,1) double: single value for all DOFs
%                        - (1,6) double: [Rx Ry Rz Tx Ty Tz] deltas
%                        Rotations always in rad.  Translation units set
%                        by 'delta_units'.  Default 1e-8.
%     'delta_units'      'si' (default) | 'base'.  Units of the
%                        TRANSLATION entries of 'delta': 'si' = SI metres,
%                        'base' = prescription BaseUnits (converted to
%                        metres via CBM).  Rotations are rad either way.
%                        Example: on an mm Rx, delta=1e-5 delta_units=
%                        'base' is the same 10 nm translation poke as the
%                        1e-8 SI default.
%     'method'           'central' (default) | 'forward'.
%     'exit_pupil_elt'   element id at which to evaluate the OPD.
%                        Default -1 = nElt-1 (XP convention).
%     'verbose'          logical.  Default false.
%     'reload_rx'        logical.  Default true.  Pass false from a
%                        multi-field supervisor so per-field set_src_fov
%                        survives.
%     'ngridpts'         ray-grid sampling override (nGridPts).  Default
%                        [] = keep the .in-file value.  Clamped by the
%                        engine to [3, model-size limit] (warns).
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
    opts.elts                (:,1) double = []
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
    opts.groups              = []   % containers.Map name -> col vec,
                                    % or [] = no extras
    opts.groups_auto         (1,1) logical = false
    opts.group_coords        (1,:) char {mustBeMember( ...
                                opts.group_coords, ...
                                {'global','local'})} = 'global'
    opts.group_fp_mode       (1,:) char {mustBeMember( ...
                                opts.group_fp_mode, ...
                                {'auto','none','sxp','srs'})} = 'auto'
    opts.group_stop_mode     (1,:) char {mustBeMember( ...
                                opts.group_stop_mode, ...
                                {'obj','elt','none'})} = 'obj'
    opts.group_stop_pos      (1,3) double = [0 0 0]
    opts.rot_output          (1,:) char {mustBeMember( ...
        opts.rot_output, {'natural','base-per-rad'})} = 'natural'
    opts.delta               (:,:) double {mustBeDeltaSize} = 1e-8
    opts.delta_units         (1,:) char {mustBeMember(opts.delta_units, ...
                                {'si','base'})} = 'si'
    opts.method              (1,:) char {mustBeMember(opts.method, ...
                                {'central','forward'})} = 'central'
    opts.exit_pupil_elt      (1,1) double {mustBeInteger} = -1
    opts.verbose             (1,1) logical = false
    opts.reload_rx           (1,1) logical = true
    opts.ngridpts            double {mustBeScalarOrEmpty} = []
    opts.src_samp            double {mustBeScalarOrEmpty, mustBeInteger} = []
    opts.compute_los         (1,1) logical = false
    opts.spot_elt            double {mustBeScalarOrEmpty, mustBeInteger} = []
    opts.orient (1,:) char {mustBeMember(opts.orient, {'raw','xy'})} = 'raw'   % OPD array orientation (doc/opd_conventions.md)
    opts.sign   (1,:) char {mustBeMember(opts.sign, {'opl','wavefront'})} = 'opl' % OPD sign convention
end

if ~isempty(opts.stop_elt) && ~isempty(opts.stop_obj_pos)
    error('macos:dw_dx:stop', ...
        'stop_elt and stop_obj_pos are mutually exclusive');
end

if opts.reload_rx
    session.load_rx(rx_path);
end
apply_ngridpts(session, opts.ngridpts, 'dw_dx');

% Apply source sampling if specified
if ~isempty(opts.src_samp)
    session.set_src_sampling(opts.src_samp);
    session.modify();  % Flush cache so the new sampling takes effect
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

% Resolve the FD step to SI metres (translations) / rad (rotations)
% before the inner loop.  'base' scales ONLY the translation entries by
% CBM; a scalar delta is expanded so its translation use is scaled too.
delta_si = opts.delta;
if strcmp(opts.delta_units, 'base')
    if isscalar(delta_si)
        delta_si = repmat(delta_si, 1, 6);
    end
    delta_si(4:6) = delta_si(4:6) * cbm;   % BaseUnits -> metres
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
    'elts', opts.elts, ...
    'fp_mode', opts.fp_mode, ...
    'ep_elt', opts.ep_elt, ...
    'include_non_optics', opts.include_non_optics)];

% Group channels (appended AFTER per-element).
groups = containers.Map('KeyType','char','ValueType','any');
if opts.groups_auto
    groups = macos.channels.parse_rx_groups(rx_path);
end
if isa(opts.groups, 'containers.Map')
    k = keys(opts.groups);
    for kk = 1:numel(k)
        groups(k{kk}) = opts.groups(k{kk});
    end
end
if groups.Count > 0
    grp_chans = macos.channels.grouped_rigid_body_channels(session, ...
        groups, ...
        'dofs', opts.dofs, ...
        'rx_path', rx_path, ...
        'fp_mode', opts.group_fp_mode, ...
        'ep_elt', opts.ep_elt, ...
        'coords', opts.group_coords, ...
        'stop_mode', opts.group_stop_mode, ...
        'stop_obj_pos', opts.group_stop_pos, ...
        'stop_elt', 0);
    channels = [channels; grp_chans];
end

if isempty(channels)
    error('macos:dw_dx:nochan', 'no channels found');
end

% Output scale: IDENTITY -- the Jacobian's OPD numerator emits in the
% deck's BaseUnits (Dave, 2026-08-25), matching w_nom/opd() and the
% dwdz/dwdsurf/dwdgrid rungs, so `wall = dwdx*x + w0` is unit-consistent
% on any deck.  (Historically dwdx alone multiplied by CBM to emit
% OPD-metres, except rotations under 'base-per-rad' -- that option is
% now a no-op, kept for API compatibility.)  The poke DENOMINATOR is
% untouched: rad for rotations, SI metres for translations.
function s = output_scale_fn(~)
    s = 1;
end

wf_func = @() local_wf(session, wf_elt);

% Create spot_func if LOS computation requested
if opts.compute_los
    if isempty(opts.spot_elt)
        spot_elt_use = n_elt;  % Default to focal plane
    else
        spot_elt_use = opts.spot_elt;
    end
    spot_func = @() local_spot(spot_elt_use);
else
    spot_func = [];
end

[dwdx, w_nom_2d, w_nom_vec, indx, names, dcdx, spot_pos, spot_neg, spot_nom, spot_pert] = ...
    macos.dwdx_for_current_source(channels, wf_func, delta_si, ...
        'method', opts.method, ...
        'output_scale_fn', @output_scale_fn, ...
        'verbose', opts.verbose, ...
        'spot_func', spot_func);

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
out.delta_units   = opts.delta_units;
out.method        = opts.method;
out.wf_elt        = wf_elt;
out.rot_output    = opts.rot_output;
out.cbm           = cbm;
out.base_units    = base_units;

out = apply_opd_convention(out, opts.orient, opts.sign);
% Add LOS fields if SPOT was computed
if opts.compute_los
    out.dcdx      = dcdx;
    out.spot_elt  = spot_elt_use;
    if strcmp(opts.method, 'central')
        out.spot_pos = spot_pos;
        out.spot_neg = spot_neg;
    else  % forward
        out.spot_nom  = spot_nom;
        out.spot_pert = spot_pert;
    end
end
end


function W = local_wf(session, wf_elt)
session.trace(wf_elt);
W = session.opd();
end

function S = local_spot(spot_elt)
S = macos.spot(spot_elt, 'ref', 'tout', 'at', 'chief');
end

function mustBeDeltaSize(d)
    if ~(isequal(size(d), [1 1]) || isequal(size(d), [1 6]))
        error('macos:dw_dx:deltaSize', ...
            'delta must be (1,1) or (1,6)');
    end
end
