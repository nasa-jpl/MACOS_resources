function out = dw_dz_zernike(session, rx_path, opts)
%MACOS.DW_DZ_ZERNIKE  Single-field dw/dz_Zernike sensitivity driver.
%   out = macos.dw_dz_zernike(SESSION, RX_PATH) loads RX_PATH on
%   SESSION (an open macos.Session), discovers MonZern + Zern
%   eligible elements from the Rx, builds one channel per
%   (element, mode) pair, runs the finite-difference sweep at the
%   current source state, and returns a struct of the result.
%
%   Required positional inputs:
%     SESSION   macos.Session handle.
%     RX_PATH   path to .in file (loaded onto SESSION).
%
%   Name-value pairs:
%     'kinds'         cellstr subset of {'monzern','ffzern','zern'}.
%                     Default {'monzern','zern'}.
%     'zmode_start'   lowest Zernike mode to perturb.  Default 4
%                     (skip piston / tip / tilt -- redundant with
%                     rigid-body Tz/Ry/Rx).
%     'n_zcoef'       highest Zernike mode.  Default 15.
%     'delta'         finite-difference step.  Default 1e-6.
%     'method'        'central' | 'forward'.  Default central.
%     'exit_pupil_elt'  element id at which to evaluate the wavefront;
%                       default nElt-1 (the XP convention).
%     'verbose'       logical, prints per-channel RMS.  Default false.
%     'ngridpts'      ray-grid sampling override (nGridPts).  Default
%                     [] = keep the .in-file value.  Clamped by the
%                     engine to [3, model-size limit] (warns).
%
%   Output struct fields:
%     dwdz        Nw × Nz finite-difference Jacobian.
%     w_nom_2d    N × N nominal OPD canvas.
%     w_nom_vec   Nw × 1 nominal OPD values at non-zero mask positions.
%     indx        m2v.m bookkeeping struct.
%     channel_names  Nz × 1 cell of channel names.
%     iElt, mode  Nz × 1 vectors -- the (element, mode) decomposition.
%     kind        Nz × 1 cell of kind labels ('MonZern' / 'Zern' / ...).
%     rx_path     echo of the prescription path.
%     wf_elt      element index of the wavefront evaluation.
%     delta       echo of finite-difference step.
%     method      echo of FD method.
%
%   See also: macos.dwdz_for_current_source, macos.channels.

arguments
    session
    rx_path     (1,:) char {mustBeNonempty}
    opts.kinds                cell    = {'monzern','zern'}
    opts.zmode_start          (1,1) double {mustBeInteger, mustBePositive} = 4
    opts.n_zcoef              (1,1) double {mustBeInteger, mustBePositive} = 15
    opts.delta                (1,1) double = 1e-6
    opts.method               (1,:) char {mustBeMember(opts.method, ...
                                  {'central','forward'})} = 'central'
    opts.exit_pupil_elt       (1,1) double {mustBeInteger} = -1
    opts.verbose              (1,1) logical = false
    opts.reload_rx            (1,1) logical = true
    opts.ngridpts             double {mustBeScalarOrEmpty} = []
end

% reload_rx=true is the right default for a standalone single-field
% call (load Rx, compute Jacobian, done).  Multi-field supervisors
% should pass reload_rx=false so the source-FoV state established
% before each per-field call survives -- otherwise load_rx resets
% ChfRayDir back to nominal and every field sees the same nominal
% OPD.
if opts.reload_rx
    session.load_rx(rx_path);
end
apply_ngridpts(session, opts.ngridpts, 'dw_dz_zernike');
n_elt = session.num_elt();
if opts.exit_pupil_elt < 0
    wf_elt = n_elt - 1;
else
    wf_elt = opts.exit_pupil_elt;
end

target_modes = (opts.zmode_start : opts.n_zcoef).';
if isempty(target_modes)
    error('macos:dw_dz_zernike:modes', ...
        'zmode_start (%d) must be <= n_zcoef (%d)', ...
        opts.zmode_start, opts.n_zcoef);
end

% Discover eligibility + build (element, mode) channel lists.
ff_elts = session.find_freeform_elts();
ze_elts = session.find_zern_elts(rx_path);

mp_ff = containers.Map('KeyType','int32','ValueType','any');
for k = 1:numel(ff_elts)
    mp_ff(int32(ff_elts(k))) = target_modes;
end
mp_ze = containers.Map('KeyType','int32','ValueType','any');
for k = 1:numel(ze_elts)
    mp_ze(int32(ze_elts(k))) = target_modes;
end

% Build channels in CANONICAL ORDER: kind-major, element-minor,
% mode-minor.  Users work with en-bloc Jacobians (all MonZern, then
% all FFZern, then all Zern), so the natural block layout is the
% right output order.  The kind blocks always appear in the fixed
% order monzern -> ffzern -> zern regardless of the order in which
% the caller listed them in 'kinds'.  Within each block, elements
% are in element-id order (the builders iterate find_*_elts() sorted)
% and modes are in mode-index order.
channels = {};
kinds_l = lower(opts.kinds);
if any(strcmp(kinds_l, 'monzern'))
    channels = [channels; macos.channels.freeform_monzern_channels( ...
        session, rx_path, 'modes_per_elt', mp_ff)];
end
if any(strcmp(kinds_l, 'ffzern'))
    channels = [channels; macos.channels.freeform_ffzern_channels( ...
        session, rx_path, 'modes_per_elt', mp_ff)];
end
if any(strcmp(kinds_l, 'zern'))
    channels = [channels; macos.channels.zernike_channels( ...
        session, rx_path, 'modes_per_elt', mp_ze)];
end
if isempty(channels)
    error('macos:dw_dz_zernike:nochan', 'no channels found');
end

wf_func = @() local_wf(session, wf_elt);

[dwdz, w_nom_2d, w_nom_vec, indx, names] = ...
    macos.dwdz_for_current_source(channels, wf_func, opts.delta, ...
        'method', opts.method, 'verbose', opts.verbose);

iElt_out = zeros(numel(channels), 1);
mode_out = zeros(numel(channels), 1);
kind_out = cell(numel(channels), 1);
for k = 1:numel(channels)
    iElt_out(k) = channels{k}.iElt;
    mode_out(k) = channels{k}.mode;
    kind_out{k} = channels{k}.kind;
end

out = struct();
out.dwdz          = dwdz;
out.w_nom_2d      = w_nom_2d;
out.w_nom_vec     = w_nom_vec;
out.indx          = indx;
out.channel_names = names;
out.iElt          = iElt_out;
out.mode          = mode_out;
out.kind          = kind_out;
out.rx_path       = rx_path;
out.wf_elt        = wf_elt;
out.delta         = opts.delta;
out.method        = opts.method;
end


function W = local_wf(session, wf_elt)
session.trace(wf_elt);
W = session.opd();
end
