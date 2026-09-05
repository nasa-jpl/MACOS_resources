function out = dw_dsurf(session, rx_path, opts)
%MACOS.DW_DSURF  Single-field dw/dKr + dw/dKc sensitivity driver.
%   out = macos.dw_dsurf(SESSION, RX_PATH) loads RX_PATH on SESSION,
%   discovers the POWERED optics (Element= Reflector / Refractor /
%   NSReflector / NSRefractor / Segment with |Kr| << the flat sentinel
%   1e22 -- engine-queried, see macos.find_powered_elts), builds one
%   channel per (optic, param)
%   with param in {Kr, Kc}, runs the finite-difference sweep at the current
%   source state, and returns a struct of the result.
%
%   This is the powered-surface (radius + conic) companion to
%   macos.dw_dz_zernike, built on the same channel + FD machinery.
%
%   Required positional inputs:
%     SESSION   macos.Session handle.
%     RX_PATH   path to .in file (loaded onto SESSION).
%
%   Name-value pairs:
%     'params'         cellstr subset of {'Kr','Kc'}.  Default {'Kr','Kc'}.
%     'elts'           Vector of element IDs to include in the sensitivity
%                      calculation.  Default [] (auto-detect all powered
%                      optics from the loaded prescription).  An explicitly
%                      requested id that is not powered-capable ERRORS with
%                      a named reason (macos:channels:eltNotEligible) -- it
%                      is never dropped silently.
%                      Example: 'elts', [2; 4; 6] includes only elements 2, 4, 6.
%     'delta'          finite-difference step (Kr in BaseUnits, Kc
%                      dimensionless).  Default 1e-6.
%     'method'         'central' | 'forward'.  Default central.
%     'exit_pupil_elt' element id at which to evaluate the wavefront;
%                      default nElt-1 (the XP convention).
%     'verbose'        logical, prints per-channel RMS.  Default false.
%     'remove_ptt'     project piston + tip + tilt out of each Kr/Kc
%                      response column (default false).  A radius/conic
%                      error re-focuses and re-points; that global piston +
%                      pointing is normally aligned out during assembly, so
%                      removing it leaves the surviving higher-order figure
%                      -- the quantity a sensitivity budget wants.  Fit over
%                      each column's own aperture footprint (private/
%                      remove_ptt_columns).
%     'reload_rx'      reload the Rx first (default true; pass false from a
%                      multi-field supervisor that has set the source FoV).
%     'ngridpts'       ray-grid sampling override (nGridPts).  Default [] =
%                      keep the .in-file value.  Clamped by the engine to
%                      [3, model-size limit] (warns).
%
%   Output struct fields:
%     dwds          Nw × Ns finite-difference Jacobian (OPD per Kr/Kc unit).
%     w_nom_2d      N × N nominal OPD canvas.
%     w_nom_vec     Nw × 1 nominal OPD at non-zero mask positions.
%     indx          m2v.m bookkeeping struct.
%     channel_names Ns × 1 cell of channel names ('Elt k Kr' / 'Elt k Kc').
%     iElt          Ns × 1 element ids.
%     param         Ns × 1 cell of 'Kr' / 'Kc'.
%     rx_path, wf_elt, delta, method  echoes.
%
%   See also: macos.dw_dz_zernike, macos.channels.surf_channels,
%             macos.find_powered_elts.
arguments
    session
    rx_path     (1,:) char {mustBeNonempty}
    opts.params               cell    = {'Kr','Kc'}
    opts.elts                 (:,1) double = []
    opts.delta                (1,1) double = 1e-6
    opts.method               (1,:) char {mustBeMember(opts.method, ...
                                  {'central','forward'})} = 'central'
    opts.exit_pupil_elt       (1,1) double {mustBeInteger} = -1
    opts.verbose              (1,1) logical = false
    opts.reload_rx            (1,1) logical = true
    opts.ngridpts             double {mustBeScalarOrEmpty} = []
    opts.src_samp             double {mustBeScalarOrEmpty, mustBeInteger} = []
    opts.compute_los          (1,1) logical = false
    opts.spot_elt             double {mustBeScalarOrEmpty, mustBeInteger} = []
    opts.orient (1,:) char {mustBeMember(opts.orient, {'raw','xy'})} = 'raw'   % OPD array orientation (doc/opd_conventions.md)
    opts.sign   (1,:) char {mustBeMember(opts.sign, {'opl','wavefront'})} = 'opl' % OPD sign convention
    opts.remove_ptt (1,1) logical = false   % project piston+tip+tilt out of
                                            % each Kr/Kc response (aligned out
                                            % during assembly) -- default OFF
                                            % so the raw response is unchanged
end

if opts.reload_rx
    session.load_rx(rx_path);
end
apply_ngridpts(session, opts.ngridpts, 'dw_dsurf');

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

channels = macos.channels.surf_channels(session, rx_path, ...
    'params', opts.params, ...
    'elts', opts.elts);
if isempty(channels)
    error('macos:dw_dsurf:nochan', ...
        ['no powered optics (Reflector/Refractor/NSReflector/' ...
         'NSRefractor/Segment, |Kr|<<1e22) found in %s'], rx_path);
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

[dwds, w_nom_2d, w_nom_vec, indx, names, dcdx, spot_pos, spot_neg, spot_nom, spot_pert] = ...
    macos.dwdz_for_current_source(channels, wf_func, opts.delta, ...
        'method', opts.method, 'verbose', opts.verbose, 'spot_func', spot_func);

% Optionally project piston + tip + tilt out of each Kr/Kc response: a
% radius/conic error re-focuses and re-points, and that global piston +
% pointing is normally ALIGNED OUT during assembly.  Per column = per
% (optic, param); each dwdsurf optic (SM/TM) is full-beam, so its footprint
% is the whole exit-pupil aperture and one plane per column is the per-optic
% removal.  Done on the raw response, in the orientation indx describes,
% before the orient/sign convention (a transpose/negate commutes with it).
if opts.remove_ptt
    dwds = remove_ptt_columns(dwds, indx);
end

iElt_out  = zeros(numel(channels), 1);
param_out = cell(numel(channels), 1);
for k = 1:numel(channels)
    iElt_out(k)  = channels{k}.iElt;
    param_out{k} = channels{k}.param;
end

out = struct();
out.dwds          = dwds;
out.w_nom_2d      = w_nom_2d;
out.w_nom_vec     = w_nom_vec;
out.indx          = indx;
out.channel_names = names;
out.iElt          = iElt_out;
out.param         = param_out;
out.rx_path       = rx_path;
out.wf_elt        = wf_elt;
out.delta         = opts.delta;
out.method        = opts.method;

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
