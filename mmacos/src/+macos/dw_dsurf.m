function out = dw_dsurf(session, rx_path, opts)
%MACOS.DW_DSURF  Single-field dw/dKr + dw/dKc sensitivity driver.
%   out = macos.dw_dsurf(SESSION, RX_PATH) loads RX_PATH on SESSION,
%   discovers the POWERED optics (Element= Reflector / Refractor with
%   |Kr| << the flat sentinel 1e22), builds one channel per (optic, param)
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
%     'delta'          finite-difference step (Kr in BaseUnits, Kc
%                      dimensionless).  Default 1e-6.
%     'method'         'central' | 'forward'.  Default central.
%     'exit_pupil_elt' element id at which to evaluate the wavefront;
%                      default nElt-1 (the XP convention).
%     'verbose'        logical, prints per-channel RMS.  Default false.
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
    opts.delta                (1,1) double = 1e-6
    opts.method               (1,:) char {mustBeMember(opts.method, ...
                                  {'central','forward'})} = 'central'
    opts.exit_pupil_elt       (1,1) double {mustBeInteger} = -1
    opts.verbose              (1,1) logical = false
    opts.reload_rx            (1,1) logical = true
    opts.ngridpts             double {mustBeScalarOrEmpty} = []
end

if opts.reload_rx
    session.load_rx(rx_path);
end
apply_ngridpts(session, opts.ngridpts, 'dw_dsurf');
n_elt = session.num_elt();
if opts.exit_pupil_elt < 0
    wf_elt = n_elt - 1;
else
    wf_elt = opts.exit_pupil_elt;
end

channels = macos.channels.surf_channels(session, rx_path, 'params', opts.params);
if isempty(channels)
    error('macos:dw_dsurf:nochan', ...
        'no powered optics (Reflector/Refractor, |Kr|<<1e22) found in %s', ...
        rx_path);
end

wf_func = @() local_wf(session, wf_elt);

[dwds, w_nom_2d, w_nom_vec, indx, names] = ...
    macos.dwdz_for_current_source(channels, wf_func, opts.delta, ...
        'method', opts.method, 'verbose', opts.verbose);

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
end


function W = local_wf(session, wf_elt)
session.trace(wf_elt);
W = session.opd();
end
