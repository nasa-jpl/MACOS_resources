function [dwdx, w_nom_2d, w_nom_vec, indx, names] = ...
    dwdx_for_current_source(channels, wf_func, delta, opts)
%MACOS.DWDX_FOR_CURRENT_SOURCE  Single-field dw/dx inner FD loop.
%   Computes the rigid-body channel Jacobian at the macos source
%   state CURRENTLY loaded.  Caller owns load_rx / src_fov / STOP /
%   channel setup -- this function just runs the finite-difference
%   sweep.
%
%   Inputs:
%     channels  cell array of channel handles (RigidBodyChannel /
%               SourceChannel / FocalPlaneChannel).
%     wf_func   function handle () -> 2D OPD matrix.  Typically
%               @() (session.trace(wf_elt); session.opd()).
%     delta     finite-difference step (scalar; rotations in rad,
%               translations in SI metres).
%
%   Name-value pairs:
%     'method'           'central' (default) | 'forward'.
%     'output_scale_fn'  function handle  ch -> scale (default 1).
%                        Multiplied into each column after the
%                        difference.  Use for CBM rescaling
%                        (OPD_BaseUnits -> OPD_metres) etc.
%     'verbose'          logical (default false).
%
%   Outputs:
%     dwdx       Nw × Nz finite-difference Jacobian, AFTER output_scale.
%     w_nom_2d   N × N nominal OPD canvas.
%     w_nom_vec  Nw × 1 nominal OPD values at non-zero mask positions.
%     indx       m2v.m bookkeeping (i / j / size).
%     names      Nz × 1 cell of channel-name strings.
%
%   See also: macos.dwdz_for_current_source, macos.m2v.

arguments
    channels             cell
    wf_func              function_handle
    delta                (1,1) double
    opts.method          (1,:) char {mustBeMember(opts.method, ...
                            {'central','forward'})} = 'central'
    opts.output_scale_fn = []  % function handle or []
    opts.verbose         (1,1) logical = false
end

if isempty(opts.output_scale_fn)
    opts.output_scale_fn = @(~) 1;
end

w_nom_2d = wf_func();
[w_nom_vec, indx] = macos.m2v(w_nom_2d);
Nw = numel(w_nom_vec);
Nz = numel(channels);
dwdx = zeros(Nw, Nz);
names = cell(Nz, 1);

for k = 1:Nz
    ch = channels{k};
    switch opts.method
        case 'central'
            ch.apply(+delta);
            w_plus = macos.m2v(wf_func(), indx);
            ch.apply(-delta);
            w_minus = macos.m2v(wf_func(), indx);
            ch.restore();
            dwdx(:, k) = (w_plus - w_minus) / (2 * delta);
        case 'forward'
            ch.apply(+delta);
            w_plus = macos.m2v(wf_func(), indx);
            ch.restore();
            dwdx(:, k) = (w_plus - w_nom_vec) / delta;
    end
    dwdx(:, k) = dwdx(:, k) * opts.output_scale_fn(ch);
    names{k} = ch.name();
    if opts.verbose
        col_rms = sqrt(mean(dwdx(:, k).^2));
        fprintf('[dwdx] %3d/%d  %-24s  RMS dw/dx = %.3e\n', ...
            k, Nz, ch.name(), col_rms);
    end
end
end
