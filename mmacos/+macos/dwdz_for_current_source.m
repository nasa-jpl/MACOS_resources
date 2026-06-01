function [dwdz, w_nom_2d, w_nom_vec, indx, names] = ...
    dwdz_for_current_source(channels, wf_func, delta, opts)
%MACOS.DWDZ_FOR_CURRENT_SOURCE  Single-field dw/dz_Zernike inner loop.
%   Computes the Zernike-channel Jacobian at the macos source state
%   CURRENTLY loaded.  Caller owns: load_rx + any STOP / src_fov /
%   channel setup -- this function just runs the finite-difference
%   sweep.
%
%   Inputs:
%     channels  cell array of macos.channels.ZernikeCoefChannel.
%     wf_func   function handle () -> 2D OPD matrix.  Typically
%               @() macos.opd() after a fresh trace_rays.
%     delta     finite-difference step (scalar).
%
%   Name-value pairs:
%     'method'   'central' (default) | 'forward'.
%     'verbose'  logical (default false) -- print per-channel RMS.
%
%   Outputs:
%     dwdz      Nw × Nz  finite-difference Jacobian in m2v-vector space.
%     w_nom_2d  N × N    full nominal OPD canvas (un-vectorised).
%     w_nom_vec Nw × 1   nominal OPD values at non-zero mask positions.
%     indx      m2v.m bookkeeping struct (i / j / size).
%     names     Nz × 1 cell of channel-name strings.
%
%   See also: macos.m2v, macos.dw_dz_zernike.
arguments
    channels    cell
    wf_func     function_handle
    delta       (1,1) double
    opts.method (1,:) char {mustBeMember(opts.method, ...
                    {'central','forward'})} = 'central'
    opts.verbose (1,1) logical = false
end

w_nom_2d = wf_func();
[w_nom_vec, indx] = macos.m2v(w_nom_2d);
Nw = numel(w_nom_vec);
Nz = numel(channels);
dwdz = zeros(Nw, Nz);
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
            dwdz(:, k) = (w_plus - w_minus) / (2 * delta);
        case 'forward'
            ch.apply(+delta);
            w_plus = macos.m2v(wf_func(), indx);
            ch.restore();
            dwdz(:, k) = (w_plus - w_nom_vec) / delta;
    end
    names{k} = ch.name();
    if opts.verbose
        col_rms = sqrt(mean(dwdz(:, k).^2));
        fprintf('[dwdz] %3d/%d  %-24s  RMS dw/dz = %.3e\n', ...
            k, Nz, ch.name(), col_rms);
    end
end
end
