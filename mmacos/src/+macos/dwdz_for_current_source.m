function [dwdz, w_nom_2d, w_nom_vec, indx, names, dcdx, spot_pos, spot_neg, spot_nom, spot_pert] = ...
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
%     'method'    'central' (default) | 'forward'.
%     'verbose'   logical (default false) -- print per-channel RMS.
%     'spot_func' function handle () -> SPOT struct (optional).
%                 When provided, computes LOS sensitivities dC/dX.
%
%   Outputs:
%     dwdz      Nw × Nz  finite-difference Jacobian in m2v-vector space.
%     w_nom_2d  N × N    full nominal OPD canvas (un-vectorised).
%     w_nom_vec Nw × 1   nominal OPD values at non-zero mask positions.
%     indx      m2v.m bookkeeping struct (i / j / size).
%     names     Nz × 1 cell of channel-name strings.
%     dcdx      Nz × 2 LOS sensitivities [dc_x/dX, dc_y/dX] (empty if no spot_func).
%     spot_pos  Nz × 1 cell of SPOT structs at +delta (central, empty otherwise).
%     spot_neg  Nz × 1 cell of SPOT structs at -delta (central, empty otherwise).
%     spot_nom  Nz × 1 cell of SPOT structs at nominal (forward, empty otherwise).
%     spot_pert Nz × 1 cell of SPOT structs at +delta (forward, empty otherwise).
%
%   See also: macos.m2v, macos.dw_dz_zernike.
arguments
    channels     cell
    wf_func      function_handle
    delta        (1,1) double
    opts.method  (1,:) char {mustBeMember(opts.method, ...
                     {'central','forward'})} = 'central'
    opts.verbose (1,1) logical = false
    opts.spot_func = []  % function handle or []
end

compute_los = ~isempty(opts.spot_func);

w_nom_2d = wf_func();
[w_nom_vec, indx] = macos.m2v(w_nom_2d);
Nw = numel(w_nom_vec);
Nz = numel(channels);
dwdz = zeros(Nw, Nz);
names = cell(Nz, 1);

% Initialize LOS outputs
if compute_los
    dcdx = zeros(Nz, 2);
    if strcmp(opts.method, 'central')
        spot_pos = cell(Nz, 1);
        spot_neg = cell(Nz, 1);
        spot_nom = {};
        spot_pert = {};
    else  % forward
        spot_nom = cell(Nz, 1);
        spot_pert = cell(Nz, 1);
        spot_pos = {};
        spot_neg = {};
    end
else
    dcdx = [];
    spot_pos = {};
    spot_neg = {};
    spot_nom = {};
    spot_pert = {};
end

for k = 1:Nz
    ch = channels{k};
    switch opts.method
        case 'central'
            ch.apply(+delta);
            w_plus = macos.m2v(wf_func(), indx);
            if compute_los
                s_plus = opts.spot_func();
                spot_pos{k} = s_plus;
                c_plus = s_plus.centroid;
            end
            ch.apply(-delta);
            w_minus = macos.m2v(wf_func(), indx);
            if compute_los
                s_minus = opts.spot_func();
                spot_neg{k} = s_minus;
                c_minus = s_minus.centroid;
            end
            ch.restore();
            dwdz(:, k) = (w_plus - w_minus) / (2 * delta);
            if compute_los
                dcdx(k, :) = (c_plus - c_minus) / (2 * delta);
            end
        case 'forward'
            if compute_los && k == 1
                s_nom_single = opts.spot_func();
                c_nom = s_nom_single.centroid;
            end
            ch.apply(+delta);
            w_plus = macos.m2v(wf_func(), indx);
            if compute_los
                s_plus = opts.spot_func();
                spot_pert{k} = s_plus;
                spot_nom{k} = s_nom_single;
                c_plus = s_plus.centroid;
            end
            ch.restore();
            dwdz(:, k) = (w_plus - w_nom_vec) / delta;
            if compute_los
                dcdx(k, :) = (c_plus - c_nom) / delta;
            end
    end
    names{k} = ch.name();
    if opts.verbose
        col_rms = sqrt(mean(dwdz(:, k).^2));
        fprintf('[dwdz] %3d/%d  %-24s  RMS dw/dz = %.3e', ...
            k, Nz, ch.name(), col_rms);
        if compute_los
            los_rms = sqrt(sum(dcdx(k, :).^2));
            fprintf('  RMS dC/dx = %.3e', los_rms);
        end
        fprintf('\n');
    end
end
end
