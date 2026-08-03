function [dwdx, w_nom_2d, w_nom_vec, indx, names, dcdx, spot_pos, spot_neg, spot_nom, spot_pert] = ...
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
%     delta     finite-difference step. Either:
%               - (1,1) double: single value for all DOFs
%               - (1,6) double: [Rx Ry Rz Tx Ty Tz] deltas
%               Rotations in rad, translations in BaseUnits.
%
%   Name-value pairs:
%     'method'           'central' (default) | 'forward'.
%     'output_scale_fn'  function handle  ch -> scale (default 1).
%                        Multiplied into each column after the
%                        difference.  Use for CBM rescaling
%                        (OPD_BaseUnits -> OPD_metres) etc.
%     'cbm'              Conversion factor: BaseUnits × CBM = metres.
%                        Required when delta has translation components.
%     'verbose'          logical (default false).
%     'spot_func'        function handle () -> SPOT struct (optional).
%                        When provided, computes LOS sensitivities dC/dX
%                        where C = [c_x, c_y] is the spot centroid.
%
%   Outputs:
%     dwdx       Nw × Nz finite-difference Jacobian, AFTER output_scale.
%     w_nom_2d   N × N nominal OPD canvas.
%     w_nom_vec  Nw × 1 nominal OPD values at non-zero mask positions.
%     indx       m2v.m bookkeeping (i / j / size).
%     names      Nz × 1 cell of channel-name strings.
%     dcdx       Nz × 2 LOS sensitivities [dc_x/dX, dc_y/dX] (empty if no spot_func).
%     spot_pos   Nz × 1 cell of SPOT structs at +delta (central method, empty otherwise).
%     spot_neg   Nz × 1 cell of SPOT structs at -delta (central method, empty otherwise).
%     spot_nom   Nz × 1 cell of SPOT structs at nominal (forward method, empty otherwise).
%     spot_pert  Nz × 1 cell of SPOT structs at +delta (forward method, empty otherwise).
%
%   See also: macos.dwdz_for_current_source, macos.m2v.

arguments
    channels             cell
    wf_func              function_handle
    delta                (:,:) double {mustBeDeltaSize}
    opts.method          (1,:) char {mustBeMember(opts.method, ...
                            {'central','forward'})} = 'central'
    opts.output_scale_fn = []  % function handle or []
    opts.cbm             (1,1) double = NaN
    opts.verbose         (1,1) logical = false
    opts.spot_func       = []  % function handle or []
end

if isempty(opts.output_scale_fn)
    opts.output_scale_fn = @(~) 1;
end

% Expand scalar delta to (1,6) if needed
if isscalar(delta)
    delta_vec = repmat(delta, 1, 6);
else
    delta_vec = delta;
end

compute_los = ~isempty(opts.spot_func);

w_nom_2d = wf_func();
[w_nom_vec, indx] = macos.m2v(w_nom_2d);
Nw = numel(w_nom_vec);
Nz = numel(channels);
dwdx = zeros(Nw, Nz);
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

    % Get the appropriate delta for this channel
    if isprop(ch, 'dof_idx')
        ch_delta = delta_vec(ch.dof_idx + 1);
        % Convert translation deltas from BaseUnits to metres
        % Rotations (dof_idx 0-2) are already in radians
        % Translations (dof_idx 3-5) are in BaseUnits, need × CBM
        if ch.dof_idx >= 3
            if isnan(opts.cbm)
                error('macos:dwdx_for_current_source:cbm', ...
                    'CBM required for translation perturbations');
            end
            ch_delta = ch_delta * opts.cbm;
        end
    else
        % Source channels or other types: use first delta
        ch_delta = delta_vec(1);
    end

    switch opts.method
        case 'central'
            ch.apply(+ch_delta);
            w_plus = macos.m2v(wf_func(), indx);
            if compute_los
                s_plus = opts.spot_func();
                spot_pos{k} = s_plus;
                c_plus = s_plus.centroid;  % [1×2]: [c_x, c_y]
            end
            ch.apply(-ch_delta);
            w_minus = macos.m2v(wf_func(), indx);
            if compute_los
                s_minus = opts.spot_func();
                spot_neg{k} = s_minus;
                c_minus = s_minus.centroid;  % [1×2]: [c_x, c_y]
            end
            ch.restore();
            dwdx(:, k) = (w_plus - w_minus) / (2 * ch_delta);
            if compute_los
                dcdx(k, :) = (c_plus - c_minus) / (2 * ch_delta);
            end
        case 'forward'
            if compute_los && k == 1
                % Capture nominal SPOT once (same for all channels)
                s_nom_single = opts.spot_func();
                c_nom = s_nom_single.centroid;  % [1×2]: [c_x, c_y]
            end
            ch.apply(+ch_delta);
            w_plus = macos.m2v(wf_func(), indx);
            if compute_los
                s_plus = opts.spot_func();
                spot_pert{k} = s_plus;
                spot_nom{k} = s_nom_single;  % Store reference to same nominal
                c_plus = s_plus.centroid;  % [1×2]: [c_x, c_y]
            end
            ch.restore();
            dwdx(:, k) = (w_plus - w_nom_vec) / ch_delta;
            if compute_los
                dcdx(k, :) = (c_plus - c_nom) / ch_delta;
            end
    end
    dwdx(:, k) = dwdx(:, k) * opts.output_scale_fn(ch);
    if compute_los
        % Apply same scaling to LOS sensitivities
        dcdx(k, :) = dcdx(k, :) * opts.output_scale_fn(ch);
    end
    names{k} = ch.name();
    if opts.verbose
        col_rms = sqrt(mean(dwdx(:, k).^2));
        fprintf('[dwdx] %3d/%d  %-24s  delta=%.3e  RMS dw/dx = %.3e', ...
            k, Nz, ch.name(), ch_delta, col_rms);
        if compute_los
            los_rms = sqrt(sum(dcdx(k, :).^2));
            fprintf('  RMS dC/dx = %.3e', los_rms);
        end
        fprintf('\n');
    end
end
end

function mustBeDeltaSize(d)
    if ~(isequal(size(d), [1 1]) || isequal(size(d), [1 6]))
        error('macos:dwdx_for_current_source:deltaSize', ...
            'delta must be (1,1) or (1,6)');
    end
end
